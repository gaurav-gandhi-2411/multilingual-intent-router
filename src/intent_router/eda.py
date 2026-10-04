from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.feature_selection import chi2  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from intent_router.data import load_data  # noqa: E402
from intent_router.dedup import compute_dup_groups  # noqa: E402

SLANG = [
    "pls", "plz", "whats", "aight", "hrs", "hr", "thx", "thanks", "nvm", "u", "ur", "tmrw",
    "tmr", "asap", "btw", "idk", "lol", "gonna", "wanna", "gotta", "rn", "ya", "yep", "nope",
    "cuz", "tho", "ok", "okay", "k", "dunno", "lemme", "gimme", "kinda", "omg", "fyi", "eta",
]  # fmt: skip
# thanks/ok/okay/eta/hr are ordinary words in logistics text; they are reported separately
# but excluded from the is_noisy decision (STRICT_SLANG).
NON_NOISY_WORDS = {"thanks", "ok", "okay", "eta", "hr", "k", "yep", "nope"}
STRICT_SLANG = [w for w in SLANG if w not in NON_NOISY_WORDS]

OTHER_ID = "other_ID [A-Z]{2,4}-\\d+"
ENTITY_PATTERNS: dict[str, str] = {
    "LD-\\d+": r"\bLD-\d+\b",
    "PO-\\d+": r"\bPO-\d+\b",
    OTHER_ID: r"\b(?!LD-|PO-)[A-Z]{2,4}-\d+\b",
    "number": r"\b\d+(?:[.,]\d+)?\b",
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "url": r"https?://\S+|www\.\S+",
    "percent": r"\d+(?:\.\d+)?\s?%",
    "time": r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:am|pm)\b",
    "date": (
        r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b"
        r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    ),
}

EMOJI_RE = re.compile("[\U0001f300-\U0001faff☀-➿\U0001f000-\U0001f2ff⭐⬆✅❌️]")


def to_py(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to plain python for json."""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return to_py(obj.tolist())
    return obj


def pct(a: np.ndarray, qs: tuple[int, ...] = (50, 95, 99)) -> dict[str, float]:
    """p50/p95/p99 plus min and max of an array."""
    out = {f"p{q}": float(np.percentile(a, q)) for q in qs}
    out.update(min=float(a.min()), max=float(a.max()))
    return out


# ----------------------------------------------------------------------------- length


def token_length_stats(df: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    """Token-count percentiles per candidate tokenizer (with special tokens, with prefix)."""
    from transformers import AutoTokenizer

    prefixes = cfg.get("query_prefix", {})
    out: dict[str, Any] = {}
    for name in cfg["candidate_tokenizers"]:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            texts = [prefixes.get(name, "") + t for t in df["text"]]
            lens = np.array([len(x) for x in tok(texts)["input_ids"]])
            out[name] = {**pct(lens), "prefix": prefixes.get(name, "")}
        except Exception as e:  # noqa: BLE001 - record and continue; one tokenizer must not kill EDA
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def decide_max_len(tok_stats: dict[str, Any], default: int) -> dict[str, Any]:
    """max_len rule: `default` if max p99 <= default, else ceil(max p99 / 16) * 16."""
    p99s = [v["p99"] for v in tok_stats.values() if "p99" in v]
    worst = max(p99s)
    max_len = default if worst <= default else int(np.ceil(worst / 16) * 16)
    return {
        "rule": f"{default} if max p99 across tokenizers <= {default} else ceil(max_p99/16)*16",
        "max_p99_across_tokenizers": worst,
        "tokenizers_used": [k for k, v in tok_stats.items() if "p99" in v],
        "max_len": max_len,
    }


# ----------------------------------------------------------------------------- language


def n_letters(text: str) -> int:
    """Count alphabetic-category characters (letters in any script, incl. CJK/Arabic)."""
    return sum(1 for c in text if unicodedata.category(c).startswith("L"))


ID_RE = re.compile(r"\b[A-Za-z]{2,4}-\d+\b")


def clean_for_lid(text: str) -> str:
    """Strip entities/emoji/digits so IDs like LD-10889 do not drive language ID."""
    text = ID_RE.sub(" ", text)
    text = re.sub(r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.-]+", " ", text)
    text = EMOJI_RE.sub(" ", text)
    text = re.sub(r"[\d_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_languages(df: pd.DataFrame, ecfg: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-row primary language, undetermined flag, non-English segment flags via lingua."""
    from lingua import Language, LanguageDetectorBuilder

    langs = [
        Language.ENGLISH, Language.SPANISH, Language.FRENCH, Language.GERMAN, Language.CHINESE,
        Language.ARABIC, Language.PORTUGUESE, Language.ITALIAN, Language.DUTCH,
        Language.JAPANESE, Language.HINDI,
    ]  # fmt: skip
    # Restricted to 11 plausible languages: a 27-language set diluted confidence on short
    # English rows (an off-topic English message scored only 0.33) and produced spurious segments.
    det = LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build()
    min_letters = ecfg["lang_min_letters"]
    min_conf = ecfg["lang_min_confidence"]
    seg_min_words = ecfg["segment_min_words"]
    seg_min_conf = ecfg["segment_min_confidence"]
    logographic = (Language.CHINESE, Language.JAPANESE, Language.ARABIC)
    rows = []
    for raw in df["text"]:
        text = clean_for_lid(raw)
        nl = n_letters(text)
        conf = det.compute_language_confidence_values(text)
        top = conf[0] if conf else None
        lang, top_conf = "undetermined", 0.0
        if top is not None:
            top_conf = float(top.value)
            if nl >= min_letters and top_conf >= min_conf:
                lang = top.language.iso_code_639_1.name.lower()
        seg_langs: list[str] = []
        if lang != "undetermined":
            for r in det.detect_multiple_languages_of(text):
                code = r.language.iso_code_639_1.name.lower()
                if r.language == Language.ENGLISH:
                    if r.word_count >= seg_min_words:
                        seg_langs.append(code)
                    continue
                # Non-English segments are re-scored in isolation: lingua's segmenter invents
                # short spurious it/fr/es spans inside plain English, so require a confident
                # standalone detection of the same language.
                seg_conf = det.compute_language_confidence_values(text[r.start_index : r.end_index])
                top_seg = seg_conf[0]
                if (
                    (r.word_count >= seg_min_words or r.language in logographic)
                    and top_seg.language == r.language
                    and top_seg.value >= seg_min_conf
                ):
                    seg_langs.append(code)
        non_en_seg = any(s != "en" for s in seg_langs)
        en_seg = any(s == "en" for s in seg_langs)
        rows.append(
            {
                "lang": lang,
                "lang_conf": top_conf,
                "any_non_en": (lang not in ("en", "undetermined")) or non_en_seg,
                "code_mixed": non_en_seg and en_seg,
                "seg_langs": ",".join(sorted(set(seg_langs))),
            }
        )
    meta = {
        "library": "lingua-language-detector (offline, preloaded models)",
        "candidate_languages": [lang.name for lang in langs],
        "primary_language": f"top confidence value; 'undetermined' if letters < {min_letters} "
        f"or top confidence < {min_conf}",
        "segments": "detect_multiple_languages_of; a segment counts if word_count >= "
        f"{seg_min_words} (or script is ZH/JA/AR, where word_count is unreliable); non-English "
        "segments must also re-score standalone as the same language with confidence >= "
        f"{seg_min_conf} (the segmenter otherwise invents it/fr/es spans in plain English)",
        "text_cleaning": "IDs (XX-123), URLs, emails, emoji and digits removed before detection",
        "any_non_en_def_B": "primary non-English OR any counted non-English segment",
        "code_mixed": "counted English segment AND counted non-English segment in same text",
        "undetermined": "reported separately, never counted as non-English",
    }
    return pd.DataFrame(rows, index=df.index), meta


# ----------------------------------------------------------------------------- noise


def load_wordlist(name: str) -> set[str]:
    """Whole-word alphabetic tokens from a tokenizer vocab (approximate English wordlist)."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(name)
    except Exception:  # noqa: BLE001 - fall back to the (identical-vocab) distilbert tokenizer
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return {t for t in tok.get_vocab() if t.isalpha() and t.isascii() and not t.startswith("##")}


def is_known_word(w: str, vocab: set[str]) -> bool:
    """Known if in vocab or after stripping a common English inflection suffix."""
    if w in vocab:
        return True
    for suf in ("s", "es", "ed", "d", "ing", "ly", "er"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            stem = w[: -len(suf)]
            if stem in vocab or stem + "e" in vocab:
                return True
    return False


def noise_features(df: pd.DataFrame, lang: pd.DataFrame, vocab: set[str]) -> pd.DataFrame:
    """Per-row noise indicators."""
    slang_re = re.compile(r"\b(" + "|".join(STRICT_SLANG) + r")\b", re.I)
    all_slang_re = re.compile(r"\b(" + "|".join(SLANG) + r")\b", re.I)
    rows = []
    for text, lg in zip(df["text"], lang["lang"], strict=True):
        letters = [c for c in text if c.isalpha()]
        lower_only = bool(letters) and all(not c.isupper() for c in letters)
        stripped = re.sub(r"\b[A-Za-z]{2,4}-\d+\b", " ", text).replace("’", "'")
        words = re.findall(r"[a-z]+", stripped.lower())
        oov = [w for w in words if len(w) > 1 and not is_known_word(w, vocab)]
        is_en = lg == "en"
        rows.append(
            {
                "all_lower": lower_only,
                "has_letters": bool(letters),
                "slang_hits": len(slang_re.findall(text)),
                "slang_any": len(all_slang_re.findall(text)) > 0,
                "emoji": bool(EMOJI_RE.search(text)),
                "oov_tokens": oov if is_en else [],
                "oov_n": len(oov) if is_en else 0,
                "oov_rate": (len(oov) / max(len(words), 1)) if is_en else np.nan,
            }
        )
    out = pd.DataFrame(rows, index=df.index)
    out["is_noisy"] = out["all_lower"] | (out["slang_hits"] > 0) | (out["oov_n"] > 0) | out["emoji"]
    return out


# ----------------------------------------------------------------------------- shortcuts


def shortcut_audit(df: pd.DataFrame) -> dict[str, Any]:
    """Per entity pattern: rows, label distribution, shares, shortcut_risk flag."""
    label_counts = df["label"].value_counts()
    out: dict[str, Any] = {}
    for name, pat in ENTITY_PATTERNS.items():
        rx = re.compile(pat, re.I if name != OTHER_ID else 0)
        has = df["text"].apply(lambda t, rx=rx: bool(rx.search(t)))
        n = int(has.sum())
        dist = df.loc[has, "label"].value_counts()
        top_label = dist.index[0] if n else None
        out[name] = {
            "n_rows": n,
            "label_distribution": dist.to_dict(),
            "top_label": top_label,
            "max_label_share": float(dist.iloc[0] / n) if n else 0.0,
            "share_of_top_label_rows_with_pattern": (
                float(dist.iloc[0] / label_counts[top_label]) if n else 0.0
            ),
            "shortcut_risk": bool(n >= 5 and dist.iloc[0] / n >= 0.8),
            "example_ids": df.loc[has, "id"].head(5).tolist(),
        }
    return out


# ----------------------------------------------------------------------------- siblings


def sibling_analysis(df: pd.DataFrame, sim: np.ndarray) -> dict[str, Any]:
    """chi2 / LR discriminative n-grams for shipment_information.* + 1-NN confusion preview."""
    sib = sorted(lb for lb in df["label"].unique() if lb.startswith("shipment_information."))
    sub = df[df["label"].isin(sib)]
    cv = CountVectorizer(ngram_range=(1, 2), lowercase=True, min_df=2, binary=True)
    x = cv.fit_transform(sub["text"])
    terms = np.array(cv.get_feature_names_out())
    res: dict[str, Any] = {"labels": sib, "n_rows": int(len(sub)), "chi2_top15": {}, "lr_top15": {}}
    y = sub["label"].to_numpy()
    for lb in sib:
        yb = (y == lb).astype(int)
        scores, _ = chi2(x, yb)
        scores = np.nan_to_num(scores)
        rate_in = np.asarray(x[yb == 1].mean(axis=0)).ravel()
        rate_out = np.asarray(x[yb == 0].mean(axis=0)).ravel()
        pos = np.where(rate_in > rate_out, scores, -1)  # only terms enriched in this class
        top = np.argsort(-pos)[:15]
        res["chi2_top15"][lb] = [
            {
                "ngram": terms[i],
                "chi2": float(scores[i]),
                "rate_in_class": float(rate_in[i]),
                "rate_in_rest": float(rate_out[i]),
            }
            for i in top
        ]
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42).fit(x, y)
    for k, lb in enumerate(lr.classes_):
        top = np.argsort(-lr.coef_[k])[:15]
        res["lr_top15"][lb] = [{"ngram": terms[i], "weight": float(lr.coef_[k][i])} for i in top]
    # 1-NN, leave-self-out, over all rows
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    labels = df["label"].to_numpy()
    pred = labels[s.argmax(axis=1)]
    res["overall_1nn_accuracy"] = float((pred == labels).mean())
    res["per_label_1nn_accuracy"] = {
        lb: float((pred[labels == lb] == lb).mean()) for lb in sorted(df["label"].unique())
    }
    conf: dict[str, dict[str, int]] = {}
    for a in sib:
        conf[a] = {b: int(((labels == a) & (pred == b)).sum()) for b in sib}
        conf[a]["<other_label>"] = int(((labels == a) & ~np.isin(pred, sib)).sum())
    res["sibling_1nn_confusion"] = conf
    return res


# ----------------------------------------------------------------------------- figures


def make_figures(df: pd.DataFrame, rows: pd.DataFrame, fig_dir: Path, ct: pd.DataFrame) -> None:
    """Class counts, length histogram, language bar chart, language x label heatmap."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    counts = df["label"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(counts.index, counts.values)
    ax.set_xlabel("rows")
    ax.set_title("Class counts")
    fig.tight_layout()
    fig.savefig(fig_dir / "class_counts.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rows["n_chars"], bins=40)
    ax.set_xlabel("characters")
    ax.set_ylabel("rows")
    ax.set_title("Text length (characters)")
    fig.tight_layout()
    fig.savefig(fig_dir / "length_hist.png", dpi=130)
    plt.close(fig)

    lc = rows["lang"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(lc.index, lc.values)
    ax.set_ylabel("rows")
    ax.set_title("Primary language (lingua; 'undetermined' = too short/ambiguous)")
    fig.tight_layout()
    fig.savefig(fig_dir / "language_counts.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(ct.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(ct.shape[1]), ct.columns, rotation=60, ha="right")
    ax.set_yticks(range(ct.shape[0]), ct.index)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, int(ct.values[i, j]), ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Language x label")
    fig.tight_layout()
    fig.savefig(fig_dir / "language_by_label.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main


def print_language_sanity(df: pd.DataFrame, lang: pd.DataFrame) -> None:
    """stdout-only sanity check of language ID (raw text must never be written to files)."""
    print("== lingua sanity check (stdout only) ==")
    tmp = df.assign(lang=lang["lang"], conf=lang["lang_conf"], segs=lang["seg_langs"])
    for lg in sorted(set(tmp["lang"]) - {"en"}):
        print(f"-- {lg} (n={int((tmp['lang'] == lg).sum())})")
        for _, r in tmp[tmp["lang"] == lg].head(6).iterrows():
            print(f"   {r['id']} conf={r['conf']:.2f} segs={r['segs']} | {r['text'][:90]}")
    print("-- English-primary rows with a counted non-English segment")
    for _, r in tmp[(tmp["lang"] == "en") & lang["any_non_en"]].head(10).iterrows():
        print(f"   {r['id']} conf={r['conf']:.2f} segs={r['segs']} | {r['text'][:90]}")
    print("-- shortest rows")
    for _, r in tmp.assign(n=tmp["text"].str.len()).sort_values("n").head(12).iterrows():
        print(f"   {r['id']} lang={r['lang']} conf={r['conf']:.2f} | {r['text'][:60]}")


def run(cfg: dict) -> dict[str, Any]:
    """Run all EDA steps; write results/eda.json, eda_rows.csv, figures."""
    df = load_data(cfg["data_path"])
    ecfg = cfg["eda"]
    res_dir = Path(cfg["results_dir"])
    (res_dir / "figures").mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"n_rows": len(df)}

    counts = df["label"].value_counts()
    out["class_counts"] = counts.to_dict()
    out["imbalance_ratio_max_over_min"] = float(counts.max() / counts.min())
    out["n_classes"] = int(len(counts))

    n_chars = df["text"].str.len().to_numpy()
    tok_stats = token_length_stats(df, cfg)
    out["length"] = {
        "chars": pct(n_chars),
        "n_texts_under_15_chars": int((n_chars < 15).sum()),
        "tokens_by_tokenizer": tok_stats,
        "max_len_decision": decide_max_len(tok_stats, ecfg["default_max_len"]),
    }

    lang, lang_meta = detect_languages(df, ecfg)
    determined = lang["lang"] != "undetermined"
    non_en_primary = determined & (lang["lang"] != "en")
    ct = pd.crosstab(lang["lang"], df["label"])
    out["language"] = {
        "method": lang_meta,
        "guards": {
            "lang_min_letters": ecfg["lang_min_letters"],
            "lang_min_confidence": ecfg["lang_min_confidence"],
            "segment_min_words": ecfg["segment_min_words"],
            "segment_min_confidence": ecfg["segment_min_confidence"],
        },
        "primary_language_counts": lang["lang"].value_counts().to_dict(),
        "undetermined_n": int((~determined).sum()),
        "pct_non_english_primary_definition_A": float(non_en_primary.mean() * 100),
        "pct_any_non_english_segment_definition_B": float(lang["any_non_en"].mean() * 100),
        "pct_code_mixed": float(lang["code_mixed"].mean() * 100),
        "pct_non_english_A_among_determined": float(non_en_primary.sum() / determined.sum() * 100),
        "pdf_claim_pct_non_english": 25,
        "n_rows": len(df),
        "language_x_label": {k: v.to_dict() for k, v in ct.iterrows()},
    }

    vocab = load_wordlist(ecfg["wordlist_tokenizer"])
    noise = noise_features(df, lang, vocab)
    en = lang["lang"] == "en"
    oov_counter = pd.Series([w for ws in noise["oov_tokens"] for w in ws], dtype=object)
    out["noise"] = {
        "wordlist": f"{len(vocab)} whole-word alphabetic tokens from {ecfg['wordlist_tokenizer']} "
        "vocab (+ inflection-suffix stripping); approximation, OOV overstates typos",
        "pct_all_lowercase_among_rows_with_letters": float(
            noise.loc[noise["has_letters"], "all_lower"].mean() * 100
        ),
        "pct_slang_rows_strict": float((noise["slang_hits"] > 0).mean() * 100),
        "pct_slang_rows_incl_thanks_ok_eta": float(noise["slang_any"].mean() * 100),
        "strict_slang_list": STRICT_SLANG,
        "excluded_from_strict": sorted(NON_NOISY_WORDS),
        "pct_emoji_rows": float(noise["emoji"].mean() * 100),
        "n_english_rows": int(en.sum()),
        "pct_english_rows_with_oov_token": float((noise.loc[en, "oov_n"] > 0).mean() * 100),
        "mean_oov_rate_english_rows": float(noise.loc[en, "oov_rate"].mean()),
        "top_oov_tokens": oov_counter.value_counts().head(40).to_dict(),
        "pct_is_noisy": float(noise["is_noisy"].mean() * 100),
        "is_noisy_definition": "all-lowercase OR strict slang hit OR >=1 OOV token (English "
        "rows only) OR emoji",
        "pct_is_noisy_excluding_lowercase": float(
            ((noise["slang_hits"] > 0) | (noise["oov_n"] > 0) | noise["emoji"]).mean() * 100
        ),
    }

    out["shortcut_audit"] = shortcut_audit(df)

    groups, pairs, sim = compute_dup_groups(df, cfg["dup_threshold"])
    out["sibling_overlap"] = sibling_analysis(df, sim)

    lab = df["label"].to_numpy()
    ids = df["id"].to_numpy()
    iu, ju = np.triu_indices_from(sim, k=1)
    sel = (sim[iu, ju] > cfg["cross_label_threshold"]) & (lab[iu] != lab[ju])
    gsizes = pd.Series(groups).value_counts()
    out["near_duplicates"] = {
        "threshold": cfg["dup_threshold"],
        "n_pairs": len(pairs),
        "pairs": [
            {
                "id_a": ids[i],
                "id_b": ids[j],
                "label_a": lab[i],
                "label_b": lab[j],
                "cosine": s,
            }
            for i, j, s in pairs
        ],
        "n_groups": int(groups.max() + 1),
        "n_groups_with_more_than_one_row": int((gsizes > 1).sum()),
        "largest_group_size": int(gsizes.max()),
        "cross_label_threshold": cfg["cross_label_threshold"],
        "n_cross_label_pairs_above_cross_label_threshold": int(sel.sum()),
        "cross_label_pairs": [
            {
                "id_a": ids[i],
                "id_b": ids[j],
                "label_a": lab[i],
                "label_b": lab[j],
                "cosine": float(sim[i, j]),
            }
            for i, j in zip(iu[sel], ju[sel], strict=True)
        ],
    }

    rows = pd.DataFrame(
        {
            "id": df["id"],
            "lang": lang["lang"],
            "code_mixed": lang["code_mixed"],
            "is_noisy": noise["is_noisy"],
            "n_chars": n_chars,
            "any_non_en": lang["any_non_en"],
        }
    )
    rows.to_csv(res_dir / "eda_rows.csv", index=False)
    make_figures(df, rows, res_dir / "figures", ct)
    (res_dir / "eda.json").write_text(json.dumps(to_py(out), indent=2), encoding="utf-8")
    print_language_sanity(df, lang)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = run(cfg)
    print("wrote results/eda.json; max_len =", out["length"]["max_len_decision"]["max_len"])


if __name__ == "__main__":
    main()
