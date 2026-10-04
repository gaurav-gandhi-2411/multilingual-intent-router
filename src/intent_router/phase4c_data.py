"""Robustness and open-set training fixes data preparation (pre-registered; generation only).

Stages: synthetic (local qwen3:8b novel-intent messages), a3_translate (NLLB train+val copies +
5% noise copies), eval_translate (m2m100 English OOF rows -> es/fr/de/zh), labse (translation
quality filter), embed (per-holdout OE leakage counts).

Confidentiality: every text (synthetic, translated) goes to outputs/ (gitignored). results/ gets
counts, ids and statistics only. The only network endpoint for LLM calls is loopback Ollama via
llm_audit's guarded client. All GPU work runs one stage at a time with exclusive GPU access.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import random
import re
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router import data as data_mod
from intent_router import gpu_lock
from intent_router.baselines import embed_e5
from intent_router.llm_audit import (
    LABEL_DESCRIPTIONS,
    _http_json,
    _ollama_whitelisted,
    validate_loopback_url,
)
from intent_router.models import prepare_text
from intent_router.ood import NON_LOCO_LABELS
from intent_router.robustness import perturb_chars, row_rng, translate_texts

STAGES = ("synthetic", "a3_translate", "eval_translate", "labse", "embed", "all")
ALL_ORDER = STAGES[:-1]
A3_LANGS = ("es", "fr", "de", "zh")
QUANTILES = (0, 5, 25, 50, 75, 95, 100)


# ------------------------------------------------------------------ small pure helpers
class BatchParseError(ValueError):
    """The model's reply was not a strict JSON array of non-empty strings."""


def lang_for_row(row_id: str, seed: int = 42, langs: Sequence[str] = A3_LANGS) -> str:
    """Target language from a seeded hash of (seed, row id); deterministic, order independent."""
    h = hashlib.sha256(f"{seed}|{row_id}".encode()).digest()
    return langs[int.from_bytes(h[:8], "little") % len(langs)]


def parse_batch(raw: str, n_requested: int, min_frac: float = 0.5) -> list[str]:
    """Strictly parse a generation reply into at most n_requested strings.

    Accepts a bare JSON array (optionally inside one ``` fence). Rejects anything else, any
    non-string / empty element, and replies with fewer than min_frac * n_requested items.
    """
    text = raw.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BatchParseError(f"not valid JSON: {exc}") from exc
    if not isinstance(obj, list):
        raise BatchParseError(f"expected a JSON array, got {type(obj).__name__}")
    items: list[str] = []
    for el in obj:
        if not isinstance(el, str) or not el.strip():
            raise BatchParseError("array elements must be non-empty strings")
        items.append(" ".join(el.split()))
    if len(items) < max(1, int(min_frac * n_requested)):
        raise BatchParseError(f"only {len(items)} items for {n_requested} requested")
    return items[:n_requested]


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def dedupe(texts: Sequence[str], emb: np.ndarray, thr: float) -> tuple[list[int], dict[str, int]]:
    """Greedy in-order dedupe: exact (case/whitespace-insensitive) then cosine > thr to a kept row.

    emb must be L2-normalised (cosine = dot product), same order as texts. Returns
    (kept indices, {"exact": n, "cosine": n}).
    """
    kept: list[int] = []
    seen: set[str] = set()
    drops = {"exact": 0, "cosine": 0}
    for i, t in enumerate(texts):
        key = _norm(t)
        if key in seen:
            drops["exact"] += 1
            continue
        if kept and float((emb[kept] @ emb[i]).max()) > thr:
            drops["cosine"] += 1
            continue
        seen.add(key)
        kept.append(i)
    return kept, drops


def assert_no_test_ids(ids: Sequence[str], test_ids: set[str]) -> None:
    """Raise if any id belongs to the test split (A3 must never touch test rows)."""
    bad = sorted(set(ids) & test_ids)
    if bad:
        raise AssertionError(f"test-split ids present: {bad[:5]} ({len(bad)} total)")


def assert_distinct_mt(eval_system: str, a3_system: str) -> None:
    """Evaluation MT must differ from the A3 training MT (no train/eval translator overlap)."""
    if eval_system == a3_system:
        raise AssertionError(f"eval MT and A3 MT are the same system: {eval_system}")


def max_cos_and_drop(
    syn_emb: np.ndarray, heldout_emb: np.ndarray, thr: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per synthetic item: max cosine to any held-out row, and whether it is dropped (> thr)."""
    mx = (syn_emb @ heldout_emb.T).max(axis=1)
    return mx, mx > thr


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    """min/q05/q25/median/q75/q95/max and mean of a 1-d array (numbers only)."""
    v = np.asarray(values, dtype=float)
    names = ("min", "q05", "q25", "q50", "q75", "q95", "max")
    out = {n: float(np.percentile(v, q)) for n, q in zip(names, QUANTILES, strict=True)}
    out["mean"] = float(v.mean())
    return out


def split_counts(total: int, quotas: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of total across keys proportional to quotas."""
    raw = {k: total * q / sum(quotas.values()) for k, q in quotas.items()}
    out = {k: int(v) for k, v in raw.items()}
    rest = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - int(raw[k])), k))[:rest]:
        out[k] += 1
    return out


def plan_batches(scfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Seeded generation plan: one (lang, kind) cell per batch, <= batch_size items each."""
    cells = split_counts(
        int(scfg["n_target"]),
        {
            f"{lang}|{kind}": lq * kq
            for lang, lq in scfg["lang_quota"].items()
            for kind, kq in scfg["kind_quota"].items()
        },
    )
    plan: list[dict[str, Any]] = []
    bs = int(scfg["batch_size"])
    for cell, count in cells.items():
        lang, kind = cell.split("|")
        n_batches = -(-count // bs)
        sizes = split_counts(count, {str(j): 1.0 for j in range(n_batches)}) if count else {}
        plan += [{"lang": lang, "kind": kind, "n": sizes[str(j)]} for j in range(n_batches)]
    for idx, b in enumerate(plan):
        b["batch_idx"] = idx
        b["seed"] = int(scfg["batch_seed_base"]) + idx
    return plan


# Topic areas rotated across batches (4 per call) for diversity: the v1 run without them lost 27%
# to dedupe and most "novel" items still fit an existing intent. Written from the logistics
# domain, not from dataset rows.
NOVEL_TOPICS: tuple[str, ...] = (
    "customs duties and tariffs", "freight pricing and quotes", "carrier onboarding and contracts",
    "warehouse stock counts and inventory levels", "driver HR, pay and shift rosters",
    "fuel, maintenance and vehicle repair", "carbon emissions and sustainability reporting",
    "cargo insurance and claims", "API and system integrations", "route planning and optimisation",
    "supplier and vendor management", "user permissions and software licences",
)  # fmt: skip
OOS_TOPICS: tuple[str, ...] = (
    "cooking and recipes", "travel and holidays", "health and fitness", "movies, music and games",
    "personal finance and taxes", "programming and gadgets", "sports", "weather and news",
    "relationships and family", "education and homework", "shopping for clothes or phones",
    "jokes, riddles and trivia",
)  # fmt: skip


def topic_hint(kind: str, batch_idx: int, k: int = 4) -> list[str]:
    """k consecutive topics (cyclic) for this batch; deterministic in (kind, batch_idx)."""
    pool = NOVEL_TOPICS if kind == "logistics_novel" else OOS_TOPICS
    return [pool[(batch_idx * k + j) % len(pool)] for j in range(k)]


def build_prompt(
    lang_name: str, kind: str, n: int, topics: Sequence[str] = ()
) -> list[dict[str, str]]:
    """Chat messages: the 12 label descriptions as EXCLUSIONS, then the generation request."""
    excl = "\n".join(f"- {d}" for d in LABEL_DESCRIPTIONS.values())
    if kind == "logistics_novel":
        clause = (
            "but are still plausible requests to a logistics / supply-chain platform assistant "
            "(a kind of task none of the intents above covers)"
        )
    else:
        clause = (
            "and are generic out-of-scope requests unrelated to logistics (the kind of thing "
            "someone might ask any general-purpose chatbot)"
        )
    user = (
        "A logistics / supply-chain platform assistant already handles these intents:\n"
        f"{excl}\n\nThe list above is an EXCLUSION list. Write {n} different short chat messages "
        f"in {lang_name} that a user might send to the assistant, that fit NONE of the intents "
        f"above, {clause}. Style: informal chat register, 3 to 25 words, with realistic typos, "
        "missing punctuation and abbreviations. Every message must differ in topic and wording. "
        + (f"Spread the messages across these areas: {', '.join(topics)}. " if topics else "")
        + f"Reply with only a JSON array of {n} strings."
    )
    system = "You write realistic user messages for assistant testing. Output JSON only."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ------------------------------------------------------------------ io / gpu plumbing
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item()), encoding="utf-8")


@contextlib.contextmanager
def gpu_stage(cfg: dict[str, Any], name: str, ollama: bool = False) -> Iterator[dict[str, Any]]:
    """One GPU stage under the cross-project lock; Ollama's own processes whitelisted if asked."""
    with contextlib.ExitStack() as st:
        if ollama:
            st.enter_context(_ollama_whitelisted())
        lock = st.enter_context(
            gpu_lock.gpu_exclusive(int(cfg["gpu_lock_expected_s"]), f"phase4c_data {name}")
        )
        yield lock


def _free_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _e5_settings(cfg: dict[str, Any]) -> tuple[str, int]:
    tr = yaml.safe_load(Path(cfg["final_config"]).read_text(encoding="utf-8"))["train"]
    return tr["model_name"], int(tr["max_len"])


def embed_with_e5(cfg: dict[str, Any], texts: list[str]) -> np.ndarray:
    """Frozen e5 (query prefix, masked mean pooling, L2-normalised), as baselines.embed_e5."""
    name, max_len = _e5_settings(cfg)
    return embed_e5([prepare_text(t, name) for t in texts], name, max_len)


def load_frozen(cfg: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Frozen-e5 features of all dataset rows (ids, feats); checked against final.yaml."""
    z = np.load(cfg["frozen_e5_path"], allow_pickle=False)
    name, max_len = _e5_settings(cfg)
    assert str(z["model_name"]) == name and int(z["max_len"]) == max_len, "frozen cache drift"
    return [str(i) for i in z["ids"]], z["feats"].astype(np.float32)


def _dataset_labels(cfg: dict[str, Any]) -> pd.DataFrame:
    """id,text,label,split for all 500 rows."""
    return data_mod.get_frame(["train", "val", "test"], cfg["data_path"], cfg["splits_path"])


# ------------------------------------------------------------------ stage: synthetic
def _chat(base_url: str, scfg: dict[str, Any], messages: list[dict[str, str]], seed: int) -> str:
    payload: dict[str, Any] = {
        "model": scfg["model_tag"],
        "messages": messages,
        "stream": False,
        "think": bool(scfg["think"]),
        "format": {"type": "array", "items": {"type": "string"}},
        "options": {
            "temperature": float(scfg["temperature"]),
            "seed": seed,
            "num_ctx": int(scfg["num_ctx"]),
            "num_predict": int(scfg["num_predict"]),
        },
        "keep_alive": "10m",
    }
    out = _http_json(f"{base_url}/api/chat", payload, float(scfg["timeout_s"]))
    return str(out["message"]["content"])


def _model_record(base_url: str, tag: str) -> dict[str, Any]:
    tags = {m["name"]: m for m in _http_json(f"{base_url}/api/tags", None, 30)["models"]}
    m = tags[tag]
    return {"tag": tag, "digest": m["digest"], "size_bytes": m["size"]}


def generate_batches(cfg: dict[str, Any], out_dir: Path) -> tuple[list[dict[str, Any]], dict]:
    """Run (or resume from cache) every planned batch; returns (batch records, run stats)."""
    scfg = cfg["synthetic"]
    base_url = validate_loopback_url(scfg["base_url"])
    plan = plan_batches(scfg)
    cache = out_dir / "synthetic_raw.jsonl"
    done: dict[int, dict[str, Any]] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["batch_idx"]] = rec
    stats = {"parse_failures": 0, "batches_resumed": len(done), "gen_seconds": 0.0}
    t0 = time.monotonic()
    with cache.open("a", encoding="utf-8") as fh:
        for b in plan:
            if b["batch_idx"] in done:
                continue
            msgs = build_prompt(
                scfg["lang_names"][b["lang"]],
                b["kind"],
                b["n"],
                topic_hint(b["kind"], b["batch_idx"]),
            )
            for attempt in range(int(scfg["max_attempts"])):
                seed = b["seed"] + 7919 * attempt
                try:
                    items = parse_batch(_chat(base_url, scfg, msgs, seed), b["n"])
                except BatchParseError as exc:
                    stats["parse_failures"] += 1
                    print(f"[synthetic] batch {b['batch_idx']} attempt {attempt}: {exc}")
                    continue
                rec = {**b, "seed_used": seed, "attempts": attempt + 1, "items": items}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[b["batch_idx"]] = rec
                print(f"[synthetic] batch {b['batch_idx']}/{len(plan)} {b['lang']} {b['kind']} "
                      f"-> {len(items)} items")  # fmt: skip
                break
            else:
                raise RuntimeError(f"batch {b['batch_idx']} failed {scfg['max_attempts']} times")
    stats["gen_seconds"] = round(time.monotonic() - t0, 1)
    with contextlib.suppress(Exception):  # free VRAM for the embedding stage
        _http_json(f"{base_url}/api/chat", {"model": scfg["model_tag"], "messages": [],
                                            "keep_alive": 0, "stream": False}, 60)  # fmt: skip
    return [done[b["batch_idx"]] for b in plan], stats


def stage_synthetic(cfg: dict[str, Any]) -> None:
    """Generate, embed, dedupe and flag the synthetic novel-intent set (texts -> outputs/)."""
    scfg = cfg["synthetic"]
    out_dir, res_dir = Path(cfg["outputs_dir"]), Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = validate_loopback_url(scfg["base_url"])
    model_rec = _model_record(base_url, scfg["model_tag"])
    with gpu_stage(cfg, "synthetic-generate", ollama=True) as lock:
        batches, gstats = generate_batches(cfg, out_dir)
    rows = [
        {
            "syn_id": "",
            "text": t,
            "lang": b["lang"],
            "kind": b["kind"],
            "batch_seed": b["seed_used"],
        }
        for b in batches
        for t in b["items"]
    ]
    for i, r in enumerate(rows, 1):
        r["syn_id"] = f"oe-{i:04d}"
    raw = pd.DataFrame(rows)
    with gpu_stage(cfg, "synthetic-embed"):
        emb = embed_with_e5(cfg, raw["text"].tolist())
        _free_cuda()
    kept, drops = dedupe(raw["text"].tolist(), emb, float(scfg["dedupe_cos"]))
    syn = raw.iloc[kept].reset_index(drop=True)
    syn_emb = emb[kept]
    syn.to_csv(out_dir / "synthetic_oe.csv", index=False)
    np.save(out_dir / "synthetic_oe_emb.npy", syn_emb.astype(np.float32))
    ids, feats = load_frozen(cfg)
    mx = (syn_emb @ feats.T).max(axis=1)
    thr = float(scfg["dataset_flag_cos"])
    flags = pd.DataFrame(
        {"syn_id": syn["syn_id"], "max_cos_any_dataset_row": mx, "flagged": mx > thr}
    )
    flags.to_csv(out_dir / "synthetic_flags.csv", index=False)
    stats = {
        "label_note": "synthetic data from local qwen3:8b; texts in outputs/phase4c only",
        "model": model_rec,
        "think": bool(scfg["think"]),
        "temperature": scfg["temperature"],
        "n_requested": int(sum(b["n"] for b in batches)),
        "n_generated_pre_dedupe": len(raw),
        "n_final": len(syn),
        "n_batches": len(batches),
        "batch_seeds": [b["seed_used"] for b in batches],
        "batches_with_retry": [b["batch_idx"] for b in batches if b["attempts"] > 1],
        "parse_failures": gstats["parse_failures"],
        "dedupe_drops": drops,
        "dedupe_cos": scfg["dedupe_cos"],
        "counts_per_lang": {k: int(v) for k, v in syn["lang"].value_counts().items()},
        "counts_per_kind": {k: int(v) for k, v in syn["kind"].value_counts().items()},
        "counts_per_lang_kind": {
            f"{lang}|{kind}": int(n)
            for (lang, kind), n in syn.groupby(["lang", "kind"]).size().items()
        },
        "dataset_flag": {
            "threshold": thr,
            "n_flagged_of_final": int((mx > thr).sum()),
            "max_cos_any_dataset_row": quantile_summary(mx),
            "note": "diagnostic only; flagged items are NOT dropped here",
        },
        "generation_seconds": gstats["gen_seconds"],
        "batches_resumed_from_cache": gstats["batches_resumed"],
        "gpu_lock_waited_s": lock["waited_s"],
    }
    _write_json(res_dir / "synthetic_stats.json", stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "batch_seeds"}, indent=1))
    n_s = int(cfg["samples"]["n_synthetic"])
    pick = sorted(random.Random(int(cfg["seed"])).sample(range(len(syn)), min(n_s, len(syn))))
    for i in pick:
        r = syn.iloc[i]
        print(f"[sample syn] {r['syn_id']} lang={r['lang']} kind={r['kind']}: {r['text']}")


# ------------------------------------------------------------------ stage: a3_translate
def _load_seq2seq(name: str, fp16: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForSeq2SeqLM

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        name, dtype=torch.float16 if fp16 and dev.type == "cuda" else torch.float32
    )
    return model.to(dev).eval(), dev


def stage_a3_translate(cfg: dict[str, Any]) -> None:
    """NLLB (beam 4) copy + 5% char-noise copy of every train+val row; test rows never touched."""
    from transformers import AutoTokenizer

    acfg = cfg["a3"]
    out_dir, res_dir = Path(cfg["outputs_dir"]), Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _dataset_labels(cfg)
    test_ids = set(frame.loc[frame["split"] == "test", "id"])
    rows = frame[frame["split"].isin(["train", "val"])].copy()
    assert_no_test_ids(rows["id"].tolist(), test_ids)
    eda = pd.read_csv(cfg["eda_rows_path"])[["id", "lang"]].rename(columns={"lang": "src_lang"})
    rows = rows.merge(eda, on="id", how="left", validate="one_to_one")
    assert rows["src_lang"].notna().all(), "eda_rows.csv does not cover every train/val id"
    rows["tgt_lang"] = [lang_for_row(i, int(cfg["seed"])) for i in rows["id"]]
    rows["src_code"] = rows["src_lang"].map(acfg["src_codes"])
    assert rows["src_code"].notna().all(), f"unmapped source langs: {set(rows['src_lang'])}"
    t0 = time.monotonic()
    mt: dict[str, str] = {}
    with gpu_stage(cfg, "a3-nllb") as lock:
        model, dev = _load_seq2seq(acfg["model"], bool(acfg["fp16"]))
        try:
            for (src_code, tgt), grp in rows.groupby(["src_code", "tgt_lang"]):
                tok = AutoTokenizer.from_pretrained(acfg["model"], src_lang=src_code)
                out = translate_texts(grp["text"].tolist(), acfg["targets"][tgt], acfg, tok, model)
                mt.update(dict(zip(grp["id"], out, strict=True)))
        finally:
            del model
            _free_cuda()
    mt_s = time.monotonic() - t0
    noise_rows = []
    for r in rows.itertuples():
        rng = row_rng(int(cfg["seed"]), acfg["noise_name"], r.id)
        noisy, _ = perturb_chars(r.text, float(acfg["noise_rate"]), acfg["noise_mode"], rng)
        noise_rows.append((r.id, "noise", r.src_lang, noisy))
    aug = pd.DataFrame(
        [(r.id, "mt", r.tgt_lang, mt[r.id]) for r in rows.itertuples()] + noise_rows,
        columns=["src_id", "kind", "lang", "text"],
    )
    assert_no_test_ids(aug["src_id"].tolist(), test_ids)
    aug.to_csv(out_dir / "a3_aug.csv", index=False)
    stats = {
        "model": acfg["model"],
        "decoding": {"num_beams": acfg["num_beams"], "max_new_tokens": acfg["max_new_tokens"]},
        "n_source_rows": len(rows),
        "n_mt": int((aug["kind"] == "mt").sum()),
        "n_noise": int((aug["kind"] == "noise").sum()),
        "mt_target_lang_counts": {k: int(v) for k, v in rows["tgt_lang"].value_counts().items()},
        "n_same_language_pairs": int((rows["src_lang"] == rows["tgt_lang"]).sum()),
        "source_lang_counts": {k: int(v) for k, v in rows["src_lang"].value_counts().items()},
        "n_empty_mt": int((aug.loc[aug["kind"] == "mt", "text"].str.strip() == "").sum()),
        "noise_perturbation": acfg["noise_name"],
        "noise_rate": acfg["noise_rate"],
        "n_noise_unchanged": int(
            sum(n == t for (_, _, _, n), t in zip(noise_rows, rows["text"], strict=True))
        ),
        "test_ids_present": 0,
        "mt_seconds": round(mt_s, 1),
        "gpu_lock_waited_s": lock["waited_s"],
    }
    _write_json(res_dir / "a3_stats.json", stats)
    print(json.dumps(stats, indent=1))


# ------------------------------------------------------------------ stage: eval_translate
def _eval_sources(cfg: dict[str, Any]) -> pd.DataFrame:
    """English OOF rows: eda lang == 'en' and split in train/val (id, text)."""
    frame = _dataset_labels(cfg)
    eda = pd.read_csv(cfg["eda_rows_path"])[["id", "lang"]]
    df = frame.merge(eda, on="id", validate="one_to_one")
    df = df[(df["lang"] == "en") & df["split"].isin(["train", "val"])]
    return df.sort_values("id").reset_index(drop=True)[["id", "text"]]


def _m2m_translate(src: pd.DataFrame, ecfg: dict[str, Any]) -> pd.DataFrame:
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ecfg["primary_model"], src_lang="en")
    model, dev = _load_seq2seq(ecfg["primary_model"], fp16=False)
    texts = src["text"].tolist()
    parts = []
    try:
        for lang in ecfg["targets"]:
            out = [""] * len(texts)
            order = np.argsort([-len(t) for t in texts], kind="stable")
            bs = int(ecfg["batch_size"])
            for s in range(0, len(order), bs):
                idx = order[s : s + bs]
                enc = tok(
                    [texts[i] for i in idx],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(ecfg["max_new_tokens"]),
                ).to(dev)
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        forced_bos_token_id=tok.get_lang_id(lang),
                        max_new_tokens=int(ecfg["max_new_tokens"]),
                        num_beams=1,
                        do_sample=False,
                    )
                for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True), strict=True):
                    out[int(i)] = t
            parts.append(pd.DataFrame({"id": src["id"], "lang": lang, "text": out}))
    finally:
        del model
        _free_cuda()
    df = pd.concat(parts, ignore_index=True)
    df["system"] = ecfg["primary_model"]
    return df


def _opus_translate(src: pd.DataFrame, ecfg: dict[str, Any]) -> pd.DataFrame:
    """Fallback: Helsinki-NLP/opus-mt-en-{lang}, greedy (used only if m2m100 is unusable)."""
    import torch
    from huggingface_hub import list_repo_files
    from transformers import AutoTokenizer

    parts = []
    for lang in ecfg["targets"]:
        name = ecfg["fallback_pattern"].format(lang=lang)
        if not list_repo_files(name):
            raise RuntimeError(f"fallback model {name} does not exist")
        tok = AutoTokenizer.from_pretrained(name)
        model, dev = _load_seq2seq(name, fp16=False)
        texts, out = src["text"].tolist(), []
        for s in range(0, len(texts), int(ecfg["batch_size"])):
            enc = tok(texts[s : s + int(ecfg["batch_size"])], return_tensors="pt", padding=True,
                      truncation=True).to(dev)  # fmt: skip
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=int(ecfg["max_new_tokens"]),
                                     num_beams=1, do_sample=False)  # fmt: skip
            out += tok.batch_decode(gen, skip_special_tokens=True)
        del model
        _free_cuda()
        parts.append(pd.DataFrame({"id": src["id"], "lang": lang, "text": out, "system": name}))
    return pd.concat(parts, ignore_index=True)


def stage_eval_translate(cfg: dict[str, Any]) -> None:
    """English OOF rows -> es/fr/de/zh with m2m100 (opus-mt fallback); texts -> outputs/."""
    ecfg = cfg["eval_mt"]
    out_dir = Path(cfg["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    src = _eval_sources(cfg)
    frame = _dataset_labels(cfg)
    assert_no_test_ids(src["id"].tolist(), set(frame.loc[frame["split"] == "test", "id"]))
    t0 = time.monotonic()
    fallback_reason: str | None = None
    with gpu_stage(cfg, "eval-mt"):
        try:
            df = _m2m_translate(src, ecfg)
        except Exception as exc:  # noqa: BLE001 - any m2m100 failure triggers the registered fallback
            fallback_reason = repr(exc)[:300]
            print(f"[eval_translate] m2m100 failed ({fallback_reason}); using opus-mt fallback")
            df = _opus_translate(src, ecfg)
    systems = sorted(set(df["system"]))
    for s in systems:
        assert_distinct_mt(s, cfg["a3"]["model"])
    df.to_csv(out_dir / "eval_mt.csv", index=False)
    _write_json(
        Path(cfg["results_dir"]) / "eval_mt_stats.json",
        {
            "systems": systems,
            "fallback_used": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "n_source_rows": len(src),
            "n_translations": len(df),
            "n_empty": int((df["text"].str.strip() == "").sum()),
            "seconds": round(time.monotonic() - t0, 1),
        },
    )
    print(f"[eval_translate] {len(df)} translations, systems={systems}")
    rng = random.Random(int(cfg["seed"]))
    en = dict(zip(src["id"], src["text"], strict=True))
    for lang in ecfg["targets"]:
        sub = df[df["lang"] == lang].reset_index(drop=True)
        for i in sorted(rng.sample(range(len(sub)), int(cfg["samples"]["n_eval_per_lang"]))):
            print(f"[sample eval {lang}] EN: {en[sub.at[i, 'id']]}\n[sample eval {lang}] MT: "
                  f"{sub.at[i, 'text']}")  # fmt: skip


# ------------------------------------------------------------------ stage: labse
def labse_embed(texts: list[str], model_name: str, max_len: int, bs: int) -> np.ndarray:
    """LaBSE sentence embeddings: BERT pooler_output (CLS -> dense -> tanh), L2-normalised.

    Matches sentence-transformers/LaBSE's pipeline (CLS pooling -> Dense 768 tanh -> Normalize);
    labse_check() verifies the Dense layer equals BertModel's pooler.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()
    out = []
    try:
        with torch.no_grad():
            for s in range(0, len(texts), bs):
                b = tok(texts[s : s + bs], padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt").to(dev)  # fmt: skip
                e = torch.nn.functional.normalize(model(**b).pooler_output, dim=-1)
                out.append(e.float().cpu().numpy())
    finally:
        del model
        _free_cuda()
    return np.concatenate(out)


def labse_check(model_name: str) -> dict[str, Any]:
    """Verify the ST Dense layer == BertModel pooler (so pooler_output is the ST embedding)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModel

    root = Path(snapshot_download(model_name, allow_patterns=["*.json", "2_Dense/*.safetensors"]))
    dense = load_file(str(root / "2_Dense" / "model.safetensors"))
    dcfg = json.loads((root / "2_Dense" / "config.json").read_text(encoding="utf-8"))
    pcfg = json.loads((root / "1_Pooling" / "config.json").read_text(encoding="utf-8"))
    model = AutoModel.from_pretrained(model_name)
    w_ok = bool(
        np.allclose(dense["linear.weight"].numpy(), model.pooler.dense.weight.detach().numpy())
    )
    b_ok = bool(np.allclose(dense["linear.bias"].numpy(), model.pooler.dense.bias.detach().numpy()))
    return {
        "st_dense_config": dcfg,
        "st_pooling_cls_token": bool(pcfg.get("pooling_mode_cls_token")),
        "dense_weight_equals_bert_pooler": w_ok,
        "dense_bias_equals_bert_pooler": b_ok,
    }


def stage_labse(cfg: dict[str, Any]) -> None:
    """LaBSE cosine of each eval translation to its English source (filter), plus A3 diagnostic."""
    ecfg = cfg["eval_mt"]
    out_dir, res_dir = Path(cfg["outputs_dir"]), Path(cfg["results_dir"])
    check = labse_check(ecfg["labse_model"])
    assert check["dense_weight_equals_bert_pooler"] and check["dense_bias_equals_bert_pooler"], (
        f"LaBSE pooler != sentence-transformers Dense: {check}"
    )
    assert check["st_pooling_cls_token"] and check["st_dense_config"][
        "activation_function"
    ].endswith("Tanh"), check  # noqa: E501
    mt = pd.read_csv(out_dir / "eval_mt.csv")
    a3 = pd.read_csv(out_dir / "a3_aug.csv")
    frame = _dataset_labels(cfg).set_index("id")["text"]
    mt_src = [frame[i] for i in mt["id"]]
    a3_mt = a3[a3["kind"] == "mt"].reset_index(drop=True)
    a3_src = [frame[i] for i in a3_mt["src_id"]]
    mlen, bs = int(ecfg["labse_max_len"]), int(ecfg["labse_batch_size"])
    with gpu_stage(cfg, "labse") as lock:
        e_src = labse_embed(mt_src, ecfg["labse_model"], mlen, bs)
        e_mt = labse_embed(mt["text"].fillna("").tolist(), ecfg["labse_model"], mlen, bs)
        a_src = labse_embed(a3_src, ecfg["labse_model"], mlen, bs)
        a_mt = labse_embed(a3_mt["text"].fillna("").tolist(), ecfg["labse_model"], mlen, bs)
    mt["cos"] = (e_src * e_mt).sum(axis=1)
    thr = float(ecfg["labse_min_cos"])
    mt["kept"] = mt["cos"] >= thr
    mt.to_csv(out_dir / "eval_mt.csv", index=False)
    a3_cos = (a_src * a_mt).sum(axis=1)
    pd.DataFrame({"src_id": a3_mt["src_id"], "lang": a3_mt["lang"], "cos": a3_cos}).to_csv(
        out_dir / "a3_labse.csv", index=False
    )
    per_lang = {
        lang: {
            "kept": int(g["kept"].sum()),
            "dropped": int((~g["kept"]).sum()),
            "cos": quantile_summary(g["cos"].to_numpy()),
        }
        for lang, g in mt.groupby("lang")
    }
    a3_lang = {
        lang: quantile_summary(a3_cos[(a3_mt["lang"] == lang).to_numpy()])
        for lang in sorted(set(a3_mt["lang"]))
    }
    _write_json(
        res_dir / "eval_mt_filter.json",
        {
            "labse_model": ecfg["labse_model"],
            "labse_verification": check,
            "threshold_min_cos": thr,
            "eval_mt_systems": sorted(set(mt["system"])),
            "overall": {
                "kept": int(mt["kept"].sum()),
                "dropped": int((~mt["kept"]).sum()),
                "cos": quantile_summary(mt["cos"].to_numpy()),
            },  # fmt: skip
            "per_language": per_lang,
            "a3_nllb_diagnostic": {
                "note": "LaBSE cos of A3 NLLB translations to their source rows; no filtering",
                "overall": quantile_summary(a3_cos),
                "per_target_language": a3_lang,
                "n_below_threshold": int((a3_cos < thr).sum()),
                "n": len(a3_cos),
            },
            "gpu_lock_waited_s": lock["waited_s"],
        },
    )
    print(json.dumps({k: v["kept"] for k, v in per_lang.items()}), "kept per lang")
    print(json.dumps({k: v["dropped"] for k, v in per_lang.items()}), "dropped per lang")


# ------------------------------------------------------------------ stage: embed
def holdout_sets(cfg: dict[str, Any], labels: list[str]) -> dict[str, list[str]]:
    """Name -> held-out classes: headline holdout plus the 10 LOCO classes (as trackb.plan_runs)."""
    tcfg = yaml.safe_load(Path(cfg["trackb_config"]).read_text(encoding="utf-8"))
    excl = set(tcfg["loco"]["excluded_classes"]) | set(NON_LOCO_LABELS)
    out = {"headline": sorted(tcfg["headline"]["holdout"])}
    out.update({f"loco_{c}": [c] for c in labels if c not in excl})
    return out


def stage_embed(cfg: dict[str, Any]) -> None:
    """Per-holdout counts of synthetic items the trainer will drop (cos > 0.85 to held-out rows)."""
    out_dir, res_dir = Path(cfg["outputs_dir"]), Path(cfg["results_dir"])
    thr = float(cfg["embed"]["leakage_cos"])
    syn = pd.read_csv(out_dir / "synthetic_oe.csv")
    syn_emb = np.load(out_dir / "synthetic_oe_emb.npy")
    assert len(syn) == len(syn_emb), "synthetic csv / embedding order mismatch"
    ids, feats = load_frozen(cfg)
    frame = _dataset_labels(cfg).set_index("id")
    lab = np.array([frame.at[i, "label"] for i in ids])
    labels = sorted(set(lab))
    detail: dict[str, Any] = {}
    per_item: dict[str, list[float]] = {}
    for name, classes in holdout_sets(cfg, labels).items():
        held = feats[np.isin(lab, classes)]
        mx, drop = max_cos_and_drop(syn_emb, held, thr)
        detail[name] = {
            "held_out_classes": classes,
            "n_heldout_rows": len(held),
            "n_synthetic": len(syn),
            "n_dropped": int(drop.sum()),
            "n_kept": int((~drop).sum()),
            "max_cos_distribution": quantile_summary(mx),
        }
        per_item[name] = [round(float(x), 5) for x in mx]
        print(f"[embed] {name}: heldout_rows={len(held)} dropped={int(drop.sum())}")
    base = {
        "threshold": thr,
        "n_synthetic": len(syn),
        "heldout_rows_source": "all 500 dataset rows of the held-out classes (frozen e5)",
    }
    _write_json(
        out_dir / "synthetic_vs_heldout.json",
        {
            **base,
            "holdouts": detail,
            "syn_ids": syn["syn_id"].tolist(),
            "max_cos_per_item": per_item,
        },
    )
    _write_json(res_dir / "oe_leakage_filter.json", {**base, "holdouts": detail})


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> None:
    """CLI: run one Robustness and open-set training fixes data stage (or all, in order)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/phase4c_data.yaml")
    ap.add_argument("--stage", choices=STAGES, default="all")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # zh/es text on cp1252
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_loopback_url(cfg["synthetic"]["base_url"])
    stages = ALL_ORDER if args.stage == "all" else (args.stage,)
    runners = {
        "synthetic": stage_synthetic,
        "a3_translate": stage_a3_translate,
        "eval_translate": stage_eval_translate,
        "labse": stage_labse,
        "embed": stage_embed,
    }
    for st in stages:
        print(f"=== phase4c_data stage: {st} ===", flush=True)
        runners[st](cfg)


if __name__ == "__main__":
    main()
