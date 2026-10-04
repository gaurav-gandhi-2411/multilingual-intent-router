"""Label-ambiguity audit with 3 local open-weight LLM judges (blind to gold/pred).

Everything reported is "LLM consensus (local open-weight models), not human ground truth".
Confidentiality: the only network endpoint is loopback Ollama; results/ holds
ids, labels and judge choices only, while prompts/responses with text go to outputs/ (gitignored).
"""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterator, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router import data as data_mod
from intent_router import gpu_lock
from intent_router.analysis import load_oof_mean

AMBIGUOUS = "ambiguous"
INVALID = "invalid"
LABEL_NOTE = "LLM consensus (local open-weight models), not human ground truth"
LOOPBACK_PORT = 11434

# One-line neutral descriptions written from the label names and the supply-chain/logistics
# chat-routing domain only (NOT derived from dataset rows).
LABEL_DESCRIPTIONS: dict[str, str] = {
    "ai_agent_performance": "questions about how well the AI assistant itself performs or is doing",
    "appointment_manager": "scheduling, rescheduling or cancelling dock or delivery appointments",
    "chitchat": "greetings, thanks and small talk with no logistics task",
    "customer_support": "help requests, complaints or issues with the service or account",
    "document_processing": "handling shipping documents such as invoices, BOLs and PODs",
    "knowledge_base": "general how-to or policy questions answered from reference material",
    "orders": "creating, looking up or changing customer or purchase orders",
    "other": "anything that fits none of the other categories",
    "shipment_information.analytics": "aggregate shipment statistics, trends and reports",
    "shipment_information.disruptions": "delays, exceptions and problems affecting shipments",
    "shipment_information.realtime_query": "current status or location of a specific shipment",
    "yard_management": "yard operations such as trailers, parking spots and dock doors",
}


# ------------------------------------------------------------------ network guard
def validate_loopback_url(url: str) -> str:
    """Return url if its host is a loopback address (127.0.0.0/8, ::1, localhost); else raise."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http":
        raise ValueError(f"refusing non-http URL {url!r}: only loopback Ollama is allowed")
    host = parsed.hostname
    if host is None:
        raise ValueError(f"refusing URL without host: {url!r}")
    if host != "localhost":
        try:
            ok = ipaddress.ip_address(host).is_loopback
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(
                f"refusing non-loopback host {host!r}: data must not leave the machine"
            )
    return url.rstrip("/")


class _NoProxy(urllib.request.ProxyHandler):
    """Empty proxy map so system proxy settings can never redirect loopback traffic."""

    def __init__(self) -> None:
        super().__init__({})


def _http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    validate_loopback_url(url)
    opener = urllib.request.build_opener(_NoProxy())
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(  # noqa: S310 - scheme/host validated loopback above
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with opener.open(req, timeout=timeout) as resp:  # noqa: S310
        out: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    return out


# ---------------------------------------------------------------------- parsing
def parse_choice(raw: str, labels: Sequence[str]) -> dict[str, Any]:
    """Strictly parse a judge reply.

    Valid: exactly a label name (case-insensitive, surrounding whitespace trimmed), or
    "ambiguous: <label A> | <label B>" with two distinct label names. Anything else is invalid.
    Returns {valid, choice, pair} with choice a label, AMBIGUOUS or INVALID.
    """
    by_lower = {label.lower(): label for label in labels}
    text = raw.strip().lower()
    if text in by_lower:
        return {"valid": True, "choice": by_lower[text], "pair": ""}
    m = re.fullmatch(r"ambiguous\s*:\s*(\S+)\s*\|\s*(\S+)", text)
    if m and m.group(1) in by_lower and m.group(2) in by_lower and m.group(1) != m.group(2):
        pair = f"{by_lower[m.group(1)]}|{by_lower[m.group(2)]}"
        return {"valid": True, "choice": AMBIGUOUS, "pair": pair}
    return {"valid": False, "choice": INVALID, "pair": ""}


# ---------------------------------------------------------------------- prompt
def build_messages(text: str, labels: Sequence[str]) -> list[dict[str, str]]:
    """System + user messages: label names with descriptions, the text, answer format."""
    lines = "\n".join(f"- {label}: {LABEL_DESCRIPTIONS[label]}" for label in labels)
    system = (
        "You route chat messages from a supply-chain / logistics assistant to exactly one "
        "intent label. The available labels are:\n"
        f"{lines}\n\n"
        "Answer with exactly one label name and nothing else. If two labels are equally "
        'defensible, answer exactly "ambiguous: <label A> | <label B>" instead.'
    )
    user = f"Message:\n{text}\n\nAnswer:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


RETRY_REMINDER = (
    "Your answer was not in the required format. Reply with exactly one label name from the "
    'list, or "ambiguous: <label A> | <label B>", and nothing else.'
)


# ---------------------------------------------------------------------- sampling
def _largest_remainder(sizes: dict[int, int], caps: dict[int, int], total: int) -> dict[int, int]:
    """Allocate total seats proportional to sizes, capped per class, largest remainder first."""
    alloc = dict.fromkeys(sizes, 0)
    active = {k for k in sizes if caps[k] > 0}
    remaining = total
    while remaining > 0 and active:
        weight = sum(sizes[k] for k in active)
        quota = {k: remaining * sizes[k] / weight for k in active}
        base = {k: min(int(quota[k]), caps[k] - alloc[k]) for k in active}
        for k, v in base.items():
            alloc[k] += v
        remaining -= sum(base.values())
        if remaining == 0:
            break
        order = sorted(active, key=lambda k: (-(quota[k] - int(quota[k])), k))
        for k in order:
            if remaining == 0:
                break
            if alloc[k] < caps[k]:
                alloc[k] += 1
                remaining -= 1
        active = {k for k in active if alloc[k] < caps[k]}
    return alloc


def sample_items(
    oof: pd.DataFrame,
    seed: int,
    n_controls: int = 20,
    n_calibration: int = 10,
    control_conf_max: float = 0.99,
    calibration_conf_min: float = 0.99,
) -> pd.DataFrame:
    """Select audit items from probability-averaged OOF rows (id, gold, pred, conf).

    errors: every row with pred != gold.
    calibration: one correct row with conf >= calibration_conf_min from each of n_calibration
      classes (classes chosen by seeded sample among classes having such a row, then a seeded
      row per class).
    controls: correct rows not in calibration; per-class counts proportional to each class's
      number of correct rows (largest remainder, capped by availability), each class filled from
      rows with conf < control_conf_max first, then (only if short) from the rest. Seeded.
    No overlap between groups. Returns id, item_type, gold, model_pred, model_conf (ints = class
    indices, as in the OOF file); gold/pred are for scoring only, never shown to judges.
    """
    rng = random.Random(seed)  # noqa: S311 - seeded sampling, not security
    oof = oof.sort_values("id").reset_index(drop=True)
    errors = oof[oof["pred"] != oof["gold"]]
    correct = oof[oof["pred"] == oof["gold"]]

    high = correct[correct["conf"] >= calibration_conf_min]
    cal_classes_avail = sorted(high["gold"].unique())
    if len(cal_classes_avail) < n_calibration:
        raise ValueError(f"only {len(cal_classes_avail)} classes have calibration candidates")
    cal_classes = sorted(rng.sample(cal_classes_avail, n_calibration))
    cal_ids = [rng.choice(sorted(high[high["gold"] == c]["id"])) for c in cal_classes]

    pool = correct[~correct["id"].isin(cal_ids)]
    sizes = {int(c): int(n) for c, n in pool["gold"].value_counts().sort_index().items()}
    caps = dict(sizes)
    alloc = _largest_remainder(sizes, caps, n_controls)
    ctrl_ids: list[str] = []
    for c in sorted(alloc):
        k = alloc[c]
        if k == 0:
            continue
        cls = pool[pool["gold"] == c]
        low_ids = sorted(cls[cls["conf"] < control_conf_max]["id"])
        rest_ids = sorted(cls[cls["conf"] >= control_conf_max]["id"])
        take = rng.sample(low_ids, min(k, len(low_ids)))
        if len(take) < k:
            take += rng.sample(rest_ids, k - len(take))
        ctrl_ids += take

    parts = [
        (errors["id"].tolist(), "error"),
        (ctrl_ids, "control"),
        (cal_ids, "calibration"),
    ]
    rows = []
    by_id = oof.set_index("id")
    for ids, kind in parts:
        for i in sorted(ids):
            r = by_id.loc[i]
            rows.append((i, kind, int(r["gold"]), int(r["pred"]), float(r["conf"])))
    df = pd.DataFrame(rows, columns=["id", "item_type", "gold", "model_pred", "model_conf"])
    if df["id"].duplicated().any():
        raise ValueError("audit items overlap between groups")
    return df


# ---------------------------------------------------------------------- statistics
def fleiss_kappa(counts: np.ndarray) -> float | None:
    """Fleiss' kappa for an (items x categories) matrix of rater counts (same n per item).

    Returns None when chance agreement is 1 (single category, kappa undefined).
    """
    counts = np.asarray(counts, dtype=float)
    n_items = counts.shape[0]
    n_raters = counts[0].sum()
    if not np.allclose(counts.sum(axis=1), n_raters):
        raise ValueError("every item needs the same number of ratings")
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_i = ((counts**2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar, p_e = p_i.mean(), (p_j**2).sum()
    if np.isclose(p_e, 1.0):
        return None
    return float((p_bar - p_e) / (1 - p_e))


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa between two equal-length label sequences (None if undefined)."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    if np.isclose(pe, 1.0):
        return None
    return float((po - pe) / (1 - pe))


def consensus(choices: Sequence[str]) -> str | None:
    """Category chosen by >= 2 judges (invalid ignored); None if no such category."""
    cnt = Counter(c for c in choices if c != INVALID)
    if not cnt:
        return None
    top, n = cnt.most_common(1)[0]
    return top if n >= 2 else None


def _breakdown(
    group: pd.DataFrame, cons: dict[str, str | None], labels: list[str]
) -> dict[str, Any]:
    out = {"n": len(group), "agrees_with_model": 0, "agrees_with_gold": 0, "ambiguous": 0,
           "other_label": 0, "no_consensus": 0}  # fmt: skip
    for r in group.itertuples():
        c = cons[r.id]
        if c is None:
            out["no_consensus"] += 1
        elif c == AMBIGUOUS:
            out["ambiguous"] += 1
        elif c == labels[r.model_pred]:
            out["agrees_with_model"] += 1
            if r.model_pred == r.gold:
                out["agrees_with_gold"] += 1
        elif c == labels[r.gold]:
            out["agrees_with_gold"] += 1
        else:
            out["other_label"] += 1
    return out


def summarise(
    items: pd.DataFrame, judgements: pd.DataFrame, labels: list[str], gate: float
) -> dict[str, Any]:
    """
    Compute all label-ambiguity audit metrics from items and per-judge judgements (no text
    involved).
    """
    judges = sorted(judgements["judge"].unique())
    wide = judgements.pivot(index="id", columns="judge", values="choice")
    valid = judgements.pivot(index="id", columns="judge", values="valid")
    cal = items[items["item_type"] == "calibration"]
    per_judge: dict[str, Any] = {}
    reasons = []
    for j in judges:
        sub = judgements[judgements["judge"] == j]
        cal_j = sub[sub["id"].isin(cal["id"])].merge(cal[["id", "gold"]], on="id")
        acc = float((cal_j["choice"] == cal_j["gold"].map(lambda g: labels[g])).mean())
        per_judge[j] = {
            "calibration_accuracy": acc,
            "calibration_n": len(cal_j),
            "valid_output_rate": float(sub["valid"].mean()),
            "n_items": len(sub),
        }
        if acc < gate:
            reasons.append(f"{j} calibration accuracy {acc:.2f} < {gate}")
    all_valid = valid[judges].all(axis=1)
    cats = [*labels, AMBIGUOUS]
    kappa_f = None
    if all_valid.any():
        rows = wide.loc[all_valid, judges]
        mat = np.array([[sum(v == c for v in r) for c in cats] for r in rows.to_numpy()])
        kappa_f = fleiss_kappa(mat)
    pairwise = {}
    for a, b in combinations(judges, 2):
        both = valid[a] & valid[b]
        pairwise[f"{a}|{b}"] = {
            "kappa": cohen_kappa(wide.loc[both, a].tolist(), wide.loc[both, b].tolist()),
            "n": int(both.sum()),
        }
    cons = {i: consensus(list(wide.loc[i, judges])) for i in wide.index}
    by_type = {
        t: _breakdown(items[items["item_type"] == t], cons, labels) for t in ("error", "control")
    }
    return {
        "label_note": LABEL_NOTE,
        "judges": judges,
        "audit_reliable": not reasons,
        "unreliable_reasons": reasons,
        "calibration_gate": gate,
        "per_judge": per_judge,
        "fleiss_kappa": {
            "value": kappa_f,
            "n_items_all_valid": int(all_valid.sum()),
            "categories": len(cats),
        },
        "pairwise_cohen_kappa": pairwise,
        "consensus_vs_model": by_type,
    }


# ---------------------------------------------------------------------- Ollama judge
class OllamaJudge:
    """One Ollama model queried over loopback with deterministic options."""

    def __init__(self, base_url: str, tag: str, think: bool | None, cfg: dict[str, Any]) -> None:
        self.base_url = validate_loopback_url(base_url)
        self.tag, self.think, self.cfg = tag, think, cfg

    def chat(self, messages: list[dict[str, str]], keep_alive: Any = None) -> str:
        """Single non-streaming chat call at temperature 0, seed 42."""
        payload: dict[str, Any] = {
            "model": self.tag,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_ctx": self.cfg["num_ctx"],
                "num_predict": self.cfg["num_predict"],
            },
            "keep_alive": self.cfg["keep_alive"] if keep_alive is None else keep_alive,
        }
        if self.think is not None:
            payload["think"] = self.think
        out = _http_json(f"{self.base_url}/api/chat", payload, self.cfg["timeout_s"])
        return str(out["message"]["content"])

    def judge(self, text: str, labels: Sequence[str]) -> dict[str, Any]:
        """Ask once, retry once with a reminder if invalid. Returns parsed result + raw replies."""
        msgs = build_messages(text, labels)
        raw1 = self.chat(msgs)
        res = parse_choice(raw1, labels)
        raws = [raw1]
        if not res["valid"]:
            raw2 = self.chat([*msgs, {"role": "assistant", "content": raw1},
                              {"role": "user", "content": RETRY_REMINDER}])  # fmt: skip
            raws.append(raw2)
            res = parse_choice(raw2, labels)
            res["retried"] = True
        else:
            res["retried"] = False
        res["raw"] = raws
        return res

    def unload(self) -> None:
        """Free VRAM: empty chat request with keep_alive 0."""
        payload = {"model": self.tag, "messages": [], "keep_alive": 0, "stream": False}
        with contextlib.suppress(urllib.error.URLError, OSError, ValueError):
            _http_json(f"{self.base_url}/api/chat", payload, self.cfg["timeout_s"])


def record_judges(cfg: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    """Write exact tags + digests (from /api/tags and /api/show) to path."""
    base = validate_loopback_url(cfg["ollama"]["base_url"])
    tags = {m["name"]: m for m in _http_json(f"{base}/api/tags", None, 30)["models"]}
    rec = []
    for j in cfg["judges"]:
        m = tags.get(j["tag"])
        if m is None:
            raise RuntimeError(f"judge model {j['tag']} not pulled (ollama pull {j['tag']})")
        show = _http_json(f"{base}/api/show", {"model": j["tag"]}, 30)
        det = show.get("details", {})
        rec.append({
            "name": j["name"], "tag": j["tag"], "digest": m["digest"], "size_bytes": m["size"],
            "family": det.get("family"), "parameter_size": det.get("parameter_size"),
            "quantization": det.get("quantization_level"), "think": j["think"],
        })  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"judges": rec}, indent=2), encoding="utf-8")
    return rec


# ---------------------------------------------------------------------- GPU access
@contextlib.contextmanager
def _ollama_whitelisted() -> Iterator[list[dict[str, Any]]]:
    """Make gpu_lock ignore processes named *ollama* (our own server/runner); record sightings."""
    seen: list[dict[str, Any]] = []
    orig = gpu_lock.foreign_gpu_processes

    def patched() -> list[dict[str, Any]]:
        out = []
        for p in orig():
            if "ollama" in str(p.get("process_name", "")).lower():
                seen.append(p)
            else:
                out.append(p)
        return out

    gpu_lock.foreign_gpu_processes = patched
    try:
        yield seen
    finally:
        gpu_lock.foreign_gpu_processes = orig


# ---------------------------------------------------------------------- runner
def _load_items(cfg: dict[str, Any]) -> pd.DataFrame:
    oof = load_oof_mean(Path(cfg["oof_path"]), cfg["n_classes"], cfg["oof_per_id"])
    ic = cfg["items"]
    return sample_items(
        oof, cfg["seed"], ic["n_controls"], ic["n_calibration"],
        ic["control_conf_max"], ic["calibration_conf_min"],
    )  # fmt: skip


def run_judges(
    cfg: dict[str, Any],
    items: pd.DataFrame,
    texts: dict[str, str],
    labels: list[str],
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Judge every item with every judge in one seeded shuffled order. Raw text -> out_dir."""
    order = sorted(items["id"])
    random.Random(cfg["seed"]).shuffle(order)  # noqa: S311
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = dict(zip(items["id"], items["item_type"], strict=True))
    rows: list[dict[str, Any]] = []
    timing: dict[str, Any] = {}
    for j in cfg["judges"]:
        judge = OllamaJudge(cfg["ollama"]["base_url"], j["tag"], j["think"], cfg["ollama"])
        t0 = time.monotonic()
        with (out_dir / f"raw_{j['name']}.jsonl").open("w", encoding="utf-8") as f:
            for i in order:
                res = judge.judge(texts[i], labels)
                rows.append({"id": i, "item_type": kind[i], "judge": j["name"],
                             "choice": res["choice"], "ambiguous_pair": res["pair"],
                             "valid": res["valid"]})  # fmt: skip
                f.write(json.dumps({"id": i, "text": texts[i], "raw": res["raw"],
                                    "retried": res["retried"]}) + "\n")  # fmt: skip
        judge.unload()
        timing[j["name"]] = {"seconds": round(time.monotonic() - t0, 1), "n": len(order)}
    return pd.DataFrame(rows), timing


def run_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    """Full audit: items -> judges (with exclusive GPU access) -> judgements.csv + summary.json."""
    df = data_mod.load_data(cfg["data_path"])
    labels = list(data_mod.LABELS)
    texts = dict(zip(df["id"], df["text"], strict=True))
    res_dir = Path(cfg["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    items = _load_items(cfg)
    items.to_csv(res_dir / "items.csv", index=False)
    record_judges(cfg, res_dir / "judges.json")
    with (
        _ollama_whitelisted() as seen,
        gpu_lock.gpu_exclusive(
            cfg["expected_duration_s"], "python -m intent_router.llm_audit"
        ) as lock,
    ):
        before = gpu_lock.gpu_snapshot()
        judgements, timing = run_judges(cfg, items, texts, labels, Path(cfg["outputs_dir"]))
        after = gpu_lock.gpu_snapshot()
    judgements.to_csv(res_dir / "judgements.csv", index=False)
    summary = summarise(items, judgements, labels, cfg["gate"]["calibration_min_accuracy"])
    summary["timing"] = timing
    summary["gpu_lock"] = {
        "waited_s": lock["waited_s"],
        "ollama_processes_whitelisted": [
            {"pid": p.get("pid"), "process_name": p.get("process_name")} for p in seen
        ],
        "snapshot_before": before,
        "snapshot_after": after,
    }
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_smoke(cfg: dict[str, Any], n: int = 3) -> dict[str, Any]:
    """Judge n VALIDATION-split rows outside the item set; outputs only to smoke_dir."""
    from intent_router.data import get_frame

    labels_df = data_mod.load_data(cfg["data_path"])
    labels = list(data_mod.LABELS)
    items = _load_items(cfg)
    val = get_frame(["val"], cfg["data_path"], cfg["splits_path"])
    val = val[~val["id"].isin(items["id"])].sort_values("id")
    pick = val.sample(n=n, random_state=cfg["seed"])
    del labels_df
    out_dir = Path(cfg["smoke_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    with _ollama_whitelisted(), gpu_lock.gpu_exclusive(600, "llm_audit smoke"):
        for j in cfg["judges"]:
            judge = OllamaJudge(cfg["ollama"]["base_url"], j["tag"], j["think"], cfg["ollama"])
            lat, parsed, outs = [], 0, []
            for r in pick.itertuples():
                t0 = time.monotonic()
                res = judge.judge(r.text, labels)
                lat.append(round(time.monotonic() - t0, 2))
                parsed += int(res["valid"])
                outs.append({"id": r.id, "choice": res["choice"], "raw": res["raw"]})
            judge.unload()
            (out_dir / f"smoke_{j['name']}.json").write_text(json.dumps(outs, indent=1))
            report[j["name"]] = {"latency_s_per_item": lat, "valid": parsed, "n": n}
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI: `--smoke` (3 val rows), `--record-judges`, or the full audit (default)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/llm_audit.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--record-judges", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_loopback_url(cfg["ollama"]["base_url"])
    if args.record_judges:
        print(json.dumps(record_judges(cfg, Path(cfg["results_dir"]) / "judges.json"), indent=2))
    elif args.smoke:
        print(json.dumps(run_smoke(cfg), indent=2))
    else:
        print(json.dumps(run_audit(cfg), indent=2))


if __name__ == "__main__":
    main()
