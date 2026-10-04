from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router import data as data_mod
from intent_router import gpu_lock
from intent_router.cv import fold_split, load_cv_frame
from intent_router.data import get_frame
from intent_router.evaluate import (
    _f1_from_confusion,
    git_sha,
    macro_f1_gold_classes,
)
from intent_router.models import QUERY_PREFIX, predict, state_dict_sha256
from intent_router.stats import accuracy
from intent_router.train import TrainConfig, train_model

STAGES = ("fold_models", "idswap", "translate", "noise", "report", "all")
MT_LABEL = "machine-translated (synthetic)"
CALL_TYPE = "robustness_inference"  # test_eval_log.jsonl call_type for every test-split call

# --------------------------------------------------------------------------- ID swap
# Same shape as models.mask_entities' regexes: an uppercase 2-4 letter prefix, a hyphen, digits.
_ID_RE = re.compile(r"\b([A-Z]{2,4})-(\d+)")
# swap name -> (source prefix or "other", replacement prefix)
IDSWAPS: dict[str, tuple[str, str]] = {
    "PO->LD": ("PO", "LD"),
    "PO->REF": ("PO", "REF"),
    "LD->PO": ("LD", "PO"),
    "LD->REF": ("LD", "REF"),
    "other->REF": ("other", "REF"),
}


def _is_source(prefix: str, src: str) -> bool:
    # "other" = any [A-Z]{2,4}-n that is not PO/LD; REF is excluded too, since REF->REF is a
    # no-op and would count an unchanged row as swapped.
    if src == "other":
        return prefix not in ("PO", "LD", "REF")
    return prefix == src


def swap_ids(text: str, src: str, dst: str) -> tuple[str, int]:
    """Replace every `src`-prefixed ID with a `dst`-prefixed one, digits kept.

    src is a prefix ("PO", "LD") or "other" (any [A-Z]{2,4}-n except PO/LD/REF). Returns
    (new_text, number_of_ids_replaced); text with no matching ID comes back untouched.
    """
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        if _is_source(m.group(1), src):
            count += 1
            return f"{dst}-{m.group(2)}"
        return m.group(0)

    return _ID_RE.sub(repl, text), count


# --------------------------------------------------------------------------- noise
# Fixed dictionary (the pre-registered "etc." is spelled out here); applied word-boundary,
# case-insensitive, longest phrase first so "what is" wins over any single-word entry.
ABBREVIATIONS: dict[str, str] = {
    "what is": "whats",
    "please": "pls",
    "thanks": "thx",
    "thank you": "thx",
    "you": "u",
    "your": "ur",
    "hours": "hrs",
    "tomorrow": "tmrw",
    "shipment": "shpmt",
    "appointment": "appt",
    "information": "info",
    "number": "no.",
    "message": "msg",
    "because": "bc",
    "between": "btwn",
    "delivery": "dlvry",
    "schedule": "sched",
    "with": "w/",
    "without": "w/o",
}
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(k).replace(r"\ ", r"\s+") for k in sorted(
        ABBREVIATIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)  # fmt: skip
CHAR_MODES = ("swap", "drop", "insert", "mixed")


def abbreviate(text: str) -> str:
    """Substitute abbreviations (word-boundary, case-insensitive; result is lowercase)."""
    return _ABBREV_RE.sub(lambda m: ABBREVIATIONS[" ".join(m.group(1).lower().split())], text)


def row_rng(seed: int, name: str, row_id: str) -> np.random.Generator:
    """Per-(perturbation, row) generator: independent of row order and of other perturbations."""
    h = hashlib.sha256(f"{seed}|{name}|{row_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def perturb_chars(text: str, rate: float, mode: str, rng: np.random.Generator) -> tuple[str, int]:
    """Apply char-level noise to round(rate * n_alphabetic) distinct alphabetic positions.

    mode: swap (with the next alphabetic neighbour, else the previous one), drop, insert (a random
    lowercase letter after the position), or mixed (each position gets a random one of the three).
    Positions are processed right-to-left so earlier indices stay valid. Returns
    (new_text, n_positions_perturbed); the count is the *attempted* operations (a swap of two
    identical letters changes nothing but still counts).
    """
    if mode not in CHAR_MODES:
        raise ValueError(f"mode must be one of {CHAR_MODES}, got {mode!r}")
    chars = list(text)
    alpha = [i for i, c in enumerate(chars) if c.isalpha()]
    k = int(rate * len(alpha) + 0.5)
    if k == 0:
        return text, 0
    pos = sorted((int(p) for p in rng.choice(alpha, size=k, replace=False)), reverse=True)
    for i in pos:
        op = mode if mode != "mixed" else ("swap", "drop", "insert")[int(rng.integers(3))]
        if op == "drop":
            del chars[i]
        elif op == "insert":
            chars.insert(i + 1, "abcdefghijklmnopqrstuvwxyz"[int(rng.integers(26))])
        elif i + 1 < len(chars) and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif i > 0 and chars[i - 1].isalpha():
            chars[i], chars[i - 1] = chars[i - 1], chars[i]
    return "".join(chars), k


def noise_types(rates: list[float]) -> list[str]:
    """Perturbation names: char_<mode>_<pct> for every rate and mode, then lowercase, abbrev."""
    names = [f"char_{m}_{round(r * 100)}" for r in rates for m in CHAR_MODES]
    return [*names, "lowercase", "abbrev"]


def apply_noise(text: str, name: str, row_id: str, seed: int) -> str:
    """Apply the named perturbation to one text (deterministic given seed, name, row id)."""
    if name == "lowercase":
        return text.lower()
    if name == "abbrev":
        return abbreviate(text)
    m = re.fullmatch(r"char_(swap|drop|insert|mixed)_(\d+)", name)
    if m is None:
        raise ValueError(f"unknown noise perturbation {name!r}")
    return perturb_chars(text, int(m.group(2)) / 100, m.group(1), row_rng(seed, name, row_id))[0]


# --------------------------------------------------------------------------- metrics
def perf(y: np.ndarray, pred: np.ndarray, n_classes: int) -> tuple[float, float]:
    """(accuracy, macro-F1 over the gold classes present): the pre-registered convention."""
    return accuracy(y, pred), macro_f1_gold_classes(y, pred, n_classes)


def paired_drop_ci(
    y: np.ndarray,
    clean_pred: np.ndarray,
    pert_pred: np.ndarray,
    n_classes: int,
    n_resamples: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap of (clean - perturbed) accuracy and macro-F1 (same resampled rows).

    Positive drop = the perturbation hurt. Macro-F1 per resample is over the gold classes present
    in that resample, matching the point estimate's convention.
    """
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    cc = (y == np.asarray(clean_pred)).astype(float)
    pc = (y == np.asarray(pert_pred)).astype(float)
    acc_d = cc[idx].mean(axis=1) - pc[idx].mean(axis=1)
    ka = y * n_classes + np.asarray(clean_pred, dtype=np.int64)
    kb = y * n_classes + np.asarray(pert_pred, dtype=np.int64)
    f1_d = np.empty(n_resamples)
    for b in range(n_resamples):
        ib = idx[b]
        gold = np.unique(y[ib])
        ca = np.bincount(ka[ib], minlength=n_classes**2).reshape(n_classes, n_classes)
        cb = np.bincount(kb[ib], minlength=n_classes**2).reshape(n_classes, n_classes)
        f1_d[b] = _f1_from_confusion(ca, False, gold) - _f1_from_confusion(cb, False, gold)
    a = (1 - level) / 2 * 100
    acc0, f10 = perf(y, clean_pred, n_classes)
    acc1, f11 = perf(y, pert_pred, n_classes)
    return {
        "accuracy_drop": {
            "point": acc0 - acc1,
            "lo": float(np.percentile(acc_d, a)),
            "hi": float(np.percentile(acc_d, 100 - a)),
        },
        "macro_f1_drop": {
            "point": f10 - f11,
            "lo": float(np.percentile(f1_d, a)),
            "hi": float(np.percentile(f1_d, 100 - a)),
        },
    }


# --------------------------------------------------------------------------- plumbing
class Paths:
    """Output locations; smoke runs are redirected under outputs/robustness_smoke/."""

    def __init__(self, cfg: dict[str, Any], smoke: bool) -> None:
        self.smoke = smoke
        if smoke:
            self.out = Path("outputs/robustness_smoke")
            self.results = self.out / "results"
            self.figures = self.out / "figures"
        else:
            self.out = Path(cfg["outputs_dir"])
            self.results = Path(cfg["results_dir"])
            self.figures = Path(cfg["figures_dir"])
        self.fold_models = self.out / "fold_models"


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: Path, obj: Any) -> None:
    """Indented JSON; numpy scalars/arrays converted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def load_configs(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(robustness cfg, final cfg); the final cfg's query prefix is checked against models."""
    rcfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    fcfg = yaml.safe_load(Path(rcfg["final_config"]).read_text(encoding="utf-8"))
    tr = fcfg["train"]
    assert QUERY_PREFIX.get(tr["model_name"], "") == tr["query_prefix"], "query prefix drift"
    return rcfg, fcfg


def load_cohort(rcfg: dict[str, Any], fcfg: dict[str, Any], smoke: bool) -> pd.DataFrame:
    """Rows to perturb: id,text,label,gold,lang,split,source('test'|'oof'),fold(-1 for test).

    Normal runs: every test row (final model) and every train+val row (OOF fold model of its
    fold-seed-s0 fold). Smoke runs: val rows of fold 0 only -- the test split is never loaded.
    """
    frame = get_frame(["train", "val", "test"], fcfg["data_path"], fcfg["splits_path"])
    eda = pd.read_csv(fcfg["analysis"]["eda_rows_path"])[["id", "lang"]]
    frame = frame.merge(eda, on="id", how="left", validate="one_to_one")
    assert frame["lang"].notna().all(), "eda_rows.csv does not cover every id"
    frame["gold"] = frame["label"].map(data_mod.label2id())
    col = f"cv_fold_s{rcfg['fold_models']['fold_seed_idx']}"
    frame["fold"] = frame[col]
    frame["source"] = np.where(frame["split"] == "test", "test", "oof")
    if smoke:
        frame = frame[(frame["split"] == "val") & (frame["fold"] == 0)]
    # optional restriction (Multi-axis selection: fold models were deleted)
    elif rcfg.get("sources"):
        frame = frame[frame["source"].isin(list(rcfg["sources"]))]
    keep = ["id", "text", "label", "gold", "lang", "split", "source", "fold"]
    return frame[keep].sort_values(["source", "id"]).reset_index(drop=True)


@contextlib.contextmanager
def wandb_disabled() -> Iterator[None]:
    """train_model opens a W&B run per call; fold models must not (they are not the story)."""
    old = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "disabled"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("WANDB_MODE", None)
        else:
            os.environ["WANDB_MODE"] = old


def append_test_log(
    path: Path, fingerprint: str, perturbation: str, stage: str, n_rows: int
) -> None:
    """One jsonl line per perturbation batch that reads test rows (before inference runs).

    Written here rather than via evaluate.evaluate_test_once: that door only accepts
    CALL_TYPES = (evaluation, determinism_inference) and these calls compute no reported metric.
    """
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_fingerprint": fingerprint,
        "call_type": CALL_TYPE,
        "git_sha": git_sha(),
        "stage": stage,
        "perturbation": perturbation,
        "n_rows": n_rows,
        "smoke": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _load_classifier(path: Path, device: Any) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(path), dtype=torch.float32, attn_implementation="eager"
    )
    return tok, model.to(device).eval()


class ModelBank:
    """Final model (test rows) and fold models (OOF rows), loaded lazily and kept on device."""

    def __init__(self, rcfg: dict[str, Any], fcfg: dict[str, Any], P: Paths, stage: str) -> None:
        import torch

        self.rcfg, self.fcfg, self.P, self.stage = rcfg, fcfg, P, stage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = fcfg["train"]["model_name"]
        self.max_len = int(fcfg["train"]["max_len"])
        self.bs = int(rcfg["predict_batch_size"])
        self.test_log = Path(fcfg["test_eval_log"])
        self._final: tuple[Any, Any] | None = None
        self.final_fingerprint: str | None = None
        self._folds: dict[int, tuple[Any, Any]] = {}
        self._clean: np.ndarray | None = None

    def _final_model(self) -> tuple[Any, Any]:
        if self._final is None:
            mdir = Path(self.fcfg["model_dir"])
            cfg = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
            assert cfg.get("query_prefix", "") == QUERY_PREFIX.get(self.name, ""), "prefix drift"
            assert int(cfg.get("max_length", self.max_len)) == self.max_len, "max_len drift"
            tok, model = _load_classifier(mdir, self.device)
            self.final_fingerprint = state_dict_sha256(model)
            expected = json.loads(
                (Path(self.fcfg["results_dir"]) / "track_a.json").read_text(encoding="utf-8")
            )["model_fingerprint"]
            assert self.final_fingerprint == expected, "outputs/final_model is not the final model"
            self._final = (tok, model)
        return self._final

    def _fold_model(self, fold: int) -> tuple[Any, Any]:
        if fold not in self._folds:
            self._folds[fold] = _load_classifier(self.P.fold_models / f"fold{fold}", self.device)
        return self._folds[fold]

    def predict(self, df: pd.DataFrame, texts: list[str], perturbation: str) -> np.ndarray:
        """Predicted class index per row (test rows -> final model, OOF rows -> its fold model).

        A call that reads any test row appends one `robustness_inference` line to the test log.
        """
        pred = np.full(len(df), -1, dtype=np.int64)
        src = df["source"].to_numpy()
        fold = df["fold"].to_numpy()
        is_test = src == "test"
        groups: list[tuple[Any, np.ndarray]] = []
        if is_test.any():
            groups.append(("final", np.flatnonzero(is_test)))
        for f in sorted(set(fold[~is_test].tolist())):
            groups.append((int(f), np.flatnonzero(~is_test & (fold == f))))
        for key, idx in groups:
            if key == "final":
                tok, model = self._final_model()
                assert self.final_fingerprint is not None
                if not self.P.smoke:
                    append_test_log(
                        self.test_log, self.final_fingerprint, perturbation, self.stage, len(idx)
                    )
            else:
                tok, model = self._fold_model(int(key))
            logits, _ = predict(
                model, tok, [texts[i] for i in idx], self.name, self.max_len, self.bs
            )
            pred[idx] = logits.argmax(axis=1)
        assert (pred >= 0).all()
        return pred

    def clean(self, df: pd.DataFrame) -> np.ndarray:
        """Clean predictions for df's rows; test rows must equal results/final/test_predictions."""
        pred = self.predict(df, df["text"].tolist(), "clean")
        test = df["source"] == "test"
        if test.any():
            ref = pd.read_csv(Path(self.fcfg["results_dir"]) / "test_predictions.csv")
            ref_pred = ref.set_index("id")["pred"]
            got = pd.Series(pred[test.to_numpy()], index=df.loc[test, "id"].to_numpy())
            assert (got.to_numpy() == ref_pred.loc[got.index].to_numpy()).all(), (
                "clean test predictions differ from results/final/test_predictions.csv"
            )
        return pred

    def meta(self) -> dict[str, Any]:
        """Provenance recorded in every stage json."""
        return {
            "git_sha": git_sha(),
            "final_model_fingerprint": self.final_fingerprint,
            "fold_models_dir": str(self.P.fold_models),
            "smoke": self.P.smoke,
        }

    def close(self) -> None:
        import torch

        self._final, self._folds = None, {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@contextlib.contextmanager
def locked_bank(
    rcfg: dict[str, Any], fcfg: dict[str, Any], P: Paths, stage: str
) -> Iterator[ModelBank]:
    """Exclusive GPU access for the whole block; models freed before it is released."""
    with gpu_lock.gpu_exclusive(
        int(rcfg["gpu_lock_expected_s"]), f"intent-router robustness {stage}"
    ):
        bank = ModelBank(rcfg, fcfg, P, stage)
        try:
            yield bank
        finally:
            bank.close()


def _label_names() -> list[str]:
    return list(data_mod.LABELS)


# ------------------------------------------------------------------- stage: fold models
def stage_fold_models(rcfg: dict[str, Any], fcfg: dict[str, Any], P: Paths) -> None:
    """Train (or reuse cached) fold-seed-s0 fold models, then save their clean OOF predictions."""
    fm = rcfg["fold_models"]
    cohort = load_cohort(rcfg, fcfg, P.smoke)
    frame = load_cv_frame(fcfg["data_path"], fcfg["splits_path"])
    folds = [0] if P.smoke else list(range(int(fm["n_folds"])))
    stop = 1 if P.smoke else int(fcfg["train"]["stop_epoch"])
    tcfg = TrainConfig.from_dict(
        {
            **fcfg["train"],
            "model_seed": int(fm["model_seed"]),
            "attn_implementation": fm["attn_implementation"],
            "stop_epoch": stop,
        }
    )
    runs: list[dict[str, Any]] = []
    for k in folds:
        fdir = P.fold_models / f"fold{k}"
        meta_path = fdir / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            assert meta["stop_epoch"] == stop and meta["model_seed"] == tcfg.model_seed, (
                f"cached fold {k} was trained with a different config; delete {fdir} to retrain"
            )
            print(f"[fold_models] fold {k}: cached")
            runs.append(meta)
            continue
        tr, ev = fold_split(frame, int(fm["fold_seed_idx"]), k)
        run_id = f"robust_fs{fm['fold_seed_idx']}_ms{tcfg.model_seed}_f{k}"
        print(f"[fold_models] fold {k}: training on {len(tr)}, held-out {len(ev)}")
        with (
            wandb_disabled(),
            gpu_lock.gpu_exclusive(
                int(fm["expected_run_s_per_fold"]), f"intent-router robustness {run_id}"
            ) as lock_info,
        ):
            res, model, tok = train_model(tcfg, tr, ev, {"run_id": run_id, "fold": k})
            try:
                fdir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(fdir)
                tok.save_pretrained(fdir)
                fingerprint = state_dict_sha256(model)
            finally:
                model = None
                gc.collect()
        meta = {
            "fold": k,
            "run_id": run_id,
            "n_train": len(tr),
            "n_eval": len(ev),
            "stop_epoch": stop,
            "model_seed": tcfg.model_seed,
            "attn_implementation": tcfg.attn_implementation,
            "fingerprint": fingerprint,
            "final_epoch_eval": res.epochs[-1],
            "wall_clock_s": res.wall_clock_s,
            "gpu_exclusive": res.gpu_exclusive,
            "lock_waited_s": lock_info.get("waited_s"),
            "git_sha": git_sha(),
        }
        write_json(meta_path, meta)
        runs.append(meta)
    oof = cohort[(cohort["source"] == "oof") & cohort["fold"].isin(folds)].reset_index(drop=True)
    with locked_bank(rcfg, fcfg, P, "fold_models") as bank:
        pred = bank.clean(oof)
    labels = _label_names()
    out = pd.DataFrame(
        {
            "id": oof["id"],
            "fold": oof["fold"],
            "split": oof["split"],
            "gold": oof["gold"],
            "pred": pred,
            "gold_label": oof["label"],
            "pred_label": [labels[i] for i in pred],
            "correct": pred == oof["gold"].to_numpy(),
        }
    )
    P.results.mkdir(parents=True, exist_ok=True)
    out.to_csv(P.results / "oof_clean_predictions.csv", index=False)
    acc, f1 = perf(oof["gold"].to_numpy(), pred, len(labels))
    write_json(
        P.results / "fold_models.json",
        {
            "n_oof_rows": len(oof),
            "oof_accuracy": acc,
            "oof_macro_f1_gold_classes": f1,
            "folds": runs,
            "note": "fold-seed s0, model seed 0, eager; scores each fold's held-out rows",
            "provenance": {"git_sha": git_sha(), "smoke": P.smoke},
        },
    )
    print(f"[fold_models] OOF n={len(oof)} accuracy={acc:.4f} macro_f1={f1:.4f}")


# ----------------------------------------------------------------------- stage: idswap
def stage_idswap(rcfg: dict[str, Any], fcfg: dict[str, Any], P: Paths) -> None:
    """ID-prefix swaps on rows containing the source ID; flip rate vs the same model's clean."""
    cohort = load_cohort(rcfg, fcfg, P.smoke)
    labels = _label_names()
    results: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    with locked_bank(rcfg, fcfg, P, "idswap") as bank:
        clean = bank.clean(cohort)
        cohort = cohort.assign(clean_pred=clean)
        for source in ("test", "oof"):
            sub_all = cohort[cohort["source"] == source]
            if sub_all.empty:
                continue
            for name, (src, dst) in IDSWAPS.items():
                swapped = [swap_ids(t, src, dst) for t in sub_all["text"]]
                keep = np.array([c > 0 for _, c in swapped])
                sub = sub_all[keep].reset_index(drop=True)
                rec: dict[str, Any] = {"source": source, "swap": name, "n": int(keep.sum())}
                if rec["n"] == 0:
                    results.append(rec)
                    continue
                texts = [t for (t, c) in swapped if c > 0]
                pred = bank.predict(sub, texts, f"idswap:{name}")
                cp = sub["clean_pred"].to_numpy()
                y = sub["gold"].to_numpy()
                flipped = pred != cp
                rec.update(
                    {
                        "n_flips": int(flipped.sum()),
                        "flip_rate": float(flipped.mean()),
                        "accuracy_clean": accuracy(y, cp),
                        "accuracy_swapped": accuracy(y, pred),
                        "flipped_ids": sub.loc[flipped, "id"].tolist(),
                    }
                )
                results.append(rec)
                for i in np.flatnonzero(flipped):
                    flips.append(
                        {
                            "source": source,
                            "swap": name,
                            "id": sub.at[i, "id"],
                            "gold_label": sub.at[i, "label"],
                            "clean_pred": labels[cp[i]],
                            "swapped_pred": labels[pred[i]],
                        }
                    )
        meta = bank.meta()
    write_json(
        P.results / "idswap.json",
        {
            "results": results,
            "note": "rows containing the source ID; flip = prediction differs from the same "
            "model's clean prediction; no texts stored",
            "provenance": meta,
        },
    )
    pd.DataFrame(
        flips, columns=["source", "swap", "id", "gold_label", "clean_pred", "swapped_pred"]
    ).to_csv(P.results / "idswap_flips.csv", index=False)
    for r in results:
        print(f"[idswap] {r['source']:4s} {r['swap']:10s} n={r['n']:3d} flips={r.get('n_flips')}")


# ------------------------------------------------------------------------ stage: noise
def stage_noise(rcfg: dict[str, Any], fcfg: dict[str, Any], P: Paths) -> None:
    """Seeded noise/lowercase/abbreviation perturbations; paired-bootstrap accuracy/F1 drops."""
    cohort = load_cohort(rcfg, fcfg, P.smoke)
    labels = _label_names()
    n_cls = len(labels)
    nz = rcfg["noise"]
    names = noise_types([float(r) for r in nz["rates"]])
    bs = nz["bootstrap"]
    n_boot = 200 if P.smoke else int(bs["n_resamples"])
    results: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    with locked_bank(rcfg, fcfg, P, "noise") as bank:
        clean = bank.clean(cohort)
        cohort = cohort.assign(clean_pred=clean)
        for source in ("test", "oof"):
            sub = cohort[cohort["source"] == source].reset_index(drop=True)
            if sub.empty:
                continue
            y = sub["gold"].to_numpy()
            cp = sub["clean_pred"].to_numpy()
            for name in names:
                texts = [
                    apply_noise(t, name, i, int(nz["seed"]))
                    for t, i in zip(sub["text"], sub["id"], strict=True)
                ]
                pred = bank.predict(sub, texts, f"noise:{name}")
                changed = np.array([a != b for a, b in zip(texts, sub["text"], strict=True)])
                acc_c, f1_c = perf(y, cp, n_cls)
                acc_p, f1_p = perf(y, pred, n_cls)
                ci = paired_drop_ci(y, cp, pred, n_cls, n_boot, int(bs["seed"]))
                results.append(
                    {
                        "source": source,
                        "perturbation": name,
                        "n": len(sub),
                        "frac_text_changed": float(changed.mean()),
                        "accuracy_clean": acc_c,
                        "accuracy_perturbed": acc_p,
                        "macro_f1_clean": f1_c,
                        "macro_f1_perturbed": f1_p,
                        "flip_rate": float((pred != cp).mean()),
                        **ci,
                    }
                )
                pred_rows.append(
                    pd.DataFrame(
                        {
                            "id": sub["id"],
                            "source": source,
                            "perturbation": name,
                            "gold": y,
                            "clean_pred": cp,
                            "pred": pred,
                            "text_changed": changed,
                        }
                    )
                )
        meta = bank.meta()
    write_json(
        P.results / "noise.json",
        {
            "results": results,
            "bootstrap": {"n_resamples": n_boot, "seed": int(bs["seed"]), "paired": True},
            "seed": int(nz["seed"]),
            "abbreviations": ABBREVIATIONS,
            "provenance": meta,
        },
    )
    pd.concat(pred_rows).to_csv(P.results / "noise_predictions.csv", index=False)
    for r in results:
        print(
            f"[noise] {r['source']:4s} {r['perturbation']:16s} acc {r['accuracy_clean']:.3f}"
            f"->{r['accuracy_perturbed']:.3f} drop "
            f"{r['accuracy_drop']['point']:+.3f} [{r['accuracy_drop']['lo']:+.3f},"
            f"{r['accuracy_drop']['hi']:+.3f}] flips {r['flip_rate']:.3f}"
        )


# -------------------------------------------------------------------- stage: translate
def translate_texts(
    texts: list[str], tgt_code: str, tr: dict[str, Any], tok: Any, model: Any
) -> list[str]:
    """Greedy NLLB translation (length-sorted batches, original order restored)."""
    import torch

    bos = tok.convert_tokens_to_ids(tgt_code)
    assert bos != tok.unk_token_id, f"{tgt_code} is not an NLLB language token"
    order = np.argsort([-len(t) for t in texts], kind="stable")
    out = [""] * len(texts)
    bsz = int(tr["batch_size"])
    for s in range(0, len(order), bsz):
        idx = order[s : s + bsz]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=int(tr["max_new_tokens"])).to(model.device)  # fmt: skip
        with torch.no_grad():
            gen = model.generate(
                **enc,
                forced_bos_token_id=bos,
                max_new_tokens=int(tr["max_new_tokens"]),
                num_beams=int(tr["num_beams"]),
                do_sample=False,
            )
        for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True), strict=True):
            out[int(i)] = t
    return out


def stage_translate(rcfg: dict[str, Any], fcfg: dict[str, Any], P: Paths) -> None:
    """Machine-translate English rows to es/fr/de/zh; agreement with the English prediction."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tr = rcfg["translation"]
    cohort = load_cohort(rcfg, fcfg, P.smoke)
    en = cohort[cohort["lang"] == "en"]
    if not tr["include_oof"]:
        en = en[en["source"] == "test"]
    if P.smoke:  # 3 English VAL rows only
        en = en.head(3)
    en = en.reset_index(drop=True)
    labels = _label_names()
    n_cls = len(labels)
    cache = P.out / "translations.csv"  # texts live here only (gitignored)
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(
        columns=["id", "lang", "text_mt"])  # fmt: skip
    timings: dict[str, Any] = {}
    if any(not set(en["id"]) <= set(have.loc[have["lang"] == lg, "id"]) for lg in tr["targets"]):
        # Fetch weights BEFORE taking exclusive GPU access: a slow download must not hold the GPU.
        from huggingface_hub import snapshot_download

        t_dl = time.perf_counter()
        snapshot_download(tr["model"], allow_patterns=["*.json", "*.model", "pytorch_model.bin"])
        timings["download_s"] = time.perf_counter() - t_dl
    with gpu_lock.gpu_exclusive(int(rcfg["gpu_lock_expected_s"]), "intent-router robustness nllb"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = model = None
        new_parts: list[pd.DataFrame] = []
        try:
            for lang, code in tr["targets"].items():
                done = set(have.loc[have["lang"] == lang, "id"])
                todo = en[~en["id"].isin(done)]
                if todo.empty:
                    continue
                if model is None:
                    t_load = time.perf_counter()
                    tok = AutoTokenizer.from_pretrained(tr["model"], src_lang=tr["src_lang"])
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        tr["model"], dtype=torch.float16 if tr["fp16"] else torch.float32
                    ).to(device).eval()  # fmt: skip
                    timings["load_s"] = time.perf_counter() - t_load
                t0 = time.perf_counter()
                mt = translate_texts(todo["text"].tolist(), code, tr, tok, model)
                timings[lang] = {
                    "n_rows": len(todo),
                    "seconds": time.perf_counter() - t0,
                    "rows_per_s": len(todo) / (time.perf_counter() - t0),
                }
                new_parts.append(pd.DataFrame({"id": todo["id"], "lang": lang, "text_mt": mt}))
                print(f"[translate] {lang}: {len(todo)} rows in {timings[lang]['seconds']:.1f}s")
        finally:
            model = tok = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if new_parts:
        have = pd.concat([have, *new_parts], ignore_index=True)
        P.out.mkdir(parents=True, exist_ok=True)
        have.to_csv(cache, index=False)
    mt_text = {(r.id, r.lang): r.text_mt for r in have.itertuples(index=False)}

    results: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    with locked_bank(rcfg, fcfg, P, "translate") as bank:
        clean = bank.clean(en)
        en = en.assign(clean_pred=clean)
        for source in ("test", "oof"):
            sub = en[en["source"] == source].reset_index(drop=True)
            if sub.empty:
                continue
            y, cp = sub["gold"].to_numpy(), sub["clean_pred"].to_numpy()
            acc_en, f1_en = perf(y, cp, n_cls)
            for lang in tr["targets"]:
                texts = [mt_text[(i, lang)] for i in sub["id"]]
                pred = bank.predict(sub, texts, f"translate:{lang}")
                acc, f1 = perf(y, pred, n_cls)
                results.append(
                    {
                        "source": source,
                        "lang": lang,
                        "data_label": MT_LABEL,
                        "n": len(sub),
                        "agreement_with_english_pred": float((pred == cp).mean()),
                        "accuracy": acc,
                        "macro_f1_gold_classes": f1,
                        "english_accuracy_same_rows": acc_en,
                        "english_macro_f1_same_rows": f1_en,
                    }
                )
                pred_rows.append(
                    pd.DataFrame(
                        {
                            "id": sub["id"],
                            "source": source,
                            "lang": lang,
                            "data_label": MT_LABEL,
                            "gold": y,
                            "english_pred": cp,
                            "pred": pred,
                        }
                    )  # fmt: skip
                )
        meta = bank.meta()
    write_json(
        P.results / "translate.json",
        {
            "results": results,
            "data_label": MT_LABEL,
            "translator": tr["model"],
            "decoding": {"num_beams": tr["num_beams"], "max_new_tokens": tr["max_new_tokens"]},
            "include_oof": bool(tr["include_oof"]),
            "timings": timings,
            "provenance": meta,
        },
    )
    pd.concat(pred_rows).to_csv(P.results / "translate_predictions.csv", index=False)
    for r in results:
        print(
            f"[translate] {r['source']:4s} {r['lang']} n={r['n']} agree="
            f"{r['agreement_with_english_pred']:.3f} acc={r['accuracy']:.3f} "
            f"(en acc {r['english_accuracy_same_rows']:.3f})"
        )
    # Spot check: stdout only, never written to results/ (translated text is not publishable).
    rng = np.random.default_rng(42)
    for lang in tr["targets"]:
        pick = rng.choice(len(en), size=min(int(tr["spot_check_per_lang"]), len(en)), replace=False)
        for i in sorted(pick):
            print(f"[spot-check {lang}] EN: {en.at[i, 'text']}\n[spot-check {lang}] MT: "
                  f"{mt_text[(en.at[i, 'id'], lang)]}")  # fmt: skip


# ------------------------------------------------------------------------ stage: report
def _plots(summary: dict[str, Any], P: Paths) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P.figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    noise = summary.get("noise", {}).get("results")
    if noise:
        df = pd.DataFrame(noise)
        names = list(dict.fromkeys(df["perturbation"]))
        fig, ax = plt.subplots(figsize=(9, 4))
        w = 0.4
        for j, src in enumerate(("test", "oof")):
            d = (
                df[df["source"] == src]
                .set_index("perturbation")
                .reindex(names)
                .dropna(subset=["n"])
            )
            if d.empty:
                continue
            pt = d["accuracy_drop"].map(lambda c: c["point"])
            lo = pt - d["accuracy_drop"].map(lambda c: c["lo"])
            hi = d["accuracy_drop"].map(lambda c: c["hi"]) - pt
            x = np.arange(len(names))[[names.index(n) for n in d.index]] + (j - 0.5) * w
            ax.bar(x, pt, w, yerr=[lo, hi], capsize=2, label=f"{src} (n={int(d['n'].iloc[0])})")
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
        ax.set_ylabel("accuracy drop vs clean (95% paired bootstrap CI)")
        ax.axhline(0, color="k", lw=0.5)
        ax.legend()
        ax.set_title("Noise robustness")
        fig.tight_layout()
        paths.append(P.figures / "robustness_noise.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)

    tl = summary.get("translate", {}).get("results")
    if tl:
        df = pd.DataFrame(tl)
        fig, ax = plt.subplots(figsize=(7, 4))
        langs = list(dict.fromkeys(df["lang"]))
        for j, src in enumerate(("test", "oof")):
            d = df[df["source"] == src].set_index("lang").reindex(langs).dropna(subset=["n"])
            if d.empty:
                continue
            x = np.arange(len(langs)) + (j - 0.5) * 0.4
            ax.bar(x, d["agreement_with_english_pred"], 0.2, label=f"{src} agreement")
            ax.bar(x + 0.2, d["accuracy"], 0.2, label=f"{src} accuracy")
        ax.set_xticks(range(len(langs)), langs)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Translation invariance ({MT_LABEL})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        paths.append(P.figures / "robustness_translation.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)

    sw = summary.get("idswap", {}).get("results")
    if sw:
        df = pd.DataFrame(sw)
        df = df[df["n"] > 0]
        fig, ax = plt.subplots(figsize=(7, 4))
        for j, src in enumerate(("test", "oof")):
            d = df[df["source"] == src]
            if d.empty:
                continue
            x = np.arange(len(d)) + (j - 0.5) * 0.4
            ax.bar(x, d["flip_rate"], 0.4, label=src)
            for xi, (n, f) in zip(x, zip(d["n"], d["n_flips"], strict=True), strict=True):
                ax.text(xi, 0.0, f"{f}/{n}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(len(d)), d["swap"])
        ax.set_ylabel("prediction flip rate")
        ax.set_title("ID-prefix swap")
        ax.legend()
        fig.tight_layout()
        paths.append(P.figures / "robustness_idswap.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    return paths


def _wandb_log(rcfg: dict[str, Any], summary: dict[str, Any], figs: list[Path]) -> None:
    """Summary metrics only (no texts); any W&B failure is reported, never fatal."""
    try:
        import wandb

        run = wandb.init(
            project=rcfg["wandb"]["project"],
            group=rcfg["wandb"]["group"],
            name="robustness-summary",
            job_type="robustness",
        )
        flat: dict[str, Any] = {}
        for r in summary.get("noise", {}).get("results", []):
            k = f"noise/{r['source']}/{r['perturbation']}"
            flat[f"{k}/accuracy_drop"] = r["accuracy_drop"]["point"]
            flat[f"{k}/flip_rate"] = r["flip_rate"]
        for r in summary.get("translate", {}).get("results", []):
            k = f"translate/{r['source']}/{r['lang']}"
            flat[f"{k}/agreement"] = r["agreement_with_english_pred"]
            flat[f"{k}/accuracy"] = r["accuracy"]
        for r in summary.get("idswap", {}).get("results", []):
            if r["n"]:
                flat[f"idswap/{r['source']}/{r['swap']}/flip_rate"] = r["flip_rate"]
        run.summary.update(flat)
        for p in figs:
            run.log({p.stem: wandb.Image(str(p))})
        run.finish()
    except Exception as exc:  # noqa: BLE001 - tracking must never fail the report
        print(f"[wandb] skipped: {exc!r}")


def stage_report(rcfg: dict[str, Any], P: Paths, use_wandb: bool) -> None:
    """Assemble summary.json from the stage jsons, draw figures, log to W&B."""
    summary: dict[str, Any] = {"git_sha": git_sha(), "smoke": P.smoke}
    for name in ("fold_models", "idswap", "translate", "noise"):
        path = P.results / f"{name}.json"
        summary[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if summary[name] is None:
            print(f"[report] missing {path}")
    write_json(P.results / "summary.json", summary)
    figs = _plots({k: v for k, v in summary.items() if isinstance(v, dict)}, P)
    print(f"[report] wrote summary.json and {len(figs)} figures")
    if use_wandb:
        _wandb_log(rcfg, summary, figs)


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Robustness evaluation")
    ap.add_argument("--config", default="configs/robustness.yaml")
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--smoke", action="store_true", help="fold 0, 1 epoch, val rows, no test")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    rcfg, fcfg = load_configs(args.config)
    P = Paths(rcfg, args.smoke)
    use_wandb = bool(rcfg["wandb"]["enabled"]) and not args.no_wandb and not args.smoke
    if not use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    stages = STAGES[:-2] if args.stage == "all" else (args.stage,)
    for st in stages:
        if st == "fold_models":
            stage_fold_models(rcfg, fcfg, P)
        elif st == "idswap":
            stage_idswap(rcfg, fcfg, P)
        elif st == "translate":
            stage_translate(rcfg, fcfg, P)
        elif st == "noise":
            stage_noise(rcfg, fcfg, P)
    if args.stage in ("report", "all"):
        stage_report(rcfg, P, use_wandb)


if __name__ == "__main__":
    main()
