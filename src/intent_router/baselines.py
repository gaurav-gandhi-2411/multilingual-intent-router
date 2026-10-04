from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router import gpu_lock
from intent_router.cv import fold_split, load_cv_frame, resolve_max_len
from intent_router.data import label2id
from intent_router.models import prepare_text
from intent_router.stats import accuracy, macro_f1


def _macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Module-level (picklable for joblib workers) macro-F1 over all classes."""
    return macro_f1(y_true, y_pred, n_classes)


def fit_predict_proba(
    kind: str,
    train_x: Any,
    y_train: np.ndarray,
    groups: np.ndarray,
    eval_x: Any,
    c_grid: list[float],
    inner_folds: int,
    n_classes: int,
) -> tuple[np.ndarray, float]:
    """Fit B0 (tfidf pipeline on raw text) or B1 (LR on embeddings); return (probs, best C)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    from sklearn.pipeline import Pipeline

    lr = LogisticRegression(class_weight="balanced", max_iter=5000)
    est: Any
    if kind == "B0":
        est = Pipeline(
            [("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))), ("lr", lr)]
        )
        grid = {"lr__C": c_grid}
    else:
        est = lr
        grid = {"C": c_grid}
    scorer = make_scorer(_macro_f1_score, n_classes=n_classes)
    cv = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=42)
    # B1: loky workers fail to unpickle once torch/CUDA is loaded in the parent (observed
    # BrokenProcessPool); the 768-d LR grid is cheap, so run it serially.
    n_jobs = -1 if kind == "B0" else 1
    gs = GridSearchCV(est, grid, scoring=scorer, cv=cv, n_jobs=n_jobs, refit=True)
    gs.fit(train_x, y_train, groups=groups)
    cls = gs.best_estimator_.classes_
    p = gs.best_estimator_.predict_proba(eval_x)
    full = np.zeros((p.shape[0], n_classes), dtype=np.float32)
    full[:, cls] = p
    return full, float(next(iter(gs.best_params_.values())))


def embed_texts(model: Any, tok: Any, texts: list[str], max_len: int, dev: Any) -> np.ndarray:
    """Frozen mean-pooled (attention-masked), L2-normalised embeddings from a loaded encoder."""
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, len(texts), 64):
            b = tok(
                texts[s : s + 64],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(dev)
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def embed_e5(texts: list[str], model_name: str, max_len: int) -> np.ndarray:
    """Frozen mean-pooled (attention-masked), L2-normalised embeddings."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()
    emb = embed_texts(model, tok, texts, max_len, dev)
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return emb


def _wandb_summary(name: str, group: str, project: str, summary: dict[str, Any]) -> str | None:
    """Log summary metrics only; never raise."""
    try:
        import wandb

        run = wandb.init(
            project=project, group=group, name=name, config=summary["config"], reinit=True
        )
        run.summary.update({k: v for k, v in summary.items() if k != "config"})
        url = run.url
        run.finish()
        return url
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] baseline logging failed: {exc!r}")
        return None


def run_baseline(
    kind: str, cfg: dict[str, Any], frame: pd.DataFrame, oof_dir: Path
) -> dict[str, Any]:
    """Evaluate B0/B1 over all fold seeds x folds; write OOF csv; return result dict."""
    bc = cfg["baselines"]
    l2i = label2id()
    n_classes = len(l2i)
    y_all = frame["label"].map(l2i).to_numpy()
    gpu: dict[str, Any] = {}
    if kind == "B1":
        # Only the embedding pass uses the GPU (the LR grid is CPU); lock just that.
        with gpu_lock.gpu_exclusive(
            int(bc.get("expected_embed_s", 300)), "intent-router B1 embeddings"
        ) as lock_info:
            snap_start = gpu_lock.gpu_snapshot()
            emb = embed_e5(
                [prepare_text(t, bc["e5_model"]) for t in frame["text"]],
                bc["e5_model"],
                resolve_max_len(cfg),
            )
            snap_end = gpu_lock.gpu_snapshot()
        foreign = [*snap_start["foreign"], *snap_end["foreign"]]
        gpu = {
            "gpu_snapshot_start": snap_start,
            "gpu_snapshot_end": snap_end,
            "gpu_exclusive": not foreign,
            "gpu_foreign_seen": foreign,
            "gpu_lock_waited_s": lock_info["waited_s"],
        }
    per_fold: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for fs in bc["fold_seed_idx"]:
        for fold in range(cfg["n_folds"]):
            tr, ev = fold_split(frame, fs, fold)
            id_index = frame.set_index("id").index
            ev_i = id_index.get_indexer(ev["id"])
            tr_i = id_index.get_indexer(tr["id"])
            if kind == "B0":
                xtr, xev = tr["text"].to_numpy(), ev["text"].to_numpy()
            else:
                xtr, xev = emb[tr_i], emb[ev_i]
            probs, best_c = fit_predict_proba(
                kind,
                xtr,
                y_all[tr_i],
                tr["dup_group"].to_numpy(),
                xev,
                bc["c_grid"],
                bc["inner_folds"],
                n_classes,
            )
            pred = probs.argmax(1)
            gold = y_all[ev_i]
            per_fold.append(
                {
                    "fold_seed_idx": fs,
                    "fold": fold,
                    "best_C": best_c,
                    "macro_f1": macro_f1(gold, pred, n_classes),
                    "accuracy": accuracy(gold, pred),
                }
            )
            df = pd.DataFrame(probs, columns=[f"prob_{i}" for i in range(n_classes)])
            df.insert(0, "pred", pred)
            df.insert(0, "gold", gold)
            df.insert(0, "fold", fold)
            df.insert(0, "model_seed", 0)
            df.insert(0, "fold_seed", fs)
            df.insert(0, "id", ev["id"].to_numpy())
            oof_frames.append(df)
            print(
                f"{kind} fs{fs} f{fold}: F1={per_fold[-1]['macro_f1']:.4f} C={best_c}", flush=True
            )
    oof_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(oof_frames, ignore_index=True).to_csv(oof_dir / f"{kind}.csv", index=False)
    f1 = np.array([p["macro_f1"] for p in per_fold])
    acc = np.array([p["accuracy"] for p in per_fold])
    return {
        "per_fold": per_fold,
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std(ddof=1)),
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)),
        "n_fold_runs": len(per_fold),
        "wall_clock_s": time.perf_counter() - t0,
        **gpu,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="B0/B1 floors")
    ap.add_argument("--config", default="configs/bakeoff.yaml")
    ap.add_argument("--kinds", nargs="*", default=["B0", "B1"], choices=["B0", "B1"])
    ap.add_argument("--fold-seeds", nargs="*", type=int, default=None, help="smoke override")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--oof-dir", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.fold_seeds is not None:
        cfg["baselines"]["fold_seed_idx"] = args.fold_seeds
    res = Path(args.results_dir or cfg["results_dir"])
    oof = Path(args.oof_dir or cfg["oof_dir"])
    frame = load_cv_frame()
    out: dict[str, Any] = {}
    for kind in args.kinds:
        out[kind] = run_baseline(kind, cfg, frame, oof)
        print(f"{kind}: F1 {out[kind]['macro_f1_mean']:.4f} +/- {out[kind]['macro_f1_std']:.4f}")
        if not args.no_wandb:
            summ = {k: v for k, v in out[kind].items() if k != "per_fold"}
            summ["config"] = {
                "baseline": kind,
                "c_grid": cfg["baselines"]["c_grid"],
                "fold_seed_idx": cfg["baselines"]["fold_seed_idx"],
            }
            out[kind]["wandb_url"] = _wandb_summary(
                kind, cfg["wandb"]["group_bakeoff"], cfg["wandb"]["project"], summ
            )
    res.mkdir(parents=True, exist_ok=True)
    (res / "baselines.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
