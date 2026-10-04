from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "wandb_final_summary", ROOT / "scripts" / "wandb_final_summary.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()


def test_lr_schedule_matches_hf_linear_warmup_decay() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    n_train, bs, epochs, wr, lr = 345, 16, 20, 0.1, 3e-5
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=lr)
    spe = -(-n_train // bs)
    total = spe * epochs
    sched = transformers.get_linear_schedule_with_warmup(opt, int(wr * total), total)
    expected = []
    for step in range(1, 9 * spe + 1):
        opt.step()
        sched.step()
        if step % spe == 0:
            expected.append(opt.param_groups[0]["lr"])
    got = m.lr_at_epoch_ends(n_train, bs, epochs, wr, lr, 9)
    assert got == pytest.approx(expected, rel=1e-9)


def test_lr_schedule_shape() -> None:
    lrs = m.lr_at_epoch_ends(320, 16, 20, 0.1, 3e-5, 9)
    assert len(lrs) == 9
    assert max(lrs) <= 3e-5 and min(lrs) > 0
    assert lrs[0] < lrs[1]  # still warming up at epoch 1
    assert lrs[-1] < lrs[2]  # decaying by epoch 9


def test_epoch_records_alignment_and_length_check() -> None:
    run = {
        "epochs": [
            {"epoch": 1, "train_loss": 2.0, "eval_loss": 1.9, "macro_f1": 0.1, "accuracy": 0.2},
            {"epoch": 2, "train_loss": 1.0, "eval_loss": 1.1, "macro_f1": 0.5, "accuracy": 0.6},
        ]
    }
    recs = m.epoch_records(run, [1e-5, 2e-5])
    assert recs[1] == {
        "epoch": 2,
        "train/loss": 1.0,
        "val/loss": 1.1,
        "val/macro_f1": 0.5,
        "val/accuracy": 0.6,
        "lr": 2e-5,
    }
    with pytest.raises(ValueError):
        m.epoch_records(run, [1e-5])  # strict zip: a length mismatch must not be silent


def test_error_rows_whitelist_drops_text() -> None:
    df = pd.DataFrame(
        {
            "id": ["a"],
            "text": ["SECRET MESSAGE"],
            "gold_label": ["x"],
            "pred_label": ["y"],
            "confidence": [0.9],
            "confidence_msp_raw": [0.95],
        }
    )
    rows = m.error_rows(df)
    assert rows == [["a", "x", "y", 0.9]]
    assert "SECRET" not in json.dumps(rows)
    with pytest.raises(ValueError):
        m.error_rows(df.drop(columns=["confidence"]))


def test_per_class_and_class_names_from_track_a() -> None:
    ta = {
        "test": {
            "classes": [{"index": 1, "label": "b"}, {"index": 0, "label": "a"}],
            "per_class": [
                {"label": "a", "precision": 1.0, "recall": 0.5, "f1": 0.66, "support": 2}
            ],
        }
    }
    assert m.class_names(ta) == ["a", "b"]
    cols, rows = m.per_class_rows(ta)
    assert cols == ["label", "precision", "recall", "f1", "support"]
    assert rows == [["a", 1.0, 0.5, 0.66, 2]]


def test_unknown_score_orientation_and_auroc() -> None:
    df = pd.DataFrame(
        {
            "set": ["cal", "eval", "eval", "eval", "eval"],
            "is_unknown": [False, False, False, True, True],
            "maha_ft": [0.0, -1.0, -2.0, -8.0, -9.0],  # higher == more known
        }
    )
    assert list(m.eval_labels(df)) == [0, 0, 1, 1]
    assert list(m.unknown_score(df, "maha_ft")) == [1.0, 2.0, 8.0, 9.0]  # cal row excluded
    assert m.roc_auroc(df, "maha_ft") == 1.0


def test_summaries_on_committed_results() -> None:
    root = ROOT / "results"
    ta = json.loads((root / "final" / "track_a.json").read_text())
    ts = json.loads((root / "final" / "train_summary.json").read_text())
    hl = json.loads((root / "trackb" / "headline.json").read_text())
    s = m.track_a_summary(ta, ts)
    assert s["test/macro_f1_ci_lo"] < s["test/macro_f1"] < s["test/macro_f1_ci_hi"]
    assert s["test/n"] == ta["test"]["n"]
    h = m.headline_summary(hl)
    assert h["trackb/maha_ft/auroc_mean"] == hl["methods"]["maha_ft"]["auroc"]["mean"]
    assert h["trackb/n_seeds"] == 3.0


def test_saved_scores_reproduce_saved_headline_auroc() -> None:
    hl = json.loads((ROOT / "results" / "trackb" / "headline.json").read_text())
    scores = {
        s: pd.read_csv(ROOT / "results" / "trackb" / "scores" / f"headline_s{s}.csv")
        for s in m.SEEDS
    }
    m.check_scores_reproduce_headline(scores, hl)  # raises on mismatch
    bad = json.loads(json.dumps(hl))
    bad["runs"]["headline_s42"]["per_method"]["maha_ft"]["auroc"] += 0.05
    with pytest.raises(AssertionError):
        m.check_scores_reproduce_headline(scores, bad)


def test_source_files_exist_and_config_has_no_text() -> None:
    mv = json.loads((ROOT / "results" / "final" / "model_version.json").read_text())
    ts = json.loads((ROOT / "results" / "final" / "train_summary.json").read_text())
    cwd = Path.cwd()
    try:
        import os

        os.chdir(ROOT)
        cfg = m.build_config(mv, ts, 345)
    finally:
        os.chdir(cwd)
    assert cfg["model_fingerprint"] == mv["model_fingerprint"]
    assert cfg["train_config"]["stop_epoch"] == 9
    assert len(cfg["source_sha256"]) == len(m.SOURCE_FILES)
    assert all(len(v) == 64 for v in cfg["source_sha256"].values())
    assert "text" not in json.dumps(cfg).lower().replace("test_", "")  # no message-text fields
