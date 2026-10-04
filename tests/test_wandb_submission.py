from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rerun = _load("wandb_final_rerun")
runs = _load("wandb_submission_runs")
report = _load("wandb_report")

MD = """# Title

## Track A

| metric | v1 | v3 |
|---|---|---|
| macro-F1 | 0.9 | 0.8 |
| acc | 0.7 | 0.6 |

text

## Verdict (rendered)

- first
- second

## Other

| a | b |
|---|---|
| 1 | 2 |
"""


def test_sha256_file_and_config_diff(tmp_path: Path) -> None:
    f = tmp_path / "x.csv"
    f.write_bytes(b"abc")
    assert rerun.sha256_file(f) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    cfg = json.loads((ROOT / "results/final/model_version.json").read_text())["train_config"]
    import yaml

    yml = yaml.safe_load((ROOT / "configs/final.yaml").read_text())["train"]
    assert rerun.train_config_matches(yml, cfg) == []  # re-run == shipped configuration
    assert rerun.train_config_matches({**yml, "lr": 1e-4}, cfg) == ["lr"]


def test_assert_safe_dirs_refuses_shipped_paths() -> None:
    rerun.assert_safe_dirs(rerun.MODEL_DIR, rerun.RESULTS_DIR)
    rerun.assert_safe_dirs(rerun.DRY_MODEL_DIR, rerun.DRY_RESULTS_DIR)
    with pytest.raises(ValueError):
        rerun.assert_safe_dirs(Path("outputs/final_model"), rerun.RESULTS_DIR)
    with pytest.raises(ValueError):
        rerun.assert_safe_dirs(rerun.MODEL_DIR, Path("results/final"))
    with pytest.raises(ValueError):  # a child of the shipped results dir is also refused
        rerun.assert_safe_dirs(rerun.MODEL_DIR, Path("results/final/sub"))


def test_fingerprint_check() -> None:
    ok = rerun.fingerprint_check(rerun.SHIPPED_FINGERPRINT)
    assert ok["fingerprint_match"] is True
    bad = rerun.fingerprint_check("0" * 64)
    assert bad["fingerprint_match"] is False
    assert bad["fingerprint_expected"] == rerun.SHIPPED_FINGERPRINT


def test_run_meta_has_provenance_and_no_paths() -> None:
    meta = rerun.build_run_meta("abc", "def", {"torch": "x"}, "query: ")
    assert meta["run_id"] == "final-v1"
    assert meta["split_sha256"] == "def" and meta["git_sha"] == "abc"
    assert meta["no_test_inference"] is True
    assert "\\" not in json.dumps(meta)


def test_parse_md_tables_and_bullets() -> None:
    tables = runs.parse_md_tables(MD)
    assert [t[0] for t in tables] == ["Track A", "Other"]
    assert tables[0][1] == ["metric", "v1", "v3"]
    assert tables[0][2] == [["macro-F1", "0.9", "0.8"], ["acc", "0.7", "0.6"]]
    assert runs.md_bullets(MD, "Verdict") == ["first", "second"]
    assert runs.md_bullets(MD, "Nope") == []


def test_slug() -> None:
    assert runs.slug("Track A test (one logged), n=74") == "track_a_test_one_logged_n_74"


def test_real_tradeoff_md_parses_into_tables() -> None:
    md = (ROOT / "results/tradeoff_v1_v3.md").read_text(encoding="utf-8")
    tables = runs.parse_md_tables(md)
    assert len(tables) >= 6 and all(len(r) == len(c) for _, c, rows in tables for r in rows)
    assert len({runs.slug(h) for h, _, _ in tables}) == len(tables)  # unique W&B table keys


def test_trackb_and_bakeoff_rows_match_saved_files() -> None:
    headline = json.loads((ROOT / "results/trackb/headline.json").read_text())
    cols, rows = runs.trackb_method_rows(headline)
    assert len(rows) == len(headline["methods"]) and all(len(r) == len(cols) for r in rows)
    maha = next(r for r in rows if r[0] == "maha_ft")
    assert maha[cols.index("auroc_mean")] == headline["methods"]["maha_ft"]["auroc"]["mean"]
    scols, srows = runs.trackb_seed_rows(headline)
    assert {r[0] for r in srows} == {42, 43, 44} and all(len(r) == len(scols) for r in srows)
    bake = json.loads((ROOT / "results/bakeoff/summary.json").read_text())
    ccols, crows = runs.bakeoff_config_rows(bake)
    assert len(crows) == len(bake["configs"]) and all(len(r) == len(ccols) for r in crows)
    sel = json.loads((ROOT / "results/bakeoff/selection.json").read_text())
    _, srows2 = runs.bakeoff_selection_rows(sel)
    assert [r[0] for r in srows2 if r[1]] == [sel["winner"]]


def test_llm_rows_include_finetuned_reference() -> None:
    summary = json.loads((ROOT / "results/llm_baseline/summary.json").read_text())
    cols, rows = runs.llm_rows(summary)
    assert len(rows) == len(summary["track_a"]["llm"]) + 1
    assert all(len(r) == len(cols) for r in rows)
    assert rows[-1][0].startswith("fine-tuned")


def test_calibration_rows() -> None:
    ta = json.loads((ROOT / "results/final/track_a.json").read_text())
    cols, rows = runs.calibration_rows(ta)
    assert [r[0] for r in rows] == ["val", "test"] and all(len(r) == len(cols) for r in rows)


def test_report_helpers() -> None:
    assert (
        report.fmt_ci({"point": 0.94651, "lo": 0.8651, "hi": 0.9892}) == "0.9465 [0.8651, 0.9892]"
    )
    links = report.merge_links({"repo": "r", "wandb": "old"}, {"wandb": "new"})
    assert links["wandb"] == "new" and links["repo"] == "r"
    assert links["hf"] is None and links["colab"] is None  # placeholders for other tools
    assert set(links) == set(report.LINK_KEYS)
    cfg = json.loads((ROOT / "results/final/model_version.json").read_text())["train_config"]
    md = report.config_markdown(cfg, "s" * 64, "f" * 64)
    assert "batch size 16" in md and "seed 42" in md and "stopped after epoch 9" in md
    assert report.run_url("x").endswith("/multilingual-intent-router/runs/x")
