from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "render_readme_results", ROOT / "scripts" / "render_readme_results.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load()


def test_formatters() -> None:
    assert R.ci({"point": 0.5, "lo": 0.25, "hi": 0.75}) == "0.500 [0.250, 0.750]"
    assert R.mean_std({"mean": 0.9, "std": 0.01}) == "0.900 ± 0.010"
    assert R.delta_cell({"delta": 0.1, "lo": -0.02, "hi": 0.2}) == "+0.100 [-0.020, +0.200]"
    # (other - shipped) deltas are flipped so the sign convention matches the other rows
    assert R.delta_cell({"delta": -0.3, "lo": -0.4, "hi": -0.2}, negate=True) == (
        "+0.300 [+0.200, +0.400]"
    )


def test_best_llm_picks_highest_macro_f1() -> None:
    metrics = {
        "a|zero": {"macro_f1": {"point": 0.5}},
        "b|five": {"macro_f1": {"point": 0.7}},
        "b|zero": {"macro_f1": {"point": 0.6}},
    }
    assert R.best_llm(metrics)[0] == "b|five"


def test_colab_url_from_repo_url_and_rejects_other_hosts() -> None:
    url = R.colab_url("https://github.com/o/r.git")
    assert url == (
        "https://colab.research.google.com/github/o/r/blob/v1.0-submission/"
        "notebooks/intent_router_colab.ipynb"
    )
    with pytest.raises(ValueError):
        R.colab_url("https://example.com/o/r")


def test_replace_block_is_idempotent_and_handles_empty_and_missing_markers() -> None:
    text = "a\n<!-- X:BEGIN -->\n<!-- X:END -->\nb\n"
    once = R.replace_block(text, "X", "body")
    assert once == "a\n<!-- X:BEGIN -->\nbody\n<!-- X:END -->\nb\n"
    assert R.replace_block(once, "X", "body") == once
    assert R.replace_block(once, "X", "new\nlines") == (
        "a\n<!-- X:BEGIN -->\nnew\nlines\n<!-- X:END -->\nb\n"
    )
    with pytest.raises(SystemExit):
        R.replace_block("no markers", "X", "body")


def test_links_block_placeholders_without_links_json(tmp_path: Path) -> None:
    out = R.links_block(tmp_path)
    assert out.count("[placeholder") == 4
    assert "colab.research.google.com" not in out


def test_links_block_uses_links_json_and_derives_colab(tmp_path: Path) -> None:
    (tmp_path / "report").mkdir()
    (tmp_path / "report" / "links.json").write_text(
        json.dumps({"repo": "https://github.com/o/r", "wandb": "https://wandb.ai/u/p"}),
        encoding="utf-8",
    )
    out = R.links_block(tmp_path)
    assert "https://wandb.ai/u/p" in out
    assert "W%26B_report" in out  # the ampersand is percent-encoded in the shields.io path
    assert "blob/v1.0-submission/notebooks/intent_router_colab.ipynb" in out
    assert out.count("[placeholder") == 2  # report.pdf and Hugging Face still pending


def test_links_block_renders_report_pdf_as_repo_relative_link(tmp_path: Path) -> None:
    (tmp_path / "report").mkdir()
    (tmp_path / "report" / "links.json").write_text(
        json.dumps({"report": "report.pdf", "hf": "https://huggingface.co/o/m"}), encoding="utf-8"
    )
    out = R.links_block(tmp_path)
    assert "](report.pdf)" in out
    assert "report.pdf [placeholder" not in out


def test_deliverables_block_uses_the_same_links(tmp_path: Path) -> None:
    (tmp_path / "report").mkdir()
    (tmp_path / "report" / "links.json").write_text(
        json.dumps(
            {
                "report": "report.pdf",
                "hf": "https://huggingface.co/o/m",
                "repo": "https://github.com/o/r",
                "wandb": "https://wandb.ai/u/p",
            }
        ),
        encoding="utf-8",
    )
    out = R.deliverables_block(tmp_path)
    for needle in ("](report.pdf)", "](https://huggingface.co/o/m)", "](https://github.com/o/r)"):
        assert needle in out
    assert "](https://wandb.ai/u/p)" in out
    assert "blob/v1.0-submission/notebooks/intent_router_colab.ipynb" in out
    assert "placeholder" not in out
    assert "placeholder" in R.deliverables_block(tmp_path / "missing")


def test_committed_links_json_has_real_urls_and_readme_matches() -> None:
    links = json.loads((ROOT / "report" / "links.json").read_text(encoding="utf-8"))
    assert all(links[k] for k in ("hf", "repo", "colab", "wandb", "report"))
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert R.render(text) == text
    assert "placeholder" not in text


def test_results_block_refuses_a_different_shipped_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "_load", lambda rel, root: {"version": "v3"})
    with pytest.raises(SystemExit):
        R.results_block(ROOT)


def test_readme_blocks_match_the_committed_results() -> None:
    """The README tables are exactly what the renderer produces from results/ (no hand edits)."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    # RESULTS block only: the LINKS block follows report/links.json, which changes at publish
    assert R.replace_block(text, "RESULTS", R.results_block(ROOT)) == text, (
        "run: python scripts/render_readme_results.py"
    )


def test_every_rendered_number_comes_from_json() -> None:
    """Spot check: the shipped test macro-F1 in the README equals track_a.json to 3 decimals."""
    ta = json.loads((ROOT / "results/final/track_a.json").read_text(encoding="utf-8"))
    point = f"{ta['test']['macro_f1']['point']:.3f}"
    assert point in (ROOT / "README.md").read_text(encoding="utf-8")


def test_quickstart_snippets_run_against_a_staged_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Execute the README quick-start code with the Hub repo id replaced by a local staged dir."""
    if not (ROOT / "serve_model" / "model.safetensors").exists():
        pytest.skip("serve_model/ (gitignored packaged model) is not available")
    import huggingface_hub

    from intent_router import hub

    stage = hub.build_hub_dir("v1", root=ROOT, stage_root=tmp_path)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Quick start")[1].split("## Reproduce")[0]
    blocks = re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL)
    assert len(blocks) == 2
    repo_line = re.search(r'^REPO = ".*"$', blocks[0], flags=re.MULTILINE)
    assert repo_line
    blocks[0] = blocks[0].replace(repo_line[0], f"REPO = {str(stage)!r}")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **_kw: str(repo))
    ns: dict[str, Any] = {}
    sys_path = list(sys.path)
    try:
        for code in blocks:
            exec(compile(code, "<readme>", "exec"), ns)  # noqa: S102
    finally:
        sys.path[:] = sys_path
        sys.modules.pop("predict", None)
    lines = capsys.readouterr().out.strip().splitlines()
    labels = json.loads((stage / "config.json").read_text(encoding="utf-8"))["id2label"].values()
    assert lines[0] in labels
    assert "'abstained'" in lines[1] and "'ood_score'" in lines[1]


def test_confirm_msp_row_uses_dash_and_has_footnote() -> None:
    block = R.track_b_block(ROOT)
    assert "not recorded in this file" not in block
    assert "| MSP baseline | " in block and "(mean only) | — | — |" in block
    assert block.rstrip().endswith("— only MSP AUROC is recorded on CONFIRM.")
