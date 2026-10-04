from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "intent_router_colab.ipynb"


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "build_colab_notebook", ROOT / "scripts" / "build_colab_notebook.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def nb() -> dict[str, Any]:
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _code_cells(nb: dict[str, Any]) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _strip_magics(src: str) -> str:
    """Drop IPython `!cmd` / `%magic` lines so the rest can be parsed by ast."""
    return "\n".join(
        "pass" if line.lstrip().startswith(("!", "%")) else line for line in src.splitlines()
    )


def test_notebook_json_valid(nb: dict[str, Any]) -> None:
    nbformat = pytest.importorskip("nbformat")  # not a project dependency; structural test below
    nbformat.validate(nbformat.reads(NB_PATH.read_text(encoding="utf-8"), as_version=4))


def test_notebook_structure(nb: dict[str, Any]) -> None:
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    for c in nb["cells"]:
        assert c["cell_type"] in {"markdown", "code"}
        assert isinstance(c["source"], list)
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None  # committed unexecuted


def test_notebook_matches_generator(nb: dict[str, Any]) -> None:
    assert _load_builder().build() == nb, "rerun scripts/build_colab_notebook.py"


def test_code_cells_parse(nb: dict[str, Any]) -> None:
    for i, src in enumerate(_code_cells(nb)):
        ast.parse(_strip_magics(src), filename=f"code-cell-{i}")


def test_required_content(nb: dict[str, Any]) -> None:
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert 'REPO_TAG = "v1.0-submission"' in text
    assert "--depth" in text and "PUBLIC_REPO_URL" in text
    assert "GH_TOKEN" not in text and "intent-router.git" not in text  # no private-repo dependence
    assert text.count("ESTIMATE") >= 5
    for flag in (
        "PUSH = False",
        "RUN_BAKEOFF = False",
        "RUN_LOCO = False",
        "RUN_LLM_BASELINE = False",
        "RUN_PHASE4C = False",
        "RUN_PHASE4E = False",
    ):
        assert flag in text


def test_setup_cell_refuses_the_placeholder_and_never_clones(
    nb: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the placeholder URL the setup cell raises before any subprocess/clone runs."""
    cell = next(s for s in _code_cells(nb) if "PUBLIC_REPO_URL" in s)
    default = 'PUBLIC_REPO_URL = "https://github.com/gaurav-gandhi-2411/multilingual-intent-router"'
    assert default in cell  # the default is the real public repo; the placeholder is still refused
    calls: list[Any] = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match="Set PUBLIC_REPO_URL"):
        exec(  # noqa: S102 - our own generated cell
            compile(cell.replace(default, "PUBLIC_REPO_URL = PLACEHOLDER"), "setup-cell", "exec"),
            {},
        )
    assert calls == []
    filled = cell.replace(default, 'PUBLIC_REPO_URL = "https://example.invalid/o/r"').split(
        "if not REPO_DIR.exists()"
    )[0]
    ns: dict[str, Any] = {}
    exec(compile(filled, "setup-cell", "exec"), ns)  # noqa: S102
    assert ns["clone_url"] == "https://example.invalid/o/r.git" and ns["REPO_DIR"].name == "r"


def test_referenced_modules_and_configs_exist(nb: dict[str, Any]) -> None:
    code = "\n".join(_code_cells(nb))
    modules = set(re.findall(r'run_py\(\s*"(\w+)"', code))
    assert {"final", "trackb"} <= modules
    for mod in modules:
        src = (ROOT / "src" / "intent_router" / f"{mod}.py").read_text(encoding="utf-8")
        assert 'if __name__ == "__main__"' in src, f"{mod} has no CLI entry"
    for name in set(re.findall(r'make_cfg\(\s*"(\w+)"', code)):
        assert (ROOT / "configs" / f"{name}.yaml").exists(), name
    for rel in set(re.findall(r'"((?:results|splits|configs)/[\w./-]+\.(?:json|csv|yaml))"', code)):
        assert (ROOT / rel).exists(), f"referenced path missing: {rel}"
    assert (ROOT / "requirements.txt").exists()


def test_cli_flags_used_exist(nb: dict[str, Any]) -> None:
    code = "\n".join(_code_cells(nb))
    for mod, flags in {
        "final": ["--config", "--stage", "--no-wandb"],
        "trackb": ["--config", "--stage", "--no-wandb"],
    }.items():
        src = (ROOT / "src" / "intent_router" / f"{mod}.py").read_text(encoding="utf-8")
        for flag in flags:
            assert f'"{flag}"' in src, f"{mod} lacks {flag}"
    for stage in ("train", "evaluate", "headline", "loco"):
        assert f'"{stage}"' in code


def test_no_secret_like_strings(nb: dict[str, Any]) -> None:
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    patterns = [
        r"hf_[A-Za-z0-9]{20,}",
        r"ghp_[A-Za-z0-9]{20,}",
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"\b[0-9a-f]{40}\b",  # wandb keys / raw tokens are 40 hex chars
        r"(WANDB_API_KEY|HF_TOKEN|GH_TOKEN)\s*=\s*[\"'][^\"']+[\"']",
    ]
    for pat in patterns:
        assert not re.search(pat, text), f"secret-looking string matches {pat}"


def test_secrets_never_printed(nb: dict[str, Any]) -> None:
    """A secret-holding name may appear in print() only wrapped in bool(...)."""
    secret_names = {"auth_url", "token", "CLONE_TOKEN"}
    for src in _code_cells(nb):
        for node in ast.walk(ast.parse(_strip_magics(src))):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print"):
                continue
            wrapped = {
                id(n)
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "bool"
                for n in ast.walk(sub)
            }
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id in secret_names:
                    assert id(sub) in wrapped, f"print() exposes {sub.id}"
                if isinstance(sub, ast.Subscript) and "environ" in ast.dump(sub.value):
                    raise AssertionError("print() reads os.environ directly")


def test_make_cfg_redirects_outputs_only(
    nb: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the data-independent helper cell: outputs move to results_colab/, inputs stay."""
    cell = next(s for s in _code_cells(nb) if "def make_cfg" in s)
    cell = cell.replace('Path("/content/colab_cfg")', f"Path({str(tmp_path)!r})")
    cell = cell.split("FINAL_CFG = make_cfg")[0]  # definitions only
    monkeypatch.chdir(ROOT)
    ns: dict[str, Any] = {
        "Path": Path,
        "USE_WANDB": False,
        "os": __import__("os"),
        "sys": sys,
        "subprocess": __import__("subprocess"),
        "REPO_DIR": ROOT,
    }
    exec(compile(cell, "helper-cell", "exec"), ns)  # noqa: S102 - our own generated cell
    final = yaml.safe_load(Path(ns["make_cfg"]("final")).read_text())
    assert final["results_dir"] == "results_colab/final"
    assert final["test_eval_log"] == "results_colab/final/test_eval_log.jsonl"
    assert final["figures_dir"] == "results_colab/figures"
    assert final["selection_path"] == "results/bakeoff/selection.json"  # input stays read-only
    assert final["analysis"]["oof_dir"] == "results/oof"
    assert final["wandb"]["enabled"] is False and final["wandb"]["group"] == "colab"
    tb = yaml.safe_load(Path(ns["make_cfg"]("trackb", final_config="X.yaml")).read_text())
    assert tb["results_dir"] == "results_colab/trackb"
    assert tb["test_inference_log"] == "results_colab/trackb/test_inference_log.jsonl"
    assert tb["final_config"] == "X.yaml"
    # The committed configs are untouched by the redirect.
    committed = yaml.safe_load((ROOT / "configs/final.yaml").read_text())
    assert committed["results_dir"] == "results/final"


# ------------------------------------------------------------------ latency cells (section 8b)
def _cell_with(nb: dict[str, Any], needle: str) -> str:
    return next(s for s in _code_cells(nb) if needle in s)


def test_latency_cells_come_after_track_a_and_before_push(nb: dict[str, Any]) -> None:
    cells = _code_cells(nb)
    idx = {n: next(i for i, s in enumerate(cells) if n in s) for n in
           ('run_py("final"', "def rows():", 'run_py("latency_colab"', "PUSH = False")}  # fmt: skip
    assert (
        idx['run_py("final"']
        < idx["def rows():"]
        < idx['run_py("latency_colab"']
        < idx["PUSH = False"]
    )
    md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert (
        "copy `results_colab/serving/latency_colab.json` to `results/serving/latency_colab.json`"
        in md
    )


def test_latency_cell_prints_the_full_json_between_markers(nb: dict[str, Any]) -> None:
    cell = _cell_with(nb, 'run_py("latency_colab"')
    begin = cell.index("BEGIN latency_colab.json")
    end = cell.index("END latency_colab.json")
    assert begin < cell.index("Path(LATENCY_OUT).read_text())", begin) < end


def test_latency_cell_has_no_unguarded_cuda_and_uses_val_only(nb: dict[str, Any]) -> None:
    cell = _cell_with(nb, 'run_py("latency_colab"')
    assert "cuda" not in cell.lower() and "test" not in cell.lower().replace("latency", "")
    mod = (ROOT / "src" / "intent_router" / "latency_colab.py").read_text(encoding="utf-8")
    assert "cuda.is_available" not in mod and ".cuda(" not in mod and 'to("cuda")' not in mod
    assert 'load_split(a.data, a.splits, "val")' in mod


def test_latency_pip_cell_installs_the_pins_from_requirements_serve(
    nb: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell_with(nb, "SERVE_PINS")
    calls: list[list[str]] = []
    monkeypatch.chdir(ROOT)
    ns: dict[str, Any] = {
        "Path": Path,
        "sys": sys,
        "subprocess": type("S", (), {"run": staticmethod(lambda cmd, **k: calls.append(cmd))}),
        "md": type("M", (), {"version": staticmethod(lambda p: "x")}),
    }
    exec(compile(cell, "latency-pip-cell", "exec"), ns)  # noqa: S102 - our own generated cell
    serve = (ROOT / "requirements-serve.txt").read_text(encoding="utf-8")
    pins = calls[0][calls[0].index("install") + 2 :]
    assert {"onnx", "onnxruntime"} <= {p.split("==")[0] for p in pins}
    assert all(p in serve.splitlines() for p in pins) and all("==" in p for p in pins)


def test_latency_cell_runs_with_the_heavy_part_monkeypatched(
    nb: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cell = _cell_with(nb, 'run_py("latency_colab"')
    st = {"p50_ms": 12.34, "p95_ms": 23.45}
    payload = {
        "n_rows": 74,
        "environment": {
            "cpu_model": "Fake CPU", "os_cpu_count": 2, "runtime_type": "Colab, CPU-only",
            "torch": "2.x", "onnxruntime": "1.x",
        },
        "latency": {"onnx_fp32": {"threads_1": st, "threads_2": st}},
    }  # fmt: skip
    calls: list[tuple[str, ...]] = []

    def fake_run_py(module: str, *args: str) -> None:
        calls.append((module, *args))
        out = Path(args[args.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload))

    monkeypatch.chdir(tmp_path)
    ns: dict[str, Any] = {
        "Path": Path,
        "OUT_ROOT": "results_colab",
        "FINAL_CFG": "cfg.yaml",
        "run_py": fake_run_py,
    }
    exec(compile(cell, "latency-cell", "exec"), ns)  # noqa: S102 - our own generated cell
    assert ns["LATENCY_OUT"] == "results_colab/serving/latency_colab.json"
    assert calls == [
        (
            "latency_colab", "--config", "cfg.yaml", "--results-dir", "results_colab/final",
            "--out", ns["LATENCY_OUT"],
        )
    ]  # fmt: skip
    out = capsys.readouterr().out
    assert "CPU: Fake CPU; vCPUs: 2; runtime: Colab, CPU-only" in out
    assert "onnx_fp32  threads_2  p50=12.3 ms" in out and "copy results_colab/serving" in out


def test_latency_module_cli_flags_used_by_the_cell_exist() -> None:
    src = (ROOT / "src" / "intent_router" / "latency_colab.py").read_text(encoding="utf-8")
    for flag in ("--config", "--out", "--results-dir"):
        assert f'"{flag}"' in src


# ------------------------------------------------- header, Hub demo, inline results, verdict
def _md(c: dict[str, Any]) -> str:
    return "".join(c["source"])


def test_header_cell_is_first_and_states_the_run_facts(nb: dict[str, Any]) -> None:
    first = nb["cells"][0]
    assert first["cell_type"] == "markdown"
    h = _md(first)
    assert "Run all with no edits" in h and "Runtime > Run all" in h
    assert "ESTIMATE" in h and "~6 min" in h and "~3-5 min" in h
    assert "NOT MEASURED" in h and "Install" in h and "latency" in h
    assert "USE_WANDB" in h and "WANDB_API_KEY" in h and "PUSH" in h and "HF_TOKEN" in h
    assert "off by default" in h


def test_cell_order(nb: dict[str, Any]) -> None:
    cells = _code_cells(nb)
    needles = [
        "PUBLIC_REPO_URL = ",
        "def make_cfg",
        'run_py("final"',
        "AutoTokenizer.from_pretrained",
        'run_py("trackb"',
        "colab_test_confusion.png",
        "def judge(",
        'run_py("latency_colab"',
        "PUSH = False",
    ]
    idx = [next(i for i, s in enumerate(cells) if n in s) for n in needles]
    assert idx == sorted(idx) and len(set(idx)) == len(idx)


def test_hf_cell_is_tolerant_of_failure(
    nb: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both Hub calls raise (private repo / no network): messages printed, no exception."""
    pytest.importorskip("transformers")
    hub = pytest.importorskip("huggingface_hub")
    import transformers

    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("401 Client Error: repository is private")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom)
    monkeypatch.setattr(hub, "snapshot_download", boom)
    ns: dict[str, Any] = {"REPO_DIR": ROOT, "sys": sys, "Path": Path}
    exec(compile(_cell_with(nb, "HF_REPO ="), "hf-cell", "exec"), ns)  # noqa: S102
    out = capsys.readouterr().out
    assert "HF model demo skipped: OSError" in out and "Continuing." in out
    assert "HF abstention demo skipped: OSError" in out


def test_hf_cell_uses_the_public_repo_and_the_repo_router(nb: dict[str, Any]) -> None:
    cell = _cell_with(nb, "HF_REPO =")
    assert 'HF_REPO = "gauravgandhi2411/multilingual-intent-router"' in cell
    assert "AutoTokenizer.from_pretrained(HF_REPO)" in cell
    assert "AutoModelForSequenceClassification.from_pretrained(HF_REPO)" in cell
    assert "hub_predict import IntentRouter" in cell and "abstained" in cell
    for field in ("label", "confidence", "ood_score", "threshold", "abstained"):
        assert field in cell
    assert "carbon emissions report per lane" in cell and "PROBES = DEMO_TEXTS + [OOD_TEXT" in cell
    assert (ROOT / "src" / "intent_router" / "hub_predict.py").exists()


def test_hf_cell_notes_temperature_scaling_from_the_router(
    nb: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The note reads T at run time from the router object (no number typed into the cell)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import transformers

    import intent_router.hub_predict as hp

    class FakeRouter:
        temperature = 1.0228
        ood_threshold = -128.06

        @classmethod
        def from_pretrained(cls, _repo: str) -> FakeRouter:
            return cls()

        def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
            return [
                {"label": "x", "confidence": 0.9, "ood_score": -1.0, "abstained": False}
                for _ in texts
            ]

    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("offline")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom)
    monkeypatch.setattr(hp, "IntentRouter", FakeRouter)
    cell = _cell_with(nb, "HF_REPO =")
    assert "1.0228" not in cell
    ns: dict[str, Any] = {"REPO_DIR": ROOT, "sys": sys, "Path": Path}
    exec(compile(cell, "hf-cell", "exec"), ns)  # noqa: S102
    out = capsys.readouterr().out
    assert "temperature-scaled, T = 1.0228" in out and "raw-softmax block printed above" in out


def _fake_results(root: Path, new_shift: float, tb_shift: float = 0.0) -> None:
    """Write committed-style and fresh-style result JSONs; fresh = committed + shift."""

    def pt(v: float, shift: float) -> dict[str, float]:
        return {"point": v + shift, "lo": v - 0.1, "hi": v + 0.05}

    def a(shift: float) -> dict[str, Any]:
        return {
            "model_fingerprint": "abcdef0123456789",
            "test": {
                "macro_f1": pt(0.90, shift),
                "accuracy": pt(0.91, shift),
                "n": 4,
                "classes": [{"index": 0, "label": "x"}, {"index": 1, "label": "y"}],
                "confusion_counts": [[2, 0], [1, 1]],
            },
        }

    def b(shift: float) -> dict[str, Any]:
        method = {
            "auroc": {"mean": 0.80 + shift},
            "strict_rejection_recall": {"mean": 0.40 + shift},
            "retention_known": {"mean": 0.95},
        }
        return {"methods": {"maha_ft": method, "msp": method}}

    for base, sa, sb in (("results", 0.0, 0.0), ("results_colab", new_shift, tb_shift)):
        (root / base / "final").mkdir(parents=True, exist_ok=True)
        (root / base / "trackb").mkdir(parents=True, exist_ok=True)
        (root / base / "final" / "track_a.json").write_text(json.dumps(a(sa)))
        (root / base / "trackb" / "headline.json").write_text(json.dumps(b(sb)))


def _run_compare(nb: dict[str, Any]) -> str:
    ns: dict[str, Any] = {"Path": Path, "OUT_ROOT": "results_colab"}
    exec(compile(_cell_with(nb, "def judge("), "compare-cell", "exec"), ns)  # noqa: S102
    return str(ns["verdict"])


def test_compare_cell_pass_verdict(
    nb: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    _fake_results(tmp_path, new_shift=0.02, tb_shift=0.02)  # inside every tolerance
    monkeypatch.chdir(tmp_path)
    assert _run_compare(nb) == "REPRODUCTION: PASS"
    out = capsys.readouterr().out
    assert "FAIL" not in out and out.count("PASS") >= 7  # 6 rows + verdict line


def test_compare_cell_fail_verdict_lists_failing_metrics(
    nb: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    # +0.05 breaks macro-F1/accuracy/AUROC (+-0.03) but not rejection@95 (+-0.10)
    _fake_results(tmp_path, new_shift=0.05, tb_shift=0.05)
    monkeypatch.chdir(tmp_path)
    verdict = _run_compare(nb)
    assert verdict.startswith("REPRODUCTION: FAIL (") and "Track A test macro_f1" in verdict
    assert "AUROC" in verdict and "rejection@95" not in verdict
    assert verdict in capsys.readouterr().out


def test_compare_cell_boundary_and_rejection_failure(
    nb: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_results(tmp_path, new_shift=0.03, tb_shift=0.0)  # exactly at the tolerance passes
    monkeypatch.chdir(tmp_path)
    assert _run_compare(nb) == "REPRODUCTION: PASS"
    _fake_results(tmp_path, new_shift=0.0, tb_shift=0.11)  # rejection@95 outside +-0.10
    verdict = _run_compare(nb)
    assert verdict.startswith("REPRODUCTION: FAIL") and "rejection@95" in verdict


def test_compare_tolerances_are_the_stated_ones(nb: dict[str, Any]) -> None:
    cell = _cell_with(nb, "def judge(")
    assert '"macro_f1": 0.03' in cell and '"rejection@95": 0.10' in cell and '"auroc": 0.03' in cell
    text = "\n".join(_md(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "REPRODUCTION: PASS" in text


def test_inline_results_cell_runs_and_prints_labels_and_numbers_only(
    nb: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg", force=True)  # plt.show() must not open a Tk window (it blocks forever)
    _fake_results(tmp_path, new_shift=0.01, tb_shift=0.01)
    monkeypatch.chdir(tmp_path)
    ns: dict[str, Any] = {"Path": Path, "OUT_ROOT": "results_colab"}
    cell = _cell_with(nb, "colab_test_confusion.png")
    exec(compile(cell, "inline-cell", "exec"), ns)  # noqa: S102
    out = capsys.readouterr().out
    assert "Track A test macro-F1" in out and "rejection@95" in out and "retention" in out
    assert "model fingerprint: fresh abcdef01, committed abcdef01" in out
    assert (tmp_path / "results_colab/figures/colab_test_confusion.png").stat().st_size > 0


def test_no_dataset_text_is_displayed(nb: dict[str, Any]) -> None:
    code = "\n".join(_code_cells(nb))
    assert '["text"]' not in code and "['text']" not in code
    assert "head(" not in code and "sample(" not in code and "iloc" not in code


# ------------------------------------------------------------------ local runner
def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_notebook_local", ROOT / "scripts" / "run_notebook_local.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_runner_localizes_only_colab_parts(nb: dict[str, Any]) -> None:
    runner = _load_runner()
    cells = {i: s for i, s in runner.code_cells(nb)}
    pip_key = 'pip", "install", "-q", "-r'
    keys = ("PLACEHOLDER =", "files.upload", pip_key, "def make_cfg", "SERVE_PINS")
    by = {k: next(i for i, s in cells.items() if k in s) for k in keys}
    setup = runner.localize(cells[by["PLACEHOLDER ="]], ROOT)
    assert "REPO_DIR = Path(" in setup and "WORKDIR / Path(PUBLIC_REPO_URL" not in setup
    data = runner.localize(cells[by["files.upload"]], ROOT)
    assert "files.upload" not in data and "dataset ids differ" in data  # all checks still run
    install = runner.localize(cells[by[pip_key]], ROOT)
    assert '"pip", "install", "-q", "-r"' not in install and '"pip", "check"' in install
    pipe = runner.localize(cells[by["def make_cfg"]], ROOT)
    assert "/content" not in pipe and "results_colab/_colab_cfg" in pipe
    lat = runner.localize(cells[by["SERVE_PINS"]], ROOT)
    assert '"install", "-q", *pins' not in lat
    for src in cells.values():  # every localized cell still parses
        ast.parse(_strip_magics(runner.localize(src, ROOT)))
    plain = next(s for s in cells.values() if "def judge(" in s)
    assert runner.localize(plain, ROOT) == plain  # no Colab-specific part: unchanged


def test_trackb_cell_aggregates_with_the_report_stage(
    nb: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug found by the local run: `--stage headline` alone never writes headline.json."""
    src = (ROOT / "src" / "intent_router" / "trackb.py").read_text(encoding="utf-8")
    assert 'write_json(P.results / "headline.json", h)' in src and "def stage_report" in src
    cell = _cell_with(nb, "Track B headline already computed")
    calls: list[tuple[str, ...]] = []
    ns: dict[str, Any] = {
        "Path": Path,
        "OUT_ROOT": "results_colab",
        "TRACKB_CFG": "t.yaml",
        "WANDB_FLAG": ["--no-wandb"],
        "run_py": lambda *a: calls.append(a),
    }
    monkeypatch.chdir(tmp_path)
    exec(compile(cell, "trackb-cell", "exec"), ns)  # noqa: S102
    assert [c[c.index("--stage") + 1] for c in calls] == ["headline", "report"]
    (tmp_path / "results_colab/trackb").mkdir(parents=True)
    (tmp_path / "results_colab/trackb/headline.json").write_text("{}")
    calls.clear()
    exec(compile(cell, "trackb-cell", "exec"), ns)  # noqa: S102
    assert calls == []  # idempotent: nothing re-run when headline.json exists


# ------------------------------------------- torch/torchvision CUDA mismatch guard (sections 2/2b)
def _idx(cells: list[str], needle: str) -> int:
    return next(i for i, s in enumerate(cells) if needle in s)


def test_uninstall_and_assert_cells_follow_the_install_and_precede_later_imports(
    nb: dict[str, Any],
) -> None:
    cells = _code_cells(nb)
    install = _idx(cells, '"pip", "install", "-q", "-r"')
    uninstall = _idx(cells, '"pip", "uninstall", "-y"')
    check = _idx(cells, "get_linear_schedule_with_warmup")
    assert install < uninstall < check
    for needle in ("torchvision", "torchaudio"):
        assert needle in cells[uninstall]
    assert "check=False" in cells[uninstall]  # an absent package must not fail the cell
    first_import = min(
        i
        for i, s in enumerate(cells)
        if i > install and ("import transformers" in s or "from transformers" in s
                            or "intent_router" in s)
        and i not in (check,)
    )  # fmt: skip
    assert check < first_import
    md = "\n".join(_md(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "PIN_TORCH = False" in md and "peft" in md


def _run_assert_cell(nb: dict[str, Any], monkeypatch: pytest.MonkeyPatch, res: Any) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_k: Any) -> Any:
        calls.append(cmd)
        return res

    monkeypatch.setattr("subprocess.run", fake_run)
    exec(compile(_cell_with(nb, "get_linear_schedule_with_warmup"), "assert-cell", "exec"), {})  # noqa: S102
    assert calls[0][0] == sys.executable and calls[0][1] == "-c"
    assert "get_linear_schedule_with_warmup" in calls[0][2]


def test_assert_cell_prints_versions_on_success(
    nb: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ok = type("R", (), {"returncode": 0, "stdout": "2.11.0+cu128 12.8 5.18.0\n", "stderr": ""})()
    _run_assert_cell(nb, monkeypatch, ok)
    assert capsys.readouterr().out.strip() == "2.11.0+cu128 12.8 5.18.0"


def test_assert_cell_fails_loudly_with_stderr_tail_and_hint(
    nb: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    stderr = "x" * 3000 + "RuntimeError: operator torchvision::nms does not exist"
    bad = type("R", (), {"returncode": 1, "stdout": "", "stderr": stderr})()
    with pytest.raises(RuntimeError) as ei:
        _run_assert_cell(nb, monkeypatch, bad)
    msg = str(ei.value)
    assert "torchvision::nms does not exist" in msg and "Disconnect and delete runtime" in msg
    assert "torch/torchvision CUDA mismatch" in msg and len(msg) < 2500  # tail, not all 3 kB


def test_runner_never_uninstalls_from_the_local_venv(nb: dict[str, Any]) -> None:
    runner = _load_runner()
    src = _cell_with(nb, '"pip", "uninstall", "-y"')
    local = runner.localize(src, ROOT)
    assert '"-m", "pip", "uninstall"' not in local and "pip install skipped" in local
