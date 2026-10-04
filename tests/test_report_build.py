"""Report build: every number is rendered from results by code, nothing is hand-typed.

Checks: loader keys, strict rendering, no numeric literals in the template, forbidden-term hashes
(HTML and, when pypdf is importable, the PDF), no dataset rows, a number-by-number diff between the
rendered spans and the loader, provenance coverage (build/provenance.json, no superscripts in the
HTML), the new executive summary / approach / open-set sections, the Open-set improvement round
facts rendered from
the (now complete) result files with a loud failure when one is missing, determinism of the
charts and the main/appendix page gates.
"""

from __future__ import annotations

import copy
import hashlib
import html as htmllib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `report/` is a top-level package, not under src/

from report import build, charts, data  # noqa: E402

HASHES = ROOT / "tests" / "forbidden_term_hashes.txt"
DATASET = ROOT / "data" / "dataset.csv"
TEMPLATE = ROOT / "report" / "template.html.j2"
LATENCY = "results/serving/latency_colab.json"
MIN_MESSAGE_CHARS = 20  # shorter messages ("thanks") collide with ordinary report words

REQUIRED_KEYS = [
    "a.f1",
    "a.acc",
    "a.n",
    "a.B0.f1",
    "a.B1.f1",
    "a.B1.delta",
    "cv.f1_mean",
    "cv.f1_std",
    "oof.f1",
    "tb.headline.auroc",
    "tb.headline.strict_recall_95",
    "tb.headline.retention_95",
    "tb.m.maha_ft.auroc",
    "tb.m.msp.auroc",
    "tb.loco_n",
    "tb.holdout",
    "eda.nonen",
    "eda.claim",
    "err.oof45.fam_share_err",
    "err.oof45.fam_share_rows",
    "hist.v1.f1",
    "hist.v3.f1",
    "srv.thr",
    "srv.det",
    "q.decision",
    "dk.size",
    "llm.qwen3_zero.f1",
    "biz.r4.unknowns_caught_strict",
    "log.eval.final_models",
    "log.infer_total",
    "log.other.total",
    "aud2.kappa",
    "sc.best.f1",
    "trd.0.d",
    "swap.LD->PO.rate",
]


# --------------------------------------------------------------------------- shared fixtures


@pytest.fixture(scope="module")
def rep() -> data.Report:
    return data.load()


@pytest.fixture(scope="module")
def figs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("figs")
    figs = {k: ROOT / v for k, v in build.FIGURE_FILES.items()}
    figs.update(charts.make_all(out))
    return figs


@pytest.fixture(scope="module")
def page(rep: data.Report, figs: dict[str, Path]) -> str:
    return build.render_html(rep, figs)


def _visible_text(html: str) -> str:
    body = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    return htmllib.unescape(re.sub(r"<[^>]+>", " ", body))


def _forbidden() -> list[tuple[int, str]]:
    out = []
    for ln in HASHES.read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.startswith("#"):
            n, h = ln.split(":")
            out.append((int(n), h))
    return out


def _contains_term(text: str, length: int, digest: str) -> bool:
    flat = re.sub(r"[^a-z0-9]", "", text.lower())
    return any(
        hashlib.sha256(flat[i : i + length].encode()).hexdigest() == digest
        for i in range(len(flat) - length + 1)
    )


# --------------------------------------------------------------------------- (a) loader


def test_loader_returns_all_required_keys(rep: data.Report) -> None:
    flat = rep.flat()
    missing = [k for k in REQUIRED_KEYS if k not in flat]
    assert missing == []
    assert len(flat) > 500  # every table cell is a registered, provenanced key
    assert set(flat) == set(rep.metrics)


def test_formatting_helpers() -> None:
    assert data.fmt("f3", 0.94651606) == "0.947"
    assert data.fmt("ci", {"point": 0.5, "lo": 0.25, "hi": 0.75}) == "0.500 [0.250, 0.750]"
    assert (
        data.fmt("dci", {"point": -0.0141, "lo": -0.0373, "hi": 0.0091})
        == "-0.014 [-0.037, +0.009]"
    )
    assert data.fmt("pct", 0.124) == "12.4%"
    assert data.fmt("pval", 7.3e-4) == "7.3e-04"
    assert data.fmt("pm3", {"mean": 0.87, "std": 0.0084}) == "0.870 ± 0.008"
    assert data._segments("a.b[2].c") == ["a", "b", 2, "c"]
    assert data._path_str(["a", "x.y", 3]) == "$.a['x.y'][3]"


def test_metric_must_fit_its_kind() -> None:
    L = data.Loader()
    with pytest.raises(TypeError):
        L.add("bad", "results/final/track_a.json", "test.macro_f1", "f3")  # dict, not a float


def test_every_metric_has_provenance(rep: data.Report) -> None:
    rows = rep.provenance()
    assert len(rows) == len(rep.metrics)
    for r in rows:
        assert (ROOT / r["file"]).exists(), r["file"]
        assert r["path"].startswith(("$", "computed:")), r
        assert r["sha"] and r["sha_source"], r
    recorded = [r for r in rows if r["sha_source"].startswith("recorded")]
    assert recorded  # e.g. track_a.json records its own git sha
    head = rep.head["full"]
    assert head and all(len(r["sha"]) >= 7 for r in rows)


# --------------------------------------------------------------------------- (b) strict render


def test_renders_without_undefined(page: str) -> None:
    assert '<div class="box exec">' in page
    assert "Undefined" not in page


def test_template_missing_variable_is_an_error(rep: data.Report, figs: dict[str, Path]) -> None:
    env = build.make_env()
    with pytest.raises(Exception, match="undefined"):
        env.from_string("{{ no_such_variable }}").render()
    r = build.Renderer(rep, figs, build.DEFAULT_LINKS)
    with pytest.raises(KeyError):
        r.n("no.such.key")


# --------------------------------------------------------------------------- (c) no literals

_JINJA = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.S)
_LITERAL = re.compile(
    r"\d+(?:\.\d+)?\s?%"  # percentages
    r"|\d+\.\d+"  # decimals
    r"|(?<![\w.#&;-])\d+(?![\w])"  # standalone integers; F1, e5, v3 etc. are fine
)


def template_literals(src: str) -> list[str]:
    """Numeric literals left in the template's static text (Jinja, tags and comments removed)."""
    text = _JINJA.sub(" ", src)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)  # attributes (colspan, ...) are not prose numbers
    return _LITERAL.findall(htmllib.unescape(text))


def test_literal_scanner_catches_hand_typed_numbers() -> None:
    assert template_literals("<p>macro-F1 is 0.94</p>") == ["0.94"]
    assert template_literals("<p>about 25%</p>") == ["25%"]
    assert template_literals("<p>we ran 3 seeds</p>") == ["3"]
    assert template_literals("<p>{{ n('a.f1') }} model v3, B0, e5-base, F1</p>") == []


def test_template_has_no_hand_typed_numbers() -> None:
    assert template_literals(TEMPLATE.read_text(encoding="utf-8")) == []


def test_rendered_text_outside_spans_has_no_decimals(page: str) -> None:
    """Numbers reaching the page through data.py strings (not n()) would show up here."""
    body = re.sub(r'<span class="n" data-k="[^"]*">[^<]*</span>', " ", page)
    found = [x for x in re.findall(r"\d+\.\d+", _visible_text(body))]
    allowed = {"3.1"}  # the model tag llama3.1:8b, a name not a measurement
    assert [x for x in found if x not in allowed] == []


# --------------------------------------------------------------------------- (d) forbidden terms


def test_forbidden_term_check_detects_its_own_term() -> None:
    (n, h), *_ = _forbidden()
    assert n > 0 and len(h) == 64
    digest = hashlib.sha256(b"probe").hexdigest()
    assert _contains_term("Some PROBE text", 5, digest)
    assert not _contains_term("other text", 5, digest)


def test_no_company_name_in_rendered_html(page: str) -> None:
    text = _visible_text(page)
    for n, h in _forbidden():
        assert not _contains_term(text, n, h)


def test_no_company_name_in_report_sources() -> None:
    for f in [*(ROOT / "report").glob("*.py"), TEMPLATE, ROOT / "report" / "report.css"]:
        src = f.read_text(encoding="utf-8")
        for n, h in _forbidden():
            assert not _contains_term(src, n, h), f.name


def test_no_company_name_in_pdf_text(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf", reason="pypdf not installed: PDF text check skipped")
    pdf = ROOT / "build" / "report_draft.pdf"
    if not pdf.exists():
        pytest.skip("build/report_draft.pdf not built")
    text = "\n".join(p.extract_text() for p in pypdf.PdfReader(str(pdf)).pages)
    assert len(text) > 1000
    for n, h in _forbidden():
        assert not _contains_term(text, n, h)


# --------------------------------------------------------------------------- (e) no dataset rows


def _dataset_texts() -> list[str]:
    import pandas as pd

    return [t for t in pd.read_csv(DATASET)["text"].astype(str) if len(t) >= MIN_MESSAGE_CHARS]


def test_no_dataset_text_in_default_report(page: str) -> None:
    if not DATASET.exists():
        pytest.skip("data/dataset.csv not available (confidential, gitignored)")
    flat = re.sub(r"\s+", " ", _visible_text(page).lower())
    leaks = [t for t in _dataset_texts() if re.sub(r"\s+", " ", t.lower()) in flat]
    assert leaks == []  # (count only: never print dataset text)


def test_include_text_flag_adds_examples_and_detector_sees_them(
    rep: data.Report, figs: dict[str, Path]
) -> None:
    """Non-vacuity: the private variant contains dataset text, and the same detector flags it."""
    if not DATASET.exists():
        pytest.skip("data/dataset.csv not available (confidential, gitignored)")
    private = build.render_html(rep, figs, include_text=True)
    assert "PRIVATE: example texts" in private
    flat = re.sub(r"\s+", " ", _visible_text(private).lower())
    texts = _dataset_texts()
    assert any(re.sub(r"\s+", " ", t.lower()) in flat for t in texts)


def test_default_report_has_no_private_appendix(page: str) -> None:
    assert "PRIVATE" not in page


# --------------------------------------------------------------------------- (f) number diff

_SPAN = re.compile(r'<span class="n" data-k="([^"]*)">([^<]*)</span>')
_TOK = re.compile(r"\.(\w+)|\['([^']+)'\]|\[(\d+)\]")


def _resolve_path(obj: Any, path: str) -> Any:
    """Independent JSON-path resolver for '$.a.b[2]['x.y']' (not shared with the loader)."""
    assert path.startswith("$")
    for m in _TOK.finditer(path[1:]):
        key, qkey, idx = m.groups()
        obj = obj[int(idx)] if idx is not None else obj[key if key is not None else qkey]
    return obj


def test_every_rendered_number_equals_loader_value(page: str, rep: data.Report) -> None:
    spans = _SPAN.findall(page)
    assert len(spans) > 700
    keys = {k for k, _ in spans}
    for raw_key, shown in spans:
        key = htmllib.unescape(raw_key)
        assert key in rep.metrics, key
        m = rep.metrics[key]
        assert htmllib.unescape(shown) == data.fmt(m.kind, m.value), key
    # and the loader values equal what is in the results files, re-read independently
    checked = 0
    for k in keys:
        m = rep.metrics[htmllib.unescape(k)]
        if m.computed or " + " in m.path or "{" in m.path:
            continue
        raw = _resolve_path(json.loads((ROOT / m.file).read_text(encoding="utf-8")), m.path)
        if isinstance(m.value, dict) and "lo" in m.value:
            assert m.value["lo"] == raw["lo"] and m.value["hi"] == raw["hi"], m.key
        elif isinstance(m.value, dict):  # mean +/- std
            assert m.value["mean"] == raw["mean"] and m.value["std"] == raw["std"], m.key
        else:
            assert m.value == raw, m.key
        checked += 1
    assert checked > 400


def test_no_superscript_provenance_markers_and_no_provenance_table(page: str) -> None:
    """The PDF carries no footnote ids; the data-k attributes (HTML) are the provenance hook."""
    assert "<sup" not in page and 'class="fn"' not in page
    assert "<h2>Provenance</h2>" not in page and 'class="prov"' not in page
    assert len(_SPAN.findall(page)) > 700  # data-k spans are all still there


def test_placeholder_links_and_override(rep: data.Report, figs: dict[str, Path]) -> None:
    default = build.render_html(rep, figs)
    assert "[HF link — filled at publish time]" in default
    custom = build.render_html(rep, figs, links={"wandb": "https://example.invalid/run"})
    assert "https://example.invalid/run" in custom
    assert "[HF link" in custom  # other placeholders stay


# --------------------------------------------------------------------------- CPU latency (Colab)

PLACEHOLDER_SENTENCE = (
    "CPU batch-1 latency (ONNX fp32 and PyTorch fp32): measured in the Colab canonical run "
    "(environment stated); filled at publish time"
)


def _synthetic_colab(threads: list[int]) -> dict[str, Any]:
    """What `intent_router.latency_colab` writes (distinctive values so spans are unambiguous)."""

    def stats(p50: float, p95: float, th: int) -> dict[str, Any]:
        return {"threads": th, "n_calls": 222, "p50_ms": p50, "p95_ms": p95, "mean_ms": p50 + 2.0}

    lat = {
        "torch": {f"threads_{t}": stats(311.4 + t, 377.6 + t, t) for t in threads},
        "onnx_fp32": {f"threads_{t}": stats(111.4 + t, 177.6 + t, t) for t in threads},
    }
    return {
        "stage": "latency",
        "source": "colab",
        "split": "val",
        "n_rows": 74,
        "settings": {"warmup": 10, "repeats": 3, "threads": threads, "backends": list(lat)},
        "environment": {
            "torch": "2.11.0+cu128",
            "onnxruntime": "1.30.0",
            "cpu_model": "Synthetic CPU 9000 @ 9.99GHz",
            "os_cpu_count": threads[-1],
            "runtime_type": "Colab, CPU-only",
        },
        "latency": lat,
    }


def _patch_colab_latency(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any] | None) -> None:
    """Make the loader see (payload) or not see (None) results/serving/latency_colab.json."""
    orig_j, orig_exists = data.Loader.j, data.Loader.exists

    def j(self: data.Loader, rel: str) -> Any:
        return (
            copy.deepcopy(payload) if rel == LATENCY and payload is not None else orig_j(self, rel)
        )

    def exists(self: data.Loader, rel: str) -> bool:
        return (payload is not None) if rel == LATENCY else orig_exists(self, rel)

    monkeypatch.setattr(data.Loader, "j", j)
    monkeypatch.setattr(data.Loader, "exists", exists)


def test_latency_placeholder_shown_when_colab_file_absent(
    monkeypatch: pytest.MonkeyPatch, figs: dict[str, Path]
) -> None:
    _patch_colab_latency(monkeypatch, None)
    html = build.render_html(data.load(), figs)
    assert PLACEHOLDER_SENTENCE in html
    assert "Latency run: measured on Colab" not in html
    assert not any(k.startswith("srv.onnx") for k, _ in _SPAN.findall(html))
    text = _visible_text(html).lower()
    assert "not final" not in text and "contended" not in text and "idle check" not in text


def test_latency_values_rendered_from_colab_file_when_present(
    monkeypatch: pytest.MonkeyPatch, figs: dict[str, Path]
) -> None:
    _patch_colab_latency(monkeypatch, _synthetic_colab([1, 4]))
    rep = data.load()
    html = build.render_html(rep, figs)
    spans = {htmllib.unescape(k): v for k, v in _SPAN.findall(html)}
    assert spans["srv.onnx_fp32.t1.p50"] == "112" and spans["srv.onnx_fp32.t1.p95"] == "179"
    assert spans["srv.onnx_fp32.tn.p50"] == "115" and spans["srv.torch.tn.p95"] == "382"
    assert spans["srv.threads_n"] == "4" and spans["srv.cpu_cores"] == "4"
    assert spans["srv.cpu_model"] == "Synthetic CPU 9000 @ 9.99GHz"
    assert spans["srv.runtime"] == "Colab, CPU-only"
    assert "Latency run: measured on Colab" in html and PLACEHOLDER_SENTENCE not in html
    text = _visible_text(html).lower()
    assert "not final" not in text and "contended" not in text
    # every rendered value equals the loader value (which comes from the file)
    for k, shown in spans.items():
        assert htmllib.unescape(shown) == data.fmt(rep.metrics[k].kind, rep.metrics[k].value), k


def test_latency_single_vcpu_runtime_renders_one_thread_columns_only(
    monkeypatch: pytest.MonkeyPatch, figs: dict[str, Path]
) -> None:
    _patch_colab_latency(monkeypatch, _synthetic_colab([1]))
    html = build.render_html(data.load(), figs)
    keys = {htmllib.unescape(k) for k, _ in _SPAN.findall(html)}
    assert "srv.onnx_fp32.t1.p50" in keys and "srv.onnx_fp32.tn.p50" not in keys


def test_latency_file_with_bad_thread_list_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _synthetic_colab([2, 4])
    _patch_colab_latency(monkeypatch, bad)
    with pytest.raises(ValueError, match="threads must start with 1"):
        data.load()


def test_report_never_reads_the_local_contended_latency_file(rep: data.Report) -> None:
    assert all("latency_v1" not in m.file for m in rep.metrics.values())


# --------------------------------------------------------------------------- charts + PDF


def test_charts_are_deterministic(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    pa, pb = charts.make_all(a), charts.make_all(b)
    for k in pa:
        assert (
            hashlib.sha256(pa[k].read_bytes()).digest()
            == hashlib.sha256(pb[k].read_bytes()).digest()
        )


def test_test_log_counts_are_derived_from_the_logs(rep: data.Report) -> None:
    flat = rep.flat()
    lines = [
        json.loads(ln)
        for ln in (ROOT / "results/final/test_eval_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    ev = [r for r in lines if r["call_type"] == "evaluation"]
    assert flat["log.total"] == len(lines)
    assert flat["log.eval.final_models"] + flat["log.eval.baselines"] == len(ev)
    assert flat["log.infer_total"] == len(lines) - len(ev)
    assert flat["log.eval.distinct_final_fps"] == flat["log.eval.final_models"]  # one per model


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:  # noqa: BLE001 - any launch problem means: skip the PDF tests
        return False


@pytest.fixture(scope="module")
def chromium() -> None:
    if not _chromium_available():
        pytest.skip("Playwright Chromium not available")


def test_pdf_main_and_appendix_page_limits(chromium: None, tmp_path: Path) -> None:
    out = tmp_path / "report.pdf"
    s = build.build(out)
    assert build.MAX_MAIN_PAGES == 4 and build.MAX_APPENDIX_PAGES == 8
    assert 1 <= s["main_pages"] <= build.MAX_MAIN_PAGES
    # the gaps section is one block that must fit one page of content (print layout, measured)
    assert 0 < s["gaps_section_px"] <= s["gaps_section_max_px"] == round(build.GAPS_MAX_PX)
    assert 1 <= s["appendix_pages"] <= build.MAX_APPENDIX_PAGES  # appendix starts a new page
    assert s["total_pages"] == s["main_pages"] + s["appendix_pages"]
    assert out.read_bytes()[:4] == b"%PDF"
    prov = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert prov["metrics"] and prov["files"] and prov["keys_used_in_text"]
    assert set(prov["keys_used_in_text"]) <= {m["key"] for m in prov["metrics"]}
    html = out.with_suffix(".html").read_text(encoding="utf-8")
    assert "<sup" not in html and len(_SPAN.findall(html)) > 700  # data-k kept, no superscripts
    # latency_colab.json exists: its numbers are rendered and no latency placeholder remains
    lat = _rj(LATENCY)["latency"]["onnx_fp32"]["threads_1"]
    spans = {htmllib.unescape(k): v for k, v in _SPAN.findall(html)}
    assert spans["srv.onnx_fp32.t1.p50"] == f"{lat['p50_ms']:.0f}"
    assert spans["srv.onnx_fp32.t1.p95"] == f"{lat['p95_ms']:.0f}"
    assert PLACEHOLDER_SENTENCE not in html
    assert "measured in the Colab canonical run" not in _visible_text(html)
    assert forbidden_wording(_visible_text(html)) == []
    try:
        import pypdf
    except ImportError:  # the HTML scan above is the always-on check
        return
    text = "\n".join(p.extract_text() for p in pypdf.PdfReader(str(out)).pages)
    assert len(text) > 1000 and forbidden_wording(text) == []


def test_provenance_records_publish_mode_and_placeholder_links(tmp_path: Path) -> None:
    """A publisher relies on provenance.json `build` to tell a final build from a draft."""
    build.build(tmp_path / "r.pdf", make_pdf=False)
    draft = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))["build"]
    assert draft["publish"] is False and draft["include_text"] is False
    assert sorted(draft["placeholder_links"]) == sorted(build.DEFAULT_LINKS)
    real = {k: f"https://example.org/{k}" for k in build.DEFAULT_LINKS}
    build.build(tmp_path / "r.pdf", links=real, make_pdf=False, publish=True)
    final = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))["build"]
    assert final["publish"] is True and final["placeholder_links"] == []


def test_page_gates_fail_loudly(chromium: None, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="main text is"):
        build.build(tmp_path / "r.pdf", max_main_pages=1)
    with pytest.raises(SystemExit, match="appendix is"):
        build.build(tmp_path / "a.pdf", max_appendix_pages=0)


# --------------------------------------------------------------------------- new sections


def _between(page: str, start: str, end: str) -> str:
    i = page.index(start)
    return page[i : page.index(end, i)]


# F1, e5, 8B, 6a, D1 and compounds like batch-1 are names, not measurements
_BARE_NUMBER = re.compile(r"(?<![\w.])(?<!\w-)\d+(?:\.\d+)?(?![\w])")


def _numbers_outside_spans(fragment: str) -> list[str]:
    body = re.sub(r'<span class="n" data-k="[^"]*">[^<]*</span>', " ", fragment)
    return _BARE_NUMBER.findall(_visible_text(body))


def test_summary_is_five_lines_first_and_only_span_numbers(page: str) -> None:
    assert page.index('<header class="title">') < page.index('<div class="box exec">')
    box = _between(page, '<div class="box exec">', '<table class="head">')
    assert len(_SPAN.findall(box)) >= 15
    assert _numbers_outside_spans(box) == []
    text = re.sub(r"\s+", " ", _visible_text(box)).strip()
    text = text.removeprefix("Summary. ")
    assert len(re.findall(r"(?<=[A-Za-z0-9)])\.(?=\s|$)", text)) == 5  # five sentences
    assert "pre-registered" not in text and "fixed before the candidates were run" in text


def _repro_box(html: str) -> str:
    return _between(html, '<div class="box repro">', "<h2>Ablations and Bake-Off Detail</h2>")


def test_repro_box_lists_deliverables_with_placeholders_until_links_given(page: str) -> None:
    box = _repro_box(page)
    assert "<h2>Reproduction and Determinism</h2>" in page
    for ph in ("HF link", "W&amp;B link", "repo link", "Colab link"):
        assert ph in box  # the link placeholders stay until links are given
    assert "report.pdf" in box and "REPRODUCTION: PASS/FAIL" in box  # how to reproduce
    # latency_colab.json exists: no CPU-latency placeholder in the box; the verdict is rendered
    assert "CPU latency:" not in box and "measured in the Colab canonical run" not in box
    assert dict(_SPAN.findall(box))["rc.verdict"] == "REPRODUCTION: PASS"
    assert page.index("<h2>Reproduction and Determinism</h2>") < page.index(
        '<div class="box repro">'
    )


def test_repro_box_keeps_the_latency_placeholder_only_when_the_json_is_absent(
    monkeypatch: pytest.MonkeyPatch, figs: dict[str, Path]
) -> None:
    _patch_colab_latency(monkeypatch, None)
    box = _repro_box(build.render_html(data.load(), figs))
    assert "filled at publish time" in box.split("CPU latency:")[1]


def test_repro_box_takes_links_from_the_links_argument(
    rep: data.Report, figs: dict[str, Path]
) -> None:
    links = {k: f"https://example.invalid/{k}" for k in ("hf", "wandb", "repo", "colab")}
    box = _repro_box(build.render_html(rep, figs, links=links))
    assert all(v in box for v in links.values()) and "link — filled at publish time" not in box


def test_links_fall_back_per_key_when_null(rep: data.Report, figs: dict[str, Path]) -> None:
    """report/links.json has only some keys filled; the null ones keep their placeholder."""
    links = {"wandb": "https://example.invalid/w", "hf": None, "repo": "", "colab": None}
    assert build.resolve_links(links) == {
        **build.DEFAULT_LINKS,
        "wandb": "https://example.invalid/w",
    }
    box = _repro_box(build.render_html(rep, figs, links=links))
    assert "https://example.invalid/w" in box
    for ph in ("HF link", "repo link", "Colab link"):
        assert ph in box
    assert "W&amp;B link" not in box and "None" not in box


def test_real_links_json_builds_with_per_key_fallback(
    rep: data.Report, figs: dict[str, Path]
) -> None:
    links = json.loads((ROOT / "report" / "links.json").read_text(encoding="utf-8"))
    box = _repro_box(build.render_html(rep, figs, links=links, publish=True))
    for k, v in links.items():
        if k in build.DEFAULT_LINKS:
            assert (v in box) if v else (build.DEFAULT_LINKS[k].replace("&", "&amp;") in box)


def test_provenance_placeholders_use_per_key_fallback(tmp_path: Path) -> None:
    links = {"wandb": "https://example.org/w", "hf": None, "repo": None, "colab": None}
    build.build(tmp_path / "r.pdf", links=links, make_pdf=False, publish=True)
    b = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))["build"]
    assert b["placeholder_links"] == ["colab", "hf", "repo"]


def test_repro_box_determinism_numbers_equal_the_json(page: str, rep: data.Report) -> None:
    det = json.loads((ROOT / "results/final/determinism.json").read_text(encoding="utf-8"))
    rr = json.loads(
        (ROOT / "results_rerun/final_wandb/rerun_summary.json").read_text(encoding="utf-8")
    )
    spans = dict(_SPAN.findall(_repro_box(page)))
    assert spans["det.seed"] == str(det["seed"])
    assert spans["det.fp"] == det["fingerprint_run1"][:8] == rr["fingerprint_got"][:8]
    assert spans["det.test_logit_diff"] == f"{det['test']['logits']['max_abs_diff']:.1e}"
    assert spans["det.test_prob_diff"] == f"{det['test']['softmax_float32']['max_abs_diff']:.1e}"
    assert spans["det.bitwise"] == ("yes" if det["bitwise_identical"] else "no")
    assert spans["det.state"] == "yes" and det["state_dict_identical"] is True
    assert spans["rr.match"] == ("yes" if rr["fingerprint_match"] else "no")
    assert spans["rr.epochs"] == str(rr["epochs_run"])
    assert _numbers_outside_spans(_repro_box(page)) == []


def test_repro_commands_are_the_readme_commands(page: str, rep: data.Report) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    spans = dict(_SPAN.findall(_repro_box(page)))
    cmds = [htmllib.unescape(spans[f"repro.cmd{i}"]) for i in range(4)]
    assert len(cmds) == 4 and all(c in readme for c in cmds)
    assert "--stage train" in cmds[1] and "--stage evaluate" in cmds[2] and "headline" in cmds[3]
    nb = re.findall(r'data-k="nb\.[a-z_]+">(\d+)<', _repro_box(page))
    assert len(nb) == 5  # Colab runtime ESTIMATE figures come from the notebook header


def test_report_has_no_author_initials(page: str) -> None:
    initials = "G" + "G"  # spelled out so this file itself carries no author initials
    assert not re.search(rf"\b{initials}\b", _visible_text(page))
    for rel in ("report/template.html.j2", "report/links.example.json", "report/links.json"):
        assert not re.search(rf"\b{initials}\b", (ROOT / rel).read_text(encoding="utf-8")), rel


def test_approach_section_is_prose_with_measured_encoder_vs_llm_numbers(page: str) -> None:
    sec = _between(page, "Approach and Model Selection</h2>", "<h2>Data Characteristics")
    keys = {k for k, _ in _SPAN.findall(sec)}
    for needed in ("bk.e5_lr3e-05.f1", "llm.f1_min", "ft.gpu1.p50_ms", "ratio.usd", "q.size.torch"):
        assert needed in keys
    assert _numbers_outside_spans(sec) == []
    assert 250 < len(_visible_text(sec).split()) < 650  # about a page of prose, plus the table
    assert "CONTENDED" not in _visible_text(sec) and "NOT FINAL" not in _visible_text(sec)


def test_ratios_are_computed_from_the_results(rep: data.Report) -> None:
    flat = rep.flat()
    llm = json.loads((ROOT / "results/llm_baseline/summary.json").read_text(encoding="utf-8"))
    cfgs = llm["latency"]["llm"]["configs"]
    rows = llm["cost"]["rows"]
    ft_p50 = rows["ft|gpu_batch1"]["latency_p50_s"]
    assert flat["ratio.lat_gpu"] == pytest.approx(min(c["p50_s"] for c in cfgs.values()) / ft_p50)
    llm_usd = [v["usd_per_1k_messages"] for k, v in rows.items() if k.startswith("llm|")]
    ft_usd = rows["ft|gpu_batch1"]["usd_per_1k_messages"]  # same $/hr basis as the LLM rows
    assert flat["ratio.usd"] == pytest.approx(min(llm_usd) / ft_usd)


def test_oof_per_class_table_matches_stored_oof_score_and_sits_next_to_test(
    rep: data.Report, page: str
) -> None:
    flat, classes = rep.flat(), rep.ctx["classes"]
    f1s = [flat[f"oofpc.{c}.f1"] for c in classes]
    stored = json.loads((ROOT / "results/final/track_a.json").read_text(encoding="utf-8"))
    point = stored["cv"]["oof_probability_averaged_426_rows"]["macro_f1"]["point"]
    assert sum(f1s) / len(f1s) == pytest.approx(point, abs=1e-9)
    assert sum(flat[f"oofpc.{c}.support"] for c in classes) == flat["oof.n"]
    table = _between(page, '<table class="pc">', "</table>")
    assert table.count("<tr>") == len(classes) + 2  # two header rows
    for c in classes:  # one row per class: 4 test spans + 4 OOF spans
        row = _between(table, f"<tr><td>{c}</td>", "</tr>")
        assert len(_SPAN.findall(row)) == 8 and f'data-k="oofpc.{c}.recall"' in row
    block = _between(page, '<table class="pc">', '<table class="split">')
    assert 'data-k="oof.n"' in block  # the caption states the OOF row count


def test_test_exposure_paragraph_is_three_sentences_plus_reproduction_note(page: str) -> None:
    box = _between(page, "<b>Test-split exposure.</b>", "</div>").split("</b>", 1)[1]
    # three sentences on the logged evaluations, plus the author's reproduction-runs sentence
    assert len(re.findall(r"\.(?=\s|$)", _visible_text(box))) == 4
    assert (
        "Reproduction runs (the notebook, locally and on Colab) re-evaluate the bit-identical v1 "
        "weights into results_colab/, recorded separately from results/; they feed no selection "
        "and no reported Track A number."
    ) in _visible_text(box)
    assert _numbers_outside_spans(box) == [] and not _SPAN.findall(box)
    appendix = page[page.index('<section class="app">') :]
    for k in ("log.eval.final_models", "log.infer_total", "log.other.total", "log.total"):
        assert f'data-k="{k}"' in appendix  # the counts moved to the appendix


def test_next_steps_c_and_limitations_use_v3_and_neutral_swap_evidence(
    rep: data.Report, page: str
) -> None:
    ax = json.loads((ROOT / "results/phase4e/axes.json").read_text(encoding="utf-8"))
    c = ax["candidates"]["a1a3"]["axes"]["c"]
    assert rep.metrics["ax.a1a3.c"].text == data.fmt(
        "dci", {"point": c["improvement"], "lo": c["lo"], "hi": c["hi"]}
    )
    steps = _next_steps_detail(page)
    assert 'data-k="ax.a1a3.c"' in steps and "A1 cut flips but failed the guard" not in steps
    lim = _other_limitations(page)
    assert "swap.LD-&gt;REF.rate" in lim or "swap.LD->REF.rate" in lim
    assert "swap.PO-&gt;LD.rate" not in lim and "swap.PO->LD.rate" not in lim
    sw = json.loads((ROOT / "results/robustness/idswap.json").read_text(encoding="utf-8"))
    r = next(x for x in sw["results"] if x["source"] == "oof" and x["swap"] == "LD->REF")
    assert r["n_flips"] / r["n"] == pytest.approx(r["flip_rate"])
    assert rep.metrics["swap.LD->REF.rate"].text == data.fmt("pct", r["flip_rate"])


def test_open_set_section_has_four_part_structure(page: str) -> None:
    lim = _limitations(page)
    parts = ("Measured gap", "Root cause", "What was tried", "What would break it")
    order = [lim.index(x) for x in parts]
    assert order == sorted(order)
    assert _numbers_outside_spans(lim) == []


# --------------------------------------------------------------------------- Open-set improvement
# round (complete)


def _rj(rel: str) -> Any:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _limitations(html: str) -> str:
    """Appendix C: the open-set gap, the D1 to D4 facts, what was tried and what would break it."""
    return _between(
        html,
        f"<h2>{APPENDIX_HEADINGS[2]}</h2>",
        "<h2>Selection History",
    )


def _other_limitations(html: str) -> str:
    return _between(html, "<h3>Other limitations</h3>", '<div class="box gap3">')


def _next_steps_detail(html: str) -> str:
    """The full next-steps paragraph (Appendix K); the main text carries three bullets."""
    return html[html.index("<h3>Next steps, detail</h3>") :].split("</section>")[0]


def test_phase6a_values_equal_the_result_files(rep: data.Report) -> None:
    """Independent re-read of the JSON: the loader's 6a values are the files' values."""
    flat = rep.flat()
    d1 = _rj(data.P6A_D1)["holdouts"]["headline"]
    ft = d1["feature_sets"]["v1_finetuned"]["all_rows"]
    assert flat["p6a.d1.v1_finetuned"]["point"] == ft["auroc_mean_of_folds"]
    assert flat["p6a.d1.v1_finetuned"]["lo"] == ft["auroc_mean_of_folds_ci"][0]
    for fid, _ in data.D1_SETS:
        assert f"p6a.d1.{fid}" in flat and f"p6a.d1.{fid}.unseen" in flat
    y = d1["unsupervised_finetuned_mahalanobis"]["phase3_results"]["mean_3_seeds"]
    assert flat["p6a.d1.maha"] == y
    # HEADLINE pool: unseen known rows only (all-rows includes rows the encoder trained on)
    un = d1["feature_sets"]["v1_finetuned"]["unseen_known_only"]
    assert flat["p6a.d1.v1_finetuned.unseen"]["point"] == un["auroc_mean_of_folds"]
    assert flat["p6a.d1.n_pos"] == un["n_pos"] and flat["p6a.d1.n_neg"] == un["n_neg"]
    assert flat["p6a.d1.headroom"] == pytest.approx(un["auroc_mean_of_folds"] - y)
    d2 = _rj(data.P6A_D2)
    assert flat["p6a.d2.reversal"] is d2["ranking_reversal"] is False
    for rec in ("v1", "a1a3", "a1"):
        for mode in ("raw", "neutral"):
            cell = d2["results"][rec][mode]["headline"]["strict_rej95"]
            assert flat[f"p6a.d2.{rec}.{mode}.strict_rej95"]["point"] == cell["point"]
    ids = d2["id_prefix_counts"]["headline"]
    assert flat["p6a.d2.ids.unknown.n_rows_with_id"] == ids["unknown"]["n_rows_with_id"]
    d3 = _rj(data.P6A_D3)["pooled_over_seeds"]
    assert flat["p6a.d3.rej.sd"] == d3["rejection_unknown"]["sd"]
    assert flat["p6a.d3.ret.hi"] == d3["retention_known"]["p97_5"]
    assert flat["p6a.d3.level"] == pytest.approx(0.95)
    d4 = _rj(data.P6A_D4)
    slope = d4["auroc"]["slope_vs_log2_fraction"]
    assert flat["p6a.d4.auroc.slope"]["point"] == slope["slope_per_doubling"]
    assert flat["p6a.d4.row0.n"] == d4["n_train_rows"]["0.25|42"]
    assert flat["p6a.d4.row3.n"] == d4["n_train_rows"]["1.0|42"]


def test_phase6a_selection_and_pre_ship_values(rep: data.Report) -> None:
    flat = rep.flat()
    sel, unguarded = _rj(data.P6A_SEL), _rj(data.P6A_SEL_UNG)
    assert flat["p6a.sel.chosen"] == sel["chosen"] == "i1b"
    assert flat["p6a.sel.unguarded.chosen"] == unguarded["chosen"] == "i6b"
    assert flat["p6a.sel.unguarded.n_eligible"] == len(unguarded["eligible"])
    axes = _rj(data.P6A_AXES)
    b = axes["candidates"]["i6b"]["axes"]["b"]["modes"]
    assert flat["p6a.sel.i6b.ret_raw"] == b["raw"]["retention"]["candidate"]
    assert flat["p6a.sel.i6b.ret_neu"] == b["neutral"]["retention"]["candidate"]
    assert flat["p6a.ax.ref.ret_raw"] == axes["ref"]["retention"]["raw"]
    # the collapse: cross-fitted threshold far above the calibration one, mean over raw DEV runs
    thr = [
        _rj(f.relative_to(ROOT).as_posix())["per_retention"]["95"]
        for f in sorted((ROOT / data.P6A / "i6b").glob("s*/views/raw/*.thr.json"))
    ]
    assert flat["p6a.sel.i6b.thr_n"] == len(thr) > 0
    assert flat["p6a.sel.i6b.thr_cf"] == pytest.approx(
        sum(t["crossfit_threshold"] for t in thr) / len(thr)
    )
    assert flat["p6a.sel.i6b.thr_cf"] > flat["p6a.sel.i6b.thr_cal"]
    conf = _rj(data.P6A_CONF)
    cell = conf["paired_delta_candidate_minus_ref"]["raw"]["headline"]["rej95"]
    shipped = flat["p6a.ship.raw.headline.rej"]
    assert shipped == {"point": cell["mean_delta"], "lo": cell["lo"], "hi": cell["hi"]}
    assert shipped["hi"] < 0  # CI entirely below zero: the pre-ship check blocks i1b
    assert flat["p6a.ref.epoch"] == _rj(data.P6A_REF_EPOCH)["e_star"] == 8
    assert flat["hist.sens.v1_e"] == 9  # the shipped v1; ref differs, which is the disclosure
    assert flat["p6a.ref.rej"]["point"] == conf["ref"]["raw"]["headline"]["ci95"]["rej95"]["point"]
    i4 = axes["candidates"]["i4a"]["axes"]
    assert (
        flat["p6a.ax.i4a.c"] == i4["c"]["delta"]
        and flat["p6a.ax.i4a.a_ci"]["point"] == i4["a"]["delta"]
    )
    assert flat["p6a.n2.diff"] == _rj(data.P6A_N2)["values"]["diff_mean_of_folds"]


def test_phase6a_log_counts_are_derived_from_the_log_files(rep: data.Report) -> None:
    flat = rep.flat()
    total = 0
    for rel in data.P6A_LOGS:
        lines = [ln for ln in (ROOT / rel).read_text(encoding="utf-8").splitlines() if ln.strip()]
        total += len(lines)
        cts = {json.loads(ln)["call_type"] for ln in lines}
        assert cts and all(f"p6a.log.{ct}" in flat for ct in cts)
    assert flat["p6a.log.total"] == total
    assert sum(flat[r["key"]] for r in rep.ctx["p6a"]["logs"]["rows"]) == total


def test_phase6a_renders_with_the_exact_d1_framing_and_no_stub(page: str) -> None:
    main = page.split("</main>")[0]
    assert "pending" not in _visible_text(main).lower()
    lim = _limitations(page)
    text = re.sub(r"\s+", " ", _visible_text(lim))
    text = re.sub(r"\s+([),;.])", lambda m: m.group(1), text)  # tags leave a space before marks
    # the author's framing, verbatim apart from the two rendered numbers
    assert re.search(
        r"Held-out intents are linearly separable in the fine-tuned feature space \(oracle AUROC "
        r"[0-9.]+ \[[0-9., ]+\] on the unseen-known pool\), yet unsupervised scoring reaches "
        r"[0-9.]+; the gap, \+[0-9.]+, is a missing negative training signal, which verified "
        r"hard-negative OOS data \(next step\) supplies\.",
        text,
    )
    assert "inflates the all-rows figure" in text and "D1 table below" in text  # pools
    assert "upper bound on separability, not achievable rejection" in text
    assert "not a significant gap" in text and "The v1 decision stands" in text  # D2 wording
    assert "thresholds must be calibrated on the shipped model's own held-out data" in text  # D3
    keys = {k for k, _ in _SPAN.findall(lim)}
    wanted = (
        "p6a.d1.v1_finetuned.unseen", "p6a.d1.maha", "p6a.d1.headroom",
        "p6a.d2.reversal", "p6a.d2.delta.raw.strict_rej95", "p6a.d2.delta.neutral.strict_rej95",
        "p6a.d3.ret.sd", "p6a.d3.rej.sd", "p6a.d3.rej.lo", "p6a.d3.cf.ret", "p6a.d3.cf.rej",
        "p6a.d4.auroc.slope", "p6a.d4.rej.slope", "p6a.d4.est0.auroc",
        "p6a.d4.sat",
    )  # fmt: skip
    assert [k for k in wanted if k not in keys] == []
    assert _numbers_outside_spans(lim) == []


def test_phase6a_selection_history_is_three_sentences_plus_the_i4_soups_line(
    rep: data.Report, page: str
) -> None:
    sec = _between(page, "<h2>Selection History and the v1/v3 Decision</h2>", "<h2>Robustness</h2>")
    para = sec[sec.index("<p><b>Open-set improvement round.</b>") :].split("</p>")[0]
    text = re.sub(r"\s+", " ", _visible_text(para)).strip()
    text = re.sub(r"\s+([),;.])", lambda m: m.group(1), text)
    # sentence ends are a letter or closing bracket followed by '. ' (decimals are not)
    body = text.removeprefix("Open-set improvement round. ").rstrip(".")
    sentences = re.split(r"(?<=[A-Za-z)])\.\s+", body)
    assert len(sentences) == 4, sentences  # three selection sentences, then the I4 / soups line
    assert "i6b" in sentences[0] and "collapsed" in sentences[0]
    assert (
        "A retention check was added after the first selection picked a candidate that kept"
        in sentences[1]
        and "of known messages" in sentences[1]
    )
    assert "i1b" in sentences[1]
    assert "held back" in sentences[2] and sentences[2].endswith("v1 ships")
    assert "I4" in sentences[3] and "soups" in sentences[3]
    keys = {k for k, _ in _SPAN.findall(para)}
    for k in (
        "p6a.sel.i6b.ret_raw",
        "p6a.sel.i6b.ret_neu",
        "p6a.ship.raw.headline.rej",
        "p6a.ax.i4a.c",
        "p6a.ax.i4b.c",
        "p6a.ax.i4a.a_ci",
        "p6a.ax.i3u.a",
        "p6a.ax.i3g.a",
    ):
        assert k in keys
    assert _numbers_outside_spans(para) == []
    assert rep.metrics["p6a.ship.raw.headline.rej"].text == "-0.033 [-0.062, -0.008]"


def test_phase6a_d4_extrapolation_is_an_independent_log_linear_estimate(
    rep: data.Report, page: str
) -> None:
    d4 = _rj(data.P6A_D4)["auroc"]["slope_vs_log2_fraction"]
    a, b = d4["intercept"], d4["slope_per_doubling"]
    lo, hi = d4["ci"]
    flat = rep.flat()
    for i, mult in enumerate((2,)):  # the 4x extrapolation was removed on review
        est = flat[f"p6a.d4.est{i}.auroc"]
        assert est["point"] == pytest.approx(min(1.0, a + b * math.log2(mult)))
        assert est["lo"] == pytest.approx(min(1.0, a + lo * math.log2(mult)))
        assert est["hi"] == pytest.approx(min(1.0, a + hi * math.log2(mult)))
        assert 0.0 < est["lo"] <= est["point"] <= est["hi"] <= 1.0
        assert flat[f"p6a.d4.est{i}.n"] == mult * flat["p6a.d4.row3.n"]
    assert flat["p6a.d4.sat"] == pytest.approx((1.0 - a) / b)
    assert "p6a.d4.est1.auroc" not in flat and "p6a.d4.est1.n" not in flat
    lim = _limitations(page)
    assert "ESTIMATE beyond" in lim and "log-linear" in lim
    assert "4&times;" not in lim and "(4)" not in _visible_text(lim)
    assert "seed CI is coarse" in lim


def test_phase6a_next_steps_carry_the_measured_tradeoffs(page: str) -> None:
    steps = _next_steps_detail(page)
    keys = {k for k, _ in _SPAN.findall(steps)}
    needed = {
        "p6a.d1.v1_finetuned.unseen", "p6a.d1.maha",  # N1 with D1 evidence
        "p6a.ax.i4a.c", "p6a.ax.i4b.c", "p6a.ax.i4a.a_ci", "p6a.ax.i4b.a_ci",  # I4 trade-off
        "p6a.n2.large", "p6a.n2.base", "p6a.n2.diff", "p6a.n2.margin", "p6a.n2.run",  # e5-large
        "p6a.ax.i1a.b_raw_rej", "p6a.ax.i6a.ret_raw",
        "p6a.ax.i6b.ret_raw",
    }  # fmt: skip
    assert sorted(needed - keys) == []
    assert "not worth it under D1" in _visible_text(steps)
    order = [
        steps.index(x) for x in ("N1,", "More labelled data", "I4, ID-consistency", "N2, e5-large")
    ]
    assert order == sorted(order)
    assert 'data-k="p6a.d4.auroc.slope"' in steps
    assert _numbers_outside_spans(steps) == []


def test_phase6a_appendix_has_tables_logs_and_axes_rows(rep: data.Report, page: str) -> None:
    app = page[page.index('<section class="app">') :]
    sec = _between(
        app,
        f"<h2>{APPENDIX_HEADINGS[2]}</h2>",
        "<h2>Robustness</h2>",
    )
    ax = rep.ctx["p6a"]["ax"]
    assert len(ax["rows"]) == 9 and len(ax["not_formed"]) == 3  # i3i4, i3i4i1, i3i4i1i6 not formed
    for r in ax["rows"] + ax["not_formed"]:
        assert r["name"] in sec
    assert sec.count("<tr") > 30
    for k in ("p6a.log.total", "p6a.log.n_files", "p6a.ref.rej", "p6a.ax.guard_margin"):
        assert f'data-k="{k}"' in app
    for r in rep.ctx["p6a"]["logs"]["rows"]:
        assert f'data-k="{r["key"]}"' in app
    assert "no new classifier test evaluation" in _visible_text(app)


def test_missing_phase6a_file_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Open-set improvement round is complete: no stubs, so an absent result file stops the build.
    """
    monkeypatch.setattr(data, "P6A_D4", "results/phase6a/diag/does_not_exist.json")
    with pytest.raises(FileNotFoundError):
        data.load()


def test_phase6a_file_with_wrong_schema_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_j = data.Loader.j

    def j(self: data.Loader, rel: str) -> Any:
        return {"unexpected": 1} if rel == data.P6A_D3 else orig_j(self, rel)

    monkeypatch.setattr(data.Loader, "j", j)
    with pytest.raises(KeyError):
        data.load()


# --------------------------------------------------------------------------- requirements


def test_playwright_is_pinned_to_the_installed_version() -> None:
    from importlib.metadata import version

    pins = [
        ln.strip()
        for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if ln.lower().startswith("playwright==")
    ]
    assert pins == [f"playwright=={version('playwright')}"]


def test_next_steps_list_n1_and_n2_with_measured_evidence_and_no_external_numbers(
    page: str,
) -> None:
    steps = _next_steps_detail(page)
    assert "N1" in steps and "unanimously" in steps and "K-plus-one" in steps
    assert "N2" in steps and "e5-large" in steps
    assert "outside these measurements" in steps  # outside OE results are flagged
    for k in ("hist.guard.a2", "hist.a2.auroc", "aud2.reliable"):
        assert f'data-k="{k}"' in steps


# ------------------------------------------------------------- review fixes: flags and wording
def test_publish_flag_drops_the_draft_wording_and_default_keeps_it(
    rep: data.Report, figs: dict[str, Path], page: str
) -> None:
    pub = build.render_html(rep, figs, publish=True)
    assert "DRAFT for private review" in page and "(draft)" in page
    for phrase in ("DRAFT", "private review", "(draft)"):
        assert phrase not in pub
    assert "<title>intent-router report</title>" in pub


def test_publish_flag_refuses_to_mix_with_include_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["report.build", "--publish", "--include-text", "--no-pdf"])
    with pytest.raises(SystemExit, match="mutually exclusive"):
        build.main()


def _next_steps_text(html: str) -> str:
    steps = _next_steps_detail(html)
    return re.sub(r"\s+", " ", _visible_text(steps)).strip()


def test_next_steps_latency_item_is_dropped_automatically_when_the_colab_json_exists(
    monkeypatch: pytest.MonkeyPatch, figs: dict[str, Path]
) -> None:
    _patch_colab_latency(monkeypatch, None)
    without = _next_steps_text(build.render_html(data.load(), figs))
    assert without.startswith("Next steps, detail (a) Run the Colab latency cell")
    assert re.findall(r"\(([a-h])\) ", without) == list("abcdef")
    _patch_colab_latency(monkeypatch, _synthetic_colab([1, 2]))
    with_json = _next_steps_text(build.render_html(data.load(), figs))
    assert "Colab latency cell" not in with_json and "Re-run" not in with_json
    assert with_json.startswith("Next steps, detail (a) N1")  # letters restart: no gap
    assert re.findall(r"\(([a-h])\) ", with_json) == list("abcde")


def test_n2_reports_both_pools_without_mixing(rep: data.Report) -> None:
    flat = rep.flat()
    fs = _rj(data.P6A_D1)["holdouts"]["headline"]["feature_sets"]
    big = fs["frozen_e5_large"]["unseen_known_only"]["auroc_mean_of_folds"]
    small = fs["frozen_e5_base"]["unseen_known_only"]["auroc_mean_of_folds"]
    assert flat["p6a.n2.diff_unseen"] == pytest.approx(big - small)
    assert flat["p6a.n2.diff_unseen"] < flat["p6a.n2.margin"]  # the cut still holds on this pool


def test_committed_links_json_leaves_no_link_placeholder_and_out_stays_in_its_directory(
    tmp_path: Path,
) -> None:
    """Fix 7: --links report/links.json fills every §13 deliverable; --out keeps outputs local."""
    links = json.loads((ROOT / "report" / "links.json").read_text(encoding="utf-8"))
    out = tmp_path / "sub" / "report_publish.pdf"
    out.parent.mkdir()
    build.build(out, links=links, make_pdf=False, publish=True)
    b = json.loads((out.parent / "provenance.json").read_text(encoding="utf-8"))["build"]
    assert b["placeholder_links"] == []
    box = _repro_box(out.with_suffix(".html").read_text(encoding="utf-8"))
    for k in build.DEFAULT_LINKS:
        assert links[k] in box.replace("&amp;", "&")
    assert "link — filled at publish time" not in box


# ------------------------------------------------- Colab latency recommendation, cost, reproduction
RUN1 = "results/colab/reproduction_run1.json"
RUN2 = "results/colab/reproduction_run2.json"


def _patch_json(monkeypatch: pytest.MonkeyPatch, rel: str, edit: Any) -> None:
    """Let the loader see `rel` after `edit(doc)` mutated a deep copy (synthetic mismatch tests)."""
    orig = data.Loader.j

    def j(self: data.Loader, r: str) -> Any:
        doc = copy.deepcopy(orig(self, r))
        if r == rel:
            edit(doc)
        return doc

    monkeypatch.setattr(data.Loader, "j", j)


def _deploy(html: str) -> str:
    """Appendix G: serving path, quantization, the full CPU latency table, recommendation, cost."""
    return _between(
        html, "<h2>Latency, Quantization and Cost</h2>", "<h2>Reproduction and Determinism</h2>"
    )


def test_cpu_recommendation_quotes_p50_and_p95_from_the_latency_json(page: str) -> None:
    lat = _rj(LATENCY)["latency"]
    sec = _deploy(page)
    spans = {htmllib.unescape(k): v for k, v in _SPAN.findall(sec)}
    text = re.sub(r"\s+", " ", _visible_text(sec))
    assert "CPU recommendation: ONNX fp32 at one thread for batch-1 serving" in text
    n_thr = _rj(LATENCY)["settings"]["threads"][-1]
    for be, short in (("onnx_fp32", "onnx_fp32"), ("torch", "torch")):
        for slot, th in (("t1", 1), ("tn", n_thr)):
            for stat in ("p50", "p95"):
                assert (
                    spans[f"srv.{short}.{slot}.{stat}"]
                    == f"{lat[be][f'threads_{th}'][f'{stat}_ms']:.0f}"
                )
    assert spans["srv.threads_n"] == str(n_thr)
    # the author's claim, checked against the data rather than assumed: more threads are slower
    for be in lat:
        assert lat[be][f"threads_{n_thr}"]["p50_ms"] > lat[be]["threads_1"]["p50_ms"], be
        assert lat[be][f"threads_{n_thr}"]["p95_ms"] > lat[be]["threads_1"]["p95_ms"], be
    assert "slower than one thread" in text and "also slower" in text
    assert "measured on Colab" in text and "Shared cloud vCPUs" in text  # environment line
    assert spans["srv.cpu_model"].startswith("Intel") and spans["srv.torch_ver"]


def test_thread_claim_follows_the_data_when_more_threads_are_not_slower(
    monkeypatch: pytest.MonkeyPatch, figs: dict[str, Path]
) -> None:
    doc = _synthetic_colab([1, 4])
    doc["latency"]["onnx_fp32"]["threads_4"]["p50_ms"] = 50.0  # faster than one thread (112.4)
    doc["latency"]["onnx_fp32"]["threads_4"]["p95_ms"] = 60.0
    _patch_colab_latency(monkeypatch, doc)
    text = re.sub(r"\s+", " ", _visible_text(_deploy(build.render_html(data.load(), figs))))
    assert "slower than one thread" not in text
    assert "thread count is not settled" in text


def test_cpu_cost_is_an_estimate_computed_from_the_p50_with_stated_assumptions(
    page: str, rep: data.Report
) -> None:
    lat = _rj(LATENCY)["latency"]
    sec = _deploy(page)
    spans = {htmllib.unescape(k): v for k, v in _SPAN.findall(sec)}
    price = rep.flat()["srv.cpu_usd_hr"]
    assert price == 0.05 and spans["srv.cpu_usd_hr"] == "$0.05"
    p50 = lat["onnx_fp32"]["threads_1"]["p50_ms"]
    mean = lat["onnx_fp32"]["threads_1"]["mean_ms"]
    # 1k sequential batch-1 calls take p50 ms x 1000 = p50 seconds on one vCPU
    assert spans["srv.onnx_fp32.usd1k_p50"] == f"${p50 / 3600 * 0.05:.5f}"
    assert spans["srv.onnx_fp32.usd1k_mean"] == f"${mean / 3600 * 0.05:.5f}"
    para = _between(sec, "<b>CPU cost per 1k messages:</b>", "</p>")
    assert 'class="tag e">estimate' in sec.split("<b>CPU cost per 1k messages:</b>")[0][-120:]
    text = re.sub(r"\s+", " ", _visible_text(para))
    assert "sequential batch-1" in text and "one vCPU" in text and "p50" in text
    assert "mean" in text  # the p50-versus-mean choice is stated
    assert _numbers_outside_spans(para) == []


def test_no_latency_placeholder_remains_and_next_steps_a_is_gone(page: str) -> None:
    text = _visible_text(page)
    for phrase in (
        "CPU latency is measured in the Colab canonical run",
        "CPU latency of the encoder is not compared here",
        "Run the Colab latency cell",
    ):
        assert phrase not in text, phrase
    steps = _next_steps_text(page)
    assert steps.startswith("Next steps, detail (a) N1")
    assert re.findall(r"\(([a-h])\) ", steps) == list("abcde")


def test_encoder_vs_llm_paragraph_states_both_latency_bases_and_forms_no_cpu_gpu_ratio(
    page: str,
) -> None:
    sec = _between(page, "<b>Why an encoder fine-tune and not an LLM.</b>", "</p>")
    text = re.sub(r"\s+", " ", _visible_text(sec))
    spans = {htmllib.unescape(k): v for k, v in _SPAN.findall(sec)}
    gpu = _rj("results/llm_baseline/ft_latency.json")["gpu_name"]
    assert "No CPU-versus-GPU ratio is formed" in text and spans["llm.gpu_name"] == gpu
    assert "shared Colab vCPUs" in text and "no LLM was timed on CPU" in text
    # the only latency ratio quoted is the like-for-like GPU one
    assert "ratio.lat_gpu" in spans and not any(k.startswith("ratio.cpu") for k in spans)


def test_reproduction_table_values_equal_the_run2_json(page: str) -> None:
    run2 = _rj(RUN2)
    box = _repro_box(page)
    spans = {htmllib.unescape(k): htmllib.unescape(v) for k, v in _SPAN.findall(box)}
    assert spans["rc.verdict"] == run2["verdict_recomputed"] == "REPRODUCTION: PASS"
    rows = run2["table_recomputed"]
    assert len(rows) == 6 and spans["rc.n_rows"] == "6" and spans["rc.n_pass"] == "6"
    for i, r in enumerate(rows):
        assert spans[f"rc.row{i}.metric"] == r["metric"]
        assert spans[f"rc.row{i}.fresh"] == f"{r['fresh']:.4f}"
        assert spans[f"rc.row{i}.committed"] == f"{r['committed']:.4f}"
        assert spans[f"rc.row{i}.delta"] == f"{r['delta']:+.4f}"
        assert spans[f"rc.row{i}.tol"] == r["tolerance"].replace("+-", "\u00b1")
        assert spans[f"rc.row{i}.result"] == r["result"]
    assert spans["rc.fp2"] == run2["fresh_fingerprint"][:8]
    assert spans["rc.fp_committed"] == run2["committed_fingerprint"][:8]
    assert _numbers_outside_spans(box) == []


def test_msp_rejection_edge_is_disclosed_with_counts_from_the_json(page: str) -> None:
    run2 = _rj(RUN2)
    row = next(
        r for r in run2["table_recomputed"] if r["metric"].startswith("Track B msp rejection")
    )
    assert abs(row["delta"]) == float(row["tolerance"].lstrip("+-"))  # exactly at the edge
    seeds = run2["trackb_per_seed_msp_vs_maha_ft"]
    fresh_n = sum(v["n_unknown_rejected"] for v in seeds["msp"].values())
    unk_n = sum(v["n_unknown"] for v in seeds["msp"].values())
    committed = _rj("results/trackb/headline.json")["methods"]["msp"]["strict_rejection_recall"]
    committed_n = round(
        sum(committed["values"]) * _rj("results/trackb/headline.json")["n_eval_unknown"]
    )
    assert (fresh_n, unk_n, committed_n) == (71, 240, 47)
    box = _repro_box(page)
    spans = {htmllib.unescape(k): v for k, v in _SPAN.findall(box)}
    assert spans["rc.edge.fresh_n"] == str(fresh_n) and spans["rc.edge.unk_n"] == str(unk_n)
    assert spans["rc.edge.committed_n"] == str(committed_n)
    assert spans["rc.edge.cal_n"] == str(seeds["msp"]["s42"]["n_cal_known"])
    text = re.sub(r"\s+", " ", _visible_text(box))
    assert "sits exactly at its tolerance" in text and "+0.1000" in text
    assert "Not a code bug" in text and "results/colab/reproduction_run2.json" in text
    assert "not in the public repository" not in text and "working repository" not in text
    assert "arithmetic coincidence" in text  # the equal 0.2958 on the Mahalanobis row


def test_colab_determinism_claim_is_rendered_from_a_programmatic_comparison(page: str) -> None:
    r1, r2 = _rj(RUN1), _rj(RUN2)
    # the claim is only printed because these hold; the test re-derives them independently
    assert r1["fresh_fingerprint"] == r2["fresh_fingerprint"]
    assert [x["fresh"] for x in r1["table_recomputed"]] == [
        x["fresh"] for x in r2["table_recomputed"]
    ]
    assert r2["fresh_fingerprint"] != r2["committed_fingerprint"]
    text = re.sub(r"\s+", " ", _visible_text(_repro_box(page)))
    fp, fpc = r2["fresh_fingerprint"][:8], r2["committed_fingerprint"][:8]
    assert "identical fingerprint " + fp in text and "identical metrics" in text
    assert "bitwise deterministic within a GPU type" in text
    flat_text = text.replace(" \u2026", "\u2026")  # a span precedes each ellipsis
    assert "differs only across GPU types" in text and f"{fp}\u2026 against {fpc}" in flat_text
    assert "Tesla T4 against NVIDIA GeForce RTX 3070" in text and "within tolerance" in text
    assert "results/colab/reproduction_run1.json" in text  # the run files are public paths
    assert "results/colab/reproduction_run2.json" in text and "working repository" not in text
    assert 'data-k="det.fp"' in _repro_box(page)  # the 3070 determinism evidence is kept
    assert 'data-k="rr.fp"' in _repro_box(page)  # ... and the W&B re-run


def test_header_points_the_key_map_at_the_public_provenance_file(page: str) -> None:
    text = re.sub(r"\s+", " ", _visible_text(page))
    assert "report/provenance.json in the public repository" in text
    assert "build/provenance.json" not in text


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.__setitem__("fresh_fingerprint", "0" * 64),
        lambda d: d["table_recomputed"][0].__setitem__("fresh", 0.5),
        lambda d: d["table_recomputed"][1].__setitem__("result", "FAIL"),
    ],
    ids=["fingerprint-differs", "metric-differs", "row-fails"],
)
def test_colab_determinism_claim_is_not_printed_when_the_runs_differ(
    monkeypatch: pytest.MonkeyPatch, figs: dict[str, Path], edit: Any
) -> None:
    _patch_json(monkeypatch, RUN1, edit)
    rep = data.load()
    assert rep.ctx["repro_runs_identical"] is False
    box = _repro_box(build.render_html(rep, figs))
    text = re.sub(r"\s+", " ", _visible_text(box))
    assert "bitwise deterministic within a GPU type" not in text
    assert "identical fingerprint" not in text
    assert "REPRODUCTION: PASS" in text  # the run-2 table itself is still shown


# ------------------------------------------------------- Gaps, fixes and the production plan

GAP_TITLE = "The 3–4-hour path."


def _gaps_section(html: str) -> str:
    return _between(html, '<section class="gaps"', "</section>")


def _gap_rows(sec: str) -> list[list[str]]:
    table = _between(sec, '<table class="gaps">', "</table>")
    rows = re.findall(r"<tr>(.*?)</tr>", table, flags=re.S)
    return [re.findall(r"<td>(.*?)</td>", r, flags=re.S) for r in rows if "<td>" in r]


def _spans_of(fragment: str) -> dict[str, str]:
    return {htmllib.unescape(k): htmllib.unescape(v) for k, v in _SPAN.findall(fragment)}


def test_gaps_table_sits_in_the_production_section(page: str) -> None:
    i_prod = page.index("<h2>Deployment and Monitoring</h2>")
    i_gaps = page.index('<section class="gaps"')
    i_next = page.index("<h2>Next Steps</h2>")
    assert i_prod < i_gaps < i_next  # the table moved into section 6
    assert page.count('<section class="gaps"') == 1
    assert "Gaps, fixes and the production plan" not in _visible_text(page)  # title gone
    assert "<h2>" not in _gaps_section(page)


def test_gaps_table_has_four_rows_and_four_columns(page: str) -> None:
    sec = _gaps_section(page)
    rows = _gap_rows(sec)
    assert len(rows) == 4 and all(len(r) == 4 for r in rows)
    heads = re.findall(r"<th>(.*?)</th>", _between(sec, '<table class="gaps">', "</table>"))
    assert heads == [
        "Limitation, with its measured number",
        "Evidence",
        "Mitigation in production today",
        "Next step",
    ]
    names = [
        "Open-set rejection",
        "Fine-tune versus frozen B1",
        "ID-prefix shortcut",
        "Small synthetic data",
    ]
    for r, name in zip(rows, names, strict=True):
        assert name in r[0]


def test_gaps_numbers_equal_the_result_files(page: str, rep: data.Report) -> None:
    sec = _gaps_section(page)
    rows = _gap_rows(sec)
    sp = [_spans_of("".join(r)) for r in rows]
    # every span in the section is the loader's value (the global diff test covers the files)
    for k, v in _spans_of(sec).items():
        assert v == data.fmt(rep.metrics[k].kind, rep.metrics[k].value), k
    # row 1: open-set rejection, re-read from the files
    head = _rj("results/trackb/headline.json")["methods"]["maha_ft"]
    conf = _rj("results/trackb_improve/confirm.json")["entries"]["base/maha_ft"]
    assert sp[0]["gap.open.head_rej"] == f"{head['strict_rejection_recall']['mean']:.3f}"
    assert sp[0]["gap.open.head_ret"] == f"{head['retention_known']['mean']:.3f}"
    c95 = conf["confirm"]["mean"]["op95"]["strict_rejection_recall"]
    assert sp[0]["gap.open.confirm_rej"] == f"{c95:.3f}"
    d1 = _rj(data.P6A_D1)["holdouts"]["headline"]
    un = d1["feature_sets"]["v1_finetuned"]["unseen_known_only"]["auroc_mean_of_folds"]
    assert sp[0]["p6a.d1.v1_finetuned.unseen"].startswith(f"{un:.3f}")
    maha = d1["unsupervised_finetuned_mahalanobis"]["phase3_results"]["mean_3_seeds"]
    assert sp[0]["p6a.d1.maha"] == f"{maha:.3f}"
    d4 = _rj(data.P6A_D4)["auroc"]["slope_vs_log2_fraction"]["slope_per_doubling"]
    assert sp[0]["p6a.d4.auroc.slope"].startswith(f"{d4:+.3f}")
    biz = _rj("results/trackb_improve/business.json")["rows"]
    assert (biz[1]["calibration_retention_target"], biz[4]["calibration_retention_target"]) == (
        0.9,
        0.95,
    )  # the business-table rows the cell cites (5% novel traffic at the 90% and 95% points)
    assert biz[1]["prevalence"] == biz[4]["prevalence"] == 0.05
    assert sp[0]["biz.r1.unknowns_misrouted_strict"] == f"{biz[1]['unknowns_misrouted_strict']:.1f}"
    assert sp[0]["biz.r4.known_wrongly_abstained"] == f"{biz[4]['known_wrongly_abstained']:.1f}"
    text0 = _visible_text(rows[0][1])
    assert "upper bound" in text0 and "oracle" in text0.lower()
    # row 2: paired comparison, CV, interval widths
    ta = _rj(data.TA)["test"]
    b1 = ta["baselines"]["B1"]
    mc = b1["mcnemar_exact_on_correctness"]
    assert sp[1]["a.B1.mc_p"] == f"{mc['p_value']:.3f}"
    assert sp[1]["a.B1.mc_a"] == str(mc["a_only_correct"])
    assert sp[1]["a.B1.mc_b"] == str(mc["b_only_correct"])
    dl = b1["delta_macro_f1_final_minus_baseline"]
    assert sp[1]["gap.b1.delta_w"] == f"{dl['hi'] - dl['lo']:.3f}"
    assert dl["lo"] < 0 < dl["hi"] and "includes zero" in _visible_text(rows[1][1])
    assert sp[1]["gap.f1_w"] == f"{ta['macro_f1']['hi'] - ta['macro_f1']['lo']:.3f}"
    ctx = _rj(data.SEL)["baseline_context_ttests"]["B1"]
    assert sp[1]["bk.nb_p_B1"] == f"{ctx['p']:.3f}"
    assert sp[1]["gap.b1.cv_diff"] == f"{ctx['mean_diff']:+.3f}"
    assert "not part of the selection rule" in _visible_text(rows[1][0])
    sel = _rj("results/trackb_improve/select.json")["dev_mean_auroc"]
    assert sp[1]["p3b.b1_b1_msp"] == f"{sel['b1/b1_msp']:.3f}"
    assert sp[1]["p3b.base_maha_ft"] == f"{sel['base/maha_ft']:.3f}"
    # row 3: LD->REF flip rate (the number the limitations paragraph uses) and the EDA audit
    swaps = _rj("results/robustness/idswap.json")["results"]
    sw = next(r for r in swaps if (r["source"], r["swap"]) == ("oof", "LD->REF"))
    assert sp[2]["swap.LD->REF.rate"] == f"{100 * sw['flip_rate']:.1f}%"
    assert sp[2]["swap.LD->REF.n"] == str(sw["n"])
    audit = _rj(data.EDA)["shortcut_audit"]["LD-\\d+"]
    assert sp[2]["eda.sc.ld.share"] == f"{100 * audit['max_label_share']:.1f}%"
    assert sp[2]["eda.sc.ld.n"] == str(audit["n_rows"])
    ax = _rj(data.P6A_AXES)["candidates"]
    assert sp[2]["p6a.ax.i4a.c"] == f"{ax['i4a']['axes']['c']['delta']:+.3f}"
    assert sp[2]["p6a.ax.i4b.a_ci"].startswith(f"{ax['i4b']['axes']['a']['delta']:+.3f}")
    # row 4
    assert sp[3]["eda.n"] == str(_rj(data.EDA)["n_rows"]) and sp[3]["a.n"] == str(ta["n"])
    assert sp[3]["gap.f1_w"] == sp[1]["gap.f1_w"]


def test_sample_size_estimate_is_computed_from_the_discordant_counts(page: str) -> None:
    from statistics import NormalDist

    ta = _rj(data.TA)["test"]
    mc = ta["baselines"]["B1"]["mcnemar_exact_on_correctness"]
    b, c, n_test = mc["a_only_correct"], mc["b_only_correct"], ta["n"]
    psi, delta = (b + c) / n_test, (b - c) / n_test
    z = NormalDist()
    root = z.inv_cdf(0.975) * math.sqrt(psi) + z.inv_cdf(0.8) * math.sqrt(psi - delta**2)
    need = math.ceil(root**2 / delta**2)
    assert data.mcnemar_required_n(b, c, n_test, 0.05, 0.80) == need
    cell = _gap_rows(_gaps_section(page))[1][3]
    sp = _spans_of(cell)
    assert sp["gap.ss.n"] == str(need) and sp["gap.ss.psi"] == f"{psi:.3f}"
    assert sp["gap.ss.delta"] == f"{delta:.3f}" and sp["gap.ss.mult"] == f"{need / n_test:.1f}"
    assert sp["gap.ss.alpha"] == "0.05" and sp["gap.ss.power"] == "80%"
    assert sp["a.B1.mc_a"] == str(b) and sp["a.B1.mc_b"] == str(c)
    text = re.sub(r"\s+", " ", _visible_text(cell))
    assert "ESTIMATE (observed discordance):" in text
    assert len(re.findall("estimate", text, re.I)) == 1  # one visible label, not a badge + word
    assert "It assumes the observed discordance rate (" in text and "stays as observed" not in text
    assert "McNemar sample-size approximation" in _visible_text(cell)
    assert _numbers_outside_spans(cell.replace(data.SS_FORMULA, "")) == []  # formula is text


def test_mcnemar_required_n_edge_cases() -> None:
    assert data.mcnemar_required_n(3, 3, 74, 0.05, 0.8) is None  # delta = 0
    assert data.mcnemar_required_n(0, 0, 74, 0.05, 0.8) is None  # no discordant rows
    assert data.mcnemar_required_n(74, 0, 74, 0.05, 0.8) is None  # psi - delta^2 = 0
    big = data.mcnemar_required_n(7, 3, 74, 0.05, 0.8)
    small = data.mcnemar_required_n(10, 2, 74, 0.05, 0.8)
    assert big is not None and small is not None and small < big  # larger gap, fewer rows


def test_sample_size_cell_falls_back_when_the_discordance_gives_no_estimate(
    monkeypatch: pytest.MonkeyPatch, figs: dict[str, Path]
) -> None:
    def edit(doc: Any) -> None:
        mc = doc["test"]["baselines"]["B1"]["mcnemar_exact_on_correctness"]
        mc["b_only_correct"] = mc["a_only_correct"]  # delta = 0
        mc["n_discordant"] = 2 * mc["a_only_correct"]

    _patch_json(monkeypatch, data.TA, edit)
    rep = data.load()
    assert "gap.ss.n" not in rep.metrics and rep.ctx["gap"]["ss_ok"] is False
    cell = _gap_rows(_gaps_section(build.render_html(rep, figs)))[1][3]
    text = re.sub(r"\s+", " ", _visible_text(cell))
    assert "several times the current" in text and "no number is formed" in text
    assert "gap.ss.n" not in cell


def test_gaps_section_text_rules(page: str) -> None:
    sec = _gaps_section(page)
    assert not re.search(r"\bG" r"G\b", _visible_text(sec))
    # measured / estimate / optimistic labels as elsewhere in the report
    assert 'class="tag">measured' in sec and 'class="tag e">estimate' in sec
    assert 'class="tag o">optimistic' in sec
    # the section is one unbreakable block (its print height is gated in build.build)
    css = (ROOT / "report" / "report.css").read_text(encoding="utf-8")
    assert re.search(r"\.gaps\s*\{[^}]*break-inside:\s*avoid", css)


def test_three_four_hour_box_has_three_lines_and_no_time_figures(page: str) -> None:
    box = _between(page, '<div class="box gap3">', "<h3>Next steps, detail</h3>")
    assert box.count("<p>") == 3 and GAP_TITLE in _visible_text(box)
    body = box.replace(f"<b>{GAP_TITLE.replace('.', '')}.</b>", "")  # only the title has digits
    assert GAP_TITLE not in _visible_text(body)
    assert _numbers_outside_spans(body) == []
    assert not re.search(
        r"\b(hours?|minutes?|mins?|hrs?|days?|weeks?)\b", _visible_text(body), re.I
    )
    text = _visible_text(box)
    for step in ("EDA", "stratified split", "fine-tune", "macro-F1", "MSP", "W&B", "Hub push"):
        assert step in text
    assert "intervals" in text and "pre-registered" in text and "Track B diagnosis" in text


def test_gaps_row2_wording_training_free_fallback_and_mcnemar_scope(page: str) -> None:
    sec = _between(page, '<section class="gaps"', "</section>")
    assert "training-free fallback: same inference cost (same frozen e5 encoder)" in sec
    assert "and a logistic-regression head to retrain" in sec
    assert "cheap fallback" not in sec
    assert "to resolve the observed accuracy difference (McNemar)" in sec


def test_no_doubled_label_in_visible_text(page: str) -> None:
    """A badge followed by the same word (ESTIMATE ESTIMATE, ...) reads as a typo."""
    rx = r"\b(ESTIMATE|MEASURED|OPTIMISTIC|REPORT[- ]ONLY)\b\W{0,3}\1\b"
    # The labels legend is exempt on purpose: it defines each badge as "BADGE = word (...)", which
    # is a deliberate repetition of the word.
    body = page.split("</header>", 1)[1]
    assert [m.group(0) for m in re.finditer(rx, _visible_text(body), re.I)] == []


def test_labels_legend_defines_each_badge(page: str) -> None:
    header = page.split("</header>", 1)[0]
    legend = _visible_text(header[header.index("Labels:") :])
    assert re.search(r"MEASURED\s*=\s*measured \(seed, split and interval stated\)", legend, re.I)
    assert re.search(r"ESTIMATE\s*=\s*estimate \(assumption stated\)", legend, re.I)
    assert re.search(r"OPTIMISTIC\s*=\s*post-selection, optimistic", legend, re.I)


# ------------------------------------------- restructure: 4-page main text, lettered appendix

MAIN_HEADINGS = [
    "Approach and Model Selection",
    "Data Characteristics and Handling",
    "Closed-Set Classification (Track A)",
    "Open-Set Abstention (Track B)",
    "Error Analysis",
    "Deployment and Monitoring",
    "Next Steps",
]
APPENDIX_HEADINGS = [
    "Debrief Answers",
    "Open-Set Evaluation in Full: Scorers, Leave-One-Class-Out and Business Framing",
    "Diagnostics (Oracle Probe, ID-Neutral Scoring, Threshold Stability, Learning Curve)",
    "Selection History and the v1/v3 Decision",
    "Robustness",
    "LLM Baseline",
    "Latency, Quantization and Cost",
    "Reproduction and Determinism",
    "Ablations and Bake-Off Detail",
    "Test-Split Exposure",
    "Further Detail",
]
# Public text states decisions and their data-driven rationale; it never points at internal
# documents or process artefacts. Case-sensitive: the plan document, the rest case-insensitive.
# The phrases are written in fragments so this file does not itself contain them.
FORBIDDEN_CS = re.compile(r"\bPL" r"AN\b")
FORBIDDEN_CI = re.compile(
    r"desi"
    r"gn/|spec\.md|the plan|production plan|amend"
    r"ment|available on"
    r" request"
    r"|pre-registration "
    r"(?:log|document)|gpu "
    r"lock|gpu_exclusive|contention"
    r"|\b(?:the|a|shared|test|selection) logs?\b|\blog files?\b",
    re.I,
)


def forbidden_wording(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    return [m.group(0) for m in (*FORBIDDEN_CS.finditer(flat), *FORBIDDEN_CI.finditer(flat))]


def test_forbidden_wording_scanner_catches_its_own_terms() -> None:
    for bad in (
        "see desi" "gn/ notes", "per spec" ".md", "the PL" "AN says", "in the plan", "Amend" "ment "
            "2",
        "available on" " request", "pre-registration" " log", "the GPU" " lock", "gpu_exclusive",
        "GPU contention", "recorded in the log", "from the log files", "a shared test log",
    ):  # fmt: skip
        assert forbidden_wording(bad), bad
    assert forbidden_wording("test_eval_log.jsonl holds logged calls; a retention check") == []


def test_no_internal_document_references_in_the_rendered_report(
    rep: data.Report, figs: dict[str, Path], page: str
) -> None:
    pub = build.render_html(rep, figs, publish=True)
    for html in (page, pub):
        assert forbidden_wording(_visible_text(html)) == []
    # the sources too: the template (minus its Jinja comment) and the email text
    for rel in ("report/template.html.j2", "scripts/render_submission_email.py"):
        if not (ROOT / rel).exists():  # the email renderer is private (not in the public snapshot)
            continue
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert forbidden_wording(re.sub(r"\{#.*?#\}", " ", src, flags=re.S)) == [], rel


def test_main_has_exactly_seven_numbered_sections_and_the_appendix_is_lettered(
    page: str,
) -> None:
    main = page.split("<main>", 1)[1].split("</main>", 1)[0]
    assert re.findall(r"<h2[^>]*>(.*?)</h2>", main) == MAIN_HEADINGS
    assert list(data.heading_map().values()) == MAIN_HEADINGS
    app = page[page.index('<section class="app">') :]
    assert re.findall(r"<h2[^>]*>(.*?)</h2>", app) == APPENDIX_HEADINGS
    assert data.appendix_map() == dict(zip("ABCDEFGHIJK", APPENDIX_HEADINGS, strict=True))
    css = (ROOT / "report" / "report.css").read_text(encoding="utf-8")
    assert "counter(app, upper-alpha)" in css and "main h2::before" in css
    contents = _visible_text(page[page.index("Appendix contents.") :].split("</p>")[0])
    for letter, title in data.appendix_map().items():
        assert f"{letter} {title}" in contents


def test_section_references_in_the_text_come_from_the_heading_map(page: str) -> None:
    sec = data.section_numbers()
    assert sec["trackb"] == 4 and sec["production"] == 6 and sec["diag"] == "C"
    assert sec["history"] == "D" and sec["robust"] == "E" and sec["more"] == "K"
    main = page.split("</main>")[0]
    assert f"details in Appendix {sec['diag']}." in _visible_text(main)  # D1 root-cause line
    src = TEMPLATE.read_text(encoding="utf-8")
    # references are rendered from the map, never typed
    typed = re.search(
        r"Appendix [A-K]\b|(?<!notebook )[Ss]ection \d", re.sub(r"\{\{.*?\}\}", " ", src)
    )
    assert typed is None


def test_header_has_the_four_links_and_the_headline_table(page: str) -> None:
    head = _between(page, '<header class="title">', "</header>")
    for ph in ("HF link", "W&amp;B link", "repo link", "Colab link"):
        assert ph in head
    box = _between(page, '<div class="box exec">', '<table class="head">')
    assert box.count("<p>") == 1
    tab = _between(page, '<table class="head">', "</table>")
    keys = {htmllib.unescape(k): v for k, v in _SPAN.findall(tab)}
    for k in (
        "a.f1", "a.acc", "tb.m.msp.auroc", "tb.headline.auroc",
        "tb.m.msp.strict_rejection_recall", "tb.headline.strict_recall_95",
        "tb.m.msp.retention_known", "tb.headline.retention_95",
    ):  # fmt: skip
        assert k in keys, k
    f1 = _rj(data.TA)["test"]["macro_f1"]
    assert keys["a.f1"] == f"{f1['point']:.3f} [{f1['lo']:.3f}, {f1['hi']:.3f}]"
    msp = _rj("results/trackb/headline.json")["methods"]["msp"]["auroc"]
    assert keys["tb.m.msp.auroc"] == f"{msp['mean']:.3f} ± {msp['std']:.3f}"
    conf = _rj(data.CONF4E)["v1"]["headline"]["ci95"]
    assert keys["tb.headline.auroc"].startswith(f"{conf['auroc']['point']:.3f} [")
    assert _numbers_outside_spans(tab) == []


def test_main_per_class_table_sits_next_to_the_confusion_matrix(
    page: str, rep: data.Report
) -> None:
    block = _between(page, '<div class="grid-fig">', "<p><b>Paired comparison")
    assert block.index('<table class="pc1">') < block.index("<figure>")
    assert "confusion matrix, test" in block
    table = _between(block, '<table class="pc1">', "</table>")
    for c in rep.ctx["classes"]:
        row = _between(table, f"<tr><td>{c}</td>", "</tr>")
        assert len(_SPAN.findall(row)) == 4  # P, R, F1, n on the test split
    css = (ROOT / "report" / "report.css").read_text(encoding="utf-8")
    assert re.search(r"\.grid-fig\s*\{[^}]*grid-template-columns:\s*1fr\s+\d+mm", css)
    line = _between(page, "<b>Paired comparison on the same test rows.</b>", "</p>")
    assert {"a.B0.delta", "a.B1.delta", "a.B1.mc_p"} <= {k for k, _ in _SPAN.findall(line)}


def _csv_rows(rel: str) -> list[dict[str, str]]:
    import csv

    with (ROOT / rel).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _op_point(rows: list[dict[str, str]], m: str, r: float) -> tuple[float, float]:
    """(strict rejection, known retention) at calibration retention r, coded independently."""
    cal = sorted((float(x[m]) for x in rows if x["set"] == "cal"), reverse=True)
    thr = cal[max(1, math.ceil(r * len(cal) - 1e-9)) - 1]
    ev = [x for x in rows if x["set"] == "eval"]
    unk = [float(x[m]) for x in ev if x["is_unknown"] == "True"]
    kn = [float(x[m]) for x in ev if x["is_unknown"] != "True"]
    return sum(u < thr for u in unk) / len(unk), sum(k >= thr for k in kn) / len(kn)


def _sample_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def test_90_percent_operating_point_is_recomputed_and_reproduces_the_stored_95(
    rep: data.Report,
) -> None:
    flat = rep.flat()
    head = _rj("results/trackb/headline.json")
    tables = [_csv_rows(f"results/trackb/scores/headline_s{s}.csv") for s in head["seeds"]]
    for m in ("msp", "maha_ft"):
        pts95 = [_op_point(t, m, 0.95) for t in tables]
        # the same rule reproduces the stored 95% per-seed values, so the 90% values are trusted
        want_rej = head["methods"][m]["strict_rejection_recall"]["values"]
        want_ret = head["methods"][m]["retention_known"]["values"]
        assert [p[0] for p in pts95] == pytest.approx(want_rej)
        assert [p[1] for p in pts95] == pytest.approx(want_ret)
        pts90 = [_op_point(t, m, 0.90) for t in tables]
        rej, ret = [p[0] for p in pts90], [p[1] for p in pts90]
        got = flat[f"tb.op90.{m}.rej"]
        assert got["mean"] == pytest.approx(sum(rej) / 3)
        assert got["std"] == pytest.approx(_sample_std(rej))
        assert flat[f"tb.op90.{m}.ret"]["mean"] == pytest.approx(sum(ret) / 3)
    # a second, independent source: the paired-CI file stores the shipped scorer's 90% per seed
    per = _rj(data.CONF4E)["v1"]["headline"]["per_unit"]
    assert [per[k]["op90"]["strict_rejection_recall"] for k in sorted(per)] == pytest.approx(
        [_op_point(t, "maha_ft", 0.90)[0] for t in tables]
    )


def test_op90_recomputation_fails_loudly_when_it_stops_reproducing_the_stored_95(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def edit(doc: Any) -> None:
        doc["methods"]["msp"]["strict_rejection_recall"]["mean"] += 0.01

    _patch_json(monkeypatch, "results/trackb/headline.json", edit)
    with pytest.raises(ValueError, match="op90 recomputation drifted"):
        data.load()


def test_trackb_main_table_has_msp_and_shipped_at_both_operating_points(page: str) -> None:
    tab = _between(page, '<table class="tbm">', "</table>")
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tab, flags=re.S)
    assert len(rows) == 3  # header, MSP, shipped
    for row, who in zip(rows[1:], ("msp", "maha_ft"), strict=True):
        keys = [htmllib.unescape(k) for k, _ in _SPAN.findall(row)]
        assert keys == [
            f"tb.m.{who}.auroc",
            f"tb.m.{who}.strict_rejection_recall",
            f"tb.m.{who}.retention_known",
            f"tb.op90.{who}.rej",
            f"tb.op90.{who}.ret",
        ]
    sec = _between(page, "<h2>Open-Set Abstention (Track B)</h2>", "<h2>Error Analysis</h2>")
    keys = {k for k, _ in _SPAN.findall(sec)}
    assert {"tb.holdout", "srv.thr", "p6a.d1.maha", "tb.headline.strict_recall_90"} <= keys
    assert f"details in Appendix {data.section_numbers()['diag']}" in _visible_text(sec)
    assert _numbers_outside_spans(sec) == []


def test_main_next_steps_are_exactly_three_bullets(page: str) -> None:
    sec = _between(page, "<h2>Next Steps</h2>", "</main>")
    assert sec.count("<li>") == 3
    text = _visible_text(sec)
    assert "hard negatives" in text and "labelled data" in text and "ID-consistency" in text
    keys = {k for k, _ in _SPAN.findall(sec)}
    assert {"p6a.d4.auroc.slope", "p6a.ax.i4a.c", "p6a.ax.i4a.a_ci"} <= keys
    assert _numbers_outside_spans(sec) == []


def test_main_production_section_has_serving_monitoring_playbook_and_gaps(page: str) -> None:
    sec = _between(page, "<h2>Deployment and Monitoring</h2>", "<h2>Next Steps</h2>")
    for word in ("Serving.", "Monitoring and drift.", "13th-class playbook."):
        assert word in sec
    keys = {k for k, _ in _SPAN.findall(sec)}
    assert {"srv.onnx_fp32.t1.p50", "srv.onnx_fp32.t1.p95"} <= keys
    assert sec.index("Serving.") < sec.index("Monitoring and drift.") < sec.index('class="gaps"')


def test_main_data_quirks_are_the_seven_one_liners(page: str) -> None:
    sec = _between(page, "<h2>Data Characteristics and Handling</h2>", "<h2>Closed-Set")
    labels = re.findall(r"<li><b>(.*?)</b>", sec)
    assert labels == [
        "Class imbalance.", "Multilingual.", "Sibling classes.", "Noise.",
        "Catch-all classes.", "Small data.", "ID-prefix shortcut.",
    ]  # fmt: skip
    assert _numbers_outside_spans(sec) == []


def test_main_encoder_vs_llm_keeps_the_like_for_like_gpu_basis(page: str) -> None:
    sec = _between(page, "<b>Encoder versus LLM.</b>", "<h2>Data Characteristics")
    text = re.sub(r"\s+", " ", _visible_text(sec))
    keys = {k for k, _ in _SPAN.findall(sec)}
    assert {"llm.f1_min", "llm.rej_max", "ft.gpu1.p50_ms", "ratio.lat_gpu", "ratio.usd"} <= keys
    assert "like-for-like" in text and "no CPU-versus-GPU ratio is formed" in text
    assert "no LLM was timed on CPU" in text and not any(k.startswith("ratio.cpu") for k in keys)


def test_every_metric_family_the_old_report_showed_is_still_rendered(page: str) -> None:
    """Nothing was dropped: every metric family of the pre-restructure report (by key prefix) is
    still rendered somewhere, main text or appendix."""
    used = {htmllib.unescape(k) for k, _ in _SPAN.findall(page)}
    for prefix in (
        "a.", "cv.", "oof.", "bk.", "abl.", "sc.", "q.", "qv.", "srv.", "dk.", "hub.", "llm.",
        "ft.", "ratio.", "tb.", "biz.", "err.", "swap.", "tr.", "nz.", "sl.", "p6a.", "p3b.",
        "hist.", "trd.", "log.", "rc.", "det.", "rr.", "nb.", "gap.", "eda.", "aud.", "aud2.",
        "cal.", "sel.", "ax.", "repro.",
    ):  # fmt: skip
        assert any(k.startswith(prefix) for k in used), prefix


def test_header_table_puts_track_a_under_the_shipped_column_with_a_dash_for_msp(page: str) -> None:
    table = _between(page, '<table class="head">', "</table>")
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.S)
    cells = [re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S) for r in rows]
    assert [c.strip() for c in cells[0][1:]] == ["MSP baseline", "Shipped scorer (v1)"]
    track_a = [c for c in cells[1:] if c[0].startswith("Track A")]
    assert len(track_a) == 2
    for c in track_a:
        assert len(c) == 3 and c[1].strip() == "&mdash;" and c[2].strip() != "&mdash;"
    for c in cells[1:]:
        assert "colspan" not in "".join(c)


# --------------------------------------------------- main-only variant (--main-only)

POINTER_MARK = "are in the complete report:"


def _main_only_html(rep: data.Report, figs: dict[str, Path], **kw: Any) -> str:
    links = json.loads((ROOT / "report" / "links.json").read_text(encoding="utf-8"))
    return build.render_html(rep, figs, links, main_only=True, publish=True, **kw)


def test_main_only_has_header_and_sections_1_to_7_and_no_appendix(
    rep: data.Report, figs: dict[str, Path]
) -> None:
    html = _main_only_html(rep, figs, appendix_pointer=True)
    assert '<header class="title">' in html and '<section class="app">' not in html
    assert "Appendix contents" not in html
    h2s = re.findall(r"<h2[^>]*>(.*?)</h2>", html)
    assert [re.sub(r"<[^>]+>", "", h).strip() for h in h2s] == list(data.heading_map().values())
    assert len(h2s) == 7
    for _letter, title in data.appendix_map().items():
        assert f"<h2>{title}</h2>" not in html


def test_pointer_line_only_in_main_only_and_built_from_links_and_appendix_map(
    rep: data.Report, figs: dict[str, Path]
) -> None:
    links = json.loads((ROOT / "report" / "links.json").read_text(encoding="utf-8"))
    letters = list(data.appendix_map())
    want = (
        f"Appendices {letters[0]}–{letters[-1]} (full evidence) are in the complete report: "
        f"{links['repo']}/blob/main/report.pdf"
    )
    html = _main_only_html(rep, figs, appendix_pointer=True)
    assert want in _visible_text(html) and letters[-1] == "K"
    # under the Links line, inside the header
    head = _between(html, '<header class="title">', "</header>")
    assert head.index('<p class="links"><b>Links.') < head.index(POINTER_MARK)
    # the page-count variant of the full build and the full report never carry it
    assert POINTER_MARK not in _main_only_html(rep, figs)
    assert POINTER_MARK not in build.render_html(rep, figs, links, publish=True)
    assert POINTER_MARK not in build.render_html(rep, figs)


def test_pointer_needs_a_real_repo_link(rep: data.Report, figs: dict[str, Path]) -> None:
    with pytest.raises(SystemExit, match="'repo' link"):
        build.render_html(rep, figs, None, main_only=True, appendix_pointer=True)


def test_main_only_pdf_is_gated_and_writes_side_files_only_next_to_out(
    chromium: None, tmp_path: Path
) -> None:
    links = json.loads((ROOT / "report" / "links.json").read_text(encoding="utf-8"))
    out = tmp_path / "scratch" / "report_main.pdf"
    s = build.build(out, links=links, publish=True, main_only=True)
    assert 1 <= s["main_pages"] <= build.MAX_MAIN_PAGES == 4
    info = json.loads((out.parent / "report_main_build.json").read_text(encoding="utf-8"))
    assert info["main_pages"] == s["main_pages"] == build.pdf_pages(out)
    assert info["pdf_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert not (out.parent / "provenance.json").exists()
    assert not (out.with_name("report_main_only.pdf")).exists()
    with pytest.raises(SystemExit, match="main text is"):
        build.build(tmp_path / "g" / "r.pdf", links=links, main_only=True, max_main_pages=1)
    try:
        import pypdf
    except ImportError:
        return
    text = re.sub(r"\s+", " ", " ".join(p.extract_text() for p in pypdf.PdfReader(str(out)).pages))
    assert POINTER_MARK in text and "Appendix contents" not in text
    assert forbidden_wording(text) == []


def test_main_only_flag_refuses_include_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["report.build", "--main-only", "--include-text"])
    with pytest.raises(SystemExit, match="--main-only cannot be combined"):
        build.main()


# ------------------------------------------- wording pass: method first, limits as facts

_MAIN_CODES = re.compile(r"\bPhase \d\w?\b|\b(?:i1b|i6b|i1a|i6a|I[1-6]|N[12]|A[1-3]|D[1-4])\b")
_APOLOGETIC = re.compile(
    r"we say so|unfortunately|\bweak\b|untested|\bnot run\b|\bcannot\b|\bunresolved\b|\bpoor\b|"
    r"cost a lot|\bfailed\b|did not beat|changed nothing that ships|\bblocked\b|\bcould not\b",
    re.I,
)


def test_summary_leads_with_method_then_results_then_the_limit_and_next_step(page: str) -> None:
    box = _between(page, '<div class="box exec">', '<table class="head">')
    text = re.sub(r"\s+", " ", _visible_text(box))
    i_method, i_results, i_limit = (
        text.index(x) for x in ("Method:", "Results:", "Limit and next")
    )
    assert i_method < i_results < i_limit
    method = text[i_method:i_results]
    for phrase in (
        "class-held-out", "duplicate-grouped", "fixed before the results",
        "rendered from saved results and recomputed by a separate check",
        "bitwise reproducible within a GPU type", "REPRODUCTION: PASS",
    ):  # fmt: skip
        assert phrase in method, phrase
    keys = [htmllib.unescape(k) for k, _ in _SPAN.findall(box)]
    assert {"a.f1", "tb.headline.strict_recall_95", "tb.m.msp.auroc", "rc.verdict"} <= set(keys)
    assert text.index("measured limit") > i_limit - 1 and "verified hard negatives" in text
    assert "supplier" not in text.lower()
    assert len(re.findall(r"(?<=[a-z)\]])\. (?=[A-Za-z])", text)) <= 5  # about five sentences
    assert _numbers_outside_spans(box) == []


def test_limits_are_stated_as_facts_with_a_next_step_not_apologetically(page: str) -> None:
    flat = re.sub(r"\s+", " ", _visible_text(page.split("</header>", 1)[1]))
    assert [m.group(0) for m in _APOLOGETIC.finditer(flat)] == []
    main = re.sub(r"\s+", " ", _visible_text(_between(page, "<main>", "</main>")))
    assert "Hypotheses (next step: targeted test)" in main
    assert "did not meet the pre-set rule" in main and "so fp32 ships" in main
    assert "gave no gain over the CV noise band" in main and "plain cross-entropy kept" in main


def test_main_body_uses_no_internal_codes_except_the_two_named_diagnostics(page: str) -> None:
    main = re.sub(r"\s+", " ", _visible_text(_between(page, "<main>", "</main>")))
    main = main.replace("oracle probe (D1)", "oracle probe").replace(
        "learning curve (D4)", "learning curve"
    )
    assert [m.group(0) for m in _MAIN_CODES.finditer(main)] == []
