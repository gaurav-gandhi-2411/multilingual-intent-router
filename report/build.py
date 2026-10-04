"""Build the report: results -> flat metrics -> Jinja2 HTML -> Playwright Chromium PDF.

    .\\.venv\\Scripts\\python.exe -m report.build                    # build/report_draft.pdf
    .\\.venv\\Scripts\\python.exe -m report.build --links links.json  # fill the link placeholders
    .\\.venv\\Scripts\\python.exe -m report.build --include-text      # PRIVATE: adds example texts
    .\\.venv\\Scripts\\python.exe -m report.build --publish --links links.json  # final build
    .\\.venv\\Scripts\\python.exe -m report.build --publish --links links.json --main-only \\
        --out build/scratch_main/report_main.pdf   # main body only; side files next to --out

The template carries no numeric literals: every number goes through `n("key")`, which emits
`<span class="n" data-k="key">...</span>` (no visible provenance marker in the PDF); the key-to-file
map is written to build/provenance.json. The build fails loudly if the main text (everything
before the appendix) exceeds MAX_MAIN_PAGES or the appendix exceeds MAX_APPENDIX_PAGES.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import jinja2
from markupsafe import Markup

if __package__ in (None, ""):  # `python report/build.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report import charts, data  # noqa: E402

ROOT = data.ROOT
PAGE_W_MM, MARGIN_X_MM = 210, 13  # A4 width and the @page side margins in report.css
PRINT_W_PX = round((PAGE_W_MM - 2 * MARGIN_X_MM) / 25.4 * 96)  # CSS px at 96 dpi
REPORT_DIR = Path(__file__).resolve().parent
BUILD = ROOT / "build"
MAX_MAIN_PAGES = 4  # task limit: main text (the assessment's report items), appendix excluded
MAX_APPENDIX_PAGES = 8  # generous ceiling: the appendix holds everything the main text omits
PAGE_H_MM, MARGIN_TOP_MM, MARGIN_BOTTOM_MM = 297, 12, 14  # A4 height and the @page margins
GAPS_MAX_PX = (PAGE_H_MM - MARGIN_TOP_MM - MARGIN_BOTTOM_MM) / 25.4 * 96  # one page of content
DEFAULT_LINKS = {
    "hf": "[HF link — filled at publish time]",
    "wandb": "[W&B link — filled at publish time]",
    "repo": "[repo link — filled at publish time]",
    "colab": "[Colab link — filled at publish time]",
}
FIGURE_FILES = {
    "confusion": "results/final/figures/final_confusion_counts.png",
    "reliability": "results/final/figures/final_reliability.png",
}


def resolve_links(links: dict[str, str | None] | None) -> dict[str, str]:
    """Defaults overlaid per key: a missing, null or empty entry keeps its placeholder."""
    return {**DEFAULT_LINKS, **{k: v for k, v in (links or {}).items() if v and k in DEFAULT_LINKS}}


def appendix_pointer_text(lk: dict[str, str]) -> str:
    """The main-only header line: appendix range from the appendix map, URL from the repo link."""
    if lk["repo"] == DEFAULT_LINKS["repo"]:
        raise SystemExit(
            "FAIL: --main-only needs a real 'repo' link (--links) for its pointer line"
        )
    letters = list(data.appendix_map())
    url = lk["repo"].rstrip("/") + "/blob/main/report.pdf"
    return (
        f"Appendices {letters[0]}–{letters[-1]} (full evidence) are in the complete report: {url}"
    )


class Renderer:
    """Template-side helpers; records every metric key the text uses."""

    def __init__(self, rep: data.Report, figs: dict[str, Path], links: dict[str, str]) -> None:
        self.rep = rep
        self.figs = figs
        self.links = links
        self.used_keys: list[str] = []

    @staticmethod
    def tag_placeholder() -> Markup:
        """The 'placeholder' tag shown in front of a link that is not filled yet."""
        return Markup('<span class="tag p">placeholder</span> ')  # noqa: S704

    def n(self, key: str) -> Markup:
        """Emit a metric value as <span data-k="key">text</span>."""
        m = self.rep.metrics.get(key)
        if m is None:
            raise KeyError(f"template asks for unknown metric key {key!r}")
        self.used_keys.append(key)
        return Markup(f'<span class="n" data-k="{html.escape(key)}">{html.escape(m.text)}</span>')  # noqa: S704


def make_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(REPORT_DIR)),
        undefined=jinja2.StrictUndefined,
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html(
    rep: data.Report,
    figs: dict[str, Path],
    links: dict[str, str | None] | None = None,
    include_text: bool = False,
    main_only: bool = False,
    root: Path = ROOT,
    used_keys: set[str] | None = None,
    publish: bool = False,
    appendix_pointer: bool = False,
) -> str:
    """Render the report HTML. Raises on any undefined template variable or unknown metric key.

    `appendix_pointer` (main-only variant) adds one line under the Links line that points to the
    complete report; it needs a real `repo` link.
    """
    lk = resolve_links(links)
    pointer = appendix_pointer_text(lk) if appendix_pointer else None
    r = Renderer(rep, figs, lk)
    examples = data.load_examples(root) if include_text else []
    env = make_env()
    ctx: dict[str, Any] = {
        **rep.ctx,
        "n": r.n,
        "t": r.n,
        "links": lk,
        "lk": lambda k: Markup(  # noqa: S704 - values are escaped here, tag is a constant
            ("" if lk[k] != DEFAULT_LINKS[k] else str(r.tag_placeholder())) + html.escape(lk[k])
        ),
        "figs": {k: v.as_uri() for k, v in figs.items()},
        "css": Markup((REPORT_DIR / "report.css").read_text(encoding="utf-8")),  # noqa: S704
        "head": rep.head,
        "sec": data.section_numbers(),  # numbers/letters cited in the text come from the headings
        "sec_titles": list(data.appendix_map().items()),  # the appendix contents line
        "include_text": include_text,
        "examples": examples,
        "main_only": main_only,
        "appendix_pointer": pointer,
        "publish": publish,
        "R90": "biz.r0.calibration_retention_target",
        "R95": "biz.r3.calibration_retention_target",
        "P1": "biz.r0.prevalence",
        "P5": "biz.r1.prevalence",
        "P10": "biz.r2.prevalence",
    }
    out = env.get_template("template.html.j2").render(**ctx)
    if used_keys is not None:
        used_keys.update(r.used_keys)
    return out


# --------------------------------------------------------------------------- PDF


def pdf_pages(pdf: Path) -> int:
    """Page count from the PDF structure (Chromium writes plain /Type /Page objects)."""
    raw = pdf.read_bytes()
    return len(re.findall(rb"/Type\s*/Page(?![s\w])", raw))


def html_to_pdf(html_path: Path, pdf_path: Path, footer: str) -> float | None:
    """Print the HTML to PDF; returns the print-layout height (CSS px) of section.gaps, if any."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": PRINT_W_PX, "height": 1100})
        page.goto(html_path.as_uri())
        page.wait_for_load_state("load")
        page.emulate_media(media="print")
        # Chromium silently shrinks the whole document to fit when anything is wider than the page,
        # which would make the page count below meaningless. Fail loudly instead.
        wide = page.evaluate(
            """() => { const W = document.documentElement.clientWidth, out = [];
            document.querySelectorAll('table, figure, img, pre').forEach(e => {
              if (e.getBoundingClientRect().right > W + 1) out.push(e.tagName + ':' +
                (e.textContent || '').trim().slice(0, 40)); });
            return out; }"""
        )
        if wide:
            raise SystemExit(
                f"FAIL: content wider than the page (shrink-to-fit would apply): {wide[:5]}"
            )
        gaps_px = page.evaluate(
            """() => { const e = document.querySelector('section.gaps');
            return e ? e.getBoundingClientRect().height : null; }"""
        )
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template=(
                '<div style="width:100%;font-size:7px;color:#7b8794;padding:0 14mm;'
                'display:flex;justify-content:space-between;font-family:Arial,sans-serif;">'
                f"<span>{html.escape(footer)}</span>"
                '<span>page <span class="pageNumber"></span> of <span class="totalPages"></span>'
                "</span></div>"
            ),
        )
        browser.close()
    return None if gaps_px is None else float(gaps_px)


def provenance_doc(rep: data.Report, used: set[str]) -> dict[str, Any]:
    """The key-to-source map that replaces the in-PDF provenance table (build/provenance.json)."""
    return {
        "build_head": rep.head,
        "files": {
            f: {"sha": sha, "sha_source": src} for f, (sha, src) in sorted(rep.file_sha.items())
        },
        "figures": rep.figures,
        "keys_used_in_text": sorted(used),
        "metrics": rep.provenance(),
    }


def build(
    out_pdf: Path,
    links: dict[str, str | None] | None = None,
    include_text: bool = False,
    max_main_pages: int = MAX_MAIN_PAGES,
    make_pdf: bool = True,
    max_appendix_pages: int = MAX_APPENDIX_PAGES,
    publish: bool = False,
    main_only: bool = False,
) -> dict[str, Any]:
    """Full build: charts, HTML, PDF, page-count gates, provenance JSON. Returns a summary.

    Fails loudly (SystemExit) when the main text exceeds `max_main_pages` or, for the public
    variant, the appendix exceeds `max_appendix_pages` (the private --include-text variant
    appends example texts, so only its main text is gated).
    """
    if main_only:
        return build_main_only(out_pdf, links, max_main_pages, publish)
    t0 = time.perf_counter()
    out_pdf = out_pdf.resolve()  # Playwright needs absolute file URIs (relative --out)
    BUILD.mkdir(exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)  # e.g. --out build/final/report.pdf
    rep = data.load()
    figs = {k: ROOT / v for k, v in FIGURE_FILES.items()}
    figs.update(charts.make_all(BUILD / "figs"))
    stem = out_pdf.with_suffix("")
    html_path = stem.with_suffix(".html")
    used: set[str] = set()
    html_text = render_html(rep, figs, links, include_text, used_keys=used, publish=publish)
    html_path.write_text(html_text, encoding="utf-8")
    prov = out_pdf.parent / ("provenance_private.json" if include_text else "provenance.json")
    prov_doc = provenance_doc(rep, used)
    placeholders = sorted(k for k, v in resolve_links(links).items() if v == DEFAULT_LINKS[k])
    # How this build was made; a PDF whose build is not final must not be published.
    prov_doc["build"] = {
        "publish": publish,
        "include_text": include_text,
        "placeholder_links": placeholders,
        "pdf_name": None,
        "pdf_sha256": None,
    }
    prov.write_text(json.dumps(prov_doc, indent=1), encoding="utf-8")
    summary: dict[str, Any] = {
        "html": str(html_path),
        "provenance": str(prov),
        "metrics": len(rep.metrics),
    }
    if make_pdf:
        kind = "report" if publish else "report draft"
        footer = f"intent-router {kind} | results at build HEAD {rep.head['short']}"
        main_html = stem.parent / (stem.name + "_main_only.html")
        main_pdf = stem.parent / (stem.name + "_main_only.pdf")
        main_html.write_text(
            render_html(rep, figs, links, include_text, main_only=True, publish=publish),
            encoding="utf-8",
        )
        gaps_px = html_to_pdf(main_html, main_pdf, footer)
        main_pages = pdf_pages(main_pdf)
        html_to_pdf(html_path, out_pdf, footer)
        total = pdf_pages(out_pdf)
        prov_doc["build"]["pdf_name"] = out_pdf.name
        prov_doc["build"]["pdf_sha256"] = hashlib.sha256(out_pdf.read_bytes()).hexdigest()
        prov.write_text(json.dumps(prov_doc, indent=1), encoding="utf-8")
        summary.update(
            main_pages=main_pages,
            appendix_pages=total - main_pages,
            total_pages=total,
            pdf=str(out_pdf),
            pdf_bytes=out_pdf.stat().st_size,
            gaps_section_px=None if gaps_px is None else round(gaps_px),
            gaps_section_max_px=round(GAPS_MAX_PX),
        )
        if gaps_px is not None and gaps_px > GAPS_MAX_PX:
            raise SystemExit(
                f"FAIL: the gaps section is {gaps_px:.0f} px tall (> {GAPS_MAX_PX:.0f} px, one "
                "page of content); shorten its cells"
            )
        if main_pages > max_main_pages:
            raise SystemExit(
                f"FAIL: main text is {main_pages} pages (> {max_main_pages}); "
                "tighten the template/CSS (appendix is excluded from this limit)"
            )
        if not include_text and total - main_pages > max_appendix_pages:
            raise SystemExit(
                f"FAIL: appendix is {total - main_pages} pages (> {max_appendix_pages}); "
                "move bulk tables to build/provenance.json or the results files"
            )
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    return summary


def build_main_only(
    out_pdf: Path,
    links: dict[str, str | None] | None,
    max_main_pages: int = MAX_MAIN_PAGES,
    publish: bool = True,
) -> dict[str, Any]:
    """Render only the main body (header + sections 1-7) into one PDF, plus the pointer line.

    Every side file (HTML, <stem>_build.json) goes next to `out_pdf`; no provenance.json is
    written. Fails loudly when the page count exceeds `max_main_pages`.
    """
    t0 = time.perf_counter()
    out_pdf = out_pdf.resolve()
    BUILD.mkdir(exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    rep = data.load()
    figs = {k: ROOT / v for k, v in FIGURE_FILES.items()}
    figs.update(charts.make_all(BUILD / "figs"))
    html_path = out_pdf.with_suffix(".html")
    html_path.write_text(
        render_html(rep, figs, links, main_only=True, publish=publish, appendix_pointer=True),
        encoding="utf-8",
    )
    kind = "report" if publish else "report draft"
    gaps_px = html_to_pdf(
        html_path, out_pdf, f"intent-router {kind} | results at build HEAD {rep.head['short']}"
    )
    pages = pdf_pages(out_pdf)
    info = {
        "main_pages": pages,
        "pdf": out_pdf.name,
        "pdf_sha256": hashlib.sha256(out_pdf.read_bytes()).hexdigest(),
        "pdf_bytes": out_pdf.stat().st_size,
    }
    out_pdf.with_name(out_pdf.stem + "_build.json").write_text(
        json.dumps(info, indent=1), encoding="utf-8"
    )
    if gaps_px is not None and gaps_px > GAPS_MAX_PX:
        raise SystemExit(
            f"FAIL: the gaps section is {gaps_px:.0f} px tall (> {GAPS_MAX_PX:.0f} px)"
        )
    if pages > max_main_pages:
        raise SystemExit(f"FAIL: main text is {pages} pages (> {max_main_pages})")
    return {**info, "html": str(html_path), "seconds": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--links", type=Path, help="JSON with optional keys hf, wandb, repo, colab")
    ap.add_argument("--out", type=Path, default=BUILD / "report_draft.pdf")
    ap.add_argument(
        "--include-text",
        action="store_true",
        help="PRIVATE review only: append example texts read from data/dataset.csv",
    )
    ap.add_argument(
        "--publish",
        action="store_true",
        help="final build: drop the DRAFT / private-review wording (links still come from --links)",
    )
    ap.add_argument(
        "--main-only",
        action="store_true",
        help="render only the main body (header + sections 1-7) with a pointer line to the full "
        "report; side files go next to --out, nothing is written to provenance.json",
    )
    ap.add_argument("--no-pdf", action="store_true", help="write HTML + provenance only")
    ap.add_argument("--max-main-pages", type=int, default=MAX_MAIN_PAGES)
    ap.add_argument("--max-appendix-pages", type=int, default=MAX_APPENDIX_PAGES)
    a = ap.parse_args()
    links = json.loads(a.links.read_text(encoding="utf-8")) if a.links else None
    out = a.out
    if a.include_text and out.name == "report_draft.pdf":
        out = out.with_name("report_draft_private.pdf")  # never overwrite the text-free draft
    if a.publish and a.include_text:
        raise SystemExit("FAIL: --publish and --include-text are mutually exclusive")
    if a.publish:
        missing = [k for k, v in resolve_links(links).items() if v == DEFAULT_LINKS[k]]
        if missing:
            print(f"WARNING: --publish with placeholder links: {missing}", file=sys.stderr)
    if a.main_only and (a.include_text or a.no_pdf):
        raise SystemExit("FAIL: --main-only cannot be combined with --include-text or --no-pdf")
    s = build(
        out,
        links,
        a.include_text,
        a.max_main_pages,
        not a.no_pdf,
        a.max_appendix_pages,
        a.publish,
        a.main_only,
    )
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
