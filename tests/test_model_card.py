"""Model cards: numbers equal the results JSON; no company name, no dataset text."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import jinja2
import pytest
import yaml

from intent_router import model_card as mc

ROOT = Path(__file__).resolve().parents[1]
HASHES = ROOT / "tests" / "forbidden_term_hashes.txt"
DATASET = ROOT / "data" / "dataset.csv"
MIN_MESSAGE_CHARS = 20  # shorter messages ("thanks") collide with ordinary card words


def _j(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cards() -> dict[str, str]:
    return {v: mc.render_card(v, ROOT) for v in ("v1", "v3")}


def test_filters_format_from_json_values() -> None:
    assert mc.f3(0.94651606) == "0.947"
    assert mc.pct(0.124) == "12.4"
    assert mc.ci({"point": 0.5, "lo": 0.25, "hi": 0.75}) == "0.500 [0.250, 0.750]"
    assert mc.ci(None) == "n/a"


@pytest.mark.parametrize(("version", "results"), [("v1", "final"), ("v3", "final_v3")])
def test_key_numbers_equal_json_sources(cards: dict[str, str], version: str, results: str) -> None:
    card = cards[version]
    ta = _j(f"results/{results}/track_a.json")
    t = ta["test"]
    for m in ("macro_f1", "accuracy"):
        d = t[m]
        assert f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]" in card
    assert f"n = {t['n']}" in card
    assert f"T = {ta['calibration']['temperature']:.4f}" in card
    ood = _j(f"results/{results}/ood_shipped.json")
    assert f"threshold = {ood['threshold']:.2f}" in card
    for c in t["per_class"]:
        assert f"| `{c['label']}` | {c['f1']:.3f} | {c['support']} |" in card
    # Track B (headline holdout) comes from the tradeoff JSON, which cites confirm_a1a3.json
    h = _j("results/tradeoff_v1_v3.json")["trackb_holdout"]["headline"]["strict_recall_95"][version]
    assert f"{h['point']:.3f} [{h['lo']:.3f}, {h['hi']:.3f}]" in card
    conf = _j("results/phase4e/confirm_a1a3.json")
    key = "v1" if version == "v1" else "candidate"
    assert abs(conf[key]["headline"]["ci95"]["strict_recall_95"]["point"] - h["point"]) < 1e-12


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_eda_language_numbers(cards: dict[str, str], version: str) -> None:
    lang = _j("results/eda.json")["language"]
    card = cards[version]
    assert f"{lang['pct_non_english_primary_definition_A']}% of messages" in card
    assert "n = 74" in card  # small-test-set limitation


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_oracle_sentence_renders_d1_numbers(cards: dict[str, str], version: str) -> None:
    d1 = _j("results/phase6a/diag/d1_oracle.json")["holdouts"]["headline"]
    o = d1["feature_sets"]["v1_finetuned"]["unseen_known_only"]["auroc_mean_of_folds"]
    u = d1["unsupervised_finetuned_mahalanobis"]["phase3_results"]["mean_3_seeds"]
    assert f"AUROC {o:.3f} on unseen-known rows against {u:.3f}" in cards[version]
    assert "upper bound" in cards[version]


def test_oracle_sentence_labelled_as_v1_only_on_the_v3_card(cards: dict[str, str]) -> None:
    assert "for the unsupervised score used here (measured on v1)." in cards["v3"]
    assert "(measured on v1)" not in cards["v1"]


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_abstain_direction_and_both_thresholds(cards: dict[str, str], version: str) -> None:
    t = {
        v: _j(f"results/{r}/ood_shipped.json")["threshold"]
        for v, r in (("v1", "final"), ("v3", "final_v3"))
    }
    assert "score falls below the threshold" in cards[version]
    assert "passes the threshold" not in cards[version]
    assert f"({t['v1']:.2f} v1 / {t['v3']:.2f} v3;" in cards[version]


SECTIONS = (
    "## Label map",
    "## Usage",
    "## Training data",
    "## Intended and out-of-scope use",
    "## Track A: closed-set classification",
    "## Track B: open-set rejection",
    "## v1 vs v3",
    "## Limitations",
    "## Links",
)


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_sections_present_and_in_order(cards: dict[str, str], version: str) -> None:
    heads = re.findall(r"^## .*$", cards[version], flags=re.M)
    assert tuple(heads) == SECTIONS
    assert cards[version].index("\n# ") < cards[version].index(SECTIONS[0])


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_cards_are_short(version: str) -> None:
    assert len(mc.render_card(version, ROOT, publish=True).splitlines()) <= 200


@pytest.mark.parametrize(("version", "results"), [("v1", "final"), ("v3", "final_v3")])
def test_label_map_has_all_classes(cards: dict[str, str], version: str, results: str) -> None:
    classes = _j(f"results/{results}/track_a.json")["test"]["classes"]
    assert len(classes) == 12
    for c in classes:
        assert f"| {c['index']} | `{c['label']}` |" in cards[version]


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_six_row_tradeoff_table_from_json(cards: dict[str, str], version: str) -> None:
    section = cards[version].split("## v1 vs v3", 1)[1].split("## Limitations", 1)[0]
    rows = [ln for ln in section.splitlines() if ln.startswith("| ") and "---" not in ln][1:]
    assert len(rows) == 6
    t = _j("results/tradeoff_v1_v3.json")
    for v in ("v1", "v3"):
        d = t["track_a_test"][v]["macro_f1"]
        assert f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]" in rows[0]
    sw = t["robustness_cv"][mc.TRADEOFF_SWAP_KEY]
    assert f"{sw['v1']:.3f}" in rows[4] and f"{sw['v3']:.3f}" in rows[4]
    tr = t["robustness_cv"][mc.TRADEOFF_TRANSLATION_KEY]
    assert f"{tr['v1']:.3f}" in rows[5] and f"{tr['v3']:.3f}" in rows[5]
    c90 = t["trackb_holdout"]["confirm"]["strict_recall_90"]["v3"]
    assert f"{c90['point']:.3f} [{c90['lo']:.3f}, {c90['hi']:.3f}]" in rows[3]


@pytest.mark.parametrize(("version", "results"), [("v1", "final"), ("v3", "final_v3")])
def test_yaml_front_matter_valid_and_equals_json(
    cards: dict[str, str], version: str, results: str
) -> None:
    from huggingface_hub import ModelCard, ModelCardData, metadata_eval_result

    card = cards[version]
    assert card.startswith("---\n")
    fm = yaml.safe_load(card.split("---\n", 2)[1])
    assert fm["license"] == "mit" and fm["language"] == ["en", "es", "fr", "de", "zh"]
    assert fm["library_name"] == "transformers" and fm["pipeline_tag"] == "text-classification"
    assert fm["base_model"] == "intfloat/multilingual-e5-base"
    assert {"intent-classification", "open-set", "abstention", "multilingual", "e5"} <= set(
        fm["tags"]
    )
    entry = fm["model-index"][0]
    assert entry["name"].startswith("multilingual-intent-router")
    assert ("robustness variant" in entry["name"]) == (version == "v3")
    (res,) = entry["results"]
    assert res["task"]["type"] == "text-classification"
    assert res["dataset"] == {
        "name": "assessment synthetic logistics intents (not redistributed)",
        "type": "custom",
        "split": "test",
    }
    t = _j(f"results/{results}/track_a.json")["test"]
    vals = {m["type"]: m for m in res["metrics"]}
    assert abs(vals["f1"]["value"] - t["macro_f1"]["point"]) < 5e-5
    assert vals["f1"]["args"] == {"average": "macro"}
    assert abs(vals["accuracy"]["value"] - t["accuracy"]["point"]) < 5e-5
    assert all(m["verified"] is False for m in res["metrics"])
    # huggingface_hub parses the card and accepts the metadata (round trip through its own types)
    parsed = ModelCard(card).data.eval_results
    assert {r.metric_type for r in parsed} == {"f1", "accuracy"}
    assert all(r.verified is False for r in parsed)
    assert ModelCardData(**{k: v for k, v in fm.items() if k != "model-index"}).to_dict()
    again = metadata_eval_result(
        model_pretty_name=entry["name"],
        task_pretty_name="Text Classification",
        task_id="text-classification",
        metrics_pretty_name=["macro-F1"],
        metrics_id=["f1"],
        metrics_value=[vals["f1"]["value"]],
        dataset_pretty_name=res["dataset"]["name"],
        dataset_id="custom",
        dataset_split="test",
    )
    assert again["model-index"][0]["results"][0]["metrics"][0]["value"] == [vals["f1"]["value"]]


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_publish_card_has_no_draft_or_private_wording(version: str) -> None:
    card = mc.render_card(version, ROOT, publish=True)
    for banned in ("DRAFT", "authorised token", "private repo", "resolve at publish"):
        assert banned not in card
    assert re.search(r"\bGG\b", card) is None
    assert _j("report/links.json")["wandb"] in card
    assert "blob/v1.0-submission/notebooks/intent_router_colab.ipynb" in card


def test_v3_card_says_robustness_variant_and_v1_says_shipped(cards: dict[str, str]) -> None:
    assert "NOT the shipped model" in cards["v3"]
    assert "robustness variant" in cards["v3"].lower()
    assert "NOT the shipped model" not in cards["v1"]
    assert "the shipped model" in cards["v1"]
    serving = _j("results/tradeoff_v1_v3.json")["serving"]
    assert f"threshold = {serving['v3']['ood_threshold']['value']:.2f}" in cards["v3"]
    assert f"threshold = {serving['v1']['ood_threshold']['value']:.2f}" in cards["v1"]


def test_template_has_no_hardcoded_decimals() -> None:
    src = (mc.TEMPLATE_DIR / mc.TEMPLATE_NAME).read_text(encoding="utf-8")
    prose = re.sub(r"\{[{%#].*?[}%#]\}", "", src, flags=re.S)
    assert re.findall(r"\d+\.\d+", prose) == []


def test_missing_field_is_an_error_not_a_blank() -> None:
    tpl = mc.make_env().get_template(mc.TEMPLATE_NAME)
    with pytest.raises(jinja2.UndefinedError):
        tpl.render()


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


def test_forbidden_term_check_detects_its_own_term() -> None:
    """Guard against a vacuous check: a text that contains a known term is flagged."""
    (n, h), *_ = _forbidden()
    assert n > 0 and len(h) == 64
    digest = hashlib.sha256(b"probe").hexdigest()
    assert _contains_term("Some PROBE text", 5, digest)
    assert not _contains_term("other text", 5, digest)


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_no_company_name_in_card(cards: dict[str, str], version: str) -> None:
    for n, h in _forbidden():
        assert not _contains_term(cards[version], n, h)


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_no_dataset_text_in_card(cards: dict[str, str], version: str) -> None:
    if not DATASET.exists():
        pytest.skip("data/dataset.csv not available (confidential, gitignored)")
    import pandas as pd

    card = re.sub(r"\s+", " ", cards[version].lower())
    texts = pd.read_csv(DATASET)["text"].astype(str)
    leaks = [t for t in texts if len(t) >= MIN_MESSAGE_CHARS and t.lower() in card]
    assert leaks == []  # (count only: never print dataset text)


def test_template_and_code_files_carry_no_company_name() -> None:
    files = [ROOT / "src/intent_router" / n for n in ("model_card.py", "hub.py", "hub_predict.py")]
    files += [mc.TEMPLATE_DIR / mc.TEMPLATE_NAME, HASHES]
    for f in files:
        for n, h in _forbidden():
            assert not _contains_term(f.read_text(encoding="utf-8"), n, h), f.name


# ------------------------------------------------------------------ review fixes (snippet, flags)
LOCAL_DIRS = {"v1": "serve_model", "v3": "serve_model_v3"}


def _first_python_block_after(card: str, heading: str) -> str:
    section = card.split(heading, 1)[1]
    m = re.search(r"```python\n(.*?)```", section, flags=re.S)
    assert m is not None
    return m.group(1)


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_minimal_snippet_comes_first_and_names_the_repo(
    cards: dict[str, str], version: str
) -> None:
    code = _first_python_block_after(cards[version], "## Usage")
    assert "AutoTokenizer" in code and "AutoModelForSequenceClassification" in code
    assert f'repo = "{mc.REPO_ID}"' in code
    assert ("revision=" in code) == (version == "v3")  # only the robustness variant pins a branch
    if version == "v3":
        assert 'revision="robust-v3"' in code
    assert '"query: "' in code and "id2label" in code
    usage = cards[version].split("## Usage", 1)[1].split("## Training data", 1)[0]
    assert "needs `predict.py` and `ood_bank.safetensors`" in usage  # abstention note
    assert usage.index("AutoTokenizer") < usage.index("snapshot_download")


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_minimal_snippet_runs_against_the_local_package(
    cards: dict[str, str], version: str, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    local = ROOT / LOCAL_DIRS[version]
    if not (local / "model.safetensors").exists():
        pytest.skip(f"local package {local.name}/ not present (gitignored build output)")
    code = _first_python_block_after(cards[version], "## Usage")
    code = re.sub(r', revision="[^"]+"', "", code)  # revision applies to the Hub, not a directory
    code = code.replace(mc.REPO_ID, str(local).replace("\\", "/"))
    id2label = _j(f"{LOCAL_DIRS[version]}/config.json")["id2label"]
    outs = []
    for text in ("where is my parcel", "book a dock appointment"):
        code_i = code.replace("where is my shipment?", text)
        exec(compile(code_i, f"<card-{version}>", "exec"), {})  # noqa: S102 - the card snippet
        outs.append(capsys.readouterr().out.strip())
    assert all(o in set(id2label.values()) for o in outs), outs


def test_publish_flag_drops_the_private_repo_remark() -> None:
    private = mc.render_card("v1", ROOT)
    public = mc.render_card("v1", ROOT, publish=True)
    assert "authorised token" in private
    assert "authorised token" not in public
    assert "snapshot_download" in public  # the snippet itself stays
    assert mc.render_card("v3", ROOT, publish=True).count("authorised token") == 0


def test_cli_publish_flag(tmp_path: Path) -> None:
    out = tmp_path / "README.md"
    mc.main(["--version", "v3", "--root", str(ROOT), "--out", str(out), "--publish"])
    assert "authorised token" not in out.read_text(encoding="utf-8")


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_cards_have_no_internal_shorthand(cards: dict[str, str], version: str) -> None:
    assert re.search(r"\bGG\b", cards[version]) is None
    assert re.search(r"\bGG\b", mc.render_card(version, ROOT, publish=True)) is None


def test_small_n_caveat_only_in_v3_card(cards: dict[str, str]) -> None:
    clause = "too few pairs to claim the shortcut is gone"
    assert clause not in cards["v1"]  # contradicts v1's own non-zero test flip rate
    assert clause in cards["v3"]  # v3's test swap flips are ~0 over few pairs: caveat warranted
    assert clause not in mc.render_card("v1", ROOT, publish=True)


def test_v3_limitation_renders_measured_swap_reduction(cards: dict[str, str]) -> None:
    sw = _j("results/phase4e/axes.json")
    v1_flip = sw["v1"]["flip_rate"]
    tr = _j("results/tradeoff_v1_v3.json")["robustness_cv"][mc.TRADEOFF_SWAP_KEY]
    assert tr["source"].startswith("phase4e/axes.json")
    assert abs(tr["v1"] - v1_flip) < 1e-12
    card = cards["v3"]
    assert "partly keys on" not in card
    assert f"{tr['v3']:.3f} against {v1_flip:.3f} for v1" in card
    idswap = _j("results/phase4e/final_v3_robustness/idswap.json")["results"]
    tests = [r for r in idswap if r["source"] == "test"]
    n = sum(r["n"] for r in tests)
    flips = sum(r["n_flips"] for r in tests)
    assert f"{flips / n:.3f} of {n} pairs" in card


@pytest.mark.parametrize(
    ("version", "results"), [("v1", "results/final"), ("v3", "results/final_v3")]
)
def test_card_renders_model_fingerprint_line(
    cards: dict[str, str], version: str, results: str
) -> None:
    fp = _j(f"{results}/model_version.json")["model_fingerprint"]
    line = f"Model fingerprint (sha256 of fp32 state dict): `{fp}`"
    card = cards[version]
    assert card.count(line) == 1
    assert card.index("\n# ") < card.index(line) < card.index("## Label map")
