"""Hub round trip: a clean download of the private repo reproduces the saved test preds.

    HUB_REVISION=main pytest tests/test_hub_roundtrip.py            # v1 (default)
    HUB_REVISION=robust-v3 pytest tests/test_hub_roundtrip.py       # v3

Skips (with the reason) without HF auth / network / the local confidential data. Does not append
to the test log: the logged runs are made with `python -m intent_router.hub_roundtrip --write`.
Tolerance: labels identical; probabilities within PROB_TOL = 1e-5 of the saved `prob_*` columns
(plain softmax). Measured max abs diffs are in results/hub/roundtrip_v{1,3}.json (about 2e-6 and
3e-6: the saved arrays came from the GPU training venv, this run is CPU fp32).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from intent_router import hub_roundtrip as rt
from intent_router.model_card import REPO_ID

ROOT = Path(__file__).resolve().parents[1]
REVISION = os.environ.get("HUB_REVISION", "main")
VERSION = os.environ.get("HUB_VERSION", "v3" if REVISION == "robust-v3" else "v1")


def _require_hub() -> None:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.whoami()
        api.model_info(REPO_ID, revision=REVISION)
    except Exception as exc:  # no token, no network, or no access to the private repo
        pytest.skip(f"Hub unavailable for {REPO_ID}@{REVISION}: {type(exc).__name__}")


@pytest.fixture(scope="module")
def result() -> dict:
    if not (ROOT / "data/dataset.csv").exists():
        pytest.skip("data/dataset.csv not available (confidential, gitignored)")
    _require_hub()
    return rt.run_roundtrip(VERSION, REVISION, root=ROOT, write=False)


def test_repo_is_private_and_has_no_onnx() -> None:
    _require_hub()
    from huggingface_hub import HfApi

    info = HfApi().model_info(REPO_ID, revision=REVISION)
    assert info.private is True
    assert not [s.rfilename for s in info.siblings if s.rfilename.endswith(".onnx")]


def test_snapshot_labels_identical_to_saved_test_predictions(result: dict) -> None:
    assert result["n"] == 74
    assert result["label_agreement"] == 1.0
    assert result["n_label_mismatch"] == 0


def test_snapshot_probabilities_within_tolerance(result: dict) -> None:
    assert result["max_abs_prob_diff"] <= rt.PROB_TOL
    assert result["max_abs_prob_diff_temperature_scaled"] <= rt.PROB_TOL


def test_snapshot_abstention_coverage_matches_shipped_ood_json(result: dict) -> None:
    import json

    from intent_router.model_card import VERSIONS

    ood = json.loads(
        (ROOT / VERSIONS[VERSION]["results"] / "ood_shipped.json").read_text(encoding="utf-8")
    )
    assert abs(result["test_coverage_at_threshold"] - ood["test_coverage_at_threshold"]) < 1e-12
    assert (
        result["model_fingerprint"]
        == json.loads(
            (ROOT / VERSIONS[VERSION]["results"] / "model_version.json").read_text(encoding="utf-8")
        )["model_fingerprint"]
    )
