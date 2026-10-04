"""Hub staging/push logic: config fields, bank conversion, private-only push."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from intent_router import hub, hub_predict
from intent_router.predict import MahalanobisBank, mahalanobis_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["a", "b", "c"]
ROUTER = {
    "labels": LABELS,
    "prefix": "query: ",
    "max_len": 64,
    "temperature": 1.25,
    "ood_method": "maha_ft",
    "ood_threshold": -3.5,
    "retention": 0.95,
    "model_fingerprint": "f" * 64,
    "version": "v9",
}
MODEL_CFG = {
    "model_type": "xlm-roberta",
    "id2label": {str(i): lab for i, lab in enumerate(LABELS)},
    "label2id": {lab: i for i, lab in enumerate(LABELS)},
}


def test_hub_config_adds_custom_fields_and_keeps_labels() -> None:
    cfg = hub.hub_config(MODEL_CFG, ROUTER)
    for k in ("ood_method", "ood_threshold", "temperature", "query_prefix", "max_length"):
        assert k in cfg
    assert (
        cfg["temperature"] == 1.25 and cfg["max_length"] == 64 and cfg["query_prefix"] == "query: "
    )
    assert cfg["id2label"] == MODEL_CFG["id2label"] and cfg["label2id"] == MODEL_CFG["label2id"]
    assert "ood_method" not in MODEL_CFG  # input not mutated


def test_hub_config_rejects_label_mismatch() -> None:
    with pytest.raises(ValueError, match="id2label"):
        hub.hub_config(MODEL_CFG, {**ROUTER, "labels": ["a", "c", "b"]})


def _bank(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    means = rng.normal(size=(3, 5))
    prec = np.eye(5) * 2.0
    npz = tmp_path / "ood_bank.npz"
    np.savez(npz, means=means, precision=prec, shrinkage=np.float64(0.25))
    return npz, tmp_path / "ood_bank.safetensors"


def test_bank_conversion_preserves_names_and_values(tmp_path: Path) -> None:
    from safetensors.numpy import load_file

    npz, st = _bank(tmp_path)
    hub.npz_to_safetensors(npz, st)
    back = load_file(str(st))
    z = np.load(npz)
    assert set(back) == {"means", "precision", "shrinkage"} == set(z.files)
    for k in z.files:
        # safetensors stores a 0-d scalar (shrinkage) as shape (1,); values are identical
        assert np.array_equal(back[k].ravel(), z[k].ravel(), equal_nan=True)


def test_predict_loader_accepts_either_bank_format(tmp_path: Path) -> None:
    npz, st = _bank(tmp_path)
    hub.npz_to_safetensors(npz, st)
    a, b = MahalanobisBank.from_file(npz), MahalanobisBank.from_file(st)
    x = np.random.default_rng(0).normal(size=(4, 5))
    assert np.array_equal(a.score(x), b.score(x))


def test_hub_predict_math_matches_serving_reference() -> None:
    rng = np.random.default_rng(1)
    x, means, prec = rng.normal(size=(6, 5)), rng.normal(size=(3, 5)), np.eye(5)
    assert np.array_equal(
        hub_predict.mahalanobis_score(x, means, prec), mahalanobis_score(x, means, prec)
    )
    p = hub_predict.softmax(rng.normal(size=(6, 3)), 1.7)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_hub_predict_is_standalone() -> None:
    import ast

    tree = ast.parse((ROOT / "src/intent_router/hub_predict.py").read_text(encoding="utf-8"))
    mods = [
        n.module if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in (n.names if isinstance(n, ast.Import) else [None])
    ]
    assert not [m for m in mods if m and m.startswith("intent_router")]


class FakeApi:
    """Records calls; reports the repo as private or public."""

    def __init__(self, private: bool, files: dict[str, int] | None = None) -> None:
        self.private = private
        self.files = files or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create_repo(self, repo_id: str, **kw: Any) -> None:
        self.calls.append(("create_repo", kw))

    def create_branch(self, repo_id: str, **kw: Any) -> None:
        self.calls.append(("create_branch", kw))

    def upload_folder(self, **kw: Any) -> None:
        self.calls.append(("upload_folder", kw))

    def model_info(self, repo_id: str, revision: str | None = None, **kw: Any) -> Any:
        sibs = [types.SimpleNamespace(rfilename=k, size=v) for k, v in self.files.items()]
        return types.SimpleNamespace(private=self.private, sha="abc123", siblings=sibs)


def _staged(root: Path, version: str) -> dict[str, int]:
    d = root / hub.STAGE_ROOT / version
    d.mkdir(parents=True)
    (d / "README.md").write_text("x", encoding="utf-8")
    (d / "config.json").write_text("{}", encoding="utf-8")
    return hub.staged_files(d)


def test_push_creates_private_repo_branch_and_uploads_to_branch(tmp_path: Path) -> None:
    files = _staged(tmp_path, "v3")
    api = FakeApi(private=True, files={**files, ".gitattributes": 10})
    out = hub.push("v3", root=tmp_path, api=api)
    names = [c[0] for c in api.calls]
    assert names == ["create_repo", "create_branch", "upload_folder"]
    assert api.calls[0][1]["private"] is True
    assert api.calls[1][1]["branch"] == "robust-v3"
    assert api.calls[2][1]["revision"] == "robust-v3"
    assert "*.onnx" in api.calls[2][1]["ignore_patterns"]
    assert out["private"] is True and out["sha"] == "abc123" and out["has_onnx"] is False


def test_push_to_main_creates_no_branch(tmp_path: Path) -> None:
    files = _staged(tmp_path, "v1")
    api = FakeApi(private=True, files=files)
    hub.push("v1", root=tmp_path, api=api)
    assert [c[0] for c in api.calls] == ["create_repo", "upload_folder"]


def test_push_aborts_before_upload_if_repo_is_not_private(tmp_path: Path) -> None:
    _staged(tmp_path, "v1")
    api = FakeApi(private=False)
    with pytest.raises(RuntimeError, match="not private"):
        hub.push("v1", root=tmp_path, api=api)
    assert "upload_folder" not in [c[0] for c in api.calls]


def test_verify_detects_size_mismatch(tmp_path: Path) -> None:
    files = _staged(tmp_path, "v1")
    bad = dict(files)
    bad["README.md"] += 1
    with pytest.raises(RuntimeError, match="differs from staged"):
        hub.verify("v1", root=tmp_path, api=FakeApi(private=True, files=bad))


def test_staged_files_ignores_bytecode(tmp_path: Path) -> None:
    files = _staged(tmp_path, "v1")
    cache = tmp_path / hub.STAGE_ROOT / "v1" / "__pycache__"
    cache.mkdir()
    (cache / "predict.cpython-312.pyc").write_bytes(b"x")
    assert hub.staged_files(tmp_path / hub.STAGE_ROOT / "v1") == files


@pytest.mark.skipif(not (ROOT / "serve_model").exists(), reason="serve_model/ not built")
def test_build_hub_dir_v1_stages_expected_files(tmp_path: Path) -> None:
    out = hub.build_hub_dir("v1", ROOT, stage_root=tmp_path)
    names = {p.name for p in out.iterdir()}
    assert names >= hub.EXPECTED_FILES
    assert not any(n.endswith(".onnx") for n in names)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    ref = json.loads((ROOT / "serve_model/router_config.json").read_text(encoding="utf-8"))
    assert cfg["temperature"] == ref["temperature"]
    assert cfg["ood_threshold"] == ref["ood_threshold"]
    assert cfg["ood_method"] == "maha_ft" and cfg["max_length"] == 64
    assert len(cfg["id2label"]) == 12 and len(cfg["label2id"]) == 12
