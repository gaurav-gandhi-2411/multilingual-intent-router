"""scripts/push_cards.py with fakes: no network, no real upload."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("push_cards", ROOT / "scripts" / "push_cards.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pcd = _load()


class FakeApi:
    """Records uploads; `private` is a list consumed per model_info call (last value repeats)."""

    def __init__(self, private: list[bool]) -> None:
        self._private, self.uploads, self.other_calls = list(private), [], []
        self.store: dict[str, bytes] = {}

    def model_info(self, repo_id: str) -> Any:
        value = self._private.pop(0) if len(self._private) > 1 else self._private[0]
        return SimpleNamespace(private=value)

    def upload_file(self, **kw: Any) -> None:
        self.uploads.append(kw)
        self.store[kw["revision"]] = kw["path_or_fileobj"]

    def __getattr__(self, name: str) -> Any:  # any other API call (visibility etc.) is a bug
        def call(*_a: object, **_k: object) -> None:
            self.other_calls.append(name)
            raise AssertionError(f"unexpected HfApi.{name}")

        return call


_REAL_BUILD_JOBS = pcd.build_jobs  # tests monkeypatch pcd.build_jobs; keep the original


def _jobs() -> list[Any]:
    render = lambda v, _r, publish: f"card {v} publish={publish}"  # noqa: E731
    return _REAL_BUILD_JOBS(ROOT, render)


def test_jobs_map_v1_to_main_and_v3_to_robust_branch_in_publish_mode() -> None:
    jobs = _jobs()
    assert [(j.version, j.branch) for j in jobs] == [("v1", "main"), ("v3", "robust-v3")]
    assert jobs[0].data == b"card v1 publish=True"
    assert jobs[0].sha256 == hashlib.sha256(jobs[0].data).hexdigest()


def test_dry_run_prints_plan_and_touches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeApi([True])
    monkeypatch.setattr(pcd, "build_jobs", lambda _root: _jobs())
    assert pcd.main(["--root", str(tmp_path)], api=api) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "revision='main'" in out and "revision='robust-v3'" in out
    assert pcd.COMMIT_MESSAGE in out and "private" in out
    assert api.uploads == [] and not (tmp_path / "results").exists()


def test_push_uploads_readme_only_to_each_branch_and_passes_on_equal_hashes() -> None:
    api = FakeApi([True, True, True])
    res = pcd.push_and_verify(_jobs(), api, lambda _repo, branch: api.store[branch])
    assert [(u["revision"], u["path_in_repo"], u["commit_message"]) for u in api.uploads] == [
        ("main", "README.md", "docs: publish-mode model card"),
        ("robust-v3", "README.md", "docs: publish-mode model card"),
    ]
    assert all(u["repo_type"] == "model" for u in api.uploads) and api.other_calls == []
    assert res["ok"] is True and res["private_after"] is True
    assert {b["result"] for b in res["branches"].values()} == {"PASS"}


def test_hash_mismatch_is_a_fail() -> None:
    api = FakeApi([True])
    res = pcd.push_and_verify(_jobs(), api, lambda _r, _b: b"something else")
    assert res["ok"] is False
    assert {b["result"] for b in res["branches"].values()} == {"FAIL"}


def test_public_repo_before_upload_aborts_without_uploading() -> None:
    api = FakeApi([False])
    with pytest.raises(RuntimeError, match="not private before"):
        pcd.push_and_verify(_jobs(), api, lambda _r, _b: b"")
    assert api.uploads == []


def test_repo_turning_public_is_reported_not_ok() -> None:
    api = FakeApi([True, False])  # private before, public after
    res = pcd.push_and_verify(_jobs(), api, lambda _repo, branch: api.store[branch])
    assert res["private_after"] is False and res["ok"] is False


def test_execute_writes_the_result_file_and_returns_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeApi([True])
    monkeypatch.setattr(pcd, "build_jobs", lambda _root: _jobs())
    monkeypatch.setattr(pcd, "hub_download", lambda _repo, branch: api.store[branch])
    assert pcd.main(["--execute", "--root", str(tmp_path)], api=api) == 0
    doc = json.loads((tmp_path / pcd.OUT).read_text(encoding="utf-8"))
    assert doc["ok"] is True and set(doc["branches"]) == {"main", "robust-v3"}
    assert "PASS  main" in capsys.readouterr().out
    monkeypatch.setattr(pcd, "hub_download", lambda _repo, _branch: b"x")
    assert pcd.main(["--execute", "--root", str(tmp_path)], api=FakeApi([True])) == 1
