"""Upload the publish-mode model cards (README.md only) to the PRIVATE Hub repo, then read back.

    python scripts/push_cards.py             # DRY RUN (default): plan only, no network, no write
    python scripts/push_cards.py --execute   # upload v1 -> branch main, v3 -> branch robust-v3

Per branch: `HfApi.upload_file(README.md, revision=<branch>)` with the commit message
'docs: publish-mode model card'. Nothing else in the repo is touched and visibility is never
changed: the repo must be private before the first upload (otherwise the run aborts) and is
asserted private again after the last one. Each README is then downloaded back from its branch
(`hf_hub_download`, `revision=<branch>`, forced fresh) and its sha256 compared with the locally
rendered card; PASS/FAIL per branch is printed and written to results/hub/cards_pushed.json.
huggingface_hub uses whatever login exists; no token is read or printed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intent_router.model_card import REPO_ID, VERSIONS, render_card

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("results") / "hub" / "cards_pushed.json"
COMMIT_MESSAGE = "docs: publish-mode model card"
README = "README.md"


@dataclass(frozen=True)
class CardJob:
    """One card to push: model version, target branch, rendered bytes and their sha256."""

    version: str
    branch: str
    data: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def build_jobs(root: Path, render: Callable[..., str] = render_card) -> list[CardJob]:
    """Publish-mode card per version, in VERSIONS order (v1 -> main, v3 -> robust-v3)."""
    return [
        CardJob(v, spec["branch"], render(v, root, True).encode("utf-8"))
        for v, spec in sorted(VERSIONS.items())
    ]


def describe(jobs: list[CardJob], repo_id: str) -> list[str]:
    """What an --execute run does, in order (printed by the dry run)."""
    lines = [f"assert {repo_id} is private (abort before any upload otherwise)"]
    for j in jobs:
        lines.append(
            f"upload_file({README}, revision={j.branch!r}, commit_message={COMMIT_MESSAGE!r}) "
            f"from the publish-mode {j.version} card (sha256 {j.sha256})"
        )
    lines.append(f"assert {repo_id} is still private")
    lines += [
        f"hf_hub_download({README}, revision={j.branch!r}), compare sha256: PASS/FAIL" for j in jobs
    ]
    lines.append(f"write {OUT.as_posix()}")
    return lines


def _assert_private(api: Any, repo_id: str, when: str) -> None:
    private = api.model_info(repo_id).private
    if private is not True:
        raise RuntimeError(f"{repo_id} is not private {when} (private={private!r}); stopping")


def push_and_verify(
    jobs: list[CardJob],
    api: Any,
    download: Callable[[str, str], bytes],
    repo_id: str = REPO_ID,
) -> dict[str, Any]:
    """Upload each card, assert privacy before/after, read each README back and compare hashes.

    `download(repo_id, branch)` returns the bytes of README.md on that branch (injected so tests
    need no network). Raises if the repo is not private before any upload.
    """
    _assert_private(api, repo_id, "before the upload")
    for j in jobs:
        api.upload_file(
            path_or_fileobj=j.data,
            path_in_repo=README,
            repo_id=repo_id,
            repo_type="model",
            revision=j.branch,
            commit_message=COMMIT_MESSAGE,
        )
    private_after = api.model_info(repo_id).private is True
    branches = {}
    for j in jobs:
        remote = hashlib.sha256(download(repo_id, j.branch)).hexdigest()
        branches[j.branch] = {
            "version": j.version,
            "local_sha256": j.sha256,
            "remote_sha256": remote,
            "result": "PASS" if remote == j.sha256 else "FAIL",
        }
    ok = private_after and all(b["result"] == "PASS" for b in branches.values())
    return {
        "repo_id": repo_id,
        "private_before": True,
        "private_after": private_after,
        "commit_message": COMMIT_MESSAGE,
        "branches": branches,
        "ok": ok,
    }


def hub_download(repo_id: str, branch: str) -> bytes:
    """README.md bytes of a branch, forced fresh (never the local cache)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, README, revision=branch, force_download=True)
    return Path(path).read_bytes()


def main(argv: list[str] | None = None, api: Any = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--execute", action="store_true", help="really upload (default: dry run)")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
    jobs = build_jobs(a.root)
    if not a.execute:
        print("push_cards: DRY RUN (no network, nothing written; pass --execute)")
        for i, line in enumerate(describe(jobs, REPO_ID), 1):
            print(f"  {i}. WOULD {line}")
        return 0
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    try:
        res = push_and_verify(jobs, api, hub_download)
    except RuntimeError as exc:
        print(f"push_cards: ABORTED: {exc}", file=sys.stderr)
        return 1
    out = a.root / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    for branch, b in res["branches"].items():
        print(f"{b['result']}  {branch}: local {b['local_sha256']} remote {b['remote_sha256']}")
    print(f"private after: {res['private_after']}; wrote {out}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
