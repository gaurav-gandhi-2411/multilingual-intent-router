"""Stage and push the model to a PRIVATE Hugging Face repo.

    python -m intent_router.hub stage --version v1
    python -m intent_router.hub push  --version v1            # branch main
    python -m intent_router.hub push  --version v3            # branch robust-v3
    python -m intent_router.hub verify --version v3

v1 (shipped) lives on branch `main`, v3 (robustness variant) on `robust-v3`. A staged directory
(outputs/hub_stage/<version>/, gitignored) holds exactly what is uploaded:

    config.json            HF config + id2label/label2id + ood_method, ood_threshold, temperature,
                           query_prefix, max_length (and ood_retention, model_fingerprint, version)
    model.safetensors, tokenizer files
    ood_bank.safetensors   the Mahalanobis bank (arrays `means`, `precision`, `shrinkage`),
                           converted from the serve dir's ood_bank.npz
    predict.py             standalone (torch, transformers, safetensors, numpy); hub_predict.py
    README.md              model card rendered from results JSON (model_card.py)

ONNX files are never uploaded. The repo is created with private=True and its privacy is
re-checked through the API before any upload (and after it); a repo that is not private aborts the
push. Auth is whatever huggingface_hub has cached; no token is read or printed here.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from intent_router.model_card import REPO_ID, VERSIONS, render_card

SERVE_DIRS = {"v1": "serve_model", "v3": "serve_model_v3"}
STAGE_ROOT = Path("outputs/hub_stage")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)
BANK_SRC = "ood_bank.npz"
BANK_DST = "ood_bank.safetensors"
EXPECTED_FILES = frozenset(
    {"config.json", "model.safetensors", BANK_DST, "predict.py", "README.md", "tokenizer.json"}
)
IGNORE = ["__pycache__", "__pycache__/*", "*.pyc", "*.onnx", "onnx/*"]  # never uploaded
TOL = 1e-6  # shipped numbers must match results JSON to float round-off


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def hub_config(model_config: dict[str, Any], router: dict[str, Any]) -> dict[str, Any]:
    """HF config.json + the custom serving fields (keeps AutoModel.from_pretrained loadable)."""
    cfg = dict(model_config)
    cfg["query_prefix"] = router["prefix"]
    cfg["max_length"] = int(router["max_len"])
    cfg["temperature"] = float(router["temperature"])
    cfg["ood_method"] = router["ood_method"]
    cfg["ood_threshold"] = float(router["ood_threshold"])
    cfg["ood_retention"] = float(router["retention"])
    cfg["model_fingerprint"] = router["model_fingerprint"]
    cfg["router_version"] = router["version"]
    if [cfg["id2label"][str(i)] for i in range(len(router["labels"]))] != router["labels"]:
        raise ValueError("config.json id2label differs from router_config labels")
    return cfg


def npz_to_safetensors(src: Path, dst: Path) -> None:
    """Convert the OOD bank, preserving array names; verifies a bit-exact round trip."""
    from safetensors.numpy import load_file, save_file

    z = np.load(src)
    arrays = {k: np.ascontiguousarray(z[k]) for k in z.files}
    save_file(arrays, str(dst), metadata={"format": "np"})
    back = load_file(str(dst))
    if set(back) != set(arrays) or any(
        not np.array_equal(back[k], arrays[k], equal_nan=True) for k in arrays
    ):
        raise ValueError(f"{dst}: safetensors bank differs from {src}")


def check_against_results(version: str, router: dict[str, Any], root: Path) -> None:
    """The serve dir must agree with the results JSON this card will quote."""
    rd = root / VERSIONS[version]["results"]
    mv, ood, ta = (
        _read(rd / n) for n in ("model_version.json", "ood_shipped.json", "track_a.json")
    )
    if router["version"] != version or router["model_fingerprint"] != mv["model_fingerprint"]:
        raise ValueError(f"{version}: serve dir fingerprint/version differs from {rd}")
    if abs(router["ood_threshold"] - ood["threshold"]) > TOL:
        raise ValueError(f"{version}: serve threshold differs from ood_shipped.json")
    if abs(router["temperature"] - ta["calibration"]["temperature"]) > TOL:
        raise ValueError(f"{version}: serve temperature differs from track_a.json")


def build_hub_dir(
    version: str,
    root: Path = Path(),
    stage_root: Path | None = None,
    serve_dir: Path | None = None,
) -> Path:
    """Stage outputs/hub_stage/<version>/ and return it."""
    if version not in VERSIONS:
        raise ValueError(f"unknown version {version!r}; expected one of {sorted(VERSIONS)}")
    src = serve_dir or root / SERVE_DIRS[version]
    out = (stage_root or root / STAGE_ROOT) / version
    for name in ("config.json", "router_config.json", "model.safetensors", BANK_SRC):
        if not (src / name).exists():
            raise FileNotFoundError(f"{src / name} missing: run intent_router.package first")
    router = _read(src / "router_config.json")
    check_against_results(version, router, root)

    out.mkdir(parents=True, exist_ok=True)  # files are overwritten; strays are rejected below
    shutil.copyfile(src / "model.safetensors", out / "model.safetensors")
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copyfile(src / name, out / name)
    cfg = hub_config(_read(src / "config.json"), router)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    npz_to_safetensors(src / BANK_SRC, out / BANK_DST)
    shutil.copyfile(Path(__file__).with_name("hub_predict.py"), out / "predict.py")
    (out / "README.md").write_text(render_card(version, root), encoding="utf-8")
    present = {p.name for p in out.iterdir()}
    if EXPECTED_FILES - present:
        raise FileNotFoundError(f"staged dir incomplete: {sorted(EXPECTED_FILES - present)}")
    stray = present - EXPECTED_FILES - set(TOKENIZER_FILES) - {"__pycache__"}  # never uploaded
    if stray:
        raise ValueError(f"{out} has unexpected files {sorted(stray)}: remove them by hand")
    return out


def staged_files(stage: Path) -> dict[str, int]:
    """{relative path: size in bytes} of a staged directory."""
    return {
        p.relative_to(stage).as_posix(): p.stat().st_size
        for p in sorted(stage.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".onnx")
    }


def remote_files(api: Any, repo_id: str, revision: str) -> tuple[str, dict[str, int]]:
    """(commit sha, {path: size}) of a repo revision as reported by the Hub."""
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    return info.sha, {s.rfilename: s.size for s in (info.siblings or [])}


def assert_private(api: Any, repo_id: str) -> None:
    """Raise unless the Hub reports the repo as private."""
    info = api.model_info(repo_id)
    if info.private is not True:
        raise RuntimeError(f"{repo_id} is not private (private={info.private!r}); aborting")


def push(
    version: str,
    branch: str | None = None,
    repo_id: str = REPO_ID,
    root: Path = Path(),
    api: Any = None,
) -> dict[str, Any]:
    """Create the repo PRIVATE, create the branch, upload the staged dir, then verify it."""
    from huggingface_hub import HfApi

    api = api or HfApi()
    branch = branch or VERSIONS[version]["branch"]
    stage = root / STAGE_ROOT / version
    if not stage.is_dir():
        raise FileNotFoundError(f"{stage} not staged: run build_hub_dir({version!r}) first")
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    assert_private(api, repo_id)  # before any upload: exist_ok may have returned a public repo
    if branch != "main":
        api.create_branch(repo_id, branch=branch, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(stage),
        revision=branch,
        ignore_patterns=IGNORE,
        commit_message=f"{version}: model, tokenizer, OOD bank, predict.py, model card",
    )
    return verify(version, branch, repo_id, root, api)


def verify(
    version: str,
    branch: str | None = None,
    repo_id: str = REPO_ID,
    root: Path = Path(),
    api: Any = None,
) -> dict[str, Any]:
    """Check privacy, that the revision exists and that its files/sizes equal the staged dir."""
    from huggingface_hub import HfApi

    api = api or HfApi()
    branch = branch or VERSIONS[version]["branch"]
    assert_private(api, repo_id)
    sha, remote = remote_files(api, repo_id, branch)
    local = staged_files(root / STAGE_ROOT / version)
    remote = {k: v for k, v in remote.items() if k != ".gitattributes"}
    if remote != local:
        diff = {
            k: (local.get(k), remote.get(k))
            for k in set(local) | set(remote)
            if local.get(k) != remote.get(k)
        }
        raise RuntimeError(f"{repo_id}@{branch} differs from staged files (local, remote): {diff}")
    return {
        "repo_id": repo_id,
        "private": True,
        "branch": branch,
        "sha": sha,
        "files": remote,
        "has_onnx": any(k.endswith(".onnx") or k.startswith("onnx/") for k in remote),
    }


def main(argv: list[str] | None = None) -> None:
    """CLI: stage / push / verify."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("stage", "push", "verify"))
    ap.add_argument("--version", choices=sorted(VERSIONS), required=True)
    ap.add_argument("--branch", default=None)
    ap.add_argument("--repo-id", default=REPO_ID)
    a = ap.parse_args(argv)
    if a.action == "stage":
        out = build_hub_dir(a.version)
        print(json.dumps({"staged": str(out), "files": staged_files(out)}, indent=2))
        return
    fn = push if a.action == "push" else verify
    kw: dict[str, Any] = {"branch": a.branch, "repo_id": a.repo_id}
    print(json.dumps(fn(a.version, **kw), indent=2))


if __name__ == "__main__":
    main()
