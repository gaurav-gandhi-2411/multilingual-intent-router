"""Generate notebooks/intent_router_colab.ipynb.

Writes raw nbformat-4.5 JSON (nbformat is not a project dependency), so the notebook is
reproducible with `python scripts/build_colab_notebook.py` and diffs stay reviewable.
All runtime estimates below are derived from committed result files; the source of each
number is cited next to it in the notebook text.
"""

# ruff: noqa: E501 - cell sources are embedded notebook text, wrapped for Colab not for 100 cols
from __future__ import annotations

import argparse
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/intent_router_colab.ipynb")


def md(text: str) -> dict:
    """A markdown cell (id assigned later)."""
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    """A code cell with no outputs (the notebook is committed unexecuted)."""
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


TITLE = r"""
# intent-router: canonical Colab run

This notebook reproduces the v1 intent router end to end on a Colab T4: it clones the public repo at a pinned tag,
trains the final model (seed 42), makes the single Track A test evaluation, runs the Track B open-set headline
(3 model seeds), shows the results inline (test confusion matrix and a summary table next to the committed
values), tries the published model from the Hugging Face Hub on three synthetic messages, compares every fresh
number with the committed one under explicit tolerances (one `REPRODUCTION: PASS/FAIL` line) and times CPU latency.

**Run all with no edits:** Runtime > Run all. Only the dataset upload dialog (section 3) needs interaction.

**Runtime on a T4, default path (ESTIMATE, from this notebook's own labelled figures below):** v1 training ~6 min
(section 6) + Track B ~3-5 min (section 7) = about 9-11 min for those two stages. Install, Track A evaluate, the
Hub demo and the latency cell are **NOT MEASURED** on Colab; add them on top.

**Optional flags (both off by default):**

- `USE_WANDB = True` (section 4) logs runs to W&B project `multilingual-intent-router`, group `colab`; needs the Colab secret `WANDB_API_KEY`.
- `PUSH = True` (section 9) uploads results to W&B / a private HF repo; needs the Colab secret `HF_TOKEN` (and `USE_WANDB` / `HF_REPO_ID`).

Runtime: **Colab free T4 GPU** (Runtime > Change runtime type > T4 GPU), Python 3.12.
The numbers in the report that are labelled "Colab" come from this notebook.

> **The setup cell clones the PUBLIC repo at a pinned tag.** `PUBLIC_REPO_URL` in the setup cell already points
> at it; the cell refuses to run with a placeholder.

What this notebook does, top to bottom:

1. clone the repo at a pinned tag, install pinned requirements;
2. load `dataset.csv` (upload or Google Drive) and check it against `splits/splits.csv`;
3. read `HF_TOKEN` / `WANDB_API_KEY` from Colab secrets (optional);
4. v1 final training (e5-base, lr 3e-5, 20-epoch schedule stopped at epoch 9, seed 42, eager attention);
5. Track A (one logged test evaluation, written to `results_colab/`, not to the committed `results/`);
6. load the published Hub model and predict 3 synthetic messages (skipped with a message if the repo is unreachable);
7. Track B headline holdout, model seeds 42/43/44;
8. inline results: test confusion matrix and Track A / Track B summary (fresh next to committed);
9. compare the fresh numbers with the committed results using explicit tolerances (PASS/FAIL per metric, one verdict);
10. CPU batch-1 latency of the fresh model (PyTorch fp32 and ONNX fp32) to `results_colab/serving/latency_colab.json`;
11. optional push (off by default) and opt-in long stages (all off by default).

Scope note: LOCO, robustness and packaging are larger stages. Here LOCO is an opt-in cell;
robustness and packaging are **not** part of this notebook.

Reproducibility caveat: a T4 is not bitwise-equal to the RTX 3070 the committed numbers came from
(different kernels, fp16 autocast). The comparison cell reports deltas against tolerances; it does not
assert equality.
"""

SETUP_MD = r"""
## 1. Setup: clone at a pinned tag

`REPO_TAG` points at the public snapshot commit. Cloning a tag with `--depth 1` pins the code, so a
rerun in a year runs the same code. `PUBLIC_REPO_URL` below is the public repository
(`https://github.com/gaurav-gandhi-2411/multilingual-intent-router`, tag `v1.0-submission`); no token
is needed. Change it only to run a fork.
"""

SETUP = r"""
import os
import subprocess
import sys
from pathlib import Path

REPO_TAG = "v1.0-submission"
PLACEHOLDER = "[PUBLIC REPO URL — filled at publish time]"
# Public repository (change only to run a fork):
PUBLIC_REPO_URL = "https://github.com/gaurav-gandhi-2411/multilingual-intent-router"
WORKDIR = Path("/content")

if PUBLIC_REPO_URL == PLACEHOLDER or not PUBLIC_REPO_URL.startswith("https://"):
    raise RuntimeError(
        "Set PUBLIC_REPO_URL in this cell to the public repository URL "
        "(https://github.com/<owner>/<repo>) before running."
    )
clone_url = PUBLIC_REPO_URL if PUBLIC_REPO_URL.endswith(".git") else PUBLIC_REPO_URL + ".git"
REPO_DIR = WORKDIR / Path(PUBLIC_REPO_URL.removesuffix(".git")).name

# Optional fallback, ONLY if the repository is ever made private again: add a Colab secret named
# CLONE_TOKEN, uncomment the next two lines and clone from auth_url instead of clone_url. Never print it.
#   from google.colab import userdata
#   auth_url = clone_url.replace("https://", f"https://{userdata.get('CLONE_TOKEN')}@")

if not REPO_DIR.exists():
    res = subprocess.run(
        ["git", "clone", "--branch", REPO_TAG, "--depth", "1", clone_url, str(REPO_DIR)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"git clone of tag {REPO_TAG!r} from {PUBLIC_REPO_URL} failed (is the repository public, "
            f"and does the tag exist?): {res.stderr}"
        )
os.chdir(REPO_DIR)
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
print(f"repo at tag {REPO_TAG}, commit {head}")
"""

INSTALL_MD = r"""
## 2. Install pinned requirements

`requirements.txt` is pinned from the local Windows venv (torch `2.11.0+cu128` from the PyTorch cu128 index).
Trade-off:

- `PIN_TORCH = True` (default): install the exact pinned torch from the cu128 index. Closest to the committed
  environment, but a large download (~GBs) and it replaces Colab's preinstalled torch.
- `PIN_TORCH = False`: keep Colab's preinstalled torch (faster, smaller), at the cost of a different
  torch build than the one that produced the committed numbers. Versions are printed either way so the
  report can state what ran.

Always dropped: `colorama` (Windows-only) and the dev-only tools `pytest`, `ruff`, `pluggy`, `iniconfig`.
Every pipeline step below runs in a fresh subprocess, so no runtime restart is needed after installing.
"""

INSTALL = r"""
import importlib.metadata as md

PIN_TORCH = True
DROP = {"colorama", "pytest", "ruff", "pluggy", "iniconfig"}
if not PIN_TORCH:
    DROP.add("torch")

kept = []
for line in Path("requirements.txt").read_text().splitlines():
    s = line.strip()
    if not s or s.startswith("#"):
        continue
    if s.startswith("--extra-index-url"):
        if PIN_TORCH:
            kept.append(s)
        continue
    if s.split("==")[0].lower() in DROP:
        continue
    kept.append(s)
Path("/content/colab_cfg").mkdir(exist_ok=True)
req = Path("/content/colab_cfg/requirements_colab.txt")
req.write_text("\n".join(kept) + "\n")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)

for pkg in ("torch", "transformers", "numpy", "pandas", "scikit-learn", "wandb", "huggingface-hub"):
    print(f"{pkg}=={md.version(pkg)}")
subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
import torch  # noqa: E402 - after install on purpose

print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
"""

CLEAN_MD = r"""
## 2b. Remove preinstalled vision/audio packages, then check the stack imports

This project is text-only. Colab preinstalls `torchvision` and `torchaudio` (built for the runtime's own CUDA,
13.0 at the time of writing) and `peft`. The pinned `torch` here is `2.11.0+cu128` (CUDA 12.8), so the preinstalled
`torchvision` no longer matches it. `transformers` imports `torchvision` when it is importable, and so does `peft`
(which `transformers` also imports when present), so a mismatch fails with a `RuntimeError` and then
`Could not import get_linear_schedule_with_warmup` at the first `transformers` import.

Our code never imports `torchvision`, `torchaudio` or `peft` (checked with grep over `src/`, `scripts/` and every
notebook cell), and `requirements.txt` pins none of them, so the cell below uninstalls all three. It runs AFTER the
pinned install so a reinstall cannot bring them back, and it does not fail when a package is absent (`-y`; pip only
warns). `accelerate` is kept: `requirements.txt` pins it.

The next cell is an assert: it imports `torch` and `transformers` in a fresh subprocess and raises with the
subprocess stderr if that fails. It also applies with `PIN_TORCH = False` (Colab's own torch): it catches any
torch/torchvision mismatch before the long stages start, instead of 10 minutes in.
"""

UNINSTALL = r"""
import subprocess
import sys

# Equivalent to: pip uninstall -y torchvision torchaudio peft
# -y: no prompt; an absent package only makes pip print a warning (exit code 0), so this never fails the cell.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio", "peft"], check=False)
"""

ASSERT_IMPORTS = r"""
import subprocess
import sys

CHECK = (
    "import torch, transformers; "
    "from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification; "
    "print(torch.__version__, torch.version.cuda, transformers.__version__)"
)
chk = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True)
if chk.returncode != 0:
    raise RuntimeError(
        "torch/torchvision CUDA mismatch: run Runtime → Disconnect and delete runtime, then re-run from the top.\n"
        "The import check `import torch, transformers; from transformers import get_linear_schedule_with_warmup` "
        f"failed (exit code {chk.returncode}). stderr (tail):\n{chk.stderr[-1500:]}"
    )
print(chk.stdout.strip())
"""

DATA_MD = r"""
## 3. Data: `dataset.csv`

The dataset is confidential and gitignored, so it is not in the clone. Set `USE_DRIVE = True` and
`DRIVE_PATH` to read it from Google Drive, otherwise the cell opens an upload dialog.

Verification (rows are never printed): the repo records no data hash, so the file is checked against what
the repo does record. Columns `id,text,label`, exactly 500 rows, and the id set equals the id set of
`splits/splits.csv`. The file's sha256 is printed for the run record; set `EXPECTED_SHA256` to enforce it.
"""

DATA = r"""
import hashlib
import shutil

import pandas as pd

USE_DRIVE = False
DRIVE_PATH = "/content/drive/MyDrive/intent-router/dataset.csv"
EXPECTED_SHA256 = None  # no hash is stored in the repo; paste one here to enforce it

Path("data").mkdir(exist_ok=True)
dest = Path("data/dataset.csv")
if USE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    shutil.copy(DRIVE_PATH, dest)
else:
    from google.colab import files

    uploaded = files.upload()
    if len(uploaded) != 1:
        raise RuntimeError(f"upload exactly one file, got {len(uploaded)}")
    dest.write_bytes(next(iter(uploaded.values())))

sha = hashlib.sha256(dest.read_bytes()).hexdigest()
if EXPECTED_SHA256 and sha != EXPECTED_SHA256:
    raise RuntimeError(f"dataset sha256 mismatch: got {sha}")
df = pd.read_csv(dest)
if list(df.columns) != ["id", "text", "label"]:
    raise RuntimeError(f"unexpected columns: {list(df.columns)}")
if len(df) != 500:
    raise RuntimeError(f"expected 500 rows, got {len(df)}")
splits = pd.read_csv("splits/splits.csv")
if set(df["id"]) != set(splits["id"]):
    raise RuntimeError("dataset ids differ from splits/splits.csv: wrong dataset file")
print(f"dataset ok: {len(df)} rows, {df['label'].nunique()} labels, sha256={sha}")
del df
"""

SECRETS_MD = r"""
## 4. Secrets (optional)

`HF_TOKEN` and `WANDB_API_KEY` come from Colab secrets and go into `os.environ` only (child processes
inherit them). Values are never printed. W&B is optional: `USE_WANDB = False` runs everything with logging
disabled. With `USE_WANDB = True` runs go to project `multilingual-intent-router`, group `colab`.
"""

SECRETS = r"""
USE_WANDB = False

for name in ("HF_TOKEN", "WANDB_API_KEY"):
    try:
        from google.colab import userdata

        os.environ[name] = userdata.get(name)
        print(f"{name}: set from Colab secrets")
    except Exception:  # noqa: BLE001 - not on Colab or secret missing
        print(f"{name}: not available")
if USE_WANDB and not os.environ.get("WANDB_API_KEY"):
    raise RuntimeError("USE_WANDB=True needs the Colab secret WANDB_API_KEY")
"""

PIPE_MD = r"""
## 5. Pipeline helpers and config overrides

The committed `results/` holds the RTX 3070 numbers and the committed test-once logs. A fresh clone would
collide with them (the test-once guard refuses a second evaluation of the same model fingerprint), and
overwriting them would destroy the comparison baseline. So this notebook writes generated copies of the
configs to `/content/colab_cfg/` (outside the repo) with every **output** path moved from `results/...` to
`results_colab/...`; inputs (e.g. `results/bakeoff/selection.json`, `results/oof`) stay read-only. The test
log therefore starts empty, and the guard works as designed within the Colab run.
"""

PIPE = r'''
import yaml

OUT_ROOT = "results_colab"
CFG_DIR = Path("/content/colab_cfg")
# Top-level keys that are OUTPUTS (or intermediate files a later stage reads from this same run).
OUTPUT_KEYS = (
    "results_dir", "figures_dir", "test_eval_log", "test_inference_log", "figure_path", "oof_dir",
    "test_predictions_path", "headline_scores_path", "headline_json_path",
)


def make_cfg(name: str, **extra) -> str:
    """Write /content/colab_cfg/<name>.yaml with outputs redirected to results_colab/; return its path."""
    cfg = yaml.safe_load(Path(f"configs/{name}.yaml").read_text())
    for key in OUTPUT_KEYS:
        if isinstance(cfg.get(key), str) and cfg[key].startswith("results/"):
            cfg[key] = cfg[key].replace("results/", f"{OUT_ROOT}/", 1)
    if "wandb" in cfg:
        cfg["wandb"]["enabled"] = USE_WANDB
        cfg["wandb"]["project"] = "multilingual-intent-router"
        for gkey in [k for k in cfg["wandb"] if k.startswith("group")]:
            cfg["wandb"][gkey] = "colab"
    for key, value in extra.items():
        cfg[key] = value
    out = CFG_DIR / f"{name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out)


def run_py(module: str, *args: str) -> None:
    """Run `python -m intent_router.<module> ...` in a subprocess, streaming its output."""
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_DIR / "src"),
        "INTENT_ROUTER_GPU_LOCK": "/content/.gpu.lock",  # the default path is a Windows path
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [sys.executable, "-u", "-m", f"intent_router.{module}", *args]
    print("$", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",  # progress bars are UTF-8; the locale codec (cp1252 on Windows) can choke
    )
    for line in proc.stdout:
        print(line, end="")
    if proc.wait() != 0:
        raise RuntimeError(f"{module} exited with code {proc.returncode}")


FINAL_CFG = make_cfg("final")
TRACKB_CFG = make_cfg("trackb", final_config=FINAL_CFG)
WANDB_FLAG = [] if USE_WANDB else ["--no-wandb"]
print(FINAL_CFG, TRACKB_CFG)
'''

FINAL_MD = r"""
## 6. v1 final training and Track A

Config values come from `configs/final.yaml` (copied with redirected outputs): `intfloat/multilingual-e5-base`,
lr 3e-5, batch 16, 20-epoch LR schedule with `stop_epoch: 9` (schedule not re-scaled), fp16 autocast, seed 42,
`attn_implementation: eager`. The `train` stage trains twice (determinism check) plus one sdpa timing run;
the `evaluate` stage makes the single logged test evaluation (Track A).

ESTIMATE of runtime on a T4: about 3 min on the RTX 3070 for the train stage (`stage_wall_clock_s` = 185.8 s
in `results/final/train_summary.json`). T4 slowdown is NOT MEASURED; assuming up to ~2x gives ~6 min.
The evaluate stage (baselines refit with inner CV, bootstrap, calibration) was not timed separately: NOT MEASURED.

The evaluate cell is skipped if `results_colab/final/track_a.json` exists: the test-once guard would refuse
a second evaluation of the same model anyway. To redo it deliberately, delete `results_colab/final/` and rerun.
"""

FINAL = r"""
if Path(f"{OUT_ROOT}/final/track_a.json").exists():
    print("Track A already computed in this run: skipping (test-once guard)")
else:
    run_py("final", "--config", FINAL_CFG, "--stage", "train", *WANDB_FLAG)
    run_py("final", "--config", FINAL_CFG, "--stage", "evaluate", *WANDB_FLAG)
"""

TRACKB_MD = r"""
## 7. Track B headline holdout (3 seeds)

Holdout classes `yard_management`, `document_processing`; model seeds 42/43/44 (from `configs/trackb.yaml`).
ESTIMATE: 3 runs x ~30 s on the RTX 3070 (`wall_clock_s` per run in `results/trackb/headline.json`: 30.5, 30.2,
30.4 s) plus the frozen-e5 feature cache; ~3-5 min on a T4 assuming a ~2x slowdown (NOT MEASURED).
"""

TRACKB = r"""
if Path(f"{OUT_ROOT}/trackb/headline.json").exists():
    print("Track B headline already computed in this run: skipping")
else:
    run_py("trackb", "--config", TRACKB_CFG, "--stage", "headline", *WANDB_FLAG)
    # the headline stage only runs the seeds; `report` aggregates them into headline.json (what sections 7b/8 read)
    run_py("trackb", "--config", TRACKB_CFG, "--stage", "report", *WANDB_FLAG)
"""

HF_MD = r"""
## 6b. Try the published model from the Hugging Face Hub

Independent of the model trained above: loads `gauravgandhi2411/multilingual-intent-router` with
`AutoTokenizer` + `AutoModelForSequenceClassification` and predicts three **synthetic** logistics messages
(written for this notebook, not taken from the dataset), printing label and temperature-scaled confidence
(temperature and the `query: ` prefix come from the Hub `config.json`). Then it makes the abstention call with the
repo's own `intent_router.hub_predict.IntentRouter` (the same code as the Hub's `predict.py`; it reads
`ood_bank.safetensors` from the same repo snapshot): `abstained` is the Mahalanobis out-of-distribution flag, shown
for the three messages plus two more synthetic probes (a far off-topic message and a novel in-domain request), as a
table of label, confidence, ood_score, threshold and abstained. Far off-topic messages are routed to the trained
catch-all `other`; the abstention gate targets novel in-domain intents, and its measured rejection is in report
section 6 (Track B). If the Hub repo cannot be reached (private, 401, no
network) each part prints a clear message and the notebook **continues**.
"""

HF = r"""
HF_REPO = "gauravgandhi2411/multilingual-intent-router"
DEMO_TEXTS = [  # synthetic, written for this notebook; not dataset rows
    "Where is my pallet of printer cartridges? The tracking page has not updated since Tuesday.",
    "Please book a dock slot at the Rotterdam warehouse for the 14th, around nine in the morning.",
    "Besoin d'une facture corrigee pour l'expedition de la semaine derniere.",
]
OOD_TEXT = "What is a good recipe for a chocolate birthday cake?"  # synthetic far-off-topic probe
NOVEL_TEXT = "can you send me the carbon emissions report per lane for Q3?"  # synthetic novel in-domain probe
PROBES = DEMO_TEXTS + [OOD_TEXT, NOVEL_TEXT]

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(HF_REPO)
    hf_model = AutoModelForSequenceClassification.from_pretrained(HF_REPO).eval()  # CPU: 3 short strings
    hf_cfg = hf_model.config
    enc = hf_tok(
        [getattr(hf_cfg, "query_prefix", "query: ") + t for t in DEMO_TEXTS],
        padding=True, truncation=True, max_length=int(getattr(hf_cfg, "max_length", 64)), return_tensors="pt",
    )
    with torch.inference_mode():
        logits = hf_model(**enc).logits.double() / float(getattr(hf_cfg, "temperature", 1.0))
    for text, p in zip(DEMO_TEXTS, torch.softmax(logits, dim=-1), strict=False):
        j = int(p.argmax())
        print(f"{hf_cfg.id2label[j]:28s} confidence={float(p[j]):.3f}  <- {text[:60]}")
except Exception as exc:  # noqa: BLE001 - never abort Run all because of the optional Hub demo
    print(f"HF model demo skipped: {type(exc).__name__}: {str(exc)[:200]}")
    print("(the Hub repo may be private, unreachable, or a dependency is missing). Continuing.")

try:
    if str(REPO_DIR / "src") not in sys.path:
        sys.path.insert(0, str(REPO_DIR / "src"))
    from intent_router.hub_predict import IntentRouter

    hub_router = IntentRouter.from_pretrained(HF_REPO)
    print(f"{'label':30s} {'confidence':>10s} {'ood_score':>10s} {'threshold':>10s} {'abstained':>9s}  probe")
    for text, res in zip(PROBES, hub_router.predict_batch(PROBES), strict=False):
        print(f"{res['label']:30s} {res['confidence']:10.3f} {res['ood_score']:10.2f} {hub_router.ood_threshold:10.2f}"
              f" {res['abstained']!s:>9}  {text[:50]}")
    print("(abstained = ood_score < threshold; the threshold is the 95%-retention cut in the Hub config)")
    print(f"(IntentRouter confidences are temperature-scaled, T = {hub_router.temperature:.4f}, so they differ slightly from the raw-softmax block printed above)")
except Exception as exc:  # noqa: BLE001 - same: print and continue
    print(f"HF abstention demo skipped: {type(exc).__name__}: {str(exc)[:200]}. Continuing.")
"""

INLINE_MD = r"""
## 7b. Inline results: confusion matrix and summary (fresh vs committed)

Shows (class names and numbers only, no dataset text) the **test** confusion matrix of the fresh Track A run and a
compact Track A / Track B summary next to the committed RTX 3070 values. Track B values are means over the 3
model seeds; `rejection@95` is the strict rejection recall of `maha_ft` at the threshold giving 95% known-class
retention on calibration rows, `retention` the achieved known-class retention on the evaluation rows. The figure
is also saved to `results_colab/figures/colab_test_confusion.png`.
"""

INLINE = r"""
import json as _json  # aliased: this cell must also run on its own

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_new_a = _json.loads(Path(f"{OUT_ROOT}/final/track_a.json").read_text())
_old_a = _json.loads(Path("results/final/track_a.json").read_text())
_new_b = _json.loads(Path(f"{OUT_ROOT}/trackb/headline.json").read_text())
_old_b = _json.loads(Path("results/trackb/headline.json").read_text())

_test = _new_a["test"]
_names = [c["label"] for c in _test["classes"]]
_cm = np.asarray(_test["confusion_counts"])
fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(_cm, cmap="Blues")
ax.set_xticks(range(len(_names)), _names, rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(_names)), _names, fontsize=7)
ax.set_xlabel("predicted")
ax.set_ylabel("gold")
ax.set_title(f"Track A test confusion (n={_test['n']}), fresh run")
for _i in range(_cm.shape[0]):
    for _j in range(_cm.shape[1]):
        if _cm[_i, _j]:
            ax.text(_j, _i, int(_cm[_i, _j]), ha="center", va="center", fontsize=7,
                    color="white" if _cm[_i, _j] > _cm.max() / 2 else "black")
fig.tight_layout()
Path(f"{OUT_ROOT}/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT_ROOT}/figures/colab_test_confusion.png", dpi=120)
plt.show()


def _ci(a: dict, m: str) -> str:
    v = a["test"][m]
    return f"{v['point']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]"


def _mean(b: dict, method: str, key: str) -> str:
    return f"{b['methods'][method][key]['mean']:.4f}"


_summary = pd.DataFrame(
    [
        ("Track A test macro-F1 [95% CI]", _ci(_new_a, "macro_f1"), _ci(_old_a, "macro_f1")),
        ("Track A test accuracy [95% CI]", _ci(_new_a, "accuracy"), _ci(_old_a, "accuracy")),
        ("Track B maha_ft AUROC (mean of seeds)", _mean(_new_b, "maha_ft", "auroc"), _mean(_old_b, "maha_ft", "auroc")),
        ("Track B maha_ft rejection@95 (strict)", _mean(_new_b, "maha_ft", "strict_rejection_recall"),
         _mean(_old_b, "maha_ft", "strict_rejection_recall")),
        ("Track B maha_ft known retention", _mean(_new_b, "maha_ft", "retention_known"),
         _mean(_old_b, "maha_ft", "retention_known")),
    ],
    columns=["metric", "fresh (this run)", "committed (RTX 3070)"],
)
print(_summary.to_string(index=False))
print(f"model fingerprint: fresh {_new_a['model_fingerprint'][:8]}, committed {_old_a['model_fingerprint'][:8]}")
"""

COMPARE_MD = r"""
## 8. Fresh vs committed numbers: PASS / FAIL

Explicit tolerances (T4 is not bitwise-equal to the RTX 3070): macro-F1 and accuracy +-0.03, rejection@95 +-0.10,
AUROC +-0.03. "rejection@95" is `strict_rejection_recall` of the shipped OOD score `maha_ft`
(threshold at 95% known-class retention on calibration rows). The cell prints deltas, a PASS/FAIL per metric and
ONE overall line: `REPRODUCTION: PASS` only if every metric is within tolerance, otherwise `REPRODUCTION: FAIL`
followed by the failing metrics. It reports, it does not raise (so Run all continues to the latency cell).
"""

COMPARE = r"""
import json

import pandas as pd

TOL = {"macro_f1": 0.03, "accuracy": 0.03, "rejection@95": 0.10, "auroc": 0.03}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def rows():
    a_new, a_old = load(f"{OUT_ROOT}/final/track_a.json"), load("results/final/track_a.json")
    b_new, b_old = load(f"{OUT_ROOT}/trackb/headline.json"), load("results/trackb/headline.json")
    for m in ("macro_f1", "accuracy"):
        yield f"Track A test {m}", m, a_new["test"][m]["point"], a_old["test"][m]["point"]
    for method in ("maha_ft", "msp"):
        yield (f"Track B {method} AUROC (mean of 3 seeds)", "auroc",
               b_new["methods"][method]["auroc"]["mean"], b_old["methods"][method]["auroc"]["mean"])
        yield (f"Track B {method} rejection@95 (mean of 3 seeds)", "rejection@95",
               b_new["methods"][method]["strict_rejection_recall"]["mean"],
               b_old["methods"][method]["strict_rejection_recall"]["mean"])


def judge(rows_iter) -> tuple[list[dict], str]:
    # Per-metric PASS/FAIL against TOL (|fresh - committed| <= tolerance) and the overall verdict line.
    table = []
    for name, kind, fresh, committed in rows_iter:
        delta = fresh - committed
        ok = abs(delta) <= TOL[kind] + 1e-12  # epsilon: a delta exactly at the tolerance passes
        table.append({"metric": name, "fresh": round(fresh, 4), "committed": round(committed, 4),
                      "delta": round(delta, 4), "tolerance": f"+-{TOL[kind]}", "result": "PASS" if ok else "FAIL"})
    failed = [r["metric"] for r in table if r["result"] == "FAIL"]
    if table and not failed:
        return table, "REPRODUCTION: PASS"
    return table, "REPRODUCTION: FAIL (" + ("; ".join(failed) if failed else "no metrics compared") + ")"


table, verdict = judge(rows())
print(pd.DataFrame(table).to_string(index=False))
print(f"\n{sum(r['result'] == 'PASS' for r in table)} of {len(table)} metrics within tolerance")
print(verdict)
"""

LATENCY_MD = r"""
## 8b. CPU batch-1 latency (the canonical latency run)

The report's CPU latency comes from this cell, not from the local laptop (which was never idle). After final
training and Track A it (1) packages the fresh model, (2) exports **ONNX fp32** with `intent_router.onnx_export`,
(3) times **PyTorch fp32** and **ONNX fp32** on the CPU through `intent_router.serve_bench.stage_latency`
(batch 1, 10 warm-up calls, 3 repeats over the 74 **val** texts, never test rows; threads = 1 and
`os.cpu_count()`), and (4) writes `LATENCY_OUT` with the CPU model, vCPU count, runtime type (Colab release,
GPU name from `nvidia-smi` or `CPU-only`) and library versions. No CUDA call is made: the accelerator is only
recorded. The ONNX libraries are installed first from the versions pinned in `requirements-serve.txt`
(`onnx`, `onnx-ir`, `onnxruntime`, `flatbuffers`, `ml-dtypes`; `protobuf` already matches `requirements.txt`).

ESTIMATE of runtime: the export plus 2 backends x 2 thread counts x (10 + 3 x 74) calls; NOT MEASURED on Colab.

**After the run:** copy `results_colab/serving/latency_colab.json` to `results/serving/latency_colab.json` in
the repo before building the report (`python -m report.build`); the report reads it from there and shows its
p50/p95, CPU model, vCPU count and runtime type. Colab vCPUs are shared cloud CPUs, so the numbers are not an
idle-machine measurement; the environment is stated next to them.
"""

LATENCY_PIP = r"""
SERVE_PINS = {"onnx", "onnx-ir", "onnxruntime", "flatbuffers", "ml-dtypes", "protobuf"}
pins = [
    ln.strip() for ln in Path("requirements-serve.txt").read_text().splitlines()
    if ln.strip() and not ln.startswith(("#", "--")) and ln.split("==")[0].strip().lower() in SERVE_PINS
]
print("installing:", pins)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pins], check=True)
for pkg in ("onnx", "onnxruntime", "onnx-ir", "flatbuffers", "ml-dtypes", "protobuf"):
    print(f"{pkg}=={md.version(pkg)}")
"""

LATENCY = r"""
import json as _json  # aliased: this cell must also run on its own

LATENCY_OUT = f"{OUT_ROOT}/serving/latency_colab.json"  # the report reads results/serving/latency_colab.json

run_py("latency_colab", "--config", FINAL_CFG, "--results-dir", f"{OUT_ROOT}/final", "--out", LATENCY_OUT)
lat = _json.loads(Path(LATENCY_OUT).read_text())
env = lat["environment"]
print(f"\nCPU: {env['cpu_model']}; vCPUs: {env['os_cpu_count']}; runtime: {env['runtime_type']}")
print(f"torch {env['torch']}, onnxruntime {env['onnxruntime']}; n_rows={lat['n_rows']} (val)")
for backend, by_threads in lat["latency"].items():
    for key, st in by_threads.items():
        print(f"{backend:10s} {key:10s} p50={st['p50_ms']:.1f} ms  p95={st['p95_ms']:.1f} ms")
print(f"\nwrote {LATENCY_OUT}")
print("----- BEGIN latency_colab.json (copy everything between the markers) -----")
print(Path(LATENCY_OUT).read_text())
print("----- END latency_colab.json -----")
print(f"NEXT: copy {LATENCY_OUT} to results/serving/latency_colab.json in the repo, then rebuild the report")
"""

PUSH_MD = r"""
## 9. Optional push (off by default)

`PUSH = False`. When enabled: (a) uploads `results_colab/` to W&B as an artifact (ids and numbers only,
no dataset text); (b) if `HF_REPO_ID` is set, uploads `outputs/final_model` to a **private** HF model repo.
The model card and the full package are handled by the hub tooling, not here.
"""

PUSH = r"""
PUSH = False
HF_REPO_ID = None  # e.g. "your-user/intent-router-v1"; the repo is created private

if PUSH:
    if USE_WANDB:
        import wandb

        run = wandb.init(project="multilingual-intent-router", group="colab", name="colab-results", job_type="push")
        art = wandb.Artifact("colab-results", type="results")
        art.add_dir(OUT_ROOT)
        run.log_artifact(art)
        run.finish()
    if HF_REPO_ID:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(HF_REPO_ID, private=True, exist_ok=True)
        api.upload_folder(repo_id=HF_REPO_ID, folder_path="outputs/final_model")
        print("uploaded (private):", HF_REPO_ID)
else:
    print("PUSH is False: nothing uploaded")
"""

OPT_MD = r"""
## 10. Opt-in stages (all off by default)

Each cell is guarded by a `RUN_*` flag. Outputs go to `results_colab/<stage>/` via the same config redirect.
Runtime figures are **ESTIMATE**s derived from RTX 3070 results in the repo; a T4 slowdown is NOT MEASURED
(treat as roughly 1.5-2x, an assumption). **None of these cells has been executed on Colab**; stages that read
upstream committed artifacts read them from `results/` (read-only).
"""

BAKEOFF = r"""
# ESTIMATE ~7.3 h on the RTX 3070: 240 fold-runs, summed `wall_clock_s` of results/bakeoff/runs/*.json =
# 26,312 s (derived by summing; the sweep/confirm/ablate stages are resumable). On a T4 expect more, and
# beyond a free Colab session limit (~12 h at best): split into stages across sessions.
RUN_BAKEOFF = False
if RUN_BAKEOFF:
    cfg = make_cfg("bakeoff")
    for stage in ("sweep", "confirm", "ablate", "timing"):
        run_py("cv", "--config", cfg, "--stage", stage)
        run_py("aggregate", "--config", cfg)
"""

LOCO = r"""
# ESTIMATE ~6 min on the RTX 3070: 10 LOCO runs, summed `wall_clock_s` of results/trackb/runs/loco_*.json
# = 331.6 s. T4: assume up to ~2x.
RUN_LOCO = False
if RUN_LOCO:
    run_py("trackb", "--config", TRACKB_CFG, "--stage", "loco", *WANDB_FLAG)
    run_py("trackb", "--config", TRACKB_CFG, "--stage", "report", *WANDB_FLAG)  # aggregates loco.json
"""

LLM = r"""
# NEEDS OLLAMA (local, loopback only; dataset text must not leave the machine): install Ollama in the Colab VM,
# `ollama serve`, and pull qwen3:8b and llama3.1:8b (configs/llm_baseline.yaml). Requires sections 6 and 7 to
# have run first (it reads the fresh test predictions / Track B scores from results_colab/).
# ESTIMATE of inference time: ~0.18 s/message on the RTX 3070 (mean_s in results/llm_baseline/llm_latency.json,
# 74 messages x 4 model/mode configs x 2 tracks is a few minutes); model download and Ollama setup time NOT MEASURED.
RUN_LLM_BASELINE = False
if RUN_LLM_BASELINE:
    cfg = make_cfg(
        "llm_baseline",
        test_predictions_path=f"{OUT_ROOT}/final/test_predictions.csv",
        headline_scores_path=f"{OUT_ROOT}/trackb/scores/headline_s42.csv",
        headline_json_path=f"{OUT_ROOT}/trackb/headline.json",
    )
    run_py("llm_baseline", "--config", cfg, "--stage", "all")
"""

P4C = r"""
# Robustness and open-set training fixes (factors: random ID prefixes, outlier exposure, translated and noisy copies; rules fixed before the runs). Stage names come from phase4c.STAGES; edit the list to
# run a subset. Outlier exposure needs synthetic data generated by local Ollama (phase4c_data.py), so only the
# non-outlier-exposure stages can run without Ollama. ESTIMATE >= 1.6 h on the RTX 3070: summed `wall_clock_s` over the 43
# result files under results/phase4c that record one is 5,875 s (a lower bound: not every run file records it).
RUN_PHASE4C = False
P4C_STAGES = ["guard_current", "folds_current", "trackb_current"]
if RUN_PHASE4C:
    cfg = make_cfg("phase4c")
    for stage in P4C_STAGES:
        run_py("phase4c", "--config", cfg, "--stage", stage, *WANDB_FLAG)
"""

P4E = r"""
# Multi-axis selection (candidates v1/a3/a1/a1a3, rules fixed before the runs). Default stage order as in scripts/run_phase4e.sh.
# ESTIMATE ~6-7 GPU-hours on the RTX 3070 (Multi-axis selection budget); the post-selection stages
# confirm_final / final_retrain make logged test-split calls and are NOT in the default list.
RUN_PHASE4E = False
P4E_STAGES = ["reuse_check", "curves", "epoch", "folds", "trackb", "analyze", "select"]
if RUN_PHASE4E:
    cfg = make_cfg("phase4e")
    for stage in P4E_STAGES:
        run_py("phase4e", "--config", cfg, "--stage", stage, *WANDB_FLAG)
"""


def build() -> dict:
    """Assemble the notebook dict."""
    cells = [
        md(TITLE),
        md(SETUP_MD),
        code(SETUP),
        md(INSTALL_MD),
        code(INSTALL),
        md(CLEAN_MD),
        code(UNINSTALL),
        code(ASSERT_IMPORTS),
        md(DATA_MD),
        code(DATA),
        md(SECRETS_MD),
        code(SECRETS),
        md(PIPE_MD),
        code(PIPE),
        md(FINAL_MD),
        code(FINAL),
        md(HF_MD),
        code(HF),
        md(TRACKB_MD),
        code(TRACKB),
        md(INLINE_MD),
        code(INLINE),
        md(COMPARE_MD),
        code(COMPARE),
        md(LATENCY_MD),
        code(LATENCY_PIP),
        code(LATENCY),
        md(PUSH_MD),
        code(PUSH),
        md(OPT_MD),
        md("### 10a. Bake-off"),
        code(BAKEOFF),
        md("### 10b. LOCO (Track B)"),
        code(LOCO),
        md("### 10c. LLM baseline (needs Ollama)"),
        code(LLM),
        md("### 10d. Robustness and open-set training fixes"),
        code(P4C),
        md("### 10e. Multi-axis selection"),
        code(P4E),
    ]
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:02d}"  # deterministic ids (nbformat 4.5 requires them)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=NOTEBOOK_PATH)
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
