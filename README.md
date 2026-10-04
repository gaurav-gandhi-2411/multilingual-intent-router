# multilingual-intent-router

<!-- LINKS:BEGIN -->
[![report.pdf](https://img.shields.io/badge/report.pdf-blue)](report.pdf) | [![Hugging Face model](https://img.shields.io/badge/Hugging_Face_model-yellow)](https://huggingface.co/gauravgandhi2411/multilingual-intent-router) | [![W&B report](https://img.shields.io/badge/W%26B_report-orange)](https://wandb.ai/gauravgandhi429-gaurav-gandhi/multilingual-intent-router/reports/Submission-summary--VmlldzoxODA1MTA3Mw==) | [![Open in Colab](https://img.shields.io/badge/Open_in_Colab-green)](https://colab.research.google.com/github/gaurav-gandhi-2411/multilingual-intent-router/blob/v1.0-submission/notebooks/intent_router_colab.ipynb) | [Code](https://github.com/gaurav-gandhi-2411/multilingual-intent-router)
<!-- LINKS:END -->

Multilingual intent classifier with open-set abstention. It fine-tunes `intfloat/multilingual-e5-base`
on a 500-row synthetic logistics intent dataset (12 known intents, several languages, short messages)
provided for an assessment; the dataset is not redistributed, so `data/` is not part of this repository
and the pipeline expects the file at `data/dataset.csv` (columns `id`, `text`, `label`). Track A scores
known-class classification (macro-F1 first, accuracy second); Track B scores whether the model abstains
on intents it never saw, using a Mahalanobis distance on the fine-tuned features with a threshold fixed
for 95% retention of known inputs. Everything is seeded (42), every number below is rendered from a
committed JSON file, and the test split was evaluated once for the shipped model (once in the committed results; notebook reproduction runs re-evaluate the bit-identical weights into results_colab/, never results/).

## Results

<!-- RESULTS:BEGIN -->
**Track A: known classes** (n = 74 test rows, unbiased (single test evaluation); 95% CIs: percentile, plain row resampling, 10000 resamples, seed 42).

| Model | Macro-F1 [95% CI] | Accuracy [95% CI] | Paired delta macro-F1, shipped minus row [95% CI] |
|---|---|---|---|
| **Fine-tuned multilingual-e5-base (v1, shipped)** | 0.947 [0.865, 0.989] | 0.946 [0.892, 0.986] | n/a |
| B0: TF-IDF + logistic regression | 0.733 [0.602, 0.822] | 0.743 [0.635, 0.838] | +0.213 [+0.113, +0.340] |
| B1: frozen e5-base embeddings + logistic regression | 0.899 [0.802, 0.956] | 0.892 [0.811, 0.959] | +0.048 [-0.009, +0.119] |
| Best local LLM: llama3.1:8b, zero-shot (of 2 models x 2 modes) | 0.680 [0.530, 0.774] | 0.703 [0.595, 0.797] | +0.266 [+0.159, +0.411] |

The paired 95% interval of the macro-F1 difference B0: excludes 0; B1: includes 0; llama3.1:8b (zero-shot): excludes 0.

Cross-validation (**post-selection (optimistic) CV estimate**: the configuration and epoch were chosen on these folds, so treat as optimistic and not comparable to the test row): out-of-fold macro-F1 0.953 [0.932, 0.971], accuracy 0.953 [0.932, 0.972] (probabilities averaged over the 9 OOF predictions per id, then scored once); mean over 45 fold-runs: macro-F1 0.939 ± 0.022.

**Track B: unknown classes** (shipped scorer `maha_ft`: Mahalanobis distance on the fine-tuned features; threshold set for 95% known-class retention on calibration rows; baseline `msp`: plain (uncalibrated) max-softmax probability).

Headline holdout, classes held out: `document_processing`, `yard_management` (62 known and 80 unknown evaluation rows; mean ± sample std over 3 model seeds [42, 43, 44]).

| Scorer | AUROC | Strict rejection recall @95% retention | Known retention |
|---|---|---|---|
| MSP baseline | 0.841 ± 0.019 | 0.196 ± 0.104 | 0.984 ± 0.016 |
| **`maha_ft` (shipped)** | 0.870 ± 0.008 | 0.342 ± 0.104 | 0.968 ± 0.000 |

CONFIRM (leave-one-class-out over 5 held-out classes, one model seed per class, row bootstrap CIs; thresholds from calibration rows; `results/trackb_improve/confirm.json`):

| Scorer | AUROC | Strict rejection recall @95% retention | Known retention |
|---|---|---|---|
| MSP baseline | 0.867 (mean only) | — | — |
| **`maha_ft` (shipped)** | 0.879 [0.850, 0.908] | 0.227 [0.174, 0.283] | 0.970 [0.950, 0.988] |

— only MSP AUROC is recorded on CONFIRM.

<sub>Rendered by `scripts/render_readme_results.py` from `results/final/track_a.json`, `results/llm_baseline/track_a.json`, `results/trackb/headline.json`, `results/trackb_improve/confirm.json`; do not edit by hand.</sub>
<!-- RESULTS:END -->

Small-sample caveat: the test split has 74 rows, so the Track A intervals are wide, and the Track B
headline holdout is two classes and three seeds (hence the CONFIRM table). Cross-validation numbers were
used for model selection and are optimistic.

## Quick start

Load the classifier with plain `transformers` (the label mapping is in `config.json`; inputs carry the
`query: ` prefix the e5 family expects):

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "gauravgandhi2411/multilingual-intent-router"
tok, model = (
    AutoTokenizer.from_pretrained(REPO),
    AutoModelForSequenceClassification.from_pretrained(REPO).eval(),
)
enc = tok(
    ["query: where is my shipment right now?"], return_tensors="pt", truncation=True, max_length=64
)
print(model.config.id2label[model(**enc).logits.argmax(-1).item()])
```

Abstention (temperature-scaled confidence, Mahalanobis `ood_score`, `abstained` flag) lives in the
`predict.py` shipped next to the weights:

```python
import sys
from huggingface_hub import snapshot_download

sys.path.insert(0, d := snapshot_download(REPO))
from predict import IntentRouter

print(IntentRouter.from_pretrained(d).predict("book me a table for two"))
```

The serving path (FastAPI + ONNX) is in `src/intent_router/serve.py` and the `Dockerfile`.

## Reproduce

Hardware used for the committed results: one RTX 3070 (8 GB), Windows 11, Python 3.12, CUDA 12.8 wheels
(`requirements.txt`, pinned). Runtimes below are **ESTIMATE**s for a re-run on that machine, derived from
the `wall_clock_s` fields of the committed result files; stages marked NOT MEASURED were never timed.

**One click:** Open in Colab (badge above; the notebook `notebooks/intent_router_colab.ipynb` clones
the repository at tag `v1.0-submission`, installs pinned dependencies, trains v1, runs Track A and the
Track B headline holdout, times CPU latency and compares fresh numbers with the committed ones). It
needs your own copy of the dataset at the path it asks for. Outputs go to `results_colab/`, never over
`results/`. A T4 slowdown over the RTX 3070 is NOT MEASURED.

**Locally**, in order (Windows paths shown; `PYTHONPATH=src`; seeds are in the configs, 42 everywhere
stochastic):

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
export PYTHONPATH=src
# 0. data: put the dataset at data/dataset.csv (not distributed)
# 1. EDA + leakage-safe split (ids only are written to splits/splits.csv)
python -m intent_router.eda   --config configs/base.yaml          # NOT MEASURED
python -m intent_router.split --config configs/base.yaml          # NOT MEASURED
# 2. (optional) model bake-off: 240 fold-runs, resumable, ~7.3 h ESTIMATE (sum of run wall clocks 26,312 s)
python -m intent_router.baselines --config configs/bakeoff.yaml
bash scripts/run_bakeoff.sh sweep confirm ablate timing
python -m intent_router.selection --config configs/bakeoff.yaml
# 3. final model (train stage ~3 min ESTIMATE, 185.8 s measured) and Track A (one logged test evaluation;
#    the evaluate stage was not timed: NOT MEASURED)
python -m intent_router.final --config configs/final.yaml --stage train
python -m intent_router.final --config configs/final.yaml --stage evaluate
# 4. Track B: headline holdout (3 seeds x ~30 s ESTIMATE) and the other stages (leave-one-class-out ~6 min ESTIMATE)
python -m intent_router.trackb --config configs/trackb.yaml --stage headline
python -m intent_router.trackb --config configs/trackb.yaml --stage loco
# 5. package, ONNX export, stage and push the model to the Hub (needs an authenticated Hugging Face login; NOT MEASURED)
python -m intent_router.package --results-dir results/final --model-dir outputs/final_model \
    --features outputs/final/features_logits.npz --out serve_model
python -m intent_router.onnx_export --serve-dir serve_model
python -m intent_router.hub stage --version v1
python -m intent_router.hub push  --version v1
# 6. Weights & Biases: the curated live run (run `final-v1-train`, project `multilingual-intent-router`) is a
#    fingerprint-asserted rerun of the final model (needs `wandb login`); the summary run is logged from saved results
python scripts/wandb_final_rerun.py
python scripts/wandb_final_summary.py
# 7. regenerate the README tables from results JSON, and the report
python scripts/render_readme_results.py
python -m report.build
```

Notes. The `evaluate` stage refuses a second test
evaluation of the same model (`results/final/test_eval_log.jsonl`), so on a checkout that already holds the
committed results, redirect the output paths in a copy of the config (the Colab notebook does exactly that).
Tests: `pytest -q`.

## Repository map

Core pipeline (what the Reproduce section runs and what serves the model):

| Module | Role |
|---|---|
| `data`, `dedup`, `split`, `eda` | load the dataset, near-duplicate groups, leakage-safe stratified split, exploratory report |
| `models`, `train`, `seeding` | tokenisation and model building, the fine-tuning loop, seeding |
| `cv`, `aggregate`, `selection`, `baselines` | grouped cross-validation bake-off, aggregation, selection rule fixed before the runs, B0/B1 floors |
| `evaluate`, `stats`, `analysis`, `ood` | metrics, calibration, bootstrap and paired tests, slice/error analysis, Mahalanobis open-set scoring |
| `final`, `trackb` | final model and Track A; Track B open-set experiments |
| `package`, `predict`, `serve`, `onnx_export`, `serve_bench` | serve directory, lean inference, FastAPI service, ONNX export, latency benchmark |
| `hub`, `hub_predict`, `hub_roundtrip`, `model_card` | stage and push to the Hub, standalone `predict.py` shipped with the weights, round-trip check, model card rendered from results JSON |
| `latency_colab` | CPU latency of the freshly trained model on the Colab runtime |

Experiment modules (side studies that informed the design choices or the report; none is needed
to train or serve the shipped model):

| Modules | Study |
|---|---|
| `phase4c`, `phase4c_data`, `phase4e`, `phase4e_final` | robustness and open-set training fixes, multi-axis model selection |
| `phase6a`, `phase6a_diag`, `phase6a_post`, `phase6a_select`, `phase6a_views`, `ood_variants`, `soup`, `twin_kl` | open-set improvement round, diagnostics, scoring variants, seed soups, twin-consistency loss |
| `trackb_improve` | Track B improvement candidates and the CONFIRM leave-one-class-out run |
| `llm_baseline`, `llm_audit` | local zero-/few-shot LLM baseline and label-ambiguity audit (loopback Ollama only) |
| `robustness`, `selection_check`, `quantization_eval` | noise/translation/ID-swap robustness, post-hoc selection check, int8 quantisation study |

Other directories: `configs/` (runtime config per stage), `scripts/` (stage runners, report helpers,
README renderer), `report/` (report builder),
`results/` (committed result JSON; index in [results/README.md](results/README.md)), `splits/`
(ids only), `notebooks/` (Colab notebook, generated by `scripts/build_colab_notebook.py`), `tests/`.

## Deliverables map

<!-- DELIVERABLES:BEGIN -->
| Assessment item | Where |
|---|---|
| 4.1 report.pdf | [report.pdf](report.pdf), built by `python -m report.build` (numbers read from `results/`) |
| 4.2 Hugging Face model with label mapping and model card | repo [gauravgandhi2411/multilingual-intent-router](https://huggingface.co/gauravgandhi2411/multilingual-intent-router) (branch `main` = shipped v1, `robust-v3` = robustness variant); label mapping in `config.json` (`id2label`), card rendered by `intent_router.model_card` from `src/intent_router/templates/model_card.md.j2`; staged by `intent_router.hub` |
| 4.3 Weights & Biases | [W&B report](https://wandb.ai/gauravgandhi429-gaurav-gandhi/multilingual-intent-router/reports/Submission-summary--VmlldzoxODA1MTA3Mw==): project `multilingual-intent-router`; curated run `final-v1-train` (display name `final-v1`) by `scripts/wandb_final_rerun.py` |
| 4.4 code | [this repository](https://github.com/gaurav-gandhi-2411/multilingual-intent-router): load and split (`data`, `split`), fine-tune (`train`, `final`), evaluate (`evaluate`, `final`, `trackb`), save and push (`package`, `hub`), W&B logging, seeded (42), documented (this README); [Colab notebook](https://colab.research.google.com/github/gaurav-gandhi-2411/multilingual-intent-router/blob/v1.0-submission/notebooks/intent_router_colab.ipynb) reproduces the results |
<!-- DELIVERABLES:END -->

## License

MIT, code only (see `LICENSE`). The dataset the pipeline was built on is not covered and not included.
