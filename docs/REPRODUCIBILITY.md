# Reproducibility: what a reviewer can check, and how

This page maps each point raised in the *Remarks on Code Availability* to a file
in this repository and to a command that verifies it. Everything here runs
without the clinical dataset. Reproducing the published **numbers** additionally
requires the multi-centre cohort and the compute described at the end.

## 1. Software environment and dependencies

| Item | Where |
|---|---|
| Exact dependency set, split by CUDA/CPU | `requirements.txt`, `requirements-torch-cu118.txt`, `requirements-torch-cpu.txt` |
| Interpreter and CUDA versions | `environment.yml`, [INSTALL.md](INSTALL.md) |
| Packaged metadata, Python requirement | `pyproject.toml` |

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
python -m pytest -q          # 20+ tests, no GPU required
```

The previous `requirements.txt` was a dump of a local conda environment
(`... @ file:///home/conda/...`), which cannot be installed anywhere else. It has
been replaced by a short, pinned list; PyTorch is installed from its own index
because the CUDA and CPU wheels differ.

## 2. Step-by-step preprocessing, training and inference

| Step | Command | Documentation |
|---|---|---|
| Build a COCO split from images + masks | `python scripts/build_coco.py --task subtype --out Data` | [DATA_FORMAT.md](DATA_FORMAT.md#4-building-a-training-split) |
| Smoke-test the training stack | `python train.py --config configs/train_multitask.yaml --dry-run` | [TRAINING_INFERENCE.md](TRAINING_INFERENCE.md#3-smoke-test-before-committing-gpu-hours) |
| Train | `python train.py --config configs/train_multitask.yaml` | same page |
| Single-image report | `python report.py --config configs/infer.yaml` | same page |
| Batch inference | `python predict.py --config configs/demo.yaml` | same page |

## 3. Example configuration files

| File | Purpose |
|---|---|
| `configs/train_multitask.yaml` | training, with α₁–α₅ = 0.8/1/1/1/1, β = 0.5, focal α = 0.75, γ = 2 |
| `configs/infer.yaml` | single-image report, paths to the three checkpoints |
| `configs/demo.yaml` | offline minimal example |

Every key documented in `configs/*.yaml` is validated against the argparse flags
by `tests/test_config.py`, so a config cannot silently drift from the code.

## 4. Input and output formats

| Direction | Specification |
|---|---|
| Input images | PNG, 224 × 224, converted to RGB |
| Input masks | PNG, binary, same stem as the image |
| Clinical table | `dataset/Baseline.csv`, columns in [DATA_FORMAT.md](DATA_FORMAT.md) |
| Class order | fixed tables in [DATA_FORMAT.md](DATA_FORMAT.md#2-class-tables), asserted by `tests/test_labels.py` |
| Training data | COCO: `images[]`, `annotations[]`, `categories[]` |
| Output (single image) | `Reports/<image>_report/{report.html, img.png, mask.png, score1-4.png}` |
| Output (batch) | `predictions.csv` with one row per image |

## 5. Pretrained weights or representative checkpoints

| Item | Where |
|---|---|
| Download links and exact target paths | [WEIGHTS.md](WEIGHTS.md) |
| Machine-readable manifest | `weights/manifest.json` |
| Checksum verification | `python scripts/verify_weights.py` |

The weights are distributed through institutional share links rather than a git
repository, because the data-use agreement covering the cohort restricts
redistribution of derived model files. The manifest records size and SHA-256 so
that a download can be verified; see WEIGHTS.md for how the checksums are frozen.

## 6. Minimal executable example without private data

```bash
python demo.py --config configs/demo.yaml --random-weights
```

* Uses only the 430 images committed to this repository (20 by default, `--limit 0`
  for all of them).
* No GPU, no checkpoint, no clinical data, no API key.
* Runs data loading → three task heads → risk score → HTML report, and writes
  `reports/demo_predictions.csv`, `reports/demo_summary.txt` and
  `reports/<image>_report/report.html`.
* With random weights the printed agreement is meaningless — the command proves
  that the pipeline runs, not that the model is accurate. Drop
  `--random-weights` after placing the checkpoints to get real predictions.

Expected console output ends with:

```
Demo finished: reports/demo_predictions.csv + reports/demo_summary.txt
```

## What cannot be reproduced from this repository

| Limitation | Reason |
|---|---|
| The published AUROC / HR / NRI figures | They require the full multi-centre cohort (159,649 patients), which is not redistributable; only a 430-image demonstration subset is public |
| The full 150-epoch training schedule | Eight A800 80 GB GPUs; the released config documents every hyper-parameter but the wall-clock cost is out of reach for a review |
| The survival analysis (Cox model, thresholds 0.53 / 3.66) | Separate analysis code on the full cohort; the score displayed by `report.py` is a different, integer score (see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)) |
| Prospective validation | Clinical study, not a code artefact |

These limits are stated in the README and in the response letter, so that the
scope of the released code is unambiguous.
