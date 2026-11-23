# B-CAREs

**Breast Cancer Assessment, Reporting, and Risk Stratification Executive System (B-CAREs)** — code,
configuration and a demonstration dataset for the multi-task ultrasound framework
described in the manuscript.

[![CI](https://github.com/DouBeeTwT/B-CAREs/actions/workflows/ci.yml/badge.svg)](https://github.com/DouBeeTwT/B-CAREs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)

> **Version 1.1.0 — reproducibility release.** The manuscript is under review.
> This release adds the model package, configuration files, installation
> instructions, input/output specifications, a weight manifest and an offline
> minimal example. See [CHANGELOG.md](CHANGELOG.md).

## Contents

| Section | What you get |
|---|---|
| [Minimal example](#minimal-example-no-data-no-gpu-no-weights) | one command, proves the pipeline runs anywhere |
| [Installation](#installation) | pinned environment, CUDA and CPU variants |
| [Data](#data) | 430 ultrasound images + masks + clinical table, with formats |
| [Model weights](#model-weights) | download links, exact target paths, checksum tool |
| [Training](#training) | YAML config, one command, dry-run smoke test |
| [Inference](#inference) | single-image report and batch CSV |
| [Reproducibility](#reproducibility) | requirement-by-requirement map for reviewers |
| [Citation](#citation) | BibTeX |

## Abstract

Breast cancer is the most common malignancy in women, and ultrasound (US) is the primary imaging tool for its initial evaluation. Although the Breast Imaging Reporting and Data System (BI-RADS) excels in morphological characterization and biopsy guidance, it does not inherently encode prognostic dimensions such as tumor invasiveness, sentinel lymph node (SLN) status, or histological subtypes. Here, we developed the Breast Cancer Assessment, Reporting, and Risk Stratification Executive System (B-CAREs), a multi-task deep learning framework with dynamic domain adaptation that processes standard US images to simultaneously perform nodule detection, BI-RADS classification, histological subtype prediction, and SLN metastasis prediction, while generating structured reports via a large language model interface. B-CAREs was developed and internally tested on 207,735 participants from five centers and externally validated on US videos from 11,514 participants across two independent centers. In the internal test set, B-CAREs achieved a nodule detection accuracy of 0.982 and AUROCs of 0.987 for malignancy, 0.903 for BI-RADS classification, 0.905 for histological subtype, and 0.893 for SLN metastasis prediction. The derived risk stratification model effectively categorized patients into low-, medium-, and high-risk groups (high-risk vs. medium: HR = 9.90; low-risk vs. medium: HR = 0.09), with a net reclassification improvement of 33.62% over BI-RADS alone, and these findings were confirmed in external validation (high-risk: HR = 8.90; low-risk: HR = 0.17). B-CAREs-generated structured reports received high expert-blind scores (9.31/10) and significantly improved physician diagnostic accuracy. These results indicate that routinely acquired US images carry image-derived information associated with histological subtype, sentinel lymph node status and outcome that is not captured by the BI-RADS category alone, and that deep learning can exploit this information to support preoperative risk stratification at the first visit.

## What is in this repository

| Path | Content |
|---|---|
| `BMIL/` | model, losses, decoder, dataset and training utilities |
| `train.py` | multi-task training entry point (YAML-driven) |
| `report.py` | single-image inference → HTML report |
| `predict.py` | batch inference → CSV |
| `demo.py` | offline minimal example, runs without weights |
| `bcares/` | configuration loading and label tables |
| `configs/` | example configuration files |
| `scripts/build_split.py` | images + masks → train/val `img` + `lab` layout |
| `scripts/verify_weights.py` | SHA-256 verification of the checkpoints |
| `dataset/` | 430 ultrasound images, their masks and `Baseline.csv` |
| `docs/` | installation, data format, training/inference, weights, reproducibility, troubleshooting |
| `tests/` | 20+ tests, no GPU required |
| `templates/` | Jinja2 template of the clinical report |
| `weights/manifest.json` | download links, target paths, size and checksum |

## System requirements

| Component | Value used for the manuscript | Minimum |
|---|---|---|
| Python | 3.11 | 3.10 |
| PyTorch / torchvision | 2.2.2 + CUDA 11.8 / 0.17.2 | 2.2 |
| NVIDIA driver | 520 | 520 |
| GPU | 8 × A800 80 GB (training) | any GPU for inference; CPU works |
| RAM / disk | 64 GB / ~10 GB | 8 GB / 2 GB |

No system CUDA toolkit is needed: the wheels bundle the CUDA runtime.

## Installation

```bash
git clone https://github.com/DouBeeTwT/B-CAREs.git
cd B-CAREs

conda env create -f environment.yml && conda activate bcares
# or: python3.11 -m venv .venv && source .venv/bin/activate

pip install -r requirements-torch-cu118.txt   # or -torch-cpu.txt
pip install -r requirements.txt
pip install -e .                              # optional
```

Full matrix and verification steps: [docs/INSTALL.md](docs/INSTALL.md).

## Minimal example (no data, no GPU, no weights)

```bash
python demo.py --config configs/demo.yaml --random-weights
```

Runs 20 of the shipped images through the three task heads, writes
`reports/demo_predictions.csv`, `reports/demo_summary.txt` and one full HTML
report in `reports/BIC_0009_report/`. No clinical data, no checkpoints, no API
key, no GPU. With random weights the printed agreement is meaningless by
construction — the command proves that data loading, model construction,
inference and report rendering work end to end.

## Data

`dataset/` contains a **demonstration subset** of 430 breast ultrasound images
(224 × 224 PNG), their binary segmentation masks, and a clinical table:

```
dataset/Baseline.csv     430 rows: PictureName, BIRADS, Lymph, Age, Gender, Size,
                         FamilyHistory, Menophania, Period, Duration, Pregnant,
                         Menopause, Smoke, Drink
dataset/test/img/        430 images, filename prefix = histological subtype
dataset/test/lab/        430 masks, same stems
```

| Index | Prefix | Code | Histological subtype | Images |
|---:|---|---|---|---:|
| 0 | FBB | BF | Benign fibroadenoma | 190 |
| 1 | BIC | IBCN | Invasive breast cancer non-special | 70 |
| 2 | DICS | DCIS | Ductal carcinoma in situ | 20 |
| 3 | IDC | IDC | Invasive ductal carcinoma | 60 |
| 4 | LBC | LCIS | Lobular carcinoma in situ | 10 |
| 5 | PBC | ILC | Invasive lobular carcinoma | 40 |
| 6 | UDF | IBCU | Invasive breast cancer unclassified | 40 |
| | | | **Total** | **430** |

BI-RADS is encoded 0–4 (`2–3, 4a, 4b, 4c, 5`) and lymph node status 0–2
(`0`, `1–2`, `≥3` nodes). The full specification — column by column,
class tables, preprocessing constants, COCO conversion and output formats — is in
[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

**Access to the full cohort.** All data that fuelled the findings are reported in
the article and its Supplementary Information. The full breast cancer dataset
(raw imaging, detailed clinical records) cannot be made public: even after
de-identification the combination of imaging and clinical variables may allow
re-identification. Researchers may request access for non-commercial academic
work from the corresponding author; requests are reviewed by the institutional
ethics committee and data custodians. Conditions: research use aligned with the
study objectives, a data-use agreement prohibiting re-identification and
redistribution, and a two-year retention limit from the date of publication.

The 430 images in this repository are released for non-commercial research and
teaching only.

## Model weights

The three checkpoints are not stored in git; they are distributed through the
institutional share links below. Place each file at the exact path in the third
column (see [docs/WEIGHTS.md](docs/WEIGHTS.md)).

| Task | File | Place at | Link |
|---|---|---|---|
| BI-RADS (5 classes) | `Best.pth` | `pth/QPMC_BIRADS_Exp001/Best.pth` | [share](https://pan.sjtu.edu.cn/web/share/ef3c18cf0557a22a5d65b35fca88e442) |
| Subtype (7 classes) | `Last.pth` | `pth/BMIL_v4_Exp010/Last.pth` | [share](https://pan.sjtu.edu.cn/web/share/34371324234abf37f70ff9b4e43a1f99) |
| Sentinel node (3 classes) | `Best.pth` | `pth/QPMC_Lymph_Exp002/Best.pth` | [share](https://pan.sjtu.edu.cn/web/share/12b42e150ee920db0388446254c7342b) |

```bash
python scripts/verify_weights.py     # checks presence, size and SHA-256
```

## Training

```bash
python scripts/build_split.py --val-fraction 0.2 --seed 1 --out Data
python train.py --config configs/train_multitask.yaml --dry-run   # smoke test, writes nothing
python train.py --config configs/train_multitask.yaml             # 150 epochs
```

`configs/train_multitask.yaml` contains the loss weights reported in the
manuscript (α₁ = 0.8, α₂–α₅ = 1, β = 0.5, focal α = 0.75, γ = 2) together with the
optimiser, schedule and model sizes. Command-line flags override the YAML, so the
old invocation style still works. Checkpoints are written to
`pth/<project>_<exp>/{Last,Best}.pth`.

Details: [docs/TRAINING_INFERENCE.md](docs/TRAINING_INFERENCE.md).

## Inference

```bash
python report.py  --config configs/infer.yaml                  # one image -> Reports/<img>_report/report.html
python predict.py --config configs/demo.yaml --limit 50        # directory -> reports/predictions.csv
```

`report.py` writes the report, the segmentation overlay and four score panels;
`--explain` adds a SHAP attention map, `--llm` adds the two narrative paragraphs
(needs `SILICONFLOW_API_KEY`, see `.env.example`). Without `--llm` everything runs
offline.

## Reproducibility

The [reproducibility page](docs/REPRODUCIBILITY.md) maps each point of a code
review to a file and a command: environment, step-by-step instructions, example
configs, input/output formats, weights and the offline example. The tests run
without a GPU:

```bash
python -m pytest -q
```

Deliberate limitations (two different risk scores, single-process training, no
resume) are listed in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) rather than
left implicit.

## Citation

The full author list follows the manuscript, which is under review at the time of
this release. Machine-readable metadata: [CITATION.cff](CITATION.cff).

```bibtex
@software{bcares2026,
  title  = {B-CAREs: Breast Cancer Assessment, Reporting, and Risk Stratification Executive System},
  author = {{The B-CAREs authors}},
  year   = {2026},
  note   = {Code release v1.1.0},
  url    = {https://github.com/DouBeeTwT/B-CAREs}
}
```

## License

Source code: MIT — see [LICENSE](LICENSE). The images, masks and clinical table
under `dataset/` are for non-commercial research use. The model weights are
provided for non-commercial academic research and must not be redistributed.

## Contact

Corresponding author, or open an issue at
<https://github.com/DouBeeTwT/B-CAREs/issues>.
