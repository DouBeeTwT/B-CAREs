# Data format

Everything in this section refers to the demonstration subset committed to this
repository. The full study cohort is not redistributable (see the *Data* section
of the README); the formats are identical.

## 1. What is shipped

```
dataset/
├── Baseline.csv          430 rows, one per image, clinical + label columns
└── test/
    ├── img/              430 ultrasound images, PNG, 224 × 224
    └── lab/              430 segmentation masks, PNG, 224 × 224, 0/255
```

* Images are 8-bit PNG. Mode varies (some are greyscale `L`, some `RGB`), so the
  loaders convert to RGB before anything else.
* Masks are binary; the lesion is `255`, the background `0`. `scripts/build_coco.py`
  derives the object bounding box from the mask.
* Filename convention: `<SUBTYPE>_<4-digit id>.png`, where `SUBTYPE` is one of
  `FBB, BIC, DICS, IDC, PBC, LBC, UDF` (see the table below). The prefix is the
  only source of the subtype label; it is not stored in `Baseline.csv`.

### `dataset/Baseline.csv`

| Column | Example | Meaning |
|---|---|---|
| `PictureName` | `BIC_0009` | image stem (no extension) |
| `BIRADS` | `3` | BI-RADS class index 0–4 (see below) |
| `Lymph` | `0` | sentinel node class index 0–2 (see below) |
| `Age` | `81` | years |
| `Gender` | `Female` | as recorded |
| `Size` | `17.8` | lesion size, mm |
| `FamilyHistory` | `0` | 0/1 |
| `Menophania` | `12` | age at menarche |
| `Period` | `33` | years of menstruation |
| `Duration` | `4` | symptom duration |
| `Pregnant` | `No` | Yes/No |
| `Menopause` | `Yes` | Yes/No |
| `Smoke` | `No` | Yes/No |
| `Drink` | `No` | Yes/No |

Distribution in the released subset (useful to spot a truncated download):

| Task | Class counts |
|---|---|
| BI-RADS (`BIRADS`) | 0: 177, 1: 133, 2: 63, 3: 49, 4: 8 |
| Sentinel node (`Lymph`) | 0: 367, 1: 28, 2: 35 |
| Subtype (from prefix) | FBB 190, IDC 60, BIC 70, PBC 40, UDF 40, DICS 20, LBC 10 |

## 2. Class tables

These tables are the contract between the checkpoints and the labels. **The order
is the order of the output units and must not be re-sorted.** The same tables are
encoded in `bcares/labels.py` and asserted by `tests/test_labels.py`.

### Histological subtype — 7 classes (`--class_num 7`)

| Index | Prefix | Code | Name | Category |
|---:|---|---|---|---|
| 0 | FBB | BF | Benign fibroadenoma | benign |
| 1 | BIC | IBCN | Invasive breast cancer, no special type | invasive |
| 2 | DICS | DCIS | Ductal carcinoma in situ | in-situ |
| 3 | IDC | IDC | Invasive ductal carcinoma | invasive |
| 4 | PBC | ILC | Invasive lobular carcinoma | invasive |
| 5 | LBC | LCIS | Lobular carcinoma in situ | in-situ |
| 6 | UDF | IBCU | Invasive breast cancer, undefined | invasive |

### BI-RADS — 5 classes (`--class_num 5`)

| Index | Reported category |
|---:|---|
| 0 | 1–3 |
| 1 | 4a |
| 2 | 4b |
| 3 | 4c |
| 4 | 5–6 |

### Sentinel lymph node metastasis — 3 classes (`--class_num 3`)

| Index | Reported category |
|---:|---|
| 0 | 0 metastatic nodes |
| 1 | 1–2 metastatic nodes |
| 2 | more than 3 metastatic nodes |

## 3. Pre-processing used by the models

| Step | Value | Where |
|---|---|---|
| Convert to RGB | always | `report.py::get_score`, `predict.py` |
| Input size | 224 × 224 | `configs/*.yaml` → `model.figure_size` |
| To tensor | `T.ToTensor()` (0–1, CHW) | `train.py`, `report.py` |
| Normalise (training) | mean `[0.2, 0.2, 0.2]`, std `[0.08, 0.08, 0.08]` | `train.py::build_loaders` |
| Normalise (inference) | none beyond `ToTensor()` | `report.py` |

The training normalisation constants are the ones used for the reported models;
inference deliberately uses the same transform as the original release so that
the released checkpoints reproduce the published predictions.

## 4. Building a training split

`train.py` expects `<root>/train` and `<root>/val`, each with the COCO-style
annotation file produced by the converter:

```bash
python scripts/build_coco.py --task subtype --val-fraction 0.2 --seed 1 --out Data
# Data/train/images/*.png, Data/train/annotations.json
# Data/val/images/*.png,   Data/val/annotations.json
```

`--task` selects which label becomes the COCO category (`subtype`, `birads` or
`sln`); the bounding box always comes from the mask. `category_id` is 1-based by
default; pass `--category-id-base 0` if your copy of `BMIL.data.dataset` expects
zero-based ids.

The 430-image subset is split deterministically (`--seed`), which is enough for a
pipeline check but **not** for measuring performance: the counts per class are
far too small and the images come from the held-out test set of the study.

## 5. Outputs

### Single-image report (`report.py`)

`<output_dir>/<image>_report/` contains

| File | Content |
|---|---|
| `report.html` | the rendered report (template: `templates/report_template.html`) |
| `img.png` | copy of the input image |
| `mask.png` | segmentation overlay with the predicted box |
| `score1.png` … `score4.png` | class-probability bars for BI-RADS, subtype, SLN, features |
| `attention.png` | SHAP attention map — only with `--explain` |

The integer `total risk score` in the report is the sum of the four contributions
shown in the table (BI-RADS index + subtype score + SLN index + feature score).
It is the score displayed by the reporting module; the prognostic score of the
manuscript (Eq. 5, Cox coefficients 0.49 / 0.51 / 0.67 and thresholds 0.53 / 3.66)
is produced by the survival analysis, not by this script. See
`docs/KNOWN_ISSUES.md`.

### Batch inference (`predict.py` / `demo.py`)

`<output_dir>/predictions.csv` (or `demo_predictions.csv`):

| Column | Meaning |
|---|---|
| `image` | image stem |
| `birads_index`, `birads_label`, `birads_prob` | predicted class, its label, its probability |
| `subtype_index`, `subtype_code`, `subtype_name`, `subtype_prob` | same for subtype |
| `sln_index`, `sln_label`, `sln_prob` | same for the sentinel node task |
| `risk_score` | integer sum of the three task contributions |

`demo.py` additionally writes `demo_summary.txt` with agreement against
`dataset/Baseline.csv` (meaningless in `--random-weights` mode).
