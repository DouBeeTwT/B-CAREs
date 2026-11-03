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
* Masks are binary or index images; the lesion is non-zero, the background `0`.
  `BMIL/data/dataset.py` derives the object bounding box from the mask.
* Filename convention: `<SUBTYPE>_<4-digit id>.png`, where `SUBTYPE` is one of
  `FBB, BIC, DICS, IDC, LBC, PBC, UDF` (see the table below). **The prefix is the
  only source of the subtype label** — `dataset/Baseline.csv` does not contain it.

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
| Subtype (from prefix) | FBB 190, BIC 70, DICS 20, IDC 60, LBC 10, PBC 40, UDF 40 |

## 2. Class tables

These tables are the contract between the checkpoints and the labels. **The order
is the order of the output units and must not be re-sorted.** The same tables are
encoded in `bcares/labels.py`, in `BMIL/data/dataset.py`, and asserted by
`tests/test_labels.py`.

### Histological subtype — 7 classes (`--class_num 7`)

| Index | Prefix | Code | Name | Category | Score |
|---:|---|---|---|---|---:|
| 0 | FBB | BF | Benign fibroadenoma | benign | 0 |
| 1 | BIC | IBCN | Invasive breast cancer, no special type | invasive | 3 |
| 2 | DICS | DCIS | Ductal carcinoma in situ | in-situ | 1 |
| 3 | IDC | IDC | Invasive ductal carcinoma | invasive | 2 |
| 4 | LBC | LCIS | Lobular carcinoma in situ | in-situ | 1 |
| 5 | PBC | ILC | Invasive lobular carcinoma | invasive | 2 |
| 6 | UDF | IBCU | Invasive breast cancer, undefined | invasive | 3 |

The `Score` column is the contribution of the predicted subtype to the integer
risk score displayed by the reporting module: benign scores 0, in-situ 1, and
invasive 2 or 3.

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

### How the label is read

`COCODataset` maps the filename prefix to an index and also accepts `A`–`E` as
indices 0–4 (the naming used for the 5-class files); `bcares.labels.from_prefix`
implements the same mapping and is tested against it. Renaming a file therefore
changes its label — keep the prefixes intact.

## 3. Pre-processing used by the models

| Step | Value | Where |
|---|---|---|
| Convert to RGB | always | `BMIL/data/dataset.py`, `report.py::get_score` |
| Input size | 224 × 224 | `configs/*.yaml` → `data.image_size` (passed as `figure_size`) |
| To tensor | `T.ToTensor()` (0–1, CHW) | `train.py`, `report.py` |
| Normalise (training) | mean `[0.2, 0.2, 0.2]`, std `[0.08, 0.08, 0.08]` | `train.py::build_loaders` |
| Normalise (inference) | none beyond `ToTensor()` | `report.py` |

The training normalisation constants are the ones used for the reported models;
inference deliberately keeps the transform of the original release so that the
released checkpoints reproduce the published predictions.

## 4. Building a training split

`train.py` passes `<root>/train` and `<root>/val` to `COCODataset`, which expects
two paired directories per split and **no annotation file**:

```bash
python scripts/build_split.py --val-fraction 0.2 --seed 1 --out Data
```

```
Data/train/img/*.png   Data/train/lab/*.png
Data/val/img/*.png     Data/val/lab/*.png
```

`build_split.py` checks that every image has a mask and a recognised prefix,
splits deterministically with `--seed`, and supports `--link` (hard links) and
`--dry-run` (report only). Class labels are never written: the loader infers them
from the filenames, so the files have to keep their names.

The 430-image subset is enough for a pipeline check but **not** for measuring
performance: the per-class counts are small and the images come from the held-out
test set of the study.

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
