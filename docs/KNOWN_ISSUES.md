# Known issues and deliberate limitations

This page records behaviour that is intentional, ambiguous, or not implemented in
v1.1.0, so that it is not mistaken for a defect discovered after release.

## 1. Two different "risk scores"

`report.py` displays an **integer** score: the sum of the BI-RADS class index, the
subtype score, the SLN class index and the imaging-feature score.

The manuscript's prognostic score (Eq. 5) is a Cox model with fitted coefficients
(0.49 / 0.51 / 0.67) and two thresholds (0.53 / 3.66) derived in the training set;
it is computed by the survival analysis on the full cohort, which is not part of
this repository.

Consequence: the number in `report.html` is the reporting module's display score.
It is **not** the 5-year risk score of the paper and must not be compared with the
thresholds.

## 2. Distributed training is not wired in

`train.py` accepts `--world_size` and `--device` for compatibility with the
original launch scripts, but the training loop is single-process. The paper was
trained on eight GPUs with an external launch script that is not part of this
release.

## 3. No checkpoint resume

Training always starts from random initialisation. `Last.pth` / `Best.pth` are
written each epoch, so an interrupted run can be evaluated, but not continued.

## 4. Feature thresholds are fixed two-bin splits

`report.py` converts each morphological feature into a 0/1 contribution with a
fixed two-bin split (`int(value * 2)`), matching the published pipeline. The
original source contained a branch for a three-bin "Mean" feature that could never
trigger because no such key exists; it was removed without changing output.

## 5. Labels live in the filenames, not in an annotation file

`COCODataset` pairs `<root>/img` with `<root>/lab` by sorted filename and derives
the class from the prefix. There is no COCO JSON and no per-image subtype column.
This keeps the released loader identical to the one used for training, but it
means that renaming a file changes its label, and that the three tasks are
prepared from different sets of filenames (`A`–`E` for the 5-class files).

## 6. SHAP panel is tied to the subtype model

`report.py --explain` explains the 7-class subtype network (`ResNet101`). The
attention map is not available for the BI-RADS or SLN heads.

## 7. Report styling

The original report used a locally installed font (`Bahnschrift.ttf`). The
released template uses the default sans-serif family so that report rendering does
not depend on a font file that cannot be redistributed.

## 8. Demonstration subset is not a benchmark

`dataset/` contains 430 images taken from the held-out test set, with 10–190
images per class. It exists to exercise the code paths. Accuracy computed on it is
not an estimate of the published performance, and `--random-weights` mode produces
numbers that are meaningless by construction.

## 9. The early Mask R-CNN baseline is not included

The original working tree contained `BMIL/model/maskrcnn.py`, a standalone
Mask R-CNN script derived from a public detection tutorial (it defines
`PennFudanDataset`). It is not imported by the released package, it requires a
private module (`my_roiu_heads`) and it plays no part in the B-CAREs pipeline, so
it is not part of this release. Ask the corresponding author if you need it.
