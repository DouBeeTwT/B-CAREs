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
It ranks the same way for a single patient's four contributions, but it is **not**
the 5-year risk score of the paper and must not be compared with the thresholds.

## 2. Subtype score of invasive lobular carcinoma

In the published score lookup, class index 4 (`PBC`, invasive lobular carcinoma)
carries score 1, the value otherwise used for in-situ lesions. The released code
reproduces that lookup exactly, so that its output matches the manuscript:

```yaml
subtype_score: paper      # default: index 4 -> 1
subtype_score: corrected  # index 4 -> 3 (value implied by the invasive category)
```

`--subtype_score corrected` is provided for sensitivity runs. Which value entered
the published analysis has to be confirmed against the analysis code; the release
keeps `paper` as the default and documents the discrepancy rather than silently
changing results.

## 3. Distributed training is not wired in

`train.py` accepts `--world_size` and `--device` for compatibility with the
original launch scripts, but the training loop is single-process. Distributed
data-parallel training was used for the paper and is not included in this release.

## 4. No checkpoint resume

Training always starts from random initialisation. `Last.pth` / `Best.pth` are
written each epoch, so an interrupted run can be evaluated, but not continued.

## 5. Feature thresholds are fixed two-bin splits

`report.py` converts each morphological feature into a 0/1 contribution with a
fixed two-bin split (`int(value * 2)`), matching the published pipeline. The
original source contained a branch for a three-bin "Mean" feature that could
never trigger because no such key exists; it was removed without changing output.

## 6. COCO category ids

`scripts/build_coco.py` writes 1-based `category_id` (standard COCO). If your copy
of `BMIL.data.dataset.COCODataset` maps categories from 0, build with
`--category-id-base 0`. The released BMIL version is the reference implementation.

## 7. SHAP panel is tied to the subtype model

`report.py --explain` explains the 7-class subtype network (`ResNet101`). The
attention map is not available for the BI-RADS or SLN heads.

## 8. Attention map font and styling

The original report used a locally installed font (`Bahnschrift.ttf`). The
released template uses the default sans-serif family so that report rendering
does not depend on a font file that cannot be redistributed.

## 9. Demonstration subset is not a benchmark

`dataset/` contains 430 images taken from the held-out test set, with 10–190
images per class. It exists to exercise the code paths. Accuracy computed on it
is not an estimate of the published performance, and `--random-weights` mode
produces numbers that are meaningless by construction.
