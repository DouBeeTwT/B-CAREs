# Changelog

All notable changes to the released code are documented here. The version
corresponds to the git tag of the same name.

## [1.1.0] — 2026-10-26

Reproducibility release, prepared in response to the *Remarks on Code
Availability* of the review. No model architecture or training hyper-parameter
was changed relative to the version used for the manuscript.

### Added

* `configs/` — YAML configuration for training, single-image inference and the
  offline demo, including the loss weights reported in the manuscript
  (α₁ = 0.8, α₂–α₅ = 1, β = 0.5, focal α = 0.75, γ = 2).
* `bcares/` — configuration loader (YAML + CLI overrides), label tables for the
  three tasks, device resolution.
* `demo.py` — offline end-to-end example that runs without the clinical dataset,
  without checkpoints and without a GPU (`--random-weights`).
* `predict.py` — batch inference over a directory, writes `predictions.csv`.
* `scripts/build_coco.py` — converts images + masks + `Baseline.csv` into the
  COCO layout expected by `train.py`; `--task subtype|birads|sln`.
* `scripts/verify_weights.py` and `weights/manifest.json` — download links, target
  paths, size and SHA-256 verification for the three checkpoints.
* `templates/report_template.html` — the report template, previously missing from
  the repository (`Temp/temp.html` was referenced but not shipped).
* `docs/` — installation, data format, training/inference, weights,
  reproducibility checklist, troubleshooting, known issues.
* `tests/` — 29 tests; the pipeline tests skip automatically when torch or BMIL
  is unavailable.
* `.github/workflows/ci.yml` — CPU CI that runs the tests, a training dry-run and
  the offline demo.
* `LICENSE` (MIT, code only), `CITATION.cff`, `.env.example`, `environment.yml`,
  `pyproject.toml`.

### Changed

* `requirements.txt` replaced: the previous file was a dump of a local conda
  environment containing 40+ `@ file:///home/conda/...` paths and both CUDA 11 and
  CUDA 12 packages, so it could not be installed on any other machine. PyTorch is
  now installed from its own index (`requirements-torch-cu118.txt` /
  `requirements-torch-cpu.txt`), the remaining dependencies are pinned and short.
* `train.py` and `report.py` take `--config` while keeping every previous flag;
  flags still override the YAML.
* `--device auto` replaces the hard-coded `cuda:1` in `report.py`; CPU is now a
  supported target.
* `train.py --learning_rate` is a float (it was declared `type=int`, so any
  explicit value such as `1e-3` failed).
* `--need_P2/6/7`, `--use_sample` and `--debug_anomaly` accept real booleans
  (`--flag true|false`); the previous `type=bool` made `--need_P2 False` evaluate
  to `True`.
* `torch.autograd.set_detect_anomaly` and `CUDA_LAUNCH_BLOCKING` are off by
  default (they were unconditionally enabled and roughly halved throughput).
* Report rendering now uses a Jinja2 template from `templates/` instead of copying
  a `Temp/temp.html` that was never committed.
* The report no longer depends on a locally installed `Bahnschrift.ttf` font.
* README rewritten: correct dataset description (430 images, not 100; breast, not
  thyroid), complete class tables, weights table, training/inference commands,
  citation, license.

### Removed

* Hard-coded API key from `report.py`; the narrative sections now read
  `SILICONFLOW_API_KEY` from the environment and stay disabled unless `--llm` is
  given.
* The dead `if key == "Mean"` branch in the feature scoring code (it could never
  be reached; behaviour is unchanged).

### Fixed

* `report.py` no longer crashes when `Temp/`, `Temp/temp.html` or the font file
  are missing.
* Checkpoint paths in the documentation and in the code now agree
  (`pth/QPMC_BIRADS_Exp001/Best.pth`, `pth/BMIL_v4_Exp010/Last.pth`,
  `pth/QPMC_Lymph_Exp002/Best.pth`).

### Security

* The SiliconFlow API key that was committed in `report.py` must be treated as
  compromised. It has to be revoked in the provider console — a code change cannot
  do that — and the scripts now read the key from the environment. Never commit
  credentials; `.env` is git-ignored and
  `.env.example` documents the variable names.
