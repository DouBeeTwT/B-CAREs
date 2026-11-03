# Installation

Three supported setups. Pick the one that matches your machine; all of them run
the offline demo, only the CUDA one is fast enough for training.

| Setup | Command set | Typical use |
|---|---|---|
| Linux + NVIDIA GPU, CUDA 11.8 | `requirements-torch-cu118.txt` | reproducing training, batch inference |
| Linux, no GPU | `requirements-torch-cpu.txt` | demo, tests, single-image inference |
| macOS (Apple silicon / Intel), no GPU | `requirements-torch-cpu.txt` | demo, tests |

## 0. Prerequisites

| Component | Version used for the paper | Minimum |
|---|---|---|
| Python | 3.11 | 3.10 |
| PyTorch | 2.2.2 | 2.2 |
| CUDA runtime | 11.8 (bundled in the wheel) | — |
| NVIDIA driver | 520 | 520 |
| GPU | 8 × A800 80 GB (training) | 1 × 16 GB for inference |
| Disk | ~10 GB (env + weights + outputs) | 2 GB |

The CUDA wheels ship their own CUDA libraries; a system CUDA toolkit is **not**
required. Only the driver has to be new enough.

## 1. Environment

### Option A — conda (recommended)

```bash
conda env create -f environment.yml      # creates python 3.11 + pip only
conda activate bcares
```

### Option B — venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## 2. PyTorch (choose one)

```bash
# Linux + CUDA 11.8 — the configuration used for the reported experiments
pip install -r requirements-torch-cu118.txt

# Linux / macOS, CPU only
pip install -r requirements-torch-cpu.txt
```

## 3. Remaining dependencies

```bash
pip install -r requirements.txt
pip install -e .            # optional: installs the `bcares` package and the
                            # bcares-{train,report,predict,demo} entry points
```

## 4. Verify

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m pytest -q
python demo.py --config configs/demo.yaml --random-weights
```

The last command needs no GPU, no checkpoints and no API key. It should finish
with `Demo finished: ...` and leave `reports/demo_predictions.csv` and
`reports/BIC_0009_report/report.html` behind.

## Notes

* `pip install -e .` is optional: all scripts also run from the repository root
  without installation, as long as the working directory is the root.
* The `BMIL` package (model, losses, decoders, dataset) is shipped in this
  repository. If it is missing from your checkout, `train.py` and `report.py`
  exit with an explicit message.
* macOS: `torchvision`'s video IO and some OpenCV codecs are not used here; the
  CPU wheels are sufficient for the demo.
* Reproducing the full training run additionally requires the multi-centre
  clinical dataset, which is not publicly distributed (see README, *Data*).
