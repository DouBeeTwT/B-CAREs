# Training and inference

Every command below is run from the repository root.

## Training

### 1. Build a COCO-style split

```bash
python scripts/build_coco.py --task subtype --val-fraction 0.2 --seed 1 --out Data
```

`--task birads` and `--task sln` build the splits for the other two heads
(5 and 3 classes). With the full cohort, point `data.root` at your own directory
that already contains `train/` and `val/` in the format of
[docs/DATA_FORMAT.md](DATA_FORMAT.md).

### 2. Configure

`configs/train_multitask.yaml` holds the settings of the reported model:

| Group | Key | Meaning |
|---|---|---|
| `data` | `root` | dataset root (`<root>/train`, `<root>/val`) |
| | `batch_size` | 32 in the paper |
| | `image_size` | 224 |
| | `num_workers` | DataLoader workers |
| `model` | `backbone_name`, `protonet_name` | encoder / prototype network |
| | `class_num` | 7 subtype, 5 BI-RADS, 3 SLN |
| | `base_num`, `attention_size`, `blender_size`, `neck_layers` | head sizes |
| `loss` | `task_weights` | α₁–α₅ of Eq. (2): `[0.8, 1, 1, 1, 1]` |
| | `alpha_cls`, `gamma_cls` | focal-loss α = 0.75, γ = 2 |
| `optim` | `learning_rate`, `weight_decay` | 1e-3, 5e-4 (Adam) |
| | `epoch_max`, `epoch_qp` | 150 epochs; class re-weighting starts at 120 |
| `runtime` | `device` | `auto`, `cpu`, `cuda:0`, … |
| | `debug_anomaly` | torch anomaly mode, off by default (it is ~10× slower) |
| `logging` | `wandb`, `project_name`, `exp_name` | optional experiment tracking |

Command-line flags override the YAML, so the historical form still works:

```bash
python train.py --config configs/train_multitask.yaml
python train.py --config configs/train_multitask.yaml --batch_size 8 --device cuda:1
python train.py --root ./Data --class_num 7 --backbone_name StarNet --epoch_max 150
```

### 3. Smoke-test before committing GPU hours

```bash
python train.py --config configs/train_multitask.yaml --dry-run
```

`--dry-run` builds the dataset, the loaders, the model, the loss and the decoder,
loads one batch and runs one forward pass. It writes nothing.

### 4. Run

```bash
python train.py --config configs/train_multitask.yaml 2>&1 | tee train.log
```

Outputs go to `pth/<project_name>_<exp_name>/`:

| File | Content |
|---|---|
| `Last.pth` | weights after the final epoch |
| `Best.pth` | weights at the best validation accuracy |

The reported experiments used eight A800 80 GB GPUs and 150 epochs. On a single
consumer GPU the same schedule takes days; for a pipeline check use
`--epoch_max 2` on the 430-image subset.

`--world_size` is accepted for compatibility but **distributed training is not
wired in this release** — see [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Inference

### Single image, full report

```bash
python report.py --config configs/infer.yaml
python report.py --config configs/infer.yaml --image dataset/test/img/IDC_0436.png --device cpu
python report.py --config configs/infer.yaml --llm        # needs SILICONFLOW_API_KEY
python report.py --config configs/infer.yaml --explain    # adds the SHAP panel (slow)
```

Output: `Reports/<image>_report/` — see
[docs/DATA_FORMAT.md](DATA_FORMAT.md#5-outputs) for the file list.

The narrative sections are optional. With `llm.enabled: false` (the default) the
report is generated entirely offline and the two narrative fields say so.

### Batch inference

```bash
python predict.py --config configs/demo.yaml --limit 50
python predict.py --image_dir dataset/test/img --output_dir reports --device cpu
```

Output: `reports/predictions.csv`, one row per image.

### Minimal example (no data, no GPU, no checkpoints)

```bash
python demo.py --config configs/demo.yaml --random-weights
```

This runs 20 images through the three task models (randomly initialised), writes
the CSV and renders one HTML report. It is the command to use when you want to
check the installation rather than reproduce a number. Drop `--random-weights`
once the checkpoints from [docs/WEIGHTS.md](WEIGHTS.md) are in place to get real
predictions.
