# Troubleshooting

## `ImportError: No module named 'BMIL'` (or "The `BMIL` package is not importable")

`train.py` and `report.py` import the model package from the repository root.

* Make sure you run the command **from the repository root** (`ls BMIL` must work), or
* `pip install -e .` once, which installs `bcares` and exposes `BMIL` on the path.

If `BMIL/` is genuinely absent from your checkout, the clone is incomplete —
re-clone the repository.

## `checkpoint not found: pth/...`

The weights are not in git. Follow [WEIGHTS.md](WEIGHTS.md): download, rename to
the file name in the table, and place at the exact `target_path`. Then run

```bash
python scripts/verify_weights.py
```

If you only want to check the installation, use

```bash
python demo.py --config configs/demo.yaml --random-weights
```

## CUDA out of memory

* Lower `data.batch_size` (`--batch_size 8`).
* Single-image inference needs ~2 GB; `--device cpu` always works, it is just slower.
* `--device cuda:1` on a machine with one GPU gives `invalid device ordinal` — use
  `--device auto` or `cuda:0`.

## `RuntimeError: Expected all tensors to be on the same device`

Some BMIL versions keep a hard-coded device inside a submodule. Use `--device`
consistently and avoid mixing `cuda:0` / `cuda:1` in one run.

## `report.py --explain` is very slow or runs out of memory

The SHAP explainer needs hundreds of forward passes. Reduce `explain.shap_max_evals`
(default 200) or leave the panel off; it is not needed for the report itself.

## `RuntimeError: SILICONFLOW_API_KEY is not set`

The narrative paragraphs need an LLM. Either export the key

```bash
export SILICONFLOW_API_KEY=...
```

or keep `llm.enabled: false` (default), which produces the report without the two
narrative fields. The key is **never** stored in the repository; `.env.example`
shows the variable name.

## The share links do not download with `curl` / `wget`

`pan.sjtu.edu.cn` share links require an interactive browser session. Open them in
a browser; automated download is not supported by the provider.

## `pip install -r requirements.txt` fails on `torch`

Install PyTorch first, from its own index:

```bash
pip install -r requirements-torch-cu118.txt   # or requirements-torch-cpu.txt
pip install -r requirements.txt
```

## `mmcv` / `mmdet` errors

They were part of the original environment dump but are not imported by any
script in this repository. They are no longer required; if a local BMIL branch
imports them, install `mmcv==2.0.1` from the OpenMMLab index.

## macOS: `libomp` / OpenMP error from matplotlib or shap

```bash
brew install libomp
```

## Tests fail with `ModuleNotFoundError: torch`

`tests/test_pipeline.py` needs torch and BMIL and is skipped automatically when
they are absent; the remaining tests (labels, config, tools) are pure Python.
