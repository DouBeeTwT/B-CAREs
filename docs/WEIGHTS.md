# Model weights

The three checkpoints are **not** stored in git (they are large and are governed
by the data-use terms below). They are distributed through the institutional
share links used for the manuscript.

| Task | File name | Place at | Share link |
|---|---|---|---|
| BI-RADS (5 classes) | `Best.pth` | `pth/QPMC_BIRADS_Exp001/Best.pth` | <https://pan.sjtu.edu.cn/web/share/ef3c18cf0557a22a5d65b35fca88e442> |
| Histological subtype (7 classes) | `Last.pth` | `pth/BMIL_v4_Exp010/Last.pth` | <https://pan.sjtu.edu.cn/web/share/34371324234abf37f70ff9b4e43a1f99> |
| Sentinel lymph node (3 classes) | `Best.pth` | `pth/QPMC_Lymph_Exp002/Best.pth` | <https://pan.sjtu.edu.cn/web/share/12b42e150ee920db0388446254c7342b> |

The same table, machine-readable, is in [`weights/manifest.json`](../weights/manifest.json).

## Download and placement

1. Open each link in a browser (the share service requires an interactive
   session; `curl` will not work).
2. Extract the `.pth` file from the archive.
3. Rename it to the file name in the table and put it at the path in the table,
   relative to the repository root:

```bash
mkdir -p pth/QPMC_BIRADS_Exp001 pth/BMIL_v4_Exp010 pth/QPMC_Lymph_Exp002
# ... move the downloaded files into those directories ...
```

4. Verify:

```bash
python scripts/verify_weights.py
```

Exit code 0 means every file is present and matches the recorded size/hash.
`missing` means the path is wrong, `hash_mismatch` means the download is corrupt
or a different release.

## Recording checksums (maintainers)

`size_bytes` and `sha256` in `weights/manifest.json` are `null` until the release
archive is frozen. Once the three files are in place:

```bash
python scripts/verify_weights.py --compute
```

paste the printed values into the manifest, commit, and re-run
`python scripts/verify_weights.py` — it must exit 0.

## Why the weights are not in git

* The clinical data-use agreement that covers the cohort does not permit
  unrestricted redistribution of derived model files; access is mediated by the
  institutional share links.
* Each checkpoint is on the order of 10²–10³ MB, which is unsuitable for a git
  repository that reviewers clone.

If the weights cannot be downloaded in your environment, the reproducibility of
the **code paths** can still be checked with

```bash
python demo.py --config configs/demo.yaml --random-weights
```

which exercises data loading, model construction, inference and report rendering
without any checkpoint.

## Terms

The weights are provided for non-commercial academic research. Do not
redistribute them, do not attempt to re-identify individuals, and cite the
manuscript in any work that uses them.
