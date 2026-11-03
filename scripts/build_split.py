#!/usr/bin/env python3
"""Assemble train/val splits in the layout expected by ``BMIL.data.dataset``.

``COCODataset`` does not read a COCO JSON file: it lists two directories and pairs
them by sorted filename,

    <root>/img/*.png      ultrasound images
    <root>/lab/*.png      binary segmentation masks, same file names

and derives the class label from the filename prefix (see the table below).  This
script builds that structure without touching the labels:

    python scripts/build_split.py --out Data                     # 80 / 20 split
    python scripts/build_split.py --out Data --val-fraction 0.1 --seed 7

    Data/train/img/..., Data/train/lab/...
    Data/val/img/...,   Data/val/lab/...

Class labels are *not* written anywhere: they are inferred by the loader from the
prefix, which is why the file names must be preserved.

Prefix -> class index (``BMIL/data/dataset.py``):

    0 FBB   1 BIC   2 DICS   3 IDC   4 LBC   5 PBC   6 UDF
    (the same loader also accepts A-E as indices 0-4, the naming used for the
     5-class BI-RADS files)

The 430-image subset shipped with this repository is meant to exercise the code
paths, not to measure performance; use your own annotated data for real training.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bcares.labels import SUBTYPE_CLASSES, from_prefix  # noqa: E402

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def collect_pairs(images_dir: str, masks_dir: str):
    """Return [(name, prefix)] for files that exist in both directories."""
    if not os.path.isdir(images_dir):
        raise SystemExit("image directory not found: {}".format(images_dir))
    if not os.path.isdir(masks_dir):
        raise SystemExit("mask directory not found: {}".format(masks_dir))

    names = sorted(name for name in os.listdir(images_dir)
                   if name.lower().endswith(IMAGE_SUFFIXES))
    missing_masks = [name for name in names if not os.path.isfile(os.path.join(masks_dir, name))]
    if missing_masks:
        print("[warn] {} image(s) have no mask and are skipped, e.g. {}".format(
            len(missing_masks), ", ".join(missing_masks[:5])), file=sys.stderr)

    pairs, unknown = [], []
    for name in names:
        if name in missing_masks:
            continue
        stem_prefix = os.path.splitext(name)[0].split("_")[0]
        try:
            from_prefix(stem_prefix)
        except KeyError:
            unknown.append(name)
            continue
        pairs.append((name, stem_prefix))
    if unknown:
        print("[warn] {} file(s) with an unrecognised prefix are skipped, e.g. {}".format(
            len(unknown), ", ".join(unknown[:5])), file=sys.stderr)
    return pairs


def materialise(names, src_images: str, src_masks: str, out_dir: str, link: bool) -> None:
    img_dir = os.path.join(out_dir, "img")
    lab_dir = os.path.join(out_dir, "lab")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    for name in names:
        for source, target_dir in ((src_images, img_dir), (src_masks, lab_dir)):
            target = os.path.join(target_dir, name)
            if os.path.exists(target):
                continue
            if link:
                try:
                    os.link(os.path.join(source, name), target)
                    continue
                except OSError:
                    pass  # different filesystem: fall back to copying
            shutil.copy2(os.path.join(source, name), target)


def histogram(pairs) -> dict:
    counts = Counter(prefix for _, prefix in pairs)
    return {prefix: counts.get(prefix, 0) for prefix in
            [cls.prefix for cls in SUBTYPE_CLASSES]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default="dataset/test/img", help="source image directory")
    parser.add_argument("--masks", default="dataset/test/lab", help="source mask directory")
    parser.add_argument("--out", default="Data", help="output root (train/ and val/ inside)")
    parser.add_argument("--val-fraction", default=0.2, type=float, help="fraction held out")
    parser.add_argument("--seed", default=1, type=int, help="split seed")
    parser.add_argument("--link", action="store_true", help="hard-link instead of copying")
    parser.add_argument("--dry-run", action="store_true", help="report the split, write nothing")
    args = parser.parse_args(argv)

    pairs = collect_pairs(args.images, args.masks)
    if not pairs:
        raise SystemExit("no usable image/mask pairs found")

    names = [name for name, _ in pairs]
    random.Random(args.seed).shuffle(names)
    n_val = max(1, int(round(len(names) * args.val_fraction))) if len(names) > 1 else 0
    val_names, train_names = names[:n_val], names[n_val:]

    lookup = dict(pairs)
    print("source : {} images, {} masks".format(
        len(os.listdir(args.images)), len(os.listdir(args.masks))))
    print("train  : {:4d} images  {}".format(len(train_names), histogram(
        [(n, lookup[n]) for n in train_names])))
    print("val    : {:4d} images  {}".format(len(val_names), histogram(
        [(n, lookup[n]) for n in val_names])))

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    materialise(train_names, args.images, args.masks, os.path.join(args.out, "train"), args.link)
    if val_names:
        materialise(val_names, args.images, args.masks, os.path.join(args.out, "val"), args.link)

    print("written: {}/train and {}/val (img/ + lab/ pairs)".format(args.out, args.out))
    print("next   : python train.py --config configs/train_multitask.yaml --root {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
