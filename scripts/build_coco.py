#!/usr/bin/env python3
"""Build a COCO-style dataset split from the images shipped in this repository.

The 430 images in ``dataset/test/img`` are a *demonstration subset*: they are
enough to smoke-test the training pipeline, not to reproduce the manuscript.
Use your own COCO annotations for real training.

    python scripts/build_coco.py --task subtype --val-fraction 0.2 --seed 1
    python scripts/build_coco.py --task birads --out Data_BIRADS
    python scripts/build_coco.py --task sln    --out Data_SLN

Produces (defaults):

    Data/train/images/*.png
    Data/train/annotations.json
    Data/val/images/*.png
    Data/val/annotations.json

COCO schema written here
------------------------
    images[]      : {id, file_name, width, height}
    annotations[] : {id, image_id, category_id, bbox: [x, y, w, h], area, iscrowd}
    categories[]  : {id, name}

``category_id`` is 1-based by default (standard COCO); pass ``--category-id-base 0``
if your copy of ``BMIL.data.dataset`` indexes categories from zero.  The label of
an image comes from ``--task``:

* ``subtype`` — filename prefix (FBB, BIC, DICS, IDC, PBC, LBC, UDF), 7 classes
* ``birads``  — ``BIRADS`` column of the label table, 5 classes
* ``sln``     — ``Lymph`` column of the label table, 3 classes

The bounding box always comes from the segmentation mask in ``dataset/test/lab``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bcares.labels import BIRADS_CLASSES, SLN_CLASSES, SUBTYPE_CLASSES, from_prefix  # noqa: E402

TASKS = {
    "subtype": [cls.code for cls in SUBTYPE_CLASSES],
    "birads": [cls.label for cls in BIRADS_CLASSES],
    "sln": [cls.label for cls in SLN_CLASSES],
}


def read_label_table(labels_csv: str) -> dict:
    """Return {image_stem: {"birads": int, "sln": int}} from Baseline.csv."""
    table = {}
    with open(labels_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("PictureName") or "").strip()
            if not name:
                continue
            table[name] = {"birads": int(row["BIRADS"]), "sln": int(row["Lymph"])}
    return table


def label_index(task: str, stem: str, table: dict):
    """Class index of one image for the requested task, or None if unavailable."""
    if task == "subtype":
        try:
            return from_prefix(stem.split("_")[0]).index
        except KeyError:
            return None
    row = table.get(stem)
    return None if row is None else row[task]


def mask_to_bbox(mask_path: str) -> list:
    """Tight [x, y, w, h] around the non-zero pixels of a mask."""
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        height, width = mask.shape
        return [0, 0, width, height]
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]) - 1)
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]) - 1)
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def write_split(records: list, out_dir: str, categories: list, category_base: int,
                link: bool) -> int:
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    images, annotations = [], []
    for index, record in enumerate(records):
        image_id = index + 1
        dest = os.path.join(image_dir, os.path.basename(record["image"]))
        if link:
            try:
                os.link(record["image"], dest)
            except OSError:
                shutil.copy2(record["image"], dest)
        else:
            shutil.copy2(record["image"], dest)

        width, height = record["size"]
        images.append({"id": image_id, "file_name": os.path.basename(record["image"]),
                       "width": width, "height": height})
        x, y, w, h = record["bbox"]
        annotations.append({"id": image_id, "image_id": image_id,
                            "category_id": record["label_index"] + category_base,
                            "bbox": [x, y, w, h], "area": float(w * h), "iscrowd": 0})

    payload = {
        "info": {"description": "B-CAREs demonstration split", "version": "1.1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + category_base, "name": name}
                       for i, name in enumerate(categories)],
    }
    with open(os.path.join(out_dir, "annotations.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("{}: {} images -> {}".format(os.path.basename(out_dir), len(images), out_dir))
    return len(images)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="subtype", choices=sorted(TASKS),
                        help="which label becomes the COCO category")
    parser.add_argument("--images", default="dataset/test/img", help="image directory")
    parser.add_argument("--masks", default="dataset/test/lab", help="segmentation mask directory")
    parser.add_argument("--labels", default="dataset/Baseline.csv", help="label table")
    parser.add_argument("--out", default="Data", help="output directory")
    parser.add_argument("--val-fraction", default=0.2, type=float, help="fraction held out for validation")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--category-id-base", default=1, type=int, choices=[0, 1])
    parser.add_argument("--link", action="store_true", help="hard-link instead of copying images")
    args = parser.parse_args(argv)

    table = read_label_table(args.labels)
    names = sorted(name for name in os.listdir(args.images)
                   if name.lower().endswith((".png", ".jpg", ".jpeg")))
    if not names:
        raise SystemExit("no images in {}".format(args.images))

    records, skipped = [], []
    for name in names:
        stem = os.path.splitext(name)[0]
        label = label_index(args.task, stem, table)
        mask_path = os.path.join(args.masks, name)
        if label is None or not os.path.isfile(mask_path):
            skipped.append(name)
            continue
        with Image.open(os.path.join(args.images, name)) as image:
            size = image.size
        records.append({"image": os.path.join(args.images, name), "label_index": label,
                        "bbox": mask_to_bbox(mask_path), "size": size})

    if not records:
        raise SystemExit("no usable images; check --task, --labels and --masks")

    random.Random(args.seed).shuffle(records)
    n_val = max(1, int(round(len(records) * args.val_fraction))) if len(records) > 1 else 0
    val, train = records[:n_val], records[n_val:]

    categories = TASKS[args.task]
    write_split(train, os.path.join(args.out, "train"), categories,
                args.category_id_base, args.link)
    if val:
        write_split(val, os.path.join(args.out, "val"), categories,
                    args.category_id_base, args.link)

    if skipped:
        print("[warn] skipped {} file(s) without a label or mask, e.g. {}".format(
            len(skipped), ", ".join(skipped[:5])))
    print("Done: {} train / {} val, task={!r}, {} classes.".format(
        len(train), len(val), args.task, len(categories)))
    print("Next: python train.py --config configs/train_multitask.yaml --root {} "
          "--class_num {}".format(args.out, len(categories)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
