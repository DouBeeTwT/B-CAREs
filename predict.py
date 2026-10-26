"""Batch inference over a directory of ultrasound images.

    python predict.py --config configs/demo.yaml
    python predict.py --image_dir dataset/test/img --limit 50 --device cpu

Writes <output_dir>/predictions.csv with one row per image:

    image, birads_index, birads_label, birads_prob,
    subtype_index, subtype_code, subtype_name, subtype_prob,
    sln_index, sln_label, sln_prob, risk_score

and, when a label table is supplied, prints agreement with it.

This script deliberately reuses :func:`report.get_score` so that single-image and
batch inference share exactly the same code path.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

import torchvision.transforms as T

from bcares.config import PREDICT_KEYMAP, parse_with_config, resolve_device
from bcares.labels import (
    BIRADS_CLASSES,
    SLN_CLASSES,
    SUBTYPE_CLASSES,
    birads_label,
    birads_score,
    from_prefix,
    sln_label,
    sln_score,
    subtype_code,
    subtype_name,
    subtype_score,
)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(directory: str, limit: int = 0) -> List[str]:
    if not os.path.isdir(directory):
        raise SystemExit("image directory not found: {}".format(directory))
    names = sorted(name for name in os.listdir(directory) if name.lower().endswith(IMAGE_SUFFIXES))
    if limit:
        names = names[:limit]
    if not names:
        raise SystemExit("no images found in {}".format(directory))
    return [os.path.join(directory, name) for name in names]


def load_labels(csv_path: Optional[str]) -> Dict[str, Dict[str, int]]:
    """Read dataset/Baseline.csv into {image_stem: {birads, subtype, sln}}."""
    if not csv_path:
        return {}
    if not os.path.isfile(csv_path):
        print("[warn] label table not found: {}".format(csv_path))
        return {}
    labels: Dict[str, Dict[str, int]] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("PictureName") or ""
            prefix = name.split("_")[0]
            try:
                labels[name] = {
                    "birads": int(row["BIRADS"]),
                    "sln": int(row["Lymph"]),
                    "subtype": from_prefix(prefix).index,
                }
            except (KeyError, ValueError):
                continue
    return labels


def infer_image(path: str, weights: Dict[str, str], device: str, subtype_score_mode: str = "paper",
                random_weights: bool = False) -> Dict[str, object]:
    """Run the three task models on one image; return a flat result dict."""
    from report import get_score  # imported lazily: report.py pulls in matplotlib/cv2

    transform = T.Compose([T.ToTensor()])
    stem = os.path.splitext(os.path.basename(path))[0]

    birads = get_score(weights.get("birads", ""), "StarNet", device, len(BIRADS_CLASSES),
                       transform, path, random_weights=random_weights)
    subtype = get_score(weights.get("subtype", ""), "ResNet101", device, len(SUBTYPE_CLASSES),
                        transform, path, random_weights=random_weights)
    sln = get_score(weights.get("lymph", ""), "StarNet", device, len(SLN_CLASSES),
                    transform, path, random_weights=random_weights)

    birads_index = int(birads["score"].argmax())
    subtype_index = int(subtype["score"].argmax())
    sln_index = int(sln["score"].argmax())

    risk_score = (birads_score(birads_index)
                  + subtype_score(subtype_index, subtype_score_mode)
                  + sln_score(sln_index))

    return {
        "image": stem,
        "birads_index": birads_index,
        "birads_label": birads_label(birads_index),
        "birads_prob": round(float(birads["score"].max()), 4),
        "subtype_index": subtype_index,
        "subtype_code": subtype_code(subtype_index),
        "subtype_name": subtype_name(subtype_index),
        "subtype_prob": round(float(subtype["score"].max()), 4),
        "sln_index": sln_index,
        "sln_label": sln_label(sln_index),
        "sln_prob": round(float(sln["score"].max()), 4),
        "risk_score": risk_score,
    }


def summarise(rows: List[Dict[str, object]], labels: Dict[str, Dict[str, int]]) -> Dict[str, object]:
    compared = [row for row in rows if row["image"] in labels]
    summary: Dict[str, object] = {"n_images": len(rows), "n_labelled": len(compared)}
    if not compared:
        return summary
    for task, key in (("birads", "birads_index"), ("subtype", "subtype_index"), ("sln", "sln_index")):
        correct = sum(1 for row in compared if int(row[key]) == labels[row["image"]][task])
        summary["{}_accuracy".format(task)] = round(correct / len(compared), 4)
    return summary


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch inference for B-CAREs.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML configuration file")
    parser.add_argument("--image_dir", default="dataset/test/img", type=str)
    parser.add_argument("--labels_csv", default="dataset/Baseline.csv", type=str)
    parser.add_argument("--birads_weights", default="pth/QPMC_BIRADS_Exp001/Best.pth", type=str)
    parser.add_argument("--subtype_weights", default="pth/BMIL_v4_Exp010/Last.pth", type=str)
    parser.add_argument("--lymph_weights", default="pth/QPMC_Lymph_Exp002/Best.pth", type=str)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--output_dir", default="reports", type=str)
    parser.add_argument("--batch_size", default=4, type=int, help="reserved; inference is per image")
    parser.add_argument("--limit", default=0, type=int, help="only the first N images (0 = all)")
    parser.add_argument("--subtype_score", default="paper", choices=["paper", "corrected"])
    parser.add_argument("--random_weights", action="store_true",
                        help="ignore checkpoints and use random initialisation")
    return parser


def main(argv=None) -> int:
    args = parse_with_config(build_parser(), PREDICT_KEYMAP, argv)
    args.device = resolve_device(args.device)

    images = list_images(args.image_dir, args.limit)
    labels = load_labels(args.labels_csv)
    weights = {"birads": args.birads_weights, "subtype": args.subtype_weights,
               "lymph": args.lymph_weights}

    rows = []
    for index, path in enumerate(images, 1):
        rows.append(infer_image(path, weights, args.device, args.subtype_score, args.random_weights))
        if index % 10 == 0 or index == len(images):
            print("[{}/{}] {}".format(index, len(images), os.path.basename(path)))

    out_csv = os.path.join(args.output_dir, "predictions.csv")
    write_csv(rows, out_csv)
    summary = summarise(rows, labels)
    print("Wrote {} rows to {}".format(len(rows), out_csv))
    print("Summary: {}".format(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
