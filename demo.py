"""Minimal executable example — runs the whole B-CAREs pipeline offline.

    python demo.py --config configs/demo.yaml --random-weights   # no checkpoints, no GPU
    python demo.py --config configs/demo.yaml                   # with the released weights

What it does
------------
1. reads images from ``input.dir`` (20 by default, ``limit: 0`` for all 430),
2. runs the three task models (BI-RADS, histological subtype, sentinel lymph node),
3. writes ``reports/demo_predictions.csv`` and ``reports/demo_summary.txt``,
4. renders one full HTML report for the first image, and
5. compares the predictions with ``dataset/Baseline.csv`` when checkpoints are used.

With ``--random-weights`` the networks are untrained, so the accuracy figures are
meaningless; the run only proves that data loading, model construction, inference
and report rendering work end-to-end.  That is exactly what a reviewer without
the clinical dataset and without GPUs can execute.
"""

from __future__ import annotations

import argparse
import os
import sys

from bcares.config import PREDICT_KEYMAP, parse_with_config, resolve_device

BANNER = "B-CAREs minimal example"


def build_parser() -> argparse.ArgumentParser:
    # Reuse the batch-inference keys; demo-specific flags are added below.
    from predict import build_parser as build_predict_parser

    parser = build_predict_parser()
    parser.description = "Minimal, offline end-to-end demo of B-CAREs."
    parser.add_argument("--report", default=True, type=lambda v: str(v).lower() not in {"false", "0", "no"},
                        help="render one HTML report (default: true)")
    return parser


def main(argv=None) -> int:
    args = parse_with_config(build_parser(), PREDICT_KEYMAP, argv)
    args.device = resolve_device(args.device)

    from predict import infer_image, list_images, load_labels, summarise, write_csv

    print("=" * 68)
    print("{} | device={} | random_weights={}".format(BANNER, args.device, args.random_weights))
    print("=" * 68)

    images = list_images(args.image_dir, args.limit)
    labels = load_labels(args.labels_csv)
    weights = {"birads": args.birads_weights, "subtype": args.subtype_weights,
               "lymph": args.lymph_weights}

    rows = []
    for index, path in enumerate(images, 1):
        rows.append(infer_image(path, weights, args.device, args.subtype_score, args.random_weights))
        if index % 5 == 0 or index == len(images):
            print("[{}/{}] done".format(index, len(images)))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "demo_predictions.csv")
    write_csv(rows, csv_path)

    summary = summarise(rows, labels)
    lines = ["B-CAREs demo summary",
             "images: {}".format(summary["n_images"]),
             "labelled images compared: {}".format(summary["n_labelled"]),
             "device: {}".format(args.device),
             "random weights: {}".format(args.random_weights)]
    for task in ("birads", "subtype", "sln"):
        key = "{}_accuracy".format(task)
        if key in summary:
            lines.append("{} agreement: {:.4f}".format(task, summary[key]))
    if args.random_weights:
        lines.append("NOTE: random weights - agreement figures are meaningless.")
    summary_path = os.path.join(args.output_dir, "demo_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    if args.report:
        from report import ABCGMS

        report_dir = ABCGMS(
            {"age": 52, "gender": "Female", "menopause": "No", "family": "No",
             "smoke": "No", "drink": "No"},
            images[0], device=args.device, weights=weights, output_root=args.output_dir,
            llm={"enabled": False}, explain=False, subtype_score_mode=args.subtype_score,
            random_weights=args.random_weights)
        print("Report: {}".format(os.path.join(report_dir, "report.html")))

    print("\nDemo finished: {} + {}".format(csv_path, summary_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
