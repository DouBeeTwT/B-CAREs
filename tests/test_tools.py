"""Smoke tests for the two command-line tools (no torch needed)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_IMAGES = ROOT / "dataset" / "test" / "img"
FIXTURE_MASKS = ROOT / "dataset" / "test" / "lab"
LABELS = ROOT / "dataset" / "Baseline.csv"


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_coco_writes_a_valid_split(tmp_path):
    build_coco = load_script("build_coco")
    images = tmp_path / "img"
    masks = tmp_path / "lab"
    images.mkdir()
    masks.mkdir()
    names = ["FBB_0006", "BIC_0009", "DICS_0054", "IDC_0436", "PBC_0038", "UDF_0040"]
    for name in names:
        shutil.copy(FIXTURE_IMAGES / f"{name}.png", images / f"{name}.png")
        shutil.copy(FIXTURE_MASKS / f"{name}.png", masks / f"{name}.png")

    out = tmp_path / "Data"
    code = build_coco.main(["--images", str(images), "--masks", str(masks),
                            "--labels", str(LABELS), "--out", str(out),
                            "--val-fraction", "0.34", "--seed", "1"])
    assert code == 0

    train = json.loads((out / "train" / "annotations.json").read_text())
    val = json.loads((out / "val" / "annotations.json").read_text())
    assert len(train["images"]) + len(val["images"]) == len(names)
    assert len(train["annotations"]) == len(train["images"])
    assert {c["id"] for c in train["categories"]} == set(range(1, 8))
    assert all(1 <= a["category_id"] <= 7 for a in train["annotations"])
    for annotation in train["annotations"]:
        x, y, w, h = annotation["bbox"]
        assert x >= 0 and y >= 0 and w > 0 and h > 0
        assert annotation["area"] == pytest.approx(w * h)


def test_build_coco_is_deterministic(tmp_path):
    build_coco = load_script("build_coco")
    args = ["--images", str(FIXTURE_IMAGES), "--masks", str(FIXTURE_MASKS),
            "--labels", str(LABELS), "--val-fraction", "0.2", "--seed", "1", "--link"]
    first, second = tmp_path / "a", tmp_path / "b"
    assert build_coco.main(args + ["--out", str(first)]) == 0
    assert build_coco.main(args + ["--out", str(second)]) == 0
    a = json.loads((first / "train" / "annotations.json").read_text())
    b = json.loads((second / "train" / "annotations.json").read_text())
    assert [i["file_name"] for i in a["images"]] == [i["file_name"] for i in b["images"]]


def test_build_coco_birads_and_sln_tasks(tmp_path):
    build_coco = load_script("build_coco")
    for task, n_classes in (("birads", 5), ("sln", 3)):
        out = tmp_path / task
        assert build_coco.main(["--task", task, "--images", str(FIXTURE_IMAGES),
                                "--masks", str(FIXTURE_MASKS), "--labels", str(LABELS),
                                "--out", str(out), "--link"]) == 0
        payload = json.loads((out / "train" / "annotations.json").read_text())
        assert len(payload["categories"]) == n_classes


def test_verify_weights_detects_mismatch(tmp_path, capsys):
    verify = load_script("verify_weights")
    payload = tmp_path / "model.pth"
    payload.write_bytes(b"not really a checkpoint")
    digest = verify.sha256(str(payload))

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"weights": [{
        "task": "birads", "target_path": str(payload),
        "size_bytes": payload.stat().st_size, "sha256": digest}]}))
    assert verify.main(["--manifest", str(good)]) == 0

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"weights": [{
        "task": "birads", "target_path": str(payload),
        "size_bytes": payload.stat().st_size, "sha256": "0" * 64}]}))
    assert verify.main(["--manifest", str(bad)]) == 1


def test_shipped_manifest_is_valid():
    payload = json.loads((ROOT / "weights" / "manifest.json").read_text())
    tasks = {entry["task"] for entry in payload["weights"]}
    assert tasks == {"birads", "subtype", "lymph"}
    for entry in payload["weights"]:
        assert entry["url"].startswith("https://")
        assert entry["target_path"].startswith("pth/")


def test_dataset_fixture_counts():
    """The README's per-class counts must match the files actually shipped."""
    suffixes = {}
    for path in FIXTURE_IMAGES.glob("*.png"):
        prefix = path.stem.split("_")[0]
        suffixes[prefix] = suffixes.get(prefix, 0) + 1
    assert suffixes == {"FBB": 190, "BIC": 70, "DICS": 20, "IDC": 60,
                        "PBC": 40, "LBC": 10, "UDF": 40}
    assert len(list(FIXTURE_MASKS.glob("*.png"))) == 430
