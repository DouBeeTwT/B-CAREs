"""Smoke tests for the command-line tools (no torch needed)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_IMAGES = ROOT / "dataset" / "test" / "img"
FIXTURE_MASKS = ROOT / "dataset" / "test" / "lab"
NAMES = ["FBB_0006", "BIC_0009", "DICS_0054", "IDC_0436", "PBC_0038", "UDF_0040"]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fixture(tmp_path: Path):
    images, masks = tmp_path / "img", tmp_path / "lab"
    images.mkdir()
    masks.mkdir()
    for name in NAMES:
        shutil.copy(FIXTURE_IMAGES / f"{name}.png", images / f"{name}.png")
        shutil.copy(FIXTURE_MASKS / f"{name}.png", masks / f"{name}.png")
    return images, masks


def test_build_split_mirrors_the_loader_layout(tmp_path):
    build = load_script("build_split")
    images, masks = make_fixture(tmp_path)
    out = tmp_path / "Data"

    assert build.main(["--images", str(images), "--masks", str(masks),
                       "--out", str(out), "--val-fraction", "0.34", "--seed", "1"]) == 0

    for split in ("train", "val"):
        img_dir, lab_dir = out / split / "img", out / split / "lab"
        assert img_dir.is_dir() and lab_dir.is_dir()
        # the loader pairs sorted listings, so names must match exactly
        assert (sorted(p.name for p in img_dir.iterdir())
                == sorted(p.name for p in lab_dir.iterdir()))

    total = sum(len(list((out / split / "img").iterdir())) for split in ("train", "val"))
    assert total == len(NAMES)


def test_build_split_is_deterministic(tmp_path):
    build = load_script("build_split")
    images, masks = make_fixture(tmp_path)
    args = ["--images", str(images), "--masks", str(masks),
            "--val-fraction", "0.2", "--seed", "3", "--link"]
    first, second = tmp_path / "a", tmp_path / "b"
    assert build.main(args + ["--out", str(first)]) == 0
    assert build.main(args + ["--out", str(second)]) == 0
    for split in ("train", "val"):
        a = sorted(p.name for p in (first / split / "img").iterdir())
        b = sorted(p.name for p in (second / split / "img").iterdir())
        assert a == b


def test_build_split_dry_run_writes_nothing(tmp_path):
    build = load_script("build_split")
    images, masks = make_fixture(tmp_path)
    out = tmp_path / "Data"
    assert build.main(["--images", str(images), "--masks", str(masks),
                       "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_build_split_skips_unknown_prefix(tmp_path):
    build = load_script("build_split")
    images, masks = make_fixture(tmp_path)
    shutil.copy(FIXTURE_IMAGES / "IDC_0436.png", images / "ZZZ_0001.png")
    shutil.copy(FIXTURE_MASKS / "IDC_0436.png", masks / "ZZZ_0001.png")
    out = tmp_path / "Data"
    assert build.main(["--images", str(images), "--masks", str(masks), "--out", str(out)]) == 0
    written = [p.name for split in ("train", "val") for p in (out / split / "img").iterdir()]
    assert "ZZZ_0001.png" not in written
    assert len(written) == len(NAMES)


def test_verify_weights_detects_mismatch(tmp_path):
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
                        "LBC": 10, "PBC": 40, "UDF": 40}
    assert len(list(FIXTURE_MASKS.glob("*.png"))) == 430
