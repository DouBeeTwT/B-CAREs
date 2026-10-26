"""End-to-end smoke tests — skipped automatically when torch/BMIL are absent.

These tests do not need model weights: they run the full pipeline with randomly
initialised networks (``--random-weights``), which is exactly the mode advertised
in the README for reviewers without the clinical dataset or a GPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch", reason="PyTorch is not installed")
pytest.importorskip("torchvision", reason="torchvision is not installed")
pytest.importorskip("BMIL", reason="the BMIL package is not available")


def test_report_single_image(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    from report import main

    code = main(["--image", "dataset/test/img/BIC_0009.png", "--device", "cpu",
                 "--output_dir", str(tmp_path), "--random_weights"])
    assert code == 0
    report_dir = tmp_path / "BIC_0009_report"
    assert (report_dir / "report.html").is_file()
    html = (report_dir / "report.html").read_text(encoding="utf-8")
    assert "B-CAREs" in html
    assert (report_dir / "mask.png").is_file()
    for name in ("score1.png", "score2.png", "score3.png", "score4.png"):
        assert (report_dir / name).is_file()


def test_demo_runs_without_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    from demo import main

    code = main(["--image_dir", "dataset/test/img", "--limit", "2", "--device", "cpu",
                 "--output_dir", str(tmp_path), "--labels_csv", "dataset/Baseline.csv",
                 "--random_weights"])
    assert code == 0
    assert (tmp_path / "demo_predictions.csv").is_file()
    assert (tmp_path / "demo_summary.txt").is_file()
    rows = (tmp_path / "demo_predictions.csv").read_text().strip().splitlines()
    assert len(rows) == 3  # header + 2 images
    assert "random weights" in (tmp_path / "demo_summary.txt").read_text()


def test_predict_batch(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    from predict import main

    code = main(["--image_dir", "dataset/test/img", "--limit", "2", "--device", "cpu",
                 "--output_dir", str(tmp_path), "--labels_csv", "dataset/Baseline.csv",
                 "--random_weights"])
    assert code == 0
    assert (tmp_path / "predictions.csv").is_file()


def test_missing_checkpoint_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    from report import main

    with pytest.raises(FileNotFoundError, match="docs/WEIGHTS.md"):
        main(["--image", "dataset/test/img/BIC_0009.png", "--device", "cpu",
              "--output_dir", str(tmp_path),
              "--birads_weights", str(tmp_path / "nope.pth")])
