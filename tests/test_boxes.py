"""Bounding-box clamping used before the mask is pasted into the report."""

from __future__ import annotations

import pytest

from bcares.boxes import sanitise_box


def test_normal_box_is_preserved():
    assert sanitise_box([10, 20, 110, 140], 224, 224) == (10, 20, 110, 140)


def test_degenerate_box_falls_back_to_the_whole_image():
    """A randomly initialised network can emit an empty box; cv2.resize would fail."""
    assert sanitise_box([0, 0, 0, 0], 224, 224) == (0, 0, 224, 224)
    assert sanitise_box([50, 50, 49, 200], 224, 224) == (0, 0, 224, 224)
    assert sanitise_box([50, 50, 200, 10], 224, 224) == (0, 0, 224, 224)


def test_negative_coordinates_are_clipped():
    row0, col0, row1, col1 = sanitise_box([-50, -30, 100, 120], 224, 224)
    assert (row0, col0) == (0, 0)
    assert row1 == 100 and col1 == 120


def test_coordinates_past_the_border_are_clipped():
    row0, col0, row1, col1 = sanitise_box([10, 10, 500, 500], 224, 224)
    assert (row0, col0) == (10, 10)
    assert row1 == 224 and col1 == 224


def test_box_at_the_last_pixel_still_has_area():
    row0, col0, row1, col1 = sanitise_box([223, 223, 223, 223], 224, 224)
    assert row1 > row0 and col1 > col0
    assert row1 <= 224 and col1 <= 224


def test_invalid_image_size_is_rejected():
    with pytest.raises(ValueError):
        sanitise_box([0, 0, 10, 10], 0, 224)
