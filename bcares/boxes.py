"""Bounding-box helpers shared by the reporting code.

The pipeline indexes with ``img[box[0]:box[2], box[1]:box[3]]``, so in this
code base ``box[0]``/``box[2]`` are row coordinates and ``box[1]``/``box[3]`` are
column coordinates.
"""

from __future__ import annotations

from typing import Sequence, Tuple

__all__ = ["sanitise_box"]


def sanitise_box(box: Sequence[float], height: int, width: int) -> Tuple[int, int, int, int]:
    """Clamp a predicted box to the image and guarantee a non-empty region.

    A randomly initialised network (``demo.py --random-weights``) can emit a
    degenerate box such as ``[0, 0, 0, 0]``; ``cv2.resize`` then fails with
    ``inv_scale_x > 0``. Rather than crash, a degenerate box falls back to the
    whole image, and partial out-of-bounds coordinates are clipped.
    """
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive, got {}x{}".format(width, height))

    row0, col0, row1, col1 = (int(round(float(value))) for value in box[:4])
    if row1 <= row0 or col1 <= col0:
        return 0, 0, height, width

    row0 = min(max(row0, 0), height - 1)
    col0 = min(max(col0, 0), width - 1)
    row1 = min(max(row1, row0 + 1), height)
    col1 = min(max(col1, col0 + 1), width)
    return row0, col0, row1, col1
