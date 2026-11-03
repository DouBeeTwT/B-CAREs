"""Every argparse dest used in a config key map must exist as a real flag.

This catches typos in ``bcares/config.py`` that would otherwise only show up when
someone actually runs a configuration file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bcares.config import INFER_KEYMAP, PREDICT_KEYMAP, TRAIN_KEYMAP

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("train.py", TRAIN_KEYMAP),
    ("report.py", INFER_KEYMAP),
    ("predict.py", PREDICT_KEYMAP),
]


@pytest.mark.parametrize("script,keymap", CASES, ids=[case[0] for case in CASES])
def test_every_mapped_dest_is_declared(script, keymap):
    source = (ROOT / script).read_text(encoding="utf-8")
    missing = sorted(dest for dest in set(keymap.values())
                     if '"--{}"'.format(dest) not in source
                     and "'--{}'".format(dest) not in source)
    assert not missing, "not declared in {}: {}".format(script, missing)


@pytest.mark.parametrize("script,keymap", CASES, ids=[case[0] for case in CASES])
def test_key_map_has_no_duplicate_destinations(script, keymap):
    destinations = list(keymap.values())
    duplicates = {dest for dest in destinations if destinations.count(dest) > 1}
    assert not duplicates, "{} maps several keys to {}".format(script, duplicates)
