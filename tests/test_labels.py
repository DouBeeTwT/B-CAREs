"""Label maps must match the trained checkpoints exactly."""

from __future__ import annotations

import pytest

from bcares import labels


def test_subtype_table_is_complete_and_ordered():
    assert len(labels.SUBTYPE_CLASSES) == 7
    assert [c.index for c in labels.SUBTYPE_CLASSES] == list(range(7))
    assert [c.prefix for c in labels.SUBTYPE_CLASSES] == ["FBB", "BIC", "DICS", "IDC",
                                                          "PBC", "LBC", "UDF"]
    assert [c.code for c in labels.SUBTYPE_CLASSES] == ["BF", "IBCN", "DCIS", "IDC",
                                                        "ILC", "LCIS", "IBCU"]


def test_birads_and_sln_tables():
    assert [c.label for c in labels.BIRADS_CLASSES] == ["1-3", "4a", "4b", "4c", "5-6"]
    assert [c.score for c in labels.BIRADS_CLASSES] == [0, 1, 2, 3, 4]
    assert [c.label for c in labels.SLN_CLASSES] == ["0", "1 or 2", "more than 3"]


def test_prefix_lookup_matches_counts_in_the_released_subset():
    """Image counts per prefix are the evidence for the class order (see docs/DATA_FORMAT.md)."""
    expected = {"FBB": 0, "BIC": 1, "DICS": 2, "IDC": 3, "PBC": 4, "LBC": 5, "UDF": 6}
    for prefix, index in expected.items():
        assert labels.from_prefix(prefix).index == index
        assert labels.from_prefix(prefix.lower()).index == index


def test_invasive_lobular_carcinoma_is_class_four():
    cls = labels.SUBTYPE_CLASSES[4]
    assert cls.prefix == "PBC"
    assert cls.code == "ILC"
    assert cls.name == "Invasive lobular carcinoma"
    assert cls.category == "invasive"


def test_subtype_score_modes_only_differ_for_ilc():
    paper = [labels.subtype_score(i, "paper") for i in range(7)]
    corrected = [labels.subtype_score(i, "corrected") for i in range(7)]
    assert paper == [0, 3, 1, 2, 1, 2, 3]
    differences = [(i, a, b) for i, (a, b) in enumerate(zip(paper, corrected)) if a != b]
    assert differences == [(4, 1, 3)]


def test_out_of_range_indices_raise():
    for func in (labels.subtype_name, labels.birads_label, labels.sln_label,
                 labels.subtype_score, labels.birads_score, labels.sln_score):
        with pytest.raises(IndexError):
            func(99)
        with pytest.raises(IndexError):
            func(-1)


def test_unknown_prefix_raises():
    with pytest.raises(KeyError):
        labels.from_prefix("XYZ")
