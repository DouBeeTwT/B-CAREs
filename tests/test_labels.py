"""Label maps must match the trained checkpoints exactly."""

from __future__ import annotations

import pytest

from bcares import labels


def test_subtype_table_is_complete_and_ordered():
    assert len(labels.SUBTYPE_CLASSES) == 7
    assert [c.index for c in labels.SUBTYPE_CLASSES] == list(range(7))
    assert [c.prefix for c in labels.SUBTYPE_CLASSES] == ["FBB", "BIC", "DICS", "IDC",
                                                          "LBC", "PBC", "UDF"]
    assert [c.code for c in labels.SUBTYPE_CLASSES] == ["BF", "IBCN", "DCIS", "IDC",
                                                        "LCIS", "ILC", "IBCU"]


def test_birads_and_sln_tables():
    assert [c.label for c in labels.BIRADS_CLASSES] == ["1-3", "4a", "4b", "4c", "5-6"]
    assert [c.score for c in labels.BIRADS_CLASSES] == [0, 1, 2, 3, 4]
    assert [c.label for c in labels.SLN_CLASSES] == ["0", "1 or 2", "more than 3"]


def test_prefix_lookup_matches_loader_order():
    """The order is the one hard-coded in BMIL/data/dataset.py::COCODataset."""
    expected = {"FBB": 0, "BIC": 1, "DICS": 2, "IDC": 3, "LBC": 4, "PBC": 5, "UDF": 6}
    for prefix, index in expected.items():
        assert labels.from_prefix(prefix).index == index
        assert labels.from_prefix(prefix.lower()).index == index


def test_alternative_prefixes_for_the_five_class_files():
    for prefix, index in {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.items():
        assert labels.from_prefix(prefix).index == index


def test_invasive_lobular_carcinoma_is_index_five():
    cls = labels.SUBTYPE_CLASSES[5]
    assert cls.prefix == "PBC"
    assert cls.code == "ILC"
    assert cls.name == "Invasive lobular carcinoma"
    assert cls.category == "invasive"


def test_in_situ_classes_are_index_two_and_four():
    """DICS and LBC are the in-situ classes; the score table must agree."""
    in_situ = [c.index for c in labels.SUBTYPE_CLASSES if c.category == "in-situ"]
    assert in_situ == [2, 4]


def test_subtype_scores():
    assert [labels.subtype_score(i) for i in range(7)] == [0, 3, 1, 2, 1, 2, 3]


def test_score_agrees_with_histological_category():
    """Benign scores 0, in-situ scores 1, invasive scores 2 or more."""
    for cls in labels.SUBTYPE_CLASSES:
        if cls.category == "benign":
            assert cls.score == 0
        elif cls.category == "in-situ":
            assert cls.score == 1
        else:
            assert cls.score >= 2, cls


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
