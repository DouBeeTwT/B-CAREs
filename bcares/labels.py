"""Label maps for the three prediction tasks.

Every table below is *ordered by model class index*, i.e. ``CLASSES[i]`` is the
class whose output unit has index ``i``.  The order is fixed by the trained
checkpoints and must not be re-sorted: it is the order used when the models were
trained and when the manuscript's confusion matrices were computed.

Source of the order and of the naming: ``report.py`` (``switch()``), the
``output_names`` passed to SHAP, and the per-class image counts in
``dataset/test/img``.  See ``docs/DATA_FORMAT.md`` for the derivation.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = [
    "SUBTYPE_CLASSES",
    "BIRADS_CLASSES",
    "SLN_CLASSES",
    "subtype_name",
    "subtype_code",
    "subtype_prefix",
    "subtype_category",
    "subtype_score",
    "birads_label",
    "birads_score",
    "sln_label",
    "sln_score",
    "from_prefix",
]


class SubtypeClass(NamedTuple):
    index: int
    prefix: str          # filename prefix in dataset/test/img, e.g. "FBB"
    code: str            # short code used in the manuscript, e.g. "BF"
    name: str            # full histological name
    category: str        # "benign" | "in-situ" | "invasive"
    score_paper: int     # risk-score contribution used for the published analysis
    score_corrected: int # value implied by the category (see docs/KNOWN_ISSUES.md)


#: Class order of the subtype model (7 classes, ``--class_num 7``).
SUBTYPE_CLASSES = (
    SubtypeClass(0, "FBB", "BF", "Benign fibroadenoma", "benign", 0, 0),
    SubtypeClass(1, "BIC", "IBCN", "Invasive breast cancer, no special type", "invasive", 3, 3),
    SubtypeClass(2, "DICS", "DCIS", "Ductal carcinoma in situ", "in-situ", 1, 1),
    SubtypeClass(3, "IDC", "IDC", "Invasive ductal carcinoma", "invasive", 2, 2),
    # PBC = invasive lobular carcinoma (40 images).  The published score table
    # assigns 1 here, which is the value for an in-situ lesion; the corrected
    # value would be 3.  The default keeps the published behaviour so that the
    # released code reproduces the manuscript; see docs/KNOWN_ISSUES.md.
    SubtypeClass(4, "PBC", "ILC", "Invasive lobular carcinoma", "invasive", 1, 3),
    SubtypeClass(5, "LBC", "LCIS", "Lobular carcinoma in situ", "in-situ", 2, 2),
    SubtypeClass(6, "UDF", "IBCU", "Invasive breast cancer, undefined", "invasive", 3, 3),
)


class BiradsClass(NamedTuple):
    index: int
    label: str           # category as reported to the clinician
    score: int           # risk-score contribution (== index, as in the paper)


#: Class order of the BI-RADS model (5 classes, ``--class_num 5``).
BIRADS_CLASSES = (
    BiradsClass(0, "1-3", 0),
    BiradsClass(1, "4a", 1),
    BiradsClass(2, "4b", 2),
    BiradsClass(3, "4c", 3),
    BiradsClass(4, "5-6", 4),
)


class SlnClass(NamedTuple):
    index: int
    label: str
    score: int


#: Class order of the sentinel lymph node model (3 classes, ``--class_num 3``).
SLN_CLASSES = (
    SlnClass(0, "0", 0),
    SlnClass(1, "1 or 2", 1),
    SlnClass(2, "more than 3", 2),
)

_PREFIX_TO_INDEX = {cls.prefix: cls.index for cls in SUBTYPE_CLASSES}


def _check(index: int, table, what: str) -> int:
    if not 0 <= int(index) < len(table):
        raise IndexError(f"{what} class index {index} outside 0..{len(table) - 1}")
    return int(index)


def subtype_name(index: int) -> str:
    return SUBTYPE_CLASSES[_check(index, SUBTYPE_CLASSES, "subtype")].name


def subtype_code(index: int) -> str:
    return SUBTYPE_CLASSES[_check(index, SUBTYPE_CLASSES, "subtype")].code


def subtype_prefix(index: int) -> str:
    return SUBTYPE_CLASSES[_check(index, SUBTYPE_CLASSES, "subtype")].prefix


def subtype_category(index: int) -> str:
    """Broad category used in the narrative part of the report."""
    return SUBTYPE_CLASSES[_check(index, SUBTYPE_CLASSES, "subtype")].category


def subtype_score(index: int, mode: str = "paper") -> int:
    """Risk-score contribution of a subtype class.

    ``mode="paper"`` (default) reproduces the published analysis;
    ``mode="corrected"`` uses the value implied by the histological category.
    """
    cls = SUBTYPE_CLASSES[_check(index, SUBTYPE_CLASSES, "subtype")]
    if mode == "paper":
        return cls.score_paper
    if mode == "corrected":
        return cls.score_corrected
    raise ValueError(f"mode must be 'paper' or 'corrected', got {mode!r}")


def birads_label(index: int) -> str:
    return BIRADS_CLASSES[_check(index, BIRADS_CLASSES, "BI-RADS")].label


def birads_score(index: int) -> int:
    return BIRADS_CLASSES[_check(index, BIRADS_CLASSES, "BI-RADS")].score


def sln_label(index: int) -> str:
    return SLN_CLASSES[_check(index, SLN_CLASSES, "SLN")].label


def sln_score(index: int) -> int:
    return SLN_CLASSES[_check(index, SLN_CLASSES, "SLN")].score


def from_prefix(prefix: str) -> SubtypeClass:
    """Look up a subtype class by file-name prefix (``"IDC"`` -> index 3)."""
    try:
        return SUBTYPE_CLASSES[_PREFIX_TO_INDEX[prefix.upper()]]
    except KeyError as exc:
        raise KeyError(
            f"unknown subtype prefix {prefix!r}; expected one of "
            f"{', '.join(sorted(_PREFIX_TO_INDEX))}"
        ) from exc
