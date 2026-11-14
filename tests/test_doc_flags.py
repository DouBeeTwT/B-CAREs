"""Flags used in the documentation must exist in the scripts they document.

Documentation drift is checked mechanically here: every ``python <script>.py ...``
invocation in README.md, docs/ and the CI workflow is parsed, and each ``--flag``
it uses has to be declared by that script. This caught, for example, a documented
``--dry-run`` against a parser that only accepted ``--dry_run``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTED = {"train": "train.py", "report": "report.py", "predict": "predict.py", "demo": "demo.py"}
# demo.py builds its parser by importing predict.build_parser(), so predict.py is
# the file that declares the flags it accepts.
DECLARED_IN = {"train": "train.py", "report": "report.py", "predict": "predict.py",
               "demo": "predict.py"}

DOC_FILES = [ROOT / "README.md", ROOT / ".github" / "workflows" / "ci.yml"]
DOC_FILES += sorted((ROOT / "docs").glob("*.md"))

INVOCATION = re.compile(r"python3?\s+(train|report|predict|demo)\.py([^\n|`]*)")
FLAG = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*")


def documented_flags():
    """Yield (script, flag) pairs found in the shipped documentation."""
    for path in DOC_FILES:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for match in INVOCATION.finditer(text):
            script = match.group(1)
            for index, flag in enumerate(FLAG.findall(match.group(2))):
                if flag == "--help":
                    continue
                yield pytest.param(script, flag,
                                   id="{}-{}-{}-{}".format(path.name, script, index, flag))


@pytest.mark.parametrize("script,flag", list(documented_flags()))
def test_documented_flag_exists(script, flag):
    target = DECLARED_IN[script]
    source = (ROOT / target).read_text(encoding="utf-8")
    assert '"{0}"'.format(flag) in source or "'{0}'".format(flag) in source, (
        "{} documents {}, but {} does not declare it".format(
            DOCUMENTED[script], flag, target))
