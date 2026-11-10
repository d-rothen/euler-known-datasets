"""The package must parse as Python 3.9, the oldest supported version."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "euler_known_datasets"


@pytest.mark.parametrize(
    "path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_parses_as_python_39(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))
