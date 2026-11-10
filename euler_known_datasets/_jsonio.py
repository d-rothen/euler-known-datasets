"""JSON reading and atomic writing.

Reading uses orjson when it is installed (``pip install euler-known-datasets[fast]``),
which matters for the multi-gigabyte tables some datasets ship. Writing always
uses the standard library so a prepared dataset is byte-identical whether or not
orjson is present.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only when the optional extra is installed
    import orjson as _orjson
except ImportError:  # pragma: no cover
    _orjson = None


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "rb") as handle:
        data = handle.read()
    if _orjson is not None:
        return _orjson.loads(data)
    return json.loads(data)


def dumps(value: Any, *, compact: bool = False) -> str:
    if compact:
        return json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def _file_mode() -> int:
    """The mode ``open()`` would give a new file under the current umask."""
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def write_text_atomic(path: str | os.PathLike[str], text: str) -> Path:
    """Write *text* to *path* through a temporary file in the same directory.

    The file gets the permissions a plain ``open()`` would give it; mkstemp's
    own 0600 would hide prepared datasets from the rest of a shared group.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, _file_mode())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return target


def write_json_atomic(
    path: str | os.PathLike[str], value: Any, *, compact: bool = False
) -> Path:
    return write_text_atomic(path, dumps(value, compact=compact))
