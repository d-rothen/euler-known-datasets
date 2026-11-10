from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("euler-known-datasets")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "0.0.0+unknown"

GENERATOR_NAME = "euler-known-datasets"


def generator_info() -> dict[str, str]:
    """The ``generator`` block recorded in every artifact this package writes."""
    return {"name": GENERATOR_NAME, "version": __version__}
