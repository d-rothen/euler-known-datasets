"""euler-known-datasets: downloaded dataset archives in, euler-ready datasets out.

>>> from euler_known_datasets import prepare, verify
>>> prepare("nuscenes", "/data/downloads/nuscenes", "/data/nuscenes",
...         cameras=["CAM_FRONT_LEFT"])                       # doctest: +SKIP
>>> verify("/data/nuscenes").ok                               # doctest: +SKIP
True
"""

from ._version import __version__
from .archives import UnsafeArchiveError, extract_archive
from .media import MEDIA_MODES, MediaError
from .prepare import (
    RECEIPT_FILENAME,
    PrepareError,
    PrepareResult,
    plan,
    prepare,
    read_receipt,
)
from .recipe import (
    Option,
    Outcome,
    Plan,
    PrepareContext,
    Recipe,
    detect_recipes,
    get_recipe,
    list_recipes,
    register,
)
from .verify import VerifyReport, verify

__all__ = [
    "MEDIA_MODES",
    "MediaError",
    "Option",
    "Outcome",
    "Plan",
    "PrepareContext",
    "PrepareError",
    "PrepareResult",
    "RECEIPT_FILENAME",
    "Recipe",
    "UnsafeArchiveError",
    "VerifyReport",
    "__version__",
    "detect_recipes",
    "extract_archive",
    "get_recipe",
    "list_recipes",
    "plan",
    "prepare",
    "read_receipt",
    "register",
    "verify",
]
