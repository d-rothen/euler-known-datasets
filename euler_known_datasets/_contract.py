"""Optional head validation against the installed euler-loading.

euler-loading registers the validator of the ``euler_loading`` head addon with
euler-dataset-contract when it is imported (for example, it only allows
``hierarchy_scope`` together with ``used_as: condition``). Without that import,
ds-crawler accepts any addon payload, so a head could pass here and be rejected
when the dataset is loaded. When euler-loading is installed, import its
validator so heads are checked by the same rules that will load them.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_state: dict[str, bool] = {}


def register_euler_loading_validator() -> bool:
    """Register euler-loading's addon validator; ``False`` if not installed."""
    if "registered" not in _state:
        try:
            importlib.import_module("euler_loading._dataset_contract")
            _state["registered"] = True
        except ImportError:
            logger.debug("euler-loading not installed; its head addon rules are not checked")
            _state["registered"] = False
    return _state["registered"]
