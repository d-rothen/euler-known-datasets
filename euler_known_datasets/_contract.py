"""Head validation by the rules euler-loading loads heads with.

The rules of the ``euler_loading`` head addon (for example, ``hierarchy_scope``
only together with ``used_as: condition``) live in euler-dataset-contract
from 0.14 and are registered explicitly. Older contracts leave them to
euler-loading, which registers them when it is imported. Either way, register
them before writing or verifying heads, so a head that passes here is one
euler-loading accepts.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_state: dict[str, bool] = {}


def register_euler_loading_validator() -> bool:
    """Make sure the ``euler_loading`` addon rules are registered.

    Returns ``False`` only when neither the contract nor an installed
    euler-loading provides them.
    """
    if "registered" not in _state:
        try:
            from euler_dataset_contract import register_euler_loading_addon_validator
        except ImportError:  # euler-dataset-contract < 0.14
            try:
                importlib.import_module("euler_loading._dataset_contract")
                _state["registered"] = True
            except ImportError:
                logger.debug("no euler_loading addon rules available; heads not fully checked")
                _state["registered"] = False
        else:
            register_euler_loading_addon_validator()
            _state["registered"] = True
    return _state["registered"]
