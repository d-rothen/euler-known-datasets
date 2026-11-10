"""Head validation by the rules euler-loading loads heads with.

euler-dataset-contract owns the rules of the ``euler_loading`` head addon (for
example, ``hierarchy_scope`` only together with ``used_as: condition``) and
registers them on request. Register them before writing or verifying heads,
so a head that passes here is one euler-loading accepts.
"""

from __future__ import annotations

from euler_dataset_contract import register_euler_loading_addon_validator


def register_euler_loading_validator() -> bool:
    """Make sure the ``euler_loading`` addon rules are registered."""
    register_euler_loading_addon_validator()
    return True
