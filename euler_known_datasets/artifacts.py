"""Writing ds-crawler artifacts for prepared datasets.

Recipes describe their files as a hierarchy of index entries and hand them to
:func:`write_scope`, which stores a complete, scoped ds-crawler artifact set
(``dataset-head.json``, ``ds-crawler.json``, ``index.json``) below
``.ds_crawler/<scope>/``. The index is *prebuilt*: its IDs and hierarchy come
from the recipe, not from regexes, so they can encode facts that file paths do
not carry (a scene, the camera frame a lidar sweep pairs with). ds-crawler loads
it through ``source.prebuilt_index_file``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from ds_crawler import build_dataset_head, create_mapped_dataset_splits

# ds-crawler does not re-export this from the package root yet; it is the
# writer behind ``index_dataset_from_path(save_index=True)`` and the dataset
# writers. Keep the import in this one place.
from ds_crawler.artifacts import save_output_artifacts

from ._contract import register_euler_loading_validator
from ._version import generator_info

INDEX_CONTRACT = {"kind": "dataset_index", "version": "1.0"}
METADATA_DIR = ".ds_crawler"

QualifiedId = tuple  # (*hierarchy_keys, file_id)


def file_entry(
    path: str, file_id: str, attributes: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """A ds-crawler file entry for *path* (relative to the dataset root)."""
    entry: dict[str, Any] = {
        "path": path,
        "id": file_id,
        "path_properties": {},
        "basename_properties": {},
    }
    if attributes:
        entry["attributes"] = dict(attributes)
    return entry


class IndexTree:
    """Accumulates file entries into a ds-crawler index tree."""

    def __init__(self) -> None:
        self._root: dict[str, Any] = {}
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def add(self, hierarchy: Sequence[str], entry: dict[str, Any]) -> None:
        node = self._root
        for key in hierarchy:
            node = node.setdefault("children", {}).setdefault(key, {})
        node.setdefault("files", []).append(entry)
        self._count += 1

    def qualified_ids(self) -> Iterator[QualifiedId]:
        def walk(node: dict[str, Any], keys: tuple) -> Iterator[QualifiedId]:
            for entry in node.get("files", []):
                yield keys + (entry["id"],)
            for key, child in node.get("children", {}).items():
                yield from walk(child, keys + (key,))

        return walk(self._root, ())

    def to_dict(self) -> dict[str, Any]:
        return self._root


def write_scope(
    root: str | os.PathLike[str],
    scope: str,
    *,
    dataset: Mapping[str, Any],
    modality: Mapping[str, Any],
    tree: IndexTree,
    addons: Optional[Mapping[str, Any]] = None,
    execution: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write one scoped ds-crawler artifact set and return the index path.

    The dataset head is built and validated by ds-crawler against the
    euler dataset contract before anything is written. Split files a previous
    run left in the scope are removed: they described the previous index.
    """
    register_euler_loading_validator()
    head = build_dataset_head(
        dataset=dict(dataset), modality=dict(modality), addons=dict(addons or {})
    )
    scope_dir = Path(root) / METADATA_DIR / scope
    if scope_dir.is_dir():
        for stale in scope_dir.glob("split_*.json"):
            stale.unlink()
    output = {
        "contract": dict(INDEX_CONTRACT),
        "generator": generator_info(),
        "execution": dict(execution or {}),
        "head": head,
        "indexing": {},
        "index": tree.to_dict(),
    }
    return save_output_artifacts(Path(root), output, metadata_scope=scope)


def write_splits(
    root: str | os.PathLike[str],
    scope: str,
    splits: Mapping[str, Sequence[QualifiedId]],
) -> list[str]:
    """Write disjoint named splits of one scope from qualified IDs.

    Empty splits are skipped. Returns the names written. ds-crawler checks
    that every ID exists in the scope's index and that no ID is in two splits.
    """
    payload = {name: [list(qid) for qid in ids] for name, ids in splits.items() if ids}
    if not payload:
        return []
    create_mapped_dataset_splits(Path(root), {"splits": payload}, metadata_scope=scope)
    return sorted(payload)
