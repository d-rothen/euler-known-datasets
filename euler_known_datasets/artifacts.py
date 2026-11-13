"""Writing ds-crawler artifacts for prepared datasets.

Two ways to write a scope, i.e. a complete artifact set (``dataset-head.json``,
``ds-crawler.json``, ``index.json``) below ``.ds_crawler/<scope>/``:

* :func:`write_scope` stores an index the recipe built entry by entry. It is
  *prebuilt*: IDs and hierarchy come from the recipe, so they can encode facts
  that file paths do not carry (a scene, the camera frame a lidar sweep pairs
  with). ds-crawler loads it through ``source.prebuilt_index_file``.
  Its hierarchy keys are ``name:value`` pairs, such as ``scene:ai_001_001``;
  the scope declares ``:`` as its hierarchy separator, so consumers can split
  them into names and values (to select sections or name output folders).
* :func:`write_regex_scope` indexes a list of paths with ds-crawler regexes and
  keeps the regexes in ``ds-crawler.json``, so the scope can be re-crawled.
  The paths need not exist yet: indexing only matches paths, so a recipe can
  index archive member names and extract just the files it indexed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from ds_crawler import (
    build_crawler_config,
    build_dataset_head,
    build_index_output,
    create_mapped_dataset_splits,
    index_dataset_from_files,
    save_dataset_index,
)

from ._contract import register_euler_loading_validator
from ._version import generator_info

METADATA_DIR = ".ds_crawler"

QualifiedId = tuple  # (*hierarchy_keys, file_id)

#: Separator between name and value in the hierarchy keys of prebuilt scopes.
HIERARCHY_SEPARATOR = ":"

#: Prepared datasets are recorded ground truth, flagged like the ecosystem's
#: other ground-truth heads.
GROUND_TRUTH_ATTRIBUTES = {"gt": True, "baseline": False}


def family_dataset(
    family: str, attributes: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """The ``dataset`` block shared by every scope of a prepared dataset.

    The dataset family, the recipe name such as ``"nuscenes"``, is both id and
    name in every scope: it is the one descriptor that unites them (euler-view
    groups datasets into families by it), and scopes differ by their modality
    key. *attributes* add to the ground-truth flags.
    """
    return {
        "id": family,
        "name": family,
        "attributes": {**GROUND_TRUTH_ATTRIBUTES, **dict(attributes or {})},
    }


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

    def hierarchy_keys(self) -> set[str]:
        """Every hierarchy key used in the tree."""
        keys: set[str] = set()

        def walk(node: dict[str, Any]) -> None:
            for key, child in node.get("children", {}).items():
                keys.add(key)
                walk(child)

        walk(self._root)
        return keys

    def qualified_ids(self) -> Iterator[QualifiedId]:
        def walk(node: dict[str, Any], keys: tuple) -> Iterator[QualifiedId]:
            for entry in node.get("files", []):
                yield keys + (entry["id"],)
            for key, child in node.get("children", {}).items():
                yield from walk(child, keys + (key,))

        return walk(self._root, ())

    def to_dict(self) -> dict[str, Any]:
        return self._root


def _save(root: Path, output: dict[str, Any], scope: str) -> Path:
    scope_dir = root / METADATA_DIR / scope
    if scope_dir.is_dir():
        # Splits describe the previous index of this scope.
        for stale in scope_dir.glob("split_*.json"):
            stale.unlink()
    return save_dataset_index(root, output, metadata_scope=scope)


def _head(
    dataset: Mapping[str, Any],
    modality: Mapping[str, Any],
    addons: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    register_euler_loading_validator()
    return build_dataset_head(
        dataset=dict(dataset), modality=dict(modality), addons=dict(addons or {})
    )


def write_scope(
    root: str | os.PathLike[str],
    scope: str,
    *,
    dataset: Mapping[str, Any],
    modality: Mapping[str, Any],
    tree: IndexTree,
    addons: Optional[Mapping[str, Any]] = None,
    execution: Optional[Mapping[str, Any]] = None,
    hierarchy_separator: Optional[str] = HIERARCHY_SEPARATOR,
) -> Path:
    """Write one scope with a prebuilt index and return the index path.

    The dataset head is validated against the euler dataset contract, including
    the ``euler_loading`` addon rules, before anything is written. Split files a
    previous run left in the scope are removed.

    *hierarchy_separator* is declared in ``ds-crawler.json``
    (``indexing.hierarchy.separator``); every hierarchy key must contain it,
    as in ``scene:ai_001_001``. ``None`` declares none.
    """
    head = _head(dataset, modality, addons)
    indexing = None
    if hierarchy_separator is not None:
        unsplittable = sorted(k for k in tree.hierarchy_keys() if hierarchy_separator not in k)
        if unsplittable:
            raise ValueError(
                f"scope {scope!r}: hierarchy keys without the separator {hierarchy_separator!r}: "
                + ", ".join(unsplittable[:5])
            )
        indexing = {"hierarchy": {"separator": hierarchy_separator}}
    output = build_index_output(
        head=head,
        index=tree.to_dict(),
        indexing=indexing,
        generator=generator_info(),
        execution=dict(execution or {}),
    )
    return _save(Path(root), output, scope)


def write_regex_scope(
    root: str | os.PathLike[str],
    scope: str,
    *,
    files: Iterable[str],
    dataset: Mapping[str, Any],
    modality: Mapping[str, Any],
    indexing: Mapping[str, Any],
    addons: Optional[Mapping[str, Any]] = None,
    execution: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Index *files* with ds-crawler regexes and write the scope.

    Args:
        root: The prepared dataset's root.
        scope: Metadata scope to write.
        files: Paths relative to *root*. Those the regexes reject are left out.
            They need not exist yet.
        dataset, modality, addons: The dataset head's parts.
        indexing: A ds-crawler ``indexing`` block (``id``, ``hierarchy``,
            ``properties``, ``files``, ``constraints``).
        execution: Provenance recorded in the index.

    Returns the hydrated index output, e.g. for :func:`write_splits` via
    ``ds_crawler.collect_qualified_ids``. Duplicate IDs raise.
    """
    root = Path(root)
    head = _head(dataset, modality, addons)
    config = build_crawler_config(head=head, indexing=dict(indexing), metadata_scope=scope)
    config["head"] = head
    output = index_dataset_from_files(
        config, [root / rel for rel in files], base_path=root, strict=True
    )
    output["generator"] = generator_info()
    output["execution"] = dict(execution or {})
    _save(root, output, scope)
    return output


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
