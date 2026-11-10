"""``verify``: check that a prepared output is loadable as promised."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from ds_crawler import (
    get_files,
    index_dataset_from_path,
    load_dataset_split,
    validate_dataset,
)

from ._contract import register_euler_loading_validator
from ._fs import DirectoryCache, missing_files
from .prepare import RECEIPT_FILENAME, read_receipt

_MAX_LISTED = 10


@dataclass
class ScopeReport:
    scope: str
    dataset_id: str = ""
    modality: str = ""
    loader: str = ""
    files: int = 0
    missing: int = 0
    missing_examples: list[str] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class VerifyReport:
    output: str
    recipe: str = ""
    status: str = ""
    scopes: list[ScopeReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and all(not scope.errors for scope in self.scopes)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ok"] = self.ok
        return result

    def format_text(self) -> str:
        lines = [f"{self.output}  [{'ok' if self.ok else 'FAILED'}]"]
        lines.append(f"  recipe {self.recipe}, status {self.status}")
        for message in self.errors:
            lines.append(f"  error    {message}")
        for message in self.warnings:
            lines.append(f"  warning  {message}")
        for scope in self.scopes:
            lines.append(f"  scope {scope.scope}  [{'ok' if not scope.errors else 'FAILED'}]")
            lines.append(
                f"    {scope.dataset_id}, modality {scope.modality}, loader {scope.loader or '-'}"
            )
            lines.append(f"    {scope.files} files indexed, {scope.missing} missing")
            for example in scope.missing_examples:
                lines.append(f"      missing {example}")
            if scope.splits:
                listed = ", ".join(f"{name} ({count})" for name, count in scope.splits.items())
                lines.append(f"    splits {listed}")
            for message in scope.errors:
                lines.append(f"    error    {message}")
            for message in scope.warnings:
                lines.append(f"    warning  {message}")
        return "\n".join(lines)


def euler_loading_inventory() -> Optional[dict[str, set[str]]]:
    """``{loader module: {functions}}`` from the installed euler-loading.

    Read from its checked-in ``loaders.json`` without importing euler-loading
    (or torch). ``None`` when euler-loading is not installed.
    """
    spec = importlib.util.find_spec("euler_loading")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        path = Path(location) / "loaders" / "generate" / "loaders.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                loader["name"]: {modality["function"] for modality in loader.get("modalities", [])}
                for loader in data.get("supportedLoaders", [])
            }
    return None


def verify(
    output: Union[str, Path], *, check_files: bool = True, check_loaders: bool = True
) -> VerifyReport:
    """Check every scope a prepared output declares.

    For each scope: the artifact set validates, the index loads through
    ds-crawler, every indexed file exists, and every split loads. With
    *check_loaders*, also report whether the installed euler-loading provides
    the loader each dataset head names.
    """
    root = Path(output).expanduser().resolve()
    report = VerifyReport(output=str(root))
    try:
        receipt = read_receipt(root)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        report.errors.append(str(exc))
        return report
    if receipt is None:
        report.errors.append(f"no {RECEIPT_FILENAME}; not prepared by euler-known-datasets")
        return report
    report.recipe = receipt.get("recipe", {}).get("name", "")
    report.status = receipt.get("status", "")
    if report.status != "complete":
        report.errors.append("prepare did not finish; re-run it to resume")

    listings = DirectoryCache()
    inventory = None
    if check_loaders:
        inventory = euler_loading_inventory()
        if inventory is None or not register_euler_loading_validator():
            report.warnings.append(
                "euler-loading is not installed; loader names and its head rules not checked"
            )

    for scope in receipt.get("scopes", []):
        scope_report = ScopeReport(scope=scope)
        report.scopes.append(scope_report)
        try:
            validate_dataset(root, metadata_scope=scope)
            index = index_dataset_from_path(root, metadata_scope=scope)
        except Exception as exc:  # noqa: BLE001
            scope_report.errors.append(f"index does not load: {exc}")
            continue
        head = index.get("head", {})
        scope_report.dataset_id = head.get("dataset", {}).get("id", "")
        scope_report.modality = head.get("modality", {}).get("key", "")
        addon = head.get("addons", {}).get("euler_loading", {})
        if addon.get("loader") and addon.get("function"):
            scope_report.loader = f"{addon['loader']}.{addon['function']}"
            if inventory is not None and addon["function"] not in inventory.get(
                addon["loader"], set()
            ):
                scope_report.warnings.append(
                    f"the installed euler-loading has no {scope_report.loader}; pass an "
                    "explicit Modality(loader=...) until it does"
                )

        paths = get_files(index)
        scope_report.files = len(paths)
        if check_files:
            missing = missing_files(root, paths, listings)
            scope_report.missing = len(missing)
            scope_report.missing_examples = missing[:_MAX_LISTED]
            if missing:
                scope_report.errors.append(f"{len(missing)} indexed files are missing")

        for split in receipt.get("splits", {}).get(scope, []):
            try:
                split_output = load_dataset_split(root, split, metadata_scope=scope)
                scope_report.splits[split] = len(get_files(split_output))
            except Exception as exc:  # noqa: BLE001
                scope_report.errors.append(f"split {split!r} does not load: {exc}")
    return report
