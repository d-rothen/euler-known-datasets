"""Recipes: the dataset-specific knowledge behind ``prepare``.

A recipe knows what a dataset's downloads look like, how to turn them into an
euler-ready directory, and which ds-crawler scopes and splits that directory
offers. Built-in recipes live in :mod:`euler_known_datasets.datasets`; other
packages can contribute recipes through the ``euler_known_datasets.recipes``
entry-point group (each entry point names a :class:`Recipe` subclass).
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Mapping, Optional, Sequence

from .archives import ExtractionLedger, ExtractionStats, archive_label, extract_archive

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "euler_known_datasets.recipes"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


@dataclass(frozen=True)
class Option:
    """A recipe option, exposed as a keyword of ``prepare`` and a CLI flag."""

    name: str
    help: str
    default: Any = None
    type: Callable[[Any], Any] = str
    multiple: bool = False
    choices: Optional[tuple] = None

    @property
    def flag(self) -> str:
        return "--" + self.name.replace("_", "-")

    def _convert(self, value: Any) -> Any:
        converted = _parse_bool(value) if self.type is bool else self.type(value)
        if self.choices is not None and converted not in self.choices:
            raise ValueError(
                f"{self.name}: {converted!r} is not one of {', '.join(map(str, self.choices))}"
            )
        return converted

    def coerce(self, value: Any) -> Any:
        if not self.multiple:
            return None if value is None else self._convert(value)
        if value is None:
            return ()
        if isinstance(value, str):
            value = [part for part in value.split(",") if part.strip()]
        items = []
        for item in value:
            if isinstance(item, str):
                items.extend(part.strip() for part in item.split(",") if part.strip())
            else:
                items.append(item)
        converted = tuple(self._convert(item) for item in items)
        if len(set(converted)) != len(converted):
            raise ValueError(f"{self.name}: duplicate values in {list(converted)}")
        return converted


@dataclass
class Plan:
    """What ``prepare`` found in the source and is about to do.

    ``inputs`` and ``steps`` are JSON-ready and shown by ``--dry-run``;
    ``details`` is private to the recipe and carries what ``prepare`` needs.
    """

    recipe: str
    inputs: list[dict[str, Any]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"recipe": self.recipe, "inputs": self.inputs, "steps": self.steps}


@dataclass
class Outcome:
    """What a recipe's ``prepare`` produced, recorded in the receipt."""

    media: str
    scopes: list[str]
    splits: dict[str, list[str]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrepareContext:
    """Everything a recipe's ``plan``/``prepare`` needs to know about the run."""

    source: Path
    output: Path
    options: dict[str, Any]
    media: str
    state_dir: Path

    @property
    def in_place(self) -> bool:
        return self.source.resolve() == self.output.resolve()

    def extract(
        self,
        archive: Path,
        *,
        destination: Optional[Path] = None,
        select: Optional[Callable[[str], bool]] = None,
        fingerprint: str = "all",
        inner: Sequence[str] = (),
    ) -> Optional[ExtractionStats]:
        """Extract *archive* (selected members only) unless already done.

        *fingerprint* identifies the selection; an archive already extracted
        with the same fingerprint into the same destination is skipped.
        *inner* extracts a zip nested in the zip *archive* instead (see
        :func:`~euler_known_datasets.archives.open_nested_zip`).
        Returns ``None`` when skipped.
        """
        destination = destination or self.output
        inner = tuple(inner)
        label = archive_label(archive, inner)
        ledger = ExtractionLedger(self.state_dir)
        if ledger.is_complete(archive, destination, fingerprint, inner):
            logger.info("%s: already extracted into %s, skipping", label, destination)
            return None
        logger.info("extracting %s into %s", label, destination)
        stats = extract_archive(archive, destination, select=select, inner=inner)
        ledger.mark_complete(archive, destination, fingerprint, stats, inner)
        logger.info(
            "%s: %d files extracted (%.2f GB), %d already present, %d not needed, %.0fs",
            label,
            stats.extracted,
            stats.bytes_written / 1e9,
            stats.skipped_existing,
            stats.skipped_unselected,
            stats.seconds,
        )
        return stats


class Recipe(abc.ABC):
    """Base class of a dataset recipe. Subclass, fill in the class attributes,
    implement :meth:`plan` and :meth:`prepare`, and decorate with :func:`register`.
    """

    #: Registry key and CLI name, e.g. ``"nuscenes"``.
    name: ClassVar[str]
    #: Human-readable dataset name.
    title: ClassVar[str]
    #: One line on what the prepared output contains.
    summary: ClassVar[str]
    homepage: ClassVar[str] = ""
    #: The dataset's own licence. Recipes never redistribute data.
    data_license: ClassVar[str] = ""
    #: The downloads the recipe accepts, for ``info`` and error messages.
    expected_inputs: ClassVar[tuple[str, ...]] = ()
    #: Bump whenever the prepared output changes shape; recorded in the receipt.
    layout_version: ClassVar[int] = 1
    options: ClassVar[tuple[Option, ...]] = ()

    def detect(self, source: Path) -> bool:
        """Whether *source* looks like this dataset's downloads."""
        return False

    def resolve_options(self, values: Mapping[str, Any]) -> dict[str, Any]:
        remaining = dict(values)
        resolved = {}
        for option in self.options:
            resolved[option.name] = option.coerce(remaining.pop(option.name, option.default))
        if remaining:
            known = ", ".join(option.name for option in self.options) or "none"
            raise TypeError(
                f"unknown option(s) for recipe {self.name!r}: {', '.join(sorted(remaining))} "
                f"(known: {known})"
            )
        return resolved

    @abc.abstractmethod
    def plan(self, ctx: PrepareContext) -> Plan:
        """Inspect the source and describe the run; raise if it cannot succeed."""

    @abc.abstractmethod
    def prepare(self, ctx: PrepareContext, plan: Plan) -> Outcome:
        """Produce the euler-ready output described by *plan*."""


_REGISTRY: dict[str, Recipe] = {}
_loaded = False


def register(cls: type[Recipe]) -> type[Recipe]:
    """Class decorator adding a recipe to the registry."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} must define a non-empty 'name'")
    existing = _REGISTRY.get(name)
    if existing is not None and type(existing) is not cls:
        raise ValueError(f"recipe {name!r} is already registered by {type(existing).__name__}")
    _REGISTRY[name] = cls()
    return cls


def _entry_points() -> Iterable[Any]:
    from importlib import metadata

    eps = metadata.entry_points()
    if hasattr(eps, "select"):
        return eps.select(group=ENTRY_POINT_GROUP)
    return eps.get(ENTRY_POINT_GROUP, ())  # Python 3.9


def _load_recipes() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import datasets  # noqa: F401  (registers the built-in recipes)

    for entry_point in _entry_points():
        try:
            recipe_cls = entry_point.load()
            if recipe_cls.name not in _REGISTRY:
                register(recipe_cls)
        except Exception:  # a broken plugin must not hide the built-in recipes
            logger.exception("could not load recipe entry point %s", entry_point.name)


def get_recipe(name: str) -> Recipe:
    _load_recipes()
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"unknown dataset recipe {name!r} (known: {known})") from None


def list_recipes() -> list[Recipe]:
    _load_recipes()
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def detect_recipes(source: str | Path) -> list[Recipe]:
    """Recipes whose downloads *source* appears to contain."""
    path = Path(source)
    return [recipe for recipe in list_recipes() if recipe.detect(path)]
