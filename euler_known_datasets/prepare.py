"""``prepare``: downloads in, euler-ready dataset out."""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from ._jsonio import load_json, write_json_atomic
from ._version import generator_info
from .media import MEDIA_MODES
from .recipe import Plan, PrepareContext, Recipe, get_recipe

logger = logging.getLogger(__name__)

RECEIPT_FILENAME = "euler-known-datasets.json"
STATE_DIRNAME = ".euler-known-datasets"
RECEIPT_CONTRACT = {"kind": "euler_known_datasets_receipt", "version": "1.0"}


class PrepareError(RuntimeError):
    """The output cannot be prepared as requested."""


@dataclass
class PrepareResult:
    recipe: str
    source: Path
    output: Path
    plan: Plan
    receipt: Optional[dict[str, Any]]

    @property
    def dry_run(self) -> bool:
        return self.receipt is None


def read_receipt(output: Union[str, Path]) -> Optional[dict[str, Any]]:
    path = Path(output) / RECEIPT_FILENAME
    if not path.is_file():
        return None
    receipt = load_json(path)
    if receipt.get("contract", {}).get("kind") != RECEIPT_CONTRACT["kind"]:
        raise PrepareError(f"{path} is not an euler-known-datasets receipt")
    return receipt


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _check_output(recipe: Recipe, source: Path, output: Path, force: bool) -> None:
    if output.exists() and not output.is_dir():
        raise PrepareError(f"output {output} exists and is not a directory")
    if source == output or not output.exists():
        return
    existing = read_receipt(output)
    if existing is not None:
        previous = existing.get("recipe", {}).get("name")
        if previous != recipe.name:
            raise PrepareError(
                f"{output} was prepared by recipe {previous!r}, not {recipe.name!r}"
            )
        return
    if (output / STATE_DIRNAME).is_dir():
        return  # an earlier run of ours was interrupted; resume it
    if any(output.iterdir()) and not force:
        raise PrepareError(
            f"{output} is not empty and was not prepared by euler-known-datasets; "
            "choose an empty directory or pass force=True (--force)"
        )


def _context(
    recipe: Recipe,
    source: Union[str, Path],
    output: Union[str, Path],
    media: str,
    options: dict[str, Any],
) -> PrepareContext:
    if media not in MEDIA_MODES:
        raise ValueError(f"media must be one of {', '.join(MEDIA_MODES)}, got {media!r}")
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"source directory {source_path} does not exist")
    output_path = Path(output).expanduser().resolve()
    return PrepareContext(
        source=source_path,
        output=output_path,
        options=recipe.resolve_options(options),
        media=media,
        state_dir=output_path / STATE_DIRNAME,
    )


def plan(
    recipe: Union[str, Recipe],
    source: Union[str, Path],
    output: Union[str, Path],
    *,
    media: str = "auto",
    **options: Any,
) -> Plan:
    """Inspect *source* and describe what :func:`prepare` would do."""
    recipe = get_recipe(recipe) if isinstance(recipe, str) else recipe
    return recipe.plan(_context(recipe, source, output, media, options))


def prepare(
    recipe: Union[str, Recipe],
    source: Union[str, Path],
    output: Union[str, Path],
    *,
    media: str = "auto",
    dry_run: bool = False,
    force: bool = False,
    **options: Any,
) -> PrepareResult:
    """Turn the downloads in *source* into an euler-ready dataset in *output*.

    Args:
        recipe: A recipe name such as ``"nuscenes"``, or a :class:`Recipe`.
        source: Directory holding the downloaded archives and/or their
            extracted contents.
        output: Directory to prepare. ``output == source`` prepares in place.
        media: How media files reach the output; see :mod:`.media`.
        dry_run: Only inspect the source and return the plan.
        force: Allow a non-empty output that euler-known-datasets did not create.
        **options: Recipe options, e.g. ``cameras=["CAM_FRONT_LEFT"]``.

    Re-running ``prepare`` with the same arguments resumes an interrupted run:
    finished archive extractions are skipped and present files are kept.
    """
    recipe = get_recipe(recipe) if isinstance(recipe, str) else recipe
    ctx = _context(recipe, source, output, media, options)
    the_plan = recipe.plan(ctx)
    if dry_run:
        return PrepareResult(recipe.name, ctx.source, ctx.output, the_plan, None)

    _check_output(recipe, ctx.source, ctx.output, force)
    ctx.output.mkdir(parents=True, exist_ok=True)
    ctx.state_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    receipt: dict[str, Any] = {
        "contract": dict(RECEIPT_CONTRACT),
        "status": "incomplete",
        "generator": generator_info(),
        "recipe": {
            "name": recipe.name,
            "title": recipe.title,
            "layout_version": recipe.layout_version,
        },
        "data_license": recipe.data_license,
        "started_at": started,
        "source": {"path": str(ctx.source), "inputs": the_plan.inputs},
        "options": _jsonable(ctx.options),
        "media": {"requested": media},
    }
    write_json_atomic(ctx.output / RECEIPT_FILENAME, receipt)

    outcome = recipe.prepare(ctx, the_plan)

    receipt.update(
        status="complete",
        finished_at=_now(),
        media={"requested": media, "used": outcome.media},
        scopes=list(outcome.scopes),
        splits={scope: list(names) for scope, names in outcome.splits.items()},
        summary=_jsonable(outcome.summary),
    )
    write_json_atomic(ctx.output / RECEIPT_FILENAME, receipt)
    logger.info("prepared %s in %s", recipe.title, ctx.output)
    return PrepareResult(recipe.name, ctx.source, ctx.output, the_plan, receipt)
