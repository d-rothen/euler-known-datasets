"""ACDC: anonymised images of drives in fog, at night, in rain and in snow, each
with a normal-condition reference image of the same place.

See ``docs/datasets/acdc.md`` for the layout and the dataset facts it relies on.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ds_crawler import collect_qualified_ids, get_files

from ...archives import selection_fingerprint
from ...artifacts import family_dataset, write_regex_scope, write_splits
from ...media import (
    IN_PLACE,
    MediaError,
    link_top_level,
    list_files,
    materialize_files,
    resolve_media,
)
from ...recipe import Option, Outcome, Plan, PrepareContext, Recipe, register
from .sources import CONDITIONS, DOCS, IMAGE, SPLITS, TYPE, AcdcSource, discover

logger = logging.getLogger(__name__)

LOADER_MODULE = "acdc"


@dataclass(frozen=True)
class ImageScope:
    """The adverse-condition images or their references."""

    scope: str
    reference: bool
    function: str
    meta: dict

    @property
    def suffix(self) -> str:
        return "rgb_ref_anon.png" if self.reference else "rgb_anon.png"

    def indexing(self) -> dict[str, Any]:
        split_dir = r"(?P<split>train|val|test)" + ("_ref" if self.reference else "")
        suffix = re.escape(self.suffix)
        return {
            # A reference image has its adverse image's sequence and frame, so
            # both scopes get the same hierarchy and IDs and join by them.
            "id": {
                "regex": rf"^(?:.*/)?{TYPE}/[^/]+/[^/]+/[^/]+/[^/]+_frame_(\d+)_{suffix}$",
                "join_char": "+",
            },
            "hierarchy": {
                "regex": rf"^(?:.*/)?{TYPE}/(?P<condition>[^/]+)/{split_dir}/(?P<sequence>[^/]+)/[^/]+_{suffix}$",
                "separator": ":",
            },
            "properties": {
                "basename": {"regex": rf"^(?P<sequence>.+?)_frame_(?P<frame>\d+)_{suffix}$"}
            },
            "files": {"extensions": [".png"]},
        }


#: Every image of the release is 1920x1080 RGB (checked 2026-11-10).
DIMENSIONS = {"height": 1080, "width": 1920, "channels": 3}
SCOPES = (
    ImageScope("rgb", False, "rgb", {"range": [0, 255], "anonymized": True, "dimensions": DIMENSIONS}),
    ImageScope("reference_rgb", True, "reference_rgb",
               {"range": [0, 255], "anonymized": True, "condition": "normal", "pixel_aligned": False,
                "dimensions": DIMENSIONS}),
)


@register
class Acdc(Recipe):
    name = "acdc"
    title = "ACDC"
    summary = (
        "Anonymised images in fog, at night, in rain and in snow, each with a "
        "normal-condition reference image of the same place; official splits."
    )
    homepage = "https://acdc.vision.ee.ethz.ch"
    data_license = (
        "ACDC licence agreement: non-commercial use only; see License.pdf in the download"
    )
    expected_inputs = (
        "rgb_anon_trainvaltest.zip as downloaded (registration required)",
        "or an archive of a whole ACDC copy holding rgb_anon/ (e.g. ACDC.tar.gz)",
        "or the extracted rgb_anon/<condition>/<split>/<sequence>/ tree",
    )
    layout_version = 1
    options = (
        Option("modalities", "Scopes to prepare: rgb (adverse conditions) and reference_rgb.",
               default=(), multiple=True, choices=tuple(s.scope for s in SCOPES)),
        Option("conditions", "Adverse conditions to prepare.", default=(), multiple=True, choices=CONDITIONS),
        Option("splits", "Official splits to prepare.", default=(), multiple=True, choices=SPLITS),
    )

    def detect(self, source: Path) -> bool:
        try:
            found = discover(Path(source))
        except OSError:
            return False
        return found.archive is not None or found.extracted is not None

    def plan(self, ctx: PrepareContext) -> Plan:
        found = discover(ctx.source)
        if found.archive is None and found.extracted is None:
            raise FileNotFoundError(
                f"no ACDC images in {ctx.source}; expected " + "; ".join(self.expected_inputs)
            )
        requested = set(ctx.options["modalities"])
        scopes = [s for s in SCOPES if not requested or s.scope in requested]
        conditions = tuple(ctx.options["conditions"]) or CONDITIONS
        splits = tuple(ctx.options["splits"]) or SPLITS
        media = resolve_media(ctx, extracted=found.extracted is not None, archives=found.archive is not None)
        steps = [
            f"prepare {', '.join(s.scope for s in scopes)} for {', '.join(conditions)} "
            f"({', '.join(splits)})",
            f"bring the images into the output (media={media})"
            + (": one pass through the archive, keeping only the selected images" if media == "extract" else ""),
            "write regex-indexed scopes and the official splits",
        ]
        return Plan(recipe=self.name, inputs=found.describe(), steps=steps,
                    details={"found": found, "scopes": scopes, "conditions": conditions,
                             "splits": splits, "media": media})

    def prepare(self, ctx: PrepareContext, plan: Plan) -> Outcome:
        found: AcdcSource = plan.details["found"]
        scopes: list[ImageScope] = plan.details["scopes"]
        conditions, splits = set(plan.details["conditions"]), set(plan.details["splits"])
        references = {s.reference for s in scopes}

        def keep(path: str) -> bool:
            match = IMAGE.match(path)
            return bool(match) and match.group("condition") in conditions and \
                match.group("split") in splits and bool(match.group("ref")) in references

        media, root, prefix = self._materialize(ctx, found, plan.details["media"], keep, scopes, conditions, splits)
        images = [path for path in list_files(root, f"{prefix}{TYPE}") if keep(path)]
        if not images:
            raise MediaError(f"no ACDC images matching the selection below {root / prefix / TYPE}")

        attributes = {"source": "ACDC", "conditions": sorted(conditions), "splits": sorted(splits)}
        execution = {"recipe": self.name, "layout_version": self.layout_version,
                     "options": {k: list(v) for k, v in ctx.options.items()}}
        counts: dict[str, dict[str, int]] = {}
        split_names: dict[str, list[str]] = {}
        for scope in scopes:
            files = [p for p in images if bool(IMAGE.match(p).group("ref")) == scope.reference]
            output = write_regex_scope(
                ctx.output,
                scope.scope,
                files=files,
                dataset=family_dataset(self.name, attributes),
                modality={"key": "rgb", "meta": dict(scope.meta)},
                addons={"euler_loading": {"version": "1.0", "loader": LOADER_MODULE, "function": scope.function}},
                indexing=scope.indexing(),
                execution=execution,
            )
            if len(get_files(output)) != len(files):
                raise MediaError(f"{scope.scope}: indexed {len(get_files(output))} of {len(files)} images")
            by_split: dict[str, list[tuple]] = defaultdict(list)
            for qid in collect_qualified_ids(output):
                by_split[qid[1].split(":", 1)[1]].append(qid)
            split_names[scope.scope] = write_splits(
                ctx.output, scope.scope, {name: sorted(by_split[name]) for name in SPLITS if by_split.get(name)}
            )
            per = Counter(f"{IMAGE.match(p).group('condition')}/{IMAGE.match(p).group('split')}" for p in files)
            counts[scope.scope] = dict(sorted(per.items()))
            logger.info("%s: %d images", scope.scope, len(files))

        summary: dict[str, Any] = {"images": counts}
        if {s.scope for s in scopes} == {"rgb", "reference_rgb"}:
            adverse = {self._key(p) for p in images if not IMAGE.match(p).group("ref")}
            reference = {self._key(p) for p in images if IMAGE.match(p).group("ref")}
            summary["pairs"] = len(adverse & reference)
            summary["unpaired"] = {"rgb": len(adverse - reference), "reference_rgb": len(reference - adverse)}
        return Outcome(media=media, scopes=[s.scope for s in scopes], splits=split_names, summary=summary)

    @staticmethod
    def _key(path: str) -> tuple:
        match = IMAGE.match(path)
        return match.group("condition", "split", "sequence", "frame")

    @staticmethod
    def _materialize(ctx, found: AcdcSource, media: str, keep, scopes, conditions, splits) -> tuple[str, Path, str]:
        """Bring the selected images into place; return the media mode, the root
        the index is relative to, and the path prefix of ``rgb_anon/`` below it."""
        if media in ("symlink", "hardlink", "copy") or (media == IN_PLACE and found.extracted is not None):
            base = found.extracted
            source_root = ctx.source / base if base else ctx.source
            if media == IN_PLACE:
                return media, ctx.source, f"{base}/" if base else ""
            docs = [d for d in DOCS if (source_root / d).is_file()]
            if media == "symlink":
                link_top_level(source_root, ctx.output, [TYPE, *docs])
            else:
                selected = [p for p in list_files(source_root, TYPE) if keep(p)]
                materialize_files(source_root, ctx.output, selected + docs, media)
            return media, ctx.output, ""

        if found.archive is None:
            raise MediaError(f"no ACDC archive to extract from in {ctx.source}")
        destination = ctx.source if media == IN_PLACE else ctx.output
        prefix = found.archive_prefix
        docs = {f"{prefix}{d}" for d in DOCS}
        # The selection is described rather than listed: listing a tarball
        # means reading it to its end, and this reads it once.
        fingerprint = selection_fingerprint([
            f"conditions={sorted(conditions)}", f"splits={sorted(splits)}",
            f"scopes={sorted(s.scope for s in scopes)}", f"prefix={prefix}",
        ])
        ctx.extract(found.archive, destination=destination,
                    select=lambda name: keep(name) or name in docs, fingerprint=fingerprint)
        if not (destination / f"{prefix}{TYPE}").is_dir():
            raise MediaError(f"{found.archive.name} held no images matching the selection")
        return media, destination, prefix
