"""MUSES: frame camera, clear-weather reference frames, lidar, GNSS/IMU,
semantic and panoptic ground truth, calibration and per-frame sensor timing.

See ``docs/datasets/muses.md`` for the prepared layout and the dataset facts it
is built on.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ds_crawler import collect_qualified_ids, get_files

from ...archives import selection_fingerprint
from ...artifacts import write_regex_scope, write_splits
from ...media import (
    IN_PLACE,
    MediaError,
    list_files,
    missing_files,
    place_extracted,
    resolve_media,
)
from ...recipe import Option, Outcome, Plan, PrepareContext, Recipe, register
from .sources import (
    CALIB,
    DOCS,
    META,
    ROOT,
    MusesSources,
    discover_sources,
    looks_like_muses,
)

logger = logging.getLogger(__name__)

LOADER_MODULE = "muses"
SPLITS = ("train", "val", "test")
WEATHERS = ("clear", "fog", "rain", "snow")
TIMES_OF_DAY = ("day", "night")


@dataclass(frozen=True)
class FrameModality:
    """One per-frame file type: ``muses/<folder>/…/<seq>_frame_<frame>_<suffix>``."""

    scope: str
    folder: str
    suffix: str
    name: str
    key: str
    function: str
    meta: dict = field(default_factory=dict)

    def indexing(self) -> dict[str, Any]:
        suffix = re.escape(self.suffix)
        folder = re.escape(self.folder)
        return {
            # Frame numbers are unique across the whole dataset, so the frame
            # alone is the ID; the split and condition form the hierarchy.
            "id": {
                "regex": rf"^{ROOT}/{folder}/[^/]+/[^/]+/[^/]+/REC\d+_frame_(\d+)_{suffix}$",
                "join_char": "+",
            },
            "hierarchy": {
                "regex": rf"^{ROOT}/{folder}/([^/]+)/([^/]+)/([^/]+)/REC\d+_frame_\d+_{suffix}$",
                "separator": ":",
            },
            "properties": {
                "basename": {"regex": rf"^(?P<sequence>REC\d+)_frame_(?P<frame>\d+)_{suffix}$"}
            },
            "files": {"extensions": [Path(self.suffix).suffix]},
            "constraints": {"flat_ids_unique": True},
        }


@dataclass(frozen=True)
class SharedFile:
    """A root-level file every sample shares: ``calib.json`` or ``meta.json``."""

    scope: str
    path: str
    name: str
    key: str
    function: str
    meta: dict = field(default_factory=dict)

    def indexing(self) -> dict[str, Any]:
        # No hierarchy: the file sits at the root and applies to every sample.
        return {
            "id": {"regex": rf"^({re.escape(self.path)})$", "override": self.scope},
            "constraints": {"flat_ids_unique": True},
        }


_LIDAR_META = {
    "representation": "point_cloud",
    "columns": ["x", "y", "z", "intensity", "ring", "timestamp"],
    "coordinate_unit": "meters",
    "frame": "lidar",
}

FRAME_MODALITIES = (
    FrameModality("rgb", "frame_camera", "frame_camera.png", "MUSES frame camera",
                  "rgb", "rgb", {"range": [0, 255]}),
    FrameModality("reference_rgb", "reference_frame", "reference_frame.png",
                  "MUSES clear-weather reference frames", "rgb", "reference_rgb",
                  {"range": [0, 255]}),
    FrameModality("lidar", "lidar", "lidar.bin", "MUSES lidar", "lidar_point_cloud",
                  "lidar_point_cloud", _LIDAR_META),
    FrameModality("gnss", "gnss", "gnss.txt", "MUSES GNSS/IMU records", "gnss", "gnss"),
    FrameModality("semantic_segmentation", "gt_semantic", "gt_labelIds.png",
                  "MUSES semantic labels (Cityscapes label IDs)", "semantic_segmentation",
                  "semantic_segmentation", {"skyclass": [23, 23, 23]}),
    FrameModality("semantic_segmentation_train_ids", "gt_semantic", "gt_labelTrainIds.png",
                  "MUSES semantic labels (Cityscapes train IDs)", "semantic_segmentation",
                  "semantic_segmentation", {"skyclass": [10, 10, 10]}),
    FrameModality("semantic_segmentation_color", "gt_semantic", "gt_labelColor.png",
                  "MUSES semantic labels (Cityscapes colours)", "semantic_segmentation_color",
                  "semantic_segmentation_color", {"skyclass": [70, 130, 180]}),
    FrameModality("panoptic_segmentation", "gt_panoptic", "gt_panoptic.png",
                  "MUSES panoptic labels", "panoptic_segmentation", "panoptic_segmentation"),
)
SHARED_FILES = (
    SharedFile("intrinsics", CALIB, "MUSES frame-camera intrinsics", "intrinsics",
               "read_intrinsics", {"camera_model": "pinhole", "dimensions": {"rows": 3, "columns": 3}}),
    SharedFile("camera_extrinsics", CALIB, "MUSES lidar-to-camera extrinsics",
               "camera_extrinsics", "read_extrinsics",
               {"source_frame": "lidar", "target_frame": "camera_optical",
                "direction": "target_from_source", "dimensions": {"rows": 4, "columns": 4}}),
    SharedFile("lidar_to_gnss", CALIB, "MUSES lidar-to-GNSS/IMU mount", "lidar_to_gnss",
               "read_lidar_to_gnss",
               {"source_frame": "lidar", "target_frame": "gnss",
                "direction": "target_from_source", "dimensions": {"rows": 4, "columns": 4}}),
    SharedFile("sensor_timing", META, "MUSES per-frame sensor timing", "sensor_timing",
               "sensor_timing"),
)
FOLDERS = tuple(dict.fromkeys(m.folder for m in FRAME_MODALITIES))


@register
class Muses(Recipe):
    name = "muses"
    title = "MUSES"
    summary = (
        "Frame camera, reference frames, lidar, GNSS/IMU, semantic and panoptic "
        "labels, calibration and per-frame sensor timing."
    )
    homepage = "https://muses.vision.ee.ethz.ch"
    data_license = (
        "ETH Zurich end user licence agreement: non-commercial research only, no "
        "redistribution; see muses/License.pdf in the download"
    )
    expected_inputs = (
        "the MUSES zip packages, e.g. frame_camera_trainvaltest.zip, "
        "lidar_trainvaltest.zip, gnss_trainvaltest.zip, "
        "reference_frame_trainvaltest.zip, gt_semantic_trainval.zip",
        "or what they extract to: muses/<type>/<split>/<weather>/<time_of_day>/..., "
        "muses/calib.json, muses/meta.json",
    )
    layout_version = 1
    options = (
        Option(
            "modalities",
            "Per-frame scopes to prepare; all = every one the source holds.",
            default=(),
            multiple=True,
            choices=tuple(m.scope for m in FRAME_MODALITIES),
        ),
        Option("splits", "Official splits to prepare.", default=(), multiple=True, choices=SPLITS),
        Option("weather", "Weather conditions to prepare.", default=(), multiple=True,
               choices=WEATHERS),
        Option("time_of_day", "Times of day to prepare.", default=(), multiple=True,
               choices=TIMES_OF_DAY),
    )

    def detect(self, source: Path) -> bool:
        try:
            return looks_like_muses(Path(source))
        except OSError:
            return False

    # -- plan -----------------------------------------------------------------

    def plan(self, ctx: PrepareContext) -> Plan:
        found = discover_sources(ctx.source, FOLDERS)
        if not found.folders:
            raise FileNotFoundError(
                f"no MUSES downloads in {ctx.source}; expected "
                + "; ".join(self.expected_inputs)
            )
        requested = ctx.options["modalities"]
        available = [m for m in FRAME_MODALITIES if m.folder in found.folders]
        if requested:
            missing = [s for s in requested if s not in {m.scope for m in available}]
            if missing:
                raise FileNotFoundError(
                    f"{ctx.source} holds no data for {', '.join(missing)}; it has "
                    f"{', '.join(sorted(found.folders))}"
                )
            selected = [m for m in available if m.scope in requested]
        else:
            selected = available
        folders = sorted({m.folder for m in selected})

        all_extracted = found.side_extracted and all(f in found.extracted for f in folders)
        all_archived = found.side_archive is not None and all(f in found.archives for f in folders)
        if not (found.side_extracted or found.side_archive is not None):
            raise FileNotFoundError(f"{ctx.source} holds neither {CALIB} nor {META}")
        # In place, each folder comes from wherever it is: extracted, or else
        # its package; elsewhere one strategy serves every folder.
        media = resolve_media(ctx, extracted=all_extracted, archives=all_archived)

        steps = [
            f"index {', '.join(m.scope for m in selected)} and the shared "
            f"{', '.join(s.scope for s in SHARED_FILES)}",
            f"bring media files into the output (media={media})",
            "write the official splits train/val/test per scope",
        ]
        return Plan(
            recipe=self.name,
            inputs=found.describe(),
            steps=steps,
            details={"sources": found, "selected": selected, "media": media},
        )

    # -- prepare --------------------------------------------------------------

    def prepare(self, ctx: PrepareContext, plan: Plan) -> Outcome:
        found: MusesSources = plan.details["sources"]
        selected: list[FrameModality] = plan.details["selected"]
        media: str = plan.details["media"]
        keep = self._condition_filter(ctx.options)
        execution = {
            "recipe": self.name,
            "layout_version": self.layout_version,
            "options": {k: list(v) for k, v in ctx.options.items()},
        }

        needed: set[str] = set()
        counts: dict[str, int] = {}
        splits: dict[str, list[str]] = {}
        split_sizes: dict[str, dict[str, int]] = {}
        for modality in selected:
            # gt_semantic holds three encodings per frame and gnss/ a format note;
            # each scope takes only its own files.
            ending = f"_{modality.suffix}"
            candidates = [
                path
                for path in self._candidates(ctx, found, modality.folder, media)
                if path.endswith(ending) and keep(path)
            ]
            if not candidates:
                # e.g. ground truth with splits=["test"]: there is none to index.
                logger.warning("%s: no files match the selected conditions; skipped", modality.scope)
                continue
            output = write_regex_scope(
                ctx.output,
                modality.scope,
                files=candidates,
                dataset={"id": f"muses_{modality.scope}", "name": modality.name},
                modality={"key": modality.key, "meta": dict(modality.meta)},
                addons=self._addon(modality.function),
                indexing=modality.indexing(),
                execution=execution,
            )
            files = get_files(output)
            needed.update(files)
            counts[modality.scope] = len(files)
            by_split: dict[str, list[tuple]] = defaultdict(list)
            for qid in collect_qualified_ids(output):
                by_split[qid[0]].append(qid)
            official = {name: sorted(by_split[name]) for name in SPLITS if by_split.get(name)}
            splits[modality.scope] = write_splits(ctx.output, modality.scope, official)
            split_sizes[modality.scope] = {name: len(ids) for name, ids in official.items()}
            logger.info("%s: %d files %s", modality.scope, len(files), split_sizes[modality.scope])

        for shared in SHARED_FILES:
            write_regex_scope(
                ctx.output,
                shared.scope,
                files=[shared.path],
                dataset={"id": f"muses_{shared.scope}", "name": shared.name},
                modality={"key": shared.key, "meta": dict(shared.meta)},
                addons=self._addon(shared.function),
                indexing=shared.indexing(),
                execution=execution,
            )
        needed.update({CALIB, META})

        used = self._materialize(ctx, found, sorted(needed), media)
        missing = missing_files(ctx.output, sorted(needed))
        if missing:
            raise MediaError(
                f"{len(missing)} indexed files are missing below {ctx.output} "
                f"(e.g. {missing[0]}); the packages may be incomplete"
            )
        summary = {
            "files": counts,
            "frames_per_split": split_sizes,
            "files_referenced": len(needed),
        }
        scopes = [m.scope for m in selected if m.scope in counts] + [s.scope for s in SHARED_FILES]
        return Outcome(media=used, scopes=scopes, splits=splits, summary=summary)

    @staticmethod
    def _addon(function: str) -> dict[str, Any]:
        return {"euler_loading": {"version": "1.0", "loader": LOADER_MODULE, "function": function}}

    @staticmethod
    def _condition_filter(options: dict[str, Any]):
        wanted = (
            set(options["splits"]) or set(SPLITS),
            set(options["weather"]) or set(WEATHERS),
            set(options["time_of_day"]) or set(TIMES_OF_DAY),
        )
        pattern = re.compile(rf"^{ROOT}/[^/]+/([^/]+)/([^/]+)/([^/]+)/")

        def keep(path: str) -> bool:
            match = pattern.match(path)
            return bool(match) and all(value in allowed for value, allowed in zip(match.groups(), wanted))

        return keep

    @staticmethod
    def _candidates(
        ctx: PrepareContext, found: MusesSources, folder: str, media: str
    ) -> list[str]:
        """Every file of *folder* the run will draw from, relative to the root."""
        extracted = folder in found.extracted
        if media in ("symlink", "hardlink", "copy") or (media == IN_PLACE and extracted):
            return list_files(ctx.source, f"{ROOT}/{folder}")
        return found.archive_files(folder)

    def _materialize(
        self, ctx: PrepareContext, found: MusesSources, needed: list[str], media: str
    ) -> str:
        docs_extracted = [d for d in DOCS if (ctx.source / d).is_file()]
        if media in ("symlink", "hardlink", "copy"):
            place_extracted(ctx, needed + docs_extracted, mode=media, top_level=[ROOT])
            return media
        if media == IN_PLACE:
            missing = set(missing_files(ctx.source, needed))
            if missing:
                self._extract(ctx, found, missing, ctx.source, with_docs=False)
            return IN_PLACE
        self._extract(ctx, found, set(needed), ctx.output, with_docs=True)
        return media

    @staticmethod
    def _extract(
        ctx: PrepareContext,
        found: MusesSources,
        files: Iterable[str],
        destination: Path,
        *,
        with_docs: bool,
    ) -> None:
        wanted = set(files)
        shared = {CALIB, META}
        # Every package carries the same calib.json and meta.json (and
        # documentation); take them from one package only.
        assignment = found.assign(wanted - shared)
        side = found.side_archive
        if side is not None:
            from_side = (wanted & shared) | (set(DOCS) if with_docs else set())
            assignment.setdefault(side, set()).update(from_side & set(found.members[side]))
        for archive, selection in sorted(assignment.items()):
            if not selection:
                continue
            ctx.extract(
                archive,
                destination=destination,
                select=selection.__contains__,
                fingerprint=selection_fingerprint(selection),
            )
