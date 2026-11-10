"""KITTI-CARLA: two cameras' RGB, depth and semantic segmentation, the lidar
sweep of every frame, intrinsics, lidar-to-camera extrinsics and lidar poses.

See ``docs/datasets/kitti_carla.md`` for the prepared layout and the dataset
facts it relies on.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from ...archives import archive_members, selection_fingerprint
from ...artifacts import (
    IndexTree,
    family_dataset,
    file_entry,
    write_regex_scope,
    write_scope,
)
from ...media import (
    IN_PLACE,
    MediaError,
    list_files,
    missing_files,
    place_extracted,
    resolve_media,
)
from ...recipe import Option, Outcome, Plan, PrepareContext, Recipe, register
from .scopes import (
    CAMERAS,
    FILE_SCOPES,
    IMAGE_SCOPES,
    LIDAR_FUNCTION,
    LIDAR_MODALITY,
    LIDAR_PATTERN,
    LIDAR_SCOPE,
    MODALITIES,
    addon,
)
from .sources import TOWNS, TownSource, discover_towns, misplaced_hint

logger = logging.getLogger(__name__)


def _hierarchy(town: str, camera: str) -> tuple[str, str]:
    return (f"town:{town}", f"camera:{camera}")


@register
class KittiCarla(Recipe):
    name = "kitti_carla"
    title = "KITTI-CARLA"
    summary = (
        "RGB, depth and semantic segmentation of both cameras, the lidar sweep of "
        "every frame, intrinsics, lidar-to-camera extrinsics and lidar poses."
    )
    homepage = "https://arxiv.org/abs/2109.00892"
    data_license = (
        "Not stated in the download (readme.txt); check the authors' terms before "
        "redistributing"
    )
    expected_inputs = (
        "Town01.zip ... Town07.zip, as unzipped from the KITTI-CARLA.zip bundle",
        "or the extracted TownNN/generated/ directories",
    )
    layout_version = 1
    options = (
        Option(
            "towns",
            "Towns to prepare; all = every town found in the source.",
            default=(),
            multiple=True,
            choices=TOWNS,
        ),
        Option("cameras", "Cameras to prepare.", default=CAMERAS, multiple=True, choices=CAMERAS),
        Option(
            "modalities",
            "Per-frame modalities to prepare; intrinsics, extrinsics and lidar poses "
            "always come along.",
            default=MODALITIES,
            multiple=True,
            choices=MODALITIES,
        ),
        Option(
            "frame_step",
            "Keep the frames whose number is a multiple of N; 1 keeps all 5000 per town.",
            default=1,
            type=int,
        ),
    )

    def detect(self, source: Path) -> bool:
        try:
            return bool(discover_towns(Path(source)))
        except OSError:
            return False

    # -- plan -----------------------------------------------------------------

    def plan(self, ctx: PrepareContext) -> Plan:
        found = discover_towns(ctx.source)
        if not found:
            raise FileNotFoundError(misplaced_hint(ctx.source))
        towns = ctx.options["towns"] or tuple(sorted(found))
        missing = [town for town in towns if town not in found]
        if missing:
            raise FileNotFoundError(
                f"{', '.join(missing)} not in {ctx.source}; found {', '.join(sorted(found))}"
            )
        if ctx.options["frame_step"] < 1:
            raise ValueError(f"frame_step must be at least 1, got {ctx.options['frame_step']}")
        if not ctx.options["modalities"]:
            raise ValueError("modalities must name at least one of " + ", ".join(MODALITIES))
        sources = {town: found[town] for town in towns}
        modalities = ", ".join(ctx.options["modalities"])
        cameras = ", ".join(ctx.options["cameras"])
        step = ctx.options["frame_step"]
        steps = [
            f"index {modalities} of cameras {cameras} in {', '.join(towns)}"
            + (f", frames whose number is a multiple of {step}" if step > 1 else ""),
            "in place; extract only missing files"
            if ctx.in_place
            else f"bring the indexed files into the output (media={ctx.media})",
            "write ds-crawler scopes "
            + ", ".join([*ctx.options["modalities"], *FILE_SCOPES]),
        ]
        inputs = [item for source in sources.values() for item in source.describe()]
        return Plan(recipe=self.name, inputs=inputs, steps=steps, details={"towns": sources})

    # -- prepare --------------------------------------------------------------

    def prepare(self, ctx: PrepareContext, plan: Plan) -> Outcome:
        towns: dict[str, TownSource] = plan.details["towns"]
        selection = {town: self._select(ctx, source) for town, source in towns.items()}
        needed = {
            town: sorted({path for paths in chosen.values() for path in paths})
            for town, chosen in selection.items()
        }
        media = self._materialize(ctx, towns, needed)
        scopes = self._write_scopes(ctx, towns, selection)
        summary = {
            "towns": {
                town: {scope: len(paths) for scope, paths in chosen.items()}
                for town, chosen in selection.items()
            },
            "cameras": list(ctx.options["cameras"]),
            "frame_step": ctx.options["frame_step"],
            "files_referenced": sum(len(paths) for paths in needed.values()),
        }
        return Outcome(media=media, scopes=scopes, splits={}, summary=summary)

    def _members(self, ctx: PrepareContext, source: TownSource) -> list[str]:
        """The town's files relative to the source: the archive is authoritative."""
        if source.archive is not None:
            names = [
                name for name in archive_members(source.archive) if name.startswith(f"{source.town}/")
            ]
        else:
            names = list_files(ctx.source, source.town)
        if not names:
            raise FileNotFoundError(f"{source.town} holds no files under {source.town}/generated/")
        return names

    def _select(self, ctx: PrepareContext, source: TownSource) -> dict[str, list[str]]:
        """Each scope's files for one town, after the town/camera/frame options."""
        cameras = set(ctx.options["cameras"])
        modalities = set(ctx.options["modalities"])
        step = ctx.options["frame_step"]
        chosen: dict[str, list[str]] = defaultdict(list)
        for name in self._members(ctx, source):
            for scope, spec in IMAGE_SCOPES.items():
                if scope not in modalities:
                    continue
                match = spec.pattern.match(name)
                if match and match["camera"] in cameras and int(match["frame"]) % step == 0:
                    chosen[scope].append(name)
            match = LIDAR_PATTERN.match(name)
            if match and LIDAR_SCOPE in modalities and int(match["frame"]) % step == 0:
                chosen[LIDAR_SCOPE].append(name)
            for scope, spec in FILE_SCOPES.items():
                match = spec.pattern.match(name)
                if match and (not spec.per_camera or match["camera"] in cameras):
                    chosen[scope].append(name)
        for scope in [*modalities, *FILE_SCOPES]:
            if not chosen.get(scope):
                raise FileNotFoundError(f"{source.town} has no files for scope {scope!r}")
        return {scope: sorted(paths) for scope, paths in chosen.items()}

    def _materialize(
        self, ctx: PrepareContext, towns: dict[str, TownSource], needed: dict[str, list[str]]
    ) -> str:
        extracted = all(
            source.extracted is not None and not missing_files(ctx.source, needed[town])
            for town, source in towns.items()
        )
        archives = all(source.archive is not None for source in towns.values())
        mode = resolve_media(ctx, extracted=extracted, archives=archives)
        logger.info("media: %s", mode)
        if mode == IN_PLACE:
            for town, source in towns.items():
                missing = missing_files(ctx.source, needed[town])
                if not missing:
                    continue
                if source.archive is None:
                    raise MediaError(
                        f"{town}: {len(missing)} files are missing (e.g. {missing[0]}) and "
                        f"there is no {town}.zip to extract them from"
                    )
                self._extract(ctx, source, missing, ctx.source)
            root = ctx.source
        elif mode == "extract":
            for town, source in towns.items():
                self._extract(ctx, source, needed[town], ctx.output)
            root = ctx.output
        else:
            relpaths = [path for paths in needed.values() for path in paths]
            place_extracted(ctx, relpaths, mode=mode, top_level=list(towns))
            root = ctx.output
        missing = missing_files(root, [path for paths in needed.values() for path in paths])
        if missing:
            raise MediaError(
                f"{len(missing)} indexed files are missing below {root} (e.g. {missing[0]}); "
                "the archive may be incomplete"
            )
        return mode

    @staticmethod
    def _extract(
        ctx: PrepareContext, source: TownSource, files: list[str], destination: Path
    ) -> None:
        selected = set(files)
        ctx.extract(
            source.archive,
            destination=destination,
            select=selected.__contains__,
            fingerprint=selection_fingerprint(selected),
        )

    def _write_scopes(
        self,
        ctx: PrepareContext,
        towns: dict[str, TownSource],
        selection: dict[str, dict[str, list[str]]],
    ) -> list[str]:
        root = ctx.output
        options = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in ctx.options.items()
        }
        execution = {"recipe": self.name, "layout_version": self.layout_version, "options": options}
        attributes: dict[str, Any] = {
            "source": "KITTI-CARLA",
            "towns": list(towns),
            "cameras": list(ctx.options["cameras"]),
            "frame_step": ctx.options["frame_step"],
        }

        def files_of(scope: str) -> list[str]:
            return [path for chosen in selection.values() for path in chosen.get(scope, [])]

        written = []
        for scope, spec in IMAGE_SCOPES.items():
            if scope not in ctx.options["modalities"]:
                continue
            write_regex_scope(
                root,
                scope,
                files=files_of(scope),
                dataset=family_dataset(self.name, attributes),
                modality=spec.modality,
                indexing=spec.indexing,
                addons=addon(spec.function),
                execution=execution,
            )
            written.append(scope)

        if LIDAR_SCOPE in ctx.options["modalities"]:
            # A sweep has no camera, but samples are per camera: index each sweep
            # under every prepared camera, so it joins that camera's images by ID
            # and takes the camera's extrinsics from the hierarchy.
            tree = IndexTree()
            for town in towns:
                frames = [
                    (LIDAR_PATTERN.match(path)["frame"], path)
                    for path in selection[town].get(LIDAR_SCOPE, [])
                ]
                for camera in ctx.options["cameras"]:
                    for frame, path in frames:
                        tree.add(_hierarchy(town, camera), file_entry(path, f"frame-{frame}"))
            write_scope(
                root,
                LIDAR_SCOPE,
                dataset=family_dataset(self.name, attributes),
                modality=LIDAR_MODALITY,
                tree=tree,
                addons=addon(LIDAR_FUNCTION),
                execution=execution,
            )
            written.append(LIDAR_SCOPE)

        for scope, spec in FILE_SCOPES.items():
            write_regex_scope(
                root,
                scope,
                files=files_of(scope),
                dataset=family_dataset(self.name, attributes),
                modality=spec.modality,
                indexing=spec.indexing,
                addons=addon(spec.function),
                execution=execution,
            )
            written.append(scope)
        return written
