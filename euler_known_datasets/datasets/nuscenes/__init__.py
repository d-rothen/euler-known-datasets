"""nuScenes: camera images, the LIDAR_TOP sweep synchronised to each image,
camera intrinsics and per-frame lidar-to-camera extrinsics.

See ``docs/datasets/nuscenes.md`` for the prepared layout and the dataset
facts it is built on.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from ..._jsonio import write_json_atomic
from ...archives import selection_fingerprint
from ...artifacts import (
    IndexTree,
    family_dataset,
    file_entry,
    write_scope,
    write_splits,
)
from ...media import (
    IN_PLACE,
    MediaError,
    link_top_level,
    materialize_files,
    missing_files,
)
from ...recipe import Option, Outcome, Plan, PrepareContext, Recipe, register
from .calibration import (
    calibration_document,
    calibration_path,
    frame_record,
    sensor_matrix,
)
from .sources import VERSIONS, VersionSources, discover_sources
from .sync import (
    ROTATION_PERIOD_US,
    keyframe_offsets_us,
    nominal_offset_us,
    pair_with_lidar,
)
from .tables import Frame, NuScenesTables, Scene

logger = logging.getLogger(__name__)

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
LIDAR = "LIDAR_TOP"
LOADER_MODULE = "nuscenes"

SCOPE_RGB = "rgb"
SCOPE_LIDAR = "lidar"
SCOPE_INTRINSICS = "intrinsics"
SCOPE_EXTRINSICS = "camera_extrinsics"
SCOPES = (SCOPE_RGB, SCOPE_LIDAR, SCOPE_INTRINSICS, SCOPE_EXTRINSICS)

#: The official scene splits that apply to each version.
OFFICIAL_SPLITS = {
    "v1.0-trainval": ("train", "val"),
    "v1.0-test": ("test",),
    "v1.0-mini": ("mini_train", "mini_val"),
}
KEYFRAME_SUFFIX = "_keyframes"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def official_scene_splits() -> dict[str, list[str]]:
    """The nuscenes-devkit scene lists (``train``, ``val``, ``test``, ``mini_*``)."""
    path = Path(__file__).with_name("scene_splits.json")
    return json.loads(path.read_text(encoding="utf-8"))["splits"]


def hierarchy_keys(scene_name: str, camera: str) -> tuple[str, str]:
    return (f"scene:{scene_name}", f"camera:{camera}")


def _meta_selector(version: str):
    prefix = f"{version}/"

    def select(name: str) -> bool:
        return name.startswith(prefix) or name.startswith("maps/") or name == "LICENSE"

    return select


def _single(values: Iterable[Any], what: str) -> Any:
    distinct = set(values)
    if len(distinct) != 1:
        raise ValueError(f"expected one {what}, found {len(distinct)}: {sorted(map(str, distinct))}")
    return distinct.pop()


@dataclass
class _Unit:
    """One scene seen by one camera: its frames and their paired sweeps."""

    version: str
    scene: Scene
    camera: str
    frames: list[Frame]
    lidar: list[Optional[Frame]]
    camera_calib: dict
    lidar_calib: Optional[dict]
    width: int
    height: int


@register
class NuScenes(Recipe):
    name = "nuscenes"
    title = "nuScenes"
    summary = (
        "Camera images with the LIDAR_TOP sweep synchronised to each image, "
        "intrinsics and per-frame lidar-to-camera extrinsics."
    )
    homepage = "https://www.nuscenes.org/nuscenes"
    data_license = (
        "CC BY-NC-SA 4.0, non-commercial use only; see the LICENSE file in the download"
    )
    expected_inputs = (
        "v1.0-trainval_meta.tgz and v1.0-trainval01_blobs.tgz ... v1.0-trainval10_blobs.tgz",
        "v1.0-test_meta.tgz and v1.0-test_blobs.tgz",
        "v1.0-mini.tgz",
        "or what they extract to: v1.0-<version>/, samples/, sweeps/, maps/",
    )
    layout_version = 1
    options = (
        Option(
            "versions",
            "Dataset versions to prepare; all = every version found in the source.",
            default=(),
            multiple=True,
            choices=VERSIONS,
        ),
        Option(
            "cameras",
            "Camera channels to prepare.",
            default=CAMERAS,
            multiple=True,
            choices=CAMERAS,
        ),
        Option(
            "keyframes_only",
            "Prepare only the 2 Hz keyframes (samples/), not the ~12 Hz sweeps.",
            default=False,
            type=bool,
        ),
        Option(
            "sync_tolerance_ms",
            "Largest deviation from a camera's nominal lidar offset for a sweep "
            "to count as synchronised with an image; must stay below half a "
            "rotation (25 ms).",
            default=20.0,
            type=float,
        ),
    )

    def detect(self, source: Path) -> bool:
        try:
            return bool(discover_sources(Path(source)))
        except OSError:
            return False

    # -- plan -----------------------------------------------------------------

    def plan(self, ctx: PrepareContext) -> Plan:
        found = discover_sources(ctx.source)
        if not found:
            raise FileNotFoundError(
                f"no nuScenes downloads in {ctx.source}; expected "
                + "; ".join(self.expected_inputs)
            )
        versions = ctx.options["versions"] or tuple(v for v in VERSIONS if v in found)
        for version in versions:
            if version not in found or not found[version].has_tables:
                raise FileNotFoundError(
                    f"{version}: {ctx.source} holds neither an extracted {version}/ with "
                    "the tables nor the archive containing them"
                )
        if "v1.0-mini" in versions and "v1.0-trainval" in versions:
            raise ValueError(
                "v1.0-mini is a subset of v1.0-trainval (the same scenes); prepare them "
                "into separate outputs by passing versions"
            )
        tolerance = ctx.options["sync_tolerance_ms"]
        if not 0 < tolerance < ROTATION_PERIOD_US / 2e3:
            raise ValueError(f"sync_tolerance_ms must be in (0, 25), got {tolerance}")

        sources = {version: found[version] for version in versions}
        steps = []
        for version, vs in sources.items():
            where = vs.tables_dir if vs.tables_dir is not None else vs.meta_archive
            steps.append(f"read the {version} tables from {where}")
        cameras = ", ".join(ctx.options["cameras"])
        frames = "keyframes" if ctx.options["keyframes_only"] else "keyframes and sweeps"
        steps.append(
            f"pair {cameras} {frames} with {LIDAR} sweeps "
            f"(at the nominal offset, ±{tolerance:g} ms)"
        )
        if ctx.in_place:
            steps.append("prepare in place; extract only missing files")
        else:
            steps.append(f"bring media files into the output (media={ctx.media})")
        steps.append("write calibration/<scene>/<camera>.json for every scene and camera")
        steps.append(
            f"write ds-crawler scopes {', '.join(SCOPES)} and the official scene splits"
        )
        inputs = [item for vs in sources.values() for item in vs.describe()]
        return Plan(
            recipe=self.name,
            inputs=inputs,
            steps=steps,
            details={"versions": versions, "sources": sources},
        )

    # -- prepare --------------------------------------------------------------

    def prepare(self, ctx: PrepareContext, plan: Plan) -> Outcome:
        versions: tuple[str, ...] = plan.details["versions"]
        sources: dict[str, VersionSources] = plan.details["sources"]
        cameras: tuple[str, ...] = ctx.options["cameras"]
        tolerance_ms: float = ctx.options["sync_tolerance_ms"]

        tables, tables_in_output = self._load_tables(ctx, versions, sources, cameras)
        nominal = self._nominal_offsets(tables, cameras)
        units, needed, stats = self._pair(ctx, tables, cameras, nominal, tolerance_ms)
        media = self._materialize(ctx, versions, sources, needed, tables_in_output)
        trees, split_ids, unassigned = self._write_calibration_and_index(
            ctx, units, nominal, tolerance_ms
        )
        splits = self._write_scopes(ctx, versions, cameras, trees, split_ids)

        summary = {
            "versions": stats,
            "nominal_lidar_offset_ms": {cam: round(nominal[cam] / 1e3, 3) for cam in cameras},
            "calibration_files": len(units),
            "unassigned_scenes": unassigned,
            "files_referenced": sum(len(paths) for paths in needed.values()),
        }
        return Outcome(media=media, scopes=list(SCOPES), splits=splits, summary=summary)

    def _load_tables(
        self,
        ctx: PrepareContext,
        versions: tuple[str, ...],
        sources: dict[str, VersionSources],
        cameras: tuple[str, ...],
    ) -> tuple[dict[str, NuScenesTables], set[str]]:
        tables: dict[str, NuScenesTables] = {}
        tables_in_output: set[str] = set()
        for version in versions:
            vs = sources[version]
            if vs.tables_dir is not None:
                tables_dir = vs.tables_dir
            else:
                ctx.extract(
                    vs.meta_archive,
                    select=_meta_selector(version),
                    fingerprint=f"meta:{version}",
                )
                tables_dir = ctx.output / version
                tables_in_output.add(version)
            logger.info("reading %s tables from %s", version, tables_dir)
            tables[version] = NuScenesTables(version, tables_dir, (*cameras, LIDAR))
        return tables, tables_in_output

    @staticmethod
    def _lidar_keyframes(tables: NuScenesTables) -> dict[str, Frame]:
        return {
            frame.sample_token: frame
            for scene in tables.scenes.values()
            for frame in tables.frames(scene.token, LIDAR)
            if frame.is_key_frame
        }

    def _nominal_offsets(
        self, tables: dict[str, NuScenesTables], cameras: tuple[str, ...]
    ) -> dict[str, float]:
        nominal = {}
        lidar_keyframes = {v: self._lidar_keyframes(t) for v, t in tables.items()}
        for camera in cameras:
            offsets: list[int] = []
            for version, t in tables.items():
                for scene in t.scenes.values():
                    offsets += keyframe_offsets_us(
                        t.frames(scene.token, camera), lidar_keyframes[version]
                    )
            nominal[camera] = nominal_offset_us(offsets)
            logger.info(
                "%s fires %.2f ms before the end of its %s sweep (median of %d keyframes)",
                camera,
                nominal[camera] / 1e3,
                LIDAR,
                len(offsets),
            )
            if not 0 <= nominal[camera] <= ROTATION_PERIOD_US:
                logger.warning(
                    "%s: nominal offset %.1f ms is outside one %d ms rotation; the "
                    "sweep timing differs from v1.0, check the pairing",
                    camera,
                    nominal[camera] / 1e3,
                    ROTATION_PERIOD_US // 1000,
                )
        return nominal

    def _pair(
        self,
        ctx: PrepareContext,
        tables: dict[str, NuScenesTables],
        cameras: tuple[str, ...],
        nominal: dict[str, float],
        tolerance_ms: float,
    ) -> tuple[list[_Unit], dict[str, set[str]], dict[str, Any]]:
        units: list[_Unit] = []
        needed: dict[str, set[str]] = defaultdict(set)
        stats: dict[str, Any] = {}
        for version, t in tables.items():
            lidar_keyframes = self._lidar_keyframes(t)
            per_camera = {
                cam: {
                    "frames": 0,
                    "synced": 0,
                    "keyframes": 0,
                    "keyframes_synced": 0,
                    "keyframes_paired_as_sample": 0,
                }
                for cam in cameras
            }
            stats[version] = {"scenes": len(t.scenes), "cameras": per_camera}
            for scene in t.scenes_by_name():
                if not _SAFE_NAME.match(scene.name):
                    raise ValueError(f"{version}: scene name {scene.name!r} is not a safe path")
                lidar_frames = t.frames(scene.token, LIDAR)
                lidar_us = np.array([f.timestamp for f in lidar_frames], dtype=np.int64)
                lidar_calib = None
                if lidar_frames:
                    token = _single(
                        (f.calibrated_sensor_token for f in lidar_frames),
                        f"{LIDAR} calibration in {scene.name}",
                    )
                    lidar_calib = t.calibrated_sensor[token]
                for camera in cameras:
                    frames = t.frames(scene.token, camera)
                    if ctx.options["keyframes_only"]:
                        frames = [f for f in frames if f.is_key_frame]
                    if not frames:
                        logger.warning("%s %s has no %s frames", version, scene.name, camera)
                        continue
                    camera_us = np.array([f.timestamp for f in frames], dtype=np.int64)
                    if np.unique(camera_us).size != camera_us.size:
                        raise ValueError(f"{scene.name} {camera}: duplicate timestamps")
                    index, _, synced = pair_with_lidar(
                        camera_us, lidar_us, nominal[camera], tolerance_ms * 1e3
                    )
                    paired = [
                        lidar_frames[i] if ok else None for i, ok in zip(index, synced)
                    ]
                    calib_token = _single(
                        (f.calibrated_sensor_token for f in frames),
                        f"{camera} calibration in {scene.name}",
                    )
                    width = _single((f.width for f in frames), f"{camera} width")
                    height = _single((f.height for f in frames), f"{camera} height")
                    units.append(
                        _Unit(
                            version=version,
                            scene=scene,
                            camera=camera,
                            frames=frames,
                            lidar=paired,
                            camera_calib=t.calibrated_sensor[calib_token],
                            lidar_calib=lidar_calib,
                            width=width,
                            height=height,
                        )
                    )
                    counts = per_camera[camera]
                    counts["frames"] += len(frames)
                    counts["synced"] += int(synced.sum())
                    for frame, lidar in zip(frames, paired):
                        needed[version].add(frame.filename)
                        if lidar is not None:
                            needed[version].add(lidar.filename)
                        if frame.is_key_frame:
                            counts["keyframes"] += 1
                            if lidar is None:
                                continue
                            counts["keyframes_synced"] += 1
                            own = lidar_keyframes.get(frame.sample_token)
                            if own is not None and lidar.token == own.token:
                                counts["keyframes_paired_as_sample"] += 1
            for camera, counts in per_camera.items():
                counts["unsynced"] = counts["frames"] - counts["synced"]
                # nuScenes pairs every keyframe camera with its sample's sweep; a
                # different choice means the timing model does not fit this data.
                mismatched = counts["keyframes_synced"] - counts["keyframes_paired_as_sample"]
                if mismatched:
                    logger.warning(
                        "%s %s: %d synchronised keyframes paired with a sweep other "
                        "than their sample's own",
                        version,
                        camera,
                        mismatched,
                    )
                logger.info(
                    "%s %s: %d frames, %d with a synchronised sweep",
                    version,
                    camera,
                    counts["frames"],
                    counts["synced"],
                )
        return units, needed, stats

    def _materialize(
        self,
        ctx: PrepareContext,
        versions: tuple[str, ...],
        sources: dict[str, VersionSources],
        needed: dict[str, set[str]],
        tables_in_output: set[str],
    ) -> str:
        all_needed = sorted(set().union(*needed.values())) if needed else []

        if ctx.in_place:
            for version in versions:
                missing = missing_files(ctx.source, sorted(needed[version]))
                if missing:
                    self._extract_blobs(ctx, sources[version], missing, ctx.source)
            self._require_present(ctx.source, all_needed)
            return IN_PLACE

        mode = ctx.media
        if mode == "auto":
            missing = missing_files(ctx.source, all_needed)
            mode = "symlink" if not missing else "extract"
            logger.info(
                "media=auto: %s",
                "every needed file is extracted in the source, linking it"
                if mode == "symlink"
                else f"{len(missing)} needed files are not extracted, extracting archives",
            )

        if mode in ("symlink", "hardlink", "copy"):
            missing = missing_files(ctx.source, all_needed)
            if missing:
                raise MediaError(
                    f"{len(missing)} needed files are not extracted in {ctx.source} "
                    f"(e.g. {missing[0]}); use media='extract' with the blob archives"
                )
        if mode == "symlink":
            names = ["samples", "sweeps"]
            names += [v for v in versions if v not in tables_in_output]
            names += [n for n in ("maps", "LICENSE") if not (ctx.output / n).exists()]
            link_top_level(ctx.source, ctx.output, names)
        elif mode in ("hardlink", "copy"):
            files = list(all_needed)
            for version in versions:
                if version not in tables_in_output:
                    files += self._extracted_meta_files(ctx.source, version)
            counts = materialize_files(ctx.source, ctx.output, files, mode)
            logger.info("media=%s: %s", mode, counts)
        elif mode == "extract":
            for version in versions:
                vs = sources[version]
                if not vs.blob_archives:
                    raise MediaError(f"{version}: no blob archives in {ctx.source} to extract")
                self._extract_blobs(ctx, vs, needed[version], ctx.output)
                if version in tables_in_output:
                    continue
                if vs.meta_archive is not None:
                    ctx.extract(
                        vs.meta_archive,
                        select=_meta_selector(version),
                        fingerprint=f"meta:{version}",
                    )
                else:
                    materialize_files(
                        ctx.source,
                        ctx.output,
                        self._extracted_meta_files(ctx.source, version),
                        "copy",
                    )
        else:  # pragma: no cover - validated by prepare()
            raise ValueError(f"unknown media mode {mode!r}")
        self._require_present(ctx.output, all_needed)
        return mode

    @staticmethod
    def _extracted_meta_files(source: Path, version: str) -> list[str]:
        files = [f"{version}/{p.name}" for p in sorted((source / version).glob("*.json"))]
        maps = source / "maps"
        if maps.is_dir():
            files += [f"maps/{p.name}" for p in sorted(maps.iterdir()) if p.is_file()]
        if (source / "LICENSE").is_file():
            files.append("LICENSE")
        return files

    @staticmethod
    def _extract_blobs(
        ctx: PrepareContext, vs: VersionSources, files: Iterable[str], destination: Path
    ) -> None:
        if not vs.blob_archives:
            raise MediaError(
                f"{vs.version}: needed files are missing and there are no blob archives"
            )
        selection = set(files)
        fingerprint = selection_fingerprint(selection)
        for archive in vs.blob_archives:
            ctx.extract(
                archive,
                destination=destination,
                select=selection.__contains__,
                fingerprint=fingerprint,
            )

    @staticmethod
    def _require_present(root: Path, files: list[str]) -> None:
        missing = missing_files(root, files)
        if missing:
            raise MediaError(
                f"{len(missing)} needed files are still missing below {root} "
                f"(e.g. {missing[0]}); the archives may be incomplete"
            )

    def _write_calibration_and_index(
        self,
        ctx: PrepareContext,
        units: list[_Unit],
        nominal: dict[str, float],
        tolerance_ms: float,
    ) -> tuple[dict[str, IndexTree], dict[str, dict[str, list[tuple]]], list[str]]:
        official = official_scene_splits()
        trees = {scope: IndexTree() for scope in SCOPES}
        split_ids: dict[str, dict[str, list[tuple]]] = {
            SCOPE_RGB: defaultdict(list),
            SCOPE_LIDAR: defaultdict(list),
        }
        unassigned: set[str] = set()
        for unit in units:
            split = next(
                (
                    name
                    for name in OFFICIAL_SPLITS.get(unit.version, ())
                    if unit.scene.name in official[name]
                ),
                None,
            )
            if split is None:
                unassigned.add(unit.scene.name)
            keys = hierarchy_keys(unit.scene.name, unit.camera)
            camera_to_ego = sensor_matrix(unit.camera_calib)
            lidar_to_ego = (
                sensor_matrix(unit.lidar_calib) if unit.lidar_calib is not None else None
            )
            records = {}
            for frame, lidar in zip(unit.frames, unit.lidar):
                file_id = str(frame.timestamp)
                records[file_id] = frame_record(frame, camera_to_ego, lidar, lidar_to_ego)
                trees[SCOPE_RGB].add(
                    keys,
                    file_entry(
                        frame.filename,
                        file_id,
                        {"is_key_frame": frame.is_key_frame, "sample_data_token": frame.token},
                    ),
                )
                qid = keys + (file_id,)
                if split is not None:
                    split_ids[SCOPE_RGB][split].append(qid)
                    if frame.is_key_frame:
                        split_ids[SCOPE_RGB][split + KEYFRAME_SUFFIX].append(qid)
                if lidar is None:
                    continue
                trees[SCOPE_LIDAR].add(
                    keys,
                    file_entry(
                        lidar.filename,
                        file_id,
                        {
                            "sample_data_token": lidar.token,
                            "lidar_timestamp": lidar.timestamp,
                            "offset_ms": (lidar.timestamp - frame.timestamp) / 1e3,
                        },
                    ),
                )
                if split is not None:
                    split_ids[SCOPE_LIDAR][split].append(qid)
                    if frame.is_key_frame:
                        split_ids[SCOPE_LIDAR][split + KEYFRAME_SUFFIX].append(qid)

            relpath = calibration_path(unit.scene.name, unit.camera)
            document = calibration_document(
                version=unit.version,
                scene=unit.scene,
                camera_calib=unit.camera_calib,
                lidar_calib=unit.lidar_calib,
                channel=unit.camera,
                lidar_channel=LIDAR,
                width=unit.width,
                height=unit.height,
                nominal_offset_ms=nominal[unit.camera] / 1e3,
                sync_tolerance_ms=tolerance_ms,
                frames=records,
            )
            write_json_atomic(ctx.output / relpath, document, compact=True)
            trees[SCOPE_INTRINSICS].add(keys, file_entry(relpath, "intrinsics"))
            trees[SCOPE_EXTRINSICS].add(keys, file_entry(relpath, "extrinsics"))
        if unassigned:
            logger.warning(
                "%d scenes are in no official split: %s",
                len(unassigned),
                ", ".join(sorted(unassigned)[:10]),
            )
        return trees, split_ids, sorted(unassigned)

    def _write_scopes(
        self,
        ctx: PrepareContext,
        versions: tuple[str, ...],
        cameras: tuple[str, ...],
        trees: dict[str, IndexTree],
        split_ids: dict[str, dict[str, list[tuple]]],
    ) -> dict[str, list[str]]:
        execution = {
            "recipe": self.name,
            "layout_version": self.layout_version,
            "options": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in ctx.options.items()
            },
        }
        attributes = {"source": "nuScenes", "versions": list(versions), "cameras": list(cameras)}

        def addon(function: str) -> dict[str, Any]:
            # Only the loader: roles such as used_as belong to an experiment,
            # not to the dataset.
            return {
                "euler_loading": {"version": "1.0", "loader": LOADER_MODULE, "function": function}
            }

        heads = {
            SCOPE_RGB: (
                {"key": "rgb", "meta": {"range": [0, 255]}},
                addon("rgb"),
            ),
            SCOPE_LIDAR: (
                {
                    "key": "lidar_point_cloud",
                    # The euler-dataset-contract's conventions, as in its
                    # reference heads (fixtures/heads/valid/phase0-*.json).
                    "meta": {
                        "representation": "point_cloud",
                        "columns": ["x", "y", "z", "intensity", "ring"],
                        "coordinate_unit": "meters",
                        "frame": LIDAR,
                        "axes": "x right, y forward, z up",
                    },
                },
                addon("lidar_point_cloud"),
            ),
            SCOPE_INTRINSICS: (
                {
                    "key": "intrinsics",
                    "meta": {"camera_model": "pinhole", "dimensions": {"rows": 3, "columns": 3}},
                },
                addon("read_intrinsics"),
            ),
            SCOPE_EXTRINSICS: (
                {
                    "key": "camera_extrinsics",
                    "meta": {
                        "source_frame": LIDAR,
                        "target_frame": "camera_optical",
                        "direction": "target_from_source",
                        "dimensions": {"rows": 4, "columns": 4},
                    },
                },
                addon("read_extrinsics"),
            ),
        }
        for scope, (modality, addons) in heads.items():
            path = write_scope(
                ctx.output,
                scope,
                dataset=family_dataset(self.name, attributes),
                modality=modality,
                addons=addons,
                tree=trees[scope],
                execution=execution,
            )
            logger.info("wrote %s (%d files)", path, len(trees[scope]))

        splits: dict[str, list[str]] = {}
        for scope, by_name in split_ids.items():
            official = {k: v for k, v in by_name.items() if not k.endswith(KEYFRAME_SUFFIX)}
            keyframes = {k: v for k, v in by_name.items() if k.endswith(KEYFRAME_SUFFIX)}
            names = write_splits(ctx.output, scope, official)
            if not ctx.options["keyframes_only"]:
                names += write_splits(ctx.output, scope, keyframes)
            splits[scope] = sorted(names)
        return splits
