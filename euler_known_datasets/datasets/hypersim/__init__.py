"""Hypersim: photorealistic synthetic indoor scenes with HDR colour, depth and
exact camera parameters (Roberts et al., ICCV 2021).

See ``docs/datasets/hypersim.md`` for the layout and the dataset facts it
relies on, and :mod:`.camera` for the camera model.
"""

from __future__ import annotations

import csv
import fnmatch
import io
import logging
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
from ...geometry import matrix_to_list
from ...media import IN_PLACE, MediaError, missing_files, place_extracted, resolve_media
from ...recipe import Option, Outcome, Plan, PrepareContext, Recipe, register
from .camera import (
    SceneCamera,
    calibration_document,
    camera_parameters,
    camera_to_world,
    rigid_frames,
)
from .sources import (
    CAMERA_CSV,
    CAMERA_FILES,
    REPOSITORY_PATHS,
    SCENE_FILES,
    SPLIT_CSV,
    SPLITS,
    Metadata,
    SceneSource,
    discover,
    find_metadata,
    read_scene_file,
)

logger = logging.getLogger(__name__)

LOADER_MODULE = "hypersim"
MODALITIES = ("rgb", "depth", "radial_distance", "intrinsics", "camera_pose")
DEFAULT_MODALITIES = MODALITIES
IMAGE_SCOPES = ("rgb", "depth", "radial_distance")
#: Hypersim's tone-mapping operator (code/python/tools/scene_generate_images_tonemap.py).
TONEMAP = {"operator": "hypersim", "gamma": 1 / 2.2, "percentile": 90, "percentile_target": 0.8,
           "brightness": "0.3 R + 0.59 G + 0.11 B", "valid_pixels": "render_entity_id != -1"}
#: Most distances are a few meters (median 5 m), but geometry seen through
#: windows reaches 1900 m (all 18,660 frames of every 4th public frame,
#: 2026-11-13); the receipt's summary has the maximum of the prepared frames.
DEPTH_RANGE = [0, 2000]
_INVALID_ZERO = {"non_finite": False, "sentinel": 0}
FRAME_KEYS = ("_detail/{camera}/camera_keyframe_frame_indices.hdf5",)


def _addon(function: str) -> dict[str, Any]:
    return {"euler_loading": {"version": "1.0", "loader": LOADER_MODULE, "function": function}}


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "preparing Hypersim reads HDF5; install h5py (pip install 'euler-known-datasets[hypersim]')"
        ) from exc
    return h5py


def _read_dataset(path) -> np.ndarray:
    """The ``dataset`` array of an HDF5 file (a path or a binary stream)."""
    with _h5py().File(path, "r") as handle:
        return handle["dataset"][()]


def frame_statistics(job: tuple[Optional[str], Optional[str], Optional[str]]) -> dict[str, Any]:
    """Tone-mapping scale and depth statistics of one frame (run in worker processes).

    *job* holds the paths of the colour, render-entity and depth files (any
    may be ``None``). The scale is Hypersim's: the 90th-percentile brightness
    of the pixels with geometry becomes 0.8 after gamma 1/2.2.
    """
    color_path, entity_path, depth_path = job
    stats: dict[str, Any] = {}
    if color_path is not None:
        color = _read_dataset(Path(color_path)).astype(np.float32)
        brightness = 0.3 * color[..., 0] + 0.59 * color[..., 1] + 0.11 * color[..., 2]
        finite = np.isfinite(brightness)
        if entity_path is not None:
            valid = (_read_dataset(Path(entity_path)) != -1) & finite
        else:
            valid = finite
        stats["entity_ids"] = entity_path is not None
        stats["nonfinite_color"] = int((~np.isfinite(color)).any(axis=-1).sum())
        if not valid.any():
            stats["tonemap_scale"] = 1.0
        else:
            current = float(np.percentile(brightness[valid], TONEMAP["percentile"]))
            target = TONEMAP["percentile_target"] ** (1.0 / TONEMAP["gamma"])
            stats["tonemap_scale"] = 0.0 if current < 1e-4 else target / current
    if depth_path is not None:
        distance = _read_dataset(Path(depth_path)).astype(np.float32)
        good = np.isfinite(distance) & (distance > 0)
        stats["invalid_fraction"] = float(1.0 - good.mean())
        stats["max_distance"] = float(distance[good].max()) if good.any() else 0.0
    return stats


@dataclass
class _Trajectory:
    scene: str
    camera: str
    frames: list  # [(frame, split)]


@register
class Hypersim(Recipe):
    name = "hypersim"
    title = "Hypersim"
    summary = (
        "Photorealistic synthetic indoor scenes: tone-mapped HDR renders, planar "
        "depth, exact per-scene intrinsics and metric camera poses; official splits."
    )
    homepage = "https://github.com/apple/ml-hypersim"
    data_license = (
        "CC BY-SA 3.0 (the dataset); the split and camera CSVs come from the "
        "ml-hypersim repository, under its licence"
    )
    expected_inputs = (
        "the extracted scene folders ai_VVV_NNN/{_detail,images}/ (all, or the files a "
        "partial download fetched), or the per-scene archives ai_VVV_NNN.zip",
        f"ml-hypersim's {SPLIT_CSV} and {CAMERA_CSV}, at the source root, in metadata/, "
        "or at their repository paths below ml-hypersim/",
    )
    layout_version = 1
    options = (
        Option("modalities", "Scopes to write.", default=DEFAULT_MODALITIES, multiple=True, choices=MODALITIES),
        Option("scenes", "Scenes to prepare, as names or patterns such as ai_00*; all = every one found.",
               default=(), multiple=True),
        Option("splits", "Official splits to prepare.", default=SPLITS, multiple=True, choices=SPLITS),
        Option("frame_step", "Keep only frames whose number is a multiple of this.", default=1, type=int),
        Option("workers", "Processes reading the images for the tone-mapping scale (0 = all CPUs, at most 16).",
               default=0, type=int),
        Option("keep_dark_frames", "Keep frames too dark to tone-map (Hypersim's operator renders them black).",
               default=False, type=bool),
    )

    def detect(self, source: Path) -> bool:
        try:
            scenes, _ = discover(Path(source), scan=set())
        except OSError:
            return False
        return bool(scenes)

    # ------------------------------------------------------------------ plan

    def plan(self, ctx: PrepareContext) -> Plan:
        split_csv, camera_csv = find_metadata(ctx.source)
        if split_csv is None or camera_csv is None:
            missing = [name for name, path in ((SPLIT_CSV, split_csv), (CAMERA_CSV, camera_csv)) if path is None]
            raise FileNotFoundError(
                f"{', '.join(missing)} not found in {ctx.source}; copy "
                + " and ".join(f"{REPOSITORY_PATHS[name]}" for name in missing)
                + " from https://github.com/apple/ml-hypersim into it (e.g. below ml-hypersim/)"
            )
        if ctx.options["frame_step"] < 1:
            raise ValueError("frame_step must be at least 1")
        metadata = Metadata(split_csv, camera_csv)
        public = metadata.public_frames()
        found, prefix = discover(ctx.source, scan=set())
        if not found:
            raise FileNotFoundError(
                f"no Hypersim scenes in {ctx.source}; expected " + "; ".join(self.expected_inputs)
            )
        patterns = ctx.options["scenes"]
        names = sorted(n for n in found if not patterns or any(fnmatch.fnmatchcase(n, p) for p in patterns))
        if patterns and not names:
            raise FileNotFoundError(f"no scene in {ctx.source} matches {', '.join(patterns)}")
        not_public = [n for n in names if n not in public]
        names = [n for n in names if n in public]
        found, _ = discover(ctx.source, scan=set(names))
        scenes = {n: found[n] for n in names}

        modalities = tuple(ctx.options["modalities"])
        roles = set()
        if "rgb" in modalities:
            roles.add("color")
        if {"depth", "radial_distance"} & set(modalities):
            roles.add("depth")
        splits, step = set(ctx.options["splits"]), ctx.options["frame_step"]
        trajectories: list[_Trajectory] = []
        skipped: Counter = Counter()
        other_camera: list[str] = []
        for name, scene in scenes.items():
            by_camera: dict[str, list] = defaultdict(list)
            for (camera, frame), split in sorted(public[name].items()):
                if split not in splits or int(frame) % step:
                    continue
                files = scene.frames.get((camera, frame), {})
                if not roles <= set(files):
                    skipped["frames_without_files"] += 1
                    continue
                by_camera[camera].append((frame, split))
            for camera, frames in sorted(by_camera.items()):
                needed = {f.format(camera=camera) for f in CAMERA_FILES + SCENE_FILES}
                if {"intrinsics", "camera_pose"} & set(modalities) and not needed <= scene.files:
                    skipped["frames_without_camera_files"] += len(frames)
                    continue
                # Frames whose stored orientation is stretched or mirrored were
                # rendered by another camera than the scene's (see camera.py).
                orientations = f"_detail/{camera}/camera_keyframe_orientations.hdf5"
                if orientations in scene.files:
                    stored = io.BytesIO(read_scene_file(ctx.source, scene, orientations))
                    rigid = rigid_frames(_read_dataset(stored))
                    kept = [(f, s) for f, s in frames if int(f) < len(rigid) and rigid[int(f)]]
                    other_camera += [f"{name}/{camera}/{f}" for f, s in frames if (f, s) not in kept]
                    skipped["frames_rendered_with_another_camera"] += len(frames) - len(kept)
                    frames = kept
                if frames:
                    trajectories.append(_Trajectory(name, camera, frames))
        skipped = Counter({k: v for k, v in skipped.items() if v})
        if not trajectories:
            raise FileNotFoundError(
                f"no complete frames in {ctx.source} for splits {', '.join(sorted(splits))} "
                f"and frame_step {step}"
            )
        extracted = all(scenes[t.scene].extracted for t in trajectories)
        archived = all(scenes[t.scene].archive is not None for t in trajectories)
        media = resolve_media(ctx, extracted=extracted, archives=archived)
        frames = sum(len(t.frames) for t in trajectories)
        by_split = Counter(split for t in trajectories for _, split in t.frames)
        inputs = [{
            "scenes": len({t.scene for t in trajectories}),
            "trajectories": len(trajectories),
            "frames": frames,
            "by_split": dict(sorted(by_split.items())),
            "extracted_below": prefix if extracted else None,
            "archives": sum(1 for t in {t.scene for t in trajectories} if scenes[t].archive is not None),
            "split_csv": str(split_csv.relative_to(ctx.source)),
            "camera_csv": str(camera_csv.relative_to(ctx.source)),
        }]
        steps = []
        if not_public:
            steps.append(f"note: {len(not_public)} scenes are not in the public release and are skipped")
        if skipped:
            steps.append(f"note: skipping {dict(skipped)}")
        steps += [
            f"prepare {', '.join(modalities)} for {frames} frames (frame_step {step})",
            f"bring the colour and depth files into the output (media={media})",
            "compute each frame's tone-mapping scale from its colour and render-entity images"
            if "rgb" in modalities else "read the depth maps for their statistics",
            "write calibration/<scene>/<camera>.json (exact pinhole intrinsics, metric poses)",
            "write prebuilt scopes and the official splits",
        ]
        return Plan(recipe=self.name, inputs=inputs, steps=steps, details={
            "metadata": metadata, "scenes": scenes, "trajectories": trajectories, "media": media,
            "modalities": modalities, "skipped": dict(skipped), "not_public": not_public,
            "other_camera": other_camera,
        })

    # --------------------------------------------------------------- prepare

    def prepare(self, ctx: PrepareContext, plan: Plan) -> Outcome:
        details = plan.details
        scenes: dict[str, SceneSource] = details["scenes"]
        trajectories: list[_Trajectory] = details["trajectories"]
        modalities: tuple = details["modalities"]
        media: str = details["media"]
        cameras = camera_parameters(details["metadata"].camera_rows())
        missing_camera = sorted({t.scene for t in trajectories if t.scene not in cameras})
        if missing_camera:
            logger.warning("%d scenes have no camera parameters and are skipped: %s",
                           len(missing_camera), ", ".join(missing_camera[:5]))
            trajectories = [t for t in trajectories if t.scene in cameras]

        def in_source(scene: str, rel: str) -> str:
            """A scene file's path relative to the source."""
            prefix = scenes[scene].prefix
            return f"{prefix}/{scene}/{rel}" if prefix else f"{scene}/{rel}"

        # Files the scopes index, and those only read while preparing.
        indexed: dict[str, list[str]] = defaultdict(list)
        auxiliary: dict[str, list[str]] = defaultdict(list)
        for t in trajectories:
            scene = scenes[t.scene]
            for frame, _ in t.frames:
                files = scene.frames[(t.camera, frame)]
                if "rgb" in modalities:
                    indexed[t.scene].append(files["color"])
                if {"depth", "radial_distance"} & set(modalities):
                    indexed[t.scene].append(files["depth"])
                if "rgb" in modalities and "render_entity_id" in files:
                    auxiliary[t.scene].append(files["render_entity_id"])
            for pattern in CAMERA_FILES + SCENE_FILES + FRAME_KEYS:
                rel = pattern.format(camera=t.camera)
                if rel in scene.files:
                    auxiliary[t.scene].append(rel)
        root, read_root, layout = self._materialize(ctx, scenes, media, indexed, auxiliary, in_source)

        def indexed_path(scene: str, rel: str) -> str:
            return in_source(scene, rel) if layout == "source" else f"{scene}/{rel}"

        def readable(scene: str, rel: str) -> Path:
            return read_root / (in_source(scene, rel) if read_root == ctx.source else f"{scene}/{rel}")

        # Per-frame statistics: tone-mapping scales and depth ranges.
        jobs, keys = [], []
        for t in trajectories:
            files_of = scenes[t.scene].frames
            for frame, _ in t.frames:
                files = files_of[(t.camera, frame)]
                color = str(readable(t.scene, files["color"])) if "rgb" in modalities else None
                entity = str(readable(t.scene, files["render_entity_id"])) if color and "render_entity_id" in files else None
                depth = str(readable(t.scene, files["depth"])) if "depth" in files and (
                    {"depth", "radial_distance"} & set(modalities)) else None
                jobs.append((color, entity, depth))
                keys.append((t.scene, t.camera, frame))
        stats = dict(zip(keys, self._run(jobs, ctx.options["workers"])))
        # A render whose 90th-percentile brightness is below 1e-4 gets scale 0:
        # Hypersim's operator turns it black (whole scenes, e.g. ai_003_001).
        dark = sorted("/".join(key) for key, value in stats.items() if value.get("tonemap_scale") == 0.0)
        if dark and not ctx.options["keep_dark_frames"]:
            logger.warning("leaving out %d frames too dark to tone-map", len(dark))
            dropped = set(dark)
            for t in trajectories:
                t.frames = [(f, s) for f, s in t.frames if f"{t.scene}/{t.camera}/{f}" not in dropped]
            trajectories = [t for t in trajectories if t.frames]
            stats = {key: value for key, value in stats.items() if "/".join(key) not in dropped}
            details["skipped"] = {**details["skipped"], "dark_frames": len(dark)}
        details["dark_frames"] = dark

        trees = {scope: IndexTree() for scope in modalities}
        split_ids: dict[str, dict[str, list]] = {scope: defaultdict(list) for scope in IMAGE_SCOPES}
        orientation_notes: dict[str, Any] = {}
        frame_key_mismatch: list[str] = []
        scene_cameras: dict[str, SceneCamera] = {}
        for t in trajectories:
            camera = cameras[t.scene]
            scene_cameras[t.scene] = camera
            keys_ = (f"scene:{t.scene}", f"camera:{t.camera}")
            matrix = matrix_to_list(camera.intrinsics)
            files_of = scenes[t.scene].frames
            for frame, split in t.frames:
                files = files_of[(t.camera, frame)]
                frame_stats = stats[(t.scene, t.camera, frame)]
                if "rgb" in trees:
                    trees["rgb"].add(keys_, file_entry(indexed_path(t.scene, files["color"]), frame,
                                                       {"tonemap_scale": frame_stats["tonemap_scale"]}))
                for scope in ("depth", "radial_distance"):
                    if scope in trees:
                        trees[scope].add(keys_, file_entry(indexed_path(t.scene, files["depth"]), frame,
                                                           {"intrinsics": matrix}))
                for scope in IMAGE_SCOPES:
                    if scope in trees:
                        split_ids[scope][split].append(keys_ + (frame,))
            if {"intrinsics", "camera_pose"} & set(modalities):
                relpath = f"calibration/{t.scene}/{t.camera}.json"
                document, notes, mismatch = self._calibration(t, camera, lambda rel, s=t.scene: readable(s, rel))
                if notes:
                    orientation_notes[f"{t.scene}/{t.camera}"] = notes
                if mismatch:
                    frame_key_mismatch.append(f"{t.scene}/{t.camera}")
                write_json_atomic(ctx.output / relpath, document, compact=True)
                for scope in ("intrinsics", "camera_pose"):
                    if scope in trees:
                        trees[scope].add(keys_, file_entry(relpath, scope))

        sizes = {(c.height, c.width) for c in scene_cameras.values()}
        grid = {"height": sizes.copy().pop()[0], "width": sizes.copy().pop()[1]} if len(sizes) == 1 else None
        splits = self._write_scopes(ctx, trees, split_ids, modalities, grid)
        summary = self._summary(trajectories, stats, scene_cameras, orientation_notes, frame_key_mismatch,
                                details, ctx.options)
        return Outcome(media=media, scopes=list(modalities), splits=splits, summary=summary)

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _materialize(ctx, scenes, media, indexed, auxiliary, in_source):
        """Bring the indexed files into place.

        Returns the dataset root, the root to read files from while preparing,
        and whether indexed paths follow the source's layout (``"source"``) or
        start at the scene folders (``"scenes"``).
        """
        if media == IN_PLACE:
            for name in indexed:
                scene = scenes[name]
                if not scene.extracted:
                    wanted = {f"{name}/{rel}" for rel in indexed[name] + auxiliary[name]}
                    ctx.extract(scene.archive, destination=ctx.source, select=wanted.__contains__,
                                fingerprint=selection_fingerprint(wanted))
                    scene.prefix = ""
            return ctx.source, ctx.source, "source"
        if media == "extract":
            for name in sorted(indexed):
                wanted = {f"{name}/{rel}" for rel in indexed[name] + auxiliary[name]}
                ctx.extract(scenes[name].archive, destination=ctx.output, select=wanted.__contains__,
                            fingerprint=selection_fingerprint(wanted))
            needed = [f"{name}/{rel}" for name in indexed for rel in indexed[name]]
            missing = missing_files(ctx.output, needed)
            if missing:
                raise MediaError(f"{len(missing)} files are still missing below {ctx.output} (e.g. {missing[0]})")
            return ctx.output, ctx.output, "scenes"
        needed = [in_source(name, rel) for name in indexed for rel in indexed[name]]
        prefixes = {scenes[name].prefix for name in indexed}
        top_level = sorted(prefixes) if prefixes != {""} else sorted(indexed)
        place_extracted(ctx, needed, mode=media, top_level=top_level)
        # Hard links and copies hold only the indexed files: the camera and
        # render-entity files are read from the source.
        return ctx.output, ctx.source, "source"

    @staticmethod
    def _run(jobs: list, workers: int) -> list[dict[str, Any]]:
        if not jobs:
            return []
        if workers <= 0:
            try:
                workers = len(os.sched_getaffinity(0))
            except AttributeError:  # pragma: no cover - not on Linux
                workers = os.cpu_count() or 1
        workers = max(1, min(16, workers))
        if workers == 1 or len(jobs) < 64:
            return [frame_statistics(job) for job in jobs]
        logger.info("reading %d frames with %d processes", len(jobs), workers)
        with ProcessPoolExecutor(workers) as pool:
            return list(pool.map(frame_statistics, jobs, chunksize=16))

    @staticmethod
    def _calibration(t: _Trajectory, camera: SceneCamera, readable) -> tuple[dict, dict, bool]:
        orientations = _read_dataset(readable(f"_detail/{t.camera}/camera_keyframe_orientations.hdf5"))
        positions = _read_dataset(readable(f"_detail/{t.camera}/camera_keyframe_positions.hdf5"))
        with open(readable("_detail/metadata_scene.csv"), newline="") as handle:
            rows = {row["parameter_name"]: row["parameter_value"] for row in csv.DictReader(handle)}
        meters = float(rows["meters_per_asset_unit"])
        poses, valid, notes = camera_to_world(camera, orientations, positions, meters)
        frame_ids = list(range(len(poses)))
        mismatch = False
        try:
            indices = _read_dataset(readable(FRAME_KEYS[0].format(camera=t.camera)))
            if list(np.asarray(indices).ravel()) != frame_ids:
                mismatch = True
                frame_ids = [int(i) for i in np.asarray(indices).ravel()]
        except (FileNotFoundError, OSError, KeyError):
            pass
        document = calibration_document(
            scene=t.scene, camera_name=t.camera, camera=camera, meters_per_asset_unit=meters,
            poses={f"{frame:04d}": pose for frame, pose, ok in zip(frame_ids, poses, valid) if ok},
        )
        missing = [frame for frame, _ in t.frames if frame not in document["frames"]]
        if missing:
            raise ValueError(f"{t.scene} {t.camera}: no camera pose for frames {', '.join(missing[:5])}")
        return document, notes, mismatch

    def _write_scopes(self, ctx, trees, split_ids, modalities, grid) -> dict[str, list[str]]:
        execution = {"recipe": self.name, "layout_version": self.layout_version,
                     "options": {k: list(v) if isinstance(v, tuple) else v for k, v in ctx.options.items()}}
        attributes = {"source": "Hypersim", "frame_step": ctx.options["frame_step"]}
        dims = {"dimensions": grid} if grid else {}
        dims_rgb = {"dimensions": {**grid, "channels": 3}} if grid else {}
        stored = "Euclidean distance to the camera centre (depth_meters.hdf5, float16); "
        heads = {
            "rgb": ({"key": "rgb", "meta": {
                "range": [0, 1], "stored": "linear HDR colour (color.hdf5, float16)", "tonemap": TONEMAP,
                "tonemap_scale": "per file, attributes.tonemap_scale", **dims_rgb,
            }}, "rgb"),
            "depth": ({"key": "depth", "meta": {
                "radial_depth": False, "scale_to_meters": 1.0, "range": list(DEPTH_RANGE),
                "invalid": dict(_INVALID_ZERO), **dims,
                "stored": stored + "the loader divides it by ||K^-1 [x, y, 1]|| with K from attributes.intrinsics",
            }}, "depth"),
            "radial_distance": ({"key": "depth", "meta": {
                "radial_depth": True, "scale_to_meters": 1.0, "range": list(DEPTH_RANGE),
                "invalid": dict(_INVALID_ZERO), **dims, "stored": stored + "returned as stored",
            }}, "radial_distance"),
            "intrinsics": ({"key": "intrinsics", "meta": {
                "camera_model": "pinhole", "dimensions": {"rows": 3, "columns": 3},
                "skew": "non-zero in the scenes rendered with a horizontal lens shift",
            }}, "read_intrinsics"),
            "camera_pose": ({"key": "camera_extrinsics", "meta": {
                "source_frame": "camera_optical", "target_frame": "world", "direction": "target_from_source",
                "dimensions": {"rows": 4, "columns": 4}, "translation_unit": "meters",
            }}, "camera_pose"),
        }
        splits: dict[str, list[str]] = {}
        for scope in modalities:
            modality, function = heads[scope]
            path = write_scope(ctx.output, scope, dataset=family_dataset(self.name, attributes),
                               modality=modality, addons=_addon(function), tree=trees[scope],
                               execution=execution)
            logger.info("wrote %s (%d files)", path, len(trees[scope]))
            if scope in IMAGE_SCOPES:
                splits[scope] = write_splits(ctx.output, scope, split_ids[scope])
        return splits

    @staticmethod
    def _summary(trajectories, stats, scene_cameras, orientation_notes, frame_key_mismatch, details, options):
        by_split = Counter(split for t in trajectories for _, split in t.frames)
        summary: dict[str, Any] = {
            "scenes": len({t.scene for t in trajectories}),
            "trajectories": len(trajectories),
            "frames": sum(by_split.values()),
            "by_split": dict(sorted(by_split.items())),
            "frame_step": options["frame_step"],
            "skipped": details["skipped"],
            "scenes_not_public": details["not_public"],
        }
        tilts = [c.tilt_deg for c in scene_cameras.values()]
        skews = [abs(float(c.intrinsics[0, 1])) for c in scene_cameras.values()]
        summary["cameras"] = {
            "tilted_scenes": sum(1 for v in tilts if v > 0.01),
            "max_tilt_deg": round(max(tilts), 2) if tilts else 0.0,
            "skewed_scenes": sum(1 for v in skews if v > 0.01),
            "max_skew_px": round(max(skews), 3) if skews else 0.0,
            "orientation_notes": orientation_notes,
            "frames_rendered_with_another_camera": details["other_camera"],
            "trajectories_with_frame_indices_not_0_to_n": frame_key_mismatch,
        }
        values = list(stats.values())
        scales = [v["tonemap_scale"] for v in values if "tonemap_scale" in v]
        if scales:
            summary["tonemap"] = {
                "frames": len(scales),
                "frames_without_entity_ids": sum(1 for v in values if v.get("entity_ids") is False),
                "frames_with_nonfinite_color": sum(1 for v in values if v.get("nonfinite_color")),
                "scale_median": round(float(np.median(scales)), 6),
                "dark_frames": len(details["dark_frames"]),
                "dark_frames_kept": bool(details["dark_frames"]) and options["keep_dark_frames"],
                "dark_frames_by_scene": dict(Counter(f.split("/")[0] for f in details["dark_frames"])),
            }
        distances = [v["max_distance"] for v in values if "max_distance" in v]
        if distances:
            invalid = [v["invalid_fraction"] for v in values if "invalid_fraction" in v]
            summary["depth"] = {
                "max_distance_m": round(max(distances), 3),
                "frames_beyond_range": sum(1 for d in distances if d > DEPTH_RANGE[1]),
                "mean_invalid_fraction": round(float(np.mean(invalid)), 6),
                "frames_with_invalid_pixels": sum(1 for f in invalid if f > 0),
            }
        return summary
