"""CEREMA Pixel-Accurate Depth Benchmark: four static scenes recorded in the
CEREMA weather chamber in clear weather, fog and rain, by day and by night, with
stereo and gated cameras, projected lidar, dense laser-scanned ground truth and
the outputs of four benchmarked depth methods.

See ``docs/datasets/cerema.md`` for the layout and the dataset facts it relies on.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from ...archives import selection_fingerprint
from ...artifacts import IndexTree, family_dataset, file_entry, write_scope
from ...media import (
    IN_PLACE,
    MediaError,
    link_top_level,
    materialize_files,
    missing_files,
    resolve_media,
)
from ...recipe import Option, Outcome, Plan, PrepareContext, Recipe, register
from .sources import (
    CALIBRATION,
    FRAME,
    FRAME_FOLDERS,
    SCENE,
    SCENE_FOLDERS,
    FolderSource,
    discover,
)

LOADER_MODULE = "cerema"
SCENES = ("1", "2", "3", "4")
TIMES_OF_DAY = ("day", "night")
#: The benchmark evaluates rgb_left depth inside this crop, clipped to 28 m.
BENCHMARK = {"crop": {"top": 270, "right": 20, "bottom": 20, "left": 170}, "max_depth_m": 28}
RGB_GRID = {"height": 1024, "width": 1920}
GATED_GRID = {"height": 720, "width": 1280}
#: Value ranges of the release, in meters (checked 2026-11-10 on all files):
#: ground truth reaches 58.9 m, lidar 106 m, SGM and PSMNet a few hundred
#: meters. Monodepth's near-zero disparities give far larger outliers.
GROUND_TRUTH_RANGE = [0, 60]
LIDAR_RANGE = [0, 120]
METHOD_RANGE = [0, 500]

_INVALID_ZERO = {"non_finite": False, "sentinel": 0}


def _depth_meta(grid: dict, value_range: list, **extra: Any) -> dict[str, Any]:
    return {"radial_depth": False, "scale_to_meters": 1.0, "range": list(value_range),
            "invalid": dict(_INVALID_ZERO), "dimensions": dict(grid), **extra}


#: scope -> (folder, modality key, head meta, euler-loading function, method or None)
SCOPES: dict[str, tuple[str, str, dict[str, Any], str, Optional[str]]] = {
    "rgb": ("rgb_left_8bit", "rgb", {"range": [0, 255], "camera": "rgb_left", "dimensions": {**RGB_GRID, "channels": 3}}, "rgb", None),
    "rgb_right": ("rgb_right_8bit", "rgb", {"range": [0, 255], "camera": "rgb_right", "dimensions": {**RGB_GRID, "channels": 3}}, "rgb", None),
    "gated0": ("gated0_10bit", "gated", {"bit_depth": 10, "dark_level": 87, "slice": 0, "dimensions": dict(GATED_GRID)}, "gated", None),
    "gated17": ("gated17_10bit", "gated", {"bit_depth": 10, "dark_level": 87, "slice": 17, "dimensions": dict(GATED_GRID)}, "gated", None),
    "gated31": ("gated31_10bit", "gated", {"bit_depth": 10, "dark_level": 87, "slice": 31, "dimensions": dict(GATED_GRID)}, "gated", None),
    "lidar": ("lidar_hdl64_rgb_left", "depth",
              _depth_meta(RGB_GRID, LIDAR_RANGE, sparse=True, camera="rgb_left", sensor="Velodyne HDL64-S3"), "depth", None),
    "lidar_gated": ("lidar_hdl64_gated", "depth",
                    _depth_meta(GATED_GRID, LIDAR_RANGE, sparse=True, camera="gated", sensor="Velodyne HDL64-S3"), "depth", None),
    "depth": ("intermetric_rgb_left", "depth",
              _depth_meta(RGB_GRID, GROUND_TRUTH_RANGE, camera="rgb_left", per_scene=True, benchmark=BENCHMARK), "depth", None),
    "depth_gated": ("intermetric_gated", "depth",
                    _depth_meta(GATED_GRID, GROUND_TRUTH_RANGE, camera="gated", per_scene=True), "depth", None),
    "intrinsics": (CALIBRATION, "intrinsics",
                   {"camera_model": "pinhole", "camera": "rgb_left", "dimensions": {"rows": 3, "columns": 3}}, "read_intrinsics", None),
    "gated_intrinsics": (CALIBRATION, "intrinsics",
                         {"camera_model": "pinhole", "camera": "gated", "dimensions": {"rows": 3, "columns": 3}}, "read_intrinsics", None),
    "sgm": ("sgm", "depth", _depth_meta(RGB_GRID, METHOD_RANGE, sparse=True, camera="rgb_left"), "sgm_depth", "SGM"),
    "psmnet": ("psmnet", "depth", _depth_meta(RGB_GRID, METHOD_RANGE, camera="rgb_left"), "psmnet_depth", "PSMNet"),
    "monodepth": ("monodepth", "depth", _depth_meta(RGB_GRID, METHOD_RANGE, camera="rgb_left"), "monodepth_depth", "Monodepth"),
    "sparse2dense": ("sparse2dense", "depth", _depth_meta(RGB_GRID, METHOD_RANGE, camera="rgb_left"), "depth", "Sparse-to-Dense"),
}
CALIBRATION_FILE = {"intrinsics": "calib_cam_stereo_left.json", "gated_intrinsics": "calib_gated_bwv.json"}


def _weather_selected(weather: str, wanted: tuple[str, ...]) -> bool:
    return not wanted or weather in wanted or re.match(r"[a-z]+", weather).group(0) in wanted


def _weather_order(weather: str) -> tuple[int, int]:
    """clear, then fog by visibility, then rain by rate."""
    match = re.fullmatch(r"([a-z]+?)(\d*)", weather)
    kind, level = match.group(1), int(match.group(2) or 0)
    return ({"clear": 0, "fog": 1, "rain": 2}.get(kind, 3), level)


def condition_attributes(weather: str) -> dict[str, int]:
    """Per-frame attributes of a weather name: fog visibility or rain rate."""
    match = re.fullmatch(r"(fog|rain)(\d+)", weather)
    if not match:
        return {}
    return {"fog_visibility_m" if match.group(1) == "fog" else "rain_mm_h": int(match.group(2))}


@register
class Cerema(Recipe):
    name = "cerema"
    title = "CEREMA Pixel-Accurate Depth Benchmark"
    summary = (
        "Stereo and gated cameras, projected lidar and dense laser-scanned ground truth "
        "of four static scenes in clear weather, fog and rain, day and night; plus the "
        "outputs of four benchmarked depth methods."
    )
    homepage = "https://github.com/gruberto/PixelAccurateDepthBenchmark"
    data_license = (
        "No licence file in the release; the terms are those accepted with the download "
        "request. The benchmark code is MIT-licensed."
    )
    expected_inputs = (
        "pixel_accurate_depth_benchmark.zip as downloaded (zips nested inside a zip)",
        "or its inner <folder>.zip archives",
        "or their extracted folders (rgb_left_8bit/, intermetric_rgb_left/, calibration/, ...)",
    )
    layout_version = 1
    options = (
        Option("modalities", "Scopes to prepare; all = every scope whose files are found.",
               default=(), multiple=True, choices=tuple(SCOPES)),
        Option("scenes", "Scenes to prepare (1-4).", default=(), multiple=True, choices=SCENES),
        Option("time_of_day", "day and/or night.", default=(), multiple=True, choices=TIMES_OF_DAY),
        Option("weather", "Conditions: clear, fog, rain, or one level such as fog40 or rain15.",
               default=(), multiple=True),
    )

    def detect(self, source: Path) -> bool:
        try:
            found, _ = discover(Path(source))
        except OSError:
            return False
        return "rgb_left_8bit" in found or "intermetric_rgb_left" in found

    # -- planning -----------------------------------------------------------

    def plan(self, ctx: PrepareContext) -> Plan:
        found, notes = discover(ctx.source)
        if not found:
            raise FileNotFoundError(
                f"no CEREMA Pixel-Accurate Depth Benchmark downloads in {ctx.source}; expected "
                + "; ".join(self.expected_inputs)
            )
        requested = list(ctx.options["modalities"])
        available = [
            scope for scope, (folder, *_rest) in SCOPES.items()
            if folder in found and (folder != CALIBRATION or CALIBRATION_FILE[scope] in found[folder].names)
        ]
        missing = [scope for scope in requested if scope not in available]
        if missing:
            raise FileNotFoundError(
                f"no files for scope(s) {', '.join(missing)} in {ctx.source}; found "
                f"{', '.join(available) or 'none'}"
            )
        scopes = requested or available

        scenes = tuple(ctx.options["scenes"])
        times = tuple(ctx.options["time_of_day"])
        weathers = tuple(ctx.options["weather"])
        # Every frame of the release, from all frame folders; ground truth applies to each.
        frames = sorted({
            name.rsplit(".", 1)[0]
            for folder in FRAME_FOLDERS if folder in found
            for name in found[folder].names
        })
        selected_frames = []
        for stem in frames:
            match = FRAME.match(stem + ".png")
            if ((not scenes or match.group("scene") in scenes)
                    and (not times or match.group("time_of_day") in times)
                    and _weather_selected(match.group("weather"), weathers)):
                selected_frames.append(stem)
        if not selected_frames and any(SCOPES[s][0] != CALIBRATION for s in scopes):
            raise FileNotFoundError("no frames match the selected scenes, time of day and weather")

        folders = sorted({SCOPES[scope][0] for scope in scopes})
        inputs = []
        for folder in folders:
            src = found[folder]
            item: dict[str, Any] = {"folder": folder, "files": len(src.names)}
            if src.extracted:
                item["extracted"] = src.extracted
            if src.archive is not None:
                item["archive"] = src.archive_label
            inputs.append(item)
        steps = [f"note: {note}" for note in notes]
        steps.append(f"prepare {len(selected_frames)} frames: scopes {', '.join(scopes)}")
        steps.append(f"bring the files into the output (media={ctx.media})")
        steps.append("write ds-crawler scopes with prebuilt indexes")
        return Plan(recipe=self.name, inputs=inputs, steps=steps,
                    details={"found": found, "notes": notes, "scopes": scopes, "frames": selected_frames})

    # -- preparing ----------------------------------------------------------

    def prepare(self, ctx: PrepareContext, plan: Plan) -> Outcome:
        found: dict[str, FolderSource] = plan.details["found"]
        scopes: list[str] = plan.details["scopes"]
        frames: list[str] = plan.details["frames"]
        scenes_needed = sorted({FRAME.match(f + ".png").group("scene") for f in frames})

        frame_set = set(frames)
        needed: dict[str, list[str]] = defaultdict(list)  # folder -> file names
        for scope in scopes:
            folder = SCOPES[scope][0]
            names = found[folder].names
            if folder in FRAME_FOLDERS:
                wanted = [n for n in names if n.rsplit(".", 1)[0] in frame_set]
            elif folder in SCENE_FOLDERS:
                wanted = [n for n in names if SCENE.match(n).group("scene") in scenes_needed]
            else:
                wanted = [CALIBRATION_FILE[scope]]
            needed[folder] = sorted(set(needed[folder]) | set(wanted))

        media, bases = self._materialize(ctx, found, needed)

        attributes = {"source": "Pixel-Accurate Depth Benchmark (CEREMA)",
                      "scenes": scenes_needed,
                      "time_of_day": sorted({FRAME.match(f + ".png").group("time_of_day") for f in frames}),
                      "weather": sorted({FRAME.match(f + ".png").group("weather") for f in frames},
                                        key=_weather_order)}
        execution = {"recipe": self.name, "layout_version": self.layout_version,
                     "options": {k: list(v) if isinstance(v, tuple) else v for k, v in ctx.options.items()}}
        files: dict[str, int] = {}
        missing: dict[str, int] = {}
        for scope in scopes:
            folder, key, meta, function, method = SCOPES[scope]
            location = bases[folder]  # the folder's directory, relative to the output root
            tree = IndexTree()
            if folder == CALIBRATION:
                tree.add((), file_entry(f"{location}/{CALIBRATION_FILE[scope]}", scope))
            else:
                present = set(needed[folder])
                skipped = 0
                for stem in frames:
                    match = FRAME.match(stem + ".png")
                    scene, tod, weather, index = match.group("scene", "time_of_day", "weather", "index")
                    name = f"scene{scene}.npz" if folder in SCENE_FOLDERS else f"{stem}.{'png' if folder.endswith('bit') else 'npz'}"
                    if name not in present:
                        skipped += 1
                        continue
                    tree.add(
                        (f"scene:{scene}", f"time_of_day:{tod}", f"weather:{weather}"),
                        file_entry(f"{location}/{name}", f"frame-{index}", condition_attributes(weather)),
                    )
                if skipped:
                    missing[scope] = skipped
            dataset_attributes = dict(attributes)
            if method:
                dataset_attributes.update({"gt": False, "baseline": True, "method": method})
            write_scope(
                ctx.output,
                scope,
                dataset=family_dataset(self.name, dataset_attributes),
                modality={"key": key, "meta": meta},
                addons={"euler_loading": {"version": "1.0", "loader": LOADER_MODULE, "function": function}},
                tree=tree,
                execution=execution,
            )
            files[scope] = len(tree)

        summary: dict[str, Any] = {
            "files": files,
            "frames": len(frames),
            "conditions": attributes,
            "notes": plan.details["notes"],
            "lidar": "projected depth maps in the rgb_left and gated grids; no point clouds in the release",
            "baselines": {s: SCOPES[s][4] for s in scopes if SCOPES[s][4]},
        }
        if missing:
            summary["frames_without_file"] = missing
        return Outcome(media=media, scopes=scopes, splits={}, summary=summary)

    @staticmethod
    def _materialize(
        ctx: PrepareContext, found: dict[str, FolderSource], needed: dict[str, list[str]]
    ) -> tuple[str, dict[str, str]]:
        """Bring the needed files into the output; return the media mode and,
        per folder, the directory holding it relative to the output root."""
        extracted = all(found[f].extracted for f in needed)
        archived = all(found[f].archive is not None for f in needed)
        mode = resolve_media(ctx, extracted=extracted, archives=archived)
        bases: dict[str, str] = {}
        for folder, names in sorted(needed.items()):
            src = found[folder]
            if mode == IN_PLACE and src.extracted:
                bases[folder] = src.extracted
                continue
            bases[folder] = folder
            if mode in ("symlink", "hardlink", "copy"):
                location = PurePosixPath(src.extracted)
                root = ctx.source.joinpath(*location.parent.parts) if location.parent.parts else ctx.source
                missing = missing_files(ctx.source / src.extracted, names)
                if missing:
                    raise MediaError(f"{len(missing)} needed files are not in {ctx.source / src.extracted} (e.g. {missing[0]})")
                if location.name != folder:  # e.g. calib/ for calibration/
                    raise MediaError(f"cannot place {src.extracted}/ as {folder}/; rename it, "
                                     "or prepare from the archives with --media extract")
                if mode == "symlink":
                    link_top_level(root, ctx.output, [folder])
                else:
                    materialize_files(root, ctx.output, [f"{folder}/{n}" for n in names], mode)
                continue
            # Extract from the archive (in place: into the source).
            destination = ctx.source if mode == IN_PLACE else ctx.output
            if src.archive is None:
                raise MediaError(f"{folder} has no archive to extract from in {ctx.source}")
            if not src.member_prefix:  # calibration.zip: files at its root
                destination = destination / folder
            elif src.member_prefix != f"{folder}/":
                raise MediaError(f"{src.archive_label} keeps {folder} files below {src.member_prefix}, "
                                 f"not {folder}/")
            selection = {src.member(n) for n in names}
            ctx.extract(src.archive, destination=destination, select=selection.__contains__,
                        fingerprint=selection_fingerprint(selection), inner=src.inner)
        for folder, names in needed.items():
            root = ctx.output if mode != IN_PLACE else ctx.source
            missing = missing_files(root / bases[folder], names)
            if missing:
                raise MediaError(f"{len(missing)} files are still missing below {root / bases[folder]} "
                                 f"(e.g. {missing[0]})")
        return mode, bases
