"""SHIFT (discrete view, front camera): synthetic CARLA drives in fixed weather
and light, with planar depth, semantic labels, intrinsics and camera poses.

See ``docs/datasets/shift.md`` for the layout and the dataset facts it relies on.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

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
    resolve_media,
)
from ...recipe import Option, Outcome, Plan, PrepareContext, Recipe, register
from .camera import HEIGHT, WIDTH, calibration_document
from .labels import SKYCLASS, label_set_meta
from .selection import (
    IMAGE_RATE_STEP,
    SPLITS,
    TIMES_OF_DAY,
    WEATHERS,
    Selection,
    select,
    selection_parameters,
)
from .sources import ShiftSource, discover

logger = logging.getLogger(__name__)

LOADER_MODULE = "shift"
#: Planar depth in CARLA's 24-bit code; the code 2**24 - 1 is the 1000 m far plane.
DEPTH_FAR_PLANE_M = 1000.0
DEPTH_SCALE_TO_METERS = DEPTH_FAR_PLANE_M / (2**24 - 1)
#: The ego vehicle's hood, the same in every frame (checked on 100 frames of 100 sequences).
EGO_HOOD = ("an arc over the bottom rows: from row 770 at the image centre and row 787 a quarter "
            "width to either side, not reaching the sides; 1.0-1.2 m away")
GRID = {"height": HEIGHT, "width": WIDTH}
#: Per-frame scope -> the release group holding its files.
FRAME_SCOPES = {"rgb": "img", "depth": "depth", "semantic_segmentation": "semseg"}
SCOPES = (*FRAME_SCOPES, "intrinsics", "camera_pose")
#: ``det_2d.json`` attributes copied onto every frame's file entries (as numbers).
CONDITION_ATTRIBUTES = ("fog_density", "fog_distance", "fog_falloff", "precipitation",
                        "precipitation_deposits", "cloudiness", "wind_intensity", "sun_altitude_angle")


def _number(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def heads(decoded: bool) -> dict[str, tuple[dict[str, Any], str]]:
    """Scope -> (modality, euler-loading function); *decoded*: frames decoded from the videos."""
    rgb_stored = ("PNG of frame n of the sequence's MPEG-4 video, decoded with ffmpeg" if decoded
                  else "JPEG as released")
    return {
        "rgb": ({"key": "rgb", "meta": {
            "range": [0, 255], "dimensions": {**GRID, "channels": 3}, "stored": rgb_stored, "ego_hood": EGO_HOOD,
        }}, "rgb"),
        "depth": ({"key": "depth", "meta": {
            "radial_depth": False, "scale_to_meters": DEPTH_SCALE_TO_METERS, "range": [0, DEPTH_FAR_PLANE_M],
            "invalid": {"non_finite": False, "sentinel": None}, "encoding": "carla_rgb24",
            "far_plane_m": DEPTH_FAR_PLANE_M, "dimensions": dict(GRID), "ego_hood": EGO_HOOD,
            "stored": "8-bit RGB PNG, (R + 256 G + 65536 B) / (2**24 - 1) * 1000 m; the sky and anything "
                      "beyond the far plane are code 16777210",
        }}, "depth"),
        "semantic_segmentation": ({"key": "semantic_segmentation", "meta": {
            "skyclass": list(SKYCLASS), "encoding": "id_in_red", "dimensions": dict(GRID),
            **label_set_meta(decoded),
        }}, "semantic_segmentation"),
        "intrinsics": ({"key": "intrinsics", "meta": {
            "camera_model": "pinhole", "dimensions": {"rows": 3, "columns": 3}, "image_size": [HEIGHT, WIDTH],
        }}, "read_intrinsics"),
        "camera_pose": ({"key": "camera_extrinsics", "meta": {
            "source_frame": "camera_optical", "target_frame": "world", "direction": "target_from_source",
            "dimensions": {"rows": 4, "columns": 4}, "translation_unit": "meters",
            "world": "CARLA's world with y negated (right-handed, z up)",
        }}, "camera_pose"),
    }


@register
class Shift(Recipe):
    name = "shift"
    title = "SHIFT"
    summary = (
        "SHIFT's discrete set, front camera: CARLA drives in fixed weather and light with planar "
        "depth, semantic labels, intrinsics and camera poses; frames chosen by condition, spacing "
        "and ego motion; official train/val splits."
    )
    homepage = "https://www.vis.xyz/shift"
    data_license = "CC BY-NC-SA 4.0 (SHIFT, per the shift-dev repository)"
    expected_inputs = (
        "discrete/images/<split>/front/{img,semseg,depth}.zip with det_2d.json and seq.csv, as "
        "shift-dev's download.py stores the image-rate set",
        "or their extracted folders img/, semseg/, depth/ (<sequence>/<frame>_<group>_front.<ext>)",
        "or frames decoded from the video-rate set into that layout by slurm/tools/shift_decode.py; "
        "det_2d.json and seq.csv may then stay in discrete/videos/<split>/front/",
    )
    layout_version = 1
    options = (
        Option("modalities", "Scopes to prepare; all = every scope whose files are found.",
               default=(), multiple=True, choices=SCOPES),
        Option("splits", "Official splits to prepare; all = every split found.",
               default=(), multiple=True, choices=SPLITS),
        Option("weather", "Coarse weather classes to keep; all by default.",
               default=(), multiple=True, choices=WEATHERS),
        Option("time_of_day", "Coarse times of day to keep; all by default.",
               default=(), multiple=True, choices=TIMES_OF_DAY),
        Option("frame_step", "Keep the frames whose number is a multiple of this; 10 keeps every "
               "image-rate frame (1 fps), 50 one every 5 s.", default=IMAGE_RATE_STEP, type=int),
        Option("min_motion_m", "Keep a frame only if the camera moved this far since the last kept "
               "frame of its sequence, or turned min_rotation_deg; 0 = off.", default=0.0, type=float),
        Option("min_rotation_deg", "Keep a frame only if the camera's yaw changed this much since the last "
               "kept frame, or it moved min_motion_m; 0 = off.", default=0.0, type=float),
    )

    def detect(self, source: Path) -> bool:
        try:
            return bool(discover(Path(source)).splits)
        except OSError:
            return False

    # -- planning -----------------------------------------------------------

    def plan(self, ctx: PrepareContext) -> Plan:
        found = discover(ctx.source)
        if not found.splits:
            raise FileNotFoundError(f"no SHIFT front-camera frames in {ctx.source}; expected "
                                    + "; ".join(self.expected_inputs))
        options = ctx.options
        splits = [s for s in (options["splits"] or SPLITS) if s in found.splits]
        if not splits:
            raise FileNotFoundError(f"none of the splits {list(options['splits'])} in {ctx.source}; "
                                    f"found {', '.join(found.splits)}")
        for split in splits:
            src = found.splits[split]
            if src.det2d is None or src.seq_csv is None:
                raise FileNotFoundError(f"{split}: needs det_2d.json and seq.csv in {src.folder} or "
                                        f"discrete/videos/{split}/front")
        available = [s for s in SCOPES if s not in FRAME_SCOPES
                     or any(FRAME_SCOPES[s] in found.splits[split].groups for split in splits)]
        missing = [s for s in options["modalities"] if s not in available]
        if missing:
            raise FileNotFoundError(f"no files for scope(s) {', '.join(missing)}; found {', '.join(available)}")
        scopes = [s for s in SCOPES if s in (options["modalities"] or available)]
        if options["frame_step"] <= 0 or options["frame_step"] % IMAGE_RATE_STEP:
            raise ValueError(f"frame_step must be a positive multiple of {IMAGE_RATE_STEP}, "
                             f"got {options['frame_step']}")
        parameters = selection_parameters(
            weather=options["weather"], time_of_day=options["time_of_day"], frame_step=options["frame_step"],
            min_motion_m=options["min_motion_m"], min_rotation_deg=options["min_rotation_deg"])
        steps = [f"note: {note}" for note in found.notes]
        for split in splits:
            decoded = found.splits[split].decoded
            if decoded is not None and decoded.get("selection") != parameters:
                steps.append(f"note: {split}: its frames were decoded for selection {decoded.get('selection')}; "
                             f"this run selects {parameters}, and frames without files are left out")
        steps += [
            f"select frames of {', '.join(splits)}: {parameters}",
            "keep the frames that have a file in every selected per-frame scope",
            f"bring the files into the output (media={ctx.media})",
            "write calibration/<split>/<sequence>.json: intrinsics and per-frame camera poses",
            f"write ds-crawler scopes {', '.join(scopes)} and the official splits",
        ]
        return Plan(recipe=self.name, inputs=found.describe(), steps=steps,
                    details={"found": found, "splits": splits, "scopes": scopes, "parameters": parameters})

    # -- preparing ----------------------------------------------------------

    def prepare(self, ctx: PrepareContext, plan: Plan) -> Outcome:
        found: ShiftSource = plan.details["found"]
        scopes: list[str] = plan.details["scopes"]
        parameters: dict[str, Any] = plan.details["parameters"]
        groups = sorted({FRAME_SCOPES[s] for s in scopes if s in FRAME_SCOPES})
        selections: dict[str, Selection] = {}
        complete: dict[str, dict[str, list[int]]] = {}
        dropped: dict[str, dict[str, list[str]]] = {}
        for split in plan.details["splits"]:
            src = found.splits[split]
            selection = select(ctx.source / src.seq_csv, ctx.source / src.det2d, split,
                               weather=parameters["weather"], time_of_day=parameters["time_of_day"],
                               frame_step=parameters["frame_step"], min_motion_m=parameters["min_motion_m"],
                               min_rotation_deg=parameters["min_rotation_deg"])
            selections[split] = selection
            complete[split], dropped[split] = {}, {}
            for sequence, frames in sorted(selection.frames.items()):
                kept = []
                for frame in frames:
                    absent = [g for g in groups
                              if g not in src.groups or (sequence, frame) not in src.groups[g].files]
                    for g in absent:
                        dropped[split].setdefault(g, []).append(f"{sequence}/{frame:08d}")
                    if not absent:
                        kept.append(frame)
                if kept:
                    complete[split][sequence] = kept
            logger.info("%s: %s; %d frames have every file", split, selection.counts,
                        sum(len(v) for v in complete[split].values()))
        complete = {split: sequences for split, sequences in complete.items() if sequences}
        if not complete:
            raise MediaError("no selected frame has a file in every selected scope")

        media, paths = self._materialize(ctx, found, complete, groups)

        decoded = any(found.splits[s].decoded is not None for s in complete)
        scope_heads = heads(decoded)
        attributes = {"source": "SHIFT", "set": "discrete", "camera": "front", "splits": sorted(complete),
                      "selection": parameters, "decoded_from_videos": decoded}
        execution = {"recipe": self.name, "layout_version": self.layout_version,
                     "options": {k: list(v) if isinstance(v, tuple) else v for k, v in ctx.options.items()}}
        trees = {scope: IndexTree() for scope in scopes}
        split_ids: dict[str, dict[str, list[tuple]]] = {s: {} for s in scopes if s in FRAME_SCOPES}
        for split, sequences in sorted(complete.items()):
            selection = selections[split]
            for sequence, frames in sorted(sequences.items()):
                info = selection.sequences[sequence]
                keys = info.hierarchy
                for frame in frames:
                    conditions = selection.records[sequence][frame].get("attributes") or {}
                    file_attributes = {
                        "town": info.town, "weather_fine": info.weather_fine,
                        "time_of_day_fine": info.time_of_day_fine,
                        **{k: _number(conditions[k]) for k in CONDITION_ATTRIBUTES if k in conditions},
                    }
                    for scope in split_ids:
                        path = paths[(split, FRAME_SCOPES[scope], sequence, frame)]
                        trees[scope].add(keys, file_entry(path, f"{frame:08d}", file_attributes))
                        split_ids[scope].setdefault(split, []).append(keys + (f"{frame:08d}",))
                if {"intrinsics", "camera_pose"} & set(scopes):
                    relpath = f"calibration/{split}/{sequence}.json"
                    document = calibration_document(info, {f: selection.records[sequence][f] for f in frames})
                    write_json_atomic(ctx.output / relpath, document, compact=True)
                    for scope in ("intrinsics", "camera_pose"):
                        if scope in trees:
                            trees[scope].add(keys, file_entry(relpath, scope))

        splits_written: dict[str, list[str]] = {}
        for scope in scopes:
            modality, function = scope_heads[scope]
            write_scope(
                ctx.output,
                scope,
                dataset=family_dataset(self.name, attributes),
                modality=modality,
                addons={"euler_loading": {"version": "1.0", "loader": LOADER_MODULE, "function": function}},
                tree=trees[scope],
                execution=execution,
            )
            if scope in split_ids:
                splits_written[scope] = write_splits(ctx.output, scope, split_ids[scope])
            logger.info("%s: %d files", scope, len(trees[scope]))

        summary: dict[str, Any] = {"files": {scope: len(tree) for scope, tree in trees.items()}, "splits": {}}
        for split, selection in selections.items():
            frames = complete.get(split, {})
            summary["splits"][split] = {
                **selection.counts,
                "frames": sum(len(v) for v in frames.values()),
                "sequences_with_frames": len(frames),
                "frames_by_condition": dict(sorted(Counter(
                    f"{selection.sequences[s].weather}/{selection.sequences[s].time_of_day}"
                    for s, fs in frames.items() for _ in fs).items())),
            }
            if dropped[split]:
                summary["splits"][split]["frames_without_file"] = {
                    g: {"count": len(ids), "frames": ids[:20]} for g, ids in sorted(dropped[split].items())}
        if found.notes:
            summary["notes"] = found.notes
        return Outcome(media=media, scopes=scopes, splits=splits_written, summary=summary)

    @staticmethod
    def _materialize(
        ctx: PrepareContext, found: ShiftSource, complete: dict[str, dict[str, list[int]]], groups: list[str]
    ) -> tuple[str, dict[tuple[str, str, str, int], str]]:
        """Bring the needed files into the output. Returns the media mode and each
        file's path relative to the output root, by (split, group, sequence, frame)."""
        sources = [(split, found.splits[split].groups[g]) for split in complete for g in groups]
        mode = resolve_media(ctx, extracted=all(g.extracted for _, g in sources),
                             archives=all(g.archive is not None for _, g in sources))
        root = ctx.source if mode == IN_PLACE else ctx.output
        paths: dict[tuple[str, str, str, int], str] = {}
        for split, src in sources:
            wanted = [(sequence, f) for sequence, frames in sorted(complete[split].items()) for f in frames]
            if src.extracted and mode in (IN_PLACE, "symlink", "hardlink", "copy"):
                relpaths = [f"{src.extracted}/{src.name(sequence, f)}" for sequence, f in wanted]
                paths.update({(split, src.group, s, f): rel for (s, f), rel in zip(wanted, relpaths)})
                if mode in ("hardlink", "copy"):
                    materialize_files(ctx.source, ctx.output, relpaths, mode)
                continue
            if src.archive is None:
                raise MediaError(f"{split}/{src.group} has no zip to extract from in {ctx.source}")
            folder = f"{found.splits[split].folder}/{src.group}"
            members = {(s, f): src.member(s, f) for s, f in wanted}
            paths.update({(split, src.group, s, f): f"{folder}/{m}" for (s, f), m in members.items()})
            selection = set(members.values())
            ctx.extract(src.archive, destination=root / folder, select=selection.__contains__,
                        fingerprint=selection_fingerprint(selection))
        if mode == "symlink":
            link_top_level(ctx.source, ctx.output, sorted({p.split("/", 1)[0] for p in paths.values()}))
        absent = missing_files(root, sorted(paths.values()))
        if absent:
            raise MediaError(f"{len(absent)} files are still missing below {root} (e.g. {absent[0]})")
        return mode, paths
