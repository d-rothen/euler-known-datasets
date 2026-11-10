"""ScatterNeRF: rectified stereo frames of foggy drives with per-image
intrinsics and camera-to-world poses (no lidar in the release).

See ``docs/datasets/scatternerf.md`` for the layout and the dataset facts it
relies on.
"""

from __future__ import annotations

import datetime
import logging
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ...archives import selection_fingerprint
from ...artifacts import IndexTree, family_dataset, file_entry, write_scope
from ...media import IN_PLACE, MediaError, missing_files, place_extracted, resolve_media
from ...recipe import Option, Outcome, Plan, PrepareContext, Recipe, register
from .sources import EXTENSIONS, MEMBER, SequenceSource, discover, open_chain

logger = logging.getLogger(__name__)

#: The stereo baseline of the SeeingThroughFog rig, whose rectified images
#: ScatterNeRF uses (identical fx = 2355.722801): P[0, 3] = -fx * b in its
#: calib_cam_stereo_right.json.
BASELINE_M = 478.200589 / 2355.722801
CAMERAS = ("left", "right")
LOADER_MODULE = "scatternerf"
SCOPES = {"rgb": "rgb", "intrinsics": "intrinsics", "camera_pose": "pose"}


def _addon(function: str) -> dict[str, Any]:
    return {"euler_loading": {"version": "1.0", "loader": LOADER_MODULE, "function": function}}


def _frame_key(match: "re.Match[str]") -> tuple[str, str, str]:
    """``(camera, frame, timestamp)`` of a matched member name."""
    return match.group("camera"), match.group("frame"), match.group("timestamp")


def _matrix(text: str) -> np.ndarray:
    return np.array(text.split(), dtype=np.float64).reshape(4, 4)


def stereo_scale(poses: dict[tuple[str, str, str], np.ndarray]) -> dict[str, Any]:
    """Metric scale of one sequence from its stereo pairs.

    *poses* maps ``(camera, frame, timestamp)`` to the stored camera-to-world
    pose. A pair is a left and right pose with the same frame and timestamp;
    the right camera's position in the left camera's axes is the baseline.
    """
    baselines = []
    for (camera, frame, timestamp), left in poses.items():
        if camera != "left":
            continue
        right = poses.get(("right", frame, timestamp))
        if right is None:
            continue
        offset = (np.linalg.inv(left) @ right)[:3, 3]
        baselines.append(offset)
    if not baselines:
        return {"stereo_pairs": 0}
    offsets = np.array(baselines)
    lengths = np.linalg.norm(offsets, axis=1)
    baseline = float(np.median(lengths))
    return {
        "stereo_pairs": len(baselines),
        "baseline_units": round(baseline, 6),
        "baseline_spread": round(float(np.std(lengths) / baseline), 4),
        # A rectified pair is offset along the left camera's x axis.
        "baseline_along_x": round(float(np.median(offsets[:, 0] / lengths)), 4),
        "scale_to_meters": round(BASELINE_M / baseline, 6),
    }


@register
class ScatterNeRF(Recipe):
    name = "scatternerf"
    title = "ScatterNeRF"
    summary = (
        "Rectified stereo frames of foggy drives with per-image intrinsics and "
        "camera-to-world poses; no lidar."
    )
    homepage = ""
    data_license = (
        "Research use under the terms accepted with the download request; the "
        "archives carry no licence file"
    )
    expected_inputs = (
        "sequence_<n>.zip archives as published (they may wrap one another; see the docs)",
        "or their extracted <n>/{rgb,intrinsics,pose}/ folders",
    )
    layout_version = 1
    options = (
        Option(
            "sequences",
            "Sequences to prepare, by the number of their folder; all = every one found.",
            default=(),
            multiple=True,
        ),
        Option("cameras", "Stereo cameras to prepare.", default=CAMERAS, multiple=True, choices=CAMERAS),
    )

    def detect(self, source: Path) -> bool:
        try:
            return any(discover(Path(source))[0])
        except (OSError, zipfile.BadZipFile, ValueError):
            return False

    def plan(self, ctx: PrepareContext) -> Plan:
        found, notes = discover(ctx.source)
        if not found:
            raise FileNotFoundError(
                f"no ScatterNeRF downloads in {ctx.source}; expected "
                + "; ".join(self.expected_inputs)
            )
        requested = [str(int(s)) for s in ctx.options["sequences"]]
        missing = sorted(set(requested) - set(found), key=int)
        if missing:
            raise FileNotFoundError(
                f"sequence(s) {', '.join(missing)} are not in {ctx.source}; found "
                f"{', '.join(sorted(found, key=int))}"
            )
        selected = {s: found[s] for s in sorted(requested or found, key=int)}
        inputs = []
        for sequence, src in selected.items():
            item = {"sequence": sequence, "files": len(src.members)}
            if src.archive is not None:
                item["archive"] = src.archive.name
                if src.chain:
                    item["inside"] = " > ".join(src.chain)
                if len(src.archives) > 1:
                    item["duplicate_archives"] = sorted(set(src.archives) - {src.archive.name})
            item["extracted"] = src.extracted
            inputs.append(item)
        steps = [f"note: {note}" for note in notes]
        steps.append(f"prepare sequences {', '.join(selected)} for cameras {', '.join(ctx.options['cameras'])}")
        steps.append(f"bring the files into the output (media={ctx.media})")
        steps.append("derive each sequence's metric scale from its stereo baseline")
        steps.append("write ds-crawler scopes rgb, intrinsics and camera_pose")
        return Plan(recipe=self.name, inputs=inputs, steps=steps,
                    details={"sequences": selected, "notes": notes})

    def prepare(self, ctx: PrepareContext, plan: Plan) -> Outcome:
        sequences: dict[str, SequenceSource] = plan.details["sequences"]
        cameras = set(ctx.options["cameras"])
        # Only frames with an image, intrinsics and pose are prepared.
        wanted: dict[str, list[str]] = {}
        incomplete: dict[str, int] = {}
        for sequence, src in sequences.items():
            kinds: dict[tuple[str, str, str], set[str]] = defaultdict(set)
            for name in src.members:
                match = MEMBER.match(name)
                if match.group("camera") in cameras:
                    kinds[_frame_key(match)].add(match.group("kind"))
            complete = {key for key, found in kinds.items() if found == set(EXTENSIONS)}
            incomplete[sequence] = len(kinds) - len(complete)
            if incomplete[sequence]:
                logger.warning("sequence %s: %d frames lack an image, intrinsics or pose; "
                               "skipping them", sequence, incomplete[sequence])
            wanted[sequence] = sorted(
                name for name in src.members if _frame_key(MEMBER.match(name)) in complete
            )
        media = self._materialize(ctx, sequences, wanted)
        root = ctx.source if media == IN_PLACE else ctx.output

        trees = {scope: IndexTree() for scope in SCOPES}
        summary: dict[str, Any] = {"sequences": {}, "notes": plan.details["notes"],
                                   "lidar": "not part of the ScatterNeRF release"}
        for sequence, names in wanted.items():
            # The scale comes from every stereo pair, whichever cameras are prepared.
            poses = {}
            for name in sequences[sequence].members:
                match = MEMBER.match(name)
                if match.group("kind") == "pose":
                    poses[_frame_key(match)] = _matrix(self._read(ctx, sequences[sequence], root, name))
            scale = stereo_scale(poses)
            if not scale["stereo_pairs"]:
                logger.warning("sequence %s has no stereo pairs; poses get no metric scale", sequence)
            elif abs(scale["baseline_along_x"]) < 0.95 or scale["baseline_spread"] > 0.1:
                logger.warning("sequence %s: stereo pairs are not a clean x baseline: %s", sequence, scale)

            frames = sorted({_frame_key(MEMBER.match(name)) for name in names})
            counts = {camera: 0 for camera in sorted(cameras)}
            for camera, frame, timestamp in frames:
                keys = (f"sequence:{sequence}", f"camera:{camera}")
                file_id = f"frame-{frame}"
                stem = f"{camera}_{frame}_{timestamp}"
                time_attr = {"timestamp_ns": int(timestamp)}
                trees["rgb"].add(keys, file_entry(f"{sequence}/rgb/{stem}.png", file_id, time_attr))
                trees["intrinsics"].add(keys, file_entry(f"{sequence}/intrinsics/{stem}.txt", file_id, time_attr))
                pose_attr = dict(time_attr)
                if "scale_to_meters" in scale:
                    pose_attr["scale_to_meters"] = scale["scale_to_meters"]
                trees["camera_pose"].add(keys, file_entry(f"{sequence}/pose/{stem}.txt", file_id, pose_attr))
                counts[camera] += 1
            stamps = sorted(int(key[2]) for key in frames)
            summary["sequences"][sequence] = {
                "frames": counts,
                "incomplete_frames": incomplete[sequence],
                "recorded_utc": datetime.datetime.fromtimestamp(
                    stamps[0] / 1e9, tz=datetime.timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S") if stamps else None,
                **scale,
            }

        attributes = {"source": "ScatterNeRF", "sequences": list(wanted), "cameras": sorted(cameras)}
        execution = {"recipe": self.name, "layout_version": self.layout_version,
                     "options": {k: list(v) if isinstance(v, tuple) else v for k, v in ctx.options.items()}}
        heads = {
            "rgb": ({"key": "rgb", "meta": {"range": [0, 255]}}, "rgb"),
            "intrinsics": (
                {"key": "intrinsics",
                 "meta": {"camera_model": "pinhole", "dimensions": {"rows": 3, "columns": 3}}},
                "read_intrinsics",
            ),
            "camera_pose": (
                {"key": "camera_extrinsics",
                 "meta": {"source_frame": "camera_optical", "target_frame": "world",
                          "direction": "target_from_source", "dimensions": {"rows": 4, "columns": 4}}},
                "camera_pose",
            ),
        }
        for scope, (modality, function) in heads.items():
            path = write_scope(
                ctx.output,
                scope,
                dataset=family_dataset(self.name, attributes),
                modality=modality,
                addons=_addon(function),
                tree=trees[scope],
                execution=execution,
            )
            logger.info("wrote %s (%d files)", path, len(trees[scope]))
        return Outcome(media=media, scopes=list(SCOPES), splits={}, summary=summary)

    @staticmethod
    def _read(ctx: PrepareContext, src: SequenceSource, root: Path, name: str) -> str:
        """Read *name* from the output, else the extracted source, else its archive.

        Poses of a camera that is not prepared are only in the source.
        """
        for base in (root, ctx.source) if src.extracted else (root,):
            try:
                return (base / name).read_text()
            except FileNotFoundError:
                continue
        if src.archive is None:
            raise MediaError(f"{name} is neither extracted nor in an archive below {ctx.source}")
        with open_chain(src.archive, src.chain) as zf:
            return zf.read(name).decode("utf-8")

    @staticmethod
    def _materialize(
        ctx: PrepareContext, sequences: dict[str, SequenceSource], wanted: dict[str, list[str]]
    ) -> str:
        extracted = all(src.extracted for src in sequences.values())
        archived = all(src.archive is not None for src in sequences.values())
        mode = resolve_media(ctx, extracted=extracted, archives=archived)
        needed = sorted(name for names in wanted.values() for name in names)
        if mode in ("symlink", "hardlink", "copy"):
            place_extracted(ctx, needed, mode=mode, top_level=list(sequences))
            return mode
        destination = ctx.source if mode == IN_PLACE else ctx.output
        for sequence, src in sequences.items():
            if mode == IN_PLACE and src.extracted:
                continue
            if src.archive is None:
                raise MediaError(f"sequence {sequence} has no archive to extract from")
            # A nested copy is read through its wrapping archives in place.
            selection = set(wanted[sequence])
            ctx.extract(src.archive, destination=destination, select=selection.__contains__,
                        fingerprint=selection_fingerprint(selection), inner=src.chain)
        missing = missing_files(destination, needed)
        if missing:
            raise MediaError(f"{len(missing)} files are still missing below {destination} (e.g. {missing[0]})")
        return mode
