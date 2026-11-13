"""UrbanSyn: 7539 path-traced frames of three synthetic city districts, with
planar depth, Cityscapes semantic labels, the camera's intrinsics and the
per-frame camera poses recorded in the depth maps' headers.

See ``docs/datasets/urbansyn.md`` for the layout and the dataset facts it relies on.
"""

from __future__ import annotations

import logging
from collections import Counter
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
from .sources import CAMERA_FILE, FolderSource, UrbanSynSource, discover

logger = logging.getLogger(__name__)

LOADER_MODULE = "urbansyn"

#: (scene, city, first frame, last frame). The frames are numbered by scene;
#: each depth EXR names its scene's project (``urbansyn_<scene>_orbx``) in its
#: header (checked on all 7539 frames, 2026-11-13).
SCENES = (
    ("poblenou", "Barcelona", 1, 2688),
    ("torstrasse", "Berlin", 2689, 5593),
    ("veteranenstrasse", "Berlin", 5594, 7539),
)
SCENE_NAMES = tuple(name for name, *_ in SCENES)

GRID = {"height": 1024, "width": 2048}
#: Cityscapes training classes by train ID; 19 marks everything else.
CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]
SKY_TRAIN_ID = 10
OTHER_ID = 19
#: Planar depth in meters: valid values reach 1360 m, the sky being a backdrop
#: 360-1180 m away (5th-95th percentile; checked on all frames, 2026-11-13).
DEPTH_RANGE = [0, 1500]
#: Frames whose depth map lacks the static scene: 29-54 % of their non-sky
#: pixels hit nothing (stored 1.0), where only people, vehicles and poles
#: remain. They are the frames whose labels UrbanSyn fixed on 2026-10-19; the
#: depth was not re-released. Every other frame misses at most 0.08 %.
INCOMPLETE_DEPTH = frozenset(f"{n:04d}" for n in range(6973, 7030))

#: scope -> (release folder or None for the camera file, modality key, head meta, function)
SCOPES: dict[str, tuple[Optional[str], str, dict[str, Any], str]] = {
    "rgb": ("rgb", "rgb", {"range": [0, 255], "dimensions": {**GRID, "channels": 3}}, "rgb"),
    "depth": ("depth", "depth", {
        "radial_depth": False, "scale_to_meters": 100000.0, "range": list(DEPTH_RANGE),
        "invalid": {"non_finite": False, "sentinel": 0}, "dimensions": dict(GRID),
        "stored": "OctaneRender Z-depth pass, float32 EXR channel Y in units of 1e5 m; 1.0 where a ray hits nothing",
    }, "depth"),
    "semantic_segmentation": ("ss", "semantic_segmentation", {
        "skyclass": [SKY_TRAIN_ID] * 3, "label_ids": "cityscapes_train_ids", "classes": list(CLASSES),
        "other_id": OTHER_ID, "encoding": "grey_rgb", "stored": "RGB PNG with R = G = B = train ID",
        "dimensions": dict(GRID),
    }, "semantic_segmentation"),
    "intrinsics": (None, "intrinsics", {
        "camera_model": "pinhole", "dimensions": {"rows": 3, "columns": 3}, "image_size": [1024, 2048],
    }, "read_intrinsics"),
    # The poses live in the depth EXRs' headers, so this scope indexes them too.
    "camera_pose": ("depth", "camera_extrinsics", {
        "source_frame": "camera_optical", "target_frame": "world", "direction": "target_from_source",
        "dimensions": {"rows": 4, "columns": 4}, "translation_unit": "meters",
        "world": "the district's render frame, y up; one per scene",
        "stored": "camera pos, target and up in the depth EXR header (attribute octane)",
    }, "camera_pose"),
}


def scene_of(frame: str) -> Optional[tuple[str, str]]:
    """``(scene, city)`` of a frame number such as ``"0001"``."""
    number = int(frame)
    for name, city, first, last in SCENES:
        if first <= number <= last:
            return name, city
    return None


@register
class UrbanSyn(Recipe):
    name = "urbansyn"
    title = "UrbanSyn"
    summary = (
        "Path-traced frames of three synthetic districts of Barcelona and Berlin with planar "
        "depth, Cityscapes semantic labels, the camera intrinsics and per-frame camera poses; no splits."
    )
    homepage = "https://www.urbansyn.org"
    data_license = (
        "CC BY-SA 4.0 with UrbanSyn's terms of use: research, teaching and commercial use; "
        "no generative-AI reverse engineering of its assets or stock-media content made from it"
    )
    expected_inputs = (
        "rgb.zip, depth.zip and ss.zip as downloaded from urbansyn.org, and camera_metadata.json",
        "or their extracted rgb/, depth/ and ss/ folders (also one folder down)",
    )
    layout_version = 1
    options = (
        Option("modalities", "Scopes to prepare; all = every scope whose files are found.",
               default=(), multiple=True, choices=tuple(SCOPES)),
        Option("scenes", "Scenes to prepare.", default=(), multiple=True, choices=SCENE_NAMES),
        Option("keep_incomplete_depth",
               "Also index the 57 depth maps that lack the static scene (frames 6973-7029).",
               default=False, type=bool),
    )

    def detect(self, source: Path) -> bool:
        try:
            found = discover(Path(source))
        except OSError:
            return False
        return "rgb" in found.folders or "depth" in found.folders

    # -- planning -----------------------------------------------------------

    def plan(self, ctx: PrepareContext) -> Plan:
        found = discover(ctx.source)
        if not found.folders:
            raise FileNotFoundError(
                f"no UrbanSyn downloads in {ctx.source}; expected " + "; ".join(self.expected_inputs)
            )
        available = [scope for scope, (folder, *_rest) in SCOPES.items()
                     if (folder in found.folders if folder else found.camera_file is not None)]
        requested = list(ctx.options["modalities"])
        missing = [scope for scope in requested if scope not in available]
        if missing:
            raise FileNotFoundError(
                f"no files for scope(s) {', '.join(missing)} in {ctx.source}; found "
                f"{', '.join(available) or 'none'}"
            )
        scopes = requested or available
        folders = sorted({SCOPES[s][0] for s in scopes if SCOPES[s][0]})
        frames = sorted(set().union(*(found.folders[f].frames for f in folders))) if folders else []
        unknown = [f for f in frames if scene_of(f) is None]
        if unknown:
            raise ValueError(f"frames {unknown[:5]} lie outside the known UrbanSyn scenes {SCENES}")
        wanted_scenes = set(ctx.options["scenes"]) or set(SCENE_NAMES)
        frames = [f for f in frames if scene_of(f)[0] in wanted_scenes]
        if folders and not frames:
            raise FileNotFoundError("no frames of the selected scenes")
        steps = [f"note: {note}" for note in found.notes]
        steps.append(f"prepare {len(frames)} frames of {', '.join(s for s in SCENE_NAMES if s in wanted_scenes)}: "
                     f"scopes {', '.join(scopes)}")
        dropped = sorted(INCOMPLETE_DEPTH & set(frames))
        if "depth" in scopes and dropped and not ctx.options["keep_incomplete_depth"]:
            steps.append(f"leave the {len(dropped)} depth maps without the static scene out of depth "
                         f"(frames {dropped[0]}-{dropped[-1]})")
        steps.append(f"bring the files into the output (media={ctx.media})")
        steps.append("write ds-crawler scopes with prebuilt indexes")
        return Plan(recipe=self.name, inputs=found.describe(), steps=steps,
                    details={"found": found, "scopes": scopes, "frames": frames, "folders": folders})

    # -- preparing ----------------------------------------------------------

    def prepare(self, ctx: PrepareContext, plan: Plan) -> Outcome:
        found: UrbanSynSource = plan.details["found"]
        scopes: list[str] = plan.details["scopes"]
        frames: list[str] = plan.details["frames"]
        keep_incomplete = ctx.options["keep_incomplete_depth"]
        # Frames each scope indexes; the folder's files are extracted for any of its scopes.
        selected = {
            scope: [f for f in frames if scope != "depth" or keep_incomplete or f not in INCOMPLETE_DEPTH]
            for scope in scopes if SCOPES[scope][0]
        }
        needed: dict[str, list[str]] = {}
        for scope, wanted in selected.items():
            folder = SCOPES[scope][0]
            needed[folder] = sorted((set(needed.get(folder, ())) | set(wanted)) & found.folders[folder].frames)
        camera = "intrinsics" in scopes
        media, bases, camera_path = self._materialize(ctx, found, needed, camera)

        scenes = sorted({scene_of(f)[0] for f in frames}, key=SCENE_NAMES.index)
        attributes = {"source": "UrbanSyn", "scenes": scenes,
                      "cities": sorted({scene_of(f)[1] for f in frames})}
        execution = {"recipe": self.name, "layout_version": self.layout_version,
                     "options": {k: list(v) if isinstance(v, tuple) else v for k, v in ctx.options.items()}}
        files: dict[str, int] = {}
        missing: dict[str, list[str]] = {}
        for scope in scopes:
            folder, key, meta, function = SCOPES[scope]
            tree = IndexTree()
            if folder is None:
                tree.add((), file_entry(camera_path, scope))
            else:
                present = set(needed[folder])
                source = found.folders[folder]
                for frame in selected[scope]:
                    if frame not in present:
                        missing.setdefault(scope, []).append(frame)
                        continue
                    scene, city = scene_of(frame)
                    file_attributes = {"city": city}
                    if scope == "depth" and frame in INCOMPLETE_DEPTH:
                        file_attributes["incomplete_depth"] = True
                    tree.add((f"scene:{scene}",),
                             file_entry(f"{bases[folder]}/{source.name(frame)}", frame, file_attributes))
            write_scope(
                ctx.output,
                scope,
                dataset=family_dataset(self.name, attributes),
                modality={"key": key, "meta": meta},
                addons={"euler_loading": {"version": "1.0", "loader": LOADER_MODULE, "function": function}},
                tree=tree,
                execution=execution,
            )
            files[scope] = len(tree)
            logger.info("%s: %d files", scope, len(tree))

        summary: dict[str, Any] = {
            "files": files,
            "frames": len(frames),
            "frames_per_scene": dict(Counter(scene_of(f)[0] for f in frames)),
            "notes": found.notes,
        }
        if missing:
            summary["frames_without_file"] = {scope: len(names) for scope, names in missing.items()}
        if "depth" in selected:
            left_out = sorted(set(frames) - set(selected["depth"]))
            summary["incomplete_depth"] = {
                "frames": len(INCOMPLETE_DEPTH & set(frames)),
                "indexed": keep_incomplete,
                **({"left_out": f"{left_out[0]}-{left_out[-1]}"} if left_out else {}),
            }
        return Outcome(media=media, scopes=scopes, splits={}, summary=summary)

    @staticmethod
    def _materialize(
        ctx: PrepareContext, found: UrbanSynSource, needed: dict[str, list[str]], camera: bool
    ) -> tuple[str, dict[str, str], Optional[str]]:
        """Bring the needed files into the output; return the media mode, each
        folder's directory relative to the output root, and the camera file's path."""
        extracted = all(found.folders[f].extracted for f in needed)
        archived = all(found.folders[f].archive is not None for f in needed)
        mode = resolve_media(ctx, extracted=extracted, archives=archived)
        root = ctx.source if mode == IN_PLACE else ctx.output
        bases: dict[str, str] = {}
        for folder, frames in sorted(needed.items()):
            src: FolderSource = found.folders[folder]
            names = [src.name(f) for f in frames]
            if mode == IN_PLACE and src.extracted:
                bases[folder] = src.extracted
                continue
            bases[folder] = folder
            if mode in ("symlink", "hardlink", "copy"):
                location = PurePosixPath(src.extracted)
                parent = ctx.source.joinpath(*location.parent.parts) if location.parent.parts else ctx.source
                absent = missing_files(ctx.source / src.extracted, names)
                if absent:
                    raise MediaError(f"{len(absent)} needed files are not in {ctx.source / src.extracted} "
                                     f"(e.g. {absent[0]})")
                if mode == "symlink":
                    link_top_level(parent, ctx.output, [folder])
                else:
                    materialize_files(parent, ctx.output, [f"{folder}/{n}" for n in names], mode)
                continue
            if src.archive is None:
                raise MediaError(f"{folder} has no zip to extract from in {ctx.source}")
            if src.member_prefix != f"{folder}/":
                raise MediaError(f"{src.archive.name} keeps {folder} files below {src.member_prefix!r}, "
                                 f"not {folder}/")
            selection = {src.member(f) for f in frames}
            ctx.extract(src.archive, destination=root, select=selection.__contains__,
                        fingerprint=selection_fingerprint(selection))
        for folder, frames in needed.items():
            absent = missing_files(root / bases[folder], [found.folders[folder].name(f) for f in frames])
            if absent:
                raise MediaError(f"{len(absent)} files are still missing below {root / bases[folder]} "
                                 f"(e.g. {absent[0]})")

        camera_path = None
        if camera:
            if found.camera_file is None:
                raise MediaError(f"no {CAMERA_FILE} in {ctx.source}")
            if mode == IN_PLACE:
                camera_path = found.camera_file
            else:
                location = PurePosixPath(found.camera_file)
                parent = ctx.source.joinpath(*location.parent.parts) if location.parent.parts else ctx.source
                if mode == "symlink":
                    link_top_level(parent, ctx.output, [CAMERA_FILE])
                else:  # hardlink and copy as asked; extract copies, the file is no archive member
                    materialize_files(parent, ctx.output, [CAMERA_FILE], "hardlink" if mode == "hardlink" else "copy")
                camera_path = CAMERA_FILE
        return mode, bases, camera_path
