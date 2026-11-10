"""DrivingStereo: rectified stereo images with lidar-derived depth and disparity.

See ``docs/datasets/driving_stereo.md`` for the downloads, the prepared layout
and the dataset facts the recipe relies on. Decoding lives in euler-loading's
``driving_stereo`` loader module; this recipe only decides which files form
which scope.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ds_crawler import collect_qualified_ids, get_files

from ...archives import archive_members, selection_fingerprint
from ...artifacts import (
    IndexTree,
    family_dataset,
    file_entry,
    write_regex_scope,
    write_scope,
    write_splits,
)
from ...media import (
    IN_PLACE,
    MediaError,
    list_files,
    materialize_files,
    missing_files,
    place_extracted,
    resolve_media,
)
from ...recipe import Option, Outcome, Plan, PrepareContext, Recipe, register
from .layout import (
    INTRINSICS_SCOPE,
    SCOPES,
    SPLITS,
    STREAMS,
    SUBSETS,
    ArchiveInfo,
    classify_archive,
    classify_calibration,
    subset_of,
)

logger = logging.getLogger(__name__)

LOADER_MODULE = "driving_stereo"
_MEDIA_SUFFIXES = (".jpg", ".png")


def _addon(function: str) -> dict[str, Any]:
    return {"euler_loading": {"version": "1.0", "loader": LOADER_MODULE, "function": function}}


def _key_value(key: str) -> str:
    return key.split(":", 1)[1]


@register
class DrivingStereo(Recipe):
    name = "driving_stereo"
    title = "DrivingStereo"
    summary = (
        "Rectified stereo images with lidar-derived depth and disparity maps on the "
        "half-size left-image grid, per-sequence intrinsics, train/test and weather subsets."
    )
    homepage = "https://drivingstereo-dataset.github.io/"
    data_license = (
        "Research use; see the terms on the DrivingStereo website and the LICENSE "
        "shipped with the download"
    )
    expected_inputs = (
        "train/<kind>/<sequence>.zip, kind = left-image, right-image, depth-map, disparity-map",
        "test/<half|full>-size/<kind>/<sequence>.zip",
        "weather/<sunny|cloudy|foggy|rainy>/<kind>-<half|full>-size.zip",
        "calibration/<train|test>/half-image-calib/<sequence>.txt",
        "or the archives unzipped in the directory that holds them",
    )
    layout_version = 1
    options = (
        Option(
            "subsets",
            "Subsets to prepare: train, test and the weather subsets; all = every "
            "subset found in the source.",
            default=(),
            multiple=True,
            choices=SUBSETS,
        ),
        Option(
            "modalities",
            "Scopes to prepare; all = every scope the downloads provide.",
            default=(),
            multiple=True,
            choices=SCOPES,
        ),
    )

    # -- discovery ------------------------------------------------------------

    @staticmethod
    def _scan(source: Path) -> dict[str, Any]:
        """Classify the source's files from one directory walk."""
        files = list_files(source)
        archives: list[ArchiveInfo] = []
        calibration: dict[tuple[str, str], str] = {}
        extracted: dict[str, list[str]] = defaultdict(list)
        for relpath in files:
            if relpath.endswith(".zip"):
                info = classify_archive(relpath)
                if info is not None:
                    archives.append(info)
                continue
            if relpath.endswith(".txt"):
                found = classify_calibration(relpath)
                if found is not None:
                    calibration[found] = relpath
                continue
            if relpath.endswith(_MEDIA_SUFFIXES):
                for stream in STREAMS:
                    if stream.matches(relpath):
                        extracted[stream.scope].append(relpath)
                        break
        return {
            "files": set(files),
            "archives": archives,
            "calibration": calibration,
            "extracted": dict(extracted),
        }

    def detect(self, source: Path) -> bool:
        try:
            scan = self._scan(Path(source))
        except OSError:
            return False
        return bool(scan["archives"] or scan["extracted"])

    # -- plan -----------------------------------------------------------------

    def plan(self, ctx: PrepareContext) -> Plan:
        scan = self._scan(ctx.source)
        found = {info.subset for info in scan["archives"]}
        found |= {subset_of(p) for paths in scan["extracted"].values() for p in paths}
        found.discard(None)
        if not found:
            raise FileNotFoundError(
                f"no DrivingStereo downloads in {ctx.source}; expected "
                + "; ".join(self.expected_inputs)
            )
        subsets = ctx.options["subsets"] or tuple(s for s in SUBSETS if s in found)
        absent = [s for s in subsets if s not in found]
        if absent:
            raise FileNotFoundError(
                f"subset(s) {', '.join(absent)} not in {ctx.source}; found: "
                + ", ".join(s for s in SUBSETS if s in found)
            )
        modalities = ctx.options["modalities"] or SCOPES
        archives = [
            info
            for info in scan["archives"]
            if info.subset in subsets and info.stream.scope in modalities
        ]
        extracted = {
            scope: [p for p in paths if subset_of(p) in subsets]
            for scope, paths in scan["extracted"].items()
            if scope in modalities
        }
        extracted = {scope: paths for scope, paths in extracted.items() if paths}
        if not archives and not extracted:
            raise FileNotFoundError(
                f"no files of the requested modalities ({', '.join(modalities)}) for "
                f"subsets {', '.join(subsets)} in {ctx.source}"
            )

        inputs = [
            {"kind": "archive", "subset": a.subset, "scope": a.stream.scope, "path": a.relpath}
            for a in archives
        ]
        for scope, paths in sorted(extracted.items()):
            inputs.append({"kind": "extracted", "scope": scope, "files": len(paths)})
        if scan["calibration"]:
            inputs.append({"kind": "calibration", "files": len(scan["calibration"])})
        steps = [
            f"index subsets {', '.join(subsets)} from archive member names and extracted files",
            "write one ds-crawler scope per modality, hierarchy /subset:<subset>/sequence:<sequence>",
        ]
        if INTRINSICS_SCOPE in modalities:
            steps.append(
                "attach calibration/<train|test>/half-image-calib/<sequence>.txt per "
                "subset and sequence (scope intrinsics)"
                if scan["calibration"]
                else "no calibration files found: the intrinsics scope is skipped"
            )
        steps.append(
            "prepare in place; extract only missing files"
            if ctx.in_place
            else f"bring the indexed files into the output (media={ctx.media})"
        )
        steps.append("write one split per subset")
        return Plan(
            recipe=self.name,
            inputs=inputs,
            steps=steps,
            details={
                "subsets": subsets,
                "modalities": modalities,
                "archives": archives,
                "extracted": extracted,
                "calibration": scan["calibration"],
                "files": scan["files"],
            },
        )

    # -- prepare --------------------------------------------------------------

    def prepare(self, ctx: PrepareContext, plan: Plan) -> Outcome:
        details = plan.details
        subsets: tuple[str, ...] = details["subsets"]
        modalities: tuple[str, ...] = details["modalities"]
        source_files: set[str] = details["files"]
        execution = {
            "recipe": self.name,
            "layout_version": self.layout_version,
            "options": {k: list(v) if isinstance(v, tuple) else v for k, v in ctx.options.items()},
        }
        attributes = {"source": "DrivingStereo", "subsets": list(subsets)}

        # Candidate paths per scope: extracted files, and archive members at the
        # path unzipping the archive in its own directory gives them.
        candidates: dict[str, set[str]] = defaultdict(set)
        origin: dict[str, tuple[ArchiveInfo, str]] = {}
        for scope, paths in details["extracted"].items():
            candidates[scope].update(paths)
        for info in details["archives"]:
            for member in archive_members(ctx.source / info.relpath):
                relpath = f"{info.directory}/{member}" if info.directory else member
                if info.stream.matches(relpath) and subset_of(relpath) == info.subset:
                    candidates[info.stream.scope].add(relpath)
                    origin.setdefault(relpath, (info, member))

        outputs: dict[str, dict[str, Any]] = {}
        for stream in STREAMS:
            paths = candidates.get(stream.scope)
            if not paths:
                continue
            output = write_regex_scope(
                ctx.output,
                stream.scope,
                files=sorted(paths),
                dataset=family_dataset(self.name, attributes),
                modality={"key": stream.key, "meta": stream.meta},
                addons=_addon(stream.function),
                indexing=stream.indexing(),
                execution=execution,
            )
            indexed = len(get_files(output))
            if indexed != len(paths):
                logger.warning(
                    "%s: %d of %d candidate files indexed", stream.scope, indexed, len(paths)
                )
            outputs[stream.scope] = output
            logger.info("wrote scope %s (%d files)", stream.scope, indexed)

        needed = {path for output in outputs.values() for path in get_files(output)}
        intrinsics = self._intrinsics(ctx, outputs, details["calibration"], modalities, execution, attributes)
        needed |= set(intrinsics["files"])

        media = self._materialize(ctx, needed, source_files, origin)

        splits: dict[str, list[str]] = {}
        counts: dict[str, dict[str, int]] = defaultdict(dict)
        sequences: dict[str, set[str]] = defaultdict(set)
        for scope, output in outputs.items():
            by_subset: dict[str, list[tuple]] = defaultdict(list)
            for qid in sorted(collect_qualified_ids(output)):
                subset = _key_value(qid[0])
                by_subset[subset].append(qid)
                sequences[subset].add(_key_value(qid[1]))
            for subset, ids in by_subset.items():
                counts[subset][scope] = len(ids)
            splits[scope] = write_splits(ctx.output, scope, by_subset)

        scopes = list(outputs)
        if intrinsics["written"]:
            scopes.append(INTRINSICS_SCOPE)
        summary = {
            "subsets": {
                subset: {"sequences": len(sequences[subset]), "files": counts[subset]}
                for subset in SUBSETS
                if subset in counts
            },
            "intrinsics": {
                "subset_sequences": intrinsics["nodes"],
                "calibration_files": len(intrinsics["files"]),
                "missing": intrinsics["missing"],
            },
            "files_referenced": len(needed),
        }
        return Outcome(media=media, scopes=scopes, splits=splits, summary=summary)

    @staticmethod
    def _intrinsics(
        ctx: PrepareContext,
        outputs: dict[str, dict[str, Any]],
        calibration: dict[tuple[str, str], str],
        modalities: tuple[str, ...],
        execution: dict[str, Any],
        attributes: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach each sequence's half-size calibration at /subset/sequence."""
        result: dict[str, Any] = {"written": False, "files": [], "nodes": 0, "missing": []}
        if INTRINSICS_SCOPE not in modalities:
            return result
        nodes = sorted({qid[:2] for output in outputs.values() for qid in collect_qualified_ids(output)})
        tree = IndexTree()
        files = set()
        for node in nodes:
            subset, sequence = _key_value(node[0]), _key_value(node[1])
            # Weather subsets are drawn from training and test sequences, whose
            # calibration lives with their split.
            splits = (subset,) if subset in SPLITS else SPLITS
            relpath = next(
                (calibration[(split, sequence)] for split in splits if (split, sequence) in calibration),
                None,
            )
            if relpath is None:
                result["missing"].append(f"{subset}/{sequence}")
                continue
            tree.add(node, file_entry(relpath, "intrinsics"))
            files.add(relpath)
        if result["missing"]:
            logger.warning(
                "no half-image-calib file for %d subset/sequence pairs, e.g. %s",
                len(result["missing"]),
                result["missing"][0],
            )
        if len(tree):
            write_scope(
                ctx.output,
                INTRINSICS_SCOPE,
                dataset=family_dataset(DrivingStereo.name, attributes),
                modality={"key": "intrinsics", "meta": {"camera_model": "pinhole", "dimensions": {"rows": 3, "columns": 3}}},
                addons=_addon("read_intrinsics"),
                tree=tree,
                execution=execution,
            )
            result.update(written=True, files=sorted(files), nodes=len(tree))
        return result

    @staticmethod
    def _extract(
        ctx: PrepareContext,
        relpaths: Iterable[str],
        origin: dict[str, tuple[ArchiveInfo, str]],
        root: Path,
    ) -> None:
        by_archive: dict[str, tuple[ArchiveInfo, set[str]]] = {}
        for relpath in relpaths:
            info, member = origin[relpath]
            by_archive.setdefault(info.relpath, (info, set()))[1].add(member)
        for info, members in by_archive.values():
            ctx.extract(
                ctx.source / info.relpath,
                destination=root / info.directory if info.directory else root,
                select=members.__contains__,
                fingerprint=selection_fingerprint(members),
            )

    def _materialize(
        self,
        ctx: PrepareContext,
        needed: set[str],
        source_files: set[str],
        origin: dict[str, tuple[ArchiveInfo, str]],
    ) -> str:
        paths = sorted(needed)
        not_extracted = [p for p in paths if p not in source_files]
        unavailable = [p for p in not_extracted if p not in origin]
        if unavailable:  # pragma: no cover - every candidate has one of the two
            raise MediaError(f"{len(unavailable)} files have no source, e.g. {unavailable[0]}")
        mode = resolve_media(ctx, extracted=not not_extracted, archives=bool(origin))
        if mode == IN_PLACE:
            self._extract(ctx, not_extracted, origin, ctx.source)
            root = ctx.source
        elif mode in ("symlink", "hardlink", "copy"):
            top_level = sorted({p.split("/", 1)[0] for p in paths})
            place_extracted(ctx, paths, mode=mode, top_level=top_level)
            root = ctx.output
        else:  # extract
            self._extract(ctx, [p for p in paths if p in origin], origin, ctx.output)
            rest = [p for p in paths if p not in origin]  # e.g. plain calibration files
            if rest:
                materialize_files(ctx.source, ctx.output, rest, "copy")
            root = ctx.output
        missing = missing_files(root, paths)
        if missing:
            raise MediaError(
                f"{len(missing)} indexed files are missing below {root} (e.g. {missing[0]}); "
                "the archives may be incomplete"
            )
        return mode
