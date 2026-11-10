# Roadmap

Ideas agreed for later, with enough context to start on them. Each item names
what exists today, the open design questions, and how to tell it is done.

- [1. Zip-native and packed datasets](#1-zip-native-and-packed-datasets)
- [2. Downloads and checksums](#2-downloads-and-checksums)
- [3. Typed descriptions of prepared modalities](#3-typed-descriptions-of-prepared-modalities)

## 1. Zip-native and packed datasets

**Why.** Many datasets ship as zip archives (KITTI-CARLA `TownNN.zip` at about
37 GB each, DrivingStereo, MUSES), and euler-loading reads a zip root
natively, with one `ZipFile` handle per DataLoader worker. On Lustre each file
open or `stat()` is a metadata round trip; we measured about 22 ms per `stat()`
on `/cluster/work`. A prepared dataset of loose files therefore costs at least
one metadata operation per sample while training. A zip costs one per worker.

**Today.** Recipes extract zip and tar downloads, safely, selectively and
resumably (`archives.extract_archive`, `archive_members`). The prepared output
is always loose files: linked, copied or extracted.

**1a. Use downloaded zips in place (`--media zip-native`).** Index the members
of the downloaded zips without extracting them.

- *Obstacle.* A ds-crawler root is a directory or one zip, and a zip root keeps
  its `.ds_crawler/` metadata *inside* the archive. Writing there modifies the
  user's download (and rewrites a 37 GB file).
- *Options:*
  - i. ds-crawler and euler-loading accept metadata stored next to a zip root,
    e.g. `Town01.zip` plus `Town01.zip.ds_crawler/`, or a prepared directory
    whose scopes point into named zips.
  - ii. Copy the zip into the output and write the metadata into the copy.
  - iii. Accept modifying the download.
- *Preferred:* (i). It needs a small ds-crawler/euler-loading feature, and
  it keeps downloads read-only.
- *Constraint:* one modality root is one zip. A dataset split across several
  zips per modality (KITTI-CARLA: one per town) needs either one modality per
  zip or (i) with multi-archive scopes.

**1b. Pack prepared media into zips (`--media zip`).** Write each scope's
files into one zip per scope, so every modality root is a single archive.

- *Tooling.* ds-crawler's `ZipDatasetWriter` already writes a zip together
  with its index. Store JPEG/PNG/`.pcd.bin` members uncompressed: they are
  already compressed or small. Large scopes need zip64.
- *Layout.* The prepared directory keeps the receipt, and each scope becomes
  `<scope>.zip` with its metadata inside: `Modality("/data/nuscenes/rgb.zip")`.
  Hierarchical scopes (calibration) can share one small zip.
- *Measure first.* Benchmark DataLoader throughput on Lustre for loose files
  against one zip per scope, with 1, 4 and 8 workers, on nuScenes
  CAM_FRONT_LEFT (34,902 frames).

**Done when** `prepare --media zip` (and/or `zip-native`) produces outputs
that `verify` passes and euler-loading loads unchanged, the benchmark shows the
metadata saving, and the docs say when to choose which mode.

## 2. Downloads and checksums

**Why.** `prepare` starts from downloads the user fetched. Two gaps remain:
the archives' integrity is not checked, and fetching is manual.

**Today.**
- The extraction ledger records each archive's name, size and modification
  time, and the receipt lists the inputs.
- nuScenes' `.v1.0-*.txt` side files contain only a thank-you note, not
  checksums, and its downloads need a login.

**2a. Checksums.**
- Recipes declare checksums where publishers provide them
  (`expected_inputs` with sha256/md5).
- `prepare --verify-checksums` streams each archive once to check it. It is
  opt-in, since that reads every byte (57 GB for nuScenes test).
- Without published checksums, `--record-checksums` computes sha256 once and
  stores it in the receipt. A later `prepare` or `verify` can then detect a
  changed download.
- Keep size and mtime as the cheap default.

**2b. Fetching.**
- An opt-in `euler-known-datasets download <recipe> DIR`, only for datasets
  with public direct URLs. DrivingStereo, for instance, has a Hugging Face
  mirror.
- Downloads resume with HTTP range requests.
- The dataset's licence is shown and must be accepted explicitly
  (`--accept-license`).
- Datasets behind a login, such as nuScenes, stay manual; `info` says where to
  get them.
- `prepare` itself stays offline.

**Done when** recipes can declare checksums, and both flags are covered by
tests with a fake archive. Separately, `download` works for at least one
public dataset and records what it fetched in a manifest that `prepare` reads.

## 3. Typed descriptions of prepared modalities

**Why.** euler-dataset-contract can describe what a modality's *decoded* values
are in a typed `RepresentationProfile`. It carries `kind` (image, depth,
intrinsics, extrinsics, point_cloud, …), `layout`, `shape`, `dtype`, `unit`,
`invalid`, plus `frame`, `image_plane`, `camera_model`, `components`,
`source_frame`, `target_frame` and `direction` where they apply. Profiles
travel in the `euler_representation` head addon and in the field bindings of
`euler_transforms`. They are what makes a resize/crop recipe, its replay and a
geometry-aware evaluator checkable rather than guessed. A prepared dataset
should therefore be describable by profiles.

**Today.**

- Prepared heads carry the contract's optional geometry meta, which the
  contract validates since 0.15: transform frames and `direction`,
  `camera_model`, point `representation`, `columns` and `coordinate_unit`.
- The loader functions declare dtype, layout per variant, unit and range in
  euler-loading's `loaders.json`.
- `euler_representation` is written only by materialized outputs. ds-crawler's
  `ValidatedDatasetWriter` freezes it in the output head, and euler-loading's
  materialization drops a source head's `euler_representation` when it builds
  that output head.
- No component reads a profile from a *source* dataset. Descriptor authors
  supply profiles by hand, or they are inferred from loader annotations at
  export time.

### Problem

Given a prepared dataset, produce for each scope a `RepresentationProfile`
that is:

1. **correct**: it agrees with what the named loader actually returns;
2. **complete**: every field the profile's `kind` requires is known;
3. **single-sourced**: no fact is stated twice where the copies could
   disagree.

It must also hold up under the following constraints.

**C1. Profiles depend on the decoder variant.** A profile describes decoded
values, but a head names one loader module, and euler-loading decodes it in two
variants. The same file is `(3, H, W)` from the GPU loaders and `(H, W, 3)` from
the CPU ones. A profile stored in the head would have to pick a variant, or
describe one canonical variant and leave the other to be inferred.

**C2. One scope can span several frames and image planes.**
- The `rgb` scope of a multi-camera nuScenes output covers several
  `image_plane`s. Its `camera_extrinsics` scope has a different `target_frame`
  per camera.
- KITTI-CARLA's lidar scope indexes each sweep under every camera.

A head-level profile cannot name one `image_plane` or `target_frame` for these.
The value depends on the sample's position in the hierarchy (`camera:<channel>`).

**C3. Shapes vary per sample.**
- Lidar sweeps have `N` points.
- DrivingStereo images are 879 or 881 pixels wide depending on the subset.
- nuScenes images are 1600×900, but a crop changes that.

Profiles must allow symbolic dimensions (the contract permits tokens), and
concrete shapes come from the data.

**C4. The vocabularies disagree.**

| Concept | Head meta (0.15) | Modality inventory | Representation profile |
|---|---|---|---|
| point layout | `representation` | `form` | `kind` + `layout` |
| coordinate unit | `coordinate_unit: meters` | `unit` | `unit` (e.g. `meter`) |
| point columns | `columns` (all, incl. intensity) | `columns` | `components` (coordinates only, ≥ 3) |
| camera frame | free text (`camera_optical`) | `frame` | `frame`, plus a separate `image_plane` |

A derivation needs one mapping between them, or the vocabularies need to be
merged.

**C5. Explicit descriptions must still win.** A materialized output asserts its
profile after transforms. Any derivation must defer to an explicit
`euler_representation`, or check against it, and never override it.

### Proposed design: derive, don't store

1. **Contract: a pure derivation.** Add
   `derive_representation(head, annotation, *, variant, hierarchy=None)`.
   - Inputs: the head's modality key and meta, the loader's `loaders.json`
     annotation, the decoder variant, and optionally the sample's hierarchy
     keys.
   - Output: a `RepresentationAddon`, or a list of the facts that are missing
     (e.g. "no `frame` for kind `point_cloud`").
   - It uses no array libraries. Golden cases go in the conformance corpus as
     (head, annotation, variant, hierarchy) → profile.
2. **Contract: one vocabulary.** Settle C4: map head meta to profile fields,
   define unit spellings, and document a table. Deprecate rather than rename.
   Then the geometry meta is exactly the head's share of the profile facts.
3. **Hierarchy bindings for C2.** A rule maps a hierarchy key to profile fields,
   e.g. `camera:<channel>` gives `image_plane = <channel>` and
   `frame = <channel>_optical`. It could be declared in the head, in an addon
   such as `euler_layout`, or by the recipe. The alternative is one scope per
   camera, which avoids the rule but multiplies scopes.
4. **euler-loading: an accessor.** `MultiModalDataset.representation(name)`
   derives the profile from the resolved loader's annotation and variant. The
   transform export (`bind_to_dataset`, `SamplePreprocessor`) uses it instead
   of its own inference. An explicit `euler_representation` in the head takes
   precedence and is checked against the derivation (C5).
5. **euler-known-datasets: completeness in `verify`.**
   - `verify` derives every scope's profile and reports missing facts as
     warnings, then as errors once the recipes comply.
   - Recipes add whatever meta the derivation needs, e.g. `frame` for depth.
   - An optional `prepare --write-representation` stores the derived profiles
     (for the canonical variant) in the heads, for consumers that cannot
     derive, such as euler-view or non-Python tools.

### Open questions

- Which variant is canonical when a profile is stored (C1): the CPU layout,
  since it is closest to the file, or a variant-neutral form plus per-variant
  layouts?
- Where hierarchy bindings live (C2): a contract rule, the `euler_layout` addon,
  or recipe-written attributes on index nodes?
- Is `camera_optical` acceptable as a generic frame name, or must every frame be
  qualified by its sensor?
- Should `columns` beyond the coordinates become typed (`intensity`, `ring`,
  `timestamp`, with units)?

**Done when**

- The contract has the derivation and its golden cases.
- euler-loading exposes `representation(name)` for both variants.
- `verify` reports complete profiles for every scope of the four recipes,
  including a multi-camera nuScenes output.
- One resize/crop export binds a prepared nuScenes source through derived
  profiles, with no hand-written profile.

