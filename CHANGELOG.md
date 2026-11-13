# Changelog

## Unreleased

- Recipe `shift` for SHIFT's discrete set, front camera: images (`rgb`),
  planar depth in CARLA's 24-bit code (`depth`), semantic labels with the ID
  in red (`semantic_segmentation`, sky `[13, 0, 0]`), and, from a
  `calibration/<split>/<sequence>.json` written per sequence, the intrinsics
  and per-frame camera-to-world poses (`camera_pose`). The poses mirror
  CARLA's left-handed world by negating y, with the rotation
  `Rz(-r2) Ry(r1) Rx(-r0)`; depth reprojects between frames with a median
  error of 0.14 % on 30 real frame pairs. The hierarchy is
  `/split/weather/time_of_day/sequence/` with coarse conditions, and the fine
  conditions and CARLA weather parameters are file attributes. Options select
  scopes, splits, weather, time of day, a frame step, and a motion filter
  (`min_motion_m`, `min_rotation_deg`) that drops frames where the camera
  barely moved; the selection code is shared with the tool that decodes
  frames from the video-rate mirror (`selection.py`). A frame is kept only if
  every selected scope has its file. The recipe reads shift-dev's image-rate
  zips, their extracted folders, or frames decoded from the videos (merged
  label set, `decoded_from_videos.json`); `det_2d.json` (several GB) is
  streamed.
- Recipe `urbansyn` for UrbanSyn: images (`rgb`), planar depth in units of
  1e5 m (`depth`), Cityscapes train-ID labels (`semantic_segmentation`, sky
  `[10, 10, 10]`), the intrinsics from `camera_metadata.json`, and per-frame
  camera-to-world poses read from the depth EXRs' headers (`camera_pose`),
  with the district as hierarchy (`/scene:<scene>/<frame>`). The depth
  convention, units, intrinsics and poses were checked together by
  reprojecting depth between frames. The 57 depth maps that lack the static
  scene (frames 6973–7029) are left out of `depth` unless
  `keep_incomplete_depth` is set. The recipe reads the per-modality zips or
  their extracted folders. Options select scopes and scenes.
- Recipe `hypersim` for Hypersim: the HDR colour (`rgb`, with each frame's
  tone-mapping scale computed by Hypersim's operator), planar depth (`depth`,
  from the released Euclidean distances; `radial_distance` keeps those), per-scene
  intrinsics and per-frame camera-to-world poses in meters, in
  `calibration/<scene>/<camera>.json`. The camera model decomposes Hypersim's
  `M_cam_from_uv` into an exact pinhole `K` and the image's optical frame, so
  the tilt-shift scenes (272 of 482, up to 41.5°) are handled; checked against
  Hypersim's world-space positions to within float16 precision. The recipe
  reads extracted scene folders, partial downloads or the per-scene zips, with
  the official v1 splits. Options select scopes, scenes, splits and a frame
  step. Frame 0 of 38 trajectories, rendered with a stretched or mirrored
  camera, and the black renders of two scenes are left out and listed in the
  receipt (option `keep_dark_frames` keeps the latter). Needs `h5py` (extra
  `hypersim`).
- Recipe `acdc` for ACDC: the anonymised adverse-condition images (`rgb`) and
  their normal-condition references (`reference_rgb`), joined by ID, with the
  official splits. It reads the official zip, an archive of a whole ACDC copy
  (one streaming pass with a path filter), or the extracted tree. Options
  select scopes, conditions and splits.
- Recipe `cerema` for the CEREMA Pixel-Accurate Depth Benchmark: stereo RGB,
  gated slices, lidar projected into both cameras, dense per-scene ground
  truth, intrinsics, and the SGM / PSMNet / Monodepth / Sparse-to-Dense
  outputs as baseline scopes (`gt: false, baseline: true, method`). Options
  select scopes, scenes, time of day and weather.
- Nested zips are read in place: `extract_archive`, `archive_members` and
  `PrepareContext.extract` take `inner=` (member names leading to a zip inside
  a zip), and `open_nested_zip` opens one. A stored inner zip is read through
  a window onto its parent; a compressed one is spooled to a temporary file.
  The extraction ledger tells nested archives apart. `scatternerf` uses this
  instead of unpacking nested copies.
- A dataset family may include baseline scopes: the family test and docs
  accept `gt: false, baseline: true` next to ground truth.
- Recipe `scatternerf` for ScatterNeRF: left/right RGB, per-image intrinsics and
  camera-to-world poses, each sequence's metric scale (from the 0.203 m
  stereo baseline) in the pose files' `scale_to_meters`. It follows the
  release's nested, duplicated archives and reads each sequence once. The
  release has no lidar.
- Every scope of a prepared dataset shares one `dataset` block: `id` and `name`
  are the dataset family, i.e. the recipe name, and the attributes carry
  `gt: true, baseline: false`. Previously each scope had its own id, such as
  `nuscenes_lidar`. `artifacts.family_dataset` builds the block, and `verify`
  warns when scopes do not share the family. `kitti_carla` and `muses` match the
  families euler-view already uses.
- `docs/roadmap.md` §3 formalizes typed descriptions (`euler_representation`)
  for prepared modalities: the problem, its constraints, a derive-don't-store
  design, and open questions.
- Recipes `driving_stereo`, `kitti_carla` and `muses`. Their indexing
  knowledge (regexes, hierarchy, head meta) moves here from euler-loading's
  documentation and tests. Each accepts the official zip downloads or their
  extracted contents and extracts only the files it indexes.
- Zip downloads: `extract_archive` reads zip archives as well as tar (with the
  same safety, selection and resume rules), and `archive_members` lists an
  archive's files, from the zip central directory without reading it.
- `artifacts.write_regex_scope` indexes paths with ds-crawler regexes, which
  stay in `ds-crawler.json`. The paths need not exist yet, so a recipe can
  index archive member names and then extract only what it indexed.
- `media.resolve_media` and `media.place_extracted` bring extracted downloads
  into an output, and `media.list_files` lists files with one directory read
  each (no per-file `stat()`).
- Prepared indexes are written with ds-crawler's public `build_index_output` and
  `save_dataset_index`. Requires `ds-crawler>=2.17.0`.
- Heads are checked against the `euler_loading` addon rules, which
  euler-dataset-contract owns from 0.14, with no need to import euler-loading.
  Requires `euler-dataset-contract>=0.14.0`; tests use euler-loading 2.40's
  `nuscenes` loaders.
- Geometry heads use the contract's vocabulary: transforms declare
  `source_frame`, `target_frame`, `direction: target_from_source` and
  `dimensions`; intrinsics `camera_model`; point clouds `representation`,
  `columns`, `coordinate_unit`.
- `docs/roadmap.md`: zip-native and packed outputs; checksums and downloads.

## 0.1.0

First version.

- `prepare(recipe, source, output, media=..., **options)` and the
  `euler-known-datasets prepare <recipe> SOURCE OUTPUT` command turn a
  dataset's downloads into a directory with ds-crawler scopes, splits and a
  receipt (`euler-known-datasets.json`).
- `verify(output)` / `euler-known-datasets verify` checks every scope, file and
  split of a prepared output, and whether the installed euler-loading provides
  the loaders its heads name.
- Media strategies `auto`, `symlink`, `hardlink`, `copy`, `extract`, and in-place
  preparation. Archive extraction is selective, resumable and rejects members
  that would leave the output.
- Recipes are registered with `@register` or contributed by other packages
  through the `euler_known_datasets.recipes` entry-point group.
- Recipe `nuscenes` (layout version 1): camera RGB, the `LIDAR_TOP` sweep
  synchronised to each frame, intrinsics and per-frame lidar→camera extrinsics
  for v1.0-trainval, v1.0-test and v1.0-mini, with the official scene splits
  and keyframe-only splits.
