# Changelog

## Unreleased

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
