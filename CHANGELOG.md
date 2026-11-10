# Changelog

## Unreleased

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
  `save_dataset_index` (2.17+); older ds-crawler falls back to the previous
  internal call.
- Heads are checked against the `euler_loading` addon rules from
  euler-dataset-contract 0.14+, with no need to import euler-loading. `verify`
  reports the rules and the loader inventory separately.
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
