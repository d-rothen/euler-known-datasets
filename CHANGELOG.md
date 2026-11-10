# Changelog

## Unreleased

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
