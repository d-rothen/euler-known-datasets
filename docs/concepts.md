# Concepts

- [Who knows what](#who-knows-what)
- [Recipes](#recipes)
- [Media strategies](#media-strategies)
- [The prepared output](#the-prepared-output)
- [The receipt](#the-receipt)
- [Resuming](#resuming)
- [Verifying](#verifying)
- [Network filesystems](#network-filesystems)

## Who knows what

| Package | Knows |
|---|---|
| ds-crawler | How files become a versioned index: IDs, hierarchy, scopes, splits. Nothing about any dataset. |
| euler-loading | How to decode a file (a JPEG, a `.pcd.bin`, a calibration file) and join modalities by ID. |
| **euler-known-datasets** | What a specific dataset's downloads look like, and everything about it that file paths don't say: joins hidden in metadata tables, sensor timing, coordinate conventions, official splits. |

A prepared dataset is the hand-over point. Its dataset heads name the
euler-loading loader for each scope (`addons.euler_loading`), so
`Modality(root, metadata_scope="rgb")` needs no further configuration.

## Recipes

A recipe is a `Recipe` subclass registered under a name (`nuscenes`). It
declares:

- `expected_inputs`: the downloads it accepts, shown by `info` and in errors;
- `options`: typed, documented, exposed both as `prepare(..., name=value)` and
  as `--name` flags;
- `layout_version`: bumped whenever the prepared output changes shape;
- `data_license`: the dataset's own licence, recorded in every receipt.

It implements two methods:

- `plan(ctx)` inspects the source and returns a `Plan`: the inputs it found and
  the steps it will take. It raises if the run cannot succeed, before anything
  is written. `--dry-run` stops here.
- `prepare(ctx, plan)` produces the output and returns an `Outcome`: the media
  strategy used, the scopes and splits written, and a summary.

Other packages can contribute recipes through the
`euler_known_datasets.recipes` entry-point group:

```toml
[project.entry-points."euler_known_datasets.recipes"]
my_dataset = "my_package.recipes:MyDataset"
```

## Media strategies

Media files (images, point clouds) are never decoded or converted by
`prepare`; the only question is how they reach the output:

| `--media` | Effect |
|---|---|
| `auto` (default) | `symlink` if the source already holds every needed file extracted, otherwise `extract`. |
| `symlink` | Link the source's top-level directories into the output. Instant and free; the output depends on the source staying where it is. |
| `hardlink` | Hard-link each needed file. Free, survives deleting the source, needs the same filesystem. |
| `copy` | Copy each needed file. |
| `extract` | Extract only the needed files from the archives. |

`OUTPUT == SOURCE` prepares in place: nothing moves, and only missing files
are extracted from the archives next to them.

`extract`, `hardlink` and `copy` bring in only the files the prepared dataset
references. For nuScenes `CAM_FRONT_LEFT`, that is 1 of 12 sensor directories
plus the lidar sweeps paired with its frames.

## The prepared output

```
<output>/
├── euler-known-datasets.json     the receipt
├── .euler-known-datasets/        resume state (safe to delete once complete)
├── .ds_crawler/
│   ├── scopes.json
│   └── <scope>/                  dataset-head.json, ds-crawler.json, index.json, split_<name>.json
└── ...                           the dataset's media and recipe-specific files
```

Every scope's index is *prebuilt* (`source.prebuilt_index_file`). Its IDs and
hierarchy come from the recipe rather than from path regexes, so they can carry
facts paths do not encode: a scene, or the camera frame a lidar sweep belongs
to. ds-crawler and euler-loading read it like any other index.

## The receipt

`euler-known-datasets.json` records:

| Field | Meaning |
|---|---|
| `status` | `incomplete` while `prepare` runs, `complete` when it finished |
| `generator` | `euler-known-datasets` and its version |
| `recipe` | name, title and `layout_version` |
| `data_license` | the dataset's licence |
| `source` | the source path and the inputs found in it |
| `options` | the resolved recipe options |
| `media` | the requested and the used strategy |
| `scopes`, `splits` | what the output offers |
| `summary` | recipe-specific counts, e.g. frames and synchronised frames per camera |

## Resuming

`prepare` is safe to re-run with the same arguments. Archives that were fully
extracted with the same selection into the same destination are skipped, which
the ledger in `.euler-known-datasets/` records. Within an archive, files that
already exist with the right size are kept. Files are written under a temporary
name and renamed into place. An output that holds a receipt or resume state
from euler-known-datasets is reused; any other non-empty output is refused
unless `force=True` (`--force`).

## Verifying

```bash
euler-known-datasets verify /data/nuscenes      # --json for machines
```

For every scope the receipt lists, it checks that:

- the artifact set validates and the index loads through ds-crawler;
- every indexed file exists;
- every split loads.

With euler-loading installed, it also checks each dataset head against
euler-loading's addon rules, and warns when the named `loader.function` is
missing from the installed euler-loading. It reads euler-loading's
`loaders.json` for that, without importing torch. The exit status is non-zero
on errors, so it works as a job prologue.

## Network filesystems

On Lustre and GPFS every `stat()` is a round trip to a metadata server, about
20 ms on the Euler cluster. Checking 70,000 files one by one would take over
twenty minutes. All existence checks therefore read directory listings, once
per directory, and `stat()` only files that exist, e.g. when resuming. The same
cost applies to training: one small file per sample means one metadata round
trip per sample.
