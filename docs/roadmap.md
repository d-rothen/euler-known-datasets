# Roadmap

Ideas agreed for later, with enough context to start on them. Each item names
what exists today, the open design questions, and how to tell it is done.

- [1. Zip-native and packed datasets](#1-zip-native-and-packed-datasets)
- [2. Downloads and checksums](#2-downloads-and-checksums)

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
