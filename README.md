<!-- euler header — shared across the euler packages.
     Per package, change only: the <h1>, the tagline, and the badge URLs. -->
<p align="center">
  <img src="https://files.chronodle.com/icons/euler.svg" alt="euler" width="96" height="96">
</p>

<h1 align="center">euler-known-datasets</h1>

<p align="center">
  <em>Downloaded dataset archives in, euler-ready datasets out — the dataset-specific knowledge lives here.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/euler-known-datasets/"><img alt="PyPI" src="https://img.shields.io/pypi/v/euler-known-datasets.svg"></a>
  <a href="https://pypi.org/project/euler-known-datasets/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/euler-known-datasets.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://github.com/d-rothen/euler-known-datasets/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/d-rothen/euler-known-datasets/actions/workflows/ci.yml/badge.svg"></a>
</p>

---

[ds-crawler](https://github.com/d-rothen/ds-crawler) indexes files and
[euler-loading](https://github.com/d-rothen/euler-loading) loads them, but
neither should have to know that nuScenes keeps its calibration in relational
JSON tables, or that a nuScenes camera fires 43 ms before the end of the lidar
sweep it belongs to. That knowledge lives here, once per dataset, as a
**recipe**:

```bash
euler-known-datasets prepare nuscenes /data/downloads/nuscenes /data/nuscenes \
    --cameras CAM_FRONT_LEFT
```

reads the official downloads (`v1.0-test_meta.tgz`, `v1.0-test_blobs.tgz`, …,
or what they extracted to) and writes a directory that ds-crawler and
euler-loading use as is:

```python
from euler_loading import Modality, MultiModalDataset

root = "/data/nuscenes"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb", split="test"),
        "lidar": Modality(root, metadata_scope="lidar", split="test"),
    },
    hierarchical_modalities={
        "intrinsics": Modality(root, metadata_scope="intrinsics", collapse_single=True),
        "extrinsics": Modality(root, metadata_scope="camera_extrinsics", collapse_single=True),
    },
)
sample = dataset[0]   # full_id "/scene:scene-0077/camera:CAM_FRONT_LEFT/1533153857904799"
```

## Install

```bash
pip install "euler-known-datasets[fast]"   # [fast] adds orjson for large metadata tables
```

Python 3.9 or newer.

## What `prepare` does

1. **Finds the downloads.** Each recipe knows its dataset's archive names and
   extracted layout, and accepts either or both.
2. **Brings media into the output** without duplicating it unless asked:
   `--media auto` links an already-extracted source and otherwise extracts only
   the files the prepared dataset uses. `symlink`, `hardlink`, `copy` and
   `extract` force a strategy; `OUTPUT == SOURCE` prepares in place.
3. **Resolves the dataset's peculiarities**: joins, sensor synchronisation,
   coordinate conventions. Where a dataset keeps per-frame facts in tables, the
   recipe writes them into small per-scene files in a documented format.
4. **Writes ds-crawler artifacts**: one scope per modality below
   `.ds_crawler/<scope>/`, with dataset heads that name their euler-loading
   loader, plus the dataset's official splits.
5. **Writes a receipt**, `euler-known-datasets.json`: recipe and layout version,
   options, inputs, and a summary of what was produced.

Extraction is safe (no absolute paths, `..` or links leave the output),
selective, and resumable: re-running an interrupted `prepare` skips finished
archives and present files.

```bash
euler-known-datasets verify /data/nuscenes
```

checks that every scope loads through ds-crawler, every indexed file exists,
every split loads, and the installed euler-loading provides the loaders the
heads name.

## Known datasets

| Recipe | Dataset | Prepared modalities |
|---|---|---|
| `cerema` | CEREMA Pixel-Accurate Depth Benchmark (Gruber et al., 3DV 2019) | stereo RGB, gated slices, lidar projected into both cameras, dense ground-truth depth, intrinsics; SGM / PSMNet / Monodepth / Sparse-to-Dense outputs as baselines |
| `driving_stereo` | [DrivingStereo](https://drivingstereo-dataset.github.io/) train / test / weather subsets | left/right RGB, depth, disparity (half- and full-size), per-sequence intrinsics; one split per subset |
| `kitti_carla` | [KITTI-CARLA](https://arxiv.org/abs/2109.00892) towns | RGB, depth, semantic segmentation of both cameras, lidar sweeps, intrinsics, lidar→camera extrinsics, lidar poses |
| `muses` | [MUSES](https://muses.vision.ee.ethz.ch/) | frame camera, reference frames, lidar, GNSS/IMU, semantic and panoptic labels, calibration, sensor timing; train/val/test splits |
| `nuscenes` | [nuScenes](https://www.nuscenes.org/nuscenes) v1.0 trainval / test / mini | camera RGB, synchronised `LIDAR_TOP` sweep, intrinsics, per-frame lidar→camera extrinsics; official scene splits |
| `scatternerf` | ScatterNeRF (Ramazzina et al., ICCV 2023) foggy stereo sequences | left/right RGB, per-image intrinsics, camera-to-world poses with a per-sequence metric scale; no lidar in the release |

`euler-known-datasets list` and `euler-known-datasets info <recipe>` show the
same from the command line. Per-dataset documentation:
[CEREMA](docs/datasets/cerema.md),
[DrivingStereo](docs/datasets/driving_stereo.md),
[KITTI-CARLA](docs/datasets/kitti_carla.md), [MUSES](docs/datasets/muses.md),
[nuScenes](docs/datasets/nuscenes.md), [ScatterNeRF](docs/datasets/scatternerf.md).

## Documentation

| Guide | Covers |
|---|---|
| [Concepts](docs/concepts.md) | Recipes, media strategies, the output contract, the receipt, resuming, verification |
| [Adding a recipe](docs/adding-a-recipe.md) | Writing and testing a recipe for a new dataset |
| [CEREMA](docs/datasets/cerema.md) | The nested release zip, scopes and baselines, ground truth per scene, the benchmark's protocol |
| [DrivingStereo](docs/datasets/driving_stereo.md) | Downloads, subsets, scopes, calibration matching |
| [KITTI-CARLA](docs/datasets/kitti_carla.md) | Town archives, options, lidar indexed per camera, conventions |
| [MUSES](docs/datasets/muses.md) | Packages, IDs, scopes including the ego-motion inputs |
| [nuScenes](docs/datasets/nuscenes.md) | Downloads, options, prepared layout, synchronisation, calibration file format |
| [ScatterNeRF](docs/datasets/scatternerf.md) | Nested duplicate archives, metric scale from the stereo baseline, why there is no lidar |
| [Roadmap](docs/roadmap.md) | Planned: zip-native and packed outputs, checksums and downloads |

## Licences

This package is MIT licensed. It never redistributes dataset content: each
recipe states its dataset's licence (`euler-known-datasets info <recipe>`),
and the receipt records it. Data shipped with the package itself is listed
in [NOTICE](NOTICE).

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest                       # synthetic datasets, no downloads needed
NUSCENES_ROOT=/data/nuscenes uv run pytest -m real
```

See [CONTRIBUTING.md](CONTRIBUTING.md).
