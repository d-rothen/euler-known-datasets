# DrivingStereo — recipe `driving_stereo`

Rectified stereo images with lidar-derived depth and disparity, for
[DrivingStereo](https://drivingstereo-dataset.github.io/): the training and
test sets and the four weather subsets, with per-sequence intrinsics.
Decoding is done by euler-loading's `driving_stereo` loader module.

- [Downloads](#downloads)
- [Usage](#usage)
- [Options](#options)
- [The prepared dataset](#the-prepared-dataset)
- [What the recipe knows about DrivingStereo](#what-the-recipe-knows-about-drivingstereo)
- [Reading the output](#reading-the-output)
- [Limits](#limits)

## Downloads

| Subset | Archives |
|---|---|
| train | `train/<kind>/<sequence>.zip`, half size |
| test | `test/half-size/<kind>/<sequence>.zip`, `test/full-size/<kind>/<sequence>.zip` |
| weather | `weather/<sunny\|cloudy\|foggy\|rainy>/<kind>-<half\|full>-size.zip` |
| calibration | `calibration/<train\|test>/<half\|full>-image-calib/<sequence>.txt` |

`<kind>` is `left-image`, `right-image`, `depth-map` or `disparity-map`. Not
every subset ships every kind at both sizes. Among the weather subsets, for
example, only rainy and sunny have full-size images. The recipe prepares
whatever is present. The source may hold the archives, the archives unzipped
in the directory that holds them, or both.

This layout is the one of the dataset's Hugging Face mirror, read from its file
tree. Inside the archives:

- A weather archive holds `<weather>/<kind>-<size>-size/<sequence>_<timestamp>.<ext>`.
  This was checked on the foggy subset.
- The inner layout of the train and test archives has not been checked. The
  recipe therefore accepts any directories between the archive's folder and
  the file names.

## Usage

```bash
euler-known-datasets prepare driving_stereo /data/downloads/DrivingStereo /data/driving_stereo
euler-known-datasets verify /data/driving_stereo
```

```python
from euler_known_datasets import prepare

prepare("driving_stereo", "/data/downloads/DrivingStereo", "/data/driving_stereo",
        subsets=["foggy"])
```

On the foggy subset (six archives, 0.6 GB), extracting from the zips on Lustre
takes 46 s.

## Options

| Option | Default | Meaning |
|---|---|---|
| `subsets` | every subset found | `train`, `test`, `sunny`, `cloudy`, `foggy`, `rainy` |
| `modalities` | every scope found | Scopes to prepare, from the table below. Leaving out `rgb_right`, for example, also skips extracting the right images. |

## The prepared dataset

```
/data/driving_stereo/
├── euler-known-datasets.json
├── weather/foggy/foggy/left-image-half-size/<sequence>_<timestamp>.jpg   (each archive unzipped in place)
├── train/left-image/…   test/half-size/…   test/full-size/…
├── calibration/<train|test>/half-image-calib/<sequence>.txt
└── .ds_crawler/<scope>/
```

**Sample IDs.** Each sample's full ID is
`/subset:<subset>/sequence:<sequence>/frame-<timestamp>`, e.g.
`/subset:foggy/sequence:2018-10-25-07-37-26/frame-2018-10-25-07-41-58-446`.

- **The subset is part of the hierarchy.** The weather subsets reuse
  training and test sequences, so one output can hold all subsets without ID
  clashes, and every subset is its own split.
- **Scopes join by ID.** All media scopes share these IDs, so euler-loading
  joins them.
- **Intrinsics apply per sequence.** The intrinsics file sits at
  `/subset:<subset>/sequence:<sequence>/` and applies to every frame below
  it.

| Scope | Modality key | euler-loading function | Files |
|---|---|---|---|
| `rgb` | `rgb` | `driving_stereo.rgb` | left images, half size |
| `rgb_right` | `rgb` | `driving_stereo.rgb` | right images, half size |
| `depth` | `depth` | `driving_stereo.depth` | `depth-map`, half size |
| `disparity` | `disparity` | `driving_stereo.disparity` | `disparity-map`, half size |
| `rgb_full_size`, `rgb_right_full_size` | `rgb` | `driving_stereo.rgb` | full-size images (full-size grid) |
| `depth_full_size` | `depth` | `driving_stereo.depth_full_size` | `depth-map`, full size, resampled onto the half-size grid |
| `disparity_full_size` | `disparity` | `driving_stereo.disparity_full_size` | `disparity-map`, full size, resampled onto the half-size grid |
| `intrinsics` | `intrinsics` | `driving_stereo.read_intrinsics` | `half-image-calib/<sequence>.txt`, one per subset and sequence |

The media scopes keep their regexes in `ds-crawler.json`, so they can be
re-crawled; `intrinsics` is a prebuilt index. A scope is written only when its
files are present. `intrinsics` is skipped when there are no calibration
files, and sequences without one are listed in the receipt's
`summary.intrinsics.missing`.

The heads declare the stored encodings: `range` for images. Depth heads add
`radial_depth: false`, `scale_to_meters: 1/256` and `range`. Depth and
disparity heads mark the maps as `sparse`, with `invalid.sentinel: 0` for
pixels without lidar. euler-loading's preprocessor resizes them without
blending in those zeros. Every head also records `resolution` (`half` or
`full`).

**Splits.** Every media scope gets one split per subset (`train`, `test`,
`foggy`, …), the dataset's own separation. Pass `split=` to the media scopes
only.

## What the recipe knows about DrivingStereo

### Grids and units

Every euler-loading function returns data on the pixel grid of
`left-image-half-size` (879×400 on the foggy subset, about 881×400 elsewhere):

- **Depth and disparity belong to the rectified left camera.** The half-size
  maps have the left image's size and grid.
- **Half-size maps** are uint16 / 256: metres for depth, half-size pixels for
  disparity.
- **Full-size maps.** Full-size depth is uint16 / 256 metres. Full-size
  disparity is uint16 / 128 in full-size pixels.
- **Exact doubling.** The full-size calibration is the half-size one doubled
  exactly, with no half-pixel offset. Half-size pixel `(y, x)` sits on
  full-size pixel `(2y, 2x)`, and `depth_full_size` and `disparity_full_size`
  resample onto that grid.
- **Half-size maps are not a downsampling of the full-size ones.** Where both
  are valid, the resampled full-size depth is within 5% of the half-size
  depth at 97.4% of pixels (foggy subset, 10 samples). Prefer the half-size
  maps where both sizes exist.
- **Depth is planar.** `depth × disparity = fx × baseline` holds per pixel.
  On the foggy subset, the median over 10 samples is 545.5 against
  1001.496 × 0.544915 = 545.7.

### Calibration

- The intrinsics are the rectified `P_rect_101[:, :3]` from the
  **`half-image-calib`** file. The right camera's `P_rect_103[:, :3]` is the
  same matrix, so no separate scope is needed.
- The `full-image-calib` file describes the same cameras at twice the
  resolution and is not used.
- In the half-size files, `P_rect_103[0, 3]` was not halved and still holds
  the full-size `-fx × baseline`. Take the baseline from `T_103` (0.545 m).
- `K_*`, `D_*` and `S_*` describe the raw, unrectified sensor images, which
  are not released.
- Calibration is per sequence and split. The 38 training and 4 test sequences
  share no name. Weather subsets reuse those sequences: foggy's
  `2018-10-25-07-37-26` is a training sequence. A weather frame therefore
  takes the calibration of its sequence from whichever split has it.

## Reading the output

```python
from euler_loading import Modality, MultiModalDataset

root = "/data/driving_stereo"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb", split="foggy"),
        "depth": Modality(root, metadata_scope="depth", split="foggy"),
        "disparity": Modality(root, metadata_scope="disparity", split="foggy"),
    },
    hierarchical_modalities={
        "intrinsics": Modality(root, metadata_scope="intrinsics", collapse_single=True),
    },
)
sample = dataset[0]  # full_id "/subset:foggy/sequence:2018-10-25-07-37-26/frame-2018-10-25-07-41-58-446"
sample["rgb"].shape, sample["depth"].shape  # (3, 400, 879), (1, 400, 879)
```

Automatic resolution uses euler-loading's torch variant. On an install
without torch, pass `loader=` from `euler_loading.loaders.cpu.driving_stereo`,
together with the matching `writer=`; otherwise writer resolution still
imports the torch variant.

## Limits

- The inner layout of the train and test archives has not been checked
  against real downloads. The regexes accept any folders inside them, and the
  synthetic tests cover two layouts.
- Calibration is read only as the plain `.txt` files the mirror ships. A
  zipped calibration download is not recognised.
- Each archive is extracted. Reading the zips in place is on the
  euler-known-datasets roadmap.
