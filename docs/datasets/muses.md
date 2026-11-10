# MUSES — recipe `muses`

The frame camera, clear-weather reference frames, lidar, GNSS/IMU records,
semantic (and panoptic) ground truth, the shared calibration and per-frame
sensor timing of [MUSES](https://muses.vision.ee.ethz.ch), the multi-sensor
adverse-condition driving dataset, with its official train/val/test splits.

- [Downloads](#downloads)
- [Usage](#usage)
- [Options](#options)
- [The prepared dataset](#the-prepared-dataset)
- [Ego-motion inputs](#ego-motion-inputs)
- [What the recipe knows about MUSES](#what-the-recipe-knows-about-muses)
- [Limits](#limits)

## Downloads

MUSES is downloaded per data type, after registering on its website, as zip
packages such as:

| Package | Holds |
|---|---|
| `frame_camera_trainvaltest.zip` | `muses/frame_camera/…_frame_camera.png` (RGB, 1920×1080) |
| `reference_frame_trainvaltest.zip` | `muses/reference_frame/…_reference_frame.png` |
| `lidar_trainvaltest.zip` | `muses/lidar/…_lidar.bin` |
| `gnss_trainvaltest.zip` | `muses/gnss/…_gnss.txt`, `muses/gnss/dataformat.txt` |
| `gt_semantic_trainval.zip` | `muses/gt_semantic/…_gt_labelIds.png`, `…_gt_labelTrainIds.png`, `…_gt_labelColor.png` |

Every package extracts into the same tree, as the dataset's README instructs
(`unzip <package>.zip -d /path/to`), and carries identical copies of
`muses/calib.json`, `muses/meta.json`, `README.md` and `License.pdf`. The recipe
recognises packages by their members, not their names, so the source may hold
the packages, the extracted `muses/` tree, or both. A `gt_panoptic` package is
recognised too, from the layout its README documents. Radar and event-camera
data are ignored, as euler-loading has no reader for them. MUSES is licensed
under an ETH Zurich end user licence agreement: non-commercial research only,
no redistribution (`muses/License.pdf`).

## Usage

```bash
euler-known-datasets prepare muses /data/downloads/muses /data/muses
euler-known-datasets verify /data/muses
```

```python
from euler_known_datasets import prepare

prepare("muses", "/data/downloads/muses", "/data/muses", modalities=["rgb", "lidar"], splits=["val"])
```

From the zip packages, `prepare` extracts only the files it indexes, reading
each through the zip's central directory. Extracting means inflating deflate
data, which is CPU-bound: on an ETH Euler login node it ran at about 6 MB/s,
so the val split (1.7 GB) took 3.3 minutes. Run a full prepare (13 GB) in a
compute job. With the packages already extracted, `media=auto` links the
`muses/` tree instead.

## Options

| Option | Default | Meaning |
|---|---|---|
| `modalities` | every one the source holds | Per-frame scopes to prepare: `rgb`, `reference_rgb`, `lidar`, `gnss`, `semantic_segmentation`, `semantic_segmentation_train_ids`, `semantic_segmentation_color`, `panoptic_segmentation`. |
| `splits` | all | `train`, `val`, `test`. |
| `weather` | all | `clear`, `fog`, `rain`, `snow`. |
| `time_of_day` | all | `day`, `night`. |

The calibration and timing scopes are always written; they cost two shared
files.

## The prepared dataset

```
/data/muses/
├── euler-known-datasets.json
├── muses/                        the official tree: <type>/<split>/<weather>/<time_of_day>/…,
│                                 calib.json, meta.json, README.md, License.pdf
└── .ds_crawler/<scope>/          one scope per modality
```

The output is still a MUSES root for the official SDK.

**Sample IDs.** Hierarchy `/<split>/<weather>/<time_of_day>/`, ID the six-digit
frame number, e.g. `/test/fog/day/670832`. This is the scheme existing MUSES
indexes use, and so do datasets derived from them with euler-preprocess (crops,
motion-compensated lidar). Those keep joining with a prepared output. The
sequence and frame are also available per file as `basename_properties`
(`{"sequence": "REC0176", "frame": "670832"}`).

| Scope | Modality key | Files | euler-loading function |
|---|---|---|---|
| `rgb` | `rgb` | `frame_camera/…_frame_camera.png` | `muses.rgb` |
| `reference_rgb` | `rgb` | `reference_frame/…_reference_frame.png` | `muses.reference_rgb` |
| `lidar` | `lidar_point_cloud` | `lidar/…_lidar.bin` | `muses.lidar_point_cloud` |
| `gnss` | `gnss` | `gnss/…_gnss.txt` (not `dataformat.txt`) | `muses.gnss` |
| `semantic_segmentation` | `semantic_segmentation` (`skyclass` `[23, 23, 23]`) | `gt_semantic/…_gt_labelIds.png` | `muses.semantic_segmentation` |
| `semantic_segmentation_train_ids` | `semantic_segmentation` (`skyclass` `[10, 10, 10]`) | `gt_semantic/…_gt_labelTrainIds.png` | `muses.semantic_segmentation` |
| `semantic_segmentation_color` | `semantic_segmentation_color` (`skyclass` `[70, 130, 180]`) | `gt_semantic/…_gt_labelColor.png` | `muses.semantic_segmentation_color` |
| `panoptic_segmentation` | `panoptic_segmentation` | `gt_panoptic/…_gt_panoptic.png` | `muses.panoptic_segmentation` |
| `intrinsics` | `intrinsics` | `calib.json`, root level | `muses.read_intrinsics` |
| `camera_extrinsics` | `camera_extrinsics` | `calib.json`, root level | `muses.read_extrinsics` (lidar → camera) |
| `lidar_to_gnss` | `lidar_to_gnss` | `calib.json`, root level | `muses.read_lidar_to_gnss` |
| `sensor_timing` | `sensor_timing` | `meta.json`, root level | `muses.sensor_timing` (selected per sample) |

The per-frame scopes keep their ds-crawler regexes in `ds-crawler.json`, so
they can be re-crawled. The root-level files apply to every sample; load them
as hierarchical modalities with `collapse_single=True`. The sky label lets
`Modality(…, function="sky_mask")` read any segmentation scope as a sky mask.

**Splits.** Every per-frame scope gets `split_train`, `split_val` and
`split_test`, as far as it has data. Ground truth exists for train and val
only.

```python
from euler_loading import Modality, MultiModalDataset

root = "/data/muses"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb", split="val"),
        "lidar": Modality(root, metadata_scope="lidar", split="val"),
        "gnss": Modality(root, metadata_scope="gnss", split="val"),
    },
    hierarchical_modalities={
        name: Modality(root, metadata_scope=name, collapse_single=True)
        for name in ("intrinsics", "camera_extrinsics", "lidar_to_gnss", "sensor_timing")
    },
)
```

## Ego-motion inputs

The MUSES lidar sweeps its field of view top to bottom in about 85 ms (84.7 ms
in the sweep we measured) while the car moves, so its points need ego-motion
compensation before they line up with the frame camera. The official SDK moves
each point to the camera's mid-exposure using the frame's GNSS record. Three
scopes supply that SDK's inputs per sample; the compensation itself is left to
consumers:

- `gnss`: the frame's u-blox record, joined by ID like `rgb` and `lidar`.
  `dataformat.txt` describes its 28 integer fields and is left out. The
  loader returns physical units, among them `ground_speed_m_s`,
  `roll_rate_deg_s`, `pitch_rate_deg_s` and `heading_rate_deg_s`. The heading
  rate is about the negative yaw axis, i.e. positive clockwise seen from above.
- `sensor_timing`: `meta.json`, decoded once. Each sample then gets its own
  `{"frame", "camera_exposure_start_us", "camera_exposure_end_us",
  "lidar_start_us", "lidar_end_us", "radar_start_us", "radar_end_us"}`
  record. It is found through the native file names the entries list under
  `path_to_*`, which the prepared output keeps.
- `lidar_to_gnss`: `extrinsics.lidar2gnss`, which maps lidar points into the
  GNSS record's frame.

The per-point lidar timestamp (column 6, seconds) together with these inputs
is all the compensation needs.

## What the recipe knows about MUSES

These facts were established on the downloaded packages:

- **Layout.** Every member is
  `muses/<type>/<split>/<weather>/<time_of_day>/<sequence>_frame_<frame>_<type>.<ext>`,
  as the packages' README documents. All 2,500 frames' split, weather and
  time of day in their paths agree with `meta.json`.
- **Splits.** 1,500 train, 250 val, 750 test frames, according to
  `meta.json`'s `split` field and the paths. `gt_semantic` covers train and
  val only.
- **IDs.** Frame numbers are unique across all 2,500 frames in 136 sequences,
  so the frame number alone identifies a frame. The recipe indexes with
  `flat_ids_unique`, so a violating download fails instead of silently
  merging frames.
- **Shared files.** `calib.json` and `meta.json` are byte-identical in every
  package, and are taken from one.
- **Reference frames** exist only for adverse conditions: 2,000 frames, none
  for clear day. Joining `rgb` with `reference_rgb` drops the clear-day frames.
- **Labels.** Sky is label ID 23, train ID 10 and colour `(70, 130, 180)` in
  the three encodings, checked on the released images; the heads declare it
  as `skyclass`.
- **Calibration.** `calib.json` holds `intrinsics.rgb.K` (and `D`) and the
  extrinsics `lidar2rgb`, `lidar2gnss` and others. `lidar2rgb` maps lidar
  points into optical camera axes: every point of a sweep lands in front of
  the camera, about two thirds of them inside the image. The released frames
  are already undistorted, so project with plain `K`.
- **Lidar** `.bin` files are float64 `(N, 6)`: x, y, z, intensity, ring,
  timestamp (seconds).

## Limits

- Radar, event camera, uncertainty and detection ground truth are not
  prepared; euler-loading has no readers for them.
- `gt_panoptic` support follows the README's layout but was not run on a real
  download.
- Zip packages are extracted. Reading them in place, without extraction, is
  on the roadmap.
