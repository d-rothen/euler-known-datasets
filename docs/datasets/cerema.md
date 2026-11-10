# CEREMA Pixel-Accurate Depth Benchmark — recipe `cerema`

Four static scenes recorded in the CEREMA weather chamber, each in clear
weather, in fog at 17 visibilities and in rain at two rates, by day and by
night. The data includes rectified stereo images, three gated-camera slices,
Velodyne HDL64-S3 lidar projected into the cameras, dense laser-scanned ground
truth, and the outputs of the four methods the benchmark evaluates. Source:
T. Gruber, M. Bijelic, F. Heide, W. Ritter, K. Dietmayer, "Pixel-Accurate
Depth Evaluation in Realistic Driving Scenarios", 3DV 2019
([code](https://github.com/gruberto/PixelAccurateDepthBenchmark)). Decoding is
done by euler-loading's `cerema` loader module.

- [Downloads](#downloads)
- [Usage](#usage)
- [Options](#options)
- [The prepared dataset](#the-prepared-dataset)
- [What the recipe knows about the benchmark](#what-the-recipe-knows-about-the-benchmark)
- [Reading the output](#reading-the-output)
- [Limits](#limits)

## Downloads

The dataset is distributed on request as a temporary download link to one
archive, `pixel_accurate_depth_benchmark.zip` (26.1 GB). It holds a stored zip
per folder in `pixel_accurate_depth_benchmark/`:

| Folder | Files | Content |
|---|---|---|
| `rgb_left_8bit`, `rgb_right_8bit` | 1600 PNG each | rectified stereo pair, 1920×1024, 8-bit |
| `gated0_10bit`, `gated17_10bit`, `gated31_10bit` | 1600 PNG each | gated slices, 1280×720, 16-bit PNG |
| `lidar_hdl64_rgb_left`, `lidar_hdl64_gated` | 1600 npz each | HDL64 lidar projected into the left camera and the gated camera |
| `intermetric_rgb_left`, `intermetric_gated` | 4 npz each | dense ground-truth depth, one per scene |
| `sgm`, `psmnet`, `monodepth`, `sparse2dense` | 1600 npz each | outputs of the benchmarked methods |
| `calibration` | 3 JSON | `calib_cam_stereo_left.json`, `calib_gated_bwv.json`, `calib_tf_tree.json` |

Every inner zip holds `<folder>/<file>`, except `calibration.zip`, whose files
sit at its root. Frames are named `scene<1-4>_<day|night>_<weather>_<0-9>`,
and the weather is `clear`, `fog20` … `fog100` (in steps of 5) or `rain15` /
`rain55`. That gives 4 scenes × 2 × 20 conditions × 10 frames = 1600 frames.

The source may hold the release zip, its inner zips (directly or in
`pixel_accurate_depth_benchmark/`), their extracted folders (also in `data/`,
where the benchmark code expects them), or a mix. The recipe reads the inner
zips inside the release zip in place, with no unpacked copy. An extracted
folder is used only when it holds every file of its archive.

## Usage

```bash
euler-known-datasets prepare cerema /data/downloads/CEREMA /data/cerema
euler-known-datasets verify /data/cerema
```

```python
from euler_known_datasets import prepare

prepare("cerema", "/data/downloads/CEREMA", "/data/cerema",
        modalities=["rgb", "lidar", "depth", "intrinsics"], weather=["clear", "fog"])
```

On Lustre, preparing everything from the release zip takes about 4 minutes
and writes 25 GB. A re-run with everything extracted takes a few seconds.

## Options

| Option | Default | Meaning |
|---|---|---|
| `modalities` | every scope found | Scopes to prepare, from the table below. Only their folders are extracted. |
| `scenes` | all | `1` … `4` |
| `time_of_day` | both | `day`, `night` |
| `weather` | all | `clear`, `fog`, `rain`, or one level such as `fog40` or `rain15` |

## The prepared dataset

```
/data/cerema/
├── euler-known-datasets.json
├── rgb_left_8bit/scene1_day_clear_0.png  …   (each folder as released)
├── intermetric_rgb_left/scene1.npz  …
├── calibration/calib_cam_stereo_left.json, calib_gated_bwv.json
└── .ds_crawler/<scope>/
```

**Sample IDs.** Each sample's full ID is
`/scene:<n>/time_of_day:<day|night>/weather:<weather>/frame-<0-9>`, e.g.
`/scene:2/time_of_day:night/weather:fog40/frame-3`.

- **Scopes join by ID.** All frame scopes share these IDs, so euler-loading
  joins them.
- **Condition attributes.** Fog frames carry the file attribute
  `fog_visibility_m`, and rain frames `rain_mm_h`.
- **Ground truth is indexed once per frame.** The ground-truth scopes list
  the scene's single file under every frame of that scene, so they are
  regular modalities that join like any other.
- **Calibration is shared.** The intrinsics scopes index their JSON file at the
  root, so it applies to every sample.

| Scope | Modality key | euler-loading function | Files |
|---|---|---|---|
| `rgb`, `rgb_right` | `rgb` | `cerema.rgb` | stereo images |
| `gated0`, `gated17`, `gated31` | `gated` | `cerema.gated` | gated slices |
| `lidar` | `depth` (sparse) | `cerema.depth` | `lidar_hdl64_rgb_left` |
| `lidar_gated` | `depth` (sparse) | `cerema.depth` | `lidar_hdl64_gated` |
| `depth` | `depth` | `cerema.depth` | `intermetric_rgb_left`, the scene's file for every frame |
| `depth_gated` | `depth` | `cerema.depth` | `intermetric_gated`, likewise |
| `intrinsics` | `intrinsics` | `cerema.read_intrinsics` | `calibration/calib_cam_stereo_left.json` |
| `gated_intrinsics` | `intrinsics` | `cerema.read_intrinsics` | `calibration/calib_gated_bwv.json` |
| `sgm` | `depth` (sparse) | `cerema.sgm_depth` | SGM output |
| `psmnet` | `depth` | `cerema.psmnet_depth` | PSMNet output |
| `monodepth` | `depth` | `cerema.monodepth_depth` | Monodepth output |
| `sparse2dense` | `depth` | `cerema.depth` | Sparse-to-Dense output |

**Families and baselines.** Every scope belongs to the family `cerema`. The
sensor and ground-truth scopes carry `gt: true, baseline: false`. The four
method scopes carry `gt: false, baseline: true` and `method` (`SGM`,
`PSMNet`, `Monodepth`, `Sparse-to-Dense`), so euler-view lists them as
baselines next to model predictions.

**Head meta.**
- Depth heads declare `radial_depth: false`, `scale_to_meters: 1`, `range`,
  `invalid.sentinel: 0` and the grid's `dimensions`.
- The lidar heads and SGM's head add `sparse: true`.
- The `depth` head also records the benchmark's evaluation crop and maximum
  depth (`benchmark`).
- Gated heads record the dark level, 87.

**Splits.** There are none: the release is a test benchmark and defines no
split. The hierarchy selects conditions.

## What the recipe knows about the benchmark

### Grids and calibration

- **The stereo images are rectified.** Their camera matrix is `P[:, :3]` of
  `calib_cam_stereo_left.json` (fx = fy = 2355.72, principal point (988.1,
  508.1)). The right image shares it.
- **`K` and `D` describe the raw sensor.** In that file, `K` (fx 2612.86)
  and `D` belong to the unrectified sensor, and the benchmark never uses
  them.
- **The stereo baseline is 0.202993 m.** It is the benchmark code's constant;
  the release has no right-camera calibration.
- **The gated camera has its own grid.** It is 1280×720 with
  `calib_gated_bwv.json` (fx = fy = 2322.4), and `lidar_hdl64_gated` and
  `intermetric_gated` lie on it.
- **`calib_tf_tree.json`** holds the vehicle's sensor frames (ROS tf). It is
  not indexed, since the lidar comes projected already.

### Depth

- **Depth is z-depth in meters**, with `0` where invalid, for the ground
  truth, the lidar and `sparse2dense`. In clear weather the projected lidar
  agrees with the ground truth with a median ratio of 1.000–1.004 in every
  scene and in both grids, and shows no trend with the off-axis angle. A
  radial range would grow by up to 10 % towards the image corners.
- **Value ranges.** The ground truth is valid at every pixel. It reaches
  58.9 m in scene 1 and 26.8 m in scenes 2–4. Lidar returns reach 106 m
  (median density 0.3 % of the pixels).
- **The ground truth is a static scan per scene.** It was acquired with a
  Leica ScanStation P30 from several accumulated positions; mannequins stood
  in for pedestrians so that the scene stayed fixed. The benchmark code
  applies the scene's file to every condition and frame.
- **The method outputs are stored as each method produced them.** The
  loaders convert them as the benchmark's evaluation code does, with
  `depth = 0.202993 × 2355.722801 / disparity`:
  - `sgm`: half-resolution disparity, 412×960, covering rows 70–893 of the
    full grid. Resized ×2 and doubled. NaN marks pixels without a match.
  - `psmnet`: disparity in pixels of a 960-wide image, 512×960. Resized to
    the full grid and doubled.
  - `monodepth`: disparity relative to the image width, 256×512. Resized and
    multiplied by 1920. The paper fine-tuned it on SeeingThroughFog's
    training split.
  - `sparse2dense`: metric depth on the full grid already.
- **Monodepth has extreme outliers.** Its near-zero disparities become
  depths of up to 10¹² m. The benchmark clips predictions at 28 m, so they
  count as far; filter or clip them before use.
- **The conversions check out.** Measured against the ground truth inside
  the benchmark crop and up to 28 m, without hole filling, the median
  absolute relative error (median over frames) is:

  | Condition | SGM | PSMNet | Monodepth | Sparse-to-Dense |
  |---|---|---|---|---|
  | day, clear | 0.026 | 0.048 | 0.134 | 0.050 |
  | day, fog 40 m | 0.041 | 0.063 | 0.186 | 0.453 |
  | day, rain 55 | 0.051 | 0.085 | 0.187 | 0.395 |
  | night, clear | 0.039 | 0.064 | 0.188 | 0.051 |
  | night, fog 40 m | 0.137 | 0.124 | 0.163 | 0.333 |

  SGM matches only about half of the pixels in the crop (median coverage
  0.49); the others are `0`.

### Evaluation protocol (as published)

- **Crop.** Predictions are compared on the `rgb_left` grid, cropped by 270
  rows at the top, 20 at the bottom, 170 columns on the left and 20 on the
  right.
- **Clipping.** Both prediction and ground truth are clipped to
  `[0.001, 28]` m.
- **Hole filling.** Before cropping, methods with less than full density
  (lidar, SGM) are filled with nearest-neighbour interpolation. The loaders
  do not do this.
- **Reporting.** Metrics are averaged per condition over the 40 frames of the
  four scenes, and also binned by ground-truth depth in 14 bins over 0–28 m.

### Conditions

- **Fog** levels are the chamber's meteorological visibility in meters, from
  20 to 100 in steps of 5.
- **Rain** levels are 15 and 55, light and heavy rain. The paper prints the
  unit as "mm/h/m²"; the attribute name says mm/h.
- **Day and night** come from the chamber's transparent roof being left open
  or covered.

## Reading the output

```python
from euler_loading import Modality, MultiModalDataset

root = "/data/cerema"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb"),
        "lidar": Modality(root, metadata_scope="lidar"),
        "depth": Modality(root, metadata_scope="depth"),
        "psmnet": Modality(root, metadata_scope="psmnet"),
    },
    hierarchical_modalities={
        "intrinsics": Modality(root, metadata_scope="intrinsics", collapse_single=True),
    },
)
sample = dataset[0]  # full_id "/scene:1/time_of_day:day/weather:clear/frame-0"
sample["rgb"].shape, sample["depth"].shape, sample["psmnet"].shape  # (3, 1024, 1920), (1, 1024, 1920) twice
```

Automatic resolution needs an euler-loading release newer than 2.41.0, the
first with the `cerema` module. Without torch, pass `loader=` from
`euler_loading.loaders.cpu.cerema`, together with a `writer=`.

## Limits

- **The lidar comes only projected**, as depth maps in two camera grids.
  Back-projecting a map with its camera's intrinsics gives camera-frame
  points, rounded to pixel centres.
- **The release ships three gated slices** (`0`, `17`, `31`, as named). The
  recipe does not interpret the slice numbers. Their raw values reach about
  1240, so the benchmark's dark-level subtraction and clipping to 10 bits
  saturates the brightest pixels.
