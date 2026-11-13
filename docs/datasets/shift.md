# SHIFT — recipe `shift`

SHIFT (T. Sun, M. Segu, J. Postels, Y. Wang, L. Van Gool, B. Schiele,
F. Tombari, F. Yu, "SHIFT: A Synthetic Driving Dataset for Continuous
Multi-Task Domain Adaptation", CVPR 2022;
[website](https://www.vis.xyz/shift), [devkit](https://github.com/SysCV/shift-dev))
renders drives in CARLA under many weathers and times of day. Its *discrete*
set has 4250 sequences of 50 s, each with fixed conditions. The recipe
prepares its **front camera**: images, planar depth, semantic labels, the
intrinsics and per-frame camera poses. It chooses frames by condition, by
spacing and by ego motion. Decoding is done by euler-loading's `shift` loader
module. The data is licensed CC BY-NC-SA 4.0.

- [Downloads](#downloads)
- [Decoding frames from the video-rate mirror](#decoding-frames-from-the-video-rate-mirror)
- [Usage](#usage)
- [Options](#options)
- [The prepared dataset](#the-prepared-dataset)
- [What the recipe knows about SHIFT](#what-the-recipe-knows-about-shift)
- [Reading the output](#reading-the-output)
- [Limits](#limits)

## Downloads

shift-dev's `download.py` filters by view, split, data group and frame rate.
For the image rate (1 fps: frames 0, 10, …, 490 of each sequence) it stores
`discrete/images/<split>/front/` with:

| File | Content |
|---|---|
| `img.zip` | `<seq>/<frame>_img_front.jpg` |
| `depth.zip` | `<seq>/<frame>_depth_front.png`, 24-bit depth |
| `semseg.zip` | `<seq>/<frame>_semseg_front.png`, class ID in red |
| `det_2d.json` | Scalabel frames: intrinsics, extrinsics, conditions, 2D boxes |
| `seq.csv` | each sequence's town, weather and time of day |

The recipe reads these zips, their extracted folders (`img/`, `depth/`,
`semseg/`, as shift-dev's `FileBackend` expects them), or both.

**The official server is gone.** `dl.cv.ethz.ch` returns NXDOMAIN and the
project page is unreachable (shift-dev issues #78–#80, 2026). The only mirror
found, the Hugging Face dataset `KevinConnorLee/SHIFT` (revision
`443326746aa5`; `vanthanh/SHIFT` is a copy), holds only the **video rate**
(10 fps) of `discrete/videos/{train,val}/front/`:

| File | train | val | Content |
|---|---|---|---|
| `img.tar` | 59.6 GB | 9.6 GB | one MPEG-4 Part 2 video per sequence (yuv420p, 10 fps, 501 frames) |
| `semseg.tar` | 21.1 GB | 3.6 GB | one H.264 4:4:4 video per sequence, lossless (`qp=0`) |
| `det_2d.json` | 3.1 GB | 0.5 GB | frames 0–499 of every sequence |
| `seq.csv` | 118 KB | 20 KB | |
| `depth.part.aa`…`af` | 722 GB | none | split zip of the train `depth.zip`: 3000 × 501 stored PNGs |

There is no image rate, no test split, no val depth, no other camera and no
continuous set. The image rate is every 10th video frame of the same
sequences, under the same names (checked against an archived copy of the
image-rate `det_2d.json`). `slurm/tools/remote_zip_subset.py` reads selected
members of the remote depth zip over HTTP range requests. It copied frames
0:500:10 of the train depth (150,000 PNGs, 72 GB) into
`discrete/videos/train/front/depth_1fps.zip`. Four of them sit in zero-filled
holes of the mirror's parts and are missing:

- `0917-57e3` frames 410 and 430;
- `13f9-6d84` frames 300 and 310.

None of them is a multiple of 40.

## Decoding frames from the video-rate mirror

The recipe never decodes media. `slurm/tools/shift_decode.py`, run by
`slurm/shift-decode.sbatch` (it needs ffmpeg:
`module load stack/2025-06 gcc/12.2.0 ffmpeg/7.0.2`), decodes the selected
frames into the extracted image-rate layout below
`discrete/images/<split>/front/`. Selection options work as for the recipe
(it imports the recipe's `selection.py`).

- **`img/<seq>/<frame>_img_front.png`** is frame n of `<seq>.mp4`, decoded to
  RGB and stored losslessly.
- **`semseg/<seq>/<frame>_semseg_front.png`** holds the labels of frame n's
  YUV triples (see [labels](#semantic-labels)), as an RGB PNG with R = label.
- **`depth/<seq>/<frame>_depth_front.png`** is copied unchanged from
  `depth_1fps.zip`.
- **`decoded_from_videos.json`** records the sources, the selection, the
  ffmpeg version and the frames without depth.

`det_2d.json` and `seq.csv` stay in `discrete/videos/<split>/front/`; the
recipe looks there when the image-rate folder has none. Video frame n is
frame n:

- the decoded sky matches the depth's far plane at the same index (IoU 1.0
  on the checked frames) and differs at n ± 1 (IoU 0.91–0.97);
- the image's edges align best with the depth at n.

On 16 cores, decoding 11,647 frames took 14 minutes, three of them listing
the tars. It wrote 13.1 GB: images 7.5 GB, depth 5.3 GB, labels 0.3 GB. No
label pixel fell outside the 14 triples.

## Usage

```bash
sbatch slurm/shift-decode.sbatch /data/SHIFT              # once, from the mirror's files
euler-known-datasets prepare shift /data/SHIFT /data/SHIFT_euler --media hardlink \
    --splits train --weather clear,cloudy,overcast --time-of-day daytime,dawn_dusk \
    --frame-step 40 --min-motion-m 3 --min-rotation-deg 5
euler-known-datasets verify /data/SHIFT_euler
```

```python
from euler_known_datasets import prepare

prepare("shift", "/data/SHIFT", "/data/SHIFT_euler", media="hardlink", splits=["train"],
        weather=["clear", "cloudy", "overcast"], time_of_day=["daytime", "dawn_dusk"],
        frame_step=40, min_motion_m=3, min_rotation_deg=5)
```

`--media hardlink` adds no data on the same filesystem. On a compute node the
prepare takes about 3 minutes, 40 s of it streaming `det_2d.json` (3.1 GB).

## Options

| Option | Default | Meaning |
|---|---|---|
| `modalities` | all found | `rgb`, `depth`, `semantic_segmentation`, `intrinsics`, `camera_pose` |
| `splits` | all found | `train`, `val` |
| `weather` | all | coarse weather: `clear`, `cloudy`, `overcast`, `rainy`, `foggy` |
| `time_of_day` | all | coarse time of day: `daytime`, `dawn_dusk`, `night` |
| `frame_step` | `10` | keep frames whose number is a multiple of this; a multiple of 10 (the image rate) |
| `min_motion_m` | `0` | motion filter, distance; `0` = off |
| `min_rotation_deg` | `0` | motion filter, yaw change; `0` = off |

The coarse classes group the fine ones:

| Coarse class | Fine conditions |
|---|---|
| weather `clear` | clear, slight cloudy |
| weather `cloudy` | partial cloudy |
| weather `overcast` | overcast |
| weather `rainy` | small rain, mid rain, heavy rain |
| weather `foggy` | small fog, heavy fog |
| time `daytime` | noon, morning/afternoon |
| time `dawn_dusk` | dawn/dusk, sunrise/sunset |
| time `night` | night, dark night |

The motion filter keeps a sequence's first candidate frame, then each
candidate whose camera moved at least `min_motion_m` (3D distance) **or**
turned at least `min_rotation_deg` (yaw, `rotation[2]`) since the last kept
frame. A frame enters the output only if every selected per-frame scope has
its file; the others are counted in the receipt (`frames_without_file`).

### The prepared selection

`SHIFT_euler` on /cluster/work uses train only, dry weather, day and
twilight, 4 s spacing and a 3 m / 5° motion filter (fog will be synthesised
on it). The counts after each step:

| Step | Sequences | Frames |
|---|---|---|
| train, `seq.csv` | 3000 | |
| weather clear / slight cloudy / partial cloudy / overcast, time noon / morning-afternoon / dawn-dusk / sunrise-sunset | 1243 | |
| every 4 s (`frame_step` 40: 0, 40, …, 480) | | 16,159 |
| moved ≥ 3 m or turned ≥ 5° | | 11,647 |
| with every file (no hole is a multiple of 40) | | 11,647 |

Every 5 s with the same filter gives 9,298 frames. Relaxing the filter adds
few (1 m / 3°: 9,660), because the frames it drops are standstills. The
selected sequences still carry CARLA's light haze: `fog_density` 0.1 (208
sequences), 1.0 (614) or 3.0 (421), with `fog_distance` 0.75 m.

## The prepared dataset

```
/data/SHIFT_euler/
├── discrete/images/train/front/
│   ├── img/<seq>/<frame>_img_front.png       (or .jpg from the official zips)
│   ├── depth/<seq>/<frame>_depth_front.png
│   └── semseg/<seq>/<frame>_semseg_front.png
├── calibration/train/<seq>.json              intrinsics and per-frame poses
├── euler-known-datasets.json                 the receipt
└── .ds_crawler/<scope>/                      one scope per modality
```

**Sample IDs.** The hierarchy is
`/split:<split>/weather:<coarse>/time_of_day:<coarse>/sequence:<seq>/` and the
ID is the 8-digit frame number, e.g.
`/split:train/weather:cloudy/time_of_day:dawn_dusk/sequence:0bb0-2222/00000040`.
Each per-frame file carries these attributes:

- the town;
- the fine weather and time of day;
- CARLA's weather parameters as numbers: `fog_density`, `fog_distance`,
  `fog_falloff`, `precipitation`, `precipitation_deposits`, `cloudiness`,
  `wind_intensity` and `sun_altitude_angle`.

| Scope | Modality key | Files | euler-loading function |
|---|---|---|---|
| `rgb` | `rgb` | `img/…` | `shift.rgb` |
| `depth` | `depth` (`radial_depth: false`, `scale_to_meters` 1000 / (2²⁴ − 1), range [0, 1000]) | `depth/…` | `shift.depth` |
| `semantic_segmentation` | `semantic_segmentation` (`skyclass` `[13, 0, 0]`) | `semseg/…` | `shift.semantic_segmentation` |
| `intrinsics` | `intrinsics` (per sequence) | `calibration/…` | `shift.read_intrinsics` |
| `camera_pose` | `camera_extrinsics` (per sequence, selected per frame) | `calibration/…` | `shift.camera_pose` |

The per-frame scopes have the official splits (`train`, `val`). The
calibration file, format `euler_known_datasets.shift.camera_frames` version 1,
holds these fields:

- `camera.intrinsics` (3×3), `width` and `height`;
- for each frame, `frames[<frame>].camera_to_world` (4×4, row-major, optical
  axes) and the `source` name, `location` and `rotation` from
  `det_2d.json`.

## What the recipe knows about SHIFT

**Sequences and frames.** Train has 3000 sequences and val 500. `det_2d.json`
annotates frames 0–499 at 10 fps; the videos and the depth have 501 frames
(0–500). Images are 1280×800. The conditions of a sequence never change: no
attribute of `det_2d.json` varies within any of the 3500 sequences, and
`seq.csv`'s start conditions match them.

**Intrinsics.** `fx = fy = 640`, principal point `(640, 400)`, skew 0, on all
1.75 million frames of train and val: an ideal 90° pinhole, the same for every
sequence.

**Depth** is CARLA's depth camera, stored as an 8-bit RGB PNG:
`(R + 256 G + 65536 B) / (2²⁴ − 1) × 1000` m, as shift-dev's loader decodes it.

- **It is planar (z-buffer) depth.** On the road, depth is constant along an
  image row: the edge-to-centre ratio is 0.97–1.03, where radial distance
  would give 1.08–1.41. With f = 640 and cy = 400, the road's depth gives the
  camera height, about 1.55 m, which is the height in the extrinsics.
- **The sky** is code 16777210 (999.9997 m), the largest code that occurs;
  2²⁴ − 1 never does. Every sky pixel of the segmentation has it. So do a few
  distant non-sky pixels beyond the far plane: 0.06 % of the pixels in 120
  frames. Use the segmentation's sky, not a depth threshold, for a sky mask.
- **No invalid pixels.** There are no zeros; finite depth reaches about
  1000 m.
- **The ego vehicle's hood** is in every frame, at the same pixels: an arc
  over the bottom rows, from row 770 at the image centre and row 787 a quarter
  width to either side, not reaching the sides, 1.0–1.2 m away. It is left in
  the data.

**Semantic labels.** CARLA 0.9.10–0.9.13's 23 tags, sky `13`, with the ID in
red and green = blue = 0 (shift-dev reads channel 0). The mirror's
`semseg.tar` encodes the colour `(ID, 0, 0)` after a BT.601 limited-range RGB
to YUV conversion, so nine pairs of tags become the same YUV triple. Every
decoded pixel holds one of the 14 triples. Decoded frames therefore use a
merged set:

- **Exact tags:** the sky (`13`) and four others keep their CARLA ID:
  `10` vehicle, `14` ground, `17` guard rail, `22` terrain.
- **Merged pairs** get IDs 23–31:

| ID | Merged tags |
|---|---|
| 23 | unlabeled, building |
| 24 | fence, other |
| 25 | pedestrian, pole |
| 26 | road line, road |
| 27 | sidewalk, vegetation |
| 28 | wall, traffic sign |
| 29 | bridge, rail track |
| 30 | traffic light, static |
| 31 | dynamic, water |

The head's `label_ids` says which set a scope holds (`carla_0.9.10` or
`shift_video_merged`), and `classes` names the IDs. SHIFT's official server
had replaced these videos with PNGs (`semseg.zip`, 2023); the mirror does not
have them.

**Images.** RGB decoded from the videos is lossy (MPEG-4 Part 2, 2.7–6.7
Mbit/s), unlike the official JPEGs; it is stored as PNG to avoid a second
loss.

**Poses.** Each frame's `extrinsics` give the camera's `location` (meters)
and `rotation` `[r0, r1, r2]` (degrees; shift-dev's loader takes them for
radians, issue #77) in CARLA's world. That world is left-handed: x and y in
the ground plane, z up. The recipe mirrors it into a right-handed world by
negating y. The camera-to-world pose of the optical frame (x right, y down,
z forward) is then:

```
R = Rz(-r2) · Ry(r1) · Rx(-r0) · B,   t = (x, -y, z)
```

`B` maps optical axes to body axes (x forward, y left, z up). `r2` is the
heading of the motion (within 5° in 96 % of moving frame pairs); `r1` is
minus the road's slope (correlation −0.96); `r0` is a roll within ±3.5°.

**How the poses were checked.** Depth of one frame was back-projected with
`K`, moved with the two frames' poses and compared with the other frame's
depth: 30 pairs of train frames 1–5 s apart in six sequences, rows above the
hood, depth below 80 m.

- With this convention the median relative error is **0.14 %**, and pairs
  through 34–57° turns stay at 0.6–5 %.
- Taking the world as given with `Rz(r2)` (a right-handed reading) gives 0.7 %
  overall but fails on turns: 47 % on a 9° turn.
- Flipping the roll's sign raises the error to 0.19 %, and flipping the
  pitch's to 2.3 %.

`tests/test_shift.py` repeats the reprojection on the fake dataset. Its
renderer builds the camera from the heading and pitch, not from the recipe's
matrices. `tests/test_real_shift.py` repeats it on the prepared data.

## Reading the output

```python
from euler_loading import Modality, MultiModalDataset

root = "/data/SHIFT_euler"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb"),
        "depth": Modality(root, metadata_scope="depth"),
        "sky": Modality(root, metadata_scope="semantic_segmentation", function="sky_mask"),
    },
    hierarchical_modalities={
        "intrinsics": Modality(root, metadata_scope="intrinsics"),
        "pose": Modality(root, metadata_scope="camera_pose"),
    },
)
sample = dataset[0]
sample["attributes"]["rgb"]["weather_fine"], sample["attributes"]["rgb"]["fog_density"]
```

`radial_distance` (function of the `depth` scope) turns planar depth into
distance along the ray.

## Limits

- Only the front camera of the discrete set; the continuous set, the other
  cameras and lidar are not on the mirror.
- Val has no depth on the mirror, and test none at all.
- Decoded RGB carries MPEG-4 artefacts; decoded labels merge nine class pairs.
- Selection is per frame number. The image-rate set holds every 10th frame,
  so `frame_step` must be a multiple of 10.
