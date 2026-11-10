# ScatterNeRF — recipe `scatternerf`

Rectified stereo frames of foggy drives with per-image intrinsics and
camera-to-world poses, as released for ScatterNeRF (Ramazzina et al., ICCV
2023). The release has **no lidar**. Decoding is done by euler-loading's
`scatternerf` loader module.

- [Downloads](#downloads)
- [Usage](#usage)
- [Options](#options)
- [The prepared dataset](#the-prepared-dataset)
- [What the recipe knows about ScatterNeRF](#what-the-recipe-knows-about-scatternerf)
- [Reading the output](#reading-the-output)
- [Limits](#limits)

## Downloads

The dataset is distributed on request as temporary download links, one per
`sequence_<n>.zip`. Each archive holds one sequence in the NSVF layout used by
NeRF pipelines:

```
<n>/rgb/<camera>_<frame>_<timestamp ns>.png         camera: left | right
<n>/intrinsics/<camera>_<frame>_<timestamp ns>.txt  4x4, row-major
<n>/pose/<camera>_<frame>_<timestamp ns>.txt        4x4 camera-to-world, row-major
```

As published (checked 2026-11-10), the ten archives hold only **three
distinct sequences**. Most archives wrap another archive, often several levels
deep:

| Archive | Holds |
|---|---|
| `sequence_0.zip` | sequence 0 |
| `sequence_4.zip` | `squence_0.zip` (sic), a byte-identical copy of sequence 0 |
| `sequence_6.zip` | sequence 6 |
| `sequence_8.zip` … `sequence_12.zip` | each wraps the previous one, down to `sequence_6.zip`: copies of sequence 6 |
| `sequence_13.zip` | sequence 13 |
| `sequence_14.zip` | `sequence_13.zip`, a copy of sequence 13 |

The recipe therefore never trusts an archive's name. It follows the wrapping,
takes the sequence number from the folder inside the innermost archive, and
compares duplicate copies by their members' sizes and CRCs. Identical copies
are noted in the receipt and the least nested one is read. Two copies with
different content stop the run. The source may hold the archives, their
extracted `<n>/` folders, or both.

## Usage

```bash
euler-known-datasets prepare scatternerf /data/downloads/princeton_scatternerf /data/scatternerf
euler-known-datasets verify /data/scatternerf
```

```python
from euler_known_datasets import prepare

prepare("scatternerf", "/data/downloads/princeton_scatternerf", "/data/scatternerf",
        sequences=["0"], cameras=["left"])
```

All three sequences (345 frames, 0.4 GB) are extracted from the zips on
Lustre in about 35 s.

## Options

| Option | Default | Meaning |
|---|---|---|
| `sequences` | every sequence found | Sequence numbers, as in the archives' folders (`0`, `6`, `13`) |
| `cameras` | `left`, `right` | Stereo cameras to prepare. The metric scale still uses both cameras' poses. |

## The prepared dataset

```
/data/scatternerf/
├── euler-known-datasets.json
├── 0/  6/  13/            {rgb,intrinsics,pose}/<camera>_<frame>_<timestamp ns>.{png,txt}
└── .ds_crawler/{rgb,intrinsics,camera_pose}/
```

**Sample IDs.** Each sample's full ID is
`/sequence:<n>/camera:<left|right>/frame-<frame>`, e.g.
`/sequence:6/camera:left/frame-00015`. The three scopes share these IDs, so
euler-loading joins them. The left and right image of one stereo pair have
the same frame number. Only frames with an image, intrinsics and pose are
indexed; the receipt counts the others as `incomplete_frames` (none in the
release).

| Scope | Modality key | euler-loading function | Files |
|---|---|---|---|
| `rgb` | `rgb` | `scatternerf.rgb` | `<n>/rgb/*.png`, 1704×908 |
| `intrinsics` | `intrinsics` | `scatternerf.read_intrinsics` | `<n>/intrinsics/*.txt`, one per image |
| `camera_pose` | `camera_extrinsics` | `scatternerf.camera_pose` | `<n>/pose/*.txt`, one per image |

All scopes are prebuilt indexes. Every file carries its `timestamp_ns` as an
attribute, and pose files also carry their sequence's `scale_to_meters`.
There are no splits: the release defines none.

The heads use the contract's geometry vocabulary. Intrinsics declare
`camera_model: pinhole`. Poses declare `source_frame: camera_optical`,
`target_frame: world` and `direction: target_from_source`, with a world frame
chosen per sequence.

The receipt's `summary.sequences.<n>` records per sequence the frames per
camera, the UTC time of the first frame, and the stereo-baseline measurement
behind the metric scale.

## What the recipe knows about ScatterNeRF

### Images and intrinsics

- **The images are the rectified SeeingThroughFog stereo pair.** They are
  cropped to 1704×908, with `fx = fy = 2355.7228` and the principal point at
  the image centre `(852, 454)`.
- **Intrinsics are an ideal pinhole.** Every file stores the same NSVF 4×4
  `[[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]]`, with no skew
  and no distortion. `read_intrinsics` checks this and returns the 3×3 block.

### Poses and metric scale

- **Poses are camera-to-world in optical axes** (x right, y down, z
  forward). The relative motion between two frames is
  `inv(pose_a) @ pose_b`.
- **Each sequence is in its own normalized units**, as usual for NeRF data.
  Its stereo baseline, the right camera's position in the left camera's axes,
  measures 0.021 to 0.038 units.
- **The metric scale comes from the stereo rig.** The SeeingThroughFog
  calibration (`calib_cam_stereo_right.json`) has `P[0, 3] = -fx × b =
  -478.200589` with `fx = 2355.722801`, so `b = 0.2030 m`. The recipe divides
  this by the median baseline of each sequence's stereo pairs:

  | Sequence | Recorded (UTC) | Frames left / right | Stereo pairs | Baseline (units) | `scale_to_meters` |
  |---|---|---|---|---|---|
  | 0 | 2018-10-08 07:02 | 41 / 50 | 29 | 0.021039 | 9.64864 |
  | 6 | 2023-02-14 07:08 | 69 / 65 | 37 | 0.033780 | 6.009308 |
  | 13 | 2023-02-14 07:34 | 61 / 59 | 30 | 0.038135 | 5.323105 |

  The baselines vary by 1–2% within a sequence and point along the left
  camera's +x axis (cosine ≥ 0.996). The recipe warns if a sequence's pairs
  are not such a clean x offset.

### No lidar

ScatterNeRF releases no lidar or depth. It cannot be recovered from
SeeingThroughFog either:

- Sequence 0 is a stretch of SeeingThroughFog's recording
  `2018-10-08_08-50-46`, frames 06880–06965. The released SeeingThroughFog
  lidar has no frames in that range.
- Sequences 6 and 13 were recorded on 2023-02-14, after SeeingThroughFog, and
  are not part of it.

Depth for these sequences would have to come from the stereo pairs themselves
(the rectified images with the 0.203 m baseline) or from a NeRF fit.

## Reading the output

```python
from euler_loading import Modality, MultiModalDataset

root = "/data/scatternerf"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb"),
        "intrinsics": Modality(root, metadata_scope="intrinsics"),
        "pose": Modality(root, metadata_scope="camera_pose", function="camera_pose_metric"),
    },
)
sample = dataset[0]  # full_id "/sequence:0/camera:left/frame-06880"
sample["rgb"].shape, sample["intrinsics"].shape, sample["pose"].shape  # (3, 908, 1704), (3, 3), (4, 4)
```

- `camera_pose` returns the poses as stored, in normalized units.
- `function="camera_pose_metric"` multiplies the translations by the file's
  `scale_to_meters`, so a stereo pair's baseline is 0.203 m.

Automatic resolution needs an euler-loading release newer than 2.41.0, the
first with the `scatternerf` module. Until then, `verify` warns that the
installed euler-loading lacks these functions. Without torch, pass `loader=`
from `euler_loading.loaders.cpu.scatternerf`, together with a `writer=`.

## Limits

- Only sequences 0, 6 and 13 are in the release. The archive names suggest
  sequences 4 and 8–12 and 14, but those archives hold copies (see
  [Downloads](#downloads)).
- When the only copy of a sequence is nested, it is read through its
  wrapping archives without unpacking them (a compressed wrapping is spooled
  to a temporary file).
- The images are not undistorted or re-rectified further: they are used as
  released.
