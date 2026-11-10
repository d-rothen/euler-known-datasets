# nuScenes — recipe `nuscenes`

Camera images, the `LIDAR_TOP` sweep synchronised to each image, camera
intrinsics, and the per-frame lidar→camera transform, for
[nuScenes](https://www.nuscenes.org/nuscenes) v1.0-trainval, v1.0-test and
v1.0-mini, with the official scene splits.

- [Downloads](#downloads)
- [Usage](#usage)
- [Options](#options)
- [The prepared dataset](#the-prepared-dataset)
- [What the recipe knows about nuScenes](#what-the-recipe-knows-about-nuscenes)
- [Calibration file format, version 1](#calibration-file-format-version-1)
- [Reading the output](#reading-the-output)
- [Limits](#limits)

## Downloads

| Version | Archives |
|---|---|
| v1.0-trainval | `v1.0-trainval_meta.tgz`, `v1.0-trainval01_blobs.tgz` … `v1.0-trainval10_blobs.tgz` |
| v1.0-test | `v1.0-test_meta.tgz`, `v1.0-test_blobs.tgz` |
| v1.0-mini | `v1.0-mini.tgz` |

All of them extract into one layout: `v1.0-<version>/` (the JSON tables),
`samples/` (keyframes), `sweeps/` (everything else) and `maps/`. The source
directory may hold the archives, their extracted contents, or both. Other
downloads (CAN bus, map expansion, lidarseg) are ignored. nuScenes is licensed
CC BY-NC-SA 4.0 for non-commercial use.

## Usage

```bash
euler-known-datasets prepare nuscenes /data/downloads/nuscenes /data/nuscenes \
    --cameras CAM_FRONT_LEFT
euler-known-datasets verify /data/nuscenes
```

```python
from euler_known_datasets import prepare

prepare("nuscenes", "/data/downloads/nuscenes", "/data/nuscenes", cameras=["CAM_FRONT_LEFT"])
```

On v1.0-test with `CAM_FRONT_LEFT`, already extracted on Lustre, `prepare` takes
33 s at 1.4 GB peak memory (the tables are read once), and `verify` takes 7 s.
The output adds 68 MB of indexes and 35 MB of calibration files. The tables of
v1.0-trainval are about six times larger.

## Options

| Option | Default | Meaning |
|---|---|---|
| `versions` | every version found | `v1.0-trainval`, `v1.0-test`, `v1.0-mini`. Mini is a subset of trainval and cannot share an output with it. |
| `cameras` | all six | Camera channels to prepare. |
| `keyframes_only` | `false` | Only the 2 Hz keyframes in `samples/`. Frames from `sweeps/` (~12 Hz) are included by default. |
| `sync_tolerance_ms` | `20` | Largest deviation from a camera's nominal lidar offset for a sweep to count as synchronised. Must stay below 25 ms; see [synchronisation](#synchronisation). |

## The prepared dataset

```
/data/nuscenes/
├── euler-known-datasets.json
├── samples/  sweeps/  maps/  v1.0-test/  LICENSE     (linked, hard-linked, copied or extracted)
├── calibration/<scene>/<camera>.json
└── .ds_crawler/
    ├── rgb/                 camera images
    ├── lidar/               LIDAR_TOP sweeps, indexed under the camera frame they pair with
    ├── intrinsics/          calibration/<scene>/<camera>.json, read as K
    └── camera_extrinsics/   the same files, read as the frame's lidar→camera transform
```

The output is still a valid nuScenes root for the devkit.

**Sample IDs.** Every sample is `/scene:<scene>/camera:<camera>/<timestamp>`.
The timestamp is that of the camera image, in microseconds, e.g.
`/scene:scene-0830/camera:CAM_FRONT_LEFT/1538450717104844`. The `rgb` and
`lidar` scopes share these IDs, so euler-loading joins each image with its
sweep. The two calibration scopes hold one file per scene and camera at
`/scene:<scene>/camera:<camera>/`, which applies to every frame below it.

| Scope | Modality key | One entry per | euler-loading function |
|---|---|---|---|
| `rgb` | `rgb` | camera frame | `nuscenes.rgb` |
| `lidar` | `lidar_point_cloud` | camera frame with a synchronised sweep | `nuscenes.lidar_point_cloud` |
| `intrinsics` | `intrinsics` | scene and camera | `nuscenes.read_intrinsics` |
| `camera_extrinsics` | `camera_extrinsics` | scene and camera | `nuscenes.read_extrinsics` (selected per sample) |

Index attributes: `rgb` entries carry `is_key_frame` and the
`sample_data_token`. `lidar` entries carry the sweep's `sample_data_token`,
`lidar_timestamp` and `offset_ms` (`t_lidar - t_cam`).

**Splits.** `rgb` and `lidar` get the official splits from the nuscenes-devkit
(`train`/`val` for trainval, `test`, `mini_train`/`mini_val`), plus
`<split>_keyframes`, restricted to keyframes. Pass `split=` to the regular
modalities only.

## What the recipe knows about nuScenes

These facts were measured on v1.0-test and are re-checked on every run (see the
receipt's `summary`).

### Calibration and poses

- `calibrated_sensor` has one row per scene and sensor, but the values are
  constant within a recording log. v1.0-test has only two distinct
  `CAM_FRONT_LEFT` calibrations. Intrinsics therefore sit at the
  scene/camera level.
- `ego_pose` has one row per `sample_data`: every image and every sweep has its
  own vehicle pose, at its own timestamp. The lidar→camera transform is
  therefore **per frame**, chained through both poses (the devkit's
  `map_pointcloud_to_image`):

  `lidar_to_camera = inv(camera_to_ego) · inv(ego_to_global @ t_cam) · (ego_to_global @ t_lidar) · lidar_to_ego`

  Dropping the two poses, i.e. using `inv(camera_to_ego) · lidar_to_ego`, is
  off by 0.44 m at 10.6 m/s, which is visible in projections.
- `LIDAR_TOP` axes are x right, y forward, z up; the vehicle frame is x
  forward, y left, z up; cameras use optical axes (x right, y down, z
  forward). The intrinsics are a pinhole `K` without distortion coefficients.

### Synchronisation

A `LIDAR_TOP` timestamp marks the **end** of a 50 ms clockwise rotation. Each
camera is triggered when the beam crosses its view centre, so it fires a fixed
time before its sweep ends:

| Camera | Azimuth in the lidar frame | Fires before the sweep ends |
|---|---|---|
| CAM_FRONT_LEFT | 145.4° | 43.0 ms |
| CAM_FRONT | 90.6° | 35.4 ms |
| CAM_FRONT_RIGHT | 32.4° | 27.4 ms |
| CAM_BACK_RIGHT | −22.5° | 19.8 ms |
| CAM_BACK | −90.5° | 10.3 ms |
| CAM_BACK_LEFT | −161.4° | 0.4 ms |

The offsets are linear in azimuth, 360° to 50 ms. The recipe estimates each
camera's nominal offset from nuScenes' own keyframe association: the median
over keyframes of the camera and LIDAR_TOP rows of the same `sample`. It then
pairs every frame, keyframe or sweep, with the sweep whose `t_lidar - t_cam` is
closest to that offset:

- **Not the nearest timestamp.** For `CAM_FRONT_LEFT`, the nearest sweep is
  usually the *previous* one (about 7 ms earlier), whose points in the
  camera's view are about 50 ms older than the image. For 5848 of 6008
  keyframes, nearest-timestamp pairing picks a different sweep than nuScenes
  pairs them with.
- **Keyframes agree with nuScenes.** The rule pairs all 6008 keyframes with
  their sample's own sweep.
- **Jitter.** Some sweeps are stamped up to ~14 ms late, and the next one is
  stamped correspondingly early; their periods still add up to 100 ms. The
  sweep is still the right one.
- **Dropped sweeps.** Above the jitter there is a clean gap: deviations run
  continuously up to 13.7 ms, then jump to 46.3 ms. Past the gap, the sweep that
  covered the image was dropped. That is 121 of 34,902 `CAM_FRONT_LEFT` frames
  (0.35%), none of them keyframes. These frames stay in `rgb` but have no entry
  in `lidar`, so joining the two drops them, and euler-loading reports them as
  unmatched.

The default tolerance of 20 ms sits in that gap. It also stays below half a
rotation, so a neighbouring sweep can never be accepted.

Camera frames are also on the lidar's 50 ms grid between keyframes (consecutive
frames are 50, 100 or 150 ms apart), so sweeps from `sweeps/` are synchronised
just like keyframes.

## Calibration file format, version 1

`calibration/<scene>/<camera>.json`, compact JSON:

```json
{
  "format": {"name": "euler_known_datasets.nuscenes.camera_frames", "version": 1},
  "generator": {"name": "euler-known-datasets", "version": "0.1.0"},
  "dataset_version": "v1.0-test",
  "scene": {"name": "scene-0077", "token": "bc6a757d637f4832be68986833ec17ac", "log": "n008-2018-08-01-16-03-27-0400",
            "location": "boston-seaport", "vehicle": "n008"},
  "conventions": {"matrices": "row-major; X_to_Y maps homogeneous points from frame X into frame Y", "…": "…"},
  "camera": {"channel": "CAM_FRONT_LEFT", "calibrated_sensor_token": "…", "width": 1600, "height": 900,
             "intrinsics": [[3x3]], "camera_to_ego": [[4x4]]},
  "lidar": {"channel": "LIDAR_TOP", "calibrated_sensor_token": "…", "lidar_to_ego": [[4x4]],
            "nominal_offset_ms": 43.008, "sync_tolerance_ms": 20.0},
  "frames": {
    "1533153857904799": {
      "sample_data_token": "…", "sample_token": "…", "is_key_frame": true,
      "path": "samples/CAM_FRONT_LEFT/n008-2018-08-01-16-03-27-0400__CAM_FRONT_LEFT__1533153857904799.jpg",
      "camera_to_global": [[4x4]],
      "lidar": {"sample_data_token": "47d2f6131ba449f6a03ced4bde2c6dbe", "timestamp": 1533153857947444,
                "offset_ms": 42.645,
                "path": "samples/LIDAR_TOP/n008-2018-08-01-16-03-27-0400__LIDAR_TOP__1533153857947444.pcd.bin",
                "lidar_to_camera": [[4x4]]}
    }
  }
}
```

- `frames` keys are the sample IDs, so a reader selects a sample's record by
  its ID.
- `lidar` is `null` for a frame without a synchronised sweep, and the
  top-level `lidar` is `null` for a scene without any sweeps.
- `camera_to_global` places the camera in the scene's map frame, e.g. for
  multi-frame or pose supervision. Relative motion between two frames of a
  scene is `inv(camera_to_global_a) @ camera_to_global_b`.
- Every value traces back to the nuScenes tables through the tokens.

A change to this format bumps `format.version` and the recipe's
`layout_version`.

## Reading the output

`.pcd.bin` files are float32, five columns: x, y, z (metres, LIDAR_TOP frame),
intensity and ring index. With euler-loading's `nuscenes` module:

```python
import numpy as np
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
sample = dataset[0]
points = np.asarray(sample["lidar"])[:, :3]
camera = (np.asarray(sample["extrinsics"]) @ np.c_[points, np.ones(len(points))].T)[:3]
uvw = np.asarray(sample["intrinsics"]) @ camera[:, camera[2] > 1.0]
pixels, depth = uvw[:2] / uvw[2], uvw[2]
```

euler-loading's `nuscenes` module (in the release after 2.39.0) reads the
four scopes, and the heads name it, so the dataset loads without explicit
loaders; `verify` warns when the installed euler-loading predates it. Its
functions:

| Function | Reads | Returns |
|---|---|---|
| `rgb` | `.jpg` | image in `[0, 1]` |
| `lidar_point_cloud` | `.pcd.bin` | `(N, 5)` float32 |
| `read_intrinsics` | calibration file | `camera.intrinsics`, 3×3 |
| `read_extrinsics` | calibration file, selected per sample by ID | `frames[id].lidar.lidar_to_camera`, 4×4 |

## Limits

- Annotations, radar and the lidarseg/panoptic expansions are not prepared.
- One sweep per frame. Accumulating neighbouring sweeps, as the devkit's
  `from_file_multisweep` does, needs the sweeps' poses; `camera_to_global`
  together with the lidar index's tokens is the starting point.
- The timing facts above were measured on v1.0-test. Each run re-estimates the
  nominal offsets and reports in its summary how many keyframes agree with
  nuScenes' own pairing, so a deviating version shows up there.
