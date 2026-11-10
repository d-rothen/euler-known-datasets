# KITTI-CARLA — recipe `kitti_carla`

RGB, depth and semantic segmentation of both cameras, the lidar sweep of every
frame, camera intrinsics, lidar→camera extrinsics and lidar poses for
[KITTI-CARLA](https://arxiv.org/abs/2109.00892). It is rendered with
CARLA 0.9.10 from a vehicle with KITTI's sensor layout: an HDL-64E-like
lidar and two 1392×1024 cameras with a 72° horizontal field of view, 5000
frames in each of seven towns.

- [Downloads](#downloads)
- [Usage](#usage)
- [Options](#options)
- [The prepared dataset](#the-prepared-dataset)
- [What the recipe knows about KITTI-CARLA](#what-the-recipe-knows-about-kitti-carla)
- [Limits](#limits)

## Downloads

The dataset comes as one bundle, `KITTI-CARLA.zip`. It holds
`KITTI-CARLA/dataset/Town01.zip` … `Town07.zip` (36–44 GB each), the paper and
two scripts. Unzip the bundle once. The recipe's source is the directory that
holds the town archives, `KITTI-CARLA/dataset/`. That directory may also hold
extracted `TownNN/` directories, alongside the archives or instead of them.
Each town archive holds `TownNN/generated/` with:

| Path | Content |
|---|---|
| `images_rgb/<frame>_<camera>.png` | RGBA images, cameras `0` and `1` |
| `images_ss/<frame>_1<camera>.png` | CARLA-palette semantic segmentation |
| `images_depth/<frame>_2<camera>.png` | depth, CARLA's 24-bit code |
| `frames/frame_<frame>.ply` | the lidar sweep, binary PLY |
| `lidar_to_cam0.txt`, `lidar_to_cam1.txt` | lidar→camera calibration |
| `full_poses_lidar.txt` | lidar poses at 1 kHz |
| `full_ts_camera.txt`, `poses_lidar.ply`, `screenshot.png` | not prepared |

The download states no licence (`readme.txt`); check the authors' terms
before redistributing.

## Usage

```bash
euler-known-datasets prepare kitti_carla /data/KITTI-CARLA/dataset /data/kitti_carla
euler-known-datasets prepare kitti_carla /data/KITTI-CARLA/dataset /data/kitti_carla_small \
    --towns Town01 --cameras 0 --modalities rgb depth lidar --frame-step 10
euler-known-datasets verify /data/kitti_carla
```

The town archives are read through their central directory. Selecting towns,
cameras, modalities or a frame step therefore reads and extracts only the
selected files. Every 1000th frame of Town01 (38 files, 0.18 GB) comes out of
its 37.8 GB archive in 12 s. Extraction from Lustre ran at 9–13 MB/s on a
login node, so a whole town takes about an hour; run it in a batch job.
Already extracted towns are linked with the default `--media auto`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `towns` | every town found | `Town01` … `Town07`. |
| `cameras` | `0 1` | Cameras to prepare. |
| `modalities` | `rgb depth semantic_segmentation lidar` | Per-frame scopes to prepare. `intrinsics`, `camera_extrinsics` and `lidar_pose` always come along. |
| `frame_step` | `1` | Keep the frames whose number is a multiple of N. Consecutive frames are 0.1 s apart. |

All towns of one run must be available in the same way: all extracted for
`symlink`/`hardlink`/`copy`, or all as archives for `extract`.

## The prepared dataset

```
/data/kitti_carla/
├── euler-known-datasets.json
├── Town01/generated/…            (linked, hard-linked, copied or extracted)
└── .ds_crawler/
    ├── rgb/  depth/  semantic_segmentation/      regex-indexed per camera
    ├── lidar/                                    each sweep under every prepared camera
    ├── intrinsics/  camera_extrinsics/           lidar_to_cam<N>.txt per town and camera
    └── lidar_pose/                               full_poses_lidar.txt per town
```

**Sample IDs.** Every sample is `/town:<town>/camera:<camera>/frame-<frame>`,
e.g. `/town:Town01/camera:0/frame-0000`. The frame is written as in the file
names. These are the IDs of euler-loading's KITTI-CARLA tests and of the
earlier in-archive indexes, so datasets derived from those indexes still join a
prepared town. The `1`/`2` that segmentation and depth names put before the
camera digit belong to neither the ID nor the hierarchy.

| Scope | Modality key | One entry per | euler-loading function | Index |
|---|---|---|---|---|
| `rgb` | `rgb` | town, camera, frame | `kitti_carla.rgb` | regexes |
| `depth` | `depth` | town, camera, frame | `kitti_carla.depth` | regexes |
| `semantic_segmentation` | `semantic_segmentation` | town, camera, frame | `kitti_carla.class_segmentation` (`sky_mask` with `Modality.function`) | regexes |
| `lidar` | `lidar_point_cloud` | town, camera, frame | `kitti_carla.lidar_point_cloud` | prebuilt |
| `intrinsics` | `intrinsics` | town and camera | `kitti_carla.read_intrinsics` | regexes, `id.override` |
| `camera_extrinsics` | `camera_extrinsics` | town and camera | `kitti_carla.read_extrinsics` | regexes, `id.override` |
| `lidar_pose` | `lidar_pose` | town | `kitti_carla.lidar_pose` (selected per sample) | regexes, `id.override` |

The regex-indexed scopes keep their regexes in `ds-crawler.json`, so they
can be re-crawled. Those regexes describe every frame and camera; the index,
not the regexes, carries the selection of a run.

**Why the lidar scope is per camera.** A sweep has no camera, but samples do.
The recipe indexes each `frames/frame_<frame>.ply` under every prepared camera
with that camera's frame ID. An image therefore joins its sweep by ID and takes
its camera's extrinsics from the hierarchy, and lidar projects into both
cameras with no other setup. Without it, joining images and lidar meant
restricting the images to one camera and dropping the camera level.

```python
from euler_loading import Modality, MultiModalDataset

root = "/data/kitti_carla"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb"),
        "depth": Modality(root, metadata_scope="depth"),
        "semantic": Modality(root, metadata_scope="semantic_segmentation"),
        "lidar": Modality(root, metadata_scope="lidar"),
    },
    hierarchical_modalities={
        "intrinsics": Modality(root, metadata_scope="intrinsics", collapse_single=True),
        "extrinsics": Modality(root, metadata_scope="camera_extrinsics", collapse_single=True),
        "pose": Modality(root, metadata_scope="lidar_pose", collapse_single=True),
    },
)
sample = dataset[0]  # full_id "/town:Town01/camera:0/frame-0000"
```

With the NumPy loaders (`euler_loading.loaders.cpu.kitti_carla`) on an
install without torch, pass each modality's `writer=` as well: euler-loading
otherwise resolves a writer from the head, which imports the torch loaders.

## What the recipe knows about KITTI-CARLA

The layout below is the same in all seven town archives: 5000 frames, cameras
`0` and `1`, 35,012 entries each. The real `Town01.zip` also carries a
`.ds_crawler/` directory written by earlier in-archive indexing; the recipe
ignores it.

- **Depth.** The PNG stores CARLA's 24-bit code
  `(R + 256 G + 65536 B)`, normalised to the 1000 m far plane. It is planar
  (z-buffer) depth. The head declares `radial_depth: false`,
  `scale_to_meters: 1000 / (2**24 - 1)` and `range: [0, 1000]`. euler-loading
  decodes it to metres itself. Sky is the far plane, exactly 1000 m.
- **Segmentation** uses CARLA 0.9.10's 23-tag palette; sky is
  `(70, 130, 180)`, declared as `skyclass`.
- **Camera model.** No intrinsics file exists. euler-loading's
  `read_intrinsics` computes the ideal pinhole from the paper (1392×1024, 72°).
  It reads a `lidar_to_cam<N>.txt` only to place the camera in the hierarchy.
- **Extrinsics.** `lidar_to_cam<N>.txt` maps lidar points into the camera in
  vehicle axes (x forward, y left, z up); `read_extrinsics` returns it in
  optical axes, so lidar projects directly. Prepared from the real
  `Town01.zip` (every 1000th frame), loaded through euler-loading and projected
  with the prepared intrinsics and extrinsics, a sweep matches `images_depth`
  (nearest pixel, sky excluded):

  | Sample | Projected points | Median depth error | 90th percentile | Same segmentation tag |
  |---|---|---|---|---|
  | `/town:Town01/camera:0/frame-0000` | 24,710 | 1.02 cm | 11.2 cm | 97.2% |
  | `/town:Town01/camera:1/frame-0000` | 24,716 | 1.04 cm | 13.5 cm | 96.0% |
  | `/town:Town01/camera:1/frame-4000` | 25,872 | 1.64 cm | 42.3 cm | 91.2% |

  The large errors and tag mismatches sit at object edges, where a point
  rounds onto the neighbouring surface.
- **Timing.** Sweep `i` covers `[0.1 i, 0.1 i + 0.1)` s and camera frame `i` is
  taken at its start (`full_ts_camera.txt`: 0.1 s steps). Images and sweeps of
  the same frame number belong together. Points are not ego-motion
  compensated; the PLY's `timestamp` column says when each was measured.
- **Poses.** `full_poses_lidar.txt` holds lidar→world at 1 kHz from the first
  step, which is the world origin. euler-loading's `lidar_pose` selects each
  sample's pose by the frame number in its file names. The prepared names are
  the native ones, so that works.

## Limits

- No official splits exist; none are written.
- The bundle `KITTI-CARLA.zip` is not read directly; unzip it once. Reading
  the town archives without extracting them is on the roadmap.
- `full_ts_camera.txt` and `poses_lidar.ply` are not prepared. The first
  restates the 10 Hz timing; the second duplicates the poses.
