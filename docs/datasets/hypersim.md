# Hypersim — recipe `hypersim`

Hypersim (M. Roberts et al., "Hypersim: A Photorealistic Synthetic Dataset for
Holistic Indoor Scene Understanding", ICCV 2021;
[repository](https://github.com/apple/ml-hypersim)) holds 74,619 publicly
released renders of 457 indoor scenes. They are 1024×768, rendered with V-Ray
along 769 camera trajectories. The recipe prepares:
- the HDR colour, tone-mapped by the loader with Hypersim's own operator;
- planar depth, converted from the released distances, and those distances
  themselves (`radial_distance`, for fog synthesis);
- exact per-scene intrinsics;
- metric camera poses.

Decoding is done by euler-loading's `hypersim` loader module. The dataset is
licensed CC BY-SA 3.0.

- [Downloads](#downloads)
- [Usage](#usage)
- [Options](#options)
- [The prepared dataset](#the-prepared-dataset)
- [What the recipe knows about Hypersim](#what-the-recipe-knows-about-hypersim)
- [Reading the output](#reading-the-output)
- [Limits](#limits)

## Downloads

Hypersim is published as one zip per scene, `ai_VVV_NNN.zip`: 457 zips, 1.9 TB
in all. Their members are stored uncompressed, which makes a partial download
practical. The recipe needs these files:

```
ai_VVV_NNN/_detail/metadata_scene.csv                                  meters_per_asset_unit
ai_VVV_NNN/_detail/cam_XX/camera_keyframe_{orientations,positions}.hdf5 camera trajectory
ai_VVV_NNN/images/scene_cam_XX_final_hdf5/frame.IIII.color.hdf5         HDR colour, float16 (3.2 MB)
ai_VVV_NNN/images/scene_cam_XX_geometry_hdf5/frame.IIII.render_entity_id.hdf5   (25 kB, for the tone mapping)
ai_VVV_NNN/images/scene_cam_XX_geometry_hdf5/frame.IIII.depth_meters.hdf5       (0.2 MB)
```

Two more files come from the ml-hypersim repository, not from the zips:
`evermotion_dataset/analysis/metadata_images_split_scene_v1.csv` (the
official split) and `contrib/mikeroberts3000/metadata_camera_parameters.csv`
(the per-scene camera matrices). The recipe looks for them in this order:
- at the source root;
- in `metadata/`;
- at their repository paths below `ml-hypersim/`.

The source may hold the scene zips, the extracted scene folders, or a partial
download of them. The scene folders may sit directly in the source or one
folder below it.

The euler cluster copy was fetched with `slurm/tools/download_hypersim.py`
(`slurm/download-hypersim.sbatch`). The tool reads each zip's central
directory over HTTP range requests and fetches only the selected members, one
CRC-checked request each. The copy holds every public frame whose number is a
multiple of 4: 18,660 frames, 60,739 files, 71.2 GB. Fetching it took
51 minutes at 23–58 MB/s (2026-11-13).

## Usage

```bash
euler-known-datasets prepare hypersim /data/downloads/hypersim /data/hypersim --media hardlink
euler-known-datasets verify /data/hypersim
```

```python
from euler_known_datasets import prepare

prepare("hypersim", "/data/downloads/hypersim", "/data/hypersim_val", splits=["val"], frame_step=10)
```

Preparing needs `h5py` (`pip install 'euler-known-datasets[hypersim]'`). It
reads every colour and render-entity image once to compute its tone-mapping
scale, in parallel worker processes.

On the cluster copy (every 4th public frame), a hard-linked prepare with 16
processes took 17 minutes. It gives 18,572 frames (train 14,808, val 1,844,
test 1,920) on 766 trajectories, after leaving out 38 frames rendered with
another camera and 50 dark frames (2026-11-13).

## Options

| Option | Default | Meaning |
|---|---|---|
| `modalities` | all: `rgb`, `depth`, `radial_distance`, `intrinsics`, `camera_pose` | scopes to write |
| `scenes` | all | names or patterns, e.g. `ai_00*` |
| `splits` | all | `train`, `val`, `test` |
| `frame_step` | 1 | keep frames whose number is a multiple of it |
| `workers` | 0 (all CPUs, at most 16) | processes computing the tone-mapping scales |
| `keep_dark_frames` | false | keep the black renders (tone-mapping scale 0) |

## The prepared dataset

```
/data/hypersim/
├── euler-known-datasets.json
├── ai_VVV_NNN/images/scene_cam_XX_{final,geometry}_hdf5/…   (as released)
├── calibration/<scene>/<camera>.json
└── .ds_crawler/{rgb,depth,radial_distance,intrinsics,camera_pose}/
```

**Sample IDs and joins.**
- Each sample's full ID is `/scene:<scene>/camera:<trajectory>/<frame>`, e.g.
  `/scene:ai_037_002/camera:cam_00/0040`.
- The image scopes join by ID.
- `intrinsics` and `camera_pose` are indexed once per trajectory, as
  hierarchical modalities.

**Splits.** The official v1 split (`train`, `val` and `test`, by scene) is
written for every image scope. Only frames in the public release are
prepared.

| Scope | Modality key | euler-loading function | Files |
|---|---|---|---|
| `rgb` | `rgb` | `hypersim.rgb` | `…_final_hdf5/frame.IIII.color.hdf5` |
| `depth` | `depth` | `hypersim.depth` | `…_geometry_hdf5/frame.IIII.depth_meters.hdf5` |
| `radial_distance` | `depth` | `hypersim.radial_distance` | the same files |
| `intrinsics` | `intrinsics` | `hypersim.read_intrinsics` | `calibration/<scene>/<camera>.json` |
| `camera_pose` | `camera_extrinsics` | `hypersim.camera_pose` | the same file, selected per sample by frame |

**Per-file attributes.**
- Each `rgb` entry carries its `tonemap_scale`.
- Each `depth` and `radial_distance` entry carries the `intrinsics` matrix that
  converts the released distances to planar depth.

**Head meta.**
- `depth` declares `radial_depth: false`. This is what the loader returns, so
  euler-eval treats the values as planar depth.
- `radial_distance` declares `radial_depth: true`: the released distances,
  as fog synthesis needs them (the distance along each ray).
- Both declare `range: [0, 2000]` and that `0` means invalid. Most
  distances are a few meters (median 5 m), but geometry seen through windows
  reaches 1900 m.

**Calibration file.** Format `euler_known_datasets.hypersim.camera_frames`,
version 1. It holds:
- `camera.intrinsics` (3×3) and `camera.width` / `camera.height`;
- `camera.optical_tilt_deg`;
- `camera.stored_camera_from_optical`: the rotation from the image's optical
  frame into Hypersim's stored camera axes;
- `camera.camera_from_uv`: the published matrix;
- `meters_per_asset_unit`;
- `frames["IIII"].camera_to_world`: a 4×4 pose in meters for every frame of
  the trajectory.

Matrices are row-major. The optical frame is x right, y down, z forward.

## What the recipe knows about Hypersim

- **`depth_meters` holds distances, not depth.** The files hold the
  Euclidean distance to the camera centre in meters, as the README says.
  Planar depth is `distance / ||K⁻¹ [x, y, 1]||`, with pixel centres at
  integer coordinates. Pixels without geometry are NaN in the files and 0
  after loading.
- **Many scenes use tilt-shift lenses.** The camera matrices of 272 of the 482
  scenes in the camera CSV are not plain pinhole matrices of the stored camera
  axes:
  - 99 are tilted by more than 5°, up to 41.5° (ai_026_006);
  - their principal points range over cx 304–775 and cy 120–971 px;
  - 31 have skew of up to 16 px.

  The common shortcut (f = 886.81 with a centred principal point) misplaces
  points by 0.57 m (median) in ai_037_002. The recipe instead decomposes
  `S · M_cam_from_uv · A` into a rotation and an upper-triangular `K`, where
  `A` maps pixels to the [-1, 1] uv range and `S = diag(1, -1, -1)` flips y
  and z (see `datasets/hypersim/camera.py`). `K` is exact, and poses and depth
  refer to the image's own optical frame.

  Checked against Hypersim's world-space `position` images, the points agree
  to about 1 mm in ai_001_001 and ai_037_002, and to 6 mm in ai_026_006,
  which is within float16 precision at 10–30 m (2026-11-13).
- **Frame 0 of 67 trajectories has an odd orientation matrix.** Hypersim
  renders with the stored matrix as it is (its rays match the `position`
  images only that way), and normalises the rays. Every other frame stores a
  rotation. The odd frame-0 matrices fall into two groups:
  - **Uniformly scaled (29 trajectories).** For example, ai_026_006 carries a
    scale of 0.25. This changes nothing about the image, so the recipe uses
    the nearest rotation.
  - **Stretched or mirrored (38 trajectories).** For example, one axis of
    ai_015_003 cam_00 is scaled by 0.54, and ai_008_003 cam_01 is mirrored.
    These frames were rendered by another camera than their scene's: the
    nearest rotation is up to 14.5° off. They have no pose and no depth
    conversion in the scene's pinhole model, so the recipe leaves them out of
    every scope.

  The receipt lists both groups (checked on the 769 trajectories, 2026-11-13).
- **Tone mapping.** The released colour is linear HDR. Hypersim's operator
  (`scene_generate_images_tonemap.py`) scales the image so that the 90th
  percentile of the brightness `0.3R + 0.59G + 0.11B` becomes 0.8 after gamma
  1/2.2. The percentile is taken over the pixels with geometry
  (`render_entity_id != -1`). The recipe computes that scale per frame, and
  the loader applies it. The JPEG previews in the zips (about 50 kB each) are
  lossy, and Hypersim advises against training on them.
- **The split.** `metadata_images_split_scene_v1.csv` lists 82,900 images,
  74,619 of them public: 59,543 train, 7,386 val and 7,690 test.

## Reading the output

```python
from euler_loading import Modality, MultiModalDataset

root = "/data/hypersim"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb", split="train"),
        "depth": Modality(root, metadata_scope="depth", split="train"),
    },
    hierarchical_modalities={
        "intrinsics": Modality(root, metadata_scope="intrinsics"),
        "pose": Modality(root, metadata_scope="camera_pose"),
    },
)
sample = dataset[0]  # rgb (3, 768, 1024), depth (1, 768, 1024) in meters
```

Automatic resolution needs an euler-loading release with the `hypersim`
module and `h5py` (`pip install 'euler-loading[hypersim]'`).

## Limits

- **Only colour, depth and the camera.** Normals, segmentation and the
  diffuse/residual decomposition are in the zips but not prepared.
- **Some frames are left out.** The recipe skips:
  - frames outside the public release;
  - the frames of the 25 scenes the camera CSV has but the split does not
    release;
  - the 38 frames rendered with a stretched or mirrored camera;
  - frames too dark to tone-map: the renders of ai_003_001 and ai_004_009 are
    black (peak radiance about 1e-6). Hypersim's operator gives them scale 0,
    so they would load as black images. Option `keep_dark_frames` keeps them.

  The last check needs the trajectory's orientation file. A partial download
  without camera files cannot be checked.
- **Pixels without geometry stay 0.** These are views through windows onto
  nothing (`render_entity_id == -1`): 1.2 % of the pixels on average, more
  than half of the image in a few frames. Fog synthesis treats depth 0 as
  "no fog".
