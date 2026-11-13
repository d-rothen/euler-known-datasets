# UrbanSyn — recipe `urbansyn`

UrbanSyn (J. L. Gómez, M. Silva, A. Seoane, A. Borrás, M. Noriega, G. Ros,
J. A. Iglesias-Guitian, A. M. López, "All for one, and one for all: UrbanSyn
Dataset, the third Musketeer of synthetic driving scenes", Neurocomputing 637,
2025; [website](https://www.urbansyn.org),
[preprint](https://arxiv.org/abs/2312.12176)) has 7539 photorealistic frames of
three procedurally built city districts, path-traced with OctaneRender at
2048×1024. The recipe prepares the images, the depth maps, the semantic labels,
the camera's intrinsics and the per-frame camera poses, which UrbanSyn does not
document but records in every depth map's header. Decoding is done by
euler-loading's `urbansyn` loader module.

- [Downloads](#downloads)
- [Usage](#usage)
- [Options](#options)
- [The prepared dataset](#the-prepared-dataset)
- [What the recipe knows about UrbanSyn](#what-the-recipe-knows-about-urbansyn)
- [Reading the output](#reading-the-output)
- [Limits](#limits)

## Downloads

The website links every archive directly, with no registration, from two
mirrors (`pegaso.citic.udc.es` and `datasets.cvc.uab.es`); a copy is also on
[Hugging Face](https://huggingface.co/datasets/UrbanSyn/UrbanSyn). The data is
licensed CC BY-SA 4.0 under UrbanSyn's terms of use, which forbid using
generative AI to reverse-engineer its assets or to make stock-media content
from it. Each zip holds one flat folder:

| Download | Size | Content | Used |
|---|---|---|---|
| `rgb.zip` | 21.5 GB | `rgb/rgb_<NNNN>.png`, 8-bit sRGB (and a stray `rgb/Thumbs.db`) | yes |
| `depth.zip` | 51.7 GB (63.4 GB unpacked) | `depth/depth_<NNNN>.exr`, float32, one channel `Y`, uncompressed | yes |
| `ss.zip` | 0.40 GB | `ss/ss_<NNNN>.png`, Cityscapes train IDs as RGB with R = G = B | yes |
| `camera_metadata.json` | 524 B | focal length, field of view and sensor size of the camera | yes |
| `ss_color.zip` | 0.41 GB | the labels in Cityscapes colours | no |
| `panoptic.zip` | 0.11 GB | instance masks of the dynamic classes (RG instance, B class) | no |
| `bbox2d.zip` | 6 MB | 2D boxes of dynamic objects, JSON | no |

Frames are numbered `0001` to `7539` in every folder. `ss.zip` and
`ss_color.zip` were replaced on 2026-10-19 to fix the labels of frames
6973–7029; the other zips date from 2023-12-01.

The source may hold the zips, their extracted folders, or both, directly or
one folder down. An extracted folder is used only when it holds every frame
of its zip. Only the indexed files are extracted.

## Usage

```bash
euler-known-datasets prepare urbansyn /data/downloads/URBANSYN /data/urbansyn
euler-known-datasets verify /data/urbansyn
```

```python
from euler_known_datasets import prepare

prepare("urbansyn", "/data/downloads/URBANSYN", "/data/urbansyn_berlin",
        modalities=["rgb", "depth", "intrinsics"], scenes=["torstrasse", "veteranenstrasse"])
```

On Lustre, preparing everything from the zips extracts 85 GB.

## Options

| Option | Default | Meaning |
|---|---|---|
| `modalities` | every scope found | `rgb`, `depth`, `semantic_segmentation`, `intrinsics`, `camera_pose`. Only their folders are extracted. |
| `scenes` | all | `poblenou`, `torstrasse`, `veteranenstrasse` |
| `keep_incomplete_depth` | `false` | Also index the 57 depth maps that lack the static scene (see below), marked with the file attribute `incomplete_depth: true`. |

## The prepared dataset

```
/data/urbansyn/
├── euler-known-datasets.json
├── rgb/rgb_0001.png …            (each folder as released)
├── depth/depth_0001.exr …
├── ss/ss_0001.png …
├── camera_metadata.json
└── .ds_crawler/{rgb,depth,semantic_segmentation,intrinsics,camera_pose}/
```

**Sample IDs.** Each sample's full ID is `/scene:<scene>/<NNNN>`, e.g.
`/scene:torstrasse/3000`, with the release's frame number as the file ID.

- **Scopes join by ID.** Every frame scope has the same IDs. `depth` lacks
  frames 6973–7029, so joining it drops them.
- **City.** Every file carries the attribute `city` (`Barcelona` or `Berlin`).
- **The intrinsics are shared.** The `intrinsics` scope indexes
  `camera_metadata.json` at the root, so it applies to every sample.
- **The poses come from the depth maps.** `camera_pose` indexes the depth
  EXRs, whose headers hold the camera, so it needs `depth/` extracted. It
  covers all frames, including 6973–7029.
- **Splits.** There are none: the release defines no split. The hierarchy
  selects scenes.

| Scope | Modality key | euler-loading function | Files |
|---|---|---|---|
| `rgb` | `rgb` | `urbansyn.rgb` | `rgb/rgb_<NNNN>.png` |
| `depth` | `depth` | `urbansyn.depth` | `depth/depth_<NNNN>.exr` |
| `semantic_segmentation` | `semantic_segmentation` | `urbansyn.semantic_segmentation` | `ss/ss_<NNNN>.png` |
| `intrinsics` | `intrinsics` | `urbansyn.read_intrinsics` | `camera_metadata.json` |
| `camera_pose` | `camera_extrinsics` | `urbansyn.camera_pose` | `depth/depth_<NNNN>.exr` (header) |

**Head meta.**
- The depth head declares `radial_depth: false`, `scale_to_meters: 100000`
  (stored value × 1e5 = meters), `range: [0, 1500]` in meters and
  `invalid.sentinel: 0`.
- The label head declares `skyclass: [10, 10, 10]`, `label_ids:
  cityscapes_train_ids`, the 19 class names, and `other_id: 19`.
- The intrinsics head declares `camera_model: pinhole` and `image_size:
  [1024, 2048]`.
- The pose head declares `source_frame: camera_optical`, `target_frame:
  world`, `direction: target_from_source` and `translation_unit: meters`.
- Every image head declares `dimensions`.

## What the recipe knows about UrbanSyn

The facts below were checked on all 7539 frames on 2026-11-13, by reading
every depth map, label map and EXR header.

### Scenes

- **Three districts.** Frames 0001–2688 show Poblenou (Barcelona), 2689–5593
  Torstraße and 5594–7539 Veteranenstraße (Berlin). Each depth EXR names its
  district's render project in its header (`urbansyn_poblenou_orbx`,
  `urbansyn_torstrasse_orbx`, `urbansyn_veteranentrasse_orbx`).
- **The paper's four ODDs follow the same numbering.** Their sizes (1319,
  1369, 2905 and 1946 images) add up to the district boundaries; the second
  and the fourth use participating media. The recipe records districts only:
  nothing in the files marks the boundary between the first two ODDs.
- **The scene changes between frames.** Buildings are generated
  procedurally and the people and vehicles are placed per frame, so two
  frames of a district at the same place need not show the same objects.
- **Frames often come in pairs** a few meters apart, one camera looking
  forward and one backward.

### Depth

- **Planar (z) depth.** The depth maps are OctaneRender's Z-depth pass, the
  distance "measured parallel to the view vector" (Octane documentation).
- **Units of 1e5 m.** The site states "the measurement units are 1e5 meters"
  (corrected on 2024-07-23 from "10e-5 meters"): 10 m is stored as `1e-4`.
- **The convention and the units check out.** Back-projected with the
  intrinsics below and moved with the poses below, a frame's road, sidewalk,
  building and vegetation pixels land at the depth another frame measures
  there. For four frame pairs that see the same street the median error is
  0.01–0.15 %, with 82–98 % of the points within 1 %; reading the depth as
  distance along the ray gives 1.6–8 %, and a mirrored camera 2.5–19 %.
  Pairs whose street was regenerated in between agree less (0.8–1.7 %).
- **A ray that hits nothing is stored as `1.0`** (100 km), the pass's
  maximum. The loader returns it as `0`. No depth map holds NaN, zero or
  values above 1.
- **57 depth maps lack the static scene.** In frames 6973–7029, 29–54 % of
  the non-sky pixels hit nothing: only people, vehicles and poles were
  rendered. These are the frames whose labels UrbanSyn fixed on 2026-10-19;
  the depth was not re-released. The recipe leaves them out of `depth` unless
  `keep_incomplete_depth` is set. In every other frame at most 0.08 % of the
  non-sky pixels miss.
- **The sky is a finite backdrop**, 360–1180 m away (5th–95th percentile over
  the frames). Mask it with the labels (`sky_mask`) before using depth there.
  Valid depth reaches 1360 m and starts at the 0.3 m near clip.

### Labels

- **Cityscapes train IDs**: `0` road … `10` sky … `18` bicycle, and `19` for
  everything else (benches, planters and other street furniture), the role
  Cityscapes gives `255`. All 20 values occur.
- **Stored as RGB PNGs with three equal channels** (in every file), so
  `skyclass` is `[10, 10, 10]`.

### Camera

- **Intrinsics.** `camera_metadata.json` gives a 4.9 mm lens on a 5.8 mm ×
  2.9 mm sensor (horizontal FOV 61.24°). At 2048×1024 that is `fx = fy =
  1730.21` px. The renderer's camera has no lens shift, so the principal point
  is the image centre, `(1023.5, 511.5)` with pixel centres at integer
  coordinates. Every EXR header records the same camera (thin lens, no
  distortion, square pixels, 2048×1024).
- **Poses.** The EXR header attribute `octane` holds the render settings as
  JSON, among them the camera's `pos`, `target` and `up` in the district's
  world frame (y up, meters). `camera_pose` turns them into a camera-to-world
  matrix in optical axes: z towards the target, y against up, x = y × z. The
  camera sits 0.78–1.69 m up the world's y axis.

## Reading the output

```python
from euler_loading import Modality, MultiModalDataset

root = "/data/urbansyn"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb"),
        "depth": Modality(root, metadata_scope="depth"),
        "labels": Modality(root, metadata_scope="semantic_segmentation"),
        "sky": Modality(root, metadata_scope="semantic_segmentation", function="sky_mask"),
        "pose": Modality(root, metadata_scope="camera_pose"),
    },
    hierarchical_modalities={
        "intrinsics": Modality(root, metadata_scope="intrinsics", collapse_single=True),
    },
)
sample = dataset[0]  # full_id "/scene:poblenou/0001"
sample["rgb"].shape, sample["depth"].shape, sample["pose"].shape  # (3, 1024, 2048), (1, 1024, 2048), (4, 4)
```

Automatic resolution needs an euler-loading release newer than 2.43.0, the
first with the `urbansyn` module, and depth and poses need OpenEXR
(`pip install 'euler-loading[urbansyn]'`). Without torch, pass `loader=` from
`euler_loading.loaders.cpu.urbansyn`, together with a `writer=`.

## Limits

- **Not prepared:** the colour-coded labels, the panoptic masks and the 2D
  boxes. They are separate downloads.
- **No splits.** Choose your own; the districts make natural held-out sets.
- **Each district has its own world frame**, and the scene content varies
  between frames, so poses relate frames of one district only.
