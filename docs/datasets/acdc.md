# ACDC — recipe `acdc`

The Adverse Conditions Dataset with Correspondences (C. Sakaridis, D. Dai,
L. Van Gool, ICCV 2021; [website](https://acdc.vision.ee.ethz.ch)) has 4006
images of drives in fog, at night, in rain and in snow. Each image comes with a
reference image of the same place taken in normal conditions. The recipe
prepares the anonymised images (`rgb_anon`), which ACDC distributes by default,
and their references. ACDC's semantic labels (`gt/`) are not prepared yet.
Decoding is done by euler-loading's `acdc` loader module.

- [Downloads](#downloads)
- [Usage](#usage)
- [Options](#options)
- [The prepared dataset](#the-prepared-dataset)
- [What the recipe knows about ACDC](#what-the-recipe-knows-about-acdc)
- [Reading the output](#reading-the-output)
- [Limits](#limits)

## Downloads

ACDC is downloaded from its website after registering, so the recipe cannot
fetch it. The images are in `rgb_anon_trainvaltest.zip`:

```
rgb_anon/<condition>/<split>/<sequence>/<sequence>_frame_<frame>_rgb_anon.png
rgb_anon/<condition>/<split>_ref/<sequence>/<sequence>_frame_<frame>_rgb_ref_anon.png
README.md, License.pdf
```

`<condition>` is `fog`, `night`, `rain` or `snow`, and `<split>` is `train`,
`val` or `test`. The source may hold that zip, an archive of a whole ACDC copy
that holds the same `rgb_anon/` tree (such as an `ACDC.tar.gz` with `gt/` and
`lists_file_names/` beside it), or the extracted tree, possibly one folder
down. Only `rgb_anon/` images and the documentation are extracted.

A tarball cannot be listed without decompressing it whole. The recipe
therefore extracts in a single pass with a path filter, then indexes the files
on disk. It recognises an archive from its first members.

## Usage

```bash
euler-known-datasets prepare acdc /data/downloads/ACDC /data/acdc
euler-known-datasets verify /data/acdc
```

```python
from euler_known_datasets import prepare

prepare("acdc", "/data/downloads/ACDC", "/data/acdc_fog", conditions=["fog"])
```

From a 17.3 GB tarball of a whole copy on Lustre, preparing everything takes
about 5 minutes and writes 16.8 GB.

## Options

| Option | Default | Meaning |
|---|---|---|
| `modalities` | both | `rgb` (adverse conditions), `reference_rgb` (their references) |
| `conditions` | all | `fog`, `night`, `rain`, `snow` |
| `splits` | all | `train`, `val`, `test` |

## The prepared dataset

```
/data/acdc/
├── euler-known-datasets.json
├── README.md, License.pdf
├── rgb_anon/<condition>/<split>[_ref]/<sequence>/…png   (as released)
└── .ds_crawler/{rgb,reference_rgb}/
```

**Sample IDs.** Each sample's full ID is
`/condition:<condition>/split:<split>/sequence:<sequence>/<frame>`, e.g.
`/condition:fog/split:train/sequence:GOPR0475/000041`.

- **Scopes join by ID.** A reference image has the same ID as its
  adverse-condition image.
- **Splits.** Each scope has the official splits `train`, `val` and `test`.
  Pass `split=` to select one.

| Scope | Modality key | euler-loading function | Files |
|---|---|---|---|
| `rgb` | `rgb` | `acdc.rgb` | `rgb_anon/<condition>/<split>/…_rgb_anon.png` |
| `reference_rgb` | `rgb` | `acdc.reference_rgb` | `rgb_anon/<condition>/<split>_ref/…_rgb_ref_anon.png` |

**Head meta and indexes.**
- Both heads declare `range: [0, 255]`, `anonymized: true` and `dimensions`.
  Every image is 1920×1080 RGB.
- The reference head adds `condition: normal` and `pixel_aligned: false`.
- The scopes keep their regexes in `ds-crawler.json`, so they can be
  re-crawled.
- The receipt counts the images per condition and split, and the adverse
  images paired with a reference.

## What the recipe knows about ACDC

- **Pairing by name.** Every adverse image has exactly one reference image in
  the matching `<split>_ref/` folder, with the same sequence and frame number.
  This was checked on all 4006 images (2026-11-10).
- **The release's lists are incomplete.** The `lists_file_names/` files a full
  copy ships list only part of the references (`val_ref` 203 of 406,
  `test_ref` 500 of 2000), so the recipe reads the files instead.
- **References are not pixel-aligned.** A reference image shows the place of
  its adverse image but was recorded on another drive.
- **Counts per condition.** `train` has 400 images; `val` has 100 (night: 106);
  `test` has 500. Test labels are withheld, which does not matter for images.

## Reading the output

```python
from euler_loading import Modality, MultiModalDataset

root = "/data/acdc"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb", split="train"),
        "reference": Modality(root, metadata_scope="reference_rgb", split="train"),
    },
)
sample = dataset[0]  # full_id "/condition:fog/split:train/sequence:GOPR0475/000041"
```

Automatic resolution needs an euler-loading release newer than 2.41.0, the
first with the `acdc` module. Without torch, pass `loader=` from
`euler_loading.loaders.cpu.acdc`, together with a `writer=`.

## Limits

- **Images only.** The semantic labels and invalid masks (`gt/`) are not
  prepared.
- **Anonymised images only.** The original images (`rgb/`) need a separate
  access request and are not recognised.
