# Adding a recipe

A recipe turns one dataset's downloads into an euler-ready directory. The
nuScenes recipe (`euler_known_datasets/datasets/nuscenes/`) is the reference.

## 1. Establish the facts first

Before writing code, write down what the recipe will rely on and check it on
the real download: archive names and their layout, how IDs and hierarchy can
be formed, where calibration lives and at what granularity, sensor timing,
coordinate conventions, official splits. Each fact goes into
`docs/datasets/<recipe>.md` together with how it was established. Recipes are
where these facts are written down once.

## 2. Write the recipe

```python
from pathlib import Path

from euler_known_datasets import Option, Outcome, Plan, PrepareContext, Recipe, register
from euler_known_datasets.artifacts import IndexTree, file_entry, write_scope, write_splits


@register
class MyDataset(Recipe):
    name = "my_dataset"
    title = "My Dataset"
    summary = "RGB and depth for ..."
    homepage = "https://example.org/my-dataset"
    data_license = "CC BY 4.0"
    expected_inputs = ("my_dataset_rgb.tar.gz", "my_dataset_depth.tar.gz")
    layout_version = 1
    options = (Option("subset", "Which subset to prepare.", default="all"),)

    def detect(self, source: Path) -> bool:
        return (source / "my_dataset_rgb.tar.gz").is_file()

    def plan(self, ctx: PrepareContext) -> Plan:
        archive = ctx.source / "my_dataset_rgb.tar.gz"
        if not archive.is_file():
            raise FileNotFoundError(f"{archive} is missing")
        return Plan(
            recipe=self.name,
            inputs=[{"kind": "archive", "path": str(archive)}],
            steps=["extract rgb", "write scope rgb"],
            details={"archive": archive},
        )

    def prepare(self, ctx: PrepareContext, plan: Plan) -> Outcome:
        ctx.extract(plan.details["archive"])  # tar archives; resumable and safe
        tree = IndexTree()
        for path in sorted((ctx.output / "rgb").rglob("*.png")):
            relpath = path.relative_to(ctx.output).as_posix()
            scene = path.parent.name
            tree.add((f"scene:{scene}",), file_entry(relpath, path.stem))
        write_scope(
            ctx.output,
            "rgb",
            dataset={"id": "my_dataset_rgb", "name": "My Dataset RGB"},
            modality={"key": "rgb", "meta": {"range": [0, 255]}},
            addons={"euler_loading": {"version": "1.0", "loader": "generic_dense_depth", "function": "rgb"}},
            tree=tree,
        )
        return Outcome(media="extract", scopes=["rgb"])
```

Import the module from `euler_known_datasets/datasets/__init__.py` (or expose it
through the `euler_known_datasets.recipes` entry point from another package).

Guidelines:

- **IDs and hierarchy.** Regular modalities that should be joined need equal
  hierarchy keys and IDs. Put whatever groups samples semantically (scene,
  sequence, camera) into the hierarchy, and let per-scene files (calibration)
  sit at the level they apply to.
- **Prebuilt indexes** let IDs encode joins that paths cannot, such as a lidar
  sweep indexed under the ID of the camera frame it pairs with.
- **Per-frame facts from tables** go into small, versioned, documented files,
  read by a hierarchical loader with `@selects_per_sample` in euler-loading.
  Include the dataset's own tokens so values trace back to the source.
- **Never stat files one by one**; use `euler_known_datasets._fs.DirectoryCache`
  or `media.missing_files` (see [network filesystems](concepts.md#network-filesystems)).
- **Heads** name an euler-loading loader that exists, or one added there in
  step. Keep experiment roles (`used_as`, `slot`) out of dataset heads.

## 3. Test it without the dataset

Unit tests build a small synthetic copy of the downloads, archives included,
that reproduces the dataset's peculiarities (`tests/fake_nuscenes.py`). Cover
every media strategy, re-runs, options, and loading the output through
euler-loading. Add a `real`-marked test that runs the recipe on the actual
download when an environment variable points at it.
