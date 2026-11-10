# Contributing

Thanks for improving euler-known-datasets. New recipes are the most welcome
contribution; please read [Adding a recipe](docs/adding-a-recipe.md) first.

## Development setup

```bash
git clone https://github.com/d-rothen/euler-known-datasets.git
cd euler-known-datasets
uv sync --extra dev
```

## Before opening a pull request

```bash
uv run ruff check .
uv run pytest
uv build
```

`pytest` runs against small synthetic datasets that the tests build, archives
included, so it needs no downloads. Tests marked `real` run a recipe on the
actual dataset and are deselected by default:

```bash
NUSCENES_ROOT=/data/downloads/nuscenes uv run pytest -m real
```

Please keep the unit suite runnable without any dataset on disk.

## Rules for recipes

- **Never redistribute dataset content.** Recipes read what the user
  downloaded. Small facts that are not in the downloads (such as official split
  lists) may be vendored with attribution in [NOTICE](NOTICE).
- **Document every dataset fact the recipe relies on** in
  `docs/datasets/<recipe>.md`, with how it was established.
- **Bump `layout_version`** whenever the prepared output changes shape, and
  note it in the changelog: prepared directories record the version they were
  made with.
- **Keep heads honest.** The `addons.euler_loading` loader a head names must
  exist in euler-loading (or be added there in step); `verify` reports it.
- Python 3.9 is the floor: `from __future__ import annotations`, no `match`,
  no runtime `X | Y` unions.

## Releasing

Bump `version` in `pyproject.toml`, move the changelog's Unreleased entries
under the new version, and tag `v<version>`.
