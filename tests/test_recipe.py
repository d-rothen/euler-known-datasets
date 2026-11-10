from __future__ import annotations

import pytest

from euler_known_datasets import Option, detect_recipes, get_recipe, list_recipes

from .fake_nuscenes import build_fake_nuscenes


def test_multiple_option_accepts_lists_and_comma_strings():
    option = Option("cameras", "", default=(), multiple=True, choices=("A", "B"))
    assert option.coerce("A,B") == ("A", "B")
    assert option.coerce(["A", "B"]) == ("A", "B")
    assert option.coerce(["A,B"]) == ("A", "B")
    assert option.coerce(None) == ()


def test_option_rejects_unknown_choice_and_duplicates():
    option = Option("cameras", "", default=(), multiple=True, choices=("A", "B"))
    with pytest.raises(ValueError, match="not one of"):
        option.coerce(["C"])
    with pytest.raises(ValueError, match="duplicate"):
        option.coerce(["A", "A"])


@pytest.mark.parametrize("raw, expected", [("yes", True), ("0", False), (True, True)])
def test_bool_option(raw, expected):
    assert Option("flag", "", default=False, type=bool).coerce(raw) is expected


def test_float_option_converts_strings():
    assert Option("tol", "", default=1.0, type=float).coerce("2.5") == 2.5


def test_registry_knows_nuscenes():
    assert "nuscenes" in [recipe.name for recipe in list_recipes()]
    assert get_recipe("nuscenes").title == "nuScenes"


def test_unknown_recipe():
    with pytest.raises(KeyError, match="unknown dataset recipe"):
        get_recipe("no-such-dataset")


def test_unknown_option_is_rejected():
    with pytest.raises(TypeError, match="unknown option"):
        get_recipe("nuscenes").resolve_options({"camera": "CAM_FRONT"})


def test_detect(tmp_path):
    assert detect_recipes(tmp_path) == []
    build_fake_nuscenes(tmp_path)
    assert [recipe.name for recipe in detect_recipes(tmp_path)] == ["nuscenes"]
