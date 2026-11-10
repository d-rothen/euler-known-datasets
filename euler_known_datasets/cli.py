"""``euler-known-datasets`` command line.

    euler-known-datasets list
    euler-known-datasets info nuscenes
    euler-known-datasets detect /data/downloads/nuscenes
    euler-known-datasets prepare nuscenes SOURCE OUTPUT [--cameras ...] [--dry-run]
    euler-known-datasets verify OUTPUT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional, Sequence

from ._version import __version__
from .media import MEDIA_MODES
from .prepare import prepare
from .recipe import Option, Recipe, detect_recipes, get_recipe, list_recipes
from .verify import verify


def _add_option(parser: argparse.ArgumentParser, option: Option) -> None:
    default_text = option.default
    if option.multiple:
        default_text = ", ".join(option.default) if option.default else "all"
    help_text = f"{option.help} Default: {default_text}."
    if option.type is bool:
        parser.add_argument(
            option.flag,
            dest=option.name,
            action=argparse.BooleanOptionalAction,
            default=None,
            help=help_text,
        )
        return
    kwargs = {"dest": option.name, "default": None, "help": help_text}
    if option.multiple:
        kwargs["nargs"] = "+"
        kwargs["metavar"] = option.name.upper().rstrip("S") if option.choices else None
        if option.choices:
            help_text += f" Choices: {', '.join(map(str, option.choices))}."
            kwargs["help"] = help_text
    elif option.choices:
        kwargs["choices"] = option.choices
    parser.add_argument(option.flag, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="euler-known-datasets",
        description="Turn downloaded dataset archives into euler-ready datasets.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list the known datasets")

    info = commands.add_parser("info", help="describe a dataset recipe")
    info.add_argument("recipe")

    detect = commands.add_parser("detect", help="which known datasets a directory holds")
    detect.add_argument("source")

    prepare_parser = commands.add_parser(
        "prepare", help="prepare a dataset from its downloads"
    )
    recipes = prepare_parser.add_subparsers(dest="recipe", required=True, metavar="RECIPE")
    for recipe in list_recipes():
        sub = recipes.add_parser(recipe.name, help=recipe.summary, description=recipe.summary)
        sub.add_argument("source", help="directory with the downloads (archives or extracted)")
        sub.add_argument("output", help="directory to prepare; equal to SOURCE for in place")
        sub.add_argument(
            "--media",
            choices=MEDIA_MODES,
            default="auto",
            help="how media files reach the output (default: auto)",
        )
        sub.add_argument("--dry-run", action="store_true", help="only show the plan")
        sub.add_argument(
            "--force", action="store_true", help="allow a non-empty output we did not create"
        )
        for option in recipe.options:
            _add_option(sub, option)

    check = commands.add_parser("verify", help="check a prepared output")
    check.add_argument("output")
    check.add_argument("--no-files", action="store_true", help="skip file existence checks")
    check.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def _describe(recipe: Recipe) -> str:
    lines = [f"{recipe.name}: {recipe.title}", f"  {recipe.summary}"]
    if recipe.homepage:
        lines.append(f"  homepage      {recipe.homepage}")
    if recipe.data_license:
        lines.append(f"  data licence  {recipe.data_license}")
    lines.append(f"  layout        version {recipe.layout_version}")
    if recipe.expected_inputs:
        lines.append("  downloads")
        lines += [f"    {item}" for item in recipe.expected_inputs]
    if recipe.options:
        lines.append("  options")
        for option in recipe.options:
            lines.append(f"    {option.flag:<22} {option.help}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "list":
        for recipe in list_recipes():
            print(f"{recipe.name:<16} {recipe.title} - {recipe.summary}")
        return 0
    if args.command == "info":
        print(_describe(get_recipe(args.recipe)))
        return 0
    if args.command == "detect":
        found = detect_recipes(args.source)
        for recipe in found:
            print(recipe.name)
        return 0 if found else 1
    if args.command == "verify":
        report = verify(args.output, check_files=not args.no_files)
        print(json.dumps(report.to_dict(), indent=2) if args.json else report.format_text())
        return 0 if report.ok else 1

    recipe = get_recipe(args.recipe)
    options = {
        option.name: getattr(args, option.name)
        for option in recipe.options
        if getattr(args, option.name) is not None
    }
    try:
        result = prepare(
            recipe,
            args.source,
            args.output,
            media=args.media,
            dry_run=args.dry_run,
            force=args.force,
            **options,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if result.dry_run:
        print(f"{recipe.title}: {result.source} -> {result.output}")
        print("inputs")
        for item in result.plan.inputs:
            print("  " + ", ".join(f"{key}={value}" for key, value in item.items()))
        print("steps")
        for step in result.plan.steps:
            print(f"  - {step}")
        return 0
    print(f"prepared {recipe.title} in {result.output}")
    print(json.dumps(result.receipt.get("summary", {}), indent=2))
    return 0
