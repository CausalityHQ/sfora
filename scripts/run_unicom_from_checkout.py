#!/usr/bin/env python3
"""Run a registered UniCOM experiment script against this checkout's sources."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REGISTERED_SCRIPTS = (
    "build_unicom_full_width_pair_config.py",
    "compare_unicom_full_width_profiles.py",
    "confirm_unicom_full_width_objective.py",
    "decide_unicom_full_width_objective.py",
    "evaluate_unicom_full_width_objective.py",
    "profile_unicom_training_step.py",
    "train_unicom_inshop.py",
)


def authenticate_paths(requested_target: Path) -> tuple[Path, Path]:
    """Return the physical checkout source and a registered physical target."""

    scripts = Path(__file__).resolve().parent
    root = scripts.parent
    src = root / "src"
    if (
        not src.is_dir()
        or src.is_symlink()
        or src.resolve().parent != root.resolve()
    ):
        raise ValueError("checkout source directory differs")
    if not isinstance(requested_target, Path):
        raise TypeError("target must be a pathlib.Path")
    if requested_target.is_symlink():
        raise ValueError("target is not a registered experiment script")
    target = requested_target.resolve()
    registered = tuple((scripts / name).resolve() for name in REGISTERED_SCRIPTS)
    if (
        target not in registered
        or not target.is_file()
        or target.is_symlink()
        or target.parent != scripts
    ):
        raise ValueError("target is not a registered experiment script")
    return src.resolve(), target


def prepare_checkout_imports(requested_target: Path) -> tuple[Path, Path]:
    """Authenticate paths and make this checkout win module resolution."""

    src, target = authenticate_paths(requested_target)
    if any(name == "sfora" or name.startswith("sfora.") for name in sys.modules):
        raise ValueError("sfora was imported before checkout authentication")
    sys.path.insert(0, str(src))
    return src, target


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    try:
        if not arguments:
            raise ValueError("registered experiment script is required")
        src, target = prepare_checkout_imports(Path(arguments[0]))
    except Exception as error:
        print(f"structural failure: {error}", file=sys.stderr)
        return 2
    original_argv = sys.argv
    try:
        sys.argv = [str(target), *arguments[1:]]
        runpy.run_path(str(target), run_name="__main__")
        return 0
    finally:
        sys.argv = original_argv
        if sys.path and sys.path[0] == str(src):
            del sys.path[0]


if __name__ == "__main__":
    raise SystemExit(main())
