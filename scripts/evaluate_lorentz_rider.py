#!/usr/bin/env python3
"""Evaluate the registered Lorentz L0/L1 rider on one embedding bundle."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sfora.lorentz_rider_io import evaluate_l0_l1, publish_lorentz_report
from sfora.unicom_audit_io import load_embedding_bundle


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-index", required=True, type=int, choices=(0, 1, 2))
    return parser.parse_args(arguments)


def run(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if not args.bundle.is_absolute() or not args.output.is_absolute():
        print("evaluation failed: bundle and output paths must be absolute", file=sys.stderr)
        return 2
    if not args.output.parent.is_dir() or args.output.parent.is_symlink():
        print(
            f"evaluation failed: output parent is not a directory: {args.output.parent}",
            file=sys.stderr,
        )
        return 2
    owned_temporaries = tuple(args.output.parent.glob(f".{args.output.name}.*.tmp"))
    if args.bundle.is_symlink():
        print(f"evaluation failed: bundle is a symlink: {args.bundle}", file=sys.stderr)
        return 2
    if args.output.exists() or args.output.is_symlink() or owned_temporaries:
        print(
            f"evaluation failed: output exists or temporary residue remains: {args.output}",
            file=sys.stderr,
        )
        return 2
    try:
        bundle = load_embedding_bundle(args.bundle)
        report = evaluate_l0_l1(bundle, dataset_index=args.dataset_index)
        publish_lorentz_report(args.output, report)
    except Exception as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 2
    print(f"evaluation complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
