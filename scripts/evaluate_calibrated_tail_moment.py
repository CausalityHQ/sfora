#!/usr/bin/env python3
"""Evaluate calibrated tail moment on one strict UNICOM embedding bundle."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sfora.calibrated_tail_moment_io import evaluate_bundle, publish_ctm_report
from sfora.unicom_audit_io import load_embedding_bundle

REGISTERED_WIDTHS = (64, 128, 256, 512)
REGISTERED_NEIGHBORS = 50
REGISTERED_NULL_SEEDS = range(206, 238)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(arguments)


def run(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    owned_temporaries = tuple(args.output.parent.glob(f".{args.output.name}.*.tmp"))
    if args.output.exists() or args.output.is_symlink() or owned_temporaries:
        print(f"evaluation failed: output exists: {args.output}", file=sys.stderr)
        return 2
    try:
        report = evaluate_bundle(
            load_embedding_bundle(args.bundle),
            widths=REGISTERED_WIDTHS,
            neighbors=REGISTERED_NEIGHBORS,
            null_seeds=REGISTERED_NULL_SEEDS,
        )
        publish_ctm_report(
            args.output,
            report,
            expected_widths=REGISTERED_WIDTHS,
            expected_neighbors=REGISTERED_NEIGHBORS,
            expected_null_count=len(REGISTERED_NULL_SEEDS),
        )
    except Exception as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 2
    print(f"evaluation complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
