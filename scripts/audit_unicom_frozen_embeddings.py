#!/usr/bin/env python3
"""Run the frozen UNICOM geometry and class-shard audits."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sfora.unicom_audit_io import (
    build_audit_report,
    load_embedding_bundle,
    publish_json_no_clobber,
    validate_official_audit_report,
)
from sfora.unicom_retrieval_audit import audit_deployment_geometry
from sfora.unicom_shard_audit import audit_shard_sensitivity


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(arguments)


def run_geometry(bundle):
    return audit_deployment_geometry(
        bundle.query_embeddings,
        bundle.gallery_embeddings,
        bundle.query_labels,
        bundle.gallery_labels,
    )


def run_sharding(bundle):
    return audit_shard_sensitivity(bundle.train_embeddings, bundle.train_labels)


def run(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        bundle = load_embedding_bundle(args.bundle)
        geometry = run_geometry(bundle)
        sharding = run_sharding(bundle)
        report = build_audit_report(bundle, geometry, sharding)
        validate_official_audit_report(report)
        publish_json_no_clobber(args.output, report)
    except Exception as error:
        print(f"audit failed: {error}", file=sys.stderr)
        return 2
    if not geometry.reproduction_passed:
        print(f"audit reproduction failed: {args.output}", file=sys.stderr)
        return 1
    print(f"audit complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
