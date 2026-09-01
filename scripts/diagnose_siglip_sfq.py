#!/usr/bin/env python3
"""Run the local optimization-only SigLIP SFQ fold diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import cast

from sfora.siglip_head_screen import build_feature_split_authority
from sfora.siglip_sfq import (
    run_sfq_fold_diagnostic,
    sfq_label_vector_sha256,
    validate_sfq_result_bytes,
)
from sfora.token_set_screen import F1_TRAIN_CLASSES

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))

from diagnose_siglip_head_screen import (  # noqa: E402
    _MODEL_NAME,
    _MODEL_REVISION,
    CachedFeatureBand,
    _canonical_bytes,
    _load_band,
    _read_regular,
    _write_new,
)


def load_optimization_feature_cache(
    path: Path, *, expected_sha256: str
) -> tuple[str, str, CachedFeatureBand]:
    """Authenticate the cache authority while opening only optimization rows."""

    raw = _read_regular(path, role="feature manifest")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("feature manifest digest differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("feature manifest JSON differs") from error
    if (
        type(value) is not dict
        or raw != _canonical_bytes(value)
        or set(value)
        != {
            "schema",
            "claim_eligible",
            "source_manifest_sha256",
            "control_manifest_file",
            "source_commit",
            "model_name",
            "model_revision",
            "bands",
        }
    ):
        raise ValueError("feature manifest schema differs")
    source_digest = value["source_manifest_sha256"]
    source_commit = value["source_commit"]
    bands = value["bands"]
    if (
        value["schema"] != "sfora-siglip-head-feature-cache-v1"
        or value["claim_eligible"] is not False
        or type(source_digest) is not str
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
        or value["control_manifest_file"] != "control-manifest.json"
        or type(source_commit) is not str
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or value["model_name"] != _MODEL_NAME
        or value["model_revision"] != _MODEL_REVISION
        or type(bands) is not dict
        or set(bands) != {"optimization", "clean_validation", "burned_diagnostic"}
    ):
        raise ValueError("feature manifest authority differs")
    optimization = _load_band(
        path.parent,
        cast(dict[str, object], bands)["optimization"],
        expected_role="optimization-train",
        expected_classes=F1_TRAIN_CLASSES,
    )
    return source_digest, source_commit, optimization


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected one lowercase SHA-256 digest")
    return value


def _commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected one lowercase Git commit")
    return value


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise argparse.ArgumentTypeError("paths must be normalized absolute paths")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse an explicit local-only execution request."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--feature-manifest", type=_absolute_path, required=True)
    parser.add_argument("--feature-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--feature-source-commit", type=_commit, required=True)
    parser.add_argument("--result", type=_absolute_path, required=True)
    parser.add_argument("--execute-sfq-folds", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


def main(argv: list[str] | None = None) -> int:
    """Authenticate the cache and run four optimization-only folds."""

    args = parse_args(argv)
    source_manifest_sha256, source_commit, optimization = load_optimization_feature_cache(
        args.feature_manifest,
        expected_sha256=args.feature_manifest_sha256,
    )
    if source_commit != args.feature_source_commit:
        raise ValueError("feature cache source revision differs")
    split_authority = build_feature_split_authority(
        source_manifest_sha256=source_manifest_sha256,
        role=optimization.role,
        official_test_access=False,
        ordered_example_ids=optimization.example_ids,
        features=optimization.features,
    )
    raw = run_sfq_fold_diagnostic(
        optimization.features,
        optimization.labels,
        split_authority=split_authority,
        feature_cache_manifest_sha256=args.feature_manifest_sha256,
        output_dimensions=min(512, *optimization.features.shape),
        fold_count=4,
    )
    validate_sfq_result_bytes(
        raw,
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_feature_cache_manifest_sha256=args.feature_manifest_sha256,
        expected_ordered_example_ids_sha256=split_authority.ordered_example_ids_sha256,
        expected_feature_matrix_sha256=split_authority.feature_matrix_sha256,
        expected_label_vector_sha256=sfq_label_vector_sha256(optimization.labels),
    )
    _write_new(args.result, raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
