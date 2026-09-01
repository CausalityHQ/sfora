#!/usr/bin/env python3
"""Run the local optimization-only SigLIP CDGA fold diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

from sfora.siglip_cdga import run_cdga_fold_diagnostic, validate_cdga_result_bytes
from sfora.siglip_head_screen import build_feature_split_authority
from sfora.siglip_sfq import sfq_label_vector_sha256

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))

from diagnose_siglip_head_screen import _write_new  # noqa: E402
from diagnose_siglip_sfq import (  # noqa: E402
    _absolute_path,
    _commit,
    _sha256,
    load_optimization_feature_cache,
)

_MASTER_SEED_SHA256 = hashlib.sha256(b"sfora-siglip-cdga-v1").hexdigest()
_FOLD_COUNT = 4
_TRAIN_STEPS = 20
_EXAMPLES_PER_CLASS = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one explicit local-only execution request."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--feature-manifest", type=_absolute_path, required=True)
    parser.add_argument("--feature-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--feature-source-commit", type=_commit, required=True)
    parser.add_argument("--result", type=_absolute_path, required=True)
    parser.add_argument("--execute-cdga-folds", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


def main(argv: list[str] | None = None) -> int:
    """Authenticate one optimization band, run CDGA folds, and publish once."""

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
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    raw = run_cdga_fold_diagnostic(
        optimization.features,
        optimization.labels,
        split_authority=split_authority,
        feature_cache_manifest_sha256=args.feature_manifest_sha256,
        master_seed_sha256=_MASTER_SEED_SHA256,
        output_dimensions=min(512, *optimization.features.shape),
        fold_count=_FOLD_COUNT,
        train_steps=_TRAIN_STEPS,
        examples_per_class=_EXAMPLES_PER_CLASS,
        device="cpu",
    )
    result = validate_cdga_result_bytes(raw)
    expected = {
        "source_manifest_sha256": source_manifest_sha256,
        "feature_cache_manifest_sha256": args.feature_manifest_sha256,
        "ordered_example_ids_sha256": split_authority.ordered_example_ids_sha256,
        "feature_matrix_sha256": split_authority.feature_matrix_sha256,
        "label_vector_sha256": sfq_label_vector_sha256(optimization.labels),
        "master_seed_sha256": _MASTER_SEED_SHA256,
        "fold_count": _FOLD_COUNT,
        "train_steps": _TRAIN_STEPS,
        "examples_per_class": _EXAMPLES_PER_CLASS,
        "input_dimensions": optimization.features.shape[1],
        "output_dimensions": min(512, *optimization.features.shape),
    }
    if any(getattr(result, field) != value for field, value in expected.items()):
        raise ValueError("CDGA result binding differs")
    _write_new(args.result, raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
