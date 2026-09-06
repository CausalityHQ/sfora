#!/usr/bin/env python3
"""Materialize the fixed calibration split and project its authenticated authority."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from probe_siglip_attention_readout_recovery import _canonical_json, _image_basename
from run_siglip_proxy_control import load_control_examples, write_control_manifest_artifacts
from run_siglip_rsta_stage_a import project_stage_a_authority

from sfora.siglip_coverage_calibration import coverage_support_indexes
from sfora.siglip_proxy_control import SiglipProxyControlConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the preparation-only local-file interface."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--optimization-image-root", type=Path, required=True)
    parser.add_argument("--support-image-root", type=Path, required=True)
    parser.add_argument("--heldout-image-root", type=Path, required=True)
    parser.add_argument("--execute-preparation", action="store_true")
    parsed = parser.parse_args(argv)
    if not parsed.execute_preparation:
        parser.error("explicit --execute-preparation is required")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Write the exact manifests and disjoint physical image partitions."""
    arguments = parse_args(argv)
    authority = arguments.authority
    authority.mkdir()
    control_manifest = authority / "control-manifest.json"
    evaluation_manifest = authority / "evaluation-manifest.json"
    bands = load_control_examples()
    config = SiglipProxyControlConfig()
    write_control_manifest_artifacts(
        output=control_manifest,
        optimization_image_root=arguments.optimization_image_root,
        bands=bands,
        png_compress_level=0,
    )
    arguments.support_image_root.mkdir()
    arguments.heldout_image_root.mkdir()
    candidate_ids = tuple(example.example_id for example in bands.clean_validation)
    candidate_labels = tuple(example.label for example in bands.clean_validation)
    support = set(coverage_support_indexes(candidate_ids, candidate_labels))
    for index, example in enumerate(bands.clean_validation):
        stream = io.BytesIO()
        example.image.save(stream, format="PNG", optimize=False, compress_level=0)
        root = arguments.support_image_root if index in support else arguments.heldout_image_root
        (root / _image_basename(example.example_id)).write_bytes(stream.getvalue())
    evaluation_manifest.write_bytes(
        _canonical_json(
            {
                "schema": "sfora-attention-readout-evaluation-v1",
                "claim_eligible": False,
                "dataset_id": config.dataset_name,
                "dataset_revision": config.dataset_revision,
                "examples": [
                    {"example_id": example.example_id, "label": example.label}
                    for example in bands.clean_validation
                ],
            }
        )
    )
    projected = project_stage_a_authority(
        seed_receipts=tuple(
            arguments.control / f"seed-{seed:03d}.receipt.json" for seed in (17, 29, 43)
        ),
        aggregate_receipt=arguments.control / "control.receipt.json",
        checkpoints=tuple(
            arguments.control / f"seed-{seed:03d}/checkpoints/seed-{seed:03d}-epoch-060.pt"
            for seed in (17, 29, 43)
        ),
        control_manifest=control_manifest,
    )
    (authority / "control-binding.json").write_bytes(projected.control_binding_bytes)
    (authority / "optimization-manifest.json").write_bytes(projected.optimization_manifest_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
