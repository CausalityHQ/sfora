#!/usr/bin/env python3
"""Generate the repository-only local comparator embedding fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sfora.data import materialize_image
from sfora.foundation_pareto import LocalCheckpointFoundationSpec, load_foundation_encoder

SPEC = LocalCheckpointFoundationSpec(
    arm="inshop-pa-bninception-seed2",
    checkpoint_path=Path(
        "/home/riomus/sfora-inshop-seed2/reports/checkpoints/inshop_corrected_pa_seed2.pt"
    ),
    pretrained_backbone_path=Path(
        "/home/riomus/.cache/torch/hub/checkpoints/bn_inception-52deb4733.pth"
    ),
    checkpoint_sha256="f11aaf526efa4ce690a01ee19c5587842c27f78ac47be6943221c2b9f20acf7f",
    resolved_config_sha256="e64cdb33c7bc694671c20693a4babdbed201c53afbabca42c2e2e7ac8b189880",
    pretrained_backbone_sha256=("52deb473314542a5c2f87e9e6f26f4ca42fe863d15f986414dbae8c2dfdd2353"),
    transform_id="proxy-anchor-eval-224-v1",
    embedding_width=512,
    pooling="embedding",
    dtype="float32",
    normalize=True,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    encoder = load_foundation_encoder(SPEC)
    output = encoder.encode(
        [materialize_image(args.image)], batch_size=1, normalize_embeddings=True
    )
    if output.dtype.name != "float32" or output.shape != (1, 512):
        raise ValueError("local comparator fixture output differs from frozen dtype/shape")
    payload = {
        "schema_version": "foundation-embedding-fixture-v1",
        "image_paths": ["../../assets/sfora-logo.png"],
        "reference_embedding": output.tolist(),
    }
    args.output.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
