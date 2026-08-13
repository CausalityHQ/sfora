#!/usr/bin/env python3
"""Generate the repository-only identity-disjoint comparator embedding fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sfora.data import materialize_image
from sfora.foundation_pareto import LocalCheckpointFoundationSpec, load_foundation_encoder

SPEC = LocalCheckpointFoundationSpec(
    arm="inshop-pa-bninception-disjoint-seed2",
    checkpoint_path=Path(
        "/home/riomus/group-learning/reports/checkpoints/"
        "foundation_identity_disjoint_comparator_"
        "3267fc6e2a21c1d346a5fe98f4bbd42e292cafb9_seed2.pt"
    ),
    pretrained_backbone_path=Path(
        "/home/riomus/.cache/torch/hub/checkpoints/bn_inception-52deb4733.pth"
    ),
    checkpoint_sha256="2eb588846cde6846fbd1ca7f9894a60eb1491239f23e099dd2136bfe739fe08b",
    resolved_config_sha256="0e6b2030f661b61b62502de1cd02f3294a0b5f61d116156b34bd02260538a7b0",
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
