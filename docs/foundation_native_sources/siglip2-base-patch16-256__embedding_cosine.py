#!/usr/bin/env python3
"""Generate the frozen SigLIP2 embedding fixture from authenticated upstream bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = "google/siglip2-base-patch16-256"
REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"
RESOLUTION = 256
EXPECTED_FILES = {
    "config.json": "7b5aedcb8893e31376e129c1ffd7a5392f1a806dbc793ce53eda220c2ec59edf",
    "preprocessor_config.json": "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff",
    "model.safetensors": "6125cacc01fa93bdc98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for name, expected in EXPECTED_FILES.items():
        observed = _sha256(args.snapshot / name)
        if observed != expected:
            raise ValueError(f"{MODEL_ID} {name} digest differs: {observed}")

    processor = AutoImageProcessor.from_pretrained(
        str(args.snapshot), revision=REVISION, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(args.snapshot), revision=REVISION, local_files_only=True, torch_dtype=torch.float32
    )
    model.eval()
    with Image.open(args.image) as opened:
        image = opened.convert("RGB")
    values = processor(
        images=[image],
        return_tensors="pt",
        size={"height": RESOLUTION, "width": RESOLUTION},
    )
    with torch.no_grad():
        result = model.get_image_features(**values)
        output = result if torch.is_tensor(result) else result.pooler_output
        output = torch.nn.functional.normalize(output, p=2, dim=-1)
    if output.dtype != torch.float32 or tuple(output.shape) != (1, 768):
        raise ValueError("SigLIP2 fixture output differs from frozen dtype/shape")
    payload = {
        "schema_version": "foundation-embedding-fixture-v1",
        "image_paths": ["../../assets/sfora-logo.png"],
        "reference_embedding": output.cpu().tolist(),
    }
    args.output.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
