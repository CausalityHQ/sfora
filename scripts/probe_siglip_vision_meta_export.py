"""CPU-only fixed-shape export falsifier; no weights, images, or latency claim."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoConfig, SiglipVisionModel
import transformers


class Pooler(torch.nn.Module):
    def __init__(self, vision: torch.nn.Module) -> None:
        super().__init__()
        self.vision = vision

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vision(pixel_values=pixels).pooler_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.config.read_bytes()).hexdigest()
    assert digest == "172e39dbf0143b8fe22d2f08921730eb8c397967e58cb37d133161e28aa34104"
    started = time.monotonic()
    config = AutoConfig.for_model(**json.loads(args.config.read_text()))
    with torch.device("meta"):
        vision = SiglipVisionModel(config.vision_config).eval().half()
        pixels = torch.zeros((32, 3, 256, 256), dtype=torch.float16)
    assert all(p.device.type == "meta" for p in vision.parameters())
    with torch.inference_mode():
        exported = torch.export.export(Pooler(vision), (pixels,), strict=True)
        pooled = exported.module()(pixels)
    assert pooled.shape == (32, 1024) and pooled.dtype == torch.float16
    assert pooled.device.type == "meta"
    calls = Counter(str(n.target) for n in exported.graph.nodes if n.op == "call_function")
    receipt = {
        "schema": "sfora-siglip-vision-meta-export-v1",
        "decision": "STRUCTURAL_PASS_ONLY",
        "config_sha256": digest,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "attention_implementation": vision.config._attn_implementation,
        "input_shape": list(pixels.shape), "output_shape": list(pooled.shape),
        "dtype": str(pooled.dtype), "device": str(pooled.device),
        "wall_seconds": time.monotonic() - started,
        "call_function_counts": dict(sorted(calls.items())),
        "limitations": "No GPU compilation, real output parity, quality, or latency measured.",
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
