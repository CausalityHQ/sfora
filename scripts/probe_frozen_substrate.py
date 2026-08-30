#!/usr/bin/env python3
"""Claim-ineligible Cars train-band screen for a pinned frozen backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import torch
from PIL import Image

from sfora.data import _HF_DATASET_REVISIONS, load_image_retrieval_examples, materialize_image
from sfora.substrate_screen import (
    SUBSTRATE_F0_CLASSES,
    score_frozen_substrate,
    validate_substrate_holdout,
)


class RegisteredSubstrate(NamedTuple):
    model_name: str
    model_revision: str
    readout: str


_SUBSTRATES = {
    "dinov2-large": RegisteredSubstrate(
        "facebook/dinov2-large",
        "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
        "last_hidden_state_cls",
    ),
    "siglip2-so400m": RegisteredSubstrate(
        "google/siglip2-so400m-patch14-384",
        "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
        "vision_pooler_output",
    ),
    "siglip-so400m": RegisteredSubstrate(
        "google/siglip-so400m-patch14-384",
        "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        "vision_pooler_output",
    ),
}
_DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
_EXPECTED_QUERIES = 1345
_RECALL_MINIMUM = 0.94
_NORM_TOLERANCE = 1.0e-6


def substrate_passed(*, correct: int) -> bool:
    """Apply the single preregistered substrate-headroom gate."""

    return correct / _EXPECTED_QUERIES >= _RECALL_MINIMUM


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _materialize_rgb(image: Any) -> Image.Image:
    materialized = materialize_image(image)
    if not isinstance(materialized, Image.Image):
        raise TypeError("image materialization did not produce a PIL image")
    converted = materialized.convert("RGB")
    if converted.mode != "RGB":
        raise RuntimeError("image materialization did not produce RGB")
    return converted


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    if partial.exists():
        raise FileExistsError(f"refusing pre-existing partial {partial}")
    partial.write_bytes(payload)
    partial.replace(path)


def _encode(
    examples: list[Any],
    *,
    model_name: str,
    model_revision: str,
    readout: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        model_name, revision=model_revision, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        model_name, revision=model_revision, local_files_only=True
    ).eval().to(device)
    rows: list[torch.Tensor] = []
    observed_shape: tuple[int, int] | None = None
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            images = [_materialize_rgb(row.image) for row in examples[start : start + batch_size]]
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
            shape = (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
            if observed_shape is None:
                observed_shape = shape
            elif shape != observed_shape:
                raise RuntimeError("processor emitted inconsistent image shapes")
            if readout == "last_hidden_state_cls":
                output = model(pixel_values=pixel_values.to(device))
                descriptor = output.last_hidden_state[:, 0, :].float()
            elif readout == "vision_pooler_output":
                output = model.vision_model(pixel_values=pixel_values.to(device))
                descriptor = output.pooler_output.float()
            else:
                raise RuntimeError("unregistered substrate readout")
            rows.append(torch.nn.functional.normalize(descriptor, dim=-1).cpu())
    if observed_shape is None:
        raise RuntimeError("the holdout is empty")
    del model
    return torch.cat(rows), observed_shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-tree-digest", required=True)
    parser.add_argument("--cell", choices=sorted(_SUBSTRATES), default="dinov2-large")
    parser.add_argument("--model-name")
    parser.add_argument("--model-revision")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-block", type=int, default=32)
    args = parser.parse_args()
    substrate = _SUBSTRATES[args.cell]
    model_name = args.model_name or substrate.model_name
    model_revision = args.model_revision or substrate.model_revision
    if (model_name, model_revision) != (
        substrate.model_name,
        substrate.model_revision,
    ):
        raise ValueError("model authority differs from the registered substrate cell")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        raise ValueError("source revision must be an exact 40-character commit")
    if re.fullmatch(r"[0-9a-f]{64}", args.source_tree_digest) is None:
        raise ValueError("source tree digest must be an exact SHA-256")
    if args.batch_size < 1 or args.query_block < 1:
        raise ValueError("batch and query block sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("the substrate screen requires CUDA")
    if _HF_DATASET_REVISIONS["tanganke/stanford_cars"] != _DATASET_REVISION:
        raise RuntimeError("the Cars dataset revision authority differs")

    examples = load_image_retrieval_examples(dataset_name="cars", split="train")
    dataset_examples_sha256 = hashlib.sha256(
        _canonical_bytes(
            {
                "examples": sorted(
                    (str(row.example_id), int(row.label)) for row in examples
                )
            }
        )
    ).hexdigest()
    holdout = [row for row in examples if int(row.label) in SUBSTRATE_F0_CLASSES]
    if len(holdout) != _EXPECTED_QUERIES:
        raise RuntimeError("the registered Cars holdout cardinality differs")
    labels = torch.tensor([int(row.label) for row in holdout], dtype=torch.int64)
    validate_substrate_holdout(split="train", labels=labels)
    descriptors, image_shape = _encode(
        holdout,
        model_name=model_name,
        model_revision=model_revision,
        readout=substrate.readout,
        batch_size=args.batch_size,
        device=torch.device("cuda"),
    )
    metrics = score_frozen_substrate(
        descriptors.cuda(), labels.cuda(), query_block=args.query_block
    )
    result = {
        "schema": "sfora-frozen-substrate-screen-v1",
        "claim_eligible": False,
        "source_revision": args.source_revision,
        "source_tree_digest": args.source_tree_digest,
        "dataset": "cars",
        "dataset_revision": _DATASET_REVISION,
        "dataset_examples_sha256": dataset_examples_sha256,
        "split": "train",
        "holdout_classes": sorted(SUBSTRATE_F0_CLASSES),
        "cell": args.cell,
        "model_name": model_name,
        "model_revision": model_revision,
        "readout": substrate.readout,
        "compute_dtype": "float32",
        "processor_image_shape": list(image_shape),
        "descriptors_validated": True,
        "norm_tolerance": _NORM_TOLERANCE,
        "metrics": {
            "correct": metrics.correct,
            "queries": metrics.queries,
            "recall_at_1": metrics.recall_at_1,
        },
        "gates": {
            "expected_queries": _EXPECTED_QUERIES,
            "recall_at_1_minimum": _RECALL_MINIMUM,
        },
        "passed": substrate_passed(correct=metrics.correct),
    }
    payload = _canonical_bytes(result)
    _write_new(args.output, payload)
    summary = {
        "output": str(args.output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "passed": result["passed"],
        **result["metrics"],
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
