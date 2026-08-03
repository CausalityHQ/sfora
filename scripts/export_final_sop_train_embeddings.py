#!/usr/bin/env python3
"""Export an official SOP split from an explicitly final checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from sfora.data import load_image_retrieval_examples
from sfora.image_end_to_end import (
    ImageEndToEndConfig,
    _default_transform_factory,
    _encode_model,
    _resolve_training_schedule,
    _TorchImageDataset,
    _torchvision_model_factory,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    report = json.loads(args.report.read_text(encoding="utf-8"))
    config = ImageEndToEndConfig.model_validate(report["config"])
    if config.dataset_name != "sop" or config.objectives != ("proxy_anchor",):
        raise ValueError("exporter requires a single-objective SOP Proxy Anchor report")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("artifact_selection") not in {None, "final_training_state"}:
        raise ValueError("checkpoint is not labeled as a final training state")
    examples = load_image_retrieval_examples(
        dataset_name="sop", split=args.split, seed=config.seed
    )
    expected = {"train": (59_551, 11_318), "test": (60_502, 11_316)}[args.split]
    observed = (len(examples), len({example.label for example in examples}))
    if observed != expected:
        raise ValueError(
            f"official SOP {args.split} split count mismatch: {observed} != {expected}"
        )
    training_examples = (
        examples
        if args.split == "train"
        else load_image_retrieval_examples(dataset_name="sop", split="train", seed=config.seed)
    )
    resolved_steps, _, _ = _resolve_training_schedule(
        config,
        optimization_example_count=len(training_examples),
        optimization_labels=[example.label for example in training_examples],
    )
    if checkpoint.get("training_step") not in {None, resolved_steps}:
        raise ValueError(
            "checkpoint training step differs from the resolved official training schedule: "
            f"{checkpoint.get('training_step')} != {resolved_steps}"
        )
    transform = _default_transform_factory(config, False)
    loader: Any = DataLoader(
        cast(Any, _TorchImageDataset(examples, transform)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model: Any = _torchvision_model_factory(config)
    state = {
        key: value
        for key, value in checkpoint["state_dict"].items()
        if key not in {"metric_proxies", "metric_proxy_labels"}
    }
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    embeddings, labels = _encode_model(model, loader, device, torch)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=np.asarray([example.example_id for example in examples]),
        artifact_selection=np.asarray("final_training_state"),
        split=np.asarray(args.split),
        checkpoint_sha256=np.asarray(sha256(args.checkpoint)),
        report_sha256=np.asarray(sha256(args.report)),
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(examples),
                "classes": len(set(labels.tolist())),
                "split": args.split,
                "artifact_selection": "final_training_state",
                "checkpoint_sha256": sha256(args.checkpoint),
                "report_sha256": sha256(args.report),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
