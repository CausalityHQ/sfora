#!/usr/bin/env python3
"""Export and independently score an explicitly final Cars196 PFML checkpoint."""

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

EXPECTED_CARS_PARTITION = {"train": (8_054, 98), "test": (8_131, 98)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_partition(
    train_examples: list[Any], test_examples: list[Any]
) -> dict[str, Any]:
    """Fail closed on Cars196 split sizes, identities, IDs, and source paths."""
    summaries: dict[str, dict[str, Any]] = {}
    for split, examples in (("train", train_examples), ("test", test_examples)):
        labels = {int(example.label) for example in examples}
        example_ids = {str(example.example_id) for example in examples}
        source_paths = {str(Path(example.image).resolve()) for example in examples}
        observed = (len(examples), len(labels))
        if observed != EXPECTED_CARS_PARTITION[split]:
            raise ValueError(
                f"official Cars196 {split} split count mismatch: "
                f"{observed} != {EXPECTED_CARS_PARTITION[split]}"
            )
        if len(example_ids) != len(examples):
            raise ValueError(f"duplicate Cars196 {split} example IDs")
        if len(source_paths) != len(examples):
            raise ValueError(f"duplicate Cars196 {split} source paths")
        summaries[split] = {
            "labels": labels,
            "example_ids": example_ids,
            "source_paths": source_paths,
        }

    if summaries["train"]["labels"] & summaries["test"]["labels"]:
        raise ValueError("Cars196 train and test identities overlap")
    if summaries["train"]["example_ids"] & summaries["test"]["example_ids"]:
        raise ValueError("Cars196 train and test example IDs overlap")
    if summaries["train"]["source_paths"] & summaries["test"]["source_paths"]:
        raise ValueError("Cars196 train and test source paths overlap")
    return {
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "train_classes": len(summaries["train"]["labels"]),
        "test_classes": len(summaries["test"]["labels"]),
        "train_test_identity_overlap": 0,
        "train_test_example_id_overlap": 0,
        "train_test_source_path_overlap": 0,
    }


def independent_leave_one_out_recall_at_1(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    chunk_size: int = 512,
) -> float:
    """Compute normalized test-to-test R@1 without the production scorer."""
    vectors = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if vectors.ndim != 2 or len(vectors) != len(labels):
        raise ValueError("embeddings and labels have incompatible shapes")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(~np.isfinite(vectors)) or np.any(~np.isfinite(norms)):
        raise ValueError("independent retrieval received a non-finite embedding")
    if np.any(norms <= 0):
        raise ValueError("independent retrieval received a zero-norm embedding")
    vectors = vectors / norms
    correct = 0
    for start in range(0, len(vectors), chunk_size):
        stop = min(start + chunk_size, len(vectors))
        similarities = vectors[start:stop] @ vectors.T
        row_indices = np.arange(stop - start)
        similarities[row_indices, np.arange(start, stop)] = -np.inf
        nearest = np.argmax(similarities, axis=1)
        correct += int(np.sum(labels[nearest] == labels[start:stop]))
    return correct / len(vectors)


def reported_final_recall_at_1(report: dict[str, Any], expected_steps: int) -> float:
    methods = report.get("methods")
    if not isinstance(methods, dict) or len(methods) != 1:
        raise ValueError("Cars196 PFML report must contain exactly one method")
    metrics = next(iter(methods.values()))
    if metrics.get("objective") != "pfml":
        raise ValueError("Cars196 report method is not PFML")
    if metrics.get("executed_train_steps") != expected_steps:
        raise ValueError("Cars196 report did not execute the resolved final training step")
    value = float(metrics.get("recall_at_1", float("nan")))
    if not np.isfinite(value):
        raise ValueError("Cars196 report final R@1 is not finite")
    return value


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
    if config.dataset_name != "cars" or config.objectives != ("pfml",):
        raise ValueError("exporter requires a single-objective Cars196 PFML report")
    if config.checkpoint_selection_interval != 0:
        raise ValueError("final-state exporter refuses checkpoint-selected training")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("artifact_selection") != "final_training_state":
        raise ValueError("checkpoint is not explicitly labeled as a final training state")
    if checkpoint.get("evaluation_model_source") != "student":
        raise ValueError("Cars196 PFML checkpoint is not the trained student")
    expected_arch = {
        "backbone_name": config.backbone_name,
        "pretrained_weights": config.pretrained_weights,
        "head_pooling": config.head_pooling,
        "embedding_dimensions": config.embedding_dimensions,
        "embedding_head_init": config.embedding_head_init,
        "embedding_layer_norm": config.embedding_layer_norm,
    }
    if checkpoint.get("arch") != expected_arch:
        raise ValueError("checkpoint architecture metadata does not match the report config")
    if checkpoint.get("training_config") != config.model_dump(mode="json"):
        raise ValueError("checkpoint training_config does not match the report config")

    train_examples = load_image_retrieval_examples(
        dataset_name="cars", split="train", seed=config.seed
    )
    test_examples = load_image_retrieval_examples(
        dataset_name="cars", split="test", seed=config.seed
    )
    partition_audit = verify_official_partition(train_examples, test_examples)
    resolved_steps, _, _ = _resolve_training_schedule(
        config,
        optimization_example_count=len(train_examples),
        optimization_labels=[example.label for example in train_examples],
    )
    if checkpoint.get("training_step") != resolved_steps:
        raise ValueError(
            "checkpoint training step differs from the resolved official training schedule: "
            f"{checkpoint.get('training_step')} != {resolved_steps}"
        )
    reported_final_recall = reported_final_recall_at_1(report, resolved_steps)

    examples = train_examples if args.split == "train" else test_examples
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
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("exported Cars196 embeddings contain non-finite values")
    independent_recall = (
        independent_leave_one_out_recall_at_1(embeddings, labels)
        if args.split == "test"
        else None
    )
    if independent_recall is not None and not np.isclose(
        independent_recall, reported_final_recall, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "independently exported final R@1 disagrees with the report: "
            f"{independent_recall} != {reported_final_recall}"
        )

    checkpoint_digest = sha256(args.checkpoint)
    report_digest = sha256(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=np.asarray([example.example_id for example in examples]),
        source_paths=np.asarray([str(Path(example.image)) for example in examples]),
        artifact_selection=np.asarray("final_training_state"),
        split=np.asarray(args.split),
        checkpoint_sha256=np.asarray(checkpoint_digest),
        report_sha256=np.asarray(report_digest),
        independent_recall_at_1=np.asarray(
            np.nan if independent_recall is None else independent_recall
        ),
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "split": args.split,
                "artifact_selection": "final_training_state",
                "checkpoint_sha256": checkpoint_digest,
                "report_sha256": report_digest,
                "independent_recall_at_1": independent_recall,
                "reported_final_recall_at_1": reported_final_recall,
                "partition_audit": partition_audit,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
