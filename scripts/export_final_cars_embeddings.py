#!/usr/bin/env python3
"""Export and independently score an explicitly final Cars196 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np

from sfora.data import load_image_retrieval_examples
from sfora.image_end_to_end import (
    ImageEndToEndConfig,
    _default_transform_factory,
    _encode_model,
    _resolve_training_schedule,
    _score_end_to_end_retrieval,
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


def image_content_sha256(image: object) -> str:
    """Hash filesystem-backed or already-decoded image content deterministically."""
    if isinstance(image, (str, os.PathLike)):
        path = Path(image).resolve()
        if not path.is_file():
            raise ValueError(f"Cars196 source path is missing: {path}")
        return sha256(path)
    raw_bytes = getattr(image, "tobytes", None)
    if callable(raw_bytes):
        digest = hashlib.sha256()
        digest.update(str(getattr(image, "mode", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(getattr(image, "size", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw_bytes())
        return digest.hexdigest()
    raise ValueError(f"unsupported Cars196 image source type: {type(image).__name__}")


def image_source_reference(image: object) -> str:
    """Return a path when one exists; decoded HF images intentionally have none."""
    if isinstance(image, (str, os.PathLike)):
        return str(Path(image).resolve())
    return ""


def verify_official_partition(
    train_examples: list[Any], test_examples: list[Any]
) -> dict[str, Any]:
    """Fail closed on Cars196 split sizes, identities, IDs, and source paths."""
    summaries: dict[str, dict[str, Any]] = {}
    for split, examples in (("train", train_examples), ("test", test_examples)):
        labels = {int(example.label) for example in examples}
        class_counts = {
            label: sum(int(example.label) == label for example in examples) for label in labels
        }
        example_ids = {str(example.example_id) for example in examples}
        source_references = [image_source_reference(example.image) for example in examples]
        nonempty_source_references = {value for value in source_references if value}
        content_hashes = {image_content_sha256(example.image) for example in examples}
        observed = (len(examples), len(labels))
        if observed != EXPECTED_CARS_PARTITION[split]:
            raise ValueError(
                f"official Cars196 {split} split count mismatch: "
                f"{observed} != {EXPECTED_CARS_PARTITION[split]}"
            )
        if len(example_ids) != len(examples):
            raise ValueError(f"duplicate Cars196 {split} example IDs")
        if any(count < 2 for count in class_counts.values()):
            raise ValueError(
                f"Cars196 {split} contains a class with fewer than two examples; "
                "leave-one-out R@1 would not be defined for every query"
            )
        if nonempty_source_references and len(nonempty_source_references) != len(examples):
            raise ValueError(f"duplicate Cars196 {split} source paths")
        summaries[split] = {
            "labels": labels,
            "example_ids": example_ids,
            "source_references": nonempty_source_references,
            "content_hashes": content_hashes,
            "content_duplicate_rows": len(examples) - len(content_hashes),
            "minimum_class_examples": min(class_counts.values()),
        }

    if summaries["train"]["labels"] & summaries["test"]["labels"]:
        raise ValueError("Cars196 train and test identities overlap")
    if summaries["train"]["example_ids"] & summaries["test"]["example_ids"]:
        raise ValueError("Cars196 train and test example IDs overlap")
    if summaries["train"]["source_references"] & summaries["test"]["source_references"]:
        raise ValueError("Cars196 train and test source paths overlap")
    content_overlap = summaries["train"]["content_hashes"] & summaries["test"]["content_hashes"]
    if content_overlap:
        raise ValueError("Cars196 train and test decoded image content overlaps")
    return {
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "train_classes": len(summaries["train"]["labels"]),
        "test_classes": len(summaries["test"]["labels"]),
        "train_test_identity_overlap": 0,
        "train_test_example_id_overlap": 0,
        "train_test_source_reference_overlap": 0,
        "train_test_content_overlap": 0,
        "train_content_duplicate_rows": summaries["train"]["content_duplicate_rows"],
        "test_content_duplicate_rows": summaries["test"]["content_duplicate_rows"],
        "train_minimum_class_examples": summaries["train"]["minimum_class_examples"],
        "test_minimum_class_examples": summaries["test"]["minimum_class_examples"],
    }


def independent_leave_one_out_recall_at_1(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    chunk_size: int = 1024,
    use_partial_top_k: bool = True,
) -> float:
    """Compute test-to-test squared-L2 R@1 without the production scorer.

    The model export is already L2-normalized in torch. Preserve those exported
    float32 coordinates (cast to float64 for scoring) rather than normalizing a
    second time: standard DML scoring ranks the exported unit vectors by L2, and
    a second normalization can change near ties because float32 norms are not
    mathematically exact ones.
    """
    vectors = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if vectors.ndim != 2 or len(vectors) != len(labels):
        raise ValueError("embeddings and labels have incompatible shapes")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(~np.isfinite(vectors)) or np.any(~np.isfinite(norms)):
        raise ValueError("independent retrieval received a non-finite embedding")
    if np.any(norms <= 0):
        raise ValueError("independent retrieval received a zero-norm embedding")
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-5):
        raise ValueError("exported retrieval embeddings are not L2-normalized")
    squared_norms = np.sum(vectors * vectors, axis=1)
    label_counts = dict(zip(*np.unique(labels, return_counts=True), strict=True))
    max_relevant_count = max(int(label_counts[label]) - 1 for label in label_counts)
    top_k = min(len(vectors) - 1, max(30, max_relevant_count))
    correct = 0
    for start in range(0, len(vectors), chunk_size):
        stop = min(start + chunk_size, len(vectors))
        distances = (
            squared_norms[start:stop, None]
            + squared_norms[None, :]
            - (2.0 * vectors[start:stop] @ vectors.T)
        )
        distances = np.maximum(distances, 0.0)
        row_indices = np.arange(stop - start)
        distances[row_indices, np.arange(start, stop)] = np.inf
        if use_partial_top_k and top_k < len(vectors):
            top_indices = np.argpartition(distances, kth=top_k - 1, axis=1)[:, :top_k]
            top_distances = np.take_along_axis(distances, top_indices, axis=1)
            top_order = np.argsort(top_distances, axis=1, kind="stable")
            nearest = np.take_along_axis(top_indices, top_order, axis=1)[:, 0]
        else:
            nearest = np.argsort(distances, axis=1, kind="stable")[:, 0]
        correct += int(np.sum(labels[nearest] == labels[start:stop]))
    return correct / len(vectors)


def reported_final_recall_at_1(
    report: dict[str, Any],
    expected_steps: int,
    *,
    expected_objective: str = "pfml",
) -> float:
    methods = report.get("methods")
    if not isinstance(methods, dict) or len(methods) != 1:
        raise ValueError("Cars196 report must contain exactly one method")
    metrics = next(iter(methods.values()))
    if metrics.get("objective") != expected_objective:
        raise ValueError(f"Cars196 report method is not {expected_objective}")
    if metrics.get("executed_train_steps") != expected_steps:
        raise ValueError("Cars196 report did not execute the resolved final training step")
    value = float(metrics.get("recall_at_1", float("nan")))
    if not np.isfinite(value):
        raise ValueError("Cars196 report final R@1 is not finite")
    return value


def validate_export_recipe(
    config: Any,
    *,
    expected_recipe_id: str,
    expected_recipe_digest: str,
) -> None:
    if getattr(config, "recipe_id", None) != expected_recipe_id:
        raise ValueError("Cars196 export recipe ID differs")
    if (
        len(expected_recipe_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_recipe_digest)
        or getattr(config, "recipe_digest", None) != expected_recipe_digest
    ):
        raise ValueError("Cars196 export recipe digest differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-recipe-id", required=True)
    parser.add_argument("--expected-recipe-digest", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    report = json.loads(args.report.read_text(encoding="utf-8"))
    config = ImageEndToEndConfig.model_validate(report["config"])
    if config.dataset_name != "cars" or len(config.objectives) != 1:
        raise ValueError("exporter requires a single-objective Cars196 report")
    objective = config.objectives[0]
    if objective not in {"pfml", "proxy_anchor"}:
        raise ValueError("exporter supports only Cars196 PFML or Proxy Anchor")
    validate_export_recipe(
        config,
        expected_recipe_id=args.expected_recipe_id,
        expected_recipe_digest=args.expected_recipe_digest,
    )
    if config.checkpoint_selection_interval != 0:
        raise ValueError("final-state exporter refuses checkpoint-selected training")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("artifact_selection") != "final_training_state":
        raise ValueError("checkpoint is not explicitly labeled as a final training state")
    if checkpoint.get("evaluation_model_source") != "student":
        raise ValueError("Cars196 checkpoint is not the trained student")
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
    reported_final_recall = reported_final_recall_at_1(
        report, resolved_steps, expected_objective=objective
    )

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
        independent_leave_one_out_recall_at_1(embeddings, labels) if args.split == "test" else None
    )
    stable_full_order_recall = (
        independent_leave_one_out_recall_at_1(embeddings, labels, use_partial_top_k=False)
        if args.split == "test"
        else None
    )
    production_rescore = (
        _score_end_to_end_retrieval(
            query_embeddings=embeddings,
            query_labels=labels,
            gallery_embeddings=None,
            gallery_labels=None,
            query_limit=None,
            random_state=config.seed,
            region_grid=config.region_grid,
        ).recall_at_1
        if args.split == "test"
        else None
    )
    if independent_recall is not None:
        if production_rescore is None:
            raise RuntimeError("test export did not produce a production-scorer result")
        if not np.isclose(independent_recall, reported_final_recall, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "independently exported final R@1 disagrees with the report: "
                f"independent={independent_recall}, production_rescore={production_rescore}, "
                f"stable_full_order={stable_full_order_recall}, "
                f"report={reported_final_recall}"
            )
        if not np.isclose(independent_recall, production_rescore, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "independent and production scorers disagree on the same exported embeddings: "
                f"{independent_recall} != {production_rescore}"
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
        source_references=np.asarray(
            [image_source_reference(example.image) for example in examples]
        ),
        content_sha256=np.asarray([image_content_sha256(example.image) for example in examples]),
        artifact_selection=np.asarray("final_training_state"),
        split=np.asarray(args.split),
        checkpoint_sha256=np.asarray(checkpoint_digest),
        report_sha256=np.asarray(report_digest),
        independent_recall_at_1=np.asarray(
            np.nan if independent_recall is None else independent_recall
        ),
        production_rescore_at_1=np.asarray(
            np.nan if production_rescore is None else production_rescore
        ),
        stable_full_order_recall_at_1=np.asarray(
            np.nan if stable_full_order_recall is None else stable_full_order_recall
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
                "production_rescore_at_1": production_rescore,
                "stable_full_order_recall_at_1": stable_full_order_recall,
                "reported_final_recall_at_1": reported_final_recall,
                "partition_audit": partition_audit,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
