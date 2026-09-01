#!/usr/bin/env python3
"""Claim-ineligible Cars train-band screen for a pinned frozen backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from PIL import Image

from sfora.data import _HF_DATASET_REVISIONS, load_image_retrieval_examples, materialize_image
from sfora.substrate_screen import (
    SUBSTRATE_F0_CLASSES,
    SubstrateRetrievalError,
    score_frozen_substrate_evidence,
    validate_substrate_holdout,
)
from sfora.twin_reachability import (
    TwinReachabilityAuthority,
    build_twin_reachability,
    build_twin_reachability_inference,
    canonical_twin_reachability_artifact_bytes,
    canonical_twin_reachability_inference_artifact_bytes,
    twin_reachability_inference_seeds,
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
_F1_BATCH_SIZE = 8
_F1_QUERY_BLOCK = 32
_F1_EXPECTED_CORRECT = 1242
_ERROR_EVIDENCE_AUTHORITIES = {
    "dinov2-large": (32, _F1_QUERY_BLOCK, 1196),
    "siglip2-so400m": (_F1_BATCH_SIZE, _F1_QUERY_BLOCK, 1227),
    "siglip-so400m": (_F1_BATCH_SIZE, _F1_QUERY_BLOCK, _F1_EXPECTED_CORRECT),
}


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


def _require_new_outputs(paths: tuple[Path, ...]) -> None:
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("output paths must be distinct")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        partial = path.with_name(f".{path.name}.partial")
        if partial.exists():
            raise FileExistsError(f"refusing pre-existing partial {partial}")


def _publish_new_outputs(outputs: tuple[tuple[Path, bytes], ...]) -> None:
    paths = tuple(path for path, _ in outputs)
    _require_new_outputs(paths)
    partials: list[Path] = []
    published: list[Path] = []
    try:
        for path, payload in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f".{path.name}.partial")
            with partial.open("xb") as stream:
                stream.write(payload)
            partials.append(partial)
        for (path, _), partial in zip(outputs, partials, strict=True):
            os.link(partial, path)
            published.append(path)
            partial.unlink()
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        for partial in partials:
            partial.unlink(missing_ok=True)
        raise


def _validate_error_evidence_request(
    *, cell: str, batch_size: int, query_block: int, expected_correct: int
) -> None:
    if _ERROR_EVIDENCE_AUTHORITIES.get(cell) != (
        batch_size,
        query_block,
        expected_correct,
    ):
        raise ValueError("request differs from the registered substrate execution authority")


def _descriptor_sha256(descriptors: torch.Tensor) -> str:
    canonical = descriptors.detach().to(device="cpu", dtype=torch.float32).contiguous()
    header = _canonical_bytes(
        {"dtype": "float32-le", "shape": [int(size) for size in canonical.shape]}
    )
    values = canonical.numpy().astype("<f4", copy=False).tobytes(order="C")
    return hashlib.sha256(header + values).hexdigest()


def _build_twin_reachability_artifacts(
    *,
    examples: Sequence[Any],
    descriptors: torch.Tensor,
    source_revision: str,
    source_tree_digest: str,
    dataset_examples_sha256: str,
    model_name: str,
    model_revision: str,
) -> tuple[bytes, bytes]:
    if (
        not isinstance(descriptors, torch.Tensor)
        or descriptors.ndim != 2
        or descriptors.shape[0] != len(examples)
    ):
        raise TypeError("twin descriptor row authority differs")
    positions = [
        index
        for index, example in enumerate(examples)
        if getattr(example, "label", None) in {82, 83}
    ]
    example_ids = tuple(getattr(examples[index], "example_id", None) for index in positions)
    labels = tuple(getattr(examples[index], "label", None) for index in positions)
    if (
        len(positions) < 40
        or min(labels.count(82), labels.count(83)) < 20
        or len(set(example_ids)) != len(example_ids)
        or any(type(value) is not str or not value for value in example_ids)
        or set(labels) != {82, 83}
        or any(type(value) is not int for value in labels)
    ):
        raise ValueError("twin example authority differs")
    selected = descriptors[positions].detach().to(device="cpu", dtype=torch.float32).contiguous()
    evidence = build_twin_reachability(
        "frozen-pooled",
        selected.numpy().astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
    )
    authority = TwinReachabilityAuthority(
        plane="frozen-pooled",
        source_revision=source_revision,
        source_tree_digest=source_tree_digest,
        dataset_revision=_DATASET_REVISION,
        dataset_manifest_sha256=dataset_examples_sha256,
        model_name=model_name,
        model_revision=model_revision,
        producer_kind="frozen-model",
        producer_identity=model_revision,
        ordered_example_ids_sha256=hashlib.sha256(
            _canonical_bytes({"example_ids": list(example_ids)})
        ).hexdigest(),
        label_vector_sha256=hashlib.sha256(
            _canonical_bytes({"labels": list(labels)})
        ).hexdigest(),
        descriptor_sha256=_descriptor_sha256(selected),
    )
    bootstrap_seed, permutation_seed = twin_reachability_inference_seeds(authority)
    inference = build_twin_reachability_inference(
        "frozen-pooled",
        selected.numpy().astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
        bootstrap_seed=bootstrap_seed,
        permutation_seed=permutation_seed,
        expected_evidence=evidence,
    )
    return (
        canonical_twin_reachability_artifact_bytes(authority, evidence),
        canonical_twin_reachability_inference_artifact_bytes(
            authority,
            evidence,
            inference,
        ),
    )


def _build_error_manifest(
    *,
    examples: list[Any],
    errors: tuple[SubstrateRetrievalError, ...],
    source_revision: str,
    source_tree_digest: str,
    dataset_examples_sha256: str,
    descriptor_sha256: str,
    batch_size: int,
    query_block: int,
    cell: str,
    model_name: str,
    model_revision: str,
    class_names: Sequence[str],
) -> dict[str, Any]:
    if len(class_names) != 196:
        raise ValueError("Cars class-name authority is incomplete")
    rows: list[dict[str, Any]] = []
    previous = -1
    for error in errors:
        if error.query_position <= previous:
            raise ValueError("retrieval errors must be ordered by unique query position")
        if not 0 <= error.query_position < len(examples):
            raise ValueError("retrieval error query position is out of range")
        if not 0 <= error.nearest_position < len(examples):
            raise ValueError("retrieval error nearest position is out of range")
        query = examples[error.query_position]
        nearest = examples[error.nearest_position]
        if int(query.label) != error.query_label or int(nearest.label) != error.nearest_label:
            raise ValueError("retrieval error labels differ from example authority")
        rows.append(
            {
                "query_position": error.query_position,
                "query_example_id": str(query.example_id),
                "query_label": error.query_label,
                "nearest_position": error.nearest_position,
                "nearest_example_id": str(nearest.example_id),
                "nearest_label": error.nearest_label,
            }
        )
        previous = error.query_position
    return {
        "schema": "sfora-frozen-substrate-errors-v1",
        "claim_eligible": False,
        "source_revision": source_revision,
        "source_tree_digest": source_tree_digest,
        "dataset": "cars",
        "dataset_revision": _DATASET_REVISION,
        "dataset_examples_sha256": dataset_examples_sha256,
        "descriptor_sha256": descriptor_sha256,
        "batch_size": batch_size,
        "query_block": query_block,
        "split": "train",
        "holdout_classes": sorted(SUBSTRATE_F0_CLASSES),
        "class_names": [
            {"id": label, "name": class_names[label]}
            for label in sorted(SUBSTRATE_F0_CLASSES)
        ],
        "cell": cell,
        "model_name": model_name,
        "model_revision": model_revision,
        "error_count": len(rows),
        "errors": rows,
    }


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


def _load_cars_class_names() -> tuple[str, ...]:
    from datasets import load_dataset

    dataset = load_dataset(
        "tanganke/stanford_cars",
        split="train",
        revision=_DATASET_REVISION,
    )
    label_feature = dataset.features.get("label")
    names = getattr(label_feature, "names", None)
    if not isinstance(names, list) or not all(
        isinstance(name, str) and name for name in names
    ):
        raise RuntimeError("Cars class-name authority is unavailable")
    return tuple(names)


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
    parser.add_argument("--error-manifest", type=Path)
    parser.add_argument("--twin-reachability", type=Path)
    parser.add_argument("--twin-reachability-inference", type=Path)
    parser.add_argument("--expected-correct", type=int)
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
    if (args.error_manifest is None) != (args.expected_correct is None):
        raise ValueError("error manifest and expected correct count must be specified together")
    if (args.twin_reachability is None) != (args.twin_reachability_inference is None):
        raise ValueError("twin reachability outputs must be specified together")
    if args.twin_reachability is not None and args.error_manifest is None:
        raise ValueError("twin reachability requires the registered error-manifest execution")
    if args.error_manifest is not None:
        assert args.expected_correct is not None
        _validate_error_evidence_request(
            cell=args.cell,
            batch_size=args.batch_size,
            query_block=args.query_block,
            expected_correct=args.expected_correct,
        )
    outputs: tuple[Path, ...] = (args.output,)
    if args.error_manifest is not None:
        outputs += (args.error_manifest,)
    if args.twin_reachability is not None:
        outputs += (args.twin_reachability, args.twin_reachability_inference)
    _require_new_outputs(outputs)
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
    evidence = score_frozen_substrate_evidence(
        descriptors.cuda(), labels.cuda(), query_block=args.query_block
    )
    metrics = evidence.metrics
    descriptor_sha256 = _descriptor_sha256(descriptors)
    error_manifest_sha256: str | None = None
    error_payload: bytes | None = None
    twin_payload: bytes | None = None
    twin_inference_payload: bytes | None = None
    if args.error_manifest is not None:
        class_names = _load_cars_class_names()
        if metrics.correct != args.expected_correct:
            raise RuntimeError("sealed substrate correct count did not reproduce")
        if len(evidence.errors) != metrics.queries - metrics.correct:
            raise RuntimeError("retrieval error cardinality differs from metrics")
        error_manifest = _build_error_manifest(
            examples=holdout,
            errors=evidence.errors,
            source_revision=args.source_revision,
            source_tree_digest=args.source_tree_digest,
            dataset_examples_sha256=dataset_examples_sha256,
            descriptor_sha256=descriptor_sha256,
            batch_size=args.batch_size,
            query_block=args.query_block,
            cell=args.cell,
            model_name=model_name,
            model_revision=model_revision,
            class_names=class_names,
        )
        error_payload = _canonical_bytes(error_manifest)
        error_manifest_sha256 = hashlib.sha256(error_payload).hexdigest()
        if args.twin_reachability is not None:
            twin_payload, twin_inference_payload = _build_twin_reachability_artifacts(
                examples=holdout,
                descriptors=descriptors,
                source_revision=args.source_revision,
                source_tree_digest=args.source_tree_digest,
                dataset_examples_sha256=dataset_examples_sha256,
                model_name=model_name,
                model_revision=model_revision,
            )
    result = {
        "schema": (
            "sfora-frozen-substrate-screen-v2"
            if args.error_manifest is not None
            else "sfora-frozen-substrate-screen-v1"
        ),
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
    if args.error_manifest is not None:
        result.update(
            {
                "batch_size": args.batch_size,
                "query_block": args.query_block,
                "descriptor_shape": list(descriptors.shape),
                "descriptor_sha256": descriptor_sha256,
                "error_manifest_sha256": error_manifest_sha256,
            }
        )
        if twin_payload is not None:
            result["twin_reachability_sha256"] = hashlib.sha256(twin_payload).hexdigest()
            assert twin_inference_payload is not None
            result["twin_reachability_inference_sha256"] = hashlib.sha256(
                twin_inference_payload
            ).hexdigest()
    payload = _canonical_bytes(result)
    if args.error_manifest is None:
        _write_new(args.output, payload)
    else:
        assert error_payload is not None
        publication: list[tuple[Path, bytes]] = [(args.error_manifest, error_payload)]
        if args.twin_reachability is not None:
            assert twin_payload is not None
            assert twin_inference_payload is not None
            publication.append((args.twin_reachability, twin_payload))
            publication.append(
                (args.twin_reachability_inference, twin_inference_payload)
            )
        publication.append((args.output, payload))
        _publish_new_outputs(tuple(publication))
    summary = {
        "output": str(args.output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "passed": result["passed"],
        **result["metrics"],
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
