#!/usr/bin/env python3
"""Evaluate the Proxy Anchor authors' published In-Shop checkpoint locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
from export_final_inshop_embeddings import (
    independent_query_gallery_sensitivity,
    verify_official_partition,
)

from sfora.data import load_image_retrieval_bundle
from sfora.image_end_to_end import (
    ImageEndToEndConfig,
    _default_transform_factory,
    _encode_model,
    _TorchImageDataset,
    _torchvision_model_factory,
)
from sfora.image_recipes import reference_recipe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upstream_recall_at_1(
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    chunk_size: int = 512,
) -> float:
    """Reproduce upstream's strict-negative-rank definition exactly in float32."""
    query = np.asarray(query_embeddings, dtype=np.float32)
    gallery = np.asarray(gallery_embeddings, dtype=np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    correct = 0
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        similarity = query[start:stop] @ gallery.T
        labels = query_labels[start:stop]
        positive = gallery_labels[None, :] == labels[:, None]
        if not np.all(positive.any(axis=1)):
            raise ValueError("query without a positive gallery item")
        best_positive = np.where(positive, similarity, -np.inf).max(axis=1)
        better_negative_count = (
            (~positive) & (similarity > best_positive[:, None])
        ).sum(axis=1)
        correct += int(np.sum(better_negative_count < 1))
    return correct / len(query)


def _save_embeddings_atomic(
    path: Path,
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    example_ids: np.ndarray,
    source_paths: np.ndarray,
    split: str,
    checkpoint_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=example_ids,
        source_paths=source_paths,
        artifact_selection=np.asarray("published_upstream_checkpoint"),
        split=np.asarray(split),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
    )
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--query-output", type=Path, required=True)
    parser.add_argument("--gallery-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    recipe = reference_recipe("proxy_anchor", "inshop")
    config = ImageEndToEndConfig.model_validate(
        {"dataset_name": "inshop", **recipe.config, "num_workers": args.num_workers}
    )
    bundle = load_image_retrieval_bundle(
        dataset_name="inshop", dataset_root=args.dataset_root, seed=0
    )
    partition_audit = verify_official_partition(bundle)
    assert bundle.gallery is not None

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if set(checkpoint) != {"model_state_dict"}:
        raise ValueError(f"unexpected upstream checkpoint keys: {sorted(checkpoint)}")
    model: Any = _torchvision_model_factory(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    transform = _default_transform_factory(config, False)

    def loader(examples: list[Any]) -> Any:
        return DataLoader(
            cast(Any, _TorchImageDataset(examples, transform)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    query_embeddings, query_labels = _encode_model(model, loader(bundle.query), device, torch)
    gallery_embeddings, gallery_labels = _encode_model(
        model, loader(bundle.gallery), device, torch
    )
    expected_query_labels = np.asarray([example.label for example in bundle.query])
    expected_gallery_labels = np.asarray([example.label for example in bundle.gallery])
    if not np.array_equal(query_labels, expected_query_labels):
        raise ValueError("encoded query labels differ from official order")
    if not np.array_equal(gallery_labels, expected_gallery_labels):
        raise ValueError("encoded gallery labels differ from official order")
    for split, embeddings in (("query", query_embeddings), ("gallery", gallery_embeddings)):
        if not np.isfinite(embeddings).all():
            raise ValueError(f"non-finite {split} embeddings")
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=2e-5, rtol=2e-5):
            raise ValueError(f"non-unit {split} embeddings")

    checkpoint_digest = _sha256(args.checkpoint)
    _save_embeddings_atomic(
        args.query_output,
        embeddings=query_embeddings,
        labels=query_labels,
        example_ids=np.asarray([example.example_id for example in bundle.query]),
        source_paths=np.asarray([str(Path(example.image)) for example in bundle.query]),
        split="query",
        checkpoint_sha256=checkpoint_digest,
    )
    _save_embeddings_atomic(
        args.gallery_output,
        embeddings=gallery_embeddings,
        labels=gallery_labels,
        example_ids=np.asarray([example.example_id for example in bundle.gallery]),
        source_paths=np.asarray([str(Path(example.image)) for example in bundle.gallery]),
        split="gallery",
        checkpoint_sha256=checkpoint_digest,
    )
    upstream_r1 = _upstream_recall_at_1(
        query_embeddings, query_labels, gallery_embeddings, gallery_labels
    )
    payload = {
        "artifact_selection": "published_upstream_checkpoint",
        "checkpoint_sha256": checkpoint_digest,
        "upstream_strict_negative_rank_recall_at_1": upstream_r1,
        "registered_interval_pass": 0.917 <= upstream_r1 <= 0.921,
        **independent_query_gallery_sensitivity(
            query_embeddings, query_labels, gallery_embeddings, gallery_labels
        ),
        **partition_audit,
    }
    _write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
