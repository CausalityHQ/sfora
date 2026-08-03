#!/usr/bin/env python3
"""Independently reconstruct final-state In-Shop query/gallery retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from sfora.data import load_image_retrieval_bundle
from sfora.image_end_to_end import (
    ImageEndToEndConfig,
    _default_transform_factory,
    _encode_model,
    _resolve_training_schedule,
    _TorchImageDataset,
    _torchvision_model_factory,
)

EXPECTED_INSHOP_PARTITION = {
    "train": (25_882, 3_997),
    "query": (14_218, 3_985),
    "gallery": (12_612, 3_985),
}
EXPECTED_INSHOP_CONTENT_PROFILE = {
    "train": {
        "duplicate_groups": 19,
        "duplicate_rows": 40,
        "cross_identity_groups": 0,
        "cross_identity_rows": 0,
    },
    "query": {
        "duplicate_groups": 7,
        "duplicate_rows": 14,
        "cross_identity_groups": 1,
        "cross_identity_rows": 2,
    },
    "gallery": {
        "duplicate_groups": 7,
        "duplicate_rows": 14,
        "cross_identity_groups": 1,
        "cross_identity_rows": 2,
    },
}
EXPECTED_INSHOP_CONTENT_OVERLAP = {
    "train_query": {"groups": 0, "rows": 0, "cross_identity_groups": 0},
    "train_gallery": {"groups": 0, "rows": 0, "cross_identity_groups": 0},
    "query_gallery": {"groups": 19, "rows": 38, "cross_identity_groups": 3},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_query_gallery_recall_at_1(
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    chunk_size: int = 512,
) -> float:
    """Recompute In-Shop R@1 without calling the benchmark scorer."""
    query = np.asarray(query_embeddings, dtype=np.float64)
    gallery = np.asarray(gallery_embeddings, dtype=np.float64)
    query_norms = np.linalg.norm(query, axis=1, keepdims=True)
    gallery_norms = np.linalg.norm(gallery, axis=1, keepdims=True)
    if np.any(query_norms <= 0) or np.any(gallery_norms <= 0):
        raise ValueError("independent retrieval received a zero-norm embedding")
    query = query / query_norms
    gallery = gallery / gallery_norms
    correct = 0
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        nearest = np.argmax(query[start:stop] @ gallery.T, axis=1)
        correct += int(np.sum(gallery_labels[nearest] == query_labels[start:stop]))
    return correct / len(query)


def independent_query_gallery_sensitivity(
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    chunk_size: int = 512,
) -> dict[str, float | int]:
    """Compare the declared scorer with cosine and exact-tie expected R@1."""
    query = np.asarray(query_embeddings, dtype=np.float64)
    gallery = np.asarray(gallery_embeddings, dtype=np.float64)
    query_labels = np.asarray(query_labels, dtype=np.int64)
    gallery_labels = np.asarray(gallery_labels, dtype=np.int64)
    query_sq_norms = np.sum(query * query, axis=1)
    gallery_sq_norms = np.sum(gallery * gallery, axis=1)
    query_norms = np.sqrt(query_sq_norms)
    gallery_norms = np.sqrt(gallery_sq_norms)
    if np.any(query_norms <= 0) or np.any(gallery_norms <= 0):
        raise ValueError("independent retrieval received a zero-norm embedding")

    euclidean_correct = 0
    cosine_correct = 0
    tie_expected_correct = 0.0
    multiway_tie_queries = 0
    mixed_identity_tie_queries = 0
    gallery_label_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(gallery_labels, return_counts=True), strict=True)
    }
    max_relevant_count = max(
        (gallery_label_counts.get(int(label), 0) for label in query_labels), default=0
    )
    canonical_top_k = min(len(gallery), max(30, max_relevant_count))
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        dot = query[start:stop] @ gallery.T
        distances = query_sq_norms[start:stop, None] + gallery_sq_norms[None, :] - 2.0 * dot
        if canonical_top_k < len(gallery):
            top_indices = np.argpartition(distances, kth=canonical_top_k - 1, axis=1)[
                :, :canonical_top_k
            ]
            top_distances = np.take_along_axis(distances, top_indices, axis=1)
            top_order = np.argsort(top_distances, axis=1, kind="stable")
            euclidean_nearest = np.take_along_axis(top_indices, top_order, axis=1)[:, 0]
        else:
            euclidean_nearest = np.argsort(distances, axis=1, kind="stable")[:, 0]
        cosine = dot / (query_norms[start:stop, None] * gallery_norms[None, :])
        cosine_nearest = np.argmax(cosine, axis=1)
        chunk_labels = query_labels[start:stop]
        euclidean_correct += int(np.sum(gallery_labels[euclidean_nearest] == chunk_labels))
        cosine_correct += int(np.sum(gallery_labels[cosine_nearest] == chunk_labels))

        minima = distances.min(axis=1, keepdims=True)
        ties = distances == minima
        tie_counts = ties.sum(axis=1)
        correct_ties = (ties & (gallery_labels[None, :] == chunk_labels[:, None])).sum(axis=1)
        tie_expected_correct += float(np.sum(correct_ties / tie_counts))
        multiway_tie_queries += int(np.sum(tie_counts > 1))
        mixed_identity_tie_queries += int(np.sum((correct_ties > 0) & (correct_ties < tie_counts)))

    count = len(query)
    return {
        "canonical_float64_euclidean_recall_at_1": euclidean_correct / count,
        "float64_cosine_recall_at_1": cosine_correct / count,
        "exact_tie_expected_recall_at_1": tie_expected_correct / count,
        "multiway_nearest_tie_queries": multiway_tie_queries,
        "mixed_identity_nearest_tie_queries": mixed_identity_tie_queries,
    }


def _content_partition_audit(splits: dict[str, list[Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, list[tuple[int, str]]]] = {}
    manifests: dict[str, str] = {}
    profiles: dict[str, dict[str, int]] = {}
    for split, examples in splits.items():
        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        manifest = hashlib.sha256()
        for example in examples:
            path = Path(example.image).resolve()
            if not path.is_file():
                raise ValueError(f"In-Shop {split} references missing source path: {path}")
            digest = sha256(path)
            groups[digest].append((int(example.label), str(path)))
            manifest.update(str(path).encode("utf-8"))
            manifest.update(b"\0")
            manifest.update(digest.encode("ascii"))
            manifest.update(b"\n")
        duplicates = [rows for rows in groups.values() if len(rows) > 1]
        cross_identity = [rows for rows in duplicates if len({row[0] for row in rows}) > 1]
        profile = {
            "duplicate_groups": len(duplicates),
            "duplicate_rows": sum(len(rows) for rows in duplicates),
            "cross_identity_groups": len(cross_identity),
            "cross_identity_rows": sum(len(rows) for rows in cross_identity),
        }
        if profile != EXPECTED_INSHOP_CONTENT_PROFILE[split]:
            raise ValueError(
                f"unexpected In-Shop {split} source-content profile: "
                f"{profile} != {EXPECTED_INSHOP_CONTENT_PROFILE[split]}"
            )
        by_split[split] = groups
        manifests[split] = manifest.hexdigest()
        profiles[split] = profile

    overlaps: dict[str, dict[str, int]] = {}
    for left, right in (("train", "query"), ("train", "gallery"), ("query", "gallery")):
        shared = set(by_split[left]) & set(by_split[right])
        rows = [by_split[left][digest] + by_split[right][digest] for digest in shared]
        overlap = {
            "groups": len(shared),
            "rows": sum(len(group) for group in rows),
            "cross_identity_groups": sum(len({row[0] for row in group}) > 1 for group in rows),
        }
        key = f"{left}_{right}"
        if overlap != EXPECTED_INSHOP_CONTENT_OVERLAP[key]:
            raise ValueError(
                f"unexpected In-Shop {left}/{right} source-content overlap: "
                f"{overlap} != {EXPECTED_INSHOP_CONTENT_OVERLAP[key]}"
            )
        overlaps[key] = overlap
    return {
        "content_profiles": profiles,
        "content_manifest_sha256": manifests,
        "content_overlaps": overlaps,
    }


def verify_official_partition(bundle: Any) -> dict[str, Any]:
    """Verify all within- and cross-split identity/row invariants."""
    if bundle.gallery is None:
        raise ValueError("In-Shop bundle lacks its official gallery")
    splits = {
        "train": bundle.train,
        "query": bundle.query,
        "gallery": bundle.gallery,
    }
    identities: dict[str, set[int]] = {}
    ids: dict[str, set[str]] = {}
    paths: dict[str, set[str]] = {}
    for split, examples in splits.items():
        identities[split] = {int(example.label) for example in examples}
        ids[split] = {str(example.example_id) for example in examples}
        paths[split] = {str(Path(example.image).resolve()) for example in examples}
        observed = (len(examples), len(identities[split]))
        if observed != EXPECTED_INSHOP_PARTITION[split]:
            raise ValueError(f"unexpected official In-Shop {split} counts: {observed}")
        if len(ids[split]) != len(examples):
            raise ValueError(f"duplicate In-Shop {split} example IDs")
        if len(paths[split]) != len(examples):
            raise ValueError(f"duplicate In-Shop {split} source paths")

    if identities["query"] != identities["gallery"]:
        raise ValueError("In-Shop query and gallery identity sets differ")
    if identities["train"] & identities["query"]:
        raise ValueError("In-Shop train and test identities overlap")
    for left, right in (("train", "query"), ("train", "gallery"), ("query", "gallery")):
        if ids[left] & ids[right]:
            raise ValueError(f"In-Shop {left}/{right} example IDs overlap")
        if paths[left] & paths[right]:
            raise ValueError(f"In-Shop {left}/{right} source paths overlap")
    return {
        "train_query_identity_overlap": 0,
        "train_gallery_identity_overlap": 0,
        "train_query_example_id_overlap": 0,
        "train_gallery_example_id_overlap": 0,
        "query_gallery_example_id_overlap": 0,
        "train_query_source_path_overlap": 0,
        "train_gallery_source_path_overlap": 0,
        "query_gallery_source_path_overlap": 0,
        "query_gallery_identity_sets_equal": True,
        **_content_partition_audit(splits),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_embeddings(
    path: Path,
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    example_ids: np.ndarray,
    source_paths: np.ndarray,
    split: str,
    checkpoint_sha256: str,
    report_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=example_ids,
        source_paths=source_paths,
        artifact_selection=np.asarray("final_training_state"),
        split=np.asarray(split),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        report_sha256=np.asarray(report_sha256),
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--query-output", type=Path, required=True)
    parser.add_argument("--gallery-output", type=Path, required=True)
    parser.add_argument("--retrieval-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    report = json.loads(args.report.read_text(encoding="utf-8"))
    config = ImageEndToEndConfig.model_validate(report["config"])
    if config.dataset_name != "inshop" or config.objectives != ("proxy_anchor",):
        raise ValueError("exporter requires one-objective In-Shop Proxy Anchor")
    if config.checkpoint_selection_interval != 0:
        raise ValueError("final-state exporter refuses checkpoint-selected training")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("artifact_selection") != "final_training_state":
        raise ValueError("checkpoint is not explicitly labeled as a final training state")
    expected_training_config = config.model_dump(mode="json")
    if checkpoint.get("training_config") != expected_training_config:
        raise ValueError("checkpoint training_config does not match the report config")

    bundle = load_image_retrieval_bundle(
        dataset_name="inshop",
        dataset_root=args.dataset_root,
        seed=config.seed,
    )
    partition_audit = verify_official_partition(bundle)
    assert bundle.gallery is not None

    resolved_steps, _, _ = _resolve_training_schedule(
        config,
        optimization_example_count=len(bundle.train),
        optimization_labels=[example.label for example in bundle.train],
    )
    if checkpoint.get("training_step") != resolved_steps:
        raise ValueError(
            f"checkpoint step {checkpoint.get('training_step')} != resolved {resolved_steps}"
        )

    transform = _default_transform_factory(config, False)

    def loader(examples: list[Any]) -> Any:
        return DataLoader(
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
    query_embeddings, query_labels = _encode_model(model, loader(bundle.query), device, torch)
    gallery_embeddings, gallery_labels = _encode_model(model, loader(bundle.gallery), device, torch)
    expected_query_labels = np.asarray([example.label for example in bundle.query])
    expected_gallery_labels = np.asarray([example.label for example in bundle.gallery])
    if not np.array_equal(query_labels, expected_query_labels):
        raise ValueError("encoded query labels differ from official query order")
    if not np.array_equal(gallery_labels, expected_gallery_labels):
        raise ValueError("encoded gallery labels differ from official gallery order")
    for split, embeddings in (
        ("query", query_embeddings),
        ("gallery", gallery_embeddings),
    ):
        if not np.isfinite(embeddings).all():
            raise ValueError(f"non-finite In-Shop {split} embeddings")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-5, rtol=2e-5):
            raise ValueError(f"In-Shop {split} embeddings are not unit normalized")
    checkpoint_digest = sha256(args.checkpoint)
    report_digest = sha256(args.report)
    save_embeddings(
        args.query_output,
        embeddings=query_embeddings,
        labels=query_labels,
        example_ids=np.asarray([example.example_id for example in bundle.query]),
        source_paths=np.asarray([str(Path(example.image)) for example in bundle.query]),
        split="query",
        checkpoint_sha256=checkpoint_digest,
        report_sha256=report_digest,
    )
    save_embeddings(
        args.gallery_output,
        embeddings=gallery_embeddings,
        labels=gallery_labels,
        example_ids=np.asarray([example.example_id for example in bundle.gallery]),
        source_paths=np.asarray([str(Path(example.image)) for example in bundle.gallery]),
        split="gallery",
        checkpoint_sha256=checkpoint_digest,
        report_sha256=report_digest,
    )
    sensitivity = independent_query_gallery_sensitivity(
        query_embeddings,
        query_labels,
        gallery_embeddings,
        gallery_labels,
    )
    method = report["methods"][next(iter(report["methods"]))]
    reported_final_r1 = float(method["recall_at_1"])
    canonical_r1 = float(sensitivity["canonical_float64_euclidean_recall_at_1"])
    if canonical_r1 != reported_final_r1:
        raise ValueError(
            f"independent canonical R@1 {canonical_r1} != report final R@1 {reported_final_r1}"
        )
    payload = {
        "artifact_selection": "final_training_state",
        "checkpoint_sha256": checkpoint_digest,
        "report_sha256": report_digest,
        "resolved_training_steps": resolved_steps,
        "reported_final_recall_at_1": reported_final_r1,
        "independent_recall_at_1": canonical_r1,
        **sensitivity,
        **partition_audit,
    }
    write_json_atomic(args.retrieval_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
