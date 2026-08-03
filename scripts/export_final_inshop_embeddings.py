#!/usr/bin/env python3
"""Independently reconstruct final-state In-Shop query/gallery retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from sfora.data import load_image_retrieval_bundle
from sfora.image_benchmark import image_query_gallery_retrieval_score
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("artifact_selection") != "final_training_state":
        raise ValueError("checkpoint is not explicitly labeled as a final training state")

    bundle = load_image_retrieval_bundle(
        dataset_name="inshop",
        dataset_root=args.dataset_root,
        seed=config.seed,
    )
    counts = (
        len(bundle.train),
        len({example.label for example in bundle.train}),
        len(bundle.query),
        len(bundle.gallery or []),
    )
    if counts != (25_882, 3_997, 14_218, 12_612):
        raise ValueError(f"unexpected official In-Shop counts: {counts}")
    if bundle.gallery is None:
        raise ValueError("In-Shop bundle lacks its official gallery")

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
    query_embeddings, query_labels = _encode_model(
        model, loader(bundle.query), device, torch
    )
    gallery_embeddings, gallery_labels = _encode_model(
        model, loader(bundle.gallery), device, torch
    )
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
    retrieval = image_query_gallery_retrieval_score(
        query_embeddings,
        query_labels,
        gallery_embeddings,
        gallery_labels,
    )
    payload = {
        "artifact_selection": "final_training_state",
        "checkpoint_sha256": checkpoint_digest,
        "report_sha256": report_digest,
        "resolved_training_steps": resolved_steps,
        "retrieval": asdict(retrieval),
    }
    args.retrieval_output.parent.mkdir(parents=True, exist_ok=True)
    args.retrieval_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
