#!/usr/bin/env python3
"""Export authenticated UniCOM final and pre-projection Cars196 features."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

DATASET_ID = "tanganke/stanford_cars"
DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
CHECKPOINT_SHA256 = "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
BASELINE_FEATURE_SHA256 = "6ddfd7e2c9fd489dff51fa33697c62abf92a45247b52b336c0371f5482231ab3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_preprojection_batch(
    model: torch.nn.Module,
    rows: torch.Tensor,
    *,
    preprojection_dimensions: int,
    final_dimensions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact first-BN output and final teacher output for one batch."""

    feature = getattr(model, "feature", None)
    if not isinstance(feature, torch.nn.Sequential) or len(feature) != 4:
        raise ValueError("teacher feature layout differs")
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], value: torch.Tensor):
        captured.append(value.detach())

    handle = feature[1].register_forward_hook(capture)
    try:
        final = model(rows)
    finally:
        handle.remove()
    if (
        len(captured) != 1
        or captured[0].shape != (len(rows), preprojection_dimensions)
        or final.shape != (len(rows), final_dimensions)
        or not torch.isfinite(captured[0]).all()
        or not torch.isfinite(final).all()
    ):
        raise ValueError("teacher feature layout differs")
    rebuilt = feature[3](feature[2](captured[0]))
    if not torch.equal(rebuilt, final):
        raise ValueError("teacher pre-projection binding differs")
    return captured[0], final


def _require_checkout(checkout: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != UNICOM_REVISION or status:
        raise ValueError("teacher checkout authority differs")


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--execute-export", action="store_true", required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    if args.output.exists() or args.batch_size < 1:
        raise ValueError("pre-projection export authority differs")
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("teacher checkpoint authority differs")
    if sha256(args.baseline_features) != BASELINE_FEATURE_SHA256:
        raise ValueError("baseline feature authority differs")
    _require_checkout(args.checkout)

    with np.load(args.baseline_features, allow_pickle=False) as archive:
        fit_final_reference = np.ascontiguousarray(archive["fit_embeddings"], dtype=np.float32)
        fit_labels_reference = np.ascontiguousarray(archive["fit_labels"], dtype=np.int64)
        evaluation_final_reference = np.ascontiguousarray(
            archive["evaluation_embeddings"], dtype=np.float32
        )
        evaluation_labels_reference = np.ascontiguousarray(
            archive["evaluation_labels"], dtype=np.int64
        )

    from datasets import load_dataset

    source = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    records = [source[split] for split in ("train", "test")]
    labels = np.concatenate(
        [np.asarray(dataset["label"], dtype=np.int64) for dataset in records]
    )
    fit_mask = labels < 98
    if (
        labels.shape != (16_185,)
        or int(fit_mask.sum()) != 8_054
        or int((~fit_mask).sum()) != 8_131
        or not np.array_equal(labels[fit_mask], fit_labels_reference)
        or not np.array_equal(labels[~fit_mask], evaluation_labels_reference)
    ):
        raise ValueError("Cars196 row authority differs")

    package_root = (args.checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    if Path(unicom.__file__).resolve().parent != package_root / "unicom":
        raise ValueError("teacher import authority differs")
    model, transform = unicom.load("ViT-L/14@336px", download_root=str(args.checkpoint.parent))
    model = model.cuda().eval()

    preprojection_rows: list[np.ndarray] = []
    final_rows: list[np.ndarray] = []
    pending: list[torch.Tensor] = []
    with torch.inference_mode():
        for dataset in records:
            for row in dataset:
                pending.append(transform(row["image"].convert("RGB")))
                if len(pending) == args.batch_size:
                    preprojection, final = capture_preprojection_batch(
                        model,
                        torch.stack(pending).cuda(),
                        preprojection_dimensions=1_024,
                        final_dimensions=768,
                    )
                    preprojection_rows.append(preprojection.float().cpu().numpy())
                    final_rows.append(final.float().cpu().numpy())
                    pending.clear()
        if pending:
            preprojection, final = capture_preprojection_batch(
                model,
                torch.stack(pending).cuda(),
                preprojection_dimensions=1_024,
                final_dimensions=768,
            )
            preprojection_rows.append(preprojection.float().cpu().numpy())
            final_rows.append(final.float().cpu().numpy())

    preprojection = np.ascontiguousarray(np.concatenate(preprojection_rows), dtype=np.float32)
    final = np.ascontiguousarray(np.concatenate(final_rows), dtype=np.float32)
    reference = np.concatenate((fit_final_reference, evaluation_final_reference))
    cosine = np.einsum("nd,nd->n", final, reference) / (
        np.linalg.norm(final, axis=1) * np.linalg.norm(reference, axis=1)
    )
    if (
        preprojection.shape != (16_185, 1_024)
        or final.shape != (16_185, 768)
        or not np.isfinite(preprojection).all()
        or not np.isfinite(final).all()
        or float(cosine.min()) < 0.9999
    ):
        raise ValueError("teacher reproduction differs")

    metadata = {
        "schema": "sfora-unicom-preprojection-cars196-v1",
        "claim_eligible": False,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_fingerprints": {
            split: str(source[split]._fingerprint) for split in ("train", "test")
        },
        "unicom_revision": UNICOM_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "baseline_feature_sha256": BASELINE_FEATURE_SHA256,
        "preprojection_dimensions": 1_024,
        "final_dimensions": 768,
        "minimum_reproduction_cosine": float(cosine.min()),
        "mean_reproduction_cosine": float(cosine.mean()),
    }
    np.savez(
        args.output,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        fit_preprojection=preprojection[fit_mask],
        fit_final=final[fit_mask],
        fit_labels=labels[fit_mask],
        evaluation_preprojection=preprojection[~fit_mask],
        evaluation_final=final[~fit_mask],
        evaluation_labels=labels[~fit_mask],
    )
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), **metadata}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
