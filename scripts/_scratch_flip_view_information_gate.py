#!/usr/bin/env python3
"""Fit-only fail-fast gate for information added by horizontal-flip views."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torchvision.datasets import Food101, OxfordIIITPet

from sfora.compact_metric import _compact_metric_lexicographic_topk
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def food_fit(root: Path, fit_archive: Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    dataset = Food101(root, split="test", download=False)
    labels = np.asarray(dataset._labels, dtype=np.int64)
    values = sorted(
        (str(value) for value in np.unique(labels)),
        key=lambda value: hashlib.sha256(
            f"food-whitening-selector-v1:{value}".encode()
        ).digest(),
    )
    selected = set(values[:50])
    mask = np.asarray([str(value) in selected for value in labels])
    mapping = {value: index for index, value in enumerate(values)}
    mapped = np.asarray([mapping[str(value)] for value in labels[mask]], dtype=np.int64)
    with np.load(fit_archive, allow_pickle=False) as archive:
        reference = np.ascontiguousarray(archive["embeddings"], dtype=np.float32)
        expected_labels = np.ascontiguousarray(archive["labels"], dtype=np.int64)
    if not np.array_equal(mapped, expected_labels):
        raise ValueError("Food fit-row authority differs")
    paths = [path for path, keep in zip(dataset._image_files, mask, strict=True) if keep]
    return paths, mapped, reference


def pet_fit(root: Path, fit_archive: Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    dataset = OxfordIIITPet(root, split="trainval", target_types="category", download=False)
    labels = np.asarray(dataset._labels, dtype=np.int64)
    with np.load(fit_archive, allow_pickle=False) as archive:
        reference = np.ascontiguousarray(archive["embeddings"], dtype=np.float32)
        expected_labels = np.ascontiguousarray(archive["labels"], dtype=np.int64)
    mask = np.isin(labels, np.unique(expected_labels))
    if not np.array_equal(labels[mask], expected_labels):
        raise ValueError("Pet fit-row authority differs")
    paths = [path for path, keep in zip(dataset._images, mask, strict=True) if keep]
    return paths, expected_labels, reference


def load_model(checkout: Path, checkpoint: Path) -> tuple[torch.nn.Module, object]:
    package_root = (checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    model, transform = unicom.load("ViT-L/14@336px", download_root=str(checkpoint.parent))
    return model.cuda().eval(), transform


def encode_views(
    model: torch.nn.Module,
    transform: object,
    paths: list[Path],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    identity: np.ndarray | None = None
    flipped: np.ndarray | None = None
    for start in range(0, len(paths), batch_size):
        ordinary = []
        mirrors = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                ordinary.append(transform(image))
                mirrors.append(transform(ImageOps.mirror(image)))
        with torch.inference_mode():
            ordinary_values = F.normalize(model(torch.stack(ordinary).cuda()).float(), dim=1)
        torch.cuda.synchronize()
        ordinary_host = ordinary_values.cpu().numpy().copy()
        with torch.inference_mode():
            mirror_values = F.normalize(model(torch.stack(mirrors).cuda()).float(), dim=1)
        torch.cuda.synchronize()
        mirror_host = mirror_values.cpu().numpy().copy()
        if identity is None or flipped is None:
            identity = np.empty((len(paths), ordinary_host.shape[1]), dtype=np.float32)
            flipped = np.empty_like(identity)
        stop = start + len(ordinary_host)
        identity[start:stop] = ordinary_host
        flipped[start:stop] = mirror_host
    if identity is None or flipped is None:
        raise ValueError("flip-view input is empty")
    return torch.from_numpy(identity), torch.from_numpy(flipped)


def top1_correct(values: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    normalized = F.normalize(values.float(), dim=1).cuda()
    labels_cpu = labels.cpu()
    rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(normalized), 256):
            similarities = normalized[start : start + 256] @ normalized.T
            local = torch.arange(start, min(start + 256, len(normalized)), device="cuda")
            similarities[torch.arange(len(local), device="cuda"), local] = -torch.inf
            rows.append(_compact_metric_lexicographic_topk(similarities, 1).cpu()[:, 0])
    neighbours = torch.cat(rows)
    return (labels_cpu[neighbours] == labels_cpu).numpy()


def clustered_lower_bound(
    datasets: dict[str, dict[str, object]], *, draws: int = 10_000
) -> float:
    rng = np.random.default_rng(1701)
    deltas = []
    clusters: dict[str, list[np.ndarray]] = {}
    for name, row in datasets.items():
        labels = np.asarray(row["labels"])
        difference = np.asarray(row["average_correct"], dtype=np.int8) - np.asarray(
            row["identity_correct"], dtype=np.int8
        )
        clusters[name] = [difference[labels == label] for label in np.unique(labels)]
    for _ in range(draws):
        total = 0
        count = 0
        for rows in clusters.values():
            for index in rng.integers(0, len(rows), size=len(rows)):
                selected = rows[int(index)]
                total += int(selected.sum())
                count += len(selected)
        deltas.append(total / count)
    return float(np.quantile(np.asarray(deltas), 0.025))


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--food-fit", type=Path, required=True)
    parser.add_argument("--pet-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-fit-only-gate", action="store_true", required=True)
    args = parser.parse_args()
    preregistration = json.loads(args.preregistration.read_text())
    if (
        args.output.exists()
        or sha256(Path(__file__)) != args.script_sha256
        or sha256(args.preregistration) != args.preregistration_sha256
        or preregistration["schema"] != "sfora-flip-view-information-preregistration-v1"
        or preregistration["script_sha256"] != args.script_sha256
        or preregistration["source_commit"] != args.source_commit
        or sha256(args.checkpoint) != preregistration["checkpoint_sha256"]
        or sha256(args.food_fit) != preregistration["fit_archives"]["food101"]
        or sha256(args.pet_fit) != preregistration["fit_archives"]["pet"]
    ):
        raise ValueError("flip-view gate authority differs")
    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    started = time.monotonic()
    model, transform = load_model(args.unicom_checkout, args.checkpoint)
    inputs = {
        "food101": food_fit(args.raw_root, args.food_fit),
        "pet": pet_fit(args.raw_root, args.pet_fit),
    }
    internal: dict[str, dict[str, object]] = {}
    reported: dict[str, object] = {}
    for name, (paths, labels_array, reference) in inputs.items():
        identity, flipped = encode_views(model, transform, paths, batch_size=64)
        reference_values = F.normalize(torch.from_numpy(reference).float(), dim=1)
        identity_delta = float(torch.max(torch.abs(identity - reference_values)).item())
        if identity_delta > 2e-4:
            raise ValueError(f"{name} identity replay differs: {identity_delta}")
        averaged = F.normalize(identity + flipped, dim=1)
        labels = torch.from_numpy(labels_array)
        identity_correct = top1_correct(identity, labels)
        average_correct = top1_correct(averaged, labels)
        internal[name] = {
            "labels": labels_array,
            "identity_correct": identity_correct,
            "average_correct": average_correct,
        }
        reported[name] = {
            "rows": len(paths),
            "classes": len(np.unique(labels_array)),
            "mean_view_cosine": float(torch.mean(torch.sum(identity * flipped, dim=1)).item()),
            "identity_recall_at_1": float(identity_correct.mean()),
            "averaged_recall_at_1": float(average_correct.mean()),
            "recall_at_1_delta": float(average_correct.mean() - identity_correct.mean()),
            "identity_replay_max_abs": identity_delta,
        }
    total_rows = sum(int(row["rows"]) for row in reported.values())
    pooled_delta = math.fsum(
        int(row["rows"]) * float(row["recall_at_1_delta"]) for row in reported.values()
    ) / total_rows
    lower = clustered_lower_bound(internal)
    passed = (
        max(float(row["mean_view_cosine"]) for row in reported.values()) < 0.98
        and pooled_delta >= 0.003
        and lower > 0.0
    )
    result = {
        "schema": "sfora-flip-view-information-gate-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "script_sha256": args.script_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "datasets": reported,
        "decision": {
            "passed": passed,
            "pooled_recall_at_1_delta": pooled_delta,
            "class_clustered_bootstrap_lower_95": lower,
            "next": "expand exact view to six datasets" if passed else "close horizontal-flip view",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
