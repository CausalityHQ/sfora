#!/usr/bin/env python3
"""Measure the preregistered OPIS/retrieval-error relationship."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


RNG_SEED = 20260802
NEGATIVE_SAMPLE_COUNT = 5_000_000
GRID_SIZE = 101
FAR_BOUNDS = (0.01, 0.10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_negative_distances(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Uniformly sample ordered cross-class pairs by rejection."""
    rng = np.random.default_rng(seed)
    accepted = np.empty(count, dtype=np.float32)
    filled = 0
    n = labels.shape[0]
    while filled < count:
        draw = min(max((count - filled) * 2, 4096), 2_000_000)
        left = rng.integers(0, n, size=draw)
        right = rng.integers(0, n, size=draw)
        valid = (left != right) & (labels[left] != labels[right])
        left = left[valid]
        right = right[valid]
        take = min(left.size, count - filled)
        similarities = np.einsum(
            "ij,ij->i", embeddings[left[:take]], embeddings[right[:take]]
        )
        accepted[filled : filled + take] = np.sqrt(
            np.maximum(0.0, 2.0 - 2.0 * similarities)
        )
        filled += take
    return accepted


def measure(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    negative_sample_count: int = NEGATIVE_SAMPLE_COUNT,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    vectors = np.asarray(embeddings, dtype=np.float32).copy()
    labels = np.asarray(labels)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    classes = np.unique(labels)
    class_indices = [np.flatnonzero(labels == label) for label in classes]
    if any(indices.size < 2 for indices in class_indices):
        raise ValueError("every class must contain at least two images")

    sampled_negatives = _sample_negative_distances(
        vectors, labels, count=negative_sample_count, seed=seed
    )
    d_min, d_max = np.quantile(sampled_negatives, FAR_BOUNDS)
    thresholds = np.linspace(d_min, d_max, GRID_SIZE)

    utilities = np.empty((classes.size, GRID_SIZE), dtype=np.float64)
    class_r1 = np.empty(classes.size, dtype=np.float64)
    for row, indices in enumerate(class_indices):
        members = vectors[indices]
        similarities = members @ vectors.T

        # Leave-one-out R@1 for this class.
        similarities[np.arange(indices.size), indices] = -np.inf
        nearest = np.argmax(similarities, axis=1)
        class_r1[row] = np.mean(labels[nearest] == labels[indices])

        # OPIS positives are unordered within-class pairs.
        within = members @ members.T
        upper = np.triu_indices(indices.size, 1)
        positive_distances = np.sqrt(
            np.maximum(0.0, 2.0 - 2.0 * within[upper])
        )

        # Negatives have exactly one endpoint in this class.
        negative_mask = labels[None, :] != labels[indices, None]
        negative_similarities = similarities[negative_mask]
        negative_distances = np.sqrt(
            np.maximum(0.0, 2.0 - 2.0 * negative_similarities)
        )
        positive_distances.sort()
        negative_distances.sort()
        sensitivity = np.searchsorted(
            positive_distances, thresholds, side="right"
        ) / positive_distances.size
        specificity = 1.0 - (
            np.searchsorted(negative_distances, thresholds, side="right")
            / negative_distances.size
        )
        denominator = sensitivity + specificity
        utilities[row] = np.divide(
            2.0 * sensitivity * specificity,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )

    mean_utility = utilities.mean(axis=0)
    contributions = np.mean((utilities - mean_utility) ** 2, axis=1)
    rho, p_value = spearmanr(contributions, 1.0 - class_r1)
    return {
        "class_count": int(classes.size),
        "image_count": int(labels.size),
        "far_bounds": list(FAR_BOUNDS),
        "negative_sample_count": int(negative_sample_count),
        "rng_seed": int(seed),
        "grid_size": GRID_SIZE,
        "calibration_distance_bounds": [float(d_min), float(d_max)],
        "opis": float(contributions.mean()),
        "mean_class_leave_one_out_r1": float(class_r1.mean()),
        "spearman_opis_contribution_vs_class_error": float(rho),
        "spearman_two_sided_p": float(p_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    analyzer_path = Path(__file__).resolve()
    runs = []
    for pack_path in args.packs:
        with np.load(pack_path, allow_pickle=False) as payload:
            result = measure(payload["embeddings"], payload["labels"])
        runs.append(
            {
                "pack": str(pack_path),
                "pack_sha256": _sha256(pack_path),
                **result,
            }
        )

    rhos = np.asarray(
        [run["spearman_opis_contribution_vs_class_error"] for run in runs]
    )
    opis = np.asarray([run["opis"] for run in runs])
    aggregate = {
        "analyzer_sha256": _sha256(analyzer_path),
        "runs": runs,
        "median_spearman": float(np.median(rhos)),
        "opis_coefficient_of_variation": float(
            np.std(opis, ddof=1) / np.mean(opis) if opis.size > 1 else 0.0
        ),
    }
    rendered = json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
