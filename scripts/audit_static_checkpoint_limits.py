#!/usr/bin/env python3
"""Audit margin sufficiency and cross-identity near-duplicates on a final pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.fft import dctn
from scipy.optimize import minimize
from scipy.stats import chi2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.all(np.isfinite(rows)):
        raise ValueError("embeddings must be finite and nonzero")
    return rows / norms


def compute_observables(
    embeddings: np.ndarray,
    labels: np.ndarray,
    proxies: np.ndarray,
    proxy_labels: np.ndarray,
    *,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    """Independently compute margins, agreement, errors, and foreign neighbours."""
    x = _normalize(np.asarray(embeddings, dtype=np.float64))
    y = np.asarray(labels, dtype=np.int64)
    p = _normalize(np.asarray(proxies, dtype=np.float64))
    pl = np.asarray(proxy_labels, dtype=np.int64)
    proxy_position = {int(label): index for index, label in enumerate(pl)}
    if set(proxy_position) != set(map(int, np.unique(y))):
        raise ValueError("proxy and training label sets differ")

    n = len(x)
    proxy_margin = np.empty(n, dtype=np.float64)
    image_margin = np.empty(n, dtype=np.float64)
    agreement = np.empty(n, dtype=bool)
    error = np.empty(n, dtype=bool)
    foreign_index = np.empty(n, dtype=np.int64)
    same_similarity = np.empty(n, dtype=np.float64)
    foreign_similarity = np.empty(n, dtype=np.float64)
    own_proxy_position = np.asarray([proxy_position[int(label)] for label in y])

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        rows = np.arange(start, stop)
        similarity = x[start:stop] @ x.T
        similarity[np.arange(stop - start), rows] = -np.inf
        nearest = similarity.argmax(axis=1)
        error[start:stop] = y[nearest] != y[start:stop]

        same = np.where(y[start:stop, None] == y[None, :], similarity, -np.inf)
        foreign = np.where(y[start:stop, None] != y[None, :], similarity, -np.inf)
        same_similarity[start:stop] = same.max(axis=1)
        foreign_index[start:stop] = foreign.argmax(axis=1)
        foreign_similarity[start:stop] = foreign.max(axis=1)
        image_margin[start:stop] = same_similarity[start:stop] - foreign_similarity[start:stop]

        proxy_similarity = x[start:stop] @ p.T
        own = proxy_similarity[np.arange(stop - start), own_proxy_position[start:stop]]
        proxy_similarity[pl[None, :] == y[start:stop, None]] = -np.inf
        nearest_proxy = proxy_similarity.argmax(axis=1)
        proxy_margin[start:stop] = own - proxy_similarity[np.arange(stop - start), nearest_proxy]
        agreement[start:stop] = y[foreign_index[start:stop]] == pl[nearest_proxy]

    return {
        "proxy_margin": proxy_margin,
        "image_margin": image_margin,
        "agreement": agreement,
        "error": error,
        "foreign_index": foreign_index,
        "same_similarity": same_similarity,
        "foreign_similarity": foreign_similarity,
    }


def _fit_logistic(design: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(design, dtype=np.float64)
    target = np.asarray(outcome, dtype=np.float64)

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        linear = matrix @ coefficients
        loss = np.logaddexp(0.0, linear).sum() - target @ linear
        probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
        gradient = matrix.T @ (probability - target)
        return float(loss), gradient

    initial = np.zeros(matrix.shape[1], dtype=np.float64)
    initial[0] = np.log((target.sum() + 0.5) / (len(target) - target.sum() + 0.5))
    result = minimize(objective, initial, method="BFGS", jac=True, options={"maxiter": 2000})
    if not result.success and np.linalg.norm(result.jac, ord=np.inf) > 1e-5:
        raise RuntimeError(f"logistic fit failed: {result.message}; gradient={result.jac}")
    return {
        "coefficients": result.x,
        "log_likelihood": -float(result.fun),
        "gradient_inf_norm": float(np.linalg.norm(result.jac, ord=np.inf)),
    }


def margin_sufficiency(observables: dict[str, np.ndarray]) -> dict[str, Any]:
    margins = []
    for name in ("proxy_margin", "image_margin"):
        values = observables[name]
        margins.append((values - values.mean()) / values.std(ddof=0))
    proxy, image = margins
    base = np.column_stack(
        [np.ones(len(proxy)), proxy, image, proxy * proxy, image * image, proxy * image]
    )
    extended = np.column_stack([base, observables["agreement"].astype(np.float64)])
    base_fit = _fit_logistic(base, observables["error"])
    extended_fit = _fit_logistic(extended, observables["error"])
    statistic = max(0.0, 2.0 * (extended_fit["log_likelihood"] - base_fit["log_likelihood"]))
    return {
        "events": int(observables["error"].sum()),
        "rows": len(proxy),
        "base_log_likelihood": base_fit["log_likelihood"],
        "extended_log_likelihood": extended_fit["log_likelihood"],
        "likelihood_ratio": statistic,
        "p_value_chi2_df1": float(chi2.sf(statistic, 1)),
        "agreement_coefficient": float(extended_fit["coefficients"][-1]),
        "base_coefficients": base_fit["coefficients"].tolist(),
        "extended_coefficients": extended_fit["coefficients"].tolist(),
        "base_gradient_inf_norm": base_fit["gradient_inf_norm"],
        "extended_gradient_inf_norm": extended_fit["gradient_inf_norm"],
    }


def perceptual_hash(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        coefficients = dctn(np.asarray(grayscale, dtype=np.float64), type=2, norm="ortho")[:8, :8]
    flat = coefficients.ravel()
    return flat > np.median(flat[1:])


def grayscale_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float64
        ).ravel()
    centered = pixels - pixels.mean()
    norm = np.linalg.norm(centered)
    return centered / norm if norm > 0 else centered


def near_duplicate_audit(
    observables: dict[str, np.ndarray],
    labels: np.ndarray,
    source_paths: np.ndarray,
) -> dict[str, Any]:
    foreign = observables["foreign_index"]
    threshold = float(np.median(observables["same_similarity"]))
    pairs: list[dict[str, Any]] = []
    flagged_error_rows: set[int] = set()
    considered = 0
    for left, right_value in enumerate(foreign):
        right = int(right_value)
        if left >= right or int(foreign[right]) != left:
            continue
        cosine = float(observables["foreign_similarity"][left])
        if cosine < threshold:
            continue
        considered += 1
        left_path = Path(str(source_paths[left]))
        right_path = Path(str(source_paths[right]))
        hamming = int(np.count_nonzero(perceptual_hash(left_path) != perceptual_hash(right_path)))
        left_pixels = grayscale_pixels(left_path)
        right_pixels = grayscale_pixels(right_path)
        correlation = float(left_pixels @ right_pixels)
        flagged = hamming <= 4 or correlation >= 0.98
        if not flagged:
            continue
        for row in (left, right):
            if observables["error"][row] and int(foreign[row]) in {left, right}:
                flagged_error_rows.add(row)
        pairs.append(
            {
                "left_index": left,
                "right_index": right,
                "left_label": int(labels[left]),
                "right_label": int(labels[right]),
                "left_path": str(left_path),
                "right_path": str(right_path),
                "cosine": cosine,
                "phash_hamming": hamming,
                "grayscale_correlation": correlation,
            }
        )
    return {
        "nearest_same_class_cosine_median": threshold,
        "mutual_foreign_pairs_above_threshold": considered,
        "flagged_pairs": pairs,
        "flagged_pair_count": len(pairs),
        "flagged_loo_error_rows": sorted(flagged_error_rows),
        "flagged_loo_error_count": len(flagged_error_rows),
        "material_threshold_errors": 13,
        "material": len(flagged_error_rows) >= 13,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pack-sha256")
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args()

    pack_digest = sha256(args.pack)
    checkpoint_digest = sha256(args.checkpoint)
    if args.expected_pack_sha256 and pack_digest != args.expected_pack_sha256:
        raise ValueError("pack digest does not match the registered artifact")
    if args.expected_checkpoint_sha256 and checkpoint_digest != args.expected_checkpoint_sha256:
        raise ValueError("checkpoint digest does not match the registered artifact")

    with np.load(args.pack, allow_pickle=False) as pack:
        embeddings = pack["embeddings"]
        labels = pack["labels"]
        source_paths = pack["source_paths"]
        artifact_selection = str(pack["artifact_selection"])
        bound_checkpoint = str(pack["checkpoint_sha256"])
    if artifact_selection != "final_training_state" or bound_checkpoint != checkpoint_digest:
        raise ValueError("pack is not bound to the supplied final-state checkpoint")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_selection") != "final_training_state":
        raise ValueError("checkpoint is not the final training state")
    arch = checkpoint.get("arch", {})
    if arch.get("backbone_name") != "bn_inception" or arch.get("embedding_dimensions") != 512:
        raise ValueError(f"unexpected corrected In-Shop architecture: {arch}")
    state = checkpoint["state_dict"]
    observables = compute_observables(
        embeddings,
        labels,
        state["metric_proxies"].detach().cpu().numpy(),
        state["metric_proxy_labels"].detach().cpu().numpy(),
    )
    payload = {
        "pack_sha256": pack_digest,
        "checkpoint_sha256": checkpoint_digest,
        "artifact_selection": artifact_selection,
        "architecture": arch,
        "descriptive": {
            "loo_error": float(observables["error"].mean()),
            "agreement": float(observables["agreement"].mean()),
            "error_given_agreement": float(
                observables["error"][observables["agreement"]].mean()
            ),
            "error_given_disagreement": float(
                observables["error"][~observables["agreement"]].mean()
            ),
        },
        "margin_sufficiency": margin_sufficiency(observables),
        "near_duplicates": near_duplicate_audit(observables, labels, source_paths),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
