#!/usr/bin/env python3
"""Throwaway native MET noise-shaped product-quantization probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from sfora.product_quantization import (
    ProductQuantizer,
    balanced_product_quantization_spec,
    fit_product_quantizer,
)

DIMENSIONS = 768
BLOCKS = 128
BLOCK_WIDTH = 6
CODEBOOK_SIZE = 16
PACKED_BYTES = 64
SEEDS = (50, 51, 52)
MAXIMUM_ITERATIONS = 20
MAXIMUM_ROUNDS = 10
THRESHOLD = 0.2
BOOTSTRAPS = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def pseudoquery_mask(labels: np.ndarray, paths: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(labels), dtype=np.bool_)
    for label in np.unique(labels):
        indexes = np.flatnonzero(labels == label)
        if len(indexes) >= 2:
            chosen = indexes[np.argsort(paths[indexes], kind="stable")[0]]
            mask[chosen] = True
    return mask


def quality(
    rankings: torch.Tensor, query_labels: np.ndarray, gallery_labels: np.ndarray
) -> dict[str, object]:
    indexes = rankings.detach().cpu().numpy()
    if indexes.shape != (len(query_labels), 5):
        raise ValueError("native PQ ranking shape differs")
    counts = Counter(int(label) for label in gallery_labels.tolist())
    matches = gallery_labels[indexes] == query_labels[:, None]
    mmp: list[float] = []
    r1: list[float] = []
    for row, label in zip(matches, query_labels, strict=True):
        relevant = min(counts[int(label)], 5)
        if relevant < 1:
            raise ValueError("native PQ query lacks gallery positive")
        mmp.append(float(row.sum()) / relevant)
        r1.append(float(row[0]))
    return {
        "mmp_at_5": float(np.mean(mmp, dtype=np.float64)),
        "recall_at_1": float(np.mean(r1, dtype=np.float64)),
        "per_query_mmp_at_5": mmp,
        "per_query_r1": r1,
    }


def score_packed(
    quantizer: ProductQuantizer,
    packed: torch.Tensor,
    queries: torch.Tensor,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    metric: Literal["dot", "squared_l2", "normalized_squared_l2"],
) -> tuple[dict[str, object], float]:
    started = time.monotonic()
    raw = unpack_nibbles(packed)
    decoded = quantizer.hard_decode(raw)
    rankings = topk_decoded(queries, decoded, width=5, metric=metric)
    if queries.device.type == "cuda":
        torch.cuda.synchronize(queries.device)
    elapsed = time.monotonic() - started
    return quality(rankings, query_labels, gallery_labels), elapsed


def reconstruction_diagnostics(
    values: torch.Tensor,
    quantizer: ProductQuantizer,
    packed: torch.Tensor,
) -> dict[str, float]:
    raw = unpack_nibbles(packed)
    reconstructed = quantizer.hard_decode(raw)
    residual = values - reconstructed
    parallel = (values * residual).sum(dim=1)
    norms = torch.linalg.vector_norm(reconstructed.double(), dim=1)
    return {
        "mean_squared_error": float(residual.double().square().sum(dim=1).mean()),
        "mean_parallel_squared_error": float(parallel.double().square().mean()),
        "reconstruction_norm_mean": float(norms.mean()),
        "reconstruction_norm_std": float(norms.std(unbiased=False)),
    }


def bootstrap_interval(values: np.ndarray, *, seed: int) -> list[float]:
    if values.shape != (3_050,) or not np.isfinite(values).all():
        raise ValueError("native PQ bootstrap authority differs")
    generator = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(BOOTSTRAPS, dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 100):
        stop = min(start + 100, BOOTSTRAPS)
        indexes = generator.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indexes].mean(axis=1, dtype=np.float64)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def recall_hit_delta(candidate: list[float], baseline: list[float]) -> int:
    """Return an exact integer hit-count delta for binary Recall@1 outcomes."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if (
        candidate_array.ndim != 1
        or candidate_array.shape != baseline_array.shape
        or candidate_array.size < 1
        or not np.isin(candidate_array, (0.0, 1.0)).all()
        or not np.isin(baseline_array, (0.0, 1.0)).all()
    ):
        raise ValueError("Recall@1 hit authority differs")
    return int(candidate_array.sum(dtype=np.int64) - baseline_array.sum(dtype=np.int64))


def meets_lower_bound(value: float, bound: float) -> bool:
    """Compare a floating metric to a frozen inclusive bound with roundoff tolerance."""

    if not math.isfinite(value) or not math.isfinite(bound):
        raise ValueError("metric threshold authority differs")
    return value >= bound - 1e-12


def fit_seed(
    gallery_cpu: torch.Tensor,
    gallery: torch.Tensor,
    pseudo: torch.Tensor,
    official: torch.Tensor,
    pseudo_labels: np.ndarray,
    official_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    spec = balanced_product_quantization_spec(
        dimensions=DIMENSIONS, bytes_per_vector=BLOCKS, codebook_size=CODEBOOK_SIZE
    )
    fit_started = time.monotonic()
    fitted = fit_product_quantizer(
        gallery_cpu, spec, seed=seed, maximum_iterations=MAXIMUM_ITERATIONS
    )
    fit_seconds = time.monotonic() - fit_started
    quantizer = fitted.to(gallery.device).eval()
    codebooks = torch.stack(tuple(quantizer.codebooks)).contiguous()
    encode_started = time.monotonic()
    public_isotropic_raw = quantizer.hard_encode(gallery)
    if gallery.device.type == "cuda":
        torch.cuda.synchronize(gallery.device)
    public_isotropic_encode_seconds = time.monotonic() - encode_started
    squared_norms = gallery.double().square().sum(dim=1)
    parallel_fraction = THRESHOLD * THRESHOLD / squared_norms
    multipliers = parallel_fraction / ((1.0 - parallel_fraction) / (DIMENSIONS - 1.0))
    encode_started = time.monotonic()
    isotropic_raw, anisotropic_raw = noise_shaped_product_encode(
        gallery,
        codebooks,
        parallel_multiplier=multipliers,
        maximum_rounds=MAXIMUM_ROUNDS,
    )
    isotropic = pack_nibbles(isotropic_raw)
    anisotropic = pack_nibbles(anisotropic_raw)
    if gallery.device.type == "cuda":
        torch.cuda.synchronize(gallery.device)
    anisotropic_encode_seconds = time.monotonic() - encode_started
    if isotropic.shape != anisotropic.shape or isotropic.shape[1] != PACKED_BYTES:
        raise RuntimeError("native PQ packed geometry differs")
    if not unpack_nibbles(isotropic).equal(isotropic_raw) or not unpack_nibbles(anisotropic).equal(
        anisotropic_raw
    ):
        raise RuntimeError("native PQ packed round trip differs")
    recentered_codebooks, recentered_counts = recenter_codebooks(
        gallery_cpu,
        anisotropic_raw.detach().cpu(),
        codebooks.detach().cpu(),
    )
    recentered_quantizer = ProductQuantizer(
        spec, tuple(recentered_codebooks[index] for index in range(BLOCKS))
    ).to(gallery.device).eval()
    arms: dict[str, object] = {}
    for name, scorer, packed in (
        ("isotropic", quantizer, isotropic),
        ("anisotropic", quantizer, anisotropic),
        ("recentered_anisotropic", recentered_quantizer, anisotropic),
    ):
        metrics: dict[str, object] = {}
        for metric in ("dot", "squared_l2", "normalized_squared_l2"):
            pseudo_score, pseudo_seconds = score_packed(
                scorer,
                packed,
                pseudo,
                pseudo_labels,
                gallery_labels,
                metric=metric,
            )
            official_score, official_seconds = score_packed(
                scorer,
                packed,
                official,
                official_labels,
                gallery_labels,
                metric=metric,
            )
            metrics[metric] = {
                "pseudo": pseudo_score,
                "pseudo_score_seconds": pseudo_seconds,
                "official": official_score,
                "official_score_seconds": official_seconds,
            }
        arms[name] = {
            "packed_codes_sha256": tensor_sha256(packed),
            "payload_bytes": packed.numel(),
            "diagnostics": reconstruction_diagnostics(gallery, scorer, packed),
            "metrics": metrics,
        }
    return {
        "seed": seed,
        "fit_seconds": fit_seconds,
        "public_isotropic_encode_seconds": public_isotropic_encode_seconds,
        "anisotropic_encode_seconds": anisotropic_encode_seconds,
        "codebooks_sha256": tensor_sha256(codebooks),
        "codebook_bytes": codebooks.numel() * codebooks.element_size(),
        "recentered_codebooks_sha256": tensor_sha256(recentered_codebooks),
        "recentered_empty_cells": int((recentered_counts == 0).sum()),
        "public_source_isotropic_code_difference_fraction": float(
            (public_isotropic_raw != isotropic_raw).double().mean()
        ),
        "code_churn_fraction": float((anisotropic_raw != isotropic_raw).double().mean()),
        "parallel_multiplier_minimum": float(multipliers.min()),
        "parallel_multiplier_maximum": float(multipliers.max()),
        "arms": arms,
    }


def summarize(
    seeds: list[dict[str, object]], *, candidate_arm: str = "anisotropic"
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for metric in ("dot", "squared_l2", "normalized_squared_l2"):
        seed_deltas: list[dict[str, object]] = []
        per_query_mmp = []
        isotropic_mmp = []
        for seed_result in seeds:
            arms = seed_result["arms"]
            baseline = arms["isotropic"]["metrics"][metric]
            candidate = arms[candidate_arm]["metrics"][metric]
            row: dict[str, object] = {"seed": seed_result["seed"]}
            for split in ("pseudo", "official"):
                row[split] = {
                    name: candidate[split][name] - baseline[split][name]
                    for name in ("mmp_at_5", "recall_at_1")
                }
            row["official"]["recall_at_1_hit_delta"] = recall_hit_delta(
                candidate["official"]["per_query_r1"], baseline["official"]["per_query_r1"]
            )
            seed_deltas.append(row)
            isotropic_mmp.append(baseline["pseudo"]["mmp_at_5"])
            per_query_mmp.append(
                np.asarray(candidate["pseudo"]["per_query_mmp_at_5"], dtype=np.float64)
                - np.asarray(baseline["pseudo"]["per_query_mmp_at_5"], dtype=np.float64)
            )
        mean_query_delta = np.mean(np.stack(per_query_mmp), axis=0, dtype=np.float64)
        mean_pseudo_mmp = float(
            np.mean([row["pseudo"]["mmp_at_5"] for row in seed_deltas], dtype=np.float64)
        )
        mean_pseudo_r1 = float(
            np.mean([row["pseudo"]["recall_at_1"] for row in seed_deltas], dtype=np.float64)
        )
        mean_official_mmp = float(
            np.mean([row["official"]["mmp_at_5"] for row in seed_deltas], dtype=np.float64)
        )
        official_recall_hit_delta = int(
            sum(row["official"]["recall_at_1_hit_delta"] for row in seed_deltas)
        )
        baseline_spread = float(max(isotropic_mmp) - min(isotropic_mmp))
        interval = bootstrap_interval(mean_query_delta, seed=51_337)
        mechanism_supported = (
            meets_lower_bound(mean_pseudo_mmp, 0.002)
            and interval[0] > 0.0
            and all(row["pseudo"]["mmp_at_5"] > 0.0 for row in seed_deltas)
            and meets_lower_bound(mean_pseudo_r1, -0.003)
        )
        seed_floor_cleared = mean_pseudo_mmp > baseline_spread
        summary[metric] = {
            "seed_deltas": seed_deltas,
            "mean_pseudo_mmp_at_5_delta": mean_pseudo_mmp,
            "mean_pseudo_recall_at_1_delta": mean_pseudo_r1,
            "mean_query_mmp_at_5_delta_bootstrap_95": interval,
            "mean_official_mmp_at_5_delta": mean_official_mmp,
            "official_recall_at_1_total_hit_delta": official_recall_hit_delta,
            "original_official_noninferiority_pass": all(
                meets_lower_bound(row["official"]["mmp_at_5"], -0.005)
                and row["official"]["recall_at_1_hit_delta"] >= -1
                for row in seed_deltas
            ),
            "isotropic_seed_spread_mmp_at_5": baseline_spread,
            "mechanism_supported": mechanism_supported,
            "seed_floor_cleared": seed_floor_cleared,
            "promotion_supported": mechanism_supported and seed_floor_cleared,
        }
    return summary


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack consecutive 4-bit assignments low nibble first."""

    if (
        type(codes) is not torch.Tensor
        or codes.dtype != torch.uint8
        or codes.ndim != 2
        or codes.shape[0] < 1
        or codes.shape[1] < 2
        or codes.shape[1] % 2 != 0
        or bool((codes >= 16).any())
    ):
        raise ValueError("nibble code authority differs")
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack low/high nibbles into consecutive uint8 assignments."""

    if (
        type(packed) is not torch.Tensor
        or packed.dtype != torch.uint8
        or packed.ndim != 2
        or packed.shape[0] < 1
        or packed.shape[1] < 1
    ):
        raise ValueError("packed nibble authority differs")
    result = torch.empty(
        (packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8, device=packed.device
    )
    result[:, 0::2] = packed & 0x0F
    result[:, 1::2] = packed >> 4
    return result.contiguous()


def topk_decoded(
    queries: torch.Tensor,
    decoded: torch.Tensor,
    *,
    width: int,
    metric: Literal["dot", "squared_l2", "normalized_squared_l2"],
    batch_rows: int = 64,
) -> torch.Tensor:
    """Return exact exhaustive rankings with stable lowest-ordinal ties."""

    if (
        type(queries) is not torch.Tensor
        or type(decoded) is not torch.Tensor
        or queries.dtype != torch.float32
        or decoded.dtype != torch.float32
        or queries.ndim != 2
        or decoded.ndim != 2
        or queries.shape[0] < 1
        or decoded.shape[0] < width
        or queries.shape[1] != decoded.shape[1]
        or queries.device != decoded.device
        or type(width) is not int
        or width < 1
        or metric not in ("dot", "squared_l2", "normalized_squared_l2")
        or type(batch_rows) is not int
        or batch_rows < 1
        or not bool(torch.isfinite(queries).all())
        or not bool(torch.isfinite(decoded).all())
    ):
        raise ValueError("decoded ranking authority differs")
    if metric == "normalized_squared_l2":
        queries = torch.nn.functional.normalize(queries, dim=1)
        decoded = torch.nn.functional.normalize(decoded, dim=1)
        metric = "squared_l2"
    batches = []
    decoded_norms = decoded.square().sum(dim=1)
    with torch.no_grad():
        for start in range(0, queries.shape[0], batch_rows):
            query = queries[start : start + batch_rows]
            dots = query @ decoded.T
            if metric == "dot":
                values, order = torch.topk(dots, k=width, dim=1, largest=True, sorted=True)
                boundary_ties = (dots == values[:, -1:]).sum(dim=1) > 1
                internal_ties = (values[:, 1:] == values[:, :-1]).any(dim=1)
                if bool((boundary_ties | internal_ties).any()):
                    order = torch.argsort(dots, dim=1, descending=True, stable=True)[:, :width]
            else:
                distances = query.square().sum(dim=1, keepdim=True) + decoded_norms - 2 * dots
                values, order = torch.topk(distances, k=width, dim=1, largest=False, sorted=True)
                boundary_ties = (distances == values[:, -1:]).sum(dim=1) > 1
                internal_ties = (values[:, 1:] == values[:, :-1]).any(dim=1)
                if bool((boundary_ties | internal_ties).any()):
                    order = torch.argsort(distances, dim=1, descending=False, stable=True)[
                        :, :width
                    ]
            batches.append(order)
    return torch.cat(batches, dim=0).contiguous()


def parallel_cost_multiplier(*, threshold: float, squared_norm: float, dimensions: int) -> float:
    """Return ScaNN's exact residual-parallel multiplier."""

    if (
        not math.isfinite(threshold)
        or threshold < 0.0
        or not math.isfinite(squared_norm)
        or squared_norm <= threshold * threshold
        or type(dimensions) is not int
        or dimensions < 2
    ):
        raise ValueError("noise-shaped multiplier authority differs")
    parallel = threshold * threshold / squared_norm
    perpendicular = (1.0 - parallel) / (dimensions - 1.0)
    return parallel / perpendicular


def noise_shaped_product_encode(
    values: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    parallel_multiplier: float | torch.Tensor,
    maximum_rounds: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce ScaNN product-code noise shaping over fixed block codebooks."""

    if (
        type(values) is not torch.Tensor
        or type(codebooks) is not torch.Tensor
        or values.dtype != torch.float32
        or codebooks.dtype != torch.float32
        or values.ndim != 2
        or codebooks.ndim != 3
        or codebooks.shape[0] * codebooks.shape[2] != values.shape[1]
        or not 2 <= codebooks.shape[1] <= 256
        or values.device != codebooks.device
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(codebooks).all())
        or (
            type(parallel_multiplier) is float
            and (not math.isfinite(parallel_multiplier) or parallel_multiplier < 0.0)
        )
        or (
            type(parallel_multiplier) is torch.Tensor
            and (
                parallel_multiplier.ndim != 1
                or parallel_multiplier.shape[0] != values.shape[0]
                or parallel_multiplier.device != values.device
                or not bool(torch.isfinite(parallel_multiplier).all())
                or bool((parallel_multiplier < 0.0).any())
            )
        )
        or type(parallel_multiplier) not in (float, torch.Tensor)
        or type(maximum_rounds) is not int
        or maximum_rounds < 1
    ):
        raise ValueError("noise-shaped product encoding authority differs")
    rows = values.shape[0]
    blocks, centers_count, width = codebooks.shape
    row_indexes = torch.arange(rows, device=values.device)
    block_values = values.reshape(rows, blocks, width).double()
    score_codebooks = codebooks.double()
    inverse_norms = torch.rsqrt(values.double().square().sum(dim=1))
    multipliers = (
        torch.full((rows,), parallel_multiplier, dtype=torch.float64, device=values.device)
        if type(parallel_multiplier) is float
        else parallel_multiplier.double()
    )
    residual_norms = torch.empty(
        (rows, blocks, centers_count), dtype=torch.float64, device=values.device
    )
    parallel_components = torch.empty_like(residual_norms)
    with torch.no_grad():
        for stage in range(blocks):
            residuals = block_values[:, stage, None, :] - score_codebooks[stage, None, :, :]
            residual_norms[:, stage, :] = residuals.square().sum(dim=2)
            parallel_components[:, stage, :] = (
                (block_values[:, stage, None, :] * residuals).sum(dim=2)
                * inverse_norms[:, None]
            )
        isotropic_codes = residual_norms.argmin(dim=2).to(torch.uint8).contiguous()
        codes = isotropic_codes.clone()
        selected_parallel = parallel_components.gather(2, codes.long()[:, :, None]).squeeze(2)
        parallel_component = selected_parallel.sum(dim=1)
        initial_residual_norms = residual_norms.gather(
            2, codes.long()[:, :, None]
        ).squeeze(2)
        stage_order = torch.argsort(initial_residual_norms, dim=1, descending=True, stable=True)
        for _round in range(maximum_rounds):
            changed = False
            for order_index in range(blocks):
                stage = stage_order[:, order_index]
                incumbent = codes[row_indexes, stage].long()
                candidate_norms = residual_norms[row_indexes, stage]
                candidate_parallel = parallel_components[row_indexes, stage]
                incumbent_norm = candidate_norms[row_indexes, incumbent]
                incumbent_parallel = candidate_parallel[row_indexes, incumbent]
                new_parallel = (
                    parallel_component[:, None] - incumbent_parallel[:, None] + candidate_parallel
                )
                parallel_delta = new_parallel.square() - parallel_component[:, None].square()
                cost_delta = (
                    candidate_norms
                    - incumbent_norm[:, None]
                    + (multipliers[:, None] - 1.0) * parallel_delta
                )
                cost_delta = torch.where(
                    parallel_delta <= 0.0,
                    cost_delta,
                    torch.full_like(cost_delta, torch.inf),
                )
                cost_delta[row_indexes, incumbent] = torch.inf
                best_delta, candidate = torch.min(cost_delta, dim=1)
                use_candidate = best_delta < 0.0
                replacement = torch.where(use_candidate, candidate, incumbent)
                replacement_parallel = new_parallel.gather(1, replacement[:, None]).squeeze(1)
                codes[row_indexes, stage] = replacement.to(torch.uint8)
                parallel_component = torch.where(
                    use_candidate, replacement_parallel, parallel_component
                )
                changed = changed or bool(use_candidate.any())
            if not changed:
                break
    return isotropic_codes, codes.contiguous()


def scalar_noise_shaped_product_encode(
    values: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    parallel_multiplier: float,
    maximum_rounds: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent scalar rendition of ScaNN's coordinate updates for self-test."""

    rows, dimensions = values.shape
    blocks, centers_count, width = codebooks.shape
    if rows < 1 or blocks * width != dimensions:
        raise ValueError("scalar noise-shaped fixture differs")
    initial = torch.empty((rows, blocks), dtype=torch.uint8)
    result = torch.empty_like(initial)
    value_rows = values.double().cpu().tolist()
    center_rows = codebooks.double().cpu().tolist()
    for row_index, row in enumerate(value_rows):
        inverse_norm = 1.0 / math.sqrt(sum(coordinate * coordinate for coordinate in row))
        norms = [[0.0] * centers_count for _ in range(blocks)]
        parallels = [[0.0] * centers_count for _ in range(blocks)]
        for block in range(blocks):
            offset = block * width
            for center in range(centers_count):
                for lane in range(width):
                    coordinate = row[offset + lane]
                    residual = coordinate - center_rows[block][center][lane]
                    norms[block][center] += residual * residual
                    parallels[block][center] += residual * coordinate * inverse_norm
        codes = [
            min(range(centers_count), key=lambda center: norms[block][center])
            for block in range(blocks)
        ]
        for block, code in enumerate(codes):
            initial[row_index, block] = code
        parallel = sum(parallels[block][code] for block, code in enumerate(codes))
        order = sorted(range(blocks), key=lambda block: (-norms[block][codes[block]], block))
        for _round in range(maximum_rounds):
            changed = False
            for block in order:
                incumbent = codes[block]
                best = incumbent
                best_delta = 0.0
                best_parallel = parallel
                for candidate in range(centers_count):
                    if candidate == incumbent:
                        continue
                    candidate_parallel = (
                        parallel - parallels[block][incumbent] + parallels[block][candidate]
                    )
                    parallel_delta = candidate_parallel * candidate_parallel - parallel * parallel
                    if parallel_delta > 0.0:
                        continue
                    residual_delta = norms[block][candidate] - norms[block][incumbent]
                    cost_delta = residual_delta + (parallel_multiplier - 1.0) * parallel_delta
                    if cost_delta < best_delta:
                        best = candidate
                        best_delta = cost_delta
                        best_parallel = candidate_parallel
                if best != incumbent:
                    codes[block] = best
                    parallel = best_parallel
                    changed = True
            if not changed:
                break
        result[row_index] = torch.tensor(codes, dtype=torch.uint8)
    return initial, result


def reconstruction_objective(
    values: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    *,
    parallel_multiplier: float,
) -> torch.Tensor:
    blocks = codebooks.shape[0]
    selected = codebooks[torch.arange(blocks)[None, :], codes.long()]
    residual = values.reshape(values.shape[0], blocks, codebooks.shape[2]) - selected
    inverse_norm = torch.rsqrt(values.double().square().sum(dim=1))
    parallel = (values.reshape_as(residual) * residual).sum(dim=(1, 2)) * inverse_norm
    return residual.square().sum(dim=(1, 2)) + (parallel_multiplier - 1.0) * parallel.square()


def recenter_codebooks(
    values: torch.Tensor,
    codes: torch.Tensor,
    initial_codebooks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return conditional-mean centers for fixed assignments.

    Empty cells retain their initial center.  The control runs on CPU so each
    center mean has a deterministic row-reduction order.
    """

    if (
        type(values) is not torch.Tensor
        or values.dtype != torch.float32
        or values.device.type != "cpu"
        or values.ndim != 2
        or type(codes) is not torch.Tensor
        or codes.dtype != torch.uint8
        or codes.device.type != "cpu"
        or codes.ndim != 2
        or codes.shape[0] != values.shape[0]
        or type(initial_codebooks) is not torch.Tensor
        or initial_codebooks.dtype != torch.float32
        or initial_codebooks.device.type != "cpu"
        or initial_codebooks.ndim != 3
        or initial_codebooks.shape[0] != codes.shape[1]
        or initial_codebooks.shape[0] * initial_codebooks.shape[2] != values.shape[1]
        or bool((codes >= initial_codebooks.shape[1]).any())
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(initial_codebooks).all())
    ):
        raise ValueError("recentered codebook authority differs")
    blocks, centers, width = initial_codebooks.shape
    blocked = values.reshape(values.shape[0], blocks, width)
    result = initial_codebooks.clone()
    counts = torch.zeros((blocks, centers), dtype=torch.int64)
    for block in range(blocks):
        for center in range(centers):
            selected = codes[:, block] == center
            count = int(selected.sum())
            counts[block, center] = count
            if count:
                result[block, center] = blocked[selected, block].double().mean(dim=0).float()
    return result.contiguous(), counts


def self_test() -> None:
    assert meets_lower_bound(0.002, 0.002)
    assert meets_lower_bound(0.002 - 5e-13, 0.002)
    assert not meets_lower_bound(0.002 - 2e-12, 0.002)
    assert recall_hit_delta([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == 0
    assert recall_hit_delta([1.0, 0.0, 0.0], [1.0, 0.0, 1.0]) == -1
    assert recall_hit_delta([0.0, 0.0, 0.0], [1.0, 0.0, 1.0]) == -2
    recenter_values = torch.tensor(
        [[1.0, 3.0, 10.0, 12.0], [3.0, 5.0, 14.0, 16.0], [9.0, 11.0, 20.0, 22.0]],
        dtype=torch.float32,
    )
    recenter_codes = torch.tensor([[0, 1], [0, 1], [1, 1]], dtype=torch.uint8)
    recenter_initial = torch.tensor(
        [
            [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]],
            [[-7.0, -8.0], [-9.0, -10.0], [-11.0, -12.0]],
        ],
        dtype=torch.float32,
    )
    recentered, counts = recenter_codebooks(
        recenter_values, recenter_codes, recenter_initial
    )
    assert counts.tolist() == [[2, 1, 0], [0, 3, 0]]
    torch.testing.assert_close(
        recentered,
        torch.tensor(
            [
                [[2.0, 4.0], [9.0, 11.0], [-5.0, -6.0]],
                [[-7.0, -8.0], [44.0 / 3.0, 50.0 / 3.0], [-11.0, -12.0]],
            ]
        ),
    )
    values = torch.tensor([[0.8, 0.4, 0.3, 0.3]], dtype=torch.float32)
    values = torch.nn.functional.normalize(values, dim=1)
    codebooks = torch.tensor(
        [
            [[0.7, 0.5], [0.9, 0.1]],
            [[0.4, 0.2], [0.1, 0.5]],
        ],
        dtype=torch.float32,
    )
    eta = parallel_cost_multiplier(threshold=0.6, squared_norm=1.0, dimensions=4)
    assert math.isclose(eta, 1.6875, rel_tol=0.0, abs_tol=1e-12)
    initial, codes = noise_shaped_product_encode(
        values, codebooks, parallel_multiplier=eta, maximum_rounds=10
    )
    scalar_initial, scalar_codes = scalar_noise_shaped_product_encode(
        values, codebooks, parallel_multiplier=eta, maximum_rounds=10
    )
    assert initial.equal(scalar_initial)
    assert codes.equal(scalar_codes)
    before = reconstruction_objective(values, codebooks, initial, parallel_multiplier=eta)
    after = reconstruction_objective(values, codebooks, codes, parallel_multiplier=eta)
    assert bool((after <= before).all())
    _, converged = noise_shaped_product_encode(
        values, codebooks, parallel_multiplier=eta, maximum_rounds=10
    )
    assert converged.equal(codes)
    mutation_values = torch.tensor(
        [[0.1451240331, -0.3857758343, -0.7382100224, -0.3190383911, 0.4241273403, 0.05909820646]],
        dtype=torch.float32,
    )
    mutation_codebooks = torch.tensor(
        [
            [
                [-0.7419588566, -0.6375496387],
                [0.6793804765, -0.3946876228],
                [1.3271791935, 0.1106321141],
            ],
            [
                [0.3374040127, -0.3201435804],
                [0.5688963532, 0.1333054304],
                [1.7421891689, 0.5880212784],
            ],
            [
                [0.2755166888, -0.9336423278],
                [-0.1527392566, -0.5299631953],
                [1.4535353184, -1.5414217710],
            ],
        ],
        dtype=torch.float32,
    )
    batch_initial, batch_codes = noise_shaped_product_encode(
        mutation_values,
        mutation_codebooks,
        parallel_multiplier=10.0,
        maximum_rounds=10,
    )
    scalar_initial, scalar_codes = scalar_noise_shaped_product_encode(
        mutation_values,
        mutation_codebooks,
        parallel_multiplier=10.0,
        maximum_rounds=10,
    )
    assert batch_initial.equal(scalar_initial)
    assert batch_codes.equal(scalar_codes)
    assert batch_initial.tolist() == [[1, 0, 1]]
    assert batch_codes.tolist() == [[1, 0, 2]]
    raw = torch.tensor([[0, 1, 2, 15], [15, 2, 1, 0]], dtype=torch.uint8)
    packed = pack_nibbles(raw)
    assert packed.tolist() == [[16, 242], [47, 1]]
    assert unpack_nibbles(packed).equal(raw)
    decoded = torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    queries = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    assert topk_decoded(queries, decoded, width=2, metric="dot").tolist() == [[0, 1]]
    assert topk_decoded(queries, decoded, width=2, metric="squared_l2").tolist() == [[0, 1]]
    unequal_norms = torch.tensor([[0.9, 0.0], [1.0, 1.0]], dtype=torch.float32)
    axis_query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    assert topk_decoded(axis_query, unequal_norms, width=1, metric="dot").tolist() == [[1]]
    assert topk_decoded(axis_query, unequal_norms, width=1, metric="squared_l2").tolist() == [[0]]
    assert topk_decoded(
        axis_query, unequal_norms, width=1, metric="normalized_squared_l2"
    ).tolist() == [[0]]


def main() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("native PQ matmul precision differs")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--features-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("SELF_TEST_OK")
        return
    if None in (args.features, args.features_sha256, args.output):
        raise ValueError("native noise-shaped PQ scientific arguments required")
    if args.output.exists() or sha256_file(args.features) != args.features_sha256:
        raise ValueError("native noise-shaped PQ input authority differs")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("native noise-shaped PQ device differs")
    with np.load(args.features, allow_pickle=False) as archive:
        train = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.ascontiguousarray(archive["train_labels"], dtype=np.int64)
        paths = np.ascontiguousarray(archive["train_paths"])
        official_array = np.ascontiguousarray(archive["val_embeddings"], dtype=np.float32)
        official_labels = np.ascontiguousarray(archive["val_labels"], dtype=np.int64)
    held_out = pseudoquery_mask(labels, paths)
    if (
        train.shape != (38_307, DIMENSIONS)
        or official_array.shape != (129, DIMENSIONS)
        or int(held_out.sum()) != 3_050
    ):
        raise ValueError("native noise-shaped PQ population differs")
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    official_array /= np.linalg.norm(official_array, axis=1, keepdims=True)
    gallery_array = np.ascontiguousarray(train[~held_out])
    gallery_labels = np.ascontiguousarray(labels[~held_out])
    pseudo_array = np.ascontiguousarray(train[held_out])
    pseudo_labels = np.ascontiguousarray(labels[held_out])
    if len(np.unique(pseudo_labels)) != 3_050:
        raise ValueError("native noise-shaped PQ pseudo-query classes differ")
    gallery_cpu = torch.from_numpy(gallery_array.copy()).contiguous()
    gallery = gallery_cpu.to(device)
    pseudo = torch.from_numpy(pseudo_array.copy()).to(device)
    official = torch.from_numpy(official_array.copy()).to(device)
    seed_results = []
    for seed in SEEDS:
        result = fit_seed(
            gallery_cpu,
            gallery,
            pseudo,
            official,
            pseudo_labels,
            official_labels,
            gallery_labels,
            seed=seed,
        )
        seed_results.append(result)
        print(
            json.dumps(
                {
                    "completed_seed": seed,
                    "fit_seconds": result["fit_seconds"],
                    "anisotropic_encode_seconds": result["anisotropic_encode_seconds"],
                    "code_churn_fraction": result["code_churn_fraction"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = summarize(seed_results)
    recentered_summary = summarize(seed_results, candidate_arm="recentered_anisotropic")
    payload = {
        "schema": "scratch-met-small-native-noise-shaped-pq128x4-recentered-v1",
        "claim_eligible": False,
        "features_sha256": args.features_sha256,
        "fit_gallery_rows": len(gallery_array),
        "pseudo_queries": len(pseudo_array),
        "official_validation_queries": len(official_array),
        "geometry": {
            "dimensions": DIMENSIONS,
            "blocks": BLOCKS,
            "block_width": BLOCK_WIDTH,
            "codebook_size": CODEBOOK_SIZE,
            "bits_per_assignment": 4,
            "packed_bytes_per_vector": PACKED_BYTES,
            "packed_low_nibble_first": True,
        },
        "fit": {
            "seeds": list(SEEDS),
            "maximum_kmeans_iterations": MAXIMUM_ITERATIONS,
            "maximum_noise_shaping_rounds": MAXIMUM_ROUNDS,
            "anisotropic_threshold": THRESHOLD,
            "scann_source_commit": "4700efb9afa54286b0e04473ba80a13e8461e25f",
        },
        "seeds": seed_results,
        "summary": summary,
        "recentered_summary": recentered_summary,
        "public_codec_followup_supported": bool(
            summary["dot"]["mechanism_supported"]
            and summary["squared_l2"]["promotion_supported"]
        ),
        "scorer_compatibility_diagnostic_supported": bool(
            summary["dot"]["promotion_supported"]
            and not summary["squared_l2"]["promotion_supported"]
        ),
        "normalization_resolves_scorer_mismatch": bool(
            summary["dot"]["promotion_supported"]
            and not summary["squared_l2"]["promotion_supported"]
            and summary["normalized_squared_l2"]["promotion_supported"]
        ),
        "recentered_decoder_supported": bool(
            recentered_summary["squared_l2"]["promotion_supported"]
        ),
        "generic_supported": False,
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
