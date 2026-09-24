"""Paired-block tail-latency summaries for image-to-top-k benchmark receipts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import numpy as np


def paired_order(block: int) -> tuple[str, str]:
    """Alternate the arm order within each paired timing block."""

    if block < 0:
        raise ValueError("paired block index differs")
    return ("oml", "trained_b16") if block % 2 == 0 else ("trained_b16", "oml")


def validate_certification_shape(blocks: int, calls_per_block: int) -> None:
    """Require the prespecified minimum tail sample count and pairing."""

    if type(blocks) is not int or blocks < 20:
        raise ValueError("p99 certification requires at least twenty paired blocks")
    if blocks % 2:
        raise ValueError("p99 certification requires an even paired block count")
    if type(calls_per_block) is not int or calls_per_block < 1 or blocks * calls_per_block < 10_000:
        raise ValueError("p99 certification requires at least 10000 calls per cell")


def _higher_percentile(values: np.ndarray, fraction: float) -> int:
    index = math.ceil(values.size * fraction) - 1
    return int(np.partition(values, index)[index])


def block_bootstrap_p99_ratio(
    candidate: Sequence[Sequence[int]],
    reference: Sequence[Sequence[int]],
    *,
    draws: int = 5000,
    seed: int = 179019,
) -> dict[str, float | int | bool | str]:
    """Resample paired time blocks and bound the candidate/reference p99 ratio."""

    if len(candidate) != len(reference) or not candidate:
        raise ValueError("paired block shape differs")
    block_size = len(candidate[0])
    if block_size == 0 or any(
        len(left) != block_size or len(right) != block_size
        for left, right in zip(candidate, reference, strict=True)
    ):
        raise ValueError("paired block shape differs")
    validate_certification_shape(len(candidate), block_size)
    if type(draws) is not int or draws < 100 or type(seed) is not int or seed < 0:
        raise ValueError("paired bootstrap configuration differs")
    if any(
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or not 0 < value <= np.iinfo(np.int64).max
        for arm in (candidate, reference)
        for block in arm
        for value in block
    ):
        raise ValueError("paired block samples differ")
    left = np.asarray(candidate, dtype=np.int64)
    right = np.asarray(reference, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 2 or np.any(left <= 0) or np.any(right <= 0):
        raise ValueError("paired block samples differ")
    candidate_p99 = _higher_percentile(left.reshape(-1), 0.99)
    reference_p99 = _higher_percentile(right.reshape(-1), 0.99)
    candidate_p50 = _higher_percentile(left.reshape(-1), 0.5)
    reference_p50 = _higher_percentile(right.reshape(-1), 0.5)
    mean_latency_ratio = float(left.mean() / right.mean())
    random = np.random.default_rng(seed)
    ratios = np.empty(draws, dtype=np.float64)
    left_pairs = left.reshape(left.shape[0] // 2, 2, block_size)
    right_pairs = right.reshape(right.shape[0] // 2, 2, block_size)
    for draw in range(draws):
        selected = random.integers(0, left_pairs.shape[0], size=left_pairs.shape[0])
        numerator = _higher_percentile(left_pairs[selected].reshape(-1), 0.99)
        denominator = _higher_percentile(right_pairs[selected].reshape(-1), 0.99)
        ratios[draw] = numerator / denominator
    ci95_lower = float(np.quantile(ratios, 0.025, method="higher"))
    ci95_upper = float(np.quantile(ratios, 0.975, method="higher"))
    wins = sum(
        _higher_percentile(left_pair.reshape(-1), 0.99)
        < _higher_percentile(right_pair.reshape(-1), 0.99)
        for left_pair, right_pair in zip(left_pairs, right_pairs, strict=True)
    )
    superblocks = left_pairs.shape[0]
    sign_p = sum(math.comb(superblocks, k) for k in range(wins, superblocks + 1)) / (2**superblocks)
    point_p50_ratio = candidate_p50 / reference_p50
    return {
        "blocks": left.shape[0],
        "superblocks": left_pairs.shape[0],
        "calls_per_block": block_size,
        "candidate_p99_ns": candidate_p99,
        "reference_p99_ns": reference_p99,
        "point_ratio": candidate_p99 / reference_p99,
        "candidate_p50_ns": candidate_p50,
        "reference_p50_ns": reference_p50,
        "point_p50_ratio": point_p50_ratio,
        "mean_latency_ratio": mean_latency_ratio,
        "superblock_p99_wins": wins,
        "one_sided_sign_p": sign_p,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
        "latency_gate_passed": (
            ci95_upper < 1.0
            and point_p50_ratio <= 1.0
            and mean_latency_ratio <= 1.0
            and sign_p < 0.05
        ),
        "latency_gate_rule": (
            "p99 ratio block-bootstrap upper 95% < 1; p50 and mean latency ratios <= 1; "
            "one-sided paired-superblock p99 sign p < 0.05"
        ),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "ci_interpretation": "conditional on exchangeable consecutive AB/BA superblocks",
    }
