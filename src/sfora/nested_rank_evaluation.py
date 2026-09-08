"""Dataset-agnostic nested-rank retrieval evidence and promotion decisions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from sfora.nested_rank_protocol import cluster_bootstrap_lower_bound


def _validated_arrays(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    sample_ids: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    if (
        type(embeddings) is not np.ndarray
        or embeddings.dtype != np.float32
        or embeddings.ndim != 2
        or embeddings.shape[0] < 2
        or embeddings.shape[1] == 0
        or not np.isfinite(embeddings).all()
        or type(labels) is not np.ndarray
        or labels.dtype != np.int64
        or labels.shape != (embeddings.shape[0],)
        or type(sample_ids) is not np.ndarray
        or sample_ids.dtype != np.int64
        or sample_ids.shape != labels.shape
        or len(set(int(value) for value in sample_ids)) != sample_ids.size
    ):
        raise ValueError("nested-rank evaluation arrays differ")
    _, counts = np.unique(labels, return_counts=True)
    if np.any(counts < 2):
        raise ValueError("nested-rank evaluation class inventory differs")
    values = embeddings.astype(np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0.0) or not np.isfinite(norms).all():
        raise ValueError("nested-rank evaluation embedding norms differ")
    return values / norms, labels, sample_ids


def _bounded_top_indices(
    distances: NDArray[np.float64], sample_ids: NDArray[np.int64], retained: int
) -> NDArray[np.int64]:
    """Select exact top-R by distance/sample ID without a full ranked buffer."""

    if (
        type(distances) is not np.ndarray
        or distances.dtype != np.float64
        or distances.ndim != 1
        or np.isnan(distances).any()
        or type(sample_ids) is not np.ndarray
        or sample_ids.dtype != np.int64
        or sample_ids.shape != distances.shape
        or len(set(int(value) for value in sample_ids)) != sample_ids.size
        or type(retained) is not int
        or not 0 < retained <= distances.size
    ):
        raise ValueError("nested-rank bounded ranking authority differs")
    return _bounded_top_indices_validated(distances, sample_ids, retained)


def _bounded_top_indices_validated(
    distances: NDArray[np.float64], sample_ids: NDArray[np.int64], retained: int
) -> NDArray[np.int64]:
    cutoff = float(np.partition(distances, retained - 1)[retained - 1])
    strict = np.flatnonzero(distances < cutoff)
    ties = np.flatnonzero(distances == cutoff)
    missing = retained - strict.size
    selected_ties = ties[np.argsort(sample_ids[ties], kind="stable")[:missing]]
    selected = np.concatenate((strict, selected_ties))
    order = np.lexsort((sample_ids[selected], distances[selected]))
    return np.asarray(selected[order], dtype=np.int64)


def rank_self_retrieval(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    sample_ids: NDArray[np.int64],
    *,
    block_rows: int = 256,
) -> list[dict[str, object]]:
    """Return deterministic leave-one-out top-R rankings and per-query evidence."""

    values, labels, sample_ids = _validated_arrays(embeddings, labels, sample_ids)
    if type(block_rows) is not int or block_rows <= 0:
        raise ValueError("nested-rank evaluation block size differs")
    label_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(labels, return_counts=True), strict=True)
    }
    result: list[dict[str, object]] = []
    for start in range(0, values.shape[0], block_rows):
        stop = min(start + block_rows, values.shape[0])
        distances = 1.0 - values[start:stop] @ values.T
        for local, query_index in enumerate(range(start, stop)):
            distances[local, query_index] = np.inf
            relevant = label_counts[int(labels[query_index])] - 1
            order = _bounded_top_indices_validated(distances[local], sample_ids, relevant)
            matches = labels[order] == labels[query_index]
            precisions = np.cumsum(matches, dtype=np.int64) / np.arange(1, relevant + 1)
            ap_at_r = float(np.sum(precisions * matches, dtype=np.float64) / relevant)
            result.append(
                {
                    "query_sample_id": int(sample_ids[query_index]),
                    "ranked_sample_ids": [int(sample_ids[index]) for index in order],
                    "ap_at_r": ap_at_r,
                    "recall_at_1": bool(matches[0]),
                }
            )
    return result


def rank_self_retrieval_int8(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    sample_ids: NDArray[np.int64],
    *,
    block_rows: int = 256,
) -> list[dict[str, object]]:
    """Return deterministic leave-one-out rankings from fixed-scale int8 codes."""

    values, labels, sample_ids = _validated_arrays(embeddings, labels, sample_ids)
    if type(block_rows) is not int or block_rows <= 0:
        raise ValueError("nested-rank evaluation block size differs")
    codes = np.clip(np.rint(values * 127.0), -127, 127).astype(np.int8)
    integer_codes = codes.astype(np.int32)
    code_norms = np.linalg.norm(integer_codes.astype(np.float64), axis=1)
    if np.any(code_norms == 0.0) or not np.isfinite(code_norms).all():
        raise ValueError("nested-rank int8 geometry differs")
    label_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(labels, return_counts=True), strict=True)
    }
    result: list[dict[str, object]] = []
    for start in range(0, values.shape[0], block_rows):
        stop = min(start + block_rows, values.shape[0])
        dots = integer_codes[start:stop] @ integer_codes.T
        scores = dots.astype(np.float64) / (code_norms[start:stop, None] * code_norms[None, :])
        for local, query_index in enumerate(range(start, stop)):
            scores[local, query_index] = -np.inf
            relevant = label_counts[int(labels[query_index])] - 1
            order = _bounded_top_indices_validated(-scores[local], sample_ids, relevant)
            matches = labels[order] == labels[query_index]
            precisions = np.cumsum(matches, dtype=np.int64) / np.arange(1, relevant + 1)
            ap_at_r = float(np.sum(precisions * matches, dtype=np.float64) / relevant)
            result.append(
                {
                    "query_sample_id": int(sample_ids[query_index]),
                    "ranked_sample_ids": [int(sample_ids[index]) for index in order],
                    "ap_at_r": ap_at_r,
                    "recall_at_1": bool(matches[0]),
                }
            )
    return result


def recompute_self_retrieval(
    evidence: Sequence[object],
    labels: NDArray[np.int64],
    sample_ids: NDArray[np.int64],
) -> dict[str, float | int]:
    """Validate ranked-ID evidence and independently recompute aggregate metrics."""

    if (
        type(labels) is not np.ndarray
        or labels.dtype != np.int64
        or labels.ndim != 1
        or type(sample_ids) is not np.ndarray
        or sample_ids.dtype != np.int64
        or sample_ids.shape != labels.shape
        or len(set(int(value) for value in sample_ids)) != sample_ids.size
        or type(evidence) not in {list, tuple}
        or len(evidence) != labels.size
    ):
        raise ValueError("nested-rank evaluation evidence inventory differs")
    label_by_id = {
        int(sample_id): int(label) for sample_id, label in zip(sample_ids, labels, strict=True)
    }
    counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(labels, return_counts=True), strict=True)
    }
    expected_queries = [int(value) for value in sample_ids]
    average_precisions: list[float] = []
    recalls: list[bool] = []
    for expected_query, row in zip(expected_queries, evidence, strict=True):
        if (
            type(row) is not dict
            or set(row) != {"query_sample_id", "ranked_sample_ids", "ap_at_r", "recall_at_1"}
            or type(row["query_sample_id"]) is not int
            or row["query_sample_id"] != expected_query
            or type(row["ranked_sample_ids"]) is not list
            or any(type(value) is not int for value in row["ranked_sample_ids"])
            or type(row["ap_at_r"]) is not float
            or not math.isfinite(row["ap_at_r"])
            or type(row["recall_at_1"]) is not bool
        ):
            raise ValueError("nested-rank query evidence differs")
        ranked = row["ranked_sample_ids"]
        relevant = counts[label_by_id[expected_query]] - 1
        if (
            len(ranked) != relevant
            or len(set(ranked)) != len(ranked)
            or expected_query in ranked
            or any(value not in label_by_id for value in ranked)
        ):
            raise ValueError("nested-rank ranked sample IDs differ")
        matches = np.asarray(
            [label_by_id[value] == label_by_id[expected_query] for value in ranked],
            dtype=np.bool_,
        )
        precisions = np.cumsum(matches, dtype=np.int64) / np.arange(1, relevant + 1)
        ap_at_r = float(np.sum(precisions * matches, dtype=np.float64) / relevant)
        recall_at_1 = bool(matches[0])
        if row["ap_at_r"] != ap_at_r or row["recall_at_1"] is not recall_at_1:
            raise ValueError("nested-rank query metric evidence differs")
        average_precisions.append(ap_at_r)
        recalls.append(recall_at_1)
    return {
        "map_at_r": math.fsum(average_precisions) / len(average_precisions),
        "recall_at_1": math.fsum(recalls) / len(recalls),
        "query_count": len(average_precisions),
    }


def class_bootstrap_lower_bound(
    candidate_ap: NDArray[np.float64],
    control_ap: NDArray[np.float64],
    class_ids: NDArray[np.int64],
    *,
    seed: int,
    replicates: int = 10_000,
) -> float:
    """Return the one-sided 5% class-cluster bootstrap bound for mAP@R delta."""

    if (
        type(candidate_ap) is not np.ndarray
        or candidate_ap.dtype != np.float64
        or type(control_ap) is not np.ndarray
        or control_ap.dtype != np.float64
        or control_ap.shape != candidate_ap.shape
    ):
        raise ValueError("nested-rank paired bootstrap authority differs")
    return cluster_bootstrap_lower_bound(
        candidate_ap - control_ap,
        class_ids,
        seed=seed,
        draws=replicates,
        quantile=0.05,
    )


def classify_promotion(
    *,
    candidate_map_at_r: float,
    control_map_at_r: float,
    bootstrap_lower_bound: float,
    candidate_recall_at_1: float,
    control_recall_at_1: float,
) -> dict[str, float | str]:
    """Apply the frozen nested-rank screen gates."""

    values = (
        candidate_map_at_r,
        control_map_at_r,
        bootstrap_lower_bound,
        candidate_recall_at_1,
        control_recall_at_1,
    )
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError("nested-rank promotion metrics differ")
    map_delta = candidate_map_at_r - control_map_at_r
    recall_delta = candidate_recall_at_1 - control_recall_at_1
    status = (
        "PROMOTE"
        if map_delta + 1e-15 >= 0.005
        and bootstrap_lower_bound > 0.0
        and recall_delta + 1e-15 >= -0.001
        else "REJECT"
    )
    return {
        "status": status,
        "map_at_r_delta": map_delta,
        "bootstrap_lower_bound": bootstrap_lower_bound,
        "recall_at_1_delta": recall_delta,
    }


def evaluate_paired_rankings(
    candidate_evidence: Sequence[object],
    control_evidence: Sequence[object],
    labels: NDArray[np.int64],
    sample_ids: NDArray[np.int64],
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, object]:
    """Recompute a paired candidate/control decision from immutable ranked rows."""

    candidate = recompute_self_retrieval(candidate_evidence, labels, sample_ids)
    control = recompute_self_retrieval(control_evidence, labels, sample_ids)
    candidate_ap = np.asarray(
        [row["ap_at_r"] for row in candidate_evidence],  # type: ignore[index]
        dtype=np.float64,
    )
    control_ap = np.asarray(
        [row["ap_at_r"] for row in control_evidence],  # type: ignore[index]
        dtype=np.float64,
    )
    lower = class_bootstrap_lower_bound(
        candidate_ap,
        control_ap,
        labels,
        seed=seed,
        replicates=replicates,
    )
    decision = classify_promotion(
        candidate_map_at_r=float(candidate["map_at_r"]),
        control_map_at_r=float(control["map_at_r"]),
        bootstrap_lower_bound=lower,
        candidate_recall_at_1=float(candidate["recall_at_1"]),
        control_recall_at_1=float(control["recall_at_1"]),
    )
    return {"candidate": candidate, "control": control, "decision": decision}
