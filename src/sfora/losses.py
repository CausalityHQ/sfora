from typing import Literal

import numpy as np
from numpy.typing import NDArray

Reduction = Literal["mean", "sum"]


def triplet_margin_loss(
    anchors: NDArray[np.floating],
    positives: NDArray[np.floating],
    negatives: NDArray[np.floating],
    *,
    margin: float = 1.0,
    reduction: Reduction = "mean",
) -> float:
    """Compute standard Euclidean triplet margin loss for aligned batches."""
    anchor_array = _as_float_array(anchors, expected_ndim=2, name="anchors")
    positive_array = _as_float_array(positives, expected_ndim=2, name="positives")
    negative_array = _as_float_array(negatives, expected_ndim=2, name="negatives")
    _validate_same_shape(anchor_array, positive_array, negative_array)

    positive_distances = _euclidean_distance(anchor_array, positive_array)
    negative_distances = _euclidean_distance(anchor_array, negative_array)
    losses = np.maximum(positive_distances - negative_distances + margin, 0.0)
    return _reduce(losses, reduction)


def group_triplet_margin_loss(
    anchor_groups: NDArray[np.floating],
    positive_groups: NDArray[np.floating],
    negative_groups: NDArray[np.floating],
    *,
    margin: float = 1.0,
    hard_weight: float = 0.5,
    spread_weight: float = 0.1,
    reduction: Reduction = "mean",
) -> float:
    """Compute a group-aware triplet loss over batches of point sets.

    Each input has shape ``batch x group_size x embedding_dim``. The loss combines
    centroid triplet separation, hardest member separation, and within-group
    compactness. Singleton groups reduce to centroid triplet behavior when both
    group-specific weights are zero.
    """
    anchor_array = _as_float_array(anchor_groups, expected_ndim=3, name="anchor_groups")
    positive_array = _as_float_array(positive_groups, expected_ndim=3, name="positive_groups")
    negative_array = _as_float_array(negative_groups, expected_ndim=3, name="negative_groups")
    _validate_group_shapes(anchor_array, positive_array, negative_array)

    anchor_centroids = anchor_array.mean(axis=1)
    positive_centroids = positive_array.mean(axis=1)
    negative_centroids = negative_array.mean(axis=1)

    centroid_losses = _margin_violation(
        _euclidean_distance(anchor_centroids, positive_centroids),
        _euclidean_distance(anchor_centroids, negative_centroids),
        margin,
    )

    hard_positive_distances = _max_distance_to_centroid(positive_array, anchor_centroids)
    hard_negative_distances = _min_distance_to_centroid(negative_array, anchor_centroids)
    hard_losses = _margin_violation(hard_positive_distances, hard_negative_distances, margin)

    spread_penalties = (
        _mean_group_spread(anchor_array, anchor_centroids)
        + _mean_group_spread(positive_array, positive_centroids)
        + _mean_group_spread(negative_array, negative_centroids)
    )

    losses = centroid_losses + (hard_weight * hard_losses) + (spread_weight * spread_penalties)
    return _reduce(losses, reduction)


def _as_float_array(
    values: NDArray[np.floating],
    *,
    expected_ndim: int,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != expected_ndim:
        raise ValueError(f"{name} must be a {expected_ndim}D array")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one batch item")
    return array


def _validate_same_shape(*arrays: NDArray[np.float64]) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError("anchors, positives, and negatives must have the same shape")


def _validate_group_shapes(*arrays: NDArray[np.float64]) -> None:
    batch_dims = {array.shape[0] for array in arrays}
    embedding_dims = {array.shape[2] for array in arrays}
    empty_group = any(array.shape[1] == 0 for array in arrays)

    if len(batch_dims) != 1 or len(embedding_dims) != 1:
        raise ValueError("group triplets must share batch size and embedding dimension")
    if empty_group:
        raise ValueError("group triplets must contain at least one point per group")


def _euclidean_distance(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.linalg.norm(left - right, axis=-1)


def _margin_violation(
    positive_distances: NDArray[np.float64],
    negative_distances: NDArray[np.float64],
    margin: float,
) -> NDArray[np.float64]:
    return np.maximum(positive_distances - negative_distances + margin, 0.0)


def _max_distance_to_centroid(
    groups: NDArray[np.float64],
    centroids: NDArray[np.float64],
) -> NDArray[np.float64]:
    distances = np.linalg.norm(groups - centroids[:, np.newaxis, :], axis=-1)
    return distances.max(axis=1)


def _min_distance_to_centroid(
    groups: NDArray[np.float64],
    centroids: NDArray[np.float64],
) -> NDArray[np.float64]:
    distances = np.linalg.norm(groups - centroids[:, np.newaxis, :], axis=-1)
    return distances.min(axis=1)


def _mean_group_spread(
    groups: NDArray[np.float64],
    centroids: NDArray[np.float64],
) -> NDArray[np.float64]:
    distances = np.linalg.norm(groups - centroids[:, np.newaxis, :], axis=-1)
    return distances.mean(axis=1)


def _reduce(losses: NDArray[np.float64], reduction: Reduction) -> float:
    if reduction == "mean":
        return float(losses.mean())
    if reduction == "sum":
        return float(losses.sum())
    raise ValueError(f"unsupported reduction: {reduction}")
