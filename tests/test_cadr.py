from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sfora.cadr import balanced_log_loss, build_pairs, fit_cadr, select_lambda, split_labels


def _unit(values: list[list[float]]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_split_labels_matches_literal_hash_oracle_and_excludes_singletons() -> None:
    labels = np.repeat(np.arange(-5, 5, dtype=np.int64), 2)
    labels = np.concatenate([labels, np.asarray([99], dtype=np.int64)])
    expected = sorted(
        range(-5, 5),
        key=lambda v: (
            hashlib.sha256(b"CADR-split-v1:" + np.asarray(v, dtype="<i8").tobytes()).digest(),
            v,
        ),
    )
    split = split_labels(labels)
    assert split.fit_labels.tolist() == expected[:8]
    assert split.validation_labels.tolist() == expected[8:]
    assert 99 not in split.fit_labels and 99 not in split.validation_labels


def test_split_labels_rejects_wrong_type_and_empty_partition() -> None:
    with pytest.raises(ValueError):
        split_labels(np.asarray([1, 1], dtype=np.int32))
    with pytest.raises(ValueError):
        split_labels(np.asarray([1, 1, 2], dtype=np.int64))


def test_build_pairs_matches_positive_and_stable_boundary_oracle() -> None:
    embeddings = _unit(
        [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [-1, 0], [-0.9, 0.1]]
    )
    labels = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64)
    ids = np.asarray([f"id-{i}" for i in range(6)])
    pairs = build_pairs(embeddings, labels, ids, np.asarray([1, 2, 3], dtype=np.int64), k=1)
    assert pairs.positive_indices.tolist() == [[0, 1], [2, 3], [4, 5]]
    assert pairs.negative_indices.tolist() == [[0, 3], [1, 2], [1, 3], [2, 4], [2, 5]]
    assert pairs.features.dtype == np.float64
    assert pairs.targets.tolist() == [1.0] * 3 + [-1.0] * 5


def test_build_pairs_is_pool_isolated_and_validates_inputs() -> None:
    embeddings = _unit([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
    labels = np.asarray([1, 1, 2, 2], dtype=np.int64)
    ids = np.asarray(["a", "b", "c", "d"])
    pairs = build_pairs(embeddings, labels, ids, np.asarray([1], dtype=np.int64), k=1)
    assert pairs.positive_indices.tolist() == [[0, 1]]
    assert pairs.negative_indices.size == 0
    with pytest.raises(ValueError):
        build_pairs(embeddings * 2, labels, ids, np.asarray([1, 2], dtype=np.int64))
    with pytest.raises(ValueError):
        build_pairs(embeddings, labels, np.asarray(["a", "a", "c", "d"]), np.asarray([1, 2]))


def test_fit_cadr_gradient_and_diagonal_signal() -> None:
    rng = np.random.default_rng(7)
    positive = np.column_stack([rng.normal(0.5, 0.1, 80), rng.normal(0, 0.1, 80)])
    negative = np.column_stack([rng.normal(-0.5, 0.1, 80), rng.normal(0, 0.1, 80)])
    features = np.vstack([positive, negative])
    targets = np.asarray([1.0] * 80 + [-1.0] * 80)
    model = fit_cadr(features, targets, lambda_value=0.01)
    assert model.weights[0] > model.weights[1]
    assert model.converged


def test_balanced_log_loss_and_lambda_selection_are_deterministic() -> None:
    features = np.asarray(
        [[0.6, 0.0], [0.5, 0.0], [-0.6, 0.0], [-0.5, 0.0]], dtype=np.float64
    )
    targets = np.asarray([1.0, 1.0, -1.0, -1.0])
    model, records = select_lambda(features, targets, features, targets, (0.01, 1.0))
    assert [record.lambda_value for record in records] == [0.01, 1.0]
    assert model.weights.shape == (2,)
    assert balanced_log_loss(features, targets, model) == min(
        record.validation_loss for record in records
    )
