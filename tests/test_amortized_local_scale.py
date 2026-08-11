from __future__ import annotations

import numpy as np
import pytest

from sfora.amortized_local_scale import (
    fit_ridge_potential,
    nonself_density,
    predict_potential,
    select_ridge_lambda,
    split_labels,
)


def _unit(rows: list[list[float]]) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_nonself_density_uses_explicit_ids_and_is_block_invariant() -> None:
    embeddings = _unit([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]])
    row_ids = np.asarray(["d", "a", "c", "b"])
    expected = np.asarray([0.8, 0.8, 0.6, 0.0], dtype=np.float64)
    assert nonself_density(
        embeddings, row_ids, k=1, block_size=1
    ) == pytest.approx(expected)

    order = np.asarray([2, 0, 3, 1])
    assert nonself_density(
        embeddings[order], row_ids[order], k=1, block_size=3
    ) == pytest.approx(expected[order])


def test_split_labels_is_hash_ordered_and_class_disjoint() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    fit, validation = split_labels(labels)
    assert fit.dtype == validation.dtype == np.int64
    assert fit.size == 8
    assert validation.size == 2
    assert set(fit).isdisjoint(validation)


def test_ridge_recovers_affine_target_with_unregularized_intercept() -> None:
    embeddings = np.asarray(
        [[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32
    )
    targets = np.asarray([0.5, 0.2, 0.1, 0.4], dtype=np.float64)
    model = fit_ridge_potential(embeddings, targets, ridge_lambda=1e-6)
    assert model.weights.dtype == np.float64
    assert model.intercept == pytest.approx(0.3, abs=1e-12)
    assert predict_potential(model, embeddings) == pytest.approx(
        targets, abs=2e-7
    )


def test_select_ridge_lambda_breaks_equal_mse_by_grid_order() -> None:
    embeddings = np.zeros((6, 2), dtype=np.float32)
    targets = np.full(6, 0.4, dtype=np.float64)
    model, rows = select_ridge_lambda(
        embeddings[:4],
        targets[:4],
        embeddings[4:],
        targets[4:],
        (1e-6, 1e-4),
    )
    assert model.ridge_lambda == 1e-6
    assert [row["ridge_lambda"] for row in rows] == [1e-6, 1e-4]
    assert [row["validation_mse"] for row in rows] == pytest.approx([0.0, 0.0])
