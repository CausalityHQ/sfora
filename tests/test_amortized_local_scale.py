from __future__ import annotations

import numpy as np
import pytest

from sfora.amortized_local_scale import (
    compare_potential,
    compare_potentials,
    decide_alsp,
    density_diagnostics,
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


def test_compare_potential_reports_stable_ranking_and_paired_counts() -> None:
    gallery = _unit([[1, 0], [0.8, 0.6], [0, 1]])
    queries = _unit([[1, 0], [1, 0], [0, 1]])
    gallery_labels = np.asarray([1, 0, 2], dtype=np.int64)
    query_labels = np.asarray([1, 0, 2], dtype=np.int64)
    potential = np.asarray([0.5, 0.0, 0.0], dtype=np.float64)

    result = compare_potential(
        queries,
        query_labels,
        gallery,
        gallery_labels,
        potential,
        block_size=1,
    )

    assert result.raw_recall == pytest.approx(2 / 3)
    assert result.corrected_recall == pytest.approx(2 / 3)
    assert result.gain == pytest.approx(0.0)
    assert result.wrong_to_right == 1
    assert result.right_to_wrong == 1
    assert result.p_value == pytest.approx(1.0)


def test_compare_potentials_matches_literal_arm_results() -> None:
    gallery = _unit([[1, 0], [0.8, 0.6], [0, 1]])
    queries = _unit([[1, 0], [1, 0], [0, 1]])
    gallery_labels = np.asarray([1, 0, 2], dtype=np.int64)
    query_labels = np.asarray([1, 0, 2], dtype=np.int64)
    results = compare_potentials(
        queries,
        query_labels,
        gallery,
        gallery_labels,
        {
            "raw": np.zeros(3, dtype=np.float64),
            "shifted": np.asarray([0.5, 0.0, 0.0], dtype=np.float64),
        },
        block_size=2,
    )
    assert list(results) == ["raw", "shifted"]
    assert results["raw"].wrong_to_right == 0
    assert results["raw"].right_to_wrong == 0
    assert results["shifted"].wrong_to_right == 1
    assert results["shifted"].right_to_wrong == 1


def test_density_diagnostics_uses_average_tied_ranks() -> None:
    predicted = np.asarray([1.0, 2.0, 2.0, 4.0], dtype=np.float64)
    observed = np.asarray([4.0, 1.0, 1.0, 0.0], dtype=np.float64)
    result = density_diagnostics(predicted, observed)
    assert result == pytest.approx(
        {
            "pearson": -0.8411910241920598,
            "spearman": -1.0,
            "mse": 6.75,
        }
    )


def test_density_diagnostics_treats_constant_prediction_as_zero_correlation() -> None:
    predicted = np.ones(4, dtype=np.float64)
    observed = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    assert density_diagnostics(predicted, observed) == pytest.approx(
        {"pearson": 0.0, "spearman": 0.0, "mse": 1.5}
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("correlation", 0.199999),
        ("alsp_gain", 0.000999),
        ("oracle_gain", 0.003334),
        ("alsp_p_value", 0.05),
        ("permuted_gain", 0.00075),
        ("random_null_p95", 0.001),
    ],
)
def test_decide_alsp_rejects_each_failed_boundary(field: str, value: float) -> None:
    arguments = {
        "correlation": 0.20,
        "alsp_gain": 0.001,
        "oracle_gain": 0.0033333333333333335,
        "alsp_p_value": 0.049,
        "permuted_gain": 0.000749,
        "random_null_p95": 0.0008,
    }
    arguments[field] = value
    passed, predicates = decide_alsp(**arguments)
    assert not passed
    assert not predicates[
        {
            "correlation": "correlation",
            "alsp_gain": "absolute_gain",
            "oracle_gain": "oracle_recovery",
            "alsp_p_value": "paired_significance",
            "permuted_gain": "permuted_control",
            "random_null_p95": "random_direction_null",
        }[field]
    ]


def test_decide_alsp_accepts_exact_inclusive_gain_boundaries() -> None:
    passed, predicates = decide_alsp(
        correlation=0.20,
        alsp_gain=0.001,
        oracle_gain=0.0033333333333333335,
        alsp_p_value=0.049,
        permuted_gain=0.000749,
        random_null_p95=0.0008,
    )
    assert passed
    assert list(predicates) == [
        "correlation",
        "absolute_gain",
        "oracle_recovery",
        "paired_significance",
        "permuted_control",
        "random_direction_null",
    ]
