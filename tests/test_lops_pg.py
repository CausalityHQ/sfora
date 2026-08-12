from __future__ import annotations

import numpy as np
import pytest

from sfora.lops_pg import (
    ARM_ORDER,
    batch_hard_triplet_tangent,
    decide_lops_pg,
    evaluate_cohort,
    positive_centroid_tangent,
    project_positive_safe,
    summarize_evaluations,
)


def _unit(value: list[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array / np.linalg.norm(array)


def test_positive_centroid_tangent_matches_independent_finite_difference() -> None:
    anchor = _unit([1.0, 0.3, -0.2])
    siblings = np.stack([_unit([0.8, 0.5, -0.1]), _unit([0.9, 0.2, 0.1]), _unit([0.7, 0.6, -0.2])])
    centroid = _unit(siblings.mean(axis=0).tolist())
    expected = centroid - anchor * float(anchor @ centroid)

    actual = positive_centroid_tangent(anchor, siblings)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
    epsilon = 1e-7
    plus = _unit((anchor + epsilon * actual).tolist())
    minus = _unit((anchor - epsilon * actual).tolist())
    finite_difference = float((plus @ centroid - minus @ centroid) / (2.0 * epsilon))
    assert finite_difference == pytest.approx(float(actual @ actual), rel=1e-7)
    siblings[:] = 0.0
    assert np.any(actual != 0.0)


def test_positive_safe_projection_changes_only_conflicts() -> None:
    tangent = np.asarray([0.0, 2.0, 0.0], dtype=np.float64)
    conflicting = np.asarray([3.0, 4.0, 5.0], dtype=np.float64)
    projected, active = project_positive_safe(conflicting, tangent)
    assert active is True
    np.testing.assert_array_equal(projected, np.asarray([3.0, 0.0, 5.0]))
    assert float(projected @ tangent) == 0.0

    safe = np.asarray([3.0, -4.0, 5.0], dtype=np.float64)
    unchanged, active = project_positive_safe(safe, tangent)
    assert active is False
    np.testing.assert_array_equal(unchanged, safe)
    assert not np.shares_memory(unchanged, safe)

    with pytest.raises(ValueError, match="degenerate"):
        project_positive_safe(conflicting, tangent * 1e-10)


def test_batch_hard_triplet_uses_stable_example_id_ties() -> None:
    anchor = _unit([1.0, 0.0])
    peers = np.stack(
        [
            _unit([0.8, 0.6]),
            _unit([0.8, -0.6]),
            _unit([0.6, 0.8]),
            _unit([0.6, -0.8]),
        ]
    )
    labels = np.asarray([1, 1, 2, 3], dtype=np.int64)
    ids = np.asarray(["positive-z", "positive-a", "negative-z", "negative-a"])
    actual = batch_hard_triplet_tangent(anchor, 1, peers, labels, ids)
    expected = peers[3] - peers[1]
    expected -= anchor * float(anchor @ expected)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_evaluate_cohort_builds_independent_equal_arc_arms() -> None:
    angles = np.asarray([0.00, 0.08, 0.16, 0.24, 1.0, 1.08, 1.16, 1.24])
    embeddings = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    ids = np.asarray([f"row-{index}" for index in range(8)])
    result = evaluate_cohort(embeddings, labels, ids, fold=1, index=7)
    assert tuple(result.margin_changes) == ARM_ORDER
    assert result.example_ids.tolist() == ids.tolist()
    assert result.primary_mask.sum() == 2
    assert result.conflict_mask.shape == (8,)
    for values in result.margin_changes.values():
        assert values.shape == (8,)
        assert np.isfinite(values).all()
    assert not np.shares_memory(result.directions["lops_pg"], result.directions["proxy_anchor"])
    active = np.linalg.norm(result.directions["lops_pg"], axis=1) >= 1e-8
    stepped = np.sum(
        embeddings[active]
        * (
            np.cos(0.01) * embeddings[active]
            - np.sin(0.01)
            * result.directions["lops_pg"][active]
            / np.linalg.norm(result.directions["lops_pg"][active], axis=1)[:, None]
        ),
        axis=1,
    )
    np.testing.assert_allclose(np.arccos(stepped), 0.01, rtol=0.0, atol=2e-14)


def test_summary_and_decision_are_driven_by_paired_primary_rows() -> None:
    evaluations = []
    for fold in (1, 2, 3):
        angles = np.linspace(0.0, 2.6, 180, dtype=np.float64) + fold * 1e-3
        embeddings = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        labels = np.repeat(np.arange(45, dtype=np.int64) + fold * 100, 4)
        ids = np.asarray([f"f{fold}-r{row}" for row in range(180)])
        evaluations.append(evaluate_cohort(embeddings, labels, ids, fold=fold, index=0))
    summary = summarize_evaluations(evaluations, bootstrap_replicates=100)
    assert summary["coverage"]["eligible_rows"] == 540
    assert summary["coverage"]["primary_rows"] == 135
    assert len(summary["fold_mean_advantages"]) == 3
    predicates, status = decide_lops_pg(summary)
    assert [item["name"] for item in predicates] == [
        "coverage",
        "raw_advantage",
        "fold_consistency",
        "control_superiority",
        "material_effect",
        "positive_similarity",
        "constraint_integrity",
    ]
    assert status in {"PASS", "KILL"}
