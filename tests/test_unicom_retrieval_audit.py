from __future__ import annotations

import numpy as np
import pytest

from sfora.unicom_retrieval_audit import (
    audit_deployment_geometry,
    geometry_decision,
    l2_normalize,
    paired_r1_interval,
    random_masks,
    retrieval_view,
)


def test_official_and_prefix_unit_use_different_normalization_order() -> None:
    query = np.array([[3.0, 4.0, 12.0]], dtype=np.float32)
    gallery = np.array(
        [[3.0, 4.0, 0.0], [0.0, 5.0, 12.0]],
        dtype=np.float32,
    )
    query_labels = np.array(["a"])
    gallery_labels = np.array(["a", "b"])

    official = retrieval_view(
        query,
        gallery,
        query_labels,
        gallery_labels,
        coordinates=np.array([0, 1]),
        normalize_before=True,
    )
    corrected = retrieval_view(
        query,
        gallery,
        query_labels,
        gallery_labels,
        coordinates=np.array([0, 1]),
        normalize_before=False,
    )

    assert official.top1_indices.tolist() != corrected.top1_indices.tolist()


def test_random_masks_are_sorted_unique_and_seed_exact() -> None:
    masks = random_masks(dimension=8, selected=4, count=2)
    expected = tuple(
        np.sort(np.random.Generator(np.random.PCG64(seed)).choice(8, 4, replace=False))
        for seed in range(2)
    )

    assert len(masks) == len(expected)
    for actual, oracle in zip(masks, expected, strict=True):
        assert np.array_equal(actual, oracle)
        assert np.array_equal(actual, np.unique(actual))


def test_stable_gallery_order_breaks_exact_distance_ties() -> None:
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    gallery = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    result = retrieval_view(
        query,
        gallery,
        np.array(["right"]),
        np.array(["wrong", "right"]),
        coordinates=np.array([0, 1]),
        normalize_before=False,
    )

    assert result.top1_indices.tolist() == [0]
    assert result.top1_correct.tolist() == [False]


def test_recall_and_map_at_r_match_hand_computed_fixture() -> None:
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )
    result = retrieval_view(
        queries,
        gallery,
        np.array(["a", "b"]),
        np.array(["a", "x", "b", "b"]),
        coordinates=np.array([0, 1]),
        normalize_before=False,
    )

    assert result.recall[1] == 1.0
    assert result.recall[10] == 1.0
    assert result.recall[20] == 1.0
    assert result.recall[30] == 1.0
    assert result.map_at_r == 1.0


@pytest.mark.parametrize(
    (
        "delta_norm",
        "norm_lb",
        "delta_full",
        "full_lb",
        "delta_mask",
        "mask_wins",
        "disagree",
        "primary",
    ),
    [
        (0.002, 1e-9, 0.0, 0.0, 0.002, 24, 0.10, "EVALUATOR_REPAIR"),
        (0.002, 1e-9, 0.002, 1e-9, 0.002, 24, 0.10, "FULL_DIMENSION_CONTROL"),
        (0.0, -1e-9, 0.0, 0.0, 0.002, 24, 0.10, "COORDINATE_NONEXCHANGEABILITY"),
        (0.0, 0.0, 0.0, 0.0, 0.001999999, 32, 1.0, "GEOMETRY_NULL"),
    ],
)
def test_geometry_decision_boundaries(
    delta_norm: float,
    norm_lb: float,
    delta_full: float,
    full_lb: float,
    delta_mask: float,
    mask_wins: int,
    disagree: float,
    primary: str,
) -> None:
    decision = geometry_decision(
        delta_norm=delta_norm,
        norm_lower_bound=norm_lb,
        delta_full=delta_full,
        full_lower_bound=full_lb,
        delta_mask=delta_mask,
        mask_wins=mask_wins,
        disagree=disagree,
    )

    assert decision.primary == primary


def test_reproduction_gate_uses_published_full_768_view() -> None:
    query = np.array([[3.1690565, 3.7732935, 4.94705]], dtype=np.float32)
    gallery = np.array(
        [[4.7299457, 8.226437, 1.7401735], [8.516344, 8.891572, 0.7644352]],
        dtype=np.float32,
    )

    result = audit_deployment_geometry(
        query,
        gallery,
        np.array(["correct"]),
        np.array(["correct", "wrong"]),
        selected=2,
        random_count=2,
        bootstrap_samples=32,
        expected_official_r1=1.0,
        reproduction_tolerance=0.0,
    )

    assert result.official.recall[1] == 0.0
    assert result.full_unit.recall[1] == 1.0
    assert result.reproduction_passed is True


def test_paired_interval_uses_exact_registered_stream() -> None:
    baseline = np.array([False, True, False, True])
    candidate = np.array([True, True, False, True])
    interval = paired_r1_interval(baseline, candidate, samples=10_000, seed=205)

    generator = np.random.Generator(np.random.PCG64(205))
    indices = generator.integers(0, 4, size=(10_000, 4))
    deltas = candidate[indices].mean(axis=1) - baseline[indices].mean(axis=1)
    oracle = np.percentile(deltas, [2.5, 97.5])

    assert interval == pytest.approx((float(oracle[0]), float(oracle[1])))


@pytest.mark.parametrize(
    "values",
    [
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([[np.nan, 1.0]], dtype=np.float32),
        np.array([[np.inf, 1.0]], dtype=np.float32),
        np.array([[1.0, 2.0]], dtype=np.float64),
    ],
)
def test_l2_normalize_rejects_invalid_embeddings(values: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        l2_normalize(values)


def test_deployment_audit_records_negative_gallery_energy_bias() -> None:
    query = np.array([[3.0, 4.0, 12.0]], dtype=np.float32)
    gallery = np.array(
        [[3.0, 4.0, 0.0], [0.0, 5.0, 12.0]],
        dtype=np.float32,
    )
    result = audit_deployment_geometry(
        query,
        gallery,
        np.array(["a"]),
        np.array(["a", "b"]),
        selected=2,
        random_count=2,
        bootstrap_samples=32,
        expected_official_r1=0.0,
        reproduction_tolerance=0.0,
    )

    assert result.reproduction_passed is True
    assert result.energy_disagreement_count == 1
    assert result.energy_gap_mean is not None
    assert result.energy_gap_mean < 0.0


def test_deployment_audit_marks_reproduction_failure_without_scientific_decision() -> None:
    query = np.array([[1.0, 0.1, 0.1, 0.1]], dtype=np.float32)
    gallery = np.array([[1.0, 0.1, 0.1, 0.1]], dtype=np.float32)
    result = audit_deployment_geometry(
        query,
        gallery,
        np.array(["a"]),
        np.array(["a"]),
        selected=2,
        random_count=2,
        bootstrap_samples=32,
        expected_official_r1=0.746,
        reproduction_tolerance=0.002,
    )

    assert result.reproduction_passed is False
    assert result.decision.primary == "REPRODUCTION_FAILED"


def test_deployment_audit_is_exactly_reproducible() -> None:
    gallery = np.array(
        [
            [1.0, 0.1, 0.1, 0.1],
            [0.1, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.1],
            [0.1, 0.1, 0.1, 1.0],
        ],
        dtype=np.float32,
    )
    query = np.ascontiguousarray(gallery[:2])
    labels = np.array(["a", "b", "c", "d"])
    kwargs = {
        "selected": 2,
        "random_count": 2,
        "bootstrap_samples": 64,
        "expected_official_r1": 1.0,
        "reproduction_tolerance": 0.0,
    }

    first = audit_deployment_geometry(query, gallery, labels[:2], labels, **kwargs)
    second = audit_deployment_geometry(query, gallery, labels[:2], labels, **kwargs)

    assert first == second
