from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from sfora.idgp import (
    ARM_ORDER,
    BootstrapSummary,
    CohortEvaluation,
    DensityPools,
    assign_label_folds,
    build_cohorts,
    build_density_pools,
    cluster_bootstrap,
    decide_idgp,
    evaluate_cohort,
    evaluate_idgp,
    geodesic_step,
    global_tangent,
    local_excess_tangent,
    normalize_rows,
    project_one_sided,
    project_two_sided,
    proxy_anchor_surrogate_tangent,
    retrieval_geometry,
    summarize_evaluations,
)


def _independent_fold(label: int) -> int:
    digest = hashlib.sha256(b"LE-IDGP-fold-v1:" + struct.pack("<q", label)).digest()
    return digest[0] & 3


def _labels_in_fold(fold: int, count: int) -> list[int]:
    result: list[int] = []
    label = 0
    while len(result) < count:
        if _independent_fold(label) == fold:
            result.append(label)
        label += 1
    return result


def test_assign_label_folds_uses_frozen_low_bit_hash() -> None:
    labels = np.asarray([91, -7, 0, 91, 12], dtype=np.int64)
    actual = assign_label_folds(labels)
    assert actual == {label: _independent_fold(label) for label in sorted(set(labels.tolist()))}

    with pytest.raises(ValueError, match="int64"):
        assign_label_folds(labels.astype(np.int32))
    with pytest.raises(ValueError, match="one-dimensional"):
        assign_label_folds(labels[:, None])


def test_build_cohorts_is_order_invariant_and_uses_exact_rows() -> None:
    selected_labels = _labels_in_fold(2, 91)
    rows: list[tuple[int, str]] = []
    for label in selected_labels:
        for suffix in ("echo", "alpha", "delta", "bravo", "charlie"):
            rows.append((label, f"label-{label}-{suffix}"))
    labels = np.asarray([row[0] for row in rows], dtype=np.int64)
    ids = np.asarray([row[1] for row in rows])

    cohorts = build_cohorts(labels, ids)
    assert len(cohorts) == 2
    assert [(cohort.fold, cohort.index) for cohort in cohorts] == [(2, 0), (2, 1)]
    assert all(cohort.row_indices.shape == (180,) for cohort in cohorts)

    for label in np.unique(labels[cohorts[0].row_indices]):
        candidates = ids[labels == label].tolist()
        expected = sorted(
            candidates,
            key=lambda value: (
                hashlib.sha256(b"LE-IDGP-row-v1:" + value.encode("utf-8")).digest(),
                value,
            ),
        )[:4]
        assert sorted(
            ids[cohorts[0].row_indices][labels[cohorts[0].row_indices] == label]
        ) == sorted(expected)

    permutation = np.random.default_rng(8).permutation(labels.size)
    permuted = build_cohorts(labels[permutation], ids[permutation])
    assert [ids[c.row_indices].tolist() for c in cohorts] == [
        ids[permutation][c.row_indices].tolist() for c in permuted
    ]
    assert [c.ordered_sha256 for c in cohorts] == [c.ordered_sha256 for c in permuted]


def test_build_cohorts_rejects_invalid_ids_and_discards_incomplete_tail() -> None:
    labels = np.repeat(np.asarray(_labels_in_fold(0, 44), dtype=np.int64), 4)
    ids = np.asarray([f"id-{index}" for index in range(labels.size)])
    assert build_cohorts(labels, ids) == ()

    enough_labels = np.repeat(np.asarray(_labels_in_fold(0, 90), dtype=np.int64), 4)
    duplicate_ids = np.asarray([f"id-{index}" for index in range(enough_labels.size)])
    duplicate_ids[1] = duplicate_ids[0]
    with pytest.raises(ValueError, match="unique"):
        build_cohorts(enough_labels, duplicate_ids)
    with pytest.raises(ValueError, match="Unicode"):
        build_cohorts(enough_labels, np.arange(enough_labels.size, dtype=np.int64))


def test_build_density_pools_is_order_invariant_and_disjoint() -> None:
    labels = np.repeat(np.arange(60, dtype=np.int64), 4)
    ids = np.asarray([f"pool-{label}-{sample}" for label in range(60) for sample in range(4)])
    pools = build_density_pools(labels, ids, pool_size=130)
    assert isinstance(pools, DensityPools)
    assert pools.primary_row_indices.shape == (130,)
    assert pools.alternate_row_indices.shape == (110,)
    assert set(pools.primary_row_indices.tolist()).isdisjoint(pools.alternate_row_indices.tolist())
    permutation = np.random.default_rng(61).permutation(labels.size)
    permuted = build_density_pools(labels[permutation], ids[permutation], pool_size=130)
    assert ids[pools.primary_row_indices].tolist() == ids[permutation][
        permuted.primary_row_indices
    ].tolist()
    assert pools.primary_sha256 == permuted.primary_sha256
    assert pools.alternate_sha256 == permuted.alternate_sha256


def _density(anchor: np.ndarray, peers: np.ndarray, peer_ids: np.ndarray, k: int) -> float:
    similarities = peers @ anchor
    order = sorted(range(peers.shape[0]), key=lambda index: (-similarities[index], peer_ids[index]))
    return float(similarities[order[:k]].mean() - similarities.mean())


def test_local_excess_tangent_matches_stopped_peer_finite_difference() -> None:
    z = normalize_rows(
        np.asarray(
            [
                [1.0, 0.2, -0.1],
                [0.9, 0.3, 0.1],
                [0.8, 0.5, -0.2],
                [0.1, 1.0, 0.1],
                [-0.4, 0.7, 0.6],
                [-0.8, 0.1, 0.5],
            ],
            dtype=np.float64,
        )
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    ids = np.asarray(["a", "b", "c", "d", "e", "f"])
    tangent, density = local_excess_tangent(
        z[:2], labels[:2], z[2:], labels[2:], ids[2:], k=2
    )

    peers = z[labels != labels[0]]
    peer_ids = ids[labels != labels[0]]
    numeric = np.empty(z.shape[1], dtype=np.float64)
    for axis in range(z.shape[1]):
        offset = np.zeros(z.shape[1], dtype=np.float64)
        offset[axis] = 1e-6
        plus = normalize_rows((z[0] + offset)[None, :])[0]
        minus = normalize_rows((z[0] - offset)[None, :])[0]
        numeric[axis] = (
            _density(plus, peers, peer_ids, 2) - _density(minus, peers, peer_ids, 2)
        ) / (2e-6)
    np.testing.assert_allclose(tangent[0], numeric, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(np.sum(tangent * z[:2], axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(density[0], _density(z[0], peers, peer_ids, 2), rtol=0.0, atol=1e-15)


def test_local_plus_global_recovers_topk_mean_tangent_with_stable_ties() -> None:
    z = normalize_rows(
        np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.5, -0.5], [-1.0, 0.0]],
            dtype=np.float64,
        )
    )
    labels = np.asarray([0, 0, 1, 2, 3], dtype=np.int64)
    ids = np.asarray(["anchor", "positive", "z-tie", "a-tie", "far"])
    local, _ = local_excess_tangent(z[:2], labels[:2], z[2:], labels[2:], ids[2:], k=1)
    global_value = global_tangent(z[:2], labels[:2], z[2:], labels[2:])
    selected = z[3]
    expected_top = selected - z[0] * float(z[0] @ selected)
    np.testing.assert_allclose(local[0] + global_value[0], expected_top, atol=1e-12)


def test_one_sided_projection_removes_only_conflicting_components() -> None:
    g = np.asarray([[-1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [2.0, 1.0]])
    h = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1e-10, 0.0]])
    projected, dots, skipped = project_one_sided(g, h, cutoff=1e-8)
    np.testing.assert_allclose(dots, [-1.0, 1.0, 0.0, 2e-10])
    np.testing.assert_allclose(projected[0] @ h[0], 0.0, atol=1e-15)
    np.testing.assert_array_equal(projected[1:], g[1:])
    np.testing.assert_array_equal(skipped, [False, False, True, True])

    two_sided = project_two_sided(g, h, cutoff=1e-8)
    np.testing.assert_allclose(two_sided[0] @ h[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(two_sided[1] @ h[1], 0.0, atol=1e-15)
    np.testing.assert_array_equal(two_sided[2:], g[2:])


def test_geodesic_step_tangent_matches_all_nonzero_arm_distances() -> None:
    z = normalize_rows(np.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
    directions = np.asarray([[4.0, -1.0, 2.0], [3.0, 7.0, -2.0]])
    stepped = geodesic_step(z, directions, epsilon=0.01)
    angles = np.arccos(np.clip(np.sum(z * stepped, axis=1), -1.0, 1.0))
    np.testing.assert_allclose(angles, 0.01, atol=2e-14)
    np.testing.assert_allclose(np.linalg.norm(stepped, axis=1), 1.0, atol=1e-15)

    radial = geodesic_step(z, z * 5.0, epsilon=0.01)
    np.testing.assert_allclose(radial, z, rtol=0.0, atol=2e-16)


def test_retrieval_geometry_moves_only_the_supplied_anchor() -> None:
    anchor = normalize_rows(np.asarray([[1.0, 0.2]]))[0]
    peers = normalize_rows(np.asarray([[0.9, 0.1], [0.7, 0.7], [-1.0, 0.0]]))
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    peers_before = peers.copy()
    margin_before, positive_before = retrieval_geometry(anchor, peers, labels, 0)
    moved = geodesic_step(anchor[None, :], np.asarray([[0.0, 1.0]]), epsilon=0.01)[0]
    margin_after, positive_after = retrieval_geometry(moved, peers, labels, 0)
    np.testing.assert_array_equal(peers, peers_before)
    assert margin_before != margin_after
    assert positive_before != positive_after


def test_proxy_anchor_surrogate_gradient_is_sphere_tangent() -> None:
    z = normalize_rows(
        np.asarray(
            [
                [1.0, 0.0, 0.1],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.1],
                [0.1, 0.9, 0.0],
            ]
        )
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    gradient = proxy_anchor_surrogate_tangent(z, labels)
    np.testing.assert_allclose(np.sum(gradient * z, axis=1), 0.0, atol=1e-12)
    assert np.linalg.norm(gradient) > 0.0


def _synthetic_cohort(label_offset: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    bases = normalize_rows(rng.normal(size=(45, 16)))
    rows: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    for position, base in enumerate(bases):
        label = label_offset + position
        for sample in range(4):
            rows.append(base + 0.08 * rng.normal(size=16))
            labels.append(label)
            ids.append(f"label-{label}-sample-{sample}")
    return (
        normalize_rows(np.asarray(rows)),
        np.asarray(labels, dtype=np.int64),
        np.asarray(ids),
    )


def test_evaluate_cohort_uses_fixed_arms_primary_rows_and_independent_controls() -> None:
    z, labels, ids = _synthetic_cohort(0, 12)
    reference_z, reference_labels, reference_ids = _synthetic_cohort(100, 13)
    z_before = z.copy()
    evaluation = evaluate_cohort(
        z,
        labels,
        ids,
        fold=1,
        index=2,
        density_embeddings=reference_z,
        density_labels=reference_labels,
        density_ids=reference_ids,
        alternate_embeddings=_synthetic_cohort(200, 14)[0],
        alternate_labels=_synthetic_cohort(200, 14)[1],
        alternate_ids=_synthetic_cohort(200, 14)[2],
    )
    repeated = evaluate_cohort(
        z,
        labels,
        ids,
        fold=1,
        index=2,
        density_embeddings=reference_z,
        density_labels=reference_labels,
        density_ids=reference_ids,
        alternate_embeddings=_synthetic_cohort(200, 14)[0],
        alternate_labels=_synthetic_cohort(200, 14)[1],
        alternate_ids=_synthetic_cohort(200, 14)[2],
    )
    np.testing.assert_array_equal(z, z_before)
    assert tuple(evaluation.margin_changes) == ARM_ORDER
    assert tuple(evaluation.positive_similarity_changes) == ARM_ORDER
    bottom = np.zeros(180, dtype=np.bool_)
    ordered = sorted(range(180), key=lambda row: (evaluation.pre_margins[row], ids[row]))
    bottom[ordered[:45]] = True
    np.testing.assert_array_equal(
        evaluation.primary_mask,
        evaluation.bottom_mask & (evaluation.conflict_dots < 0.0) & ~evaluation.skipped_mask,
    )
    np.testing.assert_array_equal(evaluation.bottom_mask, bottom)
    for arm in ARM_ORDER:
        np.testing.assert_array_equal(evaluation.margin_changes[arm], repeated.margin_changes[arm])
    assert not np.array_equal(
        evaluation.margin_changes["shuffled_local"],
        evaluation.margin_changes["random_tangent"],
    )
    assert np.isfinite(evaluation.reference_conflict_fraction)
    assert np.isfinite(evaluation.reference_cosine_mean)
    np.testing.assert_array_equal(
        evaluation.margin_changes["le_idgp"][evaluation.primary_mask],
        evaluation.margin_changes["two_sided"][evaluation.primary_mask],
    )
    assert not np.array_equal(
        evaluation.margin_changes["le_idgp"], evaluation.margin_changes["two_sided"]
    )


def test_alternate_pool_changes_only_reference_diagnostics() -> None:
    z, labels, ids = _synthetic_cohort(0, 18)
    reference_a = _synthetic_cohort(100, 19)
    reference_b = _synthetic_cohort(200, 20)
    first = evaluate_cohort(
        z,
        labels,
        ids,
        fold=0,
        index=0,
        density_embeddings=reference_a[0],
        density_labels=reference_a[1],
        density_ids=reference_a[2],
        alternate_embeddings=reference_b[0],
        alternate_labels=reference_b[1],
        alternate_ids=reference_b[2],
    )
    second = evaluate_cohort(
        z,
        labels,
        ids,
        fold=0,
        index=0,
        density_embeddings=reference_a[0],
        density_labels=reference_a[1],
        density_ids=reference_a[2],
        alternate_embeddings=_synthetic_cohort(300, 21)[0],
        alternate_labels=_synthetic_cohort(300, 21)[1],
        alternate_ids=_synthetic_cohort(300, 21)[2],
    )
    for arm in ARM_ORDER:
        np.testing.assert_array_equal(first.margin_changes[arm], second.margin_changes[arm])
    assert (
        first.reference_conflict_fraction != second.reference_conflict_fraction
        or first.reference_cosine_mean != second.reference_cosine_mean
    )


def _independent_cluster_replicates(
    values: np.ndarray, labels: np.ndarray, seed: int, replicates: int
) -> np.ndarray:
    unique = np.asarray(sorted(np.unique(labels).tolist()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    result = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = unique[rng.integers(0, unique.size, size=unique.size)]
        rows = np.concatenate([values[labels == label] for label in selected])
        result[replicate] = rows.mean()
    return result


def test_cluster_bootstrap_resamples_labels_and_hashes_exact_replicates() -> None:
    values = np.asarray([1.0, 2.0, 10.0, -3.0, -2.0, -1.0], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int64)
    expected = _independent_cluster_replicates(values, labels, seed=31, replicates=50)
    actual = cluster_bootstrap(values, labels, seed=31, replicates=50)
    assert isinstance(actual, BootstrapSummary)
    np.testing.assert_array_equal(actual.replicates, expected)
    assert actual.mean == float(values.mean())
    assert actual.lower_bound == float(np.percentile(expected, 1.0, method="linear"))
    assert actual.standard_error == float(expected.std(ddof=1))
    assert actual.mde == 2.576 * actual.standard_error
    assert actual.label_count == 3
    assert actual.median_rows_per_label == 2.0
    assert actual.replicate_sha256 == hashlib.sha256(expected.astype("<f8").tobytes()).hexdigest()

    permutation = np.asarray([5, 0, 3, 2, 1, 4])
    permuted = cluster_bootstrap(values[permutation], labels[permutation], seed=31, replicates=50)
    np.testing.assert_array_equal(permuted.replicates, expected)


def _passing_summary() -> dict[str, object]:
    return {
        "coverage": {
            "eligible_rows": 720,
            "conflicting_rows": 120,
            "conflict_fraction": 1.0 / 6.0,
            "primary_rows": 110,
            "primary_labels": 105,
            "folds_with_primary": 4,
        },
        "fold_mean_advantages": [0.01, 0.02, 0.01, -0.001],
        "pooled": {
            "raw_advantage_mean": 0.01,
            "raw_advantage_lower": 0.004,
            "shuffled_contrast_lower": 0.003,
            "random_contrast_lower": 0.003,
            "global_contrast_lower": 0.002,
            "zero_abs_margin_change_median": 0.1,
            "positive_similarity_mean": -5e-5,
            "positive_similarity_lower": -4e-4,
            "le_minus_two_sided_mean": 0.001,
        },
    }


@pytest.mark.parametrize(
    ("path", "value", "predicate"),
    [
        (("coverage", "conflict_fraction"), 0.09, "coverage"),
        (("coverage", "primary_labels"), 99, "coverage"),
        (("pooled", "raw_advantage_lower"), 0.0, "raw_advantage"),
        (("fold_mean_advantages", 2), -0.01, "fold_consistency"),
        (("pooled", "shuffled_contrast_lower"), 0.0, "control_superiority"),
        (("pooled", "random_contrast_lower"), 0.0, "control_superiority"),
        (("pooled", "global_contrast_lower"), 0.0, "control_superiority"),
        (("pooled", "raw_advantage_mean"), 0.0049, "material_effect"),
        (("pooled", "positive_similarity_lower"), -0.0006, "positive_similarity"),
        (("pooled", "le_minus_two_sided_mean"), -1e-6, "one_sided_specificity"),
    ],
)
def test_decide_idgp_rejects_each_frozen_predicate(
    path: tuple[str | int, ...], value: float, predicate: str
) -> None:
    summary = _passing_summary()
    node: object = summary
    for key in path[:-1]:
        node = node[key]  # type: ignore[index]
    node[path[-1]] = value  # type: ignore[index]
    passed, predicates = decide_idgp(summary)
    assert passed is False
    failed = [record["name"] for record in predicates if record["passed"] is False]
    assert predicate in failed


def test_decide_idgp_accepts_only_complete_passing_summary() -> None:
    passed, predicates = decide_idgp(_passing_summary())
    assert passed is True
    assert [record["name"] for record in predicates] == [
        "coverage",
        "raw_advantage",
        "fold_consistency",
        "control_superiority",
        "material_effect",
        "positive_similarity",
        "one_sided_specificity",
    ]
    assert all(record["passed"] is True for record in predicates)


def _evaluation_for_summary(fold: int, advantage: float) -> CohortEvaluation:
    rows = 8
    labels = np.asarray([fold * 10 + value for value in (0, 0, 1, 1, 2, 2, 3, 3)])
    primary = np.asarray([True, True, True, True, False, False, False, False])
    zero = np.asarray([0.02, -0.01, 0.03, -0.02, 0.0, 0.0, 0.0, 0.0])
    le = zero + advantage
    margin_changes = {
        "zero": zero,
        "le_idgp": le,
        "shuffled_local": zero + advantage / 4.0,
        "random_tangent": zero + advantage / 5.0,
        "global_centering": zero + advantage / 3.0,
        # One- and two-sided projections coincide on conflict rows.  Safe
        # bottom-quartile rows carry the actual specificity contrast.
        "two_sided": np.concatenate([le[:4], zero[4:] - advantage]),
    }
    positive = {arm: np.zeros(rows, dtype=np.float64) for arm in ARM_ORDER}
    return CohortEvaluation(
        fold=fold,
        index=0,
        example_ids=np.asarray([f"f{fold}-r{row}" for row in range(rows)]),
        labels=labels.astype(np.int64),
        pre_margins=np.linspace(-1.0, 1.0, rows),
        conflict_dots=np.asarray([-1.0] * 4 + [1.0] * 4),
        skipped_mask=np.zeros(rows, dtype=np.bool_),
        bottom_mask=np.ones(rows, dtype=np.bool_),
        primary_mask=primary,
        margin_changes=margin_changes,
        positive_similarity_changes=positive,
        analytic_density_reduction=np.ones(rows),
        local_norms=np.ones(rows),
        global_norms=np.ones(rows),
        local_global_cosine_mean=0.1,
        collective_idgp_advantage=0.0,
        reference_conflict_fraction=0.25,
        reference_cosine_mean=0.2,
    )


def test_summarize_evaluations_recomputes_primary_contrasts() -> None:
    evaluations = [_evaluation_for_summary(fold, 0.01 + 0.001 * fold) for fold in range(4)]
    summary = summarize_evaluations(evaluations, bootstrap_replicates=100)
    assert summary["coverage"]["eligible_rows"] == 32
    assert summary["coverage"]["conflicting_rows"] == 16
    assert summary["coverage"]["primary_rows"] == 16
    assert summary["coverage"]["primary_labels"] == 8
    np.testing.assert_allclose(summary["fold_mean_advantages"], [0.01, 0.011, 0.012, 0.013])
    assert summary["pooled"]["raw_advantage_mean"] == pytest.approx(0.0115)
    assert summary["pooled"]["le_minus_two_sided_mean"] == pytest.approx(0.0115)


def test_evaluate_idgp_builds_complete_cohorts_and_frozen_decision() -> None:
    selected_labels = _labels_in_fold(3, 90)
    rng = np.random.default_rng(81)
    bases = normalize_rows(rng.normal(size=(90, 64)))
    rows: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    for position, label in enumerate(selected_labels):
        for sample in range(4):
            rows.append(bases[position] + 0.04 * rng.normal(size=64))
            labels.append(label)
            ids.append(f"id-{label}-{sample}")
    result = evaluate_idgp(
        normalize_rows(np.asarray(rows)),
        np.asarray(labels, dtype=np.int64),
        np.asarray(ids),
        density_pool_size=180,
        bootstrap_replicates=100,
    )
    assert len(result.evaluations) == 2
    assert result.summary["coverage"]["eligible_rows"] == 360
    assert result.passed is (result.status == "PASS")
    assert result.status in {"PASS", "KILL"}
    assert [record["name"] for record in result.predicates] == [
        "coverage",
        "raw_advantage",
        "fold_consistency",
        "control_superiority",
        "material_effect",
        "positive_similarity",
        "one_sided_specificity",
    ]


def test_one_sided_projection_prevents_conflicting_margin_damage() -> None:
    anchor = np.asarray([[1.0, 0.0]])
    positive = np.asarray([[np.cos(0.15), np.sin(0.15)]])
    foreign = np.asarray([[np.cos(-0.05), np.sin(-0.05)]])
    peers = np.concatenate([positive, foreign], axis=0)
    peer_labels = np.asarray([0, 1], dtype=np.int64)
    nuisance = np.asarray([[0.0, -1.0]])
    conflicting_gradient = -nuisance
    projected, dots, skipped = project_one_sided(conflicting_gradient, nuisance)
    assert dots[0] < 0.0
    assert skipped[0] is np.False_
    raw_anchor = geodesic_step(anchor, conflicting_gradient)[0]
    idgp_anchor = geodesic_step(anchor, projected)[0]
    raw_margin, _ = retrieval_geometry(raw_anchor, peers, peer_labels, 0)
    idgp_margin, _ = retrieval_geometry(idgp_anchor, peers, peer_labels, 0)
    assert idgp_margin > raw_margin
