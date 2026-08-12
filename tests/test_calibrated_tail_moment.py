from __future__ import annotations

import numpy as np
import pytest

from sfora.calibrated_tail_moment import (
    cluster_lambda_interval,
    ctm_decision,
    encode_tail_moment,
    evaluate_inner_product,
    evaluate_width,
    fit_projection_basis,
    fit_tail_moment,
    head_neighbor_pairs,
    permuted_tail_null,
    project_unit,
    query_identity_interval,
)
from sfora.unicom_retrieval_audit import retrieval_view


def _unit_fixture() -> np.ndarray:
    return np.asarray(
        [
            [0.8, 0.0, 0.6, 0.0],
            [0.8, 0.0, 0.0, 0.6],
            [0.0, 0.8, 0.6, 0.0],
            [0.0, 0.8, 0.0, 0.6],
        ],
        dtype=np.float32,
    )


def test_head_pairs_use_only_head_inner_product_and_stable_row_ties() -> None:
    unit = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    pairs = head_neighbor_pairs(unit, width=2, neighbors=1)

    assert pairs.tolist() == [[0, 1], [1, 0], [2, 0]]


def test_fit_matches_the_registered_ordered_fp64_formula() -> None:
    unit = np.asarray(
        [
            [0.8, 0.0, 0.6, 0.0],
            [0.8, 0.0, 0.0, 0.6],
            [0.0, 0.8, 0.6, 0.0],
        ],
        dtype=np.float32,
    )
    fit = fit_tail_moment(unit, width=2, basis_kind="native", neighbors=1)
    pairs = fit.pairs
    tail = unit[:, 2:].astype(np.float64)
    radius = np.linalg.norm(tail, axis=1)
    x = radius[pairs[:, 0]] * radius[pairs[:, 1]]
    y = np.sum(tail[pairs[:, 0]] * tail[pairs[:, 1]], axis=1, dtype=np.float64)
    expected = float(np.sum(x * y, dtype=np.float64) / np.sum(x * x, dtype=np.float64))

    assert fit.lambda_raw == expected
    assert fit.lambda_value == min(1.0, max(0.0, expected))
    encoded = encode_tail_moment(unit, fit, basis_kind="native")
    assert encoded.shape == (3, 3)
    assert encoded.dtype == np.float32


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fit_rejects_nonfinite_values(bad: float) -> None:
    values = np.eye(3, dtype=np.float32)
    values[0, 0] = bad

    with pytest.raises(ValueError, match="finite"):
        fit_tail_moment(values, width=1, basis_kind="native", neighbors=1)


def test_zero_denominator_has_zero_coefficient_and_exact_zero_scalar() -> None:
    unit = np.eye(3, dtype=np.float32)
    fit = fit_tail_moment(unit, width=2, basis_kind="native", neighbors=1)

    encoded = encode_tail_moment(unit, fit, basis_kind="native")

    assert fit.lambda_value == 0.0
    assert encoded[:, -1].tolist() == [0.0, 0.0, 0.0]


def test_pair_selection_cannot_observe_tail() -> None:
    unit = _unit_fixture()
    changed = unit.copy()
    changed[:, 2:] = changed[[1, 0, 3, 2], 2:]

    assert np.array_equal(
        head_neighbor_pairs(unit, width=2, neighbors=1),
        head_neighbor_pairs(changed, width=2, neighbors=1),
    )


def test_negative_raw_fit_clips_to_zero_and_encoding_is_exact_lambda_zero() -> None:
    unit = np.asarray([[0.8, 0.0, 0.6, 0.0], [0.8, 0.0, -0.6, 0.0]], dtype=np.float32)
    fit = fit_tail_moment(unit, width=2, basis_kind="native", neighbors=1)

    encoded = encode_tail_moment(unit, fit, basis_kind="native")

    assert fit.lambda_raw < 0.0
    assert fit.lambda_value == 0.0
    assert encoded[:, -1].tobytes() == np.zeros(2, dtype=np.float32).tobytes()


def test_encoding_rejects_wrong_dimension_and_basis() -> None:
    fit = fit_tail_moment(_unit_fixture(), width=2, basis_kind="native", neighbors=1)

    with pytest.raises(ValueError, match="basis/dimension"):
        encode_tail_moment(_unit_fixture(), fit, basis_kind="pca")
    with pytest.raises(ValueError, match="basis/dimension"):
        encode_tail_moment(np.eye(3, dtype=np.float32), fit, basis_kind="native")


def test_neighbor_selection_only_sorts_the_requested_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False, dtype=np.float32)
    unit = np.stack((np.cos(angles), np.sin(angles), np.zeros(64)), axis=1).astype(np.float32)
    original = np.lexsort
    sizes: list[int] = []

    def bounded(keys):
        sizes.append(len(keys[0]))
        assert len(keys[0]) == 4
        return original(keys)

    monkeypatch.setattr(np, "lexsort", bounded)

    pairs = head_neighbor_pairs(unit, width=2, neighbors=4, chunk_size=7)

    assert pairs.shape == (256, 2)
    assert sizes == [4] * 64


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.astype(np.float64), "float32"),
        (lambda value: np.asfortranarray(value), "C-contiguous"),
        (lambda value: value[0], "matrix"),
        (lambda value: value * np.float32(0.5), "unit"),
    ],
)
def test_fit_rejects_invalid_embedding_contract(mutation, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        fit_tail_moment(mutation(_unit_fixture()), width=2, basis_kind="native", neighbors=1)


@pytest.mark.parametrize(
    "width, neighbors",
    [(True, 1), (0, 1), (4, 1), (2, True), (2, 0), (2, 4)],
)
def test_fit_rejects_invalid_width_or_neighbor_count(width, neighbors) -> None:
    with pytest.raises((TypeError, ValueError)):
        fit_tail_moment(_unit_fixture(), width=width, basis_kind="native", neighbors=neighbors)


def test_pca_basis_is_fit_from_train_only_and_is_sign_canonical() -> None:
    train = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

    basis = fit_projection_basis(train, kind="pca")

    assert basis.kind == "pca"
    assert basis.mean.dtype == np.float32
    assert basis.matrix.dtype == np.float32
    assert basis.mean.flags.c_contiguous
    assert basis.matrix.flags.c_contiguous
    for column in basis.matrix.T:
        pivot = int(np.argmax(np.abs(column)))
        assert column[pivot] >= 0.0


def test_native_projection_returns_the_already_normalized_input_unchanged() -> None:
    train = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.asarray([[0.6, 0.8]], dtype=np.float32)
    basis = fit_projection_basis(train, kind="native")

    projected = project_unit(query, basis)

    assert projected is query


def test_pca_projection_subtracts_only_the_frozen_train_mean() -> None:
    train = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    query = np.asarray([[0.6, 0.8]], dtype=np.float32)
    basis = fit_projection_basis(train, kind="pca")
    expected = (query.astype(np.float64) - basis.mean.astype(np.float64)) @ basis.matrix
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)

    projected = project_unit(query, basis)

    assert np.allclose(projected, expected.astype(np.float32), rtol=0.0, atol=2e-7)


def test_cluster_interval_reuses_observed_pairs_and_uses_two_way_identity_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = np.asarray(
        [
            [0.8, 0.0, 0.6, 0.0],
            [0.8, 0.0, 0.6, 0.0],
            [0.0, 0.8, 0.6, 0.0],
            [0.0, 0.8, 0.0, 0.6],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(["a", "a", "b", "b"])
    calls = 0
    original = head_neighbor_pairs

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr("sfora.calibrated_tail_moment.head_neighbor_pairs", counted)

    interval = cluster_lambda_interval(
        unit,
        labels,
        width=2,
        basis_kind="native",
        samples=1,
        seed=1,
        neighbors=1,
    )

    assert calls == 1
    assert interval.samples == 1
    assert interval.seed == 1
    assert interval.point == 0.5
    assert interval.lower == 0.0
    assert interval.upper == 0.0


def test_tail_null_keeps_radii_and_pairs_fixed_and_reports_exact_p_value() -> None:
    unit = np.asarray(
        [
            [0.8, 0.0, 0.6, 0.0],
            [0.6, 0.0, 0.0, 0.8],
            [0.0, 0.9539392, 0.3, 0.0],
        ],
        dtype=np.float32,
    )
    observed = fit_tail_moment(unit, width=2, basis_kind="native", neighbors=1)

    null = permuted_tail_null(
        unit,
        width=2,
        basis_kind="native",
        neighbors=1,
        seeds=range(206, 210),
    )

    tail = unit[:, 2:].astype(np.float64)
    radius = np.linalg.norm(tail, axis=1)
    direction = tail / radius[:, None]
    permutation = np.random.Generator(np.random.PCG64(206)).permutation(unit.shape[0])
    permuted_tail = radius[:, None] * direction[permutation]
    pairs = observed.pairs
    x = radius[pairs[:, 0]] * radius[pairs[:, 1]]
    y = np.sum(
        permuted_tail[pairs[:, 0]] * permuted_tail[pairs[:, 1]],
        axis=1,
        dtype=np.float64,
    )
    expected_first = float(np.sum(x * y) / np.sum(x * x))
    expected_p = (1 + sum(value >= observed.lambda_raw for value in null.lambda_raw_values)) / 5

    assert np.array_equal(null.pairs, observed.pairs)
    assert null.seeds == (206, 207, 208, 209)
    assert null.lambda_raw_values[0] == expected_first
    assert null.p_value == expected_p


@pytest.mark.parametrize("kind", ["", "PCA", 1, True])
def test_projection_rejects_unknown_basis_kind(kind) -> None:
    with pytest.raises((TypeError, ValueError), match="kind"):
        fit_projection_basis(_unit_fixture(), kind=kind)


def test_inner_product_breaks_score_ties_by_gallery_row() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    gallery = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    view = evaluate_inner_product(
        query,
        gallery,
        np.asarray(["x"]),
        np.asarray(["y", "x"]),
        values_per_row=2,
    )

    assert view.top1_indices.tolist() == [0]
    assert view.top1_correct.tolist() == [False]


def test_inner_product_matches_existing_unit_evaluator_metrics() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    query_labels = np.asarray(["a", "b"])
    gallery_labels = np.asarray(["a", "x", "b", "b"])
    existing = retrieval_view(
        query,
        gallery,
        query_labels,
        gallery_labels,
        coordinates=np.asarray([0, 1], dtype=np.int64),
        normalize_before=False,
    )

    candidate = evaluate_inner_product(
        query,
        gallery,
        query_labels,
        gallery_labels,
        values_per_row=2,
    )

    assert candidate.recall == existing.recall
    assert candidate.map_at_r == existing.map_at_r
    assert np.array_equal(candidate.top1_indices, existing.top1_indices)
    assert np.array_equal(candidate.top1_correct, existing.top1_correct)
    assert candidate.values_per_row == 2
    assert candidate.total_bytes == gallery.nbytes


def test_query_bootstrap_keeps_gallery_fixed_and_clusters_identity_rows() -> None:
    labels = np.asarray(["a", "a", "b", "c"])
    baseline = np.asarray([False, True, False, True], dtype=np.bool_)
    candidate = np.asarray([True, True, False, True], dtype=np.bool_)

    result = query_identity_interval(baseline, candidate, labels, samples=10_000, seed=205)

    assert result.samples == 10_000
    assert result.seed == 205
    assert result.point == 0.25
    assert result.lower <= result.point <= result.upper


def test_width_grid_has_exact_controls_and_storage_accounting() -> None:
    train = _unit_fixture()
    query = train[:2]
    gallery = train[2:]
    query_labels = np.asarray(["a", "b"])
    gallery_labels = np.asarray(["a", "b"])
    native_basis = fit_projection_basis(train, kind="native")
    pca_basis = fit_projection_basis(train, kind="pca")
    train_pca = project_unit(train, pca_basis)
    query_pca = project_unit(query, pca_basis)
    gallery_pca = project_unit(gallery, pca_basis)
    native_fit = fit_tail_moment(train, width=2, basis_kind="native", neighbors=1)
    pca_fit = fit_tail_moment(train_pca, width=2, basis_kind="pca", neighbors=1)
    pca_bytes = pca_basis.mean.nbytes + pca_basis.matrix.nbytes

    views = evaluate_width(
        query,
        gallery,
        query_pca,
        gallery_pca,
        query_labels,
        gallery_labels,
        width=2,
        native_fit=native_fit,
        pca_fit=pca_fit,
        pca_fixed_bytes=pca_bytes,
        official_width=3,
    )

    assert tuple(views) == (
        "native_ctm",
        "pca_ctm",
        "renormalized_prefix_plus_zero",
        "plain_prefix_plus_zero",
        "lambda_zero",
        "unicom_tail_energy",
        "pca_renormalized_prefix_plus_zero",
        "tail_sign_control",
        "tail_permuted_control",
        "official_512",
        "full_width_768",
    )
    assert views["native_ctm"].values_per_row == 3
    assert views["native_ctm"].total_bytes == gallery.shape[0] * 3 * 4
    assert views["pca_ctm"].total_bytes == gallery.shape[0] * 3 * 4 + pca_bytes
    assert views["pca_renormalized_prefix_plus_zero"].fixed_bytes == pca_bytes
    assert views["official_512"].values_per_row == 3
    assert views["full_width_768"].values_per_row == 4
    for view in views.values():
        assert type(view.descriptor_build_seconds) is float
        assert view.descriptor_build_seconds >= 0.0
        assert type(view.search_seconds) is float
        assert view.search_seconds >= 0.0
    assert native_basis.kind == "native"


def test_registered_width_may_equal_official_geometry_width() -> None:
    train = _unit_fixture()
    basis = fit_projection_basis(train, kind="pca")
    projected = project_unit(train, basis)

    views = evaluate_width(
        train[:2],
        train[2:],
        projected[:2],
        projected[2:],
        np.asarray(["a", "b"]),
        np.asarray(["a", "b"]),
        width=2,
        native_fit=fit_tail_moment(train, width=2, basis_kind="native", neighbors=1),
        pca_fit=fit_tail_moment(projected, width=2, basis_kind="pca", neighbors=1),
        pca_fixed_bytes=basis.mean.nbytes + basis.matrix.nbytes,
        official_width=2,
    )

    assert views["native_ctm"].values_per_row == 3
    assert views["official_512"].values_per_row == 2


def _passing_decision(**changes):
    values = {
        "ctm_r1": 0.91,
        "ctm_map_at_r": 0.70,
        "renormalized_r1": 0.906,
        "renormalized_map_at_r": 0.70,
        "official_512_r1": 0.912,
        "paired_lower": 0.001,
        "control_r1": {
            "renormalized_prefix_plus_zero": 0.906,
            "plain_prefix_plus_zero": 0.905,
            "lambda_zero": 0.905,
            "unicom_tail_energy": 0.904,
            "pca_renormalized_prefix_plus_zero": 0.908,
            "tail_sign_control": 0.90,
            "tail_permuted_control": 0.899,
        },
        "ctm_total_bytes": 516,
        "control_total_bytes": {
            "renormalized_prefix_plus_zero": 516,
            "plain_prefix_plus_zero": 516,
            "lambda_zero": 516,
            "unicom_tail_energy": 516,
            "tail_sign_control": 516,
            "tail_permuted_control": 516,
        },
        "lambda_raw": 0.2,
        "lambda_lower": 0.1,
        "null_p_value": 1.0 / 33.0,
        "width_gains": ((64, 0.0), (128, 0.004), (256, 0.0), (512, 0.0)),
        "replication_status": "PENDING",
    }
    values.update(changes)
    return ctm_decision(**values)


def test_decision_passes_exact_boundaries_and_waits_for_replication() -> None:
    decision = _passing_decision(
        ctm_r1=0.909,
        ctm_map_at_r=0.699,
        official_512_r1=0.912,
        width_gains=((64, 0.0), (128, 0.003), (256, 0.0), (512, 0.0)),
        null_p_value=0.05,
    )

    assert decision.status == "REPLICATE"
    assert decision.r1_gain_passed is True
    assert decision.map_passed is True
    assert decision.gap_recovery_passed is True
    assert decision.some_width_signal_passed is True


def test_decision_uses_simpler_descriptor_when_it_matches_official_anchor() -> None:
    decision = _passing_decision(renormalized_r1=0.912, official_512_r1=0.912)

    assert decision.status == "USE_RENORMALIZED"


def test_decision_rejects_a_storage_mismatched_control() -> None:
    control_bytes = {
        "renormalized_prefix_plus_zero": 516,
        "plain_prefix_plus_zero": 516,
        "lambda_zero": 516,
        "unicom_tail_energy": 516,
        "tail_sign_control": 516,
        "tail_permuted_control": 520,
    }

    decision = _passing_decision(control_total_bytes=control_bytes)

    assert decision.equal_storage_passed is False
    assert decision.status == "CLOSE"


@pytest.mark.parametrize(
    "changes",
    [
        {"lambda_raw": 0.0},
        {"lambda_lower": 0.0},
        {"null_p_value": 0.051},
        {"paired_lower": 0.0},
        {"width_gains": ((64, 0.0009), (128, 0.0009))},
        {"replication_status": "FAILED"},
    ],
)
def test_decision_closes_on_each_falsifier(changes) -> None:
    assert _passing_decision(**changes).status == "CLOSE"


def test_decision_becomes_general_only_after_replication() -> None:
    assert _passing_decision(replication_status="PASSED").status == "GENERAL_CLAIM_READY"
