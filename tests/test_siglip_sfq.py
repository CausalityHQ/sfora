"""Tests for the optimization-only Shrunk-Fisher-Quotient diagnostic."""

from __future__ import annotations

import json
import math

import pytest
import torch

from sfora.siglip_head_screen import FeatureSplitAuthority, build_feature_split_authority
from sfora.siglip_sfq import (
    _spectral_projection,
    build_sfq_fold_schedule,
    fit_sfq_projection,
    run_sfq_fold_diagnostic,
    validate_sfq_result_bytes,
)


def _four_pair_features() -> tuple[torch.Tensor, torch.Tensor]:
    centers = (
        (1.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
        (1.00, -0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
        (0.00, 0.00, 1.00, 0.05, 0.00, 0.00, 0.00, 0.00),
        (0.00, 0.00, 1.00, -0.05, 0.00, 0.00, 0.00, 0.00),
        (0.00, 0.00, 0.00, 0.00, 1.00, 0.05, 0.00, 0.00),
        (0.00, 0.00, 0.00, 0.00, 1.00, -0.05, 0.00, 0.00),
        (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.05),
        (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, -0.05),
    )
    rows: list[torch.Tensor] = []
    labels: list[int] = []
    for label, center in enumerate(centers):
        base = torch.tensor(center, dtype=torch.float32)
        for offset in (-0.01, 0.0, 0.01):
            row = base.clone()
            row[(2 * (label // 2) + 1) % row.numel()] += offset
            rows.append(row)
            labels.append(label)
    return torch.stack(rows), torch.tensor(labels, dtype=torch.int64)


def _authority(
    features: torch.Tensor, *, role: str = "optimization-train"
) -> FeatureSplitAuthority:
    return build_feature_split_authority(
        source_manifest_sha256="1" * 64,
        role=role,
        official_test_access=False,
        ordered_example_ids=tuple(f"row-{row:04d}" for row in range(features.shape[0])),
        features=features,
    )


def test_sfq_folds_are_deterministic_class_disjoint_and_keep_twin_pairs() -> None:
    """A broken edge order or allocator must not leak classes or split nearest twins."""

    features, labels = _four_pair_features()

    first = build_sfq_fold_schedule(features, labels, _authority(features), fold_count=4)
    second = build_sfq_fold_schedule(features, labels, _authority(features), fold_count=4)

    assert first == second
    assert len(first.sha256) == 64
    assert sorted(label for fold in first.folds for label in fold.validation_labels) == list(
        range(8)
    )
    assert all(set(fold.fit_labels).isdisjoint(fold.validation_labels) for fold in first.folds)
    assert {frozenset(fold.validation_labels) for fold in first.folds} == {
        frozenset((0, 1)),
        frozenset((2, 3)),
        frozenset((4, 5)),
        frozenset((6, 7)),
    }
    reassigned = labels.clone()
    left = int(torch.nonzero(labels == 0, as_tuple=False)[0])
    right = int(torch.nonzero(labels == 1, as_tuple=False)[0])
    reassigned[left], reassigned[right] = reassigned[right].clone(), reassigned[left].clone()
    rebound = build_sfq_fold_schedule(features, reassigned, _authority(features), fold_count=4)
    assert rebound.folds == first.folds
    assert rebound.sha256 != first.sha256


def test_sfq_authority_rejects_evaluation_role_and_feature_drift() -> None:
    """Removing role or digest validation must expose this test to forbidden input."""

    features, labels = _four_pair_features()
    try:
        build_sfq_fold_schedule(
            features,
            labels,
            _authority(features, role="clean-validation"),
            fold_count=4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("SFQ accepted evaluation-role authority")

    authority = _authority(features)
    features[0, 0] += 0.25
    try:
        build_sfq_fold_schedule(features, labels, authority, fold_count=4)
    except ValueError:
        pass
    else:
        raise AssertionError("SFQ accepted post-authority feature drift")


def _spiked_features() -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    labels: list[int] = []
    for label in range(8):
        center = torch.zeros(8, dtype=torch.float32)
        center[label] = 1.0
        for sample in range(12):
            row = center.clone()
            row[(label + 1) % 8] += 0.015 * ((sample % 3) - 1)
            row[(label + 3) % 8] += 0.01 * (((sample // 3) % 3) - 1)
            rows.append(row)
            labels.append(label)
    return torch.stack(rows), torch.tensor(labels, dtype=torch.int64)


def _shared_mean_null_features() -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    labels: list[int] = []
    base = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    residuals = []
    for axis in range(1, 6):
        for sign in (-1.0, 1.0):
            residual = torch.zeros(6, dtype=torch.float32)
            residual[axis] = 0.03 * sign
            residuals.append(residual)
    for label in range(8):
        for residual in residuals:
            rows.append(base + residual)
            labels.append(label)
    return torch.stack(rows), torch.tensor(labels, dtype=torch.int64)


def test_sfq_projection_is_finite_deterministic_and_has_exact_shape() -> None:
    """Wrong BBP scaling, factor orientation, or sign handling must fail this test."""

    features, labels = _spiked_features()

    first = fit_sfq_projection(features, labels, output_dimensions=3)
    second = fit_sfq_projection(features, labels, output_dimensions=3)

    assert first.weight.shape == (3, 8)
    assert first.whitening_weight.shape == (3, 8)
    assert torch.equal(first.weight, second.weight)
    assert torch.equal(first.whitening_weight, second.whitening_weight)
    assert torch.isfinite(first.weight).all()
    assert 0.0 <= first.ledoit_wolf_shrinkage <= 1.0
    assert first.minimum_within_eigenvalue > 0.0
    assert first.maximum_within_eigenvalue >= first.minimum_within_eigenvalue
    assert first.reliable_rank == len(first.retained_spikes) == len(first.gains)
    assert first.reliable_rank >= 1
    root = math.sqrt(8 - 0.5)
    expected_threshold = ((2 * root) ** 2 + 2.023449 * (2 * root) * (2 / root) ** (1 / 3)) / 8
    assert math.isclose(first.bbp_threshold, expected_threshold, rel_tol=0.0, abs_tol=1e-12)
    for sample_spike, retained_spike, gain in zip(
        first.sample_spikes[: first.reliable_rank],
        first.retained_spikes,
        first.gains,
        strict=True,
    ):
        assert sample_spike == retained_spike
        theta = (
            retained_spike - 1.0 - 1.0 + math.sqrt((retained_spike - 1.0 - 1.0) ** 2 - 4.0)
        ) / 2.0
        alignment = (1.0 - 1.0 / theta**2) / (1.0 + 1.0 / theta)
        assert math.isclose(gain, alignment * theta, rel_tol=1e-12, abs_tol=1e-12)


def test_sfq_spectral_comparator_is_row_scale_invariant() -> None:
    """The dimension-matched comparator must share SFQ's row normalization."""

    features, _labels = _spiked_features()
    scales = torch.linspace(0.5, 2.0, features.shape[0], dtype=torch.float32).unsqueeze(1)

    baseline = _spectral_projection(features, output_dimensions=3).double()
    rescaled = _spectral_projection(features * scales, output_dimensions=3).double()
    assert torch.allclose(
        baseline.T @ baseline,
        rescaled.T @ rescaled,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_sfq_projection_rejects_zero_reliable_rank_and_invalid_dimensions() -> None:
    """A null between-class spectrum or impossible output width must not emit a head."""

    null_features, null_labels = _shared_mean_null_features()
    try:
        fit_sfq_projection(null_features, null_labels, output_dimensions=3)
    except ValueError as error:
        assert "reliable rank" in str(error)
    else:
        raise AssertionError("SFQ accepted a zero-spike shared-mean null")

    features, labels = _spiked_features()
    try:
        fit_sfq_projection(features, labels, output_dimensions=9)
    except ValueError:
        pass
    else:
        raise AssertionError("SFQ accepted an output wider than its feature space")


def _valid_result_bytes() -> bytes:
    features, labels = _spiked_features()
    return run_sfq_fold_diagnostic(
        features,
        labels,
        split_authority=_authority(features),
        feature_cache_manifest_sha256="2" * 64,
        output_dimensions=3,
        fold_count=4,
    )


def test_sfq_fold_result_recomputes_counts_gates_and_canonical_bytes() -> None:
    """Dropping one fold or deriving recall from floats must invalidate the receipt."""

    features, labels = _spiked_features()
    raw = _valid_result_bytes()

    result = validate_sfq_result_bytes(raw)

    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert result.schema == "sfora-siglip-sfq-fold-diagnostic-v1"
    assert result.claim_eligible is False
    assert result.official_test_access is False
    assert result.source_manifest_sha256 == "1" * 64
    assert result.input_dimensions == features.shape[1]
    assert result.output_dimensions == 3
    assert result.fold_count == 4 == len(result.folds)
    assert result.query_count == features.shape[0]
    assert result.sfq_hits == sum(fold.sfq_hits for fold in result.folds)
    assert result.whitening_hits == sum(fold.whitening_hits for fold in result.folds)
    assert result.raw_hits == sum(fold.raw_hits for fold in result.folds)
    assert result.spectral_hits == sum(fold.spectral_hits for fold in result.folds)
    assert result.sfq_recall_ppm == result.sfq_hits * 1_000_000 // result.query_count
    assert result.passed is (
        result.sfq_recall_ppm - result.whitening_recall_ppm >= 2_000
        and result.sfq_hits >= result.spectral_hits
    )
    assert result.passed is True
    rejected = validate_sfq_result_bytes(
        run_sfq_fold_diagnostic(
            features,
            labels,
            split_authority=_authority(features),
            feature_cache_manifest_sha256="2" * 64,
            output_dimensions=1,
            fold_count=4,
        )
    )
    assert rejected.passed is False


def test_sfq_result_rejects_derived_and_identity_drift() -> None:
    """Trusting serialized totals, gates, or schedule identity must fail this mutation test."""

    for field, mutate in (
        ("query_count", lambda value: value + 1),
        ("sfq_hits", lambda value: value - 1),
        ("passed", lambda value: not value),
        ("fold_schedule_sha256", lambda _value: "f" * 64),
    ):
        value = json.loads(_valid_result_bytes())
        value[field] = mutate(value[field])
        mutated = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        try:
            validate_sfq_result_bytes(mutated)
        except ValueError:
            pass
        else:
            raise AssertionError(f"SFQ result accepted mutated {field}")

    value = json.loads(_valid_result_bytes())
    value["folds"][0]["fit_count"] += 1
    mutated = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(ValueError):
        validate_sfq_result_bytes(mutated)

    parsed = validate_sfq_result_bytes(_valid_result_bytes())
    with pytest.raises(ValueError, match="registered identity"):
        validate_sfq_result_bytes(
            _valid_result_bytes(),
            expected_source_manifest_sha256="2" * 64,
            expected_feature_cache_manifest_sha256=parsed.feature_cache_manifest_sha256,
            expected_ordered_example_ids_sha256=parsed.ordered_example_ids_sha256,
            expected_feature_matrix_sha256=parsed.feature_matrix_sha256,
            expected_label_vector_sha256=parsed.label_vector_sha256,
        )
