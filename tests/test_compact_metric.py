from __future__ import annotations

import math

import pytest
import torch

import sfora
from sfora.compact_metric import (
    CompactMetricConfig,
    CompactMetricEncoder,
    CompactMetricModule,
    CompactMetricSelectionFold,
    CompactMetricSelectionResult,
    PowerWhiteningFitResult,
    WithinClassWhiteningFitResult,
    _positive_rows,
    _score_compact_metric_codes,
    choose_compact_metric_projection,
    fit_compact_metric_projection,
    fit_power_whitening_projection,
    fit_within_class_whitening_projection,
    select_compact_metric_projection,
)
from sfora.joint_relational_compaction import PackedInt8Embeddings
from sfora.representation_ceiling import fit_centered_pca


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(23)
    centers = torch.randn(4, 6, generator=generator)
    values = torch.cat(
        [center + 0.05 * torch.randn(4, 6, generator=generator) for center in centers]
    )
    labels = torch.arange(1, 5, dtype=torch.int64).repeat_interleave(4)
    return torch.nn.functional.normalize(values, dim=1), labels


def test_compact_metric_api_is_public() -> None:
    """Catch the validated trainer becoming inaccessible to library users."""

    assert sfora.CompactMetricConfig is CompactMetricConfig
    assert sfora.CompactMetricEncoder is CompactMetricEncoder
    assert sfora.CompactMetricModule is CompactMetricModule
    assert sfora.CompactMetricSelectionFold is CompactMetricSelectionFold
    assert sfora.CompactMetricSelectionResult is CompactMetricSelectionResult
    assert sfora.PowerWhiteningFitResult is PowerWhiteningFitResult
    assert sfora.WithinClassWhiteningFitResult is WithinClassWhiteningFitResult
    assert sfora.choose_compact_metric_projection is choose_compact_metric_projection
    assert sfora.fit_compact_metric_projection is fit_compact_metric_projection
    assert sfora.fit_power_whitening_projection is fit_power_whitening_projection
    assert sfora.fit_within_class_whitening_projection is fit_within_class_whitening_projection
    assert sfora.select_compact_metric_projection is select_compact_metric_projection


def test_power_whitening_projection_is_deterministic_and_explicit() -> None:
    generator = torch.Generator().manual_seed(809)
    centers = torch.randn(8, 6, generator=generator)
    scales = torch.tensor([0.6, 0.3, 0.15, 0.08, 0.04, 0.02])
    embeddings = torch.cat(
        [center + torch.randn(20, 6, generator=generator) * scales for center in centers]
    ).float().contiguous()
    labels = torch.arange(8, dtype=torch.int64).repeat_interleave(20).contiguous()
    original = embeddings.clone()

    first = fit_power_whitening_projection(
        embeddings,
        labels,
        alpha=0.75,
        regularization=1.0,
        output_dimensions=4,
    )
    second = fit_power_whitening_projection(
        embeddings.clone(),
        labels.clone(),
        alpha=0.75,
        regularization=1.0,
        output_dimensions=4,
    )

    assert isinstance(first, PowerWhiteningFitResult)
    assert first.alpha == 0.75
    assert first.regularization == 1.0
    assert first.output_dimensions == 4
    assert first.encoder.sha256 == second.encoder.sha256
    torch.testing.assert_close(first.encoder.weight, second.encoder.weight, atol=0, rtol=0)
    torch.testing.assert_close(first.encoder.bias, second.encoder.bias, atol=0, rtol=0)
    torch.testing.assert_close(embeddings, original, atol=0, rtol=0)
    assert first.within_minimum_eigenvalue > 0.0
    assert first.within_maximum_eigenvalue >= first.within_minimum_eigenvalue
    assert first.encoder.encode(embeddings).shape == (160, 4)


@pytest.mark.parametrize(
    ("alpha", "regularization"),
    [(-0.25, 0.0), (1.25, 0.0), (0.5, -0.01), (0.5, float("nan"))],
)
def test_power_whitening_projection_rejects_unregistered_geometry(
    alpha: float, regularization: float
) -> None:
    embeddings, labels = _fixture()

    with pytest.raises(ValueError, match="power whitening authority differs"):
        fit_power_whitening_projection(
            embeddings,
            labels,
            alpha=alpha,
            regularization=regularization,
            output_dimensions=3,
        )


def test_within_class_whitening_is_deterministic_and_does_not_mutate_inputs() -> None:
    """Catch fitting from total covariance or mutating caller-owned teacher features."""

    generator = torch.Generator().manual_seed(101)
    centers = torch.randn(8, 4, generator=generator)
    scales = torch.tensor([0.45, 0.12, 0.035, 0.01])
    embeddings = (
        torch.cat([center + torch.randn(20, 4, generator=generator) * scales for center in centers])
        .float()
        .contiguous()
        .requires_grad_(True)
    )
    labels = torch.arange(8, dtype=torch.int64).repeat_interleave(20).contiguous()
    original = embeddings.detach().clone()

    first = fit_within_class_whitening_projection(embeddings, labels, output_dimensions=3)
    second = fit_within_class_whitening_projection(
        embeddings.clone(), labels.clone(), output_dimensions=3
    )

    assert isinstance(first, WithinClassWhiteningFitResult)
    torch.testing.assert_close(embeddings.detach(), original, atol=0, rtol=0)
    torch.testing.assert_close(first.encoder.weight, second.encoder.weight, atol=0, rtol=0)
    torch.testing.assert_close(first.encoder.bias, second.encoder.bias, atol=0, rtol=0)
    assert first.shrinkage == second.shrinkage
    assert 0.0 <= first.shrinkage <= 1.0
    assert first.within_minimum_eigenvalue > 0.0
    assert first.within_maximum_eigenvalue >= first.within_minimum_eigenvalue
    assert (
        first.encoder.sha256 == "d9b2c4ea5574255736d51c25f5e367001ddb0cd27dd1593ef3195edb703eebd1"
    )
    normalized = torch.nn.functional.normalize(embeddings.detach().double(), dim=1)
    projected = normalized @ first.encoder.weight.double().T + first.encoder.bias.double()
    centered_origin = normalized.mean(dim=0) @ first.encoder.weight.double().T
    centered_origin += first.encoder.bias.double()
    assert float(centered_origin.abs().max()) < 2e-7
    class_means = torch.stack([projected[labels == value].mean(dim=0) for value in range(8)])
    residuals = torch.cat([projected[labels == value] - class_means[value] for value in range(8)])
    covariance = residuals.T.double() @ residuals.double() / len(residuals)
    condition = torch.linalg.cond(covariance).item()
    raw_means = torch.stack([normalized[labels == value].mean(dim=0) for value in range(8)])
    raw_residuals = torch.cat(
        [normalized[labels == value] - raw_means[value] for value in range(8)]
    )
    raw_covariance = raw_residuals.T.double() @ raw_residuals.double() / len(raw_residuals)
    assert condition < torch.linalg.cond(raw_covariance).item() / 5.0


def test_compact_metric_accepts_explicit_whitening_initializer_without_changing_default() -> None:
    """Catch reintroducing the research monkeypatch or changing the PCA default path."""

    embeddings, labels = _fixture()
    config = CompactMetricConfig(
        output_dimensions=3,
        cycles=1,
        anchor_epochs_per_cycle=0.5,
        hard_negatives=3,
    )
    baseline = fit_compact_metric_projection(
        embeddings, labels, config=config, device=torch.device("cpu")
    )
    repeated = fit_compact_metric_projection(
        embeddings, labels, config=config, initial_encoder=None, device=torch.device("cpu")
    )
    whitening = fit_within_class_whitening_projection(embeddings, labels, output_dimensions=3)
    whitened = fit_compact_metric_projection(
        embeddings,
        labels,
        config=config,
        initial_encoder=whitening.encoder,
        device=torch.device("cpu"),
    )

    assert baseline.parameter_sha256 == repeated.parameter_sha256
    assert baseline.losses == repeated.losses
    assert (
        baseline.parameter_sha256
        == "b4a72755f750bafac225ce72d6f716d6b3e4adb18808016dfcd2cc95aa4c2358"
    )
    assert whitened.parameter_sha256 != baseline.parameter_sha256
    assert whitened.losses != baseline.losses
    assert whitened.total_updates == baseline.total_updates
    assert whitened.encoder.weight.shape == (3, 6)
    assert torch.isfinite(whitened.encoder.weight).all()
    assert torch.isfinite(whitened.encoder.bias).all()


def test_within_class_whitening_rejects_a_degenerate_retained_boundary() -> None:
    """Catch choosing an arbitrary axis when retained and excluded variances tie."""

    embeddings = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, 0.0, -1.0],
            [-2.0, 1.0, 0.0],
            [-2.0, -1.0, 0.0],
            [-2.0, 0.0, 1.0],
            [-2.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64).contiguous()

    with pytest.raises(ValueError, match="within-class whitening projection differs"):
        fit_within_class_whitening_projection(embeddings, labels, output_dimensions=2)


def test_compact_metric_rejects_subnormal_input_geometry() -> None:
    """Catch epsilon-clamped normalization retaining input-magnitude dependence."""

    embeddings, labels = _fixture()
    embeddings[0].mul_(1e-15)
    encoder = CompactMetricEncoder(weight=torch.eye(6), bias=torch.zeros(6))

    with pytest.raises(ValueError, match="compact metric transform authority differs"):
        encoder.transform(embeddings[:1].contiguous())
    with pytest.raises(ValueError, match="compact metric training authority differs"):
        fit_compact_metric_projection(
            embeddings,
            labels,
            config=CompactMetricConfig(output_dimensions=3),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="within-class whitening authority differs"):
        fit_within_class_whitening_projection(embeddings, labels, output_dimensions=3)


def test_compact_metric_encoder_emits_the_complete_search_wire_representation() -> None:
    """Catch exposing code bytes without the inverse norms required by exact search."""

    generator = torch.Generator().manual_seed(911)
    embeddings = torch.randn(5, 6, generator=generator).float().contiguous()
    encoder = CompactMetricEncoder(
        weight=torch.randn(3, 6, generator=generator).float().contiguous(),
        bias=torch.randn(3, generator=generator).float().contiguous(),
    )

    packed = encoder.encode_packed(embeddings)

    assert isinstance(packed, PackedInt8Embeddings)
    assert torch.equal(packed.codes, encoder.encode(embeddings))
    assert packed.codes.shape == (5, 3)
    assert packed.inverse_norms.shape == (5,)
    assert len(packed.to_bytes()) == 5 * (3 + 2)


def test_compact_metric_selection_policy_uses_quality_gain_and_recall_guard() -> None:
    """Catch accepting a learned projection that misses either frozen gate."""

    assert (
        choose_compact_metric_projection(
            map_at_r_delta=0.003,
            recall_at_1_delta=0.0,
        )
        == "learned_projection"
    )
    assert (
        choose_compact_metric_projection(
            map_at_r_delta=0.002999,
            recall_at_1_delta=0.1,
        )
        == "pca_fallback"
    )
    assert (
        choose_compact_metric_projection(
            map_at_r_delta=0.1,
            recall_at_1_delta=-1e-9,
        )
        == "pca_fallback"
    )


def test_fit_only_selector_returns_full_fit_pca_fallback() -> None:
    """Catch test-split selection or returning a fold-local fallback encoder."""

    generator = torch.Generator().manual_seed(41)
    labels_by_class = (5, 7, 0, 1, 2, 3)
    centers = torch.randn(len(labels_by_class), 6, generator=generator)
    embeddings = torch.cat(
        [center + 0.08 * torch.randn(4, 6, generator=generator) for center in centers]
    )
    embeddings = torch.nn.functional.normalize(embeddings, dim=1).contiguous()
    labels = torch.tensor(labels_by_class, dtype=torch.int64).repeat_interleave(4)
    config = CompactMetricConfig(
        output_dimensions=3,
        cycles=1,
        anchor_epochs_per_cycle=0.5,
        hard_negatives=3,
    )

    result = select_compact_metric_projection(
        embeddings,
        labels,
        config=config,
        minimum_map_gain=1.0,
        device=torch.device("cpu"),
    )

    assert result.selected == "pca_fallback"
    assert len(result.folds) == 3
    assert [fold.validation_class_count for fold in result.folds] == [2, 2, 2]
    assert [fold.validation_row_count for fold in result.folds] == [8, 8, 8]
    assert math.isfinite(result.map_at_r_delta)
    assert math.isfinite(result.recall_at_1_delta)
    expected = fit_centered_pca(
        torch.nn.functional.normalize(embeddings, dim=1).contiguous(), dimensions=3
    )
    expected_bias = (-(expected.components.double() @ expected.mean.double())).float()
    torch.testing.assert_close(result.encoder.weight, expected.components, atol=0, rtol=0)
    torch.testing.assert_close(result.encoder.bias, expected_bias, atol=0, rtol=0)


def test_fit_only_selector_names_missing_deterministic_fold() -> None:
    """Catch small label sets failing after partial training with an opaque authority error."""

    embeddings = torch.nn.functional.normalize(
        torch.randn(10, 6, generator=torch.Generator().manual_seed(71)), dim=1
    ).contiguous()
    labels = torch.arange(5, dtype=torch.int64).repeat_interleave(2)

    with pytest.raises(
        ValueError,
        match=r"compact metric selector has no eligible classes in deterministic fold 0",
    ):
        select_compact_metric_projection(
            embeddings,
            labels,
            config=CompactMetricConfig(
                output_dimensions=2,
                cycles=1,
                anchor_epochs_per_cycle=0.5,
                rows_per_class=2,
                hard_negatives=2,
            ),
            device=torch.device("cpu"),
        )


def test_fit_only_selector_excludes_singletons_from_folds_but_not_full_fit() -> None:
    """Catch rejecting realistic fit sets or dropping singleton rows from the final PCA."""

    generator = torch.Generator().manual_seed(53)
    labels_by_class = (5, 7, 0, 1, 2, 3)
    centers = torch.randn(len(labels_by_class), 6, generator=generator)
    embeddings = torch.cat(
        [center + 0.08 * torch.randn(4, 6, generator=generator) for center in centers]
        + [torch.randn(1, 6, generator=generator)]
    )
    embeddings = torch.nn.functional.normalize(embeddings, dim=1).contiguous()
    labels = torch.cat(
        [torch.tensor(labels_by_class, dtype=torch.int64).repeat_interleave(4), torch.tensor([4])]
    ).contiguous()
    config = CompactMetricConfig(
        output_dimensions=3,
        cycles=1,
        anchor_epochs_per_cycle=0.5,
        hard_negatives=3,
    )

    result = select_compact_metric_projection(
        embeddings,
        labels,
        config=config,
        minimum_map_gain=1.0,
        device=torch.device("cpu"),
    )

    assert result.selected == "pca_fallback"
    assert sum(fold.validation_row_count for fold in result.folds) == 24
    expected = fit_centered_pca(
        torch.nn.functional.normalize(embeddings, dim=1).contiguous(), dimensions=3
    )
    expected_bias = (-(expected.components.double() @ expected.mean.double())).float()
    torch.testing.assert_close(result.encoder.weight, expected.components, atol=0, rtol=0)
    torch.testing.assert_close(result.encoder.bias, expected_bias, atol=0, rtol=0)


def test_fit_only_selector_refits_the_selected_learned_projection() -> None:
    """Catch returning a fold-local learned encoder instead of the full-data fit."""

    generator = torch.Generator().manual_seed(41)
    labels_by_class = (5, 7, 0, 1, 2, 3)
    centers = torch.randn(len(labels_by_class), 6, generator=generator)
    embeddings = torch.cat(
        [center + 0.08 * torch.randn(4, 6, generator=generator) for center in centers]
    )
    embeddings = torch.nn.functional.normalize(embeddings, dim=1).contiguous()
    labels = torch.tensor(labels_by_class, dtype=torch.int64).repeat_interleave(4)
    config = CompactMetricConfig(
        output_dimensions=3,
        cycles=1,
        anchor_epochs_per_cycle=0.5,
        hard_negatives=3,
    )

    result = select_compact_metric_projection(
        embeddings,
        labels,
        config=config,
        minimum_map_gain=0.0,
        device=torch.device("cpu"),
    )
    expected = fit_compact_metric_projection(
        torch.nn.functional.normalize(embeddings, dim=1).contiguous(),
        labels,
        config=config,
        device=torch.device("cpu"),
    )

    assert result.selected == "learned_projection"
    assert result.map_at_r_delta == 0.0
    assert result.recall_at_1_delta == 0.0
    torch.testing.assert_close(result.encoder.weight, expected.encoder.weight, atol=0, rtol=0)
    torch.testing.assert_close(result.encoder.bias, expected.encoder.bias, atol=0, rtol=0)


def test_fit_only_selector_detaches_caller_autograd_graph() -> None:
    """Catch selector PCA retaining an input graph and failing on sign canonicalization."""

    generator = torch.Generator().manual_seed(41)
    labels_by_class = (5, 7, 0, 1, 2, 3)
    centers = torch.randn(len(labels_by_class), 6, generator=generator)
    embeddings = torch.cat(
        [center + 0.08 * torch.randn(4, 6, generator=generator) for center in centers]
    )
    embeddings = torch.nn.functional.normalize(embeddings, dim=1).contiguous()
    labels = torch.tensor(labels_by_class, dtype=torch.int64).repeat_interleave(4)
    config = CompactMetricConfig(
        output_dimensions=3,
        cycles=1,
        anchor_epochs_per_cycle=0.5,
        hard_negatives=3,
    )
    expected = select_compact_metric_projection(
        embeddings,
        labels,
        config=config,
        minimum_map_gain=1.0,
        device=torch.device("cpu"),
    )

    actual = select_compact_metric_projection(
        embeddings.clone().requires_grad_(True),
        labels,
        config=config,
        minimum_map_gain=1.0,
        device=torch.device("cpu"),
    )

    assert actual.selected == expected.selected
    assert actual.map_at_r_delta == expected.map_at_r_delta
    assert actual.recall_at_1_delta == expected.recall_at_1_delta
    torch.testing.assert_close(actual.encoder.weight, expected.encoder.weight, atol=0, rtol=0)
    torch.testing.assert_close(actual.encoder.bias, expected.encoder.bias, atol=0, rtol=0)


def test_fit_only_selector_disables_ambient_cpu_autocast() -> None:
    """Catch caller autocast changing selector quantization or fold decisions."""

    generator = torch.Generator().manual_seed(41)
    labels_by_class = (5, 7, 0, 1, 2, 3)
    centers = torch.randn(len(labels_by_class), 6, generator=generator)
    embeddings = torch.cat(
        [center + 0.08 * torch.randn(4, 6, generator=generator) for center in centers]
    )
    embeddings = torch.nn.functional.normalize(embeddings, dim=1).contiguous()
    labels = torch.tensor(labels_by_class, dtype=torch.int64).repeat_interleave(4)
    config = CompactMetricConfig(
        output_dimensions=3,
        cycles=1,
        anchor_epochs_per_cycle=0.5,
        hard_negatives=3,
    )
    expected = select_compact_metric_projection(
        embeddings,
        labels,
        config=config,
        minimum_map_gain=1.0,
        device=torch.device("cpu"),
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = select_compact_metric_projection(
            embeddings,
            labels,
            config=config,
            minimum_map_gain=1.0,
            device=torch.device("cpu"),
        )

    assert actual.selected == expected.selected
    assert actual.map_at_r_delta == expected.map_at_r_delta
    assert actual.recall_at_1_delta == expected.recall_at_1_delta
    torch.testing.assert_close(actual.encoder.weight, expected.encoder.weight, atol=0, rtol=0)
    torch.testing.assert_close(actual.encoder.bias, expected.encoder.bias, atol=0, rtol=0)


@pytest.mark.parametrize("context_name", ["no_grad", "inference_mode"])
def test_fit_only_selector_enables_training_inside_disabled_grad_context(
    context_name: str,
) -> None:
    """Catch caller inference contexts disabling the selector's internal training graph."""

    generator = torch.Generator().manual_seed(41)
    labels_by_class = (5, 7, 0, 1, 2, 3)
    centers = torch.randn(len(labels_by_class), 6, generator=generator)
    embeddings = torch.cat(
        [center + 0.08 * torch.randn(4, 6, generator=generator) for center in centers]
    )
    embeddings = torch.nn.functional.normalize(embeddings, dim=1).contiguous()
    labels = torch.tensor(labels_by_class, dtype=torch.int64).repeat_interleave(4)
    config = CompactMetricConfig(
        output_dimensions=3,
        cycles=1,
        anchor_epochs_per_cycle=0.5,
        hard_negatives=3,
    )
    expected = select_compact_metric_projection(
        embeddings,
        labels,
        config=config,
        minimum_map_gain=1.0,
        device=torch.device("cpu"),
    )
    context = torch.no_grad() if context_name == "no_grad" else torch.inference_mode()

    with context:
        actual = select_compact_metric_projection(
            embeddings,
            labels,
            config=config,
            minimum_map_gain=1.0,
            device=torch.device("cpu"),
        )

    assert actual.selected == expected.selected
    assert actual.map_at_r_delta == expected.map_at_r_delta
    assert actual.recall_at_1_delta == expected.recall_at_1_delta
    torch.testing.assert_close(actual.encoder.weight, expected.encoder.weight, atol=0, rtol=0)
    torch.testing.assert_close(actual.encoder.bias, expected.encoder.bias, atol=0, rtol=0)


def test_selector_scorer_matches_hand_derived_map_at_r_with_cutoff_ties() -> None:
    """Catch non-stable ties or full-list AP replacing exact mAP@R semantics."""

    codes = torch.tensor(
        [[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]],
        dtype=torch.int8,
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)

    mean_ap, recall_at_1, per_query_ap, per_query_recall = _score_compact_metric_codes(
        codes,
        labels,
        device=torch.device("cpu"),
    )

    assert per_query_ap == (1.0, 1.0, 1.0, 0.0, 0.5, 0.5)
    assert per_query_recall == (1.0, 1.0, 1.0, 0.0, 1.0, 1.0)
    assert mean_ap == 2.0 / 3.0
    assert recall_at_1 == 5.0 / 6.0


def test_fit_compact_metric_projection_adapts_schedule_and_emits_exact_codes() -> None:
    """Catch fixed 64-class schedules and anchors that retain an autograd graph."""

    embeddings, labels = _fixture()
    config = CompactMetricConfig(
        output_dimensions=3,
        cycles=2,
        anchor_epochs_per_cycle=1.0,
        classes_per_update=64,
        rows_per_class=2,
        hard_negatives=4,
    )

    result = fit_compact_metric_projection(
        embeddings, labels, config=config, device=torch.device("cpu")
    )

    assert result.eligible_class_count == 4
    assert result.classes_per_update == 4
    assert result.updates_per_cycle == 2
    assert result.total_updates == 4
    assert result.effective_hard_negatives == 4
    assert len(result.losses) == 4
    assert math.isfinite(result.losses[-1])
    assert result.encoder.weight.shape == (3, 6)
    assert result.encoder.bias.shape == (3,)
    transformed = result.encoder.transform(embeddings)
    assert transformed.shape == (16, 3)
    torch.testing.assert_close(
        torch.linalg.vector_norm(transformed, dim=1), torch.ones(16), atol=1e-6, rtol=0
    )
    codes = result.encoder.encode(embeddings)
    assert codes.dtype == torch.int8
    assert codes.shape == (16, 3)
    assert codes.numel() == 16 * 3
    module = result.encoder.to_module(torch.device("cpu"))
    assert not module.training
    assert all(not parameter.requires_grad for parameter in module.parameters())
    torch.testing.assert_close(module(embeddings), transformed, atol=0, rtol=0)
    torch.testing.assert_close(module.encode(embeddings), codes, atol=0, rtol=0)


def test_fit_compact_metric_projection_is_reproducible() -> None:
    """Catch random scheduling or initialization drift under the same seed."""

    embeddings, labels = _fixture()
    config = CompactMetricConfig(
        output_dimensions=3,
        cycles=1,
        anchor_epochs_per_cycle=0.5,
        classes_per_update=3,
        rows_per_class=2,
        hard_negatives=3,
        seed=29,
    )

    first = fit_compact_metric_projection(
        embeddings, labels, config=config, device=torch.device("cpu")
    )
    second = fit_compact_metric_projection(
        embeddings.clone(), labels.clone(), config=config, device=torch.device("cpu")
    )

    torch.testing.assert_close(first.encoder.weight, second.encoder.weight, atol=0, rtol=0)
    torch.testing.assert_close(first.encoder.bias, second.encoder.bias, atol=0, rtol=0)
    assert first.schedule_sha256 == second.schedule_sha256
    assert first.parameter_sha256 == second.parameter_sha256
    assert first.parameter_sha256 == first.encoder.sha256
    assert len(first.parameter_sha256) == 64
    assert first.losses == second.losses

    changed = CompactMetricEncoder(weight=first.encoder.weight + 1e-3, bias=first.encoder.bias)
    assert changed.sha256 != first.encoder.sha256


def test_fit_compact_metric_projection_detaches_frozen_teacher_embeddings() -> None:
    """Catch metric fitting retaining or backpropagating into the teacher graph."""

    embeddings, labels = _fixture()
    embeddings.requires_grad_(True)

    result = fit_compact_metric_projection(
        embeddings,
        labels,
        config=CompactMetricConfig(
            output_dimensions=3,
            cycles=1,
            anchor_epochs_per_cycle=0.5,
            hard_negatives=3,
        ),
        device=torch.device("cpu"),
    )

    assert result.total_updates > 0
    assert embeddings.grad is None


@pytest.mark.parametrize("context_name", ["no_grad", "inference_mode"])
def test_fit_compact_metric_projection_enables_training_inside_disabled_grad_context(
    context_name: str,
) -> None:
    """Catch caller inference contexts disabling the public fit function's gradient graph."""

    embeddings, labels = _fixture()
    config = CompactMetricConfig(
        output_dimensions=3,
        cycles=1,
        anchor_epochs_per_cycle=0.5,
        hard_negatives=3,
    )
    expected = fit_compact_metric_projection(
        embeddings,
        labels,
        config=config,
        device=torch.device("cpu"),
    )
    context = torch.no_grad() if context_name == "no_grad" else torch.inference_mode()

    with context:
        actual = fit_compact_metric_projection(
            embeddings,
            labels,
            config=config,
            device=torch.device("cpu"),
        )

    assert actual.losses == expected.losses
    torch.testing.assert_close(actual.encoder.weight, expected.encoder.weight, atol=0, rtol=0)
    torch.testing.assert_close(actual.encoder.bias, expected.encoder.bias, atol=0, rtol=0)


def test_compact_metric_float32_contract_ignores_ambient_autocast() -> None:
    """Catch caller autocast changing training or deployment precision."""

    embeddings, labels = _fixture()
    config = CompactMetricConfig(
        output_dimensions=3,
        cycles=1,
        anchor_epochs_per_cycle=0.5,
        hard_negatives=3,
    )
    expected = fit_compact_metric_projection(
        embeddings, labels, config=config, device=torch.device("cpu")
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        observed = fit_compact_metric_projection(
            embeddings, labels, config=config, device=torch.device("cpu")
        )
        projected = observed.encoder.to_module(torch.device("cpu"))(embeddings)

    torch.testing.assert_close(observed.encoder.weight, expected.encoder.weight, atol=0, rtol=0)
    torch.testing.assert_close(observed.encoder.bias, expected.encoder.bias, atol=0, rtol=0)
    assert observed.losses == expected.losses
    assert projected.dtype == torch.float32


def test_fit_compact_metric_projection_changes_the_pca_initialization() -> None:
    """Catch an inert optimizer returning the initialization as a learned encoder."""

    embeddings, labels = _fixture()

    result = fit_compact_metric_projection(
        embeddings,
        labels,
        config=CompactMetricConfig(
            output_dimensions=3,
            cycles=2,
            anchor_epochs_per_cycle=0.5,
            hard_negatives=3,
        ),
        device=torch.device("cpu"),
    )

    assert result.updates_per_cycle == 1
    assert len(result.losses) == 2
    assert result.losses[1] != result.losses[0]


def test_fit_compact_metric_projection_is_invariant_to_global_input_scale() -> None:
    """Catch removal of the teacher-embedding normalization boundary."""

    embeddings, labels = _fixture()
    config = CompactMetricConfig(
        output_dimensions=3,
        cycles=1,
        anchor_epochs_per_cycle=0.5,
        hard_negatives=99,
    )

    reference = fit_compact_metric_projection(
        embeddings, labels, config=config, device=torch.device("cpu")
    )
    scaled = fit_compact_metric_projection(
        embeddings * 4.0, labels, config=config, device=torch.device("cpu")
    )

    torch.testing.assert_close(reference.encoder.weight, scaled.encoder.weight, atol=0, rtol=0)
    torch.testing.assert_close(reference.encoder.bias, scaled.encoder.bias, atol=0, rtol=0)
    assert reference.losses == scaled.losses
    assert reference.effective_hard_negatives == 12


def test_fit_compact_metric_projection_uses_the_centered_pca_initialization() -> None:
    """Catch random initialization or omission of the PCA centering bias."""

    embeddings, labels = _fixture()
    expected = fit_centered_pca(
        torch.nn.functional.normalize(embeddings, dim=1).contiguous(), dimensions=3
    )
    expected_bias = (-(expected.components.double() @ expected.mean.double())).float()

    result = fit_compact_metric_projection(
        embeddings,
        labels,
        config=CompactMetricConfig(
            output_dimensions=3,
            cycles=1,
            anchor_epochs_per_cycle=0.5,
            hard_negatives=3,
            learning_rate=1e-30,
        ),
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(result.encoder.weight, expected.components, atol=0, rtol=0)
    torch.testing.assert_close(result.encoder.bias, expected_bias, atol=0, rtol=0)


def test_anchor_regularization_restrains_projection_drift() -> None:
    """Catch the frozen-start anchor term being omitted from the objective."""

    embeddings, labels = _fixture()
    initial = fit_centered_pca(
        torch.nn.functional.normalize(embeddings, dim=1).contiguous(), dimensions=3
    )
    unanchored = fit_compact_metric_projection(
        embeddings,
        labels,
        config=CompactMetricConfig(
            output_dimensions=3,
            cycles=2,
            anchor_epochs_per_cycle=1.0,
            hard_negatives=3,
            learning_rate=1e-2,
            anchor_weight=0.0,
        ),
        device=torch.device("cpu"),
    )
    anchored = fit_compact_metric_projection(
        embeddings,
        labels,
        config=CompactMetricConfig(
            output_dimensions=3,
            cycles=2,
            anchor_epochs_per_cycle=1.0,
            hard_negatives=3,
            learning_rate=1e-2,
            anchor_weight=10.0,
        ),
        device=torch.device("cpu"),
    )
    unanchored_drift = torch.linalg.vector_norm(unanchored.encoder.weight - initial.components)
    anchored_drift = torch.linalg.vector_norm(anchored.encoder.weight - initial.components)

    assert anchored_drift < unanchored_drift


def test_ragged_positive_rows_preserve_the_padding_mask() -> None:
    """Catch padded rows becoming false positives for larger classes."""

    labels = torch.tensor([1, 1, 1, 2, 2, 2, 2], dtype=torch.int64).numpy()
    groups = {
        1: torch.tensor([0, 1, 2], dtype=torch.int64).numpy(),
        2: torch.tensor([3, 4, 5, 6], dtype=torch.int64).numpy(),
    }

    rows, mask = _positive_rows(torch.tensor([0, 3]).numpy(), labels, groups)

    assert rows.tolist() == [[1, 2, 0], [4, 5, 6]]
    assert mask.tolist() == [[True, True, False], [True, True, True]]


@pytest.mark.parametrize("mutation", ["zero-row", "one-class", "wrong-label-dtype"])
def test_fit_compact_metric_projection_rejects_invalid_training_authority(
    mutation: str,
) -> None:
    """Catch malformed embeddings or labels reaching PCA and metric training."""

    embeddings, labels = _fixture()
    if mutation == "zero-row":
        embeddings[0].zero_()
    elif mutation == "one-class":
        labels.fill_(1)
    elif mutation == "wrong-label-dtype":
        labels = labels.float()
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="compact metric training authority differs"):
        fit_compact_metric_projection(
            embeddings,
            labels,
            config=CompactMetricConfig(output_dimensions=3),
            device=torch.device("cpu"),
        )


def test_compact_metric_encoder_rejects_mutable_or_incompatible_state() -> None:
    """Catch non-CPU or mismatched affine state entering the deployment boundary."""

    with pytest.raises(ValueError, match="compact metric encoder authority differs"):
        CompactMetricEncoder(weight=torch.eye(3), bias=torch.zeros(2))


def test_compact_metric_encoder_digest_binds_parameter_shapes() -> None:
    """Catch distinct affine shapes sharing one concatenated-byte digest."""

    values = torch.arange(12, dtype=torch.float32)
    first = CompactMetricEncoder(weight=values[:10].reshape(2, 5), bias=values[10:])
    second = CompactMetricEncoder(weight=values[:9].reshape(3, 3), bias=values[9:])

    assert first.sha256 != second.sha256


def test_device_module_encode_has_no_scalar_read_synchronization() -> None:
    """Catch validation scalar reads serializing the high-throughput device path."""

    encoder = CompactMetricEncoder(weight=torch.eye(3), bias=torch.zeros(3))
    module = encoder.to_module(torch.device("cpu"))
    embeddings = torch.ones((2, 3), dtype=torch.float32)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True
    ) as profile:
        codes = module.encode(embeddings)

    assert codes.dtype == torch.int8
    assert "aten::item" not in {event.key for event in profile.key_averages()}


@pytest.mark.parametrize("mutation", ["one-dimensional-code", "singleton-positive-batch"])
def test_compact_metric_config_rejects_unexecutable_shapes(mutation: str) -> None:
    """Catch configurations that the hard-negative metric contract cannot execute."""

    with pytest.raises(ValueError, match="compact metric config differs"):
        if mutation == "one-dimensional-code":
            CompactMetricConfig(output_dimensions=1)
        elif mutation == "singleton-positive-batch":
            CompactMetricConfig(rows_per_class=1)
        else:  # pragma: no cover
            raise AssertionError(mutation)


@pytest.mark.parametrize("scale", [1e-15, 3e38])
def test_compact_metric_encoder_rejects_degenerate_projected_geometry(scale: float) -> None:
    """Catch accepted finite affine state producing nonunit or nonfinite outputs."""

    encoder = CompactMetricEncoder(
        weight=torch.eye(3, dtype=torch.float32) * scale,
        bias=torch.zeros(3, dtype=torch.float32),
    )
    embeddings = torch.ones((1, 3), dtype=torch.float32).contiguous()

    with pytest.raises(ValueError, match="compact metric transform geometry differs"):
        encoder.transform(embeddings)
