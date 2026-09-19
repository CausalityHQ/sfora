from __future__ import annotations

import math

import pytest
import torch

import sfora
from sfora.compact_metric import (
    CompactMetricConfig,
    CompactMetricEncoder,
    CompactMetricModule,
    _positive_rows,
    fit_compact_metric_projection,
)
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
    assert sfora.fit_compact_metric_projection is fit_compact_metric_projection


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
