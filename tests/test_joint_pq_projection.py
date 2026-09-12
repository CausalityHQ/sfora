from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from sfora.joint_pq_projection import (
    JointPqProjection,
    JointPqTrainingSpec,
    fit_joint_pq_projection,
)
from sfora.product_quantization import ProductQuantizationSpec, ProductQuantizer


def _quantizer() -> ProductQuantizer:
    return ProductQuantizer.from_codebooks(
        ProductQuantizationSpec(block_dimensions=(2, 2), codebook_size=3),
        (
            torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
            torch.tensor([[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]]),
        ),
    )


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = F.normalize(
        torch.tensor(
            [
                [1.0, 0.2, 0.0, 0.1],
                [0.9, 0.3, 0.1, 0.0],
                [0.1, 1.0, 0.2, 0.0],
                [0.2, 0.9, 0.0, 0.1],
                [-1.0, 0.1, 0.2, 0.0],
                [-0.9, 0.2, 0.1, 0.1],
                [0.0, -1.0, 0.1, 0.2],
                [0.1, -0.9, 0.2, 0.1],
            ]
        ),
        dim=1,
    )
    candidates = torch.tensor(
        [[1, 2, 4], [0, 3, 5], [3, 0, 6], [2, 1, 7], [5, 6, 0], [4, 7, 1], [7, 4, 2], [6, 5, 3]],
        dtype=torch.int64,
    )
    pairs = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
    return inputs, candidates, pairs


def test_joint_projection_initialization_preserves_rows_and_hard_codes() -> None:
    values, _candidates, _pairs = _fixture()
    model = JointPqProjection.from_components(
        input_dimensions=4,
        initial_weight=torch.eye(4),
        initial_bias=torch.zeros(4),
        quantizer=_quantizer(),
    )

    projected = model.project(values)

    torch.testing.assert_close(projected, values, rtol=0.0, atol=1e-7)
    assert torch.equal(model.quantizer.hard_encode(projected), _quantizer().hard_encode(values))


def test_joint_projection_loss_backpropagates_to_projection_and_codebooks() -> None:
    values, candidates, pairs = _fixture()
    model = JointPqProjection.from_components(
        input_dimensions=4,
        initial_weight=torch.eye(4),
        initial_bias=torch.zeros(4),
        quantizer=_quantizer(),
    )
    anchors = torch.tensor([0, 2], dtype=torch.int64)

    loss = model.loss(
        values[anchors],
        values[candidates[anchors]],
        values[anchors],
        values[candidates[anchors]],
        neighbor_pairs=pairs,
        temperature=0.2,
        float_weight=0.25,
        reconstruction_weight=0.1,
        differential_weight=0.4,
    )
    loss.total.backward()  # type: ignore[no-untyped-call]

    assert model.projection.weight.grad is not None
    assert float(model.projection.weight.grad.abs().sum()) > 0.0
    assert all(codebook.grad is not None for codebook in model.quantizer.codebooks)
    assert float(loss.differential.detach()) > 0.0


def test_joint_projection_accepts_a_wider_teacher_similarity_space() -> None:
    values, candidates, pairs = _fixture()
    teacher = F.normalize(
        torch.cat((values, values[:, :2] * torch.tensor([0.7, -0.4])), dim=1),
        dim=1,
    )
    model = JointPqProjection.from_components(
        input_dimensions=4,
        initial_weight=torch.eye(4),
        initial_bias=torch.zeros(4),
        quantizer=_quantizer(),
    )
    anchors = torch.tensor([0, 2], dtype=torch.int64)

    observed = model.loss(
        values[anchors],
        values[candidates[anchors]],
        teacher[anchors],
        teacher[candidates[anchors]],
        neighbor_pairs=pairs,
        temperature=0.2,
        float_weight=0.25,
        reconstruction_weight=0.1,
        differential_weight=0.4,
    )

    assert torch.isfinite(observed.total)


def test_joint_projection_fit_is_deterministic_and_returns_finite_trace() -> None:
    values, candidates, pairs = _fixture()
    spec = JointPqTrainingSpec(
        updates=4,
        batch_size=3,
        projection_learning_rate=1e-3,
        codebook_learning_rate=2e-3,
        weight_decay=1e-4,
        temperature=0.2,
        float_weight=0.25,
        reconstruction_weight=0.1,
        differential_weight=0.4,
        gradient_norm_cap=5.0,
        seed=7,
    )

    results = []
    for _ in range(2):
        model = JointPqProjection.from_components(
            input_dimensions=4,
            initial_weight=torch.eye(4),
            initial_bias=torch.zeros(4),
            quantizer=_quantizer(),
        )
        results.append(
            fit_joint_pq_projection(
                model,
                inputs=values,
                teacher_values=values,
                candidate_indexes=candidates,
                neighbor_pairs=pairs,
                spec=spec,
            )
        )

    assert results[0].losses == results[1].losses
    assert results[0].trace == results[1].trace
    assert len(results[0].losses) == 4
    assert len(results[0].trace) == 4
    assert all(value > 0.0 for value in results[0].losses)
    torch.testing.assert_close(
        results[0].model.projection.weight,
        results[1].model.projection.weight,
        rtol=0.0,
        atol=0.0,
    )


def test_joint_projection_rejects_nonfinite_or_invalid_training_geometry() -> None:
    values, candidates, pairs = _fixture()
    model = JointPqProjection.from_components(
        input_dimensions=4,
        initial_weight=torch.eye(4),
        initial_bias=torch.zeros(4),
        quantizer=_quantizer(),
    )
    bad = values.clone()
    bad[0, 0] = float("nan")
    spec = JointPqTrainingSpec(
        updates=1,
        batch_size=2,
        projection_learning_rate=1e-3,
        codebook_learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.2,
        float_weight=0.25,
        reconstruction_weight=0.1,
        differential_weight=0.0,
        gradient_norm_cap=5.0,
        seed=0,
    )

    with pytest.raises(ValueError, match="joint PQ training"):
        fit_joint_pq_projection(
            model,
            inputs=bad,
            teacher_values=values,
            candidate_indexes=candidates,
            neighbor_pairs=pairs,
            spec=spec,
        )
