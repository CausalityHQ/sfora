"""Tests for SigLIP spatial-tail extraction and recovery models."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import SiglipVisionConfig, SiglipVisionModel

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_siglip_spatial_tail_recovery.py"
_SPEC = importlib.util.spec_from_file_location(
    "scripts.probe_siglip_spatial_tail_recovery", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

stream_spatial_tail_fit_inputs = _MODULE.stream_spatial_tail_fit_inputs
residual_channel_scale = _MODULE.residual_channel_scale
TokenwiseTailControl = _MODULE.TokenwiseTailControl
LatentInteractionTail = _MODULE.LatentInteractionTail
FrozenTeacherReadout = _MODULE.FrozenTeacherReadout
fit_spatial_tail_arm = _MODULE.fit_spatial_tail_arm
spatial_tail_loss = _MODULE.spatial_tail_loss


class _MeanHead(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)


def _tiny_model() -> tuple[SiglipVisionModel, nn.Linear]:
    torch.manual_seed(9)
    model = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            image_size=16,
            patch_size=8,
        )
    ).eval()
    return model, nn.Linear(16, 8, bias=False).eval()


def test_stream_extracts_exact_source_target_and_teacher_descriptor() -> None:
    model, projection = _tiny_model()
    pixels = torch.arange(6 * 3 * 16 * 16, dtype=torch.float32).reshape(6, 3, 16, 16) / 4096
    batches = (pixels[:2], pixels[2:5], pixels[5:])

    actual = stream_spatial_tail_fit_inputs(
        model,
        projection,
        batches,
        source_depth=1,
        target_depth=3,
        device=torch.device("cpu"),
    )
    with torch.inference_mode():
        outputs = [
            model(pixel_values=batch, output_hidden_states=True, return_dict=True)
            for batch in batches
        ]
        h1 = torch.cat([output.hidden_states[1].half() for output in outputs]).contiguous()
        h3 = torch.cat([output.hidden_states[3].half() for output in outputs]).contiguous()
        targets = torch.cat(
            [F.normalize(projection(output.pooler_output.float()), dim=1) for output in outputs]
        ).contiguous()
    assert torch.equal(actual.source_tokens, h1)
    assert torch.equal(actual.target_tokens, h3)
    assert torch.equal(actual.teacher_descriptors, targets)
    assert actual.cache_bytes == sum(
        value.numel() * value.element_size()
        for value in (actual.source_tokens, actual.target_tokens, actual.teacher_descriptors)
    )


def test_residual_scale_matches_fixed_fp64_oracle_and_floors_zero_channel() -> None:
    source = torch.tensor(
        [[[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]], [[3.0, 4.0, 4.0], [4.0, 5.0, 4.0]]],
        dtype=torch.float16,
    )
    target = source.clone()
    target[..., 0] += 1
    target[..., 1] += torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
    scale = residual_channel_scale(source.contiguous(), target.contiguous())
    expected = torch.tensor([1.0, (30.0 / 4.0) ** 0.5, 0.001], dtype=torch.float32)
    assert torch.equal(scale, expected)


def test_models_preserve_shape_and_only_interaction_arm_crosses_tokens() -> None:
    torch.manual_seed(3)
    control = TokenwiseTailControl(16, bottleneck_width=8).eval()
    treatment = LatentInteractionTail(16, latent_width=8, latent_count=4, heads=2).eval()
    tokens = torch.randn(2, 5, 16)
    changed = tokens.clone()
    changed[:, 4] += 2.0

    with torch.inference_mode():
        control_left = control(tokens)
        control_right = control(changed)
        treatment_left = treatment(tokens)
        treatment_right = treatment(changed)
    assert control_left.shape == treatment_left.shape == tokens.shape
    assert torch.equal(control_left[:, :4], control_right[:, :4])
    assert not torch.equal(treatment_left[:, :4], treatment_right[:, :4])


def test_frozen_teacher_readout_returns_normalized_descriptors_without_gradients() -> None:
    model, projection = _tiny_model()
    readout = FrozenTeacherReadout(model.post_layernorm, model.head, projection).eval()
    tokens = torch.randn(3, 5, 16, requires_grad=True)
    descriptors = readout(tokens)
    descriptors.sum().backward()
    assert torch.allclose(torch.linalg.vector_norm(descriptors, dim=1), torch.ones(3))
    assert tokens.grad is not None
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in readout.parameters()
    )


def test_loss_matches_normalized_residual_mse_plus_descriptor_cosine() -> None:
    predicted = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    predicted_descriptor = F.normalize(torch.tensor([[1.0, 1.0]]), dim=1)
    teacher_descriptor = torch.tensor([[1.0, 0.0]])
    scale = torch.tensor([2.0, 1.0])
    expected = torch.mean(((predicted - target) / scale) ** 2) + (
        1.0 - torch.sum(predicted_descriptor * teacher_descriptor, dim=1)
    ).mean()
    assert torch.equal(
        spatial_tail_loss(
            predicted,
            target,
            predicted_descriptor,
            teacher_descriptor,
            scale,
        ),
        expected,
    )


def test_matched_trainer_is_deterministic_reduces_loss_and_preserves_readout() -> None:
    torch.manual_seed(21)
    source = torch.randn(8, 3, 4).half().contiguous()
    target = (source.float() + 0.1 * torch.tanh(source.float())).half().contiguous()
    layernorm = nn.LayerNorm(4).eval()
    head = _MeanHead().eval()
    projection = nn.Linear(4, 3, bias=False).eval()
    readout = FrozenTeacherReadout(layernorm, head, projection).eval()
    with torch.inference_mode():
        teacher = readout(target.float()).float().contiguous()
    scale = residual_channel_scale(source, target)
    initial = TokenwiseTailControl(4, bottleneck_width=4)
    initial_state = copy.deepcopy(initial.state_dict())
    readout_before = {name: value.clone() for name, value in readout.state_dict().items()}

    left = fit_spatial_tail_arm(
        initial,
        readout,
        source,
        target,
        teacher,
        scale,
        device=torch.device("cpu"),
        updates=20,
        batch_size=4,
    )
    right_model = TokenwiseTailControl(4, bottleneck_width=4)
    right_model.load_state_dict(initial_state)
    right = fit_spatial_tail_arm(
        right_model,
        readout,
        source,
        target,
        teacher,
        scale,
        device=torch.device("cpu"),
        updates=20,
        batch_size=4,
    )
    assert left.final_loss < left.initial_loss
    assert left.index_sha256 == right.index_sha256
    assert left.final_loss == right.final_loss
    for name, value in readout.state_dict().items():
        assert torch.equal(value, readout_before[name])


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nonfinite"])
def test_residual_scale_rejects_malformed_token_authority(mutation: str) -> None:
    source = torch.ones((2, 3, 4), dtype=torch.float16)
    target = source.clone()
    if mutation == "shape":
        target = target[:, :2]
    elif mutation == "dtype":
        target = target.float()
    else:
        target[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="residual scale authority differs"):
        residual_channel_scale(source.contiguous(), target.contiguous())
