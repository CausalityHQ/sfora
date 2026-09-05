from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from sfora.siglip_proxy_control import PooledProxyAnchorModel

_PATH = Path(__file__).resolve().parents[1] / "scripts/probe_siglip_spatial_tail_latency.py"
_SPEC = importlib.util.spec_from_file_location("probe_siglip_spatial_tail_latency", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

SpatialTailInference = _MODULE.SpatialTailInference
truncate_siglip_vision = _MODULE.truncate_siglip_vision
parse_args = _MODULE.parse_args
spatial_tail_batch_starts = _MODULE.spatial_tail_batch_starts
teacher_latency_descriptor = _MODULE.teacher_latency_descriptor


class _EncoderOutput:
    def __init__(self, value: torch.Tensor) -> None:
        self.last_hidden_state = value


class _Encoder(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(4, 4, bias=False) for _ in range(depth))
        for index, layer in enumerate(self.layers, start=1):
            nn.init.eye_(layer.weight)
            layer.weight.data.mul_(1.0 + index / 100.0)

    def forward(self, *, inputs_embeds: torch.Tensor, return_dict: bool) -> _EncoderOutput:
        assert return_dict is True
        value = inputs_embeds
        for layer in self.layers:
            value = layer(value)
        return _EncoderOutput(value)


class _Config:
    num_hidden_layers = 27


class _Vision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Identity()
        self.encoder = _Encoder(27)
        self.config = _Config()


class _Normalize(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(tokens.mean(dim=1), dim=1)


def test_truncated_inference_reproduces_exact_depth18_field_without_mutating_teacher() -> None:
    teacher = _Vision().eval()
    student = truncate_siglip_vision(teacher, retained_blocks=18)
    inference = SpatialTailInference(student, nn.Identity().eval(), _Normalize().eval()).eval()
    pixels = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 24.0

    with torch.inference_mode():
        actual = inference(pixels)
        hidden = pixels
        for layer in teacher.encoder.layers[:18]:
            hidden = layer(hidden)
        expected = torch.nn.functional.normalize(hidden.mean(dim=1), dim=1)

    assert torch.equal(actual, expected)
    assert len(student.encoder.layers) == student.config.num_hidden_layers == 18
    assert len(teacher.encoder.layers) == teacher.config.num_hidden_layers == 27


@pytest.mark.parametrize("retained_blocks", [0, 27, True])
def test_truncation_rejects_noncompact_or_nonconcrete_depth(retained_blocks: object) -> None:
    with pytest.raises(ValueError, match="spatial tail latency topology differs"):
        truncate_siglip_vision(_Vision().eval(), retained_blocks=retained_blocks)


def _args(tmp_path: Path) -> list[str]:
    return [
        "--control-root",
        str(tmp_path / "control"),
        "--spatial-artifact",
        str(tmp_path / "tail.safetensors"),
        "--spatial-artifact-sha256",
        "a" * 64,
        "--output",
        str(tmp_path / "latency.json"),
        "--execute-spatial-tail-latency",
    ]


def test_cli_requires_only_local_sealed_artifact_and_explicit_latency_execution(
    tmp_path: Path,
) -> None:
    parsed = parse_args(_args(tmp_path))
    assert parsed.spatial_artifact_sha256 == "a" * 64
    assert parsed.execute_spatial_tail_latency is True
    for mutation in (
        _args(tmp_path)[:-1],
        _args(tmp_path) + ["--spatial-artifact", str(tmp_path / "duplicate")],
        _args(tmp_path) + ["--evaluation-manifest", str(tmp_path / "forbidden")],
        _args(tmp_path) + ["--aws-profile", "forbidden"],
    ):
        with pytest.raises(SystemExit):
            parse_args(mutation)


def test_latency_schedule_measures_every_authenticated_image() -> None:
    starts = spatial_tail_batch_starts()
    assert len(starts) == 110
    assert {index for start in starts[10:] for index in range(start, start + 8)} == set(
        range(128)
    )
    assert all(0 <= start <= 120 and start % 8 == 0 for start in starts)


def test_teacher_latency_descriptor_matches_registered_output_without_validation_sync() -> None:
    model = PooledProxyAnchorModel(
        tower=nn.Identity(), input_dimensions=4, embedding_dimensions=3, class_count=2
    ).eval()
    inputs = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 1
    expected = model.encode(inputs)

    def forbidden(_inputs: torch.Tensor) -> torch.Tensor:
        raise AssertionError("validated encode must not run inside the timed region")

    model.encode = forbidden  # type: ignore[method-assign]
    actual = teacher_latency_descriptor(model, inputs)

    assert torch.equal(actual, expected)
