"""Tests for the streamed local-only intermediate SigLIP readout screen."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose_siglip_intermediate_readout.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "scripts.diagnose_siglip_intermediate_readout", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_layernorm = nn.LayerNorm(4)


class _TinyVisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = _TinyTransformer()
        self.calls = 0

    def forward(self, *, pixel_values, output_hidden_states, return_dict):
        assert output_hidden_states is True and return_dict is True
        self.calls += 1
        base = pixel_values.flatten(1).unsqueeze(1)
        hidden_states = (base, base + 1.0, base + 2.0, base + 3.0)
        return SimpleNamespace(hidden_states=hidden_states)


class _TinyControl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tower = SimpleNamespace(vision_model=_TinyVisionModel())
        self.projection = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))


def test_streamed_readout_uses_one_forward_post_ln_mean_projection_and_no_token_cache() -> None:
    model = _TinyControl()
    batches = (
        torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=torch.float32),
        torch.tensor([[[[2.0, 1.0]], [[4.0, 3.0]]]], dtype=torch.float32),
    )

    planes = _MODULE.stream_intermediate_descriptor_planes(
        model,
        batches,
        expected_depth_count=3,
        tower_width=4,
        output_dimensions=2,
        device=torch.device("cpu"),
    )

    assert model.tower.vision_model.calls == len(batches)
    assert len(planes) == 3
    assert all(value.shape == (2, 2) for value in planes)
    assert all(value.device.type == "cpu" and value.dtype == torch.float32 for value in planes)
    assert all(
        torch.allclose(torch.linalg.vector_norm(value, dim=1), torch.ones(2)) for value in planes
    )
    first_hidden = batches[0].flatten(1).unsqueeze(1) + 1.0
    expected = F.normalize(
        model.projection(
            model.tower.vision_model.vision_model.post_layernorm(first_hidden).mean(dim=1)
        ),
        dim=1,
    )
    assert torch.allclose(planes[0][0], expected[0])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda output: output.hidden_states.__setitem__(1, torch.zeros(1)),
        lambda output: output.hidden_states[-1].fill_(torch.nan),
    ],
)
def test_streamed_readout_rejects_hidden_topology_and_nonfinite_values(mutation) -> None:
    model = _TinyControl()
    original = model.tower.vision_model.forward

    def broken(**kwargs):
        output = original(**kwargs)
        output.hidden_states = list(output.hidden_states)
        mutation(output)
        return output

    model.tower.vision_model.forward = broken
    with pytest.raises(ValueError, match="intermediate hidden-state authority differs"):
        _MODULE.stream_intermediate_descriptor_planes(
            model,
            (torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]),),
            expected_depth_count=3,
            tower_width=4,
            output_dimensions=2,
            device=torch.device("cpu"),
        )


def test_cli_requires_local_optimization_authority_and_refuses_evaluation_capabilities() -> None:
    base = [
        "--control-binding",
        "/authority/control.json",
        "--control-binding-sha256",
        "1" * 64,
        "--checkpoint-seed17",
        "/authority/seed17.pt",
        "--optimization-manifest",
        "/authority/optimization.json",
        "--optimization-manifest-sha256",
        "2" * 64,
        "--image-root",
        "/authority/images",
        "--result",
        "/result/intermediate.json",
        "--execute-intermediate-readout",
    ]
    parsed = _MODULE.parse_args(base)
    assert parsed.checkpoint_seed17 == Path("/authority/seed17.pt")
    assert parsed.execute_intermediate_readout is True
    for forbidden in (
        "--clean-validation",
        "--burned-diagnostic",
        "--official-test",
        "--checkpoint-seed29",
        "--model-revision",
        "--network",
    ):
        with pytest.raises(SystemExit):
            _MODULE.parse_args([*base, forbidden, "/forbidden"])
