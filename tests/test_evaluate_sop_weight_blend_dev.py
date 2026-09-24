"""The development blend must preserve both pretrained and trained endpoints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_sop_weight_blend_dev.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("evaluate_sop_weight_blend_dev", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_blend_state_preserves_endpoints_and_interpolates_float_tensors() -> None:
    source = {"weight": torch.tensor([1.0, 3.0]), "counter": torch.tensor(2)}
    trained = {"weight": torch.tensor([5.0, -1.0]), "counter": torch.tensor(2)}
    assert MODULE.blend_state(source, trained, 0.0)["weight"] is source["weight"]
    assert MODULE.blend_state(source, trained, 1.0)["weight"] is trained["weight"]
    assert torch.equal(MODULE.blend_state(source, trained, 0.0)["weight"], source["weight"])
    assert torch.equal(MODULE.blend_state(source, trained, 1.0)["weight"], trained["weight"])
    assert torch.equal(
        MODULE.blend_state(source, trained, 0.25)["weight"], torch.tensor([2.0, 2.0])
    )


def test_blend_state_rejects_incompatible_states_or_alpha() -> None:
    source = {"weight": torch.tensor([1.0]), "counter": torch.tensor(2)}
    trained = {"weight": torch.tensor([5.0]), "counter": torch.tensor(3)}
    with pytest.raises(ValueError, match="blend state"):
        MODULE.blend_state(source, trained, 0.5)
    with pytest.raises(ValueError, match="blend state"):
        MODULE.blend_state(source, {"other": torch.tensor([5.0])}, 0.5)
    with pytest.raises(ValueError, match="blend state"):
        MODULE.blend_state(source, source, 1.5)
    with pytest.raises(ValueError, match="blend state"):
        MODULE.blend_state(
            {"weight": torch.tensor([float("nan")])},
            {"weight": torch.tensor([1.0])},
            0.0,
        )


def test_blend_state_preserves_batchnorm_counter_endpoints() -> None:
    key = "feature.1.num_batches_tracked"
    source = {key: torch.tensor(0), "feature.1.running_mean": torch.tensor([1.0])}
    trained = {key: torch.tensor(48000), "feature.1.running_mean": torch.tensor([3.0])}
    assert MODULE.blend_state(source, trained, 0.0)[key] is source[key]
    assert MODULE.blend_state(source, trained, 1.0)[key] is trained[key]
    assert MODULE.blend_state(source, trained, 0.5)[key] is source[key]
    assert torch.equal(
        MODULE.blend_state(source, trained, 0.5)["feature.1.running_mean"],
        torch.tensor([2.0]),
    )
