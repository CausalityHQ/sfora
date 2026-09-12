from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_sop_projection_parameterizations.py"
sys.path.insert(0, str(_SCRIPT.parent))


def _subject() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_sop_projection_parameterizations", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parameterizations_start_from_the_same_effective_affine_map() -> None:
    subject = _subject()
    weight = torch.tensor([[0.2, -0.4, 0.7], [0.5, 0.1, -0.3]], dtype=torch.float32)
    bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
    teacher = torch.tensor([[1.0, 2.0, -1.0], [-0.5, 0.25, 0.75]], dtype=torch.float32)

    models = subject.initialize_matched_parameterizations(weight, bias, device=torch.device("cpu"))
    base = torch.nn.functional.normalize(torch.nn.functional.linear(teacher, weight, bias), dim=-1)
    restricted = subject.encode_parameterization(
        "restricted_adapter", models["restricted_adapter"], teacher, base
    )
    direct = subject.encode_parameterization(
        "direct_projection", models["direct_projection"], teacher, base
    )

    torch.testing.assert_close(restricted, base, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(direct, base, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(models["direct_projection"].weight, weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(models["direct_projection"].bias, bias, rtol=0.0, atol=0.0)


def test_parameterization_encoder_rejects_unknown_arm() -> None:
    subject = _subject()
    model = torch.nn.Linear(2, 2)
    rows = torch.eye(2)
    with pytest.raises(ValueError, match="parameterization arm"):
        subject.encode_parameterization("unknown", model, rows, rows)


def test_scientific_projection_shape_is_exactly_768_to_128() -> None:
    subject = _subject()
    subject.validate_scientific_projection_shape(torch.ones((128, 768)), torch.zeros(128))
    with pytest.raises(ValueError, match="scientific projection shape"):
        subject.validate_scientific_projection_shape(torch.ones((127, 768)), torch.zeros(127))


def test_restricted_adapter_folds_to_one_deployable_affine_head() -> None:
    subject = _subject()
    base_weight = torch.tensor([[0.2, -0.4, 0.7], [0.5, 0.1, -0.3]], dtype=torch.float32)
    base_bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
    adapter_weight = torch.tensor([[1.1, 0.2], [-0.3, 0.8]], dtype=torch.float32)
    teacher = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float32)

    folded_weight, folded_bias = subject.fold_restricted_adapter(
        adapter_weight, base_weight, base_bias
    )

    expected = torch.nn.functional.linear(
        torch.nn.functional.linear(teacher, base_weight, base_bias), adapter_weight
    )
    observed = torch.nn.functional.linear(teacher, folded_weight, folded_bias)
    torch.testing.assert_close(observed, expected, rtol=2e-6, atol=2e-7)


def test_parameterization_state_uses_role_specific_schema_and_digest() -> None:
    subject = _subject()
    models = subject.initialize_matched_parameterizations(
        torch.ones((2, 3)), torch.zeros(2), device=torch.device("cpu")
    )

    restricted_state, restricted_sha = subject.parameterization_state(
        "restricted_adapter", models["restricted_adapter"]
    )
    direct_state, direct_sha = subject.parameterization_state(
        "direct_projection", models["direct_projection"]
    )

    assert tuple(restricted_state) == ("weight",)
    assert tuple(direct_state) == ("weight", "bias")
    assert restricted_sha == subject.artifacts.linear_weight_sha256(restricted_state["weight"])
    assert direct_sha == subject.artifacts.affine_parameters_sha256(
        direct_state["weight"], direct_state["bias"]
    )


def test_direct_projection_learning_rate_matches_deployed_map_scale() -> None:
    subject = _subject()
    assert subject.parameterization_learning_rate("restricted_adapter") == pytest.approx(
        subject.coverage.LEARNING_RATE
    )
    assert subject.parameterization_learning_rate("direct_projection") == pytest.approx(
        subject.coverage.LEARNING_RATE * np.sqrt(128.0 / 768.0)
    )


@pytest.mark.parametrize("name", ("restricted_adapter", "direct_projection"))
def test_train_parameterization_arm_preserves_matched_evidence_shape(name: str) -> None:
    subject = _subject()
    teacher = torch.tensor(
        [
            [1.0, 0.0, 0.2],
            [0.9, 0.1, 0.2],
            [0.0, 1.0, -0.1],
            [0.1, 0.9, -0.1],
            [-1.0, 0.0, 0.3],
            [-0.9, 0.1, 0.3],
        ],
        dtype=torch.float32,
    )
    weight = torch.tensor([[1.0, 0.1, 0.0], [0.0, 1.0, 0.1]])
    bias = torch.tensor([0.05, -0.05])
    base = torch.nn.functional.normalize(torch.nn.functional.linear(teacher, weight, bias), dim=-1)
    labels = (1, 1, 2, 2, 3, 3)
    negative_table = subject.controls.frozen_hard_negative_index(
        base, labels, device=torch.device("cpu"), k=2, block_size=3
    )
    model = subject.initialize_matched_parameterizations(weight, bias, device=torch.device("cpu"))[
        name
    ]

    result, state = subject.train_parameterization_arm(
        name,
        model,
        teacher,
        teacher,
        base,
        base,
        weight,
        bias,
        labels,
        labels,
        (np.asarray([0, 2, 4]),),
        negative_table,
        device=torch.device("cpu"),
    )

    expected_result_keys = {
        "deployed_relative_frobenius_displacement",
        "deployment_equivalence",
        "deployed_head_sha256",
        "final_loss",
        "mean_last_100_loss",
        "parameter_sha256",
        "score",
    }
    if name == "restricted_adapter":
        expected_result_keys.add("base_head_sha256")
    assert set(result) == expected_result_keys
    assert np.isfinite(result["final_loss"])
    assert result["deployed_relative_frobenius_displacement"] >= 0.0
    if name == "restricted_adapter":
        assert result["deployment_equivalence"]["maximum_absolute_code_delta"] <= 2e-6
        assert set(state) == {"weight", "base_head_weight", "base_head_bias"}
    else:
        assert result["deployment_equivalence"] == "identical-parameterization"
        assert set(state) == {"weight", "bias"}


def test_projection_decision_uses_paired_class_cluster_evidence() -> None:
    subject = _subject()
    restricted = {
        "score": {
            "packed_map_at_r": 0.50,
            "packed_per_query_ap": [0.4, 0.6, 0.4, 0.6],
            "packed_r1": 0.70,
        }
    }
    direct = {
        "score": {
            "packed_map_at_r": 0.51,
            "packed_per_query_ap": [0.41, 0.61, 0.41, 0.61],
            "packed_r1": 0.71,
        }
    }

    decision = subject.projection_decision(restricted, direct, validation_labels=(1, 1, 2, 2))

    assert decision["gates"] == {
        "class_clustered_lower_bound": 0.0,
        "packed_map_at_r_gain": 0.003,
        "packed_r1_gain": 0.0,
    }
    assert decision["observed"]["packed_map_at_r_gain"] == pytest.approx(0.01)
    assert decision["passed"] is True


def test_projection_parameterization_driver_requires_explicit_execution() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "--execute-projection-parameterizations" in result.stderr
