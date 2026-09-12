from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.nn import functional as F

from sfora.product_quantization import ProductQuantizationSpec, ProductQuantizer

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_sop_single_stage_differential_pq.py"
sys.path.insert(0, str(_SCRIPT.parent))


def _subject() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_sop_single_stage_differential_pq", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quantizer() -> ProductQuantizer:
    return ProductQuantizer.from_codebooks(
        ProductQuantizationSpec(block_dimensions=(2, 2), codebook_size=3),
        (
            torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
            torch.tensor([[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]]),
        ),
    )


def test_four_arms_start_with_identical_rows_and_codes() -> None:
    subject = _subject()
    teacher = F.normalize(torch.arange(1, 25, dtype=torch.float32).reshape(4, 6), dim=1)
    weight = torch.tensor(
        [
            [0.4, -0.1, 0.2, 0.0, 0.1, -0.2],
            [0.0, 0.3, -0.2, 0.2, 0.1, 0.0],
            [0.1, 0.0, 0.4, -0.1, 0.2, 0.1],
            [-0.2, 0.1, 0.0, 0.3, -0.1, 0.2],
        ]
    )
    bias = torch.tensor([0.01, -0.02, 0.03, -0.04])
    incumbent = F.normalize(F.linear(teacher, weight, bias), dim=1)
    basis = subject.projection_row_space_basis(weight)
    arms = subject.initialize_joint_pq_arms(
        weight, bias, _quantizer(), row_space_basis=basis, device=torch.device("cpu")
    )

    assert (
        tuple(arms)
        == subject.JOINT_PQ_ARM_NAMES
        == (
            "restricted_rank",
            "restricted_differential",
            "full_rank",
            "full_differential",
        )
    )
    baseline_codes = _quantizer().hard_encode(incumbent)
    for model in arms.values():
        projected = model.project(teacher)
        torch.testing.assert_close(projected, incumbent, rtol=0.0, atol=0.0)
        assert torch.equal(model.quantizer.hard_encode(projected), baseline_codes)
    assert len({id(model.quantizer) for model in arms.values()}) == 4


def test_restricted_arms_cannot_learn_from_incumbent_null_space() -> None:
    subject = _subject()
    weight = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    bias = torch.zeros(2)
    quantizer = ProductQuantizer.from_codebooks(
        ProductQuantizationSpec(block_dimensions=(2,), codebook_size=3),
        (torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),),
    )
    basis = subject.projection_row_space_basis(weight)
    arms = subject.initialize_joint_pq_arms(
        weight, bias, quantizer, row_space_basis=basis, device=torch.device("cpu")
    )
    base = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
    null_shifted = torch.tensor([[1.0, 0.5, 2.0, -3.0]])
    update = torch.tensor([[0.0, 0.0, 0.5, 0.2], [0.0, 0.0, -0.3, 0.4]])
    with torch.no_grad():
        for model in arms.values():
            model.projection.weight.add_(update)

    for name in ("restricted_rank", "restricted_differential"):
        torch.testing.assert_close(
            arms[name].project(base), arms[name].project(null_shifted), rtol=0.0, atol=0.0
        )
    assert not torch.equal(arms["full_rank"].project(base), arms["full_rank"].project(null_shifted))


def test_deployment_contract_is_exactly_24_bytes_and_24_lookups() -> None:
    subject = _subject()

    assert subject.joint_pq_deployment_contract() == {
        "bytes_per_vector": 24,
        "codebook_entries": 256,
        "database_sidecar_bytes_per_vector": 0,
        "distance_accumulations_per_candidate": 24,
        "lookup_tables_per_query": 24,
        "stored_code_dtype": "uint8",
    }


def test_decision_uses_absolute_paired_and_mechanism_gates() -> None:
    subject = _subject()
    qualities = {
        "restricted_rank": {"map_at_r": 0.5790, "r1": 0.8240, "pq32_paired_lower": -0.001},
        "restricted_differential": {
            "map_at_r": 0.5800,
            "r1": 0.8240,
            "pq32_paired_lower": 0.0001,
        },
        "full_rank": {"map_at_r": 0.5840, "r1": 0.8240, "pq32_paired_lower": 0.0002},
        "full_differential": {
            "map_at_r": 0.5880,
            "r1": 0.8320,
            "pq32_paired_lower": 0.0003,
        },
    }

    observed = subject.joint_pq_decision(qualities)

    assert observed["classification"] == "strong-go"
    assert observed["selected_arm"] == "full_differential"
    assert observed["passed"] is True
    assert observed["primary_arm"] == "full_differential"
    assert observed["single_stage_mechanism_supported"] is True
    assert observed["differential_mechanism_supported"] is True

    killed = {
        name: {"map_at_r": 0.5771, "r1": 0.83, "pq32_paired_lower": 0.0}
        for name in subject.JOINT_PQ_ARM_NAMES
    }
    assert subject.joint_pq_decision(killed)["classification"] == "kill"

    uncertain = {name: dict(value) for name, value in qualities.items()}
    uncertain["full_differential"]["pq32_paired_lower"] = -0.0001
    assert subject.joint_pq_decision(uncertain)["classification"] == "positive-not-significant"

    with pytest.raises(ValueError, match="joint PQ quality authority differs"):
        subject.joint_pq_decision(
            {**qualities, "full_rank": {**qualities["full_rank"], "r1": True}}
        )


def test_receipt_is_canonical_claim_ineligible_and_strict() -> None:
    subject = _subject()
    receipt = {
        "arms": {name: {"map_at_r": 0.58, "r1": 0.82} for name in subject.JOINT_PQ_ARM_NAMES},
        "claim_eligible": False,
        "dataset": "fixture",
        "decision": {"classification": "ambiguous"},
        "deployment": subject.joint_pq_deployment_contract(),
        "inputs": {},
        "model_checkpoint": {},
        "official_test_touched": False,
        "partition": {},
        "recipe": {},
        "runtime": {},
        "schema": "sfora-single-stage-differential-pq-v1",
        "seed": 0,
        "source": {},
    }

    wire = subject.canonical_joint_pq_receipt(receipt)

    assert wire.endswith(b"\n") and not wire.endswith(b"\n\n")
    assert json.loads(wire) == receipt
    assert wire == subject.canonical_joint_pq_receipt(dict(reversed(tuple(receipt.items()))))
    with pytest.raises(ValueError, match="joint PQ receipt authority differs"):
        subject.canonical_joint_pq_receipt({**receipt, "official_test_touched": 0})


def test_cli_requires_explicit_execution_and_rejects_unknown_flags(tmp_path: Path) -> None:
    subject = _subject()
    required = [
        "--source-snapshot",
        str(tmp_path / "source.pt"),
        "--source-sha256",
        "1" * 64,
        "--teacher-snapshot",
        str(tmp_path / "teacher.pt"),
        "--teacher-sha256",
        "2" * 64,
        "--direct-checkpoint",
        str(tmp_path / "direct.pt"),
        "--direct-checkpoint-sha256",
        "3" * 64,
        "--parent-codec-checkpoint",
        str(tmp_path / "codec.pt"),
        "--parent-codec-checkpoint-sha256",
        "4" * 64,
        "--parent-codec-receipt",
        str(tmp_path / "codec.json"),
        "--parent-codec-receipt-sha256",
        "5" * 64,
        "--receipt",
        str(tmp_path / "receipt.json"),
        "--model-output",
        str(tmp_path / "model.pt"),
        "--seed",
        "0",
        "--source-revision",
        "6" * 40,
        "--driver-sha256",
        "7" * 64,
    ]

    with pytest.raises(SystemExit):
        subject.parse_arguments(required)
    with pytest.raises(SystemExit):
        subject.parse_arguments(
            [*required, "--execute-single-stage-differential-pq", "--bucket", "x"]
        )
    parsed = subject.parse_arguments([*required, "--execute-single-stage-differential-pq"])
    assert parsed.seed == 0
