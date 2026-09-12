from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sfora.teacher_anchored_distillation import (
    multi_similarity_hard_negative_loss,
    positive_coverage_hard_negative_loss,
    stable_different_class_topk,
    supervised_contrastive_hard_negative_loss,
)

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_sop_similarity_loss_controls.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("run_sop_similarity_loss_controls", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
controls = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = controls
_SPEC.loader.exec_module(controls)


@pytest.mark.parametrize("name", ("pooled", "coverage", "mean_logit", "supcon", "multi_similarity"))
def test_matched_control_loss_dispatches_published_objectives(name: str) -> None:
    positives = torch.tensor([[0.9, 0.2]], dtype=torch.float32, requires_grad=True)
    mask = torch.tensor([[True, True]])
    negatives = torch.tensor([[0.3, -0.5]], dtype=torch.float32, requires_grad=True)
    self_similarities = torch.tensor([0.8], dtype=torch.float32, requires_grad=True)

    observed = controls.matched_control_loss(
        name,
        positives,
        mask,
        negatives,
        self_similarities,
    )
    if name in {"pooled", "coverage", "mean_logit"}:
        expected = positive_coverage_hard_negative_loss(
            positives,
            mask,
            negatives,
            self_similarities,
            temperature=controls.TEMPERATURE,
            margin=controls.MARGIN,
            anchor_weight=controls.ANCHOR_WEIGHT,
            positive_aggregation=name,
        )
    elif name == "supcon":
        expected = supervised_contrastive_hard_negative_loss(
            positives,
            mask,
            negatives,
            self_similarities,
            temperature=controls.TEMPERATURE,
            margin=controls.MARGIN,
            anchor_weight=controls.ANCHOR_WEIGHT,
        )
    else:
        expected = multi_similarity_hard_negative_loss(
            positives,
            mask,
            negatives,
            self_similarities,
            alpha=2.0,
            beta=50.0,
            base=0.5,
            mining_margin=0.1,
            anchor_weight=controls.ANCHOR_WEIGHT,
        )
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_matched_control_loss_rejects_unknown_objective() -> None:
    values = torch.ones((1, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="matched control objective"):
        controls.matched_control_loss(
            "unknown",
            values,
            torch.ones((1, 1), dtype=torch.bool),
            values,
            values[:, 0],
        )


def test_similarity_loss_control_driver_requires_explicit_execution() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--execute-matched-controls" in result.stderr


def test_frozen_negative_index_matches_direct_mining_across_blocks() -> None:
    base = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [-1.0, 0.0],
                [-0.9, 0.1],
            ]
        ),
        dim=-1,
    )
    labels = (0, 0, 1, 1, 2, 2)
    label_tensor = torch.tensor(labels, dtype=torch.int64)
    expected = stable_different_class_topk(
        base,
        base,
        label_tensor,
        label_tensor,
        k=2,
    )

    observed = controls.frozen_hard_negative_index(
        base,
        labels,
        device=torch.device("cpu"),
        k=2,
        block_size=2,
    )

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("parameter_sha256", "expected_map", "expected_r1"),
    (("0" * 64, 0.5, 0.75), ("A" * 64, 0.5, 0.75), ("0" * 64, np.nan, 0.75)),
)
def test_base_authority_requires_lowercase_digest_and_finite_scores(
    parameter_sha256: str, expected_map: float, expected_r1: float
) -> None:
    if parameter_sha256 == "0" * 64 and np.isfinite(expected_map):
        assert controls.validate_base_authority(
            parameter_sha256=parameter_sha256,
            expected_map=expected_map,
            expected_r1=expected_r1,
        ) == (parameter_sha256, expected_map, expected_r1)
    else:
        with pytest.raises(ValueError, match="base authority"):
            controls.validate_base_authority(
                parameter_sha256=parameter_sha256,
                expected_map=expected_map,
                expected_r1=expected_r1,
            )


def test_canonical_arm_result_excludes_wall_clock_timing() -> None:
    weight = torch.eye(2)
    result = controls.canonical_arm_result(
        losses=[2.0, 1.0],
        weight=weight,
        score={"packed_map_at_r": 0.5, "packed_r1": 0.75},
    )

    assert result == {
        "final_loss": 1.0,
        "mean_last_100_loss": 1.5,
        "parameter_sha256": controls.coverage._parameter_sha256(weight),
        "score": {"packed_map_at_r": 0.5, "packed_r1": 0.75},
    }


def test_integer_tensor_sha256_binds_shape_dtype_and_values() -> None:
    value = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    expected = controls.integer_tensor_sha256(value)

    assert controls.integer_tensor_sha256(value.clone()) == expected
    assert controls.integer_tensor_sha256(value.reshape(1, 4)) != expected
    mutated = value.clone()
    mutated[0, 0] += 1
    assert controls.integer_tensor_sha256(mutated) != expected


def test_base_receipt_authenticates_parent_identity(tmp_path: Path) -> None:
    parameter = "1" * 64
    receipt = {
        "claim_eligible": False,
        "dataset": "sop-official-train-class-disjoint-validation",
        "expected_packed_map_at_r": 0.5,
        "expected_packed_r1": 0.75,
        "final_head_sha256": parameter,
        "official_test_touched": False,
        "schema": "sfora-retrieval-local-rank-replay-v1",
        "source_revision": "2" * 40,
    }
    path = tmp_path / "base.json"
    wire = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(wire)
    digest = hashlib.sha256(wire).hexdigest()

    assert controls.load_base_receipt_authority(
        path=path,
        sha256=digest,
        parameter_sha256=parameter,
        expected_map=0.5,
        expected_r1=0.75,
    ) == {
        "sha256": digest,
        "source_revision": "2" * 40,
    }

    receipt["official_test_touched"] = True
    path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    drift_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="base receipt authority"):
        controls.load_base_receipt_authority(
            path=path,
            sha256=drift_digest,
            parameter_sha256=parameter,
            expected_map=0.5,
            expected_r1=0.75,
        )


def test_base_receipt_authenticates_current_v2_source_and_inputs(tmp_path: Path) -> None:
    parameter = "1" * 64
    source_sha = "3" * 64
    teacher_sha = "4" * 64
    receipt = {
        "claim_eligible": False,
        "dataset": "sop-official-train-class-disjoint-validation",
        "expected_head_sha256": parameter,
        "expected_packed_map_at_r": 0.5,
        "expected_packed_r1": 0.75,
        "final_head_sha256": parameter,
        "inputs": {
            "source_snapshot_sha256": source_sha,
            "teacher_snapshot_sha256": teacher_sha,
        },
        "matched": True,
        "official_test_touched": False,
        "schema": "sfora-retrieval-local-rank-replay-v2",
        "source": {"driver_sha256": "5" * 64, "source_revision": "2" * 40},
    }
    path = tmp_path / "base-v2.json"
    wire = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(wire)
    digest = hashlib.sha256(wire).hexdigest()

    assert controls.load_base_receipt_authority(
        path=path,
        sha256=digest,
        parameter_sha256=parameter,
        expected_map=0.5,
        expected_r1=0.75,
        source_snapshot_sha256=source_sha,
        teacher_snapshot_sha256=teacher_sha,
    ) == {"sha256": digest, "source_revision": "2" * 40}

    receipt["inputs"]["teacher_snapshot_sha256"] = "6" * 64
    path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="base receipt authority"):
        controls.load_base_receipt_authority(
            path=path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            parameter_sha256=parameter,
            expected_map=0.5,
            expected_r1=0.75,
            source_snapshot_sha256=source_sha,
            teacher_snapshot_sha256=teacher_sha,
        )
