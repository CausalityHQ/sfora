from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "screen_unicom_rank_finish.py"
SPEC = importlib.util.spec_from_file_location("screen_unicom_rank_finish", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _metrics(map_at_r: float, r1: float = 0.9862, r10: float = 0.9975):
    return {"map_at_r": map_at_r, "recall_at_1": r1, "recall_at_10": r10}


@pytest.mark.parametrize(
    ("epoch6", "epoch8", "expected"),
    (
        (_metrics(0.8945), None, "ABORT_EPOCH6"),
        (_metrics(0.8980), None, "CONTINUE_EPOCH6"),
        (_metrics(0.8980), _metrics(0.8990), "REJECT"),
        (_metrics(0.8980), _metrics(0.9020), "EXPLORATORY_IMPROVEMENT"),
        (_metrics(0.8980), _metrics(0.9080), "PROMOTE"),
        (_metrics(0.8980), _metrics(0.9080, r1=0.9850), "REJECT"),
        (_metrics(0.8980), _metrics(0.9080, r10=0.9960), "REJECT"),
    ),
)
def test_classify_rank_finish_applies_frozen_gates(epoch6, epoch8, expected) -> None:
    assert MODULE.classify_rank_finish(epoch6, epoch8)["status"] == expected


def test_canonical_result_is_claim_ineligible_and_newline_terminated() -> None:
    result = {
        "schema": "unicom-rank-finish-screen-v1",
        "claim_eligible": False,
        "status": "REJECT",
    }

    payload = MODULE.canonical_result_bytes(result)

    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    assert json.loads(payload) == result
    assert payload == (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def test_parse_args_requires_explicit_execution_and_all_authorities(tmp_path: Path) -> None:
    required = [
        "--source-commit", "a" * 40,
        "--unicom-checkout", str(tmp_path / "unicom"),
        "--official-checkpoint", str(tmp_path / "official.pt"),
        "--official-checkpoint-sha256", "b" * 64,
        "--dataset-root", str(tmp_path / "dataset"),
        "--partition-sha256", "c" * 64,
        "--resume-checkpoint", str(tmp_path / "epoch-0004.pt"),
        "--resume-checkpoint-sha256", "d" * 64,
        "--resume-run-receipt", str(tmp_path / "run-receipt.json"),
        "--resume-run-receipt-sha256", "e" * 64,
        "--finish-seed", "1",
        "--model-output", str(tmp_path / "model.pt"),
        "--output", str(tmp_path / "result.json"),
    ]

    with pytest.raises(SystemExit):
        MODULE.parse_args(required)
    parsed = MODULE.parse_args([*required, "--execute-rank-finish"])
    assert parsed.execute_rank_finish is True
    assert parsed.finish_seed == 1
    with pytest.raises(SystemExit):
        MODULE.parse_args([*required, "--execute-rank-finish", "--unknown"])


def test_only_robustness_seed_can_skip_model_publication() -> None:
    assert MODULE.model_publication_required(1, skip=False) is True
    assert MODULE.model_publication_required(2, skip=True) is False
    with pytest.raises(ValueError):
        MODULE.model_publication_required(0, skip=True)
    with pytest.raises(ValueError):
        MODULE.model_publication_required(1, skip=True)


def test_finish_seed_resets_cpu_rng_deterministically() -> None:
    MODULE.seed_rank_finish_rng(2)
    first = torch.rand(8)
    MODULE.seed_rank_finish_rng(2)
    second = torch.rand(8)
    MODULE.seed_rank_finish_rng(1)
    different = torch.rand(8)

    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_inference_checkpoint_bytes_bind_seed_model_and_parent(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    path = tmp_path / "seed-1-model.pt"

    authority = MODULE.publish_inference_checkpoint(
        path,
        model=model,
        finish_seed=1,
        source_commit="a" * 40,
        parent_checkpoint_sha256="b" * 64,
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "unicom-rank-finish-inference-v1"
    assert payload["finish_seed"] == 1
    assert payload["source_commit"] == "a" * 40
    assert payload["parent_checkpoint_sha256"] == "b" * 64
    assert tuple(payload["model"]) == tuple(model.state_dict())
    assert authority == {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
