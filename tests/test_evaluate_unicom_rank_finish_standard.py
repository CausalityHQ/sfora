from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "evaluate_unicom_rank_finish_standard.py"
)
SPEC = importlib.util.spec_from_file_location("rank_finish_standard", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _metrics(map_at_r: float, r1: float, r10: float):
    return {
        "map_at_r": map_at_r,
        "recall_at_1": r1,
        "recall_at_10": r10,
        "recall_at_20": 0.999,
        "recall_at_30": 1.0,
    }


def test_standard_gate_requires_map_gain_and_recall_noninferiority() -> None:
    baseline = _metrics(0.760, 0.940, 0.990)

    passed = MODULE.classify_standard(
        baseline, _metrics(0.764, 0.9395, 0.9895)
    )
    failed_map = MODULE.classify_standard(
        baseline, _metrics(0.7629, 0.940, 0.990)
    )
    failed_recall = MODULE.classify_standard(
        baseline, _metrics(0.764, 0.9389, 0.990)
    )

    assert passed["status"] == "RELEASE"
    assert failed_map["status"] == "REJECT"
    assert failed_recall["status"] == "REJECT"


def test_inference_loader_authenticates_seed_parent_and_source(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    payload = {
        "schema": "unicom-rank-finish-inference-v1",
        "finish_seed": 1,
        "source_commit": "a" * 40,
        "parent_checkpoint_sha256": "b" * 64,
        "model": {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
    }
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    state = MODULE.load_inference_checkpoint(
        path,
        expected_sha256=digest,
        expected_seed=1,
        expected_source_commit="a" * 40,
        expected_parent_checkpoint_sha256="b" * 64,
    )

    assert torch.equal(state["weight"], payload["model"]["weight"])
    with pytest.raises(ValueError):
        MODULE.load_inference_checkpoint(
            path,
            expected_sha256=digest,
            expected_seed=2,
            expected_source_commit="a" * 40,
            expected_parent_checkpoint_sha256="b" * 64,
        )
