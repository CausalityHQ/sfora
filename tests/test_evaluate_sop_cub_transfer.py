"""The CUB transfer check must bind to one train-selected SOP checkpoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_sop_cub_transfer.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("evaluate_sop_cub_transfer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_transfer_authority_requires_selected_checkpoint_and_training_receipt() -> None:
    official = {
        "schema": "sfora-sop-reference-official-test-v1",
        "claim_eligible": False,
        "selection_step": 32000,
        "seed": 179019,
        "embedding_width": 128,
        "inputs": {
            "selected_checkpoint_sha256": "a" * 64,
            "training_receipt_sha256": {"32000": "b" * 64},
        },
    }
    training = {
        "schema": "sfora-sop-compact-training-diagnostic-v1",
        "arm": "arcface",
        "recipe": "reference",
        "seed": 179019,
        "step": 32000,
        "checkpoint_sha256": "a" * 64,
    }
    MODULE.validate_transfer_authority(official, training, "a" * 64, "b" * 64)
    for target, key, bad in (
        (official, "selection_step", 16000),
        (official, "embedding_width", 768),
        (training, "seed", 7),
        (training, "checkpoint_sha256", "c" * 64),
    ):
        changed = target.copy()
        changed[key] = bad
        with pytest.raises(ValueError, match="transfer authority"):
            MODULE.validate_transfer_authority(
                changed if target is official else official,
                changed if target is training else training,
                "a" * 64,
                "b" * 64,
            )
    with pytest.raises(ValueError, match="transfer authority"):
        MODULE.validate_transfer_authority(official, training, "a" * 64, "c" * 64)
