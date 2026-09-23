"""Official-test evaluation must be bound to a completed training receipt."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_sop_compact_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("evaluate_sop_compact_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
sys.path.insert(0, str(SCRIPT.parent))
SPEC.loader.exec_module(MODULE)
sys.path.pop(0)


def _receipt():
    return {
        "schema": "sfora-sop-compact-full-backbone-v1",
        "arm": "packed_rank",
        "seed": 179019,
        "updates": 1000,
        "fit_images": 53700,
        "validation_images": 5851,
        "schedule_sha256": "c" * 64,
        "fit_row_indexes_sha256": "d" * 64,
        "validation_row_indexes_sha256": "e" * 64,
        "validation": {"packed": {"map_at_r": 0.73, "recall_at_1": 0.92}},
        "inputs": {
            "checkpoint_output_sha256": "a" * 64,
            "checkpoint_sha256": MODULE.CHECKPOINT_SHA256,
            "source_sha256": MODULE.source_manifest(),
            "initial_head_sha256": "f" * 64,
            "initial_classifier_sha256": "1" * 64,
        },
    }


def test_training_receipt_binds_checkpoint_bytes_and_split():
    assert MODULE.validate_training_receipt(_receipt(), "a" * 64) == (
        "packed_rank",
        179019,
        1000,
    )
    with pytest.raises(ValueError, match="training receipt"):
        MODULE.validate_training_receipt(_receipt(), "b" * 64)
    wrong_split = _receipt()
    wrong_split["validation_images"] = 6000
    with pytest.raises(ValueError, match="training receipt"):
        MODULE.validate_training_receipt(wrong_split, "a" * 64)


def test_selection_uses_only_matched_train_holdout_packed_map():
    receipts = []
    for arm, score in (("arcface", 0.72), ("float_rank", 0.74), ("packed_rank", 0.73)):
        receipt = _receipt()
        receipt["arm"] = arm
        receipt["validation"]["packed"]["map_at_r"] = score
        receipts.append(receipt)
    assert MODULE.select_training_arm(receipts) == "float_rank"
    receipts[2]["schedule_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="selection receipts"):
        MODULE.select_training_arm(receipts)


def test_selection_tie_breaks_with_recall_then_arm_name():
    receipts = []
    for arm, recall in (("arcface", 0.92), ("float_rank", 0.93), ("packed_rank", 0.93)):
        receipt = _receipt()
        receipt["arm"] = arm
        receipt["validation"]["packed"]["recall_at_1"] = recall
        receipts.append(receipt)
    assert MODULE.select_training_arm(receipts) == "float_rank"
