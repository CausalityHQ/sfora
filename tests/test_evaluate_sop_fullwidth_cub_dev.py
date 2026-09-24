"""Full-width transfer development evaluation must preserve split and model authority."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_sop_fullwidth_cub_dev.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("evaluate_sop_fullwidth_cub_dev", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_development_selection_excludes_cub_test_classes() -> None:
    def record(split: str, image_id: int, label: int):
        return MODULE.CubRecord(
            split, image_id, label, "class", "class/image.jpg", Path("/tmp/image.jpg")
        )

    rows = (
        record("train", 1, 1),
        record("train", 2, 100),
        record("test", 3, 101),
        record("test", 4, 200),
    )
    assert MODULE.select_development_records(rows, expected_count=2) == rows[:2]
    with pytest.raises(ValueError, match="CUB development split"):
        MODULE.select_development_records(rows, expected_count=3)
    with pytest.raises(ValueError, match="CUB development split"):
        MODULE.select_development_records((record("train", 5, 101), *rows), expected_count=3)


def test_pca_authority_rejects_checkpoint_or_projection_swap() -> None:
    fit_sha = "e1bc2b8d71de3f8ae8723d2d8548a259bceef2e19d68f8ca2d391d0d519e2199"
    receipt = {
        "schema": "sfora-sop-matched-fullwidth-fit-pca128-v1",
        "claim_eligible": False,
        "step": 48000,
        "seed": 179019,
        "projection_archive_sha256": "p" * 64,
        "projection": {"fit_row_indexes_sha256": fit_sha},
        "inputs": {"fullwidth_checkpoint_sha256": "c" * 64},
    }
    projection = {
        "schema": "sfora-sop-fit-only-fullwidth-pca128-projection-v1",
        "step": 48000,
        "fullwidth_checkpoint_sha256": "c" * 64,
        "fit_row_indexes_sha256": fit_sha,
    }
    MODULE.validate_pca_authority(receipt, projection, "c" * 64, "p" * 64)
    with pytest.raises(ValueError, match="PCA authority"):
        MODULE.validate_pca_authority(receipt, projection, "x" * 64, "p" * 64)
    with pytest.raises(ValueError, match="PCA authority"):
        MODULE.validate_pca_authority(receipt, projection, "c" * 64, "x" * 64)
    with pytest.raises(ValueError, match="PCA authority"):
        MODULE.validate_pca_authority({**receipt, "seed": 3}, projection, "c" * 64, "p" * 64)
    with pytest.raises(ValueError, match="PCA authority"):
        MODULE.validate_pca_authority(
            {**receipt, "projection": {"fit_row_indexes_sha256": "x" * 64}},
            {**projection, "fit_row_indexes_sha256": "x" * 64},
            "c" * 64,
            "p" * 64,
        )


def test_selected_authority_rejects_swapped_receipt_and_checkpoint() -> None:
    receipt = {
        "schema": "sfora-sop-compact-training-diagnostic-v1",
        "claim_eligible": False,
        "arm": "arcface",
        "recipe": "reference",
        "seed": 179019,
        "embedding_width": 768,
        "step": 48000,
        "checkpoint_sha256": MODULE.SELECTED_CHECKPOINT_SHA256,
    }
    MODULE.validate_selected_authority(
        receipt, MODULE.SELECTED_CHECKPOINT_SHA256, MODULE.SELECTED_TRAINING_RECEIPT_SHA256
    )
    with pytest.raises(ValueError, match="selected authority"):
        MODULE.validate_selected_authority(
            receipt, "x" * 64, MODULE.SELECTED_TRAINING_RECEIPT_SHA256
        )
    with pytest.raises(ValueError, match="selected authority"):
        MODULE.validate_selected_authority(receipt, MODULE.SELECTED_CHECKPOINT_SHA256, "x" * 64)
    with pytest.raises(ValueError, match="selected authority"):
        MODULE.validate_selected_authority(
            {**receipt, "claim_eligible": True},
            MODULE.SELECTED_CHECKPOINT_SHA256,
            MODULE.SELECTED_TRAINING_RECEIPT_SHA256,
        )
