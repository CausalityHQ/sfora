"""Train-only selection and provenance checks for the long SOP recipe."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sfora.sop_reference_selection import select_reference_checkpoint

STEPS = (4000, 8000, 16000, 32000, 48000, 53760)
SOURCE = {"scripts/train_sop_compact_backbone.py": "b" * 64}
IDS = list(range(5851))
LABELS = [index // 5 for index in IDS]


def _candidate(tmp_path: Path, step: int, map_at_r: float, recall: float):
    checkpoint = tmp_path / (
        "arcface-seed179019-53760.pt"
        if step == 53760
        else f"arcface-seed179019-53760.step{step}.pt"
    )
    checkpoint.write_bytes(f"checkpoint {step}".encode())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    shared = {
        "claim_eligible": False,
        "recipe": "reference",
        "arm": "arcface",
        "seed": 179019,
        "schedule_sha256": "c" * 64,
        "fit_row_indexes_sha256": "d" * 64,
        "validation_row_indexes_sha256": "e" * 64,
        "validation_image_ids": IDS,
        "validation_labels": LABELS,
        "validation": {"packed": {"map_at_r": map_at_r, "recall_at_1": recall}},
    }
    source_inputs = {
        "checkpoint_sha256": "1" * 64,
        "features_archive_sha256": "2" * 64,
        "sop_train_metadata_sha256": "3" * 64,
        "source_sha256": SOURCE,
        "upstream_retrieval_sha256": "4" * 64,
        "upstream_launch_sha256": "5" * 64,
        "train_transform_sha256": "6" * 64,
    }
    if step == 53760:
        receipt = {
            **shared,
            "schema": "sfora-sop-compact-full-backbone-v1",
            "embedding_width": 128,
            "updates": step,
            "fit_images": 53700,
            "validation_images": 5851,
            "batch_size": 64,
            "images_per_identity": 4,
            "split_seed": 179019,
            "selection_checkpoint_steps": list(STEPS),
            "inputs": {
                **source_inputs,
                "checkpoint_output_sha256": digest,
            },
        }
    else:
        receipt = {
            **shared,
            "schema": "sfora-sop-compact-training-diagnostic-v1",
            "resumable": False,
            "embedding_width": 128,
            "step": step,
            "total_updates": 53760,
            "input_checkpoint_sha256": source_inputs["checkpoint_sha256"],
            "features_archive_sha256": source_inputs["features_archive_sha256"],
            "sop_train_metadata_sha256": source_inputs["sop_train_metadata_sha256"],
            "source_sha256": SOURCE,
            "upstream_retrieval_sha256": source_inputs["upstream_retrieval_sha256"],
            "upstream_launch_sha256": source_inputs["upstream_launch_sha256"],
            "train_transform_sha256": source_inputs["train_transform_sha256"],
            "checkpoint_sha256": digest,
        }
    return checkpoint, receipt


def test_selects_best_completed_train_holdout_checkpoint(tmp_path: Path):
    candidates = [
        _candidate(tmp_path, step, 0.80 if step in (8000, 16000) else 0.70, 0.90) for step in STEPS
    ]
    chosen = select_reference_checkpoint(candidates, source_manifest=SOURCE)
    assert chosen.step == 8000
    assert chosen.checkpoint == candidates[1][0]


def test_rejects_missing_final_and_mismatched_source(tmp_path: Path):
    candidates = [_candidate(tmp_path, step, 0.7, 0.9) for step in STEPS]
    with pytest.raises(ValueError, match="complete reference run"):
        select_reference_checkpoint(candidates[:-1], source_manifest=SOURCE)
    with pytest.raises(ValueError, match="training source"):
        select_reference_checkpoint(candidates, source_manifest={"other": "f" * 64})


def test_rejects_changed_checkpoint_or_selection_metadata(tmp_path: Path):
    candidates = [_candidate(tmp_path, step, 0.7, 0.9) for step in STEPS]
    candidates[0][0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint digest"):
        select_reference_checkpoint(candidates, source_manifest=SOURCE)
    candidates[0] = _candidate(tmp_path, STEPS[0], 0.7, 0.9)
    candidates[1][1]["validation_image_ids"] = list(reversed(IDS))
    with pytest.raises(ValueError, match="matched training inventory"):
        select_reference_checkpoint(candidates, source_manifest=SOURCE)


def test_b8_receipts_without_embedding_width_select_128_dimensional_run(tmp_path: Path):
    candidates = [_candidate(tmp_path, step, 0.7, 0.9) for step in STEPS]
    for _, receipt in candidates:
        receipt.pop("embedding_width")
    chosen = select_reference_checkpoint(candidates, source_manifest=SOURCE)
    assert chosen.step == STEPS[0]


def test_rejects_checkpoint_from_another_run_even_if_bytes_match(tmp_path: Path):
    candidates = [_candidate(tmp_path, step, 0.7, 0.9) for step in STEPS]
    alternate = tmp_path / "other.step4000.pt"
    alternate.write_bytes(candidates[0][0].read_bytes())
    candidates[0] = (alternate, candidates[0][1])
    with pytest.raises(ValueError, match="run checkpoint path"):
        select_reference_checkpoint(candidates, source_manifest=SOURCE)
