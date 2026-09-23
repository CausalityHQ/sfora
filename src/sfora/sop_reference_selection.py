"""Frozen train-only checkpoint selection for the long SOP training recipe."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REFERENCE_STEPS = (4000, 8000, 16000, 32000, 48000, 53760)
_SOURCE_KEYS = (
    "checkpoint_sha256",
    "features_archive_sha256",
    "sop_train_metadata_sha256",
    "source_sha256",
    "upstream_retrieval_sha256",
    "upstream_launch_sha256",
    "train_transform_sha256",
)


@dataclass(frozen=True)
class ReferenceSelection:
    checkpoint: Path
    receipt: Mapping[str, object]
    step: int
    packed_map_at_r: float
    packed_recall_at_1: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(receipt: Mapping[str, object], final: bool) -> tuple[object, ...]:
    if final:
        inputs = receipt.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("reference training inputs differ")
        source = inputs
    else:
        source = receipt
    return tuple(
        source.get("input_checkpoint_sha256" if not final and key == "checkpoint_sha256" else key)
        for key in _SOURCE_KEYS
    )


def select_reference_checkpoint(
    candidates: Sequence[tuple[Path, Mapping[str, object]]],
    *,
    source_manifest: Mapping[str, str],
) -> ReferenceSelection:
    """Require a completed matched run, then choose by holdout packed mAP@R."""

    if len(candidates) != len(REFERENCE_STEPS):
        raise ValueError("complete reference run requires all recorded checkpoints")
    if not source_manifest or not all(
        isinstance(key, str) and isinstance(value, str) and len(value) == 64
        for key, value in source_manifest.items()
    ):
        raise ValueError("reference training source differs")
    normalized: list[ReferenceSelection] = []
    baseline_inventory: tuple[object, ...] | None = None
    steps: set[int] = set()
    for checkpoint, receipt in candidates:
        if not isinstance(checkpoint, Path) or not isinstance(receipt, Mapping):
            raise ValueError("reference candidate differs")
        final = receipt.get("schema") == "sfora-sop-compact-full-backbone-v1"
        step = receipt.get("updates" if final else "step")
        if (
            type(step) is not int
            or step not in REFERENCE_STEPS
            or (final != (step == REFERENCE_STEPS[-1]))
            or receipt.get("schema")
            not in (
                "sfora-sop-compact-full-backbone-v1",
                "sfora-sop-compact-training-diagnostic-v1",
            )
            or receipt.get("claim_eligible") is not False
            or receipt.get("recipe") != "reference"
            or receipt.get("arm") != "arcface"
            or type(receipt.get("seed")) is not int
            or receipt.get("embedding_width", 128) not in (128, 768)
        ):
            raise ValueError("reference candidate recipe differs")
        if final:
            if (
                receipt.get("fit_images") != 53_700
                or receipt.get("validation_images") != 5_851
                or receipt.get("batch_size") != 64
                or receipt.get("images_per_identity") != 4
                or receipt.get("split_seed") != 179019
                or receipt.get("selection_checkpoint_steps") != list(REFERENCE_STEPS)
            ):
                raise ValueError("complete reference run differs")
            checkpoint_digest = (
                receipt["inputs"].get("checkpoint_output_sha256")
                if isinstance(receipt.get("inputs"), Mapping)
                else None
            )
        else:
            if (
                receipt.get("resumable") is not False
                or receipt.get("total_updates") != REFERENCE_STEPS[-1]
            ):
                raise ValueError("complete reference run differs")
            checkpoint_digest = receipt.get("checkpoint_sha256")
        if not checkpoint.is_file() or _sha256(checkpoint) != checkpoint_digest:
            raise ValueError("reference checkpoint digest differs")
        source_inputs = _provenance(receipt, final)
        if source_inputs[_SOURCE_KEYS.index("source_sha256")] != dict(source_manifest):
            raise ValueError("reference training source differs")
        if any(
            not isinstance(value, str) or len(value) != 64
            for index, value in enumerate(source_inputs)
            if _SOURCE_KEYS[index] != "source_sha256"
        ):
            raise ValueError("reference training inputs differ")
        ids = receipt.get("validation_image_ids")
        labels = receipt.get("validation_labels")
        if (
            not isinstance(ids, list)
            or not isinstance(labels, list)
            or len(ids) != 5851
            or len(labels) != 5851
            or any(type(value) is not int for value in ids)
            or any(type(value) is not int for value in labels)
        ):
            raise ValueError("matched training inventory differs")
        inventory = (
            receipt.get("seed"),
            receipt.get("embedding_width", 128),
            receipt.get("schedule_sha256"),
            receipt.get("fit_row_indexes_sha256"),
            receipt.get("validation_row_indexes_sha256"),
            tuple(ids),
            tuple(labels),
            source_inputs,
        )
        if baseline_inventory is None:
            baseline_inventory = inventory
        elif inventory != baseline_inventory:
            raise ValueError("matched training inventory differs")
        try:
            packed = receipt["validation"]["packed"]
            map_at_r = packed["map_at_r"]
            recall = packed["recall_at_1"]
        except (KeyError, TypeError):
            raise ValueError("reference holdout metric differs") from None
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (map_at_r, recall)
        ):
            raise ValueError("reference holdout metric differs")
        steps.add(step)
        normalized.append(
            ReferenceSelection(checkpoint, receipt, step, float(map_at_r), float(recall))
        )
    if steps != set(REFERENCE_STEPS):
        raise ValueError("complete reference run requires all recorded checkpoints")
    final_checkpoint = next(
        item.checkpoint for item in normalized if item.step == REFERENCE_STEPS[-1]
    )
    for item in normalized:
        expected = (
            final_checkpoint
            if item.step == REFERENCE_STEPS[-1]
            else final_checkpoint.with_name(
                f"{final_checkpoint.stem}.step{item.step}{final_checkpoint.suffix}"
            )
        )
        if item.checkpoint.resolve() != expected.resolve():
            raise ValueError("reference run checkpoint path differs")
    return min(
        normalized, key=lambda item: (-item.packed_map_at_r, -item.packed_recall_at_1, item.step)
    )


__all__ = ["REFERENCE_STEPS", "ReferenceSelection", "select_reference_checkpoint"]
