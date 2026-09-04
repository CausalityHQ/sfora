"""Frozen protocol primitives for the UniCOM finish causal panel."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from enum import StrEnum

import numpy as np
import torch

from sfora.unicom_rank_finish import identity_balanced_batches
from sfora.unicom_training import padded_epoch_indices


class FinishArm(StrEnum):
    """Registered phase-one continuation arms."""

    CLASSIFICATION_PADDED = "classification-padded"
    CLASSIFICATION_PK = "classification-pk"
    SMOOTH_AP_PK = "smooth-ap-pk"


_CONFIG_KEYS = {
    "schema",
    "claim_eligible",
    "finish_seed",
    "epochs",
    "steps_per_epoch",
    "batch_size",
    "images_per_identity",
    "arms",
    "partition_sha256",
    "parent_checkpoint_sha256",
    "parent_receipt_sha256",
    "optimization_images",
    "optimization_identities",
    "schedule_sha256",
    "source_commit",
    "official_checkpoint_sha256",
    "unicom_revision",
    "environment_sha256",
}


def validate_finish_config(value: object) -> dict[str, object]:
    """Validate the immutable phase-one scientific shape."""

    expected_arms = [arm.value for arm in FinishArm]
    if (
        type(value) is not dict
        or set(value) != _CONFIG_KEYS
        or value.get("schema") != "unicom-finish-ablation-config-v1"
        or value.get("claim_eligible") is not False
        or value.get("finish_seed") != 3
        or value.get("epochs") != [5, 6, 7, 8]
        or value.get("steps_per_epoch") != 161
        or value.get("batch_size") != 128
        or value.get("images_per_identity") != 4
        or value.get("arms") != expected_arms
        or value.get("optimization_images") != 20_650
        or value.get("optimization_identities") != 3_200
        or any(
            type(value[key]) is not int
            for key in (
                "finish_seed",
                "steps_per_epoch",
                "batch_size",
                "images_per_identity",
                "optimization_images",
                "optimization_identities",
            )
        )
        or any(type(epoch) is not int for epoch in value["epochs"])
        or any(
            type(value.get(key)) is not str
            or len(value[key]) != 64
            or any(character not in "0123456789abcdef" for character in value[key])
            for key in (
                "partition_sha256",
                "parent_checkpoint_sha256",
                "parent_receipt_sha256",
                "official_checkpoint_sha256",
                "environment_sha256",
            )
        )
        or any(
            type(value.get(key)) is not str
            or len(value[key]) != 40
            or any(character not in "0123456789abcdef" for character in value[key])
            for key in ("source_commit", "unicom_revision")
        )
        or type(value.get("schedule_sha256")) is not dict
        or tuple(value["schedule_sha256"]) != tuple(expected_arms)
        or any(
            type(value["schedule_sha256"][arm]) is not str
            or len(value["schedule_sha256"][arm]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value["schedule_sha256"][arm]
            )
            for arm in expected_arms
        )
        or value["schedule_sha256"][FinishArm.CLASSIFICATION_PK.value]
        != value["schedule_sha256"][FinishArm.SMOOTH_AP_PK.value]
    ):
        raise ValueError("finish ablation config differs")
    return dict(value)


def build_finish_batches(
    labels: Sequence[str],
    *,
    arm: FinishArm,
    seed: int,
    epoch: int,
    steps: int,
) -> tuple[tuple[int, ...], ...]:
    """Build one registered epoch schedule for a causal arm."""

    if type(arm) is not FinishArm or seed != 3 or epoch not in {5, 6, 7, 8} or steps <= 0:
        raise ValueError("finish ablation schedule authority differs")
    if arm is FinishArm.CLASSIFICATION_PADDED:
        indices = padded_epoch_indices(
            size=len(labels), global_batch=128, epoch=epoch - 1, seed=seed
        )
        required = steps * 128
        if len(indices) < required:
            raise ValueError("finish ablation padded schedule differs")
        return tuple(tuple(indices[offset : offset + 128]) for offset in range(0, required, 128))
    return identity_balanced_batches(
        labels,
        batch_size=128,
        images_per_identity=4,
        seed=seed,
        epoch=epoch,
        steps=steps,
    )


def schedule_sha256(batches: Sequence[Sequence[int]]) -> str:
    """Hash an ordered schedule without platform-dependent binary packing."""

    if not batches or any(not batch for batch in batches):
        raise ValueError("finish ablation schedule differs")
    payload = (json.dumps(batches, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def build_finish_config(
    labels: Sequence[str],
    *,
    partition_sha256: str,
    parent_checkpoint_sha256: str,
    parent_receipt_sha256: str,
    source_commit: str,
    official_checkpoint_sha256: str,
    unicom_revision: str,
    environment_sha256: str,
) -> dict[str, object]:
    """Build the frozen phase-one config from the authenticated inventory."""

    if (
        type(labels) is not tuple
        or len(labels) != 20_650
        or any(type(label) is not str or not label for label in labels)
        or len(set(labels)) != 3_200
    ):
        raise ValueError("finish ablation optimization inventory differs")
    digests = {}
    for arm in FinishArm:
        batches = tuple(
            batch
            for epoch in (5, 6, 7, 8)
            for batch in build_finish_batches(
                labels, arm=arm, seed=3, epoch=epoch, steps=161
            )
        )
        digests[arm.value] = schedule_sha256(batches)
    value = {
        "schema": "unicom-finish-ablation-config-v1",
        "claim_eligible": False,
        "finish_seed": 3,
        "epochs": [5, 6, 7, 8],
        "steps_per_epoch": 161,
        "batch_size": 128,
        "images_per_identity": 4,
        "arms": [arm.value for arm in FinishArm],
        "partition_sha256": partition_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_receipt_sha256": parent_receipt_sha256,
        "source_commit": source_commit,
        "official_checkpoint_sha256": official_checkpoint_sha256,
        "unicom_revision": unicom_revision,
        "environment_sha256": environment_sha256,
        "optimization_images": len(labels),
        "optimization_identities": len(set(labels)),
        "schedule_sha256": digests,
    }
    return validate_finish_config(value)


def capture_rng_state() -> dict[str, object]:
    """Capture global stochastic streams around evaluation."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else (),
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore a state returned by :func:`capture_rng_state`."""

    if type(state) is not dict or set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("finish ablation RNG state differs")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda = state["cuda"]
    if cuda:
        if not torch.cuda.is_available():
            raise ValueError("finish ablation CUDA RNG state differs")
        torch.cuda.set_rng_state_all(list(cuda))


__all__ = [
    "FinishArm",
    "build_finish_batches",
    "build_finish_config",
    "capture_rng_state",
    "restore_rng_state",
    "schedule_sha256",
    "validate_finish_config",
]
