from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from sfora.unicom_finish_protocol import (
    FinishArm,
    build_finish_batches,
    build_finish_config,
    capture_rng_state,
    restore_rng_state,
    schedule_sha256,
    validate_finish_config,
)
from sfora.unicom_training import padded_epoch_indices


def _labels() -> tuple[str, ...]:
    return tuple(f"identity-{index // 4:04d}" for index in range(128))


def test_finish_config_freezes_phase_one_inventory() -> None:
    labels = tuple(f"identity-{index % 3_200:04d}" for index in range(20_650))
    value = build_finish_config(
        labels,
        partition_sha256="a" * 64,
        parent_checkpoint_sha256="b" * 64,
        parent_receipt_sha256="c" * 64,
        source_commit="d" * 40,
        official_checkpoint_sha256="e" * 64,
        unicom_revision="f" * 40,
        environment_sha256="1" * 64,
    )

    assert value["arms"] == [
        "classification-padded",
        "classification-pk",
        "smooth-ap-pk",
    ]
    assert value["optimization_images"] == 20_650
    assert value["optimization_identities"] == 3_200
    assert value["schedule_sha256"]["classification-pk"] == value[
        "schedule_sha256"
    ]["smooth-ap-pk"]
    validate_finish_config(value)
    with pytest.raises(ValueError):
        validate_finish_config({**value, "finish_seed": 4})


def test_original_arm_matches_parent_padded_schedule() -> None:
    labels = _labels()
    batches = build_finish_batches(
        labels, arm=FinishArm.CLASSIFICATION_PADDED, seed=3, epoch=5, steps=1
    )

    assert batches == (padded_epoch_indices(size=128, global_batch=128, epoch=4, seed=3),)


def test_pk_arms_have_identical_schedule_and_digest() -> None:
    labels = _labels()
    classification = build_finish_batches(
        labels, arm=FinishArm.CLASSIFICATION_PK, seed=3, epoch=5, steps=2
    )
    smooth_ap = build_finish_batches(labels, arm=FinishArm.SMOOTH_AP_PK, seed=3, epoch=5, steps=2)

    assert classification == smooth_ap
    assert schedule_sha256(classification) == schedule_sha256(smooth_ap)
    assert len(classification) == 2
    assert all(len(batch) == 128 for batch in classification)


def test_rng_capture_restore_replays_python_numpy_and_torch() -> None:
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(4))

    restore_rng_state(state)
    observed = (random.random(), np.random.random(), torch.rand(4))

    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])
