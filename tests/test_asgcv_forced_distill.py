from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from sfora.asgcv_forced_distill import (
    ASGCV_FORCED_DISTILL_SHAPE,
    ForcedDistillCapture,
    build_forced_distill_schedule,
    canonical_forced_distill_capture_bytes,
    relation_correct_gradient,
    validate_forced_distill_capture_bytes,
)

SOURCE_COMMIT = "12" * 20
LAUNCH_SHA256 = "34" * 32


def _manifest(classes: int) -> tuple[tuple[str, ...], tuple[int, ...]]:
    ids = tuple(
        f"class-{label:03d}/image-{image}.jpg" for label in range(classes) for image in range(4)
    )
    labels = tuple(label for label in range(classes) for _ in range(4))
    return ids, labels


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    patches = np.ones(ASGCV_FORCED_DISTILL_SHAPE, dtype=np.float32)
    gradient = np.full(ASGCV_FORCED_DISTILL_SHAPE, 2.0, dtype=np.float32)
    return patches, gradient


def _capture() -> ForcedDistillCapture:
    patches, gradient = _arrays()
    return ForcedDistillCapture(
        source_commit=SOURCE_COMMIT,
        launch_authority_sha256=LAUNCH_SHA256,
        schedule_sha256="56" * 32,
        role="train",
        pair_ordinal=0,
        pair_indices=(0, 1),
        relation_sign=1,
        patch_sha256=hashlib.sha256(patches.tobytes()).hexdigest(),
        gradient_sha256=hashlib.sha256(gradient.tobytes()).hexdigest(),
        array_shape=ASGCV_FORCED_DISTILL_SHAPE,
    )


def test_forced_distill_schedules_are_balanced_role_separated_and_deterministic() -> None:
    train_ids, train_labels = _manifest(64)
    validation_ids, validation_labels = _manifest(17)
    train = build_forced_distill_schedule(
        train_ids,
        train_labels,
        source_commit=SOURCE_COMMIT,
        launch_authority_sha256=LAUNCH_SHA256,
        role="train",
    )
    repeated = build_forced_distill_schedule(
        train_ids,
        train_labels,
        source_commit=SOURCE_COMMIT,
        launch_authority_sha256=LAUNCH_SHA256,
        role="train",
    )
    validation = build_forced_distill_schedule(
        validation_ids,
        validation_labels,
        source_commit=SOURCE_COMMIT,
        launch_authority_sha256=LAUNCH_SHA256,
        role="validation",
    )
    assert train == repeated
    assert train.pair_count == 128
    assert validation.pair_count == 32
    assert train.schedule_seed_sha256 != validation.schedule_seed_sha256
    assert sum(pair.relation_sign == 1 for pair in train.pairs) == 64
    assert sum(pair.relation_sign == -1 for pair in train.pairs) == 64
    assert (
        len({index for pair in train.pairs for index in (pair.left_index, pair.right_index)}) == 256
    )


def test_forced_distill_capture_binds_exact_arrays_and_canonical_bytes() -> None:
    patches, gradient = _arrays()
    capture = _capture()
    raw = canonical_forced_distill_capture_bytes(capture)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert json.loads(raw)["official_test_access"] is False
    assert (
        validate_forced_distill_capture_bytes(
            raw,
            patch_tokens=patches,
            exact_gradient=gradient,
        )
        == capture
    )
    changed = gradient.copy()
    changed[0, 0, 0] = 3.0
    with pytest.raises(ValueError, match="gradient digest"):
        validate_forced_distill_capture_bytes(
            raw,
            patch_tokens=patches,
            exact_gradient=changed,
        )
    value = json.loads(raw)
    value["relation_sign"] = True
    mutated = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(ValueError):
        validate_forced_distill_capture_bytes(
            mutated,
            patch_tokens=patches,
            exact_gradient=gradient,
        )


def test_relation_correct_gradient_is_exact_sign_orientation() -> None:
    _, gradient = _arrays()
    positive = relation_correct_gradient(gradient, 1)
    negative = relation_correct_gradient(gradient, -1)
    assert positive.dtype == np.float32 and positive.flags.c_contiguous
    assert np.array_equal(positive, gradient)
    assert np.array_equal(negative, -gradient)
    assert not np.shares_memory(positive, gradient)
    with pytest.raises(ValueError):
        relation_correct_gradient(gradient, True)
