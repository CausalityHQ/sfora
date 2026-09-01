#!/usr/bin/env python3
"""Offline, phase-separated ASG-CV E0 capture orchestration."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from sfora.asgcv import validate_gradient_sample_bundle, validate_gradient_sample_inputs
from sfora.asgcv_protocol import (
    AsgcvCompletionGroup,
    AsgcvCompletionProtocol,
    AsgcvEligibleSchedule,
    AsgcvPairSchedule,
    AsgcvRolloutAuthority,
    assemble_asgcv_eligible_schedule,
    classify_asgcv_completion_group,
    derive_asgcv_rollout_seeds,
    validate_asgcv_protocol_bundle,
)

ASGCV_CAPTURE_IMAGES = 2
ASGCV_CAPTURE_PATCHES = 49


class EligibilityAdapter(Protocol):
    """Minimal generation-only capability available to the eligibility phase."""

    def prepare_image_pair(
        self,
        images: object,
        prompt_utf8: object,
        attribute_token_span: object,
        patch_tokens_per_image: object,
    ) -> object: ...

    def generate(
        self,
        pair: object,
        seed: int,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[int, ...]: ...


def _capture_paths(directory: Path, ordinal: int) -> tuple[Path, Path, Path]:
    if not isinstance(directory, Path) or not directory.is_dir():
        raise ValueError("ASG-CV capture directory differs")
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("ASG-CV capture ordinal differs")
    return (
        directory / f"sample-{ordinal:06d}.json",
        directory / f"patch-{ordinal:06d}.npy",
        directory / f"gradient-{ordinal:06d}.npy",
    )


def _capture_array(value: object, *, name: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float32)
        or value.ndim != 3
        or value.shape[0] != ASGCV_CAPTURE_IMAGES
        or value.shape[1] != ASGCV_CAPTURE_PATCHES
        or value.shape[2] <= 0
        or not bool(np.isfinite(value).all())
    ):
        raise ValueError(f"ASG-CV capture {name} shape differs")
    return np.ascontiguousarray(value)


def _load_array(path: Path, *, name: str) -> np.ndarray:
    try:
        with path.open("rb") as stream:
            value = np.load(stream, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"ASG-CV capture {name} file differs") from error
    return _capture_array(value, name=name)


def _validate_triple(
    receipt_path: Path,
    patch_path: Path,
    gradient_path: Path,
    *,
    ordinal: int,
) -> tuple[bytes, np.ndarray, np.ndarray]:
    receipt = receipt_path.read_bytes()
    patch = _load_array(patch_path, name="patch-token")
    gradient = _load_array(gradient_path, name="gradient")
    value = validate_gradient_sample_inputs(
        receipt,
        patch_tokens=patch,
        exact_gradient=gradient,
    )
    if value["eligible_pair_ordinal"] != ordinal:
        raise ValueError("ASG-CV capture receipt ordinal differs")
    return receipt, patch, gradient


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def write_capture_triple(
    directory: Path,
    *,
    ordinal: int,
    receipt: bytes,
    patch_tokens: object,
    exact_gradient: object,
) -> str:
    """Atomically publish one authenticated sample, with its receipt as commit marker."""

    receipt_path, patch_path, gradient_path = _capture_paths(directory, ordinal)
    patch = _capture_array(patch_tokens, name="patch-token")
    gradient = _capture_array(exact_gradient, name="gradient")
    if patch.shape != gradient.shape or type(receipt) is not bytes:
        raise ValueError("ASG-CV capture triple shape differs")
    receipt_value = validate_gradient_sample_inputs(
        receipt,
        patch_tokens=patch,
        exact_gradient=gradient,
    )
    if receipt_value["eligible_pair_ordinal"] != ordinal:
        raise ValueError("ASG-CV capture receipt ordinal differs")
    existing = tuple(path.exists() for path in (receipt_path, patch_path, gradient_path))
    if all(existing):
        old_receipt, old_patch, old_gradient = _validate_triple(
            receipt_path,
            patch_path,
            gradient_path,
            ordinal=ordinal,
        )
        if (
            old_receipt != receipt
            or not np.array_equal(old_patch, patch)
            or not np.array_equal(old_gradient, gradient)
        ):
            raise ValueError("ASG-CV capture existing triple differs")
        return "reused"
    if any(existing):
        raise ValueError("ASG-CV capture partial triple differs")

    partials = tuple(
        path.with_name(path.name + ".partial") for path in (receipt_path, patch_path, gradient_path)
    )
    if any(path.exists() for path in partials):
        raise ValueError("ASG-CV capture partial file differs")
    try:
        _write_array(partials[1], patch)
        _write_array(partials[2], gradient)
        _write_bytes(partials[0], receipt)
        os.replace(partials[1], patch_path)
        os.replace(partials[2], gradient_path)
        os.replace(partials[0], receipt_path)
    finally:
        for partial in partials:
            if partial.exists():
                partial.unlink()
    _validate_triple(
        receipt_path,
        patch_path,
        gradient_path,
        ordinal=ordinal,
    )
    return "written"


def validated_capture_prefix(directory: Path, *, expected_count: int) -> int:
    """Reopen the contiguous committed prefix and return its first absent ordinal."""

    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("ASG-CV capture expected count differs")
    if tuple(directory.glob("*.partial")):
        raise ValueError("ASG-CV capture partial file differs")
    for ordinal in range(expected_count):
        paths = _capture_paths(directory, ordinal)
        existing = tuple(path.exists() for path in paths)
        if all(existing):
            _validate_triple(*paths, ordinal=ordinal)
            continue
        if any(existing):
            raise ValueError("ASG-CV capture partial triple differs")
        for later in range(ordinal + 1, expected_count):
            if any(path.exists() for path in _capture_paths(directory, later)):
                raise ValueError("ASG-CV capture ordinal gap differs")
        return ordinal
    return expected_count


def capture_schedule(
    directory: Path,
    *,
    protocol: AsgcvCompletionProtocol,
    rollout_authority: AsgcvRolloutAuthority,
    candidate_schedule: AsgcvPairSchedule,
    completion_groups: tuple[AsgcvCompletionGroup, ...],
    eligible_schedule: AsgcvEligibleSchedule,
    example_ids: tuple[str, ...],
    labels: tuple[int, ...],
    capture_one: Callable[[int, int], tuple[bytes, np.ndarray, np.ndarray]],
) -> int:
    """Capture the first absent eligible row after reopening all sealed context."""

    if not callable(capture_one):
        raise ValueError("ASG-CV capture callback differs")
    validate_asgcv_protocol_bundle(
        protocol,
        rollout_authority,
        candidate_schedule,
        completion_groups,
        eligible_schedule,
        example_ids=example_ids,
        labels=labels,
    )
    expected = eligible_schedule.target_pair_count
    prefix = validated_capture_prefix(directory, expected_count=expected)
    for eligible_ordinal in range(prefix):
        receipt_path, patch_path, gradient_path = _capture_paths(directory, eligible_ordinal)
        receipt, patch, gradient = _validate_triple(
            receipt_path,
            patch_path,
            gradient_path,
            ordinal=eligible_ordinal,
        )
        validate_gradient_sample_bundle(
            receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
            protocol=protocol,
            rollout_authority=rollout_authority,
            eligible_schedule=eligible_schedule,
            candidate_schedule=candidate_schedule,
            completion_groups=completion_groups,
            example_ids=example_ids,
            labels=labels,
        )
    for eligible_ordinal in range(prefix, expected):
        candidate_ordinal = eligible_schedule.candidate_ordinals[eligible_ordinal]
        captured = capture_one(eligible_ordinal, candidate_ordinal)
        if type(captured) is not tuple or len(captured) != 3 or type(captured[0]) is not bytes:
            raise ValueError("ASG-CV capture callback result differs")
        receipt, patch, gradient = captured
        validate_gradient_sample_bundle(
            receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
            protocol=protocol,
            rollout_authority=rollout_authority,
            eligible_schedule=eligible_schedule,
            candidate_schedule=candidate_schedule,
            completion_groups=completion_groups,
            example_ids=example_ids,
            labels=labels,
        )
        write_capture_triple(
            directory,
            ordinal=eligible_ordinal,
            receipt=receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
        )
    return validated_capture_prefix(directory, expected_count=expected)


def build_eligibility_schedule(
    adapter: EligibilityAdapter,
    *,
    images: tuple[np.ndarray, ...],
    prompt_utf8: str,
    attribute_token_span: tuple[int, int],
    patch_tokens_per_image: int,
    protocol: AsgcvCompletionProtocol,
    rollout_authority: AsgcvRolloutAuthority,
    candidate_schedule: AsgcvPairSchedule,
    target_pair_count: int,
    example_ids: tuple[str, ...],
    labels: tuple[int, ...],
) -> tuple[tuple[AsgcvCompletionGroup, ...], AsgcvEligibleSchedule]:
    """Generate and seal all completion groups before any gradient capability exists."""

    if (
        type(images) is not tuple
        or len(images) != len(example_ids)
        or len(labels) != len(example_ids)
        or any(
            type(image) is not np.ndarray
            or image.dtype != np.dtype(np.uint8)
            or image.ndim != 3
            or image.shape[-1] != 3
            or any(size <= 0 for size in image.shape)
            for image in images
        )
    ):
        raise ValueError("ASG-CV eligibility image authority differs")
    groups: list[AsgcvCompletionGroup] = []
    for pair in candidate_schedule.pairs:
        prepared = adapter.prepare_image_pair(
            (images[pair.left_index], images[pair.right_index]),
            prompt_utf8,
            attribute_token_span,
            patch_tokens_per_image,
        )
        completions = tuple(
            adapter.generate(
                prepared,
                seed,
                temperature=rollout_authority.temperature,
                top_p=rollout_authority.top_p,
                max_new_tokens=rollout_authority.max_new_tokens,
            )
            for seed in derive_asgcv_rollout_seeds(
                rollout_authority,
                candidate_pair_ordinal=pair.ordinal,
            )
        )
        groups.append(
            classify_asgcv_completion_group(
                completions,
                pair.relation_sign,
                protocol,
                rollout_authority=rollout_authority,
                candidate_pair_ordinal=pair.ordinal,
            )
        )
    sealed_groups = tuple(groups)
    eligible = assemble_asgcv_eligible_schedule(
        candidate_schedule,
        sealed_groups,
        target_pair_count=target_pair_count,
    )
    validate_asgcv_protocol_bundle(
        protocol,
        rollout_authority,
        candidate_schedule,
        sealed_groups,
        eligible,
        example_ids=example_ids,
        labels=labels,
    )
    return sealed_groups, eligible
