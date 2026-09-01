#!/usr/bin/env python3
"""Offline, phase-separated ASG-CV E0 capture orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from sfora.asgcv import validate_gradient_sample_bundle, validate_gradient_sample_inputs
from sfora.asgcv_marginal import (
    AsgcvVisionCutAuthority,
    validate_marginal_gradient_sample_bytes,
    validate_marginal_gradient_sample_context,
    validate_marginal_gradient_sample_inputs,
)
from sfora.asgcv_pilot import (
    ASGCV_P32_PAIR_COUNT,
    AsgcvP32Candidate,
    canonical_asgcv_p32_candidate_bytes,
    canonical_asgcv_p32_result_bytes,
    validate_asgcv_p32_candidate_context,
    validate_asgcv_p32_pilot_schedule,
    validate_asgcv_p32_result_bundle,
)
from sfora.asgcv_protocol import (
    AsgcvCompletionGroup,
    AsgcvCompletionProtocol,
    AsgcvEligibleSchedule,
    AsgcvMarginalSchedule,
    AsgcvPairSchedule,
    AsgcvPartitionAuthority,
    AsgcvRolloutAuthority,
    assemble_asgcv_eligible_schedule,
    assemble_asgcv_marginal_schedule,
    classify_asgcv_completion_group,
    derive_asgcv_rollout_seeds,
    validate_asgcv_marginal_protocol_bundle,
    validate_asgcv_protocol_bundle,
)

ASGCV_CAPTURE_IMAGES = 2
ASGCV_CAPTURE_PATCHES = 49
ASGCV_P32_EXACT_DIAGNOSTIC_COUNT = 4
ASGCV_P32_ROW_SCHEMA = "sfora-asgcv-p32-row-v1"
ASGCV_P32_FAILURE_SCHEMA = "sfora-asgcv-p32-failure-v1"


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


def _marginal_capture_array(
    value: object, *, name: str, cut: AsgcvVisionCutAuthority
) -> np.ndarray:
    expected_shape = (
        cut.images,
        len(cut.boundary_names) * cut.patches_per_boundary,
        cut.channel_dimensions,
    )
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float32)
        or value.shape != expected_shape
        or not bool(np.isfinite(value).all())
    ):
        raise ValueError(f"ASG-CV marginal complete-cut {name} shape differs")
    return np.ascontiguousarray(value)


def _load_marginal_array(path: Path, *, name: str, cut: AsgcvVisionCutAuthority) -> np.ndarray:
    try:
        with path.open("rb") as stream:
            value = np.load(stream, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"ASG-CV marginal capture {name} file differs") from error
    return _marginal_capture_array(value, name=name, cut=cut)


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


def _validate_marginal_triple(
    receipt_path: Path,
    patch_path: Path,
    gradient_path: Path,
    *,
    ordinal: int,
) -> tuple[bytes, np.ndarray, np.ndarray]:
    receipt = receipt_path.read_bytes()
    value = validate_marginal_gradient_sample_bytes(receipt)
    cut = AsgcvVisionCutAuthority.from_mapping(value["vision_cut_authority"])
    patch = _load_marginal_array(patch_path, name="patch-token", cut=cut)
    gradient = _load_marginal_array(gradient_path, name="gradient", cut=cut)
    validate_marginal_gradient_sample_inputs(
        receipt,
        patch_tokens=patch,
        exact_gradient=gradient,
    )
    if value["candidate_pair_ordinal"] != ordinal:
        raise ValueError("ASG-CV marginal capture receipt ordinal differs")
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


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes or path.exists():
        raise ValueError("ASG-CV P32 output differs")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise ValueError("ASG-CV P32 partial output differs")
    try:
        _write_bytes(partial, payload)
        os.replace(partial, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if partial.exists():
            partial.unlink()


def _p32_path(directory: Path, ordinal: int) -> Path:
    if (
        not isinstance(directory, Path)
        or not directory.is_dir()
        or type(ordinal) is not int
        or not 0 <= ordinal < ASGCV_P32_PAIR_COUNT
    ):
        raise ValueError("ASG-CV P32 campaign path differs")
    return directory / f"candidate-{ordinal:06d}.json"


def _canonical_p32_row_bytes(
    group: AsgcvCompletionGroup,
    candidate: AsgcvP32Candidate,
) -> bytes:
    group.validated()
    candidate.validated()
    return (
        json.dumps(
            {
                "candidate": candidate.to_mapping(),
                "completion_group": group.to_mapping(),
                "schema": ASGCV_P32_ROW_SCHEMA,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _validate_p32_row_bytes(
    raw: bytes,
) -> tuple[AsgcvCompletionGroup, AsgcvP32Candidate, bytes]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV P32 row is not canonical JSON") from error
    if (
        type(value) is not dict
        or set(value)
        != {
            "candidate",
            "completion_group",
            "schema",
        }
        or value["schema"] != ASGCV_P32_ROW_SCHEMA
    ):
        raise ValueError("ASG-CV P32 row schema differs")
    group = AsgcvCompletionGroup.from_mapping(value["completion_group"])
    candidate = AsgcvP32Candidate.from_mapping(value["candidate"])
    if _canonical_p32_row_bytes(group, candidate) != raw:
        raise ValueError("ASG-CV P32 row bytes differ")
    return group, candidate, canonical_asgcv_p32_candidate_bytes(candidate)


def _canonical_p32_failure_bytes(
    *,
    source_commit: str,
    model_revision: str,
    launch_authority_sha256: str,
    predictor_initialization_seed_sha256: str,
    partition_authority_sha256: str,
    pilot_schedule_sha256: str,
    rollout_authority_sha256: str,
    failed_candidate_ordinal: int,
    completed_candidate_sha256s: tuple[str, ...],
    failure_kind: str,
    failure_phase: str,
    error: Exception,
) -> bytes:
    if (
        type(failed_candidate_ordinal) is not int
        or not 0 <= failed_candidate_ordinal <= ASGCV_P32_PAIR_COUNT
        or failed_candidate_ordinal != len(completed_candidate_sha256s)
        or failure_kind not in {"memory-error", "authority-error", "backend-error"}
        or failure_phase not in {"candidate-execution", "result-assembly"}
        or not isinstance(error, Exception)
    ):
        raise ValueError("ASG-CV P32 failure evidence differs")
    error_identity = f"{type(error).__module__}.{type(error).__qualname__}:{error}".encode()
    value: dict[str, object] = {
        "schema": ASGCV_P32_FAILURE_SCHEMA,
        "claim_eligible": False,
        "official_test_access": False,
        "source_commit": source_commit,
        "model_revision": model_revision,
        "launch_authority_sha256": launch_authority_sha256,
        "predictor_initialization_seed_sha256": predictor_initialization_seed_sha256,
        "partition_authority_sha256": partition_authority_sha256,
        "pilot_schedule_sha256": pilot_schedule_sha256,
        "rollout_authority_sha256": rollout_authority_sha256,
        "failed_candidate_ordinal": failed_candidate_ordinal,
        "completed_candidate_sha256s": list(completed_candidate_sha256s),
        "failure_phase": failure_phase,
        "failure_kind": failure_kind,
        "error_sha256": hashlib.sha256(error_identity).hexdigest(),
    }
    value["failure_sha256"] = hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    ).hexdigest()
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _validate_p32_failure_bytes(raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV P32 failure is not canonical JSON") from error
    expected = {
        "schema",
        "claim_eligible",
        "official_test_access",
        "source_commit",
        "model_revision",
        "launch_authority_sha256",
        "predictor_initialization_seed_sha256",
        "partition_authority_sha256",
        "pilot_schedule_sha256",
        "rollout_authority_sha256",
        "failed_candidate_ordinal",
        "completed_candidate_sha256s",
        "failure_phase",
        "failure_kind",
        "error_sha256",
        "failure_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected
        or value["schema"] != ASGCV_P32_FAILURE_SCHEMA
        or value["claim_eligible"] is not False
        or value["official_test_access"] is not False
        or value["failure_phase"] not in {"candidate-execution", "result-assembly"}
        or value["failure_kind"] not in {"memory-error", "authority-error", "backend-error"}
        or type(value["failed_candidate_ordinal"]) is not int
        or not 0 <= value["failed_candidate_ordinal"] <= ASGCV_P32_PAIR_COUNT
        or type(value["completed_candidate_sha256s"]) is not list
        or len(value["completed_candidate_sha256s"]) != value["failed_candidate_ordinal"]
        or (value["failure_phase"] == "result-assembly")
        != (value["failed_candidate_ordinal"] == ASGCV_P32_PAIR_COUNT)
    ):
        raise ValueError("ASG-CV P32 failure schema differs")
    for name, length in (("source_commit", 40), ("model_revision", 40)):
        field = value[name]
        if (
            type(field) is not str
            or len(field) != length
            or any(c not in "0123456789abcdef" for c in field)
        ):
            raise ValueError("ASG-CV P32 failure identity differs")
    for name in (
        "partition_authority_sha256",
        "launch_authority_sha256",
        "predictor_initialization_seed_sha256",
        "pilot_schedule_sha256",
        "rollout_authority_sha256",
        "error_sha256",
        "failure_sha256",
    ):
        field = value[name]
        if (
            type(field) is not str
            or len(field) != 64
            or any(c not in "0123456789abcdef" for c in field)
        ):
            raise ValueError("ASG-CV P32 failure identity differs")
    completed = value["completed_candidate_sha256s"]
    if any(
        type(item) is not str or len(item) != 64 or any(c not in "0123456789abcdef" for c in item)
        for item in completed
    ):
        raise ValueError("ASG-CV P32 failure prefix differs")
    claimed = value["failure_sha256"]
    identity = dict(value)
    del identity["failure_sha256"]
    expected_digest = hashlib.sha256(
        (
            json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    ).hexdigest()
    if (
        claimed != expected_digest
        or raw
        != (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    ):
        raise ValueError("ASG-CV P32 failure bytes differ")
    return value


def _validated_p32_prefix(
    directory: Path,
    *,
    rollout_authority: AsgcvRolloutAuthority,
    pilot_schedule: AsgcvPairSchedule,
    partition_authority: AsgcvPartitionAuthority,
    source_commit: str,
    predictor_train: object,
    e0_validation: object,
    e1_optimization: object,
) -> tuple[
    tuple[AsgcvCompletionGroup, ...],
    tuple[AsgcvP32Candidate, ...],
    tuple[bytes, ...],
]:
    if (
        not isinstance(directory, Path)
        or not directory.is_dir()
        or type(rollout_authority) is not AsgcvRolloutAuthority
        or type(pilot_schedule) is not AsgcvPairSchedule
        or type(partition_authority) is not AsgcvPartitionAuthority
    ):
        raise ValueError("ASG-CV P32 campaign authority differs")
    rollout_authority.validated()
    validate_asgcv_p32_pilot_schedule(
        pilot_schedule,
        partition_authority=partition_authority,
        source_commit=source_commit,
        predictor_train=predictor_train,
        e0_validation=e0_validation,
        e1_optimization=e1_optimization,
    )
    if pilot_schedule.pair_count != ASGCV_P32_PAIR_COUNT:
        raise ValueError("ASG-CV P32 campaign schedule differs")
    if tuple(directory.glob("*.partial")):
        raise ValueError("ASG-CV P32 partial output differs")
    if tuple(directory.glob("group-*.json")):
        raise ValueError("ASG-CV P32 partial campaign differs")

    groups: list[AsgcvCompletionGroup] = []
    candidates: list[AsgcvP32Candidate] = []
    candidate_receipts: list[bytes] = []
    missing = False
    exact_count = 0
    for ordinal in range(ASGCV_P32_PAIR_COUNT):
        candidate_path = _p32_path(directory, ordinal)
        if not candidate_path.exists():
            missing = True
            continue
        if missing:
            raise ValueError("ASG-CV P32 partial campaign differs")
        group, candidate, candidate_raw = _validate_p32_row_bytes(candidate_path.read_bytes())
        validate_asgcv_p32_candidate_context(
            candidate,
            completion_group=group,
            rollout_authority=rollout_authority,
            pilot_schedule=pilot_schedule,
        )
        if (
            candidate.partition_authority_sha256 != partition_authority.sha256()
            or candidate.source_commit != source_commit
        ):
            raise ValueError("ASG-CV P32 partition context differs")
        expected_exact = (
            group.nonzero_reward_variance and exact_count < ASGCV_P32_EXACT_DIAGNOSTIC_COUNT
        )
        if candidate.exact_diagnostic is not expected_exact:
            raise ValueError("ASG-CV P32 exact diagnostic schedule differs")
        exact_count += int(candidate.exact_diagnostic)
        groups.append(group)
        candidates.append(candidate)
        candidate_receipts.append(candidate_raw)
    if (directory / "result.json").exists() and len(candidates) != ASGCV_P32_PAIR_COUNT:
        raise ValueError("ASG-CV P32 partial result differs")
    return tuple(groups), tuple(candidates), tuple(candidate_receipts)


def run_p32_campaign(
    directory: Path,
    *,
    rollout_authority: AsgcvRolloutAuthority,
    pilot_schedule: AsgcvPairSchedule,
    partition_authority: AsgcvPartitionAuthority,
    source_commit: str,
    predictor_train: object,
    e0_validation: object,
    e1_optimization: object,
    execute_one: Callable[[int, bool], tuple[AsgcvCompletionGroup, AsgcvP32Candidate]],
) -> bytes:
    """Run or resume exactly one candidate-ordered P32 campaign."""

    if not callable(execute_one):
        raise ValueError("ASG-CV P32 executor differs")
    groups, candidates, candidate_receipts = _validated_p32_prefix(
        directory,
        rollout_authority=rollout_authority,
        pilot_schedule=pilot_schedule,
        partition_authority=partition_authority,
        source_commit=source_commit,
        predictor_train=predictor_train,
        e0_validation=e0_validation,
        e1_optimization=e1_optimization,
    )
    group_rows = list(groups)
    candidate_rows = list(candidates)
    receipt_rows = list(candidate_receipts)
    exact_count = sum(candidate.exact_diagnostic for candidate in candidate_rows)
    for ordinal in range(len(candidate_rows), ASGCV_P32_PAIR_COUNT):
        diagnostic_available = exact_count < ASGCV_P32_EXACT_DIAGNOSTIC_COUNT
        executed = execute_one(ordinal, diagnostic_available)
        if (
            type(executed) is not tuple
            or len(executed) != 2
            or type(executed[0]) is not AsgcvCompletionGroup
            or type(executed[1]) is not AsgcvP32Candidate
        ):
            raise ValueError("ASG-CV P32 executor result differs")
        group, candidate = executed
        validate_asgcv_p32_candidate_context(
            candidate,
            completion_group=group,
            rollout_authority=rollout_authority,
            pilot_schedule=pilot_schedule,
        )
        if (
            candidate.partition_authority_sha256 != partition_authority.sha256()
            or candidate.source_commit != source_commit
        ):
            raise ValueError("ASG-CV P32 partition context differs")
        expected_exact = group.nonzero_reward_variance and diagnostic_available
        if candidate.exact_diagnostic is not expected_exact:
            raise ValueError("ASG-CV P32 exact diagnostic schedule differs")
        candidate_raw = canonical_asgcv_p32_candidate_bytes(candidate)
        candidate_path = _p32_path(directory, ordinal)
        _write_atomic_bytes(candidate_path, _canonical_p32_row_bytes(group, candidate))
        group_rows.append(group)
        candidate_rows.append(candidate)
        receipt_rows.append(candidate_raw)
        exact_count += int(candidate.exact_diagnostic)

    result_raw = canonical_asgcv_p32_result_bytes(tuple(candidate_rows))
    result_path = directory / "result.json"
    if result_path.exists():
        if result_path.read_bytes() != result_raw:
            raise ValueError("ASG-CV P32 existing result differs")
    else:
        _write_atomic_bytes(result_path, result_raw)
    validate_asgcv_p32_result_bundle(
        result_raw,
        candidate_receipts=tuple(receipt_rows),
        completion_groups=tuple(group_rows),
        rollout_authority=rollout_authority,
        pilot_schedule=pilot_schedule,
    )
    return result_raw


def run_p32_campaign_with_failure_terminal(
    directory: Path,
    *,
    rollout_authority: AsgcvRolloutAuthority,
    pilot_schedule: AsgcvPairSchedule,
    partition_authority: AsgcvPartitionAuthority,
    source_commit: str,
    launch_authority_sha256: str,
    predictor_initialization_seed_sha256: str,
    predictor_train: object,
    e0_validation: object,
    e1_optimization: object,
    execute_one: Callable[[int, bool], tuple[AsgcvCompletionGroup, AsgcvP32Candidate]],
) -> bytes:
    """Run P32 once and atomically preserve a canonical execution failure."""

    failure_path = directory / "failure.json"
    if failure_path.exists():
        failure = _validate_p32_failure_bytes(failure_path.read_bytes())
        _, candidates, _ = _validated_p32_prefix(
            directory,
            rollout_authority=rollout_authority,
            pilot_schedule=pilot_schedule,
            partition_authority=partition_authority,
            source_commit=source_commit,
            predictor_train=predictor_train,
            e0_validation=e0_validation,
            e1_optimization=e1_optimization,
        )
        if (
            failure["source_commit"] != source_commit
            or failure["model_revision"] != rollout_authority.model_revision
            or failure["launch_authority_sha256"] != launch_authority_sha256
            or failure["predictor_initialization_seed_sha256"]
            != predictor_initialization_seed_sha256
            or failure["partition_authority_sha256"] != partition_authority.sha256()
            or failure["pilot_schedule_sha256"] != pilot_schedule.sha256()
            or failure["rollout_authority_sha256"] != rollout_authority.sha256()
            or failure["completed_candidate_sha256s"]
            != [candidate.sha256() for candidate in candidates]
        ):
            raise ValueError("ASG-CV P32 failure terminal context differs")
        raise ValueError("ASG-CV P32 failure terminal already exists")
    try:
        return run_p32_campaign(
            directory,
            rollout_authority=rollout_authority,
            pilot_schedule=pilot_schedule,
            partition_authority=partition_authority,
            source_commit=source_commit,
            predictor_train=predictor_train,
            e0_validation=e0_validation,
            e1_optimization=e1_optimization,
            execute_one=execute_one,
        )
    except Exception as error:
        if (
            type(rollout_authority) is not AsgcvRolloutAuthority
            or type(pilot_schedule) is not AsgcvPairSchedule
            or type(partition_authority) is not AsgcvPartitionAuthority
            or type(source_commit) is not str
        ):
            raise
        _, candidates, _ = _validated_p32_prefix(
            directory,
            rollout_authority=rollout_authority,
            pilot_schedule=pilot_schedule,
            partition_authority=partition_authority,
            source_commit=source_commit,
            predictor_train=predictor_train,
            e0_validation=e0_validation,
            e1_optimization=e1_optimization,
        )
        kind = (
            "memory-error"
            if isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError"
            else "authority-error"
            if isinstance(error, ValueError)
            else "backend-error"
        )
        raw = _canonical_p32_failure_bytes(
            source_commit=source_commit,
            model_revision=rollout_authority.model_revision,
            launch_authority_sha256=launch_authority_sha256,
            predictor_initialization_seed_sha256=predictor_initialization_seed_sha256,
            partition_authority_sha256=partition_authority.sha256(),
            pilot_schedule_sha256=pilot_schedule.sha256(),
            rollout_authority_sha256=rollout_authority.sha256(),
            failed_candidate_ordinal=len(candidates),
            completed_candidate_sha256s=tuple(candidate.sha256() for candidate in candidates),
            failure_kind=kind,
            failure_phase=(
                "result-assembly"
                if len(candidates) == ASGCV_P32_PAIR_COUNT
                else "candidate-execution"
            ),
            error=error,
        )
        _write_atomic_bytes(failure_path, raw)
        return raw


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


def write_marginal_capture_triple(
    directory: Path,
    *,
    ordinal: int,
    receipt: bytes,
    patch_tokens: object,
    exact_gradient: object,
) -> str:
    """Publish one complete-cut candidate-marginal sample atomically."""

    receipt_path, patch_path, gradient_path = _capture_paths(directory, ordinal)
    if type(receipt) is not bytes:
        raise ValueError("ASG-CV marginal capture receipt differs")
    receipt_value = validate_marginal_gradient_sample_bytes(receipt)
    cut = AsgcvVisionCutAuthority.from_mapping(receipt_value["vision_cut_authority"])
    patch = _marginal_capture_array(patch_tokens, name="patch-token", cut=cut)
    gradient = _marginal_capture_array(exact_gradient, name="gradient", cut=cut)
    validate_marginal_gradient_sample_inputs(
        receipt,
        patch_tokens=patch,
        exact_gradient=gradient,
    )
    if receipt_value["candidate_pair_ordinal"] != ordinal:
        raise ValueError("ASG-CV marginal capture receipt ordinal differs")
    existing = tuple(path.exists() for path in (receipt_path, patch_path, gradient_path))
    if all(existing):
        old_receipt, old_patch, old_gradient = _validate_marginal_triple(
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
            raise ValueError("ASG-CV marginal capture existing triple differs")
        return "reused"
    if any(existing):
        raise ValueError("ASG-CV marginal capture partial triple differs")

    partials = tuple(
        path.with_name(path.name + ".partial") for path in (receipt_path, patch_path, gradient_path)
    )
    if any(path.exists() for path in partials):
        raise ValueError("ASG-CV marginal capture partial file differs")
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
    _validate_marginal_triple(
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


def validated_marginal_capture_prefix(directory: Path, *, expected_count: int) -> int:
    """Reopen a contiguous candidate-marginal prefix without legacy shape assumptions."""

    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("ASG-CV marginal capture expected count differs")
    if tuple(directory.glob("*.partial")):
        raise ValueError("ASG-CV marginal capture partial file differs")
    for ordinal in range(expected_count):
        paths = _capture_paths(directory, ordinal)
        existing = tuple(path.exists() for path in paths)
        if all(existing):
            _validate_marginal_triple(*paths, ordinal=ordinal)
            continue
        if any(existing):
            raise ValueError("ASG-CV marginal capture partial triple differs")
        for later in range(ordinal + 1, expected_count):
            if any(path.exists() for path in _capture_paths(directory, later)):
                raise ValueError("ASG-CV marginal capture ordinal gap differs")
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


def capture_marginal_schedule(
    directory: Path,
    *,
    protocol: AsgcvCompletionProtocol,
    rollout_authority: AsgcvRolloutAuthority,
    candidate_schedule: AsgcvPairSchedule,
    completion_groups: tuple[AsgcvCompletionGroup, ...],
    marginal_schedule: AsgcvMarginalSchedule,
    example_ids: tuple[str, ...],
    labels: tuple[int, ...],
    capture_one: Callable[[int, bool], tuple[bytes, np.ndarray, np.ndarray]],
) -> int:
    """Capture every candidate marginal, including exact zero-semantic rows."""

    if not callable(capture_one):
        raise ValueError("ASG-CV marginal capture callback differs")
    validate_asgcv_marginal_protocol_bundle(
        protocol,
        rollout_authority,
        candidate_schedule,
        completion_groups,
        marginal_schedule,
        example_ids=example_ids,
        labels=labels,
    )
    expected = marginal_schedule.target_pair_count
    prefix = validated_marginal_capture_prefix(directory, expected_count=expected)
    for candidate_ordinal in range(prefix):
        paths = _capture_paths(directory, candidate_ordinal)
        receipt, patch, gradient = _validate_marginal_triple(
            *paths,
            ordinal=candidate_ordinal,
        )
        validate_marginal_gradient_sample_inputs(
            receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
        )
        validate_marginal_gradient_sample_context(
            receipt,
            marginal_schedule=marginal_schedule,
            candidate_schedule=candidate_schedule,
            completion_groups=completion_groups,
        )
    for candidate_ordinal in range(prefix, expected):
        zero = marginal_schedule.zero_target_flags[candidate_ordinal]
        captured = capture_one(candidate_ordinal, zero)
        if type(captured) is not tuple or len(captured) != 3 or type(captured[0]) is not bytes:
            raise ValueError("ASG-CV marginal capture callback result differs")
        receipt, patch, gradient = captured
        validate_marginal_gradient_sample_inputs(
            receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
        )
        validate_marginal_gradient_sample_context(
            receipt,
            marginal_schedule=marginal_schedule,
            candidate_schedule=candidate_schedule,
            completion_groups=completion_groups,
        )
        write_marginal_capture_triple(
            directory,
            ordinal=candidate_ordinal,
            receipt=receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
        )
    return validated_marginal_capture_prefix(directory, expected_count=expected)


def _build_completion_groups(
    adapter: EligibilityAdapter,
    *,
    images: tuple[np.ndarray, ...],
    prompt_utf8: str,
    attribute_token_span: tuple[int, int],
    patch_tokens_per_image: int,
    protocol: AsgcvCompletionProtocol,
    rollout_authority: AsgcvRolloutAuthority,
    candidate_schedule: AsgcvPairSchedule,
    example_ids: tuple[str, ...],
    labels: tuple[int, ...],
) -> tuple[AsgcvCompletionGroup, ...]:
    """Generate all source-seeded completion groups without gradient capability."""

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
    return sealed_groups


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
    """Build the historical eligible-only refill schedule."""

    sealed_groups = _build_completion_groups(
        adapter,
        images=images,
        prompt_utf8=prompt_utf8,
        attribute_token_span=attribute_token_span,
        patch_tokens_per_image=patch_tokens_per_image,
        protocol=protocol,
        rollout_authority=rollout_authority,
        candidate_schedule=candidate_schedule,
        example_ids=example_ids,
        labels=labels,
    )
    eligible = assemble_asgcv_eligible_schedule(
        candidate_schedule, sealed_groups, target_pair_count=target_pair_count
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


def build_marginal_schedule(
    adapter: EligibilityAdapter,
    *,
    images: tuple[np.ndarray, ...],
    prompt_utf8: str,
    attribute_token_span: tuple[int, int],
    patch_tokens_per_image: int,
    protocol: AsgcvCompletionProtocol,
    rollout_authority: AsgcvRolloutAuthority,
    candidate_schedule: AsgcvPairSchedule,
    example_ids: tuple[str, ...],
    labels: tuple[int, ...],
) -> tuple[tuple[AsgcvCompletionGroup, ...], AsgcvMarginalSchedule]:
    """Seal candidate-marginal targets without outcome-conditioned refill."""

    sealed_groups = _build_completion_groups(
        adapter,
        images=images,
        prompt_utf8=prompt_utf8,
        attribute_token_span=attribute_token_span,
        patch_tokens_per_image=patch_tokens_per_image,
        protocol=protocol,
        rollout_authority=rollout_authority,
        candidate_schedule=candidate_schedule,
        example_ids=example_ids,
        labels=labels,
    )
    marginal = assemble_asgcv_marginal_schedule(candidate_schedule, sealed_groups)
    validate_asgcv_marginal_protocol_bundle(
        protocol,
        rollout_authority,
        candidate_schedule,
        sealed_groups,
        marginal,
        example_ids=example_ids,
        labels=labels,
    )
    return sealed_groups, marginal
