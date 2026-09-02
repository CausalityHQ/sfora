#!/usr/bin/env python3
"""Capture and fit the local-only ASG-CV forced-gradient student."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from sfora.asgcv_forced_distill import (
    ASGCV_FORCED_DISTILL_SHAPE,
    ForcedDistillCapture,
    canonical_forced_distill_capture_bytes,
    relation_correct_gradient,
    validate_forced_distill_capture_bytes,
)
from sfora.asgcv_protocol import AsgcvCompletionProtocol, AsgcvPair, AsgcvPairSchedule

_BOUNDARY_NAMES = ("merger", "deepstack-0", "deepstack-1", "deepstack-2")


class ForcedDistillAdapter(Protocol):
    """Only Qwen operations needed to capture one forced target."""

    def prepare_image_pair(
        self,
        images: object,
        prompt_utf8: object,
        attribute_token_span: object,
        patch_tokens_per_image: object,
    ) -> object: ...

    def collapsed_verdict_patch_gradient(
        self,
        pair: object,
        *,
        correct_completion_ids: tuple[int, ...],
        incorrect_completion_ids: tuple[int, ...],
    ) -> object: ...


def _array(value: object, *, name: str) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and type(value) is torch.Tensor:
        value = value.detach().float().cpu().contiguous().numpy()
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float32)
        or value.shape != ASGCV_FORCED_DISTILL_SHAPE
        or not bool(np.isfinite(value).all())
    ):
        raise ValueError(f"ASG-CV forced distill {name} differs")
    return np.ascontiguousarray(value)


def capture_forced_distill_pair(
    adapter: ForcedDistillAdapter,
    *,
    role: str,
    pair: AsgcvPair,
    schedule: AsgcvPairSchedule,
    images: tuple[np.ndarray, ...],
    prompt_utf8: str,
    attribute_token_span: tuple[int, int],
    patch_tokens_per_image: int,
    completion_protocol: AsgcvCompletionProtocol,
    source_commit: str,
    launch_authority_sha256: str,
) -> tuple[bytes, np.ndarray, np.ndarray]:
    """Capture one fixed-order target and orient it to the registered relation."""

    schedule.validated()
    completion_protocol.validated()
    if (
        type(pair) is not AsgcvPair
        or pair.ordinal >= schedule.pair_count
        or schedule.pairs[pair.ordinal] != pair
        or max(pair.left_index, pair.right_index) >= len(images)
    ):
        raise ValueError("ASG-CV forced distill pair differs")
    prepared = adapter.prepare_image_pair(
        (images[pair.left_index], images[pair.right_index]),
        prompt_utf8,
        attribute_token_span,
        patch_tokens_per_image,
    )
    target = adapter.collapsed_verdict_patch_gradient(
        prepared,
        correct_completion_ids=completion_protocol.same_prefix_ids,
        incorrect_completion_ids=completion_protocol.different_prefix_ids,
    )
    if getattr(target, "boundary_names", None) != _BOUNDARY_NAMES:
        raise ValueError("ASG-CV forced distill boundary differs")
    patches = _array(getattr(target, "patch_tokens", None), name="patch tokens")
    fixed_gradient = _array(getattr(target, "predicted_gradient", None), name="gradient")
    gradient = relation_correct_gradient(fixed_gradient, pair.relation_sign)
    capture = ForcedDistillCapture(
        source_commit=source_commit,
        launch_authority_sha256=launch_authority_sha256,
        schedule_sha256=schedule.sha256(),
        role=role,
        pair_ordinal=pair.ordinal,
        pair_indices=(pair.left_index, pair.right_index),
        relation_sign=pair.relation_sign,
        patch_sha256=hashlib.sha256(patches.tobytes()).hexdigest(),
        gradient_sha256=hashlib.sha256(gradient.tobytes()).hexdigest(),
        array_shape=ASGCV_FORCED_DISTILL_SHAPE,
    ).validated()
    return canonical_forced_distill_capture_bytes(capture), patches, gradient


def _paths(directory: Path, role: str, ordinal: int) -> tuple[Path, Path, Path]:
    stem = f"{role}-{ordinal:06d}"
    return (
        directory / f"{stem}.json",
        directory / f"{stem}-patch.npy",
        directory / f"{stem}-gradient.npy",
    )


def _load_array(path: Path) -> np.ndarray:
    try:
        with path.open("rb") as stream:
            value = np.load(stream, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("ASG-CV forced distill array file differs") from error
    return _array(value, name="array")


def _validate_triple(
    directory: Path,
    *,
    role: str,
    schedule: AsgcvPairSchedule,
    ordinal: int,
) -> None:
    receipt_path, patch_path, gradient_path = _paths(directory, role, ordinal)
    capture = validate_forced_distill_capture_bytes(
        receipt_path.read_bytes(),
        patch_tokens=_load_array(patch_path),
        exact_gradient=_load_array(gradient_path),
    )
    pair = schedule.pairs[ordinal]
    if (
        capture.role != role
        or capture.pair_ordinal != ordinal
        or capture.schedule_sha256 != schedule.sha256()
        or capture.pair_indices != (pair.left_index, pair.right_index)
        or capture.relation_sign != pair.relation_sign
    ):
        raise ValueError("ASG-CV forced distill capture context differs")


def _write_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _write_triple(
    directory: Path,
    *,
    role: str,
    ordinal: int,
    receipt: bytes,
    patches: np.ndarray,
    gradient: np.ndarray,
) -> None:
    final = _paths(directory, role, ordinal)
    partial = tuple(path.with_name(path.name + ".partial") for path in final)
    if any(path.exists() for path in (*final, *partial)):
        raise ValueError("ASG-CV forced distill output exists")
    try:
        with partial[0].open("xb") as stream:
            stream.write(receipt)
            stream.flush()
            os.fsync(stream.fileno())
        _write_array(partial[1], patches)
        _write_array(partial[2], gradient)
        for source, destination in zip(partial, final, strict=True):
            os.replace(source, destination)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        for path in partial:
            if path.exists():
                path.unlink()


def run_capture_phase(
    directory: Path,
    *,
    role: str,
    schedule: AsgcvPairSchedule,
    execute_one: Callable[[int], tuple[bytes, np.ndarray, np.ndarray]],
    maximum_new_rows: int | None = None,
) -> int:
    """Validate a contiguous prefix and append a bounded number of captures."""

    if (
        not isinstance(directory, Path)
        or not directory.is_dir()
        or role not in {"train", "validation"}
        or not callable(execute_one)
        or (maximum_new_rows is not None and maximum_new_rows <= 0)
        or tuple(directory.glob("*.partial"))
    ):
        raise ValueError("ASG-CV forced distill capture phase differs")
    schedule.validated()
    prefix = 0
    for ordinal in range(schedule.pair_count):
        existing = tuple(path.exists() for path in _paths(directory, role, ordinal))
        if all(existing):
            _validate_triple(directory, role=role, schedule=schedule, ordinal=ordinal)
            prefix += 1
            continue
        if any(existing):
            raise ValueError("ASG-CV forced distill partial triple differs")
        if any(
            any(path.exists() for path in _paths(directory, role, later))
            for later in range(ordinal + 1, schedule.pair_count)
        ):
            raise ValueError("ASG-CV forced distill ordinal gap differs")
        break
    stop = schedule.pair_count
    if maximum_new_rows is not None:
        stop = min(stop, prefix + maximum_new_rows)
    for ordinal in range(prefix, stop):
        receipt, patches, gradient = execute_one(ordinal)
        validate_forced_distill_capture_bytes(
            receipt,
            patch_tokens=patches,
            exact_gradient=gradient,
        )
        _write_triple(
            directory,
            role=role,
            ordinal=ordinal,
            receipt=receipt,
            patches=patches,
            gradient=gradient,
        )
        _validate_triple(directory, role=role, schedule=schedule, ordinal=ordinal)
    return stop


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the strict local-only distillation boundary."""

    values = list(sys.argv[1:] if argv is None else argv)
    option_names = [value.split("=", 1)[0] for value in values if value.startswith("--")]
    if len(option_names) != len(set(option_names)):
        raise SystemExit("duplicate ASG-CV forced distill option")
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--p32-authority", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--execute-forced-distill", required=True, action="store_true")
    parsed = parser.parse_args(values)
    if (
        type(parsed.source_commit) is not str
        or len(parsed.source_commit) != 40
        or any(character not in "0123456789abcdef" for character in parsed.source_commit)
    ):
        parser.error("source commit must be 40 lowercase hex")
    if not parsed.model_root.is_dir() or parsed.model_root.is_symlink():
        parser.error("model root must be an existing regular directory")
    for name in ("snapshot_manifest", "fixture", "p32_authority", "train_manifest"):
        path = getattr(parsed, name)
        if path.is_symlink() or not path.is_file():
            parser.error(f"{name.replace('_', ' ')} must be an existing regular file")
    if not parsed.output_directory.is_dir() or parsed.output_directory.is_symlink():
        parser.error("output directory must be an existing regular directory")
    return parsed
