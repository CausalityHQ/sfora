#!/usr/bin/env python3
"""Pressure and progress authority for the local teacher-anchored SOP panel."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, cast


class TeacherAnchoredPressureSample(NamedTuple):
    """One process-group and host-pressure observation."""

    rss_bytes: int
    psi_full_avg10: float
    swap_delta_bytes: int
    progress_age_seconds: float
    wall_seconds: float


class TeacherAnchoredProgressState(NamedTuple):
    """Last launch-bound progress cursor accepted by the watchdog."""

    sequence: int
    arm: str
    epoch: int
    update: int
    line_sha256: str


def _load_progress_validator() -> Callable[[tuple[bytes, ...], str], object]:
    path = Path(__file__).resolve().parent / "train_sop_teacher_anchored_distillation.py"
    spec = importlib.util.spec_from_file_location("sfora_teacher_anchored_progress", path)
    if spec is None or spec.loader is None:
        raise ValueError("teacher-anchored progress replay differs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    validator = getattr(module, "validate_teacher_anchored_progress_chain", None)
    if not callable(validator):
        raise ValueError("teacher-anchored progress replay differs")
    return cast(Callable[[tuple[bytes, ...], str], object], validator)


def advance_teacher_anchored_progress(
    previous: TeacherAnchoredProgressState | None,
    lines: tuple[bytes, ...],
    launch_receipt_sha256: str,
) -> TeacherAnchoredProgressState:
    """Advance only when the complete authenticated chain strictly extends."""

    if previous is not None and type(previous) is not TeacherAnchoredProgressState:
        raise ValueError("teacher-anchored progress replay differs")
    raw_state = cast(
        TeacherAnchoredProgressState,
        _load_progress_validator()(lines, launch_receipt_sha256),
    )
    try:
        current = TeacherAnchoredProgressState(
            sequence=raw_state.sequence,
            arm=raw_state.arm,
            epoch=raw_state.epoch,
            update=raw_state.update,
            line_sha256=raw_state.line_sha256,
        )
    except AttributeError as error:
        raise ValueError("teacher-anchored progress replay differs") from error
    if previous is not None and (
        current.sequence <= previous.sequence
        or current.arm != previous.arm
        or len(lines) < previous.sequence
        or hashlib.sha256(lines[previous.sequence - 1]).hexdigest() != previous.line_sha256
    ):
        raise ValueError("teacher-anchored progress replay differs")
    return current


def classify_teacher_anchored_stop(
    samples: tuple[TeacherAnchoredPressureSample, ...],
) -> str | None:
    """Return the first registered resource stop from ordered samples."""

    if type(samples) is not tuple or not samples:
        raise ValueError("teacher-anchored pressure authority differs")
    sustained = 0
    for sample in samples:
        if (
            type(sample) is not TeacherAnchoredPressureSample
            or type(sample.rss_bytes) is not int
            or sample.rss_bytes < 0
            or type(sample.psi_full_avg10) is not float
            or not math.isfinite(sample.psi_full_avg10)
            or sample.psi_full_avg10 < 0.0
            or type(sample.swap_delta_bytes) is not int
            or type(sample.progress_age_seconds) is not float
            or type(sample.wall_seconds) is not float
            or not math.isfinite(sample.progress_age_seconds)
            or not math.isfinite(sample.wall_seconds)
            or sample.progress_age_seconds < 0.0
            or sample.wall_seconds < 0.0
        ):
            raise ValueError("teacher-anchored pressure authority differs")
        if sample.wall_seconds >= 18 * 3600:
            return "wall-timeout"
        if sample.rss_bytes > 32 * 1024**3:
            return "rss-cap"
        if sample.psi_full_avg10 >= 0.79:
            return "psi-immediate"
        sustained = sustained + 1 if sample.psi_full_avg10 >= 0.50 else 0
        if sustained >= 3:
            return "psi-sustained"
        if sample.swap_delta_bytes > 2 * 1024**3:
            return "swap-delta"
        if sample.progress_age_seconds >= 900.0:
            return "progress-timeout"
    return None


def canonical_teacher_anchored_stop_bytes(
    *,
    launch_receipt_sha256: str,
    last_progress_sha256: str,
    last_arm: str,
    last_epoch: int,
    last_update: int,
    samples: tuple[TeacherAnchoredPressureSample, ...],
    reason: str,
) -> bytes:
    """Encode an outcome-blind canonical external-stop receipt."""

    classified = classify_teacher_anchored_stop(samples)
    if (
        not _is_sha256(launch_receipt_sha256)
        or not _is_sha256(last_progress_sha256)
        or last_arm not in ("head-only", "base", "anchor", "symmetric", "complete")
        or type(last_epoch) is not int
        or not 0 <= last_epoch <= 10
        or type(last_update) is not int
        or last_update < 0
        or reason != classified
    ):
        raise ValueError("teacher-anchored stop receipt differs")
    value = {
        "claim_eligible": False,
        "last_arm": last_arm,
        "last_epoch": last_epoch,
        "last_progress_sha256": last_progress_sha256,
        "last_update": last_update,
        "launch_receipt_sha256": launch_receipt_sha256,
        "reason": reason,
        "samples": [sample._asdict() for sample in samples],
        "schema": "sfora-teacher-anchored-stop-v1",
    }
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def main() -> None:
    """Refuse execution until the process-group launcher boundary is complete."""

    raise SystemExit("teacher-anchored panel launcher is not yet enabled")


if __name__ == "__main__":
    main()
