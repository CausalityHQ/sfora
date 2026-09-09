from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_subject() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "run_sop_teacher_anchored_panel.py"
    spec = importlib.util.spec_from_file_location("run_sop_teacher_anchored_panel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUBJECT = _load_subject()


def _sample(
    *,
    rss: int = 1024,
    psi: float = 0.0,
    swap_delta: int = 0,
    progress_age: float = 1.0,
    wall: float = 1.0,
) -> object:
    return SUBJECT.TeacherAnchoredPressureSample(
        rss_bytes=rss,
        psi_full_avg10=psi,
        swap_delta_bytes=swap_delta,
        progress_age_seconds=progress_age,
        wall_seconds=wall,
    )


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ((_sample(rss=32 * 1024**3 + 1),), "rss-cap"),
        ((_sample(psi=0.79),), "psi-immediate"),
        ((_sample(psi=0.5), _sample(psi=0.51), _sample(psi=0.5)), "psi-sustained"),
        ((_sample(swap_delta=2 * 1024**3 + 1),), "swap-delta"),
        ((_sample(progress_age=900.0),), "progress-timeout"),
        ((_sample(wall=float(18 * 3600)),), "wall-timeout"),
        ((_sample(psi=0.5), _sample(psi=0.0), _sample(psi=0.5)), None),
    ],
)
def test_stop_classification_is_exact_and_sustained(
    samples: tuple[object, ...], expected: str | None
) -> None:
    assert SUBJECT.classify_teacher_anchored_stop(samples) == expected


def test_canonical_stop_receipt_is_outcome_blind_and_hash_stable() -> None:
    launch = "a" * 64
    progress = "b" * 64
    samples = (_sample(psi=0.8),)
    raw = SUBJECT.canonical_teacher_anchored_stop_bytes(
        launch_receipt_sha256=launch,
        last_progress_sha256=progress,
        last_arm="complete",
        last_epoch=4,
        last_update=71,
        samples=samples,
        reason="psi-immediate",
    )
    value = json.loads(raw)
    assert raw == json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert value["claim_eligible"] is False
    assert value["reason"] == "psi-immediate"
    assert "map" not in raw.decode().lower()
    assert len(hashlib.sha256(raw).hexdigest()) == 64
    with pytest.raises(ValueError, match="stop receipt"):
        SUBJECT.canonical_teacher_anchored_stop_bytes(
            launch_receipt_sha256=launch,
            last_progress_sha256=progress,
            last_arm="complete",
            last_epoch=4,
            last_update=71,
            samples=samples,
            reason="quality-failed",
        )


def test_progress_cursor_advances_only_on_strict_authenticated_extension() -> None:
    launch = "a" * 64
    first = (
        json.dumps(
            {
                "arm": "complete",
                "epoch": 0,
                "launch_receipt_sha256": launch,
                "previous_line_sha256": "0" * 64,
                "schema": "sfora-teacher-anchored-progress-v1",
                "sequence": 1,
                "stage": "initialized",
                "update": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    second = (
        json.dumps(
            {
                "arm": "complete",
                "epoch": 1,
                "launch_receipt_sha256": launch,
                "previous_line_sha256": hashlib.sha256(first).hexdigest(),
                "schema": "sfora-teacher-anchored-progress-v1",
                "sequence": 2,
                "stage": "update",
                "update": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    state = SUBJECT.advance_teacher_anchored_progress(None, (first,), launch)
    assert state.sequence == 1
    state = SUBJECT.advance_teacher_anchored_progress(state, (first, second), launch)
    assert state.sequence == 2
    with pytest.raises(ValueError, match="progress replay"):
        SUBJECT.advance_teacher_anchored_progress(state, (first, second), launch)


def test_progress_validator_is_loaded_once_per_launcher_process() -> None:
    first = SUBJECT._load_progress_validator()
    second = SUBJECT._load_progress_validator()

    assert first is second
