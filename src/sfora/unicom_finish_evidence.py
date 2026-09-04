"""Authenticated paired-evaluation bindings for the UniCOM finish panel."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from sfora.unicom_finish_protocol import FinishArm
from sfora.unicom_retrieval_audit import (
    strict_typed_equal,
    validate_evaluation_evidence,
)

_BUNDLE_KEYS = (
    "schema",
    "claim_eligible",
    "arm",
    "finish_seed",
    "schedule_sha256",
    "evaluation_receipt",
    "metrics",
)
_METRIC_KEYS = (
    "recall_at_1",
    "recall_at_10",
    "recall_at_20",
    "recall_at_30",
    "map_at_r",
)


def _digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_shape(value: object) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != _BUNDLE_KEYS:
        raise ValueError("finish evidence bundle differs")
    binding = value["evaluation_receipt"]
    metrics = value["metrics"]
    if (
        value["schema"] != "unicom-finish-evidence-v1"
        or value["claim_eligible"] is not False
        or value["arm"] not in {arm.value for arm in FinishArm}
        or value["finish_seed"] != 3
        or type(value["finish_seed"]) is not int
        or not _digest(value["schedule_sha256"])
        or type(binding) is not dict
        or tuple(binding) != ("path", "sha256", "bytes")
        or binding["path"] != "evaluation-epoch-0008.json"
        or not _digest(binding["sha256"])
        or type(binding["bytes"]) is not int
        or binding["bytes"] <= 0
        or type(metrics) is not dict
        or tuple(metrics) != _METRIC_KEYS
        or any(
            type(metrics[key]) is not float
            or not math.isfinite(metrics[key])
            or not 0.0 <= metrics[key] <= 1.0
            for key in _METRIC_KEYS
        )
    ):
        raise ValueError("finish evidence bundle differs")
    return value


def validate_finish_evidence(
    value: object,
    evidence_root: Path,
    *,
    expected_schedule_sha256: str | None = None,
) -> None:
    """Authenticate an epoch-eight evaluation bundle and recompute its metrics."""

    bundle = _validate_shape(value)
    if expected_schedule_sha256 is not None and (
        not _digest(expected_schedule_sha256)
        or bundle["schedule_sha256"] != expected_schedule_sha256
    ):
        raise ValueError("finish evidence schedule differs")
    if not isinstance(evidence_root, Path):
        raise TypeError("finish evidence root must be a Path")
    root = evidence_root.resolve()
    if evidence_root.absolute() != root or not root.is_dir() or root.is_symlink():
        raise ValueError("finish evidence root differs")
    path = root / "evaluation-epoch-0008.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("finish evaluation receipt differs")
    payload = path.read_bytes()
    binding = bundle["evaluation_receipt"]
    if (
        len(payload) != binding["bytes"]
        or hashlib.sha256(payload).hexdigest() != binding["sha256"]
    ):
        raise ValueError("finish evaluation receipt differs")
    try:
        receipt = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("finish evaluation receipt differs") from error
    expected = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
    if payload != expected or receipt.get("epoch") != 8:
        raise ValueError("finish evaluation receipt differs")
    validate_evaluation_evidence(receipt, root)
    if not strict_typed_equal(bundle["metrics"], receipt.get("metrics")):
        raise ValueError("finish evidence metrics differ")


def bind_finish_evidence(
    *,
    arm: str,
    finish_seed: int,
    schedule_sha256: str,
    evidence_root: Path,
) -> dict[str, object]:
    """Bind an existing recomputable epoch-eight evaluation receipt to one arm."""

    root = evidence_root.resolve()
    path = root / "evaluation-epoch-0008.json"
    payload = path.read_bytes()
    try:
        receipt = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("finish evaluation receipt differs") from error
    validate_evaluation_evidence(receipt, root)
    bundle = {
        "schema": "unicom-finish-evidence-v1",
        "claim_eligible": False,
        "arm": arm,
        "finish_seed": finish_seed,
        "schedule_sha256": schedule_sha256,
        "evaluation_receipt": {
            "path": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        },
        "metrics": dict(receipt["metrics"]),
    }
    validate_finish_evidence(
        bundle, root, expected_schedule_sha256=schedule_sha256
    )
    return bundle


def canonical_finish_evidence_bytes(value: object) -> bytes:
    """Serialize a structurally valid finish-evidence binding canonically."""

    _validate_shape(value)
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


__all__ = [
    "bind_finish_evidence",
    "canonical_finish_evidence_bytes",
    "validate_finish_evidence",
]
