from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sfora.unicom_finish_evidence import (
    bind_finish_evidence,
    canonical_finish_evidence_bytes,
    validate_finish_evidence,
)


def _evaluation_receipt(root: Path) -> dict[str, object]:
    ranked = root / "evaluation-epoch-0008-ranked-prefix.json"
    query = root / "evaluation-epoch-0008-query.npy"
    gallery = root / "evaluation-epoch-0008-gallery.npy"
    ranked.write_text("[]\n")
    query.write_bytes(b"query")
    gallery.write_bytes(b"gallery")
    receipt = {
        "schema": "unicom-evaluation-evidence-v1",
        "epoch": 8,
        "metrics": {
            "recall_at_1": 0.75,
            "recall_at_10": 1.0,
            "recall_at_20": 1.0,
            "recall_at_30": 1.0,
            "map_at_r": 0.625,
        },
    }
    payload = (json.dumps(receipt, indent=2) + "\n").encode()
    (root / "evaluation-epoch-0008.json").write_bytes(payload)
    return receipt


def test_finish_evidence_binds_validated_evaluation_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _evaluation_receipt(tmp_path)
    validated: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        "sfora.unicom_finish_evidence.validate_evaluation_evidence",
        lambda value, root: validated.append((value, root)),
    )

    bundle = bind_finish_evidence(
        arm="smooth-ap-pk",
        finish_seed=3,
        schedule_sha256="ab" * 32,
        evidence_root=tmp_path,
    )

    assert validated == [
        (receipt, tmp_path.resolve()),
        (receipt, tmp_path.resolve()),
    ]
    assert bundle["metrics"] == receipt["metrics"]
    assert bundle["evaluation_receipt"]["sha256"] == hashlib.sha256(
        (tmp_path / "evaluation-epoch-0008.json").read_bytes()
    ).hexdigest()
    assert canonical_finish_evidence_bytes(bundle).endswith(b"\n")
    validate_finish_evidence(
        bundle, tmp_path, expected_schedule_sha256="ab" * 32
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "arm",
        "seed",
        "schedule",
        "receipt_hash",
        "receipt_bytes",
        "metric",
        "metric_type",
        "extra",
    ),
)
def test_finish_evidence_rejects_authority_and_metric_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    _evaluation_receipt(tmp_path)
    monkeypatch.setattr(
        "sfora.unicom_finish_evidence.validate_evaluation_evidence",
        lambda _value, _root: None,
    )
    bundle = bind_finish_evidence(
        arm="classification-padded",
        finish_seed=3,
        schedule_sha256="12" * 32,
        evidence_root=tmp_path,
    )
    changed = copy.deepcopy(bundle)
    if mutation == "arm":
        changed["arm"] = "unknown"
    elif mutation == "seed":
        changed["finish_seed"] = 4
    elif mutation == "schedule":
        changed["schedule_sha256"] = "0" * 64
    elif mutation == "receipt_hash":
        changed["evaluation_receipt"]["sha256"] = "0" * 64
    elif mutation == "receipt_bytes":
        changed["evaluation_receipt"]["bytes"] += 1
    elif mutation == "metric":
        changed["metrics"]["map_at_r"] = 0.5
    elif mutation == "metric_type":
        changed["metrics"]["recall_at_1"] = 1
    else:
        changed["extra"] = False

    with pytest.raises(ValueError):
        validate_finish_evidence(
            changed, tmp_path, expected_schedule_sha256="12" * 32
        )
