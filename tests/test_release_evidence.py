"""The release report must quote only digests and numbers its receipts carry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

_REPORT = Path("reports/factorized_residual_ann_2026-09-13.md")
_PUBLIC = Path("reports/receipts/factorized_residual_ann_2026-09-13")
_MATCHED = Path("reports/receipts/factorized_residual_ann_2026-09-19-threadmatched")


def _report() -> str:
    return _REPORT.read_text()


def test_every_public_evaluator_receipt_digest_is_quoted_by_the_report() -> None:
    receipts = sorted(_PUBLIC.glob("*.json"))
    assert receipts, "no public evaluator receipts retained"
    report = _report()
    for receipt in receipts:
        digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
        assert digest in report, f"{receipt.name} digest is not quoted"


@pytest.mark.parametrize(
    ("receipt", "field", "expected"),
    (
        ("sfora-product-evaluator-bigann100m-holdout9000-final-v6.json", "p99", "7.052182"),
        ("sfora-product-evaluator-bigann100m-holdout9000-final-v6.json", "mean", "5.469377"),
    ),
)
def test_report_latency_figures_match_the_current_receipt(
    receipt: str, field: str, expected: str
) -> None:
    payload = json.loads((_PUBLIC / receipt).read_text())
    assert f"{payload['latency_ns'][field] / 1e6:.6f}" == expected
    assert expected in _report()


def test_current_receipt_reproduces_the_frozen_prototype_output() -> None:
    payload = json.loads(
        (_PUBLIC / "sfora-product-evaluator-bigann100m-holdout9000-final-v6.json").read_text()
    )
    assert payload["strict_id_recall_ppm"] == 989583
    assert payload["outputs_sha256"] in _report()
    assert not [sample for sample in payload["latency_raw_ns"] if sample > 15_000_000]


def test_matched_thread_arms_all_report_the_same_recall() -> None:
    sfora = [
        json.loads((_MATCHED / name).read_text())["strict_id_recall_ppm"] / 1e6
        for name in ("armH-sfora-final-t1-dev1000.json", "armG-coarse-par-t20-dev1000.json")
    ]
    faiss = [
        round(json.loads((_MATCHED / name).read_text())["strict_id_recall_at_100"], 6)
        for name in ("armC2-faiss-pm1-t1-dev1000.json", "armD-faiss-pm1-t20-dev1000.json")
    ]
    # A speed comparison is only meaningful if every arm returns the same quality.
    assert set(sfora) | set(faiss) == {0.98812}


def test_faiss_control_arms_record_the_parallel_mode_they_ran_with() -> None:
    for name in ("armC2-faiss-pm1-t1-dev1000.json", "armD-faiss-pm1-t20-dev1000.json"):
        protocol = json.loads((_MATCHED / name).read_text())["protocol"]
        assert protocol["parallel_mode"] == 1, "the matched control must not run query-parallel"
        assert "threads" in protocol
