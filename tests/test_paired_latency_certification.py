"""Statistical gates for paired image-to-top-k tail measurements."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _subject():
    path = ROOT / "scripts/paired_latency_certification.py"
    spec = importlib.util.spec_from_file_location("paired_latency_certification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_order_reverses_each_block() -> None:
    subject = _subject()
    assert subject.paired_order(0) == ("oml", "trained_b16")
    assert subject.paired_order(1) == ("trained_b16", "oml")


def test_certification_requires_ten_thousand_calls_per_cell() -> None:
    subject = _subject()
    subject.validate_certification_shape(20, 500)
    with pytest.raises(ValueError, match="at least twenty paired blocks"):
        subject.validate_certification_shape(19, 1000)
    with pytest.raises(ValueError, match="at least 10000 calls"):
        subject.validate_certification_shape(20, 499)
    with pytest.raises(ValueError, match="even paired block count"):
        subject.validate_certification_shape(21, 500)


def test_block_bootstrap_resamples_paired_blocks() -> None:
    subject = _subject()
    reference = [[100] * 500 for _ in range(20)]
    candidate = [[80] * 500 for _ in range(20)]
    result = subject.block_bootstrap_p99_ratio(candidate, reference, draws=200, seed=7)
    assert result["point_ratio"] == pytest.approx(0.8)
    assert result["ci95_lower"] == pytest.approx(0.8)
    assert result["ci95_upper"] == pytest.approx(0.8)
    assert result["candidate_p99_ns"] == 80
    assert result["reference_p99_ns"] == 100
    assert result["one_sided_sign_p"] == pytest.approx(1 / 1024)
    assert result["point_p50_ratio"] == pytest.approx(0.8)
    assert result["mean_latency_ratio"] == pytest.approx(0.8)
    assert result["latency_gate_passed"] is True
    with pytest.raises(ValueError, match="paired block shape"):
        subject.block_bootstrap_p99_ratio(candidate[:-1], reference, draws=200, seed=7)


def test_paired_bootstrap_preserves_correlated_block_values() -> None:
    subject = _subject()
    reference = [[100 + 10 * (block // 2)] * 500 for block in range(20)]
    result = subject.block_bootstrap_p99_ratio(reference, reference, draws=500, seed=7)
    assert result["superblocks"] == 10
    assert result["point_ratio"] == 1.0
    assert result["ci95_lower"] == 1.0
    assert result["ci95_upper"] == 1.0
    assert result["latency_gate_passed"] is False


def test_bootstrap_rejects_noninteger_and_nonpositive_durations() -> None:
    subject = _subject()
    reference = [[100] * 500 for _ in range(20)]
    for value in (0, -1, 1.5, "10", True):
        candidate = [row.copy() for row in reference]
        candidate[0][0] = value
        with pytest.raises(ValueError, match="paired block samples differ"):
            subject.block_bootstrap_p99_ratio(candidate, reference, draws=100, seed=7)
