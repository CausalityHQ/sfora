from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "_scratch_benchmark_cutile_int8_topk.py"
_SPEC = importlib.util.spec_from_file_location("scratch_benchmark_cutile_int8_topk", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_decision = _MODULE._decision
_nearest_rank = _MODULE._nearest_rank
_summary = _MODULE._summary


def _batch(p99: int) -> dict[str, object]:
    values = list(range(1, 50)) + [p99]
    return {"native": _summary(values, batch=1)}


def test_receipt_recomputes_percentiles_and_frozen_decision() -> None:
    values = list(range(1, 51))
    summary = _summary(values, batch=32)

    assert summary["p50_ns"] == 25
    assert summary["p99_ns"] == 50
    assert summary["sample_count"] == 50
    assert _nearest_rank(values, 0.99) == 50
    assert _decision(
        {"1": _batch(2_000_000), "32": _batch(6_000_000)},
        exact=True,
        peak_rss=1_000_000_000,
    )
    assert not _decision(
        {"1": _batch(2_100_000), "32": _batch(6_000_000)},
        exact=True,
        peak_rss=1_000_000_000,
    )
    assert not _decision(
        {"1": _batch(2_000_000), "32": _batch(6_000_000)},
        exact=False,
        peak_rss=1_000_000_000,
    )
    with pytest.raises(ValueError, match="timing authority"):
        _summary(values[:-1], batch=1)
