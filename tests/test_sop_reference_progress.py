"""Paired product resampling must preserve the original query pairing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_sop_reference_progress.py"
SPEC = importlib.util.spec_from_file_location("analyze_sop_reference_progress", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_class_bootstrap_is_deterministic_and_respects_class_sizes() -> None:
    labels = np.asarray([1, 1, 1, 2], dtype=np.int64)
    # Resampling products, then their complete query groups, yields either a
    # 3:1 weighted contrast or a pure product contrast.
    delta = np.asarray([1.0, 1.0, 1.0, -1.0])
    first = MODULE.product_bootstrap(delta, labels, seed=7, replicates=256)
    second = MODULE.product_bootstrap(delta, labels, seed=7, replicates=256)
    assert first == second
    assert first["point"] == pytest.approx(0.5)
    assert first["lower_95"] == pytest.approx(-1.0)
    assert first["upper_95"] == pytest.approx(1.0)


def test_paired_class_bootstrap_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="paired product inputs differ"):
        MODULE.product_bootstrap(
            np.asarray([1.0, np.nan]), np.asarray([1, 2]), seed=7, replicates=8
        )
    with pytest.raises(ValueError, match="paired product inputs differ"):
        MODULE.product_bootstrap(np.asarray([1.0]), np.asarray([1, 2]), seed=7, replicates=8)


def test_real_holdout_receipts_require_the_same_products_and_recomputed_metrics() -> None:
    evidence = Path(__file__).parents[1] / "docs/evidence/compact_metric"
    earlier = json.loads(
        (evidence / "sop-reference-arcface-seed179019-step4000-v1.json").read_text()
    )
    later = json.loads((evidence / "sop-reference-arcface-seed179019-step8000-v1.json").read_text())
    observed = MODULE.analyze(earlier, later, seed=7, replicates=32)
    assert observed["metrics"]["map_at_r"]["point"] == pytest.approx(
        later["validation"]["packed"]["map_at_r"] - earlier["validation"]["packed"]["map_at_r"]
    )

    later["validation_labels"][0] += 1
    with pytest.raises(ValueError, match="SOP holdout pairing differs"):
        MODULE.analyze(earlier, later, seed=7, replicates=32)
