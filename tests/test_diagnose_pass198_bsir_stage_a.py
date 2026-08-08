from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_pass198_bsir_stage_a.py"
SPEC = importlib.util.spec_from_file_location("diagnose_pass198_bsir_stage_a", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compare_tail_records_identity_and_correctness_flips() -> None:
    gallery = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    gallery_labels = np.asarray([10, 20, 30])
    canonical = np.asarray([[1.0, 0.0], [0.8, 0.6]])
    legacy = np.asarray([[0.0, 1.0], [0.6, 0.8]])
    query_labels = np.asarray([10, 20])

    result = MODULE.compare_tail_retrieval(
        canonical,
        legacy,
        query_labels,
        gallery,
        gallery_labels,
        row_offset=100,
    )

    assert result["nearest_identity_flips"] == 2
    assert result["correct_to_wrong"] == 1
    assert result["wrong_to_correct"] == 1
    assert result["absolute_correctness_changes"] == 2
    assert result["net_correctness_change"] == 0
    assert [row["query_row"] for row in result["rows"]] == [100, 101]


def test_stability_certificate_uses_canonical_top_two_margin() -> None:
    gallery = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    result = MODULE.compare_tail_retrieval(
        np.asarray([[1.0, 0.0]]),
        np.asarray([[0.9999995, 0.001]]),
        np.asarray([1]),
        gallery,
        np.asarray([1, 2, 3]),
        row_offset=0,
    )
    row = result["rows"][0]
    assert np.isclose(row["canonical_top1_top2_margin"], 0.2)
    assert row["stable_by_two_l2_bound"] is True


def test_verdict_pass_fail_and_unresolved_boundaries() -> None:
    assert MODULE.stage_a_verdict([3, 3, 3, 3])["stage_a"] == "PASS_ONWARD"
    assert MODULE.stage_a_verdict([0, 0, 0, 0])["stage_a"] == "FAIL"
    assert MODULE.stage_a_verdict([3, 3, 0, 0])["stage_a"] == "UNRESOLVED"


def test_compare_rejects_nonunit_or_misaligned_inputs() -> None:
    with np.testing.assert_raises(ValueError):
        MODULE.compare_tail_retrieval(
            np.asarray([[2.0, 0.0]]),
            np.asarray([[1.0, 0.0]]),
            np.asarray([1]),
            np.asarray([[1.0, 0.0]]),
            np.asarray([1]),
            row_offset=0,
        )
