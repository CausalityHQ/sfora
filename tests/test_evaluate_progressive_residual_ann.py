from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_progressive_residual_ann.py"
SPEC = importlib.util.spec_from_file_location("evaluate_progressive_residual_ann", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUBJECT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUBJECT
SPEC.loader.exec_module(SUBJECT)


def test_candidate_containment_uses_full_candidate_width_not_truth_width() -> None:
    candidates = np.arange(1_000, dtype=np.int64).reshape(1, 1_000)
    truth = np.array([[0, 999]], dtype=np.int64)

    full = SUBJECT.candidate_containment_hits(
        candidates,
        truth,
        candidate_width=1_000,
        truth_width=2,
    )
    truncated = SUBJECT.candidate_containment_hits(
        candidates,
        truth,
        candidate_width=10,
        truth_width=2,
    )

    assert full.tolist() == [2]
    assert truncated.tolist() == [1]


@pytest.mark.parametrize(
    ("candidate_width", "truth_width", "mutation"),
    [
        (True, 2, None),
        (1_001, 2, None),
        (1_000, 0, None),
        (1_000, 3, None),
        (1_000, 2, "candidate-duplicate"),
        (1_000, 2, "truth-duplicate"),
        (1_000, 2, "negative"),
        (1_000, 2, "dtype"),
    ],
)
def test_candidate_containment_rejects_width_schema_and_identity_drift(
    candidate_width: object,
    truth_width: object,
    mutation: str | None,
) -> None:
    candidates = np.arange(1_000, dtype=np.int64).reshape(1, 1_000)
    truth = np.array([[0, 999]], dtype=np.int64)
    if mutation == "candidate-duplicate":
        candidates[0, -1] = 0
    elif mutation == "truth-duplicate":
        truth[0, -1] = 0
    elif mutation == "negative":
        candidates[0, -1] = -1
    elif mutation == "dtype":
        candidates = candidates.astype(np.int32)

    with pytest.raises(ValueError, match="candidate containment authority differs"):
        SUBJECT.candidate_containment_hits(
            candidates,
            truth,
            candidate_width=candidate_width,
            truth_width=truth_width,
        )
