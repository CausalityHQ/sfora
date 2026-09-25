"""Small exact ranking cases for the SOP fit-only diagnosis."""

import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_sop_siglip2_fit_errors.py"
SPEC = importlib.util.spec_from_file_location("diagnose_sop_siglip2_fit_errors", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
classify_scores = MODULE.classify_scores


def test_fit_error_rank_excludes_self_and_breaks_score_ties_by_ordinal() -> None:
    labels = torch.tensor([1, 1, 2, 2], dtype=torch.int64)
    scores = torch.tensor([[1.0, 0.4, 0.5, 0.4], [0.6, 1.0, 0.3, 0.3]], dtype=torch.float32)
    top, best_mate, mate_rank = classify_scores(
        scores, labels, torch.tensor([0, 1], dtype=torch.int64)
    )
    assert top.tolist() == [2, 0]
    assert best_mate.tolist() == [1, 0]
    assert mate_rank.tolist() == [2, 1]
