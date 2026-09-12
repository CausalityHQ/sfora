from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_sop_pq_candidate_reranking.py"
sys.path.insert(0, str(_SCRIPT.parent))


def _subject() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_sop_pq_candidate_reranking", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("map_at_r", "r1", "classification"),
    (
        (0.577, 0.83, "candidate-set-reranker-failed-pq32"),
        (0.582, 0.83, "candidate-set-reranker-improved"),
        (0.586, 0.819, "candidate-set-reranker-r1-regressed"),
        (0.586, 0.821, "candidate-set-reranker-target-passed"),
    ),
)
def test_candidate_reranking_decision_uses_frozen_quality_gates(
    map_at_r: float, r1: float, classification: str
) -> None:
    subject = _subject()

    decision = subject.candidate_reranking_decision(
        map_at_r=map_at_r,
        r1=r1,
        pq24_map_at_r=0.5707206723026712,
        pq32_map_at_r=0.5789231844165709,
        pq24_r1=0.8222334768372321,
        seed_standard_deviation=0.001,
    )

    assert decision["classification"] == classification
    assert decision["passed"] is (classification == "candidate-set-reranker-target-passed")
    assert decision["gates"] == {
        "match_pq32_map_at_r": 0.5789231844165709,
        "minimum_r1": pytest.approx(0.8202334768372321),
        "minimum_signal_over_seed_standard_deviation": 0.001,
        "target_map_at_r": 0.58563,
    }


def test_candidate_reranking_rejects_r1_regression_before_improvement() -> None:
    subject = _subject()

    decision = subject.candidate_reranking_decision(
        map_at_r=0.582,
        r1=0.80,
        pq24_map_at_r=0.5707206723026712,
        pq32_map_at_r=0.5789231844165709,
        pq24_r1=0.8222334768372321,
        seed_standard_deviation=0.001,
    )

    assert decision["classification"] == "candidate-set-reranker-r1-regressed"
    assert decision["passed"] is False


def test_candidate_reranking_rejects_target_gain_within_seed_variation() -> None:
    subject = _subject()

    decision = subject.candidate_reranking_decision(
        map_at_r=0.586,
        r1=0.83,
        pq24_map_at_r=0.5707206723026712,
        pq32_map_at_r=0.5789231844165709,
        pq24_r1=0.8222334768372321,
        seed_standard_deviation=0.02,
    )

    assert decision["classification"] == "candidate-set-reranker-within-seed-variation"
    assert decision["passed"] is False


def test_fit_diagnostics_are_batched_and_match_full_loss() -> None:
    subject = _subject()
    generator = torch.Generator().manual_seed(11)
    baseline = torch.randn((7, 4), generator=generator)
    features = torch.randn((7, 4, 7), generator=generator)
    raw_pairwise = torch.randn((7, 4, 4), generator=generator)
    pairwise = 0.5 * (raw_pairwise + raw_pairwise.transpose(1, 2))
    teacher = torch.randn((7, 4), generator=generator)
    model = subject.CandidateSetReranker(blocks=2, model_dimensions=8, heads=2, layers=1).eval()

    observed = subject._fit_diagnostics(model, baseline, features, pairwise, teacher, block_size=3)
    with torch.inference_mode():
        scores = model(baseline, features, pairwise)
        expected = subject.candidate_set_distillation_loss(
            scores,
            teacher,
            temperature=subject.TEMPERATURE,
            score_weight=subject.SCORE_WEIGHT,
        )
        probabilities = torch.softmax(teacher / subject.TEMPERATURE, dim=-1)
        effective_support = torch.exp(
            -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum(dim=-1).mean()
        )

    assert observed == pytest.approx(
        {
            "listwise_kl": float(expected.listwise_kl),
            "score_mse": float(expected.score_mse),
            "target_effective_support": float(effective_support),
            "total": float(expected.total),
        }
    )


def test_score_ranked_candidates_recomputes_map_at_r_and_r1() -> None:
    subject = _subject()
    labels = (1, 1, 1, 2, 2, 2)
    rankings = torch.tensor([[1, 3], [0, 2], [4, 0], [4, 0], [3, 5], [1, 4]], dtype=torch.int64)

    score = subject.score_ranked_candidates(rankings, labels)

    assert score["map_at_r"] == pytest.approx((0.5 + 1.0 + 0.25 + 0.5 + 1.0 + 0.25) / 6)
    assert score["r1"] == pytest.approx(4 / 6)
    assert len(score["per_query_ap"]) == 6


def test_candidate_reranking_requires_explicit_execution() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "--execute-candidate-set-reranking" in result.stderr


def test_parent_receipt_binds_checkpoint_and_all_representation_inputs() -> None:
    subject = _subject()
    receipt: dict[str, object] = {
        key: None
        for key in (
            "claim_eligible",
            "codec",
            "codebook_checkpoint",
            "dataset",
            "decision",
            "fit_seconds",
            "fit_seconds_by_arm",
            "implementation",
            "inputs",
            "matched_controls",
            "official_test_touched",
            "optimized_product",
            "partition",
            "relative_validation_squared_error",
            "restored_norms",
            "runtime",
            "schema",
            "score",
            "seed",
            "source",
            "used_codewords_per_stage",
        )
    }
    receipt.update(
        {
            "claim_eligible": False,
            "codebook_checkpoint": {"sha256": "aa" * 32},
            "dataset": "sop-official-train-class-disjoint-validation",
            "inputs": {
                "direct_checkpoint_sha256": "bb" * 32,
                "source_snapshot_sha256": "cc" * 32,
                "teacher_snapshot_sha256": "dd" * 32,
            },
            "official_test_touched": False,
            "partition": {"split_seed": 17},
            "schema": "sfora-additive-codec-preflight-v2",
            "seed": 0,
        }
    )

    subject.validate_parent_receipt(
        receipt,
        parent_checkpoint_sha256="aa" * 32,
        direct_checkpoint_sha256="bb" * 32,
        source_snapshot_sha256="cc" * 32,
        teacher_snapshot_sha256="dd" * 32,
    )
    receipt["inputs"] = {
        "direct_checkpoint_sha256": "bb" * 32,
        "source_snapshot_sha256": "cc" * 32,
        "teacher_snapshot_sha256": "ee" * 32,
    }
    with pytest.raises(ValueError, match="parent codec receipt authority"):
        subject.validate_parent_receipt(
            receipt,
            parent_checkpoint_sha256="aa" * 32,
            direct_checkpoint_sha256="bb" * 32,
            source_snapshot_sha256="cc" * 32,
            teacher_snapshot_sha256="dd" * 32,
        )
