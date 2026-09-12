from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_sop_pq_lookup_distillation.py"
sys.path.insert(0, str(_SCRIPT.parent))


def _subject() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_sop_pq_lookup_distillation", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_label_free_pool_is_unique_self_free_and_seeded() -> None:
    subject = _subject()
    float_rankings = torch.tensor([[1], [0], [1], [2], [3]], dtype=torch.int64)
    compressed_rankings = torch.tensor(
        [[2, 4, 1], [3, 4, 0], [4, 0, 1], [1, 4, 2], [0, 1, 2]], dtype=torch.int64
    )

    observed = subject.build_label_free_candidate_pool(
        float_rankings,
        compressed_rankings,
        compressed_width=1,
        uniform_width=1,
        uniform_seed=2901,
    )

    assert [row[:2] for row in observed.tolist()] == [
        [1, 2],
        [0, 3],
        [1, 4],
        [2, 1],
        [3, 0],
    ]
    assert all(row not in candidates for row, candidates in enumerate(observed.tolist()))
    replay = subject.build_label_free_candidate_pool(
        float_rankings,
        compressed_rankings,
        compressed_width=1,
        uniform_width=1,
        uniform_seed=2901,
    )
    torch.testing.assert_close(observed, replay, rtol=0.0, atol=0.0)


def test_uniform_tail_is_independently_sampled_across_queries() -> None:
    subject = _subject()
    rows = 300
    float_rankings = torch.tensor(
        [[(row + 1) % rows, (row + 2) % rows] for row in range(rows)], dtype=torch.int64
    )
    compressed_rankings = torch.tensor(
        [
            [(row + 2) % rows, (row + 3) % rows, (row + 4) % rows, (row + 5) % rows]
            for row in range(rows)
        ],
        dtype=torch.int64,
    )

    observed = subject.build_label_free_candidate_pool(
        float_rankings,
        compressed_rankings,
        compressed_width=2,
        uniform_width=4,
        uniform_seed=2901,
    )

    tails = {tuple(row) for row in observed[:, 4:].tolist()}
    assert len(tails) > 250


def test_float_shortlist_excludes_self_and_repairs_retention_boundary_ties() -> None:
    subject = _subject()
    values = torch.tensor([[1.0, 0.0]] * 5, dtype=torch.float32)

    observed = subject._exact_float_shortlists(values, width=2, query_block_size=3)

    assert observed.tolist() == [[1, 2], [0, 2], [0, 1], [0, 1], [0, 1]]


def test_fixed_candidate_ranking_breaks_corrected_score_ties_by_row_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = _subject()

    def zero_distances(
        queries: torch.Tensor, codes: torch.Tensor, _quantizer: object
    ) -> torch.Tensor:
        return torch.zeros((len(queries), codes.shape[1]), dtype=torch.float32)

    class BaselineModel:
        def __call__(
            self, _queries: torch.Tensor, _codes: torch.Tensor, baseline: torch.Tensor
        ) -> torch.Tensor:
            return baseline

    monkeypatch.setattr(subject, "aligned_product_squared_distances", zero_distances)
    observed = subject._fixed_candidate_rankings(
        BaselineModel(),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((5, 1), dtype=torch.uint8),
        object(),
        torch.tensor([[4, 2, 3, 1]], dtype=torch.int64),
    )

    assert observed.tolist() == [[1, 2, 3, 4]]


def test_pairwise_agreement_does_not_count_one_sided_ties_as_agreement() -> None:
    subject = _subject()

    class BaselineModel:
        def __call__(
            self, _queries: torch.Tensor, _codes: torch.Tensor, baseline: torch.Tensor
        ) -> torch.Tensor:
            return baseline

    observed = subject._pairwise_agreement(
        BaselineModel(),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((1, 2, 1), dtype=torch.uint8),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        torch.tensor([[[0, 1]]], dtype=torch.int64),
    )

    assert observed == 0.0


def test_registered_pair_positions_have_contested_and_contrast_strata() -> None:
    subject = _subject()

    pairs = subject.registered_pair_positions(
        rows=2,
        candidates=384,
        teacher_width=128,
        compressed_width=128,
        uniform_width=128,
        pairs_per_stratum=128,
    )

    assert pairs.shape == (2, 256, 2)
    assert pairs.dtype == torch.int64
    assert bool((pairs[:, :64] < 32).all())
    assert bool((pairs[:, 64:128] >= 128).all())
    assert bool((pairs[:, 64:128] < 160).all())
    assert bool((pairs[:, 128:, 0] < 160).all())
    assert bool((pairs[:, 128:, 1] >= 256).all())
    assert not bool((pairs[:, :, 0] == pairs[:, :, 1]).any())
    assert len({tuple(pair) for pair in pairs[0, :128].tolist()}) == 128
    assert len({tuple(pair) for pair in pairs[0, 128:].tolist()}) == 128
    assert len(set(pairs[0, :64].flatten().tolist())) == 32
    assert len(set((pairs[0, 64:128] - 128).flatten().tolist())) == 32


def test_registered_pair_positions_reject_non_integer_pair_count() -> None:
    subject = _subject()

    with pytest.raises(ValueError, match="lookup pair authority differs"):
        subject.registered_pair_positions(
            rows=2,
            candidates=384,
            teacher_width=128,
            compressed_width=128,
            uniform_width=128,
            pairs_per_stratum="128",
        )


@pytest.mark.parametrize(
    (
        "exhaustive_map_at_r",
        "fixed_candidate_map_at_r",
        "r1",
        "objective_improved",
        "classification",
    ),
    (
        (0.5900, 0.5900, 0.83, False, "fixed-code-lookup-optimization-failed"),
        (0.5788, 0.5900, 0.83, True, "fixed-code-lookup-failed-pq32"),
        (0.5820, 0.5790, 0.83, True, "fixed-code-lookup-within-pq-variation"),
        (0.5820, 0.5810, 0.83, True, "fixed-code-lookup-improved"),
        (0.5860, 0.5810, 0.820, True, "fixed-code-lookup-r1-regressed"),
        (0.5860, 0.5810, 0.822, True, "fixed-code-lookup-target-screen"),
    ),
)
def test_lookup_decision_uses_frozen_quality_and_variation_gates(
    exhaustive_map_at_r: float,
    fixed_candidate_map_at_r: float,
    r1: float,
    objective_improved: bool,
    classification: str,
) -> None:
    subject = _subject()

    observed = subject.lookup_distillation_decision(
        exhaustive_map_at_r=exhaustive_map_at_r,
        fixed_candidate_map_at_r=fixed_candidate_map_at_r,
        r1=r1,
        pq24_r1=0.8222334768,
        pq32_map_at_r=0.5789231844,
        prior_reranker_map_at_r=0.5786155157,
        pq_codebook_variation=0.0008777193,
        objective_improved=objective_improved,
    )

    assert observed["classification"] == classification
    assert observed["passed"] is False
    assert observed["observed"] == {
        "exhaustive_map_at_r": exhaustive_map_at_r,
        "fixed_pq24_top32_map_at_r": fixed_candidate_map_at_r,
        "r1": r1,
    }


def test_lookup_distillation_requires_explicit_execution() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "--execute-pq-lookup-distillation" in result.stderr


def test_prior_reranker_receipt_binds_comparator_and_inputs() -> None:
    subject = _subject()
    receipt: dict[str, object] = {
        key: None
        for key in (
            "claim_eligible",
            "dataset",
            "decision",
            "fit_seconds",
            "fit_teacher_loss",
            "inputs",
            "matched_controls",
            "model_checkpoint",
            "model_parameter_bytes",
            "official_test_touched",
            "partition",
            "pq24_seed_stability",
            "recipe",
            "rankings",
            "reranked",
            "reranked_paired_delta",
            "reranker_seed_stability",
            "runtime",
            "schema",
            "seed",
            "source",
        )
    }
    receipt.update(
        {
            "claim_eligible": False,
            "dataset": "sop-official-train-class-disjoint-validation",
            "decision": {"classification": "candidate-set-reranker-failed-pq32"},
            "inputs": {
                "direct_checkpoint_sha256": "aa" * 32,
                "parent_codec_checkpoint_sha256": "bb" * 32,
                "parent_codec_receipt_sha256": "cc" * 32,
                "source_snapshot_sha256": "dd" * 32,
                "teacher_snapshot_sha256": "ee" * 32,
            },
            "official_test_touched": False,
            "partition": {"fit_rows": 47704, "split_seed": 17, "validation_rows": 11847},
            "reranked": {"map_at_r": subject.PRIOR_RERANKER_MAP, "r1": 0.8226555246},
            "schema": "sfora-pq-candidate-set-reranking-v1",
            "seed": 0,
        }
    )

    subject.validate_prior_reranker_receipt(
        receipt,
        direct_checkpoint_sha256="aa" * 32,
        parent_codec_checkpoint_sha256="bb" * 32,
        parent_codec_receipt_sha256="cc" * 32,
        source_snapshot_sha256="dd" * 32,
        teacher_snapshot_sha256="ee" * 32,
    )
    receipt["reranked"] = {"map_at_r": subject.PRIOR_RERANKER_MAP + 0.001, "r1": 0.8226555246}
    with pytest.raises(ValueError, match="prior reranker receipt authority"):
        subject.validate_prior_reranker_receipt(
            receipt,
            direct_checkpoint_sha256="aa" * 32,
            parent_codec_checkpoint_sha256="bb" * 32,
            parent_codec_receipt_sha256="cc" * 32,
            source_snapshot_sha256="dd" * 32,
            teacher_snapshot_sha256="ee" * 32,
        )
