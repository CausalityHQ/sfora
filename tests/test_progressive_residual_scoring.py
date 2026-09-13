from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from sfora.product_quantization import (
    OptimizedProductQuantizer,
    ProductQuantizationSpec,
    ProductQuantizer,
)
from sfora.progressive_residual_quantization import (
    ProgressiveResidualQuantizer,
    ProgressiveResidualSpec,
)
from sfora.progressive_residual_scoring import (
    CompiledProgressiveCandidateScorer,
    ProgressiveScoringSpec,
    compile_progressive_candidate_scorer,
    progressive_candidate_scores,
)


def _codec() -> ProgressiveResidualQuantizer:
    base_spec = ProductQuantizationSpec(block_dimensions=(2,), codebook_size=4)
    base = ProductQuantizer.from_codebooks(
        base_spec,
        (
            torch.tensor(
                [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [4.0, 4.0]],
                dtype=torch.float32,
            ),
        ),
    )
    return ProgressiveResidualQuantizer(
        ProgressiveResidualSpec(
            metric="squared_l2",
            base_spec=base_spec,
            maximum_residual_bits=3,
        ),
        base,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        {"residual_bits": True},
        {"residual_bits": 0},
        {"candidate_width": 0},
        {"return_width": 0},
        {"return_width": 4},
        {"compiled_batch_rows": 0},
        {"boundary_repair_width": 1},
        {"boundary_repair_width": 4},
    ),
)
def test_progressive_scoring_spec_rejects_concrete_type_and_geometry_drift(
    mutation: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "residual_bits": 3,
        "candidate_width": 3,
        "return_width": 2,
        "compiled_batch_rows": 2,
        "boundary_repair_width": 2,
    }
    values.update(mutation)

    with pytest.raises(ValueError, match="progressive scoring spec differs"):
        ProgressiveScoringSpec(**values)  # type: ignore[arg-type]


def test_progressive_candidate_scores_match_hand_derived_l2_order_and_bytes() -> None:
    codec = _codec()
    gallery = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [4.0, 4.0]], dtype=torch.float32
    )
    codes = codec.encode(gallery)
    candidates = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    queries = torch.tensor([[1.9, 0.0], [3.9, 4.0]], dtype=torch.float32)
    spec = ProgressiveScoringSpec(
        residual_bits=3,
        candidate_width=3,
        return_width=2,
        compiled_batch_rows=2,
        boundary_repair_width=2,
    )

    result = progressive_candidate_scores(codec, codes, queries, candidates, spec)
    reference = codec.score_candidates(
        queries,
        codes,
        candidates,
        residual_bits=3,
        return_width=2,
    )

    assert result.ordinals.tolist() == [[1, 0], [3, 2]]
    assert torch.allclose(
        result.scores,
        torch.tensor(
            [[0.010197225, 3.6062908], [0.010197201, 16.200432]], dtype=torch.float32
        ),
        atol=1e-7,
        rtol=0.0,
    )
    assert result.base_bytes_read == 2 * 3
    assert result.residual_bytes_read == 2 * 3 * (2 + 3)
    assert torch.equal(result.ordinals, reference.ordinals)
    assert torch.equal(result.scores, reference.scores)


def test_progressive_candidate_scores_match_angular_opq_reference() -> None:
    base_spec = ProductQuantizationSpec(block_dimensions=(2,), codebook_size=4)
    optimized = OptimizedProductQuantizer.from_components(
        base_spec,
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        (
            torch.tensor(
                [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]],
                dtype=torch.float32,
            ),
        ),
    )
    codec = ProgressiveResidualQuantizer(
        ProgressiveResidualSpec(
            metric="angular",
            base_spec=base_spec,
            maximum_residual_bits=3,
        ),
        optimized,
    )
    gallery = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=torch.float32
    )
    codes = codec.encode(gallery)
    candidates = torch.tensor([[3, 2, 1, 0]], dtype=torch.int64)
    query = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    spec = ProgressiveScoringSpec(
        residual_bits=3,
        candidate_width=4,
        return_width=1,
        compiled_batch_rows=1,
        boundary_repair_width=1,
    )

    result = progressive_candidate_scores(codec, codes, query, candidates, spec)
    reference = codec.score_candidates(
        query,
        codes,
        candidates,
        residual_bits=3,
        return_width=1,
    )

    assert result.ordinals.tolist() == [[0]]
    assert float(result.scores[0, 0]) > 0.999
    assert torch.equal(result.ordinals, reference.ordinals)
    assert torch.equal(result.scores, reference.scores)


def test_compiled_progressive_scorer_pads_query_tail_and_owns_replayed_scores() -> None:
    codec = _codec()
    gallery = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [4.0, 4.0]], dtype=torch.float32
    )
    codes = codec.encode(gallery)
    queries = torch.tensor([[1.9, 0.0], [3.9, 4.0], [0.0, 2.9]], dtype=torch.float32)
    candidates = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 1, 2]], dtype=torch.int64)
    shared: torch.Tensor | None = None

    def compiler(
        function: Callable[..., torch.Tensor],
    ) -> Callable[..., torch.Tensor]:
        def compiled(*arguments: torch.Tensor) -> torch.Tensor:
            nonlocal shared
            assert arguments[0].shape[0] == 2
            observed = function(*arguments)
            if shared is None:
                shared = torch.empty_like(observed)
            shared.copy_(observed)
            return shared

        return compiled

    spec = ProgressiveScoringSpec(
        residual_bits=3,
        candidate_width=3,
        return_width=2,
        compiled_batch_rows=2,
        boundary_repair_width=2,
    )
    scorer = compile_progressive_candidate_scorer(
        codec,
        codes,
        spec,
        calibration_queries=queries[:2],
        calibration_candidates=candidates[:2],
        compiler=compiler,
    )

    result = scorer.score(queries, candidates)
    reference = progressive_candidate_scores(codec, codes, queries, candidates, spec)

    assert isinstance(scorer, CompiledProgressiveCandidateScorer)
    assert result.backend == "compiled"
    assert result.fallback_reason is None
    assert result.membership_contract == "validated-approximate"
    assert torch.equal(result.ordinals, reference.ordinals)
    assert torch.equal(result.scores, reference.scores)


def test_progressive_scorer_falls_back_when_compilation_fails() -> None:
    codec = _codec()
    gallery = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [4.0, 4.0]], dtype=torch.float32
    )
    codes = codec.encode(gallery)
    queries = torch.tensor([[1.9, 0.0]], dtype=torch.float32)
    candidates = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    spec = ProgressiveScoringSpec(
        residual_bits=3,
        candidate_width=3,
        return_width=2,
        compiled_batch_rows=1,
        boundary_repair_width=2,
    )

    def compiler(function: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        del function
        raise RuntimeError("compiler unavailable")

    scorer = compile_progressive_candidate_scorer(
        codec,
        codes,
        spec,
        calibration_queries=queries,
        calibration_candidates=candidates,
        compiler=compiler,
    )
    result = scorer.score(queries, candidates)

    assert result.ordinals.tolist() == [[1, 0]]
    assert result.backend == "eager"
    assert result.fallback_reason == "compile-failed:RuntimeError"
    assert result.membership_contract == "eager-reference"


def test_full_boundary_repair_rescores_unseen_compiled_score_drift() -> None:
    codec = _codec()
    gallery = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [4.0, 4.0]], dtype=torch.float32
    )
    codes = codec.encode(gallery)
    query = torch.tensor([[1.9, 0.0]], dtype=torch.float32)
    candidates = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    calls = 0

    def compiler(
        function: Callable[..., torch.Tensor],
    ) -> Callable[..., torch.Tensor]:
        def compiled(*arguments: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            scores = function(*arguments)
            return scores if calls == 1 else -scores

        return compiled

    spec = ProgressiveScoringSpec(
        residual_bits=3,
        candidate_width=3,
        return_width=2,
        compiled_batch_rows=1,
        boundary_repair_width=3,
    )
    scorer = compile_progressive_candidate_scorer(
        codec,
        codes,
        spec,
        calibration_queries=query,
        calibration_candidates=candidates,
        compiler=compiler,
    )

    result = scorer.score(query, candidates)

    assert result.ordinals.tolist() == [[1, 0]]
    assert result.backend == "compiled"
    assert result.boundary_reread_bytes == 3 * (1 + 2 + 3)
    assert result.membership_contract == "full-reference"


def test_progressive_scorer_falls_back_on_runtime_compiler_failure() -> None:
    codec = _codec()
    gallery = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [4.0, 4.0]], dtype=torch.float32
    )
    codes = codec.encode(gallery)
    query = torch.tensor([[1.9, 0.0]], dtype=torch.float32)
    candidates = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    calls = 0

    def compiler(
        function: Callable[..., torch.Tensor],
    ) -> Callable[..., torch.Tensor]:
        def compiled(*arguments: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            if calls == 1:
                return function(*arguments)
            raise RuntimeError("runtime compiler failure")

        return compiled

    spec = ProgressiveScoringSpec(
        residual_bits=3,
        candidate_width=3,
        return_width=2,
        compiled_batch_rows=1,
        boundary_repair_width=2,
    )
    scorer = compile_progressive_candidate_scorer(
        codec,
        codes,
        spec,
        calibration_queries=query,
        calibration_candidates=candidates,
        compiler=compiler,
    )

    result = scorer.score(query, candidates)

    assert result.ordinals.tolist() == [[1, 0]]
    assert result.backend == "eager"
    assert result.fallback_reason == "compiled-execution-failed:RuntimeError"
