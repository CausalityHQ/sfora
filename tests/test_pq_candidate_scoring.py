from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from sfora.pq_candidate_scoring import (
    CompiledPqCandidateScorer,
    PqCandidateResult,
    PqCandidateScoringSpec,
    compile_pq_candidate_scorer,
)
from sfora.product_quantization import (
    OptimizedProductQuantizer,
    ProductQuantizationSpec,
    ProductQuantizer,
)


def _codec() -> ProductQuantizer:
    return ProductQuantizer.from_codebooks(
        ProductQuantizationSpec(block_dimensions=(2,), codebook_size=4),
        (
            torch.tensor(
                [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [4.0, 4.0]],
                dtype=torch.float32,
            ),
        ),
    )


@pytest.mark.parametrize(
    "arguments",
    (
        {"metric": "dot"},
        {"candidate_width": True},
        {"candidate_width": 0},
        {"compiled_batch_rows": 0},
        {"row_tile": 0},
        {"row_tile": 3},
    ),
)
def test_candidate_scoring_spec_rejects_concrete_type_and_geometry_drift(
    arguments: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "metric": "squared_l2",
        "candidate_width": 2,
        "compiled_batch_rows": 2,
        "row_tile": 4,
    }
    values.update(arguments)

    with pytest.raises(ValueError, match="PQ candidate scoring spec differs"):
        PqCandidateScoringSpec(**values)  # type: ignore[arg-type]


def test_compiled_candidate_scorer_masks_gallery_tail_and_matches_eager() -> None:
    codec = _codec()
    gallery_codes = torch.tensor([[0], [1], [2]], dtype=torch.uint8)
    queries = torch.tensor([[1.9, 0.0], [0.0, 2.9], [3.9, 4.0]], dtype=torch.float32)
    spec = PqCandidateScoringSpec(
        metric="squared_l2",
        candidate_width=2,
        compiled_batch_rows=2,
        row_tile=4,
    )

    scorer = compile_pq_candidate_scorer(
        codec,
        gallery_codes,
        spec,
        calibration_queries=queries[:2],
        compiler=lambda function: function,
    )
    result = scorer.score(queries)

    assert isinstance(scorer, CompiledPqCandidateScorer)
    assert isinstance(result, PqCandidateResult)
    assert result.ordinals.tolist() == [[1, 0], [2, 0], [2, 1]]
    assert result.padding_rows == 1
    assert result.codes_bytes_scanned == 4 * 4
    assert result.backend == "compiled"
    assert result.fallback_reason is None
    assert result.membership_contract == "calibrated-approximate"
    assert bool((result.ordinals < len(gallery_codes)).all())


def test_compiled_candidate_scorer_owns_every_replayed_output() -> None:
    codec = _codec()
    gallery_codes = torch.tensor([[0], [1], [2]], dtype=torch.uint8)
    queries = torch.tensor([[1.9, 0.0], [0.0, 2.9], [3.9, 4.0]], dtype=torch.float32)
    shared: torch.Tensor | None = None

    def compiler(
        function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        def compiled(query_batch: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
            nonlocal shared
            observed = function(query_batch, codes)
            if shared is None:
                shared = torch.empty_like(observed)
            shared.copy_(observed)
            return shared

        return compiled

    scorer = compile_pq_candidate_scorer(
        codec,
        gallery_codes,
        PqCandidateScoringSpec(
            metric="squared_l2",
            candidate_width=2,
            compiled_batch_rows=2,
            row_tile=4,
        ),
        calibration_queries=queries[:2],
        compiler=compiler,
    )

    assert scorer.score(queries).ordinals.tolist() == [[1, 0], [2, 0], [2, 1]]


def test_candidate_scorer_applies_angular_and_opq_query_preparation() -> None:
    base = ProductQuantizer.from_codebooks(
        ProductQuantizationSpec(block_dimensions=(2,), codebook_size=4),
        (
            torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                dtype=torch.float32,
            ),
        ),
    )
    rotation = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    codec = OptimizedProductQuantizer.from_components(
        base.spec,
        rotation,
        base.detached_codebooks(),
    )
    gallery_codes = torch.tensor([[0], [1], [2]], dtype=torch.uint8)
    queries = torch.tensor([[0.0, 2.0], [3.0, 0.0]], dtype=torch.float32)
    scorer = compile_pq_candidate_scorer(
        codec,
        gallery_codes,
        PqCandidateScoringSpec(
            metric="angular",
            candidate_width=2,
            compiled_batch_rows=2,
            row_tile=4,
        ),
        calibration_queries=queries,
        compiler=lambda function: function,
    )

    assert scorer.score(queries).ordinals.tolist() == [[1, 0], [2, 0]]


def test_candidate_scorer_falls_back_when_calibration_membership_differs() -> None:
    codec = _codec()
    gallery_codes = torch.tensor([[0], [1], [2]], dtype=torch.uint8)
    queries = torch.tensor([[1.9, 0.0], [0.0, 2.9]], dtype=torch.float32)

    def corrupting_compiler(
        function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        del function

        def corrupted(query_batch: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
            del codes
            return torch.zeros((query_batch.shape[0], 2), dtype=torch.int64)

        return corrupted

    scorer = compile_pq_candidate_scorer(
        codec,
        gallery_codes,
        PqCandidateScoringSpec(
            metric="squared_l2",
            candidate_width=2,
            compiled_batch_rows=2,
            row_tile=4,
        ),
        calibration_queries=queries,
        compiler=corrupting_compiler,
    )
    result = scorer.score(queries)

    assert result.ordinals.tolist() == [[1, 0], [2, 0]]
    assert result.backend == "eager"
    assert result.fallback_reason == "calibration-membership-differs"
    assert result.membership_contract == "eager-reference"


def test_candidate_scorer_falls_back_when_compilation_fails() -> None:
    def failing_compiler(
        function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        del function
        raise RuntimeError("compiler unavailable")

    scorer = compile_pq_candidate_scorer(
        _codec(),
        torch.tensor([[0], [1]], dtype=torch.uint8),
        PqCandidateScoringSpec(
            metric="squared_l2",
            candidate_width=1,
            compiled_batch_rows=2,
            row_tile=4,
        ),
        calibration_queries=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        compiler=failing_compiler,
    )

    result = scorer.score(torch.tensor([[1.9, 0.0]], dtype=torch.float32))
    assert result.ordinals.tolist() == [[1]]
    assert result.backend == "eager"
    assert result.fallback_reason == "compile-failed:RuntimeError"


def test_candidate_scorer_falls_back_on_invalid_runtime_compiled_output() -> None:
    calls = 0

    def compiler(
        function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        def compiled(query_batch: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            if calls == 1:
                return function(query_batch, codes)
            return torch.zeros((query_batch.shape[0], 2), dtype=torch.int64)

        return compiled

    queries = torch.tensor([[1.9, 0.0], [0.0, 2.9]], dtype=torch.float32)
    scorer = compile_pq_candidate_scorer(
        _codec(),
        torch.tensor([[0], [1], [2]], dtype=torch.uint8),
        PqCandidateScoringSpec(
            metric="squared_l2",
            candidate_width=2,
            compiled_batch_rows=2,
            row_tile=4,
        ),
        calibration_queries=queries,
        compiler=compiler,
    )

    result = scorer.score(queries)
    assert result.ordinals.tolist() == [[1, 0], [2, 0]]
    assert result.backend == "eager"
    assert result.fallback_reason == "compiled-output-differs"
    assert result.codes_bytes_scanned == 14


def test_candidate_scorer_rejects_finite_inputs_that_overflow_adc_scores() -> None:
    blocks = 20
    codeword = torch.sqrt(
        torch.tensor(torch.finfo(torch.float32).max / blocks, dtype=torch.float64)
    ).float()
    codeword = torch.nextafter(codeword, torch.tensor(0.0, dtype=torch.float32))
    codec = ProductQuantizer.from_codebooks(
        ProductQuantizationSpec(block_dimensions=(1,) * blocks, codebook_size=2),
        tuple(torch.tensor([[codeword], [codeword]], dtype=torch.float32) for _ in range(blocks)),
    )

    with pytest.raises(ValueError, match="PQ candidate scores differ"):
        compile_pq_candidate_scorer(
            codec,
            torch.ones((1, blocks), dtype=torch.uint8),
            PqCandidateScoringSpec(
                metric="squared_l2",
                candidate_width=1,
                compiled_batch_rows=1,
                row_tile=1,
            ),
            calibration_queries=torch.zeros((1, blocks), dtype=torch.float32),
            compiler=lambda function: function,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.to(torch.int16),
        lambda value: value.T,
        lambda value: torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8)[:, :1],
        lambda value: value[:0],
        lambda value: torch.tensor([[0], [4]], dtype=torch.uint8),
    ),
)
def test_candidate_scorer_rejects_gallery_code_authority_drift(mutation: object) -> None:
    with pytest.raises(ValueError, match="PQ candidate gallery codes differ"):
        compile_pq_candidate_scorer(
            _codec(),
            mutation(torch.tensor([[0], [1]], dtype=torch.uint8)),  # type: ignore[operator]
            PqCandidateScoringSpec(
                metric="squared_l2",
                candidate_width=1,
                compiled_batch_rows=1,
                row_tile=4,
            ),
            calibration_queries=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            compiler=lambda function: function,
        )
