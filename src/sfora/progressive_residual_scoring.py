"""Validated tensor scoring for progressive residual candidates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch.nn import functional as F

from sfora.product_quantization import OptimizedProductQuantizer, ProductQuantizer
from sfora.progressive_residual_quantization import (
    ProgressiveCandidateResult,
    ProgressiveResidualCodes,
    ProgressiveResidualQuantizer,
)


@dataclass(frozen=True, slots=True)
class ProgressiveScoringSpec:
    """Fixed candidate and execution geometry for progressive scoring."""

    residual_bits: int
    candidate_width: int
    return_width: int
    compiled_batch_rows: int
    boundary_repair_width: int

    def __post_init__(self) -> None:
        if (
            type(self.residual_bits) is not int
            or not 1 <= self.residual_bits <= 8
            or type(self.candidate_width) is not int
            or self.candidate_width < 1
            or type(self.return_width) is not int
            or self.return_width < 1
            or type(self.compiled_batch_rows) is not int
            or self.compiled_batch_rows < 1
            or type(self.boundary_repair_width) is not int
            or not self.return_width
            <= self.boundary_repair_width
            <= self.candidate_width
        ):
            raise ValueError("progressive scoring spec differs")


@dataclass(frozen=True, slots=True)
class ProgressiveScoringResult:
    """Progressive ranking with execution and physical-read evidence."""

    ordinals: torch.Tensor
    scores: torch.Tensor
    base_bytes_read: int
    residual_bytes_read: int
    boundary_reread_bytes: int
    backend: Literal["compiled", "eager"]
    fallback_reason: str | None
    membership_contract: Literal[
        "validated-approximate", "full-reference", "eager-reference"
    ]


def _decode_base(
    codec: ProgressiveResidualQuantizer,
    base_codes: torch.Tensor,
) -> torch.Tensor:
    base = codec.base_quantizer
    if type(base) is ProductQuantizer:
        quantizer = base
    else:
        quantizer = cast(OptimizedProductQuantizer, base).quantizer
    decoded = torch.cat(
        [
            codebook[base_codes[:, index].long()]
            for index, codebook in enumerate(quantizer.codebooks)
        ],
        dim=1,
    )
    if type(base) is OptimizedProductQuantizer:
        decoded = torch.matmul(decoded, base._rotation.T)
    return decoded


def _unpack_prefix(
    residual_planes: torch.Tensor,
    *,
    dimensions: int,
    residual_bits: int,
) -> torch.Tensor:
    plane_bytes = (dimensions + 7) // 8
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=residual_planes.device)
    prefix = torch.zeros(
        (residual_planes.shape[0], dimensions),
        dtype=torch.int16,
        device=residual_planes.device,
    )
    for plane_index in range(residual_bits):
        bits = torch.bitwise_and(
            torch.bitwise_right_shift(residual_planes[:, plane_index, :, None], shifts),
            1,
        ).reshape(residual_planes.shape[0], plane_bytes * 8)[:, :dimensions]
        prefix = torch.bitwise_or(torch.bitwise_left_shift(prefix, 1), bits.to(torch.int16))
    return prefix.float()


def _progressive_raw_score_graph(
    codec: ProgressiveResidualQuantizer,
    prepared_queries: torch.Tensor,
    gathered_base: torch.Tensor,
    gathered_scales: torch.Tensor,
    gathered_planes: torch.Tensor,
    spec: ProgressiveScoringSpec,
) -> torch.Tensor:
    prefix = _unpack_prefix(
        gathered_planes,
        dimensions=codec.spec.dimensions,
        residual_bits=spec.residual_bits,
    )
    step = 1 << (8 - spec.residual_bits)
    centers = prefix * step + (step - 1) / 2.0 - 127.5
    decoded = _decode_base(codec, gathered_base)
    reconstructed = decoded + centers * gathered_scales.float()[:, None]
    if codec.spec.metric == "angular":
        reconstructed = F.normalize(reconstructed, dim=1)
    reconstructed = reconstructed.reshape(
        prepared_queries.shape[0],
        spec.candidate_width,
        codec.spec.dimensions,
    )
    if codec.spec.metric == "angular":
        return torch.einsum("bd,bkd->bk", prepared_queries, reconstructed)
    return (reconstructed - prepared_queries[:, None, :]).square().sum(dim=-1)


def _select_scores(
    candidate_ordinals: torch.Tensor,
    scores: torch.Tensor,
    *,
    return_width: int,
    descending: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    ordinal_order = torch.argsort(candidate_ordinals, dim=1, stable=True)
    ordered_ordinals = candidate_ordinals.gather(1, ordinal_order)
    ordered_scores = scores.gather(1, ordinal_order)
    score_order = torch.argsort(
        ordered_scores,
        dim=1,
        descending=descending,
        stable=True,
    )[:, :return_width]
    return (
        ordered_ordinals.gather(1, score_order).contiguous(),
        ordered_scores.gather(1, score_order).contiguous(),
    )


@torch.no_grad()
def progressive_candidate_scores(
    codec: ProgressiveResidualQuantizer,
    codes: ProgressiveResidualCodes,
    queries: torch.Tensor,
    candidate_ordinals: torch.Tensor,
    spec: ProgressiveScoringSpec,
) -> ProgressiveCandidateResult:
    """Score one exact candidate table through the tensor-only eager graph."""

    if type(codec) is not ProgressiveResidualQuantizer or type(spec) is not ProgressiveScoringSpec:
        raise ValueError("progressive scoring authority differs")
    codec._validate_codes(codes)
    if spec.residual_bits > codec.spec.maximum_residual_bits:
        raise ValueError("progressive scoring authority differs")
    prepared = codec.prepare_values(queries)
    if (
        type(candidate_ordinals) is not torch.Tensor
        or candidate_ordinals.dtype != torch.int64
        or candidate_ordinals.shape != (prepared.shape[0], spec.candidate_width)
        or candidate_ordinals.device != codec.device
        or not candidate_ordinals.is_contiguous()
        or bool((candidate_ordinals < 0).any())
        or bool((candidate_ordinals >= codes.rows).any())
    ):
        raise ValueError("progressive scoring candidates differ")
    sorted_candidates = torch.sort(candidate_ordinals, dim=1).values
    if spec.candidate_width > 1 and bool(
        (sorted_candidates[:, 1:] == sorted_candidates[:, :-1]).any()
    ):
        raise ValueError("progressive scoring candidates differ")
    flat = candidate_ordinals.reshape(-1)
    gathered_base = codes.base_codes.index_select(0, flat).contiguous()
    gathered_scales = codes.scales.index_select(0, flat).contiguous()
    gathered_planes = (
        codes.residual_planes[:, : spec.residual_bits].index_select(0, flat).contiguous()
    )
    raw_scores = _progressive_raw_score_graph(
        codec,
        prepared,
        gathered_base,
        gathered_scales,
        gathered_planes,
        spec,
    )
    ordinals, scores = _select_scores(
        candidate_ordinals,
        raw_scores,
        return_width=spec.return_width,
        descending=codec.spec.metric == "angular",
    )
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("progressive scoring scores differ")
    query_rows = prepared.shape[0]
    return ProgressiveCandidateResult(
        ordinals=ordinals,
        scores=scores,
        base_bytes_read=query_rows * spec.candidate_width * codec.spec.base_spec.bytes_per_vector,
        residual_bytes_read=(
            query_rows
            * spec.candidate_width
            * (2 + spec.residual_bits * codec.spec.residual_plane_bytes)
        ),
    )


_CompiledScoreFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]
_ScoreCompiler = Callable[[_CompiledScoreFunction], _CompiledScoreFunction]


def _default_compiler(function: _CompiledScoreFunction) -> _CompiledScoreFunction:
    return cast(
        _CompiledScoreFunction,
        torch.compile(function, fullgraph=True, dynamic=False, mode="max-autotune"),
    )


class CompiledProgressiveCandidateScorer:
    """Fixed-shape progressive score graph with an eager reference fallback."""

    def __init__(
        self,
        codec: ProgressiveResidualQuantizer,
        codes: ProgressiveResidualCodes,
        spec: ProgressiveScoringSpec,
        compiled: _CompiledScoreFunction | None,
        fallback_reason: str | None,
    ) -> None:
        self._codec = codec
        self._codes = codes
        self.spec = spec
        self._compiled = compiled
        self._fallback_reason = fallback_reason

    def _validate_inputs(
        self,
        queries: torch.Tensor,
        candidate_ordinals: torch.Tensor,
    ) -> torch.Tensor:
        prepared = self._codec.prepare_values(queries)
        if (
            type(candidate_ordinals) is not torch.Tensor
            or candidate_ordinals.dtype != torch.int64
            or candidate_ordinals.shape != (prepared.shape[0], self.spec.candidate_width)
            or candidate_ordinals.device != self._codec.device
            or not candidate_ordinals.is_contiguous()
            or bool((candidate_ordinals < 0).any())
            or bool((candidate_ordinals >= self._codes.rows).any())
        ):
            raise ValueError("progressive scoring candidates differ")
        sorted_candidates = torch.sort(candidate_ordinals, dim=1).values
        if self.spec.candidate_width > 1 and bool(
            (sorted_candidates[:, 1:] == sorted_candidates[:, :-1]).any()
        ):
            raise ValueError("progressive scoring candidates differ")
        return prepared

    def _compiled_scores(
        self,
        prepared: torch.Tensor,
        candidate_ordinals: torch.Tensor,
    ) -> torch.Tensor:
        compiled = self._compiled
        if compiled is None:
            raise RuntimeError("compiled progressive scorer is unavailable")
        rows = []
        for start in range(0, prepared.shape[0], self.spec.compiled_batch_rows):
            query_batch = prepared[start : start + self.spec.compiled_batch_rows]
            candidate_batch = candidate_ordinals[start : start + self.spec.compiled_batch_rows]
            logical_rows = query_batch.shape[0]
            if logical_rows < self.spec.compiled_batch_rows:
                padding_rows = self.spec.compiled_batch_rows - logical_rows
                query_batch = torch.cat(
                    (query_batch, query_batch[-1:].expand(padding_rows, -1)), dim=0
                )
                candidate_batch = torch.cat(
                    (candidate_batch, candidate_batch[-1:].expand(padding_rows, -1)), dim=0
                )
            flat = candidate_batch.reshape(-1)
            gathered_base = self._codes.base_codes.index_select(0, flat).contiguous()
            gathered_scales = self._codes.scales.index_select(0, flat).contiguous()
            gathered_planes = (
                self._codes.residual_planes[:, : self.spec.residual_bits]
                .index_select(0, flat)
                .contiguous()
            )
            owned = compiled(
                query_batch,
                gathered_base,
                gathered_scales,
                gathered_planes,
            ).clone()
            rows.append(owned[:logical_rows])
        return torch.cat(rows, dim=0).contiguous()

    @torch.no_grad()
    def score(
        self,
        queries: torch.Tensor,
        candidate_ordinals: torch.Tensor,
    ) -> ProgressiveScoringResult:
        """Score fixed-width candidates and own every replayed score buffer."""

        if self._compiled is None:
            eager = progressive_candidate_scores(
                self._codec,
                self._codes,
                queries,
                candidate_ordinals,
                self.spec,
            )
            return ProgressiveScoringResult(
                ordinals=eager.ordinals,
                scores=eager.scores,
                base_bytes_read=eager.base_bytes_read,
                residual_bytes_read=eager.residual_bytes_read,
                boundary_reread_bytes=0,
                backend="eager",
                fallback_reason=self._fallback_reason,
                membership_contract="eager-reference",
            )
        prepared = self._validate_inputs(queries, candidate_ordinals)
        physical_query_rows = (
            (prepared.shape[0] + self.spec.compiled_batch_rows - 1)
            // self.spec.compiled_batch_rows
            * self.spec.compiled_batch_rows
        )
        compiled_base_bytes = (
            physical_query_rows
            * self.spec.candidate_width
            * self._codec.spec.base_spec.bytes_per_vector
        )
        compiled_residual_bytes = (
            physical_query_rows
            * self.spec.candidate_width
            * (2 + self.spec.residual_bits * self._codec.spec.residual_plane_bytes)
        )
        try:
            raw_scores = self._compiled_scores(prepared, candidate_ordinals)
        except Exception as error:
            eager = progressive_candidate_scores(
                self._codec,
                self._codes,
                queries,
                candidate_ordinals,
                self.spec,
            )
            return ProgressiveScoringResult(
                ordinals=eager.ordinals,
                scores=eager.scores,
                base_bytes_read=compiled_base_bytes + eager.base_bytes_read,
                residual_bytes_read=compiled_residual_bytes + eager.residual_bytes_read,
                boundary_reread_bytes=0,
                backend="eager",
                fallback_reason=f"compiled-execution-failed:{type(error).__name__}",
                membership_contract="eager-reference",
            )
        if raw_scores.shape != candidate_ordinals.shape or not bool(
            torch.isfinite(raw_scores).all()
        ):
            eager = progressive_candidate_scores(
                self._codec,
                self._codes,
                queries,
                candidate_ordinals,
                self.spec,
            )
            return ProgressiveScoringResult(
                ordinals=eager.ordinals,
                scores=eager.scores,
                base_bytes_read=compiled_base_bytes + eager.base_bytes_read,
                residual_bytes_read=compiled_residual_bytes + eager.residual_bytes_read,
                boundary_reread_bytes=0,
                backend="eager",
                fallback_reason="compiled-output-differs",
                membership_contract="eager-reference",
            )
        repair_ordinals, _approximate_scores = _select_scores(
            candidate_ordinals,
            raw_scores,
            return_width=self.spec.boundary_repair_width,
            descending=self._codec.spec.metric == "angular",
        )
        repair_spec = ProgressiveScoringSpec(
            residual_bits=self.spec.residual_bits,
            candidate_width=self.spec.boundary_repair_width,
            return_width=self.spec.return_width,
            compiled_batch_rows=self.spec.compiled_batch_rows,
            boundary_repair_width=self.spec.boundary_repair_width,
        )
        repaired = progressive_candidate_scores(
            self._codec,
            self._codes,
            queries,
            repair_ordinals,
            repair_spec,
        )
        return ProgressiveScoringResult(
            ordinals=repaired.ordinals,
            scores=repaired.scores,
            base_bytes_read=compiled_base_bytes,
            residual_bytes_read=compiled_residual_bytes,
            boundary_reread_bytes=repaired.base_bytes_read + repaired.residual_bytes_read,
            backend="compiled",
            fallback_reason=None,
            membership_contract=(
                "full-reference"
                if self.spec.boundary_repair_width == self.spec.candidate_width
                else "validated-approximate"
            ),
        )


def compile_progressive_candidate_scorer(
    codec: ProgressiveResidualQuantizer,
    codes: ProgressiveResidualCodes,
    spec: ProgressiveScoringSpec,
    *,
    calibration_queries: torch.Tensor,
    calibration_candidates: torch.Tensor,
    compiler: _ScoreCompiler = _default_compiler,
) -> CompiledProgressiveCandidateScorer:
    """Compile and calibrate one fixed progressive score geometry."""

    if type(codec) is not ProgressiveResidualQuantizer or type(spec) is not ProgressiveScoringSpec:
        raise ValueError("progressive scoring authority differs")
    codec._validate_codes(codes)
    if spec.residual_bits > codec.spec.maximum_residual_bits or not callable(compiler):
        raise ValueError("progressive scoring authority differs")

    def graph(
        prepared_queries: torch.Tensor,
        gathered_base: torch.Tensor,
        gathered_scales: torch.Tensor,
        gathered_planes: torch.Tensor,
    ) -> torch.Tensor:
        return _progressive_raw_score_graph(
            codec,
            prepared_queries,
            gathered_base,
            gathered_scales,
            gathered_planes,
            spec,
        )

    expected = progressive_candidate_scores(
        codec,
        codes,
        calibration_queries,
        calibration_candidates,
        spec,
    )
    try:
        scorer = CompiledProgressiveCandidateScorer(
            codec,
            codes,
            spec,
            compiler(graph),
            None,
        )
        observed = scorer.score(calibration_queries, calibration_candidates)
    except Exception as error:
        return CompiledProgressiveCandidateScorer(
            codec,
            codes,
            spec,
            None,
            f"compile-failed:{type(error).__name__}",
        )
    if not torch.equal(observed.ordinals, expected.ordinals) or not torch.allclose(
        observed.scores,
        expected.scores,
        atol=1e-5,
        rtol=1e-5,
    ):
        return CompiledProgressiveCandidateScorer(
            codec,
            codes,
            spec,
            None,
            "calibration-differs",
        )
    return scorer
