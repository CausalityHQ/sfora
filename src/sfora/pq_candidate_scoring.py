"""Tail-safe compiled asymmetric PQ candidate scoring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import torch

from sfora.product_quantization import OptimizedProductQuantizer, ProductQuantizer

PqCandidateMetric = Literal["angular", "squared_l2"]
_CompiledCandidateFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
_CandidateCompiler = Callable[[_CompiledCandidateFunction], _CompiledCandidateFunction]


@dataclass(frozen=True, slots=True)
class PqCandidateScoringSpec:
    """Fixed metric and execution geometry for compiled ADC selection."""

    metric: PqCandidateMetric
    candidate_width: int
    compiled_batch_rows: int
    row_tile: int = 64

    def __post_init__(self) -> None:
        if (
            type(self.metric) is not str
            or self.metric not in ("angular", "squared_l2")
            or type(self.candidate_width) is not int
            or self.candidate_width < 1
            or type(self.compiled_batch_rows) is not int
            or self.compiled_batch_rows < 1
            or type(self.row_tile) is not int
            or self.row_tile < 1
            or self.row_tile & (self.row_tile - 1) != 0
        ):
            raise ValueError("PQ candidate scoring spec differs")


@dataclass(frozen=True, slots=True)
class PqCandidateResult:
    """Candidate ordinals and physical code-read accounting."""

    ordinals: torch.Tensor
    padding_rows: int
    codes_bytes_scanned: int
    backend: Literal["compiled", "eager"]
    fallback_reason: str | None

    def __post_init__(self) -> None:
        if (
            type(self.ordinals) is not torch.Tensor
            or self.ordinals.dtype != torch.int64
            or self.ordinals.ndim != 2
            or self.ordinals.shape[0] < 1
            or self.ordinals.shape[1] < 1
            or not self.ordinals.is_contiguous()
            or type(self.padding_rows) is not int
            or self.padding_rows < 0
            or type(self.codes_bytes_scanned) is not int
            or self.codes_bytes_scanned < 1
            or type(self.backend) is not str
            or self.backend not in ("compiled", "eager")
            or (self.backend == "compiled") != (self.fallback_reason is None)
        ):
            raise ValueError("PQ candidate result differs")


def _product_quantizer(
    quantizer: ProductQuantizer | OptimizedProductQuantizer,
) -> ProductQuantizer:
    if type(quantizer) is ProductQuantizer:
        return quantizer
    if type(quantizer) is OptimizedProductQuantizer:
        return quantizer.quantizer
    raise ValueError("PQ candidate quantizer differs")


def _prepare_queries(
    queries: torch.Tensor,
    quantizer: ProductQuantizer | OptimizedProductQuantizer,
    metric: PqCandidateMetric,
) -> torch.Tensor:
    base = _product_quantizer(quantizer)
    if (
        type(queries) is not torch.Tensor
        or queries.dtype != torch.float32
        or queries.ndim != 2
        or queries.shape[0] < 1
        or queries.shape[1] != base.spec.dimensions
        or queries.device != base.codebooks[0].device
        or not queries.is_contiguous()
        or not bool(torch.isfinite(queries).all())
    ):
        raise ValueError("PQ candidate queries differ")
    prepared = queries
    if metric == "angular":
        norms = torch.linalg.vector_norm(queries, dim=1, keepdim=True, dtype=torch.float64)
        if not bool(torch.isfinite(norms).all()) or bool((norms <= 1e-12).any()):
            raise ValueError("PQ candidate queries differ")
        prepared = (queries.double() / norms).float()
    if type(quantizer) is OptimizedProductQuantizer:
        prepared = torch.matmul(prepared, quantizer._rotation)
    return prepared.contiguous()


def _candidate_core(
    quantizer: ProductQuantizer,
    *,
    gallery_rows: int,
    candidate_width: int,
) -> _CompiledCandidateFunction:
    block_dimensions = quantizer.spec.block_dimensions
    codebooks = tuple(quantizer.codebooks)

    def score(query_batch: torch.Tensor, gallery_codes: torch.Tensor) -> torch.Tensor:
        distances = torch.zeros(
            (query_batch.shape[0], gallery_codes.shape[0]),
            dtype=torch.float32,
            device=query_batch.device,
        )
        start = 0
        for block_index, width in enumerate(block_dimensions):
            query_block = query_batch[:, start : start + width]
            table = (
                (query_block[:, None, :] - codebooks[block_index][None, :, :])
                .square()
                .sum(dim=-1)
            )
            distances = distances + table[:, gallery_codes[:, block_index].long()]
            start += width
        scores_are_finite = torch.isfinite(distances[:, :gallery_rows]).all()
        if gallery_rows < gallery_codes.shape[0]:
            distances[:, gallery_rows:] = torch.inf
        ordinals = torch.topk(
            distances,
            k=candidate_width,
            dim=1,
            largest=False,
            sorted=True,
        ).indices
        return torch.where(scores_are_finite, ordinals, torch.full_like(ordinals, -1))

    return score


def _default_compiler(function: _CompiledCandidateFunction) -> _CompiledCandidateFunction:
    return cast(
        _CompiledCandidateFunction,
        torch.compile(function, fullgraph=True, dynamic=False, mode="max-autotune"),
    )


class CompiledPqCandidateScorer:
    """Own fixed-shape compiled outputs and fail safely to eager ADC."""

    def __init__(
        self,
        quantizer: ProductQuantizer | OptimizedProductQuantizer,
        gallery_codes: torch.Tensor,
        padded_codes: torch.Tensor,
        spec: PqCandidateScoringSpec,
        compiled: _CompiledCandidateFunction | None,
        fallback_reason: str | None,
    ) -> None:
        self._quantizer = quantizer
        self._base = _product_quantizer(quantizer)
        self._gallery_codes = gallery_codes
        self._padded_codes = padded_codes
        self.spec = spec
        self._compiled = compiled
        self._fallback_reason = fallback_reason

    def _compiled_ordinals(self, prepared: torch.Tensor) -> torch.Tensor:
        if self._compiled is None:
            raise RuntimeError("compiled PQ candidate scorer is unavailable")
        batches = []
        for start in range(0, prepared.shape[0], self.spec.compiled_batch_rows):
            batch = prepared[start : start + self.spec.compiled_batch_rows]
            logical_rows = batch.shape[0]
            if logical_rows < self.spec.compiled_batch_rows:
                padding = torch.zeros(
                    (self.spec.compiled_batch_rows - logical_rows, prepared.shape[1]),
                    dtype=prepared.dtype,
                    device=prepared.device,
                )
                batch = torch.cat((batch, padding), dim=0)
            # CUDA graph replays may alias a previously returned output buffer.
            owned = self._compiled(batch, self._padded_codes).clone()
            batches.append(owned[:logical_rows])
        return torch.cat(batches, dim=0).contiguous()

    def _eager_ordinals(self, prepared: torch.Tensor) -> torch.Tensor:
        batches = []
        for start in range(0, prepared.shape[0], self.spec.compiled_batch_rows):
            distances = self._base.asymmetric_squared_distances(
                prepared[start : start + self.spec.compiled_batch_rows],
                self._gallery_codes,
            )
            if not bool(torch.isfinite(distances).all()):
                raise ValueError("PQ candidate scores differ")
            batches.append(
                torch.topk(
                    distances,
                    k=self.spec.candidate_width,
                    dim=1,
                    largest=False,
                    sorted=True,
                ).indices
            )
        return torch.cat(batches, dim=0).contiguous()

    def _valid_ordinals(self, ordinals: torch.Tensor, *, query_rows: int) -> bool:
        if (
            type(ordinals) is not torch.Tensor
            or ordinals.dtype != torch.int64
            or ordinals.shape != (query_rows, self.spec.candidate_width)
            or not bool((ordinals >= 0).all())
            or not bool((ordinals < self._gallery_codes.shape[0]).all())
        ):
            return False
        ordered = torch.sort(ordinals, dim=1).values
        return self.spec.candidate_width == 1 or not bool(
            (ordered[:, 1:] == ordered[:, :-1]).any()
        )

    def score(self, queries: torch.Tensor) -> PqCandidateResult:
        """Return bounded candidates for every query without exposing padded rows."""

        prepared = _prepare_queries(queries, self._quantizer, self.spec.metric)
        fallback_reason = self._fallback_reason
        code_width = self._gallery_codes.shape[1]
        eager_code_rows = prepared.shape[0] * self._gallery_codes.shape[0]
        compiled_query_rows = (
            (prepared.shape[0] + self.spec.compiled_batch_rows - 1)
            // self.spec.compiled_batch_rows
            * self.spec.compiled_batch_rows
        )
        compiled_code_rows = compiled_query_rows * self._padded_codes.shape[0]
        if self._compiled is None:
            ordinals = self._eager_ordinals(prepared)
            scanned_code_rows = eager_code_rows
            backend: Literal["compiled", "eager"] = "eager"
        else:
            try:
                ordinals = self._compiled_ordinals(prepared)
            except Exception as error:
                ordinals = self._eager_ordinals(prepared)
                scanned_code_rows = compiled_code_rows + eager_code_rows
                backend = "eager"
                fallback_reason = f"compiled-execution-failed:{type(error).__name__}"
            else:
                if self._valid_ordinals(ordinals, query_rows=prepared.shape[0]):
                    scanned_code_rows = compiled_code_rows
                    backend = "compiled"
                else:
                    ordinals = self._eager_ordinals(prepared)
                    scanned_code_rows = compiled_code_rows + eager_code_rows
                    backend = "eager"
                    fallback_reason = "compiled-output-differs"
        return PqCandidateResult(
            ordinals=ordinals,
            padding_rows=self._padded_codes.shape[0] - self._gallery_codes.shape[0],
            codes_bytes_scanned=scanned_code_rows * code_width,
            backend=backend,
            fallback_reason=fallback_reason,
        )


def compile_pq_candidate_scorer(
    quantizer: ProductQuantizer | OptimizedProductQuantizer,
    gallery_codes: torch.Tensor,
    spec: PqCandidateScoringSpec,
    *,
    calibration_queries: torch.Tensor,
    compiler: _CandidateCompiler = _default_compiler,
) -> CompiledPqCandidateScorer:
    """Compile fixed-shape ADC/top-k and retain eager fallback on calibration drift."""

    if type(spec) is not PqCandidateScoringSpec or not callable(compiler):
        raise ValueError("PQ candidate scorer authority differs")
    base = _product_quantizer(quantizer)
    try:
        base._validate_codes(gallery_codes)
    except ValueError as error:
        raise ValueError("PQ candidate gallery codes differ") from error
    if not gallery_codes.is_contiguous() or spec.candidate_width > gallery_codes.shape[0]:
        raise ValueError("PQ candidate gallery codes differ")
    gallery_codes = gallery_codes.detach().clone().contiguous()
    padding_rows = (-gallery_codes.shape[0]) % spec.row_tile
    if padding_rows:
        padding = torch.zeros(
            (padding_rows, gallery_codes.shape[1]),
            dtype=gallery_codes.dtype,
            device=gallery_codes.device,
        )
        padded_codes = torch.cat((gallery_codes, padding), dim=0).contiguous()
    else:
        padded_codes = gallery_codes.clone()
    prepared = _prepare_queries(calibration_queries, quantizer, spec.metric)
    core = _candidate_core(
        base,
        gallery_rows=gallery_codes.shape[0],
        candidate_width=spec.candidate_width,
    )
    try:
        compiled = compiler(core)
        scorer = CompiledPqCandidateScorer(
            quantizer,
            gallery_codes,
            padded_codes,
            spec,
            compiled,
            None,
        )
        observed = scorer._compiled_ordinals(prepared)
    except Exception as error:
        return CompiledPqCandidateScorer(
            quantizer,
            gallery_codes,
            padded_codes,
            spec,
            None,
            f"compile-failed:{type(error).__name__}",
        )
    expected = scorer._eager_ordinals(prepared)
    if not scorer._valid_ordinals(observed, query_rows=prepared.shape[0]) or not torch.equal(
        torch.sort(observed, dim=1).values, torch.sort(expected, dim=1).values
    ):
        return CompiledPqCandidateScorer(
            quantizer,
            gallery_codes,
            padded_codes,
            spec,
            None,
            "calibration-membership-differs",
        )
    return scorer
