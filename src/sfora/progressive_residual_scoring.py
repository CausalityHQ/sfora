"""Validated tensor scoring for progressive residual candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

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


def _progressive_score_graph(
    codec: ProgressiveResidualQuantizer,
    prepared_queries: torch.Tensor,
    candidate_ordinals: torch.Tensor,
    gathered_base: torch.Tensor,
    gathered_scales: torch.Tensor,
    gathered_planes: torch.Tensor,
    spec: ProgressiveScoringSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        candidate_ordinals.shape[0],
        spec.candidate_width,
        codec.spec.dimensions,
    )
    if codec.spec.metric == "angular":
        scores = torch.einsum("bd,bkd->bk", prepared_queries, reconstructed)
        descending = True
    else:
        scores = (reconstructed - prepared_queries[:, None, :]).square().sum(dim=-1)
        descending = False
    ordinal_order = torch.argsort(candidate_ordinals, dim=1, stable=True)
    ordered_ordinals = candidate_ordinals.gather(1, ordinal_order)
    ordered_scores = scores.gather(1, ordinal_order)
    score_order = torch.argsort(
        ordered_scores,
        dim=1,
        descending=descending,
        stable=True,
    )[:, : spec.return_width]
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
    ordinals, scores = _progressive_score_graph(
        codec,
        prepared,
        candidate_ordinals,
        gathered_base,
        gathered_scales,
        gathered_planes,
        spec,
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
