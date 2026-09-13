"""Physically progressive residual bitplanes for bounded vector refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn
from torch.nn import functional as F

from sfora.product_quantization import (
    ProductQuantizationSpec,
    ProductQuantizer,
    fit_product_quantizer,
)

ProgressiveMetric = Literal["angular", "squared_l2"]


@dataclass(frozen=True, slots=True)
class ProgressiveResidualSpec:
    """Exact metric and wire geometry for one progressive residual codec."""

    metric: ProgressiveMetric
    base_spec: ProductQuantizationSpec
    maximum_residual_bits: int = 8

    def __post_init__(self) -> None:
        if (
            type(self.metric) is not str
            or self.metric not in ("angular", "squared_l2")
            or type(self.base_spec) is not ProductQuantizationSpec
            or type(self.maximum_residual_bits) is not int
            or not 1 <= self.maximum_residual_bits <= 8
        ):
            raise ValueError("progressive residual spec differs")

    @property
    def dimensions(self) -> int:
        """Return the represented vector width."""

        return self.base_spec.dimensions

    @property
    def residual_plane_bytes(self) -> int:
        """Return bytes in one physically stored residual bitplane."""

        return (self.dimensions + 7) // 8

    def bytes_per_vector(self, *, residual_bits: int) -> int:
        """Return base code, float16 scale, and prefix-plane payload bytes."""

        if (
            type(residual_bits) is not int
            or residual_bits < 1
            or residual_bits > self.maximum_residual_bits
        ):
            raise ValueError("progressive residual prefix differs")
        return self.base_spec.bytes_per_vector + 2 + residual_bits * self.residual_plane_bytes


@dataclass(frozen=True, slots=True)
class ProgressiveResidualCodes:
    """One concrete batch of base codes, scales, and eight residual bitplanes."""

    base_codes: torch.Tensor
    scales: torch.Tensor
    residual_planes: torch.Tensor

    def __post_init__(self) -> None:
        if (
            type(self.base_codes) is not torch.Tensor
            or self.base_codes.dtype != torch.uint8
            or self.base_codes.ndim != 2
            or self.base_codes.shape[0] < 1
            or self.base_codes.shape[1] < 1
            or not self.base_codes.is_contiguous()
            or type(self.scales) is not torch.Tensor
            or self.scales.dtype != torch.float16
            or self.scales.shape != (self.base_codes.shape[0],)
            or not self.scales.is_contiguous()
            or not bool(torch.isfinite(self.scales).all())
            or bool((self.scales <= 0).any())
            or type(self.residual_planes) is not torch.Tensor
            or self.residual_planes.dtype != torch.uint8
            or self.residual_planes.ndim != 3
            or self.residual_planes.shape[0] != self.base_codes.shape[0]
            or self.residual_planes.shape[1] != 8
            or self.residual_planes.shape[2] < 1
            or not self.residual_planes.is_contiguous()
            or self.scales.device != self.base_codes.device
            or self.residual_planes.device != self.base_codes.device
        ):
            raise ValueError("progressive residual codes differ")

    @property
    def rows(self) -> int:
        """Return the number of encoded rows."""

        return self.base_codes.shape[0]

    @property
    def device(self) -> torch.device:
        """Return the common tensor device."""

        return self.base_codes.device


@dataclass(frozen=True, slots=True)
class ProgressiveCandidateResult:
    """Stable bounded candidate ranking with explicit physical byte accounting."""

    ordinals: torch.Tensor
    scores: torch.Tensor
    base_bytes_read: int
    residual_bytes_read: int


def _validate_dimensions(dimensions: int) -> None:
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("progressive residual indexes differ")


def pack_residual_bitplanes(indexes: torch.Tensor, *, dimensions: int) -> torch.Tensor:
    """Pack unsigned eight-bit coordinate indexes into MSB-first bitplanes."""

    _validate_dimensions(dimensions)
    if (
        type(indexes) is not torch.Tensor
        or indexes.dtype != torch.uint8
        or indexes.ndim != 2
        or indexes.shape[0] < 1
        or indexes.shape[1] != dimensions
        or not indexes.is_contiguous()
    ):
        raise ValueError("progressive residual indexes differ")
    plane_bytes = (dimensions + 7) // 8
    padded = torch.zeros(
        (indexes.shape[0], plane_bytes * 8),
        dtype=torch.uint8,
        device=indexes.device,
    )
    padded[:, :dimensions] = indexes
    weights = torch.tensor(
        (128, 64, 32, 16, 8, 4, 2, 1),
        dtype=torch.int16,
        device=indexes.device,
    )
    planes = []
    for shift in range(7, -1, -1):
        bits = torch.bitwise_and(torch.bitwise_right_shift(padded, shift), 1)
        packed = (bits.reshape(indexes.shape[0], plane_bytes, 8).to(torch.int16) * weights).sum(
            dim=-1
        )
        planes.append(packed.to(torch.uint8))
    return torch.stack(planes, dim=1).contiguous()


def unpack_residual_prefix(
    planes: torch.Tensor,
    *,
    dimensions: int,
    residual_bits: int,
) -> torch.Tensor:
    """Unpack the first residual bitplanes into unsigned prefix indexes."""

    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("progressive residual planes differ")
    plane_bytes = (dimensions + 7) // 8
    if (
        type(residual_bits) is not int
        or not 1 <= residual_bits <= 8
        or type(planes) is not torch.Tensor
        or planes.dtype != torch.uint8
        or planes.ndim != 3
        or planes.shape[0] < 1
        or planes.shape[1] != 8
        or planes.shape[2] != plane_bytes
        or not planes.is_contiguous()
    ):
        raise ValueError("progressive residual planes differ")
    remainder = dimensions % 8
    if remainder:
        padding_mask = (1 << (8 - remainder)) - 1
        if bool(torch.bitwise_and(planes[:, :, -1], padding_mask).any()):
            raise ValueError("progressive residual planes differ")
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=planes.device)
    prefix = torch.zeros(
        (planes.shape[0], dimensions),
        dtype=torch.int16,
        device=planes.device,
    )
    for plane_index in range(residual_bits):
        bits = torch.bitwise_and(
            torch.bitwise_right_shift(planes[:, plane_index, :, None], shifts),
            1,
        ).reshape(planes.shape[0], plane_bytes * 8)[:, :dimensions]
        prefix = torch.bitwise_or(torch.bitwise_left_shift(prefix, 1), bits.to(torch.int16))
    return prefix.to(torch.uint8).contiguous()


def _prepare_metric_values(
    values: torch.Tensor,
    spec: ProgressiveResidualSpec,
    *,
    device: torch.device,
) -> torch.Tensor:
    if (
        type(values) is not torch.Tensor
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != spec.dimensions
        or values.device != device
        or not values.is_contiguous()
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("progressive residual input differs")
    if spec.metric == "squared_l2":
        return values.clone().contiguous()
    norms = torch.linalg.vector_norm(values.double(), dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 1e-12).any()):
        raise ValueError("progressive residual input differs")
    prepared = (values.double() / norms).float().contiguous()
    if not bool(torch.isfinite(prepared).all()):
        raise ValueError("progressive residual input differs")
    return cast(torch.Tensor, prepared)


class ProgressiveResidualQuantizer(nn.Module):
    """Full-dimensional product codes with one nested residual bitstream."""

    def __init__(
        self,
        spec: ProgressiveResidualSpec,
        base_quantizer: ProductQuantizer,
    ) -> None:
        super().__init__()
        if (
            type(spec) is not ProgressiveResidualSpec
            or type(base_quantizer) is not ProductQuantizer
            or base_quantizer.spec != spec.base_spec
        ):
            raise ValueError("progressive residual codec differs")
        self.spec = spec
        self.base_quantizer = base_quantizer
        self.base_quantizer.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        """Return the common codebook device."""

        return torch.device(self.base_quantizer.codebooks[0].device)

    def prepare_values(self, values: torch.Tensor) -> torch.Tensor:
        """Validate rows and apply only the registered metric preparation."""

        return _prepare_metric_values(values, self.spec, device=self.device)

    @torch.no_grad()
    def encode(self, values: torch.Tensor) -> ProgressiveResidualCodes:
        """Encode base assignments and one maximum-rate residual bitstream."""

        prepared = self.prepare_values(values)
        base_codes = self.base_quantizer.hard_encode(prepared)
        decoded = self.base_quantizer.hard_decode(base_codes)
        residual = prepared - decoded
        minimum = torch.tensor(
            torch.finfo(torch.float16).tiny,
            dtype=torch.float32,
            device=prepared.device,
        )
        scales = (residual.abs().amax(dim=1) / 127.5).clamp_min(minimum).to(torch.float16)
        if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
            raise ValueError("progressive residual scale differs")
        scale32 = scales.float()
        indexes = (
            torch.round(residual / scale32[:, None] + 127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .contiguous()
        )
        planes = pack_residual_bitplanes(indexes, dimensions=self.spec.dimensions)
        return ProgressiveResidualCodes(
            base_codes=base_codes,
            scales=scales.contiguous(),
            residual_planes=planes,
        )

    def _validate_codes(self, codes: ProgressiveResidualCodes) -> None:
        if (
            type(codes) is not ProgressiveResidualCodes
            or codes.base_codes.shape[1] != self.spec.base_spec.bytes_per_vector
            or codes.residual_planes.shape[2] != self.spec.residual_plane_bytes
            or codes.device != self.device
        ):
            raise ValueError("progressive residual codes differ")
        self.base_quantizer._validate_codes(codes.base_codes)

    @torch.no_grad()
    def decode_prefix(
        self,
        codes: ProgressiveResidualCodes,
        *,
        residual_bits: int,
    ) -> torch.Tensor:
        """Decode one physical residual prefix under the registered metric."""

        self._validate_codes(codes)
        self.spec.bytes_per_vector(residual_bits=residual_bits)
        prefix = unpack_residual_prefix(
            codes.residual_planes,
            dimensions=self.spec.dimensions,
            residual_bits=residual_bits,
        ).float()
        step = 1 << (8 - residual_bits)
        centers = prefix * step + (step - 1) / 2.0 - 127.5
        decoded = self.base_quantizer.hard_decode(codes.base_codes)
        reconstructed = decoded + centers * codes.scales.float()[:, None]
        if self.spec.metric == "angular":
            reconstructed = F.normalize(reconstructed, dim=1)
        if not bool(torch.isfinite(reconstructed).all()):
            raise RuntimeError("progressive residual decode is nonfinite")
        return reconstructed.contiguous()

    @torch.no_grad()
    def score_candidates(
        self,
        queries: torch.Tensor,
        codes: ProgressiveResidualCodes,
        candidate_ordinals: torch.Tensor,
        *,
        residual_bits: int,
        return_width: int,
    ) -> ProgressiveCandidateResult:
        """Refine and stably rank only caller-supplied candidate rows."""

        self._validate_codes(codes)
        self.spec.bytes_per_vector(residual_bits=residual_bits)
        prepared_queries = self.prepare_values(queries)
        if (
            type(candidate_ordinals) is not torch.Tensor
            or candidate_ordinals.dtype != torch.int64
            or candidate_ordinals.ndim != 2
            or candidate_ordinals.shape[0] < 1
            or candidate_ordinals.shape[1] < 1
            or candidate_ordinals.shape[0] != prepared_queries.shape[0]
            or candidate_ordinals.device != self.device
            or not candidate_ordinals.is_contiguous()
            or bool((candidate_ordinals < 0).any())
            or bool((candidate_ordinals >= codes.rows).any())
            or type(return_width) is not int
            or return_width < 1
            or return_width > candidate_ordinals.shape[1]
        ):
            raise ValueError("progressive candidate authority differs")
        sorted_candidates = torch.sort(candidate_ordinals, dim=1).values
        if candidate_ordinals.shape[1] > 1 and bool(
            (sorted_candidates[:, 1:] == sorted_candidates[:, :-1]).any()
        ):
            raise ValueError("progressive candidate authority differs")
        flat = candidate_ordinals.reshape(-1)
        gathered = ProgressiveResidualCodes(
            base_codes=codes.base_codes[flat].contiguous(),
            scales=codes.scales[flat].contiguous(),
            residual_planes=codes.residual_planes[flat].contiguous(),
        )
        decoded = self.decode_prefix(gathered, residual_bits=residual_bits).reshape(
            candidate_ordinals.shape[0],
            candidate_ordinals.shape[1],
            self.spec.dimensions,
        )
        if self.spec.metric == "angular":
            raw_scores = torch.einsum("bd,bkd->bk", prepared_queries, decoded)
            descending = True
        else:
            raw_scores = (decoded - prepared_queries[:, None, :]).square().sum(dim=-1)
            descending = False
        ordinal_order = torch.argsort(candidate_ordinals, dim=1, stable=True)
        ordered_ordinals = candidate_ordinals.gather(1, ordinal_order)
        ordered_scores = raw_scores.gather(1, ordinal_order)
        score_order = torch.argsort(
            ordered_scores,
            dim=1,
            descending=descending,
            stable=True,
        )[:, :return_width]
        query_count, candidate_count = candidate_ordinals.shape
        return ProgressiveCandidateResult(
            ordinals=ordered_ordinals.gather(1, score_order).contiguous(),
            scores=ordered_scores.gather(1, score_order).contiguous(),
            base_bytes_read=(
                query_count * candidate_count * self.spec.base_spec.bytes_per_vector
            ),
            residual_bytes_read=(
                query_count
                * candidate_count
                * (2 + residual_bits * self.spec.residual_plane_bytes)
            ),
        )

    def export_artifact(self) -> dict[str, object]:
        """Return a versioned CPU-portable codec artifact."""

        return {
            "schema": "sfora-progressive-residual-quantizer-v1",
            "metric": self.spec.metric,
            "maximum_residual_bits": self.spec.maximum_residual_bits,
            "block_dimensions": self.spec.base_spec.block_dimensions,
            "codebook_size": self.spec.base_spec.codebook_size,
            "codebooks": self.base_quantizer.detached_codebooks(),
        }

    @classmethod
    def from_artifact(cls, artifact: object) -> ProgressiveResidualQuantizer:
        """Restore and strictly validate one codec artifact."""

        if type(artifact) is not dict or set(artifact) != {
            "schema",
            "metric",
            "maximum_residual_bits",
            "block_dimensions",
            "codebook_size",
            "codebooks",
        }:
            raise ValueError("progressive residual artifact differs")
        if artifact.get("schema") != "sfora-progressive-residual-quantizer-v1":
            raise ValueError("progressive residual artifact differs")
        block_dimensions = artifact.get("block_dimensions")
        codebook_size = artifact.get("codebook_size")
        codebooks = artifact.get("codebooks")
        if (
            type(block_dimensions) is not tuple
            or type(codebook_size) is not int
            or type(codebooks) is not tuple
        ):
            raise ValueError("progressive residual artifact differs")
        try:
            base_spec = ProductQuantizationSpec(
                block_dimensions=block_dimensions,
                codebook_size=codebook_size,
            )
            spec = ProgressiveResidualSpec(
                metric=artifact.get("metric"),  # type: ignore[arg-type]
                base_spec=base_spec,
                maximum_residual_bits=artifact.get("maximum_residual_bits"),  # type: ignore[arg-type]
            )
            base = ProductQuantizer.from_codebooks(base_spec, codebooks)
            return cls(spec, base)
        except (TypeError, ValueError) as error:
            raise ValueError("progressive residual artifact differs") from error


def fit_progressive_residual_quantizer(
    values: torch.Tensor,
    spec: ProgressiveResidualSpec,
    *,
    seed: int,
    maximum_iterations: int,
) -> ProgressiveResidualQuantizer:
    """Fit the base codebooks from only the caller-supplied metric rows."""

    if (
        type(spec) is not ProgressiveResidualSpec
        or type(seed) is not int
        or seed < 0
        or type(maximum_iterations) is not int
        or maximum_iterations < 1
        or type(values) is not torch.Tensor
        or values.device.type != "cpu"
    ):
        raise ValueError("progressive residual fit differs")
    prepared = _prepare_metric_values(values, spec, device=torch.device("cpu"))
    try:
        base = fit_product_quantizer(
            prepared,
            spec.base_spec,
            seed=seed,
            maximum_iterations=maximum_iterations,
        )
    except ValueError as error:
        raise ValueError("progressive residual fit differs") from error
    return ProgressiveResidualQuantizer(spec, base)
