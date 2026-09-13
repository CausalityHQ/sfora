"""Physically progressive residual bitplanes for bounded vector refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from sfora.product_quantization import ProductQuantizationSpec

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
