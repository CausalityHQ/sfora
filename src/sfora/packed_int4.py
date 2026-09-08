"""Canonical signed-int4 storage for unit-normalized embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


def _unit_rows(value: torch.Tensor) -> bool:
    # Keep this module independent of the source file bound into historical evidence.
    if not bool(torch.isfinite(value).all()):
        return False
    norms = torch.linalg.vector_norm(value.detach().double(), dim=1)
    return bool((torch.abs(norms - 1.0) <= 2e-5).all())


@dataclass(frozen=True, slots=True)
class PackedInt4Embeddings:
    """Row-major signed-int4 embeddings with one f16 inverse norm per row."""

    packed_codes: torch.Tensor
    inverse_norms: torch.Tensor
    dimensions: int

    def __post_init__(self) -> None:
        if (
            type(self.dimensions) is not int
            or self.dimensions < 2
            or self.dimensions % 2 != 0
            or type(self.packed_codes) is not torch.Tensor
            or self.packed_codes.device.type != "cpu"
            or self.packed_codes.dtype != torch.uint8
            or self.packed_codes.ndim != 2
            or self.packed_codes.shape[0] < 1
            or self.packed_codes.shape[1] != self.dimensions // 2
            or not self.packed_codes.is_contiguous()
            or type(self.inverse_norms) is not torch.Tensor
            or self.inverse_norms.device.type != "cpu"
            or self.inverse_norms.dtype != torch.float16
            or self.inverse_norms.shape != (self.packed_codes.shape[0],)
            or not self.inverse_norms.is_contiguous()
            or not bool(torch.isfinite(self.inverse_norms).all())
            or bool((self.inverse_norms <= 0).any())
        ):
            raise ValueError("packed int4 embedding authority differs")
        codes = self.signed_codes()
        norms = torch.linalg.vector_norm(codes.float(), dim=1)
        expected_inverse_norms = norms.reciprocal().to(torch.float16)
        lower = torch.nextafter(
            expected_inverse_norms,
            torch.full_like(expected_inverse_norms, -torch.inf),
        )
        upper = torch.nextafter(
            expected_inverse_norms,
            torch.full_like(expected_inverse_norms, torch.inf),
        )
        if (
            bool((codes == -8).any())
            or bool((norms <= 0).any())
            or bool((self.inverse_norms < lower).any())
            or bool((self.inverse_norms > upper).any())
        ):
            raise ValueError("packed int4 embedding authority differs")

    @property
    def bytes_per_vector(self) -> int:
        """Return the exact wire width of one packed vector."""

        return self.dimensions // 2 + 2

    def signed_codes(self) -> torch.Tensor:
        """Decode canonical nibbles into contiguous signed int8 coordinates."""

        signed = self.packed_codes.to(torch.int8)
        low = (signed << 4) >> 4
        high = signed >> 4
        return torch.stack((low, high), dim=2).reshape(-1, self.dimensions).contiguous()

    def restore(self) -> torch.Tensor:
        """Restore unit-like float32 rows without retaining expanded storage."""

        return self.signed_codes().float() * self.inverse_norms.float().unsqueeze(1)

    def cosine_similarity(
        self,
        other: PackedInt4Embeddings,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute pairwise cosine scores from packed signed-int4 rows."""

        if type(other) is not PackedInt4Embeddings or other.dimensions != self.dimensions:
            raise ValueError("packed int4 similarity authority differs")
        if device is None:
            device = torch.device("cpu")
        if type(device) is not torch.device:
            raise ValueError("packed int4 similarity authority differs")
        integer_dots = (
            self.signed_codes().to(device=device, dtype=torch.float32)
            @ other.signed_codes().to(device=device, dtype=torch.float32).T
        )
        return (
            integer_dots
            * self.inverse_norms.to(device=device, dtype=torch.float32).unsqueeze(1)
            * other.inverse_norms.to(device=device, dtype=torch.float32).unsqueeze(0)
        )

    def float_query_similarity(
        self,
        queries: torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Score unit float queries against this persistent int4 gallery."""

        if (
            type(queries) is not torch.Tensor
            or queries.device.type != "cpu"
            or queries.dtype != torch.float32
            or queries.ndim != 2
            or queries.shape[0] < 1
            or queries.shape[1] != self.dimensions
            or not queries.is_contiguous()
            or not _unit_rows(queries)
        ):
            raise ValueError("packed int4 float query authority differs")
        if device is None:
            device = torch.device("cpu")
        if type(device) is not torch.device:
            raise ValueError("packed int4 float query authority differs")
        integer_gallery = self.signed_codes().to(device=device, dtype=torch.float32)
        return (queries.to(device) @ integer_gallery.T) * self.inverse_norms.to(
            device=device, dtype=torch.float32
        ).unsqueeze(0)

    def to_bytes(self) -> bytes:
        """Serialize packed nibbles followed by one little-endian f16 per row."""

        count = self.packed_codes.shape[0]
        wire = np.empty((count, self.bytes_per_vector), dtype=np.uint8)
        wire[:, : self.dimensions // 2] = self.packed_codes.numpy()
        inverse_bytes = self.inverse_norms.numpy().astype("<f2", copy=False).view(np.uint8)
        wire[:, self.dimensions // 2 :] = inverse_bytes.reshape(count, 2)
        return wire.tobytes(order="C")

    @classmethod
    def from_bytes(
        cls,
        wire: bytes,
        *,
        count: int,
        dimensions: int,
    ) -> PackedInt4Embeddings:
        """Restore and validate a canonical signed-int4 wire artifact."""

        if (
            type(wire) is not bytes
            or type(count) is not int
            or count < 1
            or type(dimensions) is not int
            or dimensions < 2
            or dimensions % 2 != 0
            or len(wire) != count * (dimensions // 2 + 2)
        ):
            raise ValueError("packed int4 byte authority differs")
        rows = np.frombuffer(wire, dtype=np.uint8).reshape(count, dimensions // 2 + 2)
        packed_codes = torch.from_numpy(rows[:, : dimensions // 2].copy()).contiguous()
        inverse_bytes = rows[:, dimensions // 2 :].copy().reshape(-1)
        inverse_values = inverse_bytes.view("<f2").astype(np.float16, copy=True)
        inverse_norms = torch.from_numpy(inverse_values).contiguous()
        return cls(
            packed_codes=packed_codes,
            inverse_norms=inverse_norms,
            dimensions=dimensions,
        )


def fixed_int4_unit_codes(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize unit rows with per-vector scaling to signed int4 codes."""

    if (
        type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] < 2
        or value.shape[1] % 2 != 0
        or not _unit_rows(value)
    ):
        raise ValueError("joint relational quantization authority differs")
    maximum = torch.amax(torch.abs(value), dim=1, keepdim=True)
    codes = torch.round(value / maximum * 7.0).clamp(-7, 7).to(torch.int8).contiguous()
    restored = F.normalize(codes.float(), dim=1).contiguous()
    if not bool(torch.isfinite(restored).all()):
        raise ValueError("joint relational quantization geometry differs")
    return codes, restored


def pack_int4_unit_embeddings(value: torch.Tensor) -> PackedInt4Embeddings:
    """Quantize unit rows into the exact half-dimensions-plus-two-byte format."""

    codes, _restored = fixed_int4_unit_codes(value)
    low = codes[:, 0::2].to(torch.int16) & 0x0F
    high = (codes[:, 1::2].to(torch.int16) & 0x0F) << 4
    packed_codes = (low | high).to(torch.uint8).contiguous()
    norms = torch.linalg.vector_norm(codes.float(), dim=1)
    inverse_norms = norms.reciprocal().to(torch.float16).contiguous()
    return PackedInt4Embeddings(
        packed_codes=packed_codes,
        inverse_norms=inverse_norms,
        dimensions=codes.shape[1],
    )
