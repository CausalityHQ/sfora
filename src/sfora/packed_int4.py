"""Canonical signed-int4 storage for unit-normalized embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import numpy as np
import torch
from torch.nn import functional as F

_MAX_EXACT_INT4_DOT_DIMENSIONS = (1 << 24) // 49


@lru_cache(maxsize=1)
def _cpu_int_mm_available() -> bool:
    """Probe the private CPU int8 kernel once without relying on error text."""

    integer_mm = getattr(torch, "_int_mm", None)
    if not callable(integer_mm):
        return False
    left = torch.ones((1, 2), dtype=torch.int8)
    right = torch.ones((2, 1), dtype=torch.int8)
    try:
        observed = integer_mm(left, right)
    except (NotImplementedError, RuntimeError):
        return False
    return bool(
        type(observed) is torch.Tensor
        and observed.device.type == "cpu"
        and observed.dtype == torch.int32
        and observed.shape == (1, 1)
        and observed.item() == 2
    )


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

        if (
            type(other) is not PackedInt4Embeddings
            or other.dimensions != self.dimensions
            or self.dimensions > _MAX_EXACT_INT4_DOT_DIMENSIONS
        ):
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


@dataclass(frozen=True, slots=True)
class ResidentInt4Gallery:
    """CPU int8 execution layout derived from the canonical int4 artifact."""

    gallery_codes_transposed: torch.Tensor
    inverse_norms: torch.Tensor
    dimensions: int

    def __post_init__(self) -> None:
        if (
            type(self.dimensions) is not int
            or self.dimensions < 2
            or self.dimensions % 2 != 0
            or self.dimensions > _MAX_EXACT_INT4_DOT_DIMENSIONS
            or type(self.gallery_codes_transposed) is not torch.Tensor
            or self.gallery_codes_transposed.device.type != "cpu"
            or self.gallery_codes_transposed.dtype != torch.int8
            or self.gallery_codes_transposed.ndim != 2
            or self.gallery_codes_transposed.shape[0] != self.dimensions
            or self.gallery_codes_transposed.shape[1] < 1
            or not self.gallery_codes_transposed.is_contiguous()
            or type(self.inverse_norms) is not torch.Tensor
            or self.inverse_norms.device.type != "cpu"
            or self.inverse_norms.dtype != torch.float16
            or self.inverse_norms.shape != (self.gallery_codes_transposed.shape[1],)
            or not self.inverse_norms.is_contiguous()
            or not bool(torch.isfinite(self.inverse_norms).all())
            or bool((self.inverse_norms <= 0).any())
        ):
            raise ValueError("resident int4 gallery authority differs")
        codes = self.gallery_codes_transposed
        norms = torch.linalg.vector_norm(codes.float(), dim=0)
        expected = norms.reciprocal().to(torch.float16)
        lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
        upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
        if (
            bool((codes < -7).any())
            or bool((codes > 7).any())
            or bool((norms <= 0).any())
            or bool((self.inverse_norms < lower).any())
            or bool((self.inverse_norms > upper).any())
        ):
            raise ValueError("resident int4 gallery authority differs")

    @property
    def resident_bytes_per_vector(self) -> int:
        """Return layout bytes per vector, excluding source artifact and score outputs."""

        return self.dimensions + 2

    @property
    def integer_backend_available(self) -> bool:
        """Return whether the exact CPU int8 matrix kernel passed its probe."""

        return _cpu_int_mm_available()

    @classmethod
    def from_packed(cls, value: PackedInt4Embeddings) -> ResidentInt4Gallery:
        """Decode a canonical packed gallery once into the integer execution layout."""

        if type(value) is not PackedInt4Embeddings:
            raise ValueError("resident int4 gallery authority differs")
        return cls(
            gallery_codes_transposed=value.signed_codes().T.contiguous(),
            inverse_norms=value.inverse_norms.clone().contiguous(),
            dimensions=value.dimensions,
        )

    def score_queries(
        self, *, queries: PackedInt4Embeddings, require_integer: bool = True
    ) -> torch.Tensor:
        """Score through CPU int8, or explicitly allow a float32 materializing fallback."""

        if (
            type(queries) is not PackedInt4Embeddings
            or queries.dimensions != self.dimensions
            or type(require_integer) is not bool
        ):
            raise ValueError("resident int4 similarity authority differs")

        def float_dots() -> torch.Tensor:
            with torch.autocast(device_type="cpu", enabled=False):
                return queries.signed_codes().float() @ self.gallery_codes_transposed.float()

        integer_mm = getattr(torch, "_int_mm", None)
        if self.integer_backend_available and callable(integer_mm):
            integer_dots = cast(
                torch.Tensor,
                integer_mm(queries.signed_codes(), self.gallery_codes_transposed),
            ).to(torch.float32)
        else:
            if require_integer:
                raise RuntimeError("resident int4 integer kernel is unavailable")
            integer_dots = float_dots()
        return (
            integer_dots
            * queries.inverse_norms.to(torch.float32).unsqueeze(1)
            * self.inverse_norms.to(torch.float32).unsqueeze(0)
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
