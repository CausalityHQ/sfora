"""Canonical packed-int8 embeddings and fixed-scale unit-row quantization."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

_PACKED_INT8_ARTIFACT_MAGIC = b"SFORA-PACKED-INT8-v1\0"


_SHA256_BYTES = 32


def _unit_rows(value: torch.Tensor) -> bool:
    if not bool(torch.isfinite(value).all()):
        return False
    norms = torch.linalg.vector_norm(value.detach().double(), dim=1)
    return bool((torch.abs(norms - 1.0) <= 2e-5).all())


@dataclass(frozen=True, slots=True)
class PackedInt8Embeddings:
    """Row-major int8 embeddings with one little-endian f16 inverse norm per row."""

    codes: torch.Tensor
    inverse_norms: torch.Tensor

    def __post_init__(self) -> None:
        if (
            type(self.codes) is not torch.Tensor
            or self.codes.device.type != "cpu"
            or self.codes.dtype != torch.int8
            or self.codes.ndim != 2
            or self.codes.shape[0] < 1
            or self.codes.shape[1] < 2
            or not self.codes.is_contiguous()
            or type(self.inverse_norms) is not torch.Tensor
            or self.inverse_norms.device.type != "cpu"
            or self.inverse_norms.dtype != torch.float16
            or self.inverse_norms.shape != (self.codes.shape[0],)
            or not self.inverse_norms.is_contiguous()
            or not bool(torch.isfinite(self.inverse_norms).all())
            or bool((self.inverse_norms <= 0).any())
        ):
            raise ValueError("packed int8 embedding authority differs")
        norms = torch.linalg.vector_norm(self.codes.float(), dim=1)
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
            bool((norms <= 0).any())
            or bool((self.inverse_norms < lower).any())
            or bool((self.inverse_norms > upper).any())
        ):
            raise ValueError("packed int8 embedding authority differs")

    @property
    def bytes_per_vector(self) -> int:
        """Return the exact wire width of one packed vector."""

        return self.codes.shape[1] + 2

    def restore(self) -> torch.Tensor:
        """Restore unit-like float32 rows without retaining expanded storage."""

        return self.codes.float() * self.inverse_norms.float().unsqueeze(1)

    def cosine_similarity(
        self,
        other: PackedInt8Embeddings,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute pairwise cosine scores directly from packed row metadata."""

        if type(other) is not PackedInt8Embeddings or other.codes.shape[1] != self.codes.shape[1]:
            raise ValueError("packed int8 similarity authority differs")
        if device is None:
            device = torch.device("cpu")
        if type(device) is not torch.device:
            raise ValueError("packed int8 similarity authority differs")
        integer_dots = (
            self.codes.to(device=device, dtype=torch.float32)
            @ other.codes.to(device=device, dtype=torch.float32).T
        )
        return (
            integer_dots
            * self.inverse_norms.to(device=device, dtype=torch.float32).unsqueeze(1)
            * other.inverse_norms.to(device=device, dtype=torch.float32).unsqueeze(0)
        )

    def to_bytes(self) -> bytes:
        """Serialize each row as signed code bytes followed by one little-endian f16."""

        count, dimensions = self.codes.shape
        wire = np.empty((count, dimensions + 2), dtype=np.uint8)
        wire[:, :dimensions] = self.codes.numpy().view(np.uint8)
        inverse_bytes = self.inverse_norms.numpy().astype("<f2", copy=False).view(np.uint8)
        wire[:, dimensions:] = inverse_bytes.reshape(count, 2)
        return wire.tobytes(order="C")

    def save(self, path: Path) -> None:
        """Persist a self-describing packed batch with an exact SHA-256 trailer."""

        if not isinstance(path, Path):
            raise ValueError("packed int8 artifact path differs")
        count, dimensions = self.codes.shape
        payload = (
            _PACKED_INT8_ARTIFACT_MAGIC + struct.pack("<QQ", count, dimensions) + self.to_bytes()
        )
        path.write_bytes(payload + hashlib.sha256(payload).digest())

    @classmethod
    def load(cls, path: Path) -> PackedInt8Embeddings:
        """Load a packed batch only when framing, dimensions, and digest are exact."""

        if not isinstance(path, Path):
            raise ValueError("packed int8 artifact path differs")
        artifact = path.read_bytes()
        header_bytes = len(_PACKED_INT8_ARTIFACT_MAGIC) + 16
        if len(artifact) < header_bytes + _SHA256_BYTES:
            raise ValueError("packed int8 artifact differs")
        payload = artifact[:-_SHA256_BYTES]
        if (
            not payload.startswith(_PACKED_INT8_ARTIFACT_MAGIC)
            or hashlib.sha256(payload).digest() != artifact[-_SHA256_BYTES:]
        ):
            raise ValueError("packed int8 artifact differs")
        count, dimensions = struct.unpack(
            "<QQ", payload[len(_PACKED_INT8_ARTIFACT_MAGIC) : header_bytes]
        )
        expected_bytes = header_bytes + count * (dimensions + 2)
        if count < 1 or dimensions < 2 or len(payload) != expected_bytes:
            raise ValueError("packed int8 artifact differs")
        try:
            return cls.from_bytes(payload[header_bytes:], count=count, dimensions=dimensions)
        except ValueError as error:
            raise ValueError("packed int8 artifact differs") from error

    @classmethod
    def from_bytes(cls, wire: bytes, *, count: int, dimensions: int) -> PackedInt8Embeddings:
        """Parse an exact packed batch with no trailing or missing bytes."""

        if (
            type(wire) is not bytes
            or type(count) is not int
            or count < 1
            or type(dimensions) is not int
            or dimensions < 2
            or len(wire) != count * (dimensions + 2)
        ):
            raise ValueError("packed int8 byte authority differs")
        rows = np.frombuffer(wire, dtype=np.uint8).reshape(count, dimensions + 2)
        codes = torch.from_numpy(rows[:, :dimensions].copy().view(np.int8))
        inverse_values = (
            rows[:, dimensions:].copy().reshape(-1).view("<f2").astype(np.float16, copy=True)
        )
        inverse = torch.from_numpy(inverse_values)
        return cls(codes=codes.contiguous(), inverse_norms=inverse.contiguous())


def fixed_int8_unit_codes(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a unit embedding to fixed-scale int8 and return its unit restoration."""

    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.float32
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] < 2
        or not _unit_rows(value)
    ):
        raise ValueError("joint relational quantization authority differs")
    codes = torch.round(value * 127.0).clamp(-127, 127).to(torch.int8).contiguous()
    restored = F.normalize(codes.float(), dim=1).contiguous()
    if not bool(torch.isfinite(restored).all()):
        raise ValueError("joint relational quantization geometry differs")
    return codes, restored


def pack_int8_unit_embeddings(value: torch.Tensor) -> PackedInt8Embeddings:
    """Quantize unit rows into the exact dimensions-plus-two-byte wire format."""

    codes, _restored = fixed_int8_unit_codes(value)
    norms = torch.linalg.vector_norm(codes.float(), dim=1)
    inverse_norms = norms.reciprocal().to(torch.float16).contiguous()
    return PackedInt8Embeddings(codes=codes, inverse_norms=inverse_norms)


# Keep historical pickle references without importing the training module here.
_unit_rows.__module__ = "sfora.joint_relational_compaction"
PackedInt8Embeddings.__module__ = "sfora.joint_relational_compaction"
fixed_int8_unit_codes.__module__ = "sfora.joint_relational_compaction"
pack_int8_unit_embeddings.__module__ = "sfora.joint_relational_compaction"
