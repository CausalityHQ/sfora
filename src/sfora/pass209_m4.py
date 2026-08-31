"""Authenticated primitives for the Pass209 M4 objective-rescue measurement."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

DESCRIPTOR_MAGIC = b"SFORA-M4-F32-V1\n"
_HEADER_LENGTH = struct.Struct("<Q")
_NORM_TOLERANCE = 1.0e-6
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class M4DescriptorHeader:
    """Exact authority header for one normalized descriptor plane."""

    schema: str
    source_revision: str
    source_tree_digest: str
    dataset: str
    dataset_revision: str
    dataset_examples_sha256: str
    cell: str
    model_name: str
    model_revision: str
    readout: str
    rows: int
    dimensions: int
    payload_bytes: int
    payload_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return the concrete JSON object in the registered schema."""

        return asdict(self)


_HEADER_KEYS = frozenset(M4DescriptorHeader.__dataclass_fields__)


def canonical_json_bytes(value: dict[str, object]) -> bytes:
    """Encode canonical sorted compact JSON with exactly one trailing LF."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _require_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"descriptor header {name} must be a nonempty string")
    return value


def _require_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"descriptor header {name} must be a positive integer")
    return value


def _validate_header(header: M4DescriptorHeader) -> None:
    if header.schema != "sfora-pass209-m4-descriptor-v1":
        raise ValueError("descriptor header schema differs")
    if header.dataset != "cars":
        raise ValueError("descriptor header dataset differs")
    for name in ("cell", "model_name", "readout"):
        _require_string(getattr(header, name), name)
    for name in ("source_revision", "dataset_revision", "model_revision"):
        if _REVISION.fullmatch(_require_string(getattr(header, name), name)) is None:
            raise ValueError(f"descriptor header {name} is not a revision")
    for name in (
        "source_tree_digest",
        "dataset_examples_sha256",
        "payload_sha256",
    ):
        if _SHA256.fullmatch(_require_string(getattr(header, name), name)) is None:
            raise ValueError(f"descriptor header {name} is not a SHA-256 digest")
    rows = _require_integer(header.rows, "rows")
    dimensions = _require_integer(header.dimensions, "dimensions")
    payload_bytes = _require_integer(header.payload_bytes, "payload_bytes")
    if payload_bytes != rows * dimensions * 4:
        raise ValueError("descriptor payload byte count differs from shape")


def _descriptor_payload(descriptors: torch.Tensor) -> bytes:
    if descriptors.ndim != 2 or descriptors.shape[0] <= 0 or descriptors.shape[1] <= 0:
        raise ValueError("descriptor tensor must be a nonempty matrix")
    values = descriptors.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("descriptor values must be finite")
    norms = torch.linalg.vector_norm(values, dim=1)
    if not bool(torch.isfinite(norms).all()) or bool(
        torch.any(torch.abs(norms - 1.0) > _NORM_TOLERANCE)
    ):
        raise ValueError("descriptor rows must have unit norm")
    return values.numpy().astype("<f4", copy=False).tobytes(order="C")


def encode_descriptor_file(
    header: M4DescriptorHeader, descriptors: torch.Tensor
) -> bytes:
    """Encode one descriptor plane after validating all header bindings."""

    _validate_header(header)
    payload = _descriptor_payload(descriptors)
    if tuple(descriptors.shape) != (header.rows, header.dimensions):
        raise ValueError("descriptor tensor shape differs from header")
    if len(payload) != header.payload_bytes:
        raise ValueError("descriptor payload byte count differs from header")
    if hashlib.sha256(payload).hexdigest() != header.payload_sha256:
        raise ValueError("descriptor payload digest differs from header")
    header_bytes = canonical_json_bytes(header.to_dict())
    return DESCRIPTOR_MAGIC + _HEADER_LENGTH.pack(len(header_bytes)) + header_bytes + payload


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("descriptor header contains a duplicate key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"descriptor header contains nonfinite JSON value {value}")


def _parse_header(payload: bytes) -> M4DescriptorHeader:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("descriptor header is not valid JSON") from error
    if type(value) is not dict or set(value) != _HEADER_KEYS:
        raise ValueError("descriptor header key set differs")
    if canonical_json_bytes(value) != payload:
        raise ValueError("descriptor header bytes are not canonical")
    try:
        header = M4DescriptorHeader(**value)
    except TypeError as error:
        raise ValueError("descriptor header fields differ") from error
    _validate_header(header)
    return header


def decode_descriptor_file(payload: bytes) -> tuple[M4DescriptorHeader, torch.Tensor]:
    """Decode and authenticate one complete framed descriptor file."""

    if not payload.startswith(DESCRIPTOR_MAGIC):
        raise ValueError("descriptor file magic differs")
    length_offset = len(DESCRIPTOR_MAGIC)
    if len(payload) < length_offset + _HEADER_LENGTH.size:
        raise ValueError("descriptor file is truncated before header length")
    (header_length,) = _HEADER_LENGTH.unpack_from(payload, length_offset)
    if header_length == 0 or header_length > 65_536:
        raise ValueError("descriptor header length is invalid")
    header_start = length_offset + _HEADER_LENGTH.size
    header_end = header_start + header_length
    if header_end > len(payload):
        raise ValueError("descriptor file is truncated in header")
    header = _parse_header(payload[header_start:header_end])
    descriptor_payload = payload[header_end:]
    if len(descriptor_payload) != header.payload_bytes:
        raise ValueError("descriptor payload byte count differs from header")
    if hashlib.sha256(descriptor_payload).hexdigest() != header.payload_sha256:
        raise ValueError("descriptor payload digest differs from header")
    values = np.frombuffer(descriptor_payload, dtype="<f4")
    descriptors = torch.from_numpy(values.copy()).reshape(header.rows, header.dimensions)
    if _descriptor_payload(descriptors) != descriptor_payload:
        raise ValueError("descriptor payload does not round trip exactly")
    return header, descriptors


def _fsync_directories(paths: tuple[Path, ...]) -> None:
    for parent in sorted({path.parent.resolve() for path in paths}, key=str):
        descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish_new_outputs(outputs: tuple[tuple[Path, bytes], ...]) -> None:
    """Atomically publish a nonempty set of create-new files or roll it back."""

    if not outputs:
        raise ValueError("at least one output is required")
    paths = tuple(path for path, _ in outputs)
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("output paths must be distinct")
    partials = tuple(path.with_name(f".{path.name}.partial") for path in paths)
    for path, partial in zip(paths, partials, strict=True):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        if partial.exists():
            raise FileExistsError(f"refusing pre-existing partial {partial}")

    published: list[Path] = []
    written: list[Path] = []
    try:
        for (path, payload), partial in zip(outputs, partials, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            with partial.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            written.append(partial)
        for path, partial in zip(paths, partials, strict=True):
            os.link(partial, path)
            published.append(path)
        _fsync_directories(paths)
        for partial in partials:
            partial.unlink()
        _fsync_directories(paths)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        for partial in written:
            partial.unlink(missing_ok=True)
        _fsync_directories(paths)
        raise
