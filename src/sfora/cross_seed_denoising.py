"""Deterministic local artifacts for cross-seed denoising experiments."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

import torch

_SCHEMA = "sfora-cross-seed-tensor-artifact-v1"
_MANIFEST_KEYS = {
    "bindings",
    "claim_eligible",
    "role",
    "schema",
    "state_sha256",
    "tensors",
}
_TENSOR_KEYS = {"bytes", "dtype", "file", "name", "sha256", "shape"}
_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}
_DTYPE_NAMES = {value: key for key, value in _DTYPES.items()}


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _is_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_bindings(value: object) -> dict[str, str]:
    if type(value) is not dict or not value:
        raise ValueError("bindings must be a non-empty concrete string mapping")
    bindings = cast(dict[object, object], value)
    if any(type(key) is not str or type(item) is not str for key, item in bindings.items()):
        raise ValueError("bindings must contain concrete strings")
    return {cast(str, key): cast(str, item) for key, item in sorted(bindings.items())}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _state_digest(
    *,
    role: str,
    bindings: Mapping[str, str],
    records: list[dict[str, object]],
    payloads: list[bytes],
) -> str:
    digest = hashlib.sha256()
    header = {
        "bindings": dict(bindings),
        "role": role,
        "tensors": [
            {
                "bytes": record["bytes"],
                "dtype": record["dtype"],
                "name": record["name"],
                "shape": record["shape"],
            }
            for record in records
        ],
    }
    header_bytes = _canonical_json_bytes(header)
    digest.update(len(header_bytes).to_bytes(8, "little"))
    digest.update(header_bytes)
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def write_tensor_artifact(
    root: Path,
    state: object,
    *,
    role: str,
    bindings: object,
) -> bytes:
    """Write a deterministic, content-addressed tensor artifact."""

    if type(role) is not str or not role:
        raise ValueError("role must be a non-empty concrete string")
    normalized_bindings = _validate_bindings(bindings)
    if not isinstance(state, OrderedDict) or not state:
        raise ValueError("state must be a non-empty OrderedDict")
    if root.exists():
        raise ValueError("artifact root already exists")

    records: list[dict[str, object]] = []
    payloads: list[bytes] = []
    for ordinal, (name, tensor_value) in enumerate(sorted(state.items())):
        if type(name) is not str or not name:
            raise ValueError("tensor name must be a non-empty concrete string")
        if not isinstance(tensor_value, torch.Tensor):
            raise TypeError("state values must be tensors")
        tensor = tensor_value.detach().cpu().contiguous()
        dtype_name = _DTYPE_NAMES.get(tensor.dtype)
        if dtype_name is None:
            raise ValueError(f"unsupported tensor dtype: {tensor.dtype}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"tensor {name!r} must be finite")
        payload = _tensor_bytes(tensor)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        records.append(
            {
                "bytes": len(payload),
                "dtype": dtype_name,
                "file": f"tensors/{ordinal:05d}-{payload_sha256}.bin",
                "name": name,
                "sha256": payload_sha256,
                "shape": list(tensor.shape),
            }
        )
        payloads.append(payload)

    state_sha256 = _state_digest(
        role=role,
        bindings=normalized_bindings,
        records=records,
        payloads=payloads,
    )
    manifest = {
        "bindings": normalized_bindings,
        "claim_eligible": False,
        "role": role,
        "schema": _SCHEMA,
        "state_sha256": state_sha256,
        "tensors": records,
    }
    manifest_bytes = _canonical_json_bytes(manifest)

    tensors_root = root / "tensors"
    tensors_root.mkdir(parents=True)
    try:
        for record, payload in zip(records, payloads, strict=True):
            path = root / cast(str, record["file"])
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        with (root / "manifest.json").open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        for record in records:
            path = root / cast(str, record["file"])
            if path.is_file() and not path.is_symlink():
                path.unlink()
        manifest_path = root / "manifest.json"
        if manifest_path.is_file() and not manifest_path.is_symlink():
            manifest_path.unlink()
        if tensors_root.is_dir():
            tensors_root.rmdir()
        if root.is_dir():
            root.rmdir()
        raise
    return manifest_bytes


def read_tensor_artifact(
    root: Path,
    manifest_bytes: bytes,
    *,
    role: str,
) -> OrderedDict[str, torch.Tensor]:
    """Authenticate and load a deterministic tensor artifact."""

    if type(manifest_bytes) is not bytes:
        raise TypeError("manifest bytes must be concrete bytes")
    try:
        value = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest is not valid JSON") from exc
    if type(value) is not dict or _canonical_json_bytes(value) != manifest_bytes:
        raise ValueError("manifest is not canonical JSON")
    if set(value) != _MANIFEST_KEYS:
        raise ValueError("manifest schema differs")
    if value["schema"] != _SCHEMA or type(value["claim_eligible"]) is not bool:
        raise ValueError("manifest schema differs")
    if value["claim_eligible"] is not False:
        raise ValueError("manifest claim eligibility differs")
    if type(role) is not str or value["role"] != role:
        raise ValueError("manifest role differs")
    bindings = _validate_bindings(value["bindings"])
    if not _is_hex(value["state_sha256"], 64):
        raise ValueError("state digest differs")
    if type(value["tensors"]) is not list or not value["tensors"]:
        raise ValueError("tensor schema differs")

    tensors = cast(list[object], value["tensors"])
    records: list[dict[str, object]] = []
    payloads: list[bytes] = []
    expected_names: list[str] = []
    for ordinal, raw_record in enumerate(tensors):
        if type(raw_record) is not dict or set(raw_record) != _TENSOR_KEYS:
            raise ValueError("tensor schema differs")
        record = cast(dict[str, object], raw_record)
        name = record["name"]
        if type(name) is not str or not name:
            raise ValueError("tensor name differs")
        expected_names.append(name)
        if expected_names != sorted(expected_names) or len(set(expected_names)) != len(
            expected_names
        ):
            raise ValueError("tensor order differs")
        dtype_name = record["dtype"]
        if type(dtype_name) is not str or dtype_name not in _DTYPES:
            raise ValueError("tensor dtype differs")
        shape = record["shape"]
        if type(shape) is not list or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise ValueError("tensor shape differs")
        byte_count = record["bytes"]
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError("tensor length differs")
        payload_sha256 = record["sha256"]
        if not _is_hex(payload_sha256, 64):
            raise ValueError("tensor digest differs")
        expected_file = f"tensors/{ordinal:05d}-{payload_sha256}.bin"
        file_value = record["file"]
        if type(file_value) is not str or file_value != expected_file:
            raise ValueError("tensor path differs")
        pure_path = PurePosixPath(file_value)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ValueError("tensor path differs")
        path = root / file_value
        if path.is_symlink():
            raise ValueError("tensor payload must not be a symlink")
        if not path.is_file():
            raise ValueError("tensor payload is missing")
        payload = path.read_bytes()
        if len(payload) != byte_count:
            raise ValueError("tensor payload length differs")
        if hashlib.sha256(payload).hexdigest() != payload_sha256:
            raise ValueError("tensor payload digest differs")
        dtype = _DTYPES[dtype_name]
        expected_elements = 1
        for dimension in shape:
            expected_elements *= dimension
        expected_bytes = expected_elements * torch.empty((), dtype=dtype).element_size()
        if expected_bytes != byte_count:
            raise ValueError("tensor shape and length differ")
        records.append(record)
        payloads.append(payload)

    computed_state_sha256 = _state_digest(
        role=role,
        bindings=bindings,
        records=records,
        payloads=payloads,
    )
    if computed_state_sha256 != value["state_sha256"]:
        raise ValueError("bindings or state digest differs")

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for record, payload in zip(records, payloads, strict=True):
        dtype = _DTYPES[cast(str, record["dtype"])]
        owned = bytearray(payload)
        tensor = torch.frombuffer(owned, dtype=dtype).clone()
        result[cast(str, record["name"])] = tensor.reshape(cast(list[int], record["shape"]))
    return result
