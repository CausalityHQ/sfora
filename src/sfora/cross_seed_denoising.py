"""Deterministic local artifacts for cross-seed denoising experiments."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
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
_SEEDS = (17, 29, 43)
_SQRT2 = 2.0**0.5
_SQRT3 = 3.0**0.5


@dataclass(frozen=True)
class GroupEvidence:
    """Deterministic evidence for one named tensorwise Wiener group."""

    name: str
    cosines: tuple[float, float, float]
    rho: float
    beta: float
    g_js: float


@dataclass(frozen=True)
class SpectralEvidence:
    """Deterministic evidence for one matrix-shaped spectral estimate."""

    name: str
    edge: float
    tolerance: float
    kept_rank: int
    total_rank: int
    retained_energy: float
    total_energy: float
    singular_values: tuple[float, ...]
    retained: tuple[bool, ...]


@dataclass(frozen=True)
class CandidateStates:
    """The three preregistered candidate towers and their construction evidence."""

    tower_soup: OrderedDict[str, torch.Tensor]
    wiener_denoise: OrderedDict[str, torch.Tensor]
    spectral_denoise: OrderedDict[str, torch.Tensor]
    groups: tuple[GroupEvidence, ...]
    spectral: tuple[SpectralEvidence, ...]


def wiener_gain(rho: float) -> float:
    """Return the fixed three-replicate Wiener gain for a bounded correlation."""

    if type(rho) is not float or not torch.isfinite(torch.tensor(rho)) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be a finite float in [0, 1]")
    return 3.0 * rho / (1.0 + 2.0 * rho)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    value = float(torch.dot(left.reshape(-1), right.reshape(-1)) / (left_norm * right_norm))
    endpoint_tolerance = 8.0 * torch.finfo(torch.float64).eps
    if abs(value - 1.0) <= endpoint_tolerance:
        return 1.0
    if abs(value + 1.0) <= endpoint_tolerance:
        return -1.0
    return min(max(value, -1.0), 1.0)


def _wiener_evidence(
    name: str,
    updates: tuple[torch.Tensor, ...],
) -> tuple[GroupEvidence, torch.Tensor]:
    cosines = (
        _cosine(updates[0], updates[1]),
        _cosine(updates[0], updates[2]),
        _cosine(updates[1], updates[2]),
    )
    if any(float(torch.linalg.vector_norm(update)) == 0.0 for update in updates):
        rho = 0.0
    else:
        rho = min(max(sum(cosines) / 3.0, 0.0), 1.0)
    beta = wiener_gain(float(rho))
    mean = sum(updates[1:], updates[0].clone()) / 3.0
    residual_energy = sum(
        (float(torch.dot((update - mean).reshape(-1), (update - mean).reshape(-1))))
        for update in updates
    )
    noise = residual_energy / 6.0
    mean_energy = float(torch.dot(mean.reshape(-1), mean.reshape(-1)))
    positive = max(mean_energy - noise, 0.0)
    denominator = positive + noise
    g_js = positive / denominator if denominator > 0.0 else 0.0
    evidence = GroupEvidence(
        name=name,
        cosines=cosines,
        rho=float(rho),
        beta=beta,
        g_js=g_js,
    )
    return evidence, mean * beta


def _spectral_estimate(
    name: str,
    updates: tuple[torch.Tensor, ...],
) -> tuple[SpectralEvidence, torch.Tensor]:
    original_shape = updates[0].shape
    rows = original_shape[0]
    columns = updates[0].numel() // rows
    matrices = tuple(update.reshape(rows, columns) for update in updates)
    mean = sum(matrices[1:], matrices[0].clone()) / 3.0
    contrasts = (
        (matrices[0] - matrices[1]) / _SQRT2,
        (matrices[0] - matrices[2]) / _SQRT2,
        (matrices[1] - matrices[2]) / _SQRT2,
    )
    edge = max(float(torch.linalg.matrix_norm(contrast, ord=2)) for contrast in contrasts) / _SQRT3
    left, singular_values, right = torch.linalg.svd(mean, full_matrices=False)
    sigma1 = float(singular_values[0]) if singular_values.numel() else 0.0
    tolerance = (
        64.0
        * torch.finfo(torch.float64).eps
        * max(rows, columns)
        * max(sigma1, edge, 1.0)
    )
    if sigma1 == 0.0 and edge == 0.0:
        retained = torch.zeros_like(singular_values, dtype=torch.bool)
    else:
        if bool(torch.any(torch.abs(singular_values - edge) <= tolerance)):
            raise ValueError(f"tensor {name!r} has a singular value at the spectral edge")
        retained = singular_values > edge
        # Adjacent values inside the tolerance are one decision cluster. A cluster
        # cannot straddle the edge because that would have triggered the guard above.
        start = 0
        while start < singular_values.numel():
            end = start + 1
            while end < singular_values.numel() and (
                abs(float(singular_values[end - 1] - singular_values[end])) <= tolerance
            ):
                end += 1
            decision = bool(retained[start])
            retained[start:end] = decision
            start = end
    filtered = singular_values * retained.to(singular_values.dtype)
    estimate = ((left * filtered.unsqueeze(0)) @ right).reshape(original_shape)
    energies = singular_values.square()
    total_energy = float(energies.sum())
    retained_energy = float((energies * retained.to(energies.dtype)).sum())
    evidence = SpectralEvidence(
        name=name,
        edge=edge,
        tolerance=tolerance,
        kept_rank=int(retained.sum()),
        total_rank=int(singular_values.numel()),
        retained_energy=retained_energy,
        total_energy=total_energy,
        singular_values=tuple(float(value) for value in singular_values),
        retained=tuple(bool(value) for value in retained),
    )
    return evidence, estimate


def build_cross_seed_candidates(
    initial: object,
    endpoints: object,
) -> CandidateStates:
    """Build the fixed soup, Wiener, and symmetric spectral candidate towers."""

    if not isinstance(initial, OrderedDict) or not initial:
        raise ValueError("initial state must be a non-empty OrderedDict")
    if type(endpoints) is not dict or set(endpoints) != set(_SEEDS):
        raise ValueError("endpoints must contain exactly seeds 17, 29, and 43")
    endpoint_map = cast(dict[int, object], endpoints)
    names = tuple(sorted(initial))
    for seed in _SEEDS:
        state = endpoint_map[seed]
        if not isinstance(state, OrderedDict) or set(state) != set(names):
            raise ValueError(f"seed {seed} state names differ")

    soup: OrderedDict[str, torch.Tensor] = OrderedDict()
    wiener: OrderedDict[str, torch.Tensor] = OrderedDict()
    spectral: OrderedDict[str, torch.Tensor] = OrderedDict()
    group_rows: list[GroupEvidence] = []
    spectral_rows: list[SpectralEvidence] = []
    for name in names:
        initial_value = initial[name]
        if not isinstance(initial_value, torch.Tensor):
            raise TypeError("state values must be tensors")
        values: list[torch.Tensor] = []
        for seed in _SEEDS:
            endpoint_value = cast(OrderedDict[str, object], endpoint_map[seed])[name]
            if not isinstance(endpoint_value, torch.Tensor):
                raise TypeError("state values must be tensors")
            if (
                endpoint_value.shape != initial_value.shape
                or endpoint_value.dtype != initial_value.dtype
            ):
                raise ValueError(f"tensor {name!r} shape or dtype differs")
            values.append(endpoint_value)
        if not initial_value.is_floating_point():
            if any(not torch.equal(initial_value, value) for value in values):
                raise ValueError(f"non-floating tensor {name!r} differs")
            soup[name] = initial_value.detach().cpu().contiguous().clone()
            wiener[name] = initial_value.detach().cpu().contiguous().clone()
            spectral[name] = initial_value.detach().cpu().contiguous().clone()
            continue
        all_values = (initial_value, *values)
        if any(not bool(torch.isfinite(value).all()) for value in all_values):
            raise ValueError(f"floating tensor {name!r} must be finite")
        base = initial_value.detach().cpu().to(torch.float64)
        updates = tuple(value.detach().cpu().to(torch.float64) - base for value in values)
        group, wiener_update = _wiener_evidence(name, updates)
        mean = sum(updates[1:], updates[0].clone()) / 3.0
        if initial_value.ndim >= 2:
            spectral_row, spectral_update = _spectral_estimate(name, updates)
            spectral_rows.append(spectral_row)
        else:
            spectral_update = wiener_update
        soup[name] = (base + mean).to(torch.float32).contiguous()
        wiener[name] = (base + wiener_update).to(torch.float32).contiguous()
        spectral[name] = (base + spectral_update).to(torch.float32).contiguous()
        group_rows.append(group)
    return CandidateStates(
        tower_soup=soup,
        wiener_denoise=wiener,
        spectral_denoise=spectral,
        groups=tuple(group_rows),
        spectral=tuple(spectral_rows),
    )


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
