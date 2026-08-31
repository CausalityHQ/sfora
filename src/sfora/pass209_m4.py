"""Authenticated primitives for the Pass209 M4 objective-rescue measurement."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image
from PIL import __version__ as pillow_version
from PIL import features as pillow_features

DESCRIPTOR_MAGIC = b"SFORA-M4-F32-V1\n"
_HEADER_LENGTH = struct.Struct("<Q")
_NORM_TOLERANCE = 1.0e-6
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SCORER_CONFIGURED = False


@dataclass(frozen=True)
class M4Example:
    """Identity and label authority for one descriptor row."""

    position: int
    example_id: str
    label: int


@dataclass(frozen=True)
class QueryEvidence:
    """Exact row-level retrieval evidence emitted by the reference scorer."""

    query_position: int
    query_example_id: str
    query_label: int
    nearest_position: int
    nearest_example_id: str
    nearest_label: int
    nearest_score_bits: int
    best_same_position: int
    best_same_score_bits: int
    best_different_position: int
    best_different_score_bits: int
    margin_bits: int
    correct: bool


@dataclass(frozen=True)
class ScorerEnvironment:
    """Pinned software and CPU authority for reference retrieval scoring."""

    schema: str
    torch_version: str
    torch_build_config: str
    cpu_architecture: str
    cpu_capability: str
    cpu_isa_flags: tuple[str, ...]
    intraop_threads: int
    interop_threads: int
    deterministic_algorithms: bool
    uv_lock_sha256: str
    pillow_version: str
    libjpeg_version: str
    transformers_version: str


@dataclass(frozen=True)
class RescueRow:
    """One frozen source error and its derived cross-device reachability."""

    error_ordinal: int
    query_label: int
    nearest_label: int
    reachable: bool


@dataclass(frozen=True)
class M4SourceError:
    """One exact row from the frozen SigLIP-so400m error population."""

    error_ordinal: int
    query_position: int
    query_example_id: str
    query_label: int
    nearest_position: int
    nearest_example_id: str
    nearest_label: int


@dataclass(frozen=True)
class M4ObjectiveEvidence:
    """Derived objective rescue evidence over the frozen source errors."""

    source_error_count: int
    reachable_count: int
    universal_three_device_error_count: int
    dinov2_rescued: int
    siglip2_rescued: int
    selecting_rescued: int
    selecting_cpu_correct_on_source_errors: int
    selecting_cpu_cuda_correctness_disagreements: int
    dinov2_rescue_rate: float
    siglip2_rescue_rate: float
    selecting_rescue_rate: float
    dominant_pair: tuple[int, int]
    dominant_pair_count: int
    dominant_pair_dinov2_rescued: int
    dominant_pair_siglip2_rescued: int
    dominant_pair_rescuable: bool
    bootstrap: BootstrapEvidence
    rows: tuple[M4RescueDetail, ...]
    pair_panels: tuple[M4PairPanel, ...]


@dataclass(frozen=True)
class M4RescueDetail:
    """One frozen source error with exact three-device rescue evidence."""

    error_ordinal: int
    query_position: int
    query_example_id: str
    query_label: int
    source_nearest_position: int
    source_nearest_example_id: str
    source_nearest_label: int
    dinov2_correct: bool
    siglip2_correct: bool
    selecting_correct: bool
    selecting_cpu_correct: bool
    reachable: bool
    universal_three_device_error: bool
    dinov2_margin_bits: int
    siglip2_margin_bits: int
    selecting_margin_bits: int


@dataclass(frozen=True)
class M4PairPanel:
    """Exact raw rescue counts for one preregistered confusion pair."""

    pair: tuple[int, int]
    count: int
    dinov2_rescued: int
    siglip2_rescued: int
    selecting_rescued: int
    dinov2_rescue_rate: float
    siglip2_rescue_rate: float
    selecting_rescue_rate: float


@dataclass(frozen=True)
class M4DuplicateEvidence:
    """Exact different-label RGB-record matches for one source-error query."""

    error_ordinal: int
    query_position: int
    query_example_id: str
    query_label: int
    rgb_sha256: str
    matching_positions: tuple[int, ...]
    matching_example_ids: tuple[str, ...]
    matching_labels: tuple[int, ...]


@dataclass(frozen=True)
class M4CpuCudaDivergence:
    """One exact CPU/CUDA nearest-row divergence for a frozen cell."""

    query_position: int
    query_example_id: str
    query_label: int
    cpu_nearest_position: int
    cpu_nearest_example_id: str
    cpu_nearest_label: int
    cpu_nearest_score_bits: int
    cpu_margin_bits: int
    cuda_nearest_position: int
    cuda_nearest_example_id: str
    cuda_nearest_label: int
    cuda_nearest_score_bits: int
    cuda_margin_bits: int


@dataclass(frozen=True)
class M4CellPaths:
    """Three local artifact paths for one terminal M4 cell."""

    receipt: Path
    descriptor: Path
    queries: Path


@dataclass(frozen=True)
class M4Cell:
    """One fully authenticated and independently recomputed M4 cell."""

    spec: M4CellSpec
    receipt: dict[str, object]
    header: M4DescriptorHeader
    descriptors: torch.Tensor
    queries: tuple[QueryEvidence, ...]
    cuda_queries: tuple[QueryEvidence, ...]
    receipt_sha256: str
    descriptor_sha256: str
    query_sha256: str


@dataclass(frozen=True)
class BootstrapEvidence:
    """Canonical summary of the unordered-pair clustered reachable bootstrap."""

    schema: str
    seed: int
    cluster_count: int
    sample_count: int
    observed_share: float
    bootstrap_mean: float
    p2_5: float
    p10: float
    p97_5: float
    samples_sha256: str


@dataclass(frozen=True)
class M4DescriptorHeader:
    """Exact authority header for one normalized descriptor plane."""

    schema: str
    source_revision: str
    source_tree_digest: str
    dataset: str
    dataset_revision: str
    dataset_examples_sha256: str
    dataset_examples_ordered_sha256: str
    split: str
    holdout_classes: tuple[int, ...]
    compute_dtype: str
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


@dataclass(frozen=True)
class M4CellSpec:
    """Frozen identity and reproduction gates for one original fp32 cell."""

    cell: str
    model_name: str
    model_revision: str
    readout: str
    batch_size: int
    processor_image_shape: tuple[int, int]
    expected_rows: int
    descriptor_dimensions: int
    expected_correct: int
    prerequisite_sha256: str
    prerequisite_schema: str
    legacy_descriptor_sha256: str | None


REGISTERED_M4_CELLS: dict[str, M4CellSpec] = {
    "dinov2-large": M4CellSpec(
        cell="dinov2-large",
        model_name="facebook/dinov2-large",
        model_revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
        readout="last_hidden_state_cls",
        batch_size=32,
        processor_image_shape=(224, 224),
        expected_rows=1345,
        descriptor_dimensions=1024,
        expected_correct=1196,
        prerequisite_sha256=("8d01a2aa7cb122e9db0786e40a397a4dfe64ccec9430f6346a80d3b6a3b973a1"),
        prerequisite_schema="sfora-frozen-substrate-screen-v1",
        legacy_descriptor_sha256=None,
    ),
    "siglip2-so400m": M4CellSpec(
        cell="siglip2-so400m",
        model_name="google/siglip2-so400m-patch14-384",
        model_revision="e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
        readout="vision_pooler_output",
        batch_size=8,
        processor_image_shape=(384, 384),
        expected_rows=1345,
        descriptor_dimensions=1152,
        expected_correct=1227,
        prerequisite_sha256=("55c66314017aac208dd76c542f0b2be5f969b18a4ca422e56a15ef14b15b7f9e"),
        prerequisite_schema="sfora-frozen-substrate-screen-v1",
        legacy_descriptor_sha256=None,
    ),
    "siglip-so400m": M4CellSpec(
        cell="siglip-so400m",
        model_name="google/siglip-so400m-patch14-384",
        model_revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        readout="vision_pooler_output",
        batch_size=8,
        processor_image_shape=(384, 384),
        expected_rows=1345,
        descriptor_dimensions=1152,
        expected_correct=1242,
        prerequisite_sha256=("c95088621cdacea5286f1e4634f580ee83d9bed183284f23fc1be9b93bff5089"),
        prerequisite_schema="sfora-frozen-substrate-screen-v2",
        legacy_descriptor_sha256=(
            "4031dc2da90588dcc39005eab92c6c519f3058c581222421ca917501dd3df071"
        ),
    ),
}


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
        "dataset_examples_ordered_sha256",
        "payload_sha256",
    ):
        if _SHA256.fullmatch(_require_string(getattr(header, name), name)) is None:
            raise ValueError(f"descriptor header {name} is not a SHA-256 digest")
    if header.split != "train":
        raise ValueError("descriptor header split differs")
    if header.holdout_classes != tuple(range(82, 98)):
        raise ValueError("descriptor header holdout classes differ")
    if header.compute_dtype != "float32":
        raise ValueError("descriptor header compute dtype differs")
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


def encode_descriptor_file(header: M4DescriptorHeader, descriptors: torch.Tensor) -> bytes:
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
    holdout_classes = value.get("holdout_classes")
    if type(holdout_classes) is not list:
        raise ValueError("descriptor header holdout classes differ")
    value["holdout_classes"] = tuple(holdout_classes)
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


_CELL_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "claim_eligible",
        "source_revision",
        "source_tree_digest",
        "dataset",
        "dataset_revision",
        "dataset_examples_sha256",
        "dataset_examples_ordered_sha256",
        "split",
        "holdout_classes",
        "compute_dtype",
        "error_manifest_sha256",
        "cell",
        "model_name",
        "model_revision",
        "readout",
        "batch_size",
        "query_block",
        "processor_image_shape",
        "descriptor_shape",
        "descriptor_file_sha256",
        "descriptor_payload_sha256",
        "legacy_descriptor_sha256",
        "query_evidence_sha256",
        "correct",
        "cpu_reference_correct",
        "expected_historical_correct",
        "historical_cuda_correct",
        "historical_cuda_errors",
        "legacy_descriptor_passed",
        "historical_count_passed",
        "historical_errors_passed",
        "reproduction_passed",
        "queries",
        "prerequisite_sha256",
        "prerequisite_schema",
        "scorer_environment",
        "cuda_environment",
    }
)
_QUERY_TABLE_KEYS = frozenset(
    {
        "schema",
        "claim_eligible",
        "cell",
        "dataset_examples_sha256",
        "dataset_examples_ordered_sha256",
        "descriptor_file_sha256",
        "query_block",
        "rows",
        "historical_cuda_rows",
    }
)
_QUERY_ROW_KEYS = frozenset(QueryEvidence.__dataclass_fields__)
_SOURCE_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "claim_eligible",
        "source_revision",
        "source_tree_digest",
        "dataset",
        "dataset_revision",
        "dataset_examples_sha256",
        "descriptor_sha256",
        "batch_size",
        "query_block",
        "split",
        "holdout_classes",
        "class_names",
        "cell",
        "model_name",
        "model_revision",
        "error_count",
        "errors",
    }
)
_SOURCE_ERROR_KEYS = frozenset(
    {
        "query_position",
        "query_example_id",
        "query_label",
        "nearest_position",
        "nearest_example_id",
        "nearest_label",
    }
)


def _parse_canonical_object(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"{name} is not canonical")
    return value


def load_m4_source_errors(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, object], tuple[M4SourceError, ...]]:
    """Authenticate and parse the frozen SigLIP-so400m error population."""

    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("M4 source manifest registered digest differs")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("M4 source manifest digest differs")
    value = _parse_canonical_object(payload, name="M4 source manifest")
    spec = REGISTERED_M4_CELLS["siglip-so400m"]
    expected: dict[str, object] = {
        "schema": "sfora-frozen-substrate-errors-v1",
        "claim_eligible": False,
        "dataset": "cars",
        "batch_size": 8,
        "query_block": 32,
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "cell": spec.cell,
        "model_name": spec.model_name,
        "model_revision": spec.model_revision,
        "error_count": 103,
    }
    if set(value) != _SOURCE_MANIFEST_KEYS:
        raise ValueError("M4 source manifest schema differs")
    for key, expected_value in expected.items():
        _require_exact_json(value.get(key), expected_value, name=f"M4 source manifest {key}")
    for name, length in (
        ("source_revision", 40),
        ("source_tree_digest", 64),
        ("dataset_revision", 40),
        ("dataset_examples_sha256", 64),
        ("descriptor_sha256", 64),
    ):
        field = value.get(name)
        if type(field) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", field) is None:
            raise ValueError(f"M4 source manifest {name} differs")
    class_names = value.get("class_names")
    if type(class_names) is not list or len(class_names) != 16:
        raise ValueError("M4 source manifest class names differ")
    for expected_id, row in zip(range(82, 98), class_names, strict=True):
        if (
            type(row) is not dict
            or set(row) != {"id", "name"}
            or type(row.get("id")) is not int
            or row.get("id") != expected_id
            or type(row.get("name")) is not str
            or not row.get("name")
        ):
            raise ValueError("M4 source manifest class names differ")
    raw_errors = value.get("errors")
    if type(raw_errors) is not list or len(raw_errors) != 103:
        raise ValueError("M4 source manifest error cardinality differs")
    errors: list[M4SourceError] = []
    previous = -1
    for ordinal, raw in enumerate(raw_errors):
        if type(raw) is not dict or set(raw) != _SOURCE_ERROR_KEYS:
            raise ValueError("M4 source manifest row schema differs")
        integer_fields = (
            "query_position",
            "query_label",
            "nearest_position",
            "nearest_label",
        )
        string_fields = ("query_example_id", "nearest_example_id")
        if any(type(raw.get(name)) is not int for name in integer_fields) or any(
            type(raw.get(name)) is not str or not raw.get(name) for name in string_fields
        ):
            raise ValueError("M4 source manifest row concrete type differs")
        query_position = raw["query_position"]
        assert isinstance(query_position, int)
        if query_position <= previous:
            raise ValueError("M4 source manifest row order differs")
        previous = query_position
        errors.append(
            M4SourceError(
                error_ordinal=ordinal,
                query_position=query_position,
                query_example_id=raw["query_example_id"],
                query_label=raw["query_label"],
                nearest_position=raw["nearest_position"],
                nearest_example_id=raw["nearest_example_id"],
                nearest_label=raw["nearest_label"],
            )
        )
    return value, tuple(errors)


def _require_exact_json(actual: object, expected: object, *, name: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(f"{name} concrete type differs")
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        if set(actual) != set(expected):
            raise ValueError(f"{name} key set differs")
        for key, value in expected.items():
            _require_exact_json(actual[key], value, name=f"{name}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        if len(actual) != len(expected):
            raise ValueError(f"{name} cardinality differs")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _require_exact_json(left, right, name=f"{name}[{index}]")
    elif actual != expected:
        raise ValueError(f"{name} value differs")


def _parse_query_rows(value: object, *, expected_rows: int) -> tuple[QueryEvidence, ...]:
    if type(value) is not list or len(value) != expected_rows:
        raise ValueError("M4 query row cardinality differs")
    rows: list[QueryEvidence] = []
    integer_fields = {
        "query_position",
        "query_label",
        "nearest_position",
        "nearest_label",
        "nearest_score_bits",
        "best_same_position",
        "best_same_score_bits",
        "best_different_position",
        "best_different_score_bits",
        "margin_bits",
    }
    string_fields = {"query_example_id", "nearest_example_id"}
    for ordinal, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != _QUERY_ROW_KEYS:
            raise ValueError("M4 query row schema differs")
        if any(type(raw[field]) is not int for field in integer_fields):
            raise ValueError("M4 query row integer field differs")
        if any(type(raw[field]) is not str or not raw[field] for field in string_fields):
            raise ValueError("M4 query row string field differs")
        if type(raw["correct"]) is not bool or raw["query_position"] != ordinal:
            raise ValueError("M4 query row order differs")
        try:
            rows.append(QueryEvidence(**raw))
        except TypeError as error:
            raise ValueError("M4 query row fields differ") from error
    return tuple(rows)


def _validate_query_identities(
    rows: tuple[QueryEvidence, ...],
    examples: tuple[M4Example, ...],
    *,
    name: str,
) -> None:
    _validate_examples(examples, len(rows))
    for row, expected in zip(rows, examples, strict=True):
        if (
            row.query_position != expected.position
            or row.query_example_id != expected.example_id
            or row.query_label != expected.label
        ):
            raise ValueError(f"{name} dataset example identity differs")
    for row in rows:
        positions = (
            row.nearest_position,
            row.best_same_position,
            row.best_different_position,
        )
        if any(
            type(position) is not int or not 0 <= position < len(examples) for position in positions
        ):
            raise ValueError(f"{name} gallery position differs")
        nearest = examples[row.nearest_position]
        if (
            row.nearest_example_id != nearest.example_id
            or row.nearest_label != nearest.label
            or row.correct != (row.query_label == row.nearest_label)
            or examples[row.best_same_position].label != row.query_label
            or examples[row.best_different_position].label == row.query_label
        ):
            raise ValueError(f"{name} gallery identity differs")
        for bits in (
            row.nearest_score_bits,
            row.best_same_score_bits,
            row.best_different_score_bits,
            row.margin_bits,
        ):
            if type(bits) is not int or not 0 <= bits <= 0xFFFF_FFFF:
                raise ValueError(f"{name} score bits differ")
            value = struct.unpack("<f", struct.pack("<I", bits))[0]
            if not math.isfinite(value):
                raise ValueError(f"{name} score is nonfinite")


def _load_m4_cell(
    paths: M4CellPaths,
    spec: M4CellSpec,
    *,
    expected_examples: tuple[M4Example, ...],
) -> M4Cell:
    receipt_payload = paths.receipt.read_bytes()
    receipt = _parse_canonical_object(receipt_payload, name="M4 cell receipt")
    if set(receipt) != _CELL_RECEIPT_KEYS:
        raise ValueError("M4 cell receipt schema differs")
    descriptor_payload = paths.descriptor.read_bytes()
    descriptor_sha256 = hashlib.sha256(descriptor_payload).hexdigest()
    header, descriptors = decode_descriptor_file(descriptor_payload)
    query_payload = paths.queries.read_bytes()
    query = _parse_canonical_object(query_payload, name="M4 query evidence")
    if set(query) != _QUERY_TABLE_KEYS:
        raise ValueError("M4 query evidence schema differs")
    queries = _parse_query_rows(query.get("rows"), expected_rows=spec.expected_rows)
    cuda_queries = _parse_query_rows(
        query.get("historical_cuda_rows"), expected_rows=spec.expected_rows
    )
    _validate_query_identities(queries, expected_examples, name="M4 CPU query")
    _validate_query_identities(cuda_queries, expected_examples, name="M4 CUDA query")

    expected_receipt: dict[str, object] = {
        "schema": "sfora-pass209-m4-cell-receipt-v1",
        "claim_eligible": False,
        "dataset": "cars",
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "compute_dtype": "float32",
        "cell": spec.cell,
        "model_name": spec.model_name,
        "model_revision": spec.model_revision,
        "readout": spec.readout,
        "batch_size": spec.batch_size,
        "query_block": 32,
        "processor_image_shape": list(spec.processor_image_shape),
        "descriptor_shape": [spec.expected_rows, spec.descriptor_dimensions],
        "descriptor_file_sha256": descriptor_sha256,
        "descriptor_payload_sha256": header.payload_sha256,
        "query_evidence_sha256": hashlib.sha256(query_payload).hexdigest(),
        "expected_historical_correct": spec.expected_correct,
        "historical_cuda_correct": spec.expected_correct,
        "legacy_descriptor_passed": True,
        "historical_count_passed": True,
        "historical_errors_passed": True,
        "reproduction_passed": True,
        "queries": spec.expected_rows,
        "prerequisite_sha256": spec.prerequisite_sha256,
        "prerequisite_schema": spec.prerequisite_schema,
    }
    for key, expected in expected_receipt.items():
        _require_exact_json(receipt.get(key), expected, name=f"M4 receipt {key}")
    if receipt.get("correct") != receipt.get("cpu_reference_correct"):
        raise ValueError("M4 CPU correct count differs")
    cpu_correct = receipt.get("cpu_reference_correct")
    if type(cpu_correct) is not int or cpu_correct != sum(row.correct for row in queries):
        raise ValueError("M4 CPU correct count differs")
    for name in (
        "source_revision",
        "source_tree_digest",
        "dataset_revision",
        "dataset_examples_sha256",
        "dataset_examples_ordered_sha256",
        "error_manifest_sha256",
        "legacy_descriptor_sha256",
    ):
        value = receipt.get(name)
        length = 40 if name in {"source_revision", "dataset_revision"} else 64
        if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
            raise ValueError(f"M4 receipt {name} differs")
    if (
        spec.legacy_descriptor_sha256 is not None
        and receipt["legacy_descriptor_sha256"] != spec.legacy_descriptor_sha256
    ):
        raise ValueError("M4 legacy descriptor digest differs")
    if (
        type(receipt.get("scorer_environment")) is not dict
        or type(receipt.get("cuda_environment")) is not dict
    ):
        raise ValueError("M4 environment authority differs")
    historical_errors = receipt.get("historical_cuda_errors")
    if type(historical_errors) is not list:
        raise ValueError("M4 historical CUDA errors differ")
    expected_error_count = spec.expected_rows - spec.expected_correct
    if len(historical_errors) != expected_error_count:
        raise ValueError("M4 historical CUDA error count differs")
    historical_error_keys = {
        "query_position",
        "nearest_position",
        "query_label",
        "nearest_label",
    }
    previous_error_position = -1
    expected_historical_errors = []
    for query_row in cuda_queries:
        if not query_row.correct:
            expected_historical_errors.append(
                {
                    "query_position": query_row.query_position,
                    "nearest_position": query_row.nearest_position,
                    "query_label": query_row.query_label,
                    "nearest_label": query_row.nearest_label,
                }
            )
    for row in historical_errors:
        if (
            type(row) is not dict
            or set(row) != historical_error_keys
            or any(type(row.get(key)) is not int for key in historical_error_keys)
            or row["query_position"] <= previous_error_position
            or row["query_position"] == row["nearest_position"]
            or row["query_label"] == row["nearest_label"]
        ):
            raise ValueError("M4 historical CUDA error row differs")
        previous_error_position = row["query_position"]
    if historical_errors != expected_historical_errors:
        raise ValueError("M4 historical CUDA error rows differ from query authority")
    if receipt.get("historical_cuda_correct") != sum(row.correct for row in cuda_queries):
        raise ValueError("M4 historical CUDA correct count differs")

    receipt_holdout_classes = receipt["holdout_classes"]
    assert isinstance(receipt_holdout_classes, list)
    header_expected = {
        "source_revision": receipt["source_revision"],
        "source_tree_digest": receipt["source_tree_digest"],
        "dataset": receipt["dataset"],
        "dataset_revision": receipt["dataset_revision"],
        "dataset_examples_sha256": receipt["dataset_examples_sha256"],
        "dataset_examples_ordered_sha256": receipt["dataset_examples_ordered_sha256"],
        "split": receipt["split"],
        "holdout_classes": tuple(receipt_holdout_classes),
        "compute_dtype": receipt["compute_dtype"],
        "cell": receipt["cell"],
        "model_name": receipt["model_name"],
        "model_revision": receipt["model_revision"],
        "readout": receipt["readout"],
        "rows": spec.expected_rows,
        "dimensions": spec.descriptor_dimensions,
    }
    for key, expected in header_expected.items():
        if getattr(header, key) != expected:
            raise ValueError(f"M4 descriptor header {key} differs")
    expected_query = {
        "schema": "sfora-pass209-m4-query-evidence-v1",
        "claim_eligible": False,
        "cell": spec.cell,
        "dataset_examples_sha256": receipt["dataset_examples_sha256"],
        "dataset_examples_ordered_sha256": receipt["dataset_examples_ordered_sha256"],
        "descriptor_file_sha256": descriptor_sha256,
        "query_block": 32,
    }
    for key, expected in expected_query.items():
        _require_exact_json(query.get(key), expected, name=f"M4 query {key}")
    validate_query_evidence(queries, descriptors, expected_examples, block_size=32)
    return M4Cell(
        spec=spec,
        receipt=receipt,
        header=header,
        descriptors=descriptors,
        queries=queries,
        cuda_queries=cuda_queries,
        receipt_sha256=hashlib.sha256(receipt_payload).hexdigest(),
        descriptor_sha256=descriptor_sha256,
        query_sha256=hashlib.sha256(query_payload).hexdigest(),
    )


def load_m4_cells(
    paths: tuple[M4CellPaths, ...],
    *,
    expected_examples: tuple[M4Example, ...],
) -> tuple[M4Cell, M4Cell, M4Cell]:
    """Authenticate and recompute the exact three registered M4 cells."""

    if type(paths) is not tuple or len(paths) != 3:
        raise ValueError("M4 requires exactly three cell artifact triples")
    expected_cells = tuple(REGISTERED_M4_CELLS)
    cells = tuple(
        _load_m4_cell(
            path_set,
            REGISTERED_M4_CELLS[cell],
            expected_examples=expected_examples,
        )
        for path_set, cell in zip(paths, expected_cells, strict=True)
    )
    shared_fields = (
        "source_revision",
        "source_tree_digest",
        "dataset",
        "dataset_revision",
        "dataset_examples_sha256",
        "dataset_examples_ordered_sha256",
        "error_manifest_sha256",
        "split",
        "holdout_classes",
        "compute_dtype",
        "scorer_environment",
        "cuda_environment",
    )
    reference = cells[0].receipt
    for cell in cells[1:]:
        for name in shared_fields:
            _require_exact_json(
                cell.receipt.get(name),
                reference.get(name),
                name=f"M4 shared {name}",
            )
    return cells[0], cells[1], cells[2]


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


def _configure_reference_scorer() -> None:
    global _SCORER_CONFIGURED  # noqa: PLW0603
    if _SCORER_CONFIGURED:
        return
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        if torch.get_num_interop_threads() != 1:
            raise ValueError("reference scorer requires one interop thread") from error
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("reference scorer thread authority differs")
    _SCORER_CONFIGURED = True


def _cpu_isa_flags() -> tuple[str, ...]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return ()
    for line in cpuinfo.read_text(errors="strict").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in {"flags", "features"}:
            return tuple(sorted(set(value.split())))
    return ()


def configure_reference_scorer(uv_lock_path: Path) -> ScorerEnvironment:
    """Configure scoring and return its complete environment authority."""

    _configure_reference_scorer()
    lock_bytes = uv_lock_path.read_bytes()
    jpeg_version = pillow_features.version("jpg")
    return ScorerEnvironment(
        schema="sfora-pass209-m4-scorer-environment-v1",
        torch_version=str(torch.__version__),
        torch_build_config=torch.__config__.show(),
        cpu_architecture=platform.machine(),
        cpu_capability=torch.backends.cpu.get_cpu_capability(),
        cpu_isa_flags=_cpu_isa_flags(),
        intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        uv_lock_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        pillow_version=pillow_version,
        libjpeg_version="unavailable" if jpeg_version is None else jpeg_version,
        transformers_version=str(transformers.__version__),
    )


def _validate_examples(examples: tuple[M4Example, ...], rows: int) -> None:
    if len(examples) != rows:
        raise ValueError("example authority cardinality differs from descriptors")
    seen_ids: set[str] = set()
    label_counts: dict[int, int] = {}
    for position, example in enumerate(examples):
        if type(example.position) is not int or example.position != position:
            raise ValueError("example authority position differs")
        if type(example.example_id) is not str or not example.example_id:
            raise ValueError("example authority ID differs")
        if example.example_id in seen_ids:
            raise ValueError("example authority IDs must be unique")
        if type(example.label) is not int or example.label < 0:
            raise ValueError("example authority label differs")
        seen_ids.add(example.example_id)
        label_counts[example.label] = label_counts.get(example.label, 0) + 1
    if any(count < 2 for count in label_counts.values()):
        raise ValueError("every query label requires another gallery row")


def _float32_bits(value: torch.Tensor) -> int:
    if value.dtype != torch.float32 or value.numel() != 1:
        raise ValueError("score must be one float32 value")
    return int(struct.unpack("<I", struct.pack("<f", float(value.item())))[0])


def _first_argmax(scores: torch.Tensor) -> int:
    if scores.ndim != 1 or not bool(torch.isfinite(scores).any()):
        raise ValueError("retrieval candidate scores are empty")
    # torch.argmax returns the first maximal ordinal, which is the registered tie rule.
    return int(torch.argmax(scores).item())


def score_descriptor_plane(
    descriptors: torch.Tensor,
    examples: tuple[M4Example, ...],
    *,
    block_size: int = 32,
) -> tuple[QueryEvidence, ...]:
    """Score one authenticated descriptor plane using the CPU fp32 authority."""

    _configure_reference_scorer()
    if descriptors.device.type != "cpu" or descriptors.dtype != torch.float32:
        raise ValueError("reference scorer requires CPU float32 descriptors")
    _descriptor_payload(descriptors.detach().contiguous())
    return _score_descriptor_plane_on_device(
        torch.nn.functional.normalize(descriptors, dim=-1),
        examples,
        block_size=block_size,
    )


def score_descriptor_plane_cuda(
    descriptors: torch.Tensor,
    examples: tuple[M4Example, ...],
    *,
    block_size: int = 32,
) -> tuple[QueryEvidence, ...]:
    """Score one descriptor plane with the historical CUDA fp32 arithmetic."""

    if descriptors.device.type != "cuda" or descriptors.dtype != torch.float32:
        raise ValueError("historical scorer requires CUDA float32 descriptors")
    return _score_descriptor_plane_on_device(
        torch.nn.functional.normalize(descriptors, dim=-1),
        examples,
        block_size=block_size,
    )


def _score_descriptor_plane_on_device(
    descriptors: torch.Tensor,
    examples: tuple[M4Example, ...],
    *,
    block_size: int,
) -> tuple[QueryEvidence, ...]:
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("reference scorer block size must be positive")
    canonical = descriptors.detach().contiguous()
    if canonical.ndim != 2 or not bool(torch.isfinite(canonical).all()):
        raise ValueError("descriptor plane differs")
    norms = torch.linalg.vector_norm(canonical, dim=1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=_NORM_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("descriptor norms differ")
    _validate_examples(examples, int(canonical.shape[0]))
    labels = torch.tensor(
        [example.label for example in examples],
        dtype=torch.int64,
        device=canonical.device,
    )
    rows: list[QueryEvidence] = []
    with torch.inference_mode(), torch.autocast(device_type=canonical.device.type, enabled=False):
        for start in range(0, len(examples), block_size):
            stop = min(start + block_size, len(examples))
            scores = canonical[start:stop] @ canonical.T
            if scores.dtype != torch.float32 or not bool(torch.isfinite(scores).all()):
                raise ValueError("reference scorer produced invalid scores")
            for offset, query_position in enumerate(range(start, stop)):
                row = scores[offset].clone()
                row[query_position] = -torch.inf
                nearest_position = _first_argmax(row)
                same_scores = row.masked_fill(labels != labels[query_position], -torch.inf)
                different_scores = row.masked_fill(labels == labels[query_position], -torch.inf)
                best_same_position = _first_argmax(same_scores)
                best_different_position = _first_argmax(different_scores)
                margin = same_scores[best_same_position] - different_scores[best_different_position]
                query = examples[query_position]
                nearest = examples[nearest_position]
                rows.append(
                    QueryEvidence(
                        query_position=query_position,
                        query_example_id=query.example_id,
                        query_label=query.label,
                        nearest_position=nearest_position,
                        nearest_example_id=nearest.example_id,
                        nearest_label=nearest.label,
                        nearest_score_bits=_float32_bits(row[nearest_position]),
                        best_same_position=best_same_position,
                        best_same_score_bits=_float32_bits(same_scores[best_same_position]),
                        best_different_position=best_different_position,
                        best_different_score_bits=_float32_bits(
                            different_scores[best_different_position]
                        ),
                        margin_bits=_float32_bits(margin),
                        correct=query.label == nearest.label,
                    )
                )
    return tuple(rows)


def validate_query_evidence(
    evidence: tuple[QueryEvidence, ...],
    descriptors: torch.Tensor,
    examples: tuple[M4Example, ...],
    *,
    block_size: int = 32,
) -> None:
    """Recompute and require exact query-evidence identity."""

    expected = score_descriptor_plane(descriptors, examples, block_size=block_size)
    if evidence != expected:
        raise ValueError("query evidence differs from descriptor authority")


def rgb_record_sha256(image: Image.Image) -> str:
    """Hash the exact registered RGB record for gallery duplicate detection."""

    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("duplicate authority requires a materialized RGB image")
    width, height = image.size
    if not 0 < width <= 0xFFFF_FFFF or not 0 < height <= 0xFFFF_FFFF:
        raise ValueError("RGB image dimensions are outside u32 authority")
    rgb = image.tobytes("raw", "RGB")
    if len(rgb) != width * height * 3:
        raise ValueError("RGB image byte count differs from dimensions")
    framed = b"SFORA-M4-RGB-V1\n" + struct.pack("<II", width, height) + rgb
    return hashlib.sha256(framed).hexdigest()


def audit_exact_rgb_duplicates(
    *,
    source_errors: tuple[M4SourceError, ...],
    examples: tuple[M4Example, ...],
    rgb_sha256: tuple[str, ...],
) -> tuple[M4DuplicateEvidence, ...]:
    """Audit every source query against all different-label RGB records."""

    _validate_examples(examples, len(examples))
    if len(source_errors) != 103 or len(rgb_sha256) != len(examples):
        raise ValueError("M4 duplicate-audit cardinality differs")
    by_digest: dict[str, list[M4Example]] = {}
    for example, digest in zip(examples, rgb_sha256, strict=True):
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ValueError("M4 RGB digest authority differs")
        by_digest.setdefault(digest, []).append(example)
    rows: list[M4DuplicateEvidence] = []
    for ordinal, source in enumerate(source_errors):
        if source.error_ordinal != ordinal or not 0 <= source.query_position < len(examples):
            raise ValueError("M4 duplicate source-error authority differs")
        query = examples[source.query_position]
        if query.example_id != source.query_example_id or query.label != source.query_label:
            raise ValueError("M4 duplicate query identity differs")
        if not 0 <= source.nearest_position < len(examples):
            raise ValueError("M4 duplicate nearest identity differs")
        nearest = examples[source.nearest_position]
        if nearest.example_id != source.nearest_example_id or nearest.label != source.nearest_label:
            raise ValueError("M4 duplicate nearest identity differs")
        digest = rgb_sha256[source.query_position]
        matches = tuple(
            candidate
            for candidate in by_digest[digest]
            if candidate.position != query.position and candidate.label != query.label
        )
        rows.append(
            M4DuplicateEvidence(
                error_ordinal=ordinal,
                query_position=query.position,
                query_example_id=query.example_id,
                query_label=query.label,
                rgb_sha256=digest,
                matching_positions=tuple(candidate.position for candidate in matches),
                matching_example_ids=tuple(candidate.example_id for candidate in matches),
                matching_labels=tuple(candidate.label for candidate in matches),
            )
        )
    return tuple(rows)


def dominant_pair_rescuable(*, rescued: int, count: int) -> bool:
    """Apply the descriptive materiality floor to the exact 63-row census."""

    if type(rescued) is not int or type(count) is not int:
        raise ValueError("dominant-pair counts must be concrete integers")
    if count != 63:
        raise ValueError("dominant-pair census must contain exactly 63 rows")
    if not 0 <= rescued <= count:
        raise ValueError("dominant-pair rescued count is outside the census")
    return rescued / count >= 0.25


def _validate_rescue_rows(rows: tuple[RescueRow, ...]) -> None:
    if not rows:
        raise ValueError("reachable bootstrap requires source-error rows")
    for ordinal, row in enumerate(rows):
        if type(row.error_ordinal) is not int or row.error_ordinal != ordinal:
            raise ValueError("source-error ordinal authority differs")
        if type(row.query_label) is not int or type(row.nearest_label) is not int:
            raise ValueError("source-error labels must be concrete integers")
        if row.query_label < 0 or row.nearest_label < 0:
            raise ValueError("source-error labels must be nonnegative")
        if row.query_label == row.nearest_label:
            raise ValueError("source-error pair labels must be distinct")
        if type(row.reachable) is not bool:
            raise ValueError("source-error reachable flag must be a concrete boolean")


def bootstrap_reachable(rows: tuple[RescueRow, ...]) -> BootstrapEvidence:
    """Bootstrap global reachability over whole unordered confusion-pair blocks."""

    _validate_rescue_rows(rows)
    grouped: dict[tuple[int, int], list[bool]] = {}
    for row in rows:
        pair = (
            min(row.query_label, row.nearest_label),
            max(row.query_label, row.nearest_label),
        )
        grouped.setdefault(pair, []).append(row.reachable)
    blocks = tuple(grouped[pair] for pair in sorted(grouped))
    counts = np.asarray([len(block) for block in blocks], dtype=np.int64)
    successes = np.asarray([sum(block) for block in blocks], dtype=np.int64)
    if bool(np.any(counts <= 0)):
        raise ValueError("reachable bootstrap contains an empty pair block")

    seed = int.from_bytes(hashlib.sha256(b"pass209-m4-objective-bootstrap-v3").digest()[:16], "big")
    generator = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(10_000, dtype=np.float64)
    for index in range(samples.size):
        selected = generator.integers(0, len(blocks), size=len(blocks))
        denominator = int(counts[selected].sum())
        if denominator <= 0:
            raise ValueError("reachable bootstrap replicate denominator is empty")
        samples[index] = float(successes[selected].sum()) / denominator
    percentiles = np.percentile(samples, (2.5, 10.0, 97.5), method="inverted_cdf")
    sample_bytes = samples.astype("<f8", copy=False).tobytes(order="C")
    return BootstrapEvidence(
        schema="sfora-pass209-m4-pair-bootstrap-v1",
        seed=seed,
        cluster_count=len(blocks),
        sample_count=int(samples.size),
        observed_share=sum(row.reachable for row in rows) / len(rows),
        bootstrap_mean=float(samples.mean()),
        p2_5=float(percentiles[0]),
        p10=float(percentiles[1]),
        p97_5=float(percentiles[2]),
        samples_sha256=hashlib.sha256(sample_bytes).hexdigest(),
    )


def _query_rows_by_position(
    rows: tuple[QueryEvidence, ...], *, name: str
) -> dict[int, QueryEvidence]:
    result: dict[int, QueryEvidence] = {}
    for ordinal, row in enumerate(rows):
        if type(row.query_position) is not int or row.query_position != ordinal:
            raise ValueError(f"{name} query positions differ")
        if row.query_position in result:
            raise ValueError(f"{name} query positions differ")
        result[row.query_position] = row
    return result


def analyze_rescue_evidence(
    *,
    source_errors: tuple[M4SourceError, ...],
    dinov2_queries: tuple[QueryEvidence, ...],
    siglip2_queries: tuple[QueryEvidence, ...],
    selecting_queries: tuple[QueryEvidence, ...],
    selecting_cuda_queries: tuple[QueryEvidence, ...],
) -> M4ObjectiveEvidence:
    """Recompute P2/P3 evidence from exact source errors and query tables."""

    if len(source_errors) != 103:
        raise ValueError("M4 source-error population must contain exactly 103 rows")
    query_tables = {
        "DINOv2": _query_rows_by_position(dinov2_queries, name="DINOv2"),
        "SigLIP2": _query_rows_by_position(siglip2_queries, name="SigLIP2"),
        "selecting CPU": _query_rows_by_position(selecting_queries, name="selecting CPU"),
        "selecting CUDA": _query_rows_by_position(selecting_cuda_queries, name="selecting CUDA"),
    }
    rescue_rows: list[RescueRow] = []
    dinov2_rescued = 0
    siglip2_rescued = 0
    selecting_rescued = 0
    selecting_cpu_correct_on_source_errors = 0
    selecting_cpu_cuda_correctness_disagreements = 0
    pair_counts: dict[tuple[int, int], int] = {}
    pair_dinov2: dict[tuple[int, int], int] = {}
    pair_siglip2: dict[tuple[int, int], int] = {}
    pair_selecting: dict[tuple[int, int], int] = {}
    details: list[M4RescueDetail] = []
    previous_position = -1
    for ordinal, source in enumerate(source_errors):
        if (
            type(source.error_ordinal) is not int
            or source.error_ordinal != ordinal
            or type(source.query_position) is not int
            or source.query_position <= previous_position
            or type(source.query_example_id) is not str
            or not source.query_example_id
            or type(source.query_label) is not int
            or type(source.nearest_position) is not int
            or source.nearest_position < 0
            or type(source.nearest_example_id) is not str
            or not source.nearest_example_id
            or type(source.nearest_label) is not int
            or source.query_label == source.nearest_label
        ):
            raise ValueError("M4 source-error row authority differs")
        previous_position = source.query_position
        selected: dict[str, QueryEvidence] = {}
        for name, table in query_tables.items():
            try:
                query = table[source.query_position]
            except KeyError as error:
                raise ValueError(f"{name} query table misses a source error") from error
            if (
                query.query_example_id != source.query_example_id
                or query.query_label != source.query_label
            ):
                raise ValueError(f"{name} source-query identity differs")
            selected[name] = query
        dino_correct = selected["DINOv2"].correct
        siglip2_correct = selected["SigLIP2"].correct
        selecting_cpu_correct = selected["selecting CPU"].correct
        selecting_correct = selected["selecting CUDA"].correct
        if type(dino_correct) is not bool or type(siglip2_correct) is not bool:
            raise ValueError("M4 rescue correctness must be concrete booleans")
        if type(selecting_cpu_correct) is not bool:
            raise ValueError("M4 selecting CPU correctness must be a concrete boolean")
        if type(selecting_correct) is not bool or selecting_correct:
            raise ValueError("M4 selecting device must remain incorrect")
        pair = (
            min(source.query_label, source.nearest_label),
            max(source.query_label, source.nearest_label),
        )
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        pair_dinov2[pair] = pair_dinov2.get(pair, 0) + int(dino_correct)
        pair_siglip2[pair] = pair_siglip2.get(pair, 0) + int(siglip2_correct)
        pair_selecting[pair] = pair_selecting.get(pair, 0) + int(selecting_correct)
        dinov2_rescued += int(dino_correct)
        siglip2_rescued += int(siglip2_correct)
        selecting_rescued += int(selecting_correct)
        selecting_cpu_correct_on_source_errors += int(selecting_cpu_correct)
        selecting_cpu_cuda_correctness_disagreements += int(
            selecting_cpu_correct != selecting_correct
        )
        rescue_rows.append(
            RescueRow(
                error_ordinal=ordinal,
                query_label=source.query_label,
                nearest_label=source.nearest_label,
                reachable=dino_correct or siglip2_correct,
            )
        )
        details.append(
            M4RescueDetail(
                error_ordinal=ordinal,
                query_position=source.query_position,
                query_example_id=source.query_example_id,
                query_label=source.query_label,
                source_nearest_position=source.nearest_position,
                source_nearest_example_id=source.nearest_example_id,
                source_nearest_label=source.nearest_label,
                dinov2_correct=dino_correct,
                siglip2_correct=siglip2_correct,
                selecting_correct=selecting_correct,
                selecting_cpu_correct=selecting_cpu_correct,
                reachable=dino_correct or siglip2_correct,
                universal_three_device_error=not (dino_correct or siglip2_correct),
                dinov2_margin_bits=selected["DINOv2"].margin_bits,
                siglip2_margin_bits=selected["SigLIP2"].margin_bits,
                selecting_margin_bits=selected["selecting CUDA"].margin_bits,
            )
        )
    dominant_pair = (82, 83)
    dominant_count = pair_counts.get(dominant_pair, 0)
    if dominant_count != 63:
        raise ValueError("M4 dominant-pair census differs")
    dino_dominant = pair_dinov2.get(dominant_pair, 0)
    siglip2_dominant = pair_siglip2.get(dominant_pair, 0)
    reachable_count = sum(row.reachable for row in rescue_rows)
    bootstrap = bootstrap_reachable(tuple(rescue_rows))
    registered_pairs = ((82, 83), (85, 86), (89, 90))
    if any(pair_counts.get(pair, 0) <= 0 for pair in registered_pairs):
        raise ValueError("M4 registered pair panel is empty")
    pair_panels = tuple(
        M4PairPanel(
            pair=pair,
            count=pair_counts.get(pair, 0),
            dinov2_rescued=pair_dinov2.get(pair, 0),
            siglip2_rescued=pair_siglip2.get(pair, 0),
            selecting_rescued=pair_selecting.get(pair, 0),
            dinov2_rescue_rate=(pair_dinov2.get(pair, 0) / pair_counts[pair]),
            siglip2_rescue_rate=(pair_siglip2.get(pair, 0) / pair_counts[pair]),
            selecting_rescue_rate=(pair_selecting.get(pair, 0) / pair_counts[pair]),
        )
        for pair in registered_pairs
    )
    return M4ObjectiveEvidence(
        source_error_count=len(source_errors),
        reachable_count=reachable_count,
        universal_three_device_error_count=len(source_errors) - reachable_count,
        dinov2_rescued=dinov2_rescued,
        siglip2_rescued=siglip2_rescued,
        selecting_rescued=selecting_rescued,
        selecting_cpu_correct_on_source_errors=selecting_cpu_correct_on_source_errors,
        selecting_cpu_cuda_correctness_disagreements=(selecting_cpu_cuda_correctness_disagreements),
        dinov2_rescue_rate=dinov2_rescued / len(source_errors),
        siglip2_rescue_rate=siglip2_rescued / len(source_errors),
        selecting_rescue_rate=selecting_rescued / len(source_errors),
        dominant_pair=dominant_pair,
        dominant_pair_count=dominant_count,
        dominant_pair_dinov2_rescued=dino_dominant,
        dominant_pair_siglip2_rescued=siglip2_dominant,
        dominant_pair_rescuable=(
            dominant_pair_rescuable(rescued=dino_dominant, count=dominant_count)
            or dominant_pair_rescuable(
                rescued=siglip2_dominant,
                count=dominant_count,
            )
        ),
        bootstrap=bootstrap,
        rows=tuple(details),
        pair_panels=pair_panels,
    )


def classify_m3_transfer(
    ratios: tuple[float | None, float | None, float | None],
) -> str:
    """Classify the exact three M3 ratios using the frozen cut points."""

    if type(ratios) is not tuple or len(ratios) != 3:
        raise ValueError("M3 requires exactly three ratios")
    for value in ratios:
        if value is not None and (type(value) is not float or not math.isfinite(value)):
            raise ValueError("M3 ratios must be finite concrete floats or null")
    if any(value is None for value in ratios):
        return "T-undefined"
    defined = tuple(value for value in ratios if value is not None)
    if all(value >= 0.50 for value in defined):
        return "T-high"
    if all(value <= 0.35 for value in defined):
        return "T-low"
    return "T-mid"


def adapt_m3_m4(*, m3_state: str, reachable_p10: float, dominant_rescuable: bool) -> str:
    """Apply the frozen first-match M3/M4 broad-family adapter."""

    if m3_state not in {"T-high", "T-low", "T-mid", "T-undefined"}:
        raise ValueError("M3 state differs")
    if (
        type(reachable_p10) is not float
        or not math.isfinite(reachable_p10)
        or not 0.0 <= reachable_p10 <= 1.0
        or type(dominant_rescuable) is not bool
    ):
        raise ValueError("M4 adapter state differs")
    if m3_state == "T-low" and reachable_p10 >= 0.25:
        return "F4-TRANSFER"
    if m3_state == "T-high" and dominant_rescuable:
        return "F4-CAPACITY"
    return "F4-NONE"


def _cpu_cuda_divergences(cell: M4Cell) -> tuple[M4CpuCudaDivergence, ...]:
    if len(cell.queries) != len(cell.cuda_queries):
        raise ValueError("M4 CPU/CUDA query cardinality differs")
    rows: list[M4CpuCudaDivergence] = []
    for cpu, cuda in zip(cell.queries, cell.cuda_queries, strict=True):
        if (
            cpu.query_position != cuda.query_position
            or cpu.query_example_id != cuda.query_example_id
            or cpu.query_label != cuda.query_label
        ):
            raise ValueError("M4 CPU/CUDA query identity differs")
        if cpu.nearest_position == cuda.nearest_position:
            continue
        rows.append(
            M4CpuCudaDivergence(
                query_position=cpu.query_position,
                query_example_id=cpu.query_example_id,
                query_label=cpu.query_label,
                cpu_nearest_position=cpu.nearest_position,
                cpu_nearest_example_id=cpu.nearest_example_id,
                cpu_nearest_label=cpu.nearest_label,
                cpu_nearest_score_bits=cpu.nearest_score_bits,
                cpu_margin_bits=cpu.margin_bits,
                cuda_nearest_position=cuda.nearest_position,
                cuda_nearest_example_id=cuda.nearest_example_id,
                cuda_nearest_label=cuda.nearest_label,
                cuda_nearest_score_bits=cuda.nearest_score_bits,
                cuda_margin_bits=cuda.margin_bits,
            )
        )
    return tuple(rows)


def m4_receipt_bytes(
    *,
    cells: tuple[M4Cell, M4Cell, M4Cell],
    source_errors: tuple[M4SourceError, ...],
    error_manifest_sha256: str,
    examples: tuple[M4Example, ...],
    rgb_sha256: tuple[str, ...],
) -> bytes:
    """Recompute and serialize the canonical objective-rescue receipt."""

    if _SHA256.fullmatch(error_manifest_sha256) is None:
        raise ValueError("M4 error-manifest digest differs")
    expected_cells = tuple(REGISTERED_M4_CELLS)
    if tuple(cell.spec.cell for cell in cells) != expected_cells:
        raise ValueError("M4 receipt cell order differs")
    _validate_examples(examples, cells[0].spec.expected_rows)
    for cell in cells:
        if cell.receipt.get("error_manifest_sha256") != error_manifest_sha256:
            raise ValueError("M4 receipt error-manifest binding differs")
        if cell.receipt.get("reproduction_passed") is not True:
            raise ValueError("M4 receipt rejects a failed reproduction")
    selecting_historical_errors = cells[2].receipt.get("historical_cuda_errors")
    if type(selecting_historical_errors) is not list:
        raise ValueError("M4 selecting historical errors differ")
    historical_positions = tuple(
        row.get("query_position") for row in selecting_historical_errors if type(row) is dict
    )
    source_positions = tuple(row.query_position for row in source_errors)
    if historical_positions != source_positions:
        raise ValueError("M4 selecting historical error population differs")
    objective = analyze_rescue_evidence(
        source_errors=source_errors,
        dinov2_queries=cells[0].queries,
        siglip2_queries=cells[1].queries,
        selecting_queries=cells[2].queries,
        selecting_cuda_queries=cells[2].cuda_queries,
    )
    duplicates = audit_exact_rgb_duplicates(
        source_errors=source_errors,
        examples=examples,
        rgb_sha256=rgb_sha256,
    )
    duplicate_query_count = sum(bool(row.matching_positions) for row in duplicates)
    source = cells[0].receipt
    value: dict[str, object] = {
        "schema": "sfora-pass209-m4-objective-rescue-v1",
        "claim_eligible": False,
        "source_revision": source["source_revision"],
        "source_tree_digest": source["source_tree_digest"],
        "dataset": "cars",
        "dataset_revision": source["dataset_revision"],
        "dataset_examples_sha256": source["dataset_examples_sha256"],
        "dataset_examples_ordered_sha256": source["dataset_examples_ordered_sha256"],
        "error_manifest_sha256": error_manifest_sha256,
        "cells": [
            {
                "cell": cell.spec.cell,
                "receipt_sha256": cell.receipt_sha256,
                "descriptor_sha256": cell.descriptor_sha256,
                "query_sha256": cell.query_sha256,
                "cpu_cuda_nearest_divergence_count": len(_cpu_cuda_divergences(cell)),
                "cpu_cuda_nearest_divergences": [
                    asdict(row) for row in _cpu_cuda_divergences(cell)
                ],
            }
            for cell in cells
        ],
        "duplicate_query_count": duplicate_query_count,
        "duplicates": [asdict(row) for row in duplicates],
        "objective": asdict(objective),
        "gates": {
            "reachable_p10_minimum": 0.25,
            "dominant_pair_rescue_minimum": 0.25,
        },
        "passed": (objective.bootstrap.p10 >= 0.25 or objective.dominant_pair_rescuable),
    }
    return canonical_json_bytes(value)
