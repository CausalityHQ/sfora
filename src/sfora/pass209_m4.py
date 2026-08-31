"""Authenticated primitives for the Pass209 M4 objective-rescue measurement."""

from __future__ import annotations

import hashlib
import json
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
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("reference scorer block size must be positive")
    canonical = descriptors.detach().contiguous()
    _descriptor_payload(canonical)
    _validate_examples(examples, int(canonical.shape[0]))
    labels = torch.tensor([example.label for example in examples], dtype=torch.int64)
    rows: list[QueryEvidence] = []
    with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
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

    seed = int.from_bytes(
        hashlib.sha256(b"pass209-m4-objective-bootstrap-v3").digest()[:16], "big"
    )
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
