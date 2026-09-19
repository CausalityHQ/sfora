"""Immutable factorized residual ANN artifacts and search boundaries."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import heapq
import json
import mmap
import os
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

FactorizedResidualMetric = Literal["squared_l2"]
FactorizedResidualVectorDtype = Literal["uint8", "float32"]


@dataclass(frozen=True, slots=True)
class FactorizedResidualSpec:
    """Dataset-independent metric, codec, routing, and result geometry."""

    metric: FactorizedResidualMetric
    vector_dtype: FactorizedResidualVectorDtype
    dimensions: int
    list_count: int
    subquantizers: int
    bits_per_subquantizer: int
    probe_count: int
    shortlist_width: int
    return_width: int

    def __post_init__(self) -> None:
        integer_values = (
            self.dimensions,
            self.list_count,
            self.subquantizers,
            self.bits_per_subquantizer,
            self.probe_count,
            self.shortlist_width,
            self.return_width,
        )
        if (
            type(self.metric) is not str
            or self.metric != "squared_l2"
            or type(self.vector_dtype) is not str
            or self.vector_dtype not in ("uint8", "float32")
            or any(type(value) is not int for value in integer_values)
            or self.dimensions < 1
            or self.list_count < 1
            or self.subquantizers < 1
            or self.dimensions % self.subquantizers != 0
            or not 1 <= self.bits_per_subquantizer <= 8
            or not 1 <= self.probe_count <= self.list_count
            or self.shortlist_width < 1
            or not 1 <= self.return_width <= self.shortlist_width
        ):
            raise ValueError("factorized residual spec differs")

    @property
    def codebook_size(self) -> int:
        """Return the number of codewords in each subquantizer."""

        return 1 << self.bits_per_subquantizer

    @property
    def subvector_dimensions(self) -> int:
        """Return dimensions represented by each subquantizer."""

        return self.dimensions // self.subquantizers

    @property
    def code_bytes(self) -> int:
        """Return densely packed code bytes per posting."""

        return (self.subquantizers * self.bits_per_subquantizer + 7) // 8


def _owned_read_only(value: NDArray[np.generic]) -> NDArray[np.generic]:
    contiguous = np.ascontiguousarray(value)
    owned_bytes = bytes(memoryview(contiguous).cast("B"))
    return np.frombuffer(owned_bytes, dtype=contiguous.dtype).reshape(contiguous.shape)


@dataclass(frozen=True, slots=True)
class FactorizedResidualComponents:
    """Owned routing centroids and residual product-quantizer codebooks."""

    spec: FactorizedResidualSpec
    coarse_centroids: NDArray[np.float32]
    pq_codebooks: NDArray[np.float32]

    def __post_init__(self) -> None:
        if (
            type(self.spec) is not FactorizedResidualSpec
            or type(self.coarse_centroids) is not np.ndarray
            or self.coarse_centroids.dtype.str != "<f4"
            or self.coarse_centroids.shape != (self.spec.list_count, self.spec.dimensions)
            or not bool(np.isfinite(self.coarse_centroids).all())
            or type(self.pq_codebooks) is not np.ndarray
            or self.pq_codebooks.dtype.str != "<f4"
            or self.pq_codebooks.shape
            != (
                self.spec.subquantizers,
                self.spec.codebook_size,
                self.spec.subvector_dimensions,
            )
            or not bool(np.isfinite(self.pq_codebooks).all())
        ):
            raise ValueError("factorized residual components differ")
        object.__setattr__(self, "coarse_centroids", _owned_read_only(self.coarse_centroids))
        object.__setattr__(self, "pq_codebooks", _owned_read_only(self.pq_codebooks))


@dataclass(frozen=True, slots=True)
class FactorizedResidualPostings:
    """Owned list offsets, dense internal IDs, and packed PQ codes."""

    spec: FactorizedResidualSpec
    offsets: NDArray[np.uint64]
    ids: NDArray[np.uint32]
    codes: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if (
            type(self.spec) is not FactorizedResidualSpec
            or type(self.offsets) is not np.ndarray
            or self.offsets.dtype.str != "<u8"
            or self.offsets.shape != (self.spec.list_count + 1,)
            or int(self.offsets[0]) != 0
            or bool((self.offsets[1:] < self.offsets[:-1]).any())
            or type(self.ids) is not np.ndarray
            or self.ids.dtype.str != "<u4"
            or self.ids.ndim != 1
            or self.ids.shape[0] < 1
            or int(self.offsets[-1]) != self.ids.shape[0]
            or type(self.codes) is not np.ndarray
            or self.codes.dtype != np.dtype(np.uint8)
            or self.codes.shape != (self.ids.shape[0], self.spec.code_bytes)
            or not self._codes_have_canonical_trailing_bits()
            or not self._ids_are_dense_permutation()
        ):
            raise ValueError("factorized residual postings differ")
        object.__setattr__(self, "offsets", _owned_read_only(self.offsets))
        object.__setattr__(self, "ids", _owned_read_only(self.ids))
        object.__setattr__(self, "codes", _owned_read_only(self.codes))

    @property
    def rows(self) -> int:
        """Return the number of posting rows."""

        return int(self.ids.shape[0])

    def _ids_are_dense_permutation(self) -> bool:
        rows = self.ids.shape[0]
        seen = bytearray((rows + 7) // 8)
        for raw_value in self.ids:
            value = int(raw_value)
            if value >= rows:
                return False
            byte_index = value >> 3
            mask = 1 << (value & 7)
            if seen[byte_index] & mask:
                return False
            seen[byte_index] |= mask
        return True

    def _codes_have_canonical_trailing_bits(self) -> bool:
        trailing_bits = (
            self.spec.code_bytes * 8 - self.spec.subquantizers * self.spec.bits_per_subquantizer
        )
        if trailing_bits == 0:
            return True
        trailing_mask = ((1 << trailing_bits) - 1) << (8 - trailing_bits)
        for begin in range(0, self.codes.shape[0], 8 << 20):
            if bool((self.codes[begin : begin + (8 << 20), -1] & trailing_mask).any()):
                return False
        return True


@dataclass(frozen=True, slots=True)
class VectorStoreIdentity:
    """Exact immutable vector-file content and row-layout authority."""

    sha256: str
    logical_bytes: int
    physical_bytes: int
    rows: int
    dimensions: int
    dtype: FactorizedResidualVectorDtype
    header_bytes: int
    row_stride: int
    zero_padding_bytes: int
    generation: str

    def __post_init__(self) -> None:
        integers = (
            self.logical_bytes,
            self.physical_bytes,
            self.rows,
            self.dimensions,
            self.header_bytes,
            self.row_stride,
            self.zero_padding_bytes,
        )
        item_bytes = 1 if self.dtype == "uint8" else 4
        expected_stride = self.dimensions * item_bytes
        expected_payload = self.rows * expected_stride
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
            or any(type(value) is not int for value in integers)
            or self.rows < 1
            or self.rows >= 2**32
            or self.dimensions < 1
            or type(self.dtype) is not str
            or self.dtype not in ("uint8", "float32")
            or self.header_bytes != 8
            or self.row_stride != expected_stride
            or self.logical_bytes != self.header_bytes + expected_payload
            or self.zero_padding_bytes < 0
            or self.physical_bytes != self.logical_bytes + self.zero_padding_bytes
            or type(self.generation) is not str
            or not self.generation
        ):
            raise ValueError("vector store identity differs")

    @property
    def payload_bytes(self) -> int:
        """Return logical vector payload bytes after the header."""

        return self.logical_bytes - self.header_bytes


_ARTIFACT_ROLES = (
    "coarse.f32",
    "pq.f32",
    "offsets.u64",
    "ids.u32",
    "codes.u8",
    "norms.u8",
    "norm-low.f32",
    "norm-scale.f32",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _sha256_fd(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        block = os.pread(fd, min(8 << 20, size - offset), offset)
        if not block:
            raise ValueError("factorized residual artifact differs")
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


def _write_array_bounded(path: Path, value: NDArray[np.generic]) -> None:
    raw = memoryview(value).cast("B")
    with path.open("wb") as stream:
        for offset in range(0, len(raw), 8 << 20):
            stream.write(raw[offset : offset + (8 << 20)])
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _code_indexes(
    codes: NDArray[np.uint8],
    spec: FactorizedResidualSpec,
    begin: int,
    end: int,
    subquantizer: int,
) -> NDArray[np.uint16]:
    bit = subquantizer * spec.bits_per_subquantizer
    byte = bit >> 3
    shift = bit & 7
    selected_codes = codes[begin:end]
    values = selected_codes[:, byte].astype(np.uint16) >> shift
    if shift + spec.bits_per_subquantizer > 8:
        values |= selected_codes[:, byte + 1].astype(np.uint16) << (8 - shift)
    return cast(
        NDArray[np.uint16],
        values & np.uint16(spec.codebook_size - 1),
    )


def _reconstructed_norms_from_arrays(
    spec: FactorizedResidualSpec,
    coarse_centroids: NDArray[np.float32],
    pq_codebooks: NDArray[np.float32],
    codes: NDArray[np.uint8],
    begin: int,
    end: int,
    list_id: int,
) -> NDArray[np.float64]:
    reconstructed = np.broadcast_to(
        coarse_centroids[list_id].astype(np.float64),
        (end - begin, spec.dimensions),
    ).copy()
    for subquantizer in range(spec.subquantizers):
        start = subquantizer * spec.subvector_dimensions
        stop = start + spec.subvector_dimensions
        indexes = _code_indexes(codes, spec, begin, end, subquantizer)
        reconstructed[:, start:stop] += pq_codebooks[subquantizer, indexes].astype(np.float64)
    norms = np.zeros(end - begin, dtype=np.float64)
    for dimension in range(spec.dimensions):
        coordinate = reconstructed[:, dimension]
        norms += coordinate * coordinate
    return norms


def _reconstructed_norms(
    components: FactorizedResidualComponents,
    postings: FactorizedResidualPostings,
    begin: int,
    end: int,
    list_id: int,
) -> NDArray[np.float64]:
    return _reconstructed_norms_from_arrays(
        components.spec,
        components.coarse_centroids,
        components.pq_codebooks,
        postings.codes,
        begin,
        end,
        list_id,
    )


def _stored_norm_parameters(low: float, high: float) -> tuple[np.float32, np.float32]:
    scale = (high - low) / 255.0
    maximum_float32 = float(np.finfo(np.float32).max)
    if (
        not np.isfinite(low)
        or not np.isfinite(scale)
        or low > maximum_float32
        or scale > maximum_float32
    ):
        raise ValueError("factorized residual norms differ")
    stored_low = np.float32(low)
    stored_scale = np.float32(scale)
    with np.errstate(over="ignore", invalid="ignore"):
        decoded_maximum = np.float32(stored_low + np.float32(stored_scale * np.float32(255.0)))
    if (high > low and stored_scale == 0.0) or not np.isfinite(decoded_maximum):
        raise ValueError("factorized residual norms differ")
    return stored_low, stored_scale


def _encode_norm_roles(
    components: FactorizedResidualComponents,
    postings: FactorizedResidualPostings,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32]]:
    spec = components.spec
    norm_codes = np.zeros(postings.rows, dtype=np.uint8)
    lows = np.zeros(spec.list_count, dtype="<f4")
    scales = np.zeros(spec.list_count, dtype="<f4")
    chunk_rows = 65_536
    for list_id in range(spec.list_count):
        begin = int(postings.offsets[list_id])
        end = int(postings.offsets[list_id + 1])
        if begin == end:
            continue
        low = np.inf
        high = -np.inf
        for start in range(begin, end, chunk_rows):
            norms = _reconstructed_norms(
                components, postings, start, min(start + chunk_rows, end), list_id
            )
            low = min(low, float(norms.min()))
            high = max(high, float(norms.max()))
        stored_low, stored_scale = _stored_norm_parameters(low, high)
        lows[list_id] = stored_low
        scales[list_id] = stored_scale
        if stored_scale == 0.0:
            continue
        for start in range(begin, end, chunk_rows):
            stop = min(start + chunk_rows, end)
            norms = _reconstructed_norms(components, postings, start, stop, list_id)
            encoded = np.rint((norms - float(stored_low)) / float(stored_scale))
            norm_codes[start:stop] = np.clip(encoded, 0, 255).astype(np.uint8)
    return norm_codes, lows, scales


def _validate_derived_norm_roles(
    spec: FactorizedResidualSpec,
    arrays: dict[str, NDArray[np.generic]],
) -> None:
    coarse = cast(NDArray[np.float32], arrays["coarse.f32"])
    pq = cast(NDArray[np.float32], arrays["pq.f32"])
    offsets = cast(NDArray[np.uint64], arrays["offsets.u64"])
    codes = cast(NDArray[np.uint8], arrays["codes.u8"])
    norm_codes = cast(NDArray[np.uint8], arrays["norms.u8"])
    lows = cast(NDArray[np.float32], arrays["norm-low.f32"])
    scales = cast(NDArray[np.float32], arrays["norm-scale.f32"])
    chunk_rows = 65_536
    for list_id in range(spec.list_count):
        begin = int(offsets[list_id])
        end = int(offsets[list_id + 1])
        if begin == end:
            if lows[list_id] != 0.0 or scales[list_id] != 0.0:
                raise ValueError("factorized residual artifact differs")
            continue
        low = np.inf
        high = -np.inf
        for start in range(begin, end, chunk_rows):
            norms = _reconstructed_norms_from_arrays(
                spec,
                coarse,
                pq,
                codes,
                start,
                min(start + chunk_rows, end),
                list_id,
            )
            low = min(low, float(norms.min()))
            high = max(high, float(norms.max()))
        try:
            expected_low, expected_scale = _stored_norm_parameters(low, high)
        except ValueError as error:
            raise ValueError("factorized residual artifact differs") from error
        if lows[list_id] != expected_low or scales[list_id] != expected_scale:
            raise ValueError("factorized residual artifact differs")
        for start in range(begin, end, chunk_rows):
            stop = min(start + chunk_rows, end)
            norms = _reconstructed_norms_from_arrays(spec, coarse, pq, codes, start, stop, list_id)
            if expected_scale == 0.0:
                expected_codes = np.zeros(stop - start, dtype=np.uint8)
            else:
                encoded = np.rint((norms - float(expected_low)) / float(expected_scale))
                expected_codes = np.clip(encoded, 0, 255).astype(np.uint8)
            if not np.array_equal(norm_codes[start:stop], expected_codes):
                raise ValueError("factorized residual artifact differs")


def _store_identity_json(identity: VectorStoreIdentity) -> dict[str, object]:
    return {
        "dimensions": identity.dimensions,
        "dtype": identity.dtype,
        "generation": identity.generation,
        "header_bytes": identity.header_bytes,
        "logical_bytes": identity.logical_bytes,
        "physical_bytes": identity.physical_bytes,
        "row_stride": identity.row_stride,
        "rows": identity.rows,
        "sha256": identity.sha256,
        "zero_padding_bytes": identity.zero_padding_bytes,
    }


def _require_exact_keys(value: object, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("factorized residual artifact differs")
    return value


def _vector_store_identity_from_json(value: object) -> VectorStoreIdentity:
    data = _require_exact_keys(
        value,
        {
            "dimensions",
            "dtype",
            "generation",
            "header_bytes",
            "logical_bytes",
            "physical_bytes",
            "row_stride",
            "rows",
            "sha256",
            "zero_padding_bytes",
        },
    )
    try:
        return VectorStoreIdentity(**data)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("factorized residual artifact differs") from error


class FactorizedResidualArtifact:
    """Authenticated read-only mappings for one immutable candidate artifact."""

    spec: FactorizedResidualSpec
    rows: int
    vector_store_identity: VectorStoreIdentity
    resident_bytes: int
    _arrays: dict[str, NDArray[np.generic]]
    _mappings: list[mmap.mmap]
    _descriptors: list[int]
    _closed: bool
    _closing: bool
    _active_searches: int
    _search_condition: threading.Condition

    __slots__ = (
        "_arrays",
        "_active_searches",
        "_closed",
        "_closing",
        "_descriptors",
        "_mappings",
        "_search_condition",
        "resident_bytes",
        "rows",
        "spec",
        "vector_store_identity",
    )

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("factorized residual artifact is immutable")

    def __init__(
        self,
        *,
        spec: FactorizedResidualSpec,
        rows: int,
        vector_store_identity: VectorStoreIdentity,
        resident_bytes: int,
        arrays: dict[str, NDArray[np.generic]],
        mappings: list[mmap.mmap],
        descriptors: list[int],
    ) -> None:
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "vector_store_identity", vector_store_identity)
        object.__setattr__(self, "resident_bytes", resident_bytes)
        object.__setattr__(self, "_arrays", arrays)
        object.__setattr__(self, "_mappings", mappings)
        object.__setattr__(self, "_descriptors", descriptors)
        object.__setattr__(self, "_closed", False)
        object.__setattr__(self, "_closing", False)
        object.__setattr__(self, "_active_searches", 0)
        object.__setattr__(self, "_search_condition", threading.Condition())

    def _acquire_search(self) -> dict[str, NDArray[np.generic]]:
        with self._search_condition:
            if self._closed or self._closing or not self._arrays:
                raise ValueError("factorized residual artifact differs")
            object.__setattr__(self, "_active_searches", self._active_searches + 1)
            return self._arrays

    def _release_search(self) -> None:
        with self._search_condition:
            object.__setattr__(self, "_active_searches", self._active_searches - 1)
            self._search_condition.notify_all()

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        manifest_sha256: str | None = None,
    ) -> FactorizedResidualArtifact:
        """Authenticate the canonical manifest and map every declared role read-only."""

        root = Path(path)
        descriptors: list[int] = []
        mappings: list[mmap.mmap] = []
        arrays: dict[str, NDArray[np.generic]] = {}
        try:
            expected_entries = {*_ARTIFACT_ROLES, "manifest.json"}
            if (
                not root.is_dir()
                or root.is_symlink()
                or {entry.name for entry in root.iterdir()} != expected_entries
            ):
                raise ValueError("factorized residual artifact differs")
            open_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
            manifest_fd = os.open(root / "manifest.json", open_flags)
            descriptors.append(manifest_fd)
            manifest_stat = os.fstat(manifest_fd)
            if (
                not stat.S_ISREG(manifest_stat.st_mode)
                or not 2 <= manifest_stat.st_size <= 64 << 10
            ):
                raise ValueError("factorized residual artifact differs")
            manifest_bytes = os.pread(manifest_fd, manifest_stat.st_size, 0)
            if len(manifest_bytes) != manifest_stat.st_size:
                raise ValueError("factorized residual artifact differs")
            observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if manifest_sha256 is not None and (
                type(manifest_sha256) is not str or manifest_sha256 != observed_manifest_sha256
            ):
                raise ValueError("factorized residual artifact differs")
            try:
                manifest_value = json.loads(manifest_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("factorized residual artifact differs") from error
            if _canonical_json_bytes(manifest_value) != manifest_bytes:
                raise ValueError("factorized residual artifact differs")
            manifest_after = os.fstat(manifest_fd)
            stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(
                getattr(manifest_stat, field) != getattr(manifest_after, field)
                for field in stable_fields
            ):
                raise ValueError("factorized residual artifact differs")
            manifest = _require_exact_keys(
                manifest_value,
                {
                    "bits_per_subquantizer",
                    "claim_eligible",
                    "code_bytes",
                    "dimensions",
                    "list_count",
                    "metric",
                    "probe_count",
                    "resident_bytes",
                    "return_width",
                    "roles",
                    "rows",
                    "schema",
                    "shortlist_width",
                    "subquantizers",
                    "vector_dtype",
                    "vector_store_identity",
                },
            )
            if (
                manifest["schema"] != "sfora-factorized-residual-ann-v1"
                or manifest["claim_eligible"] is not False
                or type(manifest["rows"]) is not int
                or not 1 <= manifest["rows"] < 2**32
            ):
                raise ValueError("factorized residual artifact differs")
            try:
                spec = FactorizedResidualSpec(
                    metric=manifest["metric"],  # type: ignore[arg-type]
                    vector_dtype=manifest["vector_dtype"],  # type: ignore[arg-type]
                    dimensions=manifest["dimensions"],  # type: ignore[arg-type]
                    list_count=manifest["list_count"],  # type: ignore[arg-type]
                    subquantizers=manifest["subquantizers"],  # type: ignore[arg-type]
                    bits_per_subquantizer=manifest["bits_per_subquantizer"],  # type: ignore[arg-type]
                    probe_count=manifest["probe_count"],  # type: ignore[arg-type]
                    shortlist_width=manifest["shortlist_width"],  # type: ignore[arg-type]
                    return_width=manifest["return_width"],  # type: ignore[arg-type]
                )
            except (TypeError, ValueError) as error:
                raise ValueError("factorized residual artifact differs") from error
            rows = manifest["rows"]
            if type(manifest["code_bytes"]) is not int or manifest["code_bytes"] != spec.code_bytes:
                raise ValueError("factorized residual artifact differs")
            store_identity = _vector_store_identity_from_json(manifest["vector_store_identity"])
            if (
                store_identity.rows != rows
                or store_identity.dimensions != spec.dimensions
                or store_identity.dtype != spec.vector_dtype
            ):
                raise ValueError("factorized residual artifact differs")
            role_values = _require_exact_keys(manifest["roles"], set(_ARTIFACT_ROLES))
            expected_geometry: dict[str, tuple[np.dtype[np.generic], tuple[int, ...]]] = {
                "coarse.f32": (np.dtype("<f4"), (spec.list_count, spec.dimensions)),
                "pq.f32": (
                    np.dtype("<f4"),
                    (spec.subquantizers, spec.codebook_size, spec.subvector_dimensions),
                ),
                "offsets.u64": (np.dtype("<u8"), (spec.list_count + 1,)),
                "ids.u32": (np.dtype("<u4"), (rows,)),
                "codes.u8": (np.dtype("u1"), (rows, spec.code_bytes)),
                "norms.u8": (np.dtype("u1"), (rows,)),
                "norm-low.f32": (np.dtype("<f4"), (spec.list_count,)),
                "norm-scale.f32": (np.dtype("<f4"), (spec.list_count,)),
            }
            observed_resident_bytes = 0
            for name in _ARTIFACT_ROLES:
                authority = _require_exact_keys(role_values[name], {"bytes", "sha256"})
                dtype, shape = expected_geometry[name]
                expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if (
                    type(authority["bytes"]) is not int
                    or authority["bytes"] != expected_bytes
                    or type(authority["sha256"]) is not str
                    or len(authority["sha256"]) != 64
                ):
                    raise ValueError("factorized residual artifact differs")
                fd = os.open(root / name, open_flags)
                descriptors.append(fd)
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
                    raise ValueError("factorized residual artifact differs")
                if _sha256_fd(fd, expected_bytes) != authority["sha256"]:
                    raise ValueError("factorized residual artifact differs")
                after = os.fstat(fd)
                if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
                    raise ValueError("factorized residual artifact differs")
                mapping = mmap.mmap(fd, expected_bytes, access=mmap.ACCESS_READ)
                mappings.append(mapping)
                array = np.ndarray(shape, dtype=dtype, buffer=mapping)
                array.flags.writeable = False
                arrays[name] = array
                observed_resident_bytes += expected_bytes
            if (
                type(manifest["resident_bytes"]) is not int
                or manifest["resident_bytes"] != observed_resident_bytes
                or not bool(np.isfinite(arrays["coarse.f32"]).all())
                or not bool(np.isfinite(arrays["pq.f32"]).all())
                or not bool(np.isfinite(arrays["norm-low.f32"]).all())
                or not bool(np.isfinite(arrays["norm-scale.f32"]).all())
                or bool((cast(NDArray[np.float32], arrays["norm-scale.f32"]) < 0).any())
            ):
                raise ValueError("factorized residual artifact differs")
            offsets = cast(NDArray[np.uint64], arrays["offsets.u64"])
            ids = cast(NDArray[np.uint32], arrays["ids.u32"])
            if (
                int(offsets[0]) != 0
                or int(offsets[-1]) != rows
                or bool((offsets[1:] < offsets[:-1]).any())
            ):
                raise ValueError("factorized residual artifact differs")
            seen = bytearray((rows + 7) // 8)
            for raw_id in ids:
                internal_id = int(raw_id)
                if internal_id >= rows:
                    raise ValueError("factorized residual artifact differs")
                byte_index = internal_id >> 3
                mask = 1 << (internal_id & 7)
                if seen[byte_index] & mask:
                    raise ValueError("factorized residual artifact differs")
                seen[byte_index] |= mask
            trailing_bits = spec.code_bytes * 8 - spec.subquantizers * spec.bits_per_subquantizer
            if trailing_bits:
                trailing_mask = ((1 << trailing_bits) - 1) << (8 - trailing_bits)
                codes = cast(NDArray[np.uint8], arrays["codes.u8"])
                for begin in range(0, rows, 8 << 20):
                    if bool((codes[begin : begin + (8 << 20), -1] & trailing_mask).any()):
                        raise ValueError("factorized residual artifact differs")
            _validate_derived_norm_roles(spec, arrays)
            return cls(
                spec=spec,
                rows=rows,
                vector_store_identity=store_identity,
                resident_bytes=observed_resident_bytes,
                arrays=arrays,
                mappings=mappings,
                descriptors=descriptors,
            )
        except BaseException as original_error:
            error_type = type(original_error)
            error_args = original_error.args
            error_notes = list(getattr(original_error, "__notes__", ()))
            original_error.__traceback__ = None
            arrays.clear()
            if "array" in locals():
                del array
            if "offsets" in locals():
                del offsets
            if "ids" in locals():
                del ids
            if "codes" in locals():
                del codes
            cleanup_notes: list[str] = []
            for mapping in reversed(mappings):
                try:
                    mapping.close()
                except BaseException as cleanup_error:
                    cleanup_notes.append(repr(cleanup_error))
            for fd in reversed(descriptors):
                try:
                    os.close(fd)
                except BaseException as cleanup_error:
                    cleanup_notes.append(repr(cleanup_error))
            detached_error = error_type(*error_args)
            for error_note in error_notes:
                detached_error.add_note(error_note)
            for cleanup_note in cleanup_notes:
                detached_error.add_note(f"cleanup also failed: {cleanup_note}")
            raise detached_error from None

    def close(self) -> None:
        """Release mappings and descriptors exactly once."""

        with self._search_condition:
            while self._closing and not self._closed:
                self._search_condition.wait()
            if self._closed:
                return
            object.__setattr__(self, "_closing", True)
            try:
                while self._active_searches:
                    self._search_condition.wait()
            except BaseException:
                object.__setattr__(self, "_closing", False)
                self._search_condition.notify_all()
                raise
        self._arrays.clear()
        first_error: BaseException | None = None
        failed_mappings: list[mmap.mmap] = []
        for mapping in reversed(self._mappings):
            try:
                mapping.close()
            except BaseException as error:
                failed_mappings.append(mapping)
                if first_error is None:
                    first_error = error
        self._mappings[:] = reversed(failed_mappings)
        for fd in reversed(self._descriptors):
            try:
                os.close(fd)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self._descriptors.clear()
        object.__setattr__(
            self,
            "_closed",
            not self._mappings and not self._descriptors,
        )
        with self._search_condition:
            object.__setattr__(self, "_closing", False)
            self._search_condition.notify_all()
        if first_error is not None:
            raise first_error


def write_factorized_residual_artifact(
    path: str | Path,
    spec: FactorizedResidualSpec,
    components: FactorizedResidualComponents,
    postings: FactorizedResidualPostings,
    vector_store_identity: VectorStoreIdentity,
) -> str:
    """Atomically write one canonical immutable factorized-residual artifact."""

    destination = Path(path)
    if (
        type(spec) is not FactorizedResidualSpec
        or type(components) is not FactorizedResidualComponents
        or type(postings) is not FactorizedResidualPostings
        or components.spec != spec
        or postings.spec != spec
        or type(vector_store_identity) is not VectorStoreIdentity
        or vector_store_identity.rows != postings.rows
        or vector_store_identity.dimensions != spec.dimensions
        or vector_store_identity.dtype != spec.vector_dtype
        or destination.exists()
        or not destination.parent.is_dir()
    ):
        raise ValueError("factorized residual artifact differs")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    known_files = (*_ARTIFACT_ROLES, "manifest.json")
    published = False
    try:
        norm_codes, norm_lows, norm_scales = _encode_norm_roles(components, postings)
        roles: dict[str, NDArray[np.generic]] = {
            "coarse.f32": components.coarse_centroids,
            "pq.f32": components.pq_codebooks,
            "offsets.u64": postings.offsets,
            "ids.u32": postings.ids,
            "codes.u8": postings.codes,
            "norms.u8": norm_codes,
            "norm-low.f32": norm_lows,
            "norm-scale.f32": norm_scales,
        }
        role_authority: dict[str, dict[str, object]] = {}
        for name in _ARTIFACT_ROLES:
            role_path = temporary / name
            _write_array_bounded(role_path, roles[name])
            role_authority[name] = {
                "bytes": role_path.stat().st_size,
                "sha256": _sha256_file(role_path),
            }
        manifest = {
            "bits_per_subquantizer": spec.bits_per_subquantizer,
            "claim_eligible": False,
            "code_bytes": spec.code_bytes,
            "dimensions": spec.dimensions,
            "list_count": spec.list_count,
            "metric": spec.metric,
            "probe_count": spec.probe_count,
            "resident_bytes": sum((temporary / name).stat().st_size for name in _ARTIFACT_ROLES),
            "return_width": spec.return_width,
            "roles": role_authority,
            "rows": postings.rows,
            "schema": "sfora-factorized-residual-ann-v1",
            "shortlist_width": spec.shortlist_width,
            "subquantizers": spec.subquantizers,
            "vector_dtype": spec.vector_dtype,
            "vector_store_identity": _store_identity_json(vector_store_identity),
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        if len(manifest_bytes) > 64 << 10:
            raise ValueError("factorized residual artifact differs")
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("wb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(temporary)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            _rename_directory_no_replace(temporary, destination)
            published = True
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return hashlib.sha256(manifest_bytes).hexdigest()
    except BaseException:
        cleanup_root = destination if published else temporary
        for name in known_files:
            candidate = cleanup_root / name
            if candidate.exists():
                candidate.unlink()
        if cleanup_root.exists():
            cleanup_root.rmdir()
        raise


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Bounded work accounting for one portable candidate search."""

    backend: Literal["portable-float64", "native-c11-fma"]
    rows_scanned: int
    codes_bytes_scanned: int
    probe_count: int
    shortlist_width: int

    def __post_init__(self) -> None:
        if (
            self.backend not in {"portable-float64", "native-c11-fma"}
            or any(
                type(value) is not int
                for value in (
                    self.rows_scanned,
                    self.codes_bytes_scanned,
                    self.probe_count,
                    self.shortlist_width,
                )
            )
            or self.rows_scanned < 0
            or self.codes_bytes_scanned < 0
            or self.probe_count < 1
            or self.shortlist_width < 1
        ):
            raise ValueError("factorized residual candidate evidence differs")


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Owned approximate candidates ordered by distance then internal ID."""

    ids: NDArray[np.uint32]
    approximate_distances: NDArray[np.float64]
    probe_lists: NDArray[np.uint32]
    evidence: CandidateEvidence

    def __post_init__(self) -> None:
        if (
            type(self.ids) is not np.ndarray
            or self.ids.dtype.str != "<u4"
            or self.ids.ndim != 1
            or type(self.approximate_distances) is not np.ndarray
            or self.approximate_distances.dtype.str != "<f8"
            or self.approximate_distances.shape != self.ids.shape
            or not bool(np.isfinite(self.approximate_distances).all())
            or type(self.probe_lists) is not np.ndarray
            or self.probe_lists.dtype.str != "<u4"
            or self.probe_lists.ndim != 1
            or self.probe_lists.shape[0] < 1
            or type(self.evidence) is not CandidateEvidence
        ):
            raise ValueError("factorized residual candidate result differs")
        pairs = list(zip(self.approximate_distances.tolist(), self.ids.tolist(), strict=True))
        if pairs != sorted(pairs) or len(set(self.ids.tolist())) != self.ids.shape[0]:
            raise ValueError("factorized residual candidate result differs")
        object.__setattr__(self, "ids", _owned_read_only(self.ids))
        object.__setattr__(
            self,
            "approximate_distances",
            _owned_read_only(self.approximate_distances),
        )
        object.__setattr__(self, "probe_lists", _owned_read_only(self.probe_lists))


class PortableCandidateIndex:
    """Deterministic bounded-memory float64 reference candidate scorer."""

    def __init__(self, artifact: FactorizedResidualArtifact) -> None:
        if type(artifact) is not FactorizedResidualArtifact or artifact._closed:
            raise ValueError("factorized residual candidate index differs")
        self._artifact = artifact

    def search(
        self,
        query: NDArray[np.generic],
        *,
        probe_count: int | None = None,
        shortlist_width: int | None = None,
    ) -> CandidateResult:
        """Return approximate candidates from selected posting lists."""

        artifact = self._artifact
        spec = artifact.spec
        expected_dtype = np.dtype(np.uint8 if spec.vector_dtype == "uint8" else "<f4")
        if (
            artifact._closed
            or type(query) is not np.ndarray
            or query.dtype != expected_dtype
            or query.shape != (spec.dimensions,)
            or not query.flags.c_contiguous
            or not bool(np.isfinite(query).all())
        ):
            raise ValueError("factorized residual query differs")
        selected_probe_count = spec.probe_count if probe_count is None else probe_count
        selected_shortlist_width = (
            spec.shortlist_width if shortlist_width is None else shortlist_width
        )
        if (
            type(selected_probe_count) is not int
            or not 1 <= selected_probe_count <= spec.probe_count
            or type(selected_shortlist_width) is not int
            or not 1 <= selected_shortlist_width <= spec.shortlist_width
        ):
            raise ValueError("factorized residual search differs")
        arrays = artifact._acquire_search()
        try:
            return self._search_acquired(
                query, arrays, selected_probe_count, selected_shortlist_width
            )
        finally:
            artifact._release_search()

    def _search_acquired(
        self,
        query: NDArray[np.generic],
        arrays: dict[str, NDArray[np.generic]],
        selected_probe_count: int,
        selected_shortlist_width: int,
    ) -> CandidateResult:
        spec = self._artifact.spec
        coarse = cast(NDArray[np.float32], arrays["coarse.f32"])
        pq = cast(NDArray[np.float32], arrays["pq.f32"])
        offsets = cast(NDArray[np.uint64], arrays["offsets.u64"])
        ids = cast(NDArray[np.uint32], arrays["ids.u32"])
        codes = cast(NDArray[np.uint8], arrays["codes.u8"])
        norm_codes = cast(NDArray[np.uint8], arrays["norms.u8"])
        norm_lows = cast(NDArray[np.float32], arrays["norm-low.f32"])
        norm_scales = cast(NDArray[np.float32], arrays["norm-scale.f32"])
        query64 = query.astype(np.float64)

        probe_heap: list[tuple[float, int]] = []
        for list_id in range(spec.list_count):
            distance = 0.0
            for dimension in range(spec.dimensions):
                difference = query64[dimension] - float(coarse[list_id, dimension])
                distance += difference * difference
            pair = (-distance, -list_id)
            if len(probe_heap) < selected_probe_count:
                heapq.heappush(probe_heap, pair)
            elif pair > probe_heap[0]:
                heapq.heapreplace(probe_heap, pair)
        probe_pairs = sorted((-distance, -list_id) for distance, list_id in probe_heap)
        probe_lists = np.asarray([list_id for _, list_id in probe_pairs], dtype="<u4")

        query_norm = 0.0
        for coordinate in query64:
            query_norm += float(coordinate) * float(coordinate)
        lut = np.zeros((spec.subquantizers, spec.codebook_size), dtype="<f8")
        for subquantizer in range(spec.subquantizers):
            offset = subquantizer * spec.subvector_dimensions
            for codeword in range(spec.codebook_size):
                dot = 0.0
                for coordinate in range(spec.subvector_dimensions):
                    dot += query64[offset + coordinate] * float(
                        pq[subquantizer, codeword, coordinate]
                    )
                lut[subquantizer, codeword] = -2.0 * dot

        candidate_heap: list[tuple[float, int]] = []
        rows_scanned = 0
        chunk_rows = 65_536
        for list_id in probe_lists.tolist():
            coarse_dot = 0.0
            for dimension in range(spec.dimensions):
                coarse_dot += query64[dimension] * float(coarse[list_id, dimension])
            constant = query_norm - 2.0 * coarse_dot + float(norm_lows[list_id])
            begin = int(offsets[list_id])
            end = int(offsets[list_id + 1])
            rows_scanned += end - begin
            for start in range(begin, end, chunk_rows):
                stop = min(start + chunk_rows, end)
                scores = constant + float(norm_scales[list_id]) * norm_codes[start:stop].astype(
                    np.float64
                )
                for subquantizer in range(spec.subquantizers):
                    indexes = _code_indexes(codes, spec, start, stop, subquantizer)
                    scores += lut[subquantizer, indexes]
                if not bool(np.isfinite(scores).all()):
                    raise ValueError("factorized residual scores differ")
                for row_offset, score in enumerate(scores.tolist()):
                    internal_id = int(ids[start + row_offset])
                    pair = (-score, -internal_id)
                    if len(candidate_heap) < selected_shortlist_width:
                        heapq.heappush(candidate_heap, pair)
                    elif pair > candidate_heap[0]:
                        heapq.heapreplace(candidate_heap, pair)
        ordered = sorted((-score, -internal_id) for score, internal_id in candidate_heap)
        return CandidateResult(
            ids=np.asarray([internal_id for _, internal_id in ordered], dtype="<u4"),
            approximate_distances=np.asarray([distance for distance, _ in ordered], dtype="<f8"),
            probe_lists=probe_lists,
            evidence=CandidateEvidence(
                backend="portable-float64",
                rows_scanned=rows_scanned,
                codes_bytes_scanned=rows_scanned * spec.code_bytes,
                probe_count=selected_probe_count,
                shortlist_width=selected_shortlist_width,
            ),
        )
