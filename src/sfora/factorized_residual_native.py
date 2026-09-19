"""Explicitly compiled native candidate scoring for factorized residual indexes."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.ctypeslib import ndpointer
from numpy.typing import NDArray

from sfora.factorized_residual_ann import (
    CandidateEvidence,
    CandidateResult,
    FactorizedResidualArtifact,
)
from sfora.vector_store import (
    DirectIoVectorStore,
    ExactSearchResult,
    InsufficientCandidatesError,
    VectorReadEvidence,
)

_SOURCE = Path(__file__).with_name("_native") / "factorized_residual_ann.c"
_COMPILE_FLAGS = (
    "-std=c11",
    "-O3",
    "-ffp-contract=off",
    "-fPIC",
    "-shared",
    "-fopenmp",
)
_LINK_FLAGS = ("-lm",)
_ERROR = "native factorized residual compiler failed"
_POINTER_BYTES = ctypes.sizeof(ctypes.c_void_p)
_DIRECT_CONDITION = threading.Condition()
_DIRECT_ACTIVE = False
_DIRECT_POISONED = False


class NativeDirectBusyError(RuntimeError):
    """Raised when the bounded direct-I/O context is already active."""


class NativeDirectQuiescenceError(RuntimeError):
    """Raised after terminality becomes unprovable and direct I/O is poisoned."""


class NativeDirectUnsupportedError(RuntimeError):
    """Raised when the operating environment cannot provide the direct backend."""


def _acquire_direct_admission() -> None:
    global _DIRECT_ACTIVE
    with _DIRECT_CONDITION:
        if _DIRECT_POISONED:
            raise NativeDirectQuiescenceError("native direct I/O is permanently poisoned")
        if _DIRECT_ACTIVE:
            raise NativeDirectBusyError("native direct I/O context is busy")
        _DIRECT_ACTIVE = True


def _release_direct_admission(*, poison: bool) -> None:
    global _DIRECT_ACTIVE, _DIRECT_POISONED
    with _DIRECT_CONDITION:
        if poison:
            _DIRECT_POISONED = True
        _DIRECT_ACTIVE = False
        _DIRECT_CONDITION.notify_all()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NativeBackend:
    """Owned authenticated native shared-library handle."""

    path: Path
    source_sha256: str
    binary_sha256: str
    _library: ctypes.CDLL
    _closed: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
            or type(self.binary_sha256) is not str
            or len(self.binary_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.binary_sha256)
            or type(self._library) is not ctypes.CDLL
            or type(self._closed) is not bool
        ):
            raise ValueError("native factorized residual backend differs")

    def close(self) -> None:
        """Refuse future calls through this handle and release the read buffer.

        The direct reader keeps its read buffer mapped between queries, so an
        idle backend would otherwise retain it for the life of the process. A
        quarantined reader keeps ownership and is left alone.
        """

        object.__setattr__(self, "_closed", True)
        with contextlib.suppress(OSError):
            self._library.sfora_direct_release_buffer()


def _compiler_path(cc: str | Path | None) -> Path:
    requested = os.fspath(cc) if cc is not None else os.environ.get("CC", "cc")
    resolved = shutil.which(requested)
    if resolved is None:
        raise RuntimeError(_ERROR)
    return Path(resolved).resolve()


def _load_backend(path: Path, source_sha256: str, binary_sha256: str) -> NativeBackend:
    descriptor = -1
    load_root = Path(tempfile.mkdtemp(prefix=".factorized-residual-load-", dir=path.parent))
    load_path = load_root / "backend.so"
    try:
        os.link(path, load_path, follow_symlinks=False)
        descriptor = os.open(load_path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("native backend is not regular")
        digest = hashlib.sha256()
        offset = 0
        while offset < metadata.st_size:
            block = os.pread(descriptor, min(1 << 20, metadata.st_size - offset), offset)
            if not block:
                raise OSError("native backend changed while hashing")
            digest.update(block)
            offset += len(block)
        if digest.hexdigest() != binary_sha256:
            raise OSError("native backend digest differs")
        library = ctypes.CDLL(load_path)
        abi_version = library.sfora_factorized_backend_abi_version
        abi_version.argtypes = []
        abi_version.restype = ctypes.c_uint32
        if abi_version() != 2:
            raise OSError("native backend ABI differs")
        function = library.sfora_factorized_candidate_search
        probe = library.sfora_direct_io_probe
        alignment = library.sfora_direct_io_alignment
        context_bytes = library.sfora_exact_rerank_context_bytes
        rerank = library.sfora_exact_rerank_direct
        release_buffer = library.sfora_direct_release_buffer
    except (AttributeError, OSError) as error:
        raise RuntimeError(_ERROR) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if load_path.exists():
            load_path.unlink()
        load_root.rmdir()
    float_pointer = ndpointer(dtype=np.dtype("<f4"), ndim=1, flags=("C_CONTIGUOUS",))
    u8_pointer = ndpointer(dtype=np.dtype("u1"), ndim=1, flags=("C_CONTIGUOUS",))
    u32_pointer = ndpointer(dtype=np.dtype("<u4"), ndim=1, flags=("C_CONTIGUOUS",))
    u64_pointer = ndpointer(dtype=np.dtype("<u8"), ndim=1, flags=("C_CONTIGUOUS",))
    function.argtypes = [
        float_pointer,
        ctypes.c_size_t,
        float_pointer,
        ctypes.c_size_t,
        float_pointer,
        ctypes.c_size_t,
        u64_pointer,
        ctypes.c_size_t,
        u32_pointer,
        ctypes.c_size_t,
        u8_pointer,
        ctypes.c_size_t,
        u8_pointer,
        ctypes.c_size_t,
        float_pointer,
        ctypes.c_size_t,
        float_pointer,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        u32_pointer,
        float_pointer,
        ctypes.c_size_t,
        u32_pointer,
        ctypes.c_size_t,
        u32_pointer,
        u64_pointer,
    ]
    function.restype = ctypes.c_int
    probe.argtypes = []
    probe.restype = ctypes.c_int
    alignment.argtypes = [ctypes.c_int, u32_pointer, u32_pointer]
    alignment.restype = ctypes.c_int
    context_bytes.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        u64_pointer,
    ]
    context_bytes.restype = ctypes.c_int
    rerank.argtypes = [
        float_pointer,
        ctypes.c_size_t,
        u32_pointer,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        u32_pointer,
        ndpointer(dtype=np.dtype("<f8"), ndim=1, flags=("C_CONTIGUOUS",)),
        ctypes.c_size_t,
        u64_pointer,
        u32_pointer,
    ]
    rerank.restype = ctypes.c_int
    release_buffer.argtypes = []
    release_buffer.restype = ctypes.c_int
    return NativeBackend(path, source_sha256, binary_sha256, library)


def compile_factorized_residual_backend(
    cache_dir: str | Path, cc: str | Path | None = None
) -> NativeBackend:
    """Compile and load the packaged C backend without shell or network access."""

    cache = Path(cache_dir).resolve()
    compiler = _compiler_path(cc)
    if platform.system() != "Linux" or _POINTER_BYTES != 8 or not _SOURCE.is_file():
        raise RuntimeError(_ERROR)
    try:
        version = subprocess.run(
            [os.fspath(compiler), "--version"],
            check=True,
            capture_output=True,
            text=False,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(_ERROR) from error
    source_sha256 = _sha256(_SOURCE)
    identity = {
        "architecture": platform.machine(),
        "compiler": os.fspath(compiler),
        "compiler_version_sha256": hashlib.sha256(version).hexdigest(),
        "compile_flags": list(_COMPILE_FLAGS),
        "link_flags": list(_LINK_FLAGS),
        "platform": platform.system(),
        "pointer_bytes": _POINTER_BYTES,
        "python_abi": sys.implementation.cache_tag,
        "source_sha256": source_sha256,
    }
    cache_key = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    binary_path = cache / f"factorized-residual-{cache_key}.so"
    receipt_path = cache / f"factorized-residual-{cache_key}.json"
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache_metadata = cache.lstat()
    if (
        not stat.S_ISDIR(cache_metadata.st_mode)
        or cache_metadata.st_uid != os.geteuid()
        or cache_metadata.st_mode & 0o022
    ):
        raise RuntimeError(_ERROR)
    if binary_path.is_file() and receipt_path.is_file():
        try:
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes)
            if receipt_bytes != _canonical_bytes(receipt):
                raise ValueError
            binary_sha256 = _sha256(binary_path)
            if receipt != {"binary_sha256": binary_sha256, "identity": identity}:
                raise ValueError
            return _load_backend(binary_path, source_sha256, binary_sha256)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    build_root = Path(tempfile.mkdtemp(prefix=".factorized-residual-", dir=cache))
    temporary = build_root / "backend.so"
    receipt_temporary = build_root / "receipt.json"
    try:
        command = [
            os.fspath(compiler),
            *_COMPILE_FLAGS,
            os.fspath(_SOURCE),
            *_LINK_FLAGS,
            "-o",
            os.fspath(temporary),
        ]
        subprocess.run(command, check=True, capture_output=True, text=False)
        binary_sha256 = _sha256(temporary)
        receipt_temporary.write_bytes(
            _canonical_bytes({"binary_sha256": binary_sha256, "identity": identity})
        )
        os.replace(temporary, binary_path)
        os.replace(receipt_temporary, receipt_path)
        build_root.rmdir()
        return _load_backend(binary_path, source_sha256, binary_sha256)
    except BaseException as error:
        if temporary.exists():
            temporary.unlink()
        if receipt_temporary.exists():
            receipt_temporary.unlink()
        if build_root.exists():
            build_root.rmdir()
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError(_ERROR) from error


class NativeCandidateIndex:
    """Native float32-FMA implementation of the bounded candidate scorer."""

    def __init__(
        self,
        artifact: FactorizedResidualArtifact,
        backend: NativeBackend,
        *,
        thread_count: int = 1,
    ) -> None:
        if (
            type(artifact) is not FactorizedResidualArtifact
            or artifact._closed
            or type(backend) is not NativeBackend
            or backend._closed
            or type(thread_count) is not int
            or not 1 <= thread_count <= 256
        ):
            raise ValueError("native factorized residual index differs")
        self._artifact = artifact
        self._backend = backend
        self._thread_count = thread_count

    def search(
        self,
        query: NDArray[np.generic],
        *,
        probe_count: int | None = None,
        shortlist_width: int | None = None,
    ) -> CandidateResult:
        """Return native approximate candidates with bounded work evidence."""

        artifact = self._artifact
        backend = self._backend
        spec = artifact.spec
        expected_dtype = np.dtype(np.uint8 if spec.vector_dtype == "uint8" else "<f4")
        selected_probe = spec.probe_count if probe_count is None else probe_count
        selected_shortlist = spec.shortlist_width if shortlist_width is None else shortlist_width
        if (
            artifact._closed
            or backend._closed
            or type(query) is not np.ndarray
            or query.dtype != expected_dtype
            or query.shape != (spec.dimensions,)
            or not query.flags.c_contiguous
            or not bool(np.isfinite(query).all())
            or type(selected_probe) is not int
            or not 1 <= selected_probe <= spec.probe_count
            or type(selected_shortlist) is not int
            or not 1 <= selected_shortlist <= spec.shortlist_width
            or any(
                value >= 2**32
                for value in (
                    spec.list_count,
                    spec.dimensions,
                    spec.subquantizers,
                    selected_probe,
                    selected_shortlist,
                )
            )
        ):
            raise ValueError("native factorized residual search differs")
        arrays = artifact._acquire_search()
        try:
            return self._search_acquired(query, arrays, selected_probe, selected_shortlist)
        finally:
            artifact._release_search()

    def _search_acquired(
        self,
        query: NDArray[np.generic],
        arrays: dict[str, NDArray[np.generic]],
        selected_probe: int,
        selected_shortlist: int,
    ) -> CandidateResult:
        artifact = self._artifact
        backend = self._backend
        spec = artifact.spec
        query32 = np.array(query, dtype="<f4", order="C", copy=True)
        coarse = cast(NDArray[np.float32], arrays["coarse.f32"]).reshape(-1)
        pq = cast(NDArray[np.float32], arrays["pq.f32"]).reshape(-1)
        offsets = cast(NDArray[np.uint64], arrays["offsets.u64"]).reshape(-1)
        ids = cast(NDArray[np.uint32], arrays["ids.u32"]).reshape(-1)
        codes = cast(NDArray[np.uint8], arrays["codes.u8"]).reshape(-1)
        norms = cast(NDArray[np.uint8], arrays["norms.u8"]).reshape(-1)
        norm_low = cast(NDArray[np.float32], arrays["norm-low.f32"]).reshape(-1)
        norm_scale = cast(NDArray[np.float32], arrays["norm-scale.f32"]).reshape(-1)
        output_ids = np.empty(selected_shortlist, dtype="<u4")
        output_scores = np.empty(selected_shortlist, dtype="<f4")
        probe_lists = np.empty(selected_probe, dtype="<u4")
        output_count = np.zeros(1, dtype="<u4")
        rows_scanned = np.zeros(1, dtype="<u8")
        function = backend._library.sfora_factorized_candidate_search
        status = function(
            query32,
            query32.size,
            coarse,
            coarse.size,
            pq,
            pq.size,
            offsets,
            offsets.size,
            ids,
            ids.size,
            codes,
            codes.size,
            norms,
            norms.size,
            norm_low,
            norm_low.size,
            norm_scale,
            norm_scale.size,
            spec.list_count,
            spec.dimensions,
            spec.subquantizers,
            spec.bits_per_subquantizer,
            selected_probe,
            selected_shortlist,
            self._thread_count,
            output_ids,
            output_scores,
            output_ids.size,
            probe_lists,
            probe_lists.size,
            output_count,
            rows_scanned,
        )
        if status != 0:
            raise RuntimeError(f"native factorized residual search failed: {status}")
        actual_count = int(output_count[0])
        if actual_count > selected_shortlist:
            raise RuntimeError("native factorized residual search failed: output count")
        return CandidateResult(
            ids=output_ids[:actual_count],
            approximate_distances=output_scores[:actual_count].astype("<f8"),
            probe_lists=probe_lists,
            evidence=CandidateEvidence(
                backend="native-c11-fma",
                rows_scanned=int(rows_scanned[0]),
                codes_bytes_scanned=int(rows_scanned[0]) * spec.code_bytes,
                probe_count=selected_probe,
                shortlist_width=selected_shortlist,
            ),
        )


class NativeExactReranker:
    """Exact native reranking over one authenticated Linux direct-I/O store."""

    def __init__(
        self,
        store: DirectIoVectorStore,
        backend: NativeBackend,
        *,
        thread_count: int = 1,
    ) -> None:
        if (
            type(store) is not DirectIoVectorStore
            or store._closed
            or type(backend) is not NativeBackend
            or backend._closed
            or type(thread_count) is not int
            or not 1 <= thread_count <= 256
        ):
            raise ValueError("native exact reranker differs")
        memory_alignment = np.zeros(1, dtype="<u4")
        offset_alignment = np.zeros(1, dtype="<u4")
        probe_status = backend._library.sfora_direct_io_probe()
        if probe_status == -5:
            raise NativeDirectUnsupportedError("native direct I/O is unsupported")
        if probe_status != 0:
            raise RuntimeError(f"native direct I/O probe failed: {probe_status}")
        descriptor = store._begin_direct_read()
        try:
            alignment_status = backend._library.sfora_direct_io_alignment(
                descriptor, memory_alignment, offset_alignment
            )
        finally:
            store._end_direct_read()
        if (
            alignment_status != 0
            or int(memory_alignment[0]) < 1
            or int(offset_alignment[0]) < 1
            or store.identity.physical_bytes % int(offset_alignment[0]) != 0
        ):
            raise ValueError("native exact reranker differs")
        self._store = store
        self._backend = backend
        self._thread_count = thread_count
        self._memory_alignment = int(memory_alignment[0])
        self._offset_alignment = int(offset_alignment[0])

    def context_bytes(self, candidate_count: int) -> int:
        """Return exact native allocations and ring mappings for one rerank call."""

        if type(candidate_count) is not int or not 1 <= candidate_count <= 4096:
            raise ValueError("native exact reranker differs")
        output = np.zeros(1, dtype="<u8")
        status = self._backend._library.sfora_exact_rerank_context_bytes(
            candidate_count,
            self._store.identity.dimensions,
            int(self._store.identity.dtype == "uint8"),
            self._memory_alignment,
            self._offset_alignment,
            output,
        )
        if status != 0:
            raise RuntimeError(f"native direct context accounting failed: {status}")
        return int(output[0])

    def search(
        self,
        query: NDArray[np.generic],
        candidates: CandidateResult,
        *,
        return_width: int,
    ) -> ExactSearchResult:
        """Read candidates with O_DIRECT and return exact deterministic neighbors."""

        store = self._store
        backend = self._backend
        identity = store.identity
        if (
            type(candidates) is CandidateResult
            and type(return_width) is int
            and return_width > candidates.ids.shape[0]
        ):
            raise InsufficientCandidatesError("insufficient exact rerank candidates")
        expected_dtype = np.dtype(np.uint8 if identity.dtype == "uint8" else "<f4")
        if (
            store._closed
            or backend._closed
            or type(candidates) is not CandidateResult
            or type(return_width) is not int
            or not 1 <= return_width <= candidates.ids.shape[0]
            or candidates.ids.shape[0] > 4096
            or bool((candidates.ids >= identity.rows).any())
            or type(query) is not np.ndarray
            or query.dtype != expected_dtype
            or query.shape != (identity.dimensions,)
            or not query.flags.c_contiguous
            or not bool(np.isfinite(query).all())
        ):
            raise ValueError("native exact rerank differs")
        query32 = np.array(query, dtype="<f4", order="C", copy=True)
        output_ids = np.empty(return_width, dtype="<u4")
        output_distances = np.empty(return_width, dtype="<f8")
        physical_bytes = np.zeros(1, dtype="<u8")
        operations = np.zeros(1, dtype="<u4")
        status: int | None = None
        _acquire_direct_admission()
        try:
            descriptor = store._begin_direct_read()
            try:
                status = backend._library.sfora_exact_rerank_direct(
                    query32,
                    query32.size,
                    candidates.ids,
                    candidates.ids.size,
                    identity.rows,
                    identity.dimensions,
                    int(identity.dtype == "uint8"),
                    return_width,
                    descriptor,
                    identity.physical_bytes,
                    self._memory_alignment,
                    self._offset_alignment,
                    self._thread_count,
                    output_ids,
                    output_distances,
                    output_ids.size,
                    physical_bytes,
                    operations,
                )
            finally:
                store._end_direct_read()
        finally:
            _release_direct_admission(poison=status == -4)
        if status == -4:
            raise NativeDirectQuiescenceError("native direct I/O quiescence is unproved")
        if status == -5:
            raise NativeDirectUnsupportedError("native direct I/O is unsupported")
        if status == -7:
            raise NativeDirectBusyError("native direct I/O context is busy")
        if status != 0 or int(operations[0]) != candidates.ids.size:
            raise RuntimeError(f"native direct rerank failed: {status}")
        return ExactSearchResult(
            ids=output_ids,
            squared_distances=output_distances,
            vector_reads=VectorReadEvidence(
                requested_rows=candidates.ids.size,
                logical_bytes=candidates.ids.size * identity.row_stride,
                physical_bytes=int(physical_bytes[0]),
            ),
        )
