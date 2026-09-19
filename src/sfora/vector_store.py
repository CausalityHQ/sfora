"""Authenticated vector stores and deterministic exact reranking."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sfora.factorized_residual_ann import (
    CandidateResult,
    VectorStoreIdentity,
    _owned_read_only,
)

_HAS_PREADV = hasattr(os, "preadv")


class InsufficientCandidatesError(ValueError):
    """Raised when exact reranking cannot produce the fixed result width."""


@dataclass(frozen=True, slots=True)
class VectorReadEvidence:
    """Logical and physical bytes consumed by one vector read."""

    requested_rows: int
    logical_bytes: int
    physical_bytes: int

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int
                for value in (
                    self.requested_rows,
                    self.logical_bytes,
                    self.physical_bytes,
                )
            )
            or self.requested_rows < 1
            or self.logical_bytes < 1
            or self.physical_bytes < self.logical_bytes
        ):
            raise ValueError("vector read evidence differs")


class VectorReadContext(Protocol):
    """Owned per-query vector read state."""

    def read(self, ids: NDArray[np.uint32]) -> tuple[NDArray[np.generic], VectorReadEvidence]: ...

    def close(self) -> None: ...


class VectorStore(Protocol):
    """Immutable vector authority with isolated read contexts."""

    @property
    def identity(self) -> VectorStoreIdentity: ...

    @property
    def resident_bytes(self) -> int: ...

    def new_context(self) -> VectorReadContext: ...

    def close(self) -> None: ...


def _validate_ids(ids: NDArray[np.uint32], rows: int) -> None:
    if (
        type(ids) is not np.ndarray
        or ids.dtype.str != "<u4"
        or ids.ndim != 1
        or ids.shape[0] < 1
        or not ids.flags.c_contiguous
        or bool((ids >= rows).any())
    ):
        raise ValueError("vector read differs")


def _dtype(identity: VectorStoreIdentity) -> np.dtype[np.generic]:
    return np.dtype(np.uint8 if identity.dtype == "uint8" else "<f4")


class _MemoryReadContext:
    def __init__(self, store: MemoryVectorStore) -> None:
        self._store = store
        self._closed = False

    def read(self, ids: NDArray[np.uint32]) -> tuple[NDArray[np.generic], VectorReadEvidence]:
        store = self._store
        if self._closed or store._closed:
            raise ValueError("vector read differs")
        _validate_ids(ids, store.identity.rows)
        selected = _owned_read_only(store._vectors[ids])
        logical_bytes = ids.shape[0] * store.identity.row_stride
        return selected, VectorReadEvidence(
            requested_rows=ids.shape[0],
            logical_bytes=logical_bytes,
            physical_bytes=logical_bytes,
        )

    def close(self) -> None:
        self._closed = True


class MemoryVectorStore:
    """Owned read-only in-memory vector rows."""

    def __init__(
        self,
        vectors: NDArray[np.generic],
        identity: VectorStoreIdentity,
    ) -> None:
        expected_dtype = _dtype(identity)
        if (
            type(identity) is not VectorStoreIdentity
            or type(vectors) is not np.ndarray
            or vectors.dtype != expected_dtype
            or vectors.shape != (identity.rows, identity.dimensions)
            or not vectors.flags.c_contiguous
            or identity.zero_padding_bytes != 0
            or identity.physical_bytes != identity.logical_bytes
        ):
            raise ValueError("memory vector store differs")
        retained_vectors = _owned_read_only(vectors)
        if identity.dtype == "float32" and not bool(np.isfinite(retained_vectors).all()):
            raise ValueError("memory vector store differs")
        header = np.asarray([identity.rows, identity.dimensions], dtype="<u4").tobytes()
        digest = hashlib.sha256(header)
        digest.update(memoryview(retained_vectors).cast("B"))
        if digest.hexdigest() != identity.sha256:
            raise ValueError("memory vector store differs")
        self._identity = identity
        self._vectors = retained_vectors
        self._closed = False

    @property
    def identity(self) -> VectorStoreIdentity:
        """Return the immutable authenticated vector identity."""

        return self._identity

    @property
    def resident_bytes(self) -> int:
        """Return bytes owned by the resident vector matrix."""

        return self._vectors.nbytes

    def new_context(self) -> VectorReadContext:
        if self._closed:
            raise ValueError("memory vector store differs")
        return _MemoryReadContext(self)

    def close(self) -> None:
        self._closed = True


def _pread_exact_into(fd: int, destination: memoryview, offset: int) -> None:
    """Fill ``destination`` from ``fd`` without allocating a second copy of it."""

    size = destination.nbytes
    filled = 0
    while filled < size:
        try:
            if _HAS_PREADV:
                read = os.preadv(fd, (destination[filled:],), offset + filled)
            else:
                block = os.pread(fd, size - filled, offset + filled)
                read = len(block)
                destination[filled : filled + read] = block
        except InterruptedError:
            continue
        if read < 1:
            raise ValueError("vector read differs")
        filled += read


def _pread_exact(fd: int, size: int, offset: int) -> bytes:
    destination = bytearray(size)
    _pread_exact_into(fd, memoryview(destination), offset)
    return bytes(destination)


class _PreadReadContext:
    def __init__(self, store: PreadVectorStore) -> None:
        self._store = store
        self._closed = False

    def read(self, ids: NDArray[np.uint32]) -> tuple[NDArray[np.generic], VectorReadEvidence]:
        store = self._store
        if self._closed or store._closed:
            raise ValueError("vector read differs")
        _validate_ids(ids, store.identity.rows)
        descriptor = store._begin_read()
        try:
            values = np.empty(
                (ids.shape[0], store.identity.dimensions),
                dtype=_dtype(store.identity),
            )
            raw = memoryview(values).cast("B")
            for index, internal_id in enumerate(ids.tolist()):
                begin = index * store.identity.row_stride
                _pread_exact_into(
                    descriptor,
                    raw[begin : begin + store.identity.row_stride],
                    store.identity.header_bytes + int(internal_id) * store.identity.row_stride,
                )
        finally:
            store._end_read()
        values.flags.writeable = False
        logical_bytes = values.nbytes
        return values, VectorReadEvidence(
            requested_rows=ids.shape[0],
            logical_bytes=logical_bytes,
            physical_bytes=logical_bytes,
        )

    def close(self) -> None:
        self._closed = True


class PreadVectorStore:
    """Same-descriptor authenticated portable positional-read vector store."""

    def __init__(self, path: str | Path, identity: VectorStoreIdentity) -> None:
        if type(identity) is not VectorStoreIdentity:
            raise ValueError("pread vector store differs")
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(Path(path), flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != identity.physical_bytes:
                raise ValueError("pread vector store differs")
            digest = hashlib.sha256()
            scratch = bytearray(min(8 << 20, identity.physical_bytes))
            window = memoryview(scratch)
            offset = 0
            while offset < identity.physical_bytes:
                span = min(len(scratch), identity.physical_bytes - offset)
                block = window[:span]
                _pread_exact_into(descriptor, block, offset)
                digest.update(block)
                offset += span
            if digest.hexdigest() != identity.sha256:
                raise ValueError("pread vector store differs")
            header = _pread_exact(descriptor, 8, 0)
            if header != np.asarray([identity.rows, identity.dimensions], dtype="<u4").tobytes():
                raise ValueError("pread vector store differs")
            padding_offset = identity.logical_bytes
            remaining = identity.zero_padding_bytes
            while remaining:
                span = min(len(scratch), remaining)
                block = window[:span]
                _pread_exact_into(descriptor, block, padding_offset)
                if any(block):
                    raise ValueError("pread vector store differs")
                padding_offset += span
                remaining -= span
            after = os.fstat(descriptor)
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, field) != getattr(after, field) for field in stable):
                raise ValueError("pread vector store differs")
        except BaseException:
            os.close(descriptor)
            raise
        self._identity = identity
        self._descriptor = descriptor
        self._authenticated_metadata = tuple(getattr(after, field) for field in stable)
        self._closed = False
        self._closing = False
        self._active_reads = 0
        self._condition = threading.Condition()

    @property
    def identity(self) -> VectorStoreIdentity:
        """Return the immutable authenticated vector identity."""

        return self._identity

    @property
    def resident_bytes(self) -> int:
        """Return resident corpus bytes owned by this descriptor-only store."""

        return 0

    def _begin_read(self) -> int:
        with self._condition:
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if (
                self._closed
                or self._closing
                or tuple(getattr(os.fstat(self._descriptor), field) for field in stable)
                != self._authenticated_metadata
            ):
                raise ValueError("vector read differs")
            self._active_reads += 1
            return self._descriptor

    def _end_read(self) -> None:
        with self._condition:
            self._active_reads -= 1
            self._condition.notify_all()
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if tuple(getattr(os.fstat(self._descriptor), field) for field in stable) != (
                self._authenticated_metadata
            ):
                raise ValueError("pread vector store differs")

    def new_context(self) -> VectorReadContext:
        with self._condition:
            if self._closed or self._closing:
                raise ValueError("pread vector store differs")
            return _PreadReadContext(self)

    def close(self) -> None:
        with self._condition:
            while self._closing and not self._closed:
                self._condition.wait()
            if self._closed:
                return
            self._closing = True
            try:
                while self._active_reads:
                    self._condition.wait()
            except BaseException:
                self._closing = False
                self._condition.notify_all()
                raise
            try:
                os.close(self._descriptor)
            finally:
                self._closed = True
                self._closing = False
                self._condition.notify_all()


class DirectIoVectorStore:
    """Linux direct-I/O descriptor paired with an authenticated positional-read store."""

    def __init__(self, path: str | Path, identity: VectorStoreIdentity) -> None:
        if not hasattr(os, "O_DIRECT"):
            raise ValueError("direct I/O vector store differs")
        authenticated = PreadVectorStore(path, identity)
        descriptor = -1
        try:
            descriptor = os.open(
                Path(path),
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECT | getattr(os, "O_NOFOLLOW", 0),
            )
            authenticated_stat = os.fstat(authenticated._descriptor)
            direct_stat = os.fstat(descriptor)
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(
                getattr(authenticated_stat, field) != getattr(direct_stat, field)
                for field in stable
            ):
                raise ValueError("direct I/O vector store differs")
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            authenticated.close()
            raise
        self._authenticated = authenticated
        self._descriptor = descriptor
        self._closed = False
        self._closing = False
        self._active_reads = 0
        self._condition = threading.Condition()
        self._process_id = os.getpid()

    @property
    def identity(self) -> VectorStoreIdentity:
        """Return the immutable authenticated vector identity."""

        return self._authenticated.identity

    @property
    def resident_bytes(self) -> int:
        """Return resident corpus bytes owned by this descriptor-only store."""

        return 0

    def new_context(self) -> VectorReadContext:
        """Return the portable authenticated context for non-native callers."""

        with self._condition:
            if self._closed or self._closing or os.getpid() != self._process_id:
                raise ValueError("direct I/O vector store differs")
            return self._authenticated.new_context()

    def _begin_direct_read(self) -> int:
        with self._condition:
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if (
                self._closed
                or self._closing
                or os.getpid() != self._process_id
                or tuple(getattr(os.fstat(self._descriptor), field) for field in stable)
                != self._authenticated._authenticated_metadata
            ):
                raise ValueError("direct vector read differs")
            self._active_reads += 1
            return self._descriptor

    def _end_direct_read(self) -> None:
        with self._condition:
            self._active_reads -= 1
            self._condition.notify_all()
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if tuple(getattr(os.fstat(self._descriptor), field) for field in stable) != (
                self._authenticated._authenticated_metadata
            ):
                raise ValueError("direct I/O vector store differs")

    def close(self) -> None:
        """Wait for direct reads before closing both bound descriptors."""

        with self._condition:
            while self._closing and not self._closed:
                self._condition.wait()
            if self._closed:
                return
            self._closing = True
            try:
                while self._active_reads:
                    self._condition.wait()
            except BaseException:
                self._closing = False
                self._condition.notify_all()
                raise
            first_error: BaseException | None = None
            try:
                self._authenticated.close()
            except BaseException as error:
                if not self._authenticated._closed:
                    self._closing = False
                    self._condition.notify_all()
                    raise
                first_error = error
            descriptor = self._descriptor
            self._descriptor = -1
            try:
                os.close(descriptor)
            except BaseException as error:
                if first_error is None:
                    first_error = error
            finally:
                self._closed = True
                self._closing = False
                self._condition.notify_all()
            if first_error is not None:
                raise first_error


@dataclass(frozen=True, slots=True)
class ExactSearchResult:
    """Exact squared-L2 results and vector-read evidence."""

    ids: NDArray[np.uint32]
    squared_distances: NDArray[np.float64]
    vector_reads: VectorReadEvidence

    def __post_init__(self) -> None:
        if (
            type(self.ids) is not np.ndarray
            or self.ids.dtype.str != "<u4"
            or self.ids.ndim != 1
            or type(self.squared_distances) is not np.ndarray
            or self.squared_distances.dtype.str != "<f8"
            or self.squared_distances.shape != self.ids.shape
            or not bool(np.isfinite(self.squared_distances).all())
            or type(self.vector_reads) is not VectorReadEvidence
        ):
            raise ValueError("exact search result differs")
        object.__setattr__(self, "ids", _owned_read_only(self.ids))
        object.__setattr__(self, "squared_distances", _owned_read_only(self.squared_distances))


class ExactReranker:
    """Deterministic exact squared-L2 reranking over a vector store."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def search(
        self,
        query: NDArray[np.generic],
        candidates: CandidateResult,
        *,
        return_width: int,
    ) -> ExactSearchResult:
        identity = self._store.identity
        if (
            type(candidates) is CandidateResult
            and type(return_width) is int
            and return_width > candidates.ids.shape[0]
        ):
            raise InsufficientCandidatesError("insufficient exact rerank candidates")
        if (
            type(candidates) is not CandidateResult
            or type(return_width) is not int
            or not 1 <= return_width <= candidates.ids.shape[0]
            or type(query) is not np.ndarray
            or query.dtype != _dtype(identity)
            or query.shape != (identity.dimensions,)
            or not query.flags.c_contiguous
            or not bool(np.isfinite(query).all())
        ):
            raise ValueError("exact rerank differs")
        context = self._store.new_context()
        try:
            vectors, evidence = context.read(candidates.ids)
        finally:
            context.close()
        ranked: list[tuple[float, int]] = []
        for index, internal_id in enumerate(candidates.ids.tolist()):
            if identity.dtype == "uint8":
                distance_integer = 0
                for dimension in range(identity.dimensions):
                    difference = int(query[dimension]) - int(vectors[index, dimension])
                    distance_integer += difference * difference
                    if distance_integer > 2**64 - 1:
                        raise ValueError("exact rerank differs")
                distance = float(distance_integer)
            else:
                distance = 0.0
                for dimension in range(identity.dimensions):
                    float_difference = float(query[dimension]) - float(vectors[index, dimension])
                    distance += float_difference * float_difference
            if not np.isfinite(distance):
                raise ValueError("exact rerank differs")
            ranked.append((distance, int(internal_id)))
        ranked.sort()
        selected = ranked[:return_width]
        return ExactSearchResult(
            ids=np.asarray([internal_id for _, internal_id in selected], dtype="<u4"),
            squared_distances=np.asarray([distance for distance, _ in selected], dtype="<f8"),
            vector_reads=evidence,
        )
