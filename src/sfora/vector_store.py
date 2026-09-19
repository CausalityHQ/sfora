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

    def read(
        self, ids: NDArray[np.uint32]
    ) -> tuple[NDArray[np.generic], VectorReadEvidence]: ...

    def close(self) -> None: ...


class VectorStore(Protocol):
    """Immutable vector authority with isolated read contexts."""

    identity: VectorStoreIdentity

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

    def read(
        self, ids: NDArray[np.uint32]
    ) -> tuple[NDArray[np.generic], VectorReadEvidence]:
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

    def new_context(self) -> VectorReadContext:
        if self._closed:
            raise ValueError("memory vector store differs")
        return _MemoryReadContext(self)

    def close(self) -> None:
        self._closed = True


def _pread_exact(fd: int, size: int, offset: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        try:
            block = os.pread(fd, size - len(result), offset + len(result))
        except InterruptedError:
            continue
        if not block:
            raise ValueError("vector read differs")
        result.extend(block)
    return bytes(result)


class _PreadReadContext:
    def __init__(self, store: PreadVectorStore) -> None:
        self._store = store
        self._closed = False

    def read(
        self, ids: NDArray[np.uint32]
    ) -> tuple[NDArray[np.generic], VectorReadEvidence]:
        store = self._store
        if self._closed or store._closed:
            raise ValueError("vector read differs")
        _validate_ids(ids, store.identity.rows)
        descriptor = store._begin_read()
        try:
            raw = bytearray(ids.shape[0] * store.identity.row_stride)
            for index, internal_id in enumerate(ids.tolist()):
                row = _pread_exact(
                    descriptor,
                    store.identity.row_stride,
                    store.identity.header_bytes
                    + int(internal_id) * store.identity.row_stride,
                )
                begin = index * store.identity.row_stride
                raw[begin : begin + store.identity.row_stride] = row
        finally:
            store._end_read()
        values = np.frombuffer(bytes(raw), dtype=_dtype(store.identity)).reshape(
            ids.shape[0], store.identity.dimensions
        )
        logical_bytes = len(raw)
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
            offset = 0
            while offset < identity.physical_bytes:
                block = _pread_exact(
                    descriptor,
                    min(8 << 20, identity.physical_bytes - offset),
                    offset,
                )
                digest.update(block)
                offset += len(block)
            if digest.hexdigest() != identity.sha256:
                raise ValueError("pread vector store differs")
            header = _pread_exact(descriptor, 8, 0)
            if header != np.asarray([identity.rows, identity.dimensions], dtype="<u4").tobytes():
                raise ValueError("pread vector store differs")
            padding_offset = identity.logical_bytes
            remaining = identity.zero_padding_bytes
            while remaining:
                block = _pread_exact(descriptor, min(8 << 20, remaining), padding_offset)
                if any(block):
                    raise ValueError("pread vector store differs")
                padding_offset += len(block)
                remaining -= len(block)
            after = os.fstat(descriptor)
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, field) != getattr(after, field) for field in stable):
                raise ValueError("pread vector store differs")
        except BaseException:
            os.close(descriptor)
            raise
        self._identity = identity
        self._descriptor = descriptor
        self._closed = False
        self._closing = False
        self._active_reads = 0
        self._condition = threading.Condition()

    @property
    def identity(self) -> VectorStoreIdentity:
        """Return the immutable authenticated vector identity."""

        return self._identity

    def _begin_read(self) -> int:
        with self._condition:
            if self._closed or self._closing:
                raise ValueError("vector read differs")
            self._active_reads += 1
            return self._descriptor

    def _end_read(self) -> None:
        with self._condition:
            self._active_reads -= 1
            self._condition.notify_all()

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
            while self._active_reads:
                self._condition.wait()
            try:
                os.close(self._descriptor)
            finally:
                self._closed = True
                self._closing = False
                self._condition.notify_all()


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
                    float_difference = float(query[dimension]) - float(
                        vectors[index, dimension]
                    )
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
