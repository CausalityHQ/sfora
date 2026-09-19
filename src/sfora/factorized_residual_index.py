"""Public composition of authenticated candidates and exact vector refinement."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sfora.factorized_residual_ann import (
    CandidateResult,
    FactorizedResidualArtifact,
    PortableCandidateIndex,
    _owned_read_only,
)
from sfora.factorized_residual_native import (
    NativeBackend,
    NativeCandidateIndex,
    NativeExactReranker,
)
from sfora.vector_store import (
    DirectIoVectorStore,
    ExactReranker,
    ExactSearchResult,
    VectorReadEvidence,
    VectorStore,
)


class _CandidateIndex(Protocol):
    def search(self, query: NDArray[np.generic]) -> CandidateResult: ...


class _ExactIndex(Protocol):
    def search(
        self,
        query: NDArray[np.generic],
        candidates: CandidateResult,
        *,
        return_width: int,
    ) -> ExactSearchResult: ...


class _Closable(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FactorizedResidualMemoryLedger:
    """Pre-admission resident and per-search memory accounting."""

    artifact_resident_bytes: int
    vector_store_resident_bytes: int
    fixed_service_overhead_bytes: int
    safety_headroom_bytes: int
    context_bytes: int
    total_reserved_bytes: int
    memory_limit_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.artifact_resident_bytes,
            self.vector_store_resident_bytes,
            self.fixed_service_overhead_bytes,
            self.safety_headroom_bytes,
            self.context_bytes,
            self.total_reserved_bytes,
            self.memory_limit_bytes,
        )
        if (
            any(type(value) is not int or value < 0 for value in values)
            or self.memory_limit_bytes < 1
            or self.total_reserved_bytes
            != sum(values[:5])
            or self.total_reserved_bytes > self.memory_limit_bytes
        ):
            raise ValueError("factorized residual memory ledger differs")


@dataclass(frozen=True, slots=True)
class FactorizedResidualSearchResult:
    """Exact neighbors plus candidate, I/O, and stage timing evidence."""

    ids: NDArray[np.uint32]
    squared_distances: NDArray[np.float64]
    candidates: CandidateResult
    vector_reads: VectorReadEvidence
    candidate_ns: int
    exact_ns: int

    def __post_init__(self) -> None:
        if (
            type(self.ids) is not np.ndarray
            or self.ids.dtype.str != "<u4"
            or self.ids.ndim != 1
            or type(self.squared_distances) is not np.ndarray
            or self.squared_distances.dtype.str != "<f8"
            or self.squared_distances.shape != self.ids.shape
            or not bool(np.isfinite(self.squared_distances).all())
            or type(self.candidates) is not CandidateResult
            or type(self.vector_reads) is not VectorReadEvidence
            or type(self.candidate_ns) is not int
            or self.candidate_ns < 0
            or type(self.exact_ns) is not int
            or self.exact_ns < 0
        ):
            raise ValueError("factorized residual search result differs")
        object.__setattr__(self, "ids", _owned_read_only(self.ids))
        object.__setattr__(
            self, "squared_distances", _owned_read_only(self.squared_distances)
        )


class FactorizedResidualIndex:
    """Owned ANN index over trusted immutable files with explicit admission."""

    def __init__(
        self,
        artifact: FactorizedResidualArtifact,
        store: VectorStore,
        ledger: FactorizedResidualMemoryLedger,
        candidate_backend: NativeBackend | None,
        thread_count: int,
        exact_reranker: _ExactIndex | None,
        context_limit: int,
    ) -> None:
        self._artifact = artifact
        self._store = store
        self._backend = candidate_backend
        if candidate_backend is None:
            self._candidate_index: _CandidateIndex = PortableCandidateIndex(artifact)
            self._exact_reranker: _ExactIndex = ExactReranker(store)
        else:
            self._candidate_index = NativeCandidateIndex(
                artifact, candidate_backend, thread_count=thread_count
            )
            self._exact_reranker = exact_reranker or ExactReranker(store)
        self._ledger = ledger
        self._condition = threading.Condition()
        self._active_searches = 0
        self._context_limit = context_limit
        self._closed = False
        self._closing = False
        self._cleanup_pending = False

    @classmethod
    def open(
        cls,
        artifact: FactorizedResidualArtifact,
        store: VectorStore,
        *,
        candidate_backend: NativeBackend | None = None,
        thread_count: int = 1,
        memory_limit_bytes: int,
        fixed_service_overhead_bytes: int,
        safety_headroom_bytes: int,
    ) -> FactorizedResidualIndex:
        """Bind artifact and vectors, rejecting an over-budget owner before allocation."""

        if (
            type(artifact) is not FactorizedResidualArtifact
            or artifact._closed
            or not hasattr(store, "identity")
            or not hasattr(store, "resident_bytes")
            or store.identity != artifact.vector_store_identity
            or (candidate_backend is not None and type(candidate_backend) is not NativeBackend)
            or (type(candidate_backend) is NativeBackend and candidate_backend._closed)
            or type(thread_count) is not int
            or not 1 <= thread_count <= 256
            or type(memory_limit_bytes) is not int
            or memory_limit_bytes < 1
            or type(fixed_service_overhead_bytes) is not int
            or fixed_service_overhead_bytes < 0
            or type(safety_headroom_bytes) is not int
            or safety_headroom_bytes < 0
        ):
            raise ValueError("factorized residual index differs")
        spec = artifact.spec
        exact_reranker: _ExactIndex | None = None
        context_bytes = (
            spec.shortlist_width * (4 + 8 + store.identity.row_stride)
            + spec.probe_count * 4
            + spec.return_width * (4 + 8)
        )
        if candidate_backend is None:
            portable_chunk_rows = min(artifact.rows, 65_536)
            context_bytes += (
                (16 << 20)
                + spec.dimensions * 8
                + spec.subquantizers * spec.codebook_size * 8
                + portable_chunk_rows * 64
                + spec.probe_count * 256
            )
        elif type(candidate_backend) is NativeBackend:
            context_bytes += (
                spec.probe_count * 8
                + thread_count * spec.shortlist_width * 8
                + thread_count * 8
                + spec.subquantizers * spec.codebook_size * 4
            )
            if isinstance(store, DirectIoVectorStore):
                exact_reranker = NativeExactReranker(
                    store, candidate_backend, thread_count=thread_count
                )
                context_bytes += exact_reranker.context_bytes(spec.shortlist_width)
        # Every stage that materialises Python objects per candidate needs its own
        # allowance: the portable scorer's heap and ordered result, and the Python
        # exact reranker's ranked tuples. A native candidate backend paired with a
        # non-direct store still reranks in Python, so this cannot hang off the
        # candidate backend alone. Measured marginal cost is ~273 B/candidate.
        if candidate_backend is None or exact_reranker is None:
            context_bytes += spec.shortlist_width * 384 + spec.return_width * 64
        total = (
            artifact.resident_bytes
            + store.resident_bytes
            + fixed_service_overhead_bytes
            + safety_headroom_bytes
            + context_bytes
        )
        if total > memory_limit_bytes:
            raise MemoryError("factorized residual memory budget exceeded")
        base_bytes = total - context_bytes
        context_limit = (memory_limit_bytes - base_bytes) // context_bytes
        if candidate_backend is None or (
            type(candidate_backend) is NativeBackend
            and isinstance(store, DirectIoVectorStore)
        ):
            context_limit = min(context_limit, 1)
        return cls(
            artifact,
            store,
            FactorizedResidualMemoryLedger(
                artifact_resident_bytes=artifact.resident_bytes,
                vector_store_resident_bytes=store.resident_bytes,
                fixed_service_overhead_bytes=fixed_service_overhead_bytes,
                safety_headroom_bytes=safety_headroom_bytes,
                context_bytes=context_bytes,
                total_reserved_bytes=total,
                memory_limit_bytes=memory_limit_bytes,
            ),
            candidate_backend,
            thread_count,
            exact_reranker,
            context_limit,
        )

    def memory_ledger(self) -> FactorizedResidualMemoryLedger:
        """Return immutable admission arithmetic."""

        return self._ledger

    def search(self, query: NDArray[np.generic]) -> FactorizedResidualSearchResult:
        """Run candidate selection followed by exact deterministic reranking."""

        with self._condition:
            if self._closed or self._closing:
                raise ValueError("factorized residual index differs")
            if self._active_searches >= self._context_limit:
                raise MemoryError("factorized residual memory budget exceeded")
            self._active_searches += 1
        try:
            candidate_started = time.perf_counter_ns()
            candidates = self._candidate_index.search(query)
            candidate_ns = time.perf_counter_ns() - candidate_started
            exact_started = time.perf_counter_ns()
            exact = self._exact_reranker.search(
                query, candidates, return_width=self._artifact.spec.return_width
            )
            exact_ns = time.perf_counter_ns() - exact_started
            return FactorizedResidualSearchResult(
                ids=exact.ids,
                squared_distances=exact.squared_distances,
                candidates=candidates,
                vector_reads=exact.vector_reads,
                candidate_ns=candidate_ns,
                exact_ns=exact_ns,
            )
        finally:
            with self._condition:
                self._active_searches -= 1
                self._condition.notify_all()

    def close(self) -> None:
        """Reject new work, drain active searches, and release owned inputs."""

        with self._condition:
            while self._closing:
                self._condition.wait()
            if self._closed and not self._cleanup_pending:
                return
            self._closing = True
            if not self._closed:
                try:
                    while self._active_searches:
                        self._condition.wait()
                except BaseException:
                    self._closing = False
                    self._condition.notify_all()
                    raise
                self._closed = True
            self._cleanup_pending = True
        first_error: BaseException | None = None
        owners: list[_Closable] = [self._store, self._artifact]
        if self._backend is not None:
            owners.append(self._backend)
        for owner in owners:
            try:
                owner.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        with self._condition:
            self._cleanup_pending = first_error is not None
            self._closing = False
            self._condition.notify_all()
        if first_error is not None:
            raise first_error
