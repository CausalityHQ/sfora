from __future__ import annotations

import hashlib
import threading
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

import sfora
from sfora.factorized_residual_ann import (
    FactorizedResidualArtifact,
    FactorizedResidualComponents,
    FactorizedResidualPostings,
    FactorizedResidualSpec,
    VectorStoreIdentity,
    write_factorized_residual_artifact,
)
from sfora.factorized_residual_index import (
    FactorizedResidualIndex,
    FactorizedResidualSearchResult,
)
from sfora.factorized_residual_native import compile_factorized_residual_backend
from sfora.vector_store import DirectIoVectorStore, MemoryVectorStore, PreadVectorStore


def _open_fixture(tmp_path: Path) -> tuple[FactorizedResidualArtifact, MemoryVectorStore]:
    vectors = np.array(
        [[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0], [2, 2, 2, 2]],
        dtype=np.uint8,
    )
    payload = np.asarray([4, 4], dtype="<u4").tobytes() + vectors.tobytes()
    identity = VectorStoreIdentity(
        sha256=hashlib.sha256(payload).hexdigest(),
        logical_bytes=len(payload),
        physical_bytes=len(payload),
        rows=4,
        dimensions=4,
        dtype="uint8",
        header_bytes=8,
        row_stride=4,
        zero_padding_bytes=0,
        generation="public-index-fixture",
    )
    spec = FactorizedResidualSpec(
        metric="squared_l2",
        vector_dtype="uint8",
        dimensions=4,
        list_count=2,
        subquantizers=2,
        bits_per_subquantizer=3,
        probe_count=2,
        shortlist_width=4,
        return_width=2,
    )
    components = FactorizedResidualComponents(
        spec,
        np.array([[0, 0, 0, 0], [8, 8, 8, 8]], dtype="<f4"),
        np.zeros((2, 8, 2), dtype="<f4"),
    )
    postings = FactorizedResidualPostings(
        spec,
        np.array([0, 2, 4], dtype="<u8"),
        np.array([0, 2, 1, 3], dtype="<u4"),
        np.zeros((4, 1), dtype=np.uint8),
    )
    root = tmp_path / "artifact"
    manifest = write_factorized_residual_artifact(root, spec, components, postings, identity)
    return FactorizedResidualArtifact.open(root, manifest_sha256=manifest), MemoryVectorStore(
        vectors, identity
    )


def test_public_factorized_index_returns_exact_result_and_memory_evidence(
    tmp_path: Path,
) -> None:
    artifact, store = _open_fixture(tmp_path)
    index = FactorizedResidualIndex.open(
        artifact,
        store,
        memory_limit_bytes=64 << 20,
        fixed_service_overhead_bytes=4096,
        safety_headroom_bytes=4096,
    )

    result = index.search(np.array([1, 2, 3, 4], dtype=np.uint8))

    assert type(result) is FactorizedResidualSearchResult
    assert result.ids.tolist() == [0, 3]
    assert result.squared_distances.tolist() == [0.0, 6.0]
    assert result.candidates.evidence.backend == "portable-float64"
    assert result.vector_reads.requested_rows == 4
    assert result.candidate_ns >= 0
    assert result.exact_ns >= 0
    ledger = index.memory_ledger()
    assert ledger.artifact_resident_bytes == artifact.resident_bytes
    assert ledger.vector_store_resident_bytes == 16
    assert ledger.total_reserved_bytes <= ledger.memory_limit_bytes
    assert sfora.FactorizedResidualIndex is FactorizedResidualIndex

    index.close()
    index.close()
    with pytest.raises(ValueError, match="factorized residual index differs"):
        index.search(np.array([1, 2, 3, 4], dtype=np.uint8))


def test_portable_memory_ledger_includes_numpy_and_python_search_scratch(
    tmp_path: Path,
) -> None:
    artifact, store = _open_fixture(tmp_path)
    index = FactorizedResidualIndex.open(
        artifact,
        store,
        memory_limit_bytes=64 << 20,
        fixed_service_overhead_bytes=0,
        safety_headroom_bytes=0,
    )

    assert index.memory_ledger().context_bytes >= 16 << 20

    index.close()


def test_public_factorized_index_rejects_vector_identity_and_memory_budget(
    tmp_path: Path,
) -> None:
    artifact, store = _open_fixture(tmp_path)
    different_vectors = np.array([[0, 0, 0, 0]] * 4, dtype=np.uint8)
    different_payload = np.asarray([4, 4], dtype="<u4").tobytes() + different_vectors.tobytes()
    different = MemoryVectorStore(
        different_vectors,
        VectorStoreIdentity(
            sha256=hashlib.sha256(different_payload).hexdigest(),
            logical_bytes=len(different_payload),
            physical_bytes=len(different_payload),
            rows=4,
            dimensions=4,
            dtype="uint8",
            header_bytes=8,
            row_stride=4,
            zero_padding_bytes=0,
            generation="different",
        ),
    )

    with pytest.raises(ValueError, match="factorized residual index differs"):
        FactorizedResidualIndex.open(
            artifact,
            different,
            memory_limit_bytes=1 << 20,
            fixed_service_overhead_bytes=4096,
            safety_headroom_bytes=4096,
        )
    with pytest.raises(MemoryError, match="factorized residual memory budget exceeded"):
        FactorizedResidualIndex.open(
            artifact,
            store,
            memory_limit_bytes=1,
            fixed_service_overhead_bytes=0,
            safety_headroom_bytes=0,
        )

    different.close()
    store.close()
    artifact.close()


def test_public_factorized_index_uses_explicit_native_direct_backend(tmp_path: Path) -> None:
    vectors = np.array(
        [[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0], [2, 2, 2, 2]],
        dtype=np.uint8,
    )
    logical = np.asarray([4, 4], dtype="<u4").tobytes() + vectors.tobytes()
    physical = logical + b"\0" * (4096 - len(logical))
    vector_path = tmp_path / "vectors.bin"
    vector_path.write_bytes(physical)
    identity = VectorStoreIdentity(
        sha256=hashlib.sha256(physical).hexdigest(),
        logical_bytes=len(logical),
        physical_bytes=len(physical),
        rows=4,
        dimensions=4,
        dtype="uint8",
        header_bytes=8,
        row_stride=4,
        zero_padding_bytes=len(physical) - len(logical),
        generation="public-native-fixture",
    )
    spec = FactorizedResidualSpec(
        metric="squared_l2",
        vector_dtype="uint8",
        dimensions=4,
        list_count=2,
        subquantizers=2,
        bits_per_subquantizer=3,
        probe_count=2,
        shortlist_width=4,
        return_width=2,
    )
    components = FactorizedResidualComponents(
        spec,
        np.array([[0, 0, 0, 0], [8, 8, 8, 8]], dtype="<f4"),
        np.zeros((2, 8, 2), dtype="<f4"),
    )
    postings = FactorizedResidualPostings(
        spec,
        np.array([0, 2, 4], dtype="<u8"),
        np.array([0, 2, 1, 3], dtype="<u4"),
        np.zeros((4, 1), dtype=np.uint8),
    )
    root = tmp_path / "artifact"
    manifest = write_factorized_residual_artifact(root, spec, components, postings, identity)
    artifact = FactorizedResidualArtifact.open(root, manifest_sha256=manifest)
    store = DirectIoVectorStore(vector_path, identity)
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    index = FactorizedResidualIndex.open(
        artifact,
        store,
        candidate_backend=backend,
        thread_count=2,
        memory_limit_bytes=1 << 20,
        fixed_service_overhead_bytes=4096,
        safety_headroom_bytes=4096,
    )

    result = index.search(np.array([1, 2, 3, 4], dtype=np.uint8))

    assert result.ids.tolist() == [0, 3]
    assert result.squared_distances.tolist() == [0.0, 6.0]
    assert result.candidates.evidence.backend == "native-c11-fma"
    assert result.vector_reads.physical_bytes >= result.vector_reads.logical_bytes
    assert index.memory_ledger().context_bytes >= 4 * 512
    assert index._context_limit == 1
    index.close()
    assert backend._closed


def test_public_factorized_index_rejects_over_budget_concurrent_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, store = _open_fixture(tmp_path)
    generous = FactorizedResidualIndex.open(
        artifact,
        store,
        memory_limit_bytes=64 << 20,
        fixed_service_overhead_bytes=4096,
        safety_headroom_bytes=4096,
    )
    exact_limit = generous.memory_ledger().total_reserved_bytes
    generous._closed = True
    index = FactorizedResidualIndex.open(
        artifact,
        store,
        memory_limit_bytes=exact_limit,
        fixed_service_overhead_bytes=4096,
        safety_headroom_bytes=4096,
    )
    entered = threading.Event()
    release = threading.Event()
    real_search = index._candidate_index.search

    def blocking_search(query: np.ndarray) -> object:
        entered.set()
        assert release.wait(5)
        return real_search(query)

    monkeypatch.setattr(index._candidate_index, "search", blocking_search)
    results: list[FactorizedResidualSearchResult] = []
    thread = threading.Thread(
        target=lambda: results.append(
            index.search(np.array([1, 2, 3, 4], dtype=np.uint8))
        )
    )
    thread.start()
    assert entered.wait(5)

    with pytest.raises(MemoryError, match="factorized residual memory budget exceeded"):
        index.search(np.array([1, 2, 3, 4], dtype=np.uint8))

    release.set()
    thread.join(5)
    assert len(results) == 1
    index.close()


def test_public_index_retries_interrupted_owner_cleanup_without_reopening_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, store = _open_fixture(tmp_path)
    index = FactorizedResidualIndex.open(
        artifact,
        store,
        memory_limit_bytes=64 << 20,
        fixed_service_overhead_bytes=4096,
        safety_headroom_bytes=4096,
    )
    real_close = store.close
    calls = 0

    def interrupt_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt("fixture cleanup interruption")
        real_close()

    monkeypatch.setattr(store, "close", interrupt_once)

    with pytest.raises(KeyboardInterrupt, match="fixture cleanup interruption"):
        index.close()
    with pytest.raises(ValueError, match="factorized residual index differs"):
        index.search(np.array([1, 2, 3, 4], dtype=np.uint8))

    index.close()

    assert calls == 2
    assert store._closed
    assert artifact._closed


def _wide_shortlist_parts(
    tmp_path: Path, *, rows: int, shortlist: int, probe: int
) -> tuple[FactorizedResidualArtifact, VectorStoreIdentity, Path, np.ndarray]:
    """Build a one-dimensional artifact whose shortlist dominates the ledger."""

    lists, subquantizers, bits = 2, 1, 8
    rng = np.random.default_rng(7)
    vectors = rng.integers(0, 256, size=(rows, 1), dtype=np.uint8)
    payload = np.asarray([rows, 1], dtype="<u4").tobytes() + vectors.tobytes()
    path = tmp_path / "wide-vectors.u8"
    path.write_bytes(payload)
    identity = VectorStoreIdentity(
        sha256=hashlib.sha256(payload).hexdigest(),
        logical_bytes=len(payload),
        physical_bytes=len(payload),
        rows=rows,
        dimensions=1,
        dtype="uint8",
        header_bytes=8,
        row_stride=1,
        zero_padding_bytes=0,
        generation="wide-shortlist-fixture",
    )
    spec = FactorizedResidualSpec(
        metric="squared_l2",
        vector_dtype="uint8",
        dimensions=1,
        list_count=lists,
        subquantizers=subquantizers,
        bits_per_subquantizer=bits,
        probe_count=probe,
        shortlist_width=shortlist,
        return_width=4,
    )
    components = FactorizedResidualComponents(
        spec,
        rng.random((lists, 1), dtype=np.float32) * 255.0,
        rng.random((subquantizers, 1 << bits, 1), dtype=np.float32),
    )
    postings = FactorizedResidualPostings(
        spec,
        np.arange(lists + 1, dtype="<u8") * (rows // lists),
        np.arange(rows, dtype="<u4"),
        rng.integers(0, 256, size=(rows, subquantizers * bits // 8), dtype=np.uint8),
    )
    root = tmp_path / "wide-artifact"
    manifest = write_factorized_residual_artifact(root, spec, components, postings, identity)
    artifact = FactorizedResidualArtifact.open(root, manifest_sha256=manifest)
    return artifact, identity, path, vectors


@pytest.mark.parametrize("backend_kind", ("portable", "native"))
def test_search_peak_allocation_stays_within_the_admission_ledger(
    tmp_path: Path,
    backend_kind: str,
) -> None:
    """A whole search must fit the reservation that admitted it.

    Both stages that build Python objects per candidate are exercised: the
    portable scorer, and the Python exact reranker that a native candidate
    backend still uses when the store is not direct-I/O.
    """

    rows, shortlist = 16384, 8192
    artifact, identity, path, vectors = _wide_shortlist_parts(
        tmp_path, rows=rows, shortlist=shortlist, probe=2
    )
    backend = (
        compile_factorized_residual_backend(tmp_path / "cache")
        if backend_kind == "native"
        else None
    )
    store = (
        MemoryVectorStore(vectors, identity)
        if backend_kind == "native"
        else PreadVectorStore(path, identity)
    )
    index = FactorizedResidualIndex.open(
        artifact,
        store,
        candidate_backend=backend,
        thread_count=1,
        memory_limit_bytes=3 * 1024**3,
        fixed_service_overhead_bytes=128 * 1024**2,
        safety_headroom_bytes=64 * 1024**2,
    )
    query = np.zeros(1, dtype=np.uint8)
    index.search(query)

    tracemalloc.start()
    result = index.search(query)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result.candidates.ids.shape[0] == shortlist
    assert peak <= index.memory_ledger().context_bytes

    index.close()


def test_admission_reserves_python_object_scratch_per_candidate(tmp_path: Path) -> None:
    """The per-candidate reservation must exceed the measured ~273 B marginal cost."""

    narrow, wide = 1024, 9216
    reservations = []
    for index_number, shortlist in enumerate((narrow, wide)):
        case = tmp_path / f"case{index_number}"
        case.mkdir()
        artifact, identity, _path, vectors = _wide_shortlist_parts(
            case, rows=16384, shortlist=shortlist, probe=2
        )
        index = FactorizedResidualIndex.open(
            artifact,
            MemoryVectorStore(vectors, identity),
            memory_limit_bytes=3 * 1024**3,
            fixed_service_overhead_bytes=0,
            safety_headroom_bytes=0,
        )
        reservations.append(index.memory_ledger().context_bytes)
        index.close()

    marginal = (reservations[1] - reservations[0]) / (wide - narrow)
    assert marginal >= 384
