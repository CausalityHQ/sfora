from __future__ import annotations

import builtins
import hashlib
import threading
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

import sfora
import sfora.vector_store as vector_store_module
from sfora.factorized_residual_ann import (
    CandidateEvidence,
    CandidateResult,
    VectorStoreIdentity,
)
from sfora.vector_store import (
    DirectIoVectorStore,
    ExactReranker,
    MemoryVectorStore,
    PreadVectorStore,
    VectorReadEvidence,
)


def _identity(
    payload: bytes,
    *,
    rows: int,
    dimensions: int,
    dtype: str = "uint8",
    zero_padding_bytes: int = 0,
) -> VectorStoreIdentity:
    physical = payload + b"\0" * zero_padding_bytes
    item_bytes = 1 if dtype == "uint8" else 4
    return VectorStoreIdentity(
        sha256=hashlib.sha256(physical).hexdigest(),
        logical_bytes=8 + rows * dimensions * item_bytes,
        physical_bytes=len(physical),
        rows=rows,
        dimensions=dimensions,
        dtype=dtype,  # type: ignore[arg-type]
        header_bytes=8,
        row_stride=dimensions * item_bytes,
        zero_padding_bytes=zero_padding_bytes,
        generation="vector-fixture",
    )


def _uint8_vectors() -> np.ndarray:
    return np.array([[0, 0], [10, 10], [2, 2]], dtype=np.uint8)


def _memory_identity() -> VectorStoreIdentity:
    vectors = _uint8_vectors()
    payload = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    return _identity(payload, rows=3, dimensions=2)


def test_vector_store_api_is_public() -> None:
    assert sfora.MemoryVectorStore is MemoryVectorStore
    assert sfora.PreadVectorStore is PreadVectorStore
    assert sfora.ExactReranker is ExactReranker


def test_memory_vector_store_owns_rows_and_preserves_duplicate_read_order() -> None:
    vectors = _uint8_vectors()
    store = MemoryVectorStore(vectors, _memory_identity())
    vectors[2] = 99
    context = store.new_context()

    observed, evidence = context.read(np.array([2, 0, 2], dtype="<u4"))

    np.testing.assert_array_equal(observed, np.array([[2, 2], [0, 0], [2, 2]], np.uint8))
    assert type(evidence) is VectorReadEvidence
    assert evidence.requested_rows == 3
    assert evidence.logical_bytes == 6
    assert evidence.physical_bytes == 6
    assert not observed.flags.writeable
    context.close()
    store.close()


def test_memory_store_authenticates_the_retained_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = _uint8_vectors()
    real_owned = vector_store_module._owned_read_only

    def mutate_before_copy(value: np.ndarray) -> np.ndarray:
        vectors[0, 0] = 99
        return real_owned(value)

    monkeypatch.setattr(vector_store_module, "_owned_read_only", mutate_before_copy)

    with pytest.raises(ValueError, match="memory vector store differs"):
        MemoryVectorStore(vectors, _memory_identity())


def test_vector_store_identity_cannot_be_reassigned(tmp_path: Path) -> None:
    memory = MemoryVectorStore(_uint8_vectors(), _memory_identity())
    with pytest.raises(AttributeError):
        memory.identity = _memory_identity()
    memory.close()

    vectors = _uint8_vectors()
    payload = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    path = tmp_path / "vectors.bin"
    path.write_bytes(payload)
    pread = PreadVectorStore(path, _identity(payload, rows=3, dimensions=2))
    with pytest.raises(AttributeError):
        pread.identity = _memory_identity()
    pread.close()


def test_direct_store_close_exhausts_cleanup_without_retrying_released_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = _uint8_vectors()
    logical = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    payload = logical + b"\0" * (4096 - len(logical))
    path = tmp_path / "vectors-direct.bin"
    path.write_bytes(payload)
    store = DirectIoVectorStore(
        path,
        _identity(logical, rows=3, dimensions=2, zero_padding_bytes=4096 - len(logical)),
    )
    direct_descriptor = store._descriptor
    authenticated_descriptor = store._authenticated._descriptor
    real_close = vector_store_module.os.close
    closed: list[int] = []
    rejected_once = False

    def reject_direct_once(descriptor: int) -> None:
        nonlocal rejected_once
        closed.append(descriptor)
        if descriptor == direct_descriptor and not rejected_once:
            rejected_once = True
            raise OSError("fixture direct close interruption")
        real_close(descriptor)

    monkeypatch.setattr(vector_store_module.os, "close", reject_direct_once)

    with pytest.raises(OSError, match="fixture direct close interruption"):
        store.close()

    assert store._authenticated._closed
    assert authenticated_descriptor in closed
    assert store._closed
    store.close()
    assert closed.count(direct_descriptor) == 1


def test_direct_store_closes_direct_descriptor_when_authenticated_close_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = _uint8_vectors()
    logical = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    payload = logical + b"\0" * (4096 - len(logical))
    path = tmp_path / "vectors-direct.bin"
    path.write_bytes(payload)
    store = DirectIoVectorStore(
        path,
        _identity(logical, rows=3, dimensions=2, zero_padding_bytes=4096 - len(logical)),
    )
    direct_descriptor = store._descriptor
    real_close = vector_store_module.os.close
    closed: list[int] = []

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(vector_store_module.os, "close", record_close)

    def fail_after_authenticated_retirement() -> None:
        store._authenticated._closed = True
        raise OSError("fixture authenticated close failure")

    monkeypatch.setattr(store._authenticated, "close", fail_after_authenticated_retirement)

    with pytest.raises(OSError, match="fixture authenticated close failure"):
        store.close()

    assert direct_descriptor in closed
    assert store._descriptor == -1
    assert store._closed


def test_direct_store_refuses_use_after_fork_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = _uint8_vectors()
    logical = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    payload = logical + b"\0" * (4096 - len(logical))
    path = tmp_path / "vectors-direct.bin"
    path.write_bytes(payload)
    store = DirectIoVectorStore(
        path,
        _identity(logical, rows=3, dimensions=2, zero_padding_bytes=4096 - len(logical)),
    )

    monkeypatch.setattr(vector_store_module.os, "getpid", lambda: store._process_id + 1)

    with pytest.raises(ValueError, match="direct I/O vector store differs"):
        store.new_context()
    with pytest.raises(ValueError, match="direct vector read differs"):
        store._begin_direct_read()

    monkeypatch.undo()
    store.close()


def test_direct_store_rejects_vector_mutation_after_authentication(tmp_path: Path) -> None:
    vectors = _uint8_vectors()
    logical = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    payload = logical + b"\0" * (4096 - len(logical))
    path = tmp_path / "vectors-direct.bin"
    path.write_bytes(payload)
    store = DirectIoVectorStore(
        path,
        _identity(logical, rows=3, dimensions=2, zero_padding_bytes=4096 - len(logical)),
    )

    with path.open("r+b") as stream:
        stream.seek(8)
        stream.write(b"\xff")
        stream.flush()

    with pytest.raises(ValueError, match="direct vector read differs"):
        store._begin_direct_read()

    store.close()


def test_exact_reranker_repairs_approximate_order_with_integer_authority() -> None:
    store = MemoryVectorStore(_uint8_vectors(), _memory_identity())
    candidates = CandidateResult(
        ids=np.array([1, 2, 0], dtype="<u4"),
        approximate_distances=np.array([0.0, 1.0, 2.0], dtype="<f8"),
        probe_lists=np.array([0], dtype="<u4"),
        evidence=CandidateEvidence(
            backend="portable-float64",
            rows_scanned=3,
            codes_bytes_scanned=3,
            probe_count=1,
            shortlist_width=3,
        ),
    )

    result = ExactReranker(store).search(
        np.array([1, 1], dtype=np.uint8), candidates, return_width=2
    )

    assert result.ids.tolist() == [0, 2]
    assert result.squared_distances.tolist() == [2.0, 2.0]
    assert result.vector_reads.requested_rows == 3
    store.close()


def test_exact_reranker_float32_uses_stable_float64_distance_order() -> None:
    vectors = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype="<f4")
    payload = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    identity = _identity(payload, rows=3, dimensions=2, dtype="float32")
    store = MemoryVectorStore(vectors, identity)
    candidates = CandidateResult(
        ids=np.array([0, 1, 2], dtype="<u4"),
        approximate_distances=np.array([0.0, 1.0, 2.0], dtype="<f8"),
        probe_lists=np.array([0], dtype="<u4"),
        evidence=CandidateEvidence(
            backend="portable-float64",
            rows_scanned=3,
            codes_bytes_scanned=3,
            probe_count=1,
            shortlist_width=3,
        ),
    )

    result = ExactReranker(store).search(
        np.array([1.5, 1.5], dtype="<f4"), candidates, return_width=2
    )

    assert result.ids.tolist() == [1, 2]
    assert result.squared_distances.tolist() == [0.5, 0.5]
    store.close()


def test_exact_reranker_rejects_insufficient_candidates() -> None:
    store = MemoryVectorStore(_uint8_vectors(), _memory_identity())
    candidates = CandidateResult(
        ids=np.array([0], dtype="<u4"),
        approximate_distances=np.array([0.0], dtype="<f8"),
        probe_lists=np.array([0], dtype="<u4"),
        evidence=CandidateEvidence(
            backend="portable-float64",
            rows_scanned=1,
            codes_bytes_scanned=1,
            probe_count=1,
            shortlist_width=1,
        ),
    )

    with pytest.raises(vector_store_module.InsufficientCandidatesError):
        ExactReranker(store).search(
            np.array([0, 0], dtype=np.uint8), candidates, return_width=2
        )

    store.close()


@pytest.mark.parametrize("dtype", ("uint8", "float32"))
def test_exact_reranker_matches_independent_ordered_distance_bits(dtype: str) -> None:
    rng = np.random.default_rng(90210)
    dimensions = 257
    if dtype == "uint8":
        vectors = rng.integers(0, 256, size=(7, dimensions), dtype=np.uint8)
        query = rng.integers(0, 256, size=dimensions, dtype=np.uint8)
    else:
        vectors = (rng.normal(size=(7, dimensions)) * 1.0e10).astype("<f4")
        query = (rng.normal(size=dimensions) * 1.0e-10).astype("<f4")
    payload = np.asarray([7, dimensions], dtype="<u4").tobytes() + vectors.tobytes()
    identity = _identity(
        payload,
        rows=7,
        dimensions=dimensions,
        dtype=dtype,
    )
    store = MemoryVectorStore(vectors, identity)
    candidates = CandidateResult(
        ids=np.arange(7, dtype="<u4"),
        approximate_distances=np.arange(7, dtype="<f8"),
        probe_lists=np.array([0], dtype="<u4"),
        evidence=CandidateEvidence(
            backend="portable-float64",
            rows_scanned=7,
            codes_bytes_scanned=7,
            probe_count=1,
            shortlist_width=7,
        ),
    )
    expected: list[tuple[float, int]] = []
    for internal_id in range(7):
        if dtype == "uint8":
            integer_distance = 0
            for dimension in range(dimensions):
                difference = int(query[dimension]) - int(vectors[internal_id, dimension])
                integer_distance += difference * difference
            distance = float(integer_distance)
        else:
            distance = 0.0
            for dimension in range(dimensions):
                difference = float(query[dimension]) - float(vectors[internal_id, dimension])
                distance += difference * difference
        expected.append((distance, internal_id))
    expected.sort()

    result = ExactReranker(store).search(query, candidates, return_width=7)

    assert result.ids.tolist() == [internal_id for _, internal_id in expected]
    np.testing.assert_array_equal(
        result.squared_distances.view(np.uint64),
        np.asarray([distance for distance, _ in expected], dtype="<f8").view(np.uint64),
    )
    store.close()


def test_pread_store_authenticates_header_payload_padding_and_reads_rows(
    tmp_path: Path,
) -> None:
    vectors = _uint8_vectors()
    logical = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    payload = logical + b"\0" * 8
    path = tmp_path / "vectors.bin"
    path.write_bytes(payload)
    identity = _identity(logical, rows=3, dimensions=2, zero_padding_bytes=8)
    store = PreadVectorStore(path, identity)
    context = store.new_context()

    observed, evidence = context.read(np.array([1, 0], dtype="<u4"))

    np.testing.assert_array_equal(observed, np.array([[10, 10], [0, 0]], np.uint8))
    assert evidence.logical_bytes == 4
    assert evidence.physical_bytes == 4
    context.close()
    store.close()


def test_pread_context_does_not_duplicate_the_complete_vector_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = _uint8_vectors()
    payload = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    path = tmp_path / "vectors.bin"
    path.write_bytes(payload)
    store = PreadVectorStore(path, _identity(payload, rows=3, dimensions=2))
    context = store.new_context()

    def bounded_bytes(value: object) -> bytes:
        if len(value) > store.identity.row_stride:  # type: ignore[arg-type]
            pytest.fail("complete vector buffer was duplicated")
        return builtins.bytes(value)

    monkeypatch.setattr(vector_store_module, "bytes", bounded_bytes, raising=False)

    observed, _ = context.read(np.array([0, 2], dtype="<u4"))

    np.testing.assert_array_equal(observed, np.array([[0, 0], [2, 2]], np.uint8))
    assert observed.flags.owndata
    assert not observed.flags.writeable
    context.close()
    store.close()


def test_pread_store_rejects_vector_mutation_after_authentication(tmp_path: Path) -> None:
    vectors = _uint8_vectors()
    payload = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    path = tmp_path / "vectors.bin"
    path.write_bytes(payload)
    store = PreadVectorStore(path, _identity(payload, rows=3, dimensions=2))
    context = store.new_context()

    with path.open("r+b") as stream:
        stream.seek(8)
        stream.write(b"\xff")
        stream.flush()

    with pytest.raises(ValueError, match="vector read differs"):
        context.read(np.array([0], dtype="<u4"))

    context.close()
    store.close()


def test_pread_context_retries_eintr_and_short_successful_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = _uint8_vectors()
    payload = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    path = tmp_path / "vectors.bin"
    path.write_bytes(payload)
    store = PreadVectorStore(path, _identity(payload, rows=3, dimensions=2))
    real_preadv = vector_store_module.os.preadv
    calls = 0

    def interrupted_then_short(fd: int, buffers: tuple[memoryview, ...], offset: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError
        (destination,) = buffers
        return int(real_preadv(fd, (destination[:1],), offset))

    monkeypatch.setattr(vector_store_module.os, "preadv", interrupted_then_short)
    context = store.new_context()

    observed, _ = context.read(np.array([2, 0], dtype="<u4"))

    np.testing.assert_array_equal(observed, np.array([[2, 2], [0, 0]], np.uint8))
    assert calls == 5
    context.close()
    store.close()


def test_pread_close_waits_for_active_read_before_descriptor_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = _uint8_vectors()
    payload = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    path = tmp_path / "vectors.bin"
    path.write_bytes(payload)
    store = PreadVectorStore(path, _identity(payload, rows=3, dimensions=2))
    context = store.new_context()
    real_preadv = vector_store_module.os.preadv
    read_entered = threading.Event()
    release_read = threading.Event()
    close_finished = threading.Event()

    def blocked_preadv(fd: int, buffers: tuple[memoryview, ...], offset: int) -> int:
        if offset == 8 and not read_entered.is_set():
            read_entered.set()
            assert release_read.wait(timeout=2)
        return int(real_preadv(fd, buffers, offset))

    monkeypatch.setattr(vector_store_module.os, "preadv", blocked_preadv)
    read_thread = threading.Thread(
        target=lambda: context.read(np.array([0, 1], dtype="<u4"))
    )
    close_thread = threading.Thread(
        target=lambda: (store.close(), close_finished.set())
    )
    read_thread.start()
    assert read_entered.wait(timeout=2)
    close_thread.start()

    assert not close_finished.wait(timeout=0.05)
    release_read.set()
    read_thread.join(timeout=2)
    close_thread.join(timeout=2)
    assert not read_thread.is_alive()
    assert not close_thread.is_alive()
    assert close_finished.is_set()
    context.close()


def test_pread_context_rejects_eof_without_partial_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = _uint8_vectors()
    payload = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    path = tmp_path / "vectors.bin"
    path.write_bytes(payload)
    store = PreadVectorStore(path, _identity(payload, rows=3, dimensions=2))
    monkeypatch.setattr(vector_store_module.os, "preadv", lambda *_args: 0)
    context = store.new_context()

    with pytest.raises(ValueError, match="vector read differs"):
        context.read(np.array([0], dtype="<u4"))

    context.close()
    store.close()


def test_vector_store_close_is_idempotent_and_invalidates_context() -> None:
    store = MemoryVectorStore(_uint8_vectors(), _memory_identity())
    context = store.new_context()

    store.close()
    store.close()

    with pytest.raises(ValueError, match="vector read differs"):
        context.read(np.array([0], dtype="<u4"))


@pytest.mark.parametrize("mutation", ("header", "payload", "padding", "truncated"))
def test_pread_store_rejects_authenticated_file_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    vectors = _uint8_vectors()
    logical = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    payload = bytearray(logical + b"\0" * 8)
    identity = _identity(logical, rows=3, dimensions=2, zero_padding_bytes=8)
    if mutation == "header":
        payload[0] ^= 1
    elif mutation == "payload":
        payload[8] ^= 1
    elif mutation == "padding":
        payload[-1] = 1
    else:
        payload.pop()
    path = tmp_path / "drift.bin"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="pread vector store differs"):
        PreadVectorStore(path, identity)


@pytest.mark.parametrize("ids", ([-1], [3]))
def test_vector_read_rejects_invalid_ids(tmp_path: Path, ids: list[int]) -> None:
    store = MemoryVectorStore(_uint8_vectors(), _memory_identity())
    context = store.new_context()
    with pytest.raises(ValueError, match="vector read differs"):
        context.read(np.asarray(ids, dtype="<i8"))
    context.close()
    store.close()


def test_portable_vector_read_does_not_duplicate_wide_row_buffers(tmp_path: Path) -> None:
    rows = 4
    dimensions = 4 << 20
    header = np.asarray([rows, dimensions], dtype="<u4").tobytes()
    payload = header + bytes(rows * dimensions)
    path = tmp_path / "wide.bin"
    path.write_bytes(payload)
    identity = _identity(payload, rows=rows, dimensions=dimensions)
    store = PreadVectorStore(path, identity)
    context = store.new_context()
    ids = np.arange(rows, dtype="<u4")
    context.read(ids[:1])

    tracemalloc.start()
    vectors, _ = context.read(ids)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    destination_bytes = rows * dimensions
    assert vectors.nbytes == destination_bytes
    assert peak <= destination_bytes + dimensions + (64 << 10)

    context.close()
    store.close()


def test_portable_vector_read_uses_pread_when_preadv_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = _uint8_vectors()
    payload = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    path = tmp_path / "vectors.bin"
    path.write_bytes(payload)
    store = PreadVectorStore(path, _identity(payload, rows=3, dimensions=2))
    monkeypatch.setattr(vector_store_module, "_HAS_PREADV", False)
    real_pread = vector_store_module.os.pread
    calls = 0

    def interrupted_then_short(fd: int, size: int, offset: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError
        return bytes(real_pread(fd, min(size, 1), offset))

    monkeypatch.setattr(vector_store_module.os, "pread", interrupted_then_short)
    context = store.new_context()

    observed, _ = context.read(np.array([2, 0], dtype="<u4"))

    np.testing.assert_array_equal(observed, np.array([[2, 2], [0, 0]], np.uint8))
    assert calls == 5
    context.close()
    store.close()


def test_pread_read_releases_its_lease_when_the_destination_allocation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = _uint8_vectors()
    payload = np.array([3, 2], dtype="<u4").tobytes() + vectors.tobytes()
    path = tmp_path / "vectors.bin"
    path.write_bytes(payload)
    store = PreadVectorStore(path, _identity(payload, rows=3, dimensions=2))
    context = store.new_context()

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise MemoryError("destination allocation failure")

    monkeypatch.setattr(vector_store_module.np, "empty", refuse)
    with pytest.raises(MemoryError, match="destination allocation failure"):
        context.read(np.array([0, 1], dtype="<u4"))
    monkeypatch.undo()

    assert store._active_reads == 0
    closed = threading.Event()
    threading.Thread(target=lambda: (store.close(), closed.set()), daemon=True).start()

    assert closed.wait(timeout=5)

    context.close()
