from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import sfora
import sfora.factorized_residual_native as factorized_residual_native
from sfora.factorized_residual_ann import (
    CandidateEvidence,
    FactorizedResidualArtifact,
    FactorizedResidualComponents,
    FactorizedResidualPostings,
    FactorizedResidualSpec,
    PortableCandidateIndex,
    VectorStoreIdentity,
    write_factorized_residual_artifact,
)
from sfora.factorized_residual_native import (
    NativeCandidateIndex,
    NativeDirectQuiescenceError,
    NativeDirectUnsupportedError,
    NativeExactReranker,
    compile_factorized_residual_backend,
)
from sfora.vector_store import DirectIoVectorStore, ExactSearchResult


def _write_artifact(path: Path) -> FactorizedResidualArtifact:
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
        np.array([[0.0, 0.0, 0.0, 0.0], [8.0, 8.0, 8.0, 8.0]], dtype="<f4"),
        np.arange(32, dtype="<f4").reshape(2, 8, 2) / np.float32(16.0),
    )
    postings = FactorizedResidualPostings(
        spec,
        np.array([0, 2, 4], dtype="<u8"),
        np.array([2, 0, 3, 1], dtype="<u4"),
        np.array([[0], [9], [18], [27]], dtype=np.uint8),
    )
    store = VectorStoreIdentity(
        sha256="12" * 32,
        logical_bytes=24,
        physical_bytes=24,
        rows=4,
        dimensions=4,
        dtype="uint8",
        header_bytes=8,
        row_stride=4,
        zero_padding_bytes=0,
        generation="fixture-generation",
    )
    manifest_sha256 = write_factorized_residual_artifact(path, spec, components, postings, store)
    return FactorizedResidualArtifact.open(path, manifest_sha256=manifest_sha256)


def _pack_codes(indexes: np.ndarray, bits: int) -> np.ndarray:
    rows, subquantizers = indexes.shape
    packed = np.zeros((rows, (subquantizers * bits + 7) // 8), dtype=np.uint8)
    for row in range(rows):
        for subquantizer in range(subquantizers):
            value = int(indexes[row, subquantizer])
            start = subquantizer * bits
            for value_bit in range(bits):
                if value & (1 << value_bit):
                    bit = start + value_bit
                    packed[row, bit >> 3] |= 1 << (bit & 7)
    return packed


def _write_cross_byte_artifact(path: Path, bits: int) -> FactorizedResidualArtifact:
    random = np.random.default_rng(18_000 + bits)
    rows = 24
    spec = FactorizedResidualSpec(
        metric="squared_l2",
        vector_dtype="uint8",
        dimensions=18,
        list_count=3,
        subquantizers=9,
        bits_per_subquantizer=bits,
        probe_count=3,
        shortlist_width=8,
        return_width=4,
    )
    components = FactorizedResidualComponents(
        spec,
        random.normal(0.0, 0.25, (3, 18)).astype("<f4"),
        random.normal(
            0.0,
            0.125,
            (spec.subquantizers, spec.codebook_size, spec.subvector_dimensions),
        ).astype("<f4"),
    )
    indexes = random.integers(
        0, spec.codebook_size, size=(rows, spec.subquantizers), dtype=np.uint16
    )
    postings = FactorizedResidualPostings(
        spec,
        np.array([0, 8, 16, 24], dtype="<u8"),
        random.permutation(rows).astype("<u4"),
        _pack_codes(indexes, bits),
    )
    logical_bytes = 8 + rows * spec.dimensions
    manifest_sha256 = write_factorized_residual_artifact(
        path,
        spec,
        components,
        postings,
        VectorStoreIdentity(
            sha256="34" * 32,
            logical_bytes=logical_bytes,
            physical_bytes=logical_bytes,
            rows=rows,
            dimensions=spec.dimensions,
            dtype="uint8",
            header_bytes=8,
            row_stride=spec.dimensions,
            zero_padding_bytes=0,
            generation=f"cross-byte-{bits}",
        ),
    )
    return FactorizedResidualArtifact.open(path, manifest_sha256=manifest_sha256)


def test_native_backend_compiles_offline_and_reuses_authenticated_cache(tmp_path: Path) -> None:
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    binary = backend.path.read_bytes()

    assert backend.binary_sha256 == hashlib.sha256(binary).hexdigest()
    assert backend.path.is_file()
    assert sfora.compile_factorized_residual_backend is compile_factorized_residual_backend

    same = compile_factorized_residual_backend(tmp_path / "cache")
    assert same.path == backend.path
    assert same.binary_sha256 == backend.binary_sha256
    same.close()
    backend.close()


def test_native_backend_disables_implicit_floating_point_contraction() -> None:
    assert "-ffp-contract=off" in factorized_residual_native._COMPILE_FLAGS


def test_native_backend_accepts_relative_private_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    backend = compile_factorized_residual_backend(Path("cache"))

    assert backend.path.is_absolute()
    assert backend.path.parent == tmp_path / "cache"
    backend.close()


def test_native_backend_rejects_group_writable_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o770)
    cache.chmod(0o770)

    with pytest.raises(RuntimeError, match="native factorized residual compiler failed"):
        compile_factorized_residual_backend(cache)


def test_native_backend_rejects_unsupported_32_bit_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(factorized_residual_native, "_POINTER_BYTES", 4)

    with pytest.raises(RuntimeError, match="native factorized residual compiler failed"):
        compile_factorized_residual_backend(tmp_path / "cache")


def test_native_backend_rebuilds_corrupt_cached_binary(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys; "
                "from sfora.factorized_residual_native import "
                "compile_factorized_residual_backend as compile; "
                "backend=compile(sys.argv[1]); "
                "print(json.dumps([str(backend.path),backend.binary_sha256]))"
            ),
            os.fspath(cache),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    path_text, expected_sha256 = json.loads(completed.stdout)
    path = Path(path_text)
    path.write_bytes(b"not an ELF shared library")

    rebuilt = compile_factorized_residual_backend(cache)

    assert rebuilt.binary_sha256 == expected_sha256
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
    rebuilt.close()


def test_native_backend_rejects_library_without_required_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "wrong.c"
    source.write_text("int unrelated_symbol(void) { return 0; }\n")
    monkeypatch.setattr(factorized_residual_native, "_SOURCE", source)

    with pytest.raises(RuntimeError, match="native factorized residual compiler failed"):
        compile_factorized_residual_backend(tmp_path / "cache")


def test_frozen_prototype_and_golden_case_are_authenticated() -> None:
    root = Path(__file__).parent / "data" / "factorized_residual_golden"
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)

    assert (
        manifest_bytes
        == (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    assert manifest["reproduction_command"] == (
        "uv run --locked pytest -q tests/test_factorized_residual_native.py"
    )
    for name, authority in manifest["files"].items():
        body = (root / name).read_bytes()
        assert len(body) == authority["bytes"]
        assert hashlib.sha256(body).hexdigest() == authority["sha256"]
    assert manifest["files"]["prototype_one_lut_scanner.c"]["sha256"] == (
        "eebd53202ff5f479fe2733746adc727191e3b3b065973a28f3db42b3c0aa81d3"
    )


def test_native_backend_fails_closed_for_missing_compiler(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="native factorized residual compiler failed"):
        compile_factorized_residual_backend(tmp_path / "cache", cc=tmp_path / "missing-compiler")


def test_native_candidate_returns_actual_count_for_empty_selected_list(tmp_path: Path) -> None:
    spec = FactorizedResidualSpec(
        metric="squared_l2",
        vector_dtype="uint8",
        dimensions=4,
        list_count=2,
        subquantizers=2,
        bits_per_subquantizer=1,
        probe_count=1,
        shortlist_width=2,
        return_width=1,
    )
    components = FactorizedResidualComponents(
        spec,
        np.array([[0, 0, 0, 0], [100, 100, 100, 100]], dtype="<f4"),
        np.zeros((2, 2, 2), dtype="<f4"),
    )
    postings = FactorizedResidualPostings(
        spec,
        np.array([0, 0, 2], dtype="<u8"),
        np.array([0, 1], dtype="<u4"),
        np.zeros((2, 1), dtype=np.uint8),
    )
    logical_bytes = 16
    manifest_sha256 = write_factorized_residual_artifact(
        tmp_path / "empty-list",
        spec,
        components,
        postings,
        VectorStoreIdentity(
            sha256="56" * 32,
            logical_bytes=logical_bytes,
            physical_bytes=logical_bytes,
            rows=2,
            dimensions=4,
            dtype="uint8",
            header_bytes=8,
            row_stride=4,
            zero_padding_bytes=0,
            generation="empty-list",
        ),
    )
    artifact = FactorizedResidualArtifact.open(
        tmp_path / "empty-list", manifest_sha256=manifest_sha256
    )
    backend = compile_factorized_residual_backend(tmp_path / "cache")

    result = NativeCandidateIndex(artifact, backend).search(np.zeros(4, dtype=np.uint8))

    assert result.ids.size == 0
    assert result.approximate_distances.size == 0
    assert result.probe_lists.tolist() == [0]
    assert result.evidence.rows_scanned == 0
    artifact.close()
    backend.close()


def test_artifact_close_waits_for_native_call_to_release_mappings(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "artifact")
    compiled = compile_factorized_residual_backend(tmp_path / "cache")
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    results: list[list[int]] = []
    real_function = compiled._library.sfora_factorized_candidate_search

    class BlockingFunction:
        def __call__(self, *arguments: object) -> int:
            started.set()
            assert release.wait(5)
            return int(real_function(*arguments))

    compiled._library.sfora_factorized_candidate_search = BlockingFunction()

    def search() -> None:
        result = NativeCandidateIndex(artifact, compiled).search(
            np.array([1, 2, 3, 4], dtype=np.uint8)
        )
        results.append(result.ids.tolist())

    def close() -> None:
        artifact.close()
        closed.set()

    search_thread = threading.Thread(target=search)
    search_thread.start()
    assert started.wait(5)
    close_thread = threading.Thread(target=close)
    close_thread.start()
    assert not closed.wait(0.05)
    release.set()
    search_thread.join(5)
    close_thread.join(5)

    assert results == [[0, 2, 3, 1]]
    assert closed.is_set()
    assert artifact._closed
    compiled.close()


def test_native_direct_rerank_matches_integer_exact_oracle(tmp_path: Path) -> None:
    vectors = np.array(
        [[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0], [2, 2, 2, 2]],
        dtype=np.uint8,
    )
    logical = np.asarray([4, 4], dtype="<u4").tobytes() + vectors.tobytes()
    physical = logical + b"\0" * (4096 - len(logical))
    path = tmp_path / "vectors.bin"
    path.write_bytes(physical)
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
        generation="direct-fixture",
    )
    candidates = sfora.CandidateResult(
        ids=np.array([0, 2, 3, 1], dtype="<u4"),
        approximate_distances=np.array([0, 1, 2, 3], dtype="<f8"),
        probe_lists=np.array([0], dtype="<u4"),
        evidence=CandidateEvidence(
            backend="portable-float64",
            rows_scanned=4,
            codes_bytes_scanned=4,
            probe_count=1,
            shortlist_width=4,
        ),
    )
    store = DirectIoVectorStore(path, identity)
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    memory_alignment = np.zeros(1, dtype="<u4")
    offset_alignment = np.zeros(1, dtype="<u4")
    assert (
        backend._library.sfora_direct_io_alignment(
            store._descriptor, memory_alignment, offset_alignment
        )
        == 0
    )

    try:
        result = NativeExactReranker(store, backend).search(
            np.array([1, 2, 3, 4], dtype=np.uint8), candidates, return_width=2
        )
    except RuntimeError as error:
        store.close()
        backend.close()
        if str(error) == "native direct rerank failed: -5":
            pytest.skip("io_uring unavailable in this execution environment")
        raise

    assert result.ids.tolist() == [0, 3]
    assert result.squared_distances.tolist() == [0.0, 6.0]
    assert result.vector_reads.requested_rows == 4
    assert result.vector_reads.logical_bytes == 16
    alignment = int(offset_alignment[0])
    expected_physical = sum(
        ((8 + internal_id * identity.row_stride) % alignment + identity.row_stride + alignment - 1)
        // alignment
        * alignment
        for internal_id in candidates.ids.tolist()
    )
    assert result.vector_reads.physical_bytes == expected_physical
    store.close()
    backend.close()


def test_native_exact_reranker_owns_query_while_native_call_releases_gil(
    tmp_path: Path,
) -> None:
    vectors = np.array([[1.0, 2.0]], dtype="<f4")
    logical = np.asarray([1, 2], dtype="<u4").tobytes() + vectors.tobytes()
    physical = logical + b"\0" * (4096 - len(logical))
    path = tmp_path / "vectors.bin"
    path.write_bytes(physical)
    store = DirectIoVectorStore(
        path,
        VectorStoreIdentity(
            sha256=hashlib.sha256(physical).hexdigest(),
            logical_bytes=len(logical),
            physical_bytes=len(physical),
            rows=1,
            dimensions=2,
            dtype="float32",
            header_bytes=8,
            row_stride=8,
            zero_padding_bytes=len(physical) - len(logical),
            generation="direct-query-ownership",
        ),
    )
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    candidates = sfora.CandidateResult(
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
    entered = threading.Event()
    release = threading.Event()
    observed: list[list[float]] = []
    backend._library.sfora_direct_io_alignment = lambda _fd, memory, offset: (
        memory.__setitem__(0, 512),
        offset.__setitem__(0, 512),
        0,
    )[-1]

    def blocked_rerank(*arguments: object) -> int:
        entered.set()
        assert release.wait(5)
        observed.append(arguments[0].tolist())  # type: ignore[union-attr]
        arguments[13][0] = 0  # type: ignore[index]
        arguments[14][0] = 0.0  # type: ignore[index]
        arguments[16][0] = 512  # type: ignore[index]
        arguments[17][0] = 1  # type: ignore[index]
        return 0

    backend._library.sfora_exact_rerank_direct = blocked_rerank
    query = np.array([1.0, 2.0], dtype="<f4")
    results: list[ExactSearchResult] = []
    thread = threading.Thread(
        target=lambda: results.append(
            NativeExactReranker(store, backend).search(query, candidates, return_width=1)
        )
    )
    thread.start()
    assert entered.wait(5)
    query[:] = 99.0
    release.set()
    thread.join(5)

    assert observed == [[1.0, 2.0]]
    assert len(results) == 1
    store.close()
    backend.close()


def test_native_direct_float32_rerank_matches_ordered_float64_across_threads(
    tmp_path: Path,
) -> None:
    random = np.random.default_rng(20260913)
    vectors = random.normal(size=(5, 513)).astype("<f4")
    vectors[4] = vectors[3]
    query = random.normal(size=513).astype("<f4")
    logical = np.asarray([5, 513], dtype="<u4").tobytes() + vectors.tobytes()
    physical = logical + b"\0" * ((-len(logical)) % 4096)
    path = tmp_path / "vectors-float.bin"
    path.write_bytes(physical)
    identity = VectorStoreIdentity(
        sha256=hashlib.sha256(physical).hexdigest(),
        logical_bytes=len(logical),
        physical_bytes=len(physical),
        rows=5,
        dimensions=513,
        dtype="float32",
        header_bytes=8,
        row_stride=513 * 4,
        zero_padding_bytes=len(physical) - len(logical),
        generation="direct-float-fixture",
    )
    expected: list[tuple[float, int]] = []
    for internal_id in range(5):
        distance = 0.0
        for dimension in range(513):
            difference = float(query[dimension]) - float(vectors[internal_id, dimension])
            distance += difference * difference
        expected.append((distance, internal_id))
    expected.sort()
    candidates = sfora.CandidateResult(
        ids=np.asarray([internal_id for _, internal_id in expected], dtype="<u4"),
        approximate_distances=np.arange(5, dtype="<f8"),
        probe_lists=np.array([0], dtype="<u4"),
        evidence=CandidateEvidence(
            backend="portable-float64",
            rows_scanned=5,
            codes_bytes_scanned=5,
            probe_count=1,
            shortlist_width=5,
        ),
    )
    store = DirectIoVectorStore(path, identity)
    backend = compile_factorized_residual_backend(tmp_path / "cache")

    for thread_count in (1, 2, 4):
        result = NativeExactReranker(
            store, backend, thread_count=thread_count
        ).search(query, candidates, return_width=5)
        assert result.ids.tolist() == [internal_id for _, internal_id in expected]
        np.testing.assert_array_equal(
            result.squared_distances.view(np.uint64),
            np.asarray([distance for distance, _ in expected], dtype="<f8").view(np.uint64),
        )

    store.close()
    backend.close()


def test_native_direct_float32_rejects_nonfinite_unselected_candidate(
    tmp_path: Path,
) -> None:
    vectors = np.array([[0.0, 0.0], [np.nan, 1.0]], dtype="<f4")
    logical = np.asarray([2, 2], dtype="<u4").tobytes() + vectors.tobytes()
    physical = logical + b"\0" * (4096 - len(logical))
    path = tmp_path / "vectors-nonfinite.bin"
    path.write_bytes(physical)
    store = DirectIoVectorStore(
        path,
        VectorStoreIdentity(
            sha256=hashlib.sha256(physical).hexdigest(),
            logical_bytes=len(logical),
            physical_bytes=len(physical),
            rows=2,
            dimensions=2,
            dtype="float32",
            header_bytes=8,
            row_stride=8,
            zero_padding_bytes=len(physical) - len(logical),
            generation="direct-nonfinite-fixture",
        ),
    )
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    candidates = sfora.CandidateResult(
        ids=np.array([0, 1], dtype="<u4"),
        approximate_distances=np.array([0.0, 1.0], dtype="<f8"),
        probe_lists=np.array([0], dtype="<u4"),
        evidence=CandidateEvidence(
            backend="portable-float64",
            rows_scanned=2,
            codes_bytes_scanned=2,
            probe_count=1,
            shortlist_width=2,
        ),
    )

    with pytest.raises(RuntimeError, match="native direct rerank failed: -6"):
        NativeExactReranker(store, backend).search(
            np.array([0.0, 0.0], dtype="<f4"), candidates, return_width=1
        )

    store.close()
    backend.close()


def test_native_exact_reranker_rejects_more_than_bounded_candidates(
    tmp_path: Path,
) -> None:
    logical = np.asarray([1, 1], dtype="<u4").tobytes() + b"\0"
    physical = logical + b"\0" * (4096 - len(logical))
    path = tmp_path / "vectors.bin"
    path.write_bytes(physical)
    store = DirectIoVectorStore(
        path,
        VectorStoreIdentity(
            sha256=hashlib.sha256(physical).hexdigest(),
            logical_bytes=len(logical),
            physical_bytes=len(physical),
            rows=1,
            dimensions=1,
            dtype="uint8",
            header_bytes=8,
            row_stride=1,
            zero_padding_bytes=len(physical) - len(logical),
            generation="direct-candidate-bound",
        ),
    )
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    candidates = sfora.CandidateResult(
        ids=np.arange(4097, dtype="<u4"),
        approximate_distances=np.arange(4097, dtype="<f8"),
        probe_lists=np.array([0], dtype="<u4"),
        evidence=CandidateEvidence(
            backend="portable-float64",
            rows_scanned=4097,
            codes_bytes_scanned=4097,
            probe_count=1,
            shortlist_width=4097,
        ),
    )

    with pytest.raises(ValueError, match="native exact rerank differs"):
        NativeExactReranker(store, backend).search(
            np.array([0], dtype=np.uint8), candidates, return_width=1
        )

    store.close()
    backend.close()


def test_native_exact_reranker_rejects_unpadded_direct_tail_at_construction(
    tmp_path: Path,
) -> None:
    vectors = np.zeros((3, 128), dtype="<f4")
    logical = np.asarray([3, 128], dtype="<u4").tobytes() + vectors.tobytes()
    path = tmp_path / "vectors-unpadded.bin"
    path.write_bytes(logical)
    store = DirectIoVectorStore(
        path,
        VectorStoreIdentity(
            sha256=hashlib.sha256(logical).hexdigest(),
            logical_bytes=len(logical),
            physical_bytes=len(logical),
            rows=3,
            dimensions=128,
            dtype="float32",
            header_bytes=8,
            row_stride=512,
            zero_padding_bytes=0,
            generation="direct-unpadded-tail",
        ),
    )
    backend = compile_factorized_residual_backend(tmp_path / "cache")

    with pytest.raises(ValueError, match="native exact reranker differs"):
        NativeExactReranker(store, backend)

    store.close()
    backend.close()


def test_native_direct_quarantine_poisons_all_loaded_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = np.array([[0]], dtype=np.uint8)
    logical = np.asarray([1, 1], dtype="<u4").tobytes() + vectors.tobytes()
    physical = logical + b"\0" * (4096 - len(logical))
    path = tmp_path / "vectors.bin"
    path.write_bytes(physical)
    identity = VectorStoreIdentity(
        sha256=hashlib.sha256(physical).hexdigest(),
        logical_bytes=len(logical),
        physical_bytes=len(physical),
        rows=1,
        dimensions=1,
        dtype="uint8",
        header_bytes=8,
        row_stride=1,
        zero_padding_bytes=len(physical) - len(logical),
        generation="direct-poison-fixture",
    )
    first_store = DirectIoVectorStore(path, identity)
    second_store = DirectIoVectorStore(path, identity)
    first_backend = compile_factorized_residual_backend(tmp_path / "first-cache")
    second_backend = compile_factorized_residual_backend(tmp_path / "second-cache")
    first = NativeExactReranker(first_store, first_backend)
    second = NativeExactReranker(second_store, second_backend)
    candidates = sfora.CandidateResult(
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
    monkeypatch.setattr(factorized_residual_native, "_DIRECT_POISONED", False)
    first_backend._library.sfora_exact_rerank_direct = lambda *_arguments: -4

    with pytest.raises(NativeDirectQuiescenceError):
        first.search(np.array([0], dtype=np.uint8), candidates, return_width=1)

    second_called = False

    def reject_second_call(*_arguments: object) -> int:
        nonlocal second_called
        second_called = True
        return 0

    second_backend._library.sfora_exact_rerank_direct = reject_second_call
    with pytest.raises(NativeDirectQuiescenceError):
        second.search(np.array([0], dtype=np.uint8), candidates, return_width=1)
    assert not second_called

    first_store.close()
    second_store.close()
    first_backend.close()
    second_backend.close()


def test_native_exact_reranker_refuses_unsupported_ring_at_construction(
    tmp_path: Path,
) -> None:
    logical = np.asarray([1, 1], dtype="<u4").tobytes() + b"\0"
    physical = logical + b"\0" * (4096 - len(logical))
    path = tmp_path / "vectors.bin"
    path.write_bytes(physical)
    store = DirectIoVectorStore(
        path,
        VectorStoreIdentity(
            sha256=hashlib.sha256(physical).hexdigest(),
            logical_bytes=len(logical),
            physical_bytes=len(physical),
            rows=1,
            dimensions=1,
            dtype="uint8",
            header_bytes=8,
            row_stride=1,
            zero_padding_bytes=len(physical) - len(logical),
            generation="direct-probe-fixture",
        ),
    )
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    backend._library.sfora_direct_io_probe = lambda: -5

    with pytest.raises(NativeDirectUnsupportedError):
        NativeExactReranker(store, backend)

    store.close()
    backend.close()


@pytest.mark.parametrize("thread_count", (True, 0, 257))
def test_native_exact_reranker_rejects_unbounded_thread_count(
    tmp_path: Path,
    thread_count: object,
) -> None:
    vectors = np.array([[1, 2, 3, 4]], dtype=np.uint8)
    logical = np.asarray([1, 4], dtype="<u4").tobytes() + vectors.tobytes()
    physical = logical + b"\0" * (4096 - len(logical))
    path = tmp_path / "vectors.bin"
    path.write_bytes(physical)
    store = DirectIoVectorStore(
        path,
        VectorStoreIdentity(
            sha256=hashlib.sha256(physical).hexdigest(),
            logical_bytes=len(logical),
            physical_bytes=len(physical),
            rows=1,
            dimensions=4,
            dtype="uint8",
            header_bytes=8,
            row_stride=4,
            zero_padding_bytes=len(physical) - len(logical),
            generation="direct-thread-fixture",
        ),
    )
    backend = compile_factorized_residual_backend(tmp_path / "cache")

    with pytest.raises(ValueError, match="native exact reranker differs"):
        NativeExactReranker(store, backend, thread_count=thread_count)  # type: ignore[arg-type]

    store.close()
    backend.close()


def test_native_direct_rerank_rejects_non_power_of_two_alignment_before_io(
    tmp_path: Path,
) -> None:
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    query = np.array([1, 2, 3, 4], dtype="<f4")
    candidate_ids = np.array([0], dtype="<u4")
    output_ids = np.empty(1, dtype="<u4")
    output_distances = np.empty(1, dtype="<f8")
    physical_bytes = np.zeros(1, dtype="<u8")
    operations = np.zeros(1, dtype="<u4")

    status = backend._library.sfora_exact_rerank_direct(
        query,
        query.size,
        candidate_ids,
        candidate_ids.size,
        1,
        4,
        1,
        1,
        0,
        4096,
        3,
        512,
        1,
        output_ids,
        output_distances,
        output_ids.size,
        physical_bytes,
        operations,
    )

    assert status == -1
    assert physical_bytes[0] == 0
    assert operations[0] == 0
    backend.close()


def test_native_direct_rerank_accepts_statx_memory_alignment_below_pointer_size(
    tmp_path: Path,
) -> None:
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    query = np.array([1, 2, 3, 4], dtype="<f4")
    candidate_ids = np.array([0], dtype="<u4")
    output_ids = np.empty(1, dtype="<u4")
    output_distances = np.empty(1, dtype="<f8")
    physical_bytes = np.zeros(1, dtype="<u8")
    operations = np.zeros(1, dtype="<u4")

    status = backend._library.sfora_exact_rerank_direct(
        query,
        query.size,
        candidate_ids,
        candidate_ids.size,
        1,
        4,
        1,
        1,
        0,
        4096,
        4,
        512,
        1,
        output_ids,
        output_distances,
        output_ids.size,
        physical_bytes,
        operations,
    )

    assert status != -1
    backend.close()


@pytest.mark.parametrize(
    "base_is_u8,query",
    (
        (0, np.array([0.0, np.nan], dtype="<f4")),
        (1, np.array([0.0, 1.5], dtype="<f4")),
        (1, np.array([0.0, 256.0], dtype="<f4")),
    ),
)
def test_native_direct_rerank_rejects_invalid_query_before_io(
    tmp_path: Path,
    base_is_u8: int,
    query: np.ndarray,
) -> None:
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    candidate_ids = np.array([0], dtype="<u4")
    output_ids = np.empty(1, dtype="<u4")
    output_distances = np.empty(1, dtype="<f8")
    physical_bytes = np.zeros(1, dtype="<u8")
    operations = np.zeros(1, dtype="<u4")

    status = backend._library.sfora_exact_rerank_direct(
        query,
        query.size,
        candidate_ids,
        candidate_ids.size,
        1,
        2,
        base_is_u8,
        1,
        0,
        4096,
        4096,
        512,
        1,
        output_ids,
        output_distances,
        output_ids.size,
        physical_bytes,
        operations,
    )

    assert status == -1
    assert operations[0] == 0
    backend.close()


def test_native_candidate_ids_match_scalar_f32_control_across_threads(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(tmp_path / "artifact")
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    query = np.array([1, 2, 3, 4], dtype=np.uint8)

    expected = PortableCandidateIndex(artifact).search(query)
    golden = json.loads(
        (
            Path(__file__).parent / "data" / "factorized_residual_golden" / "golden_case.json"
        ).read_bytes()
    )
    observed_ids: list[list[int]] = []
    for threads in (1, 2, 4):
        result = NativeCandidateIndex(artifact, backend, thread_count=threads).search(query)
        observed_ids.append(result.ids.tolist())
        assert result.probe_lists.tolist() == expected.probe_lists.tolist()
        assert result.evidence.backend == "native-c11-fma"
        assert result.evidence.rows_scanned == expected.evidence.rows_scanned
        assert result.evidence.codes_bytes_scanned == expected.evidence.codes_bytes_scanned
        assert result.ids.tolist() == golden["expected_candidate_ids_u32"]
        assert [
            np.float32(value).tobytes().hex() for value in result.approximate_distances.tolist()
        ] == golden["expected_candidate_score_f32_le_hex"]

    assert observed_ids == [observed_ids[0]] * 3
    artifact.close()
    backend.close()


def test_product_native_candidate_matches_frozen_prototype_kernel(tmp_path: Path) -> None:
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        pytest.skip("C compiler unavailable")
    root = Path(__file__).parent / "data" / "factorized_residual_golden"
    prototype_library = tmp_path / "prototype.so"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O3",
            "-fPIC",
            "-shared",
            "-fopenmp",
            os.fspath(root / "prototype_one_lut_scanner.c"),
            "-lm",
            "-o",
            os.fspath(prototype_library),
        ],
        check=True,
        capture_output=True,
    )
    artifact = _write_artifact(tmp_path / "artifact")
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    query = np.array([1, 2, 3, 4], dtype="<f4")
    product = NativeCandidateIndex(artifact, backend, thread_count=4).search(query.astype(np.uint8))
    arrays = artifact._arrays
    prototype = ctypes.CDLL(prototype_library).one_lut_search
    float_pointer = np.ctypeslib.ndpointer("<f4", ndim=1, flags="C_CONTIGUOUS")
    u8_pointer = np.ctypeslib.ndpointer("u1", ndim=1, flags="C_CONTIGUOUS")
    u64_pointer = np.ctypeslib.ndpointer("<u8", ndim=1, flags="C_CONTIGUOUS")
    i64_pointer = np.ctypeslib.ndpointer("<i8", ndim=1, flags="C_CONTIGUOUS")
    prototype.argtypes = [
        float_pointer,
        float_pointer,
        float_pointer,
        u64_pointer,
        ctypes.c_void_p,
        ctypes.c_int,
        u8_pointer,
        u8_pointer,
        float_pointer,
        float_pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        i64_pointer,
        float_pointer,
    ]
    prototype.restype = ctypes.c_int
    output_ids = np.empty(4, dtype="<i8")
    output_scores = np.empty(4, dtype="<f4")
    status = prototype(
        query,
        arrays["coarse.f32"].reshape(-1),
        arrays["pq.f32"].reshape(-1),
        arrays["offsets.u64"].reshape(-1),
        ctypes.c_void_p(arrays["ids.u32"].ctypes.data),
        1,
        arrays["codes.u8"].reshape(-1),
        arrays["norms.u8"].reshape(-1),
        arrays["norm-low.f32"].reshape(-1),
        arrays["norm-scale.f32"].reshape(-1),
        2,
        4,
        2,
        3,
        2,
        4,
        4,
        0,
        None,
        -1,
        1,
        output_ids,
        output_scores,
    )

    assert status == 0
    assert output_ids.tolist() == product.ids.tolist()
    assert output_scores.tobytes() == product.approximate_distances.astype("<f4").tobytes()
    artifact.close()
    backend.close()


@pytest.mark.parametrize("bits", range(1, 9))
def test_native_candidate_is_bit_stable_across_threads_and_packed_widths(
    tmp_path: Path, bits: int
) -> None:
    artifact = _write_cross_byte_artifact(tmp_path / f"artifact-{bits}", bits)
    backend = compile_factorized_residual_backend(tmp_path / "cache")
    query = np.arange(18, dtype=np.uint8)

    scalar = NativeCandidateIndex(artifact, backend, thread_count=1).search(query)
    for thread_count in (2, 4):
        parallel = NativeCandidateIndex(artifact, backend, thread_count=thread_count).search(query)
        assert parallel.ids.tobytes() == scalar.ids.tobytes()
        assert parallel.approximate_distances.astype("<f4").tobytes() == (
            scalar.approximate_distances.astype("<f4").tobytes()
        )
        assert parallel.probe_lists.tobytes() == scalar.probe_lists.tobytes()

    artifact.close()
    backend.close()
