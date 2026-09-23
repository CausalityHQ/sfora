from __future__ import annotations

import ctypes
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import pack_int8_unit_embeddings


class _Function:
    def __init__(self, callback: object) -> None:
        self.callback = callback
        self.argtypes: list[object] = []
        self.restype: object = None
        self.raw_arguments: tuple[object, ...] = ()
        self.captured_inputs: tuple[np.ndarray, np.ndarray] | None = None

    def __call__(self, *args: object) -> int:
        self.raw_arguments = args
        if len(args) == 8 and self.argtypes[1] is ctypes.c_void_p:
            rows, dimensions, k = (int(args[index]) for index in (3, 4, 5))
            pointers = (args[index] for index in (1, 2, 6, 7))
            codes_ptr, norms_ptr, ordinals_ptr, scores_ptr = pointers
            assert all(
                isinstance(address, int) and address > 0
                for address in (codes_ptr, norms_ptr, ordinals_ptr, scores_ptr)
            )
            codes = np.ctypeslib.as_array(
                (ctypes.c_int8 * (rows * dimensions)).from_address(codes_ptr)
            )
            norms = np.ctypeslib.as_array((ctypes.c_uint16 * rows).from_address(norms_ptr))
            ordinals = np.ctypeslib.as_array(
                (ctypes.c_uint32 * (rows * k)).from_address(ordinals_ptr)
            )
            scores = np.ctypeslib.as_array((ctypes.c_float * (rows * k)).from_address(scores_ptr))
            self.captured_inputs = codes.copy(), norms.copy()
            args = (args[0], codes, norms, rows, dimensions, k, ordinals, scores)
        return int(self.callback(*args))  # type: ignore[operator]


class _Library:
    def __init__(self) -> None:
        self.destroyed = False

        def create(
            codes: np.ndarray,
            norms: np.ndarray,
            rows: int,
            dimensions: int,
            output: object,
        ) -> int:
            assert codes.shape == (int(rows) * int(dimensions),)
            assert norms.shape == (int(rows),)
            output._obj.value = 17  # type: ignore[attr-defined]
            return 0

        def search(
            handle: object,
            codes: np.ndarray,
            norms: np.ndarray,
            rows: int,
            dimensions: int,
            k: int,
            ordinals: np.ndarray,
            scores: np.ndarray,
        ) -> int:
            assert handle.value == 17  # type: ignore[attr-defined]
            assert codes.shape == (int(rows) * int(dimensions),)
            assert norms.shape == (int(rows),)
            ordinals[:] = np.tile(np.arange(int(k), dtype="<u4"), int(rows))
            scores[:] = np.arange(int(rows) * int(k), dtype="<f4")
            return 0

        def destroy(handle: object) -> int:
            assert handle.value == 17  # type: ignore[attr-defined]
            self.destroyed = True
            return 0

        self.sfora_cutile_int8_create = _Function(create)
        self.sfora_cutile_int8_search = _Function(search)
        self.sfora_cutile_int8_destroy = _Function(destroy)


def _packed(rows: int) -> tuple[np.ndarray, np.ndarray]:
    codes = np.arange(rows * 128, dtype=np.int16).reshape(rows, 128)
    codes = np.ascontiguousarray((codes % 255 - 127).astype(np.int8))
    norms = np.ascontiguousarray(
        (1.0 / np.linalg.norm(codes.astype(np.float32), axis=1)).astype("<f2")
    )
    return codes, norms


def test_explicit_native_handle_validates_searches_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library_path = tmp_path / "libsfora_cutile_int8_score.so"
    library_path.write_bytes(b"fixture")
    library = _Library()
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    gallery_codes, gallery_norms = _packed(129)

    gallery = CutilePackedInt8Gallery.open(library_path, gallery_codes, gallery_norms)
    query_codes, query_norms = _packed(32)
    ordinals, scores = gallery.search(query_codes, query_norms, k=10)

    assert ordinals.shape == (32, 10)
    assert scores.shape == (32, 10)
    assert ordinals.dtype == np.dtype("<i8")
    assert scores.dtype == np.dtype("<f4")
    assert np.array_equal(ordinals[0], np.arange(10, dtype="<u4"))
    gallery.close()
    gallery.close()
    assert library.destroyed
    with pytest.raises(RuntimeError, match="closed"):
        gallery.search(query_codes, query_norms, k=10)


def test_search_passes_validated_live_array_addresses_to_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library_path = tmp_path / "libsfora_cutile_int8_score.so"
    library_path.write_bytes(b"fixture")
    library = _Library()
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    gallery_codes, gallery_norms = _packed(129)
    query_codes, query_norms = _packed(32)

    with CutilePackedInt8Gallery.open(library_path, gallery_codes, gallery_norms) as gallery:
        ordinals, scores = gallery.search(query_codes, query_norms)

    search = library.sfora_cutile_int8_search
    assert all(isinstance(search.raw_arguments[index], int) for index in (1, 2, 6, 7))
    assert search.captured_inputs is not None
    assert np.array_equal(search.captured_inputs[0], query_codes.reshape(-1))
    assert np.array_equal(search.captured_inputs[1], query_norms.view("<u2"))
    assert np.array_equal(ordinals[0], np.arange(10, dtype=np.int64))
    assert np.array_equal(scores[0], np.arange(10, dtype=np.float32))


def test_native_gallery_accepts_the_public_packed_embedding_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library_path = tmp_path / "libsfora_cutile_int8_score.so"
    library_path.write_bytes(b"fixture")
    library = _Library()
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    gallery_values = torch.nn.functional.normalize(
        torch.arange(129 * 128, dtype=torch.float32).reshape(129, 128) + 1.0,
        dim=1,
    )
    query_values = torch.nn.functional.normalize(
        torch.arange(128, dtype=torch.float32).reshape(1, 128) + 1.0,
        dim=1,
    )
    gallery_packed = pack_int8_unit_embeddings(gallery_values)
    query_packed = pack_int8_unit_embeddings(query_values)

    with CutilePackedInt8Gallery.open_packed(library_path, gallery_packed) as gallery:
        ordinals, scores = gallery.search_packed(query_packed)

    assert ordinals.shape == (1, 10)
    assert scores.shape == (1, 10)
    assert library.destroyed


def test_native_search_batches_arbitrary_query_counts_without_partial_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library_path = tmp_path / "libsfora_cutile_int8_score.so"
    library_path.write_bytes(b"fixture")
    library = _Library()
    native_rows: list[int] = []

    def recording_search(
        _handle: object,
        _codes: np.ndarray,
        _norms: np.ndarray,
        rows: int,
        _dimensions: int,
        k: int,
        ordinals: np.ndarray,
        scores: np.ndarray,
    ) -> int:
        native_rows.append(int(rows))
        ordinals[:] = np.tile(np.arange(int(k), dtype="<u4"), int(rows))
        scores[:] = np.arange(int(rows) * int(k), dtype="<f4")
        return 0

    library.sfora_cutile_int8_search = _Function(recording_search)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    gallery_codes, gallery_norms = _packed(129)
    query_codes, query_norms = _packed(70)

    with CutilePackedInt8Gallery.open(library_path, gallery_codes, gallery_norms) as gallery:
        ordinals, scores = gallery.search(query_codes, query_norms)

    assert native_rows == [32, 32, 32]
    assert ordinals.shape == (70, 10)
    assert scores.shape == (70, 10)
    assert np.array_equal(ordinals[:, 0], np.zeros(70, dtype=np.int64))


def test_native_handle_rejects_implicit_or_invalid_arrays(tmp_path: Path) -> None:
    codes, norms = _packed(10)
    with pytest.raises(ValueError, match="absolute"):
        CutilePackedInt8Gallery.open(Path("backend.so"), codes, norms)
    library_path = tmp_path / "backend.so"
    library_path.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="gallery"):
        CutilePackedInt8Gallery.open(library_path, codes.astype(np.float32), norms)


def test_search_rejects_invalid_arrays_before_pointer_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library_path = tmp_path / "libsfora_cutile_int8_score.so"
    library_path.write_bytes(b"fixture")
    library = _Library()
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    gallery_codes, gallery_norms = _packed(129)
    codes, norms = _packed(32)
    bad_norms = norms.copy()
    bad_norms[0] = np.nan
    cases = [
        (codes[::2], norms[::2]),
        (np.asfortranarray(codes), norms),
        (codes.astype(np.float32), norms),
        (codes, norms.astype(">f2")),
        (codes, bad_norms),
    ]
    with CutilePackedInt8Gallery.open(library_path, gallery_codes, gallery_norms) as gallery:
        for bad_codes, bad_inverse_norms in cases:
            with pytest.raises(ValueError, match="query"):
                gallery.search(bad_codes, bad_inverse_norms)
            assert library.sfora_cutile_int8_search.raw_arguments == ()


def test_close_waits_for_active_native_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library_path = tmp_path / "libsfora_cutile_int8_score.so"
    library_path.write_bytes(b"fixture")
    library = _Library()
    search_entered = threading.Event()
    release_search = threading.Event()
    destroy_entered = threading.Event()

    def blocking_search(
        handle: object,
        _codes: np.ndarray,
        _norms: np.ndarray,
        rows: int,
        _dimensions: int,
        k: int,
        ordinals: np.ndarray,
        scores: np.ndarray,
    ) -> int:
        assert handle.value == 17  # type: ignore[attr-defined]
        search_entered.set()
        assert release_search.wait(timeout=2.0)
        ordinals[:] = np.tile(np.arange(int(k), dtype="<u4"), int(rows))
        scores[:] = np.arange(int(rows) * int(k), dtype="<f4")
        return 0

    def recording_destroy(handle: object) -> int:
        assert handle.value == 17  # type: ignore[attr-defined]
        destroy_entered.set()
        return 0

    library.sfora_cutile_int8_search = _Function(blocking_search)
    library.sfora_cutile_int8_destroy = _Function(recording_destroy)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    gallery_codes, gallery_norms = _packed(129)
    query_codes, query_norms = _packed(1)
    gallery = CutilePackedInt8Gallery.open(library_path, gallery_codes, gallery_norms)
    failures: list[BaseException] = []

    def run_search() -> None:
        try:
            gallery.search(query_codes, query_norms)
        except BaseException as error:
            failures.append(error)

    search_thread = threading.Thread(target=run_search)
    close_thread = threading.Thread(target=gallery.close)
    search_thread.start()
    assert search_entered.wait(timeout=1.0)
    close_thread.start()
    try:
        assert not destroy_entered.wait(timeout=0.1)
    finally:
        release_search.set()
        search_thread.join(timeout=2.0)
        close_thread.join(timeout=2.0)
    assert not search_thread.is_alive()
    assert not close_thread.is_alive()
    assert destroy_entered.is_set()
    assert failures == []


def test_native_searches_are_serialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library_path = tmp_path / "libsfora_cutile_int8_score.so"
    library_path.write_bytes(b"fixture")
    library = _Library()
    counter_lock = threading.Lock()
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    calls = 0

    def blocking_search(
        _handle: object,
        _codes: np.ndarray,
        _norms: np.ndarray,
        rows: int,
        _dimensions: int,
        k: int,
        ordinals: np.ndarray,
        scores: np.ndarray,
    ) -> int:
        nonlocal calls
        with counter_lock:
            calls += 1
            call = calls
        if call == 1:
            first_entered.set()
            assert release_first.wait(timeout=2.0)
        else:
            second_entered.set()
        ordinals[:] = np.tile(np.arange(int(k), dtype="<u4"), int(rows))
        scores[:] = np.arange(int(rows) * int(k), dtype="<f4")
        return 0

    library.sfora_cutile_int8_search = _Function(blocking_search)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    gallery_codes, gallery_norms = _packed(129)
    query_codes, query_norms = _packed(1)
    gallery = CutilePackedInt8Gallery.open(library_path, gallery_codes, gallery_norms)
    failures: list[BaseException] = []

    def run_search() -> None:
        try:
            gallery.search(query_codes, query_norms)
        except BaseException as error:
            failures.append(error)

    first = threading.Thread(target=run_search)
    second = threading.Thread(target=run_search)
    first.start()
    assert first_entered.wait(timeout=1.0)
    second.start()
    try:
        assert not second_entered.wait(timeout=0.1)
    finally:
        release_first.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)
        gallery.close()
    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()
    assert failures == []
