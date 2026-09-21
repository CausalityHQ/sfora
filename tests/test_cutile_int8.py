from __future__ import annotations

import ctypes
import threading
from pathlib import Path

import numpy as np
import pytest

from sfora.cutile_int8 import CutilePackedInt8Gallery


class _Function:
    def __init__(self, callback: object) -> None:
        self.callback = callback
        self.argtypes: list[object] = []
        self.restype: object = None

    def __call__(self, *args: object) -> int:
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


def test_native_handle_rejects_implicit_or_invalid_arrays(tmp_path: Path) -> None:
    codes, norms = _packed(10)
    with pytest.raises(ValueError, match="absolute"):
        CutilePackedInt8Gallery.open(Path("backend.so"), codes, norms)
    library_path = tmp_path / "backend.so"
    library_path.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="gallery"):
        CutilePackedInt8Gallery.open(library_path, codes.astype(np.float32), norms)


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
