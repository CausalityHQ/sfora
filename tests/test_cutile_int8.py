from __future__ import annotations

import ctypes
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
    assert ordinals.dtype == np.dtype("<u4")
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
