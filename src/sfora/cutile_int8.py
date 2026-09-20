"""Optional persistent cuTile packed-int8 similarity scorer."""

from __future__ import annotations

import contextlib
import ctypes
import os
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer
from numpy.typing import NDArray

_DIMENSIONS = 128
_TOP_K = 10
_ERROR = "cuTile packed-int8 scorer failed"


def _packed_inputs(
    codes: NDArray[np.int8], inverse_norms: NDArray[np.float16], *, role: str
) -> tuple[NDArray[np.int8], NDArray[np.uint16]]:
    if (
        not isinstance(codes, np.ndarray)
        or codes.dtype != np.dtype("i1")
        or codes.ndim != 2
        or codes.shape[1] != _DIMENSIONS
        or not codes.flags.c_contiguous
        or not isinstance(inverse_norms, np.ndarray)
        or inverse_norms.dtype != np.dtype("<f2")
        or inverse_norms.shape != (codes.shape[0],)
        or not inverse_norms.flags.c_contiguous
        or not np.isfinite(inverse_norms).all()
        or not (inverse_norms > 0).all()
    ):
        raise ValueError(f"cuTile {role} authority differs")
    return codes.reshape(-1), inverse_norms.view("<u2")


class CutilePackedInt8Gallery:
    """Owned persistent GPU gallery loaded from an explicit native library."""

    def __init__(self, library: ctypes.CDLL, handle: ctypes.c_void_p) -> None:
        self._library = library
        self._handle = handle

    @classmethod
    def open(
        cls,
        library_path: Path,
        codes: NDArray[np.int8],
        inverse_norms: NDArray[np.float16],
    ) -> CutilePackedInt8Gallery:
        """Load an explicit backend and upload one validated gallery."""

        if not isinstance(library_path, Path) or not library_path.is_absolute():
            raise ValueError("cuTile library path must be absolute")
        if not library_path.is_file():
            raise ValueError("cuTile library path differs")
        flat_codes, norm_bits = _packed_inputs(codes, inverse_norms, role="gallery")
        if codes.shape[0] < _TOP_K:
            raise ValueError("cuTile gallery authority differs")
        try:
            library = ctypes.CDLL(os.fspath(library_path))
            create = library.sfora_cutile_int8_create
            search = library.sfora_cutile_int8_search
            destroy = library.sfora_cutile_int8_destroy
        except (AttributeError, OSError) as error:
            raise RuntimeError(_ERROR) from error
        i8 = ndpointer(dtype=np.dtype("i1"), ndim=1, flags=("C_CONTIGUOUS",))
        u16 = ndpointer(dtype=np.dtype("<u2"), ndim=1, flags=("C_CONTIGUOUS",))
        u32 = ndpointer(dtype=np.dtype("<u4"), ndim=1, flags=("C_CONTIGUOUS",))
        f32 = ndpointer(dtype=np.dtype("<f4"), ndim=1, flags=("C_CONTIGUOUS",))
        create.argtypes = [
            i8,
            u16,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        create.restype = ctypes.c_int
        search.argtypes = [
            ctypes.c_void_p,
            i8,
            u16,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            u32,
            f32,
        ]
        search.restype = ctypes.c_int
        destroy.argtypes = [ctypes.c_void_p]
        destroy.restype = ctypes.c_int
        handle = ctypes.c_void_p()
        status = create(
            flat_codes,
            norm_bits,
            codes.shape[0],
            codes.shape[1],
            ctypes.byref(handle),
        )
        if status != 0 or handle.value is None:
            raise RuntimeError(f"{_ERROR}: create status {status}")
        return cls(library, handle)

    def search(
        self,
        codes: NDArray[np.int8],
        inverse_norms: NDArray[np.float16],
        *,
        k: int = _TOP_K,
    ) -> tuple[NDArray[np.uint32], NDArray[np.float32]]:
        """Return exact cosine top-k with deterministic ordinal ties."""

        if self._handle.value is None:
            raise RuntimeError("cuTile packed-int8 gallery is closed")
        flat_codes, norm_bits = _packed_inputs(codes, inverse_norms, role="query")
        if codes.shape[0] not in (1, 32) or k != _TOP_K:
            raise ValueError("cuTile query authority differs")
        ordinals = np.empty(codes.shape[0] * k, dtype="<u4")
        scores = np.empty(codes.shape[0] * k, dtype="<f4")
        status = self._library.sfora_cutile_int8_search(
            self._handle,
            flat_codes,
            norm_bits,
            codes.shape[0],
            codes.shape[1],
            k,
            ordinals,
            scores,
        )
        if status != 0:
            raise RuntimeError(f"{_ERROR}: search status {status}")
        return ordinals.reshape(codes.shape[0], k), scores.reshape(codes.shape[0], k)

    def close(self) -> None:
        """Release the native gallery exactly once."""

        if self._handle.value is None:
            return
        handle = self._handle
        self._handle = ctypes.c_void_p()
        status = self._library.sfora_cutile_int8_destroy(handle)
        if status != 0:
            raise RuntimeError(f"{_ERROR}: destroy status {status}")

    def __enter__(self) -> CutilePackedInt8Gallery:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
