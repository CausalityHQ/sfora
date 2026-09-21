"""Optional persistent cuTile packed-int8 similarity scorer."""

from __future__ import annotations

import contextlib
import ctypes
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.ctypeslib import ndpointer
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sfora.joint_relational_compaction import PackedInt8Embeddings

_DIMENSIONS = 128
_TOP_K = 10
_NATIVE_BATCH = 32
_ERROR = "cuTile packed-int8 scorer failed"


def _packed_inputs(
    codes: NDArray[np.int8], inverse_norms: NDArray[np.float16], *, role: str
) -> tuple[NDArray[np.int8], NDArray[np.uint16]]:
    if (
        not isinstance(codes, np.ndarray)
        or codes.dtype != np.dtype("i1")
        or codes.ndim != 2
        or codes.shape[0] < 1
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
        self._lifecycle_lock = threading.Lock()

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

    @classmethod
    def open_packed(
        cls,
        library_path: Path,
        embeddings: PackedInt8Embeddings,
    ) -> CutilePackedInt8Gallery:
        """Open a gallery from Sfora's complete packed embedding value."""

        from sfora.joint_relational_compaction import PackedInt8Embeddings

        if type(embeddings) is not PackedInt8Embeddings:
            raise ValueError("cuTile packed gallery authority differs")
        return cls.open(
            library_path,
            embeddings.codes.numpy(),
            embeddings.inverse_norms.numpy(),
        )

    def search(
        self,
        codes: NDArray[np.int8],
        inverse_norms: NDArray[np.float16],
        *,
        k: int = _TOP_K,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Return exact cosine top-k with deterministic ordinal ties."""

        _packed_inputs(codes, inverse_norms, role="query")
        if k != _TOP_K:
            raise ValueError("cuTile query authority differs")
        ordinal_chunks: list[NDArray[np.int64]] = []
        score_chunks: list[NDArray[np.float32]] = []
        with self._lifecycle_lock:
            if self._handle.value is None:
                raise RuntimeError("cuTile packed-int8 gallery is closed")
            for start in range(0, codes.shape[0], _NATIVE_BATCH):
                stop = min(start + _NATIVE_BATCH, codes.shape[0])
                actual_rows = stop - start
                chunk_codes = codes[start:stop]
                chunk_norms = inverse_norms[start:stop]
                if actual_rows not in (1, _NATIVE_BATCH):
                    padding = _NATIVE_BATCH - actual_rows
                    chunk_codes = np.ascontiguousarray(
                        np.concatenate(
                            (chunk_codes, np.repeat(chunk_codes[-1:], padding, axis=0)),
                            axis=0,
                        )
                    )
                    chunk_norms = np.ascontiguousarray(
                        np.concatenate((chunk_norms, np.repeat(chunk_norms[-1:], padding)))
                    )
                flat_codes, norm_bits = _packed_inputs(chunk_codes, chunk_norms, role="query")
                native_rows = chunk_codes.shape[0]
                ordinals = np.empty(native_rows * k, dtype="<u4")
                scores = np.empty(native_rows * k, dtype="<f4")
                status = self._library.sfora_cutile_int8_search(
                    self._handle,
                    flat_codes,
                    norm_bits,
                    native_rows,
                    chunk_codes.shape[1],
                    k,
                    ordinals,
                    scores,
                )
                if status != 0:
                    raise RuntimeError(f"{_ERROR}: search status {status}")
                ordinal_chunks.append(
                    ordinals.reshape(native_rows, k)[:actual_rows].astype(np.int64)
                )
                score_chunks.append(scores.reshape(native_rows, k)[:actual_rows].copy())
        return np.concatenate(ordinal_chunks), np.concatenate(score_chunks)

    def search_packed(
        self,
        embeddings: PackedInt8Embeddings,
        *,
        k: int = _TOP_K,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Search with Sfora's complete packed embedding value."""

        from sfora.joint_relational_compaction import PackedInt8Embeddings

        if type(embeddings) is not PackedInt8Embeddings:
            raise ValueError("cuTile packed query authority differs")
        return self.search(
            embeddings.codes.numpy(),
            embeddings.inverse_norms.numpy(),
            k=k,
        )

    def close(self) -> None:
        """Release the native gallery exactly once."""

        with self._lifecycle_lock:
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
