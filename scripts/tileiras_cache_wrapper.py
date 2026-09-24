#!/usr/bin/env python3
"""Cache exact tileiras cubins across fresh cuTile JIT processes.

Set CUTILE_TILEIRAS_PATH to this executable, and provide absolute
SFORA_TILEIRAS_REAL and SFORA_TILEIRAS_CACHE paths plus a toolchain identity
covering the compiler's supporting CUDA binaries and libraries.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _hash_file(digest: object, path: Path) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)  # type: ignore[attr-defined]


def main(argv: list[str]) -> int:
    real = Path(os.environ["SFORA_TILEIRAS_REAL"])
    cache = Path(os.environ["SFORA_TILEIRAS_CACHE"])
    toolchain_id = os.environ["SFORA_TILEIRAS_TOOLCHAIN_ID"]
    if (
        not real.is_absolute()
        or not real.is_file()
        or not cache.is_absolute()
        or not toolchain_id
        or argv.count("-o") != 1
        or len(argv) < 3
        or argv.index("-o") >= len(argv) - 2
    ):
        raise ValueError("tileiras cache authority differs")
    output_index = argv.index("-o") + 1
    output = Path(argv[output_index])
    source = Path(argv[-1])
    if not output.is_absolute() or not source.is_absolute() or not source.is_file():
        raise ValueError("tileiras input or output authority differs")

    digest = hashlib.sha256()
    digest.update(toolchain_id.encode())
    digest.update(b"\0")
    _hash_file(digest, real)
    for index, argument in enumerate(argv[:-1]):
        if index != output_index:
            digest.update(argument.encode())
            digest.update(b"\0")
    _hash_file(digest, source)
    key = digest.hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    cubin = cache / f"{key}.cubin"
    with (cache / f"{key}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cubin.is_file() and cubin.stat().st_size > 0:
            shutil.copyfile(cubin, output)
            print(f"tileiras cache hit {key}", file=sys.stderr)
            return 0
        output.unlink(missing_ok=True)
        result = subprocess.run([str(real), *argv], check=False)
        if result.returncode or not output.is_file() or not output.stat().st_size:
            return result.returncode or 1
        descriptor, temporary = tempfile.mkstemp(prefix=f"{key}.", dir=cache)
        try:
            with os.fdopen(descriptor, "wb") as destination, output.open("rb") as compiled:
                shutil.copyfileobj(compiled, destination)
            os.replace(temporary, cubin)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(f"tileiras cache miss {key}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
