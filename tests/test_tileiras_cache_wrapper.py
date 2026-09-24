"""The cuTile compiler cache must replay only matching native inputs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WRAPPER = Path(__file__).resolve().parents[1] / "scripts/tileiras_cache_wrapper.py"


def test_reuses_identical_bytecode_and_invalidates_changed_inputs(tmp_path: Path) -> None:
    compiler = tmp_path / "compiler.py"
    compiler.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "output = pathlib.Path(args[args.index('-o') + 1])\n"
        "source = pathlib.Path(args[-1])\n"
        "counter = pathlib.Path(__file__).with_suffix('.count')\n"
        "n = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(n))\n"
        "output.write_bytes(source.read_bytes() + bytes([n]))\n"
    )
    compiler.chmod(0o755)
    cache = tmp_path / "cache"
    environment = os.environ.copy()
    environment.update(
        SFORA_TILEIRAS_REAL=str(compiler),
        SFORA_TILEIRAS_CACHE=str(cache),
        SFORA_TILEIRAS_TOOLCHAIN_ID="toolchain-a",
    )

    def compile_bytes(
        source_name: str, payload: bytes, output_name: str, *, opt_level: str = "3"
    ) -> bytes:
        source = tmp_path / source_name
        output = tmp_path / output_name
        source.write_bytes(payload)
        subprocess.run(
            [
                sys.executable,
                str(WRAPPER),
                "--gpu-name",
                "sm_121",
                "--opt-level",
                opt_level,
                "-o",
                str(output),
                str(source),
            ],
            env=environment,
            check=True,
            capture_output=True,
        )
        return output.read_bytes()

    first = compile_bytes("one.bc", b"same ir", "one.cubin")
    assert compile_bytes("two.bc", b"same ir", "two.cubin") == first
    assert compiler.with_suffix(".count").read_text() == "1"

    assert compile_bytes("three.bc", b"changed ir", "three.cubin") != first
    assert compiler.with_suffix(".count").read_text() == "2"

    environment["SFORA_TILEIRAS_TOOLCHAIN_ID"] = "toolchain-b"
    assert compile_bytes("four.bc", b"same ir", "four.cubin") != first
    assert compiler.with_suffix(".count").read_text() == "3"

    assert compile_bytes("five.bc", b"same ir", "five.cubin", opt_level="2") != first
    assert compiler.with_suffix(".count").read_text() == "4"

    compiler.write_text(compiler.read_text() + "\n# changed compiler binary\n")
    assert compile_bytes("six.bc", b"same ir", "six.cubin") != first
    assert compiler.with_suffix(".count").read_text() == "5"


def test_does_not_cache_stale_output_when_compiler_writes_nothing(tmp_path: Path) -> None:
    compiler = tmp_path / "compiler.py"
    compiler.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n")
    compiler.chmod(0o755)
    source = tmp_path / "input.bc"
    source.write_bytes(b"new ir")
    output = tmp_path / "output.cubin"
    output.write_bytes(b"old cubin")
    cache = tmp_path / "cache"
    environment = os.environ.copy()
    environment.update(
        SFORA_TILEIRAS_REAL=str(compiler),
        SFORA_TILEIRAS_CACHE=str(cache),
        SFORA_TILEIRAS_TOOLCHAIN_ID="toolchain-a",
    )
    result = subprocess.run(
        [sys.executable, str(WRAPPER), "-o", str(output), str(source)],
        env=environment,
        capture_output=True,
    )
    assert result.returncode != 0
    assert not output.exists()
    assert not list(cache.glob("*.cubin"))
