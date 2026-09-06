"""Behavioral contract for the fail-closed local-file capability launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_landlock_launcher_allows_registered_file_and_denies_other_file(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "landlock_exec.c"
    launcher = tmp_path / "landlock-exec"
    subprocess.run(
        ["cc", "-std=c17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(launcher)],
        check=True,
    )
    allowed = tmp_path / "allowed"
    forbidden = tmp_path / "forbidden"
    allowed.write_text("allowed")
    forbidden.write_text("forbidden")
    script = 'test "$(cat "$1")" = allowed && if cat "$2"; then exit 1; else exit 0; fi'
    completed = subprocess.run(
        [
            str(launcher),
            "--ro",
            "/usr",
            "--ro",
            "/etc",
            "--ro",
            str(allowed),
            "--",
            "/bin/sh",
            "-c",
            script,
            "sh",
            str(allowed),
            str(forbidden),
        ],
        check=False,
    )
    assert completed.returncode == 0

    network = subprocess.run(
        [
            str(launcher),
            "--ro",
            "/usr",
            "--ro",
            "/etc",
            "--",
            "/usr/bin/python3",
            "-c",
            (
                "import errno,socket\n"
                "try: socket.create_connection(('127.0.0.1',9),timeout=.01)\n"
                "except OSError as error: raise SystemExit(0 if error.errno==errno.EACCES else 2)\n"
                "raise SystemExit(3)"
            ),
        ],
        check=False,
        capture_output=True,
    )
    assert network.returncode == 0

    datagram = subprocess.run(
        [
            str(launcher),
            "--ro",
            "/usr",
            "--ro",
            "/etc",
            "--",
            "/usr/bin/python3",
            "-c",
            (
                "import errno,socket\n"
                "try: socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n"
                "except OSError as error: raise SystemExit(0 if error.errno==errno.EACCES else 2)\n"
                "raise SystemExit(3)"
            ),
        ],
        check=False,
        capture_output=True,
    )
    assert datagram.returncode == 0


def test_landlock_launcher_traverses_parent_without_exposing_siblings(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "landlock_exec.c"
    launcher = tmp_path / "landlock-exec"
    subprocess.run(
        ["cc", "-std=c17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(launcher)],
        check=True,
    )
    cache = tmp_path / "cache"
    allowed = cache / "model"
    forbidden = cache / "dataset"
    allowed.mkdir(parents=True)
    forbidden.mkdir()
    (allowed / "config").write_text("registered")
    (forbidden / "labels").write_text("hidden")
    script = (
        'test "$(cat "$1/config")" = registered && if cat "$2/labels"; then exit 1; else exit 0; fi'
    )
    completed = subprocess.run(
        [
            str(launcher),
            "--ro",
            "/usr",
            "--ro",
            "/etc",
            "--traverse",
            str(cache),
            "--ro",
            str(allowed),
            "--",
            "/bin/sh",
            "-c",
            script,
            "sh",
            str(allowed),
            str(forbidden),
        ],
        check=False,
    )
    assert completed.returncode == 0


def test_coverage_fit_policy_reads_support_writes_fit_and_denies_heldout(
    tmp_path: Path,
) -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "landlock_exec.c"
    launcher = tmp_path / "landlock-exec"
    subprocess.run(
        ["cc", "-std=c17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(launcher)],
        check=True,
    )
    support = tmp_path / "support"
    heldout = tmp_path / "heldout"
    phase1 = tmp_path / "phase1"
    support.mkdir()
    heldout.mkdir()
    phase1.mkdir()
    (support / "pixel").write_bytes(b"support")
    (heldout / "pixel").write_bytes(b"heldout")
    program = (
        "import errno,os,pathlib,sys\n"
        "support,heldout,out=map(pathlib.Path,sys.argv[1:])\n"
        "assert support.read_bytes()==b'support'\n"
        "out.write_bytes(b'sealed')\n"
        "try: (heldout/'pixel').read_bytes()\n"
        "except PermissionError as e: assert e.errno==errno.EACCES\n"
        "else: raise SystemExit(3)\n"
        "try: list(os.scandir(heldout))\n"
        "except PermissionError as e: assert e.errno==errno.EACCES\n"
        "else: raise SystemExit(4)\n"
    )
    completed = subprocess.run(
        [
            str(launcher),
            "--ro",
            "/usr",
            "--ro",
            "/etc",
            "--ro",
            str(support),
            "--rw",
            str(phase1),
            "--",
            "/usr/bin/python3",
            "-c",
            program,
            str(support / "pixel"),
            str(heldout),
            str(phase1 / "map"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
