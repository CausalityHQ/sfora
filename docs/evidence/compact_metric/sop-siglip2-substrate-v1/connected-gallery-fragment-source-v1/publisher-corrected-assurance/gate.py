#!/usr/bin/env python3
"""ONE corrected serial assurance run on committed bytes.

Every tool writes its full raw output straight to a log file (no pipes), and its exact exit
status, wall time and peak RSS come from wait4. Nothing here is gated by a pipeline status.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

S = Path("/tmp/claude-1001/-home-rb-worktrees-sfora-immutable-publisher-cleanup-20261009/527f8de6-000e-4bd3-8a3f-b341f3e97e59/scratchpad")
REPO = Path("/home/rb/worktrees/sfora-immutable-publisher-cleanup-20261009")
A = "/data/cache/uv/archive-v0"
RUFF = f"{A}/FGM38RAsHizKx-KP/bin/ruff"
MYPY = f"{A}/A5hAhU0uya3PTOGc/bin/mypy"
PY314, PY312 = "/usr/bin/python3.14", "/home/rb/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
HEAD = "831ecdc30051fbf092ad7bacc5da378c6dbb8776"
BASE = "fb0e1661416e77cc320e497da7bb8f4721f00237"
PROD, TEST = "src/sfora/atomic_publication.py", "tests/test_atomic_publication.py"
EXPECT = {
    PROD: "00ffdc28ecf47a90dc1e6b74427aa4da4c3d30f3698598c9074a31af61c1415f",
    TEST: "b7e7b49706fd2b3a642a55f7c741fb169b595f288fa3c9b2615fa935f54eced3",
}
OUT = S / "gate-final"
OUT.mkdir()  # refuses to overwrite earlier evidence
BASE_TREE = OUT / "baseline-tree"
START = time.monotonic()
DEADLINE = START + 110
PYPATH = ":".join(
    ["src", f"{S}/sentinel", f"{A}/781smqp9ZslgnBns", f"{A}/b3T7675lBPxDnPfA",
     f"{A}/cfLJr8GRPzjnDaWD", f"{A}/qE2bTGbmY4iQ_5BY", f"{A}/lvdLzVHI5LWsZwq0"]
)
steps: list[dict] = []


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(name: str, argv: list[str], *, cwd: Path = REPO, env: dict | None = None,
        timeout: float = 60.0) -> dict:
    """Run one tool serially; raw stdout+stderr -> log; exit/wall/RSS via wait4."""
    log = OUT / f"{name}.log"
    full_env = dict(os.environ, **(env or {}))
    began = time.monotonic()
    with log.open("wb") as sink:
        proc = subprocess.Popen(argv, cwd=cwd, env=full_env, stdout=sink, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        timed_out = False
        while True:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() - began > timeout or time.monotonic() > DEADLINE:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                pid, status, usage = os.wait4(proc.pid, 0)
                break
            time.sleep(0.005)
    proc.returncode = os.waitstatus_to_exitcode(status)
    record = {
        "name": name, "argv": argv, "cwd": str(cwd), "exit": proc.returncode,
        "wall_s": round(time.monotonic() - began, 3), "maxrss_kb": usage.ru_maxrss,
        "timed_out": timed_out, "log": log.name, "log_bytes": log.stat().st_size,
        "log_sha256": sha(log.read_bytes()),
    }
    steps.append(record)
    return record


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True).stdout


# ---------------------------------------------------------------- 0. exact committed bytes
facts = {
    "head": git("rev-parse", "HEAD").decode().strip(),
    "parent": git("rev-parse", "HEAD^").decode().strip(),
    "status_porcelain_before": git("status", "--porcelain").decode(),
    "blob_sha256": {p: sha(git("show", f"HEAD:{p}")) for p in EXPECT},
    "worktree_sha256": {p: sha((REPO / p).read_bytes()) for p in EXPECT},
}
assert facts["head"] == HEAD and facts["parent"] == BASE, facts
assert facts["blob_sha256"] == EXPECT == facts["worktree_sha256"], facts
assert facts["status_porcelain_before"] == "", facts

# baseline tree: same layout, bytes straight from the exact base commit
for rel in ("pyproject.toml", "src/sfora/__init__.py", PROD, TEST):
    target = BASE_TREE / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(git("show", f"{BASE}:{rel}"))
assert sha((BASE_TREE / PROD).read_bytes()) == "ee0870e4e34a3bb08e82b6aa6d222a1f29dce7f17edfb331973d34dadac8ed30"

# ---------------------------------------------------------------- 1. ML sentinel self-test + versions
sent_env = {"PYTHONPATH": f"{S}/sentinel", "PYTHONDONTWRITEBYTECODE": "1"}
run("sentinel-selftest", [PY314, "-c", "import mlsentinel, numpy"], env=sent_env)
log = (OUT / "sentinel-selftest.log").read_text()
facts["sentinel_selftest_blocked_numpy"] = steps[-1]["exit"] != 0 and "ML import sentinel tripped: numpy" in log
for label, argv in (("ruff", [RUFF, "--version"]), ("mypy", [MYPY, "--version"]),
                    ("python314", [PY314, "--version"]), ("python312", [PY312, "--version"])):
    facts[f"version_{label}"] = subprocess.run(argv, capture_output=True, text=True).stdout.strip()

# ---------------------------------------------------------------- 2. pytest (plugins off, ML sentinel)
py_env = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": PYPATH, "PYTHONDONTWRITEBYTECODE": "1",
          "PYTHONPYCACHEPREFIX": str(S / "pyc")}
for tag, interpreter in (("pytest-py314", PY314), ("pytest-py312", PY312)):
    run(tag, [interpreter, "-m", "pytest", "-p", "mlsentinel", "-p", "no:cacheprovider", "-v", TEST],
        env=py_env, timeout=30)

# ---------------------------------------------------------------- 3. ruff
run("ruff-check", [RUFF, "check", "--no-cache", PROD, TEST])
run("ruff-format-check-prod", [RUFF, "format", "--check", "--no-cache", PROD])
run("ruff-format-diff-tests-current", [RUFF, "format", "--diff", "--no-cache", TEST])
run("ruff-format-diff-tests-baseline", [RUFF, "format", "--diff", "--no-cache", TEST], cwd=BASE_TREE)


def normalized_diff(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [re.sub(r"^@@ .* @@", "@@", l) if l.startswith("@@") else ("---/+++" if l.startswith(("--- ", "+++ ")) else l)
            for l in lines]


def hunks(path: Path) -> list[tuple[int, int]]:
    return [(int(m[1]), int(m[2] or 1)) for l in path.read_text().splitlines()
            if (m := re.match(r"^@@ -(\d+)(?:,(\d+))? \+", l))]


def added_ranges(rel: str) -> list[tuple[int, int]]:
    out = git("diff", "-U0", BASE, HEAD, "--", rel).decode().splitlines()
    return [(int(m[1]), int(m[2] or 1)) for l in out if (m := re.match(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", l))]


cur_hunks = hunks(OUT / "ruff-format-diff-tests-current.log")
added = added_ranges(TEST)
overlap = [(h, a) for h in cur_hunks for a in added if h[0] <= a[0] + a[1] - 1 and a[0] <= h[0] + h[1] - 1 and a[1] > 0]
facts["ruff_format_tests"] = {
    "baseline_exit": steps[-1]["exit"], "current_exit": steps[-2]["exit"],
    "normalized_diff_identical_to_baseline": normalized_diff(OUT / "ruff-format-diff-tests-current.log")
    == normalized_diff(OUT / "ruff-format-diff-tests-baseline.log"),
    "current_hunk_old_ranges": cur_hunks, "added_line_ranges": added, "hunks_overlapping_added_lines": overlap,
}

# ---------------------------------------------------------------- 4. mypy (current vs same-baseline source)
mypy_env = {"MYPYPATH": PYPATH}
mypy_common = [MYPY, "--follow-imports=silent", "--python-executable", PY314]
run("mypy-current", [*mypy_common, f"--cache-dir={OUT}/mypycache-current", PROD, TEST], env=mypy_env, timeout=60)
run("mypy-baseline", [*mypy_common, f"--cache-dir={OUT}/mypycache-baseline", PROD, TEST], cwd=BASE_TREE,
    env=mypy_env, timeout=60)

DIAG = re.compile(r"^(?P<path>\S+?):(?P<line>\d+): (?P<sev>error|note): (?P<msg>.*?)(?:  \[(?P<code>[a-z0-9-]+)\])?$")


def diagnostics(log_path: Path, root: Path) -> collections.Counter:
    found: collections.Counter = collections.Counter()
    for raw in log_path.read_text().splitlines():
        m = DIAG.match(raw)
        if not m or m["path"] == "pyproject.toml":
            continue
        source = (root / m["path"]).read_text().splitlines()[int(m["line"]) - 1].strip()
        found[(m["path"], m["sev"], m["code"], m["msg"], source)] += 1
    return found


cur_diag = diagnostics(OUT / "mypy-current.log", REPO)
base_diag = diagnostics(OUT / "mypy-baseline.log", BASE_TREE)
cur_lines = [int(DIAG.match(l)["line"]) for l in (OUT / "mypy-current.log").read_text().splitlines()
             if DIAG.match(l) and DIAG.match(l)["path"] != "pyproject.toml"]
facts["mypy"] = {
    "current_exit": steps[-2]["exit"], "baseline_exit": steps[-1]["exit"],
    "current_count": sum(cur_diag.values()), "baseline_count": sum(base_diag.values()),
    "normalized_multiset_identical": cur_diag == base_diag,
    "introduced": [list(k) for k in (cur_diag - base_diag)], "removed": [list(k) for k in (base_diag - cur_diag)],
    "verdict": ("NO_NEW_OR_REMOVED_DIAGNOSTICS_VS_BASELINE" if cur_diag == base_diag else "DIAGNOSTICS_DIFFER_FROM_BASELINE")
    + " (baseline pre-existing errors remain; this is NOT a mypy PASS)",
    "normalization": "key=(path,severity,code,message,stripped source text of the flagged line); line numbers ignored",
}

# ---------------------------------------------------------------- 5. finite inverse on the committed bytes
run("finite-inverse", [PY314, "-I", str(S / "verify_inverse.py")], timeout=30)

# ---------------------------------------------------------------- 6. cleanliness + summary
facts["status_porcelain_after"] = git("status", "--porcelain").decode()
facts["head_after"] = git("rev-parse", "HEAD").decode().strip()
facts["worktree_sha256_after"] = {p: sha((REPO / p).read_bytes()) for p in EXPECT}
facts["total_wall_s"] = round(time.monotonic() - START, 3)
facts["driver_maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
facts["rlimit_as_bytes"] = resource.getrlimit(resource.RLIMIT_AS)[0]
(OUT / "gate-result.json").write_text(json.dumps({"facts": facts, "steps": steps}, indent=2, sort_keys=True, default=str) + "\n")

print(json.dumps({"steps": [{k: s[k] for k in ("name", "exit", "wall_s", "maxrss_kb", "timed_out")} for s in steps],
                  "facts": {k: v for k, v in facts.items() if k not in ("blob_sha256", "worktree_sha256", "worktree_sha256_after")}},
                 indent=1, default=str))
