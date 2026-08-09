#!/usr/bin/env bash
set -euo pipefail

R=/home/riomus/sfora-pass201-pa-source-v2-r4
C4="$1"
PY=/home/riomus/group-learning/.venv/bin/python
OUT1=/home/riomus/sfora-pass201-pa-source-v2-r4.pass201-prelaunch-freeze-1.tmp
OUT2=/home/riomus/sfora-pass201-pa-source-v2-r4.pass201-prelaunch-freeze-2.tmp
CANON="$R/docs/pass201_pa_source_v2_prelaunch.json"
RUN="$R/reports/generated/pass201_source_v2/run-v2"

require_idle_queue() {
  local gpu_rows gpu_status controller_rows controller_status
  set +e
  gpu_rows=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits)
  gpu_status=$?
  controller_rows=$(pgrep -af 'run_pass201_pa_source_v2[.]py|pass201[^ ]*(controller|watch)|((controller|watch)[^ ]*pass201)')
  controller_status=$?
  set -e
  test "$gpu_status" -eq 0
  test -z "$gpu_rows"
  test "$controller_status" -eq 1
  test -z "$controller_rows"
}

require_clean_status() {
  local status_rows status_code
  set +e
  status_rows=$(git status --porcelain=v1)
  status_code=$?
  set -e
  test "$status_code" -eq 0
  test -z "$status_rows"
}

test "$#" -eq 1
test "$(hostname)" = spark-2751
cd /home/riomus
test "$(pwd -P)" = /home/riomus
test ! -e "$R" && test ! -L "$R"
test ! -e "$OUT1" && test ! -L "$OUT1"
test ! -e "$OUT2" && test ! -L "$OUT2"
test -x "$PY" && test -f "$PY"
require_idle_queue
test "$(git ls-remote https://github.com/CausalityHQ/sfora.git refs/heads/devbox/emafactorial | cut -f1)" = "$C4"
git clone --no-checkout --single-branch --branch devbox/emafactorial \
  https://github.com/CausalityHQ/sfora.git "$R"
cd "$R"
git checkout --detach "$C4"
test "$(pwd -P)" = "$R"
test "$(git rev-parse HEAD)" = "$C4"
test "$(git cat-file -t HEAD)" = commit
PARENT_FIELDS=()
read -r -a PARENT_FIELDS <<< "$(git rev-list --parents -n 1 HEAD)"
test "${#PARENT_FIELDS[@]}" -eq 2
test "${PARENT_FIELDS[0]}" = "$C4"
test -z "$(git symbolic-ref -q HEAD || true)"
require_clean_status
test ! -e "$OUT1" && test ! -L "$OUT1"
test ! -e "$OUT2" && test ! -L "$OUT2"
test ! -e "$CANON" && test ! -L "$CANON"
test ! -e "$RUN" && test ! -L "$RUN"
require_idle_queue
AUTHOR_IDENT=$(git var GIT_AUTHOR_IDENT)
COMMITTER_IDENT=$(git var GIT_COMMITTER_IDENT)
test -n "$AUTHOR_IDENT"
test -n "$COMMITTER_IDENT"

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

env -i \
  HOME=/home/riomus \
  PATH=/home/riomus/group-learning/.venv/bin:/usr/bin:/bin \
  PYTHONPATH="$R/src" \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED=0 \
  LC_ALL=C.UTF-8 \
  LANG=C.UTF-8 \
  TZ=UTC \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  XDG_CACHE_HOME=/home/riomus/.cache \
  TORCH_HOME=/home/riomus/.cache/torch \
  "$PY" scripts/run_pass201_pa_source_v2.py freeze-authority \
    --frozen-absence-checked-utc "$TS" \
    --output "$OUT1" &
PID1=$!
wait "$PID1"
test -f "$OUT1" && test ! -L "$OUT1"
test ! -e "$CANON" && test ! -L "$CANON"
test ! -e "$RUN" && test ! -L "$RUN"
require_clean_status

env -i \
  HOME=/home/riomus \
  PATH=/home/riomus/group-learning/.venv/bin:/usr/bin:/bin \
  PYTHONPATH="$R/src" \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED=0 \
  LC_ALL=C.UTF-8 \
  LANG=C.UTF-8 \
  TZ=UTC \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  XDG_CACHE_HOME=/home/riomus/.cache \
  TORCH_HOME=/home/riomus/.cache/torch \
  "$PY" scripts/run_pass201_pa_source_v2.py freeze-authority \
    --frozen-absence-checked-utc "$TS" \
    --output "$OUT2" &
PID2=$!
wait "$PID2"
test "$PID1" -ne "$PID2"
test -f "$OUT2" && test ! -L "$OUT2"
test ! -e "$CANON" && test ! -L "$CANON"
test ! -e "$RUN" && test ! -L "$RUN"
require_clean_status
cmp -s -- "$OUT1" "$OUT2"

MANIFEST_EVIDENCE=$(env -i \
  HOME=/home/riomus \
  PATH=/home/riomus/group-learning/.venv/bin:/usr/bin:/bin \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  LC_ALL=C.UTF-8 \
  LANG=C.UTF-8 \
  TZ=UTC \
  "$PY" - "$OUT1" "$OUT2" "$CANON" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path

out1, out2, canonical = map(Path, sys.argv[1:])

def read_fd(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)

def open_regular(path: Path) -> tuple[int, os.stat_result]:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SystemExit(f"not a regular file: {path}")
        return fd, info
    except BaseException:
        os.close(fd)
        raise

def unlink_name_if_same(path: Path, expected: os.stat_result, ordinal: int) -> None:
    quarantine = path.with_name(f".{path.name}.quarantine-{os.getpid()}-{ordinal}")
    if quarantine.exists() or quarantine.is_symlink():
        raise SystemExit(f"unlink quarantine already exists: {quarantine}")
    os.rename(path, quarantine)
    moved = os.stat(quarantine, follow_symlinks=False)
    if (moved.st_dev, moved.st_ino) == (expected.st_dev, expected.st_ino):
        os.unlink(quarantine)
        return
    try:
        os.link(quarantine, path, follow_symlinks=False)
    except BaseException as exc:
        raise SystemExit(
            f"temporary name changed; preserved unexpected inode at {quarantine}"
        ) from exc
    os.unlink(quarantine)
    raise SystemExit(f"temporary name changed; restored unexpected inode at {path}")

fd1, s1 = open_regular(out1)
fd2, s2 = open_regular(out2)
try:
    b1 = read_fd(fd1)
    b2 = read_fd(fd2)
    if b1 != b2 or s1.st_size != s2.st_size:
        raise SystemExit("temporary authority outputs differ")
    if s1.st_dev != s2.st_dev:
        raise SystemExit("temporary authority outputs use different filesystems")
    if canonical.exists() or canonical.is_symlink():
        raise SystemExit("canonical manifest already exists")
    if os.stat(canonical.parent, follow_symlinks=False).st_dev != s1.st_dev:
        raise SystemExit("canonical parent uses a different filesystem")

    os.link(out1, canonical, follow_symlinks=False)
    linked = os.stat(canonical, follow_symlinks=False)
    if (linked.st_dev, linked.st_ino) != (s1.st_dev, s1.st_ino):
        raise SystemExit("canonical manifest is not linked to accepted descriptor")
    parent_fd = os.open(canonical.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)

    unlink_name_if_same(out1, s1, 1)
    unlink_name_if_same(out2, s2, 2)
    for parent in {out1.parent, out2.parent, canonical.parent}:
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    published_fd, published_stat = open_regular(canonical)
    try:
        published = read_fd(published_fd)
    finally:
        os.close(published_fd)
    if published != b1 or published_stat.st_size != s1.st_size:
        raise SystemExit("published authority bytes changed")
finally:
    os.close(fd1)
    os.close(fd2)
digest = hashlib.sha256(published).hexdigest()
print(f"manifest_sha256={digest}")
print(f"manifest_bytes={len(published)}")
PY
)

test ! -e "$OUT1" && test ! -L "$OUT1"
test ! -e "$OUT2" && test ! -L "$OUT2"
test ! -e "$RUN" && test ! -L "$RUN"
test "$(git status --porcelain=v1)" = "?? docs/pass201_pa_source_v2_prelaunch.json"

git add -- docs/pass201_pa_source_v2_prelaunch.json
test "$(git diff --cached --name-status)" = $'A\tdocs/pass201_pa_source_v2_prelaunch.json'
test "$(git diff --cached --summary)" = " create mode 100644 docs/pass201_pa_source_v2_prelaunch.json"
git -c commit.gpgsign=false commit -m "freeze Pass201 PA source v2 launch"

A4=$(git rev-parse HEAD)
PARENT_FIELDS=()
read -r -a PARENT_FIELDS <<< "$(git rev-list --parents -n 1 HEAD)"
test "${#PARENT_FIELDS[@]}" -eq 2
test "${PARENT_FIELDS[0]}" = "$A4"
test "${PARENT_FIELDS[1]}" = "$C4"
test "$(git diff-tree --no-commit-id --name-status -r "$C4" "$A4")" = $'A\tdocs/pass201_pa_source_v2_prelaunch.json'
test "$(git ls-tree "$A4" docs/pass201_pa_source_v2_prelaunch.json)" = "100644 blob $(git hash-object "$CANON")"$'\t'"docs/pass201_pa_source_v2_prelaunch.json"
test "$(pwd -P)" = "$R"
test -z "$(git symbolic-ref -q HEAD || true)"
require_clean_status
test ! -e "$RUN" && test ! -L "$RUN"

printf 'freeze_timestamp=%s\n' "$TS"
printf 'freeze_pid_1=%s\n' "$PID1"
printf 'freeze_pid_2=%s\n' "$PID2"
printf '%s\n' "$MANIFEST_EVIDENCE"
printf 'source_commit=%s\n' "$C4"
printf 'authorization_commit=%s\n' "$A4"
printf 'checkout=%s\n' "$R"
printf 'transport_status=WITHHELD_PENDING_SSH_FETCH_AND_INDEPENDENT_REVIEW\n'
