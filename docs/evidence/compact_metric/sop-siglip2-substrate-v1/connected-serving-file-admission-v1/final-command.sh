#!/usr/bin/env bash
set -euo pipefail
cd /home/rb/worktrees/sfora-connected-serving-file-admission-20261009
ulimit -v 1048576
export PYTHONDONTWRITEBYTECODE=1
OWN2="src/sfora/connected_serving_admission.py scripts/test_connected_serving_admission.py"
sha256sum $OWN2 > /tmp/claude-1001/-home-rb-worktrees-sfora-connected-serving-file-admission-20261009/ab2699e3-9dd7-4a62-9a58-afb6d6710ae9/scratchpad/final/final-before.sha256
python3.12 --version
python3.14 --version
/data/cache/uv/archive-v0/FGM38RAsHizKx-KP/bin/ruff --version
/data/cache/uv/archive-v0/A5hAhU0uya3PTOGc/bin/mypy --version
python3.12 -B -S scripts/test_connected_serving_admission.py -v
python3.14 -B -S scripts/test_connected_serving_admission.py -v
/data/cache/uv/archive-v0/FGM38RAsHizKx-KP/bin/ruff format --check $OWN2
/data/cache/uv/archive-v0/FGM38RAsHizKx-KP/bin/ruff check $OWN2
MYPYPATH=src:scripts /data/cache/uv/archive-v0/A5hAhU0uya3PTOGc/bin/mypy --strict --follow-imports=silent --no-incremental --cache-dir=/tmp/claude-1001/-home-rb-worktrees-sfora-connected-serving-file-admission-20261009/ab2699e3-9dd7-4a62-9a58-afb6d6710ae9/scratchpad/final/mypy-cache $OWN2
sha256sum $OWN2 > /tmp/claude-1001/-home-rb-worktrees-sfora-connected-serving-file-admission-20261009/ab2699e3-9dd7-4a62-9a58-afb6d6710ae9/scratchpad/final/final-after.sha256
