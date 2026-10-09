#!/usr/bin/env bash
set -euo pipefail
cd /home/rb/worktrees/sfora-connected-serving-manifest-binder-20261009
ulimit -v 1048576
export PYTHONDONTWRITEBYTECODE=1
python3.12 --version
python3.14 --version
/data/cache/uv/archive-v0/FGM38RAsHizKx-KP/bin/ruff --version
/data/cache/uv/archive-v0/A5hAhU0uya3PTOGc/bin/mypy --version
python3.12 -B -S scripts/test_connected_serving_artifact.py -v
python3.14 -B -S scripts/test_connected_serving_artifact.py -v
/data/cache/uv/archive-v0/FGM38RAsHizKx-KP/bin/ruff format --check src/sfora/connected_serving_artifact.py scripts/test_connected_serving_artifact.py
/data/cache/uv/archive-v0/FGM38RAsHizKx-KP/bin/ruff check src/sfora/connected_serving_artifact.py scripts/test_connected_serving_artifact.py
MYPYPATH=src:scripts /data/cache/uv/archive-v0/A5hAhU0uya3PTOGc/bin/mypy --strict --follow-imports=silent --no-incremental --cache-dir=/data/cache/sfora-connected-serving-manifest-binder-20261009/mypy-cache src/sfora/connected_serving_artifact.py scripts/test_connected_serving_artifact.py
git diff --check
sha256sum src/sfora/connected_serving_artifact.py scripts/test_connected_serving_artifact.py
