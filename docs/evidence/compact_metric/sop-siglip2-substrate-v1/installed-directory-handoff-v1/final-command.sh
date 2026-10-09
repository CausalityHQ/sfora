#!/bin/bash
set -euo pipefail
ulimit -v 1048576
for py in python3.12 python3.14; do
 "$py" -B -S scripts/test_connected_directory_handoff.py
 "$py" -B -S scripts/test_connected_artifact_identity.py
 "$py" -B -S scripts/test_connected_installed_environment.py
done
/data/cache/uv/archive-v0/FGM38RAsHizKx-KP/bin/ruff format --check src/sfora/connected_artifact_identity.py src/sfora/connected_installed_environment.py scripts/test_connected_directory_handoff.py
/data/cache/uv/archive-v0/FGM38RAsHizKx-KP/bin/ruff check src/sfora/connected_artifact_identity.py src/sfora/connected_installed_environment.py scripts/test_connected_directory_handoff.py
MYPYPATH=src:scripts /data/cache/uv/archive-v0/A5hAhU0uya3PTOGc/bin/mypy --strict --follow-imports=silent --no-incremental --cache-dir=/tmp/sfora-directory-handoff-mypy-cache src/sfora/connected_artifact_identity.py src/sfora/connected_installed_environment.py scripts/test_connected_directory_handoff.py
git diff --check
