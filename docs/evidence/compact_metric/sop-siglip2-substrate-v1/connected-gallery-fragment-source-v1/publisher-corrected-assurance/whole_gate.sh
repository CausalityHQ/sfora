#!/bin/bash
set -u
A=/data/cache/uv/archive-v0; S=/tmp/claude-1001/-home-rb-worktrees-sfora-immutable-publisher-cleanup-20261009/527f8de6-000e-4bd3-8a3f-b341f3e97e59/scratchpad
export MYPYPATH="src:$A/781smqp9ZslgnBns:$A/b3T7675lBPxDnPfA:$A/cfLJr8GRPzjnDaWD:$A/qE2bTGbmY4iQ_5BY:$A/lvdLzVHI5LWsZwq0"
rc=0
$S/run_pytest.sh 15 tests/test_atomic_publication.py -q 2>&1 | tail -2 || rc=1
$A/FGM38RAsHizKx-KP/bin/ruff check --no-cache src/sfora/atomic_publication.py tests/test_atomic_publication.py || rc=1
$A/FGM38RAsHizKx-KP/bin/ruff format --check --no-cache src/sfora/atomic_publication.py || rc=1
$A/A5hAhU0uya3PTOGc/bin/mypy --follow-imports=silent --python-executable /usr/bin/python3.14 --cache-dir=$S/mypycache src/sfora/atomic_publication.py tests/test_atomic_publication.py 2>&1 | grep -v 'unused section' | tail -1
echo "gate-script-rc=$rc"
