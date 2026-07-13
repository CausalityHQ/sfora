#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# sync
rsync -az --delete --exclude .venv --exclude .git --exclude data --exclude reports/generated "${LOCAL_DIR}/" researcher@gpu.example.com:/home/researcher/group-learning/

# setup
ssh researcher@gpu.example.com 'cd /home/researcher/group-learning && (uv sync --group dev --extra research) || [ -x .venv/bin/group-learning ]'

# run
ssh researcher@gpu.example.com 'cd /home/researcher/group-learning && (uv run --group dev --extra research group-learning imdb-encoder-train --limit-per-class 128 --group-size 4 --train-steps 80 --output reports/generated/imdb_encoder_training.json) || ([ -x .venv/bin/group-learning ] && .venv/bin/group-learning imdb-encoder-train --limit-per-class 128 --group-size 4 --train-steps 80 --output reports/generated/imdb_encoder_training.json)'

# fetch-reports
rsync -az researcher@gpu.example.com:/home/researcher/group-learning/reports/generated/ "${LOCAL_DIR}/reports/generated/"
