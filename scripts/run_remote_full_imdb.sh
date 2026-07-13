#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# sync
rsync -az --delete --exclude .venv --exclude .git --exclude data --exclude reports/generated "${LOCAL_DIR}/" researcher@gpu.example.com:/home/researcher/group-learning/

# setup
ssh researcher@gpu.example.com 'cd /home/researcher/group-learning && (uv sync --group dev --extra research) || [ -x .venv/bin/sfora ]'

# run
ssh researcher@gpu.example.com 'cd /home/researcher/group-learning && (uv run --group dev --extra research sfora imdb-encoder-train --model-name sentence-transformers/paraphrase-MiniLM-L3-v2 --limit-per-class 12500 --test-limit-per-class 12500 --official-test-split --retrieval-query-limit 1024 --group-size 16 --batch-size 64 --train-steps 20 --output reports/generated/imdb_encoder_training.full.json) || ([ -x .venv/bin/sfora ] && .venv/bin/sfora imdb-encoder-train --model-name sentence-transformers/paraphrase-MiniLM-L3-v2 --limit-per-class 12500 --test-limit-per-class 12500 --official-test-split --retrieval-query-limit 1024 --group-size 16 --batch-size 64 --train-steps 20 --output reports/generated/imdb_encoder_training.full.json)'

# fetch-reports
rsync -az researcher@gpu.example.com:/home/researcher/group-learning/reports/generated/ "${LOCAL_DIR}/reports/generated/"
