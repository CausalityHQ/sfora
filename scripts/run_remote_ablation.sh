#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# sync
rsync -az --delete --exclude .venv --exclude .git --exclude data --exclude reports/generated "${LOCAL_DIR}/" researcher@gpu.example.com:/home/researcher/group-learning/

# setup
ssh researcher@gpu.example.com 'cd /home/researcher/group-learning && (uv sync --group dev --extra research) || [ -x .venv/bin/group-learning ]'

# run
ssh researcher@gpu.example.com 'cd /home/researcher/group-learning && (uv run --group dev --extra research group-learning imdb-encoder-ablation --limit-per-class 1024 --objectives triplet,group,hybrid,hybrid_xbm,hybrid_radius,hybrid_xbm_radius --train-steps-grid 20,80 --learning-rates 0.00002 --group-sizes 4,8,16 --batch-size 64 --output reports/generated/imdb_encoder_ablation.json) || ([ -x .venv/bin/group-learning ] && .venv/bin/group-learning imdb-encoder-ablation --limit-per-class 1024 --objectives triplet,group,hybrid,hybrid_xbm,hybrid_radius,hybrid_xbm_radius --train-steps-grid 20,80 --learning-rates 0.00002 --group-sizes 4,8,16 --batch-size 64 --output reports/generated/imdb_encoder_ablation.json)'

# fetch-reports
rsync -az researcher@gpu.example.com:/home/researcher/group-learning/reports/generated/ "${LOCAL_DIR}/reports/generated/"
