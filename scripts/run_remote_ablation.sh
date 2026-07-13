#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# sync
rsync -az --delete --exclude .venv --exclude .git --exclude data --exclude reports/generated "${LOCAL_DIR}/" riomus@192.168.1.35:/home/riomus/group-learning/

# setup
ssh riomus@192.168.1.35 'cd /home/riomus/group-learning && (uv sync --group dev --extra research) || [ -x .venv/bin/group-learning ]'

# run
ssh riomus@192.168.1.35 'cd /home/riomus/group-learning && (uv run --group dev --extra research group-learning imdb-encoder-ablation --limit-per-class 1024 --objectives triplet,group,hybrid,hybrid_xbm,hybrid_radius,hybrid_xbm_radius --train-steps-grid 20,80 --learning-rates 0.00002 --group-sizes 4,8,16 --batch-size 64 --output reports/generated/imdb_encoder_ablation.json) || ([ -x .venv/bin/group-learning ] && .venv/bin/group-learning imdb-encoder-ablation --limit-per-class 1024 --objectives triplet,group,hybrid,hybrid_xbm,hybrid_radius,hybrid_xbm_radius --train-steps-grid 20,80 --learning-rates 0.00002 --group-sizes 4,8,16 --batch-size 64 --output reports/generated/imdb_encoder_ablation.json)'

# fetch-reports
rsync -az riomus@192.168.1.35:/home/riomus/group-learning/reports/generated/ "${LOCAL_DIR}/reports/generated/"
