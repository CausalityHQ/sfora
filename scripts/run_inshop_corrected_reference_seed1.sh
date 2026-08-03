#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/riomus/sfora-inshop-seed1}"
PYTHON_BIN="${PYTHON_BIN:-/home/riomus/group-learning/.venv/bin/python}"
SFORA_BIN="${SFORA_BIN:-/home/riomus/group-learning/.venv/bin/sfora}"
DATASET_ROOT="${DATASET_ROOT:-/home/riomus/datasets/inshop}"
REPORT_REL="reports/generated/inshop_corrected_pa_seed1.json"
CHECKPOINT_REL="reports/checkpoints/inshop_corrected_pa_seed1.pt"

cd "$PROJECT_DIR"
test ! -e "$REPORT_REL"
test ! -e "$CHECKPOINT_REL"
mkdir -p reports/generated reports/checkpoints

env PYTHONPATH=src "$SFORA_BIN" image-end-to-end \
  --dataset-name inshop \
  --dataset-root "$DATASET_ROOT" \
  --objectives proxy_anchor \
  --recipe proxy_anchor.inshop.official-51db570 \
  --num-workers 8 \
  --seed 1 \
  --save-model-path "$CHECKPOINT_REL" \
  --output "$REPORT_REL"

env PYTHONPATH=src "$PYTHON_BIN" - "$REPORT_REL" "$CHECKPOINT_REL" <<'PY'
import json
import sys
from pathlib import Path

import torch

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checkpoint = torch.load(sys.argv[2], map_location="cpu")
config = report["config"]
if config["seed"] != 1 or config["recipe_id"] != "proxy_anchor.inshop.official-51db570":
    raise SystemExit("seed-1 report does not contain the locked recipe")
if checkpoint.get("artifact_selection") != "final_training_state":
    raise SystemExit("seed-1 checkpoint is not the final training state")
if checkpoint.get("training_step") != 8580:
    raise SystemExit("seed-1 checkpoint did not complete 8,580 steps")
if checkpoint.get("training_config") != config:
    raise SystemExit("seed-1 checkpoint/report configs differ")
PY
