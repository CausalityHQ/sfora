#!/usr/bin/env bash
# Cars196 final-state authority for the train-only method-frontier measurement.
# This is a fresh output namespace: v47 reports cannot satisfy or skip this run.
set -euo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"
REPORT_ROOT="${PROJECT}/reports/generated/cars-method-frontier-2026-08-30"
REPORT="${REPORT_ROOT}/proxy_anchor.d55241a64a5a.deterministic.seed0.json"
CHECKPOINT="${REPORT_ROOT}/proxy_anchor.d55241a64a5a.deterministic.seed0.final.pt"
TRAIN_PACK="${REPORT_ROOT}/proxy_anchor.d55241a64a5a.deterministic.seed0.final.train.npz"
FRONTIER_RESULT="${REPORT_ROOT}/proxy_anchor.d55241a64a5a.deterministic.seed0.frontier.json"
RUN_LOG="${LOG_ROOT}/cars-method-frontier-v48.proxy_anchor.seed0.log"
CONTROLLER_LOG="${LOG_ROOT}/cars-method-frontier-v48.controller.log"
RECIPE_ID="proxy_anchor.cars.official-51db570"
RECIPE_DIGEST="d55241a64a5afe9ea81be02e74fa13a6fec87e15c66e95918ad10d90337cc02a"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

mkdir -p "${LOG_ROOT}" "${REPORT_ROOT}"
cd "${PROJECT}"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${CONTROLLER_LOG}"
}

mapfile -t recipe < <(.venv/bin/python - <<'PY'
from sfora.image_recipes import recipe_digest, resolve_recipe

recipe = resolve_recipe("auto", base_method="proxy_anchor", dataset="cars")
print(recipe.recipe_id)
print(recipe_digest(recipe))
print(",".join(recipe.config["objectives"]))
PY
)
[[ "${recipe[0]}" == "${RECIPE_ID}" ]]
[[ "${recipe[1]}" == "${RECIPE_DIGEST}" ]]
[[ "${recipe[2]}" == "proxy_anchor" ]]

gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
if [[ -n "${gpu_pids//[[:space:]]/}" ]]; then
  log "STOP GPU already has a compute process"
  exit 2
fi
for path in "${REPORT}" "${CHECKPOINT}" "${TRAIN_PACK}" "${FRONTIER_RESULT}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    log "STOP refusing pre-existing authority path ${path}"
    exit 2
  fi
done
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  log "PREFLIGHT PASS exact Cars Proxy Anchor recipe, final-state-only outputs, idle GPU"
  exit 0
fi

log "START cars/proxy_anchor/seed0 final-state authority digest=${RECIPE_DIGEST}"
start=${SECONDS}
if .venv/bin/sfora image-end-to-end \
  --dataset-name cars \
  --objectives proxy_anchor \
  --recipe auto \
  --num-workers "${NUM_WORKERS}" \
  --seed 0 \
  --deterministic \
  --save-model-path "${CHECKPOINT}" \
  --output "${REPORT}" >"${RUN_LOG}" 2>&1; then
  .venv/bin/python - "${REPORT}" "${CHECKPOINT}" "${RECIPE_ID}" "${RECIPE_DIGEST}" <<'PY'
import json
import sys

import torch

report_path, checkpoint_path, recipe_id, recipe_digest = sys.argv[1:]
report = json.loads(open(report_path, encoding="utf-8").read())
config = report["config"]
methods = report["methods"]
assert config["dataset_name"] == "cars"
assert config["objectives"] == ["proxy_anchor"]
assert config["recipe_id"] == recipe_id
assert config["recipe_digest"] == recipe_digest
assert config["seed"] == 0 and type(config["seed"]) is int
assert config["deterministic"] is True
assert len(methods) == 1
method = next(iter(methods.values()))
assert method["objective"] == "proxy_anchor"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
assert checkpoint["artifact_selection"] == "final_training_state"
assert checkpoint["evaluation_model_source"] == "student"
assert checkpoint["training_config"] == config
assert checkpoint["training_step"] == method["executed_train_steps"]
PY
  .venv/bin/python scripts/export_final_cars_embeddings.py \
    --checkpoint "${CHECKPOINT}" \
    --report "${REPORT}" \
    --split train \
    --output "${TRAIN_PACK}" \
    --expected-recipe-id "${RECIPE_ID}" \
    --expected-recipe-digest "${RECIPE_DIGEST}" \
    --num-workers "${NUM_WORKERS}" >>"${RUN_LOG}" 2>&1
  .venv/bin/python scripts/diagnose_cars_method_frontier.py \
    --train-head "${TRAIN_PACK}" \
    --output "${FRONTIER_RESULT}" >>"${RUN_LOG}" 2>&1
  log "DONE cars/proxy_anchor/seed0 elapsed=$((SECONDS - start))s report=${REPORT} checkpoint=${CHECKPOINT} train_pack=${TRAIN_PACK} frontier=${FRONTIER_RESULT}"
else
  log "FAIL cars/proxy_anchor/seed0 elapsed=$((SECONDS - start))s log=${RUN_LOG}"
  exit 1
fi
