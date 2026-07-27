#!/usr/bin/env bash
# Priority-ordered experiment queue, v2.
#
# NEVER overwrite this file while it is running. bash reads a script incrementally
# by byte offset, so replacing it mid-run can drop the interpreter into the middle
# of a statement. Ship a new versioned filename instead; that is why this is v2.
#
# Changes from v1, both driven by unit-level proofs added in
# tests/test_image_end_to_end.py:
#
#   * DROPPED the CUB `_bnfix` arms. With `freeze_batch_norm=True` both teacher
#     modes force backbone BatchNorm to eval, so the fix is provably inert -- and
#     `test_teacher_normalisation_fix_is_inert_when_batch_norm_is_frozen` asserts
#     exactly that in milliseconds. Spending ~2.5 GPU-hours to re-measure a proved
#     identity is waste; the freed time goes to the arm below.
#
#   * ADDED In-Shop `herd_bnfix` (3 seeds), ahead of `pa_distill_bnfix`. The
#     HIST leg is the statistically robust one (-1.39 pt, paired t=-33.9,
#     p~0.0009) whereas the PA leg is marginal (-0.41 pt, p~0.042). Testing the
#     fix where the effect is solid is far more decisive than where it is
#     borderline. Baselines for both are already owned, so each run buys a
#     complete paired comparison.
#
# H3 predicts these arms recover most of the regression. The opposing prediction
# (inert under frozen BatchNorm) is now discharged by test, not by GPU.
#
# Usage:  nohup bash scripts/run_priority_queue_v2.sh > /dev/null 2>&1 &
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"
REPORT_ROOT="${PROJECT}/reports/generated"
STATUS_FILE="${LOG_ROOT}/status.tsv"
CONTROLLER_LOG="${LOG_ROOT}/controller.log"
NUM_WORKERS="${NUM_WORKERS:-8}"
INSHOP_ROOT="${INSHOP_ROOT:-/home/riomus/datasets/inshop}"
HIST_INSHOP_MANIFEST="reports/generated/recipe_selection.inshop.hist.json"

mkdir -p "${LOG_ROOT}" "${REPORT_ROOT}"
cd "${PROJECT}" || exit 2

log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${CONTROLLER_LOG}"; }

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'timestamp\tdataset\tmethod\tseed\tstatus\tbest_r1\tartifact\n' > "${STATUS_FILE}"
fi

recipe_metadata() {
  .venv/bin/python - "$1" "$2" "$3" "${4:-}" <<'PY'
import sys
from pathlib import Path
from typing import cast
from sfora.data import ImageDatasetName
from sfora.image_recipes import BaseMethod, recipe_digest, resolve_recipe

selector, method, dataset, manifest = sys.argv[1:5]
recipe = resolve_recipe(
    selector,
    base_method=cast(BaseMethod, method),
    dataset=cast(ImageDatasetName, dataset),
    selection_manifest=Path(manifest) if manifest else None,
)
print(recipe.recipe_id)
print(recipe_digest(recipe))
print(recipe.track)
PY
}

artifact_matches() {
  .venv/bin/python - "$1" "$2" "$3" <<'PY'
import json, sys
from pathlib import Path

path, expected_digest, expected_seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
if not path.is_file():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config"]
    objectives = set(config["objectives"])
    completed = {m["objective"] for m in payload["methods"].values() if "objective" in m}
except (KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
matches = (
    config.get("recipe_digest") == expected_digest
    and config.get("seed") == expected_seed
    and objectives <= completed
)
raise SystemExit(0 if matches else 1)
PY
}

best_r1() {
  .venv/bin/python - "$1" <<'PY' 2>/dev/null || echo "-"
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
vals = [m.get("best_test_recall_at_1") for m in d["methods"].values()]
vals = [v for v in vals if isinstance(v, (int, float))]
print(f"{max(vals):.4f}" if vals else "-")
PY
}

record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$1" "$2" "$3" "$4" "$5" "$6" >> "${STATUS_FILE}"
}

run_arm() {
  local dataset="$1" method="$2" objective="$3" selector="$4" seed="$5"
  local manifest=""
  # HIST published no In-Shop recipe; its frozen training-only selection lives here.
  if [[ "${dataset}" == "inshop" && "${objective}" == "hist" ]]; then
    manifest="${HIST_INSHOP_MANIFEST}"
  fi

  local metadata=()
  if ! mapfile -t metadata < <(recipe_metadata "${selector}" "${objective}" "${dataset}" "${manifest}"); then
    log "FAILED to resolve ${dataset}/${method}/seed${seed}"
    record "${dataset}" "${method}" "${seed}" "resolve_failed" "-" "-"
    return 1
  fi
  local recipe_id="${metadata[0]}" digest="${metadata[1]}" track="${metadata[2]}"
  local slug="${recipe_id//[^a-zA-Z0-9_.-]/_}"
  local artifact="${REPORT_ROOT}/image_end_to_end_${dataset}.${method}.${slug}.${digest:0:12}_seed${seed}.json"
  local run_log="${LOG_ROOT}/${dataset}.${method}.seed${seed}.log"

  if artifact_matches "${artifact}" "${digest}" "${seed}"; then
    log "SKIP cached ${dataset}/${method}/seed${seed} -> $(best_r1 "${artifact}")"
    record "${dataset}" "${method}" "${seed}" "cached" "$(best_r1 "${artifact}")" "${artifact}"
    return 0
  fi

  log "START ${dataset}/${method}/seed${seed} recipe=${recipe_id} digest=${digest:0:12} track=${track}"
  record "${dataset}" "${method}" "${seed}" "running" "-" "${artifact}"
  local start=${SECONDS}
  local command=(
    .venv/bin/sfora image-end-to-end
    --dataset-name "${dataset}"
    --objectives "${objective}"
    --recipe "${selector}"
    --num-workers "${NUM_WORKERS}"
    --seed "${seed}"
    --output "${artifact}"
  )
  if [[ "${dataset}" == "inshop" ]]; then
    command+=(--dataset-root "${INSHOP_ROOT}")
  fi
  if [[ -n "${manifest}" ]]; then
    command+=(--recipe-selection-manifest "${manifest}")
  fi

  if "${command[@]}" > "${run_log}" 2>&1; then
    local r1; r1="$(best_r1 "${artifact}")"
    log "DONE  ${dataset}/${method}/seed${seed} best_r1=${r1} elapsed=$(( SECONDS - start ))s"
    record "${dataset}" "${method}" "${seed}" "complete" "${r1}" "${artifact}"
  else
    log "FAIL  ${dataset}/${method}/seed${seed} elapsed=$(( SECONDS - start ))s (see ${run_log})"
    record "${dataset}" "${method}" "${seed}" "failed" "-" "${artifact}"
    return 1
  fi
}

# --- wait for any in-flight training to drain -------------------------------
# Ask the GPU what is running rather than pattern-matching command lines. v1 used
# `pgrep -f "sfora image-end-to-end"`, which matched the interactive shell that
# LAUNCHED it (that string appeared in a ps|grep in the launcher's own command
# line), deadlocking the queue with the GPU idle.
gpu_busy() {
  local pids
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"
  [[ -n "${pids//[[:space:]]/}" ]]
}

WAIT_DEADLINE=$(( SECONDS + 6 * 3600 ))
while gpu_busy; do
  if (( SECONDS > WAIT_DEADLINE )); then
    log "WARNING: GPU still busy after 6h; starting anyway rather than idling."
    break
  fi
  log "v2 waiting for in-flight run to finish (GPU busy)..."
  sleep 120
done

log "priority queue v2 start"

# 1. Finish the CUB seed-0 paired reading (pa / pa_distill land as cached).
run_arm cub  proxy_anchor proxy_anchor auto       0
run_arm cub  pa_distill   proxy_anchor pa_distill 0
run_arm cub  hist         hist         auto       0
run_arm cub  herd         hist         herd       0

# 2. H3's decisive test, on the ROBUST leg first.
#    Baseline HERD already owned: 0.8906 / 0.8892 / 0.8900 (mean 0.8899).
for seed in 0 1 2; do
  run_arm inshop herd_bnfix hist herd_bnfix "${seed}"
done

# 3. H3 on the marginal leg. Baseline PA owned: 0.9024 / 0.9048 / 0.9032.
for seed in 0 1 2; do
  run_arm inshop pa_distill_bnfix proxy_anchor pa_distill_bnfix "${seed}"
done

# 4. Independent replication of (1) on Cars.
run_arm cars proxy_anchor proxy_anchor auto       0
run_arm cars pa_distill   proxy_anchor pa_distill 0
run_arm cars hist         hist         auto       0
run_arm cars herd         hist         herd       0

# 5. Remaining seeds for the screening matrix.
for seed in 1 2; do
  for dataset in cub cars; do
    run_arm "${dataset}" proxy_anchor proxy_anchor auto       "${seed}"
    run_arm "${dataset}" pa_distill   proxy_anchor pa_distill "${seed}"
    run_arm "${dataset}" hist         hist         auto       "${seed}"
    run_arm "${dataset}" herd         hist         herd       "${seed}"
  done
done

log "priority queue v2 complete"
