#!/usr/bin/env bash
# Priority-ordered experiment queue, highest information-per-GPU-hour first.
#
# Ordering rationale:
#   1. CUB seed 0, all four arms - the paired reading on the headline dataset, and
#      the arm the in-flight run already belongs to.
#   2. CUB seed 0 `herd_bnfix` - H3's *null* prediction. CUB freezes backbone
#      BatchNorm, so teacher and student already normalise identically and the fix
#      should change essentially nothing here.
#   3. In-Shop `pa_distill_bnfix`, 3 seeds - H3's *positive* prediction, and the
#      decisive test. In-Shop trains BatchNorm, so the historical eval-mode teacher
#      normalises differently from the student. The PA baseline is already owned
#      (0.9024/0.9048/0.9032), so 3 runs buy a complete paired comparison.
#   4. Cars seed 0 - independent replication of (1).
#   5. Remaining seeds.
#
# A pair of predictions that point in OPPOSITE directions is what makes this a real
# test rather than a fishing expedition: H3 survives only if the fix is inert on CUB
# AND flips the sign on In-Shop.
#
# Waits for any in-flight `sfora image-end-to-end` process before starting, so it can
# be launched while the previous controller's last run drains.
#
# Usage:  nohup bash scripts/run_priority_queue.sh > /dev/null 2>&1 &
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"
REPORT_ROOT="${PROJECT}/reports/generated"
STATUS_FILE="${LOG_ROOT}/status.tsv"
CONTROLLER_LOG="${LOG_ROOT}/controller.log"
NUM_WORKERS="${NUM_WORKERS:-8}"
INSHOP_ROOT="${INSHOP_ROOT:-/home/riomus/datasets/inshop}"

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
  # HIST has no published In-Shop recipe; its frozen training-only selection lives here.
  if [[ "${dataset}" == "inshop" && "${objective}" == "hist" ]]; then
    manifest="reports/generated/recipe_selection.inshop.hist.json"
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
# Ask the GPU what is running rather than pattern-matching command lines. An
# earlier `pgrep -f "sfora image-end-to-end"` deadlocked this script for good:
# the interactive shell that LAUNCHED it had that exact string in its own command
# line (from a `ps | grep`), so pgrep matched the launcher and the queue waited
# forever with the GPU idle. Process-name matching is unsafe when the name is a
# substring anyone might type; GPU occupancy is the signal we actually mean.
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
  log "waiting for in-flight run to finish (GPU busy)..."
  sleep 120
done

log "priority queue start"

# 1. CUB seed 0 - paired reading on the headline dataset.
run_arm cub  proxy_anchor proxy_anchor auto       0
run_arm cub  pa_distill   proxy_anchor pa_distill 0
run_arm cub  hist         hist         auto       0
run_arm cub  herd         hist         herd       0

# 2. H3 null prediction: frozen BatchNorm => the fix should be inert here.
run_arm cub  herd_bnfix       hist         herd_bnfix       0
run_arm cub  pa_distill_bnfix proxy_anchor pa_distill_bnfix 0

# 3. H3 positive prediction: trainable BatchNorm => the fix should flip the sign.
#    Baseline PA already owned at 0.9024 / 0.9048 / 0.9032.
for seed in 0 1 2; do
  run_arm inshop pa_distill_bnfix proxy_anchor pa_distill_bnfix "${seed}"
done

# 4. Independent replication on Cars.
run_arm cars proxy_anchor proxy_anchor auto       0
run_arm cars pa_distill   proxy_anchor pa_distill 0
run_arm cars hist         hist         auto       0
run_arm cars herd         hist         herd       0
run_arm cars herd_bnfix   hist         herd_bnfix 0

# 5. Remaining seeds for the screening matrix.
for seed in 1 2; do
  for dataset in cub cars; do
    run_arm "${dataset}" proxy_anchor proxy_anchor auto       "${seed}"
    run_arm "${dataset}" pa_distill   proxy_anchor pa_distill "${seed}"
    run_arm "${dataset}" hist         hist         auto       "${seed}"
    run_arm "${dataset}" herd         hist         herd       "${seed}"
  done
done

log "priority queue complete"
