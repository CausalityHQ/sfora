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
# Usage:  nohup bash scripts/run_priority_queue_v21.sh > /dev/null 2>&1 &
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
print(",".join(recipe.config["objectives"]))
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
  # Use the RECIPE's objective, never the caller's guess. Passing --objectives
  # explicitly silently overrode the recipe, so every new OBJECTIVE
  # (local_nca / hist_sinkhorn / hist_memory) ran as plain `hist` instead.
  local recipe_objective="${metadata[3]:-${objective}}"
  local slug="${recipe_id//[^a-zA-Z0-9_.-]/_}"
  local artifact="${REPORT_ROOT}/image_end_to_end_${dataset}.${method}.${slug}.${digest:0:12}_seed${seed}.json"
  local run_log="${LOG_ROOT}/${dataset}.${method}.seed${seed}.log"

  if artifact_matches "${artifact}" "${digest}" "${seed}"; then
    log "SKIP cached ${dataset}/${method}/seed${seed} -> $(best_r1 "${artifact}")"
    record "${dataset}" "${method}" "${seed}" "cached" "$(best_r1 "${artifact}")" "${artifact}"
    return 0
  fi

  log "START ${dataset}/${method}/seed${seed} recipe=${recipe_id} digest=${digest:0:12} track=${track} objective=${recipe_objective}"
  record "${dataset}" "${method}" "${seed}" "running" "-" "${artifact}"
  local start=${SECONDS}
  local command=(
    .venv/bin/sfora image-end-to-end
    --dataset-name "${dataset}"
    --objectives "${recipe_objective}"
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

# --- adaptive escalation -----------------------------------------------------
# Read an arm's best R@1 back out of its artifact so the queue can decide, without
# a human, whether an arm earned more seeds. Screening at one seed then escalating
# only winners is how the limited overnight GPU buys the most information: a losing
# arm costs 1 run instead of 3, and a winner gets its multi-seed evidence tonight
# rather than tomorrow.
arm_best() {
  local dataset="$1" method="$2" objective="$3" selector="$4" seed="$5"
  local metadata=()
  mapfile -t metadata < <(recipe_metadata "${selector}" "${objective}" "${dataset}" "") || return 1
  local recipe_id="${metadata[0]}" digest="${metadata[1]}"
  local slug="${recipe_id//[^a-zA-Z0-9_.-]/_}"
  best_r1 "${REPORT_ROOT}/image_end_to_end_${dataset}.${method}.${slug}.${digest:0:12}_seed${seed}.json"
}

# Escalate to seeds 1 and 2 only if seed 0 beat the reference baseline by >= GATE.
GATE_POINTS="${GATE_POINTS:-0.5}"
escalate_if_promising() {
  local dataset="$1" method="$2" objective="$3" selector="$4" baseline="$5"
  local candidate; candidate="$(arm_best "${dataset}" "${method}" "${objective}" "${selector}" 0)"
  if [[ "${candidate}" == "-" || -z "${candidate}" || "${baseline}" == "-" ]]; then
    log "ESCALATE skip ${dataset}/${method}: no seed-0 result yet"
    return 0
  fi
  local delta
  delta="$(.venv/bin/python -c "print(f'{(${candidate} - ${baseline}) * 100:.3f}')")"
  if .venv/bin/python -c "import sys; sys.exit(0 if (${candidate} - ${baseline}) * 100 >= ${GATE_POINTS} else 1)"; then
    log "ESCALATE ${dataset}/${method}: seed0 ${candidate} vs base ${baseline} = ${delta} pt >= ${GATE_POINTS}; running seeds 1,2"
    run_arm "${dataset}" "${method}" "${objective}" "${selector}" 1
    run_arm "${dataset}" "${method}" "${objective}" "${selector}" 2
  else
    log "ESCALATE no  ${dataset}/${method}: seed0 ${candidate} vs base ${baseline} = ${delta} pt < ${GATE_POINTS}; not spending more seeds"
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
  log "v21 waiting for in-flight run to finish (GPU busy)..."
  sleep 120
done

log "priority queue v21 start"

# ---------------------------------------------------------------------------
# v19/v20 outcome, both legs at six paired seeds:
#
#   pa_distill - proxy_anchor  +0.91 +1.03 +0.73 +0.32 +0.86 +0.10
#                              mean +0.658  sd 0.367  6/6 positive  sign p=0.031
#   herd       - hist          -0.27 +0.34 -0.15 -0.39 +1.50 +0.76
#                              mean +0.298  sd 0.729  3/6 positive  sign p=1.000
#
# The PA effect is REAL (assumption-free p=0.031, as pre-registered). The headroom
# explanation is NOT established: the between-leg gap is +0.360 pt with SE 0.333
# (t=1.08), and resolving a gap that size needs ~40 seeds per arm. Worse, the cut of
# the data free of selection effects points the wrong way - out-of-sample, HIST gained
# +0.623 against PA's +0.427.
#
# The dissociation failed structurally: it tried to resolve a 0.36 pt difference
# between arms whose own noise is 0.37-0.73 pt. The lever was smaller than the noise,
# and more seeds only chase a 40-seed requirement.
#
# THIS QUEUE FIXES THE LEVER. Narrowing the embedding sets headroom by construction
# rather than measuring it, and multiplies the effect instead of differencing two
# similar ones. If the gain scales with headroom, a 64-d base several points below the
# ceiling should show roughly +1.3 pt against a per-seed sd of ~0.37 - resolvable at
# three seeds rather than forty. It also sidesteps the artifact that Delta = distilled
# - base is negatively correlated with base BY CONSTRUCTION, since width is fixed in
# the recipe and independent of any measurement.
#
# Pre-registered in docs/headroom_hypothesis.md section 5: monotonic Delta as width
# falls, ratio Delta/headroom within 0.2-0.6 against the 512-d value, refuted if it
# varies by more than 3x or if a weakened base gains the same as the full one.
#
# Seed-major order, so one seed of every width lands before any width gets a second -
# an early look then covers the whole curve instead of one corner of it.
# ---------------------------------------------------------------------------

for seed in 0 1 2; do
  run_arm cub narrow128         proxy_anchor narrow128         "${seed}"
  run_arm cub narrow128_distill proxy_anchor narrow128_distill "${seed}"
  run_arm cub narrow64          proxy_anchor narrow64          "${seed}"
  run_arm cub narrow64_distill  proxy_anchor narrow64_distill  "${seed}"
done

log "priority queue v21 complete"
