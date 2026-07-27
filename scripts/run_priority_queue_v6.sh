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
# Usage:  nohup bash scripts/run_priority_queue_v6.sh > /dev/null 2>&1 &
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
  log "v6 waiting for in-flight run to finish (GPU busy)..."
  sleep 120
done

log "priority queue v6 start"

# 1. Finish the CUB seed-0 paired reading (earlier arms land as cached).
run_arm cub  proxy_anchor proxy_anchor auto       0
run_arm cub  pa_distill   proxy_anchor pa_distill 0
run_arm cub  hist         hist         auto       0
run_arm cub  herd         hist         herd       0

# 2. THE NOVELTY SCREEN. `herd_hg` distils the teacher's PROPAGATED HGNN logits -
#    an n-ary target normalised by the batch's hyperedge population, which no
#    pairwise method (RKD/S2SD/STML) can express. `herd_hg_incidence` is the
#    ablation control: its target is a per-sample prototype affinity, invariant to
#    the rest of the batch, so it isolates whether any gain comes from the
#    propagation operator or merely from the Gaussian affinity.
#    Both compare against cub/hist/seed0 = 0.7183 under the same official recipe.
run_arm cub  herd_hg           hist herd_hg           0
run_arm cub  herd_hg_incidence hist herd_hg_incidence 0

# 2b. SHOT - the physics/biology-inspired structural change. Compares against
#     cub/hist/seed0 = 0.7183 under the same official recipe, and plain HIST is a
#     PROVABLE special case (iterations=0, epsilon=1) so this is a one-knob test.
#     `hist_shot_uniform` is the SwAV-style equipartition ablation.
# Ordered by how much each can actually move: the geometric coupling is the live
# variant (the HIST-compatible one is near-inert once classes separate), and IPC=4
# adds structure rather than regularisation, which is what the curves say HIST needs.
run_arm cub  hist_ipc4           hist hist_ipc4           0
run_arm cub  hist_shot_geometric hist hist_shot_geometric 0
run_arm cub  hist_shot_geo_ipc4  hist hist_shot_geo_ipc4  0
run_arm cub  hist_shot           hist hist_shot           0
run_arm cub  hist_shot_uniform   hist hist_shot_uniform   0

# 2c. Escalate ONLY the arms that cleared the gate against plain HIST on seed 0.
HIST_CUB_BASE="$(arm_best cub hist hist auto 0)"
log "CUB HIST seed-0 baseline = ${HIST_CUB_BASE}"
escalate_if_promising cub hist_ipc4           hist hist_ipc4           "${HIST_CUB_BASE}"
escalate_if_promising cub hist_shot_geo_ipc4  hist hist_shot_geo_ipc4  "${HIST_CUB_BASE}"
escalate_if_promising cub hist_shot_geometric hist hist_shot_geometric "${HIST_CUB_BASE}"
escalate_if_promising cub hist_shot           hist hist_shot           "${HIST_CUB_BASE}"
escalate_if_promising cub hist_shot_uniform   hist hist_shot_uniform   "${HIST_CUB_BASE}"

# 3. H3's decisive test, robust leg first.
#    Baseline HERD owned: 0.8906 / 0.8892 / 0.8900.
for seed in 0 1 2; do
  run_arm inshop herd_bnfix hist herd_bnfix "${seed}"
done

# 4. H3 on the marginal leg. Baseline PA owned: 0.9024 / 0.9048 / 0.9032.
for seed in 0 1 2; do
  run_arm inshop pa_distill_bnfix proxy_anchor pa_distill_bnfix "${seed}"
done

# 5. Independent replication on Cars.
run_arm cars proxy_anchor proxy_anchor auto       0
run_arm cars pa_distill   proxy_anchor pa_distill 0
run_arm cars hist         hist         auto       0
run_arm cars herd         hist         herd       0

# 6. Remaining seeds for the screening matrix, plus the hypergraph arms if they
#    cleared the gate on seed 0 (re-run this script after editing to extend).
for seed in 1 2; do
  for dataset in cub cars; do
    run_arm "${dataset}" proxy_anchor proxy_anchor auto       "${seed}"
    run_arm "${dataset}" pa_distill   proxy_anchor pa_distill "${seed}"
    run_arm "${dataset}" hist         hist         auto       "${seed}"
    run_arm "${dataset}" herd         hist         herd       "${seed}"
  done
done

log "priority queue v6 complete"
