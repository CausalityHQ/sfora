#!/usr/bin/env bash
# Six-pair deterministic Cars196 confirmation frozen in
# docs/cars_pa_distillation_confirmation_2026-08-24.md.
#
# Usage:
#   PREFLIGHT_ONLY=1 bash scripts/run_priority_queue_v47.sh
#   nohup bash scripts/run_priority_queue_v47.sh > \
#     /home/riomus/experiment-logs/reference-matrix/cars-pa-distill-v47.nohup.log 2>&1 &
set -euo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"
REPORT_ROOT="${PROJECT}/reports/generated/cars-pa-distill-confirmation-2026-08-24"
CONTROLLER_LOG="${LOG_ROOT}/cars-pa-distill-v47.controller.log"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

BASE_RECIPE_ID="proxy_anchor.cars.official-51db570"
BASE_DIGEST="d55241a64a5afe9ea81be02e74fa13a6fec87e15c66e95918ad10d90337cc02a"
CANDIDATE_RECIPE_ID="proxy_anchor.cars.official-51db570.pa_distill"
CANDIDATE_DIGEST="080a45b8c14460d43b6f5f1d352f10854adb0d6c8d434fc6d2f02f2dbd501b02"

mkdir -p "${LOG_ROOT}" "${REPORT_ROOT}"
cd "${PROJECT}"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${CONTROLLER_LOG}"
}

recipe_metadata() {
  .venv/bin/python - "$1" <<'PY'
import sys
from sfora.image_recipes import recipe_digest, resolve_recipe

selector = sys.argv[1]
recipe = resolve_recipe(selector, base_method="proxy_anchor", dataset="cars")
print(recipe.recipe_id)
print(recipe_digest(recipe))
print(",".join(recipe.config["objectives"]))
PY
}

assert_recipe() {
  local selector="$1" expected_id="$2" expected_digest="$3"
  local metadata=()
  mapfile -t metadata < <(recipe_metadata "${selector}")
  [[ "${metadata[0]}" == "${expected_id}" ]] || {
    log "STOP recipe ID drift for ${selector}: ${metadata[0]}"
    return 2
  }
  [[ "${metadata[1]}" == "${expected_digest}" ]] || {
    log "STOP recipe digest drift for ${selector}: ${metadata[1]}"
    return 2
  }
  [[ "${metadata[2]}" == "proxy_anchor" ]] || {
    log "STOP objective drift for ${selector}: ${metadata[2]}"
    return 2
  }
}

artifact_matches() {
  .venv/bin/python - "$1" "$2" "$3" "$4" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_id = sys.argv[2]
expected_digest = sys.argv[3]
expected_seed = int(sys.argv[4])
if not path.is_file() or path.is_symlink():
    raise SystemExit(1)
try:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    config = payload["config"]
    methods = payload["methods"]
    completed = {
        method["objective"]
        for method in methods.values()
        if isinstance(method, dict) and "objective" in method
    }
except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
    raise SystemExit(1)
matches = (
    config.get("recipe_id") == expected_id
    and config.get("recipe_digest") == expected_digest
    and type(config.get("seed")) is int
    and config["seed"] == expected_seed
    and config.get("deterministic") is True
    and config.get("objectives") == ["proxy_anchor"]
    and completed == {"proxy_anchor"}
)
raise SystemExit(0 if matches else 1)
PY
}

gpu_busy() {
  local pids
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  [[ -n "${pids//[[:space:]]/}" ]]
}

assert_recipe "auto" "${BASE_RECIPE_ID}" "${BASE_DIGEST}"
assert_recipe "pa_distill" "${CANDIDATE_RECIPE_ID}" "${CANDIDATE_DIGEST}"
if gpu_busy; then
  log "STOP GPU already has a compute process"
  exit 2
fi
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  log "PREFLIGHT PASS exact Cars recipes, deterministic CLI, and idle GPU"
  exit 0
fi

run_confirmation_arm() {
  local arm="$1" selector="$2" seed="$3"
  local expected_id expected_digest
  case "${arm}" in
    proxy_anchor)
      expected_id="${BASE_RECIPE_ID}"
      expected_digest="${BASE_DIGEST}"
      ;;
    pa_distill)
      expected_id="${CANDIDATE_RECIPE_ID}"
      expected_digest="${CANDIDATE_DIGEST}"
      ;;
    *)
      log "STOP unregistered arm ${arm}"
      return 2
      ;;
  esac

  assert_recipe "${selector}" "${expected_id}" "${expected_digest}"
  local artifact="${REPORT_ROOT}/${arm}.${expected_digest:0:12}.deterministic.seed${seed}.json"
  local run_log="${LOG_ROOT}/cars-pa-distill-v47.${arm}.seed${seed}.log"
  if [[ -e "${artifact}" || -L "${artifact}" ]]; then
    if artifact_matches "${artifact}" "${expected_id}" "${expected_digest}" "${seed}"; then
      log "SKIP exact cached cars/${arm}/seed${seed} artifact=${artifact}"
      return 0
    fi
    log "STOP refusing nonmatching pre-existing artifact ${artifact}"
    return 2
  fi

  log "START cars/${arm}/seed${seed} recipe=${expected_id} digest=${expected_digest} deterministic=true"
  local start=${SECONDS}
  if .venv/bin/sfora image-end-to-end \
    --dataset-name cars \
    --objectives proxy_anchor \
    --recipe "${selector}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${seed}" \
    --deterministic \
    --output "${artifact}" > "${run_log}" 2>&1; then
    artifact_matches "${artifact}" "${expected_id}" "${expected_digest}" "${seed}" || {
      log "STOP completed artifact failed exact validation ${artifact}"
      return 2
    }
    log "DONE cars/${arm}/seed${seed} elapsed=$(( SECONDS - start ))s artifact=${artifact}"
  else
    log "FAIL cars/${arm}/seed${seed} elapsed=$(( SECONDS - start ))s log=${run_log}"
    return 1
  fi
}

log "priority queue v47 start: six deterministic Cars196 pairs"
for seed in 0 1 2 3 4 5; do
  run_confirmation_arm "proxy_anchor" "auto" "${seed}"
  run_confirmation_arm "pa_distill" "pa_distill" "${seed}"
done

.venv/bin/python - "${REPORT_ROOT}" "${BASE_DIGEST}" "${CANDIDATE_DIGEST}" <<'PY' \
  | tee -a "${CONTROLLER_LOG}"
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

root = Path(sys.argv[1])
digests = {"proxy_anchor": sys.argv[2], "pa_distill": sys.argv[3]}
values: dict[tuple[str, int], tuple[float, float]] = {}
for arm, digest in digests.items():
    for seed in range(6):
        path = root / f"{arm}.{digest[:12]}.deterministic.seed{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        method_rows = [
            (float(row["best_test_recall_at_1"]), float(row["recall_at_1"]))
            for row in payload["methods"].values()
            if isinstance(row.get("best_test_recall_at_1"), (int, float))
            and isinstance(row.get("recall_at_1"), (int, float))
        ]
        if not method_rows:
            raise SystemExit(f"missing scores in {path}")
        values[(arm, seed)] = max(method_rows, key=lambda row: row[0])

best = np.asarray(
    [100.0 * (values[("pa_distill", s)][0] - values[("proxy_anchor", s)][0]) for s in range(6)],
    dtype=np.float64,
)
final = np.asarray(
    [100.0 * (values[("pa_distill", s)][1] - values[("proxy_anchor", s)][1]) for s in range(6)],
    dtype=np.float64,
)
rng = np.random.Generator(np.random.PCG64(196))
replicates = final[rng.integers(0, 6, size=(10_000, 6))].mean(axis=1)
lower, upper = np.percentile(replicates, [2.5, 97.5], method="linear")
positive = int(np.sum(final > 0.0))
tail = min(positive, 6 - positive)
sign_p = min(1.0, 2.0 * sum(math.comb(6, k) for k in range(tail + 1)) / (2**6))
paired_t = stats.ttest_1samp(final, popmean=0.0)
status = (
    "CONFIRMED"
    if float(final.mean()) >= 0.5 and positive == 6 and float(lower) > 0.0
    else "REFUTED"
    if float(final.mean()) <= 0.0 or float(upper) <= 0.0 or positive <= 3
    else "INCONCLUSIVE"
)
summary = {
    "status": status,
    "final_epoch_deltas_points": final.tolist(),
    "final_epoch_mean_points": float(final.mean()),
    "final_epoch_sample_sd_points": float(final.std(ddof=1)),
    "final_epoch_positive_pairs": positive,
    "final_epoch_exact_sign_p_two_sided": sign_p,
    "final_epoch_paired_t": float(paired_t.statistic),
    "final_epoch_paired_t_p_two_sided": float(paired_t.pvalue),
    "final_epoch_bootstrap_95": [float(lower), float(upper)],
    "best_over_training_deltas_points_sensitivity_only": best.tolist(),
    "best_over_training_mean_points_sensitivity_only": float(best.mean()),
    "all_finite": bool(np.isfinite(final).all() and np.isfinite(best).all()),
}
if not summary["all_finite"] or not all(math.isfinite(x) for x in (lower, upper)):
    raise SystemExit("nonfinite confirmation summary")
print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
PY

log "priority queue v47 complete"
