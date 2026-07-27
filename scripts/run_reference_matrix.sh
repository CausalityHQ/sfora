#!/usr/bin/env bash
# Corrected reference-recipe matrix for CUB-200 and Cars196.
#
# Runs {proxy_anchor, pa_distill, hist, herd} x {cub, cars} x seeds, using ONLY
# publication-backed `reference`-track recipes.  This is the paired comparison the
# headline claims have never had: within each base, the plain and distilled arms
# differ in `ema_distill_weight` and nothing else (`derive_recipe`), so the legacy
# LayerNorm confound cannot recur.
#
# Runs ON the DGX.  Seed-major ordering means an interrupted matrix still yields
# complete paired coverage for the seeds that finished, rather than one dataset only.
#
# Usage:  nohup bash scripts/run_reference_matrix.sh > /dev/null 2>&1 &
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"
REPORT_ROOT="${PROJECT}/reports/generated"
STATUS_FILE="${LOG_ROOT}/status.tsv"
CONTROLLER_LOG="${LOG_ROOT}/controller.log"
SEEDS="${SEEDS:-0 1 2}"
DATASETS="${DATASETS:-cub cars}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-22f7dd61f26035680627cfcc211295b10686abb4}"

mkdir -p "${LOG_ROOT}" "${REPORT_ROOT}"
cd "${PROJECT}" || exit 2

log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${CONTROLLER_LOG}"; }

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'timestamp\tdataset\tmethod\tseed\tstatus\tbest_r1\tartifact\n' > "${STATUS_FILE}"
fi

# ---------------------------------------------------------------------------
# Provenance: the remote checkout has a stale git HEAD, so pin the code by
# content hash of the source tree and record the commit it matches.
# ---------------------------------------------------------------------------
SRC_HASH="$(.venv/bin/python -c '
import hashlib, pathlib
h = hashlib.sha256()
for p in sorted(pathlib.Path("src/sfora").rglob("*.py")):
    h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"  " + str(p).encode() + b"\n")
print(h.hexdigest())
')"
cat > "${LOG_ROOT}/provenance.json" <<EOF
{
  "expected_commit": "${EXPECTED_COMMIT}",
  "src_tree_sha256": "${SRC_HASH}",
  "started": "$(date --iso-8601=seconds)",
  "host": "$(hostname)"
}
EOF
log "provenance src_tree_sha256=${SRC_HASH} expected_commit=${EXPECTED_COMMIT}"

# ---------------------------------------------------------------------------
recipe_metadata() {
  # $1 selector, $2 base method, $3 dataset -> recipe_id, digest, track
  .venv/bin/python - "$1" "$2" "$3" <<'PY'
import sys
from typing import cast
from sfora.data import ImageDatasetName
from sfora.image_recipes import BaseMethod, recipe_digest, resolve_recipe

selector, method, dataset = sys.argv[1:4]
recipe = resolve_recipe(
    selector,
    base_method=cast(BaseMethod, method),
    dataset=cast(ImageDatasetName, dataset),
)
print(recipe.recipe_id)
print(recipe_digest(recipe))
print(recipe.track)
PY
}

artifact_matches() {
  # $1 path, $2 digest, $3 seed
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
  local metadata=()
  if ! mapfile -t metadata < <(recipe_metadata "${selector}" "${objective}" "${dataset}"); then
    log "FAILED to resolve recipe ${dataset}/${method}"
    record "${dataset}" "${method}" "${seed}" "resolve_failed" "-" "-"
    return 1
  fi
  local recipe_id="${metadata[0]}" digest="${metadata[1]}" track="${metadata[2]}"
  if [[ "${track}" != "reference" ]]; then
    log "REFUSING non-reference track '${track}' for ${dataset}/${method}"
    record "${dataset}" "${method}" "${seed}" "non_reference_track" "-" "-"
    return 1
  fi
  local slug="${recipe_id//[^a-zA-Z0-9_.-]/_}"
  local artifact="${REPORT_ROOT}/image_end_to_end_${dataset}.${method}.${slug}.${digest:0:12}_seed${seed}.json"
  local run_log="${LOG_ROOT}/${dataset}.${method}.seed${seed}.log"

  if artifact_matches "${artifact}" "${digest}" "${seed}"; then
    log "SKIP digest-matched ${dataset}/${method}/seed${seed} -> $(best_r1 "${artifact}")"
    record "${dataset}" "${method}" "${seed}" "cached" "$(best_r1 "${artifact}")" "${artifact}"
    return 0
  fi

  log "START ${dataset}/${method}/seed${seed} recipe=${recipe_id} digest=${digest:0:12}"
  record "${dataset}" "${method}" "${seed}" "running" "-" "${artifact}"
  local start=${SECONDS}
  if .venv/bin/sfora image-end-to-end \
      --dataset-name "${dataset}" \
      --objectives "${objective}" \
      --recipe "${selector}" \
      --num-workers "${NUM_WORKERS}" \
      --seed "${seed}" \
      --output "${artifact}" > "${run_log}" 2>&1; then
    local r1; r1="$(best_r1 "${artifact}")"
    log "DONE  ${dataset}/${method}/seed${seed} best_r1=${r1} elapsed=$(( SECONDS - start ))s"
    record "${dataset}" "${method}" "${seed}" "complete" "${r1}" "${artifact}"
  else
    log "FAIL  ${dataset}/${method}/seed${seed} elapsed=$(( SECONDS - start ))s (see ${run_log})"
    record "${dataset}" "${method}" "${seed}" "failed" "-" "${artifact}"
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Seed-major: a full paired 8-arm picture lands after each seed.
# ---------------------------------------------------------------------------
log "matrix start seeds='${SEEDS}' datasets='${DATASETS}'"
for seed in ${SEEDS}; do
  for dataset in ${DATASETS}; do
    run_arm "${dataset}" "proxy_anchor" "proxy_anchor" "auto"       "${seed}"
    run_arm "${dataset}" "pa_distill"   "proxy_anchor" "pa_distill" "${seed}"
    run_arm "${dataset}" "hist"         "hist"         "auto"       "${seed}"
    run_arm "${dataset}" "herd"         "hist"         "herd"       "${seed}"
  done
  log "seed ${seed} complete"
done
log "matrix complete"
