#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"
INSHOP_ROOT="${INSHOP_ROOT:-/home/riomus/datasets/inshop}"
CARS_PATTERN="image_end_to_end.*cars.*recall_at_k_surrogate"

cd "${PROJECT}"
mkdir -p "${LOG_ROOT}" reports/emb reports/generated reports/checkpoints
[[ -f "${INSHOP_ROOT}/Eval/list_eval_partition.txt" && -d "${INSHOP_ROOT}/Img" ]] || {
  echo "refusing missing/invalid In-Shop root: ${INSHOP_ROOT}" >&2
  exit 2
}

EXPECTED_RECIPE_DIGEST="$(${PROJECT}/.venv/bin/python - <<'PY'
from sfora.image_recipes import recipe_digest, reference_recipe
print(recipe_digest(reference_recipe("proxy_anchor", "inshop")))
PY
)"

verify_report() {
  local report="$1"
  local expected_seed="$2"
  .venv/bin/python - "${report}" "${expected_seed}" "${EXPECTED_RECIPE_DIGEST}" <<'PY'
import json
import sys

report_path, expected_seed, expected_digest = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(report_path, encoding="utf-8") as handle:
    config = json.load(handle)["config"]
expected = {
    "dataset_name": "inshop",
    "objectives": ["proxy_anchor"],
    "recipe_id": "proxy_anchor.inshop.official-51db570",
    "recipe_digest": expected_digest,
    "train_epochs": 10,
    "train_steps": 1440,
    "eval_test_interval_epochs": 0,
    "seed": expected_seed,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"refusing mismatched fragmentation artifact: {key}="
            f"{config.get(key)!r}, expected {value!r}"
        )
PY
}

while pgrep -f "${CARS_PATTERN}" >/dev/null; do
  sleep 60
done

for seed in 1 2; do
  pack="reports/emb/inshop_pa_epoch10_operating_seed${seed}.train.npz"
  report="reports/generated/inshop_pa_epoch10_operating_seed${seed}.json"
  checkpoint="reports/checkpoints/inshop_pa_epoch10_operating_seed${seed}.pt"
  log="${LOG_ROOT}/inshop.pa.epoch10.seed${seed}.log"
  existing=0
  [[ -e "${pack}" ]] && existing=$((existing + 1))
  [[ -e "${report}" ]] && existing=$((existing + 1))
  [[ -e "${checkpoint}" ]] && existing=$((existing + 1))
  if [[ "${existing}" -ne 0 && "${existing}" -ne 3 ]]; then
    echo "refusing partial seed-${seed} artifact set" >&2
    exit 2
  fi
  if [[ "${existing}" -eq 0 ]]; then
    .venv/bin/sfora image-end-to-end \
      --dataset-name inshop \
      --dataset-root "${INSHOP_ROOT}" \
      --objectives proxy_anchor \
      --recipe auto \
      --num-workers 8 \
      --train-epochs 10 \
      --eval-test-interval-epochs 0 \
      --save-train-embeddings "${pack}" \
      --save-model-path "${checkpoint}" \
      --seed "${seed}" \
      --output "${report}" \
      >"${log}" 2>&1
  fi
  verify_report "${report}" "${seed}"
  [[ -s "${pack}" && -s "${checkpoint}" ]] || {
    echo "refusing empty seed-${seed} artifact" >&2
    exit 2
  }
  .venv/bin/python scripts/measure_spectral_class_connectivity.py "${pack}" \
    | tee "reports/generated/inshop_fragmentation_epoch10_seed${seed}.json"
  sha256sum "${pack}" "${report}" "${checkpoint}" \
    scripts/measure_spectral_class_connectivity.py \
    src/sfora/image_end_to_end.py src/sfora/image_recipes.py \
    | tee "reports/generated/inshop_fragmentation_epoch10_seed${seed}.sha256"
done
