#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"
INSHOP_ROOT="${INSHOP_ROOT:-/home/riomus/datasets/inshop}"
SEED0_REPORT="${SEED0_REPORT:-reports/generated/rspg_inshop_epoch10_operating_export.json}"
CARS_PATTERN="image_end_to_end.*cars.*recall_at_k_surrogate"

cd "${PROJECT}"
mkdir -p "${LOG_ROOT}" reports/emb reports/generated reports/checkpoints
[[ -f "${INSHOP_ROOT}/Eval/list_eval_partition.txt" && -d "${INSHOP_ROOT}/Img" ]] || {
  echo "refusing missing/invalid In-Shop root: ${INSHOP_ROOT}" >&2
  exit 2
}
[[ -s "${SEED0_REPORT}" ]] || {
  echo "refusing missing hypothesis-generating seed-0 report: ${SEED0_REPORT}" >&2
  exit 2
}

verify_report() {
  local report="$1"
  local expected_seed="$2"
  .venv/bin/python - "${report}" "${expected_seed}" "${SEED0_REPORT}" <<'PY'
import json
import sys

report_path, expected_seed, seed0_report = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(report_path, encoding="utf-8") as handle:
    config = json.load(handle)["config"]
with open(seed0_report, encoding="utf-8") as handle:
    seed0 = json.load(handle)["config"]
expected = {
    "dataset_name": "inshop",
    "objectives": ["proxy_anchor"],
    "recipe_id": "proxy_anchor.inshop.official-51db570",
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

# Modified-recipe digests include runtime output paths and the config schema, so
# seed-0's 6bb9... digest, the later checkpoint's 1ae1..., and a current rerun's
# digest need not match despite identical training behavior. Compare the locked
# behavior-bearing fields directly instead of confusing provenance with behavior.
behavior_fields = [
    "objectives", "backbone_name", "pretrained_weights", "input_size",
    "embedding_dimensions", "embedding_head_init", "embedding_layer_norm",
    "head_pooling", "optimizer", "batch_size", "samples_per_class", "group_size",
    "learning_rate", "backbone_learning_rate", "proxy_learning_rate_multiplier",
    "weight_decay", "warmup_epochs", "warmup_is_additional",
    "schedule_during_warmup", "lr_schedule", "lr_step_epochs", "lr_gamma",
    "train_augmentation", "freeze_batch_norm", "freeze_batch_norm_affine",
    "proxy_count_per_class", "proxy_anchor_alpha", "proxy_anchor_delta",
    "gradient_clip_value", "train_epochs", "train_steps", "deterministic",
]
for key in behavior_fields:
    if config.get(key) != seed0.get(key):
        raise SystemExit(
            f"refusing behavior mismatch against seed 0: {key}={config.get(key)!r}, "
            f"seed0={seed0.get(key)!r}"
        )
if config.get("recipe_source_revision") != seed0.get("recipe_source_revision"):
    raise SystemExit("refusing source-revision mismatch against seed 0")
if not config.get("recipe_digest"):
    raise SystemExit("refusing report without a recipe digest")
print(
    f"accepted behavior-equivalent seed {expected_seed}: "
    f"seed0_digest={seed0.get('recipe_digest')} rerun_digest={config.get('recipe_digest')}"
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
  .venv/bin/python scripts/measure_fragmentation_confounding.py "${pack}" \
    --output "reports/generated/inshop_fragmentation_confounding_epoch10_seed${seed}.json"
  sha256sum "${pack}" "${report}" "${checkpoint}" "${SEED0_REPORT}" \
    scripts/measure_spectral_class_connectivity.py \
    scripts/measure_fragmentation_confounding.py \
    src/sfora/image_end_to_end.py src/sfora/image_recipes.py \
    | tee "reports/generated/inshop_fragmentation_epoch10_seed${seed}.sha256"
done
