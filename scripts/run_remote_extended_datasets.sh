#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REMOTE="${REMOTE:-riomus@192.168.1.35}"
REMOTE_DIR="${REMOTE_DIR:-/home/riomus/group-learning}"
DATASETS="${DATASETS:-inshop inat2018}"
INSHOP_ROOT="${INSHOP_ROOT:-}"
INAT2018_ROOT="${INAT2018_ROOT:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FORCE_RERUN="${FORCE_RERUN:-0}"
CONTROLLER_LOG="${CONTROLLER_LOG:-logs/reference_recipes.controller.log}"
CONTROLLER_PID_FILE="logs/reference_recipes.controller.pid"
BN_INCEPTION_WEIGHT="${BN_INCEPTION_WEIGHT:-}"
BN_INCEPTION_SHA256="52deb473314542a5c2f87e9e6f26f4ca42fe863d15f986414dbae8c2dfdd2353"
REMOTE_WEIGHT_CACHE="${REMOTE_WEIGHT_CACHE:-/home/riomus/.cache/torch/hub/checkpoints/bn_inception-52deb4733.pth}"

recipe_metadata() {
  local selector="$1"
  local base_method="$2"
  local dataset="$3"
  local manifest="$4"
  .venv/bin/python - "${selector}" "${base_method}" "${dataset}" "${manifest}" <<'PY'
import sys
from pathlib import Path
from typing import cast

from sfora.data import ImageDatasetName
from sfora.image_recipes import BaseMethod, recipe_digest, resolve_recipe

selector, method, dataset, manifest = sys.argv[1:]
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
  local path="$1"
  local digest="$2"
  local seed="$3"
  .venv/bin/python - "${path}" "${digest}" "${seed}" <<'PY'
import json
import sys
from pathlib import Path

path, expected_digest, expected_seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
if not path.is_file():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config"]
    objectives = set(config["objectives"])
    completed = {
        method["objective"] for method in payload["methods"].values() if "objective" in method
    }
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

selection_manifest_matches() {
  local path="$1"
  local method="$2"
  local dataset="$3"
  .venv/bin/python - "${path}" "${method}" "${dataset}" <<'PY'
import sys
from pathlib import Path

from sfora.image_recipes import load_selected_recipe_manifest

path, method, dataset = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if not path.is_file():
    raise SystemExit(1)
try:
    recipe = load_selected_recipe_manifest(path)
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if (recipe.base_method, recipe.dataset) == (method, dataset) else 1)
PY
}

ensure_selection() {
  local method="$1"
  local manifest="reports/generated/recipe_selection.${DATASET}.${method}.json"
  if selection_manifest_matches "${manifest}" "${method}" "${DATASET}"; then
    echo "Using frozen training-only selection ${manifest}."
  else
    .venv/bin/python scripts/select_extended_recipe.py \
      --base-method "${method}" \
      --dataset "${DATASET}" \
      --dataset-root "${DATASET_ROOT}" \
      --num-workers "${NUM_WORKERS}" \
      --seed 0 \
      --output "${manifest}"
  fi
  SELECTION_MANIFEST="${manifest}"
}

run_method() {
  local method="$1"
  local objective="$2"
  local selector="$3"
  local manifest="${4:-}"
  local metadata=()
  mapfile -t metadata < <(recipe_metadata "${selector}" "${objective}" "${DATASET}" "${manifest}")
  RECIPE_ID="${metadata[0]}"
  EXPECTED_DIGEST="${metadata[1]}"
  RECIPE_TRACK="${metadata[2]}"
  RECIPE_SLUG="${RECIPE_ID//[^a-zA-Z0-9_.-]/_}"
  local output_file="reports/generated/image_end_to_end_${DATASET}.${method}.${RECIPE_SLUG}.${EXPECTED_DIGEST:0:12}_seed${SEED}.json"

  echo "Resolved ${DATASET}/${method}/seed${SEED}: ${RECIPE_ID} ${EXPECTED_DIGEST} ${RECIPE_TRACK}"
  if [[ "${FORCE_RERUN}" != "1" ]] && \
    artifact_matches "${output_file}" "${EXPECTED_DIGEST}" "${SEED}"; then
    echo "Skipping digest-matched artifact ${output_file}."
    return
  fi

  local command=(
    .venv/bin/sfora image-end-to-end
    --dataset-name "${DATASET}"
    --dataset-root "${DATASET_ROOT}"
    --objectives "${objective}"
    --recipe "${selector}"
    --num-workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --output "${output_file}"
  )
  if [[ -n "${manifest}" ]]; then
    command+=(--recipe-selection-manifest "${manifest}")
  fi
  "${command[@]}"
}

write_legacy_manifest() {
  .venv/bin/python <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("reports/generated")
paths = sorted(root.glob("image_end_to_end_inshop.*_seed*.json"))
paths += sorted(root.glob("image_end_to_end_inat2018.*_seed*.json"))
legacy = [path for path in paths if ".official-" not in path.name and ".selected-from-" not in path.name]
payload = {
    "classification": "modified_legacy",
    "reason": "global preset overrides predate publication-backed method/dataset recipes",
    "artifacts": [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in legacy
    ],
}
(root / "legacy_extended_recipe_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

controller_main() {
  cd "${LOCAL_DIR}"
  mkdir -p logs reports/generated
  write_legacy_manifest

  for DATASET in ${DATASETS}; do
    if [[ "${DATASET}" == "inshop" ]]; then
      : "${INSHOP_ROOT:?Set INSHOP_ROOT to the official DeepFashion In-Shop root}"
      DATASET_ROOT="${INSHOP_ROOT}"
    elif [[ "${DATASET}" == "inat2018" ]]; then
      : "${INAT2018_ROOT:?Set INAT2018_ROOT to the iNaturalist 2018 root}"
      DATASET_ROOT="${INAT2018_ROOT}"
    else
      echo "Unsupported extended dataset: ${DATASET}" >&2
      exit 2
    fi

    .venv/bin/sfora image-dataset-preflight \
      --dataset-name "${DATASET}" --dataset-root "${DATASET_ROOT}"

    if [[ "${DATASET}" == "inshop" ]]; then
      for SEED in 0 1 2; do
        run_method "proxy_anchor" "proxy_anchor" "auto"
        run_method "pa_distill" "proxy_anchor" "pa_distill"
      done
      ensure_selection "hist"
      HIST_MANIFEST="${SELECTION_MANIFEST}"
      for SEED in 0 1 2; do
        run_method "hist" "hist" "auto" "${HIST_MANIFEST}"
        run_method "herd" "hist" "herd" "${HIST_MANIFEST}"
      done
    else
      ensure_selection "proxy_anchor"
      PA_MANIFEST="${SELECTION_MANIFEST}"
      for SEED in 0 1 2; do
        run_method "proxy_anchor" "proxy_anchor" "auto" "${PA_MANIFEST}"
        run_method "pa_distill" "proxy_anchor" "pa_distill" "${PA_MANIFEST}"
      done
      ensure_selection "hist"
      HIST_MANIFEST="${SELECTION_MANIFEST}"
      for SEED in 0 1 2; do
        run_method "hist" "hist" "auto" "${HIST_MANIFEST}"
        run_method "herd" "hist" "herd" "${HIST_MANIFEST}"
      done
    fi
  done
}

if [[ "${1:-}" == "--controller" ]]; then
  controller_main
  exit 0
fi

if [[ "${1:-}" == "--collect" ]]; then
  rsync -az "${REMOTE}:${REMOTE_DIR}/reports/generated/" "${LOCAL_DIR}/reports/generated/"
  rsync -az "${REMOTE}:${REMOTE_DIR}/logs/" "${LOCAL_DIR}/logs/"
  exit 0
fi

install_bn_inception_weight() {
  if ssh "${REMOTE}" "test -f '${REMOTE_WEIGHT_CACHE}' && \
    echo '${BN_INCEPTION_SHA256}  ${REMOTE_WEIGHT_CACHE}' | sha256sum --check --status"; then
    echo "Official BN-Inception checkpoint already present on ${REMOTE}."
    return
  fi
  : "${BN_INCEPTION_WEIGHT:?Set BN_INCEPTION_WEIGHT to bn_inception-52deb4733.pth}"
  local actual_hash
  actual_hash="$(shasum -a 256 "${BN_INCEPTION_WEIGHT}" | awk '{print $1}')"
  if [[ "${actual_hash}" != "${BN_INCEPTION_SHA256}" ]]; then
    echo "Local BN-Inception checkpoint hash mismatch: ${actual_hash}" >&2
    exit 2
  fi
  local upload_path="${REMOTE_DIR}/bn_inception-52deb4733.pth.verified-upload"
  scp "${BN_INCEPTION_WEIGHT}" "${REMOTE}:${upload_path}"
  ssh "${REMOTE}" "echo '${BN_INCEPTION_SHA256}  ${upload_path}' | \
    sha256sum --check --status && mkdir -p '$(dirname -- "${REMOTE_WEIGHT_CACHE}")' && \
    mv '${upload_path}' '${REMOTE_WEIGHT_CACHE}'"
  echo "Installed hash-verified official BN-Inception checkpoint on ${REMOTE}."
}

: "${INSHOP_ROOT:?Set INSHOP_ROOT to the official DeepFashion In-Shop root on the remote host}"
: "${INAT2018_ROOT:?Set INAT2018_ROOT to the iNaturalist 2018 root on the remote host}"
for value in "${DATASETS}" "${INSHOP_ROOT}" "${INAT2018_ROOT}" "${NUM_WORKERS}" "${FORCE_RERUN}"; do
  if [[ "${value}" == *"'"* ]]; then
    echo "Controller arguments cannot contain a single quote." >&2
    exit 2
  fi
done

rsync -az --delete \
  --exclude .venv \
  --exclude .git \
  --exclude data \
  --exclude logs \
  --exclude reports \
  "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

ssh "${REMOTE}" "cd '${REMOTE_DIR}' && uv sync --group dev --extra research"
install_bn_inception_weight
ssh "${REMOTE}" "
  cd '${REMOTE_DIR}'
  mkdir -p logs
  existing_pid=\$(pgrep -f '^bash scripts/run_remote_extended_datasets.sh --controller\$' | head -n 1 || true)
  if test -n \"\${existing_pid}\"; then
    echo \${existing_pid} > '${CONTROLLER_PID_FILE}'
    cat '${CONTROLLER_PID_FILE}'
    exit 0
  fi
  nohup env DATASETS='${DATASETS}' INSHOP_ROOT='${INSHOP_ROOT}' \
  INAT2018_ROOT='${INAT2018_ROOT}' NUM_WORKERS='${NUM_WORKERS}' \
  FORCE_RERUN='${FORCE_RERUN}' \
  bash scripts/run_remote_extended_datasets.sh --controller \
  > '${CONTROLLER_LOG}' 2>&1 < /dev/null & \
  controller_pid=\$!
  echo \${controller_pid} > '${CONTROLLER_PID_FILE}'
  cat '${CONTROLLER_PID_FILE}'
"

echo "Corrected recipe controller is active on ${REMOTE}."
echo "PID file: ${REMOTE_DIR}/${CONTROLLER_PID_FILE}"
echo "Log: ${REMOTE_DIR}/${CONTROLLER_LOG}"
