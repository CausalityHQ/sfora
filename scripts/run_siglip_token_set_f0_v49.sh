#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"

source_revision=${SOURCE_REVISION:?SOURCE_REVISION must be the deployed 40-character commit}
output=${OUTPUT:-reports/generated/cars-token-set-f0-2026-08-30.json}
model_revision=7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed
partial=$(dirname -- "$output")/.$(basename -- "$output").partial

[[ $source_revision =~ ^[0-9a-f]{40}$ ]]
[[ ! -e $output ]]
[[ ! -e $partial ]]

authority_paths=(
  scripts/probe_siglip_token_set.py
  src/sfora/data.py
  src/sfora/kernels/__init__.py
  src/sfora/kernels/set_maxsim.py
  uv.lock
)
git ls-files --error-unmatch -- "${authority_paths[@]}" >/dev/null
[[ $(git rev-parse HEAD) == "$source_revision" ]]
[[ -z $(git status --porcelain) ]]

source_tree_digest=$(
  git ls-tree -r --full-tree "$source_revision" \
    | sha256sum \
    | awk '{print $1}'
)

uv run --offline --locked python - <<'PY'
import torch
import transformers
import triton

assert torch.cuda.is_available()
assert transformers.__version__ == "5.12.1"
assert triton.__version__ == "3.7.1"
PY

if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'PREFLIGHT PASS source=%s tree=%s model=%s\n' \
    "$source_revision" "$source_tree_digest" "$model_revision"
  exit 0
fi

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --offline --locked python scripts/probe_siglip_token_set.py \
    --output "$output" \
    --source-revision "$source_revision" \
    --source-tree-digest "$source_tree_digest" \
    --model-name google/siglip-base-patch16-224 \
    --model-revision "$model_revision" \
    --top-k 32 \
    --set-weight 0.25 \
    --batch-size 64 \
    --query-block 32

uv run --offline --locked python - "$output" "$source_revision" "$source_tree_digest" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
assert payload["schema"] == "sfora-siglip-token-set-screen-v1"
assert payload["claim_eligible"] is False
assert payload["source_revision"] == sys.argv[2]
assert payload["source_tree_digest"] == sys.argv[3]
assert payload["split"] == "train"
assert payload["holdout_classes"] == list(range(82, 98))
assert "test_split_reads" not in payload
assert payload["model_revision"] == "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
assert payload["top_k"] == 32
assert payload["set_weight"] == 0.25
assert isinstance(payload["passed"], bool)
print(json.dumps({"output": str(path), "passed": payload["passed"]}, sort_keys=True))
PY
