#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"

source_revision=${SOURCE_REVISION:?SOURCE_REVISION must be the deployed 40-character commit}
source_manifest=${SOURCE_MANIFEST:?SOURCE_MANIFEST must be the deployed SHA-256 manifest}
f0_receipt=${F0_RECEIPT:?F0_RECEIPT must name the immutable passing F0 result}
f0_receipt_sha256=${F0_RECEIPT_SHA256:?F0_RECEIPT_SHA256 must authenticate the F0 result}
output=${OUTPUT:-reports/generated/cars-token-set-f1-2026-08-30.json}

[[ $source_revision =~ ^[0-9a-f]{40}$ ]]
[[ $f0_receipt_sha256 =~ ^[0-9a-f]{64}$ ]]
[[ -f $f0_receipt ]]
[[ $(sha256sum "$f0_receipt" | awk '{print $1}') == "$f0_receipt_sha256" ]]
[[ ! -e $output ]]

PREFLIGHT_ONLY=1 OUTPUT="$output" \
  bash scripts/run_siglip_token_set_f0_v49.sh
source_tree_digest=$(sha256sum "$source_manifest" | awk '{print $1}')

export CUBLAS_WORKSPACE_CONFIG=:4096:8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --offline --locked python scripts/run_siglip_token_set_f1.py \
    --f0-receipt "$f0_receipt" \
    --f0-receipt-sha256 "$f0_receipt_sha256" \
    --output "$output" \
    --source-revision "$source_revision" \
    --source-tree-digest "$source_tree_digest" \
    --feature-batch-size 64

uv run --offline --locked python - "$output" "$source_revision" "$source_tree_digest" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
assert payload["schema"] == "sfora-siglip-token-set-f1-v1"
assert payload["claim_eligible"] is False
assert payload["source_revision"] == sys.argv[2]
assert payload["source_tree_digest"] == sys.argv[3]
assert payload["split"] == "train"
assert payload["train_classes"] == list(range(49))
assert payload["validation_classes"] == list(range(49, 82))
assert payload["seeds"] == [17, 29, 43]
assert len(payload["arms"]) == 9
assert payload["passed"] is payload["summary"]["passed"]
print(json.dumps({"output": str(path), "passed": payload["passed"]}, sort_keys=True))
PY
