#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"

source_revision=${SOURCE_REVISION:?SOURCE_REVISION must be the deployed 40-character commit}
source_manifest=${SOURCE_MANIFEST:?SOURCE_MANIFEST must be the deployed SHA-256 manifest}
uv_environment=${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT must be outside the deployed tree}
output=${OUTPUT:-reports/generated/cars-token-set-f0-2026-08-30.json}
model_revision=7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed
partial=$(dirname -- "$output")/.$(basename -- "$output").partial

[[ $source_revision =~ ^[0-9a-f]{40}$ ]]
[[ -f $source_manifest ]]
[[ $uv_environment = /* && $uv_environment != "$repo" && $uv_environment != "$repo/"* ]]
[[ ! -e $output ]]
[[ ! -e $partial ]]

/usr/bin/python3 -I -S - "$source_manifest" "$source_revision" <<'PY'
import pathlib
import re
import sys

manifest_path = pathlib.Path(sys.argv[1]).resolve()
manifest = manifest_path.read_text().splitlines()
source_revision = sys.argv[2]
required = {
    "SOURCE_REVISION",
    "scripts/probe_siglip_token_set.py",
    "src/sfora/data.py",
    "src/sfora/kernels/__init__.py",
    "src/sfora/kernels/set_maxsim.py",
    "src/sfora/token_set_proxy_anchor.py",
    "uv.lock",
}
paths = []
for line in manifest:
    if re.fullmatch(r"[0-9a-f]{64}  [^\n]+", line) is None:
        raise SystemExit("source manifest has an invalid line")
    path = line[66:]
    candidate = pathlib.PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or str(candidate) != path:
        raise SystemExit("source manifest path is not normalized and relative")
    paths.append(path)
if paths != sorted(set(paths)) or not required.issubset(paths):
    raise SystemExit("source manifest paths differ or omit F0 authority")
if pathlib.Path("SOURCE_REVISION").read_text() != f"{source_revision}\n":
    raise SystemExit("source revision differs from the manifest-bound revision")

observed = sorted(
    str(path.relative_to(pathlib.Path.cwd()))
    for path in pathlib.Path.cwd().rglob("*")
    if (path.is_file() or path.is_symlink()) and path.absolute() != manifest_path
)
if observed != paths:
    raise SystemExit("deployed source tree contains unmanifested or missing files")
PY
sha256sum --check --strict "$source_manifest" >/dev/null

source_tree_digest=$(
  sha256sum "$source_manifest" | awk '{print $1}'
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
