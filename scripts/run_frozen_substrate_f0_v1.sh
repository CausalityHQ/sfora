#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"

source_revision=${SOURCE_REVISION:?SOURCE_REVISION is required}
source_manifest=${SOURCE_MANIFEST:?SOURCE_MANIFEST is required}
uv_environment=${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT is required}
output=${OUTPUT:-reports/generated/cars-frozen-substrate-f0-2026-08-30.json}
model_name=facebook/dinov2-large
model_revision=47b73eefe95e8d44ec3623f8890bd894b6ea2d6c
partial=$(dirname -- "$output")/.$(basename -- "$output").partial

[[ $source_revision =~ ^[0-9a-f]{40}$ ]]
[[ -f $source_manifest ]]
[[ $uv_environment = /* && $uv_environment != "$repo" && $uv_environment != "$repo/"* ]]
[[ ! -e $output && ! -e $partial ]]

/usr/bin/python3 -I -S - "$source_manifest" "$source_revision" <<'PY'
import pathlib
import re
import sys

manifest = pathlib.Path(sys.argv[1]).resolve()
lines = manifest.read_text().splitlines()
paths = []
for line in lines:
    if re.fullmatch(r"[0-9a-f]{64}  [^\n]+", line) is None:
        raise SystemExit("source manifest has an invalid line")
    path = line[66:]
    candidate = pathlib.PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or str(candidate) != path:
        raise SystemExit("source manifest path is not normalized and relative")
    paths.append(path)
if paths != sorted(set(paths)):
    raise SystemExit("source manifest paths are not unique and sorted")
if pathlib.Path("SOURCE_REVISION").read_text() != f"{sys.argv[2]}\n":
    raise SystemExit("source revision differs from the manifest-bound revision")
observed = sorted(
    str(path.relative_to(pathlib.Path.cwd()))
    for path in pathlib.Path.cwd().rglob("*")
    if (path.is_file() or path.is_symlink()) and path.absolute() != manifest
)
if observed != paths:
    raise SystemExit("deployed source tree contains unmanifested or missing files")
PY
sha256sum --check --strict "$source_manifest" >/dev/null
source_tree_digest=$(sha256sum "$source_manifest" | awk '{print $1}')

uv run --offline --locked python - <<'PY'
import torch
import transformers
assert torch.cuda.is_available()
assert transformers.__version__ == "5.12.1"
PY

uv run --offline --locked python - "$model_name" "$model_revision" <<'PY'
import pathlib
import sys
from huggingface_hub import snapshot_download
path = pathlib.Path(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))
assert path.is_dir()
PY

if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'PREFLIGHT PASS source=%s tree=%s model=%s@%s\n' \
    "$source_revision" "$source_tree_digest" "$model_name" "$model_revision"
  exit 0
fi

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --offline --locked python scripts/probe_frozen_substrate.py \
    --output "$output" \
    --source-revision "$source_revision" \
    --source-tree-digest "$source_tree_digest" \
    --model-name "$model_name" \
    --model-revision "$model_revision" \
    --batch-size 32 \
    --query-block 32

uv run --offline --locked python - \
  "$output" "$source_revision" "$source_tree_digest" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload["schema"] == "sfora-frozen-substrate-screen-v1"
assert payload["claim_eligible"] is False
assert payload["source_revision"] == sys.argv[2]
assert payload["source_tree_digest"] == sys.argv[3]
assert payload["split"] == "train"
assert payload["holdout_classes"] == list(range(82, 98))
assert payload["model_name"] == "facebook/dinov2-large"
assert payload["model_revision"] == "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
assert payload["compute_dtype"] == "float32"
assert payload["descriptors_validated"] is True
assert payload["metrics"]["queries"] == 1345
assert payload["gates"] == {"expected_queries": 1345, "recall_at_1_minimum": 0.94}
assert isinstance(payload["passed"], bool)
print(json.dumps({"output": sys.argv[1], "passed": payload["passed"]}, sort_keys=True))
PY
