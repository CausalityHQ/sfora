#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
source_revision=${SOURCE_REVISION:?SOURCE_REVISION is required}
source_manifest=${SOURCE_MANIFEST:?SOURCE_MANIFEST is required}
uv_environment=${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT is required}
cell=${CELL:-siglip2-so400m}
case $cell in
  siglip2-so400m)
    model=google/siglip2-so400m-patch14-384
    revision=e8e487298228002f3d8a82e0cd5c8ea9c567f57f
    ;;
  siglip-so400m)
    model=google/siglip-so400m-patch14-384
    revision=9fdffc58afc957d1a03a25b10dba0329ab15c2a3
    ;;
  *) printf 'unregistered substrate cell: %s\n' "$cell" >&2; exit 2 ;;
esac
output=${OUTPUT:-reports/generated/cars-frozen-substrate-$cell-f0-2026-08-30.json}
partial=$(dirname -- "$output")/.$(basename -- "$output").partial
[[ $source_revision =~ ^[0-9a-f]{40}$ && -f $source_manifest ]]
[[ $uv_environment = /* && $uv_environment != "$repo" && $uv_environment != "$repo/"* ]]
[[ ! -e $output && ! -e $partial ]]

/usr/bin/python3 -I -S - "$source_manifest" "$source_revision" <<'PY'
import pathlib, re, sys
manifest = pathlib.Path(sys.argv[1]).resolve()
lines = manifest.read_text().splitlines()
paths = []
for line in lines:
    if re.fullmatch(r"[0-9a-f]{64}  [^\n]+", line) is None:
        raise SystemExit("invalid manifest line")
    paths.append(line[66:])
if paths != sorted(set(paths)):
    raise SystemExit("manifest paths differ")
if pathlib.Path("SOURCE_REVISION").read_text() != f"{sys.argv[2]}\n":
    raise SystemExit("revision differs")
observed = sorted(str(p.relative_to(pathlib.Path.cwd())) for p in pathlib.Path.cwd().rglob("*") if (p.is_file() or p.is_symlink()) and p.absolute() != manifest)
if observed != paths:
    raise SystemExit("tree differs")
PY
sha256sum --check --strict "$source_manifest" >/dev/null
tree=$(sha256sum "$source_manifest" | awk '{print $1}')
uv run --offline --locked python - "$model" "$revision" <<'PY'
import torch
from huggingface_hub import snapshot_download
import sys
assert torch.cuda.is_available()
snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True)
PY
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --offline --locked python scripts/probe_frozen_substrate.py \
  --output "$output" --source-revision "$source_revision" --source-tree-digest "$tree" \
  --cell "$cell" --batch-size 8 --query-block 32
uv run --offline --locked python - "$output" "$source_revision" "$tree" "$cell" <<'PY'
import json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["schema"] == "sfora-frozen-substrate-screen-v1" and p["claim_eligible"] is False
assert p["source_revision"] == sys.argv[2] and p["source_tree_digest"] == sys.argv[3]
assert p["cell"] == sys.argv[4] and p["metrics"]["queries"] == 1345
assert p["gates"] == {"expected_queries": 1345, "recall_at_1_minimum": 0.94}
print(json.dumps({"output": sys.argv[1], "passed": p["passed"]}, sort_keys=True))
PY
