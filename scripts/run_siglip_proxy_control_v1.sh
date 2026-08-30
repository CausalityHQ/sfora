#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"

source_revision=${SOURCE_REVISION:?SOURCE_REVISION must be the deployed 40-character commit}
source_manifest=${SOURCE_MANIFEST:?SOURCE_MANIFEST must be the deployed SHA-256 manifest}
uv_environment=${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT must be outside the deployed tree}
output_dir=${OUTPUT_DIR:?OUTPUT_DIR must be an absolute directory outside the deployed tree}
maximum_checkpoint_bytes=${MAXIMUM_CHECKPOINT_BYTES:-8589934592}

[[ $source_revision =~ ^[0-9a-f]{40}$ ]]
[[ -f $source_manifest ]]
[[ $uv_environment = /* && $uv_environment != "$repo" && $uv_environment != "$repo/"* ]]
[[ $output_dir = /* && $output_dir != "$repo" && $output_dir != "$repo/"* ]]
[[ $maximum_checkpoint_bytes =~ ^[1-9][0-9]*$ ]]

/usr/bin/python3 -I -S - "$source_manifest" "$source_revision" <<'PY'
import pathlib
import re
import sys

manifest_path = pathlib.Path(sys.argv[1]).resolve()
source_revision = sys.argv[2]
lines = manifest_path.read_text().splitlines()
required = {
    "SOURCE_REVISION",
    "scripts/run_siglip_proxy_control.py",
    "scripts/run_siglip_proxy_control_v1.sh",
    "src/sfora/data.py",
    "src/sfora/siglip_proxy_control.py",
    "src/sfora/substrate_screen.py",
    "src/sfora/token_set_proxy_anchor.py",
    "src/sfora/token_set_screen.py",
    "uv.lock",
}
paths = []
for line in lines:
    if re.fullmatch(r"[0-9a-f]{64}  [^\n]+", line) is None:
        raise SystemExit("source manifest has an invalid line")
    path = line[66:]
    candidate = pathlib.PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or str(candidate) != path:
        raise SystemExit("source manifest path is not normalized and relative")
    paths.append(path)
if paths != sorted(set(paths)) or not required.issubset(paths):
    raise SystemExit("source manifest paths differ or omit pooled-control authority")
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
source_tree_digest=$(sha256sum "$source_manifest" | awk '{print $1}')

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

uv run --offline --locked python - <<'PY'
import torch
import transformers
import torchvision

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
assert transformers.__version__
assert torchvision.__version__
PY

if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'PREFLIGHT PASS source=%s tree=%s\n' "$source_revision" "$source_tree_digest"
  exit 0
fi

mkdir -p -- "$output_dir"
smoke=$output_dir/smoke.json
if [[ ! -e $smoke ]]; then
  uv run --offline --locked python scripts/run_siglip_proxy_control.py smoke \
    --output "$smoke" \
    --source-revision "$source_revision" \
    --source-tree-digest "$source_tree_digest"
fi

seed_receipts=()
for seed in 17 29 43; do
  receipt=$(printf '%s/seed-%03d.receipt.json' "$output_dir" "$seed")
  seed_receipts+=(--seed-receipt "$receipt")
  if [[ ! -e $receipt ]]; then
    uv run --offline --locked python scripts/run_siglip_proxy_control.py train \
      --output-dir "$output_dir" \
      --smoke "$smoke" \
      --seed "$seed" \
      --source-revision "$source_revision" \
      --source-tree-digest "$source_tree_digest" \
      --maximum-checkpoint-bytes "$maximum_checkpoint_bytes" \
      --evaluation-batch-size 32 \
      --query-block 128
  fi
done

aggregate=$output_dir/control.receipt.json
[[ ! -e $aggregate ]]
uv run --offline --locked python scripts/run_siglip_proxy_control.py aggregate \
  --output "$aggregate" \
  "${seed_receipts[@]}"

/usr/bin/python3 -I -S - "$aggregate" "$source_revision" "$source_tree_digest" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_bytes())
assert payload["schema"] == "sfora-siglip-proxy-control-aggregate-v1"
assert payload["claim_eligible"] is False
assert payload["seeds"] == [17, 29, 43]
assert path.read_bytes().endswith(b"\n") and not path.read_bytes().endswith(b"\n\n")
print(json.dumps({"output": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}, sort_keys=True))
PY
