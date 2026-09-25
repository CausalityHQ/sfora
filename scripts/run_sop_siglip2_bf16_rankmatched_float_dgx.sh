#!/usr/bin/env bash
set -euo pipefail

root=/home/riomus/sfora-siglip2-bf16-rankmatched-v1
old_root=/home/riomus/sfora-siglip2-bf16-qual-v1
run_base=/home/riomus/runs
archive=/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz
model=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
native=/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so
gate="$run_base/sfora-siglip2-bf16-member-bank-multiseed-v1.json"
diagnostic="$run_base/sfora-siglip2-bf16-rank-contribution-v1.json"
python=/home/riomus/group-learning/.venv/bin/python
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
tileiras_dir="$(dirname "$CUTILE_TILEIRAS_PATH")"
export PATH="$tileiras_dir:$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

exec 9>"$run_base/.sfora-siglip2-gpu.lock"
flock -n 9 || { echo "SOP SigLIP2 GPU lock is held" >&2; exit 1; }
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || {
  echo "cannot verify an idle DGX GPU" >&2; exit 1;
}
[[ -z "$gpu_pids" ]] || { echo "DGX GPU is active" >&2; exit 1; }

check_sha() {
  local actual
  actual="$(sha256sum "$1")"
  [[ "${actual%% *}" == "$2" ]] || { echo "source hash differs: $1" >&2; exit 1; }
}
check_sha "$root/scripts/train_sop_siglip2_compact.py" 328cdfbd4d35ae8037d1130c5fc889d60fe0ec25cf185c0c9ecc714a475120e4
check_sha "$old_root/scripts/train_sop_siglip2_compact.py" ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23
check_sha "$gate" 70a682d158c914d00505df8d9fd2e53dab5abe3f067a43d472a5fd5d411ad6ce
check_sha "$diagnostic" 54685249fdc2dd199a28d59469f9b107f9e68ac4a01f34ac5abd7d5a68e7a390
check_sha "$archive" 1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a
check_sha "$native" 39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c

"$python" - "$gate" "$diagnostic" <<'PY'
import json
import sys
from pathlib import Path

gate, diagnostic = (json.loads(Path(arg).read_text()) for arg in sys.argv[1:])
if (
    gate.get("schema") != "sfora-sop-siglip2-bf16-member-bank-multiseed-v1"
    or gate.get("continuation_gate_pass") is not True
    or gate.get("seeds") != [179023, 179024, 179025]
    or diagnostic.get("schema") != "sfora-sop-siglip2-bf16-rank-contribution-v1"
    or diagnostic.get("gate_sha256")
    != "70a682d158c914d00505df8d9fd2e53dab5abe3f067a43d472a5fd5d411ad6ce"
    or diagnostic.get("rank_coefficient_float_control") != 21.93
):
    raise SystemExit("BF16 rank-matched control authority differs")
PY

canary="$run_base/sfora-siglip2-bf16-rankmatched-179023-canary-v1"
[[ ! -e "$canary" && ! -L "$canary" ]] || {
  echo "rank-matched canary output exists" >&2; exit 1;
}
for seed in 179023 179024 179025; do
  output="$run_base/sfora-siglip2-bf16-rankmatched-${seed}-float-v1"
  [[ ! -e "$output" && ! -L "$output" ]] || {
    echo "rank-matched output exists: $output" >&2; exit 1;
  }
done

echo "RUN BF16 rank-matched float canary seed=179023" >&2
"$python" "$root/scripts/train_sop_siglip2_compact.py" \
  --model-snapshot "$model" --candidate-dir "$run_base/sfora-siglip2-train-28e19203" \
  --unicom-l14-archive "$archive" --dataset-root /home/riomus/datasets/Stanford_Online_Products \
  --native-library "$native" --output-dir "$canary" --arm float_rank \
  --seed 179023 --updates 1 --batch-size 64 --workers 2 \
  --gradient-diagnostic-steps 1 --rank-coefficient 21.93 --train-vision-dtype bf16

"$python" - "$canary/receipt.json" "$run_base/sfora-siglip2-bf16-member-bank-179023-float_rank-v1/receipt.json" <<'PY'
import json
import math
import sys
from pathlib import Path

canary, full = (json.loads(Path(arg).read_text()) for arg in sys.argv[1:])
if (
    canary.get("source_sha256")
    != "328cdfbd4d35ae8037d1130c5fc889d60fe0ec25cf185c0c9ecc714a475120e4"
    or canary.get("quality") is not None
    or canary.get("updates") != 1
    or canary.get("rank_coefficient") != 21.93
    or len(canary.get("step_seconds", [])) != 1
    or not math.isfinite(canary.get("first_loss", math.nan))
    or len(canary.get("rank_to_arcface_head_gradient_ratio", [])) != 1
    or not math.isfinite(canary["rank_to_arcface_head_gradient_ratio"][0])
    or any(
        canary.get(key) != full.get(key)
        for key in ("initial_head_sha256", "initial_classifier_sha256", "model_file_sha256")
    )
    or canary.get("first_input_batch_sha256", [None])[0]
    != full.get("first_input_batch_sha256", [None])[0]
):
    raise SystemExit("BF16 rank-matched canary authority differs")
PY

for seed in 179023 179024 179025; do
  output="$run_base/sfora-siglip2-bf16-rankmatched-${seed}-float-v1"
  echo "RUN BF16 rank-matched float seed=$seed" >&2
  "$python" "$root/scripts/train_sop_siglip2_compact.py" \
    --model-snapshot "$model" --candidate-dir "$run_base/sfora-siglip2-train-28e19203" \
    --unicom-l14-archive "$archive" --dataset-root /home/riomus/datasets/Stanford_Online_Products \
    --native-library "$native" --output-dir "$output" --arm float_rank \
    --seed "$seed" --updates 1000 --batch-size 64 --workers 2 \
    --rank-coefficient 21.93 --train-vision-dtype bf16 --evaluate
  "$python" - "$output/receipt.json" "$run_base/sfora-siglip2-bf16-member-bank-${seed}-float_rank-v1/receipt.json" <<'PY'
import json
import sys
from pathlib import Path

matched, original = (json.loads(Path(arg).read_text()) for arg in sys.argv[1:])
if (
    matched.get("source_sha256")
    != "328cdfbd4d35ae8037d1130c5fc889d60fe0ec25cf185c0c9ecc714a475120e4"
    or matched.get("rank_coefficient") != 21.93
    or matched.get("updates") != 1000
    or matched.get("quality", {}).get("native_top10_exact") is not True
    or matched.get("quality", {}).get("gallery_wire_bytes_per_row") != 130
    or any(
        matched.get(key) != original.get(key)
        for key in (
            "seed", "schedule_sha256", "first_input_batch_sha256", "initial_head_sha256",
            "initial_classifier_sha256", "model_file_sha256", "source_archive_sha256",
            "query_image_ids_sha256", "native_library_sha256", "tileiras_sha256",
        )
    )
):
    raise SystemExit("BF16 rank-matched full arm authority differs")
PY
done
