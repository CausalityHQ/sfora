#!/usr/bin/env bash
set -euo pipefail

root=/home/riomus/sfora-siglip2-bf16-cached-head-v1
run_base=/home/riomus/runs
archive=/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz
candidate="$run_base/sfora-siglip2-train-28e19203"
native=/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so
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
check_sha "$root/scripts/train_sop_siglip2_cached_head_probe.py" aac9dfa0835bf199b1c6a266fba9165612d7fb1d9cb61c46bdd0cfb74366b317
check_sha "$root/scripts/train_sop_siglip2_compact.py" 328cdfbd4d35ae8037d1130c5fc889d60fe0ec25cf185c0c9ecc714a475120e4
check_sha "$root/src/sfora/live_head_bank.py" c90ca44a036cfee1998a5ad11362e6f358f013f502ca83ddd92b8318875a5c39
check_sha "$root/src/sfora/sop_compact_training.py" 12ccd15c3b943fa519ba37f8d09870b105939872d9279541fc0ab42f20a7ead9
check_sha "$root/src/sfora/unicom_rank_finish.py" b77c119aab09dae0e7ba6bab5dbffbcb9b1695d85c7b544771adaf54b0cce253
check_sha "$root/src/sfora/unicom_training.py" a40b0dac4173511787dd9a4da82e506ce44eb3ccb63710595800bc9ebcd4d272
check_sha "$root/src/sfora/representation_ceiling.py" 1608181a9c7ba18d1ae1016804898551b2bba9ab40fec9bd4cc2eaf27981771c
check_sha "$root/src/sfora/joint_relational_compaction.py" 4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67
check_sha "$root/src/sfora/cutile_int8.py" b7c57022a836774d641a829e6aac71c1d547e3f71136f716d3c9aeeedad11409
check_sha "$CUTILE_TILEIRAS_PATH" df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae
check_sha "$archive" 1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a
check_sha "$candidate/receipt.json" 3d49e039c72c0677591835752133caf1cbfd0ea483c23a901773748b44d843fb
check_sha "$candidate/train_features.npy" d15f76e459b90836df2807260a34d06692e2f2e5b8ba17b3449ed56500b8357a
check_sha "$native" 39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c

for arm in arcface live_head_bank; do
  output="$run_base/sfora-siglip2-cached-head-179023-${arm}-v1"
  [[ ! -e "$output" && ! -L "$output" ]] || {
    echo "cached-head output exists: $output" >&2; exit 1;
  }
done

for arm in arcface live_head_bank; do
  output="$run_base/sfora-siglip2-cached-head-179023-${arm}-v1"
  echo "RUN SOP cached-head arm=$arm" >&2
  /home/riomus/group-learning/.venv/bin/python \
    "$root/scripts/train_sop_siglip2_cached_head_probe.py" \
    --candidate-dir "$candidate" \
    --source-archive "$archive" \
    --native-library "$native" \
    --output-dir "$output" \
    --arm "$arm"
done
