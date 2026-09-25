#!/usr/bin/env bash
set -euo pipefail

root=/home/riomus/sfora-siglip2-step543-fork-v1
run_base=/home/riomus/runs
capture="$run_base/sfora-siglip2-arcface-step543-capture-179020-v1/step543_state.pt"
output="$run_base/sfora-siglip2-arcface-step543-backward-fork-v1.json"
model=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

expected_capture_sha256="${1:?pass the independently checked capture SHA-256}"
[[ "$expected_capture_sha256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "capture digest must be SHA-256" >&2
  exit 1
}
if systemctl --user is-active --quiet sfora-siglip2-arcface-step543-capture-179020-v1.service \
  || [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "step543 backward fork requires an idle DGX GPU" >&2
  exit 1
fi
exec 9>"$run_base/.sfora-siglip2-gpu.lock"
flock -n 9 || { echo "SOP SigLIP2 GPU lock is held" >&2; exit 1; }
[[ ! -e "$output" && ! -L "$output" ]] || {
  echo "step543 backward fork output already exists" >&2
  exit 1
}
actual="$(sha256sum "$root/scripts/fork_sop_siglip2_step543_backward.py")"
[[ "${actual%% *}" == 8731c3caff5dcb2c827b0697ab97bd5bd8860df6952dce0538d2bd6f9ef98a98 ]] || {
  echo "step543 backward fork script hash differs" >&2
  exit 1
}
actual="$(sha256sum "$capture")"
[[ "${actual%% *}" == "$expected_capture_sha256" ]] || {
  echo "step543 capture hash differs" >&2
  exit 1
}

exec /home/riomus/group-learning/.venv/bin/python \
  "$root/scripts/fork_sop_siglip2_step543_backward.py" \
  --capture "$capture" --expected-capture-sha256 "$expected_capture_sha256" \
  --model-snapshot "$model" --output "$output"
