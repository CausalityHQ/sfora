#!/usr/bin/env bash
set -euo pipefail

root=/home/riomus/sfora-siglip2-live-head-cost-v1
run_base=/home/riomus/runs
output="$run_base/sfora-siglip2-live-head-isolated-cost-v1.json"
export PYTHONPATH="$root/src:$root/scripts"

if systemctl --user is-active --quiet sfora-siglip2-member-bank-multiseed-v2.service \
  || systemctl --user is-active --quiet sfora-siglip2-member-bank-seed22-v1.service \
  || systemctl --user is-active --quiet sfora-siglip2-bank-live-paired-v1.service \
  || systemctl --user is-active --quiet sfora-siglip2-arcface-overflow-diag-179020-v1.service \
  || [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "live-head cost screen requires an idle DGX GPU" >&2
  exit 1
fi
exec 9>"$run_base/.sfora-siglip2-gpu.lock"
flock -n 9 || { echo "SOP SigLIP2 GPU lock is held" >&2; exit 1; }
test ! -e "$output" && test ! -L "$output" || {
  echo "live-head cost receipt already exists" >&2
  exit 1
}

check_sha() {
  local actual
  actual="$(sha256sum "$1")"
  [[ "${actual%% *}" == "$2" ]] || { echo "source hash differs: $1" >&2; exit 1; }
}
if [[ -s "$run_base/sfora-siglip2-member-bank-multiseed-179022-arcface-v1/receipt.json" \
  && -s "$run_base/sfora-siglip2-member-bank-multiseed-179022-float_rank-v1/receipt.json" \
  && -s "$run_base/sfora-siglip2-member-bank-multiseed-179022-bank-v1/receipt.json" ]]; then
  :
else
  check_sha "$run_base/sfora-siglip2-member-bank-aborted-seed020-pair-v1.json" a5c7caef8008ff5f0e2782e2775354dacde84ac15e2dbb468b52bd9cc6a84afc
  check_sha "$run_base/sfora-siglip2-member-bank-multiseed-v2-failure-journal.log" cb7ac80a796e807f7665c89caf4b712c5c20ab3816cc5aff84c39209b13e06a7
fi
check_sha "$root/scripts/benchmark_sop_siglip2_live_head_cost.py" 8389fe936edac600712d4734da1fdd315669f913ae5aab4ea8c160d44b82ceec
check_sha "$root/src/sfora/live_head_bank.py" c90ca44a036cfee1998a5ad11362e6f358f013f502ca83ddd92b8318875a5c39
check_sha "$root/src/sfora/deployed_code_rank.py" a57a1b8cb4722a12dbc8fd255255632b05aff918c8c66e46e5708c2da9e4854a

exec /home/riomus/group-learning/.venv/bin/python \
  "$root/scripts/benchmark_sop_siglip2_live_head_cost.py" \
  --output "$output"
