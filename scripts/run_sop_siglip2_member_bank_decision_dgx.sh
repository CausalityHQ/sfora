#!/usr/bin/env bash
set -euo pipefail

root=/home/riomus/sfora-siglip2-member-bank-v1
run_base=/home/riomus/runs
export PYTHONPATH="$root/src:$root/scripts"

exec /home/riomus/group-learning/.venv/bin/python \
  "$root/scripts/compare_sop_siglip2_member_bank_arms.py" \
  --source-archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
  --control-receipt "$run_base/sfora-siglip2-member-bank-control-1000-v1/receipt.json" \
  --treatment-receipt "$run_base/sfora-siglip2-member-bank-treatment-1000-v1/receipt.json" \
  --output "$run_base/sfora-siglip2-member-bank-decision-v1.json"
