#!/usr/bin/env bash

# Linux PSI avg10 values are percentages in [0, 100], not fractions in [0, 1].
sfora_pressure_update() {
  test "$#" = 2
  local psi_percent=$1 prior_hits=$2 hits reason=none
  [[ $prior_hits =~ ^[0-9]+$ ]]
  awk -v value="$psi_percent" 'BEGIN {
    exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0 && value <= 100)
  }' || return 1
  if awk -v value="$psi_percent" 'BEGIN { exit !(value >= 50.0) }'; then
    hits=$((prior_hits + 1))
  else
    hits=0
  fi
  if awk -v value="$psi_percent" 'BEGIN { exit !(value >= 79.0) }'; then
    reason=psi-immediate
  elif ((hits >= 3)); then
    reason=psi-sustained
  fi
  printf '%s %s\n' "$hits" "$reason"
}
