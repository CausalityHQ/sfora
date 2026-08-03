#!/usr/bin/env bash
set -u

cd /home/riomus/group-learning

while pgrep -f '[s]fora image-end-to-end.*inshop_official_pa_repaired_seed0' >/dev/null; do
  date -u +%Y-%m-%dT%H:%M:%SZ
  tail -n 1 logs/inshop_official_pa_repaired_seed0.log
  sleep 600
done

date -u +%Y-%m-%dT%H:%M:%SZ
echo TRAINER_EXITED
