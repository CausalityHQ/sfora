#!/usr/bin/env bash
# Fully source-cadenced Cars196 RS@k reference reproduction registered in
# docs/rsatk_reference_preregistration_2026-08-01.md.
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
cd "${PROJECT}" || exit 2

# Import the digest-validated runner without executing v24's historical queue body.
# shellcheck disable=SC1090
source <(sed '/^log "priority queue v24 start"/,$d' scripts/run_priority_queue_v24.sh)

log "priority queue v45 start: fully source-cadenced Cars196 RS@k reproduction"
run_arm cars rsatk recall_at_k_surrogate auto 0
log "priority queue v45 complete: stop for registered reproduction judgement"
