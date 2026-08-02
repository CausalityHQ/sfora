#!/usr/bin/env bash
# Wait for the fully source-cadenced Cars RS@k run, then perform strict analysis.
set -euo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
ARTIFACT="$PROJECT/reports/generated/image_end_to_end_cars.rsatk.recall_at_k_surrogate.cars.official-ed05202.3c72b7193c3a_seed0.json"
ANALYSIS="$PROJECT/reports/generated/cars_rsatk_faithful_analysis.json"
PATTERN='[s]fora image-end-to-end --dataset-name cars --objectives recall_at_k_surrogate'

cd "$PROJECT"
printf '%s RS@k fully source-cadenced monitor armed\n' "$(date -Iseconds)"
while pgrep -f "$PATTERN" >/dev/null; do
  sleep 60
done

if [[ ! -s "$ARTIFACT" ]]; then
  printf '%s RS@k monitor REFUSED: completed process has no registered artifact\n' \
    "$(date -Iseconds)" >&2
  exit 3
fi

sha256sum \
  "$ARTIFACT" \
  src/sfora/image_end_to_end.py \
  src/sfora/image_recipes.py \
  scripts/analyze_rsatk_reference.py \
  scripts/run_priority_queue_v45.sh
.venv/bin/python scripts/analyze_rsatk_reference.py "$ARTIFACT" --json-output "$ANALYSIS"
sha256sum "$ANALYSIS"
printf '%s RS@k fully source-cadenced monitor complete\n' "$(date -Iseconds)"
