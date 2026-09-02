#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1

source_revision=${SOURCE_REVISION:?SOURCE_REVISION is required}
source_manifest=${SOURCE_MANIFEST:?SOURCE_MANIFEST is required}
output_dir=${OUTPUT_DIR:?OUTPUT_DIR is required}
uv_environment=${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT is required}
error_manifest=${ERROR_MANIFEST:?ERROR_MANIFEST is required}
dinov2_prerequisite=${DINOV2_PREREQUISITE:?DINOV2_PREREQUISITE is required}
siglip2_prerequisite=${SIGLIP2_PREREQUISITE:?SIGLIP2_PREREQUISITE is required}
selecting_prerequisite=${SELECTING_PREREQUISITE:?SELECTING_PREREQUISITE is required}
m3_seed_017=${M3_SEED_017:?M3_SEED_017 is required}
m3_seed_029=${M3_SEED_029:?M3_SEED_029 is required}
m3_seed_043=${M3_SEED_043:?M3_SEED_043 is required}
m3_aggregate=${M3_AGGREGATE:?M3_AGGREGATE is required}

[[ $source_revision =~ ^[0-9a-f]{40}$ ]]
[[ -f $source_manifest && -f $error_manifest ]]
[[ $output_dir = /* && $uv_environment = /* ]]
[[ $output_dir != "$repo" && $output_dir != "$repo/"* ]]
[[ $uv_environment != "$repo" && $uv_environment != "$repo/"* ]]

sha256sum --check --strict "$source_manifest" >/dev/null
source_tree_digest=$(sha256sum "$source_manifest" | awk '{print $1}')
[[ -f SOURCE_REVISION && $(<SOURCE_REVISION) == "$source_revision" ]]
/usr/bin/python3 -I -S - "$repo" "$source_manifest" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
manifest = pathlib.Path(sys.argv[2]).resolve(strict=True)
if manifest.parent != root:
    raise SystemExit("source manifest must be at the source root")
registered: list[str] = []
for line in manifest.read_text(encoding="ascii").splitlines():
    match = re.fullmatch(r"[0-9a-f]{64}  ([^\0\r\n]+)", line)
    if match is None:
        raise SystemExit("source manifest syntax differs")
    relative = match.group(1)
    path = pathlib.PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or relative == manifest.name:
        raise SystemExit("source manifest path differs")
    registered.append(relative)
if registered != sorted(set(registered)):
    raise SystemExit("source manifest order differs")
observed = sorted(
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if not path.is_dir() and path.resolve(strict=True) != manifest
)
if observed != registered:
    raise SystemExit("source inventory differs")
PY

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

wait_for_control_release() {
  if pgrep -f '[r]un_siglip_proxy_control.py|[r]un_native_twin_probe.py|[a]udit_siglip_control_checkpoint.py|[p]robe_frozen_substrate.py|[r]un_siglip_rsta_stage_a.py|[d]iagnose_siglip_(rsta_stage_a|intermediate_readout|head_screen|sfq).py' >/dev/null; then
    echo 'active control has not released the DGX' >&2
    return 75
  fi
}

memory_psi_full_avg10() {
  awk '/^full / { for (i=1;i<=NF;i++) if ($i ~ /^avg10=/) { sub(/^avg10=/,"",$i); print $i; exit } }' /proc/pressure/memory
}

swap_used_kib() {
  awk '/^SwapTotal:/ { total=$2 } /^SwapFree:/ { free=$2 } END { print total-free }' /proc/meminfo
}

combined_resource_bytes() {
  local process_group=$1 cuda_bytes=$2
  local rss_bytes
  rss_bytes=$(ps -o rss= -g "$process_group" | awk '{ total += $1 } END { printf "%.0f", total*1024 }')
  printf '%s\n' "$((rss_bytes + cuda_bytes))"
}

authenticated_progress_state() {
  local progress=$1 checkpoint_dir=$2 cell=$3
  /usr/bin/python3 -I -S - \
    "$progress" "$checkpoint_dir" "$cell" "$source_revision" "$source_tree_digest" <<'PY'
import hashlib
import json
import pathlib
import sys

progress = pathlib.Path(sys.argv[1])
checkpoint_dir = pathlib.Path(sys.argv[2])
expected_cell, source_revision, source_tree_digest = sys.argv[3:]
expected_keys = {
    "cell", "checkpoint_sha256", "cuda_peak_reserved_bytes", "rows", "schema",
    "source_revision", "source_tree_digest",
}
latest = None
previous_rows = 0
previous_cuda_peak = 0
for raw in progress.read_bytes().splitlines(keepends=True):
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        continue
    if type(value) is not dict or value.get("schema") != "sfora-pass209-m4-progress-v1":
        continue
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if raw != canonical or set(value) != expected_keys:
        raise SystemExit("M4 progress authority differs")
    if (
        value["cell"] != expected_cell
        or value["source_revision"] != source_revision
        or value["source_tree_digest"] != source_tree_digest
        or type(value["rows"]) is not int
        or value["rows"] <= 0
        or type(value["cuda_peak_reserved_bytes"]) is not int
        or value["cuda_peak_reserved_bytes"] < 0
        or type(value["checkpoint_sha256"]) is not str
        or len(value["checkpoint_sha256"]) != 64
        or value["rows"] <= previous_rows
        or value["cuda_peak_reserved_bytes"] < previous_cuda_peak
    ):
        raise SystemExit("M4 progress fields differ")
    latest = value
    previous_rows = value["rows"]
    previous_cuda_peak = value["cuda_peak_reserved_bytes"]
if latest is None:
    raise SystemExit(2)
checkpoint = checkpoint_dir / f"checkpoint-{latest['rows']:04d}.bin"
try:
    payload = checkpoint.read_bytes()
except FileNotFoundError:
    raise SystemExit(2) from None
if hashlib.sha256(payload).hexdigest() != latest["checkpoint_sha256"]:
    raise SystemExit("M4 progress checkpoint binding differs")
print(latest["cuda_peak_reserved_bytes"], checkpoint.stat().st_mtime_ns // 1_000_000_000)
PY
}

cleanup_partials() {
  local path
  for path in "$output_dir"/.*.partial; do
    [[ ! -e $path ]] || unlink "$path"
  done
  for path in "$output_dir"/*.checkpoint/.*.partial; do
    [[ ! -e $path ]] || unlink "$path"
  done
}

stage=preflight
operational_stop_reason=process-exit
write_operational_stop() {
  local exit_status=$1
  local stop="$output_dir/m4-operational-stop.json"
  [[ -e $stop ]] && return 0
  /usr/bin/python3 -I -S - \
    "$stop" "$source_revision" "$source_tree_digest" "$stage" "$exit_status" \
    "$operational_stop_reason" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = {
    "claim_eligible": False,
    "consequence": "F4-NONE",
    "exit_status": int(sys.argv[5]),
    "reason": sys.argv[6],
    "schema": "sfora-pass209-m4-operational-stop-v1",
    "source_revision": sys.argv[2],
    "source_tree_digest": sys.argv[3],
    "stage": sys.argv[4],
    "status": "failed",
}
payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
with path.open("xb") as stream:
    stream.write(payload)
    stream.flush()
PY
}

ensure_campaign_authority() {
  local mode=${1:?campaign authority mode is required}
  local authority="$output_dir/campaign-authority.json"
  /usr/bin/python3 -I -S - \
    "$authority" "$source_revision" "$source_tree_digest" "$mode" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = {
    "cells": ["dinov2-large", "siglip2-so400m", "siglip-so400m"],
    "claim_eligible": False,
    "schema": "sfora-pass209-m4-campaign-authority-v1",
    "source_revision": sys.argv[2],
    "source_tree_digest": sys.argv[3],
}
expected = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
if path.exists():
    if path.read_bytes() != expected:
        raise SystemExit("M4 campaign authority differs")
elif sys.argv[4] != "create":
    raise SystemExit("M4 campaign authority is absent")
else:
    with path.open("xb") as stream:
        stream.write(expected)
        stream.flush()
PY
}

validate_resource_resume() {
  local stop="$output_dir/m4-operational-stop.json"
  local resumed_stage
  resumed_stage=$(/usr/bin/python3 -I -S - "$stop" "$source_revision" "$source_tree_digest" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("resource-stop receipt is absent")
raw = path.read_bytes()
value = json.loads(raw)
expected_keys = {
    "claim_eligible", "consequence", "exit_status", "reason", "schema",
    "source_revision", "source_tree_digest", "stage", "status",
}
if (
    set(value) != expected_keys
    or json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n" != raw
    or value["schema"] != "sfora-pass209-m4-operational-stop-v1"
    or value["claim_eligible"] is not False
    or value["consequence"] != "F4-NONE"
    or type(value["exit_status"]) is not int
    or value["exit_status"] != 125
    or value["reason"] not in {
        "combined-resource", "memory-psi", "progress-authority",
        "progress-timeout", "swap-delta",
    }
    or value["status"] != "failed"
    or value["source_revision"] != sys.argv[2]
    or value["source_tree_digest"] != sys.argv[3]
    or value["stage"] not in {"dinov2-large", "siglip2-so400m", "siglip-so400m"}
):
    raise SystemExit("resource-stop receipt differs")
print(value["stage"])
PY
  ) || return $?
  case "$resumed_stage" in
    dinov2-large|siglip2-so400m|siglip-so400m) ;;
    *) return 76;;
  esac
  /usr/bin/python3 -I -S - \
    "$output_dir" "$resumed_stage" "$source_revision" "$source_tree_digest" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
stage = sys.argv[2]
order = [
    ("dinov2-large", "dinov2"),
    ("siglip2-so400m", "siglip2"),
    ("siglip-so400m", "selecting"),
]
index = [name for name, _ in order].index(stage)
allowed_files = {"campaign-authority.json", "m4-operational-stop.json"}
allowed_directories = set()
for _, prefix in order[:index]:
    allowed_files.update({
        f"{prefix}.receipt.json",
        f"{prefix}.descriptor.bin",
        f"{prefix}.queries.json",
        f"{prefix}.progress.jsonl",
    })
prefix = order[index][1]
allowed_files.add(f"{prefix}.progress.jsonl")
checkpoint_directory = f"{prefix}.checkpoint"
allowed_directories.add(checkpoint_directory)
checkpoint_count = 0
for path in root.rglob("*"):
    relative = path.relative_to(root).as_posix()
    if path.is_symlink():
        raise SystemExit("resume namespace contains a symlink")
    if path.is_dir():
        if relative not in allowed_directories:
            raise SystemExit("resume namespace contains an extra directory")
    elif relative.startswith(f"{checkpoint_directory}/"):
        basename = relative.removeprefix(f"{checkpoint_directory}/")
        if re.fullmatch(r"checkpoint-[0-9]{4}\.bin", basename) is None:
            raise SystemExit("resume checkpoint namespace differs")
        checkpoint_count += 1
    elif re.fullmatch(r"m4-operational-stop-history-[0-9]{4}\.json", relative):
        raw = path.read_bytes()
        value = __import__("json").loads(raw)
        if (
            set(value) != {
                "claim_eligible", "consequence", "exit_status", "reason", "schema",
                "source_revision", "source_tree_digest", "stage", "status",
            }
            or value.get("claim_eligible") is not False
            or value.get("consequence") != "F4-NONE"
            or type(value.get("exit_status")) is not int
            or value.get("reason") not in {
                "combined-resource", "memory-psi", "progress-authority",
                "progress-timeout", "swap-delta",
            }
            or __import__("json").dumps(
                value, sort_keys=True, separators=(",", ":")
            ).encode() + b"\n" != raw
            or value.get("schema") != "sfora-pass209-m4-operational-stop-v1"
            or value.get("source_revision") != sys.argv[3]
            or value.get("source_tree_digest") != sys.argv[4]
            or value.get("stage") not in {name for name, _ in order}
            or value.get("status") != "failed"
        ):
            raise SystemExit("resume stop history differs")
    elif relative not in allowed_files:
        raise SystemExit("resume namespace contains an extra file")
if checkpoint_count == 0:
    progress = root / f"{prefix}.progress.jsonl"
    if progress.is_file() and b'"schema":"sfora-pass209-m4-progress-v1"' in progress.read_bytes():
        raise SystemExit("resume progress has no bound checkpoint")
PY
  local namespace_status=$?
  (( namespace_status == 0 )) || return "$namespace_status"
  if [[ -e $output_dir/m4.receipt.json || -e $output_dir/adapter.receipt.json ]]; then
    return 76
  fi
  printf '%s\n' "$resumed_stage"
}

archive_operational_stop() {
  local stop="$output_dir/m4-operational-stop.json" index target
  for index in $(seq -w 1 9999); do
    target="$output_dir/m4-operational-stop-history-$index.json"
    if [[ ! -e $target ]]; then
      mv -- "$stop" "$target"
      return 0
    fi
  done
  return 76
}

write_campaign_terminal() {
  /usr/bin/python3 -I -S - \
    "$output_dir" "$source_revision" "$source_tree_digest" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
path = root / "campaign-terminal.json"
if path.exists() or (root / "m4-operational-stop.json").exists():
    raise SystemExit("M4 campaign terminal namespace differs")
histories = []
for history in sorted(root.glob("m4-operational-stop-history-[0-9][0-9][0-9][0-9].json")):
    histories.append(
        {"name": history.name, "sha256": hashlib.sha256(history.read_bytes()).hexdigest()}
    )
value = {
    "adapter_sha256": hashlib.sha256((root / "adapter.receipt.json").read_bytes()).hexdigest(),
    "claim_eligible": False,
    "m4_sha256": hashlib.sha256((root / "m4.receipt.json").read_bytes()).hexdigest(),
    "operational_stop_history": histories,
    "schema": "sfora-pass209-m4-campaign-terminal-v1",
    "source_revision": sys.argv[2],
    "source_tree_digest": sys.argv[3],
    "status": "complete",
}
payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
with path.open("xb") as stream:
    stream.write(payload)
    stream.flush()
PY
}

on_exit() {
  local status=$?
  trap - EXIT
  cleanup_partials
  if (( status != 0 )) && [[ -d $output_dir ]]; then
    write_operational_stop "$status" || true
  fi
  exit "$status"
}

monitor_child() {
  local child=$1 progress=$2 checkpoint_dir=$3 cell=$4 swap_start=$5
  local stop_reason= progress_age_seconds=0 swap_delta_kib=0 resources=0 psi=0
  local cuda_bytes=0 progress_time=0 progress_state= progress_status=0 child_state=
  while kill -0 "$child" 2>/dev/null; do
    psi=$(memory_psi_full_avg10)
    swap_delta_kib=$(( $(swap_used_kib) - swap_start ))
    if [[ -s $progress ]]; then
      if progress_state=$(authenticated_progress_state "$progress" "$checkpoint_dir" "$cell"); then
        read -r cuda_bytes progress_time <<<"$progress_state"
        progress_age_seconds=$(( $(date +%s) - progress_time ))
      else
        progress_status=$?
        if (( progress_status == 2 )); then
          progress_age_seconds=$((progress_age_seconds + 5))
        else
          stop_reason=progress-authority
        fi
      fi
    else
      progress_age_seconds=$((progress_age_seconds + 5))
    fi
    resources=$(combined_resource_bytes "$child" "$cuda_bytes")
    (( resources <= 68719476736 )) || stop_reason=combined-resource
    awk -v value="$psi" 'BEGIN { exit !(value >= 0.50) }' && stop_reason=memory-psi || true
    (( swap_delta_kib <= 262144 )) || stop_reason=swap-delta
    (( progress_age_seconds <= 1200 )) || stop_reason=progress-timeout
    if [[ -n $stop_reason ]]; then
      operational_stop_reason=$stop_reason
      kill -TERM -- "-$child" 2>/dev/null || true
      for _ in 1 2 3 4 5 6; do
        child_state=$(ps -o stat= -p "$child" 2>/dev/null || true)
        [[ -n $child_state && $child_state != Z* ]] || break
        sleep 5
      done
      child_state=$(ps -o stat= -p "$child" 2>/dev/null || true)
      if [[ -n $child_state && $child_state != Z* ]]; then
        kill -KILL -- "-$child" 2>/dev/null || true
      fi
      wait "$child" 2>/dev/null || true
      printf 'M4 STOP reason=%s\n' "$stop_reason" >&2
      return 125
    fi
    sleep 5
  done
  wait "$child"
}

run_cell() {
  [[ ${1:-} == --cell ]]
  local cell=$2 prerequisite=$3 prefix=$4
  local receipt="$output_dir/$prefix.receipt.json"
  local descriptor="$output_dir/$prefix.descriptor.bin"
  local queries="$output_dir/$prefix.queries.json"
  local progress="$output_dir/$prefix.progress.jsonl"
  if [[ -e $receipt || -e $descriptor || -e $queries ]]; then
    [[ -s $receipt && -s $descriptor && -s $queries ]] || {
      echo 'same-cell resume artifacts are incomplete' >&2
      return 76
    }
    return 0
  fi
  setsid uv run --offline --locked python scripts/run_pass209_m4_cell.py \
    --cell "$cell" \
    --prerequisite "$prerequisite" \
    --error-manifest "$error_manifest" \
    --receipt-output "$receipt" \
    --descriptor-output "$descriptor" \
    --query-output "$queries" \
    --checkpoint-dir "$output_dir/$prefix.checkpoint" \
    --source-revision "$source_revision" \
    --source-tree-digest "$source_tree_digest" \
    --uv-lock uv.lock \
    --execute 2>"$progress" &
  local child=$!
  monitor_child \
    "$child" "$progress" "$output_dir/$prefix.checkpoint" "$cell" "$(swap_used_kib)"
}

wait_for_control_release
if [[ -e $output_dir ]]; then
  [[ -d $output_dir ]] || exit 73
  ensure_campaign_authority require || {
    echo 'refusing to reuse an unauthenticated M4 output root' >&2
    exit 73
  }
  validate_resource_resume >/dev/null || {
    echo 'refusing to duplicate a live M4 campaign or reuse an output root' >&2
    exit 73
  }
  archive_operational_stop
else
  mkdir -- "$output_dir"
  ensure_campaign_authority create
fi
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# scorer self-test
uv run --offline --locked python - <<'PY'
import torch
from huggingface_hub import snapshot_download
from pathlib import Path
from sfora.pass209_m4 import (
    M4Example,
    REGISTERED_M4_CELLS,
    configure_reference_scorer,
    score_descriptor_plane,
)

configure_reference_scorer(Path("uv.lock"))
for cell in REGISTERED_M4_CELLS.values():
    snapshot_download(
        repo_id=cell.model_name,
        revision=cell.model_revision,
        local_files_only=True,
    )
descriptors = torch.eye(4, dtype=torch.float32)
examples = tuple(M4Example(i, f"fixture-{i}", i // 2) for i in range(4))
assert len(score_descriptor_plane(descriptors, examples, block_size=32)) == 4
PY

stage=dinov2-large
run_cell --cell dinov2-large "$dinov2_prerequisite" dinov2
stage=siglip2-so400m
run_cell --cell siglip2-so400m "$siglip2_prerequisite" siglip2
stage=siglip-so400m
run_cell --cell siglip-so400m "$selecting_prerequisite" selecting

stage=analyzer
uv run --offline --locked python scripts/analyze_pass209_m4.py \
  --dinov2-receipt "$output_dir/dinov2.receipt.json" \
  --dinov2-descriptor "$output_dir/dinov2.descriptor.bin" \
  --dinov2-queries "$output_dir/dinov2.queries.json" \
  --siglip2-receipt "$output_dir/siglip2.receipt.json" \
  --siglip2-descriptor "$output_dir/siglip2.descriptor.bin" \
  --siglip2-queries "$output_dir/siglip2.queries.json" \
  --selecting-receipt "$output_dir/selecting.receipt.json" \
  --selecting-descriptor "$output_dir/selecting.descriptor.bin" \
  --selecting-queries "$output_dir/selecting.queries.json" \
  --error-manifest "$error_manifest" \
  --m3-seed-receipt "$m3_seed_017" \
  --m3-seed-receipt "$m3_seed_029" \
  --m3-seed-receipt "$m3_seed_043" \
  --m3-aggregate "$m3_aggregate" \
  --m4-output "$output_dir/m4.receipt.json" \
  --adapter-output "$output_dir/adapter.receipt.json" \
  --source-revision "$source_revision" \
  --execute
write_campaign_terminal
