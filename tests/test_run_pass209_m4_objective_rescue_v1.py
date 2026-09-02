"""Contract tests for the serialized Pass209 M4 DGX campaign."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

_RUNNER = Path(__file__).parents[1] / "scripts" / "run_pass209_m4_objective_rescue_v1.sh"


def test_runner_is_source_bound_offline_and_serializes_all_science() -> None:
    text = _RUNNER.read_text()

    subprocess.run(["bash", "-n", str(_RUNNER)], check=True)
    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "SOURCE_REVISION" in text and "SOURCE_MANIFEST" in text
    assert "sha256sum --check --strict" in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "HF_DATASETS_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    scorer = text.index("self-test")
    assert "torch.eye(4, dtype=torch.float32)" in text
    assert "for i in range(4)" in text
    dinov2 = text.index("--cell dinov2-large")
    siglip2 = text.index("--cell siglip2-so400m")
    selecting = text.index("--cell siglip-so400m")
    analyzer = text.index("analyze_pass209_m4.py")
    assert scorer < dinov2 < siglip2 < selecting < analyzer
    assert "wait_for_control_release" in text
    for process in (
        "[r]un_siglip_proxy_control.py",
        "[r]un_native_twin_probe.py",
        "[a]udit_siglip_control_checkpoint.py",
        "[p]robe_frozen_substrate.py",
    ):
        assert process in text
    assert "refusing to duplicate a live M4 campaign" in text
    assert "combined_resource_bytes" in text
    assert "68719476736" in text
    assert "memory_psi_full_avg10" in text and "0.50" in text
    assert "swap_delta_kib" in text and "262144" in text
    assert "progress_age_seconds" in text and "1200" in text
    assert "rm -rf" not in text
    assert 'cars", split="test' not in text


def _fake_campaign(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / _RUNNER.name
    runner.write_bytes(_RUNNER.read_bytes())
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    revision = "a" * 40
    (repo / "SOURCE_REVISION").write_text(f"{revision}\n")
    (repo / "uv.lock").write_text("fixture-lock\n")
    manifest = repo / "SOURCE_MANIFEST.sha256"
    lines = []
    for relative in (
        "SOURCE_REVISION",
        "scripts/run_pass209_m4_objective_rescue_v1.sh",
        "uv.lock",
    ):
        digest = __import__("hashlib").sha256((repo / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}\n")
    manifest.write_text("".join(lines))

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv_log = tmp_path / "uv.log"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >>"$FAKE_UV_LOG"
if [[ -n ${FAKE_FAIL_TOKEN:-} && "$*" == *"$FAKE_FAIL_TOKEN"* ]]; then
  exit 19
fi
receipt= descriptor= queries= checkpoint= m4= adapter= cell= source_revision= source_tree_digest=
while (($#)); do
  case "$1" in
    --cell) cell=$2; shift 2;;
    --receipt-output) receipt=$2; shift 2;;
    --descriptor-output) descriptor=$2; shift 2;;
    --query-output) queries=$2; shift 2;;
    --checkpoint-dir) checkpoint=$2; shift 2;;
    --source-revision) source_revision=$2; shift 2;;
    --source-tree-digest) source_tree_digest=$2; shift 2;;
    --m4-output) m4=$2; shift 2;;
    --adapter-output) adapter=$2; shift 2;;
    *) shift;;
  esac
done
if [[ -n $receipt ]]; then
  if [[ -n ${FAKE_STDERR_WARNING:-} ]]; then
    printf '%s\\n' "$FAKE_STDERR_WARNING" >&2
    if [[ $cell == dinov2-large && -n ${FAKE_PRE_PROGRESS_SLEEP:-} ]]; then
      sleep "$FAKE_PRE_PROGRESS_SLEEP"
    fi
  fi
  mkdir -p "$checkpoint"
  printf 'authenticated-checkpoint\\n' >"$checkpoint/checkpoint-0001.bin"
  checkpoint_sha=$(sha256sum "$checkpoint/checkpoint-0001.bin" | awk '{print $1}')
  checkpoint_sha=${FAKE_CHECKPOINT_SHA256:-$checkpoint_sha}
  progress_format='{"cell":"%s","checkpoint_sha256":"%s"'
  progress_format+=',"cuda_peak_reserved_bytes":%s,"rows":1'
  progress_format+=',"schema":"sfora-pass209-m4-progress-v1"'
  progress_format+=',"source_revision":"%s","source_tree_digest":"%s"}\\n'
  printf "$progress_format" \
    "$cell" "$checkpoint_sha" "${FAKE_CUDA_PEAK_BYTES:-0}" \
    "$source_revision" "$source_tree_digest" >&2
  if [[ -n ${FAKE_UV_SLEEP:-} ]]; then
    sleep "$FAKE_UV_SLEEP"
  fi
  printf '{"fixture":true}\\n' >"$receipt"
  printf 'descriptor\\n' >"$descriptor"
  printf '{"fixture":true}\\n' >"$queries"
fi
if [[ -n $m4 ]]; then
  printf '{"fixture":true}\\n' >"$m4"
  printf '{"fixture":true}\\n' >"$adapter"
fi
"""
    )
    fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IXUSR)
    fake_smi = fake_bin / "nvidia-smi"
    fake_smi.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"${FAKE_GPU_MEMORY_MIB:-0}\"\n")
    fake_smi.chmod(fake_smi.stat().st_mode | stat.S_IXUSR)
    fake_awk = fake_bin / "awk"
    fake_awk.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ${!#} == /proc/pressure/memory ]]; then printf '0.00\\n'; "
        'else exec /usr/bin/awk "$@"; fi\n'
    )
    fake_awk.chmod(fake_awk.stat().st_mode | stat.S_IXUSR)

    error_manifest = tmp_path / "errors.json"
    error_manifest.write_text("{}\n")
    placeholders = {}
    for name in (
        "DINOV2_PREREQUISITE",
        "SIGLIP2_PREREQUISITE",
        "SELECTING_PREREQUISITE",
        "M3_SEED_017",
        "M3_SEED_029",
        "M3_SEED_043",
        "M3_AGGREGATE",
    ):
        path = tmp_path / f"{name.lower()}.json"
        path.write_text("{}\n")
        placeholders[name] = str(path)
    output = tmp_path / "output"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_UV_LOG": str(uv_log),
        "SOURCE_REVISION": revision,
        "SOURCE_MANIFEST": str(manifest),
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv"),
        "OUTPUT_DIR": str(output),
        "ERROR_MANIFEST": str(error_manifest),
        **placeholders,
    }
    return runner, environment, uv_log


def test_runner_executes_exact_cells_then_analyzer_in_separate_processes(
    tmp_path: Path,
) -> None:
    runner, environment, uv_log = _fake_campaign(tmp_path)
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = uv_log.read_text().splitlines()
    cells = [
        next(
            part
            for part in call.split()
            if part in {"dinov2-large", "siglip2-so400m", "siglip-so400m"}
        )
        for call in calls
        if "run_pass209_m4_cell.py" in call
    ]
    assert cells == ["dinov2-large", "siglip2-so400m", "siglip-so400m"]
    assert sum("analyze_pass209_m4.py" in call for call in calls) == 1


def test_runner_rejects_unmanifested_source_files(tmp_path: Path) -> None:
    runner, environment, uv_log = _fake_campaign(tmp_path)
    (runner.parents[1] / "unexpected.py").write_text("raise SystemExit\n")

    result = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not uv_log.exists()


def test_runner_rejects_registered_source_digest_drift(tmp_path: Path) -> None:
    runner, environment, uv_log = _fake_campaign(tmp_path)
    (runner.parents[1] / "uv.lock").write_text("drifted-lock\n")

    result = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not uv_log.exists()


def test_runner_stops_when_offline_model_cache_preflight_fails(tmp_path: Path) -> None:
    runner, environment, uv_log = _fake_campaign(tmp_path)
    environment["FAKE_FAIL_TOKEN"] = "python -"

    result = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 19
    calls = uv_log.read_text().splitlines()
    assert len(calls) == 1 and calls[0].endswith("python -")
    output = Path(environment["OUTPUT_DIR"])
    stop = __import__("json").loads((output / "m4-operational-stop.json").read_bytes())
    assert stop["stage"] == "preflight"
    assert not (output / "adapter.receipt.json").exists()


@pytest.mark.parametrize("unexpected", ("unexpected.json", ".stale.partial"))
def test_runner_rejects_an_existing_unregistered_output_namespace(
    tmp_path: Path, unexpected: str
) -> None:
    runner, environment, uv_log = _fake_campaign(tmp_path)
    output = Path(environment["OUTPUT_DIR"])
    output.mkdir()
    (output / unexpected).write_text("unexpected\n")

    result = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 73
    assert not uv_log.exists()


def test_runner_refuses_to_overlap_the_live_control(tmp_path: Path) -> None:
    runner, environment, uv_log = _fake_campaign(tmp_path)
    control = subprocess.Popen(
        ["bash", "-c", "exec -a 'run_siglip_proxy_control.py train' sleep 30"]
    )
    try:
        result = subprocess.run(
            ["bash", str(runner)],
            cwd=runner.parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
    finally:
        control.terminate()
        control.wait(timeout=5)

    assert result.returncode == 75
    assert "control has not released" in result.stderr
    assert not uv_log.exists()


@pytest.mark.parametrize(
    ("token", "expected_cells", "failed_stage"),
    [
        ("dinov2-large", [], "dinov2-large"),
        ("siglip2-so400m", ["dinov2-large"], "siglip2-so400m"),
        (
            "siglip-so400m",
            ["dinov2-large", "siglip2-so400m"],
            "siglip-so400m",
        ),
        (
            "analyze_pass209_m4.py",
            ["dinov2-large", "siglip2-so400m", "siglip-so400m"],
            "analyzer",
        ),
    ],
)
def test_runner_stops_at_first_failure_and_emits_no_adapter(
    tmp_path: Path,
    token: str,
    expected_cells: list[str],
    failed_stage: str,
) -> None:
    runner, environment, uv_log = _fake_campaign(tmp_path)
    environment["FAKE_FAIL_TOKEN"] = token
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    calls = uv_log.read_text().splitlines()
    successful_cells = [
        cell
        for cell in ("dinov2-large", "siglip2-so400m", "siglip-so400m")
        if any("run_pass209_m4_cell.py" in call and cell in call for call in calls)
        and cell != token
    ]
    assert successful_cells == expected_cells
    output = Path(environment["OUTPUT_DIR"])
    assert not (output / "adapter.receipt.json").exists()
    stop = __import__("json").loads((output / "m4-operational-stop.json").read_bytes())
    assert stop["status"] == "failed"
    assert stop["consequence"] == "F4-NONE"
    assert stop["stage"] == failed_stage


def test_runner_stops_the_process_group_at_the_combined_resource_limit(
    tmp_path: Path,
) -> None:
    runner, environment, _ = _fake_campaign(tmp_path)
    environment["FAKE_CUDA_PEAK_BYTES"] = str(70 * 1024**3)
    environment["FAKE_UV_SLEEP"] = "30"
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 125
    assert "M4 STOP reason=combined-resource" in result.stderr
    output = Path(environment["OUTPUT_DIR"])
    stop = __import__("json").loads((output / "m4-operational-stop.json").read_bytes())
    assert stop["stage"] == "dinov2-large"
    assert stop["reason"] == "combined-resource"
    assert not (output / "adapter.receipt.json").exists()


def test_runner_rejects_progress_not_bound_to_the_checkpoint(tmp_path: Path) -> None:
    runner, environment, _ = _fake_campaign(tmp_path)
    environment["FAKE_CHECKPOINT_SHA256"] = "0" * 64
    environment["FAKE_UV_SLEEP"] = "30"
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 125
    assert "M4 STOP reason=progress-authority" in result.stderr
    output = Path(environment["OUTPUT_DIR"])
    stop = __import__("json").loads((output / "m4-operational-stop.json").read_bytes())
    assert stop["reason"] == "progress-authority"
    assert not (output / "adapter.receipt.json").exists()


def test_runner_tolerates_library_stderr_before_first_authenticated_progress(
    tmp_path: Path,
) -> None:
    runner, environment, _ = _fake_campaign(tmp_path)
    environment["FAKE_STDERR_WARNING"] = "Using a slow image processor"
    environment["FAKE_PRE_PROGRESS_SLEEP"] = "6"
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    output = Path(environment["OUTPUT_DIR"])
    assert (output / "adapter.receipt.json").is_file()
    assert (output / "campaign-terminal.json").is_file()


def test_resource_resume_rejects_an_extra_namespace_file(tmp_path: Path) -> None:
    runner, environment, _ = _fake_campaign(tmp_path)
    environment["FAKE_CUDA_PEAK_BYTES"] = str(70 * 1024**3)
    environment["FAKE_UV_SLEEP"] = "30"
    stopped = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert stopped.returncode == 125
    output = Path(environment["OUTPUT_DIR"])
    (output / "unexpected.json").write_text("unexpected\n")

    environment.pop("FAKE_CUDA_PEAK_BYTES")
    environment.pop("FAKE_UV_SLEEP")
    resumed = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert resumed.returncode == 73
    assert not (output / "adapter.receipt.json").exists()


def test_resource_stopped_campaign_resumes_same_cell_from_checkpoint(
    tmp_path: Path,
) -> None:
    runner, environment, _ = _fake_campaign(tmp_path)
    environment["FAKE_CUDA_PEAK_BYTES"] = str(70 * 1024**3)
    environment["FAKE_UV_SLEEP"] = "30"
    stopped = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert stopped.returncode == 125

    environment.pop("FAKE_CUDA_PEAK_BYTES")
    environment.pop("FAKE_UV_SLEEP")
    resumed = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resumed.returncode == 0, resumed.stderr
    output = Path(environment["OUTPUT_DIR"])
    assert (output / "adapter.receipt.json").is_file()
    assert not (output / "m4-operational-stop.json").exists()
    histories = tuple(output.glob("m4-operational-stop-history-*.json"))
    assert len(histories) == 1
    terminal = __import__("json").loads((output / "campaign-terminal.json").read_bytes())
    assert terminal["status"] == "complete"
    assert len(terminal["operational_stop_history"]) == 1


def test_a_later_failure_replaces_current_stop_without_laundering_history(
    tmp_path: Path,
) -> None:
    runner, environment, _ = _fake_campaign(tmp_path)
    environment["FAKE_CUDA_PEAK_BYTES"] = str(70 * 1024**3)
    environment["FAKE_UV_SLEEP"] = "30"
    first = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert first.returncode == 125

    environment.pop("FAKE_CUDA_PEAK_BYTES")
    environment.pop("FAKE_UV_SLEEP")
    environment["FAKE_FAIL_TOKEN"] = "siglip2-so400m"
    second = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert second.returncode == 19
    output = Path(environment["OUTPUT_DIR"])
    stop = __import__("json").loads((output / "m4-operational-stop.json").read_bytes())
    assert stop["stage"] == "siglip2-so400m"
    assert stop["exit_status"] == 19
    assert len(tuple(output.glob("m4-operational-stop-history-*.json"))) == 1

    environment.pop("FAKE_FAIL_TOKEN")
    third = subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert third.returncode == 73
    assert not (output / "adapter.receipt.json").exists()
