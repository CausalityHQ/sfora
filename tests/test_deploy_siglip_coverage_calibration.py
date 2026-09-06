"""Static contract for the guarded SigLIP coverage-calibration deployment."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "deploy_siglip_coverage_calibration_v1.sh"
)


def test_coverage_deployment_binds_exact_source_and_local_only_inputs() -> None:
    source = _SCRIPT.read_text()
    runner = (_SCRIPT.parent / "run_siglip_coverage_calibration_two_phase.sh").read_text()
    assert os.access(_SCRIPT, os.X_OK)
    assert "git bundle create" in source
    assert "git diff --quiet HEAD -- src/sfora scripts" in source
    for path in (
        "src/sfora/siglip_coverage_calibration.py",
        "scripts/prepare_siglip_coverage_calibration.py",
        "scripts/probe_siglip_coverage_calibration.py",
        "scripts/deploy_siglip_coverage_calibration_v1.sh",
        "scripts/run_siglip_coverage_calibration_two_phase.sh",
        "scripts/landlock_exec.c",
    ):
        assert path in source
    assert "--execute-coverage-calibration" in runner
    assert '--execution-source-commit "$revision"' in runner
    assert "HF_HUB_OFFLINE=1" in source
    assert '"$landlock" "${common_landlock[@]}"' in runner
    assert "--ro /usr --ro /etc" in runner
    assert "input_staging=$output.inputs.partial" in source
    assert "optimization_images=$input_staging/optimization-images" in source
    assert "support_images=$input_staging/support-images" in source
    assert "heldout_images=$input_staging/heldout-images" in source
    assert '--ro "$input_staging"' not in source
    assert '--rw "$staging"' not in source
    assert 'test ! -e "$input_staging" || rm -rf -- "$input_staging"' in source
    for forbidden in ("aws s3", "class-names", "--support-per-class", "--rcond"):
        assert forbidden not in source
    preflight = source.split("<<'PREFLIGHT'", 1)[1].split("PREFLIGHT", 1)[0]
    assert 'output_parent=$(dirname "$2")' in preflight
    assert 'mkdir -p "$output_parent"' in preflight
    assert 'test -d "$output_parent"' in preflight
    for seed in (17, 29, 43):
        assert f"seed-{seed:03d}/checkpoints/seed-{seed:03d}-epoch-060.pt" in preflight
        assert f"seed-{seed:03d}.receipt.json" in preflight


def test_coverage_deployment_owns_one_monitored_process_and_never_restarts() -> None:
    source = _SCRIPT.read_text()
    assert "timeout --signal=TERM --kill-after=30s 5400s" in source
    assert "coverage calibration process is already active" in source
    assert source.count("run_siglip_coverage_calibration_two_phase.sh") >= 2
    assert source.count("setsid timeout --signal=TERM --kill-after=30s 5400s") == 1
    assert 'kill -TERM -- "-$child"' in source
    assert 'pgrep -g "$target_group"' in source
    assert 'kill -KILL -- "-$target_group"' in source
    for reason in (
        "rss-cap",
        "gpu-memory-cap",
        "gpu-telemetry",
        "psi-immediate",
        "psi-sustained",
        "swap-delta",
        "progress-gap",
    ):
        assert f"stop_reason={reason}" in source
    assert "((rss <= 118111600640))" in source
    assert "((gpu_mib <= 98304))" in source
    assert "((swap <= swap0))" in source
    assert "swap-swap0 <= 262144" not in source
    assert 'test -z "$(nvidia-smi' not in source
    assert source.count("gpu_processes=$(nvidia-smi") == 2
    assert "load_control_examples" not in source
    monitored = source.index("setsid timeout --signal=TERM --kill-after=30s 5400s")
    assert monitored < source.index(
        "scripts/run_siglip_coverage_calibration_two_phase.sh", monitored
    )


def test_coverage_deployment_revalidates_canonical_result_and_map() -> None:
    source = _SCRIPT.read_text()
    assert "validate_coverage_calibration_result_bytes" in source
    assert 'value["claim_eligible"] is False' in source
    assert 'value["execution_source_commit"] == sys.argv[2]' in source
    assert 'value["inputs"]["map_artifact"]["sha256"] == sys.argv[3]' in source
    assert 'value["image_namespaces"]["support"]["sha256"]' in source
    assert 'value["image_namespaces"]["evaluation"]["sha256"]' in source
    assert 'len(value["support_ids"]) == 264' in source
    assert 'set(value["support_labels"]) == set(range(49, 82))' in source
    assert 'solver["dimensions"] == 512' in source
    assert 'solver["rank"] == 513' in source
    assert "with safe_open" in source
    assert 'metadata["optimization_images_sha256"]' in source
    assert 'metadata["support_images_sha256"]' in source
    assert "_ids_sha256" in source
    assert 'rsync -a -- "$remote_host:$remote_output/phase2/result.json" "$local_result"' in source
    assert (
        'rsync -a -- "$remote_host:$remote_output/phase1/maps.safetensors" "$local_artifact"'
        in source
    )
    assert "fit-receipt.json" in source
    assert 'sha256sum "$local_result" "$local_artifact" "$local_fit_receipt"' in source


def test_coverage_deployment_uses_single_operand_unlink() -> None:
    source = _SCRIPT.read_text()
    unlink_commands = [
        shlex.split(line.strip())
        for line in source.splitlines()
        if line.strip().startswith("unlink ")
    ]
    assert unlink_commands
    assert all(len(command) == 2 for command in unlink_commands)


def test_coverage_deployment_preserves_terminal_receipt_on_every_exit() -> None:
    source = _SCRIPT.read_text()
    assert "set +e" not in source.split("<<'PREFLIGHT'", 1)[0]
    assert "set +e" in source.split("PREFLIGHT", 2)[2].split("<<'REMOTE'", 1)[0]
    assert "execution_receipt=$output.execution.json" in source
    assert "write_execution_receipt()" in source
    assert '"schema": "sfora-siglip-coverage-execution-v1"' in source
    assert '"claim_eligible": False' in source
    assert '"group_drained": sys.argv[7] == "true"' in source
    assert "validate_coverage_execution_receipt_bytes" in source
    assert "remote_status=$?" in source
    assert source.index("remote_status=$?") < source.index(
        'rsync -a -- "$remote_host:$remote_output.execution.json"'
    )
    assert '((remote_status == 0)) || exit "$remote_status"' in source
    preflight = source.split("<<'PREFLIGHT'", 1)[1].split("PREFLIGHT", 1)[0]
    assert 'test ! -e "$2.execution.json"' in preflight
    assert "os.replace(partial, target)" not in source
    assert "os.link(partial, target)" in source
    assert "drain_group()" in source
    assert "staging_owned=0" in source
    assert "run_owned=0" in source
    assert "input_staging_owned=0" in source
    assert 'if [[ "$run_owned" != 1 ]]; then return; fi' in source
    assert 'mkdir "$staging"\nstaging_owned=1' in source
    assert 'mkdir "$input_staging"\ninput_staging_owned=1' in source
    assert 'if [[ "$staging_owned" = 1 && -e "$staging" ]]' in source
    assert 'if [[ "$input_staging_owned" = 1 && -e "$input_staging" ]]' in source
    assert 'if [[ "$group_drained" = false ]]; then return; fi' in source
    assert "drain_attempted=0" in source
    assert 'if [[ "$drain_attempted" = 1 ]]; then return; fi' in source
    assert 'test ! -e "$staging/preparation.complete" || failed_phase=fit' in source
    assert "trap 'stop_reason=signal-int; exit 130' INT" in source
    assert "trap 'stop_reason=signal-term; exit 143' TERM" in source
    assert "trap 'cleanup_remote; exit" not in source


def test_coverage_deployment_preserves_receipt_paths_after_success() -> None:
    source = _SCRIPT.read_text()
    remote = source.split("<<'REMOTE'", 1)[1].split("REMOTE", 1)[0]
    assert "staging=$output\n" in remote
    assert 'mv "$staging" "$output"' not in remote
    assert "phase1=$output/phase1" in remote
    assert "phase2=$output/phase2" in remote
    assert "authority=$output/authority" in remote
    assert "evaluation_manifest=$authority/evaluation-manifest.json" in remote
    assert 'rm -rf -- "$authority"' not in remote
