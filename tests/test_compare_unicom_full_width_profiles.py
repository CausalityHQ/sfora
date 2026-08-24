from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "compare_unicom_full_width_profiles.py"
PROFILER_PATH = ROOT / "scripts" / "profile_unicom_training_step.py"
TRAINER_PATH = ROOT / "scripts" / "train_unicom_inshop.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load(MODULE_PATH, "compare_unicom_full_width_profiles")
PROFILER = _load(PROFILER_PATH, "profile_unicom_training_step_for_comparison")
TRAINER = _load(TRAINER_PATH, "train_unicom_inshop_for_comparison")


def _timing_sample(wall: float, objective: float) -> dict[str, float]:
    components = {
        "h2d_seconds": 0.01,
        "zero_grad_seconds": 0.01,
        "backbone_forward_seconds": 0.20,
        "objective_forward_seconds": objective * 0.6,
        "head_backward_seconds": objective * 0.4,
        "backbone_backward_seconds": 0.40,
        "update_seconds": 0.20,
        "tail_seconds": 0.01,
    }
    return {
        "step_wall_seconds": wall,
        "cuda_step_seconds": math.fsum(components.values()),
        **components,
    }


def _profile(
    *, checkpoint_sha256: str, started: int, wall: float, objective: float
) -> dict[str, object]:
    timing = [_timing_sample(wall, objective) for _ in range(50)]
    fusible = [objective * 0.5 for _ in range(10)]
    return {
        "schema_version": "unicom-training-step-profile-v1",
        "run_checkpoint": f"/run/{checkpoint_sha256[:8]}.pt",
        "run_checkpoint_sha256": checkpoint_sha256,
        "trainer_sha256": "1" * 64,
        "objective_sha256": "2" * 64,
        "profiler_sha256": "3" * 64,
        "checkpoint_epoch": 16,
        "started_unix_ns": started,
        "finished_unix_ns": started + 10,
        "classifier_init": "imprinted",
        "warmup_steps": 20,
        "measure_steps": 50,
        "profiler_steps": 10,
        "timing_samples": timing,
        "fusible_samples": fusible,
        "summary": PROFILER.summarize_profile(tuple(timing), tuple(fusible)),
        "runtime": {
            "python_version": "3.12.3",
            "torch_version": "2.12.1",
            "numpy_version": "2.5.0",
            "cuda_version": "13.0",
            "device_name": "NVIDIA GB10",
        },
    }


def _receipt(
    tmp_path: Path,
    *,
    arm: str,
    checkpoint_sha256: str,
    allocated: int,
    reserved: int,
) -> dict[str, object]:
    selected = 512 if arm == "sampled_512" else 768
    return {
        "schema_version": "unicom-full-width-training-run-v1",
        "source_commit": "a" * 40,
        "trainer_sha256": "1" * 64,
        "config_path": str(tmp_path / "config.json"),
        "config_sha256": "b" * 64,
        "seed": 0,
        "arm": arm,
        "protocol": {
            "objective": "official-eight-mask",
            "selected_features": selected,
            "evaluation_features": 768,
        },
        "command": ["python", "train.py"],
        "started_unix_ns": 1,
        "finished_unix_ns": 2,
        "elapsed_seconds": 1.0,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "exit_status": 0,
        "history": {"path": "history.json", "sha256": "4" * 64, "bytes": 10},
        "checkpoints": [
            {
                "epoch": epoch,
                "path": f"epoch-{epoch:04d}.pt",
                "sha256": checkpoint_sha256 if epoch == 16 else f"{epoch // 4 + 4:x}" * 64,
                "bytes": 100,
            }
            for epoch in (4, 8, 12, 16)
        ],
        "runtime": {"python": "3.12.3", "torch": "2.12.1", "cuda": "13.0"},
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _abba_fixture(tmp_path: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    profiles: list[Path] = []
    receipts: list[Path] = []
    rows = (
        ("sampled_512", "5" * 64, 10, 1.00, 0.10, 100, 200),
        ("full_768", "6" * 64, 20, 1.01, 0.11, 101, 202),
        ("full_768", "7" * 64, 30, 1.03, 0.13, 103, 204),
        ("sampled_512", "8" * 64, 40, 1.02, 0.12, 102, 202),
    )
    for index, (arm, checkpoint, started, wall, objective, allocated, reserved) in enumerate(rows):
        profile_path = tmp_path / f"profile-{index}.json"
        receipt_path = tmp_path / f"receipt-{index}.json"
        _write(
            profile_path,
            _profile(
                checkpoint_sha256=checkpoint,
                started=started,
                wall=wall,
                objective=objective,
            ),
        )
        _write(
            receipt_path,
            _receipt(
                tmp_path,
                arm=arm,
                checkpoint_sha256=checkpoint,
                allocated=allocated,
                reserved=reserved,
            ),
        )
        profiles.append(profile_path)
        receipts.append(receipt_path)
    return tuple(profiles), tuple(receipts)


def test_compare_abba_binds_order_provenance_samples_and_position_adjusted_ratios(
    tmp_path: Path,
) -> None:
    profile_paths, receipt_paths = _abba_fixture(tmp_path)

    result = MODULE.compare_abba(profile_paths, receipt_paths)

    assert result["arms"] == ["sampled_512", "full_768", "full_768", "sampled_512"]
    assert result["ratios"]["step_wall"] == pytest.approx(2.04 / 2.02)
    assert result["ratios"]["cuda_step"] == pytest.approx(1.90 / 1.88)
    assert result["ratios"]["objective_ceiling"] == pytest.approx(0.24 / 0.22)
    assert result["ratios"]["peak_allocated"] == pytest.approx(204 / 202)
    assert result["ratios"]["peak_reserved"] == pytest.approx(406 / 402)
    assert result["checkpoint_bytes_equal"] is True
    assert result["source_commit"] == "a" * 40
    assert result["config_sha256"] == "b" * 64
    assert result["profile_sha256s"] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in profile_paths
    ]


@pytest.mark.parametrize(
    "mutation",
    ("arm_order", "checkpoint_binding", "duplicate_profile", "runtime", "sample_count"),
)
def test_compare_abba_rejects_order_provenance_and_sample_drift(
    tmp_path: Path, mutation: str
) -> None:
    profile_paths, receipt_paths = _abba_fixture(tmp_path)
    if mutation == "arm_order":
        receipt_paths = (receipt_paths[1], receipt_paths[0], *receipt_paths[2:])
    elif mutation == "checkpoint_binding":
        value = json.loads(receipt_paths[1].read_text())
        value["checkpoints"][-1]["sha256"] = "9" * 64
        _write(receipt_paths[1], value)
    elif mutation == "duplicate_profile":
        profile_paths = (profile_paths[0], profile_paths[1], profile_paths[1], profile_paths[3])
    elif mutation == "runtime":
        value = json.loads(receipt_paths[2].read_text())
        value["runtime"]["torch"] = "different"
        _write(receipt_paths[2], value)
    else:
        value = json.loads(profile_paths[2].read_text())
        value["timing_samples"].pop()
        _write(profile_paths[2], value)

    with pytest.raises((TypeError, ValueError)):
        MODULE.compare_abba(profile_paths, receipt_paths)


def test_comparison_publication_is_strict_atomic_and_no_clobber(tmp_path: Path) -> None:
    profile_paths, receipt_paths = _abba_fixture(tmp_path)
    result = MODULE.compare_abba(profile_paths, receipt_paths)
    output = tmp_path / "comparison.json"

    MODULE.write_json_atomic(output, result)

    assert json.loads(output.read_text(encoding="utf-8")) == result
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        MODULE.write_json_atomic(output, result)
    assert output.read_bytes() == original
    assert not tuple(tmp_path.glob(".comparison.json.*.tmp"))


def test_comparison_publication_rejects_invalid_result_without_output(tmp_path: Path) -> None:
    profile_paths, receipt_paths = _abba_fixture(tmp_path)
    result = MODULE.compare_abba(profile_paths, receipt_paths)
    result["ratios"].pop("step_wall")
    output = tmp_path / "comparison.json"

    with pytest.raises(ValueError, match="result"):
        MODULE.write_json_atomic(output, result)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".comparison.json.*.tmp"))


def test_comparison_publication_rolls_back_owned_link_on_directory_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_paths, receipt_paths = _abba_fixture(tmp_path)
    result = MODULE.compare_abba(profile_paths, receipt_paths)
    output = tmp_path / "comparison.json"
    real_fsync = MODULE.os.fsync
    calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(MODULE.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="directory fsync"):
        MODULE.write_json_atomic(output, result)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".comparison.json.*.tmp"))


def test_comparison_cli_publishes_once_and_preserves_first_result(tmp_path: Path) -> None:
    profile_paths, receipt_paths = _abba_fixture(tmp_path)
    output = tmp_path / "comparison.json"
    arguments = [
        "--profiles",
        *(str(path) for path in profile_paths),
        "--receipts",
        *(str(path) for path in receipt_paths),
        "--output",
        str(output),
    ]

    assert MODULE.main(arguments) == 0
    original = output.read_bytes()
    assert MODULE.main(arguments) == 2
    assert output.read_bytes() == original
