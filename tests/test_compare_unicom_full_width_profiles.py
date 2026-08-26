from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import stat
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
    rows = (
        ("5" * 64, 10, 1.00, 0.10),
        ("6" * 64, 20, 1.01, 0.11),
        ("6" * 64, 30, 1.03, 0.13),
        ("5" * 64, 40, 1.02, 0.12),
    )
    for index, (checkpoint, started, wall, objective) in enumerate(rows):
        profile_path = tmp_path / f"profile-{index}.json"
        _write(
            profile_path,
            _profile(
                checkpoint_sha256=checkpoint,
                started=started,
                wall=wall,
                objective=objective,
            ),
        )
        profiles.append(profile_path)

    receipts: list[Path] = []
    for arm, checkpoint, allocated, reserved in (
        ("sampled_512", "5" * 64, 100, 200),
        ("full_768", "6" * 64, 101, 202),
    ):
        receipt_path = tmp_path / f"receipt-{arm}.json"
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
        receipts.append(receipt_path)
    return tuple(profiles), (receipts[0], receipts[1], receipts[1], receipts[0])


def test_compare_abba_reuses_exact_two_training_receipts(tmp_path: Path) -> None:
    profiles: list[Path] = []
    for index, (checkpoint, wall, objective) in enumerate(
        (
            ("5" * 64, 1.00, 0.10),
            ("6" * 64, 1.01, 0.11),
            ("6" * 64, 1.03, 0.13),
            ("5" * 64, 1.02, 0.12),
        )
    ):
        path = tmp_path / f"profile-{index}.json"
        _write(
            path,
            _profile(
                checkpoint_sha256=checkpoint,
                started=10 * (index + 1),
                wall=wall,
                objective=objective,
            ),
        )
        profiles.append(path)

    control = tmp_path / "control-receipt.json"
    candidate = tmp_path / "candidate-receipt.json"
    _write(
        control,
        _receipt(
            tmp_path,
            arm="sampled_512",
            checkpoint_sha256="5" * 64,
            allocated=100,
            reserved=200,
        ),
    )
    _write(
        candidate,
        _receipt(
            tmp_path,
            arm="full_768",
            checkpoint_sha256="6" * 64,
            allocated=102,
            reserved=204,
        ),
    )
    receipt_paths = (control, candidate, candidate, control)

    result = MODULE.compare_abba(tuple(profiles), receipt_paths)

    control_sha = hashlib.sha256(control.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert result["receipt_sha256s"] == [
        control_sha,
        candidate_sha,
        candidate_sha,
        control_sha,
    ]


def test_compare_abba_rejects_four_distinct_training_receipts(tmp_path: Path) -> None:
    profile_paths, repeated_receipts = _abba_fixture(tmp_path)
    receipt_paths: list[Path] = []
    for index, source in enumerate(repeated_receipts):
        value = json.loads(source.read_text(encoding="utf-8"))
        value["started_unix_ns"] = 10 + index * 2
        value["finished_unix_ns"] = 11 + index * 2
        path = tmp_path / f"distinct-receipt-{index}.json"
        _write(path, value)
        receipt_paths.append(path)

    with pytest.raises(ValueError, match="identity pattern"):
        MODULE.compare_abba(profile_paths, tuple(receipt_paths))


@pytest.mark.parametrize(
    "receipt_hashes",
    (
        ["1" * 64, "2" * 64, "3" * 64, "4" * 64],
        ["1" * 64, "1" * 64, "2" * 64, "2" * 64],
        ["1" * 64, "1" * 64, "1" * 64, "1" * 64],
    ),
)
def test_comparison_result_rejects_non_abba_receipt_identity_pattern(
    tmp_path: Path, receipt_hashes: list[str]
) -> None:
    profile_paths, receipt_paths = _abba_fixture(tmp_path)
    result = MODULE.compare_abba(profile_paths, receipt_paths)
    result["receipt_sha256s"] = receipt_hashes

    with pytest.raises(ValueError, match="binding"):
        MODULE.validate_comparison_result(result)


def test_compare_abba_binds_order_provenance_samples_and_position_adjusted_ratios(
    tmp_path: Path,
) -> None:
    profile_paths, receipt_paths = _abba_fixture(tmp_path)

    result = MODULE.compare_abba(profile_paths, receipt_paths)

    assert result["arms"] == ["sampled_512", "full_768", "full_768", "sampled_512"]
    assert result["ratios"]["step_wall"] == pytest.approx(2.04 / 2.02)
    assert result["ratios"]["cuda_step"] == pytest.approx(1.90 / 1.88)
    assert result["ratios"]["objective_ceiling"] == pytest.approx(0.24 / 0.22)
    assert result["ratios"]["peak_allocated"] == pytest.approx(202 / 200)
    assert result["ratios"]["peak_reserved"] == pytest.approx(404 / 400)
    assert result["checkpoint_bytes_equal"] is True
    assert result["source_commit"] == "a" * 40
    assert result["config_sha256"] == "b" * 64
    assert result["profile_sha256s"] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in profile_paths
    ]


@pytest.mark.parametrize(
    "mutation",
    (
        "checkpoint_binding",
        "duplicate_profile",
        "runtime",
        "sample_count",
        "profiler_source",
        "objective_source",
    ),
)
def test_compare_abba_rejects_order_provenance_and_sample_drift(
    tmp_path: Path, mutation: str
) -> None:
    profile_paths, receipt_paths = _abba_fixture(tmp_path)
    if mutation == "checkpoint_binding":
        value = json.loads(receipt_paths[1].read_text())
        value["checkpoints"][-1]["sha256"] = "9" * 64
        _write(receipt_paths[1], value)
    elif mutation == "duplicate_profile":
        profile_paths = (profile_paths[0], profile_paths[1], profile_paths[1], profile_paths[3])
    elif mutation == "runtime":
        value = json.loads(receipt_paths[2].read_text())
        value["runtime"]["torch"] = "different"
        _write(receipt_paths[2], value)
    elif mutation == "sample_count":
        value = json.loads(profile_paths[2].read_text())
        value["timing_samples"].pop()
        _write(profile_paths[2], value)
    else:
        value = json.loads(profile_paths[2].read_text())
        value[f"{mutation.removesuffix('_source')}_sha256"] = "9" * 64
        _write(profile_paths[2], value)

    with pytest.raises((TypeError, ValueError)):
        MODULE.compare_abba(profile_paths, receipt_paths)


def test_compare_abba_rejects_transposed_arms_after_valid_receipt_identity_pattern(
    tmp_path: Path,
) -> None:
    profile_paths, receipt_paths = _abba_fixture(tmp_path)
    transposed = (
        receipt_paths[1],
        receipt_paths[0],
        receipt_paths[0],
        receipt_paths[1],
    )

    with pytest.raises(ValueError, match="arm order"):
        MODULE.compare_abba(profile_paths, transposed)


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


@pytest.mark.parametrize(
    "failure_phase",
    ("directory_open", "first_directory_fsync", "temporary_unlink", "final_directory_fsync"),
)
def test_comparison_publication_preserves_valid_link_after_postlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_phase: str
) -> None:
    profile_paths, receipt_paths = _abba_fixture(tmp_path)
    result = MODULE.compare_abba(profile_paths, receipt_paths)
    output = tmp_path / "comparison.json"
    temporary = output.with_name(f".{output.name}.{MODULE.os.getpid()}.tmp")
    real_open = MODULE.os.open
    real_fsync = MODULE.os.fsync
    real_link = MODULE.os.link
    real_unlink = MODULE.Path.unlink
    unlink_calls = 0
    linked = False
    unlinked = False

    def track_link(source, destination, *args, **kwargs):
        nonlocal linked
        result = real_link(source, destination, *args, **kwargs)
        linked = True
        return result

    def fail_directory_open(path, flags, *args):
        if (
            Path(path) == output.parent
            and failure_phase == "directory_open"
            and linked
            and not unlinked
        ):
            raise OSError("injected directory open failure")
        return real_open(path, flags, *args)

    def fail_directory_fsync(descriptor: int) -> None:
        if failure_phase == "first_directory_fsync" and linked and not unlinked:
            raise OSError("injected first directory fsync failure")
        if failure_phase == "final_directory_fsync" and linked and unlinked:
            raise OSError("injected final directory fsync failure")
        real_fsync(descriptor)

    def fail_temporary_unlink(path, *args, **kwargs):
        nonlocal unlink_calls, unlinked
        if Path(path) == temporary:
            unlink_calls += 1
            if failure_phase == "temporary_unlink" and unlink_calls == 1:
                raise OSError("injected temporary unlink failure")
        result = real_unlink(path, *args, **kwargs)
        if Path(path) == temporary:
            unlinked = True
        return result

    monkeypatch.setattr(MODULE.os, "open", fail_directory_open)
    monkeypatch.setattr(MODULE.os, "fsync", fail_directory_fsync)
    monkeypatch.setattr(MODULE.os, "link", track_link)
    monkeypatch.setattr(MODULE.Path, "unlink", fail_temporary_unlink)

    with pytest.raises(OSError, match="injected"):
        MODULE.write_json_atomic(output, result)

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert not tuple(tmp_path.glob(".comparison.json.*.tmp"))


def test_comparison_publication_uses_mode_0600_and_preserves_foreign_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_paths, receipt_paths = _abba_fixture(tmp_path)
    result = MODULE.compare_abba(profile_paths, receipt_paths)
    output = tmp_path / "comparison.json"
    temporary = output.with_name(f".{output.name}.{MODULE.os.getpid()}.tmp")
    temporary.write_bytes(b"foreign temporary\n")

    with pytest.raises(FileExistsError):
        MODULE.write_json_atomic(output, result)
    assert temporary.read_bytes() == b"foreign temporary\n"
    assert not output.exists()

    temporary.unlink()

    def lose_link_race(_source, destination):
        Path(destination).write_bytes(b"foreign destination\n")
        raise FileExistsError(destination)

    monkeypatch.setattr(MODULE.os, "link", lose_link_race)
    with pytest.raises(FileExistsError):
        MODULE.write_json_atomic(output, result)
    assert output.read_bytes() == b"foreign destination\n"
    assert not temporary.exists()

    output.unlink()
    monkeypatch.undo()
    MODULE.write_json_atomic(output, result)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


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
