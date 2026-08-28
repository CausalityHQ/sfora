from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_unicom_fepf_cuda_canary", ROOT / "scripts/run_unicom_fepf_cuda_canary.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": "unicom-fepf-run-config-v1",
        "source_commit": "a" * 40,
        "model": {
            "checkpoint_sha256": "b" * 64,
            "partition_sha256": "c" * 64,
        },
        "artifact_root": str(tmp_path / "artifacts"),
        "cuda_canary_receipt": "preflight/cuda_canary_v1.json",
    }


def _observation() -> dict[str, object]:
    return {
        "environment_sha256": "d" * 64,
        "device_uuid": "GPU-registered",
        "completed_steps": 512,
        "initial_head_sha256": "e" * 64,
        "final_head_sha256": "f" * 64,
        "diagnostic_sha256": "1" * 64,
        "rng_entry_sha256": "2" * 64,
        "rng_post_draw_sha256": "3" * 64,
        "rng_restored_sha256": "3" * 64,
        "raw_backbone_pre_sha256": "4" * 64,
        "raw_backbone_post_sha256": "4" * 64,
        "initial_loss": 3.0,
        "final_loss": 2.0,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
    }


def test_cpu_fake_canary_builds_and_strictly_validates_terminal_receipt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256="d" * 64,
    )
    MODULE.validate_cuda_canary_receipt(
        receipt, config, expected_device_uuid="GPU-registered",
        expected_environment_sha256="d" * 64,
    )
    assert receipt["status"] == "PASS"
    assert receipt["completed_steps"] == 512
    assert receipt["rng_post_draw_sha256"] == receipt["rng_restored_sha256"]
    assert receipt["raw_backbone_pre_sha256"] == receipt["raw_backbone_post_sha256"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "SKIP"),
        ("completed_steps", 511),
        ("device_uuid", "GPU-wrong"),
        ("environment_sha256", "0" * 64),
        ("final_loss", float("nan")),
        ("peak_reserved_bytes", True),
    ],
)
def test_canary_rejects_terminal_mutations(
    tmp_path: Path, key: str, value: object
) -> None:
    config = _config(tmp_path)
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256="d" * 64,
    )
    receipt[key] = value
    with pytest.raises(ValueError):
        MODULE.validate_cuda_canary_receipt(
            receipt, config, expected_device_uuid="GPU-registered",
            expected_environment_sha256="d" * 64,
        )


def test_canary_publication_is_config_derived_no_replace_and_strict_reload(
    tmp_path: Path
) -> None:
    config = _config(tmp_path)
    root = Path(config["artifact_root"])
    (root / "preflight").mkdir(parents=True)
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256="d" * 64,
    )
    output = MODULE.publish_cuda_canary_receipt(
        receipt, config, expected_device_uuid="GPU-registered",
        expected_environment_sha256="d" * 64,
    )
    assert output == root / "preflight" / "cuda_canary_v1.json"
    assert json.loads(output.read_bytes()) == receipt
    with pytest.raises(FileExistsError):
        MODULE.publish_cuda_canary_receipt(
            receipt, config, expected_device_uuid="GPU-registered",
            expected_environment_sha256="d" * 64,
        )


def test_canary_rejects_symlinked_output_parent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    root = Path(config["artifact_root"])
    real = tmp_path / "real"
    real.mkdir()
    root.mkdir()
    (root / "preflight").symlink_to(real, target_is_directory=True)
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="path"):
        MODULE.publish_cuda_canary_receipt(
            receipt, config, expected_device_uuid="GPU-registered",
            expected_environment_sha256="d" * 64,
        )


def test_canary_backend_must_report_real_cuda_and_never_skips(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="CUDA"):
        MODULE.run_cuda_canary(config, backend=lambda _config: {"cuda": False})
