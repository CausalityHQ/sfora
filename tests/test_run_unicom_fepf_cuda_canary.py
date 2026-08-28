from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_unicom_fepf_cuda_canary", ROOT / "scripts/run_unicom_fepf_cuda_canary.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _environment() -> dict[str, object]:
    return {
        "python_vv": "Python 3.12.3",
        "torch": "2.6.0", "torchvision": "0.21.0", "timm": "1.0.0",
        "numpy": "2.1.3", "cuda": "12.4", "cudnn": "90100",
        "compile": {"available": "True", "inductor": "registered"},
        "device_uuid": "GPU-registered",
        "gpu_inventory": ["H100, GPU-registered, 550.54"],
        "pyproject_sha256": "1" * 64, "uv_lock_sha256": "2" * 64,
        "profile": {
            "python_version": "3.12.3", "torch_version": "2.6.0",
            "numpy_version": "2.1.3", "cuda_version": "12.4",
            "device_name": "NVIDIA H100 80GB HBM3",
        },
    }


def _environment_sha256() -> str:
    return MODULE._sha256(MODULE._canonical_json(_environment()))


def _config(tmp_path: Path) -> dict[str, object]:
    checkpoint = tmp_path / "checkpoint.bin"
    partition = tmp_path / "partition.txt"
    checkpoint.write_bytes(b"checkpoint\n")
    partition.write_bytes(b"partition\n")
    return {
        "schema": "unicom-fepf-run-config-v1",
        "source_commit": "a" * 40,
        "model": {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "partition_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
        },
        "artifact_root": str(tmp_path / "artifacts"),
        "inputs": {
            "checkpoint": str(checkpoint),
            "partition": str(partition),
        },
        "cuda_canary_authority": {
            "device_uuid": "GPU-registered", "environment_sha256": _environment_sha256(),
        },
        "cuda_canary_receipt": "preflight/cuda_canary_v1.json",
    }


def _observation() -> dict[str, object]:
    return {
        "environment": _environment(),
        "environment_sha256": _environment_sha256(),
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
        expected_environment_sha256=_environment_sha256(),
    )
    MODULE.validate_cuda_canary_receipt(
        receipt, config, expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
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
        expected_environment_sha256=_environment_sha256(),
    )
    receipt[key] = value
    with pytest.raises(ValueError):
        MODULE.validate_cuda_canary_receipt(
            receipt, config, expected_device_uuid="GPU-registered",
            expected_environment_sha256=_environment_sha256(),
        )


def test_canary_publication_is_config_derived_no_replace_and_strict_reload(
    tmp_path: Path
) -> None:
    config = _config(tmp_path)
    root = Path(config["artifact_root"])
    (root / "preflight").mkdir(parents=True)
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    output = MODULE.publish_cuda_canary_receipt(
        receipt, config, expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    assert output == root / "preflight" / "cuda_canary_v1.json"
    assert json.loads(output.read_bytes()) == receipt
    with pytest.raises(FileExistsError):
        MODULE.publish_cuda_canary_receipt(
            receipt, config, expected_device_uuid="GPU-registered",
            expected_environment_sha256=_environment_sha256(),
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
        expected_environment_sha256=_environment_sha256(),
    )
    with pytest.raises(ValueError, match="path"):
        MODULE.publish_cuda_canary_receipt(
            receipt, config, expected_device_uuid="GPU-registered",
            expected_environment_sha256=_environment_sha256(),
        )


def test_canary_backend_must_report_real_cuda_and_never_skips(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="CUDA"):
        MODULE.run_cuda_canary(config, backend=lambda _config: {"cuda": False})


def test_canary_racing_destination_is_never_unlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    root = Path(config["artifact_root"])
    (root / "preflight").mkdir(parents=True)
    output = root / "preflight" / "cuda_canary_v1.json"
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    original_link = os.link

    def racing_link(source: Path, destination: Path) -> None:
        destination.write_bytes(b"racer")
        raise FileExistsError(destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        MODULE.publish_cuda_canary_receipt(
            receipt, config, expected_device_uuid="GPU-registered",
            expected_environment_sha256=_environment_sha256(),
        )
    monkeypatch.setattr(os, "link", original_link)
    assert output.read_bytes() == b"racer"


def test_review2_canary_racing_temporary_is_never_unlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    root = Path(config["artifact_root"])
    (root / "preflight").mkdir(parents=True)
    temporary = root / "preflight" / ".cuda_canary_v1.json.tmp"
    receipt = MODULE.build_cuda_canary_receipt(
        config, _observation(), expected_device_uuid="GPU-registered",
        expected_environment_sha256=_environment_sha256(),
    )
    original_read_bytes = Path.read_bytes
    substituted = False

    def replace_before_reload(path: Path) -> bytes:
        nonlocal substituted
        if path == temporary and not substituted:
            substituted = True
            path.unlink()
            path.write_bytes(b"racer-temporary")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", replace_before_reload)
    with pytest.raises(json.JSONDecodeError):
        MODULE.publish_cuda_canary_receipt(
            receipt, config, expected_device_uuid="GPU-registered",
            expected_environment_sha256=_environment_sha256(),
        )
    assert original_read_bytes(temporary) == b"racer-temporary"


def test_canary_requires_external_input_device_environment_and_task1_flow(tmp_path: Path) -> None:
    assert hasattr(MODULE, "authenticate_canary_inputs")
    assert hasattr(MODULE, "run_registered_fepf_canary")
    config = _config(tmp_path)
    root = Path(config["artifact_root"])
    (root / "preflight").mkdir(parents=True)
    called = False

    def backend(_config: dict[str, object]) -> dict[str, object]:
        nonlocal called
        called = True
        return {"cuda": True, **_observation()}

    assert MODULE.run_cuda_canary(config, backend=backend).is_file()
    assert called
    output = root / "preflight" / "cuda_canary_v1.json"
    output.unlink()
    Path(config["inputs"]["checkpoint"]).write_bytes(b"substituted\n")
    called = False
    with pytest.raises(ValueError, match="checkpoint authority"):
        MODULE.run_cuda_canary(config, backend=backend)
    assert called is False


def test_review2_canary_requires_registered_model_and_observed_provenance() -> None:
    calls: list[object] = []

    class Model:
        def to(self, device: object) -> Model:
            calls.append(("device", device))
            return self

    class Trainer:
        @staticmethod
        def _load_official_model(checkout: Path, checkpoint: Path):
            calls.append(("load", checkout, checkpoint))
            return Model(), "registered-transform"

        @staticmethod
        def raw_backbone_state_sha256(model: object) -> str:
            calls.append(("hash", model))
            return "a" * 64

        @staticmethod
        def _restore_global_rng_snapshot(snapshot: object) -> None:
            calls.append(("restore", snapshot))

        @staticmethod
        def _fepf_rng_audit(entry: object, post_draw: object) -> str:
            calls.append(("audit", entry, post_draw))
            return "registered-audit"

    config = {
        "inputs": {
            "unicom_checkout": "/registered/unicom",
            "checkpoint": "/registered/checkpoint.bin",
        }
    }
    model, transform, digest = MODULE.load_registered_canary_model(
        config, trainer=Trainer, device="cuda:registered"
    )
    assert transform == "registered-transform"
    assert digest == "a" * 64
    assert calls[:3] == [
        (
            "load",
            Path("/registered/unicom"),
            Path("/registered/checkpoint.bin"),
        ),
        ("device", "cuda:registered"),
        ("hash", model),
    ]
    assert MODULE.capture_canary_rng_audit(Trainer, "entry", "post") == (
        "registered-audit"
    )
    assert calls[-2:] == [("restore", "post"), ("audit", "entry", "post")]
    assert MODULE.validate_canary_environment_payload(
        _environment(), _environment_sha256()
    )
    mutated = _environment()
    mutated["device_uuid"] = "GPU-substituted"
    with pytest.raises(ValueError, match="environment"):
        MODULE.validate_canary_environment_payload(mutated, _environment_sha256())
