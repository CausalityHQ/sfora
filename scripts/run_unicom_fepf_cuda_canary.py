#!/usr/bin/env python3
"""Execute and publish the authenticated target-CUDA FEPF canary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

RECEIPT_KEYS = (
    "schema", "status", "config_sha256", "source_commit",
    "checkpoint_sha256", "partition_sha256", "environment", "environment_sha256",
    "device_uuid", "completed_steps", "initial_head_sha256",
    "final_head_sha256", "diagnostic_sha256", "rng_entry_sha256",
    "rng_post_draw_sha256", "rng_restored_sha256",
    "raw_backbone_pre_sha256", "raw_backbone_post_sha256",
    "initial_loss", "final_loss", "peak_allocated_bytes",
    "peak_reserved_bytes",
)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _lower_sha256(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _config_authority(config: object) -> dict[str, object]:
    if (
        type(config) is not dict
        or config.get("schema") != "unicom-fepf-run-config-v1"
        or type(config.get("source_commit")) is not str
        or len(config["source_commit"]) != 40
        or type(config.get("model")) is not dict
        or not _lower_sha256(config["model"].get("checkpoint_sha256"))
        or not _lower_sha256(config["model"].get("partition_sha256"))
        or type(config.get("artifact_root")) is not str
        or config.get("cuda_canary_receipt") != "preflight/cuda_canary_v1.json"
        or type(config.get("inputs")) is not dict
        or type(config["inputs"].get("checkpoint")) is not str
        or type(config["inputs"].get("partition")) is not str
        or type(config.get("cuda_canary_authority")) is not dict
        or type(config["cuda_canary_authority"].get("device_uuid")) is not str
        or not config["cuda_canary_authority"]["device_uuid"].startswith("GPU-")
        or not _lower_sha256(
            config["cuda_canary_authority"].get("environment_sha256")
        )
    ):
        raise ValueError("CUDA canary config differs")
    return config


def build_cuda_canary_receipt(
    config: object, observation: Mapping[str, object], *, expected_device_uuid: str,
    expected_environment_sha256: str,
) -> dict[str, object]:
    value = _config_authority(config)
    required = tuple(RECEIPT_KEYS[6:])
    if type(observation) is not dict or tuple(observation) != required:
        raise ValueError("CUDA canary observation differs")
    receipt = {
        "schema": "unicom-fepf-cuda-canary-v1",
        "status": "PASS",
        "config_sha256": _sha256(_canonical_json(value)),
        "source_commit": value["source_commit"],
        "checkpoint_sha256": value["model"]["checkpoint_sha256"],
        "partition_sha256": value["model"]["partition_sha256"],
        **dict(observation),
    }
    validate_cuda_canary_receipt(
        receipt, value, expected_device_uuid=expected_device_uuid,
        expected_environment_sha256=expected_environment_sha256,
    )
    return receipt


def validate_cuda_canary_receipt(
    receipt: object, config: object, *, expected_device_uuid: str,
    expected_environment_sha256: str,
) -> None:
    value = _config_authority(config)
    if (
        type(receipt) is not dict
        or tuple(receipt) != RECEIPT_KEYS
        or receipt["schema"] != "unicom-fepf-cuda-canary-v1"
        or receipt["status"] != "PASS"
        or receipt["config_sha256"] != _sha256(_canonical_json(value))
        or receipt["source_commit"] != value["source_commit"]
        or receipt["checkpoint_sha256"] != value["model"]["checkpoint_sha256"]
        or receipt["partition_sha256"] != value["model"]["partition_sha256"]
        or not validate_canary_environment_payload(
            receipt["environment"], expected_environment_sha256
        )
        or receipt["completed_steps"] != 512
        or type(expected_device_uuid) is not str
        or not expected_device_uuid.startswith("GPU-")
        or receipt["device_uuid"] != expected_device_uuid
        or not _lower_sha256(expected_environment_sha256)
        or receipt["environment_sha256"] != expected_environment_sha256
    ):
        raise ValueError("CUDA canary receipt differs")
    for key in (
        "config_sha256", "checkpoint_sha256", "partition_sha256",
        "environment_sha256", "initial_head_sha256", "final_head_sha256",
        "diagnostic_sha256", "rng_entry_sha256", "rng_post_draw_sha256",
        "rng_restored_sha256", "raw_backbone_pre_sha256", "raw_backbone_post_sha256",
    ):
        if not _lower_sha256(receipt[key]):
            raise ValueError("CUDA canary hash differs")
    if (
        receipt["rng_post_draw_sha256"] != receipt["rng_restored_sha256"]
        or receipt["raw_backbone_pre_sha256"] != receipt["raw_backbone_post_sha256"]
    ):
        raise ValueError("CUDA canary restoration differs")
    for key in ("initial_loss", "final_loss"):
        if type(receipt[key]) is not float or not math.isfinite(receipt[key]):
            raise ValueError("CUDA canary loss differs")
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        if type(receipt[key]) is not int or receipt[key] <= 0:
            raise ValueError("CUDA canary memory differs")
    if receipt["peak_reserved_bytes"] < receipt["peak_allocated_bytes"]:
        raise ValueError("CUDA canary memory differs")


def validate_canary_environment_payload(payload: object, expected_sha256: str) -> bool:
    if (
        type(payload) is not dict
        or tuple(payload) != (
            "python_vv", "torch", "torchvision", "timm", "numpy", "cuda",
            "cudnn", "compile", "device_uuid", "gpu_inventory",
            "pyproject_sha256", "uv_lock_sha256", "profile",
        )
        or type(payload["profile"]) is not dict
        or not _lower_sha256(expected_sha256)
        or _sha256(_canonical_json(payload)) != expected_sha256
    ):
        raise ValueError("CUDA canary environment payload differs")
    return True


def _plain_root(config: dict[str, object]) -> tuple[Path, Path]:
    root = Path(config["artifact_root"])
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("CUDA canary output path differs")
    parent = root / "preflight"
    if not parent.is_dir() or parent.is_symlink() or parent.resolve().parent != root.resolve():
        raise ValueError("CUDA canary output path differs")
    output = parent / config["cuda_canary_receipt"] .split("/", 1)[1]
    return parent, output


def publish_cuda_canary_receipt(
    receipt: object, config: object, *, expected_device_uuid: str,
    expected_environment_sha256: str,
) -> Path:
    value = _config_authority(config)
    validate_cuda_canary_receipt(
        receipt, value, expected_device_uuid=expected_device_uuid,
        expected_environment_sha256=expected_environment_sha256,
    )
    parent, output = _plain_root(value)
    if os.path.lexists(output):
        raise FileExistsError(output)
    temporary = parent / ".cuda_canary_v1.json.tmp"
    if os.path.lexists(temporary):
        raise FileExistsError(temporary)
    payload = _canonical_json(receipt)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    published = False
    owned: tuple[int, int] | None = None
    owned_descriptor: int | None = None
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            owned_descriptor = os.dup(handle.fileno())
            information = os.fstat(owned_descriptor)
            owned = (information.st_dev, information.st_ino)
        persisted = json.loads(temporary.read_bytes())
        validate_cuda_canary_receipt(
            persisted, value, expected_device_uuid=expected_device_uuid,
            expected_environment_sha256=expected_environment_sha256,
        )
        os.link(temporary, output)
        os.fsync(descriptor)
        temporary.unlink()
        os.fsync(descriptor)
        reloaded = json.loads(output.read_bytes())
        if output.read_bytes() != payload:
            raise RuntimeError("CUDA canary persisted bytes differ")
        validate_cuda_canary_receipt(
            reloaded, value, expected_device_uuid=expected_device_uuid,
            expected_environment_sha256=expected_environment_sha256,
        )
        published = True
        return output
    finally:
        if owned is not None and os.path.lexists(temporary):
            information = temporary.lstat()
            if (information.st_dev, information.st_ino) == owned:
                temporary.unlink()
                os.fsync(descriptor)
        if not published and owned is not None and os.path.lexists(output):
            information = output.lstat()
            if (information.st_dev, information.st_ino) == owned:
                output.unlink()
                os.fsync(descriptor)
        if owned_descriptor is not None:
            os.close(owned_descriptor)
        os.close(descriptor)


def _tensor_hash(tensor) -> str:
    return _sha256(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))


def authenticate_canary_inputs(config: dict[str, object]) -> dict[str, Path]:
    inputs = config.get("inputs")
    if type(inputs) is not dict:
        raise ValueError("CUDA canary input authority differs")
    paths = {
        "checkpoint": Path(inputs["checkpoint"]),
        "partition": Path(inputs["partition"]),
    }
    for name, path in paths.items():
        expected = config["model"][f"{name}_sha256"]
        if path.is_symlink() or not path.is_file() or _sha256(path.read_bytes()) != expected:
            raise ValueError(f"CUDA canary {name} authority differs")
    return paths


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_registered_canary_model(
    config: dict[str, object], *, trainer: object, device: object
) -> tuple[object, object, str]:
    loader = getattr(trainer, "_load_official_model", None)
    hasher = getattr(trainer, "raw_backbone_state_sha256", None)
    if not callable(loader) or not callable(hasher):
        raise ValueError("registered model authority differs")
    raw_model, transform = loader(
        Path(config["inputs"]["unicom_checkout"]),
        Path(config["inputs"]["checkpoint"]),
    )
    raw_model = raw_model.to(device)
    return raw_model, transform, hasher(raw_model)


def capture_canary_rng_audit(trainer: object, entry: object, post_draw: object):
    restore = getattr(trainer, "_restore_global_rng_snapshot", None)
    audit = getattr(trainer, "_fepf_rng_audit", None)
    if not callable(restore) or not callable(audit):
        raise ValueError("registered RNG authority differs")
    restore(post_draw)
    return audit(entry, post_draw)


def run_registered_fepf_canary(
    *, config: dict[str, object], device, torch, fepf, trainer, raw_model,
    eval_transform, optimization: tuple[object, ...], labels: dict[str, int],
) -> dict[str, object]:
    entry = trainer._global_rng_snapshot()
    official = torch.empty((len(labels), 768), dtype=torch.float32)
    torch.nn.init.normal_(official, std=0.01)
    post_draw = trainer._global_rng_snapshot()
    started = time.perf_counter()
    try:
        cache = trainer.build_registered_fepf_cache(
            raw_model=raw_model, optimization=optimization, labels=labels,
            eval_transform=eval_transform, device=device, batch_size=128, workers=0,
        )
        evidence = fepf.prepare_registered_fepf_evidence(
            cache, official, mode="fepf_mean", training_seed=0, device=device
        )
        fit = fepf.fit_fepf_head(
            cache, evidence, training_seed=0, device=device, steps=512
        )
        classifier = torch.nn.Parameter(fit.head.detach().to(device))
        if classifier.shape != (len(labels), 768):
            raise ValueError("registered classifier transfer differs")
        torch.cuda.synchronize(device)
        elapsed = float(time.perf_counter() - started)
    finally:
        rng_audit = capture_canary_rng_audit(trainer, entry, post_draw)
    source_sha256 = _sha256(Path(fepf.__file__).read_bytes())
    checkpoint_sha256 = config["model"]["checkpoint_sha256"]
    config_sha256 = _sha256(_canonical_json(config))
    schedule_sha256 = _sha256(
        _canonical_json({
            "steps": 512, "training_seed": 0,
            "records": [[record.label, str(record.image_path)] for record in optimization],
        })
    )
    initialization = fepf.initialization_receipt_v2(
        mode="fepf_mean", training_seed=0, holdout_fraction=0.2,
        holdout_seed=0, source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256, config_sha256=config_sha256,
        schedule_sha256=schedule_sha256, official_random_head=official,
        evidence=evidence, initialization_seconds=elapsed, cache=cache,
        rng_audit=rng_audit, fit=fit, device=device,
    )
    initialization_sha256 = fepf.canonical_initialization_receipt_v2_sha256(initialization)
    fepf.validate_initialization_receipt_v2(
        initialization,
        expected=fepf.FepfExpectedProvenance(
            mode="fepf_mean", training_seed=0, holdout_fraction=0.2,
            holdout_seed=0, source_sha256=source_sha256,
            checkpoint_sha256=checkpoint_sha256, config_sha256=config_sha256,
            schedule_sha256=schedule_sha256, receipt_sha256=initialization_sha256,
        ),
        device=device,
    )
    return {
        "initial_head_sha256": evidence.prepared_start_head_sha256,
        "final_head_sha256": fit.final_head_sha256,
        "diagnostic_sha256": _sha256(
            (
                f"{fit.diagnostic_feature_sha256}:{fit.diagnostic_mask_sha256}:"
                f"{initialization_sha256}"
            ).encode()
        ),
        "rng_entry_sha256": initialization["torch_cpu_rng_entry_sha256"],
        "rng_post_draw_sha256": initialization["torch_cpu_rng_post_draw_sha256"],
        "rng_restored_sha256": initialization["torch_cpu_rng_restored_sha256"],
        "initial_loss": fit.initial_loss,
        "final_loss": fit.final_loss,
    }


def _real_cuda_backend(config: dict[str, object]) -> dict[str, object]:
    """Run 512 real CUDA classifier steps on a deterministic registered-width fixture."""

    import numpy as np
    import timm
    import torch
    import torchvision

    from sfora import unicom_fepf as fepf
    from sfora.unicom_inshop import parse_inshop_partition

    if not torch.cuda.is_available():
        return {"cuda": False}
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is None:
        # UUID is mandatory claim evidence; never substitute a device name.
        raise RuntimeError("CUDA device UUID is unavailable")
    torch.manual_seed(23_001)
    torch.cuda.reset_peak_memory_stats(device)
    repository = Path(__file__).resolve().parents[1]
    trainer = _load_script(
        repository / "scripts/train_unicom_inshop.py", "task6_canary_trainer"
    )
    raw_model, eval_transform, raw_pre = load_registered_canary_model(
        config, trainer=trainer, device=device
    )
    records = tuple(
        record for record in parse_inshop_partition(Path(config["inputs"]["dataset_root"]))
        if record.split == "train"
    )
    optimization = records[:128]
    selected_labels = tuple(sorted({record.label for record in optimization}))
    labels = {label: index for index, label in enumerate(selected_labels)}
    if len(optimization) != 128 or len(labels) < 2:
        raise ValueError("registered canary partition fixture differs")
    registered = run_registered_fepf_canary(
        config=config, device=device, torch=torch, fepf=fepf, trainer=trainer,
        raw_model=raw_model, eval_transform=eval_transform,
        optimization=optimization, labels=labels,
    )
    torch.cuda.synchronize(device)
    gpu_inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    profile_environment = {
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "cuda_version": str(torch.version.cuda),
        "device_name": torch.cuda.get_device_name(device),
    }
    environment = {
        "python_vv": subprocess.run(
            [sys.executable, "-VV"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "compile": {
            "available": str(hasattr(torch, "compile")),
            "inductor": str(getattr(torch.version, "git_version", "unknown")),
        },
        "device_uuid": str(device_uuid),
        "gpu_inventory": gpu_inventory,
        "pyproject_sha256": _sha256((repository / "pyproject.toml").read_bytes()),
        "uv_lock_sha256": _sha256((repository / "uv.lock").read_bytes()),
        "profile": profile_environment,
    }
    raw_post = trainer.raw_backbone_state_sha256(raw_model)
    if raw_post != raw_pre:
        raise ValueError("registered raw backbone changed during canary")
    return {
        "cuda": True,
        "environment": environment,
        "environment_sha256": _sha256(_canonical_json(environment)),
        "device_uuid": str(device_uuid),
        "completed_steps": 512,
        "initial_head_sha256": registered["initial_head_sha256"],
        "final_head_sha256": registered["final_head_sha256"],
        "diagnostic_sha256": registered["diagnostic_sha256"],
        "rng_entry_sha256": registered["rng_entry_sha256"],
        "rng_post_draw_sha256": registered["rng_post_draw_sha256"],
        "rng_restored_sha256": registered["rng_restored_sha256"],
        "raw_backbone_pre_sha256": raw_pre,
        "raw_backbone_post_sha256": raw_post,
        "initial_loss": registered["initial_loss"],
        "final_loss": registered["final_loss"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def run_cuda_canary(
    config: object,
    *, backend: Callable[[dict[str, object]], dict[str, object]] = _real_cuda_backend,
) -> Path:
    value = _config_authority(config)
    authenticate_canary_inputs(value)
    observed = dict(backend(value))
    if type(observed) is not dict or observed.pop("cuda", None) is not True:
        raise RuntimeError("CUDA canary requires real CUDA")
    authority = value.get("cuda_canary_authority")
    if (
        type(authority) is not dict
        or type(authority.get("device_uuid")) is not str
        or not _lower_sha256(authority.get("environment_sha256"))
    ):
        raise ValueError("CUDA canary external authority differs")
    expected_device_uuid = authority["device_uuid"]
    expected_environment_sha256 = authority["environment_sha256"]
    receipt = build_cuda_canary_receipt(
        value, observed, expected_device_uuid=expected_device_uuid,
        expected_environment_sha256=expected_environment_sha256,
    )
    return publish_cuda_canary_receipt(
        receipt, value, expected_device_uuid=expected_device_uuid,
        expected_environment_sha256=expected_environment_sha256,
    )


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        config = json.loads(args.config.read_bytes())
        run_cuda_canary(config)
    except Exception as error:
        print(f"FEPF CUDA canary failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
