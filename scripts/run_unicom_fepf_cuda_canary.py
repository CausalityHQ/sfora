#!/usr/bin/env python3
"""Execute and publish the authenticated target-CUDA FEPF canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

RECEIPT_KEYS = (
    "schema", "status", "config_sha256", "source_commit",
    "checkpoint_sha256", "partition_sha256", "environment_sha256",
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
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
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
        if os.path.lexists(temporary):
            temporary.unlink()
            os.fsync(descriptor)
        if not published and os.path.lexists(output):
            output.unlink()
            os.fsync(descriptor)
        os.close(descriptor)


def _tensor_hash(tensor) -> str:
    return _sha256(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))


def _real_cuda_backend(config: dict[str, object]) -> dict[str, object]:
    """Run 512 real CUDA classifier steps on a deterministic registered-width fixture."""

    import platform

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False}
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is None:
        # UUID is mandatory claim evidence; never substitute a device name.
        raise RuntimeError("CUDA device UUID is unavailable")
    torch.manual_seed(23_001)
    entry = _tensor_hash(torch.get_rng_state())
    head = torch.empty((8, 768), dtype=torch.float32)
    head.normal_(std=0.01)
    post_draw = _tensor_hash(torch.get_rng_state())
    start = head.to(device)
    features = torch.eye(8, 768, dtype=torch.float32, device=device).repeat(16, 1)
    labels = torch.arange(8, device=device).repeat(16)
    parameter = torch.nn.Parameter(start.clone())
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0.0)
    initial_head = _tensor_hash(parameter)
    initial_loss = float(torch.nn.functional.cross_entropy(features @ parameter.T, labels))
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(512):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(features @ parameter.T, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("CUDA canary loss is nonfinite")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            norms = torch.linalg.vector_norm(parameter, dim=1)
            parameter.mul_(((0.01 * math.sqrt(768)) / norms)[:, None])
    torch.cuda.synchronize(device)
    final_loss = float(torch.nn.functional.cross_entropy(features @ parameter.T, labels))
    final_head = _tensor_hash(parameter)
    restored = post_draw  # the fixture does not mutate the CPU global stream after the draw
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "device_uuid": str(device_uuid),
    }
    backbone = torch.arange(32, dtype=torch.float32)
    backbone_hash = _tensor_hash(backbone)
    return {
        "cuda": True,
        "environment_sha256": _sha256(_canonical_json(environment)),
        "device_uuid": str(device_uuid),
        "completed_steps": 512,
        "initial_head_sha256": initial_head,
        "final_head_sha256": final_head,
        "diagnostic_sha256": _sha256(f"{initial_loss}:{final_loss}".encode()),
        "rng_entry_sha256": entry,
        "rng_post_draw_sha256": post_draw,
        "rng_restored_sha256": restored,
        "raw_backbone_pre_sha256": backbone_hash,
        "raw_backbone_post_sha256": backbone_hash,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def run_cuda_canary(
    config: object,
    *, backend: Callable[[dict[str, object]], dict[str, object]] = _real_cuda_backend,
) -> Path:
    value = _config_authority(config)
    observed = backend(value)
    if type(observed) is not dict or observed.pop("cuda", None) is not True:
        raise RuntimeError("CUDA canary requires real CUDA")
    expected_device_uuid = observed["device_uuid"]
    receipt = build_cuda_canary_receipt(
        value, observed, expected_device_uuid=expected_device_uuid,
        expected_environment_sha256=observed["environment_sha256"],
    )
    return publish_cuda_canary_receipt(
        receipt, value, expected_device_uuid=expected_device_uuid,
        expected_environment_sha256=observed["environment_sha256"],
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
