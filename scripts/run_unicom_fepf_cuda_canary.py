#!/usr/bin/env python3
"""Execute and publish the authenticated target-CUDA FEPF canary."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import math
import os
import pickle
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path

from sfora.atomic_publication import BudgetedPublisher, publish_bytes_noreplace

RECEIPT_KEYS = (
    "schema", "status", "config_sha256", "source_commit",
    "checkpoint_sha256", "partition_sha256", "environment", "environment_sha256",
    "device_uuid", "completed_steps", "initial_head_sha256",
    "final_head_sha256", "diagnostic_sha256", "rng_entry_sha256",
    "rng_post_draw_sha256", "rng_restored_sha256",
    "raw_backbone_pre_sha256", "raw_backbone_post_sha256",
    "evidence_manifest_sha256",
    "initial_loss", "final_loss", "peak_allocated_bytes",
    "peak_reserved_bytes",
)
CANARY_EVIDENCE_ORDER = (
    "observation.json", "initialization-receipt.json", "cache-inventory.json",
    "model-inventory.json", "rng-audit.json", "model-modes.json",
    "environment.json", "manifest.json",
)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_normalized(value: object) -> object:
    """Return the exact structure that survives canonical JSON publication."""

    return json.loads(_canonical_json(value))


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
        or (
            config["cuda_canary_authority"] != {}
            and (
                type(config["cuda_canary_authority"].get("device_uuid")) is not str
                or not config["cuda_canary_authority"]["device_uuid"].startswith("GPU-")
                or not _lower_sha256(
                    config["cuda_canary_authority"].get("environment_sha256")
                )
            )
        )
    ):
        raise ValueError("CUDA canary config differs")
    return config


def ensure_campaign_root(
    config: dict[str, object], *, physical_admission: bool = True
) -> Path:
    root = Path(config["artifact_root"])
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("CUDA canary campaign root differs")
    repository = Path(__file__).resolve().parents[1]
    builder = _load_script(
        repository / "scripts/build_unicom_fepf_run_config.py",
        "canary_storage_builder",
    )
    required_bytes = config.get("artifact_budget_bytes")
    required_inodes = config.get("artifact_budget_inodes")
    if (
        type(required_bytes) is not int
        or required_bytes <= 0
        or type(required_inodes) is not int
        or required_inodes <= 0
    ):
        raise ValueError("CUDA canary campaign budget differs")
    if not root.exists():
        if physical_admission:
            builder.prepare_artifact_root(
                root, required_bytes=required_bytes, required_inodes=required_inodes
            )
        else:
            if not root.parent.is_dir() or root.parent.is_symlink():
                raise ValueError("CUDA canary campaign parent differs")
            root.mkdir(mode=0o700)
    elif not root.is_dir():
        raise ValueError("CUDA canary campaign root differs")
    preflight = root / "preflight"
    if not preflight.exists() and any(root.iterdir()):
        raise ValueError("CUDA canary foreign campaign root differs")
    if not preflight.exists():
        preflight.mkdir(mode=0o700)
    consumed_paths = [root, *root.rglob("*")]
    if any(path.is_symlink() for path in consumed_paths):
        raise ValueError("CUDA canary campaign namespace differs")
    budget = config.get("publication_budget")
    rows = budget.get("publications", []) if type(budget) is dict else []
    registered_paths = {
        Path(row["path"])
        for row in rows
        if type(row) is dict and type(row.get("path")) is str
    }
    for existing in consumed_paths[1:]:
        relative = existing.relative_to(root)
        if relative == Path("controller-status.json"):
            continue
        if existing.is_dir():
            allowed = any(relative in candidate.parents for candidate in registered_paths)
        else:
            allowed = relative in registered_paths
        if not allowed:
            raise ValueError("CUDA canary foreign campaign namespace differs")
    consumed_bytes = sum(
        path.stat().st_size for path in consumed_paths
        if path.is_file() and not path.is_symlink()
    )
    if physical_admission:
        builder.require_remaining_capacity(
            root,
            total_budget_bytes=required_bytes,
            total_budget_inodes=required_inodes,
            consumed_bytes=consumed_bytes,
            consumed_inodes=len(consumed_paths),
        )
    payload = _canonical_json(budget)
    if _sha256(payload) != config.get("publication_budget_sha256"):
        raise ValueError("CUDA canary publication budget differs")
    path = root / config["publication_budget_path"]
    self_rows = [
        row for row in budget.get("publications", [])
        if type(row) is dict
        and row.get("name") == "campaign:publication-budget"
        and row.get("path") == config["publication_budget_path"]
    ] if type(budget) is dict else []
    if len(self_rows) != 1 or len(payload) > self_rows[0].get("persistent_bytes", -1):
        raise ValueError("CUDA canary publication budget self-row differs")
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError(path)
    else:
        published = publish_bytes_noreplace(
            path,
            payload,
            validator=lambda persisted: (
                None
                if persisted == payload
                else (_ for _ in ()).throw(ValueError("canary budget differs"))
            ),
        )
        published.close()
    return root


def validate_canary_exact_budget(
    config: dict[str, object], *, external: bool = True
) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    builder = _load_script(
        repository / "scripts/build_unicom_fepf_run_config.py",
        "canary_exact_budget_builder",
    )
    validator = (
        builder.validate_external_exact_publication_budget
        if external
        else builder.validate_exact_publication_budget
    )
    return validator(config, config.get("publication_budget"))


def validate_canary_publication_capacity(
    config: dict[str, object], *, name: str, destination: Path, payload: bytes
) -> None:
    root = Path(config["artifact_root"])
    relative = destination.resolve().relative_to(root.resolve()).as_posix()
    rows = config.get("publication_budget", {}).get("publications", [])
    matching = [
        row for row in rows
        if type(row) is dict
        and row.get("name") == f"cuda-canary:{name}"
        and row.get("path") == relative
    ]
    if len(matching) != 1:
        raise ValueError("canary publication budget row differs")
    row = matching[0]
    available = os.statvfs(root)
    if (
        len(payload) > row["persistent_bytes"]
        or available.f_bavail * available.f_frsize
        < row["persistent_bytes"] + row["temporary_bytes"]
        or available.f_favail
        < row["persistent_inodes"] + row["temporary_inodes"]
    ):
        raise OSError("canary publication capacity is insufficient")


def reconstruct_canary_authority(
    config: dict[str, object], manifest_path: Path, *, terminal: object | None = None,
    evidence_root: Path | None = None,
) -> dict[str, object]:
    root = (
        Path(config["artifact_root"]) / "preflight/canary-evidence"
        if evidence_root is None
        else evidence_root
    )
    if manifest_path.resolve().parent != root.resolve() or manifest_path.is_symlink():
        raise ValueError("canary manifest authority differs")
    manifest = json.loads(manifest_path.read_bytes())
    objects = manifest.get("objects") if type(manifest) is dict else None
    required = {
        "initialization_receipt", "cache_inventory", "model_inventory",
        "rng_audit", "model_modes", "environment", "observation",
    }
    if type(objects) is not dict or set(objects) != required:
        raise ValueError("canary manifest provenance differs")
    expected_paths = {
        "initialization_receipt": "initialization-receipt.json",
        "cache_inventory": "cache-inventory.json",
        "model_inventory": "model-inventory.json",
        "rng_audit": "rng-audit.json",
        "model_modes": "model-modes.json",
        "environment": "environment.json",
        "observation": "observation.json",
    }
    loaded = {}
    for name, binding in objects.items():
        path_value = binding.get("path") if type(binding) is dict else None
        if path_value != expected_paths[name]:
            raise ValueError("canary manifest path authority differs")
        payload = (root / path_value).read_bytes()
        if (
            (root / path_value).is_symlink()
            or _sha256(payload) != binding.get("sha256")
            or len(payload) != binding.get("bytes")
        ):
            raise ValueError("canary object authority differs")
        loaded[name] = json.loads(payload)
    validate_canary_evidence_manifest(
        {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path.read_bytes()),
            "bytes": manifest_path.stat().st_size,
        },
        evidence_root=root,
    )
    initialization = loaded["initialization_receipt"]
    rng = loaded["rng_audit"]
    if (
        type(initialization) is not dict
        or initialization.get("config_sha256") != _sha256(_canonical_json(config))
        or initialization.get("checkpoint_sha256")
        != config.get("model", {}).get("checkpoint_sha256")
    ):
        raise ValueError("canary initialization provenance differs")
    if terminal is not None and (
        type(terminal) is not dict
        or loaded["environment"] != terminal.get("environment")
        or rng.get("entry") != terminal.get("rng_entry_sha256")
        or rng.get("post_draw") != terminal.get("rng_post_draw_sha256")
        or rng.get("restored") != terminal.get("rng_restored_sha256")
    ):
        raise ValueError("canary terminal provenance authority differs")
    return manifest


def validate_registered_canary_family(
    config: dict[str, object],
    manifest_path: Path,
    terminal: object | None,
    *,
    evidence_root: Path | None = None,
) -> dict[str, object]:
    """Recompute the canary family from authorities outside its own namespace."""

    reconstruct_canary_authority(
        config, manifest_path, terminal=terminal, evidence_root=evidence_root
    )
    root = manifest_path.parent
    observation = json.loads((root / "observation.json").read_bytes())
    initialization = json.loads((root / "initialization-receipt.json").read_bytes())
    model_inventory = json.loads((root / "model-inventory.json").read_bytes())
    cache_inventory = json.loads((root / "cache-inventory.json").read_bytes())
    model_modes = json.loads((root / "model-modes.json").read_bytes())
    authenticate_canary_inputs(config)
    repository = Path(__file__).resolve().parents[1]
    trainer = _load_script(
        repository / "scripts/train_unicom_inshop.py", "canary_family_trainer"
    )
    checkout = Path(config.get("inputs", {}).get("unicom_checkout", ""))
    revision = config.get("model", {}).get("revision")
    git_revision = getattr(trainer, "_git_revision", None)
    if (
        not callable(git_revision)
        or type(revision) is not str
        or git_revision(checkout) != revision
    ):
        raise ValueError("registered UniCOM revision authority differs")
    from sfora import unicom_fepf as fepf
    from sfora.unicom_inshop import parse_inshop_partition

    records = tuple(
        record
        for record in parse_inshop_partition(Path(config["inputs"]["dataset_root"]))
        if record.split == "train"
    )[:128]
    labels = {
        label: index for index, label in enumerate(sorted({r.label for r in records}))
    }
    if len(records) != 128 or len(labels) < 2:
        raise ValueError("registered canary partition fixture differs")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("registered canary reconstruction requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    deterministic = configure_deterministic_canary_execution(
        torch, environment=os.environ
    )
    raw_model, transform, _digest = prepare_registered_canary_seeded_model(
        torch=torch,
        device=device,
        loader=lambda: load_registered_canary_model(
            config, trainer=trainer, device=device
        ),
    )
    parameters = list(raw_model.named_parameters())
    buffers = list(raw_model.named_buffers())
    parameter_names = {name for name, _tensor in parameters}
    buffer_names = {name for name, _tensor in buffers}
    if parameter_names & buffer_names:
        raise ValueError("registered canary model tensor kinds overlap")
    live_model = {
        "schema": "unicom-fepf-canary-model-v1",
        "revision": revision,
        "tensors": sorted([
            {
                "name": name,
                "kind": kind,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": _tensor_hash(tensor),
            }
            for kind, values in (
                ("parameter", parameters),
                ("buffer", buffers),
            )
            for name, tensor in values
        ], key=lambda row: (row["name"], row["kind"])),
    }
    if model_inventory != _json_normalized(live_model):
        raise ValueError("registered canary model authority differs")
    source_rows = config.get("source_files")
    source_matches = [
        row for row in source_rows if type(row) is dict
        and row.get("path") == "src/sfora/unicom_fepf.py"
    ] if type(source_rows) is list else []
    source_path = repository / "src/sfora/unicom_fepf.py"
    if (
        len(source_matches) != 1
        or source_matches[0].get("sha256") != _sha256(source_path.read_bytes())
        or source_matches[0].get("bytes") != source_path.stat().st_size
    ):
        raise ValueError("registered Task 1 source authority differs")
    schedule_sha256 = _sha256(_canonical_json({
        "steps": 512,
        "training_seed": 0,
        "records": [[record.label, str(record.image_path)] for record in records],
    }))
    expected = fepf.FepfExpectedProvenance(
        mode="fepf_mean",
        training_seed=0,
        holdout_fraction=0.2,
        holdout_seed=0,
        source_sha256=source_matches[0]["sha256"],
        checkpoint_sha256=config["model"]["checkpoint_sha256"],
        config_sha256=_sha256(_canonical_json(config)),
        schedule_sha256=schedule_sha256,
        receipt_sha256=fepf.canonical_initialization_receipt_v2_sha256(initialization),
    )
    fepf.validate_initialization_receipt_v2(
        initialization, expected=expected, device=device
    )
    environment = json.loads((root / "environment.json").read_bytes())
    import numpy as np
    import timm
    import torchvision

    properties = torch.cuda.get_device_properties(device)
    device_uuid = getattr(properties, "uuid", None)
    live_environment = {
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
        "gpu_inventory": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines(),
        "pyproject_sha256": _sha256((repository / "pyproject.toml").read_bytes()),
        "uv_lock_sha256": _sha256((repository / "uv.lock").read_bytes()),
        "deterministic_execution": deterministic,
    }
    canary_authority = config["cuda_canary_authority"]
    if environment != live_environment or (
        canary_authority
        and canary_authority.get("device_uuid") != str(device_uuid)
    ):
        raise ValueError("registered canary live environment differs")
    recomputed = recompute_registered_canary_dynamic_authority(
        config=config,
        device=device,
        torch=torch,
        fepf=fepf,
        trainer=trainer,
        raw_model=raw_model,
        eval_transform=transform,
        optimization=records,
        labels=labels,
    )
    if (
        cache_inventory != _json_normalized(recomputed["cache_inventory"])
        or model_modes != _json_normalized(recomputed["model_modes"])
    ):
        raise ValueError("registered canary cache/model-mode authority differs")
    validate_registered_canary_scientific_projection(observation, recomputed)
    if terminal is not None:
        validate_registered_canary_scientific_projection(terminal, recomputed)
    memory_bound = config.get("artifact_budget_bytes")
    if type(memory_bound) is not int or memory_bound <= 0:
        raise ValueError("registered canary memory budget differs")
    for source in (observation, recomputed, terminal):
        if source is None:
            continue
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            value = source.get(key)
            if type(value) is not int or value < 0 or value > memory_bound:
                raise ValueError("registered canary memory observation differs")
    persisted_science = {
        key: value
        for key, value in initialization.items()
        if key not in {"initialization_seconds", "fit_seconds"}
    }
    recomputed_initialization = recomputed["initialization_receipt"]
    recomputed_science = {
        key: value
        for key, value in recomputed_initialization.items()
        if key not in {"initialization_seconds", "fit_seconds"}
    }
    if (
        persisted_science != _json_normalized(recomputed_science)
        or json.loads((root / "rng-audit.json").read_bytes())
        != _json_normalized(recomputed["rng_audit"])
    ):
        raise ValueError("registered canary fitted observation reconstruction differs")
    return json.loads(manifest_path.read_bytes())


def validate_registered_canary_scientific_projection(
    observed: Mapping[str, object], recomputed: Mapping[str, object]
) -> None:
    """Compare deterministic science while validating observational telemetry."""

    scientific = (
        "completed_steps", "initial_head_sha256", "final_head_sha256",
        "diagnostic_sha256", "rng_entry_sha256", "rng_post_draw_sha256",
        "rng_restored_sha256", "raw_backbone_pre_sha256",
        "raw_backbone_post_sha256", "initial_loss", "final_loss",
    )
    if any(key not in observed or key not in recomputed for key in scientific):
        raise ValueError("registered canary fitted scientific schema differs")
    if any(
        observed[key] != recomputed[key]
        for key in scientific
    ):
        raise ValueError("registered canary fitted scientific authority differs")
    for key in (
        "initialization_seconds", "fit_seconds",
        "peak_allocated_bytes", "peak_reserved_bytes",
    ):
        for value in (observed.get(key), recomputed.get(key)):
            if value is None:
                continue
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("registered canary observational telemetry differs")


def recompute_registered_canary_dynamic_authority(
    *, config: dict[str, object], device: object, torch, fepf, trainer,
    raw_model: object, eval_transform: object, optimization: Sequence[object],
    labels: Mapping[str, int], registered_cache: object | None = None,
) -> dict[str, object]:
    """Rerun the registered deterministic fit from externally rebuilt inputs."""

    raw_pre = trainer.raw_backbone_state_sha256(raw_model)
    result = run_registered_fepf_canary(
        config=config, device=device, torch=torch, fepf=fepf, trainer=trainer,
        raw_model=raw_model, eval_transform=eval_transform,
        optimization=optimization, labels=labels, registered_cache=registered_cache,
    )
    raw_post = trainer.raw_backbone_state_sha256(raw_model)
    return {
        "completed_steps": 512,
        "initial_head_sha256": result["initial_head_sha256"],
        "final_head_sha256": result["final_head_sha256"],
        "diagnostic_sha256": result["diagnostic_sha256"],
        "rng_entry_sha256": result["rng_entry_sha256"],
        "rng_post_draw_sha256": result["rng_post_draw_sha256"],
        "rng_restored_sha256": result["rng_restored_sha256"],
        "raw_backbone_pre_sha256": raw_pre,
        "raw_backbone_post_sha256": raw_post,
        "initial_loss": result["initial_loss"],
        "final_loss": result["final_loss"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "initialization_receipt": result["initialization_receipt"],
        "rng_audit": {
            "entry": result["rng_entry_sha256"],
            "post_draw": result["rng_post_draw_sha256"],
            "restored": result["rng_restored_sha256"],
            "preimages": result["rng_preimages"],
        },
        "cache_inventory": result["cache_inventory"],
        "model_modes": result["model_modes"],
    }


def validate_non_authentic_cpu_canary_family(
    config: dict[str, object],
    manifest_path: Path,
    terminal: object,
    *,
    live_model_inventory: dict[str, object],
    live_cache_inventory: dict[str, object],
    live_environment: dict[str, object],
) -> dict[str, object]:
    """CPU-only schema/core seam; never authentic target canary evidence."""

    manifest = reconstruct_canary_authority(
        config, manifest_path, terminal=terminal
    )
    root = manifest_path.parent
    initialization = json.loads((root / "initialization-receipt.json").read_bytes())
    model = json.loads((root / "model-inventory.json").read_bytes())
    cache = json.loads((root / "cache-inventory.json").read_bytes())
    environment = json.loads((root / "environment.json").read_bytes())
    import torch

    from sfora import unicom_fepf as fepf

    expected = fepf.FepfExpectedProvenance(
        mode="fepf_mean",
        training_seed=0,
        holdout_fraction=0.2,
        holdout_seed=0,
        source_sha256=initialization["source_sha256"],
        checkpoint_sha256=config["model"]["checkpoint_sha256"],
        config_sha256=_sha256(_canonical_json(config)),
        schedule_sha256=initialization["schedule_sha256"],
        receipt_sha256=fepf.canonical_initialization_receipt_v2_sha256(initialization),
    )
    fepf._validate_initialization_receipt_v2_core(
        initialization,
        expected=expected,
        device=torch.device("cpu"),
        allow_test_device=True,
    )
    if (
        model != _json_normalized(live_model_inventory)
        or cache != _json_normalized(live_cache_inventory)
        or environment != _json_normalized(live_environment)
    ):
        raise ValueError("non-authentic CPU canary external reconstruction differs")
    return manifest


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
        "evidence_manifest_sha256",
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
            "pyproject_sha256", "uv_lock_sha256", "deterministic_execution",
        )
        or not _lower_sha256(expected_sha256)
        or _sha256(_canonical_json(payload)) != expected_sha256
    ):
        raise ValueError("CUDA canary environment payload differs")
    return True


def publish_canary_environment(
    path: Path, environment: dict[str, object], *, config: dict[str, object] | None = None
) -> dict[str, object]:
    payload = _canonical_json(environment)
    digest = _sha256(payload)
    validate_canary_environment_payload(environment, digest)
    authority = {
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": len(payload),
    }
    publisher = None
    if config is not None:
        root = Path(config["artifact_root"])
        publisher = BudgetedPublisher(
            campaign_root=root,
            budget_path=root / config["publication_budget_path"],
            budget_sha256=config["publication_budget_sha256"],
            exact_budget=config["publication_budget"],
        )
        publisher.validate_payload(
            name="cuda-canary:environment", destination=path, payload=payload
        )
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(path)
        return authority
    def validator(persisted: bytes) -> None:
        if persisted != payload:
            raise ValueError("canary evidence differs")

    if publisher is None:
        published = publish_bytes_noreplace(path, payload, validator=validator)
    else:
        published = publisher.publish_bytes(
            name="cuda-canary:environment",
            destination=path,
            payload=payload,
            validator=validator,
        )
    published.close()
    if path.read_bytes() != payload:
        raise RuntimeError("CUDA canary environment publication differs")
    return authority


def validate_canary_evidence_manifest(
    authority: object, *, evidence_root: Path
) -> dict[str, object]:
    if type(authority) is not dict or tuple(authority) != ("path", "sha256", "bytes"):
        raise ValueError("canary evidence manifest authority differs")
    path = Path(authority["path"])
    payload = path.read_bytes()
    if (
        path.is_symlink()
        or path.resolve().parent != evidence_root.resolve()
        or _sha256(payload) != authority["sha256"]
        or len(payload) != authority["bytes"]
    ):
        raise ValueError("canary evidence manifest authority differs")
    value = json.loads(payload)
    if _canonical_json(value) != payload or value.get("schema") != "unicom-fepf-canary-evidence-v1":
        raise ValueError("canary evidence manifest authority differs")
    expected_names = {
        "initialization_receipt", "cache_inventory", "model_inventory",
        "rng_audit", "model_modes", "environment", "observation",
    }
    if set(value.get("objects", {})) != expected_names:
        raise ValueError("canary evidence manifest authority differs")
    loaded: dict[str, object] = {}
    for name, binding in value["objects"].items():
        object_path = evidence_root / binding["path"]
        object_payload = object_path.read_bytes()
        if (
            object_path.is_symlink()
            or _sha256(object_payload) != binding["sha256"]
            or len(object_payload) != binding["bytes"]
            or _canonical_json(json.loads(object_payload)) != object_payload
        ):
            raise ValueError("canary evidence object authority differs")
        loaded[name] = json.loads(object_payload)
    initialization = loaded["initialization_receipt"]
    cache = loaded["cache_inventory"]
    model = loaded["model_inventory"]
    rng = loaded["rng_audit"]
    modes = loaded["model_modes"]
    if (
        type(initialization) is not dict
        or initialization.get("schema") != "initialization-receipt-v2"
        or type(cache) is not dict
        or cache.get("schema") != "unicom-fepf-canary-cache-v1"
        or type(cache.get("tensors")) is not list
        or not cache["tensors"]
        or type(model) is not dict
        or model.get("schema") != "unicom-fepf-canary-model-v1"
        or type(model.get("tensors")) is not list
        or not model["tensors"]
        or type(rng) is not dict
        or type(rng.get("preimages")) is not dict
        or type(modes) is not dict
        or tuple(modes) != ("before", "after", "restored")
        or type(modes["before"]) is not bool
        or type(modes["after"]) is not bool
        or modes["restored"] is not modes["before"]
    ):
        raise ValueError("Task 1 cache/model/RNG/mode authority differs")
    if initialization.get("schema") == "initialization-receipt-v2":
        preimages = rng.get("preimages") if type(rng) is dict else None
        if type(preimages) is not dict:
            raise ValueError("canary RNG preimage authority differs")
        for phase in ("entry", "post_draw", "restored"):
            preimage = preimages.get(phase)
            if type(preimage) is not dict:
                raise ValueError("canary RNG preimage authority differs")
            python_hash = _sha256(
                b"python-random-v1\0" + bytes.fromhex(preimage["python_pickle_v5_hex"])
            )
            numpy_hash = _sha256(
                b"numpy-random-v1\0" + bytes.fromhex(preimage["numpy_pickle_v5_hex"])
            )
            torch_hash = _sha256(
                b"torch-cpu-random-v1\0" + bytes.fromhex(preimage["torch_cpu_hex"])
            )
            cuda_hashes = [
                _sha256(
                    f"torch-cuda-random-v1:{index}".encode()
                    + b"\0"
                    + bytes.fromhex(item)
                )
                for index, item in enumerate(preimage.get("torch_cuda_hex", []))
            ]
            suffix = phase
            if (
                initialization.get(f"python_rng_{suffix}_sha256") != python_hash
                or initialization.get(f"numpy_rng_{suffix}_sha256") != numpy_hash
                or initialization.get(f"torch_cpu_rng_{suffix}_sha256") != torch_hash
                or initialization.get(f"torch_cuda_rng_{suffix}_sha256")
                != cuda_hashes
            ):
                raise ValueError("canary RNG preimage authority differs")
    return value


def _publish_or_adopt_bytes(
    path: Path,
    payload: bytes,
    *,
    publisher: BudgetedPublisher | None = None,
    name: str | None = None,
) -> None:
    if publisher is not None:
        if name is None:
            raise ValueError("canary budget name differs")
        publisher.validate_payload(name=name, destination=path, payload=payload)
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(path)
        return
    with publish_bytes_noreplace(
        path,
        payload,
        validator=lambda persisted: (
            None
            if persisted == payload
            else (_ for _ in ()).throw(ValueError("canary transaction bytes differ"))
        ),
    ):
        pass


def _rename_directory_noreplace(
    parent_descriptor: int, source_name: str, destination_name: str
) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is required for canary transaction publication")
    if renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        1,
    ):
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination_name)
        raise OSError(error, os.strerror(error), destination_name)


def _canary_transaction_paths(root: Path) -> tuple[Path, Path]:
    preflight = root / "preflight"
    return preflight / "canary-evidence.staging", preflight / "canary-evidence"


def _validate_staging_prefix(staging: Path) -> None:
    observed: set[str] = set()
    for child in staging.iterdir():
        if child.is_symlink() or not child.is_file():
            raise ValueError("canary staging prefix differs")
        payload = child.read_bytes()
        try:
            value = json.loads(payload)
        except Exception as error:
            raise ValueError("canary staging prefix differs") from error
        if type(value) is not dict or _canonical_json(value) != payload:
            raise ValueError("canary staging prefix differs")
        observed.add(child.name)
    if not any(
        observed == set(CANARY_EVIDENCE_ORDER[:length])
        for length in range(len(CANARY_EVIDENCE_ORDER) + 1)
    ):
        raise ValueError("canary staging prefix differs")


def _load_canonical_observation(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if path.is_symlink() or type(value) is not dict or _canonical_json(value) != payload:
        raise ValueError("canary observation authority differs")
    return value


def complete_canary_evidence_transaction(
    config: dict[str, object],
    *,
    backend: Callable[[dict[str, object]], dict[str, object]],
    crash_after_publication: int | None = None,
) -> tuple[dict[str, object], Path]:
    """Install one deterministic real family, adopting its observation prefix."""

    root = Path(config["artifact_root"])
    publisher = BudgetedPublisher(
        campaign_root=root,
        budget_path=root / config["publication_budget_path"],
        budget_sha256=config["publication_budget_sha256"],
        exact_budget=config["publication_budget"],
    )
    staging, evidence = _canary_transaction_paths(root)
    preflight = evidence.parent
    parent_info = preflight.lstat()
    parent_descriptor = os.open(preflight, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    opened_parent = os.fstat(parent_descriptor)
    if (parent_info.st_dev, parent_info.st_ino) != (
        opened_parent.st_dev, opened_parent.st_ino
    ):
        os.close(parent_descriptor)
        raise ValueError("canary preflight ownership differs")
    try:
        if os.path.lexists(evidence):
            if evidence.is_symlink() or os.path.lexists(staging):
                raise ValueError("canary evidence transaction state differs")
            observation = _load_canonical_observation(evidence / "observation.json")
            validate_registered_canary_family(
                config, evidence / "manifest.json", None, evidence_root=evidence
            )
            return observation, evidence / "manifest.json"
        if not os.path.lexists(staging):
            staging.mkdir(mode=0o700)
        elif staging.is_symlink() or not staging.is_dir():
            raise ValueError("canary staging authority differs")
        _validate_staging_prefix(staging)
        observation_path = staging / "observation.json"
        publication_index = 0
        if observation_path.exists():
            observation = _load_canonical_observation(observation_path)
        else:
            if any(staging.iterdir()):
                raise ValueError("canary staging lacks first observation")
            observation = dict(backend(config))
            if observation.get("cuda") is not True:
                raise RuntimeError("CUDA canary requires real CUDA")
            _publish_or_adopt_bytes(
                observation_path,
                _canonical_json(observation),
                publisher=publisher,
                name="cuda-canary:staging-observation",
            )
            publication_index = 1
            if crash_after_publication == publication_index:
                raise RuntimeError("injected canary publication crash")
        objects = observation.get("canary_objects")
        required = {
            "initialization_receipt", "cache_inventory", "model_inventory",
            "rng_audit", "model_modes", "environment",
        }
        if type(objects) is not dict or set(objects) != required:
            raise ValueError("canary observation object authority differs")
        bindings: dict[str, object] = {
            "observation": {
                "path": observation_path.name,
                "sha256": _sha256(observation_path.read_bytes()),
                "bytes": observation_path.stat().st_size,
            }
        }
        publication_index = max(publication_index, 1)
        for name in (
            "initialization_receipt", "cache_inventory", "model_inventory",
            "rng_audit", "model_modes", "environment",
        ):
            path = staging / f"{name.replace('_', '-')}.json"
            payload = _canonical_json(objects[name])
            _publish_or_adopt_bytes(
                path,
                payload,
                publisher=publisher,
                name=f"cuda-canary:staging-{name.replace('_', '-')}",
            )
            bindings[name] = {
                "path": path.name, "sha256": _sha256(payload), "bytes": len(payload),
            }
            publication_index += 1
            if crash_after_publication == publication_index:
                raise RuntimeError("injected canary publication crash")
        manifest_path = staging / "manifest.json"
        manifest_payload = _canonical_json({
            "schema": "unicom-fepf-canary-evidence-v1",
            "objects": bindings,
        })
        _publish_or_adopt_bytes(
            manifest_path,
            manifest_payload,
            publisher=publisher,
            name="cuda-canary:staging-manifest",
        )
        publication_index += 1
        if crash_after_publication == publication_index:
            raise RuntimeError("injected canary publication crash")
        validate_registered_canary_family(
            config, manifest_path, None, evidence_root=staging
        )
        directory = os.open(staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        _rename_directory_noreplace(parent_descriptor, staging.name, evidence.name)
        os.fsync(parent_descriptor)
        publication_index += 1
        if crash_after_publication == publication_index:
            raise RuntimeError("injected canary publication crash")
        return observation, evidence / "manifest.json"
    finally:
        os.close(parent_descriptor)


def validate_canary_handoff(config_path: Path, repository: Path) -> dict[str, str]:
    builder = _load_script(
        repository / "scripts/build_unicom_fepf_run_config.py", "canary_handoff_builder"
    )
    validator = getattr(builder, "validate_config_membership", None)
    if not callable(validator):
        raise ValueError("canary committed handoff validator differs")
    return validator(config_path, repository)


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
    payload = _canonical_json(receipt)
    validate_canary_publication_capacity(
        value, name="receipt", destination=output, payload=payload
    )
    publisher = BudgetedPublisher(
        campaign_root=Path(value["artifact_root"]),
        budget_path=(
            Path(value["artifact_root"]) / value["publication_budget_path"]
        ),
        budget_sha256=value["publication_budget_sha256"],
        exact_budget=value["publication_budget"],
    )
    publisher.validate_payload(
        name="cuda-canary:receipt", destination=output, payload=payload
    )
    def validate(persisted_payload: bytes) -> None:
        if persisted_payload != payload:
            raise RuntimeError("CUDA canary persisted bytes differ")
        validate_cuda_canary_receipt(
            json.loads(persisted_payload), value, expected_device_uuid=expected_device_uuid,
            expected_environment_sha256=expected_environment_sha256,
        )

    if os.path.lexists(output):
        if output.is_symlink() or not output.is_file():
            raise FileExistsError(output)
        persisted = output.read_bytes()
        validate(persisted)
        return output
    published = publisher.publish_bytes(
        name="cuda-canary:receipt",
        destination=output,
        payload=payload,
        validator=validate,
    )
    published.close()
    return output


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
    revision_loader = getattr(trainer, "_git_revision", None)
    checkout = Path(config["inputs"]["unicom_checkout"])
    expected_revision = config["model"].get("revision")
    if (
        not callable(loader)
        or not callable(hasher)
        or not callable(revision_loader)
        or type(expected_revision) is not str
        or revision_loader(checkout) != expected_revision
    ):
        raise ValueError("registered model authority differs")
    raw_model, transform = loader(
        checkout,
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


def prepare_registered_canary_seeded_model(
    *, torch: object, device: object, loader: Callable[[], tuple[object, object, str]]
) -> tuple[object, object, str]:
    """Establish the one registered seed/allocator/model-construction boundary."""

    torch.manual_seed(23_001)
    torch.cuda.reset_peak_memory_stats(device)
    return loader()


def _rng_snapshot_preimage(snapshot: tuple[object, object, object, tuple[object, ...]]):
    python_state, numpy_state, torch_state, cuda_states = snapshot
    return {
        "python_pickle_v5_hex": pickle.dumps(python_state, protocol=5).hex(),
        "numpy_pickle_v5_hex": pickle.dumps(numpy_state, protocol=5).hex(),
        "torch_cpu_hex": bytes(torch_state.tolist()).hex(),
        "torch_cuda_hex": [bytes(state.tolist()).hex() for state in cuda_states],
    }


def configure_deterministic_canary_execution(
    torch: object, *, environment: MutableMapping[str, str]
) -> dict[str, object]:
    """Establish and report the cache-reconstruction execution envelope."""

    required_workspace = ":4096:8"
    existing = environment.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in (None, required_workspace):
        raise ValueError("CUBLAS deterministic workspace authority differs")
    environment["CUBLAS_WORKSPACE_CONFIG"] = required_workspace
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "deterministic_algorithms": True,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": required_workspace,
    }


def run_registered_fepf_canary(
    *, config: dict[str, object], device, torch, fepf, trainer, raw_model,
    eval_transform, optimization: tuple[object, ...], labels: dict[str, int],
    registered_cache: object | None = None,
) -> dict[str, object]:
    previous_mode = bool(raw_model.training)
    entry = trainer._global_rng_snapshot()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    official = torch.empty((len(labels), 768), dtype=torch.float32)
    torch.nn.init.normal_(official, std=0.01)
    post_draw = trainer._global_rng_snapshot()
    mode_after = previous_mode
    try:
        cache = registered_cache
        if cache is None:
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
        try:
            rng_audit = capture_canary_rng_audit(trainer, entry, post_draw)
            restored = trainer._global_rng_snapshot()
            mode_after = bool(raw_model.training)
        finally:
            raw_model.train(previous_mode)
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
                f"{fit.final_head_sha256}"
            ).encode()
        ),
        "rng_entry_sha256": initialization["torch_cpu_rng_entry_sha256"],
        "rng_post_draw_sha256": initialization["torch_cpu_rng_post_draw_sha256"],
        "rng_restored_sha256": initialization["torch_cpu_rng_restored_sha256"],
        "initial_loss": fit.initial_loss,
        "final_loss": fit.final_loss,
        "initialization_receipt": initialization,
        "cache_inventory": {
            "schema": "unicom-fepf-canary-cache-v1",
            "tensors": sorted([
                {
                    "name": name,
                    "kind": "cache",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": _tensor_hash(value),
                }
                for name, value in sorted(vars(cache).items())
                if hasattr(value, "detach") and hasattr(value, "shape")
            ], key=lambda row: row["name"]),
            "authorities": {
                name: value
                for name, value in sorted(vars(cache).items())
                if not (hasattr(value, "detach") and hasattr(value, "shape"))
            },
        },
        "rng_preimages": {
            "entry": _rng_snapshot_preimage(entry),
            "post_draw": _rng_snapshot_preimage(post_draw),
            "restored": _rng_snapshot_preimage(restored),
        },
        "model_modes": {
            "before": previous_mode,
            "after": mode_after,
            "restored": bool(raw_model.training),
        },
    }


def _real_cuda_backend(config: dict[str, object]) -> dict[str, object]:
    """Run 512 real CUDA classifier steps on a deterministic registered-width fixture."""

    existing_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_workspace not in (None, ":4096:8"):
        raise ValueError("CUBLAS deterministic workspace authority differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import numpy as np
    import timm
    import torch
    import torchvision

    from sfora import unicom_fepf as fepf
    from sfora.unicom_inshop import parse_inshop_partition

    if not torch.cuda.is_available():
        return {"cuda": False}
    deterministic = configure_deterministic_canary_execution(
        torch, environment=os.environ
    )
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is None:
        # UUID is mandatory claim evidence; never substitute a device name.
        raise RuntimeError("CUDA device UUID is unavailable")
    repository = Path(__file__).resolve().parents[1]
    trainer = _load_script(
        repository / "scripts/train_unicom_inshop.py", "task6_canary_trainer"
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
    raw_model, eval_transform, raw_pre = prepare_registered_canary_seeded_model(
        torch=torch,
        device=device,
        loader=lambda: load_registered_canary_model(
            config, trainer=trainer, device=device
        ),
    )
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
        "deterministic_execution": deterministic,
    }
    raw_post = trainer.raw_backbone_state_sha256(raw_model)
    if raw_post != raw_pre:
        raise ValueError("registered raw backbone changed during canary")
    parameters = list(raw_model.named_parameters())
    buffers = list(raw_model.named_buffers())
    if {name for name, _tensor in parameters} & {name for name, _tensor in buffers}:
        raise ValueError("registered canary model tensor kinds overlap")
    model_inventory = {
        "schema": "unicom-fepf-canary-model-v1",
        "revision": config["model"]["revision"],
        "tensors": sorted([
            {
                "name": name,
                "kind": kind,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": _tensor_hash(tensor),
            }
            for kind, values in (
                ("parameter", parameters),
                ("buffer", buffers),
            )
            for name, tensor in values
        ], key=lambda row: (row["name"], row["kind"])),
    }
    canary_objects = {
        "initialization_receipt": registered.pop("initialization_receipt"),
        "cache_inventory": registered.pop("cache_inventory"),
        "model_inventory": model_inventory,
        "rng_audit": {
            "entry": registered["rng_entry_sha256"],
            "post_draw": registered["rng_post_draw_sha256"],
            "restored": registered["rng_restored_sha256"],
            "preimages": registered.pop("rng_preimages"),
        },
        "model_modes": registered.pop("model_modes"),
        "environment": environment,
    }
    return {
        "cuda": True,
        "canary_objects": canary_objects,
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
    crash_after_publication: int | None = None,
) -> Path:
    value = _config_authority(config)
    authenticate_canary_inputs(value)
    observed, manifest_path = complete_canary_evidence_transaction(
        value,
        backend=backend,
        crash_after_publication=crash_after_publication,
    )
    observed = dict(observed)
    if observed.pop("cuda", None) is not True:
        raise RuntimeError("CUDA canary requires real CUDA")
    observed.pop("canary_objects", None)
    observed["evidence_manifest_sha256"] = _sha256(manifest_path.read_bytes())
    observed = {key: observed[key] for key in RECEIPT_KEYS[6:]}
    authority = value.get("cuda_canary_authority")
    if type(authority) is not dict:
        raise ValueError("CUDA canary external authority differs")
    observed_environment = observed.get("environment")
    if type(observed_environment) is not dict:
        raise ValueError("CUDA canary environment differs")
    observed_environment_sha256 = _sha256(_canonical_json(observed_environment))
    observed_device_uuid = observed.get("device_uuid")
    expected_device_uuid = authority.get("device_uuid", observed_device_uuid)
    expected_environment_sha256 = authority.get(
        "environment_sha256", observed_environment_sha256
    )
    environment_authority = value.get("cuda_canary_environment")
    if type(environment_authority) is not dict:
        raise ValueError("CUDA canary environment publication authority differs")
    environment_payload = _canonical_json(observed_environment)
    validate_canary_publication_capacity(
        value,
        name="environment",
        destination=Path(environment_authority.get("path", "")),
        payload=environment_payload,
    )
    published_environment = publish_canary_environment(
        Path(environment_authority.get("path", "")),
        observed["environment"],
        config=value,
    )
    expected_published_environment = {
        "path": str(Path(environment_authority.get("path", "")).resolve()),
        "sha256": observed_environment_sha256,
        "bytes": len(_canonical_json(observed_environment)),
    }
    if published_environment != expected_published_environment:
        raise ValueError("CUDA canary environment publication authority differs")
    if crash_after_publication == 10:
        raise RuntimeError("injected canary publication crash")
    receipt = build_cuda_canary_receipt(
        value, observed, expected_device_uuid=expected_device_uuid,
        expected_environment_sha256=expected_environment_sha256,
    )
    output = publish_cuda_canary_receipt(
        receipt, value, expected_device_uuid=expected_device_uuid,
        expected_environment_sha256=expected_environment_sha256,
    )
    if crash_after_publication == 11:
        raise RuntimeError("injected canary publication crash")
    return output


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--publication-stage", default="cuda-canary")
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--authority-preflight-only", action="store_true")
    parser.add_argument("--non-authentic-synthesized-authorities", action="store_true")
    return parser.parse_args(arguments)


def main(
    arguments: Sequence[str] | None = None,
    *,
    backend: Callable[[dict[str, object]], dict[str, object]] = _real_cuda_backend,
    crash_after_publication: int | None = None,
) -> int:
    args = parse_args(arguments)
    try:
        repository = Path(__file__).resolve().parents[1]
        if args.non_authentic_synthesized_authorities:
            if not args.authority_preflight_only:
                raise ValueError("non-authentic authority seam is preflight-only")
            builder = _load_script(
                repository / "scripts/build_unicom_fepf_run_config.py",
                "canary_non_authentic_handoff_builder",
            )
            builder.validate_non_authentic_synthesized_membership(
                args.config, repository
            )
        else:
            validate_canary_handoff(args.config, repository)
        config = json.loads(args.config.read_bytes())
        validate_canary_exact_budget(
            config, external=not args.authority_preflight_only
        )
        ensure_campaign_root(
            config, physical_admission=not args.authority_preflight_only
        )
        root = Path(config["artifact_root"])
        if args.campaign_root is not None and args.campaign_root.resolve() != root.resolve():
            raise ValueError("canary campaign root differs")
        rows = config.get("publication_budget", {}).get("publications", [])
        required = {
            "cuda-canary:environment",
            "cuda-canary:manifest",
            "cuda-canary:receipt",
        }
        names = {row.get("name") for row in rows if type(row) is dict}
        budget_payload = _canonical_json(config.get("publication_budget"))
        if (
            args.publication_stage != "cuda-canary"
            or not required <= names
            or _sha256(budget_payload) != config.get("publication_budget_sha256")
        ):
            raise ValueError("canary publication budget authority differs")
        if args.authority_preflight_only:
            return 0
        run_cuda_canary(
            config,
            backend=backend,
            crash_after_publication=crash_after_publication,
        )
    except Exception as error:
        print(f"FEPF CUDA canary failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
