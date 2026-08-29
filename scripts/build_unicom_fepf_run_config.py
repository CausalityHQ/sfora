#!/usr/bin/env python3
"""Build and authenticate the immutable UniCOM FEPF campaign config."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sfora.atomic_publication import publish_bytes_noreplace

UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
UNICOM_CHECKPOINT_SHA256 = "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
INSHOP_PARTITION_SHA256 = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
PARENT_TRAINER_COMMIT = "70c760e57e6c27dec1473eecd4765e0a8cd4cf6b"
PARENT_TRAINER_PATH = "scripts/train_unicom_inshop.py"
PARENT_TRAINER_SHA256 = "6eea2dab88ff9e4c5a547f9fe326ebf56879882784c5a80c8e136f6d02b52170"
RUNTIME_ORDER = ("current", "composed", "composed", "current") * 2
CONFIRMATION_PAIRS = (
    (7, 20_260_828),
    (8, 271_828),
    (9, 314_159),
    (10, 1_618_033),
    (11, 57_721),
)
REGISTERED_SOURCE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "src/sfora/unicom_inshop.py",
    "src/sfora/unicom_fepf.py",
    "src/sfora/unicom_retrieval_audit.py",
    "src/sfora/atomic_publication.py",
    "src/sfora/cuda_authority.py",
    "scripts/train_unicom_inshop.py",
    "scripts/profile_unicom_training_step.py",
    "scripts/evaluate_unicom_fepf.py",
    "scripts/build_unicom_fepf_run_config.py",
    "scripts/run_unicom_fepf_campaign.py",
    "scripts/run_unicom_fepf_cuda_canary.py",
)
CONFIG_KEYS = (
    "schema",
    "source_commit",
    "source_files",
    "source",
    "handoff",
    "model",
    "inputs",
    "parent_trainer_commit",
    "parent_trainer_path",
    "parent_trainer_sha256",
    "live_trainer_sha256",
    "profiler_sha256",
    "fepf_inference_structure",
    "runtime_inference_signature",
    "cuda_canary_environment",
    "publication_budget",
    "publication_budget_path",
    "publication_budget_sha256",
    "legacy_runtime_authority",
    "checkout_root_template",
    "artifact_root",
    "runtime_order",
    "exploratory",
    "confirmation_pairs",
    "thresholds",
    "artifact_budget_inputs",
    "artifact_budget_bytes",
    "artifact_budget_inodes",
    "cuda_canary_authority",
    "cuda_canary_command",
    "cuda_canary_receipt",
    "commands",
)
_LOWER_COMMIT = re.compile(r"[0-9a-f]{40}").fullmatch
CANARY_EVIDENCE_ORDER = (
    "observation.json", "initialization-receipt.json", "cache-inventory.json",
    "model-inventory.json", "rng-audit.json", "model-modes.json",
    "environment.json", "manifest.json",
)

# Frozen conservative sizing authorities. The authenticated raw-backbone byte
# count is supplied by the runtime checkpoint signature; the eightfold
# multiplier covers the complete checkpoint payload rather than pretending
# state_dict tensor bytes are the serialized checkpoint size.
CLASSIFIER_STATE_BYTES = 3_997 * 768 * 4
QUALITY_CHECKPOINTS = 13 * 4
ATOMIC_COPY_FACTOR = 2


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _git(repo: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True,
        text=not binary,
    )
    return result.stdout


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_absolute_plain(path: Path, name: str) -> Path:
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ValueError(f"{name} must be a normalized absolute path")
    return path


def _validate_template(template: str) -> None:
    if type(template) is not str or template.count("{config_commit}") != 1:
        raise ValueError("checkout root template differs")
    residual = template.replace("{config_commit}", "")
    if "{" in residual or "}" in residual:
        raise ValueError("checkout root template differs")
    expanded = Path(template.replace("{config_commit}", "0" * 40))
    _require_absolute_plain(expanded, "checkout root template")


def _paths_nested(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _source_inventory(repo: Path, commit: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in REGISTERED_SOURCE_PATHS:
        payload = _git(repo, "show", f"{commit}:{relative}", binary=True)
        assert isinstance(payload, bytes)
        rows.append({"path": relative, "sha256": _sha256(payload), "bytes": len(payload)})
    return rows


def _partition_inventory(path: Path) -> dict[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("registered partition differs")
    rows = [line.split() for line in path.read_text().splitlines()[1:] if line.strip()]
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError("registered partition differs")
    queries = [row for row in rows if row[2] == "query"]
    galleries = [row for row in rows if row[2] == "gallery"]
    gallery_counts: dict[str, int] = {}
    for _image, label, _split in galleries:
        gallery_counts[label] = gallery_counts.get(label, 0) + 1
    relevant = [gallery_counts.get(label, 0) for _image, label, _split in queries]
    if not queries or not galleries or not relevant or min(relevant) <= 0:
        raise ValueError("registered partition relevance differs")
    return {
        "query_rows": len(queries), "gallery_rows": len(galleries),
        "maximum_relevant_count": max(relevant),
        "maximum_path_bytes": max(len(row[0].encode()) for row in rows),
    }


def _budget(
    partition: dict[str, int], raw_backbone_state_bytes: int
) -> tuple[dict[str, int], int, int]:
    required = ("query_rows", "gallery_rows", "maximum_relevant_count", "maximum_path_bytes")
    if tuple(partition) != required or any(type(partition[key]) is not int or partition[key] <= 0
                                            for key in required):
        raise ValueError("artifact partition inventory differs")
    if type(raw_backbone_state_bytes) is not int or raw_backbone_state_bytes <= 0:
        raise ValueError("artifact backbone inventory differs")
    inputs = {
        "quality_checkpoints": QUALITY_CHECKPOINTS,
        "raw_backbone_state_bytes": raw_backbone_state_bytes,
        "classifier_state_bytes": CLASSIFIER_STATE_BYTES,
        **partition,
        # Published evaluation arrays retain the full 768-wide descriptor;
        # runtime_inference_signature binds the deployed 512-wide prefix.
        "descriptor_dimension": 768,
        "atomic_copy_factor": ATOMIC_COPY_FACTOR,
    }
    # Aggregate capacity is derived later solely from exact publication rows.
    return inputs, 0, 0


def _inputs() -> dict[str, str]:
    return {
        "unicom_checkout": "/home/riomus/unicom-d71992e",
        "checkpoint": "/home/riomus/.cache/unicom/FP16-ViT-L-14-336px.pt",
        "dataset_root": "/home/riomus/datasets/inshop_official_standard",
        "partition": "/home/riomus/datasets/inshop_official_standard/Eval/list_eval_partition.txt",
        "runtime_checkpoint": (
            "/home/riomus/group-learning/reports/generated/"
            "unicom-full-width-objective-2026-08-25/seed-2/"
            "sampled_512/epoch-0016.pt"
        ),
        "runtime_run_receipt": (
            "/home/riomus/group-learning/reports/generated/"
            "unicom-full-width-objective-2026-08-25/seed-2/"
            "sampled_512-run-receipt.json"
        ),
    }


def _commands(
    environment: dict[str, object], budget: dict[str, object]
) -> dict[str, object]:
    python = ".venv/bin/python"
    inputs = _inputs()
    return {
        "runtime": [
            [python, "-I", "-B", "scripts/profile_unicom_training_step.py",
             "--profile-kind", "runtime", "--runtime-mode", mode,
             "--run-checkpoint", inputs["runtime_checkpoint"], "--run-receipt",
             inputs["runtime_run_receipt"], "--config",
             "docs/unicom_fepf_run_config.json", "--unicom-checkout",
             inputs["unicom_checkout"], "--initial-checkpoint", inputs["checkpoint"],
             "--dataset-root", inputs["dataset_root"], "--parent-trainer-source",
             f"{PARENT_TRAINER_COMMIT}:{PARENT_TRAINER_PATH}",
             "--environment-authority", environment["path"],
             "--environment-sha256", environment.get(
                 "sha256", "{cuda_environment_sha256}"
             ),
             "--output", "{output}"]
            for mode in RUNTIME_ORDER
        ],
        "train": [
            python, "-I", "-B", "scripts/train_unicom_inshop.py",
            "--unicom-checkout", inputs["unicom_checkout"],
            "--checkpoint", inputs["checkpoint"],
            "--dataset-root", inputs["dataset_root"],
            "--run-config", "docs/unicom_fepf_run_config.json",
            "--environment-authority", environment["path"],
            "--environment-sha256", environment.get(
                "sha256", "{cuda_environment_sha256}"
            ),
            "--publication-budget", budget["path"],
            "--publication-budget-sha256", budget["sha256"],
        ],
        "profile_quality": [
            python, "-I", "-B", "scripts/profile_unicom_training_step.py",
            "--profile-kind", "quality", "--unicom-checkout", inputs["unicom_checkout"],
            "--initial-checkpoint", inputs["checkpoint"], "--dataset-root",
            inputs["dataset_root"], "--parent-trainer-source",
            f"{PARENT_TRAINER_COMMIT}:{PARENT_TRAINER_PATH}",
            "--environment-authority", environment["path"],
            "--environment-sha256", environment.get(
                "sha256", "{cuda_environment_sha256}"
            ),
        ],
        "evaluate": [
            python, "-I", "-B", "scripts/evaluate_unicom_fepf.py",
            "--config", "docs/unicom_fepf_run_config.json",
        ],
        "quality_profile_order": ["imprinted", "fepf_mean", "fepf_mean", "imprinted"],
    }


def _checkpoint_inference_structure(path: Path) -> dict[str, object]:
    import torch

    if path.is_symlink() or not path.is_file():
        raise ValueError("registered runtime checkpoint differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if type(checkpoint) is not dict or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("registered runtime checkpoint differs")
    state = checkpoint["model"]
    ema = checkpoint.get("ema")
    parameter_names = (
        set(ema["backbone"])
        if type(ema) is dict and type(ema.get("backbone")) is dict
        else set(state)
    )
    rows = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError("registered runtime checkpoint tensor differs")
        rows.append({
            "name": name,
            "kind": "parameter" if name in parameter_names else "buffer",
            "shape": list(value.shape), "dtype": str(value.dtype),
            "numel": value.numel(), "element_size": value.element_size(),
            "bytes": value.numel() * value.element_size(),
        })
    classifier = checkpoint.get("classifier")
    if not rows or not isinstance(classifier, torch.Tensor):
        raise ValueError("registered runtime checkpoint structure differs")
    return {
        "schema": "unicom-fepf-structure-v1", "tensors": rows,
        "classifier": {
            "shape": list(classifier.shape), "dtype": str(classifier.dtype),
            "numel": classifier.numel(), "element_size": classifier.element_size(),
            "bytes": classifier.numel() * classifier.element_size(),
        },
        "operations": [
            "official_forward", "full768_l2", "prefix512", "squared_euclidean"
        ],
    }


def _checkpoint_runtime_inference_signature(path: Path) -> dict[str, object]:
    """Reload the registered signature from the authenticated runtime checkpoint."""

    import torch
    from PIL import Image

    from sfora.unicom_inshop import parse_inshop_partition

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if type(checkpoint) is not dict or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("registered runtime inference signature is absent")
    trainer_path = Path(__file__).with_name("train_unicom_inshop.py")
    spec = importlib.util.spec_from_file_location("fepf_signature_trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise ValueError("registered runtime trainer differs")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    raw_model, transform = trainer._load_official_model(
        Path(_inputs()["unicom_checkout"]), Path(_inputs()["checkpoint"])
    )
    raw_model.load_state_dict(checkpoint["model"], strict=True)
    raw_model.eval()
    records = parse_inshop_partition(Path(_inputs()["dataset_root"]))
    record = next((row for row in records if row.split == "query"), None)
    if record is None:
        raise ValueError("registered runtime descriptor record differs")
    with Image.open(record.image_path) as image, torch.no_grad():
        full = raw_model(transform(image.convert("RGB")).unsqueeze(0)).float()
        full = torch.nn.functional.normalize(full, dim=1)
        descriptor = full[:, :512].contiguous()
    return trainer.build_inference_signature(raw_model, descriptor=descriptor)


def _file_binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    if path.is_symlink() or not path.is_file():
        raise ValueError("legacy runtime authority differs")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(payload),
        "bytes": len(payload),
    }


def _registered_legacy_runtime_authority() -> dict[str, object]:
    """Bind the historical seed-2 receipt and every externally retained preimage."""

    inputs = _inputs()
    receipt_path = Path(inputs["runtime_run_receipt"])
    receipt = json.loads(receipt_path.read_bytes())
    if (
        receipt.get("schema_version") != "unicom-full-width-training-run-v1"
        or receipt.get("trainer_sha256") != PARENT_TRAINER_SHA256
        or receipt.get("seed") != 2
        or receipt.get("arm") != "sampled_512"
        or receipt.get("exit_status") != 0
    ):
        raise ValueError("legacy runtime receipt identity differs")
    config_path = Path(receipt["config_path"])
    config = _file_binding(config_path)
    if config["sha256"] != receipt.get("config_sha256"):
        raise ValueError("legacy runtime config authority differs")
    checkpoint_rows = receipt.get("checkpoints")
    if (
        type(checkpoint_rows) is not list
        or any(type(row) is not dict for row in checkpoint_rows)
        or [row.get("epoch") for row in checkpoint_rows] != [4, 8, 12, 16]
        or any(type(row.get("path")) is not str for row in checkpoint_rows)
        or checkpoint_rows[-1]["path"] != inputs["runtime_checkpoint"]
    ):
        raise ValueError("legacy runtime checkpoint authority differs")

    def receipt_bound_file(row: object) -> dict[str, object]:
        if (
            type(row) is not dict
            or type(row.get("path")) is not str
            or not Path(row["path"]).is_absolute()
            or type(row.get("sha256")) is not str
            or type(row.get("bytes")) is not int
        ):
            raise ValueError("legacy runtime file authority differs")
        actual = _file_binding(Path(row["path"]))
        if actual["sha256"] != row["sha256"] or actual["bytes"] != row["bytes"]:
            raise ValueError("legacy runtime file authority differs")
        return actual

    history = receipt_bound_file(receipt.get("history"))
    checkpoints = [
        {"epoch": row["epoch"], **receipt_bound_file(row)}
        for row in checkpoint_rows
    ]
    return {
        "run_receipt": _file_binding(receipt_path),
        "config": config,
        "history": history,
        "checkpoints": checkpoints,
    }


def build_run_config(
    *, repo: Path, checkout_root_template: str, artifact_root: Path,
    inference_structure: dict[str, object], partition_inventory: dict[str, int],
    cuda_canary_authority: dict[str, str],
    cuda_canary_environment: dict[str, object],
    publication_budget: dict[str, object],
    runtime_inference_signature: dict[str, object],
    legacy_runtime_authority: dict[str, object] | None = None,
) -> dict[str, object]:
    repo = _require_absolute_plain(repo.resolve(), "repository")
    artifact_root = _require_absolute_plain(artifact_root, "artifact root")
    _validate_template(checkout_root_template)
    expanded = Path(checkout_root_template.replace("{config_commit}", "0" * 40))
    if _paths_nested(expanded, artifact_root):
        raise ValueError("checkout and artifact roots must be distinct and non-nested")
    source_commit = str(_git(repo, "rev-parse", "HEAD")).strip()
    if _LOWER_COMMIT(source_commit) is None:
        raise ValueError("source commit differs")
    budget_inputs, budget_bytes, budget_inodes = _budget(
        partition_inventory, runtime_inference_signature["total_bytes"]
    )
    source_files = _source_inventory(repo, source_commit)
    source_hashes = {row["path"]: row["sha256"] for row in source_files}
    config = {
        "schema": "unicom-fepf-run-config-v1",
        "source_commit": source_commit,
        "source_files": source_files,
        "source": {"commit": source_commit},
        "handoff": {
            "config_parent": source_commit,
            "config_commit_paths": ["docs/unicom_fepf_run_config.json"],
            "execution_checkout": "config_commit_detached_clean",
        },
        "model": {
            "revision": UNICOM_REVISION,
            "checkpoint_sha256": UNICOM_CHECKPOINT_SHA256,
            "partition_sha256": INSHOP_PARTITION_SHA256,
        },
        "inputs": _inputs(),
        "parent_trainer_commit": PARENT_TRAINER_COMMIT,
        "parent_trainer_path": PARENT_TRAINER_PATH,
        "parent_trainer_sha256": PARENT_TRAINER_SHA256,
        "live_trainer_sha256": source_hashes["scripts/train_unicom_inshop.py"],
        "profiler_sha256": source_hashes["scripts/profile_unicom_training_step.py"],
        "fepf_inference_structure": inference_structure,
        "runtime_inference_signature": runtime_inference_signature,
        "cuda_canary_environment": dict(cuda_canary_environment),
        "publication_budget": {},
        "publication_budget_path": "preflight/publication-budget.json",
        "publication_budget_sha256": "0" * 64,
        "legacy_runtime_authority": (
            dict(legacy_runtime_authority)
            if legacy_runtime_authority is not None
            else {
                "run_receipt": {"path": "/non-authentic/run.json", "sha256": "0" * 64,
                                "bytes": 1},
                "config": {"path": "/non-authentic/config.json", "sha256": "0" * 64,
                           "bytes": 1},
                "history": {"path": "/non-authentic/history.json", "sha256": "0" * 64,
                            "bytes": 1},
                "checkpoints": [
                    {"epoch": epoch, "path": f"/non-authentic/epoch-{epoch:04d}.pt",
                     "sha256": "0" * 64, "bytes": 1}
                    for epoch in (4, 8, 12, 16)
                ],
            }
        ),
        "checkout_root_template": checkout_root_template,
        "artifact_root": str(artifact_root),
        "runtime_order": list(RUNTIME_ORDER),
        "exploratory": {
            "training_seed": 0,
            "holdout_seed": 0,
            "holdout_fraction": 0.2,
            "arms": ["imprinted", "fepf_mean", "fepf_random"],
            "evaluation_epochs": [4, 8, 12, 16],
        },
        "confirmation_pairs": [list(pair) for pair in CONFIRMATION_PAIRS],
        "thresholds": {
            "epoch4_map_delta_min": 0.003,
            "exploratory_map_delta_min": 0.010,
            "confirmation_map_delta_min": 0.010,
            "confirmation_r1_delta_min": 0.005,
            "compute_ratio_max": 1.02,
            "row_norm_rtol": 2e-6,
            "row_norm_atol": 2e-7,
        },
        "artifact_budget_inputs": budget_inputs,
        "artifact_budget_bytes": budget_bytes,
        "artifact_budget_inodes": budget_inodes,
        "cuda_canary_authority": dict(cuda_canary_authority),
        "cuda_canary_command": [
            ".venv/bin/python", "-I", "-B", "scripts/run_unicom_fepf_cuda_canary.py",
            "--config", "docs/unicom_fepf_run_config.json",
        ],
        "cuda_canary_receipt": "preflight/cuda_canary_v1.json",
        "commands": {},
    }
    budget = exact_publication_budget(config)
    config["artifact_budget_bytes"] = sum(
        row["persistent_bytes"] + row["temporary_bytes"]
        for row in budget["publications"]
    )
    config["artifact_budget_inodes"] = sum(
        row["persistent_inodes"] + row["temporary_inodes"]
        for row in budget["publications"]
    )
    budget_payload = canonical_json_bytes(budget)
    config["publication_budget"] = budget
    config["publication_budget_sha256"] = _sha256(budget_payload)
    budget_authority = {
        "path": str((artifact_root / config["publication_budget_path"]).resolve()),
        "sha256": config["publication_budget_sha256"],
        "bytes": len(budget_payload),
    }
    config["commands"] = _commands(cuda_canary_environment, budget_authority)
    return config


def _strict_shape(config: object) -> dict[str, object]:
    if type(config) is not dict or tuple(config) != CONFIG_KEYS:
        raise ValueError("FEPF config schema differs")
    return config


def _validate_config_values(config: dict[str, object]) -> None:
    structure = config.get("fepf_inference_structure")
    structure_valid = (
        type(structure) is dict
        and tuple(structure) == ("schema", "tensors", "classifier", "operations")
        and structure["schema"] == "unicom-fepf-structure-v1"
        and type(structure["tensors"]) is list
        and bool(structure["tensors"])
        and type(structure["classifier"]) is dict
        and structure["operations"]
        == ["official_forward", "full768_l2", "prefix512", "squared_euclidean"]
    )
    def file_authority(value: object) -> bool:
        return (
            type(value) is dict
            and tuple(value) == ("path", "sha256", "bytes")
            and type(value["path"]) is str
            and Path(value["path"]).is_absolute()
            and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
            and type(value["bytes"]) is int
            and value["bytes"] > 0
        )

    legacy = config.get("legacy_runtime_authority")
    legacy_valid = (
        type(legacy) is dict
        and tuple(legacy) == ("run_receipt", "config", "history", "checkpoints")
        and all(file_authority(legacy.get(name)) for name in (
            "run_receipt", "config", "history"
        ))
        and type(legacy.get("checkpoints")) is list
        and [row.get("epoch") for row in legacy["checkpoints"]] == [4, 8, 12, 16]
        and all(
            type(row) is dict
            and tuple(row) == ("epoch", "path", "sha256", "bytes")
            and file_authority({key: row[key] for key in ("path", "sha256", "bytes")})
            for row in legacy["checkpoints"]
        )
        and type(config.get("inputs")) is dict
        and type(config["inputs"].get("runtime_checkpoint")) is str
        and (
            "non-authentic" in Path(legacy["checkpoints"][-1]["path"]).parts
            or legacy["checkpoints"][-1]["path"]
            == config["inputs"]["runtime_checkpoint"]
        )
    )
    environment = config.get("cuda_canary_environment")
    canary_authority = config.get("cuda_canary_authority")
    pre_canary_environment = (
        type(environment) is dict
        and tuple(environment) == ("path",)
        and type(environment["path"]) is str
        and Path(environment["path"]).is_absolute()
        and canary_authority == {}
    )
    bound_environment = (
        file_authority(environment)
        and type(canary_authority) is dict
        and tuple(canary_authority) == ("device_uuid", "environment_sha256")
        and type(canary_authority["device_uuid"]) is str
        and canary_authority["device_uuid"].startswith("GPU-")
        and re.fullmatch(r"[0-9a-f]{64}", canary_authority["environment_sha256"])
        is not None
        and environment["sha256"] == canary_authority["environment_sha256"]
    )

    signature = config.get("runtime_inference_signature")
    def tensor_row_valid(row: object, *, with_hash: bool) -> bool:
        keys = (
            "name", "kind", "shape", "dtype", "numel", "element_size", "bytes",
            *(("sha256",) if with_hash else ()),
        )
        return (
            type(row) is dict
            and tuple(row) == keys
            and type(row["name"]) is str and bool(row["name"])
            and row["kind"] in {"parameter", "buffer"}
            and type(row["shape"]) is list
            and all(type(value) is int and value >= 0 for value in row["shape"])
            and type(row["dtype"]) is str and row["dtype"].startswith("torch.")
            and type(row["numel"]) is int and row["numel"] >= 0
            and type(row["element_size"]) is int and row["element_size"] > 0
            and row["bytes"] == row["numel"] * row["element_size"]
            and (not with_hash or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None)
        )

    signature_valid = (
        type(signature) is dict
        and tuple(signature)
        == (
            "schema", "tensors", "total_bytes", "aggregate_sha256",
            "descriptor_dtype", "descriptor_dimension", "descriptor_sha256",
            "operations",
        )
        and signature["schema"] == "unicom-inference-signature-v1"
        and type(signature["tensors"]) is list
        and bool(signature["tensors"])
        and all(tensor_row_valid(row, with_hash=True) for row in signature["tensors"])
        and type(signature["total_bytes"]) is int
        and signature["total_bytes"] > 0
        and signature["total_bytes"]
        == sum(row["bytes"] for row in signature["tensors"])
        and re.fullmatch(r"[0-9a-f]{64}", signature["aggregate_sha256"])
        is not None
        and signature["descriptor_dtype"] == "torch.float32"
        and signature["descriptor_dimension"] == 512
        and re.fullmatch(r"[0-9a-f]{64}", signature["descriptor_sha256"])
        is not None
        and signature["operations"]
        == ["official_forward", "full768_l2", "prefix512", "squared_euclidean"]
    )
    if structure_valid:
        structure_valid = all(
            tensor_row_valid(row, with_hash=False) for row in structure["tensors"]
        )
    if signature_valid and structure_valid:
        signature_structure = [
            {key: value for key, value in row.items() if key != "sha256"}
            for row in signature["tensors"]
        ]
        if signature_structure != structure["tensors"]:
            raise ValueError("runtime inference signature structure differs")
    if (
        config["schema"] != "unicom-fepf-run-config-v1"
        or _LOWER_COMMIT(config["source_commit"]) is None
        or config["model"] != {
            "revision": UNICOM_REVISION,
            "checkpoint_sha256": UNICOM_CHECKPOINT_SHA256,
            "partition_sha256": INSHOP_PARTITION_SHA256,
        }
        or config["inputs"] != _inputs()
        or config["source"] != {"commit": config["source_commit"]}
        or config["handoff"]
        != {
            "config_parent": config["source_commit"],
            "config_commit_paths": ["docs/unicom_fepf_run_config.json"],
            "execution_checkout": "config_commit_detached_clean",
        }
        or config["parent_trainer_commit"] != PARENT_TRAINER_COMMIT
        or config["parent_trainer_path"] != PARENT_TRAINER_PATH
        or config["parent_trainer_sha256"] != PARENT_TRAINER_SHA256
        or config["live_trainer_sha256"]
        != {row["path"]: row["sha256"] for row in config["source_files"]}.get(
            "scripts/train_unicom_inshop.py"
        )
        or config["profiler_sha256"]
        != {row["path"]: row["sha256"] for row in config["source_files"]}.get(
            "scripts/profile_unicom_training_step.py"
        )
        or not structure_valid
        or not signature_valid
        or not (pre_canary_environment or bound_environment)
        or type(config.get("publication_budget")) is not dict
        or config.get("publication_budget_path") != "preflight/publication-budget.json"
        or re.fullmatch(r"[0-9a-f]{64}", config.get("publication_budget_sha256", ""))
        is None
        or not legacy_valid
        or config["runtime_order"] != list(RUNTIME_ORDER)
        or config["confirmation_pairs"] != [list(pair) for pair in CONFIRMATION_PAIRS]
        or type(config["artifact_budget_bytes"]) is not int
        or config["artifact_budget_bytes"] <= 0
        or type(config["artifact_budget_inodes"]) is not int
        or config["artifact_budget_inodes"] <= 0
    ):
        raise ValueError("FEPF config protocol differs")
    threshold = config["thresholds"]
    if type(threshold) is not dict or threshold != {
        "epoch4_map_delta_min": 0.003,
        "exploratory_map_delta_min": 0.010,
        "confirmation_map_delta_min": 0.010,
        "confirmation_r1_delta_min": 0.005,
        "compute_ratio_max": 1.02,
        "row_norm_rtol": 2e-6,
        "row_norm_atol": 2e-7,
    }:
        raise ValueError("FEPF config threshold differs")
    if config["exploratory"] != {
        "training_seed": 0, "holdout_seed": 0, "holdout_fraction": 0.2,
        "arms": ["imprinted", "fepf_mean", "fepf_random"],
        "evaluation_epochs": [4, 8, 12, 16],
    } or config["commands"] != _commands(
        config["cuda_canary_environment"],
        {
            "path": str(
                (Path(config["artifact_root"]) / config["publication_budget_path"]).resolve()
            ),
            "sha256": config["publication_budget_sha256"],
            "bytes": len(canonical_json_bytes(config["publication_budget"])),
        },
    ):
        raise ValueError("FEPF config commands differ")
    partition = {
        key: config["artifact_budget_inputs"][key]
        for key in ("query_rows", "gallery_rows", "maximum_relevant_count", "maximum_path_bytes")
    }
    inputs, _budget_bytes, _budget_inodes = _budget(
        partition, config["runtime_inference_signature"]["total_bytes"]
    )
    expected_exact_budget = exact_publication_budget(config)
    exact_bytes = sum(
        row["persistent_bytes"] + row["temporary_bytes"]
        for row in expected_exact_budget["publications"]
    )
    exact_inodes = sum(
        row["persistent_inodes"] + row["temporary_inodes"]
        for row in expected_exact_budget["publications"]
    )
    if (
        config["artifact_budget_inputs"] != inputs
        or config["artifact_budget_bytes"] != exact_bytes
        or config["artifact_budget_inodes"] != exact_inodes
        or config["cuda_canary_command"]
        != [".venv/bin/python", "-I", "-B", "scripts/run_unicom_fepf_cuda_canary.py",
            "--config", "docs/unicom_fepf_run_config.json"]
        or config["cuda_canary_receipt"] != "preflight/cuda_canary_v1.json"
        or not (pre_canary_environment or bound_environment)
    ):
        raise ValueError("FEPF config budget differs")
    _validate_template(config["checkout_root_template"])
    artifact = _require_absolute_plain(Path(config["artifact_root"]), "artifact root")
    checkout = Path(config["checkout_root_template"].replace("{config_commit}", "0" * 40))
    if _paths_nested(artifact, checkout):
        raise ValueError("checkout and artifact roots must be distinct and non-nested")
    validate_exact_publication_budget(config, config["publication_budget"])


def _require_clean(repo: Path) -> None:
    status_text = str(_git(repo, "status", "--porcelain", "--untracked-files=all"))
    if status_text:
        raise ValueError("source checkout is not clean")


def validate_config_build(config: object, repo: Path) -> None:
    value = _strict_shape(config)
    _validate_config_values(value)
    repo = repo.resolve()
    _require_clean(repo)
    head = str(_git(repo, "rev-parse", "HEAD")).strip()
    if head != value["source_commit"]:
        raise ValueError("source commit differs")
    if value["source_files"] != _source_inventory(repo, head):
        raise ValueError("source files differ")
    artifact = Path(value["artifact_root"])
    checkout = Path(value["checkout_root_template"].replace("{config_commit}", "0" * 40))
    if os.path.lexists(artifact) or os.path.lexists(checkout):
        raise FileExistsError("campaign destination already exists")


def build_and_write(
    *, repo: Path, checkout_root_template: str, artifact_root: Path, output: Path,
    inference_structure: dict[str, object] | None = None,
    partition_inventory: dict[str, int] | None = None,
    cuda_canary_authority: dict[str, str],
    cuda_canary_environment: dict[str, object],
    publication_budget: dict[str, object],
    runtime_inference_signature: dict[str, object],
    legacy_runtime_authority: dict[str, object] | None = None,
) -> dict[str, object]:
    if os.path.lexists(output):
        raise FileExistsError(output)
    config = build_run_config(
        repo=repo, checkout_root_template=checkout_root_template,
        artifact_root=artifact_root,
        inference_structure=(
            inference_structure
            if inference_structure is not None
            else _checkpoint_inference_structure(Path(_inputs()["runtime_checkpoint"]))
        ),
        partition_inventory=(
            partition_inventory
            if partition_inventory is not None
            else _partition_inventory(Path(_inputs()["partition"]))
        ),
        cuda_canary_authority=cuda_canary_authority,
        cuda_canary_environment=cuda_canary_environment,
        publication_budget=publication_budget,
        runtime_inference_signature=runtime_inference_signature,
        legacy_runtime_authority=legacy_runtime_authority,
    )
    validate_config_build(config, repo)
    validate_exact_publication_budget(config, config["publication_budget"])
    output.parent.mkdir(parents=True, exist_ok=True)
    config_payload = canonical_json_bytes(config)
    published = publish_bytes_noreplace(
        output,
        config_payload,
        validator=lambda payload: (
            None
            if payload == config_payload and json.loads(payload) == config
            else (_ for _ in ()).throw(ValueError("persisted FEPF config differs"))
        ),
    )
    published.close()
    reloaded = json.loads(output.read_bytes())
    if canonical_json_bytes(reloaded) != output.read_bytes() or reloaded != config:
        raise RuntimeError("persisted FEPF config differs")
    return config


def exact_publication_budget(config: dict[str, object]) -> dict[str, object]:
    """Return the canonical, named publication inventory bound by the config."""

    inputs = config["artifact_budget_inputs"]
    checkpoint_bytes = 8 * (
        inputs["raw_backbone_state_bytes"] + inputs["classifier_state_bytes"]
    ) + 64 * 1024**2
    # NPY v1 carries a magic/version/length prefix and a padded shape/dtype
    # header. 256 bytes is a conservative format-level bound for these 2-D
    # fixed-dtype shapes, independent of future payload values.
    query_bytes = (
        inputs["query_rows"] * inputs["descriptor_dimension"] * 4 + 256
    )
    gallery_bytes = (
        inputs["gallery_rows"] * inputs["descriptor_dimension"] * 4 + 256
    )
    ranked_count = min(
        max(30, inputs["maximum_relevant_count"]), inputs["gallery_rows"]
    )
    maximum_path = "p" * inputs["maximum_path_bytes"]
    ranked_entry = {
        "gallery_index": inputs["gallery_rows"] - 1,
        "gallery_path": maximum_path,
        "gallery_label": maximum_path,
        "score": -1.7976931348623157e308,
        "correct": False,
    }
    query_envelope = {
        "query_path": maximum_path,
        "query_label": maximum_path,
        "relevant_gallery_count": inputs["maximum_relevant_count"],
        "ap_at_r": -1.7976931348623157e308,
        "query_sha256": "f" * 64,
        "complete_ranking_sha256": "f" * 64,
        "ranked_prefix": [dict(ranked_entry) for _index in range(ranked_count)],
    }
    one_query = canonical_json_bytes([query_envelope])
    two_queries = canonical_json_bytes([query_envelope, query_envelope])
    ranked_bytes = len(one_query) + (inputs["query_rows"] - 1) * (
        len(two_queries) - len(one_query)
    )

    def row(name: str, path: str, persistent: int, temporary: int | None = None):
        return {
            "name": name,
            "path": path,
            "persistent_bytes": persistent,
            "temporary_bytes": (
                persistent * (inputs["atomic_copy_factor"] - 1)
                if temporary is None
                else temporary
            ),
            "persistent_inodes": 1,
            "temporary_inodes": 1 if (temporary is None or temporary > 0) else 0,
        }

    publications: list[dict[str, object]] = []
    fresh_stages = (
        "exploratory-control-stage4", "exploratory-candidate-stage4",
        "exploratory-random-stage16",
        *(f"confirmation-{index}-{arm}" for index in range(5)
          for arm in ("control", "candidate")),
    )
    continuation_stages = (
        "exploratory-control-stage16", "exploratory-candidate-stage16",
    )
    for stage in (*fresh_stages, *continuation_stages):
        epochs = (
            (4,) if stage.endswith("stage4")
            else ((8, 12, 16) if stage in continuation_stages else (4, 8, 12, 16))
        )
        if stage in fresh_stages:
            publications.append(row(
                f"{stage}:initialization-receipt",
                f"{stage}/initialization-receipt.json", 2 * 1024**2,
            ))
        publications.extend((
            row(f"{stage}:history", f"{stage}/history.json", 2 * 1024**2),
            row(f"{stage}:run-receipt", f"{stage}/run-receipt.json", 2 * 1024**2),
        ))
        for epoch in epochs:
            stem = f"evaluation-epoch-{epoch:04d}"
            publications.extend((
                row(f"{stage}:checkpoint-epoch-{epoch:04d}",
                    f"{stage}/epoch-{epoch:04d}.pt", checkpoint_bytes),
                row(f"{stage}:{stem}-query", f"{stage}/{stem}-query.npy", query_bytes),
                row(f"{stage}:{stem}-gallery", f"{stage}/{stem}-gallery.npy",
                    gallery_bytes),
                row(f"{stage}:{stem}-ranked-prefix",
                    f"{stage}/{stem}-ranked-prefix.json", ranked_bytes),
                row(f"{stage}:{stem}", f"{stage}/{stem}.json", 4 * 1024**2),
            ))
    if sum(":checkpoint-epoch-" in row_value["name"] for row_value in publications) != (
        inputs["quality_checkpoints"]
    ):
        raise ValueError("quality checkpoint inventory differs")
    # Non-training publishers use unique campaign-root paths. They are included
    # so the controller's global remaining-capacity check covers every writer,
    # even though only training resolves the reusable local component names.
    status_temporary = row(
        "campaign:controller-status-temporary",
        ".controller-status.json.tmp",
        0,
        256 * 1024,
    )
    status_temporary["persistent_inodes"] = 0
    publications.extend((
        row(
            "campaign:controller-status", "controller-status.json", 256 * 1024, 0
        ),
        status_temporary,
        row("cuda-canary:environment", "preflight/cuda-environment.json", 2 * 1024**2),
        row("cuda-canary:manifest", "preflight/canary-evidence/manifest.json",
            2 * 1024**2),
        row("cuda-canary:receipt", "preflight/cuda_canary_v1.json", 2 * 1024**2),
    ))
    for object_name in (
        "observation", "initialization-receipt", "cache-inventory", "model-inventory",
        "rng-audit", "model-modes", "environment",
    ):
        publications.append(row(
            f"cuda-canary:evidence-{object_name}",
            f"preflight/canary-evidence/{object_name}.json",
            8 * 1024**2,
        ))
        staging_row = row(
            f"cuda-canary:staging-{object_name}",
            f"preflight/canary-evidence.staging/{object_name}.json",
            0,
            8 * 1024**2,
        )
        staging_row["persistent_inodes"] = 0
        publications.append(staging_row)
    staging_manifest = row(
        "cuda-canary:staging-manifest",
        "preflight/canary-evidence.staging/manifest.json",
        0,
        2 * 1024**2,
    )
    staging_manifest["persistent_inodes"] = 0
    publications.append(staging_manifest)
    for stage in _registered_stage_names():
        if stage == "cuda-canary":
            continue
        if stage.endswith("-decision"):
            publications.extend((
                row(f"{stage}:sources", f"{stage}-sources.json", 2 * 1024**2),
                row(f"{stage}:result", f"{stage}-result.json", 2 * 1024**2),
            ))
        elif stage.startswith("runtime-") or "profile" in stage:
            publications.append(
                row(f"{stage}:terminal", f"{stage}/terminal.json", 2 * 1024**2)
            )
    # Budget every directory entry and anonymous/named transient independently.
    # The final rows retain a temporary inode allowance because the shared
    # publisher uses O_TMPFILE. Public CLI temp names are reserved collision
    # sentinels and are not inventory entries because they are never created.
    final_paths = [row_value["path"] for row_value in publications]
    directories = sorted({
        parent.as_posix()
        for path in final_paths
        for parent in Path(path).parents
        if parent.as_posix() not in {"", "."}
    })
    for directory in directories:
        publications.append(row(
            f"campaign:directory:{directory.replace('/', ':')}",
            directory,
            0,
            0,
        ))

    # The budget is itself immutable evidence. Its bound is the least fixed
    # point of its canonical serialized size, so no caller supplies a second
    # self-size oracle.
    self_row = row(
        "campaign:publication-budget",
        config.get("publication_budget_path", "preflight/publication-budget.json"),
        0,
        0,
    )
    self_row["temporary_inodes"] = 1
    publications.append(self_row)
    result = {"schema": "unicom-fepf-publication-budget-v1", "publications": publications}
    for _iteration in range(32):
        size = len(canonical_json_bytes(result))
        if self_row["persistent_bytes"] == size:
            break
        self_row["persistent_bytes"] = size
        self_row["temporary_bytes"] = size
    else:
        raise RuntimeError("publication budget self-size did not converge")
    return result


def validate_exact_publication_budget(
    config: dict[str, object], budget: object
) -> dict[str, object]:
    expected = exact_publication_budget(config)
    if budget != expected:
        raise ValueError("exact publication budget differs")
    publications = expected["publications"]
    if (
        sum(row["persistent_bytes"] + row["temporary_bytes"] for row in publications)
        != config["artifact_budget_bytes"]
        or sum(
            row["persistent_inodes"] + row["temporary_inodes"]
            for row in publications
        )
        != config["artifact_budget_inodes"]
    ):
        raise ValueError("exact publication budget exceeds campaign capacity")
    payload = canonical_json_bytes(expected)
    if (
        config["publication_budget_sha256"] != _sha256(payload)
        or config["publication_budget"] != expected
    ):
        raise ValueError("publication budget authority differs")
    return expected


def validate_external_exact_publication_budget(
    config: dict[str, object], budget: object
) -> dict[str, object]:
    """Re-derive partition-dependent budget authority from registered bytes."""

    config = _strict_shape(config)
    _validate_config_values(config)
    partition_path = Path(config["inputs"]["partition"])
    payload = partition_path.read_bytes()
    if (
        partition_path.is_symlink()
        or _sha256(payload) != config["model"]["partition_sha256"]
    ):
        raise ValueError("publication budget partition authority differs")
    observed = _partition_inventory(partition_path)
    expected_inputs = {
        key: config["artifact_budget_inputs"][key]
        for key in (
            "query_rows", "gallery_rows", "maximum_relevant_count",
            "maximum_path_bytes",
        )
    }
    if observed != expected_inputs:
        raise ValueError("publication budget partition inventory differs")
    return validate_exact_publication_budget(config, budget)


def build_and_write_with_authorities(
    *, repo: Path, checkout_root_template: str, artifact_root: Path, output: Path,
    inference_structure: dict[str, object],
    runtime_inference_signature: dict[str, object],
    partition_inventory: dict[str, int],
    cuda_canary_authority: dict[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Compatibility wrapper returning the now config-embedded budget."""

    if os.path.lexists(output):
        raise FileExistsError(output)

    environment_path = (artifact_root / "preflight/cuda-environment.json").resolve()
    config = build_run_config(
        repo=repo,
        checkout_root_template=checkout_root_template,
        artifact_root=artifact_root,
        inference_structure=inference_structure,
        partition_inventory=partition_inventory,
        cuda_canary_authority=cuda_canary_authority,
        cuda_canary_environment={
            "path": str(environment_path),
            "sha256": cuda_canary_authority["environment_sha256"],
            "bytes": 1,
        },
        publication_budget={},
        runtime_inference_signature=runtime_inference_signature,
    )
    validate_config_build(config, repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_payload = canonical_json_bytes(config)
    published = publish_bytes_noreplace(
        output,
        config_payload,
        validator=lambda payload: (
            None
            if payload == config_payload and json.loads(payload) == config
            else (_ for _ in ()).throw(ValueError("persisted FEPF config differs"))
        ),
    )
    published.close()
    if output.read_bytes() != config_payload:
        raise RuntimeError("persisted FEPF config differs")
    return config, config["publication_budget"]


def _validate_config_handoff(
    config_path: Path, repo: Path, *, external_budget: bool,
    require_checkout_absent: bool,
) -> dict[str, str]:
    raw = config_path.read_bytes()
    config = _strict_shape(json.loads(raw))
    if canonical_json_bytes(config) != raw:
        raise ValueError("FEPF config is not canonical")
    _validate_config_values(config)
    if external_budget:
        legacy = config["legacy_runtime_authority"]
        bindings = [legacy[name] for name in ("run_receipt", "config", "history")]
        bindings.extend(legacy["checkpoints"])
        if any(
            "non-authentic" in Path(binding["path"]).parts
            for binding in bindings
        ):
            raise ValueError("authentic legacy runtime authority differs")
    repo = repo.resolve()
    _require_clean(repo)
    head = str(_git(repo, "rev-parse", "HEAD")).strip()
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"], cwd=repo,
        check=False, capture_output=True, text=True,
    ).stdout.strip()
    if _LOWER_COMMIT(head) is None or symbolic:
        raise ValueError("handoff checkout must be detached")
    parent = str(_git(repo, "rev-parse", f"{head}^")).strip()
    relative = config_path.resolve().relative_to(repo).as_posix()
    changed = str(
        _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", head)
    ).splitlines()
    committed = _git(repo, "show", f"{head}:{relative}", binary=True)
    if parent != config["source_commit"]:
        raise ValueError("config handoff parent differs")
    if changed != [relative]:
        raise ValueError("config handoff path inventory differs")
    if committed != raw:
        raise ValueError("config handoff committed bytes differ")
    if config["source_files"] != _source_inventory(repo, parent):
        raise ValueError("config handoff source inventory differs")
    if external_budget:
        validate_external_exact_publication_budget(
            config, config.get("publication_budget")
        )
    checkout = config["checkout_root_template"].replace("{config_commit}", head)
    if "{" in checkout or "}" in checkout:
        raise ValueError("resolved checkout root differs")
    if require_checkout_absent and os.path.lexists(checkout):
        raise FileExistsError("resolved execution checkout already exists")
    return {"config_commit": head, "checkout_root": checkout}


def validate_config_handoff(config_path: Path, repo: Path) -> dict[str, str]:
    return _validate_config_handoff(
        config_path, repo, external_budget=True, require_checkout_absent=True
    )


def _require_non_authentic_synthesized_authority(config_path: Path) -> None:
    config = _strict_shape(json.loads(config_path.read_bytes()))
    legacy = config["legacy_runtime_authority"]
    bindings = [legacy[name] for name in ("run_receipt", "config", "history")]
    bindings.extend(legacy["checkpoints"])
    if any(
        "non-authentic" not in Path(binding["path"]).parts
        for binding in bindings
    ):
        raise ValueError("non-authentic synthesized authority marker differs")


def validate_non_authentic_synthesized_handoff(
    config_path: Path, repo: Path
) -> dict[str, str]:
    """Validate the CPU-only synthetic handoff without claiming target authority."""

    _require_non_authentic_synthesized_authority(config_path)
    return _validate_config_handoff(
        config_path, repo, external_budget=False, require_checkout_absent=True
    )


def validate_non_authentic_synthesized_membership(
    config_path: Path, repo: Path
) -> dict[str, str]:
    """Validate the CPU-only synthetic execution checkout membership."""

    _require_non_authentic_synthesized_authority(config_path)
    return _validate_config_handoff(
        config_path, repo, external_budget=False, require_checkout_absent=False
    )


def validate_config_membership(config_path: Path, repo: Path) -> dict[str, str]:
    """Authenticate committed config bytes and checkout context only."""

    return _validate_config_handoff(
        config_path, repo, external_budget=True, require_checkout_absent=False
    )


def validate_config_document(config: object) -> dict[str, object]:
    value = _strict_shape(config)
    _validate_config_values(value)
    return value


def validate_config_membership_document(
    config: object, repo: Path
) -> dict[str, object]:
    value = validate_config_document(config)
    repo = repo.resolve()
    _require_clean(repo)
    head = str(_git(repo, "rev-parse", "HEAD")).strip()
    if head != value["source_commit"] or value["source_files"] != _source_inventory(repo, head):
        raise ValueError("config source membership differs")
    return value


def validate_transfer_handoff(config: object, execution_checkout: Path) -> None:
    validate_config_document(config)
    if os.path.lexists(execution_checkout):
        raise FileExistsError(execution_checkout)


def validate_first_launch_absence(config: object) -> None:
    """Require transfer destinations to be absent only for the first launch."""

    value = _strict_shape(config)
    artifact = Path(value["artifact_root"])
    if os.path.lexists(artifact):
        raise FileExistsError("campaign destination already exists")


def _available(stat_result: Any) -> tuple[int, int]:
    return stat_result.f_bavail * stat_result.f_frsize, stat_result.f_favail


def _plain_directory(path: Path, name: str) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{name} is not a plain directory")
    return info


def _require_capacity(
    path: Path, required_bytes: int, required_inodes: int,
    statvfs: Callable[[Path], Any],
) -> None:
    available_bytes, available_inodes = _available(statvfs(path))
    if available_bytes < required_bytes or available_inodes < required_inodes:
        raise OSError("FEPF artifact capacity is insufficient")


def prepare_artifact_root(
    root: Path, *, required_bytes: int, required_inodes: int,
    statvfs: Callable[[Path], Any] = os.statvfs,
) -> None:
    root = _require_absolute_plain(root, "artifact root")
    if type(required_bytes) is not int or type(required_inodes) is not int:
        raise ValueError("artifact budget differs")
    if os.path.lexists(root):
        raise FileExistsError(root)
    parent = root.parent
    try:
        parent_info = _plain_directory(parent, "artifact parent")
    except FileNotFoundError as error:
        raise ValueError("artifact parent is absent") from error
    _require_capacity(parent, required_bytes, required_inodes, statvfs)
    root.mkdir(mode=0o700)
    root_info = _plain_directory(root, "artifact root")
    owned = (root_info.st_dev, root_info.st_ino)
    try:
        if root_info.st_dev != parent_info.st_dev:
            raise OSError("artifact root device differs")
        _require_capacity(root, required_bytes, required_inodes, statvfs)
    except Exception:
        current = root.lstat()
        if (current.st_dev, current.st_ino) == owned and not any(root.iterdir()):
            root.rmdir()
        raise


def require_remaining_capacity(
    root: Path, *, total_budget_bytes: int, total_budget_inodes: int,
    consumed_bytes: int, consumed_inodes: int,
    statvfs: Callable[[Path], Any] = os.statvfs,
) -> None:
    _plain_directory(root, "artifact root")
    values = (total_budget_bytes, total_budget_inodes, consumed_bytes, consumed_inodes)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("artifact budget differs")
    _require_capacity(
        root, max(total_budget_bytes - consumed_bytes, 0),
        max(total_budget_inodes - consumed_inodes, 0), statvfs,
    )


def validate_campaign_resume(
    config: object, run_root: Path,
    *, terminal_validator: Callable[[Path], None],
) -> tuple[Path, ...]:
    value = _strict_shape(config)
    _validate_config_values(value)
    root = run_root.resolve()
    _plain_directory(root, "campaign root")
    inventory = set(registered_stage_inventory(value))
    allowed_root_files = {"controller-status.json", ".controller-status.json.tmp"}
    decision_names = {
        "exploratory-epoch4-decision",
        "exploratory-decision",
        "confirmation-decision",
    }
    allowed_publications = {
        f"{name}-{suffix}.json"
        for name in decision_names
        for suffix in ("sources", "result")
    }
    root_file_names = {
        child.name for child in root.iterdir()
        if child.is_file() and not child.is_symlink()
    }
    for name in tuple(root_file_names):
        if name.endswith("-sources.json"):
            continue
        elif name.endswith("-result.json"):
            sources = f"{name[:-len('-result.json')]}-sources.json"
            if sources not in root_file_names:
                raise ValueError("campaign resume publication is incomplete")
    for child in root.iterdir():
        if child.is_symlink():
            raise ValueError("campaign resume symlink differs")
        if child.is_file():
            if (
                child.name not in allowed_root_files
                and child.name not in allowed_publications
            ):
                raise ValueError("campaign resume registered path differs")
            continue
        if child.name == "preflight":
            allowed_preflight = {
                "cuda_canary_v1.json", "cuda-environment.json", "canary-evidence",
                "canary-evidence.staging", "publication-budget.json",
            }
            unexpected = [path for path in child.iterdir() if path.name not in allowed_preflight]
            if unexpected:
                raise ValueError("campaign resume registered path differs")
            evidence = child / "canary-evidence"
            staging = child / "canary-evidence.staging"
            if os.path.lexists(evidence) and os.path.lexists(staging):
                raise ValueError("campaign resume registered path differs")
            for namespace, complete in ((staging, False), (evidence, True)):
                if not os.path.lexists(namespace):
                    continue
                if namespace.is_symlink() or not namespace.is_dir():
                    raise ValueError("campaign resume registered path differs")
                observed_evidence = set()
                for path in namespace.iterdir():
                    if path.is_symlink() or not path.is_file():
                        raise ValueError("campaign resume registered path differs")
                    payload = path.read_bytes()
                    try:
                        document = json.loads(payload)
                    except Exception as error:
                        raise ValueError("campaign resume registered path differs") from error
                    if type(document) is not dict or canonical_json_bytes(document) != payload:
                        raise ValueError("campaign resume registered path differs")
                    observed_evidence.add(path.name)
                valid = (
                    observed_evidence == set(CANARY_EVIDENCE_ORDER)
                    if complete
                    else any(
                        observed_evidence == set(CANARY_EVIDENCE_ORDER[:length])
                        for length in range(len(CANARY_EVIDENCE_ORDER) + 1)
                    )
                )
                if not valid:
                    raise ValueError("campaign resume registered path differs")
            continue
        if child.name not in inventory:
            raise ValueError("campaign resume registered path differs")
        terminal_name = (
            "run-receipt.json"
            if (
                ("control" in child.name or "candidate" in child.name or "random" in child.name)
                and "profile" not in child.name
                and "decision" not in child.name
            )
            else "terminal.json"
        )
        terminal = child / terminal_name
        if not terminal.is_file() or terminal.is_symlink():
            raise ValueError("campaign resume terminal differs")
    terminals: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("campaign resume symlink differs")
        if path.is_file():
            if path.name.endswith(".tmp") or path.name.endswith(".partial"):
                raise ValueError("campaign partial artifact differs")
            if path.name in {"run-receipt.json", "profile-receipt.json", "result.json",
                             "cuda_canary_v1.json"} or path.name.endswith("-result.json"):
                terminal_validator(path)
                terminals.append(path)
    return tuple(terminals)


def registered_stage_inventory(config: object) -> tuple[str, ...]:
    _validate_config_values(_strict_shape(config))
    return _registered_stage_names()


def _registered_stage_names() -> tuple[str, ...]:
    return (
        "cuda-canary", *(f"runtime-{index:02d}" for index in range(8)),
        "exploratory-control-stage4",
        "exploratory-candidate-stage4", "exploratory-epoch4-decision",
        "exploratory-control-stage16", "exploratory-candidate-stage16",
        "exploratory-profile-control-0", "exploratory-profile-candidate-0",
        "exploratory-profile-candidate-1", "exploratory-profile-control-1",
        "exploratory-decision", "exploratory-random-stage16",
        *(f"confirmation-{index}-{suffix}" for index in range(5) for suffix in (
            "control", "candidate", "profile-control-0", "profile-candidate-0",
            "profile-candidate-1", "profile-control-1",
        )), "confirmation-decision",
    )


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkout-root-template", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-handoff", action="store_true")
    parser.add_argument("--non-authentic-synthesized-authorities", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        if args.validate_handoff:
            if args.non_authentic_synthesized_authorities:
                raise ValueError("non-authentic builder cannot validate handoff")
            validate_config_handoff(args.output, args.repo)
        else:
            legacy_runtime_authority = _registered_legacy_runtime_authority()
            if args.non_authentic_synthesized_authorities:
                bindings = [
                    legacy_runtime_authority[name]
                    for name in ("run_receipt", "config", "history")
                ]
                bindings.extend(legacy_runtime_authority["checkpoints"])
                if any(
                    "non-authentic" not in Path(binding["path"]).parts
                    for binding in bindings
                ):
                    raise ValueError(
                        "non-authentic synthesized authority marker differs"
                    )
            runtime_inference_signature = _checkpoint_runtime_inference_signature(
                Path(_inputs()["runtime_checkpoint"])
            )
            build_and_write(
                repo=args.repo, checkout_root_template=args.checkout_root_template,
                artifact_root=args.artifact_root, output=args.output,
                cuda_canary_authority={},
                cuda_canary_environment={
                    "path": str(
                        (
                            args.artifact_root
                            / "preflight/cuda-environment.json"
                        ).resolve()
                    )
                },
                publication_budget={},
                runtime_inference_signature=runtime_inference_signature,
                inference_structure=None,
                partition_inventory=None,
                legacy_runtime_authority=legacy_runtime_authority,
            )
    except Exception as error:
        print(f"FEPF config failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
