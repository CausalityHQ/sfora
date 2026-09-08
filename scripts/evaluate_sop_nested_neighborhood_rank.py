#!/usr/bin/env python3
"""Evaluate SOP nested-rank checkpoints from authenticated local artifacts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.nested_neighborhood_rank import NestedRankConfig, NestedRankHead
from sfora.nested_rank_evaluation import (
    _bounded_top_indices,
    class_bootstrap_lower_bound,
    classify_promotion,
    evaluate_paired_rankings,
    rank_query_gallery,
    rank_self_retrieval,
    rank_self_retrieval_int8,
    recompute_self_retrieval,
)
from sfora.nested_rank_protocol import class_disjoint_fold

__all__ = (
    "_bounded_top_indices",
    "class_bootstrap_lower_bound",
    "classify_promotion",
    "evaluate_paired_rankings",
    "evaluate_three_way_embeddings",
    "encode_validation_rows",
    "load_training_artifact",
    "parse_args",
    "rank_self_retrieval",
    "rank_self_retrieval_int8",
    "rank_query_gallery",
    "recompute_self_retrieval",
    "restore_candidate_state",
)


_RESULT_KEYS = {
    "schema",
    "claim_eligible",
    "status",
    "arm",
    "split_seed",
    "temperature",
    "optimization_rows",
    "optimization_classes",
    "epochs",
    "history",
    "authority",
    "model_artifact",
    "run_receipt",
}
_ARMS = {
    "proxy-anchor",
    "neighborhood",
    "combined",
    "proxy-anchor-768",
    "s2sd-768-to-128",
}


class _EvaluationRecord(Protocol):
    image_id: int
    label: int
    image_path: Path


class _UniqueStore(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may be supplied only once")
        setattr(namespace, self.dest, values)


class _UniqueTrue(argparse.Action):
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        *,
        default: object = False,
        required: bool = False,
        **kwargs: object,
    ) -> None:
        if kwargs:
            raise TypeError("unsupported boolean action configuration")
        super().__init__(option_strings, dest, nargs=0, default=default, required=required)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, False):
            parser.error(f"{option_string} may be supplied only once")
        setattr(namespace, self.dest, True)


class _ValidationDataset(torch.utils.data.Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(
        self,
        records: tuple[_EvaluationRecord, ...],
        rows: tuple[int, ...],
        transform: Callable[[object], object],
    ) -> None:
        self.records = records
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        from PIL import Image

        record = self.records[self.rows[index]]
        with Image.open(record.image_path) as image:
            value = self.transform(image.convert("RGB"))
        if type(value) is not torch.Tensor:
            raise ValueError("NNRL evaluation transform differs")
        return value, record.label, record.image_id


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _hex_argument(value: str, length: int) -> str:
    if len(value) != length or set(value) - set("0123456789abcdef"):
        raise argparse.ArgumentTypeError(f"value must be a lowercase {length}-digit digest")
    return value


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit local-only NNRL evaluation boundary."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.set_defaults(execute_evaluation=False)
    parser.add_argument(
        "--source-commit",
        required=True,
        type=lambda value: _hex_argument(value, 40),
        action=_UniqueStore,
    )
    for name in (
        "training-result",
        "model",
        "train-snapshot",
        "dataset-root",
        "unicom-checkout",
        "unicom-checkpoint",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=_absolute_path, action=_UniqueStore)
    for name in (
        "training-result-sha256",
        "train-snapshot-sha256",
        "unicom-checkpoint-sha256",
    ):
        parser.add_argument(
            f"--{name}",
            required=True,
            type=lambda value: _hex_argument(value, 64),
            action=_UniqueStore,
        )
    parser.add_argument(
        "--execute-evaluation", required=True, action=_UniqueTrue, dest="execute_evaluation"
    )
    return parser.parse_args(arguments)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_descriptor(value: object, *, expected_path: str) -> bool:
    return (
        type(value) is dict
        and set(value) == {"path", "sha256", "bytes"}
        and value["path"] == expected_path
        and _is_sha256(value["sha256"])
        and type(value["bytes"]) is int
        and value["bytes"] > 0
    )


def _validate_resolved_recipe(result: dict[str, object]) -> None:
    authority = result["authority"]
    if type(authority) is not dict:
        raise ValueError("NNRL resolved recipe differs")
    recipe = authority.get("resolved_recipe")
    if type(recipe) is not dict or set(recipe) != {
        "schema",
        "objective",
        "schedule",
        "optimizer",
        "precision",
        "data",
        "runtime",
    }:
        raise ValueError("NNRL resolved recipe differs")
    objective = recipe["objective"]
    schedule = recipe["schedule"]
    optimizer = recipe["optimizer"]
    precision = recipe["precision"]
    data = recipe["data"]
    runtime = recipe["runtime"]
    arm = result["arm"]
    history = result["history"]
    if type(history) is not list or not history or type(history[0]) is not dict:
        raise ValueError("NNRL resolved recipe differs")
    steps_per_epoch = history[0].get("steps")
    if type(steps_per_epoch) is not int or any(
        type(row) is not dict or row.get("steps") != steps_per_epoch for row in history
    ):
        raise ValueError("NNRL resolved recipe differs")
    expected_dimensions = [768] if arm == "proxy-anchor-768" else [32, 128]
    if (
        recipe["schema"] != "sfora-nnrl-sop-training-recipe-v1"
        or type(objective) is not dict
        or objective
        != {
            "arm": arm,
            "temperature": result["temperature"],
            "output_dimensions": expected_dimensions,
        }
        or type(schedule) is not dict
        or schedule
        != {
            "split_seed": result["split_seed"],
            "epochs": 10,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": 128,
        }
        or type(optimizer) is not dict
        or optimizer
        != {
            "name": "AdamW",
            "encoder_learning_rate": 1e-5,
            "head_learning_rate": 1e-3,
            "proxy_learning_rate": 1e-3,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "decay": 1e-4,
        }
        or type(precision) is not dict
        or precision
        != {
            "autocast": "float16",
            "gradient_scaler_initial_scale": 1024.0,
            "gradient_scaler_growth_interval": 2**31 - 1,
            "fail_on_nonfinite": True,
        }
        or type(data) is not dict
        or data
        != {
            "workers": 8,
            "pin_memory": True,
            "frozen_batch_norm": True,
            "transform": (
                "resize-256-bicubic/random-crop-224/random-horizontal-flip/unicom-normalize"
            ),
        }
        or type(runtime) is not dict
        or set(runtime)
        != {
            "python_version",
            "torch_version",
            "cuda_version",
            "cudnn_version",
            "gpu_name",
            "gpu_capability",
        }
        or any(
            type(runtime[key]) is not str or not runtime[key]
            for key in ("python_version", "torch_version", "cuda_version", "gpu_name")
        )
        or type(runtime["cudnn_version"]) is not int
        or runtime["cudnn_version"] <= 0
        or type(runtime["gpu_capability"]) is not list
        or len(runtime["gpu_capability"]) != 2
        or any(type(value) is not int or value < 0 for value in runtime["gpu_capability"])
    ):
        raise ValueError("NNRL resolved recipe differs")


def load_training_artifact(
    result_path: Path,
    result_sha256: str,
    model_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Authenticate and load one completed local NNRL training artifact."""

    if (
        not isinstance(result_path, Path)
        or not isinstance(model_path, Path)
        or not _is_sha256(result_sha256)
        or result_path.is_symlink()
        or model_path.is_symlink()
        or not result_path.is_file()
        or not model_path.is_file()
    ):
        raise ValueError("NNRL evaluation artifact path differs")
    result_bytes = result_path.read_bytes()
    if hashlib.sha256(result_bytes).hexdigest() != result_sha256:
        raise ValueError("NNRL result digest differs")
    try:
        result = json.loads(result_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("NNRL result JSON differs") from error
    if (
        type(result) is not dict
        or set(result) != _RESULT_KEYS
        or _canonical_json_bytes(result) != result_bytes
        or result["schema"] != "sfora-nnrl-sop-training-result-v1"
        or result["claim_eligible"] is not False
        or result["status"] != "COMPLETE"
        or result["arm"] not in _ARMS
        or type(result["split_seed"]) is not int
        or type(result["temperature"]) is not float
        or not math.isfinite(result["temperature"])
        or result["temperature"] <= 0.0
        or type(result["optimization_rows"]) is not int
        or result["optimization_rows"] <= 0
        or type(result["optimization_classes"]) is not int
        or result["optimization_classes"] <= 1
        or type(result["epochs"]) is not int
        or result["epochs"] != 10
        or type(result["history"]) is not list
        or len(result["history"]) != result["epochs"]
        or type(result["authority"]) is not dict
        or not result["authority"]
        or not _artifact_descriptor(result["model_artifact"], expected_path=model_path.name)
        or not _artifact_descriptor(result["run_receipt"], expected_path="run-receipt.json")
    ):
        raise ValueError("NNRL result authority differs")
    _validate_resolved_recipe(result)
    for epoch, row in enumerate(result["history"], start=1):
        if (
            type(row) is not dict
            or set(row) != {"epoch", "steps", "attempted_steps", "skipped_updates", "mean_loss"}
            or row["epoch"] != epoch
            or type(row["steps"]) is not int
            or row["steps"] <= 0
            or row["attempted_steps"] != row["steps"]
            or row["skipped_updates"] != 0
            or type(row["mean_loss"]) is not float
            or not math.isfinite(row["mean_loss"])
        ):
            raise ValueError("NNRL result history differs")
    model_authority = result["model_artifact"]
    if (
        model_path.stat().st_size != model_authority["bytes"]
        or _sha256_file(model_path) != model_authority["sha256"]
    ):
        raise ValueError("NNRL model authority differs")
    model = torch.load(model_path, map_location="cpu", weights_only=True)
    if type(model) is not dict or set(model) != {"encoder", "head", "raw_proxies"}:
        raise ValueError("NNRL model schema differs")
    return result, model


def _validate_state_dict(module: nn.Module, value: object) -> dict[str, torch.Tensor]:
    expected = module.state_dict()
    if (
        type(value) is not dict
        or set(value) != set(expected)
        or any(
            type(name) is not str
            or type(tensor) is not torch.Tensor
            or tensor.shape != expected[name].shape
            or tensor.dtype != expected[name].dtype
            or not torch.isfinite(tensor).all()
            for name, tensor in value.items()
        )
    ):
        raise ValueError("NNRL model state differs")
    return value


def restore_candidate_state(
    encoder: nn.Module,
    head: nn.Module,
    artifact: object,
) -> torch.Tensor:
    """Strictly restore an authenticated candidate encoder and projection head."""

    if (
        not isinstance(encoder, nn.Module)
        or not isinstance(head, nn.Module)
        or type(artifact) is not dict
        or set(artifact) != {"encoder", "head", "raw_proxies"}
    ):
        raise ValueError("NNRL model state differs")
    encoder_state = _validate_state_dict(encoder, artifact["encoder"])
    head_state = _validate_state_dict(head, artifact["head"])
    raw_proxies = artifact["raw_proxies"]
    if (
        type(raw_proxies) is not torch.Tensor
        or raw_proxies.ndim != 2
        or raw_proxies.shape[0] <= 1
        or raw_proxies.shape[1] <= 1
        or not torch.isfinite(raw_proxies).all()
    ):
        raise ValueError("NNRL model state differs")
    encoder.load_state_dict(encoder_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    return raw_proxies


def evaluate_three_way_embeddings(
    candidate: object,
    source: object,
    teacher: object,
    labels: object,
    sample_ids: object,
    *,
    block_rows: int = 256,
) -> dict[str, dict[str, object]]:
    """Evaluate candidate float/int8, source, and teacher on identical rows."""

    if (
        type(candidate) is not np.ndarray
        or type(source) is not np.ndarray
        or type(teacher) is not np.ndarray
        or type(labels) is not np.ndarray
        or type(sample_ids) is not np.ndarray
        or candidate.shape[0] != source.shape[0]
        or candidate.shape[0] != teacher.shape[0]
        or labels.shape != (candidate.shape[0],)
        or sample_ids.shape != labels.shape
    ):
        raise ValueError("NNRL three-way evaluation inventory differs")

    def evaluate(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "metrics": recompute_self_retrieval(rows, labels, sample_ids),
            "queries": rows,
        }

    return {
        "candidate_float": evaluate(
            rank_self_retrieval(candidate, labels, sample_ids, block_rows=block_rows)
        ),
        "candidate_int8": evaluate(
            rank_self_retrieval_int8(candidate, labels, sample_ids, block_rows=block_rows)
        ),
        "source": evaluate(rank_self_retrieval(source, labels, sample_ids, block_rows=block_rows)),
        "teacher": evaluate(
            rank_self_retrieval(teacher, labels, sample_ids, block_rows=block_rows)
        ),
    }


def encode_validation_rows(
    records: tuple[_EvaluationRecord, ...],
    rows: tuple[int, ...],
    transform: Callable[[object], object],
    encoder: nn.Module,
    head: nn.Module | None,
    *,
    output_width: int,
    device: torch.device,
    batch_size: int = 128,
    workers: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode exactly the registered validation rows without opening other images."""

    if (
        type(records) is not tuple
        or not records
        or type(rows) is not tuple
        or not rows
        or len(set(rows)) != len(rows)
        or any(type(row) is not int or not 0 <= row < len(records) for row in rows)
        or not callable(transform)
        or not isinstance(encoder, nn.Module)
        or (head is not None and not isinstance(head, nn.Module))
        or type(output_width) is not int
        or output_width <= 1
        or type(device) is not torch.device
        or type(batch_size) is not int
        or batch_size <= 0
        or type(workers) is not int
        or workers < 0
    ):
        raise ValueError("NNRL validation inference authority differs")
    dataset = _ValidationDataset(records, rows, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    encoder.eval()
    if head is not None:
        head.eval()
    embedding_batches: list[torch.Tensor] = []
    labels: list[int] = []
    sample_ids: list[int] = []
    with torch.inference_mode():
        for images, batch_labels, batch_ids in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                encoded = encoder(images.to(device))
            dense = F.normalize(encoded.float(), dim=1)
            if head is None:
                selected = dense
            else:
                projected = head(dense)
                if type(projected) is not dict or output_width not in projected:
                    raise ValueError("NNRL evaluation head output differs")
                selected = projected[output_width]
            if (
                type(selected) is not torch.Tensor
                or selected.dtype != torch.float32
                or selected.ndim != 2
                or selected.shape[1] != output_width
                or not torch.isfinite(selected).all()
            ):
                raise ValueError("NNRL evaluation embedding differs")
            embedding_batches.append(F.normalize(selected, dim=1).cpu())
            labels.extend(int(value) for value in batch_labels)
            sample_ids.extend(int(value) for value in batch_ids)
    embeddings = torch.cat(embedding_batches).numpy().astype(np.float32, copy=False)
    if embeddings.shape != (len(rows), output_width):
        raise ValueError("NNRL evaluation embedding inventory differs")
    return (
        embeddings,
        np.asarray(labels, dtype=np.int64),
        np.asarray(sample_ids, dtype=np.int64),
    )


def _load_trainer(repository: Path) -> Any:
    path = repository / "scripts" / "train_sop_nested_neighborhood_rank.py"
    specification = importlib.util.spec_from_file_location("sop_nnrl_trainer", path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        if sys.modules.get(specification.name) is module:
            del sys.modules[specification.name]
        raise
    return module


def _load_student_and_transform(
    checkout: Path, checkpoint: Path
) -> tuple[nn.Module, Callable[[object], object]]:
    package_root = (checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    module_file = getattr(unicom, "__file__", None)
    if (
        type(module_file) is not str
        or Path(module_file).resolve().parent != package_root / "unicom"
        or checkpoint.name != "FP16-ViT-B-16.pt"
    ):
        raise ValueError("NNRL evaluation UNICOM authority differs")
    model, transform = unicom.load("ViT-B/16", download_root=str(checkpoint.parent))
    if not isinstance(model, nn.Module) or not callable(transform):
        raise ValueError("NNRL evaluation UNICOM model differs")
    return model, cast(Callable[[object], object], transform)


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _head_config(result: dict[str, object]) -> NestedRankConfig:
    arm = result["arm"]
    class_count = result["optimization_classes"]
    if type(class_count) is not int:
        raise ValueError("NNRL evaluation head authority differs")
    if arm == "proxy-anchor-768":
        return NestedRankConfig(
            input_dim=768,
            hidden_dim=1024,
            output_dim=768,
            widths=(768,),
            class_count=class_count,
        )
    return NestedRankConfig(input_dim=768, hidden_dim=1024, class_count=class_count)


def run_evaluation(arguments: argparse.Namespace) -> dict[str, object]:
    """Run one authenticated training-only SOP validation evaluation."""

    repository = Path(__file__).resolve().parents[1]
    if (
        _git_output(repository, "rev-parse", "HEAD").strip() != arguments.source_commit
        or _git_output(repository, "status", "--porcelain")
        or arguments.output.exists()
        or arguments.output.is_symlink()
    ):
        if arguments.output.exists() or arguments.output.is_symlink():
            raise FileExistsError(arguments.output)
        raise ValueError("NNRL evaluation source authority differs")
    result, artifact = load_training_artifact(
        arguments.training_result,
        arguments.training_result_sha256,
        arguments.model,
    )
    authority = result["authority"]
    if (
        type(authority) is not dict
        or authority.get("train_snapshot_sha256") != arguments.train_snapshot_sha256
        or authority.get("student_checkpoint_sha256") != arguments.unicom_checkpoint_sha256
        or authority.get("source_commit") is None
        or _sha256_file(arguments.unicom_checkpoint) != arguments.unicom_checkpoint_sha256
        or _git_output(arguments.unicom_checkout, "rev-parse", "HEAD").strip()
        != authority.get("unicom_revision")
        or _git_output(arguments.unicom_checkout, "status", "--porcelain")
    ):
        raise ValueError("NNRL evaluation cross-object authority differs")
    trainer = _load_trainer(repository)
    snapshot = trainer.load_train_snapshot(
        arguments.train_snapshot, arguments.train_snapshot_sha256
    )
    records = trainer._load_sop_training_records(arguments.dataset_root)
    image_ids = np.asarray([record.image_id for record in records], dtype=np.int64)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    relative_paths = tuple(record.relative_path for record in records)
    trainer.bind_training_records(snapshot, image_ids, labels, relative_paths)
    split_seed = result["split_seed"]
    if type(split_seed) is not int:
        raise ValueError("NNRL evaluation split authority differs")
    fold = class_disjoint_fold(tuple(int(value) for value in image_ids), labels, seed=split_seed)
    validation_rows = fold.validation
    validation_teacher = snapshot.embeddings[np.asarray(validation_rows, dtype=np.int64)].copy()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("NNRL evaluation requires CUDA")
    encoder, transform = _load_student_and_transform(
        arguments.unicom_checkout, arguments.unicom_checkpoint
    )
    encoder = encoder.to(device)
    source, validation_labels, validation_ids = encode_validation_rows(
        records,
        validation_rows,
        transform,
        encoder,
        None,
        output_width=768,
        device=device,
    )
    head = NestedRankHead(_head_config(result)).to(device)
    proxies = restore_candidate_state(encoder, head, artifact)
    if tuple(proxies.shape) != (result["optimization_classes"], head.config.output_dim):
        raise ValueError("NNRL evaluation proxy authority differs")
    candidate, candidate_labels, candidate_ids = encode_validation_rows(
        records,
        validation_rows,
        transform,
        encoder,
        head,
        output_width=head.config.output_dim,
        device=device,
    )
    if not np.array_equal(validation_labels, candidate_labels) or not np.array_equal(
        validation_ids, candidate_ids
    ):
        raise ValueError("NNRL evaluation row binding differs")
    gc.collect()
    torch.cuda.synchronize()
    started = time.perf_counter()
    evaluation = evaluate_three_way_embeddings(
        candidate,
        source,
        validation_teacher,
        validation_labels,
        validation_ids,
    )
    source_rows = cast(list[dict[str, object]], evaluation["source"]["queries"])
    compact_rows = cast(list[dict[str, object]], evaluation["candidate_int8"]["queries"])
    comparison = evaluate_paired_rankings(
        compact_rows,
        source_rows,
        validation_labels,
        validation_ids,
        seed=split_seed,
    )
    output = {
        "schema": "sfora-nnrl-sop-evaluation-v1",
        "claim_eligible": False,
        "source_commit": arguments.source_commit,
        "training_result": {
            "path": str(arguments.training_result.resolve()),
            "sha256": arguments.training_result_sha256,
        },
        "model_artifact": result["model_artifact"],
        "train_snapshot_sha256": arguments.train_snapshot_sha256,
        "split_seed": result["split_seed"],
        "arm": result["arm"],
        "validation_rows": len(validation_rows),
        "validation_classes": len(set(int(value) for value in validation_labels)),
        "representations": evaluation,
        "candidate_int8_vs_source": comparison,
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    payload = _canonical_json_bytes(output)
    published = publish_bytes_noreplace(
        arguments.output,
        payload,
        validator=lambda observed: (
            None
            if observed == payload
            else (_ for _ in ()).throw(ValueError("NNRL evaluation result bytes differ"))
        ),
    )
    published.close()
    return output


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        parsed = parse_args(arguments)
        result = run_evaluation(parsed)
    except Exception as error:
        print(f"SOP NNRL evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(parsed.output), "arm": result["arm"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
