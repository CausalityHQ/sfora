#!/usr/bin/env python3
"""Export frozen official UNICOM embeddings for the In-Shop audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sfora.unicom_audit_io import load_embedding_bundle
from sfora.unicom_inshop import EXPECTED_COUNTS, InshopRecord, parse_inshop_partition

UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
UNICOM_B16_SHA256 = "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef"
UNICOM_L14_336_SHA256 = (
    "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
)
REGISTERED_TRAINER_SHA256 = "b2cfdaed33d46ec445141bb40b1a3f28aed0d3ca859101843ddf825866640bb1"


@dataclass(frozen=True)
class ModelSpec:
    model_identifier: str
    checkpoint_filename: str
    checkpoint_sha256: str


_MODEL_SPECS = {
    "ViT-B/16": ModelSpec(
        model_identifier="UNICOM-ViT-B/16",
        checkpoint_filename="FP16-ViT-B-16.pt",
        checkpoint_sha256=UNICOM_B16_SHA256,
    ),
    "ViT-L/14@336px": ModelSpec(
        model_identifier="UNICOM-ViT-L/14@336px",
        checkpoint_filename="FP16-ViT-L-14-336px.pt",
        checkpoint_sha256=UNICOM_L14_336_SHA256,
    ),
}

_SELECTED_STATE_FIELDS = ("model", "selection", "training_protocol")
_ENDPOINT_STATE_FIELDS = ("model", "endpoint", "training_protocol")
_SELECTION_FIELDS = (
    "name",
    "epochs",
    "checkpoints",
    "alpha",
    "metrics",
    "query_evidence",
)
_SELECTION_METRIC_FIELDS = (
    "recall_at_1",
    "recall_at_10",
    "recall_at_20",
    "recall_at_30",
    "map_at_r",
)


def _model_spec(model_name: str) -> ModelSpec:
    try:
        return _MODEL_SPECS[model_name]
    except KeyError as error:
        raise ValueError(f"unsupported UNICOM model: {model_name}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _validate_metadata_base(metadata: Mapping[str, object]) -> None:
    keys = (
        "model_identifier",
        "model_revision",
        "checkpoint_sha256",
        "image_list_sha256",
        "transform",
    )
    if type(metadata) is not dict or tuple(metadata) != keys:
        raise ValueError("export metadata key order differs")
    if metadata["model_identifier"] not in {
        spec.model_identifier for spec in _MODEL_SPECS.values()
    }:
        raise ValueError("export model identifier differs")
    for key, length in (
        ("model_revision", 40),
        ("checkpoint_sha256", 64),
        ("image_list_sha256", 64),
    ):
        value = metadata[key]
        if (
            type(value) is not str
            or len(value) != length
            or not set(value) <= set("0123456789abcdef")
        ):
            raise ValueError(f"export {key} differs")
    if type(metadata["transform"]) is not str or not metadata["transform"]:
        raise TypeError("export transform must be a nonempty string")


def export_embeddings(
    records: Sequence[InshopRecord],
    encode_batch: Callable[[tuple[Path, ...]], np.ndarray],
    metadata_base: Mapping[str, object],
    output: Path,
    *,
    batch_size: int = 64,
    expected_counts: tuple[int, int, int] = EXPECTED_COUNTS,
) -> None:
    """Encode records in official order and exclusively publish one bundle."""

    if type(records) is not tuple or not records:
        raise TypeError("records must be a nonempty tuple")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive builtin integer")
    _validate_metadata_base(metadata_base)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    parent_info = output.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or output.parent.is_symlink():
        raise ValueError("output parent must be a real directory")

    encoded_rows: list[np.ndarray] = []
    dimension: int | None = None
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        paths = tuple(record.image_path for record in batch)
        values = encode_batch(paths)
        if (
            type(values) is not np.ndarray
            or values.dtype != np.float32
            or values.ndim != 2
            or values.shape[0] != len(batch)
            or values.shape[1] == 0
            or not values.flags.c_contiguous
            or not np.isfinite(values).all()
        ):
            raise ValueError("encoded batch must be finite C-contiguous FP32 with matching rows")
        if dimension is None:
            dimension = values.shape[1]
        elif values.shape[1] != dimension:
            raise ValueError("encoded batch dimension differs")
        if np.any(np.linalg.norm(values.astype(np.float64), axis=1) == 0.0):
            raise ValueError("encoded batch contains a zero-norm row")
        encoded_rows.append(values.copy())
    assert dimension is not None
    all_embeddings = np.ascontiguousarray(np.concatenate(encoded_rows, axis=0))

    split_embeddings: dict[str, np.ndarray] = {}
    split_labels: dict[str, np.ndarray] = {}
    for split, expected_count in zip(
        ("train", "query", "gallery"), expected_counts, strict=True
    ):
        indices = [index for index, record in enumerate(records) if record.split == split]
        if len(indices) != expected_count:
            raise ValueError("record split count differs")
        split_embeddings[split] = np.ascontiguousarray(all_embeddings[indices])
        split_labels[split] = np.asarray([records[index].label for index in indices])
    if set(split_labels["query"].tolist()) != set(split_labels["gallery"].tolist()):
        raise ValueError("record query/gallery label membership differs")

    arrays = {
        "train_embeddings": split_embeddings["train"],
        "train_labels": split_labels["train"],
        "query_embeddings": split_embeddings["query"],
        "query_labels": split_labels["query"],
        "gallery_embeddings": split_embeddings["gallery"],
        "gallery_labels": split_labels["gallery"],
    }
    metadata = {
        "schema_version": 1,
        **metadata_base,
        "embedding_dimension": dimension,
        "split_counts": dict(zip(("train", "query", "gallery"), expected_counts, strict=True)),
        "array_sha256": {name: _sha256_array(values) for name, values in arrays.items()},
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    owned: tuple[int, int] | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            np.savez(
                handle,
                metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        load_embedding_bundle(
            temporary,
            expected_counts=expected_counts,
            expected_dimension=dimension,
        )
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary.unlink()
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        load_embedding_bundle(
            output,
            expected_counts=expected_counts,
            expected_dimension=dimension,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            info = temporary.lstat()
        except FileNotFoundError:
            pass
        else:
            if owned is not None and (info.st_dev, info.st_ino) == owned:
                temporary.unlink()


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--finetuned-state", type=Path)
    parser.add_argument("--finetuned-kind", choices=("selected", "endpoint"))
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-name", choices=tuple(_MODEL_SPECS), default="ViT-B/16")
    return parser.parse_args(arguments)


def _git_revision(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_selection(value: object) -> None:
    if type(value) is not dict or tuple(value) != _SELECTION_FIELDS:
        raise ValueError("fine-tuned model state differs")
    epochs = value["epochs"]
    checkpoints = value["checkpoints"]
    alpha = value["alpha"]
    metrics = value["metrics"]
    evidence = value["query_evidence"]
    if (
        type(value["name"]) is not str
        or not value["name"]
        or type(epochs) is not list
        or not epochs
        or any(type(epoch) is not int or epoch <= 0 for epoch in epochs)
        or type(checkpoints) is not list
        or len(checkpoints) != len(epochs)
        or any(type(path) is not str or not path for path in checkpoints)
        or type(alpha) is not float
        or not math.isfinite(alpha)
        or not 0.0 <= alpha <= 1.0
        or type(metrics) is not dict
        or tuple(metrics) != _SELECTION_METRIC_FIELDS
        or any(
            type(metric) is not float or not math.isfinite(metric) or not 0.0 <= metric <= 1.0
            for metric in metrics.values()
        )
        or type(evidence) is not dict
        or tuple(evidence) != ("top1_correct", "average_precision")
        or type(evidence["top1_correct"]) is not list
        or type(evidence["average_precision"]) is not list
        or not evidence["top1_correct"]
        or len(evidence["top1_correct"]) != len(evidence["average_precision"])
        or any(type(item) is not bool for item in evidence["top1_correct"])
        or any(
            type(item) is not float or not math.isfinite(item) or not 0.0 <= item <= 1.0
            for item in evidence["average_precision"]
        )
    ):
        raise ValueError("fine-tuned model state differs")
    expected_recall_at_1 = math.fsum(
        1.0 if item else 0.0 for item in evidence["top1_correct"]
    ) / len(evidence["top1_correct"])
    expected_map_at_r = math.fsum(evidence["average_precision"]) / len(
        evidence["average_precision"]
    )
    if not math.isclose(
        metrics["recall_at_1"], expected_recall_at_1, rel_tol=0.0, abs_tol=1e-12
    ) or not math.isclose(
        metrics["map_at_r"], expected_map_at_r, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("fine-tuned model state differs")


def _expected_training_protocol(
    model_spec: ModelSpec, *, partition_sha256: str, seed: int
) -> dict[str, object]:
    return {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": REGISTERED_TRAINER_SHA256,
        "unicom_revision": UNICOM_REVISION,
        "initial_checkpoint_sha256": model_spec.checkpoint_sha256,
        "partition_sha256": partition_sha256,
        "seed": seed,
        "epochs": 16,
        "batch_size": 128,
        "workers": 4,
        "learning_rate": 1e-5,
        "classifier_learning_rate": 1e-4,
        "margin": 0.25,
        "scale": 32.0,
        "objective": "official-eight-mask",
        "selected_features": 512,
        "holdout_seed": 0,
        "holdout_fraction": 0.2,
        "max_steps": None,
        "bf16": False,
        "compile": False,
        "fused": False,
    }


def _finetuned_model_state(
    payload: object,
    *,
    kind: str,
    model_spec: ModelSpec,
    partition_sha256: str,
    training_seed: int,
) -> Mapping[str, object]:
    expected_fields = _SELECTED_STATE_FIELDS if kind == "selected" else _ENDPOINT_STATE_FIELDS
    if type(payload) is not dict or tuple(payload) != expected_fields:
        raise ValueError("fine-tuned model state differs")
    state = payload["model"]
    training_protocol = payload["training_protocol"]
    expected_protocol = _expected_training_protocol(
        model_spec,
        partition_sha256=partition_sha256,
        seed=training_seed,
    )
    if (
        not isinstance(state, Mapping)
        or not state
        or type(training_protocol) is not dict
        or tuple(training_protocol) != tuple(expected_protocol)
        or any(
            type(training_protocol[key]) is not type(expected)
            for key, expected in expected_protocol.items()
        )
        or training_protocol != expected_protocol
    ):
        raise ValueError("fine-tuned model state differs")
    if kind == "selected":
        _validate_selection(payload["selection"])
    elif kind == "endpoint":
        endpoint = payload["endpoint"]
        _validate_selection(endpoint)
        if (
            endpoint["name"] != "epochs-16-alpha-1"
            or endpoint["epochs"] != [16]
            or len(endpoint["checkpoints"]) != 1
            or Path(endpoint["checkpoints"][0]).name != "epoch-0016.pt"
            or endpoint["alpha"] != 1.0
            or training_protocol.get("epochs") != 16
        ):
            raise ValueError("fine-tuned model state differs")
    else:
        raise ValueError("fine-tuned model state differs")
    return state


def _official_encoder(
    checkout: Path,
    checkpoint: Path,
    *,
    model_name: str = "ViT-B/16",
    finetuned_state: Path | None = None,
    finetuned_kind: str | None = None,
    partition_sha256: str | None = None,
    training_seed: int | None = None,
):
    model_spec = _model_spec(model_name)
    if checkpoint.name != model_spec.checkpoint_filename:
        raise ValueError(
            f"checkpoint filename must be {model_spec.checkpoint_filename}"
        )
    import torch
    from PIL import Image

    state = None
    if finetuned_state is not None:
        if (
            finetuned_kind not in ("selected", "endpoint")
            or partition_sha256 is None
            or type(training_seed) is not int
        ):
            raise ValueError("fine-tuned model state differs")
        payload = torch.load(
            finetuned_state,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state = _finetuned_model_state(
            payload,
            kind=finetuned_kind,
            model_spec=model_spec,
            partition_sha256=partition_sha256,
            training_seed=training_seed,
        )
    elif finetuned_kind is not None or training_seed is not None:
        raise ValueError("fine-tuned model state differs")

    package_root = (checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    if Path(unicom.__file__).resolve().parent != package_root / "unicom":
        raise ValueError("imported UNICOM package does not come from the pinned checkout")
    model, transform = unicom.load(
        model_name, download_root=str(checkpoint.parent)
    )
    if state is not None:
        model.load_state_dict(state, strict=True)
    model = model.cuda().eval()

    def encode(paths: tuple[Path, ...]) -> np.ndarray:
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).cuda(non_blocking=False)
        with torch.inference_mode():
            output = model(batch)
        return np.ascontiguousarray(output.float().cpu().numpy(), dtype=np.float32)

    return encode


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    try:
        if _git_revision(args.unicom_checkout) != UNICOM_REVISION:
            raise ValueError("UNICOM checkout revision differs")
        model_spec = _model_spec(args.model_name)
        if _sha256_file(args.checkpoint) != model_spec.checkpoint_sha256:
            raise ValueError("UNICOM checkpoint SHA-256 differs")
        fine_arguments = (
            args.finetuned_state,
            args.finetuned_kind,
            args.training_seed,
        )
        if any(value is None for value in fine_arguments) and any(
            value is not None for value in fine_arguments
        ):
            raise ValueError("fine-tuned model state kind differs")
        finetuned_sha256 = (
            None if args.finetuned_state is None else _sha256_file(args.finetuned_state)
        )
        partition = args.dataset_root / "Eval" / "list_eval_partition.txt"
        partition_sha256 = _sha256_file(partition)
        records = parse_inshop_partition(args.dataset_root)
        encode = _official_encoder(
            args.unicom_checkout,
            args.checkpoint,
            model_name=args.model_name,
            finetuned_state=args.finetuned_state,
            finetuned_kind=args.finetuned_kind,
            partition_sha256=partition_sha256,
            training_seed=args.training_seed,
        )
        if (
            args.finetuned_state is not None
            and _sha256_file(args.finetuned_state) != finetuned_sha256
        ):
            raise ValueError("fine-tuned model state changed while loading")
        transform = f"official UNICOM {args.model_name} load_model_and_transform"
        if finetuned_sha256 is not None:
            transform += (
                f";initial-checkpoint-sha256={model_spec.checkpoint_sha256}"
                f";finetuned-state-kind={args.finetuned_kind}"
                f";training-seed={args.training_seed}"
                f";finetuned-state-sha256={finetuned_sha256}"
            )
        export_embeddings(
            records,
            encode,
            {
                "model_identifier": model_spec.model_identifier,
                "model_revision": UNICOM_REVISION,
                "checkpoint_sha256": (
                    model_spec.checkpoint_sha256 if finetuned_sha256 is None else finetuned_sha256
                ),
                "image_list_sha256": partition_sha256,
                "transform": transform,
            },
            args.output,
            batch_size=args.batch_size,
        )
    except Exception as error:
        print(f"export failed: {error}", file=sys.stderr)
        return 2
    print(f"export complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
