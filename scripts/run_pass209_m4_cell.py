#!/usr/bin/env python3
"""Run one registered offline Pass209 M4 frozen-descriptor cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image

from sfora.data import (
    _HF_DATASET_REVISIONS,
    ImageExample,
    load_image_retrieval_examples,
    materialize_image,
)
from sfora.pass209_m4 import (
    REGISTERED_M4_CELLS,
    M4CellSpec,
    M4DescriptorHeader,
    M4Example,
    QueryEvidence,
    canonical_json_bytes,
    configure_reference_scorer,
    decode_descriptor_file,
    encode_descriptor_file,
    publish_new_outputs,
    score_descriptor_plane,
    score_descriptor_plane_cuda,
)
from sfora.substrate_screen import (
    SUBSTRATE_F0_CLASSES,
    SubstrateScreenEvidence,
    score_frozen_substrate_evidence,
    validate_substrate_holdout,
)

_DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
_EXAMPLES_SHA256 = "83a7800ee948a816e2fb9a2c9163027d9e90f167abc90052bf220619fa32240f"
_ERROR_MANIFEST_SHA256 = "64d491607d4dac144b31edac3a182130e6f94f994a272f612c195a7a72d55611"
_QUERY_BLOCK = 32
_V1_KEYS = frozenset(
    {
        "schema",
        "claim_eligible",
        "source_revision",
        "source_tree_digest",
        "dataset",
        "dataset_revision",
        "dataset_examples_sha256",
        "split",
        "holdout_classes",
        "model_name",
        "model_revision",
        "readout",
        "compute_dtype",
        "processor_image_shape",
        "descriptors_validated",
        "norm_tolerance",
        "metrics",
        "gates",
        "passed",
    }
)
_V2_EXTRA_KEYS = frozenset(
    {
        "cell",
        "batch_size",
        "query_block",
        "descriptor_shape",
        "descriptor_sha256",
        "error_manifest_sha256",
    }
)


def _revision(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("expected an exact 40-character revision")
    return value


def _sha256(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError("expected an exact lowercase SHA-256")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the closed local-file execution surface for one cell."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=tuple(REGISTERED_M4_CELLS), required=True)
    parser.add_argument("--prerequisite", type=Path, required=True)
    parser.add_argument("--error-manifest", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--descriptor-output", type=Path, required=True)
    parser.add_argument("--query-output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-revision", type=_revision, required=True)
    parser.add_argument("--source-tree-digest", type=_sha256, required=True)
    parser.add_argument("--uv-lock", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    return parser.parse_args(argv)


def _require_offline_environment() -> None:
    required = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    if any(os.environ.get(name) != "1" for name in required):
        raise RuntimeError("Pass209 M4 requires the complete offline environment")


def _concrete_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        return set(actual) == set(expected) and all(
            _concrete_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(actual, list)
        return len(actual) == len(expected) and all(
            _concrete_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def load_prerequisite(path: Path, spec: M4CellSpec) -> dict[str, object]:
    """Authenticate and validate one exact historical substrate receipt."""

    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != spec.prerequisite_sha256:
        raise ValueError("prerequisite receipt digest differs")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prerequisite receipt is not JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError("prerequisite receipt bytes differ from canonical authority")
    expected_keys = _V1_KEYS
    if spec.cell == "siglip2-so400m":
        expected_keys |= {"cell"}
    elif spec.cell == "siglip-so400m":
        expected_keys |= _V2_EXTRA_KEYS
    if set(value) != expected_keys:
        raise ValueError("prerequisite receipt schema differs")

    expected: dict[str, object] = {
        "schema": spec.prerequisite_schema,
        "claim_eligible": False,
        "dataset": "cars",
        "dataset_revision": _DATASET_REVISION,
        "dataset_examples_sha256": _EXAMPLES_SHA256,
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "model_name": spec.model_name,
        "model_revision": spec.model_revision,
        "readout": spec.readout,
        "compute_dtype": "float32",
        "processor_image_shape": list(spec.processor_image_shape),
        "descriptors_validated": True,
        "norm_tolerance": 1.0e-6,
        "metrics": {
            "correct": spec.expected_correct,
            "queries": 1345,
            "recall_at_1": spec.expected_correct / 1345,
        },
        "gates": {"expected_queries": 1345, "recall_at_1_minimum": 0.94},
        "passed": spec.expected_correct / 1345 >= 0.94,
    }
    for key, expected_value in expected.items():
        if not _concrete_equal(value.get(key), expected_value):
            raise ValueError(f"prerequisite receipt {key} differs")
    source_revision = value.get("source_revision")
    if type(source_revision) is not str or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("prerequisite receipt source revision differs")
    source_tree_digest = value.get("source_tree_digest")
    if (
        type(source_tree_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", source_tree_digest) is None
    ):
        raise ValueError("prerequisite receipt source tree differs")
    if "cell" in value and value["cell"] != spec.cell:
        raise ValueError("prerequisite receipt cell differs")
    if spec.cell == "siglip-so400m":
        v2_expected: dict[str, object] = {
            "batch_size": 8,
            "query_block": 32,
            "descriptor_shape": [1345, spec.descriptor_dimensions],
            "descriptor_sha256": spec.legacy_descriptor_sha256,
            "error_manifest_sha256": _ERROR_MANIFEST_SHA256,
        }
        for key, expected_value in v2_expected.items():
            if not _concrete_equal(value.get(key), expected_value):
                raise ValueError(f"prerequisite receipt {key} differs")
    return value


def _require_new_outputs(paths: tuple[Path, ...]) -> None:
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("output paths must be distinct")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        partial = path.with_name(f".{path.name}.partial")
        if partial.exists():
            raise FileExistsError(f"refusing pre-existing partial {partial}")


def _checkpoint_header(
    *,
    spec: M4CellSpec,
    source_revision: str,
    source_tree_digest: str,
    dataset_examples_ordered_sha256: str,
    descriptors: torch.Tensor,
) -> M4DescriptorHeader:
    raw = descriptors.numpy().astype("<f4", copy=False).tobytes(order="C")
    return M4DescriptorHeader(
        schema="sfora-pass209-m4-descriptor-v1",
        source_revision=source_revision,
        source_tree_digest=source_tree_digest,
        dataset="cars",
        dataset_revision=_DATASET_REVISION,
        dataset_examples_sha256=_EXAMPLES_SHA256,
        dataset_examples_ordered_sha256=dataset_examples_ordered_sha256,
        split="train",
        holdout_classes=tuple(range(82, 98)),
        compute_dtype="float32",
        cell=spec.cell,
        model_name=spec.model_name,
        model_revision=spec.model_revision,
        readout=spec.readout,
        rows=int(descriptors.shape[0]),
        dimensions=spec.descriptor_dimensions,
        payload_bytes=len(raw),
        payload_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _publish_encoding_checkpoint(
    directory: Path,
    *,
    spec: M4CellSpec,
    source_revision: str,
    source_tree_digest: str,
    dataset_examples_ordered_sha256: str,
    descriptors: torch.Tensor,
) -> str:
    values = descriptors.detach().to(device="cpu", dtype=torch.float32).contiguous()
    rows = int(values.shape[0])
    if (
        values.shape != (rows, spec.descriptor_dimensions)
        or not 0 < rows < spec.expected_rows
        or rows % spec.batch_size != 0
    ):
        raise ValueError("M4 checkpoint descriptor shape differs")
    header = _checkpoint_header(
        spec=spec,
        source_revision=source_revision,
        source_tree_digest=source_tree_digest,
        dataset_examples_ordered_sha256=dataset_examples_ordered_sha256,
        descriptors=values,
    )
    payload = encode_descriptor_file(header, values)
    path = directory / f"checkpoint-{rows:04d}.bin"
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("M4 checkpoint bytes differ")
        return hashlib.sha256(payload).hexdigest()
    previous = tuple(directory.glob("checkpoint-*.bin")) if directory.is_dir() else ()
    publish_new_outputs(((path, payload),))
    for old in previous:
        if old != path:
            old.unlink()
    return hashlib.sha256(payload).hexdigest()


def _load_encoding_checkpoint(
    directory: Path,
    *,
    spec: M4CellSpec,
    source_revision: str,
    source_tree_digest: str,
    dataset_examples_ordered_sha256: str,
) -> torch.Tensor:
    if not directory.exists():
        return torch.empty((0, spec.descriptor_dimensions), dtype=torch.float32)
    if not directory.is_dir():
        raise ValueError("M4 checkpoint directory differs")
    paths = tuple(sorted(directory.iterdir()))
    if not paths:
        return torch.empty((0, spec.descriptor_dimensions), dtype=torch.float32)
    candidates: list[tuple[int, torch.Tensor]] = []
    for path in paths:
        match = re.fullmatch(r"checkpoint-([0-9]{4})\.bin", path.name)
        if match is None or not path.is_file():
            raise ValueError("M4 checkpoint namespace differs")
        header, descriptors = decode_descriptor_file(path.read_bytes())
        rows = int(match.group(1))
        expected = _checkpoint_header(
            spec=spec,
            source_revision=source_revision,
            source_tree_digest=source_tree_digest,
            dataset_examples_ordered_sha256=dataset_examples_ordered_sha256,
            descriptors=descriptors,
        )
        if (
            header != expected
            or rows != header.rows
            or not 0 < rows < spec.expected_rows
            or rows % spec.batch_size != 0
        ):
            raise ValueError("M4 checkpoint source authority differs")
        candidates.append((rows, descriptors))
    return max(candidates, key=lambda item: item[0])[1]


def _clear_encoding_checkpoints(directory: Path) -> None:
    if not directory.exists():
        return
    for path in tuple(directory.iterdir()):
        if re.fullmatch(r"checkpoint-[0-9]{4}\.bin", path.name) is None:
            raise ValueError("M4 checkpoint namespace differs")
        path.unlink()
    directory.rmdir()


def _read_error_manifest(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _ERROR_MANIFEST_SHA256:
        raise ValueError("M4 source error-manifest digest differs")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("M4 source error manifest is not JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError("M4 source error manifest is not canonical")
    errors = value.get("errors")
    if (
        value.get("schema") != "sfora-frozen-substrate-errors-v1"
        or value.get("error_count") != 103
        or type(errors) is not list
        or len(errors) != 103
    ):
        raise ValueError("M4 source error-manifest schema differs")
    positions: list[int] = []
    for row in errors:
        if type(row) is not dict or type(row.get("query_position")) is not int:
            raise ValueError("M4 source error-manifest row differs")
        positions.append(row["query_position"])
    if positions != sorted(set(positions)):
        raise ValueError("M4 source error-manifest query positions differ")
    return value


def _materialize_rgb(value: object) -> Image.Image:
    image = materialize_image(value)
    if not isinstance(image, Image.Image):
        raise TypeError("Cars image materialization did not produce PIL")
    converted = image.convert("RGB")
    if converted.mode != "RGB":
        raise RuntimeError("Cars image materialization did not produce RGB")
    return converted


def _dataset_examples_sha256(examples: Sequence[ImageExample]) -> str:
    """Reproduce the historical order-insensitive receipt digest."""

    rows = []
    for row in examples:
        rows.append((str(row.example_id), int(row.label)))
    return hashlib.sha256(canonical_json_bytes({"examples": sorted(rows)})).hexdigest()


def _ordered_examples_sha256(examples: Sequence[ImageExample]) -> str:
    rows = [(str(row.example_id), int(row.label)) for row in examples]
    return hashlib.sha256(canonical_json_bytes({"examples": rows})).hexdigest()


def _legacy_descriptor_sha256(descriptors: torch.Tensor) -> str:
    values = descriptors.detach().to(device="cpu", dtype=torch.float32).contiguous()
    header = canonical_json_bytes(
        {"dtype": "float32-le", "shape": [int(size) for size in values.shape]}
    )
    payload = values.numpy().astype("<f4", copy=False).tobytes(order="C")
    return hashlib.sha256(header + payload).hexdigest()


def _gpu_environment(device: torch.device) -> dict[str, object]:
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    torch_uuid = getattr(properties, "uuid", None)
    if type(torch_uuid) is not str or re.fullmatch(r"[0-9a-f-]{36}", torch_uuid) is None:
        raise RuntimeError("CUDA device UUID is unavailable")
    expected_uuid = f"GPU-{torch_uuid}"
    identity = subprocess.run(
        [
            "nvidia-smi",
            f"--id={expected_uuid}",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parts = [part.strip() for part in identity.split(",")]
    if len(parts) != 2 or not all(parts) or parts[0] != expected_uuid:
        raise RuntimeError("CUDA device identity is unavailable")
    return {
        "schema": "sfora-pass209-m4-cuda-environment-v1",
        "product_name": properties.name,
        "uuid": parts[0],
        "compute_capability": [properties.major, properties.minor],
        "driver_version": parts[1],
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "torch_version": str(torch.__version__),
    }


def _encode(
    examples: Sequence[ImageExample],
    spec: M4CellSpec,
    *,
    device: torch.device,
    existing_descriptors: torch.Tensor,
    checkpoint_callback: Callable[[torch.Tensor], str],
    source_revision: str,
    source_tree_digest: str,
) -> tuple[torch.Tensor, tuple[int, int]]:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        spec.model_name,
        revision=spec.model_revision,
        local_files_only=True,
    )
    model = (
        AutoModel.from_pretrained(
            spec.model_name,
            revision=spec.model_revision,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        .eval()
        .to(device=device, dtype=torch.float32)
    )
    existing_rows = int(existing_descriptors.shape[0])
    if existing_descriptors.shape != (existing_rows, spec.descriptor_dimensions) or not (
        0 <= existing_rows <= len(examples)
    ):
        raise RuntimeError("M4 resumed descriptor shape differs")
    rows: list[torch.Tensor] = [existing_descriptors] if existing_rows else []
    observed_shape: tuple[int, int] | None = None
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        for start in range(existing_rows, len(examples), spec.batch_size):
            stop = min(start + spec.batch_size, len(examples))
            images = [_materialize_rgb(row.image) for row in examples[start:stop]]
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
            shape = (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
            if observed_shape is None:
                observed_shape = shape
            elif observed_shape != shape:
                raise RuntimeError("M4 processor emitted inconsistent image shapes")
            if spec.readout == "last_hidden_state_cls":
                output = model(pixel_values=pixel_values.to(device=device, dtype=torch.float32))
                descriptor = output.last_hidden_state[:, 0, :]
            elif spec.readout == "vision_pooler_output":
                output = model.vision_model(
                    pixel_values=pixel_values.to(device=device, dtype=torch.float32)
                )
                descriptor = output.pooler_output
            else:
                raise RuntimeError("M4 cell readout is unregistered")
            rows.append(torch.nn.functional.normalize(descriptor.float(), dim=-1).cpu())
            if stop < len(examples):
                checkpoint_sha256 = checkpoint_callback(torch.cat(rows))
                print(
                    json.dumps(
                        {
                            "schema": "sfora-pass209-m4-progress-v1",
                            "cell": spec.cell,
                            "checkpoint_sha256": checkpoint_sha256,
                            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                            "rows": stop,
                            "source_revision": source_revision,
                            "source_tree_digest": source_tree_digest,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    del model
    torch.cuda.empty_cache()
    if observed_shape is None:
        raise RuntimeError("M4 cell has no examples")
    return torch.cat(rows), observed_shape


def _historical_cuda_evidence(
    descriptors: torch.Tensor, labels: torch.Tensor
) -> SubstrateScreenEvidence:
    return score_frozen_substrate_evidence(
        descriptors.cuda(),
        labels.cuda(),
        query_block=_QUERY_BLOCK,
    )


def _historical_cuda_queries(
    descriptors: torch.Tensor,
    examples: tuple[M4Example, ...],
) -> tuple[QueryEvidence, ...]:
    return score_descriptor_plane_cuda(
        descriptors.cuda(),
        examples,
        block_size=_QUERY_BLOCK,
    )


def _run(args: argparse.Namespace) -> dict[str, object]:
    spec = REGISTERED_M4_CELLS[args.cell]
    outputs = (args.receipt_output, args.descriptor_output, args.query_output)
    _require_new_outputs(outputs)
    prerequisite = load_prerequisite(args.prerequisite, spec)
    manifest = _read_error_manifest(args.error_manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("Pass209 M4 descriptor inference requires CUDA")
    if _HF_DATASET_REVISIONS["tanganke/stanford_cars"] != _DATASET_REVISION:
        raise RuntimeError("Cars dataset revision authority differs")

    all_examples = load_image_retrieval_examples(dataset_name="cars", split="train")
    if _dataset_examples_sha256(all_examples) != _EXAMPLES_SHA256:
        raise RuntimeError("Cars example-sequence authority differs")
    holdout = [row for row in all_examples if int(row.label) in SUBSTRATE_F0_CLASSES]
    if len(holdout) != spec.expected_rows:
        raise RuntimeError("M4 holdout cardinality differs")
    labels = torch.tensor([int(row.label) for row in holdout], dtype=torch.int64)
    validate_substrate_holdout(split="train", labels=labels)
    examples = tuple(
        M4Example(position=index, example_id=str(row.example_id), label=int(row.label))
        for index, row in enumerate(holdout)
    )
    ordered_examples_sha256 = _ordered_examples_sha256(holdout)
    existing_descriptors = _load_encoding_checkpoint(
        args.checkpoint_dir,
        spec=spec,
        source_revision=args.source_revision,
        source_tree_digest=args.source_tree_digest,
        dataset_examples_ordered_sha256=ordered_examples_sha256,
    )

    def checkpoint_callback(values: torch.Tensor) -> str:
        return _publish_encoding_checkpoint(
            args.checkpoint_dir,
            spec=spec,
            source_revision=args.source_revision,
            source_tree_digest=args.source_tree_digest,
            dataset_examples_ordered_sha256=ordered_examples_sha256,
            descriptors=values,
        )

    device = torch.device("cuda")
    cuda_environment = _gpu_environment(device)
    descriptors, image_shape = _encode(
        holdout,
        spec,
        device=device,
        existing_descriptors=existing_descriptors,
        checkpoint_callback=checkpoint_callback,
        source_revision=args.source_revision,
        source_tree_digest=args.source_tree_digest,
    )
    if image_shape != spec.processor_image_shape:
        raise RuntimeError("M4 processor image-shape authority differs")
    if tuple(descriptors.shape) != (spec.expected_rows, spec.descriptor_dimensions):
        raise RuntimeError("M4 descriptor shape authority differs")
    legacy_digest = _legacy_descriptor_sha256(descriptors)
    legacy_descriptor_passed = (
        spec.legacy_descriptor_sha256 is None or legacy_digest == spec.legacy_descriptor_sha256
    )
    historical = _historical_cuda_evidence(descriptors, labels)
    cuda_queries = _historical_cuda_queries(descriptors, examples)
    cuda_errors = [
        {
            "query_position": row.query_position,
            "nearest_position": row.nearest_position,
            "query_label": row.query_label,
            "nearest_label": row.nearest_label,
        }
        for row in cuda_queries
        if not row.correct
    ]
    if (
        cuda_errors != [asdict(row) for row in historical.errors]
        or sum(row.correct for row in cuda_queries) != historical.metrics.correct
    ):
        raise RuntimeError("historical CUDA query evidence differs from aggregate scorer")
    historical_error_positions = {row.query_position for row in historical.errors}
    historical_count_passed = historical.metrics.correct == spec.expected_correct
    historical_errors_passed = True
    if spec.cell == "siglip-so400m":
        manifest_errors = manifest["errors"]
        assert isinstance(manifest_errors, list)
        expected_errors = {row["query_position"] for row in manifest_errors}
        historical_errors_passed = historical_error_positions == expected_errors
    reproduction_passed = (
        legacy_descriptor_passed and historical_count_passed and historical_errors_passed
    )
    scorer_environment = configure_reference_scorer(args.uv_lock)
    queries = score_descriptor_plane(descriptors, examples, block_size=_QUERY_BLOCK)
    correct = sum(row.correct for row in queries)

    raw_descriptor = descriptors.numpy().astype("<f4", copy=False).tobytes(order="C")
    descriptor_header = M4DescriptorHeader(
        schema="sfora-pass209-m4-descriptor-v1",
        source_revision=args.source_revision,
        source_tree_digest=args.source_tree_digest,
        dataset="cars",
        dataset_revision=_DATASET_REVISION,
        dataset_examples_sha256=_EXAMPLES_SHA256,
        dataset_examples_ordered_sha256=ordered_examples_sha256,
        split="train",
        holdout_classes=tuple(range(82, 98)),
        compute_dtype="float32",
        cell=spec.cell,
        model_name=spec.model_name,
        model_revision=spec.model_revision,
        readout=spec.readout,
        rows=spec.expected_rows,
        dimensions=spec.descriptor_dimensions,
        payload_bytes=len(raw_descriptor),
        payload_sha256=hashlib.sha256(raw_descriptor).hexdigest(),
    )
    descriptor_payload = encode_descriptor_file(descriptor_header, descriptors)
    query_value: dict[str, object] = {
        "schema": "sfora-pass209-m4-query-evidence-v1",
        "claim_eligible": False,
        "cell": spec.cell,
        "dataset_examples_sha256": _EXAMPLES_SHA256,
        "dataset_examples_ordered_sha256": ordered_examples_sha256,
        "descriptor_file_sha256": hashlib.sha256(descriptor_payload).hexdigest(),
        "query_block": _QUERY_BLOCK,
        "rows": [asdict(row) for row in queries],
        "historical_cuda_rows": [asdict(row) for row in cuda_queries],
    }
    query_payload = canonical_json_bytes(query_value)
    receipt: dict[str, object] = {
        "schema": "sfora-pass209-m4-cell-receipt-v1",
        "claim_eligible": False,
        "source_revision": args.source_revision,
        "source_tree_digest": args.source_tree_digest,
        "dataset": "cars",
        "dataset_revision": _DATASET_REVISION,
        "dataset_examples_sha256": _EXAMPLES_SHA256,
        "dataset_examples_ordered_sha256": ordered_examples_sha256,
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "compute_dtype": "float32",
        "error_manifest_sha256": _ERROR_MANIFEST_SHA256,
        "cell": spec.cell,
        "model_name": spec.model_name,
        "model_revision": spec.model_revision,
        "readout": spec.readout,
        "batch_size": spec.batch_size,
        "query_block": _QUERY_BLOCK,
        "processor_image_shape": list(image_shape),
        "descriptor_shape": list(descriptors.shape),
        "descriptor_file_sha256": hashlib.sha256(descriptor_payload).hexdigest(),
        "descriptor_payload_sha256": descriptor_header.payload_sha256,
        "legacy_descriptor_sha256": legacy_digest,
        "query_evidence_sha256": hashlib.sha256(query_payload).hexdigest(),
        "correct": correct,
        "cpu_reference_correct": correct,
        "expected_historical_correct": spec.expected_correct,
        "historical_cuda_correct": historical.metrics.correct,
        "historical_cuda_errors": [asdict(row) for row in historical.errors],
        "legacy_descriptor_passed": legacy_descriptor_passed,
        "historical_count_passed": historical_count_passed,
        "historical_errors_passed": historical_errors_passed,
        "reproduction_passed": reproduction_passed,
        "queries": len(queries),
        "prerequisite_sha256": spec.prerequisite_sha256,
        "prerequisite_schema": prerequisite["schema"],
        "scorer_environment": asdict(scorer_environment),
        "cuda_environment": cuda_environment,
    }
    receipt_payload = canonical_json_bytes(receipt)
    publish_new_outputs(
        (
            (args.receipt_output, receipt_payload),
            (args.descriptor_output, descriptor_payload),
            (args.query_output, query_payload),
        )
    )
    _clear_encoding_checkpoints(args.checkpoint_dir)
    return {
        "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "descriptor_sha256": hashlib.sha256(descriptor_payload).hexdigest(),
        "query_sha256": hashlib.sha256(query_payload).hexdigest(),
        "cell": spec.cell,
        "correct": correct,
        "reproduction_passed": reproduction_passed,
    }


def main() -> None:
    """Execute one registered cell and print its non-scientific terminal summary."""

    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _require_offline_environment()
    result = _run(args)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if not result["reproduction_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
