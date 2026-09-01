#!/usr/bin/env python3
"""Run the pinned local-only frozen SigLIP manufacturer-band audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F

from sfora.data import _HF_DATASET_REVISIONS, load_image_retrieval_examples, materialize_image
from sfora.siglip_band_audit import (
    SiglipBandAuditAuthority,
    canonical_siglip_band_audit_bytes,
    score_siglip_frozen_bands,
    validate_siglip_band_audit_bytes,
    validate_siglip_band_inputs,
)

_DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
_CLASS_NAMES_SHA256 = "9da9ec6333105a7a2f0d50d7a5a6afe18b1ec3ede7dd8f1df298e59eb859ce35"
_MODEL_NAME = "google/siglip-so400m-patch14-384"
_MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
_READOUT = "vision_pooler_output"
_BATCH_SIZE = 8
_QUERY_BLOCK = 32


@dataclass(frozen=True, slots=True)
class LoadedBandAudit:
    """Authenticated tensors and identities ready for pure scoring."""

    descriptors: torch.Tensor
    labels: torch.Tensor
    class_names: tuple[str, ...]
    authority: SiglipBandAuditAuthority


def _lower_hex(value: str, *, length: int, role: str) -> str:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError(
            f"{role} must be {length} lowercase hexadecimal characters"
        )
    return value


def _source_commit(value: str) -> str:
    return _lower_hex(value, length=40, role="source commit")


def _sha256(value: str) -> str:
    return _lower_hex(value, length=64, role="digest")


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise argparse.ArgumentTypeError("paths must be normalized absolute paths")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the exact local-only band-audit capability boundary."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--result", required=True, type=_absolute_path)
    parser.add_argument("--source-commit", required=True, type=_source_commit)
    parser.add_argument("--source-tree-digest", required=True, type=_sha256)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--query-block", required=True, type=int)
    parser.add_argument("--execute-band-audit", required=True, action="store_true")
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    parsed = parser.parse_args(effective)
    if (parsed.batch_size, parsed.query_block) != (_BATCH_SIZE, _QUERY_BLOCK):
        parser.error("batch-size and query-block differ from the frozen audit authority")
    return parsed


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _tensor_sha256(role: str, tensor: torch.Tensor) -> str:
    canonical = tensor.detach().to(device="cpu").contiguous()
    if canonical.dtype == torch.float32:
        dtype = "float32-le"
        values = canonical.numpy().astype("<f4", copy=False).tobytes(order="C")
    elif canonical.dtype == torch.int64:
        dtype = "int64-le"
        values = canonical.numpy().astype("<i8", copy=False).tobytes(order="C")
    else:
        raise ValueError("band-audit tensor dtype differs")
    header = _canonical(
        {
            "role": role,
            "dtype": dtype,
            "shape": [int(size) for size in canonical.shape],
        }
    )
    return hashlib.sha256(header + values).hexdigest()


def _materialize_rgb(image: Any) -> Image.Image:
    materialized = materialize_image(image)
    if not isinstance(materialized, Image.Image):
        raise TypeError("Cars image materialization did not produce PIL")
    converted = materialized.convert("RGB")
    if converted.mode != "RGB":
        raise RuntimeError("Cars image materialization did not produce RGB")
    return converted


def _load_cars_class_names() -> tuple[str, ...]:
    from datasets import load_dataset

    dataset = load_dataset(
        "tanganke/stanford_cars",
        split="train",
        revision=_DATASET_REVISION,
    )
    label_feature = dataset.features.get("label")
    names = getattr(label_feature, "names", None)
    if (
        not isinstance(names, list)
        or len(names) != 196
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise RuntimeError("Cars class-name authority is unavailable")
    result = tuple(names)
    if hashlib.sha256(_canonical(list(result))).hexdigest() != _CLASS_NAMES_SHA256:
        raise RuntimeError("Cars class-name digest differs")
    return result


def _encode(examples: list[Any], *, batch_size: int, device: torch.device) -> torch.Tensor:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        _MODEL_NAME,
        revision=_MODEL_REVISION,
        local_files_only=True,
    )
    model = (
        AutoModel.from_pretrained(
            _MODEL_NAME,
            revision=_MODEL_REVISION,
            local_files_only=True,
        )
        .eval()
        .to(device)
    )
    rows: list[torch.Tensor] = []
    observed_shape: tuple[int, int] | None = None
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            images = [_materialize_rgb(row.image) for row in examples[start : start + batch_size]]
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
            shape = (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
            if observed_shape is None:
                observed_shape = shape
            elif shape != observed_shape:
                raise RuntimeError("SigLIP processor image shape differs")
            output = model.vision_model(pixel_values=pixel_values.to(device))
            rows.append(F.normalize(output.pooler_output.float(), dim=-1).cpu())
    if observed_shape is None:
        raise RuntimeError("Cars training split is empty")
    del model
    return torch.cat(rows).contiguous()


def prepare_band_audit(
    *,
    source_commit: str,
    source_tree_digest: str,
    batch_size: int,
    query_block: int,
) -> LoadedBandAudit:
    """Load and authenticate the pinned local dataset/model evidence once."""

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen SigLIP band audit requires CUDA")
    if _HF_DATASET_REVISIONS["tanganke/stanford_cars"] != _DATASET_REVISION:
        raise RuntimeError("Cars dataset revision authority differs")
    examples = load_image_retrieval_examples(dataset_name="cars", split="train")
    class_names = _load_cars_class_names()
    labels_cpu = torch.tensor([int(row.label) for row in examples], dtype=torch.int64)
    descriptors_cpu = _encode(examples, batch_size=batch_size, device=torch.device("cuda"))
    validate_siglip_band_inputs(descriptors_cpu, labels_cpu, class_names)
    example_rows = [(str(row.example_id), int(row.label)) for row in examples]
    authority = SiglipBandAuditAuthority(
        source_commit=source_commit,
        source_tree_digest=source_tree_digest,
        dataset_revision=_DATASET_REVISION,
        dataset_examples_sha256=hashlib.sha256(
            _canonical({"examples": sorted(example_rows)})
        ).hexdigest(),
        ordered_example_ids_sha256=hashlib.sha256(
            _canonical({"example_ids": [row[0] for row in example_rows]})
        ).hexdigest(),
        descriptor_sha256=_tensor_sha256("frozen-pooler", descriptors_cpu),
        label_vector_sha256=_tensor_sha256("labels", labels_cpu),
        class_names_sha256=_CLASS_NAMES_SHA256,
        model_name=_MODEL_NAME,
        model_revision=_MODEL_REVISION,
        readout=_READOUT,
        split="train",
        batch_size=batch_size,
        query_block=query_block,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
    )
    return LoadedBandAudit(
        descriptors=descriptors_cpu.cuda(),
        labels=labels_cpu.cuda(),
        class_names=class_names,
        authority=authority,
    )


def _read_regular(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError("band-audit result path differs")
    return path.read_bytes()


def _write_exclusive(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    path.parent.mkdir(parents=True, exist_ok=True)
    installed = False
    try:
        with partial.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(path)
        installed = True
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if installed and path.is_file() and not path.is_symlink():
            path.unlink()
        if partial.is_file() and not partial.is_symlink():
            partial.unlink()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run one authenticated audit and publish one byte-exact local result."""

    arguments = parse_args(argv)
    if arguments.result.exists() or arguments.result.is_symlink():
        raise FileExistsError("band-audit result already exists")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    loaded = prepare_band_audit(
        source_commit=arguments.source_commit,
        source_tree_digest=arguments.source_tree_digest,
        batch_size=arguments.batch_size,
        query_block=arguments.query_block,
    )
    evidence = score_siglip_frozen_bands(
        loaded.descriptors,
        loaded.labels,
        loaded.class_names,
        query_block=arguments.query_block,
    )
    raw = canonical_siglip_band_audit_bytes(evidence, authority=loaded.authority)
    result_written = False
    try:
        _write_exclusive(arguments.result, raw)
        result_written = True
        written = _read_regular(arguments.result)
        if written != raw:
            raise ValueError("band-audit result differs after write")
        validate_siglip_band_audit_bytes(written, expected_authority=loaded.authority)
    except BaseException:
        if result_written and arguments.result.is_file() and not arguments.result.is_symlink():
            arguments.result.unlink()
        raise
    sys.stdout.write(
        json.dumps(
            {
                "result": str(arguments.result),
                "result_file_sha256": hashlib.sha256(raw).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
