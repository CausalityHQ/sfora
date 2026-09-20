#!/usr/bin/env python3
"""One unchanged CUB replication of the frozen SigLIP2 compact screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import tarfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image

from sfora.deterministic_similarity_runtime import (
    configure_deterministic_similarity_runtime,
)

_CARS_PATH = Path(__file__).with_name("_scratch_siglip2_so400m_cars_screen.py")
_CARS_SPEC = importlib.util.spec_from_file_location(
    "scratch_siglip2_so400m_cars_support", _CARS_PATH
)
if _CARS_SPEC is None or _CARS_SPEC.loader is None:
    raise RuntimeError("Cars SigLIP2 support module is unavailable")
_CARS = importlib.util.module_from_spec(_CARS_SPEC)
_CARS_SPEC.loader.exec_module(_CARS)

classify_candidate = _CARS.classify_candidate
enable_repeatable_fused_inference = _CARS.enable_repeatable_fused_inference
paired_class_bootstrap = _CARS.paired_class_bootstrap
validate_processor_authority = _CARS.validate_processor_authority
_fit_score = _CARS._fit_score
_ProcessorCollate = _CARS._ProcessorCollate
sha256 = _CARS.sha256

MODEL_ID = _CARS.MODEL_ID
MODEL_REVISION = _CARS.MODEL_REVISION
MODEL_SHA256 = _CARS.MODEL_SHA256
CONFIG_SHA256 = _CARS.CONFIG_SHA256
PROCESSOR_SHA256 = _CARS.PROCESSOR_SHA256
CUB_ARCHIVE_SHA256 = "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081"
CUB_ARCHIVE_BYTES = 1_150_585_339
BASELINE_FEATURE_SHA256 = "847a40bd8c0c2a5289de9b5a8eca93c935d4eafed6507e1360da3a3bfee6f62e"
CUB_PROTOCOL = "sha256-name-ordered-100-fit-train-100-eval-test-classes"


def _parse_index(path: Path, *, role: str) -> tuple[tuple[int, str], ...]:
    rows: list[tuple[int, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"CUB {role} row {line_number} differs")
        try:
            ordinal = int(fields[0])
        except ValueError as error:
            raise ValueError(f"CUB {role} row {line_number} differs") from error
        if ordinal != line_number or not fields[1]:
            raise ValueError(f"CUB {role} identity differs")
        rows.append((ordinal, fields[1]))
    if not rows:
        raise ValueError(f"CUB {role} identity differs")
    return tuple(rows)


def select_cub_protocol_rows(
    dataset_root: Path,
    metadata: Mapping[str, object],
    *,
    expected_counts: tuple[int, int] = (2_997, 2_857),
    expected_classes: tuple[int, int] = (100, 100),
) -> tuple[list[Path], np.ndarray, list[Path], np.ndarray]:
    """Reconstruct the exact baseline row order from official CUB metadata."""

    if (
        metadata.get("protocol") != CUB_PROTOCOL
        or type(expected_counts) is not tuple
        or type(expected_classes) is not tuple
        or len(expected_counts) != 2
        or len(expected_classes) != 2
    ):
        raise ValueError("CUB protocol authority differs")
    fit_names = metadata.get("fit_classes")
    evaluation_names = metadata.get("evaluation_classes")
    if (
        type(fit_names) is not list
        or type(evaluation_names) is not list
        or len(fit_names) != expected_classes[0]
        or len(evaluation_names) != expected_classes[1]
        or any(type(value) is not str for value in fit_names + evaluation_names)
        or set(fit_names) & set(evaluation_names)
    ):
        raise ValueError("CUB class authority differs")

    classes = dict(_parse_index(dataset_root / "classes.txt", role="class"))
    images = _parse_index(dataset_root / "images.txt", role="image")
    labels = _parse_index(dataset_root / "image_class_labels.txt", role="label")
    splits = _parse_index(dataset_root / "train_test_split.txt", role="split")
    if not (len(images) == len(labels) == len(splits)):
        raise ValueError("CUB row count differs")
    name_to_id = {name: class_id for class_id, name in classes.items()}
    if len(name_to_id) != len(classes):
        raise ValueError("CUB class authority differs")
    try:
        fit_ids = sorted(name_to_id[name] for name in fit_names)
        evaluation_ids = sorted(name_to_id[name] for name in evaluation_names)
    except KeyError as error:
        raise ValueError("CUB class authority differs") from error

    selected: dict[str, list[tuple[int, int, Path]]] = {"fit": [], "evaluation": []}
    for (image_id, relative_text), (label_id, class_text), (split_id, split_text) in zip(
        images, labels, splits, strict=True
    ):
        if image_id != label_id or image_id != split_id:
            raise ValueError("CUB row identity differs")
        try:
            class_id = int(class_text)
            official_train = int(split_text)
        except ValueError as error:
            raise ValueError("CUB row authority differs") from error
        if official_train not in (0, 1):
            raise ValueError("CUB split authority differs")
        relative = PurePosixPath(relative_text)
        class_name = classes.get(class_id)
        path = dataset_root / "images" / Path(*relative.parts)
        if (
            class_name is None
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != class_name
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ValueError("CUB image authority differs")
        if class_id in fit_ids and official_train == 1:
            selected["fit"].append((class_id, image_id, path))
        elif class_id in evaluation_ids and official_train == 0:
            selected["evaluation"].append((class_id, image_id, path))

    outputs: list[object] = []
    for role, class_ids, expected_count in (
        ("fit", fit_ids, expected_counts[0]),
        ("evaluation", evaluation_ids, expected_counts[1]),
    ):
        rows = sorted(selected[role])
        remap = {class_id: index for index, class_id in enumerate(class_ids)}
        paths = [path for _, _, path in rows]
        remapped = np.asarray([remap[class_id] for class_id, _, _ in rows], dtype=np.int64)
        if len(paths) != expected_count or len(set(paths)) != len(paths):
            raise ValueError(f"CUB {role} row authority differs")
        outputs.extend((paths, remapped))
    return outputs[0], outputs[1], outputs[2], outputs[3]  # type: ignore[return-value]


def verify_selected_extraction(
    archive_path: Path,
    dataset_root: Path,
    selected_paths: Sequence[Path],
) -> str:
    """Bind every consumed extracted byte to the authenticated CUB archive."""

    relative: dict[str, Path] = {
        name: dataset_root / name
        for name in (
            "classes.txt",
            "image_class_labels.txt",
            "images.txt",
            "train_test_split.txt",
        )
    }
    for path in selected_paths:
        key = path.relative_to(dataset_root).as_posix()
        relative[key] = path
    observed: dict[str, str] = {}
    prefix = "CUB_200_2011/"
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            key = member.name[len(prefix) :]
            extracted = relative.get(key)
            if extracted is None:
                continue
            if key in observed:
                raise ValueError("CUB archive member is duplicate")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("CUB archive content differs")
            digest = hashlib.sha256()
            with stream, extracted.open("rb") as local:
                while True:
                    archive_chunk = stream.read(1024 * 1024)
                    local_chunk = local.read(1024 * 1024)
                    if archive_chunk != local_chunk:
                        raise ValueError("CUB extracted content differs")
                    if not archive_chunk:
                        break
                    digest.update(archive_chunk)
            observed[key] = digest.hexdigest()
    if set(observed) != set(relative):
        raise ValueError("CUB archive content differs")
    manifest = hashlib.sha256()
    for key in sorted(observed):
        manifest.update(key.encode())
        manifest.update(b"\0")
        manifest.update(observed[key].encode())
        manifest.update(b"\n")
    return manifest.hexdigest()


class _PathDataset(torch.utils.data.Dataset):
    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = tuple(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Image.Image:
        with Image.open(self.paths[index]) as image:
            return image.convert("RGB")


def _encode_paths(
    model: torch.nn.Module,
    paths: Sequence[Path],
    processor: object,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    loader = torch.utils.data.DataLoader(
        _PathDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_ProcessorCollate(processor),
    )
    rows: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for step, images in enumerate(loader, start=1):
            output = model(images.to(device=device, dtype=torch.float16))
            rows.append(output.pooler_output.float().cpu())
            if step % 25 == 0:
                print(
                    json.dumps(
                        {"encoded_batches": step, "total_batches": len(loader)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    features = torch.cat(rows).contiguous()
    if features.shape != (len(paths), 1152) or not bool(torch.isfinite(features).all()):
        raise ValueError("SigLIP2 feature authority differs")
    return features


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--baseline-features", type=Path, required=True)
    parser.add_argument("--dataset-archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--execute-replication", action="store_true", required=True)
    return parser.parse_args(arguments)


def _feature_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    model_file = args.model_snapshot / "model.safetensors"
    config_file = args.model_snapshot / "config.json"
    processor_file = args.model_snapshot / "preprocessor_config.json"
    if (
        args.output.exists()
        or args.model_snapshot.name != MODEL_REVISION
        or args.batch_size != 32
        or args.dataset_archive.stat().st_size != CUB_ARCHIVE_BYTES
        or sha256(args.dataset_archive) != CUB_ARCHIVE_SHA256
        or sha256(model_file) != MODEL_SHA256
        or sha256(config_file) != CONFIG_SHA256
        or sha256(processor_file) != PROCESSOR_SHA256
        or sha256(args.baseline_features) != BASELINE_FEATURE_SHA256
        or sha256(args.preregistration) != args.preregistration_sha256
    ):
        raise ValueError("SigLIP2 CUB replication input authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration.get("schema") != "sfora-siglip2-so400m-cub-replication-v1"
        or preregistration.get("script_sha256") != sha256(Path(__file__))
    ):
        raise ValueError("SigLIP2 CUB preregistration authority differs")
    validate_processor_authority(json.loads(processor_file.read_text()))

    with np.load(args.baseline_features, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if (
            metadata.get("schema") != "scratch-cub-unicom-class-disjoint-v1"
            or metadata.get("counts") != {"fit": 2_997, "evaluation": 2_857}
            or metadata.get("dimensions") != 768
        ):
            raise ValueError("CUB baseline metadata differs")
        baseline_fit = torch.from_numpy(
            np.ascontiguousarray(archive["fit_embeddings"], dtype=np.float32)
        )
        fit_labels = torch.from_numpy(
            np.ascontiguousarray(archive["fit_labels"], dtype=np.int64)
        )
        baseline_evaluation = torch.from_numpy(
            np.ascontiguousarray(archive["evaluation_embeddings"], dtype=np.float32)
        )
        evaluation_labels = torch.from_numpy(
            np.ascontiguousarray(archive["evaluation_labels"], dtype=np.int64)
        )
    fit_paths, expected_fit_labels, evaluation_paths, expected_evaluation_labels = (
        select_cub_protocol_rows(args.dataset_root, metadata)
    )
    if (
        baseline_fit.shape != (2_997, 768)
        or baseline_evaluation.shape != (2_857, 768)
        or not np.array_equal(fit_labels.numpy(), expected_fit_labels)
        or not np.array_equal(evaluation_labels.numpy(), expected_evaluation_labels)
    ):
        raise ValueError("CUB baseline row authority differs")
    extraction_sha256 = verify_selected_extraction(
        args.dataset_archive,
        args.dataset_root,
        fit_paths + evaluation_paths,
    )

    configure_deterministic_similarity_runtime(17, cpu_threads=8)
    inference_backend = enable_repeatable_fused_inference()
    from transformers import AutoImageProcessor, SiglipVisionModel

    device = torch.device("cuda")
    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True
    )
    model = SiglipVisionModel.from_pretrained(
        args.model_snapshot,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != 428_225_600:
        raise ValueError("SigLIP2 parameter authority differs")

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    candidate_fit = _encode_paths(
        model, fit_paths, processor, device=device, batch_size=args.batch_size
    )
    candidate_evaluation = _encode_paths(
        model, evaluation_paths, processor, device=device, batch_size=args.batch_size
    )
    baseline = _fit_score(
        baseline_fit, fit_labels, baseline_evaluation, evaluation_labels, device=device
    )
    candidate = _fit_score(
        candidate_fit, fit_labels, candidate_evaluation, evaluation_labels, device=device
    )
    interval = paired_class_bootstrap(
        candidate["per_query_ap"],
        baseline["per_query_ap"],
        evaluation_labels.numpy(),
    )
    decision = classify_candidate(baseline, candidate, interval)
    decision["baseline_authority"] = True
    del baseline["per_query_ap"]
    del candidate["per_query_ap"]

    result = {
        "schema": "sfora-siglip2-so400m-cub-replication-result-v1",
        "claim_eligible": False,
        "dataset": "cub-200-2011-sha256-class-disjoint-fit-evaluation",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "dataset_archive_sha256": CUB_ARCHIVE_SHA256,
        "dataset_selected_extraction_sha256": extraction_sha256,
        "baseline_feature_sha256": BASELINE_FEATURE_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "config_sha256": CONFIG_SHA256,
        "processor_sha256": PROCESSOR_SHA256,
        "parameter_count": parameter_count,
        "inference_backend": inference_backend,
        "candidate_feature_sha256": {
            "fit": _feature_sha256(candidate_fit),
            "evaluation": _feature_sha256(candidate_evaluation),
        },
        "preregistration_sha256": args.preregistration_sha256,
        "script_sha256": sha256(Path(__file__)),
        "compact_representation": {
            "kind": "power-whitening-int8-128-plus-f16-inverse-norm",
            "bytes_per_item": 130,
            "alpha": 0.75,
            "regularization": 1.0,
        },
        "rows": {"fit": 2_997, "evaluation": 2_857},
        "baseline": baseline,
        "candidate": candidate,
        "paired_class_bootstrap_map_delta_95": interval,
        "decision": decision,
        "next_action": "profile-and-integrate" if decision["passed"] else "close-family",
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    print(json.dumps({"output": str(args.output), "decision": decision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
