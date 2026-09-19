#!/usr/bin/env python3
"""One-shot Stanford Dogs class-disjoint CLD transfer gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.nn import functional as F

from scratch_cub_local_metric_oracle import file_sha256
from scratch_sop_direct_head_guard import (
    evaluate_codes,
    normalized_within_class_covariance,
)
from sfora.compact_metric import CompactMetricConfig, fit_compact_metric_projection


UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
CHECKPOINT_SHA256 = "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
CUTOFFS = (1, 10, 100)


def unpack_mat_strings(path: Path) -> tuple[list[str], np.ndarray]:
    value = loadmat(path)
    paths = [str(item[0]) for item in value["file_list"].reshape(-1)]
    labels = value["labels"].reshape(-1).astype(np.int64) - 1
    if len(paths) != len(labels):
        raise ValueError("Stanford Dogs list length differs")
    return paths, labels


def encode_paths(paths: list[Path], checkout: Path, checkpoint: Path) -> np.ndarray:
    package_root = (checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    if Path(unicom.__file__).resolve().parent != package_root / "unicom":
        raise ValueError("UNICOM import authority differs")
    model, transform = unicom.load("ViT-L/14@336px", download_root=str(checkpoint.parent))
    model = model.cuda().eval()
    parts: list[np.ndarray] = []
    for start in range(0, len(paths), 64):
        tensors = []
        for path in paths[start : start + 64]:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        with torch.inference_mode():
            output = model(torch.stack(tensors).cuda())
        parts.append(output.float().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


def load_or_extract(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if args.features.exists():
        archive = np.load(args.features, allow_pickle=False)
        return (
            archive["fit_embeddings"],
            archive["fit_labels"],
            archive["evaluation_embeddings"],
            archive["evaluation_labels"],
            json.loads(str(archive["metadata_json"].item())),
        )
    revision = subprocess.run(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != UNICOM_REVISION or file_sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("teacher authority differs")
    train_paths, train_labels = unpack_mat_strings(args.root / "train_list.mat")
    test_paths, test_labels = unpack_mat_strings(args.root / "test_list.mat")
    if len(train_paths) != 12_000 or len(test_paths) != 8_580:
        raise ValueError("Stanford Dogs row authority differs")
    names = sorted({value.split("/", 1)[0] for value in train_paths + test_paths})
    if len(names) != 120:
        raise ValueError("Stanford Dogs breed authority differs")
    ordered = sorted(names, key=lambda name: (hashlib.sha256(name.encode()).hexdigest(), name))
    fit_names = set(ordered[:60])
    evaluation_names = set(ordered[60:])
    fit_rows = [i for i, value in enumerate(train_paths) if value.split("/", 1)[0] in fit_names]
    evaluation_rows = [i for i, value in enumerate(test_paths) if value.split("/", 1)[0] in evaluation_names]
    fit_files = [args.root / "Images" / train_paths[i] for i in fit_rows]
    evaluation_files = [args.root / "Images" / test_paths[i] for i in evaluation_rows]
    combined = encode_paths(fit_files + evaluation_files, args.checkout, args.checkpoint)
    fit_embeddings = combined[: len(fit_files)]
    evaluation_embeddings = combined[len(fit_files) :]
    fit_labels = train_labels[fit_rows]
    evaluation_labels = test_labels[evaluation_rows]
    metadata: dict[str, object] = {
        "schema": "scratch-stanford-dogs-sha-class-disjoint-v1",
        "claim_eligible": False,
        "unicom_revision": revision,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "images_tar_sha256": file_sha256(args.root / "images.tar"),
        "lists_tar_sha256": file_sha256(args.root / "lists.tar"),
        "fit_classes": ordered[:60],
        "evaluation_classes": ordered[60:],
        "fit_rows": len(fit_rows),
        "evaluation_rows": len(evaluation_rows),
    }
    np.savez(
        args.features,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        fit_embeddings=fit_embeddings,
        fit_labels=fit_labels,
        evaluation_embeddings=evaluation_embeddings,
        evaluation_labels=evaluation_labels,
    )
    return fit_embeddings, fit_labels, evaluation_embeddings, evaluation_labels, metadata


def packed(values: torch.Tensor) -> torch.Tensor:
    return torch.round(F.normalize(values, dim=1) * 127.0).clamp(-127, 127).to(torch.int8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    fit_np, fit_labels, evaluation_np, evaluation_labels, metadata = load_or_extract(args)
    fit = F.normalize(torch.from_numpy(fit_np).float(), dim=1)
    evaluation = F.normalize(torch.from_numpy(evaluation_np).float(), dim=1)
    fitted = fit_compact_metric_projection(
        fit,
        torch.from_numpy(fit_labels),
        config=CompactMetricConfig(),
        device=torch.device("cuda"),
    )
    fit_codes = fitted.encoder.encode(fit)
    evaluation_projected = fitted.encoder.transform(evaluation)
    evaluation_codes = fitted.encoder.encode(evaluation)
    prior = normalized_within_class_covariance(fit_codes, fit_labels)
    trained = evaluate_codes(
        evaluation_codes,
        evaluation_labels,
        protected_cutoffs=CUTOFFS,
        covariance_prior=prior,
    )
    identity = evaluate_codes(
        evaluation_codes,
        evaluation_labels,
        protected_cutoffs=CUTOFFS,
    )
    cholesky = torch.linalg.cholesky(prior)
    whitened = torch.linalg.solve_triangular(
        cholesky, evaluation_projected.cuda().T, upper=False
    ).T
    global_wccn = evaluate_codes(
        packed(whitened),
        evaluation_labels,
        protected_cutoffs=CUTOFFS,
    )
    target = trained["cld_dba_map_at_r"]
    controls = {
        "normalized_dba": trained["normalized_dba_map_at_r"],
        "identity_prior_cld_dba": identity["cld_dba_map_at_r"],
        "positive_only": trained["positive_only_map_at_r"],
        "global_prior_dba": trained["prior_dba_map_at_r"],
        "global_wccn_dba": global_wccn["normalized_dba_map_at_r"],
    }
    recall_equal = all(
        trained["cld_dba_recall"][str(k)] == trained["normalized_dba_recall"][str(k)]
        for k in CUTOFFS
    )
    payload = {
        "schema": "scratch-stanford-dogs-cld-gate-v1",
        "claim_eligible": False,
        "metadata": metadata,
        "features_sha256": file_sha256(args.features),
        "training": {
            "config": asdict(CompactMetricConfig()),
            "schedule_sha256": fitted.schedule_sha256,
            "parameter_sha256": fitted.parameter_sha256,
        },
        "protected_cutoffs": list(CUTOFFS),
        "target_map_at_r": target,
        "controls": controls,
        "target_minus_dba_map_at_r": target - controls["normalized_dba"],
        "target_recall": trained["cld_dba_recall"],
        "recall_equal": recall_equal,
        "passed": target - controls["normalized_dba"] >= 0.002
        and all(target > value for value in controls.values())
        and recall_equal,
        "elapsed_seconds": time.monotonic() - started,
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    args.output.write_text(wire)
    print(wire, end="")


if __name__ == "__main__":
    main()
