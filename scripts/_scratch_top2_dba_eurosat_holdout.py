#!/usr/bin/env python3
"""Untouched EuroSAT holdout for the frozen top-2 DBA transform."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _scratch_same_teacher_ladder import _score
from _scratch_sop_direct256_int4 import fit_pca_affine
from _scratch_top2_dba_gate import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    MINIMUM_MAP_EFFECT,
    _top2_dba_gallery,
    paired_query_bootstrap,
)
from PIL import Image
from torch.nn import functional as F
from torchvision.datasets import EuroSAT

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

EUROSAT_ARCHIVE_MD5 = "c8fa014336c82ac7804f0398fcb19387"
EUROSAT_ROWS = 27_000
UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
CHECKPOINT_SHA256 = "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
EXPECTED_CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)


def _digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _partition_rows(
    samples: list[tuple[Path, int]],
) -> dict[str, list[tuple[Path, int]]]:
    if not samples or len(samples) != len(set(samples)):
        raise ValueError("EuroSAT row authority differs")
    grouped: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for path, label in samples:
        if not isinstance(path, Path) or type(label) is not int:
            raise ValueError("EuroSAT row authority differs")
        grouped[label].append((path, label))
    result = {"fit": [], "query": [], "gallery": []}
    for label in sorted(grouped):
        rows = sorted(
            grouped[label],
            key=lambda row: (
                hashlib.sha256(
                    f"sfora-top2-dba-eurosat-v1:{row[0].as_posix()}".encode()
                ).digest(),
                row[0].as_posix(),
            ),
        )
        fit_stop = 2 * len(rows) // 5
        query_stop = fit_stop + 3 * len(rows) // 10
        result["fit"].extend(rows[:fit_stop])
        result["query"].extend(rows[fit_stop:query_stop])
        result["gallery"].extend(rows[query_stop:])
    return result


def _quality_decision(contrast: dict[str, Any], *, recall_delta: float) -> dict[str, Any]:
    passed = bool(contrast.get("survives") is True and recall_delta >= 0.0)
    return {
        "passed": passed,
        "primary_map_gate_passed": bool(contrast.get("survives") is True),
        "recall_nonregression_passed": bool(recall_delta >= 0.0),
        "next": "integrate and profile" if passed else "close top-2 DBA",
    }


def _load_model(checkout: Path, checkpoint: Path) -> tuple[torch.nn.Module, object]:
    package_root = (checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    model, transform = unicom.load("ViT-L/14@336px", download_root=str(checkpoint.parent))
    return model.cuda().eval(), transform


def _encode(
    model: torch.nn.Module,
    transform: object,
    root: Path,
    rows: list[tuple[Path, int]],
) -> tuple[torch.Tensor, tuple[int, ...]]:
    output: np.ndarray | None = None
    labels: list[int] = []
    for start in range(0, len(rows), 64):
        batch = rows[start : start + 64]
        images = []
        for relative, label in batch:
            with Image.open(root / relative) as opened:
                images.append(transform(opened.convert("RGB")))
            labels.append(label)
        with torch.inference_mode():
            values = F.normalize(model(torch.stack(images).cuda()).float(), dim=1)
        host = values.cpu().numpy().copy()
        if output is None:
            output = np.empty((len(rows), host.shape[1]), dtype=np.float32)
        output[start : start + len(host)] = host
        if start % 1024 == 0:
            print(json.dumps({"encoded": start + len(host), "rows": len(rows)}), flush=True)
    if output is None or len(labels) != len(rows):
        raise ValueError("EuroSAT encoding authority differs")
    return torch.from_numpy(output), tuple(labels)


def _sample_manifest_sha256(samples: list[tuple[Path, int]]) -> str:
    wire = "".join(f"{path.as_posix()}\t{label}\n" for path, label in samples).encode()
    return hashlib.sha256(wire).hexdigest()


def _evaluate(
    root: Path,
    checkout: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    dataset = EuroSAT(root, download=False)
    if len(dataset) != EUROSAT_ROWS or tuple(dataset.classes) != EXPECTED_CLASSES:
        raise ValueError("EuroSAT dataset authority differs")
    data_root = Path(dataset._data_folder)
    samples = [
        (Path(path).relative_to(data_root), int(label))
        for path, label in dataset.samples
    ]
    partitions = _partition_rows(samples)
    if {name: len(rows) for name, rows in partitions.items()} != {
        "fit": 10_800,
        "query": 8_100,
        "gallery": 8_100,
    }:
        raise ValueError("EuroSAT partition authority differs")
    model, transform = _load_model(checkout, checkpoint)
    fit, _fit_labels = _encode(model, transform, data_root, partitions["fit"])
    query, query_labels = _encode(model, transform, data_root, partitions["query"])
    gallery, gallery_labels = _encode(model, transform, data_root, partitions["gallery"])
    del model
    torch.cuda.empty_cache()

    weight, bias = fit_pca_affine(fit, output_dimensions=128)
    query_packed = pack_int8_unit_embeddings(F.normalize(F.linear(query, weight, bias), dim=1))
    gallery_packed = pack_int8_unit_embeddings(
        F.normalize(F.linear(gallery, weight, bias), dim=1)
    )
    device = torch.device("cuda")
    baseline = _score(
        query_packed,
        gallery_packed,
        query_labels,
        gallery_labels,
        same_rows=False,
        device=device,
    )
    candidate_gallery, neighbours = _top2_dba_gallery(gallery_packed, device=device)
    candidate = _score(
        query_packed,
        candidate_gallery,
        query_labels,
        gallery_labels,
        same_rows=False,
        device=device,
    )
    contrast = paired_query_bootstrap(
        np.asarray(candidate["per_query_ap"], dtype=np.float64),
        np.asarray(baseline["per_query_ap"], dtype=np.float64),
        seed=BOOTSTRAP_SEED,
        replicates=BOOTSTRAP_REPLICATES,
    )
    recall_delta = float(candidate["recall_at_1"] - baseline["recall_at_1"])
    return {
        "manifest_sha256": _sample_manifest_sha256(samples),
        "split": {name: len(rows) for name, rows in partitions.items()},
        "baseline": baseline,
        "top2_dba": candidate,
        "contrast": contrast,
        "recall_at_1_delta": recall_delta,
        "decision": _quality_decision(contrast, recall_delta=recall_delta),
        "representation": {
            "baseline_bytes_per_item": query_packed.bytes_per_vector,
            "candidate_bytes_per_item": candidate_gallery.bytes_per_vector,
            "neighbours": 2,
            "formula": "normalize(self+mean(top2-nonself-packed-cosine-neighbours));repack-int8",
        },
        "neighbour_authority": {
            "shape": list(neighbours.shape),
            "self_matches": int(
                sum(row in values for row, values in enumerate(neighbours.tolist()))
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-untouched-holdout", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    archive = args.raw_root / "eurosat" / "EuroSAT.zip"
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if _digest(Path(__file__), "sha256") != args.script_sha256:
        raise ValueError("EuroSAT holdout script authority differs")
    if _digest(args.preregistration, "sha256") != args.preregistration_sha256:
        raise ValueError("EuroSAT holdout preregistration authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration.get("schema") != "scratch-top2-dba-eurosat-holdout-preregistration-v1"
        or preregistration.get("source_commit") != args.source_commit
        or preregistration.get("script_sha256") != args.script_sha256
        or preregistration.get("dataset_archive_md5") != EUROSAT_ARCHIVE_MD5
        or preregistration.get("decision")
        != {
            "map_estimand": "paired per-query AP@R difference",
            "minimum_map_effect": MINIMUM_MAP_EFFECT,
            "interval": "two-sided percentile 95%",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "requires_nonnegative_recall_at_1_delta": True,
        }
    ):
        raise ValueError("EuroSAT holdout preregistration contract differs")
    if not archive.is_file() or _digest(archive, "md5") != EUROSAT_ARCHIVE_MD5:
        raise ValueError("EuroSAT archive authority differs")
    revision = subprocess.run(
        ["git", "-C", str(args.unicom_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != UNICOM_REVISION or _digest(args.checkpoint, "sha256") != CHECKPOINT_SHA256:
        raise ValueError("UNICOM authority differs")
    configure_deterministic_similarity_runtime(0, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.monotonic()
    result = _evaluate(args.raw_root, args.unicom_checkout, args.checkpoint)
    receipt = {
        "schema": "scratch-top2-dba-eurosat-holdout-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "script_sha256": args.script_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "dataset_archive_md5": EUROSAT_ARCHIVE_MD5,
        "unicom_revision": UNICOM_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        **result,
        "elapsed_seconds": time.monotonic() - started,
    }
    wire = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
