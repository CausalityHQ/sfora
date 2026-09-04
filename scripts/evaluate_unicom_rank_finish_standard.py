#!/usr/bin/env python3
"""Run the fixed seed-1 rank-finish standard-test release readout."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.unicom_inshop import parse_inshop_partition

METRIC_KEYS = (
    "map_at_r",
    "recall_at_1",
    "recall_at_10",
    "recall_at_20",
    "recall_at_30",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_metric(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"rank-finish standard {name} differs")
    return value


def classify_standard(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    """Apply the frozen final mAP and recall release gates."""

    if tuple(baseline) != METRIC_KEYS or tuple(candidate) != METRIC_KEYS:
        raise ValueError("rank-finish standard metric inventory differs")
    deltas = {
        key: _finite_metric(candidate[key], key) - _finite_metric(baseline[key], key)
        for key in METRIC_KEYS
    }
    passes = (
        deltas["map_at_r"] >= 0.003
        and deltas["recall_at_1"] >= -0.001
        and deltas["recall_at_10"] >= -0.001
    )
    return {"status": "RELEASE" if passes else "REJECT", "deltas": deltas}


def load_inference_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_seed: int,
    expected_source_commit: str,
    expected_parent_checkpoint_sha256: str,
) -> Mapping[str, torch.Tensor]:
    """Authenticate and load one inference-only rank-finish artifact."""

    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != expected_sha256
    ):
        raise ValueError("rank-finish inference bytes differ")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        type(value) is not dict
        or tuple(value)
        != (
            "schema",
            "finish_seed",
            "source_commit",
            "parent_checkpoint_sha256",
            "model",
        )
        or value["schema"] != "unicom-rank-finish-inference-v1"
        or type(value["finish_seed"]) is not int
        or value["finish_seed"] != expected_seed
        or value["source_commit"] != expected_source_commit
        or value["parent_checkpoint_sha256"]
        != expected_parent_checkpoint_sha256
        or type(value["model"]) is not dict
        or not value["model"]
        or any(
            type(name) is not str
            or type(tensor) is not torch.Tensor
            or not torch.isfinite(tensor).all()
            for name, tensor in value["model"].items()
        )
    ):
        raise ValueError("rank-finish inference authority differs")
    return value["model"]


def _load_trainer(repository: Path):
    path = repository / "scripts" / "train_unicom_inshop.py"
    specification = importlib.util.spec_from_file_location("rank_finish_trainer", path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _evaluate(trainer, model, query, gallery, transform, device) -> dict[str, object]:
    query_values, query_labels = trainer._encode_records(
        model, query, transform, device=device, batch_size=128, workers=4
    )
    gallery_values, gallery_labels = trainer._encode_records(
        model, gallery, transform, device=device, batch_size=128, workers=4
    )
    view = trainer.retrieval_view(
        query_values,
        gallery_values,
        query_labels,
        gallery_labels,
        coordinates=np.arange(512),
        normalize_before=True,
    )
    if view.average_precision is None:
        raise ValueError("rank-finish standard query evidence differs")
    metrics = {
        "map_at_r": float(view.map_at_r),
        "recall_at_1": float(view.recall[1]),
        "recall_at_10": float(view.recall[10]),
        "recall_at_20": float(view.recall[20]),
        "recall_at_30": float(view.recall[30]),
    }
    for key, value in metrics.items():
        _finite_metric(value, key)
    evidence = {
        "average_precision": [float(value) for value in view.average_precision],
        "top1_correct": [bool(value) for value in view.top1_correct],
    }
    if (
        len(evidence["average_precision"]) != len(query)
        or len(evidence["top1_correct"]) != len(query)
        or not math.isclose(
            math.fsum(evidence["average_precision"]) / len(query),
            metrics["map_at_r"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("rank-finish standard query evidence differs")
    return {"metrics": metrics, "query_evidence": evidence}


def canonical_result_bytes(result: object) -> bytes:
    if (
        type(result) is not dict
        or result.get("schema") != "unicom-rank-finish-standard-v1"
        or result.get("claim_eligible") is not False
    ):
        raise ValueError("rank-finish standard result differs")
    return (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--confirmation", required=True, type=Path)
    parser.add_argument("--confirmation-sha256", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-source-commit", required=True)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--partition-sha256", required=True)
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--official-checkpoint", required=True, type=Path)
    parser.add_argument("--official-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-standard-readout", action="store_true", required=True)
    return parser.parse_args(arguments)


def _require_sha(path: Path, expected: str, name: str) -> None:
    if (
        type(expected) is not str
        or len(expected) != 64
        or path.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != expected
    ):
        raise ValueError(f"rank-finish {name} differs")


def run(args: argparse.Namespace) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source != args.source_commit or dirty:
        raise ValueError("rank-finish standard source differs")
    for path, digest, name in (
        (args.confirmation, args.confirmation_sha256, "confirmation bytes"),
        (args.model, args.model_sha256, "model bytes"),
        (args.parent_checkpoint, args.parent_checkpoint_sha256, "parent bytes"),
        (args.official_checkpoint, args.official_checkpoint_sha256, "official bytes"),
        (
            args.dataset_root / "Eval" / "list_eval_partition.txt",
            args.partition_sha256,
            "partition bytes",
        ),
    ):
        _require_sha(path, digest, name)
    confirmation = json.loads(args.confirmation.read_bytes())
    if (
        type(confirmation) is not dict
        or confirmation.get("schema") != "unicom-rank-finish-confirmation-v1"
        or confirmation.get("claim_eligible") is not False
        or confirmation.get("status") != "CONFIRM"
    ):
        raise ValueError("rank-finish confirmation decision differs")
    seed_one = confirmation["decision"]["seeds"][1]
    if (
        seed_one.get("finish_seed") != 1
        or seed_one.get("model_artifact", {}).get("sha256") != args.model_sha256
    ):
        raise ValueError("rank-finish release candidate differs")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)

    trainer = _load_trainer(repository)
    records = parse_inshop_partition(args.dataset_root)
    query = tuple(record for record in records if record.split == "query")
    gallery = tuple(record for record in records if record.split == "gallery")
    if not query or not gallery:
        raise ValueError("rank-finish standard partition differs")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("rank-finish standard readout requires CUDA")
    model, transform = trainer._load_official_model(
        args.unicom_checkout, args.official_checkpoint
    )
    model = model.to(device)
    parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
    if type(parent) is not dict or type(parent.get("model")) is not dict:
        raise ValueError("rank-finish parent checkpoint differs")
    model.load_state_dict(parent["model"], strict=True)
    del parent
    gc.collect()
    torch.cuda.synchronize()
    started = time.perf_counter()
    baseline = _evaluate(trainer, model, query, gallery, transform, device)

    candidate_state = load_inference_checkpoint(
        args.model,
        expected_sha256=args.model_sha256,
        expected_seed=1,
        expected_source_commit=args.model_source_commit,
        expected_parent_checkpoint_sha256=args.parent_checkpoint_sha256,
    )
    model.load_state_dict(candidate_state, strict=True)
    del candidate_state
    gc.collect()
    candidate = _evaluate(trainer, model, query, gallery, transform, device)
    decision = classify_standard(baseline["metrics"], candidate["metrics"])
    result = {
        "schema": "unicom-rank-finish-standard-v1",
        "claim_eligible": False,
        "source_commit": source,
        "confirmation": {
            "path": str(args.confirmation.resolve()),
            "sha256": args.confirmation_sha256,
        },
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "source_commit": args.model_source_commit,
            "finish_seed": 1,
        },
        "parent_checkpoint_sha256": args.parent_checkpoint_sha256,
        "partition_sha256": args.partition_sha256,
        "baseline": baseline,
        "candidate": candidate,
        "decision": decision,
        "status": decision["status"],
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    payload = canonical_result_bytes(result)
    published = publish_bytes_noreplace(
        args.output,
        payload,
        validator=lambda observed: (
            None
            if observed == payload
            else (_ for _ in ()).throw(ValueError("standard result bytes differ"))
        ),
    )
    published.close()
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(arguments)
        result = run(args)
    except Exception as error:
        print(f"rank-finish standard readout failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
