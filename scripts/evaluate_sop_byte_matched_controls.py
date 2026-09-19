#!/usr/bin/env python3
"""Compare 128-byte-per-item SOP codes from different constructions, one split.

Every arm is scored on the same official Stanford Online Products test rows,
through the same symmetric packed int8 evaluator, at the same persistent budget
of 128 bytes per item. Fitting uses official train rows only; test rows are
evaluation-only. The official test split has already been observed by this
project, so the receipt is `claim_eligible=false`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

_ERROR = "byte-matched control authority differs"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _fit_pca(train_rows: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mean and top-`width` principal directions of the train rows."""

    centre = train_rows.mean(dim=0)
    centred = train_rows - centre
    # Full SVD on the covariance side keeps this deterministic and exact.
    covariance = (centred.T @ centred) / (centred.shape[0] - 1)
    values, vectors = torch.linalg.eigh(covariance.double())
    order = torch.argsort(values, descending=True)[:width]
    return centre, vectors[:, order].T.float().contiguous()


def _load_head(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    keys = set(state)
    if keys == {"weight", "bias"}:
        return state["weight"].contiguous(), state["bias"].contiguous()
    if keys == {"weight", "base_head_weight", "base_head_bias"}:
        return (
            (state["weight"] @ state["base_head_weight"]).contiguous(),
            (state["weight"] @ state["base_head_bias"]).contiguous(),
        )
    raise ValueError(_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--teacher-snapshot", type=Path, required=True)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--head", action="append", nargs=2, default=[], metavar=("NAME", "PATH"))
    parser.add_argument(
        "--reconstruction", action="append", nargs=2, default=[], metavar=("NAME", "NPZ")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-official-evaluation", action="store_true", required=True)
    args = parser.parse_args()
    if not args.execute_official_evaluation or args.output.exists():
        raise ValueError(_ERROR)

    import numpy as np
    from probe_sop_relational_linear import load_paired_archives, score_symmetric

    from sfora.deterministic_similarity_runtime import (
        configure_deterministic_similarity_runtime,
    )
    from sfora.joint_relational_compaction import pack_int8_unit_embeddings

    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    pair = load_paired_archives(
        args.source_snapshot, args.source_sha256, args.teacher_snapshot, args.teacher_sha256
    )
    teacher_train = pair["teacher_train"].float().contiguous()
    teacher_test = pair["teacher_test"].float().contiguous()
    labels = pair["test_labels"]
    device = torch.device("cuda")
    candidate_width = max(Counter(labels).values()) - 1

    def score(codes: torch.Tensor, bytes_per_item: int) -> dict[str, Any]:
        floating = score_symmetric(codes, labels, candidate_width=candidate_width, device=device)
        packed = score_symmetric(
            pack_int8_unit_embeddings(codes),
            labels,
            candidate_width=candidate_width,
            device=device,
        )
        return {
            "bytes_per_item": bytes_per_item,
            "float_map_at_r": float(floating["map_at_r"]),
            "float_r1": float(floating["r1"]),
            "packed_map_at_r": float(packed["map_at_r"]),
            "packed_r1": float(packed["r1"]),
        }

    arms: dict[str, Any] = {}
    with torch.inference_mode():
        normalized_train = F.normalize(teacher_train, dim=1).contiguous()
        normalized_test = F.normalize(teacher_test, dim=1).contiguous()

        # Baseline: PCA fitted on official train rows only, then int8 at 128 bytes.
        centre, components = _fit_pca(normalized_train, 128)
        projected = (normalized_test - centre) @ components.T
        arms["pca128_int8"] = score(F.normalize(projected, dim=1).contiguous(), 128)

        for name, path_text in args.head:
            weight, bias = _load_head(Path(path_text))
            codes = F.normalize(F.linear(normalized_test, weight, bias), dim=1).contiguous()
            arms[name] = {**score(codes, 128), "checkpoint_sha256": _file_sha256(Path(path_text))}

        for name, npz_text in args.reconstruction:
            payload = np.load(npz_text)
            rows = torch.from_numpy(np.ascontiguousarray(payload["test"], dtype="float32"))
            if rows.shape != teacher_test.shape:
                raise ValueError(_ERROR)
            arms[name] = {
                **score(F.normalize(rows, dim=1).contiguous(), int(payload["bytes_per_item"])),
                "archive_sha256": _file_sha256(Path(npz_text)),
            }

        arms["teacher_float768"] = score(normalized_test, 768 * 4)

    result = {
        "claim_eligible": False,
        "arms": arms,
        "dataset": "stanford-online-products-official-test",
        "fitting_rows": "official-train-only",
        "inputs": {
            "source_snapshot_sha256": args.source_sha256,
            "teacher_snapshot_sha256": args.teacher_sha256,
        },
        "official_test_touched": True,
        "rows": int(teacher_test.shape[0]),
        "schema": "sfora-sop-byte-matched-controls-v1",
        "split_status": "already-observed-development-surface",
    }
    wire = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    args.output.write_text(wire)
    print(wire, end="")


if __name__ == "__main__":
    sys.exit(main())
