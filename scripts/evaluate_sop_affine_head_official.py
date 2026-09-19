#!/usr/bin/env python3
"""Score deployed 768-to-128 affine heads once on the official SOP test rows.

The official SOP test split has already been observed by this project, so every
receipt this driver writes is `claim_eligible=false` and records that fact. It
exists to place already-trained heads on the same measured surface, not to
create a held-out claim.
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

_ERROR = "official affine-head authority differs"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_head(path: Path, expected_sha256: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the single deployed 768-to-128 affine map for either arm shape."""

    if _file_sha256(path) != expected_sha256:
        raise ValueError(_ERROR)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if type(state) is not dict:
        raise ValueError(_ERROR)
    keys = set(state)
    if keys == {"weight", "bias"}:
        weight, bias = state["weight"], state["bias"]
    elif keys == {"weight", "base_head_weight", "base_head_bias"}:
        # Fold the restricted adapter into the base head so both arms are scored
        # through one affine map, exactly as they deploy.
        adapter, base_weight, base_bias = (
            state["weight"],
            state["base_head_weight"],
            state["base_head_bias"],
        )
        weight = adapter @ base_weight
        bias = adapter @ base_bias
    else:
        raise ValueError(_ERROR)
    if (
        weight.dtype != torch.float32
        or weight.ndim != 2
        or weight.shape != (128, 768)
        or bias.shape != (128,)
        or not bool(torch.isfinite(weight).all())
        or not bool(torch.isfinite(bias).all())
    ):
        raise ValueError(_ERROR)
    return weight.contiguous(), bias.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--teacher-snapshot", type=Path, required=True)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument(
        "--head", action="append", nargs=3, required=True, metavar=("NAME", "PATH", "SHA256")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-official-evaluation", action="store_true", required=True)
    args = parser.parse_args()
    if not args.execute_official_evaluation or args.output.exists():
        raise ValueError(_ERROR)

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
    teacher_test = pair["teacher_test"].float().contiguous()
    labels = pair["test_labels"]
    device = torch.device("cuda")
    candidate_width = max(Counter(labels).values()) - 1

    def score(codes: torch.Tensor) -> dict[str, float]:
        floating = score_symmetric(codes, labels, candidate_width=candidate_width, device=device)
        packed = score_symmetric(
            pack_int8_unit_embeddings(codes),
            labels,
            candidate_width=candidate_width,
            device=device,
        )
        return {
            "float_map_at_r": float(floating["map_at_r"]),
            "float_r1": float(floating["r1"]),
            "packed_map_at_r": float(packed["map_at_r"]),
            "packed_r1": float(packed["r1"]),
        }

    with torch.inference_mode():
        normalized = F.normalize(teacher_test, dim=1).contiguous()
        heads: dict[str, Any] = {}
        for name, path_text, sha256 in args.head:
            weight, bias = _load_head(Path(path_text), sha256)
            codes = F.normalize(F.linear(normalized, weight, bias), dim=1).contiguous()
            heads[name] = {"checkpoint_sha256": sha256, "score": score(codes)}
        teacher_score = score(normalized)

    result = {
        "claim_eligible": False,
        "dataset": "stanford-online-products-official-test",
        "heads": heads,
        "inputs": {
            "source_snapshot_sha256": args.source_sha256,
            "teacher_snapshot_sha256": args.teacher_sha256,
        },
        "official_test_touched": True,
        "rows": int(teacher_test.shape[0]),
        "schema": "sfora-sop-official-affine-head-v1",
        "split_status": "already-observed-development-surface",
        "teacher": teacher_score,
    }
    wire = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    args.output.write_text(wire)
    print(wire, end="")


if __name__ == "__main__":
    sys.exit(main())
