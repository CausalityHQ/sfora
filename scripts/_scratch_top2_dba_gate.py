#!/usr/bin/env python3
"""Frozen top-2 database-augmentation gate on disjoint In-Shop retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _scratch_same_teacher_ladder import (
    _inshop,
    _score,
)
from _scratch_same_teacher_ladder import (
    paired_query_bootstrap as _paired_query_bootstrap,
)
from _scratch_sop_direct256_int4 import fit_pca_affine
from probe_sop_relational_linear import _lexicographic_candidates
from torch.nn import functional as F

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_921
MINIMUM_MAP_EFFECT = 0.005
EXPECTED_INSHOP_SHA256 = "05cd5901425210c06a3972f5a67acf41c961b2f4b59d6535744f3e3536d036ad"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_query_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    result = _paired_query_bootstrap(
        candidate,
        baseline,
        seed=seed,
        replicates=replicates,
    )
    result["minimum_map_effect"] = MINIMUM_MAP_EFFECT
    return result


def _decoded_unit(packed: PackedInt8Embeddings, device: torch.device) -> torch.Tensor:
    return packed.codes.to(device=device, dtype=torch.float32) * packed.inverse_norms.to(
        device=device, dtype=torch.float32
    )[:, None]


def _top2_dba_gallery(
    gallery: PackedInt8Embeddings,
    *,
    device: torch.device,
) -> tuple[PackedInt8Embeddings, torch.Tensor]:
    """Augment each packed gallery row with its two nearest non-self rows."""

    if gallery.codes.ndim != 2 or len(gallery.codes) < 3:
        raise ValueError("top-2 DBA gallery authority differs")
    unit = _decoded_unit(gallery, device)
    neighbours = torch.empty((len(unit), 2), dtype=torch.int64, device=device)
    with torch.inference_mode():
        for start in range(0, len(unit), 256):
            stop = min(start + 256, len(unit))
            scores = unit[start:stop] @ unit.T
            scores[
                torch.arange(stop - start, device=device),
                torch.arange(start, stop, device=device),
            ] = -torch.inf
            neighbours[start:stop] = _lexicographic_candidates(scores, 2)
        augmented = F.normalize(unit + unit[neighbours].mean(dim=1), dim=1)
    packed_augmented = pack_int8_unit_embeddings(augmented.cpu())
    return packed_augmented, neighbours.cpu()


def _evaluate(path: Path) -> dict[str, Any]:
    fit, _fit_labels, query, query_labels, gallery, gallery_labels, same_rows = _inshop(path)
    if same_rows:
        raise ValueError("top-2 DBA requires disjoint query/gallery")
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
    augmented_gallery, neighbours = _top2_dba_gallery(gallery_packed, device=device)
    candidate = _score(
        query_packed,
        augmented_gallery,
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
    return {
        "split": {
            "fit_rows": len(fit),
            "query_rows": len(query),
            "gallery_rows": len(gallery),
            "same_rows": False,
        },
        "baseline": baseline,
        "top2_dba": candidate,
        "contrast": contrast,
        "representation": {
            "baseline_bytes_per_item": query_packed.bytes_per_vector,
            "candidate_bytes_per_item": augmented_gallery.bytes_per_vector,
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
    parser.add_argument("--inshop", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-top2-dba-gate", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if _sha256(Path(__file__)) != args.script_sha256:
        raise ValueError("top-2 DBA script authority differs")
    if _sha256(args.preregistration) != args.preregistration_sha256:
        raise ValueError("top-2 DBA preregistration authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration.get("schema") != "scratch-top2-dba-gate-preregistration-v1"
        or preregistration.get("source_commit") != args.source_commit
        or preregistration.get("script_sha256") != args.script_sha256
        or preregistration.get("input_sha256") != EXPECTED_INSHOP_SHA256
        or preregistration.get("decision")
        != {
            "estimand": "paired per-query AP@R difference",
            "minimum_map_effect": MINIMUM_MAP_EFFECT,
            "interval": "two-sided percentile 95%",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "survives": "delta>=0.005 and lower95>0",
        }
    ):
        raise ValueError("top-2 DBA preregistration contract differs")
    if _sha256(args.inshop) != EXPECTED_INSHOP_SHA256:
        raise ValueError("In-Shop artifact authority differs")
    configure_deterministic_similarity_runtime(0, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.monotonic()
    result = _evaluate(args.inshop)
    receipt = {
        "schema": "scratch-top2-dba-gate-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "script_sha256": args.script_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "input_sha256": EXPECTED_INSHOP_SHA256,
        "dataset": "inshop-official-disjoint-query-gallery",
        **result,
        "decision": {
            "passed": result["contrast"]["survives"],
            "next": (
                "replicate once on untouched disjoint retrieval"
                if result["contrast"]["survives"]
                else "close top-2 DBA"
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    wire = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
