#!/usr/bin/env python3
"""Artifact-only retrieval-causality screen for Pass198 BSIR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


TAIL_START = 14_080
QUERY_COUNT = 14_218
TAIL_COUNT = QUERY_COUNT - TAIL_START
UNIT_TOLERANCE = 2.0e-5


def _unit_rows(rows: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1.0e-12) or not np.allclose(
        norms, 1.0, atol=UNIT_TOLERANCE, rtol=UNIT_TOLERANCE
    ):
        raise ValueError(f"{name} must contain unit rows")
    return array / norms[:, None]


def compare_tail_retrieval(
    canonical_query: np.ndarray,
    legacy_query: np.ndarray,
    query_labels: np.ndarray,
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    row_offset: int,
) -> dict[str, Any]:
    """Compare two paired query realizations against one immutable gallery."""
    canonical = _unit_rows(canonical_query, name="canonical_query")
    legacy = _unit_rows(legacy_query, name="legacy_query")
    support = _unit_rows(gallery, name="gallery")
    labels = np.asarray(query_labels, dtype=np.int64)
    support_labels = np.asarray(gallery_labels, dtype=np.int64)
    if canonical.shape != legacy.shape or labels.shape != (len(canonical),):
        raise ValueError("paired query arrays and labels must align")
    if support.shape[1] != canonical.shape[1] or support_labels.shape != (len(support),):
        raise ValueError("gallery arrays must align with query dimension")
    if len(support) < 2:
        raise ValueError("at least two gallery rows are required")

    canonical_scores = canonical @ support.T
    legacy_scores = legacy @ support.T
    canonical_order = np.argsort(-canonical_scores, axis=1, kind="stable")[:, :2]
    legacy_nearest = np.argmax(legacy_scores, axis=1)
    canonical_nearest = canonical_order[:, 0]
    canonical_correct = support_labels[canonical_nearest] == labels
    legacy_correct = support_labels[legacy_nearest] == labels
    delta = legacy - canonical
    delta_l2 = np.linalg.norm(delta, axis=1)
    dot = np.clip(np.sum(canonical * legacy, axis=1), -1.0, 1.0)
    angular = np.arccos(dot)
    top1 = canonical_scores[np.arange(len(canonical)), canonical_order[:, 0]]
    top2 = canonical_scores[np.arange(len(canonical)), canonical_order[:, 1]]
    margins = top1 - top2

    rows: list[dict[str, Any]] = []
    for index in range(len(canonical)):
        rows.append(
            {
                "query_row": int(row_offset + index),
                "query_label": int(labels[index]),
                "descriptor_l2_drift": float(delta_l2[index]),
                "descriptor_angular_drift": float(angular[index]),
                "canonical_nearest_gallery_row": int(canonical_nearest[index]),
                "legacy_nearest_gallery_row": int(legacy_nearest[index]),
                "canonical_nearest_identity": int(support_labels[canonical_nearest[index]]),
                "legacy_nearest_identity": int(support_labels[legacy_nearest[index]]),
                "canonical_correct": bool(canonical_correct[index]),
                "legacy_correct": bool(legacy_correct[index]),
                "canonical_top1_top2_margin": float(margins[index]),
                "two_l2_stability_bound": float(2.0 * delta_l2[index]),
                "stable_by_two_l2_bound": bool(margins[index] > 2.0 * delta_l2[index]),
            }
        )

    identity_flips = support_labels[canonical_nearest] != support_labels[legacy_nearest]
    correct_to_wrong = canonical_correct & ~legacy_correct
    wrong_to_correct = ~canonical_correct & legacy_correct
    return {
        "rows": rows,
        "nearest_identity_flips": int(identity_flips.sum()),
        "correct_to_wrong": int(correct_to_wrong.sum()),
        "wrong_to_correct": int(wrong_to_correct.sum()),
        "absolute_correctness_changes": int((canonical_correct != legacy_correct).sum()),
        "net_correctness_change": int(legacy_correct.sum() - canonical_correct.sum()),
        "stability_certified_rows": int(np.sum(margins > 2.0 * delta_l2)),
        "max_descriptor_l2_drift": float(delta_l2.max(initial=0.0)),
        "max_descriptor_angular_drift": float(angular.max(initial=0.0)),
    }


def stage_a_verdict(absolute_changes: list[int]) -> dict[str, Any]:
    if len(absolute_changes) != 4 or any(value < 0 for value in absolute_changes):
        raise ValueError("verdict requires four nonnegative seed counts")
    qualifying = sum(value >= 3 for value in absolute_changes)
    pooled = sum(absolute_changes)
    if qualifying >= 2 and pooled >= 12:
        status = "PASS_ONWARD"
    elif qualifying < 2 and pooled < 12:
        status = "FAIL"
    else:
        status = "UNRESOLVED"
    return {
        "stage_a": status,
        "absolute_correctness_changes_by_seed": absolute_changes,
        "seeds_with_at_least_three_changes": qualifying,
        "pooled_absolute_correctness_changes": pooled,
        "pass_requires_two_qualifying_seeds": qualifying >= 2,
        "pass_requires_twelve_pooled_changes": pooled >= 12,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_seed(entry: dict[str, dict[str, str]], *, seed: int) -> dict[str, Any]:
    import torch
    from diagnose_pass159_cotangent_stage_a import (
        _canonical_query_gallery_recall_at_1,
        _load_final_pack,
        _manifest_paths,
        _reconstruct_head,
        load_bound_seed,
    )

    bound = load_bound_seed(entry, seed=seed)
    paths = _manifest_paths(entry)
    checkpoint = torch.load(paths["checkpoint_pt"], map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    weight = state["model.embedding.weight"].detach().cpu().numpy()
    bias = state["model.embedding.bias"].detach().cpu().numpy()
    query_pack = _load_final_pack(
        paths["query_npz"],
        split="query",
        checkpoint_digest=entry["checkpoint_pt"]["sha256"],
        report_digest=entry["report_json"]["sha256"],
    )
    gallery_pack = _load_final_pack(
        paths["gallery_npz"],
        split="gallery",
        checkpoint_digest=entry["checkpoint_pt"]["sha256"],
        report_digest=entry["report_json"]["sha256"],
    )
    with np.load(paths["prehead_npz"], allow_pickle=False) as archive:
        legacy, _ = _reconstruct_head(
            np.asarray(archive["query"]), weight, bias, split="query"
        )
        legacy_labels = np.asarray(archive["query_labels"], dtype=np.int64)

    canonical = np.asarray(query_pack["embeddings"], dtype=np.float32)
    query_labels = np.asarray(query_pack["labels"], dtype=np.int64)
    gallery = np.asarray(gallery_pack["embeddings"], dtype=np.float32)
    gallery_labels = np.asarray(gallery_pack["labels"], dtype=np.int64)
    if canonical.shape != (QUERY_COUNT, 512) or legacy.shape != canonical.shape:
        raise ValueError(f"seed {seed} query shape differs from frozen In-Shop shape")
    if not np.array_equal(legacy_labels, query_labels):
        raise ValueError(f"seed {seed} legacy and canonical query labels differ")
    prefix_max = float(np.max(np.abs(legacy[:TAIL_START] - canonical[:TAIL_START])))
    if prefix_max > UNIT_TOLERANCE:
        raise ValueError(f"seed {seed} has material pre-tail drift: {prefix_max}")

    comparison = compare_tail_retrieval(
        canonical[TAIL_START:],
        legacy[TAIL_START:],
        query_labels[TAIL_START:],
        gallery,
        gallery_labels,
        row_offset=TAIL_START,
    )
    for index, row in enumerate(comparison["rows"]):
        row["seed"] = int(seed)
        row["example_id"] = str(np.asarray(query_pack["example_ids"])[TAIL_START + index])

    modified = canonical.copy()
    modified[TAIL_START:] = legacy[TAIL_START:]
    modified_r1 = _canonical_query_gallery_recall_at_1(
        modified, query_labels, gallery, gallery_labels
    )
    canonical_r1 = _canonical_query_gallery_recall_at_1(
        canonical, query_labels, gallery, gallery_labels
    )
    if canonical_r1 != bound.official_recall_at_1:
        raise ValueError(f"seed {seed} canonical R@1 lost artifact binding")
    expected_modified = canonical_r1 + comparison["net_correctness_change"] / QUERY_COUNT
    if modified_r1 != expected_modified:
        raise ValueError(f"seed {seed} modified R@1 disagrees with paired count")

    return {
        "seed": int(seed),
        "artifact_binding": bound.artifact_binding,
        "prefix_max_abs_difference": prefix_max,
        "tail_max_abs_coordinate_difference": float(
            np.max(np.abs(legacy[TAIL_START:] - canonical[TAIL_START:]))
        ),
        "canonical_recall_at_1": float(canonical_r1),
        "legacy_tail_replaced_recall_at_1": float(modified_r1),
        "recall_at_1_change_points": float(100.0 * (modified_r1 - canonical_r1)),
        **comparison,
    }


def run_manifest(manifest_path: Path, prereg_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or set(manifest.get("seeds", {})) != {
        "0", "1", "2", "3"
    }:
        raise ValueError("manifest must bind exactly seeds 0 through 3")
    per_seed = [
        _run_seed(manifest["seeds"][str(seed)], seed=seed) for seed in range(4)
    ]
    verdict = stage_a_verdict(
        [int(item["absolute_correctness_changes"]) for item in per_seed]
    )
    all_rows = [row for item in per_seed for row in item.pop("rows")]
    return {
        "schema_version": "pass198-bsir-stage-a-v1",
        "preregistration": {
            "document_path": str(prereg_path),
            "document_sha256": _sha256(prereg_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "diagnostic_source_sha256": _sha256(Path(__file__)),
            "tail_start": TAIL_START,
            "tail_count": TAIL_COUNT,
            "uses_test_data": "artifact_bound_retrieval_causality_only",
        },
        "per_seed": per_seed,
        "paired_rows": all_rows,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_manifest(args.manifest, args.preregistration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
