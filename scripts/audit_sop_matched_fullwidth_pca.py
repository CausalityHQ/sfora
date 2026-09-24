"""Score fit-only PCA-128 deployment from an authenticated trained B/16 archive.

Use matching full-width and compact diagnostic checkpoints on the official SOP
TRAIN class-disjoint holdout. This script does not read official SOP test rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
import torch
from train_sop_compact_backbone import publish_file_noreplace, score_validation_features

from sfora.representation_ceiling import deterministic_class_partition, fit_centered_pca

SEED = 179019
MATCHED_STEP = 48_000
SOURCE_ARCHIVE_SHA = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
COMPACT_RECEIPT_SHA = "42350440edf3dcb972c6d2a9372a1adc75e6363319efc31a047815e81cea29c0"
MATCHED_FIELDS = (
    "schema",
    "arm",
    "recipe",
    "seed",
    "step",
    "total_updates",
    "schedule_sha256",
    "fit_row_indexes_sha256",
    "validation_row_indexes_sha256",
    "input_checkpoint_sha256",
    "features_archive_sha256",
    "sop_train_metadata_sha256",
    "validation_image_ids",
    "validation_labels",
)
SOURCE_RELATIVES = (
    "scripts/audit_sop_matched_fullwidth_pca.py",
    "scripts/train_sop_compact_backbone.py",
    "src/sfora/representation_ceiling.py",
    "src/sfora/joint_relational_compaction.py",
    "src/sfora/sop_evaluation.py",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().astype("<f4").tobytes()
    ).hexdigest()


def product_interval(labels: np.ndarray, values: np.ndarray) -> list[float]:
    _, inverse = np.unique(labels, return_inverse=True)
    classes = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=classes)
    sums = np.bincount(inverse, weights=values, minlength=classes)
    rng = np.random.default_rng(SEED)
    replicates = np.empty(10_000, dtype=np.float64)
    for index in range(len(replicates)):
        selected = rng.integers(0, classes, size=classes)
        replicates[index] = sums[selected].sum() / counts[selected].sum()
    return [float(value) for value in np.quantile(replicates, (0.025, 0.975))]


def verify_holdout(actual: dict[str, object], receipt: dict[str, object]) -> None:
    expected = cast(dict[str, dict[str, object]], receipt["validation"])
    for arm in ("float", "packed"):
        observed = cast(dict[str, object], actual[arm])
        reference = expected[arm]
        if (
            observed["per_query_r1"] != reference["per_query_r1"]
            or abs(cast(float, observed["map_at_r"]) - cast(float, reference["map_at_r"])) > 1e-5
        ):
            raise ValueError(f"SOP full-width {arm} holdout parity differs")


def paired_result(
    candidate: dict[str, object], baseline: dict[str, object], labels: np.ndarray
) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric, per_query in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        newer = np.asarray(candidate[per_query], dtype=np.float64)
        older = np.asarray(baseline[per_query], dtype=np.float64)
        if (
            newer.shape != labels.shape
            or older.shape != labels.shape
            or not bool(np.isfinite(newer).all())
            or not bool(np.isfinite(older).all())
            or abs(float(newer.mean()) - cast(float, candidate[metric])) > 1e-7
            or abs(float(older.mean()) - cast(float, baseline[metric])) > 1e-7
        ):
            raise ValueError("SOP paired metric authority differs")
        delta = newer - older
        result[metric] = {
            "candidate": candidate[metric],
            "baseline": baseline[metric],
            "candidate_minus_baseline": float(delta.mean()),
            "product_bootstrap_ci95": product_interval(labels, delta),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trained-features", type=Path, required=True)
    parser.add_argument("--trained-features-sha256", required=True)
    parser.add_argument("--fullwidth-receipt", type=Path, required=True)
    parser.add_argument("--compact-receipt", type=Path, required=True)
    parser.add_argument("--projection-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.projection_output.exists()
        or args.output == args.projection_output
        or sha256(args.trained_features) != args.trained_features_sha256
        or sha256(args.compact_receipt) != COMPACT_RECEIPT_SHA
    ):
        raise ValueError("SOP PCA audit input or output authority differs")
    root = Path(__file__).resolve().parents[1]
    source_manifest = {relative: sha256(root / relative) for relative in SOURCE_RELATIVES}
    full = json.loads(args.fullwidth_receipt.read_text())
    compact = json.loads(args.compact_receipt.read_text())
    if (
        any(full.get(field) != compact.get(field) for field in MATCHED_FIELDS)
        or full.get("schema") != "sfora-sop-compact-training-diagnostic-v1"
        or full.get("arm") != "arcface"
        or full.get("recipe") != "reference"
        or full.get("seed") != SEED
        or full.get("step") != MATCHED_STEP
        or full.get("embedding_width") != 768
        or compact.get("embedding_width", 128) != 128
        or len(full.get("validation_labels", [])) != 5_851
        or full.get("features_archive_sha256") != SOURCE_ARCHIVE_SHA
    ):
        raise ValueError("SOP matched PCA checkpoint receipts differ")
    with np.load(args.trained_features, allow_pickle=False) as archive:
        features = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        image_ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if (
        features.shape != (59_551, 768)
        or labels.shape != (59_551,)
        or image_ids.shape != (59_551,)
        or not bool(np.isfinite(features).all())
        or not bool(np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=2e-5))
        or metadata.get("schema") != "sfora-sop-trained-train-features-v1"
        or metadata.get("rows") != 59_551
        or metadata.get("width") != 768
        or metadata.get("step") != full["step"]
        or metadata.get("trained_checkpoint_sha256") != full["checkpoint_sha256"]
        or metadata.get("training_receipt_sha256") != sha256(args.fullwidth_receipt)
        or metadata.get("source_checkpoint_sha256") != full["input_checkpoint_sha256"]
        or metadata.get("sop_train_metadata_sha256") != full["sop_train_metadata_sha256"]
    ):
        raise ValueError("SOP full-width trained archive differs")
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=SEED)
    fit = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    holdout = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    fit_hash = hashlib.sha256(fit.astype("<i4").tobytes(order="C")).hexdigest()
    if (
        len(fit) != 53_700
        or len(holdout) != 5_851
        or fit_hash != full["fit_row_indexes_sha256"]
        or image_ids[holdout].tolist() != full["validation_image_ids"]
        or labels[holdout].tolist() != full["validation_labels"]
    ):
        raise ValueError("SOP trained archive row order differs")
    torch.set_num_threads(16)
    values = torch.from_numpy(features)
    held_labels = tuple(map(int, labels[holdout]))
    started = time.perf_counter()
    full_score = score_validation_features(values[holdout], held_labels)
    verify_holdout(full_score, full)
    fitting_started = time.perf_counter()
    projection = fit_centered_pca(values[fit].contiguous(), dimensions=128)
    fitting_seconds = time.perf_counter() - fitting_started
    projected = projection.apply(values[holdout].contiguous())
    pca_score = score_validation_features(projected, held_labels)
    labels_held = labels[holdout]
    packed_pca = cast(dict[str, object], pca_score["packed"])
    packed_full = cast(dict[str, object], full_score["packed"])
    packed_compact = cast(dict[str, object], compact["validation"]["packed"])
    result = {
        "schema": "sfora-sop-matched-fullwidth-fit-pca128-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products",
        "split": (
            "official train; PCA fit on 53700 fit-identity images, score 5851 "
            "disjoint holdout images"
        ),
        "seed": SEED,
        "step": full["step"],
        "compact_step_selected_on_this_holdout": True,
        "gallery_bytes_per_item": {"fullwidth": 770, "pca128": 130, "compact128": 130},
        "fullwidth_packed": {
            "recall_at_1": packed_full["recall_at_1"],
            "map_at_r": packed_full["map_at_r"],
        },
        "pca128_packed": {
            "recall_at_1": packed_pca["recall_at_1"],
            "map_at_r": packed_pca["map_at_r"],
            "per_query_r1": packed_pca["per_query_r1"],
            "per_query_ap": packed_pca["per_query_ap"],
        },
        "compact128_packed": {
            "recall_at_1": packed_compact["recall_at_1"],
            "map_at_r": packed_compact["map_at_r"],
        },
        "pca128_minus_compact128": paired_result(packed_pca, packed_compact, labels_held),
        "pca128_minus_fullwidth": paired_result(packed_pca, packed_full, labels_held),
        "projection": {
            "mean_sha256": tensor_sha256(projection.mean),
            "components_sha256": tensor_sha256(projection.components),
            "projected_holdout_sha256": tensor_sha256(projected),
            "fit_row_indexes_sha256": fit_hash,
            "input_geometry": "l2-normalized-trained-fullwidth-head-768",
            "output_geometry": "center-project-float64-then-row-l2-normalize-f32-and-pack-int8",
        },
        "inputs": {
            "trained_features_sha256": args.trained_features_sha256,
            "fullwidth_receipt_sha256": sha256(args.fullwidth_receipt),
            "compact_receipt_sha256": sha256(args.compact_receipt),
            "fullwidth_checkpoint_sha256": full["checkpoint_sha256"],
            "compact_checkpoint_sha256": compact["checkpoint_sha256"],
        },
        "source_sha256": source_manifest,
        "fit_seconds": fitting_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "hardware": {
            "processor": platform.processor(),
            "torch": torch.__version__,
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    }
    args.projection_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write_projection(stream: BinaryIO) -> None:
        np.savez_compressed(
            stream,
            mean=projection.mean.numpy(),
            components=projection.components.numpy(),
            schema=np.asarray("sfora-sop-fit-only-fullwidth-pca128-projection-v1"),
            step=np.asarray(MATCHED_STEP, dtype=np.int64),
            fullwidth_checkpoint_sha256=np.asarray(full["checkpoint_sha256"]),
            fit_row_indexes_sha256=np.asarray(fit_hash),
            input_geometry=np.asarray("l2-normalized-trained-fullwidth-head-768"),
            output_geometry=np.asarray(
                "center-project-float64-then-row-l2-normalize-f32-and-pack-int8"
            ),
            trained_features_sha256=np.asarray(args.trained_features_sha256),
        )

    publish_file_noreplace(args.projection_output, write_projection)
    result["projection_archive_sha256"] = sha256(args.projection_output)
    if (
        sha256(args.trained_features) != args.trained_features_sha256
        or {relative: sha256(root / relative) for relative in SOURCE_RELATIVES} != source_manifest
    ):
        raise ValueError("SOP trained archive or source changed during PCA audit")

    def write_receipt(stream: BinaryIO) -> None:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())

    publish_file_noreplace(args.output, write_receipt)
    print(json.dumps({"output": str(args.output), "projection": str(args.projection_output)}))


if __name__ == "__main__":
    main()
