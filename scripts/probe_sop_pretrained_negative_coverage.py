"""Train-fit-only frozen B16 nearest-negative opportunity probe."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from sfora.representation_ceiling import deterministic_class_partition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.source_archive
    source_sha = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != source_sha:
        raise ValueError("SOP pretrained feature archive differs")
    with np.load(path, allow_pickle=False) as archive:
        vectors = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
    if vectors.shape != (59_551, 768) or labels.shape != (59_551,):
        raise ValueError("SOP pretrained feature inventory differs")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=0.9, seed=179019
    )
    fit = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    fit_vectors = np.ascontiguousarray(vectors[fit])
    fit_labels = labels[fit]
    rng = np.random.default_rng(179019)
    anchor_rows = rng.choice(len(fit), size=512, replace=False)
    classes = np.asarray(sorted(set(fit_labels)), dtype=np.int64)
    rows_by_class = {int(label): np.flatnonzero(fit_labels == label) for label in classes}
    full_neg: list[float] = []
    batch_neg: list[float] = []
    best_pos: list[float] = []
    started = time.perf_counter()
    for begin in range(0, len(anchor_rows), 32):
        selected = anchor_rows[begin : begin + 32]
        scores = fit_vectors[selected] @ fit_vectors.T
        for offset, anchor in enumerate(selected):
            label = int(fit_labels[anchor])
            positives = rows_by_class[label]
            best_pos.append(float(scores[offset, positives[positives != anchor]].max()))
            scores[offset, positives] = -np.inf
            full_neg.append(float(scores[offset].max()))
            other_classes = classes[classes != label]
            sampled = rng.choice(other_classes, size=15, replace=False)
            candidates = np.concatenate(
                [
                    rng.choice(
                        rows_by_class[int(other)],
                        size=4,
                        replace=len(rows_by_class[int(other)]) < 4,
                    )
                    for other in sampled
                ]
            )
            batch_neg.append(float(scores[offset, candidates].max()))
    full = np.asarray(full_neg)
    batch = np.asarray(batch_neg)
    pos = np.asarray(best_pos)
    result = {
        "schema": "sfora-sop-frozen-pretrained-negative-coverage-probe-v1",
        "dataset": "Stanford Online Products",
        "split": "official train fit identities only; no holdout or test",
        "encoder": "UNICOM B16@224 pretrained, 768D cosine-normalized",
        "archive_sha256": source_sha,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split_seed": 179019,
        "sample_seed": 179019,
        "anchors": len(anchor_rows),
        "fit_gallery_rows": len(fit),
        "batch_negative_rule": "15 other randomly sampled fit classes; 4 sampled images per class",
        "full_harder_fraction": float(np.mean(full > batch + 1e-7)),
        "full_vs_batch_negative_cosine_gap_median": float(np.median(full - batch)),
        "full_vs_batch_negative_cosine_gap_p90": float(np.quantile(full - batch, 0.9)),
        "full_negative_above_best_positive": float(np.mean(full > pos)),
        "batch_negative_above_best_positive": float(np.mean(batch > pos)),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_eligible": False,
    }
    output = args.output
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
