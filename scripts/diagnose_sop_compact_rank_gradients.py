#!/usr/bin/env python3
"""Train-only head-output gradient scale of the fixed SOP rank objectives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from export_unicom_sop_embeddings import load_sop_embedding_archive
from torch.nn import functional as F
from train_sop_compact_backbone import initialize_head_and_classifier

from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import (
    CompactTrainingArm,
    compact_head_features,
    compact_training_terms,
)
from sfora.unicom_rank_finish import identity_balanced_batches

ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--features-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-gradient-diagnostic", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.features_archive) != ARCHIVE_SHA256:
        raise ValueError("SOP compact gradient source differs")
    torch.set_num_threads(16)
    archive = load_sop_embedding_archive(args.features_archive)
    source_labels = tuple(int(value) for value in archive["train_labels"])
    split = deterministic_class_partition(source_labels, fit_fraction=0.9, seed=179019)
    fit_labels = tuple(source_labels[index] for index in split.fit_row_indexes)
    fit_features = torch.from_numpy(
        np.ascontiguousarray(archive["train_embeddings"][list(split.fit_row_indexes)])
    ).float()
    head, classifier = initialize_head_and_classifier(fit_features, fit_labels)
    class_names = tuple(sorted(set(fit_labels)))
    class_index = {name: index for index, name in enumerate(class_names)}
    schedule = identity_balanced_batches(
        tuple(str(label) for label in fit_labels),
        batch_size=128,
        images_per_identity=4,
        seed=179019,
        epoch=1,
        steps=1,
    )
    batch = schedule[0]
    labels = torch.tensor([class_index[fit_labels[index]] for index in batch])
    mask = torch.arange(128, dtype=torch.int64).unsqueeze(0)
    head_output = compact_head_features(fit_features[list(batch)], head).detach()
    rows = {}
    for arm in (CompactTrainingArm.FLOAT_RANK, CompactTrainingArm.PACKED_RANK):
        features = head_output.clone().requires_grad_(True)
        control, rank = compact_training_terms(features, classifier, labels, mask, arm=arm)
        control_gradient = torch.autograd.grad(control, features, retain_graph=True)[0]
        rank_gradient = torch.autograd.grad(rank, features)[0]
        control_norm = float(torch.linalg.vector_norm(control_gradient))
        rank_norm = float(torch.linalg.vector_norm(rank_gradient))
        if not all(np.isfinite(value) and value > 0 for value in (control_norm, rank_norm)):
            raise ValueError("SOP compact gradient is nonfinite or zero")
        rows[arm.value] = {
            "arcface_loss": float(control.detach()),
            "rank_loss": float(rank.detach()),
            "arcface_head_gradient_l2": control_norm,
            "rank_head_gradient_l2": rank_norm,
            "weighted_rank_to_arcface_gradient_ratio": 0.1 * rank_norm / control_norm,
            "gradient_cosine": float(
                F.cosine_similarity(control_gradient.reshape(1, -1), rank_gradient.reshape(1, -1))[
                    0
                ]
            ),
        }
    receipt = {
        "schema": "sfora-sop-compact-rank-gradient-diagnostic-v1",
        "claim_eligible": False,
        "scope": "initial PCA head, cached official SOP train fit features, first 32x4 batch",
        "archive_sha256": ARCHIVE_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "batch_indexes_sha256": hashlib.sha256(
            np.asarray(batch, dtype="<i4").tobytes(order="C")
        ).hexdigest(),
        "rank_coefficient": 0.1,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
