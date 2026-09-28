#!/usr/bin/env python3
"""Saved TRAIN-held 128-D vectors: distinguish float quality from packing loss."""

import json
from pathlib import Path

import numpy as np
import torch

import pe_l14_training as l14
import train_inshop_pe_pair as pair


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parent
    output = Path("/home/riomus/runs/sfora-pe-l14-prefetch-pilot-v1")
    result = output / "float-quality-diagnostic.json"
    assert not result.exists()
    receipt = output / "receipt.json"
    assert (
        pair.sha(receipt)
        == "0710914c9184eecfa47cfab8433126fb20fcafd8b78b751bbe3be6e82b344f7a"
    )
    r = json.loads(receipt.read_text())
    _, frozen, _ = l14.control(root)
    path = output / "pe.held.npy"
    assert pair.sha(path) == r["held_sha256"]
    vectors = torch.from_numpy(np.load(path, allow_pickle=False))
    assert vectors.shape == (12599, 128) and torch.isfinite(vectors).all()
    labels = tuple(row["product"] for row in frozen["held_manifest"])
    classes = {name: index for index, name in enumerate(sorted(set(labels)))}
    ids = torch.tensor([classes[label] for label in labels])
    query, gallery = frozen["query"], frozen["gallery"]
    relevant = torch.bincount(ids[gallery])[ids[query]]
    width = int(relevant.max())
    ranks = torch.arange(1, width + 1)
    hits, aps = [], []
    for start in range(0, len(query), 128):
        rows = query[start : start + 128]
        scores = vectors[rows] @ vectors[gallery].T
        order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :width]
        matches = ids[gallery][order] == ids[rows, None]
        counts = relevant[start : start + len(rows)]
        precision = matches.cumsum(dim=1) / ranks[None, :]
        ap = (precision * matches * (ranks[None, :] <= counts[:, None])).sum(
            dim=1
        ) / counts
        hits.extend(matches[:, 0].int().tolist())
        aps.extend(ap.tolist())
    quality = {"recall_at_1": float(np.mean(hits)), "map_at_r": float(np.mean(aps))}
    pair.smoke.save(
        result,
        {
            "script_sha256": pair.sha(Path(__file__)),
            "receipt_sha256": pair.sha(receipt),
            "held_sha256": r["held_sha256"],
            "quality": quality,
            "float_minus_packed_pp": {
                k: 100 * (v - r["quality"][k]) for k, v in quality.items()
            },
            "float_also_fails_both_quality_floors": quality["recall_at_1"] < 0.951720176
            and quality["map_at_r"] < 0.776237120,
            "quality_read": "existing TRAIN-held saved vectors only",
            "cuda": False,
            "optimizer_updates": 0,
            "new_encoder_inference": False,
        },
    )
    print(json.dumps(quality), flush=True)


if __name__ == "__main__":
    main()
