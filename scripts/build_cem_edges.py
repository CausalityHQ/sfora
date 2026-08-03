#!/usr/bin/env python3
"""Build a digestable CEM class-confusion edge file from training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sfora.cem import build_confusion_edges


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=2)
    args = parser.parse_args()
    with np.load(args.pack, allow_pickle=False) as d:
        embeddings = d["embeddings"]
        labels = d["labels"]
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ck["state_dict"]
    edges = build_confusion_edges(
        embeddings,
        labels,
        state["metric_proxies"].detach().cpu().numpy(),
        state["metric_proxy_labels"].detach().cpu().numpy(),
        min_support=args.min_support,
    )
    payload = {
        "pack_sha256": sha256(args.pack),
        "checkpoint_sha256": sha256(args.checkpoint),
        "min_support": args.min_support,
        "edges": {str(source): [target, weight] for source, (target, weight) in edges.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"edge_count": len(edges), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
