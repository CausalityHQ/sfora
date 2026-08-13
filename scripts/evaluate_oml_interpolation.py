#!/usr/bin/env python3
"""Evaluate weight-space interpolation from released OML weights to a continuation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from train_oml_anchored_triplet import evaluation_values

from sfora.foundation_finetune import (
    IdentityNeck,
    paired_retrieval_statistics,
    select_query_gallery_identity_subset,
)
from sfora.foundation_oml import configure_oml_input_size, load_oml_inshop_examples, load_oml_vit


def interpolate_state_dict(
    base: Mapping[str, torch.Tensor],
    trained: Mapping[str, torch.Tensor],
    *,
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if list(base) != list(trained):
        raise ValueError("state keys differ")
    result: dict[str, torch.Tensor] = {}
    for name, base_value in base.items():
        trained_value = trained[name]
        if base_value.shape != trained_value.shape or base_value.dtype != trained_value.dtype:
            raise ValueError(f"state tensor differs: {name}")
        if base_value.is_floating_point():
            result[name] = torch.lerp(base_value, trained_value, alpha)
        else:
            if not torch.equal(base_value, trained_value):
                raise ValueError(f"non-floating state differs: {name}")
            result[name] = base_value.clone()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--evaluation-seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sorted(set(args.alphas)) != args.alphas or any(
        alpha <= 0.0 or alpha >= 1.0 for alpha in args.alphas
    ):
        raise ValueError("alphas must be unique, sorted, and strictly inside (0, 1)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_oml_vit(str(args.base_checkpoint), device=device)
    configure_oml_input_size(model, input_size=args.input_size)
    neck = IdentityNeck(int(model.num_features)).to(device)
    base_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    base_neck = {name: value.detach().clone() for name, value in neck.state_dict().items()}
    saved = torch.load(args.trained_checkpoint, map_location=device, weights_only=False)
    trained_model = saved["model"]
    trained_neck = saved["neck"]

    query = load_oml_inshop_examples(args.partition, image_root=args.image_root, split="query")
    gallery = load_oml_inshop_examples(args.partition, image_root=args.image_root, split="gallery")
    query, gallery = select_query_gallery_identity_subset(
        query,
        gallery,
        seed=args.evaluation_seed,
        fraction=args.evaluation_fraction,
        complement=False,
    )
    initial, initial_hits, initial_ap = evaluation_values(
        model,
        neck,
        query,
        gallery,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        input_size=args.input_size,
    )
    rows: list[dict[str, object]] = []
    for index, alpha in enumerate(args.alphas):
        model.load_state_dict(interpolate_state_dict(base_model, trained_model, alpha=alpha))
        neck.load_state_dict(interpolate_state_dict(base_neck, trained_neck, alpha=alpha))
        metrics, hits, average_precision = evaluation_values(
            model,
            neck,
            query,
            gallery,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            input_size=args.input_size,
        )
        paired = paired_retrieval_statistics(
            initial_hits=initial_hits,
            final_hits=hits,
            initial_ap=initial_ap,
            final_ap=average_precision,
            seed=20_260_900 + index,
            bootstrap_replicates=10_000,
        )
        rows.append({"alpha": alpha, "metrics": metrics, "paired": paired})
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    payload = {
        "method": "linear weight interpolation from released OML to EMA continuation",
        "input_size": args.input_size,
        "initial": initial,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
