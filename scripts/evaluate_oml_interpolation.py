#!/usr/bin/env python3
"""Evaluate weight-space interpolation from released OML weights to a continuation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

import torch
from train_oml_anchored_triplet import evaluation_values, validate_registered_initial

from sfora.foundation_finetune import (
    IdentityNeck,
    paired_retrieval_statistics,
    select_query_gallery_identity_subset,
)
from sfora.foundation_oml import configure_oml_input_size, load_oml_inshop_examples, load_oml_vit

REGISTERED_MINIMUM_MAP_DELTA = 0.004
REGISTERED_MAXIMUM_RECALL_DROP = 0.0007


def registered_interpolation_decision(
    *,
    initial: Mapping[str, float],
    final: Mapping[str, float],
    paired: Mapping[str, float | int],
) -> bool:
    return bool(
        paired["map_at_r_delta"] >= REGISTERED_MINIMUM_MAP_DELTA
        and paired["map_at_r_delta_ci95_lower"] > 0.0
        and final["recall_at_1"] >= initial["recall_at_1"] - REGISTERED_MAXIMUM_RECALL_DROP
    )


def bootstrap_seed(*, alpha: float, checkpoint_sha256: str) -> int:
    material = f"{checkpoint_sha256}:{alpha.hex()}".encode()
    return int.from_bytes(sha256(material).digest()[:4], "big")


def is_registered_screen(
    *, evaluation_fraction: float, evaluation_seed: int, evaluation_role: str
) -> bool:
    return (
        evaluation_fraction == 0.2
        and evaluation_seed == 17
        and evaluation_role == "screen"
    )

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
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--evaluation-seed", type=int, default=17)
    parser.add_argument("--evaluation-role", choices=("screen", "holdout"), default="screen")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output_checkpoint is not None and args.output_checkpoint.exists():
        raise FileExistsError(args.output_checkpoint)
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
    trained_checkpoint_sha256 = sha256(args.trained_checkpoint.read_bytes()).hexdigest()
    if Path(saved["base_checkpoint"]).resolve() != args.base_checkpoint.resolve():
        raise ValueError("trained checkpoint base differs")
    if saved["student_input_size"] != args.input_size:
        raise ValueError("trained checkpoint input size differs")
    if saved["ema_decay"] != 0.995 or saved["epoch"] != 10:
        raise ValueError("trained checkpoint schedule differs")
    trained_model = saved["model"]
    trained_neck = saved["neck"]

    query = load_oml_inshop_examples(args.partition, image_root=args.image_root, split="query")
    gallery = load_oml_inshop_examples(args.partition, image_root=args.image_root, split="gallery")
    query, gallery = select_query_gallery_identity_subset(
        query,
        gallery,
        seed=args.evaluation_seed,
        fraction=args.evaluation_fraction,
        complement=args.evaluation_role == "holdout",
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
    if is_registered_screen(
        evaluation_fraction=args.evaluation_fraction,
        evaluation_seed=args.evaluation_seed,
        evaluation_role=args.evaluation_role,
    ):
        validate_registered_initial(
            seed=0,
            evaluation_role=args.evaluation_role,
            student_input_size=args.input_size,
            initial=initial,
        )
    rows: list[dict[str, object]] = []
    for alpha in args.alphas:
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
            seed=bootstrap_seed(
                alpha=alpha, checkpoint_sha256=trained_checkpoint_sha256
            ),
            bootstrap_replicates=10_000,
        )
        rows.append(
            {
                "alpha": alpha,
                "metrics": metrics,
                "paired": paired,
                "passed": registered_interpolation_decision(
                    initial=initial, final=metrics, paired=paired
                ),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    payload = {
        "method": "linear weight interpolation from released OML to EMA continuation",
        "input_size": args.input_size,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "trained_checkpoint": str(args.trained_checkpoint.resolve()),
        "trained_checkpoint_sha256": trained_checkpoint_sha256,
        "evaluation_fraction": args.evaluation_fraction,
        "evaluation_seed": args.evaluation_seed,
        "evaluation_role": args.evaluation_role,
        "batch_size": args.batch_size,
        "torch_version": str(torch.__version__),
        "timm_version": importlib.metadata.version("timm"),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "trained_ema_decay": saved["ema_decay"],
        "trained_epochs": saved["epoch"],
        "initial": initial,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_checkpoint is not None:
        if len(args.alphas) != 1:
            raise ValueError("checkpoint output requires exactly one alpha")
        torch.save(
            {
                "model": model.state_dict(),
                "neck": neck.state_dict(),
                "alpha": args.alphas[0],
                "base_checkpoint": str(args.base_checkpoint.resolve()),
                "trained_checkpoint": str(args.trained_checkpoint.resolve()),
                "trained_checkpoint_sha256": trained_checkpoint_sha256,
                "input_size": args.input_size,
            },
            args.output_checkpoint,
        )


if __name__ == "__main__":
    main()
