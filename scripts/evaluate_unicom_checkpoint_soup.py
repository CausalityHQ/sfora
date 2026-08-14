#!/usr/bin/env python3
"""Select a UNICOM trajectory soup and WiSE interpolation on train identities only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def suffix_windows(paths: tuple[Path, ...]) -> tuple[tuple[Path, ...], ...]:
    if not paths:
        raise ValueError("at least one trajectory checkpoint is required")
    return tuple(paths[index:] for index in range(len(paths) - 1, -1, -1))


def _validate_matching_states(states: tuple[Mapping[str, torch.Tensor], ...]) -> tuple[str, ...]:
    if not states:
        raise ValueError("at least one model state is required")
    keys = tuple(states[0])
    if not keys or any(tuple(state) != keys for state in states[1:]):
        raise ValueError("model state keys or order differ")
    for key in keys:
        first = states[0][key]
        if type(first) is not torch.Tensor:
            raise TypeError(f"model state value is not a tensor: {key}")
        if any(
            value.shape != first.shape or value.dtype != first.dtype
            for value in (state[key] for state in states[1:])
        ):
            raise ValueError(f"model state tensor metadata differs: {key}")
    return keys


def average_model_states(
    states: tuple[Mapping[str, torch.Tensor], ...],
) -> OrderedDict[str, torch.Tensor]:
    """Average floating tensors in FP64 and carry latest non-floating buffers."""

    keys = _validate_matching_states(states)
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in keys:
        values = tuple(state[key].detach().cpu() for state in states)
        first = values[0]
        if first.is_floating_point():
            accumulator = torch.zeros_like(first, dtype=torch.float64)
            for value in values:
                accumulator.add_(value.to(torch.float64))
            result[key] = accumulator.div_(len(values)).to(first.dtype)
        else:
            result[key] = values[-1].clone()
    return result


def interpolate_model_states(
    initial: Mapping[str, torch.Tensor],
    soup: Mapping[str, torch.Tensor],
    *,
    alpha: float,
) -> OrderedDict[str, torch.Tensor]:
    """Return initial + alpha * (soup - initial) with FP64 arithmetic."""

    if type(alpha) is not float or not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("interpolation alpha must be a finite builtin float in [0, 1]")
    keys = _validate_matching_states((initial, soup))
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in keys:
        base = initial[key].detach().cpu()
        trained = soup[key].detach().cpu()
        if base.is_floating_point():
            mixed = base.to(torch.float64).add(
                trained.to(torch.float64).sub(base.to(torch.float64)), alpha=alpha
            )
            result[key] = mixed.to(base.dtype)
        else:
            result[key] = trained.clone()
    return result


def _epoch(path: Path) -> int:
    try:
        value = int(path.stem.removeprefix("epoch-"))
    except ValueError as error:
        raise ValueError(f"checkpoint name does not encode an epoch: {path}") from error
    if value <= 0 or path.stem != f"epoch-{value:04d}":
        raise ValueError(f"checkpoint name does not encode an epoch: {path}")
    return value


def _alpha_text(alpha: float) -> str:
    return format(alpha, ".12g")


def evaluate_grid(
    model: torch.nn.Module,
    initial: Mapping[str, torch.Tensor],
    checkpoints: tuple[tuple[Path, Mapping[str, torch.Tensor]], ...],
    *,
    alphas: tuple[float, ...],
    prepare: Callable[[], None] = lambda: None,
    evaluate: Callable[[], dict[str, float]],
) -> list[dict[str, Any]]:
    """Evaluate every suffix soup and interpolation in a stable registered order."""

    if not checkpoints:
        raise ValueError("at least one trajectory checkpoint is required")
    paths = tuple(path for path, _state in checkpoints)
    epochs = tuple(_epoch(path) for path in paths)
    if tuple(sorted(paths, key=_epoch)) != paths or len(set(epochs)) != len(epochs):
        raise ValueError("trajectory checkpoints must have unique increasing epochs")
    if not alphas or any(
        type(alpha) is not float or not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0
        for alpha in alphas
    ):
        raise ValueError("alphas must be finite builtin floats in [0, 1]")
    state_by_path = dict(checkpoints)
    candidates: list[dict[str, Any]] = []
    for window in suffix_windows(paths):
        epochs = tuple(_epoch(path) for path in window)
        soup = average_model_states(tuple(state_by_path[path] for path in window))
        for alpha in alphas:
            state = interpolate_model_states(initial, soup, alpha=alpha)
            model.load_state_dict(state, strict=True)
            prepare()
            metrics = evaluate()
            if tuple(metrics) != (
                "recall_at_1",
                "recall_at_10",
                "recall_at_20",
                "recall_at_30",
                "map_at_r",
            ) and tuple(metrics) != ("recall_at_1", "map_at_r"):
                raise ValueError("candidate metric schema differs")
            candidates.append(
                {
                    "name": (
                        f"epochs-{'_'.join(str(epoch) for epoch in epochs)}"
                        f"-alpha-{_alpha_text(alpha)}"
                    ),
                    "epochs": list(epochs),
                    "checkpoints": [str(path) for path in window],
                    "alpha": alpha,
                    "metrics": metrics,
                }
            )
    return candidates


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidate list is empty")
    return max(
        candidates,
        key=lambda row: (row["metrics"]["map_at_r"], row["metrics"]["recall_at_1"]),
    )


def _load_trainer():
    path = Path(__file__).resolve().with_name("train_unicom_inshop.py")
    spec = importlib.util.spec_from_file_location("_unicom_trainer_for_soup", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load UNICOM trainer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_checkpoint_states(
    paths: tuple[Path, ...],
    *,
    holdout_seed: int,
    holdout_fraction: float,
) -> tuple[
    tuple[tuple[Path, Mapping[str, torch.Tensor]], ...],
    dict[str, object],
]:
    result: list[tuple[Path, Mapping[str, torch.Tensor]]] = []
    training_protocol: dict[str, object] | None = None
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if type(checkpoint) is not dict or checkpoint.get("epoch") != _epoch(path):
            raise ValueError(f"training checkpoint epoch differs: {path}")
        if checkpoint.get("selection_holdout") != {
            "seed": holdout_seed,
            "fraction": holdout_fraction,
        }:
            raise ValueError(f"training checkpoint selection holdout differs: {path}")
        checkpoint_protocol = checkpoint.get("training_protocol")
        if type(checkpoint_protocol) is not dict or not checkpoint_protocol:
            raise ValueError(f"training checkpoint training protocol differs: {path}")
        if training_protocol is None:
            training_protocol = checkpoint_protocol
        elif checkpoint_protocol != training_protocol:
            raise ValueError(f"training checkpoint training protocol differs: {path}")
        state = checkpoint.get("model")
        if type(state) not in (dict, OrderedDict):
            raise ValueError(f"training checkpoint model state differs: {path}")
        result.append((path, state))
    if training_protocol is None:
        raise ValueError("at least one trajectory checkpoint is required")
    return tuple(result), training_protocol


def recalibrate_batch_norm(model: torch.nn.Module, loader, *, device: torch.device) -> None:
    """Recompute candidate BatchNorm statistics on optimization identities only."""

    batch_norms = tuple(
        module
        for module in model.modules()
        if isinstance(
            module,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
            ),
        )
    )
    if not batch_norms:
        model.eval()
        return
    momenta = tuple(module.momentum for module in batch_norms)
    model.eval()
    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    batches = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                model(images.to(device, non_blocking=True))
                batches += 1
        if batches == 0:
            raise ValueError("BatchNorm recalibration loader is empty")
    finally:
        for module, momentum in zip(batch_norms, momenta, strict=True):
            module.momentum = momentum
        model.eval()


def _atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"selection output already exists: {path}")
    try:
        with temporary.open("xb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"selection output already exists: {path}")
    payload = (json.dumps(value, indent=2) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from sfora.unicom_inshop import parse_inshop_partition
    from sfora.unicom_training import identity_holdout

    trainer = _load_trainer()
    model_path = args.output_dir / "selected-model.pt"
    report_path = args.output_dir / "selection.json"
    for path in (model_path, report_path):
        if path.exists() or path.with_name(f"{path.name}.tmp").exists():
            raise FileExistsError(f"selection output already exists: {path}")
    if trainer._git_revision(args.unicom_checkout) != trainer.UNICOM_REVISION:
        raise ValueError("UNICOM checkout revision differs")
    if trainer._sha256_file(args.initial_checkpoint) != trainer.UNICOM_L14_336_SHA256:
        raise ValueError("UNICOM initial checkpoint SHA-256 differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UNICOM soup evaluation")
    records = parse_inshop_partition(args.dataset_root)
    train_records = tuple(record for record in records if record.split == "train")
    optimization, query, gallery, _labels = identity_holdout(
        train_records, fraction=args.holdout_fraction, seed=args.holdout_seed
    )
    if not query or not gallery:
        raise ValueError("soup selection requires a nonempty train-only holdout")
    model, transform = trainer._load_official_model(args.unicom_checkout, args.initial_checkpoint)
    initial = OrderedDict(
        (name, value.detach().cpu().clone()) for name, value in model.state_dict().items()
    )
    device = torch.device("cuda")
    model = model.to(device)
    checkpoints, training_protocol = _load_checkpoint_states(
        tuple(args.trajectory_checkpoints),
        holdout_seed=args.holdout_seed,
        holdout_fraction=args.holdout_fraction,
    )
    expected_protocol_fields = {
        "protocol": "unicom-inshop-official-single-device-v1",
        "unicom_revision": trainer.UNICOM_REVISION,
        "initial_checkpoint_sha256": trainer.UNICOM_L14_336_SHA256,
        "partition_sha256": trainer._sha256_file(
            args.dataset_root / "Eval" / "list_eval_partition.txt"
        ),
        "holdout_seed": args.holdout_seed,
        "holdout_fraction": args.holdout_fraction,
        "selected_features": args.selected_features,
    }
    if any(training_protocol.get(key) != value for key, value in expected_protocol_fields.items()):
        raise ValueError("training checkpoint protocol differs from soup inputs")
    calibration_loader = torch.utils.data.DataLoader(
        trainer.InshopEvalDataset(optimization, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    candidates = evaluate_grid(
        model,
        initial,
        checkpoints,
        alphas=tuple(args.alphas),
        prepare=lambda: recalibrate_batch_norm(model, calibration_loader, device=device),
        evaluate=lambda: trainer.evaluate_holdout(
            model,
            query,
            gallery,
            transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            selected_features=args.selected_features,
        ),
    )
    selected = select_candidate(candidates)
    checkpoint_by_epoch = {_epoch(path): state for path, state in checkpoints}
    soup = average_model_states(tuple(checkpoint_by_epoch[epoch] for epoch in selected["epochs"]))
    selected_state = interpolate_model_states(initial, soup, alpha=selected["alpha"])
    model.load_state_dict(selected_state, strict=True)
    recalibrate_batch_norm(model, calibration_loader, device=device)
    selected_state = OrderedDict(
        (name, value.detach().cpu().clone()) for name, value in model.state_dict().items()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        {"model": selected_state, "selection": selected},
        model_path,
    )
    report = {
        "protocol": "unicom-train-identity-holdout-suffix-soup-wise-v2",
        "holdout_seed": args.holdout_seed,
        "holdout_fraction": args.holdout_fraction,
        "training_protocol": training_protocol,
        "batch_norm_recalibration": "full-optimization-cumulative-batches",
        "selected": selected,
        "candidates": candidates,
        "model_path": str(model_path),
    }
    _atomic_json(report, report_path)
    return report


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trajectory-checkpoints", required=True, type=Path, nargs="+")
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selected-features", type=int, default=512)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        report = run(parse_args(arguments))
    except Exception as error:
        print(f"soup evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report["selected"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
