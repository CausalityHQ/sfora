#!/usr/bin/env python3
"""Fail-closed scalar-curve analysis for the fixed Cars196 PFML reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_STEPS = 16_200
EXPECTED_EVAL_EPOCHS = tuple(range(10, 201, 10))
EXPECTED_TRAIN_EXAMPLES = 8_054
EXPECTED_TEST_EXAMPLES = 8_131
EXPECTED_CONFIG: dict[str, Any] = {
    "dataset_name": "cars",
    "protocol": "pfml-resnet50-512",
    # The already-running preregistered job predates the explicit preset repair.
    # Exact official counts below prove that the legacy minimum filter was inert.
    "dataset_selection_policy": "legacy_minimum_filter",
    "objectives": ["pfml"],
    "backbone_name": "resnet50",
    "embedding_dimensions": 512,
    "optimizer": "adam",
    "batch_size": 100,
    "drop_last_train_batch": False,
    "eval_batch_size": 128,
    "train_steps": EXPECTED_STEPS,
    "train_epochs": 200,
    "learning_rate": 1.0e-4,
    "backbone_learning_rate": 1.0e-4,
    "proxy_learning_rate_multiplier": 100.0,
    "weight_decay": 1.0e-4,
    "warmup_epochs": 1,
    "warmup_is_additional": False,
    "schedule_during_warmup": True,
    "lr_schedule": "none",
    "lr_step_epochs": 10,
    "lr_milestones": [],
    "lr_gamma": 0.5,
    "samples_per_class": 4,
    "epoch_sampling_policy": "independent_balanced",
    "pretrained_weights": "v1",
    "head_pooling": "avg",
    "embedding_head_init": "default",
    "pre_embedding_layer_norm": False,
    "embedding_layer_norm": False,
    "proxy_count_per_class": 15,
    "proxy_initialization": "unit_normal",
    "potential_delta": 0.2,
    "potential_alpha": 3.0,
    "input_size": 224,
    "train_augmentation": "standard",
    "freeze_batch_norm": True,
    "freeze_batch_norm_affine": False,
    "weight_decay_exclusions": "bias_bn_proxy",
    "gradient_clip_value": None,
    "label_noise_fraction": 0.0,
    "eval_test_interval_epochs": 10,
    "eval_test_epoch_offset": 0,
    "checkpoint_selection_interval": 0,
    "retrieval_query_limit": None,
    "limit_per_class": None,
    "max_classes": None,
    "seed": 0,
    "num_workers": 8,
}


def theoretical_batch_energy_bounds(config: dict[str, Any]) -> dict[str, float | int]:
    batch_size = int(config["batch_size"])
    samples_per_class = int(config["samples_per_class"])
    proxy_count = int(config["proxy_count_per_class"])
    class_count = 98
    if batch_size != 100 or samples_per_class != 4 or proxy_count != 15:
        raise ValueError("the fixed Cars196 PFML batch/proxy interpretation changed")
    classes_per_batch = batch_size // samples_per_class
    if classes_per_batch * samples_per_class != batch_size:
        raise ValueError("fixed PFML batch is not an exact balanced-class multiple")
    points = batch_size + class_count * proxy_count
    represented_class_points = proxy_count + samples_per_class
    same_ordered_pairs = (
        classes_per_batch * represented_class_points * (represented_class_points - 1)
        + (class_count - classes_per_batch) * proxy_count * (proxy_count - 1)
    )
    all_ordered_pairs = points * (points - 1)
    different_ordered_pairs = all_ordered_pairs - same_ordered_pairs
    saturation = float(config["potential_delta"]) ** -float(config["potential_alpha"])
    far_same_baseline = saturation * different_ordered_pairs
    force_free_minimum = saturation * (different_ordered_pairs - same_ordered_pairs)
    return {
        "points_per_loss": points,
        "same_class_ordered_pairs": same_ordered_pairs,
        "different_class_ordered_pairs": different_ordered_pairs,
        "kernel_saturation": saturation,
        "far_same_no_collision_energy": far_same_baseline,
        "force_free_energy_minimum": force_free_minimum,
        "maximum_attraction_energy_drop": saturation * same_ordered_pairs,
    }


def analyze_report(report: dict[str, Any]) -> dict[str, Any]:
    config = report.get("config", {})
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"unexpected PFML config {key}: {config.get(key)!r} != {expected!r}")
    if report.get("train_examples") != EXPECTED_TRAIN_EXAMPLES:
        raise ValueError(
            "unexpected Cars196 PFML train_examples: "
            f"{report.get('train_examples')!r} != {EXPECTED_TRAIN_EXAMPLES}"
        )
    if report.get("test_examples") != EXPECTED_TEST_EXAMPLES:
        raise ValueError(
            "unexpected Cars196 PFML test_examples: "
            f"{report.get('test_examples')!r} != {EXPECTED_TEST_EXAMPLES}"
        )

    methods = report.get("methods")
    if not isinstance(methods, dict) or len(methods) != 1:
        raise ValueError("PFML report must contain exactly one method")
    metrics = next(iter(methods.values()))
    if metrics.get("objective") != "pfml" or metrics.get("executed_train_steps") != EXPECTED_STEPS:
        raise ValueError("PFML method did not complete the exact 16,200-step objective")
    losses = np.asarray(metrics.get("loss_history"), dtype=np.float64)
    recalls = np.asarray(metrics.get("test_recall_history"), dtype=np.float64)
    if losses.shape != (EXPECTED_STEPS,) or recalls.shape != (len(EXPECTED_EVAL_EPOCHS),):
        raise ValueError(
            f"unexpected PFML curve shapes: losses={losses.shape}, recalls={recalls.shape}"
        )
    if np.any(~np.isfinite(losses)) or np.any(~np.isfinite(recalls)):
        raise ValueError("PFML report contains non-finite scalar history")
    final_recall = float(metrics.get("recall_at_1", float("nan")))
    raw_best = float(np.max(recalls))
    raw_best_index = int(np.argmax(recalls))
    raw_best_epoch = EXPECTED_EVAL_EPOCHS[raw_best_index]
    if not np.isclose(final_recall, recalls[-1], rtol=0.0, atol=1.0e-12):
        raise ValueError("PFML final R@1 differs from the last fixed-cadence evaluation")
    if not np.isclose(float(metrics.get("best_test_recall_at_1")), raw_best, rtol=0.0, atol=1e-12):
        raise ValueError("PFML stored raw best differs from the fixed-cadence history")
    if int(metrics.get("best_test_epoch")) != raw_best_epoch:
        raise ValueError("PFML stored best epoch differs from the fixed-cadence history")

    bounds = theoretical_batch_energy_bounds(config)
    baseline = float(bounds["far_same_no_collision_energy"])
    maximum_drop = float(bounds["maximum_attraction_energy_drop"])
    apparent_saturation = (baseline - losses) / maximum_drop
    gates = {
        "raw_best_at_least_0_895": raw_best >= 0.895,
        "final_at_least_0_890": final_recall >= 0.890,
        "all_scalars_finite": True,
    }
    decision = (
        "passes_fixed_interpretation_metric_gates"
        if all(gates.values())
        else "fails_fixed_interpretation_metric_gates"
    )
    return {
        "decision": decision,
        "gates": gates,
        "final_recall_at_1_primary": final_recall,
        "raw_best_test_selected_recall_at_1": raw_best,
        "raw_best_epoch": raw_best_epoch,
        "test_recall_history": [
            {"epoch": epoch, "recall_at_1": float(value)}
            for epoch, value in zip(EXPECTED_EVAL_EPOCHS, recalls, strict=True)
        ],
        "loss": {
            "initial": float(losses[0]),
            "final": float(losses[-1]),
            "minimum_observed": float(np.min(losses)),
            "final_100_mean": float(np.mean(losses[-100:])),
        },
        "theoretical_batch_energy": bounds,
        "apparent_attraction_saturation_lower_bound": {
            "initial": float(apparent_saturation[0]),
            "final": float(apparent_saturation[-1]),
            "maximum": float(np.max(apparent_saturation)),
            "interpretation": (
                "Lower bound if active repulsive collisions exist; constant potential terms "
                "dominate the raw loss and exert no force."
            ),
        },
        "reporting_note": (
            "Raw best is test-selected. No selection-corrected value is claimed; final R@1 "
            "is primary and must still be independently re-encoded from the checkpoint."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_report(json.loads(args.report.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
