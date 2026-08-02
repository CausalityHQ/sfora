#!/usr/bin/env python3
"""Strict post-run judgement for the preregistered Cars196 RS@k reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sfora.image_recipes import recipe_digest, reference_recipe

EXPECTED_DATASET = "cars"
EXPECTED_OBJECTIVE = "recall_at_k_surrogate"
EXPECTED_EPOCHS = 170
PREDICTED_LOW = 0.787
PREDICTED_HIGH = 0.827
HARD_FAILURE = 0.780


def _selection_overshoot(history: list[float], half_width: int = 2) -> tuple[float, float, int]:
    if len(history) < 5:
        raise ValueError("at least five evaluations are required for selection correction")
    index = max(range(len(history)), key=history.__getitem__)
    low = max(0, index - half_width)
    high = min(len(history), index + half_width + 1)
    neighbours = [history[j] for j in range(low, high) if j != index]
    return history[index], sum(neighbours) / len(neighbours), index


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config") or {}
    recipe = reference_recipe(EXPECTED_OBJECTIVE, EXPECTED_DATASET)
    expected_digest = recipe_digest(recipe)
    errors: list[str] = []

    if config.get("dataset_name") != EXPECTED_DATASET:
        errors.append(f"dataset_name must be {EXPECTED_DATASET!r}")
    if list(config.get("objectives") or []) != [EXPECTED_OBJECTIVE]:
        errors.append(f"objectives must be [{EXPECTED_OBJECTIVE!r}]")
    if config.get("recipe_id") != recipe.recipe_id:
        errors.append(f"recipe_id must be {recipe.recipe_id!r}")
    if config.get("recipe_digest") != expected_digest:
        errors.append(f"recipe_digest must be {expected_digest}")
    if config.get("train_epochs") != EXPECTED_EPOCHS:
        errors.append(f"train_epochs must be {EXPECTED_EPOCHS}")
    if config.get("seed") != 0:
        errors.append("seed must be 0")

    methods = payload.get("methods") or {}
    if len(methods) != 1:
        errors.append("artifact must contain exactly one method")
        method: dict[str, Any] = {}
    else:
        method = next(iter(methods.values()))
    history = [float(x) for x in method.get("test_recall_history") or []]
    if len(history) != EXPECTED_EPOCHS:
        errors.append(
            f"test_recall_history must contain {EXPECTED_EPOCHS} completed evaluations, "
            f"found {len(history)}"
        )
    if errors:
        raise ValueError("invalid or incomplete RS@k artifact: " + "; ".join(errors))

    raw, corrected, index = _selection_overshoot(history)
    recorded_best = float(method.get("best_test_recall_at_1"))
    recorded_epoch = int(method.get("best_test_epoch"))
    if abs(recorded_best - raw) > 1e-12 or recorded_epoch != index + 1:
        raise ValueError(
            "artifact best fields disagree with the complete recall history: "
            f"fields=({recorded_best}, epoch {recorded_epoch}), "
            f"history=({raw}, epoch {index + 1})"
        )

    best_retrieval = method.get("best_test_retrieval") or {}
    final_retrieval = method.get("retrieval") or {}
    ks = (1, 2, 4, 8)
    best_curve = {f"r@{k}": float(best_retrieval[f"recall_at_{k}"]) for k in ks}
    final_curve = {f"r@{k}": float(final_retrieval[f"recall_at_{k}"]) for k in ks}
    if raw < HARD_FAILURE:
        verdict = "FAILS_PREREGISTERED_FAITHFUL_REPRODUCTION"
    elif PREDICTED_LOW <= raw <= PREDICTED_HIGH:
        verdict = "WITHIN_PREREGISTERED_RANGE"
    else:
        verdict = "OUTSIDE_PREDICTED_RANGE_BUT_ABOVE_HARD_FAILURE"

    return {
        "verdict": verdict,
        "recipe_id": recipe.recipe_id,
        "recipe_digest": expected_digest,
        "seed": 0,
        "epochs": EXPECTED_EPOCHS,
        "best_epoch": index + 1,
        "raw_best_r@1": raw,
        "selection_corrected_r@1": corrected,
        "selection_bonus_points": 100.0 * (raw - corrected),
        "best_checkpoint_curve": best_curve,
        "final_checkpoint_curve": final_curve,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.artifact.read_text(encoding="utf-8")))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
