#!/usr/bin/env python3
"""Select a published source recipe using only target-dataset training identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from sfora.data import ImageDatasetName, load_image_retrieval_bundle
from sfora.image_end_to_end import (
    ImageEndToEndResult,
    run_image_end_to_end_benchmark,
    write_image_end_to_end_report,
)
from sfora.image_recipes import (
    BaseMethod,
    ImageRecipe,
    RecipeCandidateScore,
    RecipeSelectionSplit,
    class_disjoint_recipe_selection_split,
    config_for_recipe,
    rank_recipe_candidates,
    reference_recipes_for_method,
    selected_extension_recipe,
    write_selection_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-method", choices=("proxy_anchor", "hist"), required=True)
    parser.add_argument("--dataset", choices=("inshop", "inat2018"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def _score_from_payload(path: Path, *, expected_digest: str) -> RecipeCandidateScore | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config", {}).get("recipe_digest") != expected_digest:
        return None
    methods = payload.get("methods", {})
    if len(methods) != 1:
        return None
    metrics = next(iter(methods.values()))
    return RecipeCandidateScore(
        recipe_id=str(payload["config"]["recipe_id"]),
        map_at_r=float(metrics["map_at_r"]),
        recall_at_1=float(metrics["recall_at_1"]),
    )


def _run_candidate(
    recipe: ImageRecipe,
    *,
    split: RecipeSelectionSplit,
    dataset_root: Path,
    seed: int,
    num_workers: int,
    output_path: Path,
) -> RecipeCandidateScore:
    from sfora.image_recipes import recipe_digest

    existing = _score_from_payload(output_path, expected_digest=recipe_digest(recipe))
    if existing is not None:
        return existing
    config = config_for_recipe(recipe).model_copy(
        update={
            "dataset_root": dataset_root,
            "seed": seed,
            "num_workers": num_workers,
            "eval_test_interval_epochs": 0,
            "progress_every": 100,
        }
    )

    def write_partial(partial: ImageEndToEndResult) -> None:
        write_image_end_to_end_report(partial, output_path)

    result: ImageEndToEndResult = run_image_end_to_end_benchmark(
        train_examples=split.optimization,
        test_examples=split.query,
        gallery_examples=split.gallery,
        config=config,
        progress_callback=write_partial,
    )
    write_image_end_to_end_report(result, output_path)
    metrics = next(iter(result.methods.values()))
    return RecipeCandidateScore(
        recipe_id=recipe.recipe_id,
        map_at_r=metrics.map_at_r,
        recall_at_1=metrics.recall_at_1,
    )


def main() -> None:
    args = _parse_args()
    base_method = cast(BaseMethod, args.base_method)
    dataset = cast(ImageDatasetName, args.dataset)
    bundle = load_image_retrieval_bundle(
        dataset_name=dataset,
        dataset_root=args.dataset_root,
        seed=args.seed,
    )
    split = class_disjoint_recipe_selection_split(
        bundle.train,
        fraction=args.selection_fraction,
        seed=args.seed,
    )
    candidates = [
        selected_extension_recipe(source, target_dataset=dataset)
        for source in reference_recipes_for_method(base_method)
    ]
    candidate_dir = args.output.parent / f"{args.output.stem}.candidates"
    scores = [
        _run_candidate(
            candidate,
            split=split,
            dataset_root=args.dataset_root,
            seed=args.seed,
            num_workers=args.num_workers,
            output_path=candidate_dir / f"{candidate.recipe_id}.json",
        )
        for candidate in candidates
    ]
    winner_id = rank_recipe_candidates(scores)[0].recipe_id
    winner = next(candidate for candidate in candidates if candidate.recipe_id == winner_id)
    write_selection_manifest(
        args.output,
        selected_recipe=winner,
        scores=scores,
        selection_seed=args.seed,
        protocol_version="class-disjoint-train-v1",
    )


if __name__ == "__main__":
    main()
