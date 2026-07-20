"""Publication-backed image training recipes and provenance.

The values in this module are transcribed from the authors' executable commands and
code defaults.  A recipe is deliberately separate from the legacy protocol presets:
the former makes a method/dataset reproduction claim, while the latter remains useful
for historical common-protocol artifacts and ablations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from sfora.data import ImageDatasetName, ImageExample

if TYPE_CHECKING:
    from sfora.image_end_to_end import ImageEndToEndConfig

BaseMethod = Literal["proxy_anchor", "hist"]
RecipeTrack = Literal["reference", "selected_extension", "modified", "modified_legacy"]
MethodStatus = Literal["reference_method", "sfora_derived"]
DerivedMethod = Literal["pa_distill", "herd"]

_PROXY_ANCHOR_REVISION = "51db57031e38f75c03f69bbdfad1a3233afd9787"
_PROXY_ANCHOR_SOURCE = "https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020"
_HIST_REVISION = "e7d650c80460f464c55bcdc2262d785923c50dc4"
_HIST_SOURCE = "https://github.com/ljin0429/HIST"


class RecipeUnavailableError(ValueError):
    """Raised when a pair has no publication-backed recipe."""


class RecipeProvenance(BaseModel):
    """Pinned primary-source identity for a recipe."""

    model_config = ConfigDict(frozen=True)

    url: str
    revision: str
    location: str
    source_method: BaseMethod
    source_dataset: ImageDatasetName
    note: str = ""


class ImageRecipe(BaseModel):
    """Complete source fields and provenance for one image-training recipe."""

    model_config = ConfigDict(frozen=True)

    recipe_id: str
    base_method: BaseMethod
    dataset: ImageDatasetName
    track: RecipeTrack
    method_status: MethodStatus = "reference_method"
    provenance: RecipeProvenance
    config: dict[str, Any]
    derived_from_recipe_id: str | None = None
    delta: dict[str, Any] = {}


class RecipeCandidateScore(BaseModel):
    """Training-only retrieval score for one source recipe candidate."""

    model_config = ConfigDict(frozen=True)

    recipe_id: str
    map_at_r: float
    recall_at_1: float


@dataclass(frozen=True)
class RecipeSelectionSplit:
    """Class-disjoint optimization and selection query/gallery collections."""

    optimization: list[ImageExample]
    query: list[ImageExample]
    gallery: list[ImageExample]


def _shared_reference_config() -> dict[str, Any]:
    return {
        "embedding_dimensions": 512,
        "train_augmentation": "reference_random_resized_crop",
        "input_size": 224,
        "head_pooling": "avg_max",
        "embedding_head_init": "kaiming_normal",
        "lr_schedule": "step",
        "proxy_count_per_class": 0,
        "checkpoint_selection_interval": 0,
        "eval_test_interval_epochs": 1,
        "warmup_is_additional": False,
        "schedule_during_warmup": True,
        "weight_decay_exclusions": "none",
        "samples_per_class": 0,
        "ema_distill_weight": 0.0,
        "ema_momentum": 0.999,
        "ema_distill_tau": 0.1,
    }


def _proxy_anchor_config(dataset: ImageDatasetName) -> dict[str, Any]:
    config = {
        **_shared_reference_config(),
        "objectives": ("proxy_anchor",),
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "train_epochs": 60,
        "proxy_count_per_class": 1,
        "proxy_learning_rate_multiplier": 100.0,
        "proxy_anchor_alpha": 32.0,
        "proxy_anchor_delta": 0.1,
        "embedding_layer_norm": False,
        "gradient_clip_value": 10.0,
    }
    if dataset in {"cub", "cars"}:
        return {
            **config,
            "backbone_name": "resnet50",
            "pretrained_weights": "v1",
            "batch_size": 120,
            "learning_rate": 1e-4,
            "backbone_learning_rate": 1e-4,
            "warmup_epochs": 5,
            "freeze_batch_norm": True,
            "freeze_batch_norm_affine": True,
            "lr_step_epochs": 5 if dataset == "cub" else 10,
            "lr_gamma": 0.5,
        }
    return {
        **config,
        "backbone_name": "bn_inception",
        "pretrained_weights": "bn_inception_52deb4733",
        "batch_size": 180,
        "learning_rate": 6e-4,
        "backbone_learning_rate": 6e-4,
        "warmup_epochs": 1,
        "freeze_batch_norm": False,
        "freeze_batch_norm_affine": False,
        "lr_step_epochs": 20,
        "lr_gamma": 0.25,
    }


def _hist_config(dataset: ImageDatasetName) -> dict[str, Any]:
    dataset_values: dict[ImageDatasetName, dict[str, Any]] = {
        "cub": {
            "train_epochs": 40,
            "learning_rate": 1.2e-4,
            "hist_lr_ds": 1e-1,
            "hist_lr_hgnn_factor": 5.0,
            "weight_decay": 5e-5,
            "lr_step_epochs": 5,
            "hist_tau": 32.0,
            "hist_alpha": 1.1,
            "freeze_batch_norm": True,
        },
        "cars": {
            "train_epochs": 50,
            "learning_rate": 1e-4,
            "hist_lr_ds": 1e-1,
            "hist_lr_hgnn_factor": 10.0,
            "weight_decay": 1e-4,
            "lr_step_epochs": 10,
            "hist_tau": 32.0,
            "hist_alpha": 0.9,
            "freeze_batch_norm": True,
        },
        "sop": {
            "train_epochs": 60,
            "learning_rate": 1e-4,
            "hist_lr_ds": 1e-2,
            "hist_lr_hgnn_factor": 10.0,
            "weight_decay": 1e-4,
            "lr_step_epochs": 10,
            "hist_tau": 16.0,
            "hist_alpha": 2.0,
            "freeze_batch_norm": False,
        },
    }
    values = dataset_values[dataset]
    freeze_batch_norm = bool(values["freeze_batch_norm"])
    return {
        **_shared_reference_config(),
        **values,
        "objectives": ("hist",),
        "backbone_name": "resnet50",
        "pretrained_weights": "v1",
        "optimizer": "adam",
        "batch_size": 32,
        "backbone_learning_rate": values["learning_rate"],
        "warmup_epochs": 1,
        "warmup_is_additional": True,
        "schedule_during_warmup": False,
        "freeze_batch_norm_affine": freeze_batch_norm,
        "lr_gamma": 0.5,
        "embedding_layer_norm": True,
        "hist_hidden": 512,
        "hist_lambda_s": 1.0,
        "hist_var_floor": 0.0,
        "gradient_clip_value": None,
    }


def _build_reference_registry() -> dict[tuple[BaseMethod, ImageDatasetName], ImageRecipe]:
    registry: dict[tuple[BaseMethod, ImageDatasetName], ImageRecipe] = {}
    proxy_anchor_datasets: tuple[ImageDatasetName, ...] = ("cub", "cars", "sop", "inshop")
    for dataset in proxy_anchor_datasets:
        source_dataset = dataset
        registry[("proxy_anchor", dataset)] = ImageRecipe(
            recipe_id=f"proxy_anchor.{dataset}.official-{_PROXY_ANCHOR_REVISION[:7]}",
            base_method="proxy_anchor",
            dataset=dataset,
            track="reference",
            provenance=RecipeProvenance(
                url=_PROXY_ANCHOR_SOURCE,
                revision=_PROXY_ANCHOR_REVISION,
                location=f"README.md#{dataset} command plus code/train.py defaults",
                source_method="proxy_anchor",
                source_dataset=source_dataset,
                note=(
                    "Official repository settings; the README states these improve "
                    "the paper result."
                ),
            ),
            config=_proxy_anchor_config(dataset),
        )
    hist_datasets: tuple[ImageDatasetName, ...] = ("cub", "cars", "sop")
    for dataset in hist_datasets:
        registry[("hist", dataset)] = ImageRecipe(
            recipe_id=f"hist.{dataset}.official-{_HIST_REVISION[:7]}",
            base_method="hist",
            dataset=dataset,
            track="reference",
            provenance=RecipeProvenance(
                url=_HIST_SOURCE,
                revision=_HIST_REVISION,
                location=f"README.md#{dataset} command plus code/train.py defaults",
                source_method="hist",
                source_dataset=dataset,
            ),
            config=_hist_config(dataset),
        )
    return registry


_REFERENCE_RECIPES = _build_reference_registry()


def reference_recipe(base_method: str, dataset: str) -> ImageRecipe:
    """Return the exact author recipe, failing if the pair was not published."""

    key = (base_method, dataset)
    recipe = _REFERENCE_RECIPES.get(key)  # type: ignore[arg-type]
    if recipe is None:
        raise RecipeUnavailableError(
            f"no published reference recipe for {base_method}/{dataset}; "
            "run training-only best-available recipe selection"
        )
    return recipe.model_copy(deep=True)


def reference_recipes_for_method(base_method: BaseMethod) -> list[ImageRecipe]:
    """Return every author recipe for a base method in stable dataset order."""

    return [
        recipe.model_copy(deep=True)
        for (method, _), recipe in sorted(
            _REFERENCE_RECIPES.items(),
            key=lambda item: item[1].recipe_id,
        )
        if method == base_method
    ]


def derive_recipe(recipe: ImageRecipe, method: DerivedMethod) -> ImageRecipe:
    """Pair a SFORA distillation variant with an otherwise unchanged base recipe."""

    expected_base: BaseMethod = "proxy_anchor" if method == "pa_distill" else "hist"
    if recipe.base_method != expected_base:
        raise ValueError(f"{method} requires a {expected_base} base recipe")
    delta = {
        "ema_distill_weight": 1.0,
        "ema_momentum": 0.999,
        "ema_distill_tau": 0.1,
    }
    return recipe.model_copy(
        deep=True,
        update={
            "recipe_id": f"{recipe.recipe_id}.{method}",
            "method_status": "sfora_derived",
            "derived_from_recipe_id": recipe.recipe_id,
            "delta": delta,
            "config": {**recipe.config, **delta},
        },
    )


def recipe_digest(recipe: ImageRecipe) -> str:
    """Return a stable digest over every recipe and provenance field."""

    payload = json.dumps(
        recipe.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def class_disjoint_recipe_selection_split(
    examples: Sequence[ImageExample],
    *,
    fraction: float,
    seed: int,
) -> RecipeSelectionSplit:
    """Hold out training labels and build deterministic selection query/gallery sets."""

    if not 0.0 < fraction < 1.0:
        raise ValueError(f"selection fraction must be between 0 and 1; got {fraction}")
    grouped: dict[int, list[ImageExample]] = {}
    for example in examples:
        grouped.setdefault(int(example.label), []).append(example)
    eligible_labels = sorted(label for label, group in grouped.items() if len(group) >= 2)
    if len(eligible_labels) < 4:
        raise ValueError(
            "recipe selection needs at least four training labels with two images each"
        )

    selection_count = min(
        max(2, int(round(len(eligible_labels) * fraction))),
        len(eligible_labels) - 2,
    )
    rng = np.random.default_rng(seed)
    selected_labels = {
        eligible_labels[int(index)]
        for index in rng.permutation(len(eligible_labels))[:selection_count]
    }
    optimization = [example for example in examples if int(example.label) not in selected_labels]
    query: list[ImageExample] = []
    gallery: list[ImageExample] = []
    for label in sorted(selected_labels):
        group = sorted(grouped[label], key=lambda example: example.example_id)
        order = rng.permutation(len(group))
        query_count = max(1, len(group) // 2)
        query.extend(group[int(index)] for index in order[:query_count])
        gallery.extend(group[int(index)] for index in order[query_count:])
    return RecipeSelectionSplit(
        optimization=optimization,
        query=query,
        gallery=gallery,
    )


def rank_recipe_candidates(
    scores: Sequence[RecipeCandidateScore],
) -> list[RecipeCandidateScore]:
    """Rank by MAP@R, Recall@1, then stable recipe ID."""

    if not scores:
        raise ValueError("recipe selection produced no candidate scores")
    return sorted(
        scores,
        key=lambda score: (-score.map_at_r, -score.recall_at_1, score.recipe_id),
    )


def selected_extension_recipe(
    source_recipe: ImageRecipe,
    *,
    target_dataset: ImageDatasetName,
) -> ImageRecipe:
    """Retarget a complete winning author recipe without hybridizing its settings."""

    source_dataset = source_recipe.provenance.source_dataset
    return source_recipe.model_copy(
        deep=True,
        update={
            "recipe_id": (
                f"{source_recipe.base_method}.{target_dataset}.selected-from-"
                f"{source_dataset}-{source_recipe.provenance.revision[:7]}"
            ),
            "dataset": target_dataset,
            "track": "selected_extension",
        },
    )


def write_selection_manifest(
    output_path: Path,
    *,
    selected_recipe: ImageRecipe,
    scores: Sequence[RecipeCandidateScore],
    selection_seed: int,
    protocol_version: str,
) -> Path:
    """Persist the frozen winner and complete training-only candidate ranking."""

    ranked = rank_recipe_candidates(scores)
    if ranked[0].recipe_id != selected_recipe.recipe_id:
        raise ValueError(
            "selected recipe does not match the highest-ranked candidate: "
            f"{selected_recipe.recipe_id} != {ranked[0].recipe_id}"
        )
    payload = {
        "protocol_version": protocol_version,
        "selection_seed": selection_seed,
        "winner": {
            "recipe_id": selected_recipe.recipe_id,
            "digest": recipe_digest(selected_recipe),
            "recipe": selected_recipe.model_dump(mode="json"),
        },
        "scores": [score.model_dump(mode="json") for score in ranked],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def load_selected_recipe_manifest(path: Path) -> ImageRecipe:
    """Load a selection winner and verify its recorded digest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        winner = payload["winner"]
        recipe = ImageRecipe.model_validate(winner["recipe"])
        recorded_digest = str(winner["digest"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid recipe selection manifest: {path}") from error
    actual_digest = recipe_digest(recipe)
    if actual_digest != recorded_digest:
        raise ValueError(
            f"recipe selection manifest digest mismatch: {recorded_digest} != {actual_digest}"
        )
    if recipe.track != "selected_extension":
        raise ValueError(
            f"selection manifest winner must be selected_extension; got {recipe.track}"
        )
    return recipe


def resolve_recipe(
    selector: str,
    *,
    base_method: BaseMethod,
    dataset: ImageDatasetName,
    selection_manifest: Path | None = None,
) -> ImageRecipe:
    """Resolve `auto`, a derived method, or an exact recipe ID."""

    try:
        base = reference_recipe(base_method, dataset)
    except RecipeUnavailableError as error:
        if selection_manifest is None:
            raise RecipeUnavailableError(
                f"selection manifest is required for unpublished {base_method}/{dataset}"
            ) from error
        base = load_selected_recipe_manifest(selection_manifest)
        if base.base_method != base_method or base.dataset != dataset:
            raise ValueError(
                "selection manifest pair mismatch: expected "
                f"{base_method}/{dataset}, got {base.base_method}/{base.dataset}"
            ) from error

    if selector == "auto":
        return base
    if selector == "pa_distill":
        return derive_recipe(base, "pa_distill")
    if selector == "herd":
        return derive_recipe(base, "herd")
    if selector != base.recipe_id:
        raise RecipeUnavailableError(
            f"recipe selector {selector!r} does not match resolved recipe {base.recipe_id!r}"
        )
    return base


def config_for_recipe(recipe: ImageRecipe) -> ImageEndToEndConfig:
    """Validate a complete recipe through the benchmark's runtime config model."""

    from sfora.image_end_to_end import ImageEndToEndConfig

    return ImageEndToEndConfig.model_validate(
        {
            **recipe.config,
            "dataset_name": recipe.dataset,
            # The legacy protocol field remains for result-schema compatibility; the
            # recipe metadata is authoritative for reproduction claims.
            "protocol": "proxy-anchor-resnet50-512",
            "recipe_id": recipe.recipe_id,
            "recipe_digest": recipe_digest(recipe),
            "recipe_track": recipe.track,
            "recipe_method_status": recipe.method_status,
            "recipe_base_method": recipe.base_method,
            "recipe_source_url": recipe.provenance.url,
            "recipe_source_revision": recipe.provenance.revision,
            "recipe_source_dataset": recipe.provenance.source_dataset,
            "recipe_derived_from_id": recipe.derived_from_recipe_id,
            "recipe_delta": recipe.delta,
            "recipe_modified_fields": {},
        }
    )


def mark_recipe_config_modified(
    reference_config: ImageEndToEndConfig,
    runtime_config: ImageEndToEndConfig,
    *,
    explicit_fields: Sequence[str],
) -> ImageEndToEndConfig:
    """Label explicit behavior overrides without treating runtime paths as recipe edits."""

    changes = {
        field: {
            "before": getattr(reference_config, field),
            "after": getattr(runtime_config, field),
        }
        for field in sorted(set(explicit_fields))
        if getattr(reference_config, field) != getattr(runtime_config, field)
    }
    if not changes:
        return runtime_config

    payload = runtime_config.model_dump(mode="json", exclude={"recipe_digest"})
    payload.update(
        {
            "recipe_track": "modified",
            "recipe_modified_fields": changes,
        }
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return runtime_config.model_copy(
        deep=True,
        update={
            "recipe_track": "modified",
            "recipe_digest": digest,
            "recipe_modified_fields": changes,
        },
    )
