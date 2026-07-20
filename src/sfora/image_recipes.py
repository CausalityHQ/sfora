"""Publication-backed image training recipes and provenance.

The values in this module are transcribed from the authors' executable commands and
code defaults.  A recipe is deliberately separate from the legacy protocol presets:
the former makes a method/dataset reproduction claim, while the latter remains useful
for historical common-protocol artifacts and ablations.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from sfora.data import ImageDatasetName

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
