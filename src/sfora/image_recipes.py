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
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

from sfora.data import ImageDatasetName, ImageExample

if TYPE_CHECKING:
    from sfora.image_end_to_end import ImageEndToEndConfig

BaseMethod = Literal["proxy_anchor", "hist"]
RecipeTrack = Literal["reference", "selected_extension", "modified", "modified_legacy"]
MethodStatus = Literal["reference_method", "sfora_derived"]
DerivedMethod = Literal[
    "pa_distill",
    "herd",
    "pa_distill_bnfix",
    "herd_bnfix",
    "tversky",
    "shepard",
    "shepard_l1",
    "narrow128",
    "narrow64",
    "narrow128_distill",
    "narrow64_distill",
    "pa_ema_avg",
]

# Base loss each derived method attaches to.
_DERIVED_BASE: dict[str, BaseMethod] = {
    "pa_distill": "proxy_anchor",
    "pa_distill_bnfix": "proxy_anchor",
    "herd": "hist",
    "herd_bnfix": "hist",
    "tversky": "proxy_anchor",
    "shepard": "proxy_anchor",
    "shepard_l1": "proxy_anchor",
    "narrow128": "proxy_anchor",
    "narrow64": "proxy_anchor",
    "narrow128_distill": "proxy_anchor",
    "narrow64_distill": "proxy_anchor",
    "pa_ema_avg": "proxy_anchor",
}

# Polyak/SWA weight averaging with the distillation term REMOVED: maintain the EMA
# teacher, score retrieval from it, and add no loss (`ema_distill_weight` stays 0).
# `pa_distill` beats `proxy_anchor` on CUB by ~+0.4 pt (6/6 seeds, sign p=0.031) with no
# established mechanism, and an EMA teacher does two separable things -- supplies a
# distillation target, and is an averaged copy of the weights. This arm isolates the
# second. If it reproduces the gain, the distillation loss is inert and the effect is
# weight averaging under another name; if it does not, the loss is doing real work.
# Momentum matches `_DISTILL_DELTA` so the average is the same one `pa_distill` builds.
_EMA_AVG_DELTA: dict[str, Any] = {
    "ema_weight_averaging": True,
    "ema_momentum": 0.999,
}

# Embedding width of each capacity-weakened arm, against the official 512.
_NARROW_DIMENSIONS: dict[str, int] = {
    "narrow128": 128,
    "narrow64": 64,
    "narrow128_distill": 128,
    "narrow64_distill": 64,
}

# Shared distillation delta.
_DISTILL_DELTA: dict[str, Any] = {
    "ema_distill_weight": 1.0,
    "ema_momentum": 0.999,
    "ema_distill_tau": 0.1,
}

# The `_bnfix` variants additionally make the EMA teacher normalisation-consistent
# with the student. Historically the teacher ran in eval mode (BatchNorm running
# statistics) while the student trained in train mode (batch statistics), and its
# buffers were hard-copied rather than EMA-blended. With frozen BatchNorm the two
# coincide; with trainable BatchNorm the teacher becomes a systematically different
# function and distillation regresses the base loss. See docs/research_reset_plan.md H3.
# These are SEPARATE recipe IDs so the digests of `pa_distill`/`herd` are unchanged.
_BN_FIX_DELTA: dict[str, Any] = {
    "ema_teacher_train_mode": True,
    "ema_teacher_ema_buffers": True,
}

# Hypergraph-native distillation: replaces the generic pairwise relational target
# with one that only exists because HIST builds a hypergraph. The pairwise term is
# switched OFF (`ema_distill_weight: 0.0`) so the arms measure the new mechanism
# rather than a sum of two. The teacher is normalisation-consistent by construction
# (`_BN_FIX_DELTA`), because an eval-mode teacher would forward `hist_module.bn1` on
# running statistics and reintroduce the H3 defect inside the new loss.
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
        "dataset_selection_policy": "full_official_partition",
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


# SHOT -- Sinkhorn Hyperedge Optimal Transport. Replaces HIST's ad-hoc degree
# normalisation of the soft incidence with the entropic-OT coupling that minimises the
# free energy <C,P> - eps*H(P) under sample/hyperedge marginals. Because the cost is
# defined so that exp(-C) IS HIST's incidence, `iterations=0, epsilon=1.0` recovers HIST
# exactly -- so these recipes change ONE thing, and plain HIST is the null arm.
# NOTE this is not a distillation method: no EMA teacher is involved.

# Balanced sampling. Official HIST sets --IPC 0, so batches are random -- and on CUB
# (5864 images / 100 classes) a batch of 32 then holds ~27 distinct classes with ONE
# or two samples each. Almost every hyperedge has a single true member, so HIST's
# "higher-order" structure is carried by soft memberships rather than by genuine
# n-ary grouping. IPC=4 gives 8 classes x 4 samples, producing real multi-member
# hyperedges for the first time. This ADDS structure rather than regularisation,
# which is what the training curves say HIST needs: it peaks at 66% of its run and
# decays only 0.16 pt, i.e. it is not overfitting and has no use for another
# regulariser. It also gives SHOT's marginal constraint something to act on.
_IPC4_DELTA: dict[str, Any] = {"samples_per_class": 4}


# Persistent ("stigmergic") hypergraph. A rolling memory of recent embeddings joins
# the live batch as CONTEXT when building the incidence and propagation operator, so
# hyperedges accumulate many members while the loss stays on the live rows only.
# Measured motivation: a CUB batch of 32 holds ~27 classes with one or two samples
# each, so almost every hyperedge has a single true member and HIST's higher-order
# structure degenerates. Balanced sampling fixes that by spending class diversity and
# fails badly (hist_ipc4, -2.74 pt); this spends staleness instead, the same trade
# cross-batch memory already makes successfully here.

# Local NCA. Replaces the collapsing objective entirely rather than adding a term:
# the positive sum moves INSIDE the log (the original NCA form), so the loss is
# satisfied once some genuine same-class instance outranks the k hardest cross-class
# ones, and a legitimately distant same-class sample is never dragged in. Cross-batch
# memory supplies positives, because unbalanced CUB batches leave ~74% of anchors
# without a same-class partner and balanced sampling costs -2.74 pt (hist_ipc4).
# It borrows only the HIST recipe's optimiser/schedule; the loss itself is unrelated.

# Region Proxy Anchor: represent an image as a SET of spatial descriptors and score
# it against a class proxy by a soft maximum over regions, rather than pooling first.
# Motivated by this repo's antihub measurement - 5-8% of CUB test images are in
# nobody's top-10 and fail their own retrieval by 21 pt - on the hypothesis that
# global pooling averages away the one locally-discriminative region that separates
# two fine-grained classes. Everything else is the untouched official Proxy Anchor
# recipe, so the comparison isolates the representation change.


# Tversky contrast similarity (Tversky, Psychological Review 1977) replacing cosine.
# Objects become feature SETS via a learnable bank; similarity is the bounded ratio
# |A n B| / (|A n B| + alpha|A - B| + beta|B - A|). alpha != beta makes it asymmetric,
# which cosine cannot be, and alpha = beta = 1 recovers Tanimoto/Jaccard. Everything
# else is the untouched official Proxy Anchor recipe, so the comparison isolates the
# similarity function.
# Shepard's exponential generalisation kernel (Science 1987) replacing the Gaussian
# one hiding inside cosine-softmax. exp(cos/T) is proportional to exp(-d^2/2T); Shepard
# derived exp(-d), linear in distance, and no temperature on cosine reproduces that.
# Only the similarity function changes, so the comparison isolates the kernel.
_SHEPARD_DELTA: dict[str, Any] = {
    "objectives": ("shepard_proxy_anchor",),
    "shepard_order": 2,
}


_TVERSKY_DELTA: dict[str, Any] = {
    "objectives": ("tversky_proxy_anchor",),
    "tversky_features": 512,
    "tversky_alpha": 1.0,
    "tversky_beta": 0.5,
}


# Capacity-weakened Proxy Anchor bases, and the same bases with distillation added.
# These exist to put arms at *intermediate* headroom: every distillation result we own
# sits either at a large gap below the dataset's best base (CUB Proxy Anchor, +0.89 pt)
# or effectively at the ceiling (everything else, |delta| <= 0.04), so the apparent
# proportionality rests on a single informative point. Narrowing the embedding weakens
# the base monotonically while changing nothing else in the recipe, and each narrowed
# base is its own paired control. See docs/headroom_hypothesis.md for the predictions
# these were registered against.
def _narrow_delta(method: str) -> dict[str, Any]:
    delta: dict[str, Any] = {"embedding_dimensions": _NARROW_DIMENSIONS[method]}
    if method.endswith("_distill"):
        delta.update(_DISTILL_DELTA)
    return delta


def derive_recipe(recipe: ImageRecipe, method: DerivedMethod) -> ImageRecipe:
    """Pair a SFORA distillation variant with an otherwise unchanged base recipe."""

    expected_base = _DERIVED_BASE[method]
    if recipe.base_method != expected_base:
        raise ValueError(f"{method} requires a {expected_base} base recipe")
    if method == "pa_ema_avg":
        delta = dict(_EMA_AVG_DELTA)
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
    if method in _NARROW_DIMENSIONS:
        delta = _narrow_delta(method)
        return recipe.model_copy(
            deep=True,
            update={
                "recipe_id": f"{recipe.recipe_id}.{method}",
                "method_status": "sfora_derived",
                "derived_from_recipe_id": recipe.recipe_id,
                # `modified`, not `reference`. Every other derived arm leaves the
                # published recipe untouched and adds a SFORA term on top; these
                # overwrite a published hyperparameter (the 512-d embedding), so
                # they are deliberate ablations and must never be readable as
                # reproductions of Proxy Anchor.
                "track": "modified",
                "delta": delta,
                "config": {**recipe.config, **delta},
            },
        )
    if method in {"tversky", "shepard", "shepard_l1"}:
        if method == "tversky":
            delta = dict(_TVERSKY_DELTA)
        else:
            delta = dict(_SHEPARD_DELTA)
            delta["shepard_order"] = 1 if method == "shepard_l1" else 2
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
    delta = dict(_DISTILL_DELTA)
    if method.endswith("_bnfix"):
        delta.update(_BN_FIX_DELTA)
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
    if selector in _DERIVED_BASE:
        return derive_recipe(base, cast("DerivedMethod", selector))
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
