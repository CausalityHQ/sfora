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

BaseMethod = Literal["proxy_anchor", "hist", "recall_at_k_surrogate"]
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
    "pa_ema_avg_fast",
    "pa_distill_fast",
    "pa_distill_avg",
    "pa_ema_avg_m95",
    "pa_ema_avg_m90",
    "pa_dual_ema",
    "pa_dual_ema_bnfix",
    "pa_ema_avg_bnfix",
    "pa_cebn",
    "pa_cebn_soft",
    "rspg",
    "rspg_soft_js",
    "rspg_distance_gate",
    "rspg_instance_gate",
    "arcg",
    "pa_cea",
    "pa_cea_distance",
    "ipsr",
    "tird",
    "pa_ipc4",
    "pa_fiedler",
    "pa_rcc",
    "pa_bep",
    "pa_cem",
    "ectr",
    "ectr_soft",
    "ectr_random",
    "ectr_plateau",
    "ectr_area",
    "pa_coalition",
    "pa_coalition_single",
    "pa_coalition_complementary",
    "pa_coalition_dropout",
    "pa_coalition_residual",
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
    "pa_ema_avg_fast": "proxy_anchor",
    "pa_distill_fast": "proxy_anchor",
    "pa_distill_avg": "proxy_anchor",
    "pa_ema_avg_m95": "proxy_anchor",
    "pa_ema_avg_m90": "proxy_anchor",
    "pa_dual_ema": "proxy_anchor",
    "pa_dual_ema_bnfix": "proxy_anchor",
    "pa_ema_avg_bnfix": "proxy_anchor",
    "pa_cebn": "proxy_anchor",
    "pa_cebn_soft": "proxy_anchor",
    "rspg": "proxy_anchor",
    "rspg_soft_js": "proxy_anchor",
    "rspg_distance_gate": "proxy_anchor",
    "rspg_instance_gate": "proxy_anchor",
    "arcg": "proxy_anchor",
    "pa_cea": "proxy_anchor",
    "pa_cea_distance": "proxy_anchor",
    "ipsr": "proxy_anchor",
    "tird": "proxy_anchor",
    "pa_ipc4": "proxy_anchor",
    "pa_fiedler": "proxy_anchor",
    "pa_rcc": "proxy_anchor",
    "pa_bep": "proxy_anchor",
    "pa_cem": "proxy_anchor",
    "ectr": "proxy_anchor",
    "ectr_soft": "proxy_anchor",
    "ectr_random": "proxy_anchor",
    "ectr_plateau": "proxy_anchor",
    "ectr_area": "proxy_anchor",
    "pa_coalition": "proxy_anchor",
    "pa_coalition_single": "proxy_anchor",
    "pa_coalition_complementary": "proxy_anchor",
    "pa_coalition_dropout": "proxy_anchor",
    "pa_coalition_residual": "proxy_anchor",
}


# A 2x2 over what an EMA teacher supplies. `pa_ema_avg_fast` showed that EVALUATING the
# averaged weights is worth >= +0.45 pt on CUB at zero training cost, while `pa_distill`
# (which evaluates the STUDENT and only distils toward the teacher) is worth +0.658. Those
# are two different uses of the same object, and until now no arm had both or neither at a
# matched momentum -- so they could not be added, separated, or ranked.
#
# Everything here runs at momentum 0.99. `pa_distill` uses 0.999, where the average still
# carries 5.3% of its initialisation; the slow/fast split (+0.07 vs >= +0.45) showed that
# contamination is not negligible, so holding momentum fixed at 0.99 is what makes the
# four cells comparable.
#
#   pa_ema_avg_fast   average only     evaluate teacher, no distillation loss
#   pa_distill_fast   distil only      evaluate student, distillation loss at 0.99
#   pa_distill_avg    both             evaluate teacher AND distil toward it
#   proxy_anchor      neither          the base
#
# If `both` exceeds each single arm, the two effects are separable and additive, and the
# combination is the arm to take to Cars and In-Shop.
_FAST_MOMENTUM = 0.99

# Polyak/SWA weight averaging with the distillation term REMOVED: maintain the EMA
# teacher, score retrieval from it, and add no loss (`ema_distill_weight` stays 0).
# `pa_distill` beats `proxy_anchor` on CUB by ~+0.4 pt (6/6 seeds, sign p=0.031) with no
# established mechanism, and an EMA teacher does two separable things -- supplies a
# distillation target, and is an averaged copy of the weights. This arm isolates the
# second. If it reproduces the gain, the distillation loss is inert and the effect is
# weight averaging under another name; if it does not, the loss is doing real work.
#
# Two momenta, because one would give an ambiguous null. At 0.999 over CUB's 2940 steps
# the EMA retains 0.999^2940 = 5.3% of its INITIALISATION -- a pretrained backbone but a
# randomly-initialised embedding head. That copy could score badly for a reason having
# nothing to do with whether averaging helps, so a null at 0.999 alone would not be
# readable. At 0.99 the initial weight is 3e-13, i.e. gone.
#
#   pa_ema_avg       momentum 0.999, matching `_DISTILL_DELTA` exactly. Isolates the
#                    teacher `pa_distill` actually distils toward, contamination and all.
#   pa_ema_avg_fast  momentum 0.99. Tests weight averaging as such, with no initialisation
#                    left in the average.
# Momentum sweep. Averaging is the strongest single intervention measured here once the
# best-over-training selection bonus is removed (+0.73 corrected at 0.99, ahead of
# distillation's +0.59), and momentum is its only real hyperparameter -- yet EMA
# EVALUATION has never been assessed on these benchmarks at all, so no prior work fixes
# it. Two sampled points is not a curve. Averaging windows over CUB's 2940 steps:
#
#   0.999   time constant 1000 steps, retains 5.3% of initialisation
#   0.99    time constant  100 steps, retains 3e-13
#   0.95    time constant   20 steps
#   0.90    time constant   10 steps
#
# At the short end the average approaches the student and the benefit must vanish; the
# question is where the optimum sits between that and contamination at the long end.
# Weight averaging on a base with TRAINABLE BatchNorm needs the EMAN buffer fix, or the
# average is only half an average. `_update_ema_teacher` hard-copies buffers by default,
# so the teacher would carry averaged WEIGHTS alongside the student's last-step BatchNorm
# running statistics. On CUB that is provably inert -- `freeze_batch_norm=True` stops the
# student's buffers moving at all, which
# `test_buffer_blending_is_a_no_op_while_batch_norm_stays_frozen` pins. On In-Shop
# (`freeze_batch_norm=False`) it is not inert, and an arm that failed there would fail for
# a reason having nothing to do with whether averaging helps -- the same shape of confound
# as the 5.3% initialisation contamination at momentum 0.999.
#
# `ema_teacher_train_mode` is deliberately NOT set: with no distillation loss the teacher
# is never forward-passed during training, and `_encode_model` forces eval() at scoring
# time, so only the buffers matter here.
_EMA_AVG_BNFIX_DELTA: dict[str, Any] = {
    "ema_weight_averaging": True,
    "ema_momentum": 0.99,
    "ema_teacher_ema_buffers": True,
}

_EMA_AVG_MOMENTUM: dict[str, float] = {
    "pa_ema_avg": 0.999,
    "pa_ema_avg_fast": 0.99,
    "pa_ema_avg_m95": 0.95,
    "pa_ema_avg_m90": 0.90,
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

# Two averages at two timescales, because the measurements say the roles want different
# ones. On CUB against proxy_anchor, at 0.999 vs 0.99:
#
#   as a distillation TARGET    0.999 -> +0.91,  0.99 -> +0.30
#   as the EVALUATED model      0.999 -> +0.07,  0.99 -> +0.45
#
# A slow teacher is a more stable thing to regress toward; a fast average tracks the
# current solution instead of dragging 5.3% of the initialisation along. A single EMA has
# to pick one and lose the other, which is what every arm before this did. This keeps the
# teacher at 0.999 for distillation and evaluates a separate 0.99 average.
#
# Prediction, registered before the run: if the roles are independent this should land
# near the sum of the two best cells rather than near either alone. It fails if it merely
# matches pa_distill (+0.658 over six seeds), which would mean the evaluated average adds
# nothing once a good teacher is present.
_DUAL_EMA_DELTA: dict[str, Any] = {
    **_DISTILL_DELTA,
    "ema_weight_averaging": True,
    "ema_eval_momentum": 0.99,
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
_RSATK_SOURCE = "https://github.com/yash0307/RecallatK_surrogate"
_RSATK_REVISION = "ed052029d258555df2f94dd82d6f7df60ef7cc6f"


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
        "proxy_initialization": "kaiming_normal",
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
        "drop_last_train_batch": True,
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


def _recall_at_k_surrogate_config(dataset: ImageDatasetName) -> dict[str, Any]:
    if dataset not in {"cub", "cars"}:
        raise RecipeUnavailableError(f"official RS@k recipe is unavailable for {dataset}")
    is_cars = dataset == "cars"
    return {
        **_shared_reference_config(),
        "objectives": ("recall_at_k_surrogate",),
        "backbone_name": "resnet50",
        "pretrained_weights": "legacy_resnet50_19c8e357",
        "optimizer": "adam",
        "batch_size": 392 if is_cars else 400,
        "eval_batch_size": 128,
        "samples_per_class": 4,
        "epoch_sampling_policy": "source_exhaustive",
        "eval_test_interval_epochs": 5,
        "eval_test_epoch_offset": 1,
        "group_size": 4,
        "train_epochs": 170 if is_cars else 40,
        "learning_rate": 1e-4,
        "backbone_learning_rate": 1e-4,
        "weight_decay": 4e-4,
        "warmup_epochs": 0,
        "lr_schedule": "multistep",
        "lr_milestones": (80, 140) if is_cars else (10, 20, 30),
        "lr_gamma": 0.3,
        "head_pooling": "gem",
        "embedding_head_init": "default",
        "pre_embedding_layer_norm": True,
        "embedding_layer_norm": False,
        "freeze_batch_norm": True,
        "freeze_batch_norm_affine": False,
        "recall_at_k_values": (1, 2, 4, 8, 16),
        "recall_at_k_rank_temperature": 0.01,
        "recall_at_k_membership_temperature": 1.0,
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
    for dataset in ("cub", "cars"):
        registry[("recall_at_k_surrogate", dataset)] = ImageRecipe(
            recipe_id=(
                f"recall_at_k_surrogate.{dataset}.official-{_RSATK_REVISION[:7]}"
            ),
            base_method="recall_at_k_surrogate",
            dataset=dataset,
            track="reference",
            provenance=RecipeProvenance(
                url=_RSATK_SOURCE,
                revision=_RSATK_REVISION,
                location="README training command plus src/main.py, losses.py, netlib.py",
                source_method="recall_at_k_surrogate",
                source_dataset=dataset,
                note="RS@k without optional SiMix; exact published ResNet-50/512-D recipe.",
            ),
            config=_recall_at_k_surrogate_config(dataset),
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
    if method == "pa_cem":
        delta = {
            "objectives": ("proxy_anchor_cem",),
            "cem_weight": 0.05,
            "cem_margin": 0.1,
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
    if method in {
        "pa_coalition",
        "pa_coalition_single",
        "pa_coalition_complementary",
        "pa_coalition_dropout",
        "pa_coalition_residual",
    }:
        delta = {
            "objectives": ("proxy_anchor_coalition",),
            "coalition_weight": 0.1,
            "coalition_mode": {
                "pa_coalition": "union",
                "pa_coalition_single": "single",
                "pa_coalition_complementary": "single_complementary",
                "pa_coalition_dropout": "dropout",
                "pa_coalition_residual": "residual",
            }[method],
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
    if method in {"ectr", "ectr_soft", "ectr_random", "ectr_plateau", "ectr_area"}:
        delta = {
            "ectr_weight": 0.5,
            "ectr_warmup_epoch": 10,
            "ectr_ramp_end_epoch": 30,
            "ectr_beta": 0.85,
            "ectr_plateau_target": 0.53,
            "ectr_switch_margin": 0.20,
            "ectr_repulsion_gap": 0.10,
            "ectr_variant": {
                "ectr": "full",
                "ectr_soft": "soft",
                "ectr_random": "random",
                "ectr_plateau": "plateau",
                "ectr_area": "area",
            }[method],
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
    if method in {"pa_ipc4", "pa_fiedler"}:
        delta: dict[str, Any] = {"samples_per_class": 4}
        if method == "pa_fiedler":
            delta.update(
                {
                    "fiedler_weight": 0.05,
                    "fiedler_temperature": 0.1,
                    "fiedler_min_class_size": 4,
                }
            )
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
    if method == "pa_rcc":
        delta = {
            "rcc_weight": 0.05,
            "rcc_tau": 0.8,
            "rcc_temperature": 0.1,
            "rcc_min_class_size": 4,
            "rcc_memory_per_class": 4,
            "samples_per_class": 0,
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
    if method == "pa_bep":
        delta = {
            "bep_weight": 0.05,
            "bep_temperature": 0.1,
            "bep_path_points": 9,
            "bep_min_class_size": 2,
            "samples_per_class": 0,
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
    if method in {"pa_dual_ema", "pa_dual_ema_bnfix"}:
        delta = dict(_DUAL_EMA_DELTA)
        if method == "pa_dual_ema_bnfix":
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
    if method in {"pa_distill_fast", "pa_distill_avg"}:
        delta = {**_DISTILL_DELTA, "ema_momentum": _FAST_MOMENTUM}
        if method == "pa_distill_avg":
            delta["ema_weight_averaging"] = True
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
    if method == "pa_ema_avg_bnfix":
        delta = dict(_EMA_AVG_BNFIX_DELTA)
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
    if method in {"pa_cebn", "pa_cebn_soft"}:
        delta = {
            "class_excluded_batch_norm": method == "pa_cebn",
            "class_excluded_batch_norm_blend": 0.70 if method == "pa_cebn_soft" else None,
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
    if method in {"rspg", "rspg_soft_js", "rspg_distance_gate", "rspg_instance_gate"}:
        delta = {
            "rspg_weight": 1.0,
            "ema_momentum": 0.99,
            "ema_teacher_ema_buffers": True,
        }
        controls = {
            "rspg": "signature_gate",
            "rspg_soft_js": "soft_js",
            "rspg_distance_gate": "distance_gate",
            "rspg_instance_gate": "instance_gate",
        }
        # Keep the already-running full arm's digest stable: signature_gate is
        # the runtime default and was absent from its frozen recipe delta.
        if method != "rspg":
            delta["rspg_control"] = controls[method]
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
    if method == "arcg":
        delta = {
            "arcg_weight": 1.0,
            "arcg_warmup_epoch": 10,
            "arcg_refresh_epoch": 40,
            "arcg_agreement_threshold": 0.5,
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
    if method in {"pa_cea", "pa_cea_distance"}:
        delta = {
            "cea_weight": 1.0,
            "cea_warmup_epoch": 10,
            "cea_refresh_epoch": 40,
            "cea_agreement_threshold": 0.47,
            "ema_teacher_ema_buffers": True,
        }
        if method == "pa_cea_distance":
            delta["cea_control"] = "distance"
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
    if method == "ipsr":
        delta = {
            "ipsr_weight": 1.0,
            "ipsr_warmup_epoch": 10,
            "ipsr_refresh_epoch": 40,
            "ipsr_agreement_threshold": 0.5,
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
    if method == "tird":
        delta = {
            "tird_weight": 1.0,
            "ema_momentum": 0.999,
            # In-Shop has trainable BatchNorm; the teacher must see the same
            # normalization regime and average its buffers as well as weights.
            **_BN_FIX_DELTA,
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
    if method in _EMA_AVG_MOMENTUM:
        delta = {
            "ema_weight_averaging": True,
            "ema_momentum": _EMA_AVG_MOMENTUM[method],
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
