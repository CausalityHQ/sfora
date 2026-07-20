from __future__ import annotations

import json
from pathlib import Path

import pytest

from sfora.data import ImageExample
from sfora.image_recipes import (
    RecipeCandidateScore,
    RecipeUnavailableError,
    class_disjoint_recipe_selection_split,
    config_for_recipe,
    derive_recipe,
    rank_recipe_candidates,
    recipe_digest,
    reference_recipe,
    selected_extension_recipe,
    write_selection_manifest,
)


def test_proxy_anchor_inshop_matches_official_command() -> None:
    recipe = reference_recipe("proxy_anchor", "inshop")

    assert recipe.recipe_id == "proxy_anchor.inshop.official-51db570"
    assert recipe.track == "reference"
    assert recipe.method_status == "reference_method"
    assert recipe.config["backbone_name"] == "bn_inception"
    assert recipe.config["batch_size"] == 180
    assert recipe.config["learning_rate"] == pytest.approx(6e-4)
    assert recipe.config["backbone_learning_rate"] == pytest.approx(6e-4)
    assert recipe.config["train_epochs"] == 60
    assert recipe.config["warmup_epochs"] == 1
    assert recipe.config["freeze_batch_norm"] is False
    assert recipe.config["lr_step_epochs"] == 20
    assert recipe.config["lr_gamma"] == pytest.approx(0.25)
    assert recipe.config["samples_per_class"] == 0
    assert recipe.config["weight_decay_exclusions"] == "none"
    assert recipe.config["gradient_clip_value"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("dataset", "step"),
    [("cub", 5), ("cars", 10)],
)
def test_proxy_anchor_resnet_reference_recipes_match_author_commands(
    dataset: str,
    step: int,
) -> None:
    recipe = reference_recipe("proxy_anchor", dataset)

    assert recipe.config["backbone_name"] == "resnet50"
    assert recipe.config["batch_size"] == 120
    assert recipe.config["learning_rate"] == pytest.approx(1e-4)
    assert recipe.config["train_epochs"] == 60
    assert recipe.config["warmup_epochs"] == 5
    assert recipe.config["freeze_batch_norm"] is True
    assert recipe.config["freeze_batch_norm_affine"] is True
    assert recipe.config["lr_step_epochs"] == step
    assert recipe.config["lr_gamma"] == pytest.approx(0.5)
    assert recipe.config["samples_per_class"] == 0


@pytest.mark.parametrize(
    (
        "dataset",
        "epochs",
        "lr",
        "lr_ds",
        "hgnn",
        "weight_decay",
        "step",
        "tau",
        "alpha",
        "freeze_bn",
    ),
    [
        ("cub", 40, 1.2e-4, 1e-1, 5.0, 5e-5, 5, 32.0, 1.1, True),
        ("cars", 50, 1e-4, 1e-1, 10.0, 1e-4, 10, 32.0, 0.9, True),
        ("sop", 60, 1e-4, 1e-2, 10.0, 1e-4, 10, 16.0, 2.0, False),
    ],
)
def test_hist_reference_recipes_match_author_commands(
    dataset: str,
    epochs: int,
    lr: float,
    lr_ds: float,
    hgnn: float,
    weight_decay: float,
    step: int,
    tau: float,
    alpha: float,
    freeze_bn: bool,
) -> None:
    recipe = reference_recipe("hist", dataset)

    assert recipe.config["backbone_name"] == "resnet50"
    assert recipe.config["optimizer"] == "adam"
    assert recipe.config["batch_size"] == 32
    assert recipe.config["samples_per_class"] == 0
    assert recipe.config["embedding_layer_norm"] is True
    assert recipe.config["warmup_epochs"] == 1
    assert recipe.config["warmup_is_additional"] is True
    assert recipe.config["schedule_during_warmup"] is False
    assert recipe.config["train_epochs"] == epochs
    assert recipe.config["learning_rate"] == pytest.approx(lr)
    assert recipe.config["backbone_learning_rate"] == pytest.approx(lr)
    assert recipe.config["hist_lr_ds"] == pytest.approx(lr_ds)
    assert recipe.config["hist_lr_hgnn_factor"] == pytest.approx(hgnn)
    assert recipe.config["weight_decay"] == pytest.approx(weight_decay)
    assert recipe.config["lr_step_epochs"] == step
    assert recipe.config["hist_tau"] == pytest.approx(tau)
    assert recipe.config["hist_alpha"] == pytest.approx(alpha)
    assert recipe.config["freeze_batch_norm"] is freeze_bn
    assert recipe.config["freeze_batch_norm_affine"] is freeze_bn


@pytest.mark.parametrize(
    ("method", "dataset"),
    [
        ("proxy_anchor", "inat2018"),
        ("hist", "inshop"),
        ("hist", "inat2018"),
    ],
)
def test_unpublished_pair_cannot_resolve_as_reference(method: str, dataset: str) -> None:
    with pytest.raises(
        RecipeUnavailableError,
        match=rf"no published reference recipe for {method}/{dataset}",
    ):
        reference_recipe(method, dataset)


def test_recipe_digest_is_stable_and_changes_with_config() -> None:
    recipe = reference_recipe("proxy_anchor", "cub")

    assert recipe_digest(recipe) == recipe_digest(recipe.model_copy(deep=True))
    changed = recipe.model_copy(update={"config": {**recipe.config, "learning_rate": 9e-4}})
    assert recipe_digest(changed) != recipe_digest(recipe)


def test_herd_derivation_retains_official_hist_layer_norm() -> None:
    hist = reference_recipe("hist", "cub")
    herd = derive_recipe(hist, "herd")

    assert herd.method_status == "sfora_derived"
    assert herd.derived_from_recipe_id == hist.recipe_id
    assert herd.config["embedding_layer_norm"] is True
    assert herd.delta == {
        "ema_distill_weight": 1.0,
        "ema_momentum": 0.999,
        "ema_distill_tau": 0.1,
    }
    changed = {
        key
        for key in set(hist.config) | set(herd.config)
        if hist.config.get(key) != herd.config.get(key)
    }
    assert changed == {"ema_distill_weight"}
    assert changed <= set(herd.delta)


def test_pa_distill_derivation_changes_only_distillation_fields() -> None:
    proxy_anchor = reference_recipe("proxy_anchor", "inshop")
    distilled = derive_recipe(proxy_anchor, "pa_distill")

    assert distilled.method_status == "sfora_derived"
    assert distilled.derived_from_recipe_id == proxy_anchor.recipe_id
    changed = {
        key
        for key in set(proxy_anchor.config) | set(distilled.config)
        if proxy_anchor.config.get(key) != distilled.config.get(key)
    }
    assert changed == {"ema_distill_weight"}
    assert changed <= set(distilled.delta)


@pytest.mark.parametrize(
    ("method", "dataset"),
    [
        ("proxy_anchor", "cub"),
        ("proxy_anchor", "cars"),
        ("proxy_anchor", "sop"),
        ("proxy_anchor", "inshop"),
        ("hist", "cub"),
        ("hist", "cars"),
        ("hist", "sop"),
    ],
)
def test_reference_recipe_resolves_to_complete_valid_config(
    method: str,
    dataset: str,
) -> None:
    recipe = reference_recipe(method, dataset)

    config = config_for_recipe(recipe)

    assert config.dataset_name == dataset
    assert config.recipe_id == recipe.recipe_id
    assert config.recipe_digest == recipe_digest(recipe)
    assert config.recipe_track == "reference"
    assert config.recipe_base_method == method
    assert config.recipe_source_revision == recipe.provenance.revision
    assert config.recipe_modified_fields == {}


def test_herd_config_differs_from_hist_only_by_declared_delta() -> None:
    hist_recipe = reference_recipe("hist", "cars")
    herd_recipe = derive_recipe(hist_recipe, "herd")

    hist = config_for_recipe(hist_recipe)
    herd = config_for_recipe(herd_recipe)

    ignored_metadata = {
        "recipe_id",
        "recipe_digest",
        "recipe_method_status",
        "recipe_delta",
        "recipe_derived_from_id",
    }
    changed = {
        key
        for key in type(hist).model_fields
        if key not in ignored_metadata and getattr(hist, key) != getattr(herd, key)
    }
    assert changed == {"ema_distill_weight"}
    assert hist.embedding_layer_norm is True
    assert herd.embedding_layer_norm is True


def test_recipe_selection_split_is_deterministic_and_class_disjoint() -> None:
    examples = [
        ImageExample(
            example_id=f"train-{label}-{index}",
            image=[float(label), float(index)],
            label=label,
        )
        for label in range(8)
        for index in range(4)
    ]

    first = class_disjoint_recipe_selection_split(examples, fraction=0.25, seed=7)
    second = class_disjoint_recipe_selection_split(examples, fraction=0.25, seed=7)

    optimization_labels = {example.label for example in first.optimization}
    query_labels = {example.label for example in first.query}
    gallery_labels = {example.label for example in first.gallery}
    assert optimization_labels.isdisjoint(query_labels)
    assert query_labels == gallery_labels
    assert len(query_labels) == 2
    assert [example.example_id for example in first.query] == [
        example.example_id for example in second.query
    ]
    assert [example.example_id for example in first.gallery] == [
        example.example_id for example in second.gallery
    ]


def test_recipe_selection_split_never_includes_external_evaluation_examples() -> None:
    training = [
        ImageExample(example_id=f"train-{label}-{index}", image=[0.0], label=label)
        for label in range(6)
        for index in range(3)
    ]
    evaluation = [
        ImageExample(example_id=f"eval-{label}-{index}", image=[0.0], label=label + 100)
        for label in range(2)
        for index in range(2)
    ]

    split = class_disjoint_recipe_selection_split(training, fraction=0.34, seed=0)

    selected_ids = {
        example.example_id for example in (*split.optimization, *split.query, *split.gallery)
    }
    assert selected_ids.isdisjoint({example.example_id for example in evaluation})


def test_recipe_candidate_ranking_uses_map_then_recall_then_recipe_id() -> None:
    scores = [
        RecipeCandidateScore(recipe_id="z", map_at_r=0.7, recall_at_1=0.8),
        RecipeCandidateScore(recipe_id="b", map_at_r=0.8, recall_at_1=0.7),
        RecipeCandidateScore(recipe_id="a", map_at_r=0.8, recall_at_1=0.7),
        RecipeCandidateScore(recipe_id="c", map_at_r=0.8, recall_at_1=0.9),
    ]

    ranked = rank_recipe_candidates(scores)

    assert [score.recipe_id for score in ranked] == ["c", "a", "b", "z"]


def test_selected_extension_retains_source_config_and_changes_target_metadata() -> None:
    source = reference_recipe("hist", "sop")

    selected = selected_extension_recipe(source, target_dataset="inshop")

    assert selected.track == "selected_extension"
    assert selected.dataset == "inshop"
    assert selected.provenance.source_dataset == "sop"
    assert selected.config == source.config
    assert selected.recipe_id.startswith("hist.inshop.selected-from-sop-")


def test_selection_manifest_persists_full_ranking(
    tmp_path: Path,
) -> None:
    selected = selected_extension_recipe(
        reference_recipe("hist", "sop"),
        target_dataset="inshop",
    )
    scores = [
        RecipeCandidateScore(recipe_id=selected.recipe_id, map_at_r=0.6, recall_at_1=0.7),
        RecipeCandidateScore(recipe_id="hist.inshop.other", map_at_r=0.5, recall_at_1=0.8),
    ]
    output = tmp_path / "selection.json"

    write_selection_manifest(
        output,
        selected_recipe=selected,
        scores=scores,
        selection_seed=0,
        protocol_version="class-disjoint-train-v1",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["winner"]["recipe_id"] == selected.recipe_id
    assert payload["winner"]["digest"] == recipe_digest(selected)
    assert [score["recipe_id"] for score in payload["scores"]] == [
        selected.recipe_id,
        "hist.inshop.other",
    ]
