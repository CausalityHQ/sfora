from __future__ import annotations

import pytest

from sfora.image_recipes import (
    RecipeUnavailableError,
    derive_recipe,
    recipe_digest,
    reference_recipe,
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
