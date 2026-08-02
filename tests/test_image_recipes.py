from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from sfora.data import ImageDatasetName, ImageExample
from sfora.image_recipes import (
    BaseMethod,
    DerivedMethod,
    RecipeCandidateScore,
    RecipeUnavailableError,
    class_disjoint_recipe_selection_split,
    config_for_recipe,
    derive_recipe,
    rank_recipe_candidates,
    recipe_digest,
    reference_recipe,
    resolve_recipe,
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


def test_fiedler_recipe_has_matched_ipc4_control() -> None:
    base = reference_recipe("proxy_anchor", "inshop")
    control = derive_recipe(base, "pa_ipc4")
    candidate = derive_recipe(base, "pa_fiedler")

    assert control.config["samples_per_class"] == 4
    assert candidate.config["samples_per_class"] == 4
    differing = {
        key for key in candidate.config if candidate.config.get(key) != control.config.get(key)
    }
    assert differing == {"fiedler_weight", "fiedler_temperature", "fiedler_min_class_size"}
    assert candidate.config["fiedler_weight"] == pytest.approx(0.05)


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
    ("dataset", "batch_size", "epochs", "milestones"),
    [("cub", 400, 40, (10, 20, 30)), ("cars", 392, 170, (80, 140))],
)
def test_recall_at_k_reference_recipes_match_author_code(
    dataset: str,
    batch_size: int,
    epochs: int,
    milestones: tuple[int, ...],
) -> None:
    recipe = reference_recipe("recall_at_k_surrogate", dataset)

    assert recipe.config["objectives"] == ("recall_at_k_surrogate",)
    assert recipe.config["batch_size"] == batch_size
    assert recipe.config["samples_per_class"] == 4
    assert recipe.config["epoch_sampling_policy"] == "source_exhaustive"
    assert recipe.config["pretrained_weights"] == "legacy_resnet50_19c8e357"
    assert recipe.config["eval_test_interval_epochs"] == 5
    assert recipe.config["eval_test_epoch_offset"] == 1
    assert recipe.config["train_epochs"] == epochs
    assert recipe.config["lr_schedule"] == "multistep"
    assert recipe.config["lr_milestones"] == milestones
    assert recipe.config["lr_gamma"] == pytest.approx(0.3)
    assert recipe.config["head_pooling"] == "gem"
    assert recipe.config["pre_embedding_layer_norm"] is True
    assert recipe.config["recall_at_k_rank_temperature"] == pytest.approx(0.01)
    assert recipe.config["recall_at_k_membership_temperature"] == pytest.approx(1.0)


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


@pytest.mark.parametrize(("narrow", "width"), [("narrow128", 128), ("narrow64", 64)])
def test_narrow_derivation_changes_only_the_embedding_width(narrow: str, width: int) -> None:
    """The capacity-weakened bases must differ from official Proxy Anchor in the
    embedding width and nothing else, or they are not measuring headroom."""
    base = reference_recipe("proxy_anchor", "cub")
    narrowed = derive_recipe(base, cast("DerivedMethod", narrow))

    changed = {
        key
        for key in set(base.config) | set(narrowed.config)
        if base.config.get(key) != narrowed.config.get(key)
    }
    assert changed == {"embedding_dimensions"}
    assert narrowed.config["embedding_dimensions"] == width
    assert base.config["embedding_dimensions"] == 512
    assert changed <= set(narrowed.delta)
    # These overwrite a published hyperparameter rather than adding a term on top of
    # an untouched base, so they must not be readable as a Proxy Anchor reproduction.
    assert base.track == "reference"
    assert narrowed.track == "modified"
    assert derive_recipe(base, cast("DerivedMethod", f"{narrow}_distill")).track == "modified"


@pytest.mark.parametrize(
    ("narrow", "narrow_distill"),
    [("narrow128", "narrow128_distill"), ("narrow64", "narrow64_distill")],
)
def test_narrow_distill_pairs_with_its_own_narrow_base(narrow: str, narrow_distill: str) -> None:
    """Each weakened base is its own paired control: the distilled arm must add the
    same distillation delta as `pa_distill` and change nothing else, so the measured
    effect at that width is comparable to the +0.89 pt measured at 512."""
    base = reference_recipe("proxy_anchor", "cub")
    narrowed = derive_recipe(base, cast("DerivedMethod", narrow))
    distilled = derive_recipe(base, cast("DerivedMethod", narrow_distill))
    full_distill = derive_recipe(base, "pa_distill")

    changed = {
        key
        for key in set(narrowed.config) | set(distilled.config)
        if narrowed.config.get(key) != distilled.config.get(key)
    }
    assert changed == {"ema_distill_weight"}
    # Identical distillation to the full-width arm, so widths are comparable.
    for field in ("ema_distill_weight", "ema_momentum", "ema_distill_tau"):
        assert distilled.config[field] == full_distill.config[field]
    assert distilled.config["embedding_dimensions"] == narrowed.config["embedding_dimensions"]


@pytest.mark.parametrize(
    ("derived", "base_method", "dataset"),
    [
        ("pa_distill_bnfix", "proxy_anchor", "cub"),
        ("pa_distill_bnfix", "proxy_anchor", "inshop"),
        ("herd_bnfix", "hist", "cub"),
        ("herd_bnfix", "hist", "sop"),
    ],
)
def test_bnfix_derivation_adds_only_teacher_normalisation_fields(
    derived: str, base_method: str, dataset: str
) -> None:
    """The `_bnfix` arms differ from their plain counterparts ONLY in the two
    EMA-teacher normalisation flags, so a paired comparison isolates H3."""
    base = reference_recipe(base_method, dataset)
    plain = derive_recipe(base, "pa_distill" if base_method == "proxy_anchor" else "herd")
    fixed = derive_recipe(base, cast("DerivedMethod", derived))

    changed = {
        key
        for key in set(plain.config) | set(fixed.config)
        if plain.config.get(key) != fixed.config.get(key)
    }
    assert changed == {"ema_teacher_train_mode", "ema_teacher_ema_buffers"}
    assert fixed.config["ema_teacher_train_mode"] is True
    assert fixed.config["ema_teacher_ema_buffers"] is True
    # The distillation itself is identical to the plain arm.
    assert fixed.config["ema_distill_weight"] == plain.config["ema_distill_weight"]
    assert fixed.config["ema_distill_tau"] == plain.config["ema_distill_tau"]
    assert changed <= set(fixed.delta)


def test_bnfix_variants_do_not_disturb_existing_derived_digests() -> None:
    """Adding the `_bnfix` recipes must not change `pa_distill`/`herd` digests -- a
    drift there would silently invalidate every in-flight and archived artifact."""
    expected: dict[tuple[BaseMethod, ImageDatasetName, str], str] = {
        ("proxy_anchor", "cub", "auto"): "534a94b64783",
        ("proxy_anchor", "cub", "pa_distill"): "47a0dea213ed",
        ("hist", "cub", "auto"): "4d6f712a59ac",
        ("hist", "cub", "herd"): "9cdd73cfef24",
        ("proxy_anchor", "cars", "auto"): "6fd195249c3a",
        ("proxy_anchor", "cars", "pa_distill"): "0a096881e5cb",
        ("hist", "cars", "auto"): "fa44e0ac30b4",
        ("hist", "cars", "herd"): "65cc88758ed1",
    }
    for (base_method, dataset, selector), digest in expected.items():
        recipe = resolve_recipe(selector, base_method=base_method, dataset=dataset)
        assert recipe_digest(recipe)[:12] == digest, f"digest drift for {selector}/{dataset}"


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
        ("recall_at_k_surrogate", "cub"),
        ("recall_at_k_surrogate", "cars"),
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
    assert config.dataset_selection_policy == "full_official_partition"


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


def test_auto_resolution_requires_manifest_for_unpublished_pair() -> None:
    with pytest.raises(
        RecipeUnavailableError,
        match="selection manifest is required for unpublished hist/inshop",
    ):
        resolve_recipe("auto", base_method="hist", dataset="inshop")


def test_auto_resolution_loads_and_verifies_selected_manifest(tmp_path: Path) -> None:
    selected = selected_extension_recipe(
        reference_recipe("hist", "sop"),
        target_dataset="inshop",
    )
    score = RecipeCandidateScore(
        recipe_id=selected.recipe_id,
        map_at_r=0.6,
        recall_at_1=0.7,
    )
    manifest = tmp_path / "selection.json"
    write_selection_manifest(
        manifest,
        selected_recipe=selected,
        scores=[score],
        selection_seed=0,
        protocol_version="class-disjoint-train-v1",
    )

    resolved = resolve_recipe(
        "auto",
        base_method="hist",
        dataset="inshop",
        selection_manifest=manifest,
    )

    assert resolved.model_dump(mode="json") == selected.model_dump(mode="json")


def test_derived_selector_uses_resolved_base_recipe() -> None:
    resolved = resolve_recipe(
        "pa_distill",
        base_method="proxy_anchor",
        dataset="inshop",
    )

    assert resolved.method_status == "sfora_derived"
    assert resolved.derived_from_recipe_id == "proxy_anchor.inshop.official-51db570"


def test_ema_averaging_momenta_bracket_the_initialisation_contamination() -> None:
    """The two averaging arms exist to disambiguate a null. At 0.999 over CUB's 2940
    steps the EMA still holds ~5% of its initialisation, including a random embedding
    head; at 0.99 that term is ~1e-13. If these momenta ever converge, the pair stops
    bracketing the confound and a null result becomes unreadable."""
    base = reference_recipe("proxy_anchor", "cub")
    slow = derive_recipe(base, "pa_ema_avg")
    fast = derive_recipe(base, "pa_ema_avg_fast")
    steps = 2940

    assert slow.config["ema_momentum"] == pytest.approx(0.999)
    assert fast.config["ema_momentum"] == pytest.approx(0.99)
    assert slow.config["ema_momentum"] ** steps > 0.01  # contaminated on purpose
    assert fast.config["ema_momentum"] ** steps < 1e-6  # initialisation is gone
    # Both must still be pure averaging, with no distillation term.
    for recipe in (slow, fast):
        assert recipe.config["ema_weight_averaging"] is True
        assert recipe.config["ema_distill_weight"] == 0.0


def test_pa_ema_avg_isolates_weight_averaging_from_the_distillation_loss() -> None:
    """The mechanism arm must differ from official Proxy Anchor ONLY by maintaining and
    scoring the EMA copy -- crucially with `ema_distill_weight` still 0. If a
    distillation term ever leaks in, the arm stops separating the two things an EMA
    teacher does and answers nothing."""
    base = reference_recipe("proxy_anchor", "cub")
    averaged = derive_recipe(base, "pa_ema_avg")
    distilled = derive_recipe(base, "pa_distill")

    changed = {
        key
        for key in set(base.config) | set(averaged.config)
        if base.config.get(key) != averaged.config.get(key)
    }
    assert changed == {"ema_weight_averaging"}
    assert averaged.config["ema_weight_averaging"] is True
    assert averaged.config["ema_distill_weight"] == 0.0
    # Same average `pa_distill` builds, so the two arms differ only in the loss term.
    assert averaged.config["ema_momentum"] == distilled.config["ema_momentum"]
    # Absent from the reference config, so it takes the field default of False -- which
    # is what keeps `pa_distill` scoring the student.
    assert distilled.config.get("ema_weight_averaging", False) is False


def test_ema_factorial_arms_vary_only_the_two_teacher_roles_at_matched_momentum() -> None:
    """An EMA teacher supplies two separable things: a distillation target, and an
    averaged copy of the weights to evaluate. These four arms are the 2x2 over those
    roles, and they are only interpretable if momentum is held fixed across the three
    non-base cells -- otherwise a difference between cells could be the momentum."""
    base = reference_recipe("proxy_anchor", "cub")
    average_only = derive_recipe(base, "pa_ema_avg_fast")
    distil_only = derive_recipe(base, "pa_distill_fast")
    both = derive_recipe(base, "pa_distill_avg")

    for arm in (average_only, distil_only, both):
        assert arm.config["ema_momentum"] == pytest.approx(0.99), arm.recipe_id

    assert average_only.config["ema_distill_weight"] == 0.0
    assert average_only.config["ema_weight_averaging"] is True

    assert distil_only.config["ema_distill_weight"] == 1.0
    assert distil_only.config.get("ema_weight_averaging", False) is False

    assert both.config["ema_distill_weight"] == 1.0
    assert both.config["ema_weight_averaging"] is True

    # `both` must be exactly the union of the two single arms, or it does not test
    # additivity -- it tests some third thing.
    assert both.config["ema_distill_weight"] == distil_only.config["ema_distill_weight"]
    assert both.config["ema_distill_tau"] == distil_only.config["ema_distill_tau"]
    assert both.config["ema_weight_averaging"] == average_only.config["ema_weight_averaging"]

    # All four cells must be distinct runs.
    digests = {recipe_digest(r) for r in (base, average_only, distil_only, both)}
    assert len(digests) == 4


@pytest.mark.parametrize(
    ("arm", "momentum"),
    [
        ("pa_ema_avg", 0.999),
        ("pa_ema_avg_fast", 0.99),
        ("pa_ema_avg_m95", 0.95),
        ("pa_ema_avg_m90", 0.90),
    ],
)
def test_averaging_sweep_spans_windows_and_stays_pure_averaging(arm: str, momentum: float) -> None:
    """The sweep is only a curve if the arms differ in momentum ALONE. Any of them
    picking up a distillation term would make its point a different method."""
    base = reference_recipe("proxy_anchor", "cub")
    recipe = derive_recipe(base, cast("DerivedMethod", arm))

    assert recipe.config["ema_momentum"] == pytest.approx(momentum)
    assert recipe.config["ema_weight_averaging"] is True
    assert recipe.config["ema_distill_weight"] == 0.0
    changed = {
        key
        for key in set(base.config) | set(recipe.config)
        if base.config.get(key) != recipe.config.get(key)
    }
    assert changed <= {"ema_momentum", "ema_weight_averaging"}


def test_averaging_sweep_windows_are_actually_distinct() -> None:
    """Guards against a sweep that looks like four points but is really two: the
    averaging windows must separate by more than a factor of two, and all four arms
    must resolve to distinct digests."""
    base = reference_recipe("proxy_anchor", "cub")
    arms = ["pa_ema_avg", "pa_ema_avg_fast", "pa_ema_avg_m95", "pa_ema_avg_m90"]
    recipes = [derive_recipe(base, cast("DerivedMethod", a)) for a in arms]

    windows = [1.0 / (1.0 - r.config["ema_momentum"]) for r in recipes]
    assert windows == sorted(windows, reverse=True)
    for longer, shorter in zip(windows[:-1], windows[1:], strict=True):
        # 1/(1-m) is not exact in binary, so 0.95 vs 0.90 gives 19.999.../10.000...
        assert longer / shorter == pytest.approx(2.0, rel=1e-6) or longer / shorter > 2.0

    assert len({recipe_digest(r) for r in recipes}) == 4


def test_dual_ema_keeps_the_slow_teacher_and_evaluates_a_fast_average() -> None:
    """The arm is only meaningful if it takes the BEST momentum for each role rather than
    compromising on one. Distillation keeps the teacher at 0.999, where it measured +0.91;
    evaluation gets its own 0.99 average, where that role measured +0.45."""
    base = reference_recipe("proxy_anchor", "cub")
    dual = derive_recipe(base, "pa_dual_ema")
    distil_only = derive_recipe(base, "pa_distill")
    average_only = derive_recipe(base, "pa_ema_avg_fast")

    # Distillation half is byte-identical to pa_distill.
    for field in ("ema_distill_weight", "ema_momentum", "ema_distill_tau"):
        assert dual.config[field] == distil_only.config[field], field
    # Evaluation half runs at the averaging arm's momentum, not the teacher's.
    assert dual.config["ema_weight_averaging"] is True
    assert dual.config["ema_eval_momentum"] == pytest.approx(0.99)
    assert dual.config["ema_eval_momentum"] == pytest.approx(average_only.config["ema_momentum"])
    # The two timescales must actually differ, or this is just pa_distill.
    assert dual.config["ema_eval_momentum"] != dual.config["ema_momentum"]
    assert recipe_digest(dual) not in {recipe_digest(distil_only), recipe_digest(average_only)}


def test_dual_ema_bnfix_is_safe_for_trainable_batch_norm() -> None:
    """Both EMA roles need coherent buffers, and the training teacher needs batch-mode
    normalization, on an In-Shop base whose BatchNorm statistics update."""
    base = reference_recipe("proxy_anchor", "inshop")
    plain = derive_recipe(base, "pa_dual_ema")
    fixed = derive_recipe(base, "pa_dual_ema_bnfix")

    assert base.config["freeze_batch_norm"] is False
    assert fixed.config["ema_momentum"] == pytest.approx(0.999)
    assert fixed.config["ema_eval_momentum"] == pytest.approx(0.99)
    assert fixed.config["ema_teacher_train_mode"] is True
    assert fixed.config["ema_teacher_ema_buffers"] is True

    changed = {
        key
        for key in set(plain.config) | set(fixed.config)
        if plain.config.get(key) != fixed.config.get(key)
    }
    assert changed == {"ema_teacher_train_mode", "ema_teacher_ema_buffers"}
    assert recipe_digest(plain) != recipe_digest(fixed)


def test_averaging_on_trainable_batch_norm_requires_the_buffer_fix() -> None:
    """Weight averaging evaluates the EMA copy, so its BatchNorm statistics must be
    averaged too. `_update_ema_teacher` hard-copies buffers by default, which would pair
    averaged WEIGHTS with the student's last-step running statistics -- half an average.

    On CUB this is inert (`freeze_batch_norm=True` stops the student's buffers moving),
    which is why `pa_ema_avg_fast` is valid there. On In-Shop BatchNorm is trainable, so
    the arm taken there must be `pa_ema_avg_bnfix` or the run measures the confound
    instead of the method."""
    cub = reference_recipe("proxy_anchor", "cub")
    inshop = reference_recipe("proxy_anchor", "inshop")

    assert cub.config["freeze_batch_norm"] is True
    assert inshop.config["freeze_batch_norm"] is False

    plain = derive_recipe(inshop, "pa_ema_avg_fast")
    fixed = derive_recipe(inshop, "pa_ema_avg_bnfix")

    assert plain.config.get("ema_teacher_ema_buffers", False) is False
    assert fixed.config["ema_teacher_ema_buffers"] is True

    # Identical in every other respect, so the pair isolates the buffer treatment.
    changed = {
        key
        for key in set(plain.config) | set(fixed.config)
        if plain.config.get(key) != fixed.config.get(key)
    }
    assert changed == {"ema_teacher_ema_buffers"}
    assert fixed.config["ema_weight_averaging"] is True
    assert fixed.config["ema_distill_weight"] == 0.0
    assert fixed.config["ema_momentum"] == pytest.approx(0.99)
    assert recipe_digest(plain) != recipe_digest(fixed)


def test_rspg_recipe_is_training_data_only_proxy_anchor_derivation() -> None:
    base = reference_recipe("proxy_anchor", "inshop")
    recipe = derive_recipe(base, "rspg")

    assert recipe.config["rspg_weight"] == 1.0
    assert recipe.config["ema_momentum"] == 0.99
    assert recipe.config["ema_teacher_ema_buffers"] is True
    assert recipe.config["ema_distill_weight"] == 0.0


def test_arcg_recipe_freezes_preregistered_graph_schedule() -> None:
    base = reference_recipe("proxy_anchor", "inshop")
    recipe = derive_recipe(base, "arcg")

    assert recipe.config["arcg_weight"] == 1.0
    assert recipe.config["arcg_warmup_epoch"] == 10
    assert recipe.config["arcg_refresh_epoch"] == 40
    assert recipe.config["arcg_agreement_threshold"] == 0.5
    assert set(recipe.delta) == {
        "arcg_weight",
        "arcg_warmup_epoch",
        "arcg_refresh_epoch",
        "arcg_agreement_threshold",
    }


def test_ipsr_recipe_freezes_preregistered_preference_schedule() -> None:
    base = reference_recipe("proxy_anchor", "inshop")
    recipe = derive_recipe(base, "ipsr")

    assert recipe.config["ipsr_weight"] == 1.0
    assert recipe.config["ipsr_warmup_epoch"] == 10
    assert recipe.config["ipsr_refresh_epoch"] == 40
    assert recipe.config["ipsr_agreement_threshold"] == 0.5
    assert set(recipe.delta) == {
        "ipsr_weight",
        "ipsr_warmup_epoch",
        "ipsr_refresh_epoch",
        "ipsr_agreement_threshold",
    }


def test_tird_recipe_is_normalization_consistent_and_has_no_ordinary_distillation() -> None:
    base = reference_recipe("proxy_anchor", "inshop")
    recipe = derive_recipe(base, "tird")

    assert recipe.config["tird_weight"] == 1.0
    assert recipe.config["ema_momentum"] == 0.999
    assert recipe.config["ema_teacher_train_mode"] is True
    assert recipe.config["ema_teacher_ema_buffers"] is True
    assert recipe.config["ema_distill_weight"] == 0.0
    assert set(recipe.delta) == {
        "tird_weight",
        "ema_momentum",
        "ema_teacher_train_mode",
        "ema_teacher_ema_buffers",
    }


@pytest.mark.parametrize(
    ("method", "control"),
    [
        ("rspg_soft_js", "soft_js"),
        ("rspg_distance_gate", "distance_gate"),
        ("rspg_instance_gate", "instance_gate"),
    ],
)
def test_rspg_ablation_recipe_changes_only_registered_control(method: str, control: str) -> None:
    base = reference_recipe("proxy_anchor", "inshop")
    full = derive_recipe(base, "rspg")
    ablation = derive_recipe(base, method)

    assert full.config.get("rspg_control", "signature_gate") == "signature_gate"
    assert ablation.config["rspg_control"] == control
    changed = {
        key
        for key in set(full.config) | set(ablation.config)
        if full.config.get(key) != ablation.config.get(key)
    }
    assert changed == {"rspg_control"}
