import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from sfora.data import ImageExample
from sfora.image_end_to_end import (
    EndToEndProtocol,
    ImageEndToEndConfig,
    config_for_protocol,
    run_image_end_to_end_benchmark,
)
from sfora.report import ReportConfig, build_site_data


def test_sota_protocol_uses_resnet50_512_adam_epochs() -> None:
    config = config_for_protocol("sota-resnet50-512", dataset_name="cub")

    assert config.backbone_name == "resnet50"
    assert config.embedding_dimensions == 512
    assert config.batch_size == 120
    assert config.optimizer == "adam"
    assert config.learning_rate == pytest.approx(5e-4)
    assert config.backbone_learning_rate == pytest.approx(1e-5)
    assert config.triplet_margin == pytest.approx(0.2)
    assert config.train_epochs == 80
    assert config.objectives == ("group_supcon_xbm_radius",)


def test_proxy_anchor_protocol_uses_repaired_resnet50_512_defaults() -> None:
    config = config_for_protocol("proxy-anchor-resnet50-512", dataset_name="cub")

    assert config.objectives == ("frozen_pretrained", "proxy_anchor")
    assert config.optimizer == "adamw"
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.backbone_learning_rate == pytest.approx(1e-4)
    assert config.weight_decay == pytest.approx(1e-4)
    assert config.warmup_epochs == 5
    assert config.lr_schedule == "step"
    assert config.lr_step_epochs == 5
    assert config.lr_gamma == pytest.approx(0.5)
    assert config.train_epochs == 60
    assert config.samples_per_class == 4
    assert config.batch_size == 120
    assert config.pretrained_weights == "v1"
    assert config.head_pooling == "avg_max"
    assert config.embedding_head_init == "kaiming_normal"
    assert config.proxy_count_per_class == 1
    assert config.proxy_anchor_alpha == pytest.approx(32.0)
    assert config.proxy_anchor_delta == pytest.approx(0.1)
    assert config.checkpoint_selection_interval == 0


def test_proxy_anchor_protocol_uses_longer_step_schedule_for_cars_and_sop() -> None:
    cars = config_for_protocol("proxy-anchor-resnet50-512", dataset_name="cars")
    sop = config_for_protocol("proxy-anchor-resnet50-512", dataset_name="sop")

    assert cars.lr_step_epochs == 10
    assert sop.lr_step_epochs == 10


def test_proxy_anchor_protocol_train_steps_override_disables_epoch_schedule() -> None:
    config = config_for_protocol("proxy-anchor-resnet50-512", dataset_name="cub", train_steps=37)

    assert config.train_steps == 37
    assert config.train_epochs is None


def test_pfml_protocol_uses_repaired_resnet50_512_defaults() -> None:
    config = config_for_protocol("pfml-resnet50-512", dataset_name="cub")

    assert config.objectives == ("frozen_pretrained", "pfml")
    assert config.optimizer == "adam"
    assert config.learning_rate == pytest.approx(5e-4)
    assert config.backbone_learning_rate == pytest.approx(5e-4)
    assert config.weight_decay == pytest.approx(1e-4)
    assert config.warmup_epochs == 5
    assert config.lr_schedule == "cosine"
    assert config.lr_step_epochs == 5
    assert config.lr_gamma == pytest.approx(0.5)
    assert config.train_epochs == 100
    assert config.samples_per_class == 4
    assert config.batch_size == 120
    assert config.pretrained_weights == "v1"
    assert config.head_pooling == "avg_max"
    assert config.embedding_head_init == "kaiming_normal"
    assert config.proxy_count_per_class == 15
    assert config.potential_delta == pytest.approx(0.2)
    assert config.potential_alpha == pytest.approx(4.0)
    assert config.checkpoint_selection_interval == 0


def test_pfml_protocol_uses_two_sop_proxies_per_class() -> None:
    config = config_for_protocol("pfml-resnet50-512", dataset_name="sop")

    assert config.proxy_count_per_class == 2


def test_pfml_protocol_train_steps_override_disables_epoch_schedule() -> None:
    config = config_for_protocol("pfml-resnet50-512", dataset_name="cub", train_steps=37)

    assert config.train_steps == 37
    assert config.train_epochs is None


def test_legacy_protocols_keep_inert_protocol_repair_defaults() -> None:
    for protocol in ("sota-resnet50-512", "hpl-resnet50-512"):
        config = config_for_protocol(protocol, dataset_name="cub")

        assert config.warmup_epochs == 0
        assert config.lr_schedule == "none"
        assert config.samples_per_class == 0
        assert config.pretrained_weights == "v2"
        assert config.head_pooling == "avg"
        assert config.embedding_head_init == "default"
        assert config.xbm_start_step == 0


def test_end_to_end_config_accepts_frozen_baseline_objective() -> None:
    config = ImageEndToEndConfig(
        dataset_name="cub",
        protocol="sota-resnet50-512",
        objectives=(
            "frozen_pretrained",
            "frozen",
            "triplet",
            "triplet_pretrained",
            "batch_hard_triplet",
            "supcon",
        ),
    )

    assert config.objectives == (
        "frozen_pretrained",
        "frozen",
        "triplet",
        "triplet_pretrained",
        "batch_hard_triplet",
        "supcon",
    )


def test_end_to_end_config_accepts_group_potential_objectives() -> None:
    config = ImageEndToEndConfig(
        dataset_name="cub",
        protocol="sota-resnet50-512",
        objectives=("group_potential", "group_potential_xbm"),
    )

    assert config.objectives == ("group_potential", "group_potential_xbm")


def test_end_to_end_config_accepts_checkpoint_selection_knobs() -> None:
    config = ImageEndToEndConfig(
        checkpoint_selection_interval=250,
        checkpoint_selection_query_limit=64,
        checkpoint_selection_metric="map_at_r",
    )

    assert config.checkpoint_selection_interval == 250
    assert config.checkpoint_selection_query_limit == 64
    assert config.checkpoint_selection_metric == "map_at_r"


def test_config_exposes_protocol_repair_fields() -> None:
    config = ImageEndToEndConfig()

    assert config.optimizer == "adam"
    assert config.warmup_epochs == 0
    assert config.lr_schedule == "none"
    assert config.lr_step_epochs == 5
    assert config.lr_gamma == 0.5
    assert config.samples_per_class == 0
    assert config.pretrained_weights == "v2"
    assert config.head_pooling == "avg"
    assert config.xbm_start_step == 0
    assert config.embedding_head_init == "default"


def test_config_accepts_adamw_and_cosine() -> None:
    config = ImageEndToEndConfig(optimizer="adamw", lr_schedule="cosine", warmup_epochs=5)

    assert config.optimizer == "adamw"


def test_trainable_pretrained_triplet_uses_pretrained_feature_model() -> None:
    from sfora.image_end_to_end import _uses_pretrained_feature_model

    assert _uses_pretrained_feature_model("frozen_pretrained", "resnet50", None) is True
    assert _uses_pretrained_feature_model("triplet_pretrained", "resnet50", None) is True
    assert _uses_pretrained_feature_model("triplet", "resnet50", None) is False


def test_batch_hard_triplet_loss_rewards_separated_classes() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _batch_hard_triplet_loss

    separated = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [-1.0, 0.0],
                [-0.9, -0.1],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    mixed = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.9, 0.1],
                [-0.9, -0.1],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    labels = torch.tensor([0, 0, 1, 1])

    separated_loss = _batch_hard_triplet_loss(
        separated,
        labels,
        margin=0.2,
        torch_module=torch,
    )
    mixed_loss = _batch_hard_triplet_loss(
        mixed,
        labels,
        margin=0.2,
        torch_module=torch,
    )

    assert separated_loss < mixed_loss


def test_triplet_objective_uses_semi_hard_mining_not_batch_hard() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.9396926, 0.3420201],
                [0.5000000, 0.8660254],
                [0.9063078, 0.4226183],
                [0.1736482, 0.9848078],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    config = ImageEndToEndConfig(triplet_margin=0.2)

    triplet_loss = _loss_for_objective(
        "triplet",
        embeddings,
        labels,
        step=1,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=None,
        proxy_labels=None,
        config=config,
        torch_module=torch,
    )
    batch_hard_loss = _loss_for_objective(
        "batch_hard_triplet",
        embeddings,
        labels,
        step=1,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=None,
        proxy_labels=None,
        config=config,
        torch_module=torch,
    )

    assert triplet_loss != pytest.approx(float(batch_hard_loss.detach().cpu()))


def test_teacher_similarity_loss_preserves_pairwise_geometry_across_dimensions() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _pairwise_similarity_preservation_loss

    student = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32),
        dim=-1,
    )
    teacher_same_geometry = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    teacher_different_geometry = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        dim=-1,
    )

    preserved = _pairwise_similarity_preservation_loss(
        student,
        teacher_same_geometry,
        torch_module=torch,
    )
    changed = _pairwise_similarity_preservation_loss(
        student,
        teacher_different_geometry,
        torch_module=torch,
    )

    assert preserved == pytest.approx(0.0)
    assert changed > preserved


def test_loss_for_objective_can_add_teacher_similarity_regularization() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [-0.8, -0.2]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    labels = torch.tensor([0, 0, 1, 1])
    teacher_embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    base_config = ImageEndToEndConfig(teacher_similarity_weight=0.0)
    regularized_config = ImageEndToEndConfig(teacher_similarity_weight=2.0)

    base_loss = _loss_for_objective(
        "triplet",
        embeddings,
        labels,
        step=1,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=None,
        proxy_labels=None,
        teacher_embeddings=teacher_embeddings,
        config=base_config,
        torch_module=torch,
    )
    regularized_loss = _loss_for_objective(
        "triplet",
        embeddings,
        labels,
        step=1,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=None,
        proxy_labels=None,
        teacher_embeddings=teacher_embeddings,
        config=regularized_config,
        torch_module=torch,
    )

    assert regularized_loss > base_loss


def test_local_potential_loss_rewards_close_positives_and_separated_negatives() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _local_potential_loss

    good_embeddings = torch.tensor(
        [
            [0.00, 0.00],
            [0.10, 0.00],
            [2.00, 0.00],
            [2.10, 0.00],
        ],
        dtype=torch.float32,
    )
    bad_embeddings = torch.tensor(
        [
            [0.00, 0.00],
            [2.00, 0.00],
            [0.05, 0.00],
            [2.05, 0.00],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1])

    good_loss = _local_potential_loss(
        good_embeddings,
        labels,
        delta=0.3,
        alpha=4.0,
        torch_module=torch,
    )
    bad_loss = _local_potential_loss(
        bad_embeddings,
        labels,
        delta=0.3,
        alpha=4.0,
        torch_module=torch,
    )

    assert good_loss < bad_loss


def test_group_supcon_xbm_radius_can_add_local_potential_with_proxies() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    labels = torch.tensor([0, 0, 1, 1])
    proxies = torch.nn.Parameter(
        torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32),
            dim=-1,
        )
    )
    proxy_labels = torch.tensor([0, 1])
    base_config = ImageEndToEndConfig(
        proxy_weight=0.0,
        potential_weight=0.0,
    )
    potential_config = ImageEndToEndConfig(
        proxy_weight=0.0,
        potential_weight=0.5,
        potential_delta=0.3,
        potential_alpha=4.0,
    )

    base_loss = _loss_for_objective(
        "group_supcon_xbm_radius",
        embeddings,
        labels,
        step=1,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        config=base_config,
        torch_module=torch,
    )
    potential_loss = _loss_for_objective(
        "group_supcon_xbm_radius",
        embeddings,
        labels,
        step=1,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        config=potential_config,
        torch_module=torch,
    )

    assert float(potential_loss.detach().cpu()) != pytest.approx(float(base_loss.detach().cpu()))


def test_group_potential_objective_uses_proxies_and_updates_them() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.92, 0.08], [-1.0, 0.0], [-0.92, -0.08]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    labels = torch.tensor([0, 0, 1, 1])
    proxies = torch.nn.Parameter(
        torch.nn.functional.normalize(
            torch.tensor([[0.7, 0.7], [-0.7, -0.7]], dtype=torch.float32),
            dim=-1,
        )
    )
    config = ImageEndToEndConfig(
        group_size=2,
        point_weight=0.25,
        group_weight=1.0,
        proxy_weight=1.0,
        potential_weight=1.0,
        potential_delta=0.3,
        potential_alpha=4.0,
    )

    loss = _loss_for_objective(
        "group_potential",
        embeddings,
        labels,
        step=1,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=proxies,
        proxy_labels=torch.tensor([0, 1]),
        config=config,
        torch_module=torch,
    )
    loss.backward()

    assert proxies.grad is not None
    assert float(proxies.grad.norm().detach().cpu()) > 0.0


def test_group_potential_loss_prefers_local_class_structure() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    labels = torch.tensor([0, 0, 1, 1])
    good_embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.94, 0.06], [-1.0, 0.0], [-0.94, -0.06]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    bad_embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [-0.94, -0.06], [0.94, 0.06], [-1.0, 0.0]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    proxies = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32),
        dim=-1,
    )
    config = ImageEndToEndConfig(
        group_size=2,
        point_weight=0.25,
        group_weight=1.0,
        proxy_weight=1.0,
        potential_weight=1.0,
        potential_delta=0.3,
        potential_alpha=4.0,
    )

    good_loss = _loss_for_objective(
        "group_potential",
        good_embeddings,
        labels,
        step=1,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=proxies,
        proxy_labels=torch.tensor([0, 1]),
        config=config,
        torch_module=torch,
    )
    bad_loss = _loss_for_objective(
        "group_potential",
        bad_embeddings,
        labels,
        step=1,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=proxies,
        proxy_labels=torch.tensor([0, 1]),
        config=config,
        torch_module=torch,
    )

    assert good_loss < bad_loss


def test_pfml_objective_config_and_display_name() -> None:
    from sfora.image_end_to_end import _objective_display_name

    config = ImageEndToEndConfig(objectives=("pfml",))

    assert config.objectives == ("pfml",)
    assert _objective_display_name("pfml") == "PFML (Potential Field)"


def test_config_accepts_zero_potential_alpha() -> None:
    # The PFML paper cross-validates alpha in {0..6}, so alpha=0 must be valid.
    config = ImageEndToEndConfig(potential_alpha=0.0)

    assert config.potential_alpha == 0.0


def test_proxy_anchor_objective_config_and_display_name() -> None:
    from sfora.image_end_to_end import _objective_display_name

    config = ImageEndToEndConfig(objectives=("proxy_anchor",))

    assert config.objectives == ("proxy_anchor",)
    assert config.proxy_anchor_alpha == pytest.approx(32.0)
    assert config.proxy_anchor_delta == pytest.approx(0.1)
    assert _objective_display_name("proxy_anchor") == "Proxy Anchor"


def test_hist_loss_trains_and_flows_gradients_to_embeddings_and_modules() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _build_hist_module, _hist_loss, _normalize

    label_to_index = {0: 0, 1: 1}
    hist_module = _build_hist_module(nb_classes=2, sz_embed=4, hidden=8, torch_module=torch)
    # Two clearly separable clusters so the softmax stays well-conditioned.
    base = torch.tensor([[1.0, 0.5, 0.0, 0.0], [-1.0, 0.5, 0.0, 0.0]], dtype=torch.float32)
    noise = 0.1 * torch.randn(8, 4, generator=torch.Generator().manual_seed(0))
    raw = (base.repeat_interleave(4, dim=0) + noise).clone().requires_grad_(True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    optimizer = torch.optim.Adam([raw, *hist_module.parameters()], lr=0.02)
    losses = []
    for _ in range(25):
        optimizer.zero_grad()
        loss = _hist_loss(
            _normalize(raw, torch),
            labels,
            hist_module=hist_module,
            label_to_index=label_to_index,
            tau=8.0,
            alpha=0.9,
            lambda_s=1.0,
            var_floor=0.0,
            torch_module=torch,
        )
        loss.backward()
        grad = raw.grad
        assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0.0
        assert any(
            p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in hist_module.parameters()
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_hist_var_floor_default_matches_relu6_and_knob_changes_it() -> None:
    torch: Any = pytest.importorskip("torch")
    from sfora.image_end_to_end import _build_hist_module, _hist_loss

    torch.manual_seed(0)
    raw = torch.randn(12, 8, requires_grad=False)
    labels = torch.tensor([0, 1, 2] * 4)
    label_to_index = {0: 0, 1: 1, 2: 2}
    module = _build_hist_module(nb_classes=3, sz_embed=8, hidden=8, torch_module=torch)
    # Force some negative log-variances so the floor actually bites.
    with torch.no_grad():
        module.log_vars.copy_(torch.full_like(module.log_vars, -2.0))

    def loss_with(floor: float) -> float:
        return float(
            _hist_loss(
                raw,
                labels,
                hist_module=module,
                label_to_index=label_to_index,
                tau=8.0,
                alpha=0.9,
                lambda_s=1.0,
                var_floor=floor,
                torch_module=torch,
            ).detach()
        )

    faithful = loss_with(0.0)
    relu6_ref = float(
        _hist_loss(
            raw,
            labels,
            hist_module=module,
            label_to_index=label_to_index,
            tau=8.0,
            alpha=0.9,
            lambda_s=1.0,
            var_floor=0.0,
            torch_module=torch,
        ).detach()
    )
    # Default floor 0.0 clamps the -2.0 log-vars up to 0.0 exactly like relu6.
    assert faithful == relu6_ref
    # A negative floor lets the -2.0 log-vars through, changing the loss.
    assert loss_with(-3.0) != faithful


def test_fused_hist_proxy_anchor_loss_sums_both_terms() -> None:
    torch: Any = pytest.importorskip("torch")
    from sfora.image_end_to_end import (
        ImageEndToEndConfig,
        _build_hist_module,
        _hist_loss,
        _loss_for_objective,
        _proxy_anchor_loss,
    )

    torch.manual_seed(0)
    emb = torch.randn(12, 8)
    labels = torch.tensor([0, 1, 2] * 4)
    hist = _build_hist_module(nb_classes=3, sz_embed=8, hidden=8, torch_module=torch)
    proxies = torch.randn(3, 8)
    plabels = torch.tensor([0, 1, 2])
    l2i = {0: 0, 1: 1, 2: 2}
    cfg = ImageEndToEndConfig(proxy_fusion_weight=0.5)

    fused = _loss_for_objective(
        "hist_proxy_anchor",
        emb,
        labels,
        step=0,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=proxies,
        proxy_labels=plabels,
        config=cfg,
        torch_module=torch,
        hist_module=hist,
        hist_label_to_index=l2i,
    )
    hist_term = _hist_loss(
        emb,
        labels,
        hist_module=hist,
        label_to_index=l2i,
        tau=cfg.hist_tau,
        alpha=cfg.hist_alpha,
        lambda_s=cfg.hist_lambda_s,
        var_floor=cfg.hist_var_floor,
        torch_module=torch,
    )
    proxy_term = _proxy_anchor_loss(
        emb,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=plabels,
        alpha=cfg.proxy_anchor_alpha,
        delta=cfg.proxy_anchor_delta,
        torch_module=torch,
    )
    assert torch.allclose(fused, hist_term + 0.5 * proxy_term)


def test_hist_objective_end_to_end_runs() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)

        def forward(self, images: object) -> object:
            return self.embedding(torch.as_tensor(images, dtype=torch.long))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return int(cast(int, image))

        return transform

    examples = [
        ImageExample(example_id=f"{label}-{index}", image=label * 4 + index, label=label)
        for label in range(2)
        for index in range(4)
    ]
    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="proxy-anchor-resnet50-512",
            objectives=("hist",),
            backbone_name="tiny",
            embedding_dimensions=4,
            batch_size=8,
            samples_per_class=4,
            hist_hidden=8,
            eval_batch_size=8,
            train_steps=2,
            train_epochs=None,
            warmup_epochs=0,
            retrieval_query_limit=8,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )
    assert "hist_end_to_end:tiny" in result.methods


def test_custom_sampler_and_custom_loss_plugins_are_invoked() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)

        def forward(self, images: object) -> object:
            return self.embedding(torch.as_tensor(images, dtype=torch.long))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        return lambda image: int(cast(int, image))

    examples = [
        ImageExample(example_id=f"{label}-{index}", image=label * 4 + index, label=label)
        for label in range(2)
        for index in range(4)
    ]
    calls = {"sampler": 0, "loss": 0}

    def sampler_factory(labels: object, config: ImageEndToEndConfig) -> list[list[int]]:
        calls["sampler"] += 1
        return [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]

    def custom_loss(embeddings: Any, labels: Any, config: Any, torch_module: Any) -> Any:
        calls["loss"] += 1
        return (embeddings * embeddings).sum()

    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="proxy-anchor-resnet50-512",
            objectives=("custom",),
            backbone_name="tiny",
            embedding_dimensions=4,
            batch_size=8,
            samples_per_class=0,
            eval_batch_size=8,
            train_steps=2,
            train_epochs=None,
            warmup_epochs=0,
            retrieval_query_limit=8,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
        sampler_factory=sampler_factory,
        custom_losses={"custom": custom_loss},
    )
    assert calls["sampler"] == 1  # the custom batch-mining strategy was used
    assert calls["loss"] > 0  # the custom loss was dispatched each step
    assert any("custom" in name for name in result.methods)


def test_mead_assignment_distillation_loss_lower_when_matched() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _mead_assignment_distillation_loss

    prototypes = torch.eye(2, dtype=torch.float32)
    center = torch.zeros(2, dtype=torch.float32)
    teacher_view = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    teacher_globals = torch.cat([teacher_view, teacher_view], dim=0)
    matched_views = [
        teacher_view.clone().requires_grad_(True),
        teacher_view.clone().requires_grad_(True),
        teacher_view.clone().requires_grad_(True),
    ]
    mismatched_view = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    mismatched_views = [
        mismatched_view.clone().requires_grad_(True),
        mismatched_view.clone().requires_grad_(True),
        mismatched_view.clone().requires_grad_(True),
    ]

    matched_loss = _mead_assignment_distillation_loss(
        matched_views,
        teacher_globals,
        prototypes,
        center,
        tau_teacher=0.05,
        tau_student=0.1,
        torch_module=torch,
    )
    mismatched_loss = _mead_assignment_distillation_loss(
        mismatched_views,
        teacher_globals,
        prototypes,
        center,
        tau_teacher=0.05,
        tau_student=0.1,
        torch_module=torch,
    )
    matched_loss.backward()  # type: ignore[no-untyped-call]

    assert float(matched_loss.detach()) < float(mismatched_loss.detach())
    for view in matched_views:
        assert view.grad is not None
        assert torch.isfinite(view.grad).all()


def test_mead_multicrop_transform_shape() -> None:
    torch: Any = pytest.importorskip("torch")
    pytest.importorskip("torchvision.transforms")
    pil_image = pytest.importorskip("PIL.Image")

    from sfora.image_end_to_end import _default_transform_factory

    transform = _default_transform_factory(
        ImageEndToEndConfig(mead_weight=1.0, mead_local_crops=3, mead_local_size=96),
        True,
    )
    crops = transform(pil_image.new("RGB", (320, 320), color="white"))

    assert isinstance(crops, tuple)
    global_crops, local_crops = crops
    assert torch.is_tensor(global_crops)
    assert torch.is_tensor(local_crops)
    assert tuple(global_crops.shape) == (2, 3, 224, 224)
    assert tuple(local_crops.shape) == (3, 3, 96, 96)

    empty_local_transform = _default_transform_factory(
        ImageEndToEndConfig(mead_weight=1.0, mead_local_crops=0, mead_local_size=96),
        True,
    )
    empty_global_crops, empty_local_crops = empty_local_transform(
        pil_image.new("RGB", (320, 320), color="white")
    )
    assert tuple(empty_global_crops.shape) == (2, 3, 224, 224)
    assert tuple(empty_local_crops.shape) == (0, 3, 96, 96)


def test_mead_multicrop_collate_stacks_global_and_local_views() -> None:
    torch: Any = pytest.importorskip("torch")

    from sfora.image_end_to_end import _mead_multicrop_collate

    samples: list[tuple[tuple[Any, Any], int]] = []
    for label in range(2):
        global_crops = torch.full((2, 3, 224, 224), float(label), dtype=torch.float32)
        local_crops = torch.full((3, 3, 96, 96), float(label + 10), dtype=torch.float32)
        samples.append(((global_crops, local_crops), label))

    (global_batch, local_batch), labels = _mead_multicrop_collate(samples)

    assert tuple(global_batch.shape) == (2, 2, 3, 224, 224)
    assert tuple(local_batch.shape) == (2, 3, 3, 96, 96)
    assert labels.tolist() == [0, 1]


def test_mead_end_to_end_smoke() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(1, 4)

        def forward(self, images: object) -> object:
            tensor = torch.as_tensor(images, dtype=torch.float32)
            pooled = tensor.mean(dim=tuple(range(1, tensor.ndim))).unsqueeze(1)
            return self.projection(pooled)

    def transform_factory(
        config: ImageEndToEndConfig,
        train: bool,
    ) -> Callable[[object], object]:
        def transform(image: object) -> object:
            value = int(cast(int, image))
            pixel_value = float(value) / 10.0
            if train and config.mead_weight > 0.0:
                global_crops = torch.full((2, 3, 224, 224), pixel_value, dtype=torch.float32)
                local_crops = torch.full(
                    (
                        config.mead_local_crops,
                        3,
                        config.mead_local_size,
                        config.mead_local_size,
                    ),
                    pixel_value,
                    dtype=torch.float32,
                )
                return global_crops, local_crops
            return torch.full((3, 224, 224), pixel_value, dtype=torch.float32)

        return transform

    examples = [
        ImageExample(example_id=f"{label}-{index}", image=label * 4 + index, label=label)
        for label in range(2)
        for index in range(4)
    ]
    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="proxy-anchor-resnet50-512",
            objectives=("hist",),
            backbone_name="tiny",
            embedding_dimensions=4,
            batch_size=8,
            samples_per_class=4,
            hist_hidden=8,
            eval_batch_size=8,
            train_steps=2,
            train_epochs=None,
            warmup_epochs=0,
            retrieval_query_limit=8,
            progress_every=0,
            num_workers=0,
            mead_weight=1.0,
            mead_local_crops=1,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    assert "hist_end_to_end:tiny" in result.methods


def test_eval_test_interval_records_best_test_recall_over_training() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)

        def forward(self, images: object) -> object:
            return self.embedding(torch.as_tensor(images, dtype=torch.long))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return int(cast(int, image))

        return transform

    examples = [
        ImageExample(example_id=f"{label}-{index}", image=label * 4 + index, label=label)
        for label in range(2)
        for index in range(4)
    ]
    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="proxy-anchor-resnet50-512",
            objectives=("supcon",),
            backbone_name="tiny",
            embedding_dimensions=4,
            batch_size=8,
            samples_per_class=4,
            eval_batch_size=8,
            train_steps=2,
            train_epochs=None,
            warmup_epochs=0,
            eval_test_interval_epochs=1,
            retrieval_query_limit=8,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )
    metrics = result.methods["supcon_end_to_end:tiny"]
    # The per-epoch test evaluation ran and recorded a best-over-training R@1.
    assert metrics.best_test_recall_at_1 is not None
    assert metrics.best_test_epoch is not None
    assert metrics.test_recall_history is not None
    assert len(metrics.test_recall_history) >= 1
    # Best is at least as good as any recorded epoch value.
    assert metrics.best_test_recall_at_1 >= max(metrics.test_recall_history)


def test_embedding_layer_norm_head_centers_and_standardizes() -> None:
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("torchvision.models")
    from sfora.image_end_to_end import _set_resnet_output_layer

    config = ImageEndToEndConfig(
        embedding_dimensions=8,
        head_pooling="avg_max",
        embedding_head_init="kaiming_normal",
        embedding_layer_norm=True,
    )
    model = models.resnet50(weights=None)
    _set_resnet_output_layer(model, config, use_embedding_head=True, torch_module=torch)
    # The head is a Sequential(Linear, LayerNorm) and its trainable params keep the
    # ``fc.`` prefix so optimizer / warmup routing still classifies them as the head.
    assert type(model.fc).__name__ == "Sequential"
    head_param_names = [name for name, _ in model.named_parameters() if name.startswith("fc.")]
    assert head_param_names == ["fc.0.weight", "fc.0.bias"]
    pooled = torch.randn(4, 2048)
    out = model.fc(pooled)
    assert out.shape == (4, 8)
    # LayerNorm(no affine) centers and standardizes each embedding across its dims.
    assert out.mean(dim=1).abs().max().item() < 1.0e-5
    assert (out.std(dim=1, unbiased=False) - 1.0).abs().max().item() < 1.0e-4


def test_bio_physical_bond_affinity_peaks_at_equilibrium_distance() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _bio_physical_bond_loss

    # A single sample and its own-class proxy: the Proxy-Anchor-style positive loss
    # is lowest when the sample sits near the LJ equilibrium bond distance sigma from
    # the proxy, higher when collapsed onto it.
    sigma = 0.4
    proxy = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    proxy_labels = torch.tensor([0])

    def loss_at(distance: float) -> float:
        sample = torch.nn.functional.normalize(
            torch.tensor([[1.0, distance]], dtype=torch.float32), dim=-1
        )
        return float(
            _bio_physical_bond_loss(
                sample,
                torch.tensor([0]),
                proxy_embeddings=proxy,
                proxy_labels=proxy_labels,
                alpha=16.0,
                delta=0.1,
                sigma=sigma,
                power=2.0,
                niche_weight=0.0,
                antico_eps=0.5,
                torch_module=torch,
            ).detach()
        )

    assert loss_at(sigma) < loss_at(sigma * 0.3)  # bonding at sigma beats collapsing


def test_bio_physical_bond_reduces_and_flows_gradients() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _bio_physical_bond_loss, _normalize

    raw = torch.tensor(
        [[1.0, 0.3], [0.9, -0.2], [-1.0, 0.3], [-0.9, -0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    proxies = torch.tensor([[0.9, 0.1], [-0.9, 0.1]], dtype=torch.float32, requires_grad=True)
    proxy_labels = torch.tensor([0, 1])
    optimizer = torch.optim.Adam([raw, proxies], lr=0.05)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = _bio_physical_bond_loss(
            _normalize(raw, torch),
            labels,
            proxy_embeddings=proxies,
            proxy_labels=proxy_labels,
            alpha=16.0,
            delta=0.1,
            sigma=0.4,
            power=2.0,
            niche_weight=0.02,
            antico_eps=0.5,
            torch_module=torch,
        )
        loss.backward()
        assert raw.grad is not None and float(raw.grad.abs().sum()) > 0.0
        assert proxies.grad is not None and float(proxies.grad.abs().sum()) > 0.0
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_bio_physical_bond_objective_requires_proxies() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    with pytest.raises(ValueError, match="bio_physical_bond.*proxy_count_per_class"):
        _loss_for_objective(
            "bio_physical_bond",
            torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32),
            torch.tensor([0, 1]),
            step=1,
            steps_per_epoch=1,
            memory_embeddings=None,
            memory_labels=None,
            proxy_embeddings=None,
            proxy_labels=None,
            config=ImageEndToEndConfig(objectives=("bio_physical_bond",)),
            torch_module=torch,
        )


def test_coding_rate_higher_for_spread_than_collapsed() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _coding_rate

    # Orthogonal (spread) features occupy more volume -> higher coding rate than
    # near-identical (collapsed) features.
    spread = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        dim=-1,
    )
    collapsed = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.01], [1.0, 0.0], [1.0, -0.01], [1.0, 0.0]], dtype=torch.float32),
        dim=-1,
    )
    r_spread = float(_coding_rate(spread, eps=0.5, torch_module=torch).detach())
    r_collapsed = float(_coding_rate(collapsed, eps=0.5, torch_module=torch).detach())
    assert r_spread > r_collapsed
    assert r_collapsed >= 0.0  # coding rate is non-negative


def test_coding_rate_maximization_spreads_collapsed_batch() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _coding_rate, _normalize

    # Start from a near-collapsed batch; MAXIMISING coding rate (minimise -R) must
    # increase the spread (mean pairwise distance) — the anti-collapse mechanism.
    raw = torch.tensor(
        [[1.0, 0.02], [1.0, 0.0], [1.0, -0.02], [1.0, 0.01]],
        dtype=torch.float32,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([raw], lr=0.1)

    def spread(z: object) -> float:
        zz = _normalize(raw, torch).detach() if z is None else z
        return float(torch.cdist(zz, zz).mean())

    start = spread(None)
    for _ in range(40):
        optimizer.zero_grad()
        loss = -_coding_rate(_normalize(raw, torch), eps=0.5, torch_module=torch)
        loss.backward()
        optimizer.step()
    assert spread(None) > start


def test_proxy_anchor_antico_reduces_to_proxy_anchor_when_weight_zero() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    embeddings = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.2], [0.8, -0.3], [-0.5, 1.0], [-1.0, -0.4]], dtype=torch.float32),
        dim=-1,
    )
    labels = torch.tensor([0, 0, 1, 1])
    proxies = torch.nn.functional.normalize(
        torch.tensor([[0.9, 0.1], [-0.6, 0.8]], dtype=torch.float32), dim=-1
    )
    proxy_labels = torch.tensor([0, 1])

    def run(objective: str, weight: float) -> float:
        obj = cast(Any, objective)
        return float(
            _loss_for_objective(
                obj,
                embeddings,
                labels,
                step=1,
                steps_per_epoch=1,
                memory_embeddings=None,
                memory_labels=None,
                proxy_embeddings=proxies,
                proxy_labels=proxy_labels,
                config=ImageEndToEndConfig(
                    objectives=(obj,), proxy_count_per_class=1, antico_weight=weight
                ),
                torch_module=torch,
            ).detach()
        )

    assert run("proxy_anchor_antico", 0.0) == pytest.approx(run("proxy_anchor", 0.0), rel=1e-5)
    assert run("proxy_anchor_antico", 0.1) != pytest.approx(run("proxy_anchor", 0.0), rel=1e-5)


def test_symmetric_potential_attracts_same_repels_different() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _symmetric_potential_loss

    # A configuration where same-class points are close and different-class far
    # should have LOWER symmetric-potential energy than the mixed-up arrangement.
    good = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.05], [1.0, -0.05], [-1.0, 0.05], [-1.0, -0.05]], dtype=torch.float32),
        dim=-1,
    )
    bad = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.05], [-1.0, 0.05], [1.0, -0.05], [-1.0, -0.05]], dtype=torch.float32),
        dim=-1,
    )
    labels = torch.tensor([0, 0, 1, 1])
    good_e = _symmetric_potential_loss(
        good,
        labels,
        proxy_embeddings=None,
        proxy_labels=None,
        delta=0.2,
        alpha=1.0,
        torch_module=torch,
    )
    bad_e = _symmetric_potential_loss(
        bad,
        labels,
        proxy_embeddings=None,
        proxy_labels=None,
        delta=0.2,
        alpha=1.0,
        torch_module=torch,
    )
    assert float(good_e.detach()) < float(bad_e.detach())


def test_symmetric_potential_repulsion_is_long_range() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _symmetric_potential_loss

    # Two different-class points far apart (> delta) must still feel a repulsive
    # gradient (unlike PFML, whose repulsion is zero-force beyond delta). Pulling
    # them further apart must reduce the energy.
    near = torch.tensor([[1.0, 0.0], [0.3, 0.954]], dtype=torch.float32, requires_grad=True)
    far = torch.tensor([[1.0, 0.0], [-0.3, 0.954]], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0, 1])
    e_near = _symmetric_potential_loss(
        torch.nn.functional.normalize(near, dim=-1),
        labels,
        proxy_embeddings=None,
        proxy_labels=None,
        delta=0.2,
        alpha=1.0,
        torch_module=torch,
    )
    e_far = _symmetric_potential_loss(
        torch.nn.functional.normalize(far, dim=-1),
        labels,
        proxy_embeddings=None,
        proxy_labels=None,
        delta=0.2,
        alpha=1.0,
        torch_module=torch,
    )
    # Farther different-class pair -> lower repulsive energy (long-range decay).
    assert float(e_far.detach()) < float(e_near.detach())
    e_near.backward()
    assert near.grad is not None and float(near.grad.abs().sum()) > 0.0


def test_symmetric_potential_end_to_end_trains_without_collapse() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _normalize, _symmetric_potential_loss

    # Two separable classes; a few Adam steps should REDUCE the energy and keep the
    # embedding from collapsing (different-class distance stays well above delta).
    raw = torch.tensor(
        [[1.0, 0.2], [0.9, -0.1], [-1.0, 0.2], [-0.9, -0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    optimizer = torch.optim.Adam([raw], lr=0.05)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        z = _normalize(raw, torch)
        loss = _symmetric_potential_loss(
            z,
            labels,
            proxy_embeddings=None,
            proxy_labels=None,
            delta=0.2,
            alpha=1.0,
            torch_module=torch,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    final = _normalize(raw, torch).detach()
    cross = float((final[0] - final[2]).norm())
    assert cross > 0.2  # classes did not collapse together


def test_symmetric_potential_balances_attraction_and_repulsion_terms() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _normalize, _symmetric_potential_loss

    # Many classes -> different-class pairs dominate 5:1; the balanced per-term means
    # must still keep same-class members compact. (The all-pairs-mean variant let
    # repulsion dominate and collapsed retrieval.) Optimising must REDUCE within-class
    # spread.
    generator = torch.Generator().manual_seed(0)
    raw = torch.randn(12, 4, generator=generator)
    raw.requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    optimizer = torch.optim.Adam([raw], lr=0.1)

    def within_class_spread(z: Any) -> float:
        return sum(float((z[labels == c][0] - z[labels == c][1]).norm()) for c in range(6))

    start_spread = within_class_spread(_normalize(raw, torch).detach())
    for _ in range(40):
        optimizer.zero_grad()
        loss = _symmetric_potential_loss(
            _normalize(raw, torch),
            labels,
            proxy_embeddings=None,
            proxy_labels=None,
            delta=0.2,
            alpha=2.0,
            torch_module=torch,
        )
        loss.backward()
        optimizer.step()
    end_spread = within_class_spread(_normalize(raw, torch).detach())
    assert end_spread < start_spread  # classes got MORE compact, not collapsed apart


def test_lennard_jones_same_class_has_equilibrium_at_sigma() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _lennard_jones_loss

    sigma = 0.3

    def same_class_energy(distance: float) -> float:
        # Two same-class points separated by `distance` along one axis.
        a = torch.tensor([1.0, 0.0], dtype=torch.float32)
        offset = torch.tensor([0.0, distance], dtype=torch.float32)
        pts = torch.stack([a, a + offset])
        labels = torch.tensor([0, 0])
        return float(
            _lennard_jones_loss(
                pts,
                labels,
                proxy_embeddings=None,
                proxy_labels=None,
                sigma=sigma,
                power=2.0,
                repulsion_weight=1.0,
                torch_module=torch,
            ).detach()
        )

    at_sigma = same_class_energy(sigma)
    too_close = same_class_energy(sigma * 0.5)
    too_far = same_class_energy(sigma * 2.0)
    # LJ minimum is at the equilibrium distance sigma: energy there is lowest.
    assert at_sigma < too_close
    assert at_sigma < too_far


def test_lennard_jones_prevents_same_class_collapse() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _lennard_jones_loss

    # Two same-class points START closer than the equilibrium sigma (but above the
    # sigma/4 numerical floor); the repulsive core must push them APART toward the
    # equilibrium distance, not let them collapse together.
    raw = torch.tensor([[1.0, 0.0], [1.0, 0.15]], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0, 0])
    optimizer = torch.optim.Adam([raw], lr=0.02)
    start = float((raw[0] - raw[1]).detach().norm())
    for _ in range(30):
        optimizer.zero_grad()
        loss = _lennard_jones_loss(
            raw,
            labels,
            proxy_embeddings=None,
            proxy_labels=None,
            sigma=0.3,
            power=2.0,
            repulsion_weight=1.0,
            torch_module=torch,
        )
        loss.backward()
        optimizer.step()
    end = float((raw[0] - raw[1]).detach().norm())
    assert end > start  # repulsive core pushed them apart, no collapse


def test_lennard_jones_end_to_end_separates_and_stays_compact() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _lennard_jones_loss, _normalize

    raw = torch.tensor(
        [[1.0, 0.3], [1.0, -0.3], [-1.0, 0.3], [-1.0, -0.3]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    optimizer = torch.optim.Adam([raw], lr=0.05)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = _lennard_jones_loss(
            _normalize(raw, torch),
            labels,
            proxy_embeddings=None,
            proxy_labels=None,
            sigma=0.3,
            power=2.0,
            repulsion_weight=1.0,
            torch_module=torch,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    final = _normalize(raw, torch).detach()
    # different classes separated, same class compact (near sigma, not collapsed)
    assert float((final[0] - final[2]).norm()) > 0.3
    assert float((final[0] - final[1]).norm()) < 1.0


def test_lennard_jones_intra_term_penalizes_collapsed_and_dispersed_classes() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _lennard_jones_intra_term, _normalize

    labels = torch.tensor([0, 0])

    def energy(distance: float) -> float:
        a = torch.tensor([1.0, 0.0], dtype=torch.float32)
        pts = _normalize(torch.stack([a, a + torch.tensor([0.0, distance])]), torch)
        return float(
            _lennard_jones_intra_term(
                pts, labels, sigma=0.3, power=2.0, torch_module=torch
            ).detach()
        )

    # The well is minimised near equilibrium sigma; collapse (tiny) and dispersion
    # (large) both cost more energy.
    assert energy(0.3) < energy(0.05)
    assert energy(0.3) < energy(1.2)


def test_proxy_anchor_lj_reduces_to_proxy_anchor_when_intra_weight_zero() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    embeddings = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.2], [0.8, -0.3], [-0.5, 1.0], [-1.0, -0.4]], dtype=torch.float32),
        dim=-1,
    )
    labels = torch.tensor([0, 0, 1, 1])
    proxies = torch.nn.functional.normalize(
        torch.tensor([[0.9, 0.1], [-0.6, 0.8]], dtype=torch.float32), dim=-1
    )
    proxy_labels = torch.tensor([0, 1])

    def run(objective: str, intra: float) -> float:
        obj = cast(Any, objective)
        return float(
            _loss_for_objective(
                obj,
                embeddings,
                labels,
                step=1,
                steps_per_epoch=1,
                memory_embeddings=None,
                memory_labels=None,
                proxy_embeddings=proxies,
                proxy_labels=proxy_labels,
                config=ImageEndToEndConfig(
                    objectives=(obj,), proxy_count_per_class=1, lj_intra_weight=intra
                ),
                torch_module=torch,
            ).detach()
        )

    assert run("proxy_anchor_lj", 0.0) == pytest.approx(run("proxy_anchor", 0.0), rel=1e-5)
    assert run("proxy_anchor_lj", 0.5) != pytest.approx(run("proxy_anchor", 0.0), rel=1e-5)


def test_lennard_jones_separate_negative_sigma_extends_repulsion_range() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _lennard_jones_loss

    # Two different-class points at distance 0.5. With sigma_neg=0.3 (< 0.5) they sit
    # outside the exclusion core and feel little repulsion; with sigma_neg=0.8 (> 0.5)
    # they are inside it and feel strong repulsion -> higher energy.
    pts = torch.tensor([[1.0, 0.0], [0.5, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1])

    def repulsion_energy(sigma_neg: float) -> float:
        return float(
            _lennard_jones_loss(
                pts,
                labels,
                proxy_embeddings=None,
                proxy_labels=None,
                sigma=0.3,
                power=2.0,
                repulsion_weight=1.0,
                sigma_neg=sigma_neg,
                torch_module=torch,
            ).detach()
        )

    assert repulsion_energy(0.8) > repulsion_energy(0.3)


def test_proxy_anchor_loss_matches_hand_computed_value() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _proxy_anchor_loss

    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1])
    proxies = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    proxy_labels = torch.tensor([0, 1])

    loss = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=1.0,
        delta=0.0,
        torch_module=torch,
    )

    expected = torch.nn.functional.softplus(torch.tensor(-1.0)) + torch.nn.functional.softplus(
        torch.tensor(0.0)
    )
    assert float(loss.detach().cpu()) == pytest.approx(float(expected), rel=1e-6)


def test_relational_distillation_matches_teacher_neighborhoods() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _relational_distillation_loss

    torch.manual_seed(0)
    teacher = torch.randn(10, 4)
    # A student equal to the teacher has the LOWEST distillation loss (perfect match);
    # a random student has a higher loss -- so minimising it aligns neighborhoods.
    matched = _relational_distillation_loss(teacher.clone(), teacher, tau=0.1, torch_module=torch)
    mismatched = _relational_distillation_loss(
        torch.randn(10, 4), teacher, tau=0.1, torch_module=torch
    )
    assert float(matched.detach()) < float(mismatched.detach())

    # Gradient descent moves a student's neighborhood distribution toward the teacher's.
    student = torch.randn(10, 4, requires_grad=True)
    optimizer = torch.optim.Adam([student], lr=0.05)
    start = float(
        _relational_distillation_loss(student, teacher, tau=0.1, torch_module=torch).detach()
    )
    for _ in range(50):
        optimizer.zero_grad()
        loss = _relational_distillation_loss(student, teacher, tau=0.1, torch_module=torch)
        loss.backward()
        assert torch.isfinite(student.grad).all()
        optimizer.step()
    end = float(
        _relational_distillation_loss(student, teacher, tau=0.1, torch_module=torch).detach()
    )
    assert end < start


def test_ema_teacher_update_moves_toward_student() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _update_ema_teacher

    teacher = torch.nn.Linear(4, 4)
    student = torch.nn.Linear(4, 4)
    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)  # push the student away from the teacher
    before = teacher.weight.detach().clone()
    _update_ema_teacher(teacher, student, momentum=0.9)
    expected = 0.9 * before + 0.1 * student.weight.detach()
    assert torch.allclose(teacher.weight.detach(), expected, atol=1e-6)


def test_gaussian_potential_uniformity_pushes_embeddings_apart() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _gaussian_potential_uniformity_loss

    torch.manual_seed(0)
    # The loss normalises to the unit sphere, so "collapsed" means all points share a
    # DIRECTION (not small magnitude). A collapsed batch has the HIGHEST (worst) loss;
    # spread directions have a lower loss -- minimising it pushes embeddings apart.
    collapsed = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(8, 1) + 0.001 * torch.randn(8, 4)
    spread = torch.randn(8, 4)
    high = _gaussian_potential_uniformity_loss(collapsed, t=2.0, torch_module=torch)
    low = _gaussian_potential_uniformity_loss(spread, t=2.0, torch_module=torch)
    assert float(high.detach()) > float(low.detach())

    # Gradient descent spreads a near-collapsed batch: mean pairwise angle grows.
    z = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(12, 1) + 0.02 * torch.randn(12, 4)
    z = z.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([z], lr=1.0)

    def _mean_pair_sim(v: object) -> float:
        n = torch.nn.functional.normalize(torch.as_tensor(v).detach(), dim=1)
        return float((n @ n.T).mean())

    start_sim = _mean_pair_sim(z)
    for _ in range(50):
        optimizer.zero_grad()
        loss = _gaussian_potential_uniformity_loss(z, t=2.0, torch_module=torch)
        loss.backward()
        assert torch.isfinite(z.grad).all()
        optimizer.step()
    assert _mean_pair_sim(z) < start_sim  # points spread out (lower mean similarity)


def test_subcenter_proxy_anchor_reduces_to_proxy_anchor_when_single_proxy() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import (
        _proxy_anchor_loss,
        _subcenter_proxy_anchor_loss,
    )

    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1])
    proxies = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    proxy_labels = torch.tensor([0, 1])  # one proxy per class -> falls back to PA

    sub = _subcenter_proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=8.0,
        delta=0.1,
        gamma=0.1,
        torch_module=torch,
    )
    base = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=8.0,
        delta=0.1,
        torch_module=torch,
    )
    assert float(sub.detach()) == pytest.approx(float(base.detach()), rel=1e-6)


def test_subcenter_proxy_anchor_fits_bimodal_class_without_collapse() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _subcenter_proxy_anchor_loss

    torch.manual_seed(0)
    # Class 0 has TWO well-separated modes; class 1 has one. Sub-centers (K=2) should
    # be able to cover both modes of class 0. Proxies contiguous per class: [c0k0,c0k1,c1k0,c1k1].
    mode_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
    mode_b = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(4, 1)
    class1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).repeat(4, 1)
    embeddings = torch.cat([mode_a, mode_b, class1], dim=0) + 0.05 * torch.randn(12, 4)
    embeddings = embeddings.clone().requires_grad_(True)
    labels = torch.tensor([0] * 8 + [1] * 4)
    proxies = torch.randn(4, 4, requires_grad=True)  # 2 classes x 2 sub-centers
    proxy_labels = torch.tensor([0, 0, 1, 1])

    optimizer = torch.optim.Adam([embeddings, proxies], lr=0.05)
    first = None
    last = None
    for _ in range(60):
        optimizer.zero_grad()
        loss = _subcenter_proxy_anchor_loss(
            embeddings,
            labels,
            proxy_embeddings=proxies,
            proxy_labels=proxy_labels,
            alpha=8.0,
            delta=0.1,
            gamma=0.1,
            torch_module=torch,
        )
        loss.backward()
        assert torch.isfinite(embeddings.grad).all()
        assert torch.isfinite(proxies.grad).all()
        optimizer.step()
        if first is None:
            first = float(loss.detach())
        last = float(loss.detach())
    assert first is not None and last is not None
    assert last < first  # the loss genuinely decreases on multimodal data
    # The two sub-centers of class 0 do NOT collapse onto each other (they cover the
    # two modes): their cosine similarity stays below a high-collapse threshold.
    from torch.nn.functional import normalize as _n

    c0 = _n(proxies[:2].detach(), dim=1)
    assert float((c0[0] * c0[1]).sum()) < 0.9


def test_proxy_anchor_group_reduces_to_proxy_anchor_when_single_proxy() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _proxy_anchor_group_loss, _proxy_anchor_loss

    embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.2], [0.8, -0.3], [-0.5, 1.0], [-1.0, -0.4]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    labels = torch.tensor([0, 0, 1, 1])
    proxies = torch.nn.functional.normalize(
        torch.tensor([[0.9, 0.1], [-0.6, 0.8]], dtype=torch.float32),
        dim=-1,
    )
    proxy_labels = torch.tensor([0, 1])

    base = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=16.0,
        delta=0.1,
        torch_module=torch,
    )
    group = _proxy_anchor_group_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=16.0,
        delta=0.1,
        tau_assign=0.1,
        torch_module=torch,
    )
    # One proxy per class -> soft assignment is trivially 1.0 -> identical loss.
    assert float(group.detach().cpu()) == pytest.approx(float(base.detach().cpu()), rel=1e-5)


def test_proxy_anchor_group_soft_assignment_specializes() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _group_soft_class_similarity

    # One class (label 0) with two proxies at orthogonal directions.
    proxies = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    proxy_labels = torch.tensor([0, 0])
    # A sample near proxy A (first direction).
    sample = torch.nn.functional.normalize(
        torch.tensor([[0.98, 0.05]], dtype=torch.float32), dim=-1
    )

    similarity, assignment = _group_soft_class_similarity(
        sample,
        proxies,
        proxy_labels,
        class_label=0,
        tau_assign=0.05,
        torch_module=torch,
    )
    # Assignment weight to proxy A must dominate.
    assert float(assignment[0, 0]) > 0.9
    # Effective similarity is close to the near proxy's cosine (~0.98).
    assert float(similarity[0]) > 0.9


def test_proxy_anchor_group_loss_decreases_on_separable_toy_data() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _normalize, _proxy_anchor_group_loss

    raw = torch.tensor(
        [[1.0, 0.1], [0.9, -0.1], [-1.0, 0.1], [-0.9, -0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    proxies = torch.tensor(
        [[0.7, 0.7], [0.7, -0.7], [-0.7, 0.7], [-0.7, -0.7]],
        dtype=torch.float32,
        requires_grad=True,
    )
    proxy_labels = torch.tensor([0, 0, 1, 1])
    optimizer = torch.optim.Adam([raw, proxies], lr=0.05)

    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        loss = _proxy_anchor_group_loss(
            _normalize(raw, torch),
            labels,
            proxy_embeddings=proxies,
            proxy_labels=proxy_labels,
            alpha=16.0,
            delta=0.1,
            tau_assign=0.1,
            torch_module=torch,
        )
        loss.backward()
        assert proxies.grad is not None and float(proxies.grad.abs().sum()) > 0.0
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    assert losses[-1] < losses[0]


def test_proxy_anchor_group_loss_for_objective_requires_proxies() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    with pytest.raises(ValueError, match="proxy_anchor_group.*proxy_count_per_class"):
        _loss_for_objective(
            "proxy_anchor_group",
            torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32),
            torch.tensor([0, 1]),
            step=1,
            steps_per_epoch=1,
            memory_embeddings=None,
            memory_labels=None,
            proxy_embeddings=None,
            proxy_labels=None,
            config=ImageEndToEndConfig(objectives=("proxy_anchor_group",)),
            torch_module=torch,
        )


def test_proxy_anchor_group_end_to_end_runs() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 2)

        def forward(self, images: object) -> object:
            return self.embedding(torch.as_tensor(images, dtype=torch.long))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return int(cast(int, image))

        return transform

    examples = [
        ImageExample(example_id=f"{label}-{index}", image=label * 4 + index, label=label)
        for label in range(2)
        for index in range(4)
    ]

    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="proxy-anchor-resnet50-512",
            objectives=("proxy_anchor_group",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=8,
            samples_per_class=4,
            proxy_count_per_class=2,
            eval_batch_size=8,
            train_steps=2,
            train_epochs=None,
            warmup_epochs=0,
            retrieval_query_limit=8,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    assert "proxy_anchor_group_end_to_end:tiny" in result.methods


def test_proxy_synthesis_augment_reduces_to_base_when_ratio_zero() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import (
        _proxy_anchor_loss,
        _proxy_synthesis_proxy_anchor_loss,
    )

    embeddings = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.1], [0.9, -0.2], [-0.4, 1.0], [-1.0, -0.3]], dtype=torch.float32),
        dim=-1,
    )
    labels = torch.tensor([0, 0, 1, 1])
    proxies = torch.nn.functional.normalize(
        torch.tensor([[0.9, 0.1], [-0.5, 0.9]], dtype=torch.float32), dim=-1
    )
    proxy_labels = torch.tensor([0, 1])

    base = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=16.0,
        delta=0.1,
        torch_module=torch,
    )
    synth = _proxy_synthesis_proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=16.0,
        delta=0.1,
        ratio=0.0,
        beta_alpha=0.4,
        generator=None,
        torch_module=torch,
    )
    assert float(synth.detach().cpu()) == pytest.approx(float(base.detach().cpu()), rel=1e-5)


def test_proxy_synthesis_creates_virtual_classes_and_flows_gradients() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _normalize, _proxy_synthesis_proxy_anchor_loss

    raw = torch.tensor(
        [[1.0, 0.1], [0.9, -0.2], [-0.4, 1.0], [-1.0, -0.3]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    proxies = torch.tensor([[0.9, 0.1], [-0.5, 0.9]], dtype=torch.float32, requires_grad=True)
    proxy_labels = torch.tensor([0, 1])
    generator = torch.Generator().manual_seed(0)

    loss = _proxy_synthesis_proxy_anchor_loss(
        _normalize(raw, torch),
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=16.0,
        delta=0.1,
        ratio=1.0,
        beta_alpha=0.4,
        generator=generator,
        torch_module=torch,
    )
    loss.backward()
    # Virtual embeddings/proxies are mixtures of real ones, so gradients reach both.
    assert raw.grad is not None and float(raw.grad.abs().sum()) > 0.0
    assert proxies.grad is not None and float(proxies.grad.abs().sum()) > 0.0


def test_confusable_pair_sampling_prefers_similar_classes() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _sample_synthesis_class_pairs

    present = [0, 1, 2]
    present_proxies = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.99, 0.14], [-1.0, 0.0]], dtype=torch.float32), dim=-1
    )
    generator = torch.Generator().manual_seed(0)
    counts: dict[tuple[int, int], int] = {}
    for _ in range(400):
        ci, cj = _sample_synthesis_class_pairs(
            present,
            present_proxies,
            mode="confusable",
            temperature=0.1,
            generator=generator,
            torch_module=torch,
        )
        lo, hi = sorted((int(ci), int(cj)))
        key = (lo, hi)
        counts[key] = counts.get(key, 0) + 1
    # Confusable pair (0,1) sampled far more than pairs involving the far class 2.
    assert counts.get((0, 1), 0) > counts.get((0, 2), 0) + counts.get((1, 2), 0)


def test_confusion_guided_synthesis_runs_and_flows_gradients() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _normalize, _proxy_synthesis_proxy_anchor_loss

    raw = torch.tensor(
        [[1.0, 0.1], [0.9, -0.2], [-0.4, 1.0], [-1.0, -0.3], [0.2, 0.9], [-0.3, -0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    proxies = torch.tensor(
        [[0.9, 0.1], [-0.5, 0.9], [0.1, -0.95]], dtype=torch.float32, requires_grad=True
    )
    proxy_labels = torch.tensor([0, 1, 2])
    generator = torch.Generator().manual_seed(0)

    loss = _proxy_synthesis_proxy_anchor_loss(
        _normalize(raw, torch),
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=16.0,
        delta=0.1,
        ratio=1.0,
        beta_alpha=0.4,
        generator=generator,
        pair_selection="confusable",
        pair_temperature=0.1,
        torch_module=torch,
    )
    loss.backward()
    assert raw.grad is not None and float(raw.grad.abs().sum()) > 0.0
    assert proxies.grad is not None and float(proxies.grad.abs().sum()) > 0.0


def test_synthesis_compactness_weight_adds_positive_penalty_and_flows_gradients() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    # Two classes, members spread away from their centroids so compactness fires.
    embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.5], [1.0, -0.5], [-1.0, 0.5], [-1.0, -0.5]],
            dtype=torch.float32,
        ),
        dim=-1,
    ).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1])
    proxies = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32), dim=-1
    ).requires_grad_(True)
    proxy_labels = torch.tensor([0, 1])
    generator = torch.Generator().manual_seed(0)

    def run(compactness: float) -> Any:
        return _loss_for_objective(
            "proxy_anchor_synthesis",
            embeddings,
            labels,
            step=1,
            steps_per_epoch=1,
            memory_embeddings=None,
            memory_labels=None,
            proxy_embeddings=proxies,
            proxy_labels=proxy_labels,
            config=ImageEndToEndConfig(
                objectives=("proxy_anchor_synthesis",),
                proxy_count_per_class=1,
                synthesis_ratio=0.0,
                synthesis_compactness_weight=compactness,
            ),
            torch_module=torch,
            generator=generator,
        )

    without = run(0.0)
    with_compactness = run(1.0)
    # The compactness term is a non-negative penalty on intra-class spread.
    assert float(with_compactness.detach()) > float(without.detach())
    with_compactness.backward()
    assert embeddings.grad is not None and float(embeddings.grad.abs().sum()) > 0.0


def test_proxy_synthesis_loss_for_objective_requires_proxies() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    with pytest.raises(ValueError, match="proxy_anchor_synthesis.*proxy_count_per_class"):
        _loss_for_objective(
            "proxy_anchor_synthesis",
            torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32),
            torch.tensor([0, 1]),
            step=1,
            steps_per_epoch=1,
            memory_embeddings=None,
            memory_labels=None,
            proxy_embeddings=None,
            proxy_labels=None,
            config=ImageEndToEndConfig(objectives=("proxy_anchor_synthesis",)),
            torch_module=torch,
        )


def test_proxy_synthesis_end_to_end_runs() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 2)

        def forward(self, images: object) -> object:
            return self.embedding(torch.as_tensor(images, dtype=torch.long))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return int(cast(int, image))

        return transform

    examples = [
        ImageExample(example_id=f"{label}-{index}", image=label * 4 + index, label=label)
        for label in range(2)
        for index in range(4)
    ]
    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="proxy-anchor-resnet50-512",
            objectives=("proxy_anchor_synthesis",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=8,
            samples_per_class=4,
            proxy_count_per_class=1,
            eval_batch_size=8,
            train_steps=2,
            train_epochs=None,
            warmup_epochs=0,
            retrieval_query_limit=8,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )
    assert "proxy_anchor_synthesis_end_to_end:tiny" in result.methods


def test_proxy_anchor_loss_for_objective_requires_proxies() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    with pytest.raises(ValueError, match="proxy_anchor.*proxy_count_per_class"):
        _loss_for_objective(
            "proxy_anchor",
            torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32),
            torch.tensor([0, 1]),
            step=1,
            steps_per_epoch=1,
            memory_embeddings=None,
            memory_labels=None,
            proxy_embeddings=None,
            proxy_labels=None,
            config=ImageEndToEndConfig(objectives=("proxy_anchor",)),
            torch_module=torch,
        )


def test_proxy_anchor_benchmark_requires_proxy_count_per_class() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(2, 2)

        def forward(self, images: object) -> object:
            return self.embedding(torch.as_tensor(images, dtype=torch.long))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.long)

        return transform

    examples = [
        ImageExample(example_id=f"{label}-{index}", image=label, label=label)
        for label in (0, 1)
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="proxy_anchor.*proxy_count_per_class"):
        run_image_end_to_end_benchmark(
            train_examples=examples,
            test_examples=examples,
            config=ImageEndToEndConfig(
                dataset_name="cub",
                protocol="sota-resnet50-512",
                objectives=("proxy_anchor",),
                backbone_name="tiny",
                embedding_dimensions=2,
                batch_size=4,
                eval_batch_size=4,
                train_steps=1,
                group_size=1,
                proxy_count_per_class=0,
                progress_every=0,
                num_workers=0,
            ),
            model_factory=lambda config: TinyModel(),
            transform_factory=transform_factory,
        )


def test_proxy_anchor_objective_trains_proxies_end_to_end_and_loss_decreases() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(2, 2)
            with torch.no_grad():
                self.embedding.weight.copy_(
                    torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=torch.float32)
                )

        def forward(self, images: object) -> object:
            return self.embedding(torch.as_tensor(images, dtype=torch.long))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.long)

        return transform

    examples = [
        ImageExample(example_id=f"{label}-{index}", image=label, label=label)
        for label in (0, 1)
        for index in range(2)
    ]
    models: list[Any] = []

    def model_factory(config: ImageEndToEndConfig) -> Any:
        model = TinyModel()
        models.append(model)
        return model

    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("proxy_anchor",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=4,
            train_steps=6,
            group_size=1,
            learning_rate=0.1,
            proxy_count_per_class=1,
            proxy_learning_rate_multiplier=1.0,
            proxy_anchor_alpha=2.0,
            proxy_anchor_delta=0.0,
            progress_every=0,
            num_workers=0,
            seed=0,
        ),
        model_factory=model_factory,
        transform_factory=transform_factory,
    )

    model = models[0]
    assert tuple(model.metric_proxies.shape) == (2, 2)
    assert model.metric_proxies.grad is not None
    assert float(model.metric_proxies.grad.norm().detach().cpu()) > 0.0
    history = result.methods["proxy_anchor_end_to_end:tiny"].loss_history
    assert len(history) == 6
    assert history[-1] < history[0]
    assert result.methods["proxy_anchor_end_to_end:tiny"].display_name == "Proxy Anchor"


def test_pfml_potential_loss_matches_hand_computed_energy() -> None:
    # Paper kernel (arXiv 2405.18560 Eq. 1-2) with delta=0.5, alpha=2:
    #   attraction: -1/delta^2 = -4 inside the margin, -1/d^2 outside
    #   repulsion:   1/d^2 inside the margin,  1/delta^2 = 4 outside
    # Total energy (Eq. 6) = all ordered pairs over embeddings + proxies.
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _pfml_potential_loss

    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.6]], dtype=torch.float32)
    labels = torch.tensor([0, 1])
    proxies = torch.tensor([[0.6, 0.8], [-1.0, 0.0]], dtype=torch.float32)
    proxy_labels = torch.tensor([0, 1])

    loss = _pfml_potential_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        delta=0.5,
        alpha=2.0,
        torch_module=torch,
    )

    # Unordered pair energies:
    #   e0-e1 (diff, d^2=0.4  >= delta): +4
    #   e0-p0 (same, d^2=0.8  >= delta): -1/0.8 = -1.25
    #   e0-p1 (diff, d^2=4.0  >= delta): +4
    #   e1-p0 (diff, d^2=0.08 <  delta): +1/0.08 = 12.5
    #   e1-p1 (same, d^2=3.6  >= delta): -1/3.6
    #   p0-p1 (diff, d^2=3.2  >= delta): +4
    expected = 2.0 * (4.0 - 1.25 + 4.0 + 12.5 - 1.0 / 3.6 + 4.0) / 12.0
    assert float(loss.detach().cpu()) == pytest.approx(expected, rel=1e-5)


def test_pfml_potential_loss_supports_zero_alpha() -> None:
    # alpha=0 (paper's cross-validation range is {0..6}) collapses both kernels
    # to constant magnitude 1: attraction -1 and repulsion +1 for every pair.
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _pfml_potential_loss

    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.6]], dtype=torch.float32)
    labels = torch.tensor([0, 1])
    proxies = torch.tensor([[0.6, 0.8], [-1.0, 0.0]], dtype=torch.float32)
    proxy_labels = torch.tensor([0, 1])

    loss = _pfml_potential_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        delta=0.5,
        alpha=0.0,
        torch_module=torch,
    )

    # 2 unordered same-label pairs at -1 and 4 different-label pairs at +1:
    # mean over the 12 ordered off-diagonal pairs = (2*4 - 2*2) / 12 = 1/3.
    assert torch.isfinite(loss)
    assert float(loss.detach().cpu()) == pytest.approx(1.0 / 3.0, rel=1e-5)


def test_pfml_potential_loss_saturates_attraction_inside_margin() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _pfml_potential_loss

    embeddings = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0])
    proxies = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss = _pfml_potential_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=torch.tensor([0]),
        delta=0.5,
        alpha=2.0,
        torch_module=torch,
    )

    assert float(loss.detach().cpu()) == pytest.approx(-4.0, rel=1e-5)


def test_pfml_proxy_proxy_pairs_interact_and_saturated_pairs_exert_no_force() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _pfml_potential_loss

    # The lone embedding belongs to a class without proxies, so all of its pairs
    # are different-class at distance sqrt(2) >= delta: constant repulsion, zero
    # force. The two proxies of different classes sit within delta of each other,
    # so only the proxy<->proxy repulsion carries gradient.
    embeddings = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([2])
    proxies = torch.nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0], [0.96, 0.28, 0.0]], dtype=torch.float32)
    )
    proxy_labels = torch.tensor([0, 1])

    loss = _pfml_potential_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        delta=0.5,
        alpha=2.0,
        torch_module=torch,
    )
    loss.backward()

    assert proxies.grad is not None
    assert float(proxies.grad.norm().detach().cpu()) > 0.0
    assert embeddings.grad is not None
    assert float(embeddings.grad.norm().detach().cpu()) == pytest.approx(0.0, abs=1e-9)


def test_pfml_loss_for_objective_requires_proxies() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    with pytest.raises(ValueError, match="pfml.*proxy_count_per_class"):
        _loss_for_objective(
            "pfml",
            torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32),
            torch.tensor([0, 1]),
            step=1,
            steps_per_epoch=1,
            memory_embeddings=None,
            memory_labels=None,
            proxy_embeddings=None,
            proxy_labels=None,
            config=ImageEndToEndConfig(objectives=("pfml",)),
            torch_module=torch,
        )


def test_pfml_benchmark_requires_proxy_count_per_class() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    examples = [
        ImageExample(
            example_id=f"{label}-{index}",
            image=[float(label), float(index)],
            label=label,
        )
        for label in (0, 1)
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="pfml.*proxy_count_per_class"):
        run_image_end_to_end_benchmark(
            train_examples=examples,
            test_examples=examples,
            config=ImageEndToEndConfig(
                dataset_name="cub",
                protocol="sota-resnet50-512",
                objectives=("pfml",),
                backbone_name="tiny",
                embedding_dimensions=2,
                batch_size=4,
                eval_batch_size=4,
                train_steps=1,
                group_size=1,
                proxy_count_per_class=0,
                progress_every=0,
                num_workers=0,
            ),
            model_factory=lambda config: TinyModel(),
            transform_factory=transform_factory,
        )


def test_pfml_objective_trains_proxies_end_to_end_and_loss_decreases() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    examples = [
        ImageExample(
            example_id=f"{label}-{index}",
            image=[float(label), float(index) - float(label)],
            label=label,
        )
        for label in (0, 1)
        for index in range(2)
    ]
    models: list[Any] = []

    def model_factory(config: ImageEndToEndConfig) -> Any:
        model = TinyModel()
        models.append(model)
        return model

    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("pfml",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=4,
            train_steps=6,
            group_size=1,
            learning_rate=0.05,
            proxy_count_per_class=2,
            proxy_learning_rate_multiplier=1.0,
            potential_delta=0.3,
            potential_alpha=2.0,
            progress_every=0,
            num_workers=0,
            seed=0,
        ),
        model_factory=model_factory,
        transform_factory=transform_factory,
    )

    model = models[0]
    assert tuple(model.metric_proxies.shape) == (4, 2)
    assert model.metric_proxies.grad is not None
    assert float(model.metric_proxies.grad.norm().detach().cpu()) > 0.0
    history = result.methods["pfml_end_to_end:tiny"].loss_history
    assert len(history) == 6
    assert history[-1] < history[0]
    assert result.methods["pfml_end_to_end:tiny"].display_name == "PFML (Potential Field)"


def test_gsi_interference_ratio_is_scale_invariant_above_variance_floor() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _gsi_interference_loss

    axes_by_class = {
        0: (
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
        )
    }
    mean = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    deviations = torch.tensor(
        [
            [0.3, 0.1, 0.0],
            [-0.3, -0.1, 0.0],
            [0.2, -0.2, 0.0],
            [-0.2, 0.2, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.zeros(4, dtype=torch.long)

    losses = {
        scale: float(
            _gsi_interference_loss(
                mean + scale * deviations,
                labels,
                axes_by_class=axes_by_class,
                floor=0.02,
                variance_floor=1e-4,
                min_group_size=4,
                torch_module=torch,
            ).detach()
        )
        for scale in (0.5, 1.0, 2.0)
    }

    assert losses[1.0] > 0.0
    assert losses[0.5] == pytest.approx(losses[1.0], rel=1e-5)
    assert losses[2.0] == pytest.approx(losses[1.0], rel=1e-5)


def test_gsi_gradient_stays_bounded_when_classes_compact_below_variance_floor() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _gsi_interference_loss

    axes_by_class = {
        0: (
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
        )
    }
    labels = torch.zeros(4, dtype=torch.long)
    deviations = (0.1 / math.sqrt(2.0)) * torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    def gradient_norm(scale: float) -> float:
        embeddings = (scale * deviations).clone().requires_grad_(True)
        loss = _gsi_interference_loss(
            embeddings,
            labels,
            axes_by_class=axes_by_class,
            floor=0.0,
            variance_floor=1e-4,
            min_group_size=4,
            torch_module=torch,
        )
        loss.backward()
        assert embeddings.grad is not None
        return float(embeddings.grad.norm().detach())

    unshrunk = gradient_norm(1.0)
    # x100 compaction drops the total variance to 1e-6, below the 1e-4 floor;
    # without the clamp the scale-invariant ratio's gradient would grow x100.
    compacted = gradient_norm(0.01)

    assert unshrunk > 0.0
    assert compacted <= 10.0 * unshrunk


def test_gsi_loss_is_zero_when_scatter_is_orthogonal_to_confusion_axes() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _confusion_axes, _gsi_interference_loss

    proxies = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    proxy_labels = torch.tensor([0, 1])
    axes_by_class = _confusion_axes(proxies, proxy_labels, top_k=1, torch_module=torch)

    # Both confusion axes live in the e0/e1 plane; all scatter goes on e2.
    scatter = torch.tensor(
        [[0.0, 0.0, 0.2], [0.0, 0.0, -0.2], [0.0, 0.0, 0.1], [0.0, 0.0, -0.1]],
        dtype=torch.float32,
    )
    embeddings = torch.cat(
        [
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32) + scatter,
            torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32) + scatter,
        ],
        dim=0,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    loss = _gsi_interference_loss(
        embeddings,
        labels,
        axes_by_class=axes_by_class,
        floor=0.0,
        variance_floor=1e-4,
        min_group_size=4,
        torch_module=torch,
    )

    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-9)


def test_gsi_loss_fires_on_axis_aligned_scatter_and_gradient_step_reduces_alignment() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _confusion_axes, _gsi_interference_loss

    proxies = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32)
    proxy_labels = torch.tensor([0, 1])
    axes_by_class = _confusion_axes(proxies, proxy_labels, top_k=1, torch_module=torch)
    labels = torch.zeros(4, dtype=torch.long)

    aligned = torch.tensor(
        [[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        dtype=torch.float32,
    )
    aligned_loss = _gsi_interference_loss(
        aligned,
        labels,
        axes_by_class=axes_by_class,
        floor=0.02,
        variance_floor=1e-4,
        min_group_size=4,
        torch_module=torch,
    )
    assert float(aligned_loss.detach()) > 0.0

    # The ratio is locally flat at pure alignment, so check the gradient
    # direction from a partially aligned (45 degree) configuration.
    partially_aligned = (0.1 / math.sqrt(2.0)) * torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    embeddings = partially_aligned.clone().requires_grad_(True)
    loss = _gsi_interference_loss(
        embeddings,
        labels,
        axes_by_class=axes_by_class,
        floor=0.0,
        variance_floor=1e-4,
        min_group_size=4,
        torch_module=torch,
    )
    loss.backward()
    assert embeddings.grad is not None

    def axis_alignment(matrix: Any) -> float:
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        axis = axes_by_class[0][0][0]
        parallel = (centered @ axis).pow(2).mean()
        total = centered.pow(2).sum(dim=1).mean()
        return float((parallel / total).detach())

    stepped = (embeddings - 1e-3 * embeddings.grad).detach()

    assert axis_alignment(stepped) < axis_alignment(partially_aligned)


def test_gsi_gradients_preserve_class_means() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _gsi_interference_loss

    axes_by_class = {
        0: (
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
        )
    }
    labels = torch.zeros(4, dtype=torch.long)
    mean = torch.tensor([[0.3, -0.2, 0.9]], dtype=torch.float32)
    deviations = (0.1 / math.sqrt(2.0)) * torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    embeddings = (mean + deviations).clone().requires_grad_(True)

    loss = _gsi_interference_loss(
        embeddings,
        labels,
        axes_by_class=axes_by_class,
        floor=0.0,
        variance_floor=1e-4,
        min_group_size=4,
        torch_module=torch,
    )
    loss.backward()

    assert embeddings.grad is not None
    assert float(embeddings.grad.norm().detach()) > 0.0
    assert torch.allclose(embeddings.grad.sum(dim=0), torch.zeros(3), atol=1e-6)


def test_gsi_ignores_classes_below_min_group_size() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _gsi_interference_loss

    axis = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    weights = torch.tensor([1.0], dtype=torch.float32)
    axes_by_class = {0: (axis, weights), 1: (axis, weights)}
    large_class = torch.tensor(
        [[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.1, 0.1, 0.0], [-0.1, -0.1, 0.0]],
        dtype=torch.float32,
    )
    small_class = torch.tensor(
        [[1.3, 0.0, 0.0], [0.7, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    def loss_value(embeddings: Any, labels: Any, min_group_size: int) -> float:
        return float(
            _gsi_interference_loss(
                embeddings,
                labels,
                axes_by_class=axes_by_class,
                floor=0.02,
                variance_floor=1e-4,
                min_group_size=min_group_size,
                torch_module=torch,
            ).detach()
        )

    combined = torch.cat([large_class, small_class], dim=0)
    combined_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    large_only = loss_value(large_class, torch.zeros(4, dtype=torch.long), 4)

    assert loss_value(combined, combined_labels, 4) == pytest.approx(large_only)
    # Sanity: the small class would contribute if the gate admitted it.
    assert loss_value(combined, combined_labels, 3) != pytest.approx(large_only)


def test_gsi_confusion_axes_are_detached_from_proxies() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _confusion_axes, _gsi_interference_loss

    proxies = torch.nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32)
    )
    proxy_labels = torch.tensor([0, 1])
    axes_by_class = _confusion_axes(proxies, proxy_labels, top_k=1, torch_module=torch)
    labels = torch.zeros(4, dtype=torch.long)
    partially_aligned = (0.1 / math.sqrt(2.0)) * torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    embeddings = partially_aligned.clone().requires_grad_(True)

    loss = _gsi_interference_loss(
        embeddings,
        labels,
        axes_by_class=axes_by_class,
        floor=0.0,
        variance_floor=1e-4,
        min_group_size=4,
        torch_module=torch,
    )
    loss.backward()

    assert proxies.grad is None
    assert embeddings.grad is not None
    assert float(embeddings.grad.norm().detach()) > 0.0


def _gsi_toy_batch(torch: Any) -> tuple[Any, Any, Any, Any]:
    embeddings = torch.tensor(
        [
            [1.2, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, -0.1, 0.0],
            [-1.2, 0.0, 0.0],
            [-0.8, 0.0, 0.0],
            [-1.0, 0.1, 0.0],
            [-1.0, -0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    proxies = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32)
    proxy_labels = torch.tensor([0, 1])
    return embeddings, labels, proxies, proxy_labels


def test_config_exposes_gsi_fields() -> None:
    config = ImageEndToEndConfig()

    assert config.gsi_weight == pytest.approx(0.3)
    assert config.gsi_floor == pytest.approx(0.02)
    assert config.gsi_top_k == 3
    assert config.gsi_min_group_size == 4
    assert config.gsi_variance_floor == pytest.approx(1e-4)
    assert config.gsi_start_epoch == 5
    assert config.gsi_axis_mode == "proxy"
    assert ImageEndToEndConfig(gsi_axis_mode="random").gsi_axis_mode == "random"
    assert ImageEndToEndConfig(gsi_axis_mode="global").gsi_axis_mode == "global"


def test_config_exposes_bgsi_fields() -> None:
    config = ImageEndToEndConfig()

    assert config.bgsi_weight == pytest.approx(0.3)
    assert config.bgsi_floor == pytest.approx(0.0)
    assert config.bgsi_top_k == 3
    assert config.bgsi_temperature == pytest.approx(0.1)
    assert config.bgsi_start_epoch == 5
    assert config.bgsi_min_group_size == 4
    assert config.bgsi_variance_floor == pytest.approx(1e-4)

    custom = ImageEndToEndConfig(
        bgsi_weight=1.0,
        bgsi_floor=0.005,
        bgsi_top_k=2,
        bgsi_temperature=0.2,
        bgsi_start_epoch=0,
    )
    assert custom.bgsi_weight == pytest.approx(1.0)
    assert custom.bgsi_floor == pytest.approx(0.005)
    assert custom.bgsi_top_k == 2
    assert custom.bgsi_temperature == pytest.approx(0.2)
    assert custom.bgsi_start_epoch == 0


def test_config_exposes_stable_bgsi_axis_fields() -> None:
    config = ImageEndToEndConfig()

    assert config.bgsi_axis_mode == "batch_boundary"
    assert config.bgsi_ema_momentum == pytest.approx(0.95)
    assert config.bgsi_min_axis_observations == 5
    assert config.bgsi_use_axis_agreement_gate is True
    assert config.bgsi_axis_agreement == pytest.approx(0.5)

    custom = ImageEndToEndConfig(
        bgsi_axis_mode="ema_boundary",
        bgsi_ema_momentum=0.8,
        bgsi_min_axis_observations=3,
        bgsi_use_axis_agreement_gate=False,
        bgsi_axis_agreement=0.25,
    )
    assert custom.bgsi_axis_mode == "ema_boundary"
    assert custom.bgsi_ema_momentum == pytest.approx(0.8)
    assert custom.bgsi_min_axis_observations == 3
    assert custom.bgsi_use_axis_agreement_gate is False
    assert custom.bgsi_axis_agreement == pytest.approx(0.25)

    for mode in ("random", "permuted", "global"):
        assert ImageEndToEndConfig(bgsi_axis_mode=mode).bgsi_axis_mode == mode


def test_bgsi_class_mean_state_updates_normalized_detached_means() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import BGSIClassMeanState

    state = BGSIClassMeanState(
        labels=[10, 20],
        embedding_dimensions=2,
        momentum=0.5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        torch_module=torch,
    )
    embeddings = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([10, 10, 20, 20])

    state.update(embeddings, labels)

    assert torch.equal(state.counts.cpu(), torch.tensor([1, 1]))
    expected_first = torch.nn.functional.normalize(torch.tensor([1.0, 1.0]), dim=0)
    expected_second = torch.nn.functional.normalize(torch.tensor([-1.0, -1.0]), dim=0)
    assert torch.allclose(state.means[0].cpu(), expected_first, atol=1e-6)
    assert torch.allclose(state.means[1].cpu(), expected_second, atol=1e-6)
    assert state.means.requires_grad is False

    second_embeddings = torch.tensor(
        [[2.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [-2.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    state.update(second_embeddings, labels)

    assert torch.equal(state.counts.cpu(), torch.tensor([2, 2]))
    blended = torch.nn.functional.normalize(
        0.5 * expected_first + 0.5 * torch.tensor([1.0, 0.0]),
        dim=0,
    )
    assert torch.allclose(state.means[0].cpu(), blended, atol=1e-6)


def test_gsi_objectives_add_weighted_gsi_term_to_base_loss() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import (
        _confusion_axes,
        _gsi_interference_loss,
        _loss_for_objective,
    )

    embeddings, labels, proxies, proxy_labels = _gsi_toy_batch(torch)
    config = ImageEndToEndConfig(
        proxy_count_per_class=1,
        proxy_anchor_alpha=2.0,
        proxy_anchor_delta=0.0,
        potential_delta=0.3,
        potential_alpha=2.0,
        gsi_weight=0.5,
        gsi_floor=0.0,
        gsi_start_epoch=0,
        gsi_min_group_size=4,
    )
    expected_gsi = _gsi_interference_loss(
        embeddings,
        labels,
        axes_by_class=_confusion_axes(
            proxies,
            proxy_labels,
            top_k=config.gsi_top_k,
            torch_module=torch,
        ),
        floor=config.gsi_floor,
        variance_floor=config.gsi_variance_floor,
        min_group_size=config.gsi_min_group_size,
        torch_module=torch,
    )
    assert float(expected_gsi.detach()) > 0.0

    for gsi_objective, base_objective in (
        ("proxy_anchor_gsi", "proxy_anchor"),
        ("pfml_gsi", "pfml"),
    ):
        losses = {
            objective: _loss_for_objective(
                objective,  # type: ignore[arg-type]
                embeddings,
                labels,
                step=1,
                steps_per_epoch=1,
                memory_embeddings=None,
                memory_labels=None,
                proxy_embeddings=proxies,
                proxy_labels=proxy_labels,
                config=config,
                torch_module=torch,
            )
            for objective in (gsi_objective, base_objective)
        }

        expected = float(losses[base_objective].detach()) + 0.5 * float(expected_gsi.detach())
        assert float(losses[gsi_objective].detach()) == pytest.approx(expected, rel=1e-5)


def test_bgsi_objective_adds_weighted_boundary_gsi_term_to_proxy_anchor() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import (
        _boundary_confusion_axes,
        _gsi_interference_loss,
        _loss_for_objective,
        _proxy_anchor_loss,
    )

    embeddings, labels, proxies, proxy_labels = _gsi_toy_batch(torch)
    config = ImageEndToEndConfig(
        objectives=("proxy_anchor_bgsi",),
        proxy_count_per_class=1,
        proxy_anchor_alpha=1.0,
        proxy_anchor_delta=0.0,
        bgsi_weight=0.5,
        bgsi_floor=0.0,
        bgsi_start_epoch=0,
        bgsi_min_group_size=4,
    )
    base = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        torch_module=torch,
    )
    axes_by_class = _boundary_confusion_axes(
        embeddings,
        labels,
        top_k=config.bgsi_top_k,
        temperature=config.bgsi_temperature,
        torch_module=torch,
    )
    bgsi = _gsi_interference_loss(
        embeddings,
        labels,
        axes_by_class=axes_by_class,
        floor=config.bgsi_floor,
        variance_floor=config.bgsi_variance_floor,
        min_group_size=config.bgsi_min_group_size,
        torch_module=torch,
    )

    diagnostics: list[dict[str, float]] = []
    loss = _loss_for_objective(
        "proxy_anchor_bgsi",
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        memory_embeddings=None,
        memory_labels=None,
        config=config,
        step=1,
        steps_per_epoch=1,
        torch_module=torch,
        gsi_step_diagnostics=diagnostics,
    )

    assert float(bgsi.detach()) > 0.0
    assert float(loss.detach()) == pytest.approx(
        float((base + 0.5 * bgsi).detach()),
        rel=1e-5,
    )
    assert diagnostics
    assert diagnostics[0]["unweighted_loss"] > 0.0


def test_bgsi_objective_uses_configured_axis_mode_state_and_records_coverage() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import (
        BGSIClassMeanState,
        _bgsi_axes_for_mode,
        _gsi_interference_loss,
        _loss_for_objective,
        _proxy_anchor_loss,
    )

    embeddings, labels, proxies, proxy_labels = _gsi_toy_batch(torch)
    state = BGSIClassMeanState(
        labels=[0, 1],
        embedding_dimensions=3,
        momentum=0.5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        torch_module=torch,
    )
    state.update(embeddings, labels)
    state.update(embeddings, labels)
    config = ImageEndToEndConfig(
        objectives=("proxy_anchor_bgsi",),
        proxy_count_per_class=1,
        proxy_anchor_alpha=1.0,
        proxy_anchor_delta=0.0,
        bgsi_weight=0.5,
        bgsi_floor=0.0,
        bgsi_start_epoch=0,
        bgsi_min_group_size=4,
        bgsi_axis_mode="ema_boundary",
        bgsi_min_axis_observations=2,
        bgsi_use_axis_agreement_gate=False,
    )

    axes_by_class = _bgsi_axes_for_mode(
        embeddings,
        labels,
        axis_mode=config.bgsi_axis_mode,
        top_k=config.bgsi_top_k,
        temperature=config.bgsi_temperature,
        generator=torch.Generator(),
        ema_state=state,
        min_axis_observations=config.bgsi_min_axis_observations,
        use_axis_agreement_gate=config.bgsi_use_axis_agreement_gate,
        axis_agreement=config.bgsi_axis_agreement,
        torch_module=torch,
    )
    expected_bgsi = _gsi_interference_loss(
        embeddings,
        labels,
        axes_by_class=axes_by_class,
        floor=config.bgsi_floor,
        variance_floor=config.bgsi_variance_floor,
        min_group_size=config.bgsi_min_group_size,
        torch_module=torch,
    )
    base = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        torch_module=torch,
    )

    diagnostics: list[dict[str, float]] = []
    loss = _loss_for_objective(
        "proxy_anchor_bgsi",
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        memory_embeddings=None,
        memory_labels=None,
        config=config,
        step=1,
        steps_per_epoch=1,
        torch_module=torch,
        generator=torch.Generator(),
        bgsi_state=state,
        gsi_step_diagnostics=diagnostics,
    )

    assert float(loss.detach()) == pytest.approx(float((base + 0.5 * expected_bgsi).detach()))
    assert diagnostics
    assert diagnostics[0]["bgsi_axis_coverage"] == pytest.approx(1.0)
    assert diagnostics[0]["bgsi_axis_count"] == pytest.approx(1.0)
    assert diagnostics[0]["bgsi_ema_ready_fraction"] == pytest.approx(1.0)


def test_bgsi_gradient_step_reduces_boundary_axis_alignment() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _boundary_confusion_axes, _gsi_interference_loss

    embeddings = torch.tensor(
        [
            [0.8, 0.2],
            [1.2, -0.2],
            [0.9, 0.1],
            [1.1, -0.1],
            [-1.0, 0.0],
            [-1.1, 0.1],
            [-0.9, -0.1],
            [-1.0, 0.2],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    axes_by_class = _boundary_confusion_axes(
        embeddings,
        labels,
        top_k=1,
        temperature=0.1,
        torch_module=torch,
    )

    def alignment(matrix: Any) -> float:
        class_embeddings = matrix[labels == 0]
        centered = class_embeddings - class_embeddings.mean(dim=0, keepdim=True)
        axis = axes_by_class[0][0][0]
        parallel = (centered @ axis).pow(2).mean()
        total = centered.pow(2).sum(dim=1).mean()
        return float((parallel / total).detach())

    before = alignment(embeddings)
    loss = _gsi_interference_loss(
        embeddings,
        labels,
        axes_by_class=axes_by_class,
        floor=0.0,
        variance_floor=1e-4,
        min_group_size=4,
        torch_module=torch,
    )
    loss.backward()
    assert embeddings.grad is not None
    stepped = (embeddings - 1e-3 * embeddings.grad).detach()

    assert alignment(stepped) < before
    assert torch.allclose(
        embeddings.grad[labels == 0].sum(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )


def test_proxy_axis_interference_diagnostics_measure_training_axes() -> None:
    from sfora.image_end_to_end import _proxy_axis_interference_diagnostics

    embeddings = np.asarray(
        [
            [1.2, 0.0],
            [0.8, 0.0],
            [-1.0, 0.2],
            [-1.0, -0.2],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    proxies = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    proxy_labels = np.asarray([0, 1], dtype=np.int64)

    diagnostics = _proxy_axis_interference_diagnostics(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        top_k=1,
        floor=0.02,
    )

    assert diagnostics is not None
    assert set(diagnostics) == {
        "proxy_axis_rho_mean",
        "proxy_axis_rho_p90",
        "proxy_axis_rho_max",
        "proxy_axis_fraction_above_floor",
    }
    assert diagnostics["proxy_axis_rho_mean"] == pytest.approx(0.5)
    assert diagnostics["proxy_axis_rho_p90"] == pytest.approx(0.9)
    assert diagnostics["proxy_axis_rho_max"] == pytest.approx(1.0)
    assert diagnostics["proxy_axis_fraction_above_floor"] == pytest.approx(0.5)


def test_boundary_axis_interference_diagnostics_measure_batch_mean_axes() -> None:
    from sfora.image_end_to_end import _boundary_axis_interference_diagnostics

    embeddings = np.asarray(
        [
            [1.2, 0.0],
            [0.8, 0.0],
            [-1.0, 0.2],
            [-1.0, -0.2],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    diagnostics = _boundary_axis_interference_diagnostics(
        embeddings,
        labels,
        top_k=1,
        floor=0.02,
        temperature=0.1,
    )

    assert diagnostics is not None
    assert set(diagnostics) == {
        "boundary_axis_rho_mean",
        "boundary_axis_rho_p90",
        "boundary_axis_rho_max",
        "boundary_axis_fraction_above_floor",
    }
    assert diagnostics["boundary_axis_rho_mean"] == pytest.approx(0.5)
    assert diagnostics["boundary_axis_rho_p90"] == pytest.approx(0.9)
    assert diagnostics["boundary_axis_rho_max"] == pytest.approx(1.0)
    assert diagnostics["boundary_axis_fraction_above_floor"] == pytest.approx(0.5)


def test_gsi_term_is_inactive_until_start_epoch() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective

    embeddings, labels, proxies, proxy_labels = _gsi_toy_batch(torch)
    config = ImageEndToEndConfig(
        proxy_count_per_class=1,
        proxy_anchor_alpha=2.0,
        proxy_anchor_delta=0.0,
        gsi_weight=0.5,
        gsi_floor=0.0,
        gsi_start_epoch=2,
        gsi_min_group_size=4,
    )

    def loss_at_step(objective: str, step: int) -> float:
        return float(
            _loss_for_objective(
                objective,  # type: ignore[arg-type]
                embeddings,
                labels,
                step=step,
                steps_per_epoch=3,
                memory_embeddings=None,
                memory_labels=None,
                proxy_embeddings=proxies,
                proxy_labels=proxy_labels,
                config=config,
                torch_module=torch,
            ).detach()
        )

    base = loss_at_step("proxy_anchor", 6)
    assert loss_at_step("proxy_anchor_gsi", 6) == pytest.approx(base)
    assert loss_at_step("proxy_anchor_gsi", 7) > base


def test_gsi_objectives_require_proxies() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _loss_for_objective, _uses_metric_proxies

    for objective in ("proxy_anchor_gsi", "pfml_gsi"):
        with pytest.raises(ValueError, match=f"{objective}.*proxy_count_per_class"):
            _uses_metric_proxies(objective, ImageEndToEndConfig(proxy_count_per_class=0))
        assert _uses_metric_proxies(objective, ImageEndToEndConfig(proxy_count_per_class=1)) is True
        with pytest.raises(ValueError, match=f"{objective}.*prox"):
            _loss_for_objective(
                objective,
                torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32),
                torch.tensor([0, 1]),
                step=1,
                steps_per_epoch=1,
                memory_embeddings=None,
                memory_labels=None,
                proxy_embeddings=None,
                proxy_labels=None,
                config=ImageEndToEndConfig(),
                torch_module=torch,
            )


def test_gsi_top_k_is_clamped_to_available_classes() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _confusion_axes, _gsi_axes_for_mode

    proxies = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32)
    proxy_labels = torch.tensor([0, 1])

    axes_by_class = _confusion_axes(proxies, proxy_labels, top_k=3, torch_module=torch)
    assert set(axes_by_class) == {0, 1}
    for axes, weights in axes_by_class.values():
        assert tuple(axes.shape) == (1, 3)
        assert tuple(weights.shape) == (1,)

    for axis_mode in ("random", "global"):
        clamped = _gsi_axes_for_mode(
            proxies,
            proxy_labels,
            axis_mode=axis_mode,
            top_k=3,
            generator=torch.Generator(),
            torch_module=torch,
        )
        assert all(tuple(axes.shape) == (1, 3) for axes, _ in clamped.values())


def test_boundary_confusion_axes_use_batch_class_means() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _boundary_confusion_axes

    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.2],
            [0.0, 1.0],
            [0.2, 1.0],
            [-1.0, 0.0],
            [-1.0, 0.2],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    axes_by_class = _boundary_confusion_axes(
        embeddings,
        labels,
        top_k=1,
        temperature=0.1,
        torch_module=torch,
    )

    assert set(axes_by_class) == {0, 1, 2}
    axes, weights = axes_by_class[0]
    assert axes.shape == (1, 2)
    assert weights.shape == (1,)
    expected_axis = torch.nn.functional.normalize(
        embeddings[labels == 1].mean(dim=0) - embeddings[labels == 0].mean(dim=0),
        dim=0,
    )
    assert torch.allclose(axes[0], expected_axis, atol=1e-6)
    assert torch.allclose(weights, torch.ones_like(weights))
    assert not axes.requires_grad
    assert not weights.requires_grad


def test_boundary_confusion_axes_rank_top_k_by_similarity() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _boundary_confusion_axes

    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [0.8, 0.2],
            [0.8, 0.3],
            [-1.0, 0.0],
            [-1.0, 0.1],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    axes_by_class = _boundary_confusion_axes(
        embeddings,
        labels,
        top_k=2,
        temperature=0.1,
        torch_module=torch,
    )

    axes, weights = axes_by_class[0]
    nearest = torch.nn.functional.normalize(
        embeddings[labels == 1].mean(dim=0) - embeddings[labels == 0].mean(dim=0),
        dim=0,
    )
    far = torch.nn.functional.normalize(
        embeddings[labels == 2].mean(dim=0) - embeddings[labels == 0].mean(dim=0),
        dim=0,
    )
    assert torch.allclose(axes[0], nearest, atol=1e-6)
    assert torch.allclose(axes[1], far, atol=1e-6)
    assert float(weights[0]) > float(weights[1])
    assert float(weights.sum()) == pytest.approx(1.0)


def test_bgsi_ema_boundary_axes_wait_for_ready_counts_and_rank_confusers() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import BGSIClassMeanState, _bgsi_axes_for_mode

    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [0.8, 0.2],
            [0.8, 0.3],
            [-1.0, 0.0],
            [-1.0, 0.1],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    state = BGSIClassMeanState(
        labels=[0, 1, 2],
        embedding_dimensions=2,
        momentum=0.5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        torch_module=torch,
    )
    state.update(embeddings, labels)

    not_ready = _bgsi_axes_for_mode(
        embeddings,
        labels,
        axis_mode="ema_boundary",
        top_k=2,
        temperature=0.1,
        generator=None,
        ema_state=state,
        min_axis_observations=2,
        use_axis_agreement_gate=False,
        axis_agreement=0.5,
        torch_module=torch,
    )
    assert not_ready == {}

    state.update(embeddings, labels)
    ready = _bgsi_axes_for_mode(
        embeddings,
        labels,
        axis_mode="ema_boundary",
        top_k=2,
        temperature=0.1,
        generator=None,
        ema_state=state,
        min_axis_observations=2,
        use_axis_agreement_gate=False,
        axis_agreement=0.5,
        torch_module=torch,
    )

    axes, weights = ready[0]
    nearest = torch.nn.functional.normalize(state.means[1] - state.means[0], dim=0)
    far = torch.nn.functional.normalize(state.means[2] - state.means[0], dim=0)
    assert torch.allclose(axes[0], nearest, atol=1e-6)
    assert torch.allclose(axes[1], far, atol=1e-6)
    assert float(weights[0]) > float(weights[1])
    assert not axes.requires_grad
    assert not weights.requires_grad


def test_bgsi_ema_boundary_agreement_gate_rejects_disagreeing_axes() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import BGSIClassMeanState, _bgsi_axes_for_mode

    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [0.9, 0.2],
            [0.9, 0.3],
            [-1.0, 0.0],
            [-1.0, 0.1],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    state = BGSIClassMeanState(
        labels=[0, 1, 2],
        embedding_dimensions=2,
        momentum=0.5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        torch_module=torch,
    )
    state.means[0] = torch.tensor([1.0, 0.0])
    state.means[1] = torch.tensor([-1.0, 0.0])
    state.means[2] = torch.tensor([0.0, 1.0])
    state.counts[:] = 3

    rejected = _bgsi_axes_for_mode(
        embeddings,
        labels,
        axis_mode="ema_boundary",
        top_k=1,
        temperature=0.1,
        generator=None,
        ema_state=state,
        min_axis_observations=2,
        use_axis_agreement_gate=True,
        axis_agreement=0.95,
        torch_module=torch,
    )
    assert 0 not in rejected


def test_bgsi_control_axis_modes_are_deterministic_and_detached() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import BGSIClassMeanState, _bgsi_axes_for_mode

    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    state = BGSIClassMeanState(
        labels=[0, 1, 2],
        embedding_dimensions=3,
        momentum=0.5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        torch_module=torch,
    )
    state.update(embeddings, labels)
    state.update(embeddings, labels)

    for mode in ("random", "permuted", "global"):
        generator = torch.Generator()
        generator.manual_seed(123)
        first = _bgsi_axes_for_mode(
            embeddings,
            labels,
            axis_mode=mode,
            top_k=2,
            temperature=0.1,
            generator=generator,
            ema_state=state,
            min_axis_observations=2,
            use_axis_agreement_gate=False,
            axis_agreement=0.5,
            torch_module=torch,
        )
        generator.manual_seed(123)
        second = _bgsi_axes_for_mode(
            embeddings,
            labels,
            axis_mode=mode,
            top_k=2,
            temperature=0.1,
            generator=generator,
            ema_state=state,
            min_axis_observations=2,
            use_axis_agreement_gate=False,
            axis_agreement=0.5,
            torch_module=torch,
        )
        assert set(first) == {0, 1, 2}
        for label in (0, 1, 2):
            axes, weights = first[label]
            assert torch.allclose(axes, second[label][0])
            assert torch.allclose(weights, second[label][1])
            assert not axes.requires_grad
            assert not weights.requires_grad
            assert torch.allclose(weights.sum(), torch.tensor(1.0))


def test_gsi_random_axis_mode_resamples_unit_axes_from_generator() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _gsi_axes_for_mode

    proxies = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    proxy_labels = torch.tensor([0, 1, 2])

    def draw(generator: Any) -> dict[int, tuple[Any, Any]]:
        return _gsi_axes_for_mode(
            proxies,
            proxy_labels,
            axis_mode="random",
            top_k=2,
            generator=generator,
            torch_module=torch,
        )

    generator = torch.Generator()
    generator.manual_seed(11)
    first = draw(generator)
    second = draw(generator)
    generator.manual_seed(11)
    replayed = draw(generator)

    for label in (0, 1, 2):
        axes, weights = first[label]
        assert tuple(axes.shape) == (2, 3)
        assert torch.allclose(axes.norm(dim=1), torch.ones(2), atol=1e-5)
        assert torch.allclose(weights, torch.full((2,), 0.5))
        assert not torch.allclose(axes, second[label][0])
        assert torch.allclose(axes, replayed[label][0])


def test_gsi_global_axis_mode_shares_proxy_principal_components() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _gsi_axes_for_mode

    proxies = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    proxy_labels = torch.tensor([0, 1, 2])

    axes_by_class = _gsi_axes_for_mode(
        proxies,
        proxy_labels,
        axis_mode="global",
        top_k=2,
        generator=None,
        torch_module=torch,
    )

    assert set(axes_by_class) == {0, 1, 2}
    reference_axes, reference_weights = axes_by_class[0]
    assert tuple(reference_axes.shape) == (2, 3)
    assert torch.allclose(reference_weights, torch.full((2,), 0.5))
    for label in (1, 2):
        assert torch.allclose(axes_by_class[label][0], reference_axes)
    # The proxy scatter is dominated by e0 (variance 2) and then e1 (2/3).
    assert abs(float(reference_axes[0] @ torch.tensor([1.0, 0.0, 0.0]))) == pytest.approx(
        1.0, abs=1e-5
    )
    assert abs(float(reference_axes[1] @ torch.tensor([0.0, 1.0, 0.0]))) == pytest.approx(
        1.0, abs=1e-5
    )


def test_gsi_global_axis_mode_clamps_axes_to_available_svd_axes() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _gsi_axes_for_mode, _gsi_interference_loss

    # Four classes in a 2-dim embedding space: SVD yields at most 2 axes even
    # though top_k=3 and (class count - 1) = 3.
    proxies = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=torch.float32,
    )
    proxy_labels = torch.tensor([0, 1, 2, 3])

    axes_by_class = _gsi_axes_for_mode(
        proxies,
        proxy_labels,
        axis_mode="global",
        top_k=3,
        generator=None,
        torch_module=torch,
    )

    assert set(axes_by_class) == {0, 1, 2, 3}
    for axes, weights in axes_by_class.values():
        assert tuple(axes.shape) == (2, 2)
        assert tuple(weights.shape) == (2,)
        assert torch.allclose(weights, torch.full((2,), 0.5))

    embeddings = torch.tensor(
        [
            [0.9, 0.1],
            [1.1, -0.1],
            [0.8, 0.2],
            [1.2, -0.2],
            [-0.9, 0.1],
            [-1.1, -0.1],
            [-0.8, 0.2],
            [-1.2, -0.2],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    loss = _gsi_interference_loss(
        embeddings,
        labels,
        axes_by_class=axes_by_class,
        floor=0.0,
        variance_floor=1e-4,
        min_group_size=4,
        torch_module=torch,
    )
    assert torch.isfinite(loss)


def test_confusion_axes_rank_multiple_confusers_by_max_cosine_with_softmax_weights() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _confusion_axes

    def unit(degrees: float) -> list[float]:
        radians = math.radians(degrees)
        return [math.cos(radians), math.sin(radians)]

    def cosine(degrees: float) -> float:
        return math.cos(math.radians(degrees))

    def normed(vector: Any) -> Any:
        return vector / vector.norm()

    # Class 1 has two proxies (10 and 80 degrees) to exercise max-cosine
    # nearest-proxy-pair selection; all proxies are unit vectors.
    proxies = torch.tensor(
        [unit(0.0), unit(10.0), unit(80.0), unit(40.0), unit(85.0)],
        dtype=torch.float32,
    )
    proxy_labels = torch.tensor([0, 1, 1, 2, 3])

    axes_by_class = _confusion_axes(proxies, proxy_labels, top_k=3, torch_module=torch)
    assert set(axes_by_class) == {0, 1, 2, 3}

    # Class 0 ranking: class 1 (cos 10 via its 10-degree proxy), class 2
    # (cos 40), class 3 (cos 85).
    axes, weights = axes_by_class[0]
    assert tuple(axes.shape) == (3, 2)
    expected_scores = torch.tensor([cosine(10.0), cosine(40.0), cosine(85.0)])
    expected_axes = torch.stack(
        [
            normed(proxies[1] - proxies[0]),
            normed(proxies[3] - proxies[0]),
            normed(proxies[4] - proxies[0]),
        ]
    )
    assert torch.allclose(axes, expected_axes, atol=1e-6)
    assert torch.allclose(weights, torch.softmax(expected_scores, dim=0), atol=1e-6)
    assert float(weights[0]) > float(weights[1]) > float(weights[2])
    assert float(weights.sum()) == pytest.approx(1.0)

    # Class 1 ranking uses the max cosine over its own two proxies: class 3
    # (cos 5 via the 80-degree proxy), class 0 (cos 10 via the 10-degree
    # proxy), class 2 (cos 30 via the 10-degree proxy).
    axes, weights = axes_by_class[1]
    expected_scores = torch.tensor([cosine(5.0), cosine(10.0), cosine(30.0)])
    expected_axes = torch.stack(
        [
            normed(proxies[4] - proxies[2]),
            normed(proxies[0] - proxies[1]),
            normed(proxies[3] - proxies[1]),
        ]
    )
    assert torch.allclose(axes, expected_axes, atol=1e-6)
    assert torch.allclose(weights, torch.softmax(expected_scores, dim=0), atol=1e-6)

    # top_k=2 keeps only the two hardest confusers with re-normalized softmax.
    top_two = _confusion_axes(proxies, proxy_labels, top_k=2, torch_module=torch)
    axes, weights = top_two[0]
    assert tuple(axes.shape) == (2, 2)
    assert torch.allclose(
        axes,
        torch.stack([normed(proxies[1] - proxies[0]), normed(proxies[3] - proxies[0])]),
        atol=1e-6,
    )
    assert torch.allclose(
        weights,
        torch.softmax(torch.tensor([cosine(10.0), cosine(40.0)]), dim=0),
        atol=1e-6,
    )


def test_gsi_objective_display_names() -> None:
    from sfora.image_end_to_end import _objective_display_name

    config = ImageEndToEndConfig(objectives=("proxy_anchor_gsi", "pfml_gsi"))

    assert config.objectives == ("proxy_anchor_gsi", "pfml_gsi")
    assert _objective_display_name("proxy_anchor_gsi") == "Proxy Anchor + GSI"
    assert _objective_display_name("pfml_gsi") == "PFML + GSI"


def test_bgsi_objective_display_name() -> None:
    from sfora.image_end_to_end import _objective_display_name

    config = ImageEndToEndConfig(objectives=("proxy_anchor_bgsi",))

    assert config.objectives == ("proxy_anchor_bgsi",)
    assert _objective_display_name("proxy_anchor_bgsi") == "Proxy Anchor + BGSI"


def test_gsi_objective_trains_end_to_end(tmp_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")

    from sfora.image_end_to_end import write_image_end_to_end_report

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 2)

        def forward(self, images: object) -> object:
            return self.embedding(torch.as_tensor(images, dtype=torch.long))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.long)

        return transform

    examples = [
        ImageExample(example_id=f"{label}-{index}", image=label * 4 + index, label=label)
        for label in (0, 1)
        for index in range(4)
    ]
    models: list[Any] = []

    def model_factory(config: ImageEndToEndConfig) -> Any:
        model = TinyModel()
        models.append(model)
        return model

    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("proxy_anchor_gsi",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=8,
            eval_batch_size=8,
            train_steps=3,
            group_size=2,
            learning_rate=0.05,
            proxy_count_per_class=1,
            proxy_learning_rate_multiplier=1.0,
            proxy_anchor_alpha=2.0,
            proxy_anchor_delta=0.0,
            gsi_weight=0.5,
            gsi_floor=0.0,
            gsi_start_epoch=0,
            gsi_min_group_size=4,
            progress_every=0,
            num_workers=0,
            seed=0,
        ),
        model_factory=model_factory,
        transform_factory=transform_factory,
    )

    model = models[0]
    assert model.metric_proxies.grad is not None
    assert float(model.metric_proxies.grad.norm().detach().cpu()) > 0.0
    metrics = result.methods["proxy_anchor_gsi_end_to_end:tiny"]
    assert len(metrics.loss_history) == 3
    assert metrics.display_name == "Proxy Anchor + GSI"
    assert metrics.gsi_diagnostics is not None
    assert set(metrics.gsi_diagnostics) == {
        "active_steps",
        "unweighted_loss_mean",
        "unweighted_loss_p90",
        "unweighted_loss_max",
        "active_fraction_mean",
        "proxy_axis_rho_mean",
        "proxy_axis_rho_p90",
        "proxy_axis_rho_max",
        "proxy_axis_fraction_above_floor",
    }
    assert metrics.gsi_diagnostics["active_steps"] == pytest.approx(3.0)
    assert metrics.gsi_diagnostics["unweighted_loss_mean"] > 0.0
    assert 0.0 <= metrics.gsi_diagnostics["active_fraction_mean"] <= 1.0

    output = tmp_path / "image_end_to_end.json"
    write_image_end_to_end_report(result, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    method = payload["methods"]["proxy_anchor_gsi_end_to_end:tiny"]
    assert method["gsi_diagnostics"] == metrics.gsi_diagnostics


def test_bgsi_objective_trains_end_to_end(tmp_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")

    from sfora.image_end_to_end import write_image_end_to_end_report

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 2)

        def forward(self, images: object) -> object:
            return self.embedding(torch.as_tensor(images, dtype=torch.long))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.long)

        return transform

    examples = [
        ImageExample(example_id=f"{label}-{index}", image=label * 4 + index, label=label)
        for label in (0, 1)
        for index in range(4)
    ]
    models: list[Any] = []

    def model_factory(config: ImageEndToEndConfig) -> Any:
        model = TinyModel()
        models.append(model)
        return model

    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("proxy_anchor_bgsi",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=8,
            eval_batch_size=8,
            train_steps=3,
            group_size=2,
            learning_rate=0.05,
            proxy_count_per_class=1,
            proxy_learning_rate_multiplier=1.0,
            proxy_anchor_alpha=2.0,
            proxy_anchor_delta=0.0,
            bgsi_weight=0.5,
            bgsi_floor=0.0,
            bgsi_start_epoch=0,
            bgsi_min_group_size=4,
            progress_every=0,
            num_workers=0,
            seed=0,
        ),
        model_factory=model_factory,
        transform_factory=transform_factory,
    )

    model = models[0]
    assert model.metric_proxies.grad is not None
    assert float(model.metric_proxies.grad.norm().detach().cpu()) > 0.0
    metrics = result.methods["proxy_anchor_bgsi_end_to_end:tiny"]
    assert len(metrics.loss_history) == 3
    assert metrics.display_name == "Proxy Anchor + BGSI"
    assert metrics.gsi_diagnostics is not None
    assert set(metrics.gsi_diagnostics) == {
        "active_steps",
        "unweighted_loss_mean",
        "unweighted_loss_p90",
        "unweighted_loss_max",
        "active_fraction_mean",
        "bgsi_axis_coverage_mean",
        "bgsi_axis_count_mean",
        "bgsi_ema_ready_fraction_mean",
        "boundary_axis_rho_mean",
        "boundary_axis_rho_p90",
        "boundary_axis_rho_max",
        "boundary_axis_fraction_above_floor",
    }
    assert metrics.gsi_diagnostics["active_steps"] == pytest.approx(3.0)
    assert metrics.gsi_diagnostics["unweighted_loss_mean"] > 0.0
    assert 0.0 <= metrics.gsi_diagnostics["active_fraction_mean"] <= 1.0
    assert 0.0 <= metrics.gsi_diagnostics["bgsi_axis_coverage_mean"] <= 1.0
    assert metrics.gsi_diagnostics["bgsi_axis_count_mean"] >= 0.0
    assert 0.0 <= metrics.gsi_diagnostics["bgsi_ema_ready_fraction_mean"] <= 1.0

    output = tmp_path / "image_end_to_end.json"
    write_image_end_to_end_report(result, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    method = payload["methods"]["proxy_anchor_bgsi_end_to_end:tiny"]
    assert method["gsi_diagnostics"] == metrics.gsi_diagnostics


def test_proxy_anchor_baseline_serializes_boundary_diagnostics() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 2)

        def forward(self, images: object) -> object:
            return self.embedding(torch.as_tensor(images, dtype=torch.long))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.long)

        return transform

    examples = [
        ImageExample(example_id=f"{label}-{index}", image=label * 4 + index, label=label)
        for label in (0, 1)
        for index in range(4)
    ]

    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("proxy_anchor",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=8,
            eval_batch_size=8,
            train_steps=1,
            group_size=2,
            learning_rate=0.05,
            proxy_count_per_class=1,
            proxy_learning_rate_multiplier=1.0,
            proxy_anchor_alpha=2.0,
            proxy_anchor_delta=0.0,
            progress_every=0,
            num_workers=0,
            seed=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    metrics = result.methods["proxy_anchor_end_to_end:tiny"]
    assert metrics.gsi_diagnostics is not None
    assert "proxy_axis_rho_mean" in metrics.gsi_diagnostics
    assert "boundary_axis_rho_mean" in metrics.gsi_diagnostics


def test_label_noise_corrupts_training_labels_deterministically() -> None:
    from sfora.image_end_to_end import _apply_training_label_noise

    examples = [
        ImageExample(
            example_id=f"example-{label}-{index}",
            image=f"image-{label}-{index}",
            label=label,
        )
        for label in (0, 1, 2)
        for index in range(10)
    ]

    first = _apply_training_label_noise(examples, fraction=0.2, seed=7)
    second = _apply_training_label_noise(examples, fraction=0.2, seed=7)

    changed = [
        (before, after)
        for before, after in zip(examples, first, strict=True)
        if before.label != after.label
    ]
    assert len(changed) == 6
    assert first == second
    assert [example.example_id for example in first] == [example.example_id for example in examples]
    assert [example.image for example in first] == [example.image for example in examples]
    assert all(before.label != after.label for before, after in changed)


def test_supcon_can_exclude_self_from_prefixed_memory_contrast() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _supervised_contrastive_loss

    anchors = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        dim=-1,
    )
    labels = torch.tensor([0, 1])
    memory = torch.nn.functional.normalize(
        torch.tensor([[-1.0, 0.0], [0.0, -1.0]], dtype=torch.float32),
        dim=-1,
    )
    memory_labels = torch.tensor([2, 3])

    loss = _supervised_contrastive_loss(
        anchors,
        labels,
        contrast_embeddings=torch.cat([anchors, memory], dim=0),
        contrast_labels=torch.cat([labels, memory_labels], dim=0),
        temperature=0.07,
        torch_module=torch,
        exclude_self=True,
    )

    assert loss == pytest.approx(0.0)


def test_resnet_factory_can_emit_pretrained_features_without_random_head() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _set_resnet_output_layer

    model = torch.nn.Module()
    model.fc = torch.nn.Linear(2048, 1000)

    _set_resnet_output_layer(
        model,
        ImageEndToEndConfig(embedding_dimensions=512),
        use_embedding_head=False,
        torch_module=torch,
    )

    assert isinstance(model.fc, torch.nn.Identity)


def test_optimizer_groups_use_lower_backbone_learning_rate() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _optimizer_parameter_groups

    class TinyResNet(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Linear(2, 2)
            self.fc = torch.nn.Linear(2, 2)

    model = TinyResNet()
    groups = _optimizer_parameter_groups(
        model,
        ImageEndToEndConfig(learning_rate=5e-4, backbone_learning_rate=1e-5),
    )

    assert [group["lr"] for group in groups] == [1e-5, 5e-4]
    assert list(groups[0]["params"]) == list(model.conv.parameters())
    assert list(groups[1]["params"]) == list(model.fc.parameters())


def test_metric_proxies_attach_one_parameter_per_train_class_proxy() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _attach_metric_proxies

    model = torch.nn.Linear(2, 2)

    _attach_metric_proxies(
        model,
        train_labels=[4, 4, 7],
        config=ImageEndToEndConfig(
            embedding_dimensions=3,
            proxy_count_per_class=2,
        ),
        torch_module=torch,
    )

    assert tuple(model.metric_proxies.shape) == (4, 3)
    assert model.metric_proxy_labels.tolist() == [4, 4, 7, 7]
    assert model.metric_proxies.requires_grad is True


def test_optimizer_groups_use_high_proxy_learning_rate() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _attach_metric_proxies, _optimizer_parameter_groups

    class TinyResNet(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Linear(2, 2)
            self.fc = torch.nn.Linear(2, 2)

    model = TinyResNet()
    config = ImageEndToEndConfig(
        learning_rate=5e-4,
        backbone_learning_rate=1e-5,
        proxy_count_per_class=1,
        proxy_learning_rate_multiplier=100.0,
    )
    _attach_metric_proxies(model, train_labels=[0, 1], config=config, torch_module=torch)

    groups = _optimizer_parameter_groups(model, config)

    assert [group["lr"] for group in groups] == [1e-5, 5e-4, 5e-2]
    assert list(groups[2]["params"]) == [model.metric_proxies]


def test_balanced_batch_indices_samples_per_class_excludes_short_classes_without_duplicates() -> (
    None
):
    from sfora.image_end_to_end import _balanced_batch_indices

    labels = [label for label in range(30) for _ in range(4)] + [99, 99, 99]

    batches = _balanced_batch_indices(
        labels,
        batch_size=120,
        group_size=8,
        samples_per_class=4,
        steps=2,
        seed=123,
    )

    assert len(batches) == 2
    for batch in batches:
        batch_labels = [labels[index] for index in batch]
        assert len(batch) == 120
        assert len(set(batch)) == 120
        assert set(batch_labels) == set(range(30))
        assert 99 not in batch_labels
        assert all(batch_labels.count(label) == 4 for label in range(30))


def test_balanced_batch_indices_samples_per_class_zero_preserves_legacy_sequence() -> None:
    from sfora.image_end_to_end import _balanced_batch_indices

    labels = [0, 0, 0, 1, 1, 2, 2, 2, 2]

    batches = _balanced_batch_indices(
        labels,
        batch_size=8,
        group_size=2,
        samples_per_class=0,
        steps=3,
        seed=123,
    )

    assert batches == [
        [0, 2, 0, 0, 8, 5, 7, 6],
        [4, 4, 4, 4, 7, 6, 8, 5],
        [2, 0, 1, 1, 4, 4, 3, 4],
    ]


def test_balanced_batch_indices_hard_class_sampling_groups_confusable_classes() -> None:
    from sfora.image_end_to_end import _balanced_batch_indices

    # 6 classes, 4 samples each; batch holds 2 classes (batch_size 8, K=4). A
    # confusability graph makes class pairs (0,1), (2,3), (4,5) each other's nearest
    # neighbour. Hard sampling (fraction 1.0) must put a seed class with its nearest
    # neighbour, never an unrelated class.
    labels = [label for label in range(6) for _ in range(4)]
    similarity = {
        0: [1, 2, 3, 4, 5],
        1: [0, 2, 3, 4, 5],
        2: [3, 0, 1, 4, 5],
        3: [2, 0, 1, 4, 5],
        4: [5, 0, 1, 2, 3],
        5: [4, 0, 1, 2, 3],
    }
    batches = _balanced_batch_indices(
        labels,
        batch_size=8,
        group_size=2,
        samples_per_class=4,
        steps=20,
        seed=7,
        class_similarity=similarity,
        hard_fraction=1.0,
    )
    pair = {frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5})}
    for batch in batches:
        classes = frozenset(labels[i] for i in batch)
        assert classes in pair  # every hard batch is a confusable pair


def test_balanced_batch_indices_hard_fraction_zero_matches_random() -> None:
    from sfora.image_end_to_end import _balanced_batch_indices

    labels = [label for label in range(6) for _ in range(4)]
    similarity = {0: [1], 1: [0], 2: [3], 3: [2], 4: [5], 5: [4]}
    plain = _balanced_batch_indices(
        labels,
        batch_size=8,
        group_size=2,
        samples_per_class=4,
        steps=5,
        seed=7,
    )
    with_sim_no_hard = _balanced_batch_indices(
        labels,
        batch_size=8,
        group_size=2,
        samples_per_class=4,
        steps=5,
        seed=7,
        class_similarity=similarity,
        hard_fraction=0.0,
    )
    assert plain == with_sim_no_hard  # hard_fraction=0 -> identical to random


def test_default_transform_uses_full_res_random_resized_crop_for_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    transforms = pytest.importorskip("torchvision.transforms")

    from sfora.image_end_to_end import _default_transform_factory

    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class CapturingTransform:
        def __init__(self, name: str, *args: object, **kwargs: object) -> None:
            calls.append((name, args, kwargs))

        def __call__(self, image: object) -> object:
            return image

    def resize(*args: object, **kwargs: object) -> CapturingTransform:
        return CapturingTransform("Resize", *args, **kwargs)

    def random_resized_crop(*args: object, **kwargs: object) -> CapturingTransform:
        return CapturingTransform("RandomResizedCrop", *args, **kwargs)

    def center_crop(*args: object, **kwargs: object) -> CapturingTransform:
        return CapturingTransform("CenterCrop", *args, **kwargs)

    monkeypatch.setattr(transforms, "Resize", resize)
    monkeypatch.setattr(transforms, "RandomResizedCrop", random_resized_crop)
    monkeypatch.setattr(transforms, "CenterCrop", center_crop)

    config = ImageEndToEndConfig(input_size=224, train_augmentation="full_res_crop")

    _default_transform_factory(config, True)
    train_calls = list(calls)
    calls.clear()
    _default_transform_factory(config, False)
    eval_calls = list(calls)
    calls.clear()
    _default_transform_factory(ImageEndToEndConfig(input_size=224), True)
    legacy_train_calls = list(calls)

    assert [name for name, _, _ in train_calls if name == "Resize"] == []
    assert ("RandomResizedCrop", (224,), {"scale": (0.16, 1.0)}) in train_calls
    assert ("Resize", (256,), {}) in eval_calls
    assert ("CenterCrop", (224,), {}) in eval_calls
    # Legacy "standard" keeps the historical Resize(256) + default-scale crop so
    # old sota/hpl protocol runs remain comparable.
    assert ("Resize", (256,), {}) in legacy_train_calls
    assert ("RandomResizedCrop", (224,), {}) in legacy_train_calls


def test_repaired_presets_use_full_res_crop_augmentation() -> None:
    repaired: tuple[EndToEndProtocol, ...] = ("proxy-anchor-resnet50-512", "pfml-resnet50-512")
    legacy: tuple[EndToEndProtocol, ...] = ("sota-resnet50-512", "hpl-resnet50-512")
    for protocol in repaired:
        assert config_for_protocol(protocol, dataset_name="cub").train_augmentation == (
            "full_res_crop"
        )
    for protocol in legacy:
        assert config_for_protocol(protocol, dataset_name="cub").train_augmentation == "standard"


def test_group_centroid_objectives_reject_starved_samples_per_class() -> None:
    pytest.importorskip("torch")

    config = ImageEndToEndConfig(
        objectives=("group_supcon",),
        samples_per_class=4,
        group_size=4,
        batch_size=16,
        train_steps=1,
    )

    with pytest.raises(ValueError, match="samples_per_class >= 2 \\* group_size"):
        run_image_end_to_end_benchmark(
            train_examples=[
                ImageExample(example_id=f"train-{label}-{index}", image=[0.0], label=label)
                for label in range(4)
                for index in range(4)
            ],
            test_examples=[
                ImageExample(example_id=f"test-{label}-{index}", image=[0.0], label=label)
                for label in range(2)
                for index in range(2)
            ],
            config=config,
        )


def test_torchvision_model_factory_selects_v1_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("torchvision.models")

    from sfora.image_end_to_end import _torchvision_model_factory

    captured: dict[str, object] = {}

    class TinyResNet(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 1000)

    def resnet50(*, weights: object) -> TinyResNet:
        captured["weights"] = weights
        return TinyResNet()

    monkeypatch.setattr(models, "resnet50", resnet50)

    _torchvision_model_factory(ImageEndToEndConfig(pretrained_weights="v1", embedding_dimensions=2))

    assert captured["weights"] is models.ResNet50_Weights.IMAGENET1K_V1


def test_set_resnet_output_layer_can_sum_gap_and_gmp() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _set_resnet_output_layer

    class TinyResNet(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            pooled = self.avgpool(images)
            return self.fc(torch.flatten(pooled, 1))

    model = TinyResNet()
    _set_resnet_output_layer(
        model,
        ImageEndToEndConfig(
            embedding_dimensions=2,
            head_pooling="avg_max",
        ),
        use_embedding_head=True,
        torch_module=torch,
    )
    with torch.no_grad():
        model.fc.weight.copy_(torch.eye(2))
        model.fc.bias.zero_()

    output = model(
        torch.tensor(
            [[[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]]],
            dtype=torch.float32,
        )
    )

    assert output.tolist() == [[pytest.approx(11.0), pytest.approx(13.0)]]


def test_set_resnet_output_layer_kaiming_initializes_embedding_head() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _set_resnet_output_layer

    class FakeHead:
        in_features = 4

    class FakeResNet:
        def __init__(self) -> None:
            self.fc: Any = FakeHead()

    torch.manual_seed(17)
    model = FakeResNet()
    _set_resnet_output_layer(
        model,
        ImageEndToEndConfig(
            embedding_dimensions=6,
            embedding_head_init="kaiming_normal",
        ),
        use_embedding_head=True,
        torch_module=torch,
    )

    torch.manual_seed(17)
    expected = torch.nn.Linear(4, 6)
    torch.nn.init.kaiming_normal_(expected.weight, mode="fan_out")
    torch.nn.init.zeros_(expected.bias)

    torch.manual_seed(17)
    default_model = FakeResNet()
    _set_resnet_output_layer(
        default_model,
        ImageEndToEndConfig(embedding_dimensions=6),
        use_embedding_head=True,
        torch_module=torch,
    )

    assert torch.allclose(model.fc.weight, expected.weight)
    assert torch.allclose(model.fc.bias, torch.zeros_like(model.fc.bias))
    assert not torch.allclose(model.fc.weight, default_model.fc.weight)


def test_resolve_training_schedule_preserves_legacy_train_steps() -> None:
    from sfora.image_end_to_end import _resolve_training_schedule

    config = ImageEndToEndConfig(batch_size=4, train_steps=5, train_epochs=None)

    assert _resolve_training_schedule(config, optimization_example_count=9) == (5, 3, 2)


def test_resolve_training_schedule_recomputes_steps_from_train_epochs() -> None:
    from sfora.image_end_to_end import _resolve_training_schedule

    config = ImageEndToEndConfig(batch_size=4, train_steps=99, train_epochs=2)

    assert _resolve_training_schedule(config, optimization_example_count=9) == (6, 3, 2)


def test_train_epochs_schedule_uses_post_split_example_count() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    examples = [
        ImageExample(
            example_id=f"{label}-{index}",
            image=[float(label), float(index)],
            label=label,
        )
        for label in (0, 1)
        for index in range(5)
    ]

    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("supcon",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=5,
            train_steps=5,
            train_epochs=2,
            group_size=1,
            checkpoint_selection_interval=100,
            checkpoint_selection_validation_fraction=0.4,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    assert len(result.methods["supcon_end_to_end:tiny"].loss_history) == 4


def test_samples_per_class_zero_uses_shuffled_loader_not_balanced_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch: Any = pytest.importorskip("torch")

    from sfora import image_end_to_end

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    def fail_if_balanced_sampler_is_used(*args: object, **kwargs: object) -> list[list[int]]:
        raise AssertionError("samples_per_class=0 must use shuffled DataLoader batches")

    monkeypatch.setattr(
        image_end_to_end,
        "_balanced_batch_indices",
        fail_if_balanced_sampler_is_used,
    )
    examples = [
        ImageExample(
            example_id=f"{label}-{index}",
            image=[float(label), float(index)],
            label=label,
        )
        for label in (0, 1)
        for index in range(5)
    ]

    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("batch_hard_triplet",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=5,
            train_steps=5,
            train_epochs=None,
            samples_per_class=0,
            group_size=1,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    assert len(result.methods["batch_hard_triplet_end_to_end:tiny"].loss_history) == 5


def test_step_lr_scheduler_steps_only_after_complete_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch: Any = pytest.importorskip("torch")

    optimizers: list[Any] = []

    class TrackingAdam(torch.optim.Adam):  # type: ignore[misc]
        def __init__(self, params: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(params, *args, **kwargs)
            optimizers.append(self)

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    examples = [
        ImageExample(
            example_id=f"{label}-{index}",
            image=[float(label), float(index)],
            label=label,
        )
        for label in (0, 1)
        for index in range(3)
    ]
    monkeypatch.setattr(torch.optim, "Adam", TrackingAdam)

    run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("supcon",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=6,
            train_steps=3,
            group_size=1,
            learning_rate=0.1,
            lr_schedule="step",
            lr_step_epochs=1,
            lr_gamma=0.5,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    assert [group["lr"] for group in optimizers[0].param_groups] == [pytest.approx(0.05)]


def test_cosine_lr_scheduler_decays_to_zero_after_configured_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch: Any = pytest.importorskip("torch")

    optimizers: list[Any] = []

    class TrackingAdam(torch.optim.Adam):  # type: ignore[misc]
        def __init__(self, params: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(params, *args, **kwargs)
            optimizers.append(self)

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    examples = [
        ImageExample(
            example_id=f"{label}-{index}",
            image=[float(label), float(index)],
            label=label,
        )
        for label in (0, 1)
        for index in range(3)
    ]
    monkeypatch.setattr(torch.optim, "Adam", TrackingAdam)

    run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("supcon",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=6,
            train_steps=99,
            train_epochs=2,
            group_size=1,
            learning_rate=0.1,
            lr_schedule="cosine",
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    assert [group["lr"] for group in optimizers[0].param_groups] == [pytest.approx(0.0, abs=1e-8)]


def test_warmup_freezes_backbone_after_optimizer_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch: Any = pytest.importorskip("torch")

    optimizers: list[Any] = []

    class TrackingAdam(torch.optim.Adam):  # type: ignore[misc]
        def __init__(self, params: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(params, *args, **kwargs)
            optimizers.append(self)

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.fc = torch.nn.Linear(2, 2)
            self.backbone_requires_grad_by_step: list[tuple[bool, bool]] = []

        def forward(self, images: object) -> object:
            if self.training:
                self.backbone_requires_grad_by_step.append(
                    tuple(parameter.requires_grad for parameter in self.backbone.parameters())
                )
            return self.fc(self.backbone(images))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    examples = [
        ImageExample(
            example_id=f"{label}-{index}",
            image=[float(label), float(index)],
            label=label,
        )
        for label in (0, 1)
        for index in range(3)
    ]
    models: list[TinyModel] = []
    monkeypatch.setattr(torch.optim, "Adam", TrackingAdam)

    def model_factory(config: ImageEndToEndConfig) -> TinyModel:
        model = TinyModel()
        models.append(model)
        return model

    run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("supcon",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=6,
            train_steps=99,
            train_epochs=2,
            group_size=1,
            samples_per_class=2,
            learning_rate=0.1,
            backbone_learning_rate=0.1,
            warmup_epochs=1,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=model_factory,
        transform_factory=transform_factory,
    )

    model = models[0]
    assert model.backbone_requires_grad_by_step == [
        (False, False),
        (False, False),
        (True, True),
        (True, True),
    ]
    optimizer_parameter_ids = {
        id(parameter) for group in optimizers[0].param_groups for parameter in group["params"]
    }
    assert {id(parameter) for parameter in model.backbone.parameters()} <= optimizer_parameter_ids
    backbone_grad_norm = sum(
        float(parameter.grad.norm().detach().cpu())
        for parameter in model.backbone.parameters()
        if parameter.grad is not None
    )
    assert backbone_grad_norm > 0.0


def test_xbm_start_step_delays_memory_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch: Any = pytest.importorskip("torch")

    import sfora.image_end_to_end as image_end_to_end

    memory_visible_by_step: list[bool] = []

    def loss_for_objective(
        objective: object,
        embeddings: Any,
        labels: Any,
        *,
        step: int,
        steps_per_epoch: int,
        memory_embeddings: Any | None,
        memory_labels: Any | None,
        proxy_embeddings: Any | None,
        proxy_labels: Any | None,
        config: ImageEndToEndConfig,
        torch_module: Any,
        teacher_embeddings: Any | None = None,
        generator: Any | None = None,
        gsi_step_diagnostics: list[dict[str, float]] | None = None,
    ) -> Any:
        del objective, labels, proxy_embeddings, proxy_labels, config, teacher_embeddings
        del step, steps_per_epoch, generator, gsi_step_diagnostics
        assert (memory_embeddings is None) == (memory_labels is None)
        memory_visible_by_step.append(memory_embeddings is not None)
        return embeddings[:, 0].mean() * torch_module.tensor(1.0, device=embeddings.device)

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    examples = [
        ImageExample(
            example_id=f"{label}-{index}",
            image=[float(label), float(index)],
            label=label,
        )
        for label in (0, 1)
        for index in range(4)
    ]
    monkeypatch.setattr(image_end_to_end, "_loss_for_objective", loss_for_objective)

    run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("supcon",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=8,
            train_steps=4,
            group_size=1,
            xbm_start_step=3,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    assert memory_visible_by_step == [False, False, False, True]


def test_adamw_optimizer_groups_disable_decay_for_bias_batch_norm_and_proxies() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _attach_metric_proxies, _optimizer_parameter_groups

    class TinyResNet(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Linear(2, 2)
            self.bn = torch.nn.BatchNorm1d(2)
            self.fc = torch.nn.Linear(2, 2)
            self.bias = torch.nn.Parameter(torch.zeros(2))

    config = ImageEndToEndConfig(
        optimizer="adamw",
        learning_rate=5e-4,
        backbone_learning_rate=1e-4,
        weight_decay=1e-4,
        proxy_count_per_class=1,
        proxy_learning_rate_multiplier=100.0,
    )
    model = TinyResNet()
    _attach_metric_proxies(model, train_labels=[0, 1], config=config, torch_module=torch)

    groups = _optimizer_parameter_groups(model, config)
    settings_by_id = {
        id(parameter): (
            float(group.get("lr", config.learning_rate)),
            float(group.get("weight_decay", config.weight_decay)),
        )
        for group in groups
        for parameter in group["params"]
    }

    assert settings_by_id[id(model.conv.weight)] == (1e-4, 1e-4)
    assert settings_by_id[id(model.conv.bias)] == (1e-4, 0.0)
    assert settings_by_id[id(model.bias)] == (1e-4, 0.0)
    assert settings_by_id[id(model.bn.weight)] == (1e-4, 0.0)
    assert settings_by_id[id(model.bn.bias)] == (1e-4, 0.0)
    assert settings_by_id[id(model.fc.weight)] == (5e-4, 1e-4)
    assert settings_by_id[id(model.fc.bias)] == (5e-4, 0.0)
    assert settings_by_id[id(model.metric_proxies)] == (5e-2, 0.0)


def test_checkpoint_selector_restores_best_validation_state() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _BestCheckpoint

    model = torch.nn.Linear(1, 1, bias=False)
    selector = _BestCheckpoint(metric_name="validation_map_at_r", mode="max")

    with torch.no_grad():
        model.weight.fill_(1.0)
    selector.update(score=0.4, step=10, model=model)

    with torch.no_grad():
        model.weight.fill_(2.0)
    selector.update(score=0.9, step=20, model=model)

    with torch.no_grad():
        model.weight.fill_(3.0)
    selector.update(score=0.6, step=30, model=model)
    selector.restore(model)

    assert model.weight.item() == pytest.approx(2.0)
    assert selector.best_step == 20
    assert selector.best_score == pytest.approx(0.9)


def test_interference_diagnostics_use_test_class_mean_axes() -> None:
    from sfora.image_end_to_end import _interference_diagnostics

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [4.0, 1.0],
            [4.0, -1.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    diagnostics = _interference_diagnostics(embeddings, labels, top_k=3)

    assert diagnostics is not None
    assert set(diagnostics) == {
        "rho_mean",
        "rho_p90",
        "rho_max",
        "fraction_above_floor_002",
        "fraction_above_floor_005",
    }
    assert diagnostics["rho_mean"] == pytest.approx(0.5)
    assert diagnostics["rho_p90"] == pytest.approx(0.9)
    assert diagnostics["rho_max"] == pytest.approx(1.0)
    assert diagnostics["fraction_above_floor_002"] == pytest.approx(0.5)
    assert diagnostics["fraction_above_floor_005"] == pytest.approx(0.5)


def test_end_to_end_run_serializes_interference_diagnostics(
    tmp_path: Path,
) -> None:
    torch: Any = pytest.importorskip("torch")

    from sfora.image_end_to_end import write_image_end_to_end_report

    class IdentityModel(torch.nn.Module):  # type: ignore[misc]
        def forward(self, images: object) -> object:
            return images

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    examples = [
        ImageExample(example_id="0-a", image=[1.0, 0.0], label=0),
        ImageExample(example_id="0-b", image=[-1.0, 0.0], label=0),
        ImageExample(example_id="1-a", image=[4.0, 1.0], label=1),
        ImageExample(example_id="1-b", image=[4.0, -1.0], label=1),
    ]

    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("frozen",),
            backbone_name="tiny",
            embedding_dimensions=2,
            eval_batch_size=4,
            retrieval_query_limit=4,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: IdentityModel(),
        transform_factory=transform_factory,
    )

    metrics = result.methods["frozen_end_to_end:tiny"]
    assert metrics.interference is not None
    assert set(metrics.interference) == {
        "rho_mean",
        "rho_p90",
        "rho_max",
        "fraction_above_floor_002",
        "fraction_above_floor_005",
    }
    assert metrics.train_interference is not None
    assert set(metrics.train_interference) == set(metrics.interference)

    output = tmp_path / "image_end_to_end.json"
    write_image_end_to_end_report(result, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    method = payload["methods"]["frozen_end_to_end:tiny"]
    assert method["interference"] == metrics.interference
    assert method["train_interference"] == metrics.train_interference


def test_end_to_end_report_keeps_supcon_baseline_separate_from_ours(tmp_path: Path) -> None:
    artifact = tmp_path / "image_end_to_end.json"
    artifact.write_text(
        json.dumps(
            {
                "name": "image-end-to-end-benchmark",
                "dataset_name": "cub",
                "protocol": "sota-resnet50-512",
                "config": {
                    "backbone_name": "resnet50",
                    "embedding_dimensions": 512,
                    "objectives": [
                        "supcon",
                        "group_supcon",
                        "group_supcon_xbm_radius",
                        "group_potential",
                        "group_potential_xbm",
                    ],
                },
                "train_examples": 10,
                "test_examples": 6,
                "methods": {
                    "supcon_end_to_end:resnet50": {
                        "model_name": "resnet50",
                        "objective": "supcon",
                        "display_name": "Supervised Contrastive",
                        "dimensions": 512,
                        "recall_at_1": 0.2,
                        "map_at_r": 0.1,
                    },
                    "group_supcon_end_to_end:resnet50": {
                        "model_name": "resnet50",
                        "objective": "group_supcon",
                        "display_name": "Group SupCon",
                        "dimensions": 512,
                        "recall_at_1": 0.3,
                        "map_at_r": 0.2,
                    },
                    "group_supcon_xbm_radius_end_to_end:resnet50": {
                        "model_name": "resnet50",
                        "objective": "group_supcon_xbm_radius",
                        "display_name": "Group SupCon + XBM + Radius",
                        "dimensions": 512,
                        "recall_at_1": 0.4,
                        "map_at_r": 0.3,
                    },
                    "group_potential_end_to_end:resnet50": {
                        "model_name": "resnet50",
                        "objective": "group_potential",
                        "display_name": "Group Potential",
                        "dimensions": 512,
                        "recall_at_1": 0.5,
                        "map_at_r": 0.35,
                    },
                    "group_potential_xbm_end_to_end:resnet50": {
                        "model_name": "resnet50",
                        "objective": "group_potential_xbm",
                        "display_name": "Group Potential + XBM",
                        "dimensions": 512,
                        "recall_at_1": 0.6,
                        "map_at_r": 0.4,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    payload = build_site_data(ReportConfig(artifact_paths=(artifact,)))

    rows = payload["endToEndRows"]
    assert [row["methodName"] for row in rows] == [
        "Supervised Contrastive (SupCon)",
        "Group SupCon",
        "Group SupCon + XBM + Radius",
        "Group Potential",
        "Group Potential + XBM",
    ]
    assert rows[0]["isOurs"] is False
    assert rows[1]["isOurs"] is True
    assert rows[2]["isOurs"] is True
    assert rows[3]["isOurs"] is True
    assert rows[4]["isOurs"] is True


def test_end_to_end_report_serializes_checkpoint_selection_metadata(tmp_path: Path) -> None:
    from sfora.image_benchmark import ImageRetrievalMetrics
    from sfora.image_end_to_end import (
        EndToEndMethodMetrics,
        ImageEndToEndResult,
        write_image_end_to_end_report,
    )

    result = ImageEndToEndResult(
        name="image-end-to-end-benchmark",
        dataset_name="cub",
        protocol="sota-resnet50-512",
        config=ImageEndToEndConfig(),
        train_examples=4,
        test_examples=4,
        methods={
            "triplet_end_to_end:tiny": EndToEndMethodMetrics(
                model_name="tiny",
                objective="triplet",
                display_name="Triplet",
                dimensions=2,
                retrieval=ImageRetrievalMetrics(
                    precision_at_1=0.5,
                    recall_at_1=0.5,
                    recall_at_2=0.75,
                    recall_at_4=1.0,
                    recall_at_8=1.0,
                    map_at_r=0.4,
                    mean_relevant_items=1.0,
                    evaluated_queries=4,
                    total_queries=4,
                ),
                precision_at_1=0.5,
                recall_at_1=0.5,
                recall_at_2=0.75,
                recall_at_4=1.0,
                recall_at_8=1.0,
                map_at_r=0.4,
                loss_history=[1.0, 0.8],
                selected_step=100,
                selection_metric="map_at_r",
                selection_score=0.42,
            )
        },
    )
    output = tmp_path / "end_to_end.json"

    write_image_end_to_end_report(result, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    method = payload["methods"]["triplet_end_to_end:tiny"]
    assert method["selected_step"] == 100
    assert method["selection_metric"] == "map_at_r"
    assert method["selection_score"] == pytest.approx(0.42)


def test_end_to_end_run_reports_partial_result_after_each_objective() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    examples = [
        ImageExample(example_id="0-a", image=[1.0, 0.0], label=0),
        ImageExample(example_id="0-b", image=[0.9, 0.1], label=0),
        ImageExample(example_id="1-a", image=[0.0, 1.0], label=1),
        ImageExample(example_id="1-b", image=[0.1, 0.9], label=1),
    ]
    snapshots: list[tuple[str, ...]] = []

    run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("frozen_pretrained", "frozen", "supcon", "group_supcon"),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=4,
            train_steps=1,
            group_size=1,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
        progress_callback=lambda result: snapshots.append(tuple(result.methods)),
    )

    assert snapshots == [
        ("frozen_pretrained_end_to_end:tiny",),
        ("frozen_pretrained_end_to_end:tiny", "frozen_end_to_end:tiny"),
        ("frozen_pretrained_end_to_end:tiny", "frozen_end_to_end:tiny", "supcon_end_to_end:tiny"),
        (
            "frozen_pretrained_end_to_end:tiny",
            "frozen_end_to_end:tiny",
            "supcon_end_to_end:tiny",
            "group_supcon_end_to_end:tiny",
        ),
    ]


def test_end_to_end_run_builds_teacher_model_for_weighted_objective() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    examples = [
        ImageExample(example_id="0-a", image=[1.0, 0.0], label=0),
        ImageExample(example_id="0-b", image=[0.9, 0.1], label=0),
        ImageExample(example_id="1-a", image=[0.0, 1.0], label=1),
        ImageExample(example_id="1-b", image=[0.1, 0.9], label=1),
    ]
    model_factory_calls = 0

    def model_factory(config: ImageEndToEndConfig) -> TinyModel:
        nonlocal model_factory_calls
        model_factory_calls += 1
        return TinyModel()

    run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("triplet",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=4,
            train_steps=1,
            group_size=1,
            teacher_similarity_weight=1.0,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=model_factory,
        transform_factory=transform_factory,
    )

    assert model_factory_calls == 2


def test_end_to_end_training_can_keep_batch_norm_layers_in_eval_mode() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyBatchNormModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.bn = torch.nn.BatchNorm1d(2)
            self.batch_norm_training_states: list[bool] = []

        def forward(self, images: object) -> object:
            self.batch_norm_training_states.append(bool(self.bn.training))
            return self.bn(self.linear(images))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    examples = [
        ImageExample(example_id="0-a", image=[1.0, 0.0], label=0),
        ImageExample(example_id="0-b", image=[0.9, 0.1], label=0),
        ImageExample(example_id="1-a", image=[0.0, 1.0], label=1),
        ImageExample(example_id="1-b", image=[0.1, 0.9], label=1),
    ]
    models: list[TinyBatchNormModel] = []

    def model_factory(config: ImageEndToEndConfig) -> TinyBatchNormModel:
        model = TinyBatchNormModel()
        models.append(model)
        return model

    run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("triplet",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=4,
            train_steps=1,
            group_size=1,
            freeze_batch_norm=True,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=model_factory,
        transform_factory=transform_factory,
    )

    assert models[0].batch_norm_training_states
    assert all(state is False for state in models[0].batch_norm_training_states)


def test_end_to_end_training_reports_selected_checkpoint_metadata() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    # Eight per class + a 0.5 validation fraction so the checkpoint train/validation
    # split has >=2 per class (retrieval needs it) — selection now refuses to fall
    # back to the test split when no validation split exists.
    examples = [
        ImageExample(example_id=f"0-{i}", image=[1.0 - 0.05 * i, 0.0 + 0.05 * i], label=0)
        for i in range(8)
    ] + [
        ImageExample(example_id=f"1-{i}", image=[0.0 + 0.05 * i, 1.0 - 0.05 * i], label=1)
        for i in range(8)
    ]

    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("triplet",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=4,
            train_steps=2,
            group_size=1,
            checkpoint_selection_interval=1,
            checkpoint_selection_validation_fraction=0.5,
            checkpoint_selection_query_limit=4,
            checkpoint_selection_metric="map_at_r",
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    metrics = result.methods["triplet_end_to_end:tiny"]
    assert metrics.selected_step in {1, 2}
    assert metrics.selection_metric == "map_at_r"
    assert metrics.selection_score is not None


def test_checkpoint_selection_uses_train_validation_split_not_test_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        def transform(image: object) -> object:
            return torch.as_tensor(image, dtype=torch.float32)

        return transform

    train_examples = [
        ImageExample(example_id=f"train-0-{index}", image=[1.0, float(index)], label=0)
        for index in range(4)
    ] + [
        ImageExample(example_id=f"train-1-{index}", image=[0.0, float(index)], label=1)
        for index in range(4)
    ]
    test_examples = [
        ImageExample(example_id=f"test-10-{index}", image=[1.0, float(index)], label=10)
        for index in range(4)
    ] + [
        ImageExample(example_id=f"test-11-{index}", image=[0.0, float(index)], label=11)
        for index in range(4)
    ]
    scored_label_sets: list[set[int]] = []

    def fake_checkpoint_score(
        model: object,
        loader: Iterable[tuple[Any, Any]],
        device: object,
        torch_module: object,
        *,
        config: ImageEndToEndConfig,
    ) -> float:
        del model, device, torch_module, config
        labels: set[int] = set()
        for _, batch_labels in loader:
            labels.update(int(label) for label in batch_labels.tolist())
        scored_label_sets.append(labels)
        return 1.0

    monkeypatch.setattr(
        "sfora.image_end_to_end._checkpoint_selection_score",
        fake_checkpoint_score,
    )

    run_image_end_to_end_benchmark(
        train_examples=train_examples,
        test_examples=test_examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("triplet",),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=8,
            train_steps=1,
            group_size=1,
            checkpoint_selection_interval=1,
            checkpoint_selection_validation_fraction=0.5,
            checkpoint_selection_metric="recall_at_1",
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    assert scored_label_sets == [{0, 1}]
