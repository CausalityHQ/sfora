from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import types
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from sfora.data import ImageDatasetName, ImageExample
from sfora.image_benchmark import ImageRetrievalMetrics, image_self_retrieval_score

if TYPE_CHECKING:
    import torch

EndToEndProtocol = Literal[
    "hpl-resnet50-512",
    "pfml-resnet50-512",
    "proxy-anchor-resnet50-512",
    "sota-resnet50-512",
]
EndToEndObjective = Literal[
    "frozen_pretrained",
    "frozen",
    "triplet",
    "triplet_pretrained",
    "batch_hard_triplet",
    "supcon",
    "group_supcon",
    "group_supcon_xbm_radius",
    "group_potential",
    "group_potential_xbm",
    "proxy_anchor",
    "proxy_anchor_group",
    "proxy_anchor_synthesis",
    "proxy_anchor_subcenter",
    "proxy_anchor_uniformity",
    "pfml",
    "symmetric_potential",
    "lennard_jones",
    "proxy_anchor_lj",
    "proxy_anchor_antico",
    "bio_physical_bond",
    "hist",
    "hist_proxy_anchor",
    "proxy_anchor_gsi",
    "proxy_anchor_bgsi",
    "pfml_gsi",
]


class TorchImageModel(Protocol):
    def train(self, mode: bool = True) -> Any: ...

    def eval(self) -> Any: ...

    def to(self, device: Any) -> Any: ...

    def parameters(self) -> Any: ...

    def named_parameters(self) -> Any: ...

    def __call__(self, images: Any) -> Any: ...


class ImageEndToEndConfig(BaseModel):
    """Configuration for end-to-end image metric-learning benchmarks."""

    dataset_name: ImageDatasetName = "cub"
    protocol: EndToEndProtocol = "hpl-resnet50-512"
    objectives: tuple[EndToEndObjective, ...] = ("group_supcon_xbm_radius",)
    backbone_name: str = "resnet50"
    embedding_dimensions: int = Field(default=512, ge=2)
    optimizer: Literal["adam", "adamw", "rmsprop"] = "adam"
    batch_size: int = Field(default=128, ge=2)
    eval_batch_size: int = Field(default=128, ge=1)
    train_steps: int = Field(default=2000, ge=1)
    train_epochs: int | None = Field(default=None, ge=1)
    learning_rate: float = Field(default=1e-6, gt=0.0)
    backbone_learning_rate: float | None = Field(default=None, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    warmup_epochs: int = Field(default=0, ge=0)
    lr_schedule: Literal["none", "step", "cosine"] = "none"
    lr_step_epochs: int = Field(default=5, ge=1)
    lr_gamma: float = Field(default=0.5, gt=0.0, le=1.0)
    samples_per_class: int = Field(default=0, ge=0)
    pretrained_weights: Literal["v1", "v2"] = "v2"
    head_pooling: Literal["avg", "avg_max"] = "avg"
    embedding_head_init: Literal["default", "kaiming_normal"] = "default"
    # Apply LayerNorm(embedding_dim, elementwise_affine=False) to the embedding, as in
    # the reference Proxy-Anchor / HIST ResNet50 head (`is_norm=1`). This centers and
    # standardizes each embedding before the downstream L2-for-cosine; omitting it is
    # the main architectural source of our ~2pt absolute offset from published numbers.
    embedding_layer_norm: bool = False
    xbm_start_step: int = Field(default=0, ge=0)
    group_size: int = Field(default=4, ge=1)
    point_weight: float = Field(default=1.0, ge=0.0)
    group_weight: float = Field(default=1.0, ge=0.0)
    xbm_weight: float = Field(default=0.25, ge=0.0)
    xbm_memory_size: int = Field(default=4096, ge=0)
    radius_weight: float = Field(default=0.01, ge=0.0)
    radius_target: float = Field(default=0.0, ge=0.0)
    proxy_weight: float = Field(default=0.0, ge=0.0)
    proxy_count_per_class: int = Field(default=0, ge=0)
    proxy_learning_rate_multiplier: float = Field(default=100.0, gt=0.0)
    proxy_anchor_alpha: float = Field(default=32.0, gt=0.0)
    proxy_anchor_delta: float = Field(default=0.1, ge=0.0)
    # SoftTriple-style intra-class softmax temperature for sub-center assignment.
    subcenter_gamma: float = Field(default=0.1, gt=0.0)
    # Thermodynamic Gaussian-potential uniformity (proxy_anchor_uniformity).
    uniformity_weight: float = Field(default=0.0, ge=0.0)
    uniformity_t: float = Field(default=2.0, gt=0.0)
    # Optional path to save final test embeddings + labels (.npz) for ensembling.
    save_test_embeddings: str | None = None
    # Optional path to save the TRAIN-split embeddings from the same best epoch, so a
    # fold/projection can be fit on train (disjoint zero-shot classes) and evaluated on
    # test — the honest, non-transductive way to compress the pack.
    save_train_embeddings: str | None = None
    # EMA-teacher relational self-distillation (any base objective). Weight 0 = off.
    ema_distill_weight: float = Field(default=0.0, ge=0.0)
    ema_momentum: float = Field(default=0.999, ge=0.0, le=1.0)
    ema_distill_tau: float = Field(default=0.1, gt=0.0)
    # Multi-crop EMA assignment distillation. Weight 0 = off.
    mead_weight: float = Field(default=0.0, ge=0.0)
    mead_local_crops: int = Field(default=4, ge=0)
    mead_local_size: int = Field(default=96, ge=1)
    mead_tau_teacher: float = Field(default=0.05, gt=0.0)
    mead_tau_student: float = Field(default=0.1, gt=0.0)
    mead_center_momentum: float = Field(default=0.9, ge=0.0, le=1.0)
    mead_proto_momentum: float = Field(default=0.9, ge=0.0, le=1.0)
    mead_global_scale_min: float = Field(default=0.4, gt=0.0, le=1.0)
    mead_local_scale_max: float = Field(default=0.4, gt=0.0, le=1.0)
    proxy_anchor_group_tau_assign: float = Field(default=0.1, gt=0.0)
    synthesis_ratio: float = Field(default=1.0, ge=0.0)
    synthesis_beta_alpha: float = Field(default=0.4, gt=0.0)
    synthesis_group_mix: bool = False
    synthesis_pair_selection: Literal["random", "confusable"] = "random"
    synthesis_pair_temperature: float = Field(default=0.1, gt=0.0)
    synthesis_compactness_weight: float = Field(default=0.0, ge=0.0)
    synthesis_compactness_target: float = Field(default=0.0, ge=0.0)
    lj_sigma: float = Field(default=0.3, gt=0.0)
    lj_sigma_neg: float | None = Field(default=None, gt=0.0)
    lj_power: float = Field(default=2.0, gt=0.0)
    lj_repulsion_weight: float = Field(default=1.0, ge=0.0)
    lj_intra_weight: float = Field(default=0.1, ge=0.0)
    antico_weight: float = Field(default=0.05, ge=0.0)
    antico_eps: float = Field(default=0.5, gt=0.0)
    antico_target: Literal["feature", "proxy", "both"] = "feature"
    bond_niche_weight: float = Field(default=0.02, ge=0.0)
    hard_class_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    hist_tau: float = Field(default=32.0, gt=0.0)
    hist_alpha: float = Field(default=0.9, gt=0.0)
    hist_hidden: int = Field(default=512, ge=2)
    hist_lambda_s: float = Field(default=1.0, ge=0.0)
    # Lower clamp on the class log-variances. The faithful HIST port uses relu6,
    # i.e. a floor of 0.0 (variance >= 1); lowering it lets a class cluster tighter
    # than unit variance, an ablation lever for fine-grained retrieval. Upper clamp
    # stays at 6.0 (relu6's ceiling) for stability.
    hist_var_floor: float = Field(default=0.0, le=6.0)
    # Weight of the Proxy Anchor term in the fused `hist_proxy_anchor` objective
    # (L = L_HIST + proxy_fusion_weight * L_ProxyAnchor). One model, both losses.
    proxy_fusion_weight: float = Field(default=1.0, ge=0.0)
    hist_lr_ds: float = Field(default=1.0e-1, gt=0.0)
    hist_lr_hgnn_factor: float = Field(default=10.0, gt=0.0)
    gsi_weight: float = Field(default=0.3, ge=0.0)
    gsi_floor: float = Field(default=0.02, ge=0.0)
    gsi_top_k: int = Field(default=3, ge=1)
    gsi_min_group_size: int = Field(default=4, ge=2)
    gsi_variance_floor: float = Field(default=1e-4, gt=0.0)
    gsi_start_epoch: int = Field(default=5, ge=0)
    gsi_axis_mode: Literal["proxy", "random", "global"] = "proxy"
    bgsi_weight: float = Field(default=0.3, ge=0.0)
    bgsi_floor: float = Field(default=0.0, ge=0.0)
    bgsi_top_k: int = Field(default=3, ge=1)
    bgsi_temperature: float = Field(default=0.1, gt=0.0)
    bgsi_start_epoch: int = Field(default=5, ge=0)
    bgsi_min_group_size: int = Field(default=4, ge=2)
    bgsi_variance_floor: float = Field(default=1e-4, gt=0.0)
    bgsi_axis_mode: Literal[
        "batch_boundary",
        "ema_boundary",
        "random",
        "permuted",
        "global",
    ] = "batch_boundary"
    bgsi_ema_momentum: float = Field(default=0.95, ge=0.0, lt=1.0)
    bgsi_min_axis_observations: int = Field(default=5, ge=1)
    bgsi_use_axis_agreement_gate: bool = True
    bgsi_axis_agreement: float = Field(default=0.5, ge=-1.0, le=1.0)
    potential_weight: float = Field(default=0.0, ge=0.0)
    potential_delta: float = Field(default=0.2, gt=0.0)
    potential_alpha: float = Field(default=4.0, ge=0.0)
    teacher_similarity_weight: float = Field(default=0.0, ge=0.0)
    label_noise_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    triplet_margin: float = Field(default=0.2, ge=0.0)
    temperature: float = Field(default=0.07, gt=0.0)
    input_size: int = Field(default=224, ge=32)
    train_augmentation: Literal["standard", "center_crop", "full_res_crop"] = "standard"
    freeze_batch_norm: bool = True
    checkpoint_selection_interval: int = Field(default=0, ge=0)
    checkpoint_selection_query_limit: int | None = Field(default=1024, ge=1)
    checkpoint_selection_metric: Literal["map_at_r", "recall_at_1"] = "map_at_r"
    checkpoint_selection_validation_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    num_workers: int = Field(default=4, ge=0)
    # Evaluate the held-out TEST classes every N epochs and record the best R@1 over
    # training (the standard DML benchmark protocol, which HIST/Proxy Anchor papers
    # report). Diagnostic only: it does NOT change model selection -- the primary
    # `recall_at_1` remains the final-epoch model. Default 0 = off (legacy behavior).
    eval_test_interval_epochs: int = Field(default=0, ge=0)
    retrieval_query_limit: int | None = Field(default=None, ge=1)
    limit_per_class: int | None = Field(default=None, ge=1)
    max_classes: int | None = Field(default=None, ge=2)
    seed: int = 0
    progress_every: int = Field(default=100, ge=0)


@dataclass(frozen=True)
class EndToEndMethodMetrics:
    model_name: str
    objective: str
    display_name: str
    dimensions: int
    retrieval: ImageRetrievalMetrics
    precision_at_1: float
    recall_at_1: float
    recall_at_2: float
    recall_at_4: float
    recall_at_8: float
    map_at_r: float
    loss_history: list[float]
    interference: dict[str, float] | None = None
    train_interference: dict[str, float] | None = None
    gsi_diagnostics: dict[str, float] | None = None
    selected_step: int | None = None
    selection_metric: str | None = None
    selection_score: float | None = None
    best_test_recall_at_1: float | None = None
    best_test_epoch: int | None = None
    test_recall_history: list[float] | None = None


@dataclass(frozen=True)
class ImageEndToEndResult:
    name: str
    dataset_name: ImageDatasetName
    protocol: str
    config: ImageEndToEndConfig
    train_examples: int
    test_examples: int
    methods: dict[str, EndToEndMethodMetrics]


def config_for_protocol(
    protocol: EndToEndProtocol,
    *,
    dataset_name: ImageDatasetName,
    train_steps: int | None = None,
) -> ImageEndToEndConfig:
    """Return defaults matching the cited paper protocol family as closely as local code can."""
    if protocol == "proxy-anchor-resnet50-512":
        return ImageEndToEndConfig(
            dataset_name=dataset_name,
            protocol=protocol,
            objectives=("frozen_pretrained", "proxy_anchor"),
            backbone_name="resnet50",
            embedding_dimensions=512,
            optimizer="adamw",
            batch_size=120,
            learning_rate=1e-4,
            backbone_learning_rate=1e-4,
            weight_decay=1e-4,
            train_steps=train_steps or 2000,
            train_epochs=None if train_steps is not None else 60,
            warmup_epochs=5,
            lr_schedule="step",
            lr_step_epochs=5 if dataset_name == "cub" else 10,
            lr_gamma=0.5,
            samples_per_class=4,
            train_augmentation="full_res_crop",
            pretrained_weights="v1",
            head_pooling="avg_max",
            embedding_head_init="kaiming_normal",
            proxy_count_per_class=1,
            proxy_anchor_alpha=32.0,
            proxy_anchor_delta=0.1,
            checkpoint_selection_interval=0,
        )
    if protocol == "pfml-resnet50-512":
        return ImageEndToEndConfig(
            dataset_name=dataset_name,
            protocol=protocol,
            objectives=("frozen_pretrained", "pfml"),
            backbone_name="resnet50",
            embedding_dimensions=512,
            optimizer="adam",
            batch_size=120,
            learning_rate=5e-4,
            backbone_learning_rate=5e-4,
            weight_decay=1e-4,
            train_steps=train_steps or 2000,
            train_epochs=None if train_steps is not None else 100,
            warmup_epochs=5,
            lr_schedule="cosine",
            lr_step_epochs=5 if dataset_name == "cub" else 10,
            lr_gamma=0.5,
            samples_per_class=4,
            train_augmentation="full_res_crop",
            pretrained_weights="v1",
            head_pooling="avg_max",
            embedding_head_init="kaiming_normal",
            proxy_count_per_class=2 if dataset_name == "sop" else 15,
            potential_delta=0.2,
            potential_alpha=4.0,
            checkpoint_selection_interval=0,
        )
    if protocol == "sota-resnet50-512":
        return ImageEndToEndConfig(
            dataset_name=dataset_name,
            protocol=protocol,
            backbone_name="resnet50",
            embedding_dimensions=512,
            optimizer="adam",
            batch_size=120,
            learning_rate=5e-4,
            backbone_learning_rate=1e-5,
            weight_decay=1e-4,
            train_steps=train_steps or 2000,
            train_epochs=None if train_steps is not None else 80,
        )
    return ImageEndToEndConfig(
        dataset_name=dataset_name,
        protocol=protocol,
        backbone_name="resnet50",
        embedding_dimensions=512,
        optimizer="rmsprop",
        batch_size=128,
        learning_rate=1e-6,
        backbone_learning_rate=1e-6,
        weight_decay=1e-4,
        train_steps=train_steps or 2000,
    )


_GROUP_CENTROID_OBJECTIVES = {
    "group_supcon",
    "group_supcon_xbm_radius",
    "group_potential",
    "group_potential_xbm",
}


def _validate_group_centroid_sampling(config: ImageEndToEndConfig) -> None:
    """Reject sampler settings that silently zero out group-centroid loss terms.

    With samples_per_class < 2 * group_size each class yields at most one group
    centroid per batch, so centroid-level SupCon has no positive pairs and the
    "group" term contributes nothing while the objective name still claims it.
    """
    if config.samples_per_class <= 0:
        return
    starved = [
        objective for objective in config.objectives if objective in _GROUP_CENTROID_OBJECTIVES
    ]
    if starved and config.samples_per_class < 2 * config.group_size:
        raise ValueError(
            "objectives "
            + ", ".join(sorted(starved))
            + " need samples_per_class >= 2 * group_size to form at least two "
            f"group centroids per class; got samples_per_class={config.samples_per_class} "
            f"and group_size={config.group_size}"
        )


def run_image_end_to_end_benchmark(
    *,
    train_examples: list[ImageExample],
    test_examples: list[ImageExample],
    config: ImageEndToEndConfig,
    model_factory: Callable[[ImageEndToEndConfig], TorchImageModel] | None = None,
    transform_factory: Callable[[ImageEndToEndConfig, bool], Callable[[object], Any]] | None = None,
    progress_callback: Callable[[ImageEndToEndResult], None] | None = None,
) -> ImageEndToEndResult:
    """Train Group SupCon + XBM + Radius with a trainable image model and evaluate retrieval."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError(
            "Install the research extra to run end-to-end image benchmarks: "
            "uv sync --group dev --extra research"
        ) from error

    _validate_group_centroid_sampling(config)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = (transform_factory or _default_transform_factory)(config, True)
    test_transform = (transform_factory or _default_transform_factory)(config, False)
    optimization_examples, checkpoint_examples = _checkpoint_train_validation_split(
        train_examples,
        fraction=(
            config.checkpoint_selection_validation_fraction
            if config.checkpoint_selection_interval > 0
            else 0.0
        ),
        seed=config.seed,
    )
    optimization_examples = _apply_training_label_noise(
        optimization_examples,
        fraction=config.label_noise_fraction,
        seed=config.seed,
    )
    train_steps, steps_per_epoch, total_epochs = _resolve_training_schedule(
        config,
        optimization_example_count=len(optimization_examples),
    )
    train_dataset = _TorchImageDataset(optimization_examples, train_transform)
    test_dataset = _TorchImageDataset(test_examples, test_transform)
    checkpoint_dataset = _TorchImageDataset(checkpoint_examples, test_transform)
    train_labels = [example.label for example in optimization_examples]
    # The test loader is shuffle=False, so encoded rows follow test_examples order.
    # Persisting the example ids lets the ensemble/thumbnail tooling prove row
    # alignment across independently-seeded runs (labels alone can't — a within-class
    # reordering would pass a label check while mixing different images).
    test_example_ids = np.asarray([example.example_id for example in test_examples])
    train_example_ids = np.asarray([example.example_id for example in optimization_examples])
    class_similarity: dict[int, list[int]] | None = None
    if config.hard_class_fraction > 0.0:
        class_similarity = _frozen_class_similarity(
            optimization_examples,
            config=config,
            model_factory=model_factory,
            transform=test_transform,
            device=device,
            torch_module=torch,
        )
    train_loader_kwargs: dict[str, Any] = {}
    if config.mead_weight > 0.0:
        train_loader_kwargs["collate_fn"] = _mead_multicrop_collate
    if config.samples_per_class > 0 or config.hard_class_fraction > 0.0:
        batch_sampler = _balanced_batch_indices(
            train_labels,
            batch_size=config.batch_size,
            group_size=config.group_size,
            samples_per_class=config.samples_per_class,
            steps=train_steps,
            seed=config.seed,
            class_similarity=class_similarity,
            hard_fraction=config.hard_class_fraction,
        )
        train_loader: Any = DataLoader(
            cast(Any, train_dataset),
            batch_sampler=batch_sampler,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            **train_loader_kwargs,
        )
    else:
        train_generator = torch.Generator()
        train_generator.manual_seed(config.seed)
        train_loader = DataLoader(
            cast(Any, train_dataset),
            batch_size=config.batch_size,
            shuffle=True,
            generator=train_generator,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            **train_loader_kwargs,
        )
    test_loader: Any = DataLoader(
        cast(Any, test_dataset),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    checkpoint_loader: Any = (
        DataLoader(
            cast(Any, checkpoint_dataset),
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if checkpoint_examples
        else test_loader
    )
    # Never select checkpoints on the test split. When selection is enabled the
    # validation split must be non-empty; the `else test_loader` above is only a
    # placeholder for the disabled path (where checkpoint_loader is never used).
    if config.checkpoint_selection_interval > 0 and not checkpoint_examples:
        raise ValueError(
            "checkpoint selection is enabled but no validation split could be built "
            "(too few examples per class). Increase "
            "--checkpoint-selection-validation-fraction or disable "
            "--checkpoint-selection-interval; refusing to select on the test set."
        )
    train_eval_dataset = _TorchImageDataset(optimization_examples, test_transform)
    train_eval_loader: Any = DataLoader(
        cast(Any, train_eval_dataset),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # `save_test_embeddings` is a single path; with several objectives each would
    # overwrite the previous one's embeddings. Require a single objective so the
    # saved artifact is unambiguous (ensemble runs already use one objective).
    if (config.save_test_embeddings or config.save_train_embeddings) and len(config.objectives) > 1:
        raise ValueError(
            "save_test_embeddings / save_train_embeddings expect a single objective, but "
            f"{len(config.objectives)} were given ({', '.join(config.objectives)}); "
            "run one objective per invocation or drop the --save-*-embeddings flags."
        )
    methods: dict[str, EndToEndMethodMetrics] = {}
    for objective in config.objectives:
        # Re-seed before each objective so its model init and batch order do not
        # depend on which objectives ran before it (order-independent comparisons).
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        if _uses_pretrained_feature_model(objective, config.backbone_name, model_factory):
            model = _torchvision_model_factory(config, use_embedding_head=False).to(device)
        else:
            model = (model_factory or _torchvision_model_factory)(config)
            if _uses_metric_proxies(objective, config):
                _attach_metric_proxies(
                    model,
                    train_labels=train_labels,
                    config=config,
                    torch_module=torch,
                )
            if objective in {"hist", "hist_proxy_anchor"}:
                _attach_hist_module(
                    model,
                    train_labels=train_labels,
                    config=config,
                    torch_module=torch,
                )
            model = model.to(device)
        history: list[float] = []
        gsi_step_diagnostics: list[dict[str, float]] = []
        best_test_recall_at_1: float | None = None
        best_test_epoch: int | None = None
        test_recall_history: list[float] = []
        selected_step: int | None = None
        selection_metric: str | None = None
        selection_score: float | None = None
        bgsi_state: BGSIClassMeanState | None = None
        if objective == "proxy_anchor_bgsi":
            bgsi_state = BGSIClassMeanState(
                labels=train_labels,
                embedding_dimensions=config.embedding_dimensions,
                momentum=config.bgsi_ema_momentum,
                device=device,
                dtype=torch.float32,
                torch_module=torch,
            )
        if objective not in {"frozen", "frozen_pretrained"}:
            teacher_model = _teacher_model_for_config(
                config,
                model_factory=model_factory,
                device=device,
            )
            optimizer_cls: Any
            if config.optimizer == "adam":
                optimizer_cls = torch.optim.Adam
            elif config.optimizer == "adamw":
                optimizer_cls = torch.optim.AdamW
            else:
                optimizer_cls = torch.optim.RMSprop
            optimizer = optimizer_cls(
                _optimizer_parameter_groups(model, config),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            scheduler: Any | None = None
            if config.lr_schedule == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=config.lr_step_epochs,
                    gamma=config.lr_gamma,
                )
            elif config.lr_schedule == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_epochs,
                )
            backbone_warmup_parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if not name.startswith("fc.")
                and not name.startswith("hist_module.")
                and name != "metric_proxies"
            ]
            warmup_steps = config.warmup_epochs * steps_per_epoch
            if config.warmup_epochs > 0:
                for parameter in backbone_warmup_parameters:
                    parameter.requires_grad_(False)
            loss_generator = torch.Generator(device=device)
            loss_generator.manual_seed(config.seed)
            memory = _XbmMemory(config.xbm_memory_size)
            mead_label_to_index: dict[int, int] | None = None
            mead_prototypes: Any | None = None
            mead_center: Any | None = None
            if config.mead_weight > 0.0:
                mead_train_labels = sorted({int(label) for label in train_labels})
                mead_label_to_index = {
                    label: index for index, label in enumerate(mead_train_labels)
                }
            ema_teacher: Any | None = None
            if config.ema_distill_weight > 0.0 or config.mead_weight > 0.0:
                import copy as _copy

                ema_teacher = _copy.deepcopy(model)
                for parameter in ema_teacher.parameters():
                    parameter.requires_grad_(False)
                ema_teacher.eval()
            checkpoint = (
                _BestCheckpoint(metric_name=config.checkpoint_selection_metric, mode="max")
                if config.checkpoint_selection_interval > 0
                else None
            )
            model.train()
            if config.freeze_batch_norm:
                _freeze_batch_norm_layers(model)
            train_batches = _iter_training_batches(train_loader)
            for step, (images, labels) in zip(
                range(1, train_steps + 1),
                train_batches,
                strict=False,
            ):
                if config.warmup_epochs > 0 and step == warmup_steps + 1:
                    for parameter in backbone_warmup_parameters:
                        parameter.requires_grad_(True)
                labels = labels.to(device, non_blocking=True)
                if config.mead_weight > 0.0:
                    if not isinstance(images, tuple) or len(images) != 2:
                        raise ValueError(
                            "MEAD training expects batches shaped as ((globals, locals), labels)"
                        )
                    global_crops, local_crops = images
                    global_crops = global_crops.to(device, non_blocking=True)
                    local_crops = local_crops.to(device, non_blocking=True)
                else:
                    images = images.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss_kwargs: dict[str, Any] = {}
                if bgsi_state is not None:
                    loss_kwargs["bgsi_state"] = bgsi_state
                if objective in {"hist", "hist_proxy_anchor"}:
                    loss_kwargs["hist_module"] = getattr(model, "hist_module", None)
                    loss_kwargs["hist_label_to_index"] = getattr(model, "hist_label_to_index", None)

                if config.mead_weight > 0.0:
                    if global_crops.ndim != 5 or int(global_crops.shape[1]) != 2:
                        raise ValueError(
                            "MEAD training expects global crops shaped "
                            f"(batch, 2, 3, 224, 224); got {tuple(global_crops.shape)}"
                        )
                    if (
                        local_crops.ndim != 5
                        or int(local_crops.shape[1]) != config.mead_local_crops
                    ):
                        raise ValueError(
                            "MEAD training expects local crops shaped "
                            f"(batch, {config.mead_local_crops}, 3, "
                            f"{config.mead_local_size}, {config.mead_local_size}); "
                            f"got {tuple(local_crops.shape)}"
                        )
                    batch_size = int(labels.shape[0])
                    global_image_shape = tuple(int(size) for size in global_crops.shape[2:])
                    global_images = global_crops.transpose(0, 1).reshape(
                        2 * batch_size,
                        *global_image_shape,
                    )
                    student_global_embeddings = _normalize(model(global_images), torch)
                    student_views = [
                        student_global_embeddings[:batch_size],
                        student_global_embeddings[batch_size : 2 * batch_size],
                    ]
                    local_count = int(local_crops.shape[1])
                    if local_count > 0:
                        local_image_shape = tuple(int(size) for size in local_crops.shape[2:])
                        local_images = local_crops.transpose(0, 1).reshape(
                            local_count * batch_size,
                            *local_image_shape,
                        )
                        student_local_embeddings = _normalize(model(local_images), torch)
                        for local_index in range(local_count):
                            local_start = local_index * batch_size
                            student_views.append(
                                student_local_embeddings[local_start : local_start + batch_size]
                            )
                    if bgsi_state is not None:
                        bgsi_state.update(student_views[0], labels)

                    teacher_global_embeddings = None
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_global_embeddings = _normalize(
                                teacher_model(global_images),
                                torch,
                            ).detach()
                    teacher_view_embeddings: list[Any | None] = [None, None]
                    if teacher_global_embeddings is not None:
                        teacher_view_embeddings = [
                            teacher_global_embeddings[:batch_size],
                            teacher_global_embeddings[batch_size : 2 * batch_size],
                        ]
                    first_global_loss = _loss_for_objective(
                        objective,
                        student_views[0],
                        labels,
                        step=step,
                        steps_per_epoch=steps_per_epoch,
                        memory_embeddings=memory.embeddings(device),
                        memory_labels=memory.labels(device),
                        proxy_embeddings=_metric_proxy_embeddings(model),
                        proxy_labels=_metric_proxy_labels(model),
                        teacher_embeddings=teacher_view_embeddings[0],
                        config=config,
                        torch_module=torch,
                        generator=loss_generator,
                        gsi_step_diagnostics=gsi_step_diagnostics,
                        **loss_kwargs,
                    )
                    second_global_loss = _loss_for_objective(
                        objective,
                        student_views[1],
                        labels,
                        step=step,
                        steps_per_epoch=steps_per_epoch,
                        memory_embeddings=memory.embeddings(device),
                        memory_labels=memory.labels(device),
                        proxy_embeddings=_metric_proxy_embeddings(model),
                        proxy_labels=_metric_proxy_labels(model),
                        teacher_embeddings=teacher_view_embeddings[1],
                        config=config,
                        torch_module=torch,
                        generator=loss_generator,
                        gsi_step_diagnostics=gsi_step_diagnostics,
                        **loss_kwargs,
                    )
                    loss = 0.5 * (first_global_loss + second_global_loss)

                    if ema_teacher is None or mead_label_to_index is None:
                        raise RuntimeError("MEAD requires an EMA teacher and train-class map")
                    with torch.no_grad():
                        ema_global_embeddings = _normalize(
                            ema_teacher(global_images),
                            torch,
                        ).detach()
                        if mead_prototypes is None:
                            mead_prototypes = ema_global_embeddings.new_zeros(
                                (
                                    len(mead_label_to_index),
                                    int(ema_global_embeddings.shape[1]),
                                )
                            )
                            mead_center = ema_global_embeddings.new_zeros(len(mead_label_to_index))
                        assert mead_center is not None
                        batch_class_indices = torch.tensor(
                            [mead_label_to_index[int(label)] for label in labels.tolist()],
                            dtype=torch.long,
                            device=device,
                        )
                        global_class_indices = batch_class_indices.repeat(2)
                        for raw_class_index in torch.unique(global_class_indices).tolist():
                            class_index = int(raw_class_index)
                            class_mask = global_class_indices == class_index
                            class_mean = ema_global_embeddings[class_mask].mean(dim=0)
                            mead_prototypes[class_index].mul_(config.mead_proto_momentum).add_(
                                class_mean, alpha=1.0 - config.mead_proto_momentum
                            )
                        normalized_prototypes = _normalize(mead_prototypes, torch).detach()
                        center_batch = (ema_global_embeddings @ normalized_prototypes.T).mean(dim=0)
                        mead_center.mul_(config.mead_center_momentum).add_(
                            center_batch,
                            alpha=1.0 - config.mead_center_momentum,
                        )
                    if step > warmup_steps:
                        loss = loss + config.mead_weight * _mead_assignment_distillation_loss(
                            student_views,
                            ema_global_embeddings,
                            normalized_prototypes,
                            mead_center,
                            tau_teacher=config.mead_tau_teacher,
                            tau_student=config.mead_tau_student,
                            torch_module=torch,
                        )
                    if config.ema_distill_weight > 0.0:
                        loss = loss + config.ema_distill_weight * _relational_distillation_loss(
                            student_global_embeddings,
                            ema_global_embeddings,
                            tau=config.ema_distill_tau,
                            torch_module=torch,
                        )
                    memory_embeddings_to_enqueue = student_views[0]
                else:
                    embeddings = _normalize(model(images), torch)
                    if bgsi_state is not None:
                        bgsi_state.update(embeddings, labels)
                    teacher_embeddings = None
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_embeddings = _normalize(teacher_model(images), torch).detach()
                    loss = _loss_for_objective(
                        objective,
                        embeddings,
                        labels,
                        step=step,
                        steps_per_epoch=steps_per_epoch,
                        memory_embeddings=memory.embeddings(device),
                        memory_labels=memory.labels(device),
                        proxy_embeddings=_metric_proxy_embeddings(model),
                        proxy_labels=_metric_proxy_labels(model),
                        teacher_embeddings=teacher_embeddings,
                        config=config,
                        torch_module=torch,
                        generator=loss_generator,
                        gsi_step_diagnostics=gsi_step_diagnostics,
                        **loss_kwargs,
                    )
                    if ema_teacher is not None:
                        with torch.no_grad():
                            ema_embeddings = _normalize(ema_teacher(images), torch).detach()
                        loss = loss + config.ema_distill_weight * _relational_distillation_loss(
                            embeddings,
                            ema_embeddings,
                            tau=config.ema_distill_tau,
                            torch_module=torch,
                        )
                    memory_embeddings_to_enqueue = embeddings
                loss.backward()
                optimizer.step()
                if ema_teacher is not None:
                    _update_ema_teacher(ema_teacher, model, momentum=config.ema_momentum)
                if step >= config.xbm_start_step:
                    memory.enqueue(memory_embeddings_to_enqueue.detach(), labels.detach())
                history.append(float(loss.detach().cpu()))
                if scheduler is not None and step % steps_per_epoch == 0:
                    scheduler.step()
                if (
                    config.eval_test_interval_epochs > 0
                    and step % steps_per_epoch == 0
                    and (step // steps_per_epoch) % config.eval_test_interval_epochs == 0
                ):
                    epoch_index = step // steps_per_epoch
                    epoch_embeddings, epoch_labels = _encode_model(
                        model, test_loader, device, torch
                    )
                    epoch_recall = image_self_retrieval_score(
                        epoch_embeddings,
                        epoch_labels,
                        query_limit=config.retrieval_query_limit,
                        random_state=config.seed,
                    ).recall_at_1
                    test_recall_history.append(float(epoch_recall))
                    if best_test_recall_at_1 is None or epoch_recall > best_test_recall_at_1:
                        best_test_recall_at_1 = float(epoch_recall)
                        best_test_epoch = int(epoch_index)
                        if config.save_test_embeddings:
                            # Persist the best-over-training test embeddings for
                            # ensembling — the standard DML protocol (same as the
                            # Proxy Anchor / HIST / PFML papers), so this selection
                            # peeks at the test split by design. Reported as such.
                            _atomic_savez(
                                Path(config.save_test_embeddings),
                                embeddings=np.asarray(epoch_embeddings, dtype=np.float32),
                                labels=np.asarray(epoch_labels, dtype=np.int64),
                                example_ids=test_example_ids,
                            )
                        if config.save_train_embeddings:
                            # Same (best) epoch's TRAIN-split embeddings, so a fold can
                            # be fit on train and evaluated on test without touching the
                            # test split. Train classes are disjoint from test (zero-shot).
                            train_embeddings, train_label_array = _encode_model(
                                model, train_eval_loader, device, torch
                            )
                            _atomic_savez(
                                Path(config.save_train_embeddings),
                                embeddings=np.asarray(train_embeddings, dtype=np.float32),
                                labels=np.asarray(train_label_array, dtype=np.int64),
                                example_ids=train_example_ids,
                            )
                    print(
                        f"{config.dataset_name} {objective} epoch {epoch_index} "
                        f"test R@1={epoch_recall:.4f} (best {best_test_recall_at_1:.4f} "
                        f"@ epoch {best_test_epoch})",
                        flush=True,
                    )
                    model.train()
                    if config.freeze_batch_norm:
                        _freeze_batch_norm_layers(model)
                if config.progress_every and step % config.progress_every == 0:
                    print(
                        f"{config.dataset_name} {objective} step "
                        f"{step}/{train_steps} loss={history[-1]:.4f}",
                        flush=True,
                    )
                if checkpoint is not None and (
                    step % config.checkpoint_selection_interval == 0 or step == train_steps
                ):
                    score = _checkpoint_selection_score(
                        model,
                        checkpoint_loader,
                        device,
                        torch,
                        config=config,
                    )
                    checkpoint.update(score=score, step=step, model=model)
                    model.train()
                    if config.freeze_batch_norm:
                        _freeze_batch_norm_layers(model)
            if checkpoint is not None and checkpoint.restore(model):
                selected_step = checkpoint.best_step
                selection_metric = checkpoint.metric_name
                selection_score = checkpoint.best_score
            del teacher_model

        test_embeddings, test_label_array = _encode_model(model, test_loader, device, torch)
        # When no per-epoch eval ran (the periodic best block never fired), save the
        # final-epoch embeddings as a fallback — for BOTH splits from the same model.
        if config.save_test_embeddings and best_test_recall_at_1 is None:
            _atomic_savez(
                Path(config.save_test_embeddings),
                embeddings=np.asarray(test_embeddings, dtype=np.float32),
                labels=np.asarray(test_label_array, dtype=np.int64),
                example_ids=test_example_ids,
            )
        if config.save_train_embeddings and best_test_recall_at_1 is None:
            train_embeddings, train_label_array = _encode_model(
                model, train_eval_loader, device, torch
            )
            _atomic_savez(
                Path(config.save_train_embeddings),
                embeddings=np.asarray(train_embeddings, dtype=np.float32),
                labels=np.asarray(train_label_array, dtype=np.int64),
                example_ids=train_example_ids,
            )
        retrieval = image_self_retrieval_score(
            test_embeddings,
            test_label_array,
            query_limit=config.retrieval_query_limit,
            random_state=config.seed,
        )
        interference = _interference_diagnostics(test_embeddings, test_label_array)
        train_eval_embeddings, train_eval_labels = _encode_model(
            model, train_eval_loader, device, torch
        )
        train_interference = _interference_diagnostics(train_eval_embeddings, train_eval_labels)
        proxy_axis_diagnostics = None
        boundary_axis_diagnostics = None
        if objective in {"proxy_anchor", "proxy_anchor_bgsi"}:
            boundary_axis_diagnostics = _boundary_axis_interference_diagnostics(
                train_eval_embeddings,
                train_eval_labels,
                top_k=config.bgsi_top_k,
                floor=config.bgsi_floor,
                temperature=config.bgsi_temperature,
            )
        proxy_embeddings = _metric_proxy_embeddings(model)
        proxy_labels = _metric_proxy_labels(model)
        if (
            objective != "proxy_anchor_bgsi"
            and proxy_embeddings is not None
            and proxy_labels is not None
        ):
            proxy_axis_diagnostics = _proxy_axis_interference_diagnostics(
                train_eval_embeddings,
                train_eval_labels,
                proxy_embeddings=np.asarray(
                    proxy_embeddings.detach().cpu().numpy(),
                    dtype=np.float64,
                ),
                proxy_labels=np.asarray(
                    proxy_labels.detach().cpu().numpy(),
                    dtype=np.int64,
                ),
                top_k=config.gsi_top_k,
                floor=config.gsi_floor,
            )
        gsi_diagnostics = _summarize_gsi_training_diagnostics(
            gsi_step_diagnostics,
            proxy_axis_diagnostics=proxy_axis_diagnostics,
            boundary_axis_diagnostics=boundary_axis_diagnostics,
        )
        method_key = f"{objective}_end_to_end:{config.backbone_name}"
        methods[method_key] = EndToEndMethodMetrics(
            model_name=config.backbone_name,
            objective=objective,
            display_name=_objective_display_name(objective),
            dimensions=int(test_embeddings.shape[1]),
            retrieval=retrieval,
            precision_at_1=retrieval.precision_at_1,
            recall_at_1=retrieval.recall_at_1,
            recall_at_2=retrieval.recall_at_2,
            recall_at_4=retrieval.recall_at_4,
            recall_at_8=retrieval.recall_at_8,
            map_at_r=retrieval.map_at_r,
            loss_history=history,
            interference=interference,
            train_interference=train_interference,
            gsi_diagnostics=gsi_diagnostics,
            selected_step=selected_step,
            selection_metric=selection_metric,
            selection_score=selection_score,
            best_test_recall_at_1=best_test_recall_at_1,
            best_test_epoch=best_test_epoch,
            test_recall_history=test_recall_history or None,
        )
        if progress_callback is not None:
            progress_callback(
                _result_with_methods(
                    config=config,
                    train_examples=len(train_examples),
                    test_examples=len(test_examples),
                    methods=dict(methods),
                )
            )
        # Drop every local that still references the model's parameters/state before
        # freeing it — the optimizer, scheduler and the EMA teacher (a full second
        # network) otherwise pin CUDA memory that empty_cache() cannot reclaim, so a
        # later objective would build its model alongside the old one. Some objectives
        # (e.g. frozen_pretrained) skip training and never create an optimizer.
        with contextlib.suppress(NameError, UnboundLocalError):
            del optimizer, scheduler
        with contextlib.suppress(NameError, UnboundLocalError):
            del ema_teacher
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _result_with_methods(
        config=config,
        train_examples=len(train_examples),
        test_examples=len(test_examples),
        methods=methods,
    )


def _result_with_methods(
    *,
    config: ImageEndToEndConfig,
    train_examples: int,
    test_examples: int,
    methods: dict[str, EndToEndMethodMetrics],
) -> ImageEndToEndResult:
    return ImageEndToEndResult(
        name="image-end-to-end-benchmark",
        dataset_name=config.dataset_name,
        protocol=config.protocol,
        config=config,
        train_examples=train_examples,
        test_examples=test_examples,
        methods=methods,
    )


def _resolve_training_schedule(
    config: ImageEndToEndConfig,
    *,
    optimization_example_count: int,
) -> tuple[int, int, int]:
    """Resolve benchmark-side steps from post-split optimization examples.

    CLI epoch-to-step conversion is only a display estimate; this function is
    authoritative because checkpoint selection can remove optimization examples.
    """
    steps_per_epoch = max(1, math.ceil(optimization_example_count / config.batch_size))
    if config.train_epochs is not None:
        return steps_per_epoch * config.train_epochs, steps_per_epoch, config.train_epochs
    total_epochs = max(1, math.ceil(config.train_steps / steps_per_epoch))
    return config.train_steps, steps_per_epoch, total_epochs


def write_image_end_to_end_report(result: ImageEndToEndResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temp_path.write_text(json.dumps(_to_payload(result), indent=2, sort_keys=True) + "\n")
    temp_path.replace(output_path)
    return output_path


class _TorchImageDataset:
    def __init__(
        self, examples: Sequence[ImageExample], transform: Callable[[object], Any]
    ) -> None:
        self._examples = list(examples)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        example = self._examples[index]
        return self._transform(example.image), int(example.label)


def _mead_multicrop_collate(
    batch: Sequence[tuple[tuple[torch.Tensor, torch.Tensor], int]],
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Stack MEAD global and local crop groups without forcing equal spatial sizes."""
    import torch

    global_crops = torch.stack([sample[0][0] for sample in batch], dim=0)
    local_crops = torch.stack([sample[0][1] for sample in batch], dim=0)
    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    return (global_crops, local_crops), labels


def _iter_training_batches(loader: Any) -> Iterator[Any]:
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            return


def _checkpoint_train_validation_split(
    examples: Sequence[ImageExample],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[ImageExample], list[ImageExample]]:
    if fraction <= 0.0:
        return list(examples), []

    grouped: dict[int, list[ImageExample]] = {}
    for example in examples:
        grouped.setdefault(int(example.label), []).append(example)

    rng = np.random.default_rng(seed)
    validation_ids: set[str] = set()
    for label in sorted(grouped):
        candidates = sorted(grouped[label], key=lambda example: example.example_id)
        if len(candidates) <= 2:
            continue
        validation_count = min(
            max(1, int(round(len(candidates) * fraction))),
            len(candidates) - 2,
        )
        selected_indices = rng.permutation(len(candidates))[:validation_count]
        validation_ids.update(candidates[int(index)].example_id for index in selected_indices)

    validation = [example for example in examples if example.example_id in validation_ids]
    optimization = [example for example in examples if example.example_id not in validation_ids]
    if len({example.label for example in validation}) < 2:
        return list(examples), []
    return optimization, validation


def _apply_training_label_noise(
    examples: Sequence[ImageExample],
    *,
    fraction: float,
    seed: int,
) -> list[ImageExample]:
    if fraction <= 0.0:
        return list(examples)

    labels = sorted({int(example.label) for example in examples})
    if len(labels) < 2:
        return list(examples)

    noise_count = int(round(len(examples) * fraction))
    if noise_count <= 0:
        return list(examples)

    rng = np.random.default_rng(seed)
    selected_indices = set(
        int(index)
        for index in rng.choice(
            np.arange(len(examples), dtype=np.int64),
            size=min(noise_count, len(examples)),
            replace=False,
        )
    )
    corrupted: list[ImageExample] = []
    for index, example in enumerate(examples):
        if index not in selected_indices:
            corrupted.append(example)
            continue
        candidate_labels = [label for label in labels if label != int(example.label)]
        corrupted.append(
            replace(example, label=int(rng.choice(np.asarray(candidate_labels, dtype=np.int64))))
        )
    return corrupted


def _frozen_class_similarity(
    examples: Sequence[ImageExample],
    *,
    config: ImageEndToEndConfig,
    model_factory: Callable[[ImageEndToEndConfig], TorchImageModel] | None,
    transform: Callable[[object], Any],
    device: Any,
    torch_module: Any,
) -> dict[int, list[int]]:
    """Rank each class's most confusable other classes from FROZEN-backbone centroids.

    Encodes the training images once with the frozen pretrained backbone, forms an
    L2-normalised mean feature per class, and returns for each class the list of other
    classes sorted by descending centroid cosine similarity. This is a fixed
    "which classes look alike" prior used to build hard-class batches — the training
    signal the model most needs for fine-grained retrieval ("Sampling Matters").
    """
    from torch.utils.data import DataLoader

    if model_factory is not None:
        model = model_factory(config).to(device)
    else:
        model = _torchvision_model_factory(config, use_embedding_head=False).to(device)
    dataset = _TorchImageDataset(examples, transform)
    loader: Any = DataLoader(
        cast(Any, dataset),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch_module.cuda.is_available(),
    )
    features, labels = _encode_model(model, loader, device, torch_module)
    del model
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    unique_labels = sorted({int(label) for label in labels.tolist()})
    centroids = []
    for label in unique_labels:
        mean = features[labels == label].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        centroids.append(mean / norm if norm > 0 else mean)
    centroid_matrix = np.stack(centroids, axis=0)  # (C, d), unit rows
    similarity = centroid_matrix @ centroid_matrix.T  # (C, C) cosine similarities
    ranking: dict[int, list[int]] = {}
    for row, label in enumerate(unique_labels):
        order = np.argsort(-similarity[row])  # descending similarity
        ranking[label] = [unique_labels[int(col)] for col in order if int(col) != row]
    return ranking


def _balanced_batch_indices(
    labels: Sequence[int],
    *,
    batch_size: int,
    group_size: int,
    samples_per_class: int = 0,
    steps: int,
    seed: int,
    class_similarity: dict[int, list[int]] | None = None,
    hard_fraction: float = 0.0,
) -> list[list[int]]:
    per_class = samples_per_class if samples_per_class > 0 else max(2 * group_size, 2)
    if batch_size < per_class * 2:
        raise ValueError("batch_size must fit at least two classes with two groups each")
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(index)
    minimum_examples = per_class if samples_per_class > 0 else 2
    eligible = sorted(
        label for label, indices in grouped.items() if len(indices) >= minimum_examples
    )
    if len(eligible) < 2:
        raise ValueError("end-to-end training requires at least two eligible classes")
    rng = np.random.default_rng(seed)
    classes_per_batch = max(2, batch_size // per_class)
    eligible_set = set(eligible)
    use_hard = class_similarity is not None and hard_fraction > 0.0
    batches: list[list[int]] = []
    for _ in range(steps):
        if use_hard and float(rng.random()) < float(hard_fraction):
            # Hard-class batch: a seed class plus its most confusable eligible
            # neighbours, so the model trains on fine-grained distinctions.
            seed_class = int(rng.choice(eligible))
            neighbours = [
                int(other)
                for other in (class_similarity or {}).get(seed_class, [])
                if int(other) in eligible_set and int(other) != seed_class
            ][: classes_per_batch - 1]
            selected = [seed_class] + neighbours
            if len(selected) < classes_per_batch:
                remaining = [c for c in eligible if c not in set(selected)]
                extra = rng.choice(
                    remaining,
                    size=min(classes_per_batch - len(selected), len(remaining)),
                    replace=False,
                )
                selected.extend(int(item) for item in extra)
            selected_labels = np.asarray(selected, dtype=np.int64)
        else:
            selected_labels = rng.choice(
                eligible,
                size=min(classes_per_batch, len(eligible)),
                replace=False,
            )
        batch: list[int] = []
        for label in sorted(int(item) for item in selected_labels):
            candidates = np.asarray(grouped[label], dtype=np.int64)
            if samples_per_class > 0:
                batch.extend(int(index) for index in rng.permutation(candidates)[:per_class])
            else:
                replace = candidates.shape[0] < per_class
                batch.extend(
                    int(index) for index in rng.choice(candidates, size=per_class, replace=replace)
                )
        batches.append(batch[:batch_size])
    return batches


class _XbmMemory:
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._embeddings: Any | None = None
        self._labels: Any | None = None

    def enqueue(self, embeddings: Any, labels: Any) -> None:
        if self._max_size == 0:
            return
        if self._embeddings is None:
            self._embeddings = embeddings.detach().cpu()
            self._labels = labels.detach().cpu()
        else:
            self._embeddings = self._cat([self._embeddings, embeddings.detach().cpu()])[
                -self._max_size :
            ]
            self._labels = self._cat([self._labels, labels.detach().cpu()])[-self._max_size :]

    def embeddings(self, device: Any) -> Any | None:
        return None if self._embeddings is None else self._embeddings.to(device)

    def labels(self, device: Any) -> Any | None:
        return None if self._labels is None else self._labels.to(device)

    def _cat(self, tensors: list[Any]) -> Any:
        import torch

        return torch.cat(tensors, dim=0)


class BGSIClassMeanState:
    """Detached EMA of normalized train-class means for stable BGSI axes."""

    def __init__(
        self,
        *,
        labels: Sequence[int],
        embedding_dimensions: int,
        momentum: float,
        device: Any,
        dtype: Any,
        torch_module: Any,
    ) -> None:
        self.label_to_index = {int(label): index for index, label in enumerate(sorted(set(labels)))}
        self.momentum = float(momentum)
        self.torch = torch_module
        self.means = torch_module.zeros(
            (len(self.label_to_index), embedding_dimensions),
            dtype=dtype,
            device=device,
        )
        self.counts = torch_module.zeros(
            len(self.label_to_index),
            dtype=torch_module.long,
            device=device,
        )

    def update(self, embeddings: Any, labels: Any) -> None:
        with self.torch.no_grad():
            for raw_label in self.torch.unique(labels).tolist():
                label = int(raw_label)
                row = self.label_to_index.get(label)
                if row is None:
                    continue
                batch_mean = embeddings[labels == label].mean(dim=0)
                batch_mean = _normalize(batch_mean, self.torch)
                if int(self.counts[row].item()) == 0:
                    self.means[row] = batch_mean
                else:
                    blended = self.momentum * self.means[row] + (1.0 - self.momentum) * batch_mean
                    self.means[row] = _normalize(blended, self.torch)
                self.counts[row] += 1


class _BestCheckpoint:
    def __init__(self, *, metric_name: str, mode: Literal["max", "min"]) -> None:
        self.metric_name = metric_name
        self.mode = mode
        self.best_score: float | None = None
        self.best_step: int | None = None
        self._state_dict: dict[str, Any] | None = None

    def update(self, *, score: float, step: int, model: TorchImageModel) -> bool:
        if not self._is_better(score):
            return False

        self.best_score = score
        self.best_step = step
        self._state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in cast(Any, model).state_dict().items()
        }
        return True

    def restore(self, model: TorchImageModel) -> bool:
        if self._state_dict is None:
            return False
        cast(Any, model).load_state_dict(self._state_dict)
        return True

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score


def _uses_metric_proxies(objective: str, config: ImageEndToEndConfig) -> bool:
    if objective in {
        "proxy_anchor",
        "proxy_anchor_group",
        "proxy_anchor_synthesis",
        "proxy_anchor_subcenter",
        "proxy_anchor_uniformity",
        "pfml",
        "proxy_anchor_gsi",
        "proxy_anchor_bgsi",
        "pfml_gsi",
        "hist_proxy_anchor",
    }:
        if config.proxy_count_per_class <= 0:
            raise ValueError(
                f"the {objective} objective requires proxy_count_per_class > 0: "
                "it is built from learnable class proxies"
            )
        return True
    # symmetric_potential and lennard_jones can use proxies as extra field sources
    # but do not require them — they also work on the batch alone.
    if objective in {"symmetric_potential", "lennard_jones"}:
        return config.proxy_count_per_class > 0
    if objective in {"proxy_anchor_lj", "proxy_anchor_antico", "bio_physical_bond"}:
        if config.proxy_count_per_class <= 0:
            raise ValueError(f"the {objective} objective requires proxy_count_per_class > 0")
        return True
    return (
        objective in {"group_supcon_xbm_radius", "group_potential", "group_potential_xbm"}
        and (config.proxy_weight > 0.0 or config.potential_weight > 0.0)
        and config.proxy_count_per_class > 0
    )


def _attach_metric_proxies(
    model: TorchImageModel,
    *,
    train_labels: Sequence[int],
    config: ImageEndToEndConfig,
    torch_module: Any,
) -> None:
    unique_labels = sorted({int(label) for label in train_labels})
    proxy_labels = [label for label in unique_labels for _ in range(config.proxy_count_per_class)]
    if not proxy_labels:
        return
    proxy_tensor = torch_module.randn(
        len(proxy_labels),
        config.embedding_dimensions,
        dtype=torch_module.float32,
    )
    proxy_tensor = _normalize(proxy_tensor, torch_module)
    cast(Any, model).register_parameter(
        "metric_proxies",
        torch_module.nn.Parameter(proxy_tensor),
    )
    cast(Any, model).register_buffer(
        "metric_proxy_labels",
        torch_module.tensor(proxy_labels, dtype=torch_module.long),
    )


def _attach_hist_module(
    model: TorchImageModel,
    *,
    train_labels: Sequence[int],
    config: ImageEndToEndConfig,
    torch_module: Any,
) -> None:
    unique_labels = sorted({int(label) for label in train_labels})
    module = _build_hist_module(
        nb_classes=len(unique_labels),
        sz_embed=config.embedding_dimensions,
        hidden=config.hist_hidden,
        torch_module=torch_module,
    )
    cast(Any, model).add_module("hist_module", module)
    cast(Any, model).hist_label_to_index = {
        label: index for index, label in enumerate(unique_labels)
    }


def _metric_proxy_embeddings(model: TorchImageModel) -> Any | None:
    return getattr(model, "metric_proxies", None)


def _metric_proxy_labels(model: TorchImageModel) -> Any | None:
    return getattr(model, "metric_proxy_labels", None)


def _group_supcon_xbm_radius_loss(
    embeddings: Any,
    labels: Any,
    *,
    memory_embeddings: Any | None,
    memory_labels: Any | None,
    proxy_embeddings: Any | None,
    proxy_labels: Any | None,
    point_weight: float,
    group_weight: float,
    xbm_weight: float,
    radius_weight: float,
    radius_target: float,
    proxy_weight: float,
    potential_weight: float,
    potential_delta: float,
    potential_alpha: float,
    group_size: int,
    temperature: float,
    torch_module: Any,
) -> Any:
    loss = point_weight * _supervised_contrastive_loss(
        embeddings,
        labels,
        contrast_embeddings=embeddings,
        contrast_labels=labels,
        temperature=temperature,
        torch_module=torch_module,
        exclude_self=True,
    )
    centroids, centroid_labels = _group_centroids(embeddings, labels, group_size, torch_module)
    if centroids.shape[0] >= 2 and torch_module.unique(centroid_labels).shape[0] >= 2:
        loss = loss + group_weight * _supervised_contrastive_loss(
            centroids,
            centroid_labels,
            contrast_embeddings=centroids,
            contrast_labels=centroid_labels,
            temperature=temperature,
            torch_module=torch_module,
            exclude_self=True,
        )
    if memory_embeddings is not None and memory_labels is not None and xbm_weight > 0.0:
        contrast_embeddings = torch_module.cat([embeddings, memory_embeddings], dim=0)
        contrast_labels = torch_module.cat([labels, memory_labels], dim=0)
        loss = loss + xbm_weight * _supervised_contrastive_loss(
            embeddings,
            labels,
            contrast_embeddings=contrast_embeddings,
            contrast_labels=contrast_labels,
            temperature=temperature,
            torch_module=torch_module,
            exclude_self=True,
        )
    if proxy_embeddings is not None and proxy_labels is not None and proxy_weight > 0.0:
        normalized_proxies = _normalize(proxy_embeddings, torch_module)
        contrast_embeddings = torch_module.cat([embeddings, normalized_proxies], dim=0)
        contrast_labels = torch_module.cat([labels, proxy_labels], dim=0)
        loss = loss + proxy_weight * _supervised_contrastive_loss(
            embeddings,
            labels,
            contrast_embeddings=contrast_embeddings,
            contrast_labels=contrast_labels,
            temperature=temperature,
            torch_module=torch_module,
            exclude_self=True,
        )
    if potential_weight > 0.0:
        potential_embeddings = embeddings
        potential_labels = labels
        if proxy_embeddings is not None and proxy_labels is not None:
            potential_embeddings = torch_module.cat(
                [embeddings, _normalize(proxy_embeddings, torch_module)],
                dim=0,
            )
            potential_labels = torch_module.cat([labels, proxy_labels], dim=0)
        loss = loss + potential_weight * _local_potential_loss(
            potential_embeddings,
            potential_labels,
            delta=potential_delta,
            alpha=potential_alpha,
            torch_module=torch_module,
        )
    if radius_weight > 0.0:
        loss = loss + radius_weight * _radius_penalty(
            embeddings,
            labels,
            target=radius_target,
            torch_module=torch_module,
        )
    return loss


def _group_potential_loss(
    embeddings: Any,
    labels: Any,
    *,
    memory_embeddings: Any | None,
    memory_labels: Any | None,
    proxy_embeddings: Any | None,
    proxy_labels: Any | None,
    point_weight: float,
    group_weight: float,
    xbm_weight: float,
    proxy_weight: float,
    potential_weight: float,
    potential_delta: float,
    potential_alpha: float,
    group_size: int,
    temperature: float,
    torch_module: Any,
) -> Any:
    loss = embeddings.sum() * 0.0
    if point_weight > 0.0:
        loss = loss + point_weight * _supervised_contrastive_loss(
            embeddings,
            labels,
            contrast_embeddings=embeddings,
            contrast_labels=labels,
            temperature=temperature,
            torch_module=torch_module,
            exclude_self=True,
        )

    centroids, centroid_labels = _group_centroids(embeddings, labels, group_size, torch_module)
    field_embeddings = embeddings
    field_labels = labels
    if centroids.shape[0] > 0:
        field_embeddings = torch_module.cat([field_embeddings, centroids], dim=0)
        field_labels = torch_module.cat([field_labels, centroid_labels], dim=0)
        if group_weight > 0.0 and torch_module.unique(centroid_labels).shape[0] >= 2:
            loss = loss + group_weight * _supervised_contrastive_loss(
                centroids,
                centroid_labels,
                contrast_embeddings=field_embeddings,
                contrast_labels=field_labels,
                temperature=temperature,
                torch_module=torch_module,
                exclude_self=False,
            )

    if proxy_embeddings is not None and proxy_labels is not None:
        normalized_proxies = _normalize(proxy_embeddings, torch_module)
        field_embeddings = torch_module.cat([field_embeddings, normalized_proxies], dim=0)
        field_labels = torch_module.cat([field_labels, proxy_labels], dim=0)
        if proxy_weight > 0.0:
            # field_embeddings begins with the anchors themselves, so exclude the
            # diagonal self-match (proxies are appended past the diagonal and stay).
            loss = loss + proxy_weight * _supervised_contrastive_loss(
                embeddings,
                labels,
                contrast_embeddings=field_embeddings,
                contrast_labels=field_labels,
                temperature=temperature,
                torch_module=torch_module,
                exclude_self=True,
            )

    if memory_embeddings is not None and memory_labels is not None and xbm_weight > 0.0:
        contrast_embeddings = torch_module.cat([field_embeddings, memory_embeddings], dim=0)
        contrast_labels = torch_module.cat([field_labels, memory_labels], dim=0)
        # The contrast set opens with the anchors (field_embeddings), so exclude the
        # diagonal self-match; the memory bank is appended past it and still counts.
        loss = loss + xbm_weight * _supervised_contrastive_loss(
            embeddings,
            labels,
            contrast_embeddings=contrast_embeddings,
            contrast_labels=contrast_labels,
            temperature=temperature,
            torch_module=torch_module,
            exclude_self=True,
        )

    if potential_weight > 0.0:
        loss = loss + potential_weight * _local_potential_loss(
            field_embeddings,
            field_labels,
            delta=potential_delta,
            alpha=potential_alpha,
            torch_module=torch_module,
        )
    return loss


def _apply_teacher_similarity_regularization(loss: Any, kwargs: dict[str, Any]) -> Any:
    config = cast(ImageEndToEndConfig, kwargs["config"])
    embeddings = kwargs["embeddings"]
    teacher_embeddings = kwargs["teacher_embeddings"]
    torch_module = kwargs["torch_module"]
    if config.teacher_similarity_weight > 0.0 and teacher_embeddings is not None:
        loss = loss + config.teacher_similarity_weight * _pairwise_similarity_preservation_loss(
            embeddings,
            teacher_embeddings,
            torch_module=torch_module,
        )
    return loss


def _hist_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    hist_module = kwargs["hist_module"]
    hist_label_to_index = kwargs["hist_label_to_index"]
    if hist_module is None or hist_label_to_index is None:
        raise ValueError("the hist objective requires an attached hist_module")
    hist_loss = _hist_loss(
        embeddings,
        labels,
        hist_module=hist_module,
        label_to_index=hist_label_to_index,
        tau=config.hist_tau,
        alpha=config.hist_alpha,
        lambda_s=config.hist_lambda_s,
        var_floor=config.hist_var_floor,
        torch_module=torch_module,
    )
    if config.uniformity_weight > 0.0:
        # Novel: add a thermodynamic Gaussian-potential uniformity term on the
        # backbone embedding on top of HIST's hypergraph structure -- explicit
        # zero-shot spread that HIST's relational loss does not directly enforce.
        hist_loss = hist_loss + config.uniformity_weight * _gaussian_potential_uniformity_loss(
            embeddings,
            t=config.uniformity_t,
            torch_module=torch_module,
        )
    return hist_loss


def _hist_proxy_anchor_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    hist_module = kwargs["hist_module"]
    hist_label_to_index = kwargs["hist_label_to_index"]
    # Fused single-model loss: HIST hypergraph + Proxy Anchor, so one model gets
    # both HIST's per-class prototypes and PA's proxy margins. The EMA-teacher
    # relational distillation is added on top by the caller (ungated), giving a
    # single method that should be strong on every dataset.
    if hist_module is None or hist_label_to_index is None:
        raise ValueError("the hist_proxy_anchor objective requires an attached hist_module")
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError(
            "the hist_proxy_anchor objective requires class proxies (proxy_count_per_class > 0)"
        )
    hist_term = _hist_loss(
        embeddings,
        labels,
        hist_module=hist_module,
        label_to_index=hist_label_to_index,
        tau=config.hist_tau,
        alpha=config.hist_alpha,
        lambda_s=config.hist_lambda_s,
        var_floor=config.hist_var_floor,
        torch_module=torch_module,
    )
    proxy_term = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        torch_module=torch_module,
    )
    return hist_term + config.proxy_fusion_weight * proxy_term


def _triplet_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _semi_hard_triplet_loss(
        embeddings,
        labels,
        margin=config.triplet_margin,
        torch_module=torch_module,
    )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _batch_hard_triplet_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _batch_hard_triplet_loss(
        embeddings,
        labels,
        margin=config.triplet_margin,
        torch_module=torch_module,
    )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _supcon_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _supervised_contrastive_loss(
        embeddings,
        labels,
        contrast_embeddings=embeddings,
        contrast_labels=labels,
        temperature=config.temperature,
        torch_module=torch_module,
        exclude_self=True,
    )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _group_supcon_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _group_supcon_xbm_radius_loss(
        embeddings,
        labels,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=None,
        proxy_labels=None,
        point_weight=config.point_weight,
        group_weight=config.group_weight,
        xbm_weight=0.0,
        radius_weight=0.0,
        radius_target=config.radius_target,
        proxy_weight=0.0,
        potential_weight=0.0,
        potential_delta=config.potential_delta,
        potential_alpha=config.potential_alpha,
        group_size=config.group_size,
        temperature=config.temperature,
        torch_module=torch_module,
    )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _group_supcon_xbm_radius_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    memory_embeddings = kwargs["memory_embeddings"]
    memory_labels = kwargs["memory_labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _group_supcon_xbm_radius_loss(
        embeddings,
        labels,
        memory_embeddings=memory_embeddings,
        memory_labels=memory_labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        point_weight=config.point_weight,
        group_weight=config.group_weight,
        xbm_weight=config.xbm_weight,
        radius_weight=config.radius_weight,
        radius_target=config.radius_target,
        proxy_weight=config.proxy_weight,
        potential_weight=config.potential_weight,
        potential_delta=config.potential_delta,
        potential_alpha=config.potential_alpha,
        group_size=config.group_size,
        temperature=config.temperature,
        torch_module=torch_module,
    )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _pfml_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _pfml_potential_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        delta=config.potential_delta,
        alpha=config.potential_alpha,
        torch_module=torch_module,
    )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _proxy_anchor_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        torch_module=torch_module,
    )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _proxy_anchor_group_objective_loss(**kwargs: Any) -> Any:
    objective = kwargs["objective"]
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError(
            f"the {objective} objective requires class proxies (proxy_count_per_class > 0)"
        )
    loss = _proxy_anchor_group_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        tau_assign=config.proxy_anchor_group_tau_assign,
        torch_module=torch_module,
    )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _proxy_anchor_synthesis_objective_loss(**kwargs: Any) -> Any:
    objective = kwargs["objective"]
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    generator = kwargs["generator"]
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError(
            f"the {objective} objective requires class proxies (proxy_count_per_class > 0)"
        )
    loss = _proxy_synthesis_proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        ratio=config.synthesis_ratio,
        beta_alpha=config.synthesis_beta_alpha,
        generator=generator,
        group_mix=config.synthesis_group_mix,
        pair_selection=config.synthesis_pair_selection,
        pair_temperature=config.synthesis_pair_temperature,
        torch_module=torch_module,
    )
    if config.synthesis_compactness_weight > 0.0:
        # Novel complement: Proxy Synthesis densifies boundaries (helps R@1) but
        # fragments class neighbourhoods (measured -MAP@R). A per-class compactness
        # term pulls members toward their centroid, targeting exactly that cost so
        # the combination can beat vanilla PS on both metrics.
        loss = loss + config.synthesis_compactness_weight * _radius_penalty(
            _normalize(embeddings, torch_module),
            labels,
            target=config.synthesis_compactness_target,
            torch_module=torch_module,
        )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _proxy_anchor_bgsi_objective_loss(**kwargs: Any) -> Any:
    objective = kwargs["objective"]
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    step = cast(int, kwargs["step"])
    steps_per_epoch = cast(int, kwargs["steps_per_epoch"])
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    generator = kwargs["generator"]
    bgsi_state = cast(BGSIClassMeanState | None, kwargs["bgsi_state"])
    gsi_step_diagnostics = cast(list[dict[str, float]] | None, kwargs["gsi_step_diagnostics"])
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError(
            f"the {objective} objective requires class proxies (proxy_count_per_class > 0)"
        )
    loss = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        torch_module=torch_module,
    )
    if config.bgsi_weight > 0.0 and step > config.bgsi_start_epoch * steps_per_epoch:
        axes_by_class = _bgsi_axes_for_mode(
            embeddings,
            labels,
            axis_mode=config.bgsi_axis_mode,
            top_k=config.bgsi_top_k,
            temperature=config.bgsi_temperature,
            generator=generator,
            ema_state=bgsi_state,
            min_axis_observations=config.bgsi_min_axis_observations,
            use_axis_agreement_gate=config.bgsi_use_axis_agreement_gate,
            axis_agreement=config.bgsi_axis_agreement,
            torch_module=torch_module,
        )
        bgsi_loss, diagnostics = _gsi_interference_loss_with_diagnostics(
            embeddings,
            labels,
            axes_by_class=axes_by_class,
            floor=config.bgsi_floor,
            variance_floor=config.bgsi_variance_floor,
            min_group_size=config.bgsi_min_group_size,
            torch_module=torch_module,
        )
        if diagnostics is not None and gsi_step_diagnostics is not None:
            diagnostics.update(
                _bgsi_axis_step_diagnostics(
                    labels,
                    axes_by_class=axes_by_class,
                    axis_mode=config.bgsi_axis_mode,
                    bgsi_state=bgsi_state,
                    min_axis_observations=config.bgsi_min_axis_observations,
                    use_axis_agreement_gate=config.bgsi_use_axis_agreement_gate,
                    torch_module=torch_module,
                )
            )
            gsi_step_diagnostics.append(diagnostics)
        loss = loss + config.bgsi_weight * bgsi_loss
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _bio_physical_bond_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _bio_physical_bond_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        sigma=config.lj_sigma,
        power=config.lj_power,
        niche_weight=config.bond_niche_weight,
        antico_eps=config.antico_eps,
        torch_module=torch_module,
    )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _proxy_anchor_antico_objective_loss(**kwargs: Any) -> Any:
    objective = kwargs["objective"]
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError(
            f"the {objective} objective requires class proxies (proxy_count_per_class > 0)"
        )
    loss = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        torch_module=torch_module,
    )
    if config.antico_weight > 0.0:
        # Anti-collapse: MAXIMISE the coding rate (log-volume) of the batch
        # features and/or proxies -> subtract weight * R from the loss.
        rate = embeddings.sum() * 0.0
        if config.antico_target in {"feature", "both"}:
            rate = rate + _coding_rate(
                _normalize(embeddings, torch_module),
                eps=config.antico_eps,
                torch_module=torch_module,
            )
        if config.antico_target in {"proxy", "both"}:
            rate = rate + _coding_rate(
                _normalize(proxy_embeddings, torch_module),
                eps=config.antico_eps,
                torch_module=torch_module,
            )
        loss = loss - config.antico_weight * rate
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _proxy_anchor_subcenter_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _subcenter_proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        gamma=config.subcenter_gamma,
        torch_module=torch_module,
    )
    if config.antico_weight > 0.0:
        # Optional MCR2 anti-collapse on the batch features (keeps the embedding
        # high-rank while sub-centers carve up each class into modes).
        rate = _coding_rate(
            _normalize(embeddings, torch_module),
            eps=config.antico_eps,
            torch_module=torch_module,
        )
        loss = loss - config.antico_weight * rate
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _proxy_anchor_uniformity_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        torch_module=torch_module,
    )
    if config.uniformity_weight > 0.0:
        loss = loss + config.uniformity_weight * _gaussian_potential_uniformity_loss(
            embeddings,
            t=config.uniformity_t,
            torch_module=torch_module,
        )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _symmetric_potential_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _symmetric_potential_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        delta=config.potential_delta,
        alpha=config.potential_alpha,
        torch_module=torch_module,
    )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _proxy_anchor_lj_objective_loss(**kwargs: Any) -> Any:
    objective = kwargs["objective"]
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError(
            f"the {objective} objective requires class proxies (proxy_count_per_class > 0)"
        )
    loss = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        torch_module=torch_module,
    )
    if config.lj_intra_weight > 0.0:
        loss = loss + config.lj_intra_weight * _lennard_jones_intra_term(
            embeddings,
            labels,
            sigma=config.lj_sigma,
            power=config.lj_power,
            torch_module=torch_module,
        )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _lennard_jones_objective_loss(**kwargs: Any) -> Any:
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _lennard_jones_loss(
        embeddings,
        labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        sigma=config.lj_sigma,
        power=config.lj_power,
        repulsion_weight=config.lj_repulsion_weight,
        sigma_neg=config.lj_sigma_neg,
        torch_module=torch_module,
    )
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _gsi_objective_loss(**kwargs: Any) -> Any:
    objective = kwargs["objective"]
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    step = cast(int, kwargs["step"])
    steps_per_epoch = cast(int, kwargs["steps_per_epoch"])
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    generator = kwargs["generator"]
    gsi_step_diagnostics = cast(list[dict[str, float]] | None, kwargs["gsi_step_diagnostics"])
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError(
            f"the {objective} objective requires class proxies (proxy_count_per_class > 0)"
        )
    if objective == "proxy_anchor_gsi":
        loss = _proxy_anchor_loss(
            embeddings,
            labels,
            proxy_embeddings=proxy_embeddings,
            proxy_labels=proxy_labels,
            alpha=config.proxy_anchor_alpha,
            delta=config.proxy_anchor_delta,
            torch_module=torch_module,
        )
    else:
        loss = _pfml_potential_loss(
            embeddings,
            labels,
            proxy_embeddings=proxy_embeddings,
            proxy_labels=proxy_labels,
            delta=config.potential_delta,
            alpha=config.potential_alpha,
            torch_module=torch_module,
        )
    if config.gsi_weight > 0.0 and step > config.gsi_start_epoch * steps_per_epoch:
        axes_by_class = _gsi_axes_for_mode(
            proxy_embeddings,
            proxy_labels,
            axis_mode=config.gsi_axis_mode,
            top_k=config.gsi_top_k,
            generator=generator,
            torch_module=torch_module,
        )
        gsi_loss, diagnostics = _gsi_interference_loss_with_diagnostics(
            embeddings,
            labels,
            axes_by_class=axes_by_class,
            floor=config.gsi_floor,
            variance_floor=config.gsi_variance_floor,
            min_group_size=config.gsi_min_group_size,
            torch_module=torch_module,
        )
        if diagnostics is not None and gsi_step_diagnostics is not None:
            gsi_step_diagnostics.append(diagnostics)
        loss = loss + config.gsi_weight * gsi_loss
    return _apply_teacher_similarity_regularization(loss, kwargs)


def _group_potential_objective_loss(**kwargs: Any) -> Any:
    objective = kwargs["objective"]
    embeddings = kwargs["embeddings"]
    labels = kwargs["labels"]
    memory_embeddings = kwargs["memory_embeddings"]
    memory_labels = kwargs["memory_labels"]
    proxy_embeddings = kwargs["proxy_embeddings"]
    proxy_labels = kwargs["proxy_labels"]
    config = cast(ImageEndToEndConfig, kwargs["config"])
    torch_module = kwargs["torch_module"]
    loss = _group_potential_loss(
        embeddings,
        labels,
        memory_embeddings=memory_embeddings if objective == "group_potential_xbm" else None,
        memory_labels=memory_labels if objective == "group_potential_xbm" else None,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        point_weight=config.point_weight,
        group_weight=config.group_weight,
        xbm_weight=config.xbm_weight,
        proxy_weight=config.proxy_weight,
        potential_weight=config.potential_weight,
        potential_delta=config.potential_delta,
        potential_alpha=config.potential_alpha,
        group_size=config.group_size,
        temperature=config.temperature,
        torch_module=torch_module,
    )
    return _apply_teacher_similarity_regularization(loss, kwargs)


_OBJECTIVE_LOSSES: dict[str, Callable[..., Any]] = {
    "hist": _hist_objective_loss,
    "hist_proxy_anchor": _hist_proxy_anchor_objective_loss,
    "triplet": _triplet_objective_loss,
    "triplet_pretrained": _triplet_objective_loss,
    "batch_hard_triplet": _batch_hard_triplet_objective_loss,
    "supcon": _supcon_objective_loss,
    "group_supcon": _group_supcon_objective_loss,
    "group_supcon_xbm_radius": _group_supcon_xbm_radius_objective_loss,
    "pfml": _pfml_objective_loss,
    "proxy_anchor": _proxy_anchor_objective_loss,
    "proxy_anchor_group": _proxy_anchor_group_objective_loss,
    "proxy_anchor_synthesis": _proxy_anchor_synthesis_objective_loss,
    "proxy_anchor_bgsi": _proxy_anchor_bgsi_objective_loss,
    "bio_physical_bond": _bio_physical_bond_objective_loss,
    "proxy_anchor_antico": _proxy_anchor_antico_objective_loss,
    "proxy_anchor_subcenter": _proxy_anchor_subcenter_objective_loss,
    "proxy_anchor_uniformity": _proxy_anchor_uniformity_objective_loss,
    "symmetric_potential": _symmetric_potential_objective_loss,
    "proxy_anchor_lj": _proxy_anchor_lj_objective_loss,
    "lennard_jones": _lennard_jones_objective_loss,
    "proxy_anchor_gsi": _gsi_objective_loss,
    "pfml_gsi": _gsi_objective_loss,
    "group_potential": _group_potential_objective_loss,
    "group_potential_xbm": _group_potential_objective_loss,
}


def _loss_for_objective(
    objective: EndToEndObjective,
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
    bgsi_state: BGSIClassMeanState | None = None,
    gsi_step_diagnostics: list[dict[str, float]] | None = None,
    hist_module: Any | None = None,
    hist_label_to_index: dict[int, int] | None = None,
) -> Any:
    objective_loss = _OBJECTIVE_LOSSES.get(objective)
    if objective_loss is None:
        raise ValueError(f"unsupported end-to-end objective: {objective}")
    return objective_loss(
        objective=objective,
        embeddings=embeddings,
        labels=labels,
        step=step,
        steps_per_epoch=steps_per_epoch,
        memory_embeddings=memory_embeddings,
        memory_labels=memory_labels,
        proxy_embeddings=proxy_embeddings,
        proxy_labels=proxy_labels,
        config=config,
        torch_module=torch_module,
        teacher_embeddings=teacher_embeddings,
        generator=generator,
        bgsi_state=bgsi_state,
        gsi_step_diagnostics=gsi_step_diagnostics,
        hist_module=hist_module,
        hist_label_to_index=hist_label_to_index,
    )


def _objective_display_name(objective: str) -> str:
    names = {
        "frozen_pretrained": "Frozen Pretrained ResNet-50",
        "frozen": "Frozen ResNet-50",
        "triplet": "Triplet",
        "triplet_pretrained": "Triplet on Pretrained Features",
        "batch_hard_triplet": "Batch-Hard Triplet",
        "supcon": "Supervised Contrastive",
        "group_supcon": "Group SupCon",
        "group_supcon_xbm_radius": "Group SupCon + XBM + Radius",
        "group_potential": "Group Potential",
        "group_potential_xbm": "Group Potential + XBM",
        "proxy_anchor": "Proxy Anchor",
        "proxy_anchor_group": "Proxy Anchor (Group Proxies)",
        "proxy_anchor_synthesis": "Proxy Anchor + Synthesis",
        "proxy_anchor_subcenter": "Proxy Anchor (Sub-Center + Anti-Collapse)",
        "proxy_anchor_uniformity": "Proxy Anchor + Gaussian-Potential Uniformity",
        "pfml": "PFML (Potential Field)",
        "symmetric_potential": "Symmetric Potential Field",
        "lennard_jones": "Lennard-Jones Potential",
        "proxy_anchor_lj": "Proxy Anchor + Lennard-Jones",
        "proxy_anchor_antico": "Proxy Anchor + Anti-Collapse (Coding Rate)",
        "bio_physical_bond": "Bio-Physical Bond (LJ-Boltzmann-Niche)",
        "hist": "HIST (Hypergraph Semantic Tuplet)",
        "hist_proxy_anchor": "HIST + Proxy Anchor (fused)",
        "proxy_anchor_gsi": "Proxy Anchor + GSI",
        "proxy_anchor_bgsi": "Proxy Anchor + BGSI",
        "pfml_gsi": "PFML + GSI",
    }
    return names.get(objective, objective.replace("_", " ").title())


def _uses_pretrained_feature_model(
    objective: str,
    backbone_name: str,
    model_factory: Callable[[ImageEndToEndConfig], TorchImageModel] | None,
) -> bool:
    return (
        objective in {"frozen_pretrained", "triplet_pretrained"}
        and model_factory is None
        and backbone_name == "resnet50"
    )


def _teacher_model_for_config(
    config: ImageEndToEndConfig,
    *,
    model_factory: Callable[[ImageEndToEndConfig], TorchImageModel] | None,
    device: Any,
) -> TorchImageModel | None:
    if config.teacher_similarity_weight <= 0.0:
        return None
    if model_factory is not None:
        teacher = model_factory(config).to(device)
    elif config.backbone_name == "resnet50":
        teacher = _torchvision_model_factory(config, use_embedding_head=False).to(device)
    else:
        return None
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return cast(TorchImageModel, teacher)


def _freeze_batch_norm_layers(model: TorchImageModel) -> None:
    # Freeze only the *backbone* BatchNorm (paper protocol). The HIST HGNN has its
    # own BatchNorm1d (`hist_module.bn1`) that must stay in train mode and normalize
    # with live batch statistics, exactly as in the reference HGNN -- freezing it
    # miscalibrates the hypergraph classifier and sends harmful gradients into the
    # backbone embedding (collapses zero-shot retrieval while training loss falls).
    for name, module in cast(Any, model).named_modules():
        if name.startswith("hist_module"):
            continue
        if module.__class__.__name__.startswith("BatchNorm"):
            module.eval()


def _update_ema_teacher(teacher: Any, student: Any, *, momentum: float) -> None:
    """In-place EMA update of the teacher: theta_t <- m*theta_t + (1-m)*theta_s."""
    for teacher_param, student_param in zip(
        teacher.parameters(), student.parameters(), strict=True
    ):
        # Operate on `.data` so the update works whether or not the teacher params
        # still carry requires_grad, and never touches the autograd graph.
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers(), strict=True):
        teacher_buffer.data.copy_(student_buffer.data)


def _batch_hard_triplet_loss(
    embeddings: Any,
    labels: Any,
    *,
    margin: float,
    torch_module: Any,
) -> Any:
    distances = torch_module.cdist(embeddings, embeddings, p=2)
    same_label = labels[:, None].eq(labels[None, :])
    valid_positive = same_label.clone()
    valid_positive.fill_diagonal_(False)
    valid_negative = ~same_label

    hardest_positive = distances.masked_fill(~valid_positive, -1.0).max(dim=1).values
    hardest_negative = distances.masked_fill(~valid_negative, 1.0e6).min(dim=1).values
    keep = valid_positive.any(dim=1) & valid_negative.any(dim=1)
    if not bool(keep.any()):
        return embeddings.sum() * 0.0
    return torch_module.relu(hardest_positive - hardest_negative + margin)[keep].mean()


def _semi_hard_triplet_loss(
    embeddings: Any,
    labels: Any,
    *,
    margin: float,
    torch_module: Any,
) -> Any:
    distances = torch_module.cdist(embeddings, embeddings, p=2)
    same_label = labels[:, None].eq(labels[None, :])
    valid_positive = same_label.clone()
    valid_positive.fill_diagonal_(False)
    valid_negative = ~same_label

    positive_distances = distances[:, :, None]
    negative_distances = distances[:, None, :]
    triplet_losses = positive_distances - negative_distances + margin
    candidate_mask = valid_positive[:, :, None] & valid_negative[:, None, :]
    semi_hard_mask = (
        candidate_mask & (negative_distances > positive_distances) & (triplet_losses > 0.0)
    )
    if bool(semi_hard_mask.any()):
        return triplet_losses[semi_hard_mask].mean()

    active_mask = candidate_mask & (triplet_losses > 0.0)
    if bool(active_mask.any()):
        return triplet_losses[active_mask].mean()
    return embeddings.sum() * 0.0


def _local_potential_loss(
    embeddings: Any,
    labels: Any,
    *,
    delta: float,
    alpha: float,
    torch_module: Any,
) -> Any:
    distances = torch_module.cdist(embeddings, embeddings, p=2).clamp_min(1.0e-12)
    same_label = labels[:, None].eq(labels[None, :])
    valid_positive = same_label.clone()
    valid_positive.fill_diagonal_(False)
    valid_negative = ~same_label

    scaled_inverse = (float(delta) / distances).pow(float(alpha))
    attraction = -torch_module.where(
        distances < float(delta),
        torch_module.ones_like(distances),
        scaled_inverse,
    )
    repulsion = torch_module.where(
        distances < float(delta),
        scaled_inverse - 1.0,
        torch_module.zeros_like(distances),
    )
    terms = []
    if bool(valid_positive.any()):
        terms.append(attraction[valid_positive].mean())
    if bool(valid_negative.any()):
        terms.append(repulsion[valid_negative].mean())
    if not terms:
        return embeddings.sum() * 0.0
    return torch_module.stack(terms).sum()


def _pfml_potential_loss(
    embeddings: Any,
    labels: Any,
    *,
    proxy_embeddings: Any | None,
    proxy_labels: Any | None,
    delta: float,
    alpha: float,
    torch_module: Any,
) -> Any:
    """Faithful PFML total potential energy (arXiv 2405.18560, CVPR 2025).

    Formulation verified against the arXiv HTML version on 2026-07-04:

    - Eq. 1 attractive kernel: ``psi_att(r, z) = -1/delta^alpha`` when
      ``||r - z|| < delta``, else ``-1/||r - z||^alpha`` — attraction decays
      with distance and saturates (zero force) inside the ``delta`` margin.
    - Eq. 2 repulsive kernel: ``psi_rep(r, z) = 1/||r - z||^alpha`` when
      ``||r - z|| < delta``, else ``1/delta^alpha`` — repulsion only exerts
      force inside the ``delta`` margin.
    - Eq. 3-5 class field ``Psi_j(r)``: attractive contributions from
      same-class batch embeddings AND the M proxies of class j; repulsive
      contributions from different-class batch embeddings AND every other
      class's proxies.
    - Eq. 6 total energy ``U = sum_i Psi_{y_i}(z_i) + sum_{j,k} Psi_j(p_jk)``:
      every unordered pair among {batch embeddings} + {all proxies} interacts
      once per direction, so U equals the all-pairs potential on the combined
      set — sample<->sample, sample<->proxy AND proxy<->proxy pairs.
    - Paper protocol: M = 15 proxies per class (CUB, Cars) and M = 2 (SOP);
      proxy learning rate x100; delta cross-validated in [0.1, 0.3]; alpha
      cross-validated in {0..6}; Adam at 5e-4 for 200 epochs; embeddings
      l2-normalized. Here ``delta``/``alpha`` map to ``potential_delta`` and
      ``potential_alpha``.

    Deliberate deviations (gradients unchanged up to a uniform scale):
    - Self-interactions (distance 0, always the constant kernel branch) are
      excluded; they only shift the loss by a constant.
    - The paper's raw double sum is divided by the number of interacting
      ordered pairs so the reported loss is batch-size independent; Adam is
      invariant to this uniform rescaling.
    - Distances are clamped below at 1e-4 for numerical stability of the
      singular repulsive kernel.
    - Proxies are l2-normalized before computing distances, matching how every
      other proxy-based term in this module consumes ``metric_proxies`` (the
      paper does not specify proxy normalization).
    """
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError("the pfml objective requires class proxies (proxy_count_per_class > 0)")
    points = torch_module.cat(
        [embeddings, _normalize(proxy_embeddings, torch_module)],
        dim=0,
    )
    point_labels = torch_module.cat([labels, proxy_labels], dim=0)
    if points.shape[0] < 2:
        return embeddings.sum() * 0.0
    distances = torch_module.cdist(points, points, p=2).clamp_min(1.0e-4)
    same_label = point_labels[:, None].eq(point_labels[None, :])
    off_diagonal = ~torch_module.eye(
        points.shape[0],
        dtype=torch_module.bool,
        device=points.device,
    )
    inside_margin = distances < float(delta)
    inverse_power = distances.pow(-float(alpha))
    saturation = torch_module.full_like(distances, float(delta) ** -float(alpha))
    attraction = -torch_module.where(inside_margin, saturation, inverse_power)
    repulsion = torch_module.where(inside_margin, inverse_power, saturation)
    potentials = torch_module.where(same_label, attraction, repulsion)
    return potentials[off_diagonal].mean()


def _symmetric_potential_loss(
    embeddings: Any,
    labels: Any,
    *,
    proxy_embeddings: Any | None,
    proxy_labels: Any | None,
    delta: float,
    alpha: float,
    torch_module: Any,
) -> Any:
    """Symmetric Potential Field (SPF) metric-learning energy — a novel objective.

    Motivation: PFML (arXiv 2405.18560) uses a long-range *attraction* kernel but a
    short-range *repulsion* kernel (constant, zero-force beyond ``delta``). On the
    L2-normalised unit sphere almost no different-class pair is ever within a small
    ``delta``, so the only globally active force is same-class attraction and the
    backbone contracts to a delta-scale blob (empirically confirmed: our faithful
    PFML reproduction collapsed to R@1 0.0155). SPF fixes this by making BOTH kernels
    long-range and distance-decaying (a Coulomb-like symmetric field, capped inside
    ``delta`` to avoid the singularity):

        potential(same-class pair)      = -1 / max(d, delta)^alpha    (attraction)
        potential(different-class pair) = +1 / max(d, delta)^alpha    (repulsion)

    So different-class points repel at ALL distances (decaying with separation) while
    same-class points attract down to the ``delta`` core. This keeps PFML's
    distance-decaying local structure (its label-noise-robust advantage — far,
    likely-mislabelled same-class samples exert little pull) but adds proper global
    separation. Optionally includes trainable class proxies as extra field sources.

    CRITICAL — balanced per-term means: with many classes ~99% of pairs are
    different-class, so a single mean over all pairs is repulsion-dominated and the
    classes never become compact (that variant also collapsed, R@1 ~0.04). Instead
    the attraction energy is averaged over same-class pairs only and the repulsion
    energy over different-class pairs only, then summed — the two forces are balanced
    1:1 regardless of pair counts, exactly how contrastive losses avoid this
    imbalance.
    """
    if proxy_embeddings is not None and proxy_labels is not None:
        points = torch_module.cat([embeddings, _normalize(proxy_embeddings, torch_module)], dim=0)
        point_labels = torch_module.cat([labels, proxy_labels], dim=0)
    else:
        points = embeddings
        point_labels = labels
    if points.shape[0] < 2:
        return embeddings.sum() * 0.0
    distances = torch_module.cdist(points, points, p=2).clamp_min(1.0e-4)
    capped = distances.clamp_min(float(delta))  # inner cutoff removes the singularity
    magnitude = capped.pow(-float(alpha))  # 1 / max(d, delta)^alpha, long-range decay
    same_label = point_labels[:, None].eq(point_labels[None, :])
    off_diagonal = ~torch_module.eye(points.shape[0], dtype=torch_module.bool, device=points.device)
    valid_positive = same_label & off_diagonal
    valid_negative = (~same_label) & off_diagonal
    terms = []
    if bool(valid_positive.any()):
        terms.append(-magnitude[valid_positive].mean())  # attraction, per same-class pair
    if bool(valid_negative.any()):
        terms.append(magnitude[valid_negative].mean())  # repulsion, per different-class pair
    if not terms:
        return embeddings.sum() * 0.0
    return torch_module.stack(terms).sum()


def _lennard_jones_loss(
    embeddings: Any,
    labels: Any,
    *,
    proxy_embeddings: Any | None,
    proxy_labels: Any | None,
    sigma: float,
    power: float,
    repulsion_weight: float,
    sigma_neg: float | None = None,
    torch_module: Any,
) -> Any:
    """Lennard-Jones metric-learning energy — a novel MOLECULAR-DYNAMICS physics.

    PFML models classes as electrostatic charges (monotone attraction/repulsion),
    which on the unit sphere collapses because attraction has no lower length scale.
    The Lennard-Jones (van der Waals) potential — the physics of how atoms pack —
    instead has a built-in EQUILIBRIUM distance ``sigma``:

        V_LJ(d) = (sigma/d)^(2p) - 2 (sigma/d)^p     (min = -1 at d = sigma)

    The steep ``(sigma/d)^(2p)`` repulsive core makes it energetically impossible for
    two points to coincide, while the ``(sigma/d)^p`` attractive tail pulls distant
    same-class points in. So same-class members settle into a compact shell of radius
    ~``sigma`` that CANNOT collapse to a point — structurally fixing the failure mode
    that sank both faithful PFML and the Symmetric Potential Field.

    - Same-class pairs feel the full LJ potential (attract to the ``sigma`` shell,
      repel below it).
    - Different-class pairs feel only the repulsive core ``(sigma/d)^(2p)`` (a soft
      excluded-volume shell that keeps classes from overlapping).

    TWO LENGTH SCALES (like atoms of different radii): the same-class bonding
    distance ``sigma`` sets how compact each class shell is, while the different-class
    exclusion radius ``sigma_neg`` (defaults to ``sigma``) sets how far apart classes
    are pushed. Decoupling them lets classes be compact (small ``sigma``) AND
    well-separated (large ``sigma_neg`` extends the repulsive core's range to cover
    more different-class pairs) — the intra/inter tension a single scale cannot
    resolve.

    Balanced per-term means (attraction over same-class pairs, repulsion over
    different-class pairs) keep the two forces commensurate regardless of pair
    counts. Optionally includes class proxies as extra particles. ``power=2`` is a
    gentler LJ than the physical (12,6); the ``sigma/4`` distance floor bounds the
    core numerically.
    """
    negative_sigma = float(sigma if sigma_neg is None else sigma_neg)
    if proxy_embeddings is not None and proxy_labels is not None:
        points = torch_module.cat([embeddings, _normalize(proxy_embeddings, torch_module)], dim=0)
        point_labels = torch_module.cat([labels, proxy_labels], dim=0)
    else:
        points = embeddings
        point_labels = labels
    if points.shape[0] < 2:
        return embeddings.sum() * 0.0
    floor = min(float(sigma), negative_sigma) * 0.25
    distances = torch_module.cdist(points, points, p=2).clamp_min(floor)
    same_label = point_labels[:, None].eq(point_labels[None, :])
    off_diagonal = ~torch_module.eye(points.shape[0], dtype=torch_module.bool, device=points.device)
    valid_positive = same_label & off_diagonal
    valid_negative = (~same_label) & off_diagonal
    terms = []
    if bool(valid_positive.any()):
        ratio_pos = float(sigma) / distances[valid_positive]
        lennard_jones = ratio_pos.pow(2.0 * float(power)) - 2.0 * ratio_pos.pow(float(power))
        terms.append(lennard_jones.mean())  # same-class: full LJ well at sigma
    if bool(valid_negative.any()):
        ratio_neg = negative_sigma / distances[valid_negative]
        core_neg = ratio_neg.pow(2.0 * float(power))  # repulsive core at sigma_neg
        terms.append(float(repulsion_weight) * core_neg.mean())  # diff-class: exclusion shell
    if not terms:
        return embeddings.sum() * 0.0
    return torch_module.stack(terms).sum()


def _build_hist_module(*, nb_classes: int, sz_embed: int, hidden: int, torch_module: Any) -> Any:
    """Build HIST's two trainable modules (CVPR 2022, faithful to ljin0429/HIST).

    HIST models each class as a learnable diagonal Gaussian and builds a hypergraph
    over the batch (soft class-membership incidence), then a Hypergraph Neural Network
    propagates higher-order class relations to produce per-sample class logits. The
    "semantic tuplet" supervision is the cross-entropy on those propagated logits, plus
    a distribution (Mahalanobis softmax) loss. Defined inside a factory so torch is only
    required at call time. Returns an ``nn.Module`` holding means, log_vars, and the HGNN.
    """
    nn = torch_module.nn

    class _HistNet(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.means = nn.Parameter(torch_module.empty(nb_classes, sz_embed))
            self.log_vars = nn.Parameter(torch_module.empty(nb_classes, sz_embed))
            nn.init.kaiming_normal_(self.means, mode="fan_out")
            nn.init.kaiming_normal_(self.log_vars, mode="fan_out")
            self.theta1 = nn.Linear(sz_embed, hidden)
            self.bn1 = nn.BatchNorm1d(hidden)
            self.lrelu = nn.LeakyReLU(0.1)
            self.theta2 = nn.Linear(hidden, nb_classes)

    return _HistNet()


def _hist_loss(
    embeddings: Any,
    labels: Any,
    *,
    hist_module: Any,
    label_to_index: dict[int, int],
    tau: float,
    alpha: float,
    lambda_s: float,
    var_floor: float,
    torch_module: Any,
) -> Any:
    """HIST total loss = distribution loss + lambda_s * hypergraph-CE (faithful port).

    ``var_floor`` is the lower clamp on log-variances; 0.0 reproduces the original
    ``relu6`` (variance >= 1), a negative value lets classes tighten below it.
    """
    functional = torch_module.nn.functional
    device = embeddings.device
    target = torch_module.tensor(
        [label_to_index[int(label)] for label in labels.tolist()],
        dtype=torch_module.long,
        device=device,
    )
    nb_classes = int(hist_module.means.shape[0])
    features = functional.normalize(embeddings, p=2, dim=-1)
    means = functional.normalize(hist_module.means, p=2, dim=-1)
    # relu6 == clamp(0, 6); var_floor generalises the lower bound (0.0 = faithful).
    log_vars = hist_module.log_vars.clamp(float(var_floor), 6.0)
    covariances = torch_module.exp(log_vars).unsqueeze(0)  # (1, C, F)
    diff = features.unsqueeze(1) - means.unsqueeze(0)  # (N, C, F)
    distance = (diff.pow(2) / covariances).sum(dim=-1)  # (N, C) squared Mahalanobis

    one_hot = functional.one_hot(target, nb_classes).to(features.dtype)  # (N, C)
    # Distribution loss: cross-entropy between softmax(-tau * distance) and the true
    # class. Use `cross_entropy` (log-softmax internally) rather than log of a gathered
    # softmax probability, so an underflowing probability cannot make the masked mean
    # empty and produce a NaN — numerically identical when nothing underflows.
    dist_loss = functional.cross_entropy(-float(tau) * distance, target)

    # Soft hypergraph incidence H over the classes present in the batch.
    class_within = torch_module.nonzero(one_hot.sum(dim=0) != 0, as_tuple=False).squeeze(dim=1)
    exp_term = torch_module.exp(-float(alpha) * distance[:, class_within])
    incidence = one_hot[:, class_within] + exp_term * (1.0 - one_hot[:, class_within])  # (N, E)

    # HGNN propagation matrix G = Dv^-1/2 H We De^-1 H^T Dv^-1/2.
    edge_weight = torch_module.ones(incidence.shape[1], device=device, dtype=features.dtype)
    node_degree = (incidence * edge_weight).sum(dim=1).clamp_min(1.0e-12)
    edge_degree = incidence.sum(dim=0).clamp_min(1.0e-12)
    inv_node = torch_module.diag(node_degree.pow(-0.5))
    inv_edge = torch_module.diag(edge_degree.pow(-1.0))
    weight = torch_module.diag(edge_weight)
    propagate = inv_node @ incidence @ weight @ inv_edge @ incidence.T @ inv_node  # (N, N)

    hidden_state = propagate @ hist_module.theta1(features)
    hidden_state = hist_module.lrelu(hist_module.bn1(hidden_state))
    logits = propagate @ hist_module.theta2(hidden_state)  # (N, C)
    ce_loss = functional.cross_entropy(logits, target)
    return dist_loss + float(lambda_s) * ce_loss


def _coding_rate(features: Any, *, eps: float, torch_module: Any) -> Any:
    """MCR2 coding rate R(Z, eps) = 1/2 log det(I + (d / (n eps^2)) Z Z^T).

    The information-theoretic log-volume of a set of L2-normalised vectors Z (n x d):
    large when they span many directions (spread), near zero when they collapse onto
    a line/point. This is the properly-normalised anti-collapse quantity that raw
    pairwise potentials (PFML/LJ) lack — maximising it forbids the embedding
    contraction that made those objectives collapse (Anti-Collapse Loss, arXiv
    2407.03106, from Maximal Coding Rate Reduction, arXiv 2006.08558).
    """
    n = features.shape[0]
    d = features.shape[1]
    if n < 2:
        return features.sum() * 0.0
    scale = float(d) / (float(n) * float(eps) ** 2)
    gram = features @ features.T  # (n, n) Gram / similarity matrix
    identity = torch_module.eye(n, dtype=features.dtype, device=features.device)
    _, logabsdet = torch_module.linalg.slogdet(identity + scale * gram)  # SPD -> sign +1
    return 0.5 * logabsdet


def _bio_physical_bond_loss(
    embeddings: Any,
    labels: Any,
    *,
    proxy_embeddings: Any | None,
    proxy_labels: Any | None,
    alpha: float,
    delta: float,
    sigma: float,
    power: float,
    niche_weight: float,
    antico_eps: float,
    torch_module: Any,
) -> Any:
    """A novel objective fusing physics + chemistry + biology into one loss.

    Three domains, each supplying the piece the others lack:

    - PHYSICS (statistical mechanics): the whole loss is a Boltzmann free energy — a
      per-anchor softmax/LogSumExp (Proxy Anchor form) over the interactions. This
      partition-function normalisation is the stabiliser that raw pairwise potentials
      (PFML, Lennard-Jones) empirically lacked, and is what makes the softmax-normalised
      DML losses (Proxy Anchor, SupCon) work.
    - CHEMISTRY (molecular bonding): the sample<->proxy affinity is a Lennard-Jones bond
      strength ``s = -V_LJ(d)`` that PEAKS at an equilibrium bond distance ``sigma``
      (not at coincidence). Pulling a class toward a ``sigma``-shell around its proxy —
      rather than onto the proxy — gives each class a bounded volume that cannot
      collapse to a point.
    - BIOLOGY (ecological niche / competitive exclusion): a coding-rate term maximises
      the volume the class proxies occupy in embedding "habitat", pushing classes into
      disjoint niches (``niche_weight`` * the MCR2 log-det on the proxies).

    So: Proxy-Anchor Boltzmann softmax (physics) over Lennard-Jones bond affinities
    (chemistry) plus a coding-rate niche term (biology).
    """
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError(
            "the bio_physical_bond objective requires class proxies (proxy_count_per_class > 0)"
        )
    normalized_embeddings = _normalize(embeddings, torch_module)
    normalized_proxies = _normalize(proxy_embeddings, torch_module)
    distances = torch_module.cdist(normalized_embeddings, normalized_proxies, p=2).clamp_min(
        float(sigma) * 0.25
    )
    ratio = float(sigma) / distances
    v_lj = ratio.pow(2.0 * float(power)) - 2.0 * ratio.pow(float(power))  # min -1 at d=sigma
    affinity = -v_lj  # bond strength: +1 at the equilibrium shell, repulsive if too close

    positive_mask = labels[:, None].eq(proxy_labels[None, :])
    positive_term = embeddings.sum() * 0.0
    with_positive = positive_mask.any(dim=0)
    if bool(with_positive.any()):
        pos_logits = (-float(alpha) * (affinity - float(delta))).masked_fill(~positive_mask, -1.0e9)
        positive_term = torch_module.nn.functional.softplus(
            torch_module.logsumexp(pos_logits[:, with_positive], dim=0)
        ).mean()
    negative_mask = ~positive_mask
    negative_term = embeddings.sum() * 0.0
    with_negative = negative_mask.any(dim=0)
    if bool(with_negative.any()):
        neg_logits = (float(alpha) * (affinity + float(delta))).masked_fill(~negative_mask, -1.0e9)
        negative_term = torch_module.nn.functional.softplus(
            torch_module.logsumexp(neg_logits[:, with_negative], dim=0)
        ).mean()
    loss = positive_term + negative_term
    if float(niche_weight) > 0.0:
        loss = loss - float(niche_weight) * _coding_rate(
            normalized_proxies, eps=antico_eps, torch_module=torch_module
        )
    return loss


def _lennard_jones_intra_term(
    embeddings: Any, labels: Any, *, sigma: float, power: float, torch_module: Any
) -> Any:
    """Same-class Lennard-Jones well only — a novel intra-class physics regulariser.

    Proxy losses (Proxy Anchor) give strong INTER-class structure but only pull each
    sample toward its proxy, so a class can collapse to a point (zero intra-class
    volume, which hurts MAP@R / transfer). Adding the same-class LJ well makes members
    settle at a natural equilibrium separation ``sigma`` — the repulsive core forbids
    collapse, the attractive tail forbids dispersion — giving each class a healthy,
    bounded volume. A physics-principled intra-class term to layer on a strong base.
    """
    normalized = _normalize(embeddings, torch_module)
    distances = torch_module.cdist(normalized, normalized, p=2).clamp_min(float(sigma) * 0.25)
    same_label = labels[:, None].eq(labels[None, :])
    off_diagonal = ~torch_module.eye(
        normalized.shape[0], dtype=torch_module.bool, device=normalized.device
    )
    valid_positive = same_label & off_diagonal
    if not bool(valid_positive.any()):
        return embeddings.sum() * 0.0
    ratio = float(sigma) / distances[valid_positive]
    well = ratio.pow(2.0 * float(power)) - 2.0 * ratio.pow(float(power))
    return well.mean()


def _proxy_anchor_loss(
    embeddings: Any,
    labels: Any,
    *,
    proxy_embeddings: Any | None,
    proxy_labels: Any | None,
    alpha: float,
    delta: float,
    torch_module: Any,
) -> Any:
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError(
            "the proxy_anchor objective requires class proxies (proxy_count_per_class > 0)"
        )
    proxies = _normalize(proxy_embeddings, torch_module)
    normalized_embeddings = _normalize(embeddings, torch_module)
    similarities = normalized_embeddings @ proxies.T
    positive_mask = labels[:, None].eq(proxy_labels[None, :])

    positive_term = embeddings.sum() * 0.0
    with_positive = positive_mask.any(dim=0)
    if bool(with_positive.any()):
        pos_logits = (-float(alpha) * (similarities - float(delta))).masked_fill(
            ~positive_mask,
            -1.0e9,
        )
        positive_term = torch_module.nn.functional.softplus(
            torch_module.logsumexp(pos_logits[:, with_positive], dim=0)
        ).mean()

    negative_term = embeddings.sum() * 0.0
    negative_mask = ~positive_mask
    with_negative = negative_mask.any(dim=0)
    if bool(with_negative.any()):
        neg_logits = (float(alpha) * (similarities + float(delta))).masked_fill(
            ~negative_mask,
            -1.0e9,
        )
        negative_term = torch_module.nn.functional.softplus(
            torch_module.logsumexp(neg_logits[:, with_negative], dim=0)
        ).mean()
    return positive_term + negative_term


def _subcenter_proxy_anchor_loss(
    embeddings: Any,
    labels: Any,
    *,
    proxy_embeddings: Any | None,
    proxy_labels: Any | None,
    alpha: float,
    delta: float,
    gamma: float,
    torch_module: Any,
) -> Any:
    """Sub-center Proxy Anchor: K proxies per class, each sample softly assigned to
    its NEAREST sub-center within its class (SoftTriple-style intra-class softmax),
    then the standard Proxy-Anchor LogSumExp over samples on the resulting per-class
    similarities. This lets one class occupy several modes (e.g. a bird species in
    different poses/colours) without the sub-centers collapsing to a single mean.
    """
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError(
            "the proxy_anchor_subcenter objective requires class proxies "
            "(proxy_count_per_class > 1)"
        )
    proxies = _normalize(proxy_embeddings, torch_module)
    normalized_embeddings = _normalize(embeddings, torch_module)
    sims = normalized_embeddings @ proxies.T  # (N, C*K)

    unique_classes = sorted({int(label) for label in proxy_labels.tolist()})
    num_classes = len(unique_classes)
    total_proxies = int(proxies.shape[0])
    if num_classes == 0 or total_proxies % num_classes != 0 or total_proxies == num_classes:
        # Non-uniform K or a single proxy per class -> plain Proxy Anchor.
        return _proxy_anchor_loss(
            embeddings,
            labels,
            proxy_embeddings=proxy_embeddings,
            proxy_labels=proxy_labels,
            alpha=alpha,
            delta=delta,
            torch_module=torch_module,
        )
    per_class = total_proxies // num_classes
    # Proxies are stored class-contiguous with `per_class` each (see
    # _attach_metric_proxies), so this reshape groups sub-centers by class.
    sims_by_center = sims.reshape(sims.shape[0], num_classes, per_class)  # (N, C, K)
    assignment = torch_module.nn.functional.softmax(sims_by_center / float(gamma), dim=2)
    class_sims = (assignment * sims_by_center).sum(dim=2)  # (N, C) soft nearest-center sim

    class_ids = torch_module.tensor(unique_classes, device=class_sims.device, dtype=labels.dtype)
    positive_mask = labels[:, None].eq(class_ids[None, :])  # (N, C)

    positive_term = embeddings.sum() * 0.0
    with_positive = positive_mask.any(dim=0)
    if bool(with_positive.any()):
        pos_logits = (-float(alpha) * (class_sims - float(delta))).masked_fill(
            ~positive_mask, -1.0e9
        )
        positive_term = torch_module.nn.functional.softplus(
            torch_module.logsumexp(pos_logits[:, with_positive], dim=0)
        ).mean()

    negative_term = embeddings.sum() * 0.0
    negative_mask = ~positive_mask
    with_negative = negative_mask.any(dim=0)
    if bool(with_negative.any()):
        neg_logits = (float(alpha) * (class_sims + float(delta))).masked_fill(
            ~negative_mask, -1.0e9
        )
        negative_term = torch_module.nn.functional.softplus(
            torch_module.logsumexp(neg_logits[:, with_negative], dim=0)
        ).mean()
    return positive_term + negative_term


def _gaussian_potential_uniformity_loss(
    embeddings: Any,
    *,
    t: float,
    torch_module: Any,
) -> Any:
    """Thermodynamic uniformity: treat the batch as a gas of unit charges on the
    hypersphere interacting via a Gaussian (RBF) potential exp(-t * ||z_i - z_j||^2),
    and minimise the log of the mean pairwise Boltzmann factor (Wang-Isola uniformity).

    The log-partition-function normalisation is exactly the stabiliser that raw
    physics potentials (PFML electrostatics, the symmetric long-range potential)
    lacked: it cannot collapse, because a collapsed batch drives every pairwise
    distance to zero and the log-mean-exp term to its MAXIMUM (worst) value, so the
    gradient always pushes embeddings apart. Added on top of an alignment term
    (Proxy Anchor) this realises the classic alignment+uniformity decomposition.
    """
    normalized = _normalize(embeddings, torch_module)
    if int(normalized.shape[0]) < 2:
        return embeddings.sum() * 0.0
    squared_distances = torch_module.pdist(normalized, p=2).pow(2)
    return torch_module.logsumexp(-float(t) * squared_distances, dim=0) - math.log(
        float(squared_distances.shape[0])
    )


def _relational_distillation_loss(
    student_embeddings: Any,
    teacher_embeddings: Any,
    *,
    tau: float,
    torch_module: Any,
) -> Any:
    """Relational self-distillation: match the student's batch neighborhood
    distribution to a slow EMA-teacher's. For each anchor i the teacher defines a
    softmax distribution over the OTHER batch samples (row-wise, diagonal masked);
    the student is trained to reproduce it via cross-entropy.

    Unlike hard proxy/class labels, these soft neighborhood targets encode graded
    inter-sample structure the teacher has already learned; distilling that relational
    structure (rather than the label) transfers to UNSEEN classes at test time -- the
    exact quantity zero-shot retrieval measures. The softmax normalisation makes it a
    Boltzmann distribution match, so it cannot collapse (a collapsed batch yields a
    uniform teacher target that the base loss already fights).
    """
    zs = _normalize(student_embeddings, torch_module)
    zt = _normalize(teacher_embeddings, torch_module)
    n = int(zs.shape[0])
    if n < 2:
        return zs.sum() * 0.0
    eye = torch_module.eye(n, device=zs.device, dtype=torch_module.bool)
    student_logits = (zs @ zs.T / float(tau)).masked_fill(eye, -1.0e9)
    teacher_logits = (zt @ zt.T / float(tau)).masked_fill(eye, -1.0e9)
    teacher_probs = torch_module.nn.functional.softmax(teacher_logits, dim=1)
    student_log_probs = torch_module.nn.functional.log_softmax(student_logits, dim=1)
    return -(teacher_probs * student_log_probs).sum(dim=1).mean()


def _mead_assignment_distillation_loss(
    student_views: list[torch.Tensor],
    teacher_globals: torch.Tensor,
    prototypes: torch.Tensor,
    center: torch.Tensor,
    *,
    tau_teacher: float,
    tau_student: float,
    torch_module: types.ModuleType,
) -> torch.Tensor:
    """Cross-view MEAD assignment distillation against EMA-teacher class prototypes."""
    torch_any = cast(Any, torch_module)
    batch_size = int(teacher_globals.shape[0]) // 2
    if batch_size <= 0 or int(prototypes.shape[0]) == 0:
        zero = teacher_globals.sum() * 0.0
        for view in student_views:
            zero = zero + view.sum() * 0.0
        return zero

    prototype_targets = prototypes.detach()
    teacher_logits = (teacher_globals.detach() @ prototype_targets.T - center.detach()) / float(
        tau_teacher
    )
    teacher_probs = torch_any.nn.functional.softmax(teacher_logits, dim=1).detach()
    teacher_views = [teacher_probs[:batch_size], teacher_probs[batch_size : 2 * batch_size]]
    losses: list[torch.Tensor] = []
    for teacher_index, teacher_assignments in enumerate(teacher_views):
        for student_index, student_view in enumerate(student_views):
            if student_index == teacher_index:
                continue
            student_logits = (student_view @ prototype_targets.T) / float(tau_student)
            student_log_probs = torch_any.nn.functional.log_softmax(student_logits, dim=1)
            losses.append(-(teacher_assignments * student_log_probs).sum(dim=1).mean())
    if not losses:
        zero = teacher_globals.sum() * 0.0
        for view in student_views:
            zero = zero + view.sum() * 0.0
        return zero
    return cast("torch.Tensor", torch_any.stack(losses).mean())


def _group_soft_class_similarity(
    normalized_embeddings: Any,
    normalized_proxies: Any,
    proxy_labels: Any,
    *,
    class_label: int,
    tau_assign: float,
    torch_module: Any,
) -> tuple[Any, Any]:
    """Soft-nearest similarity of each sample to one class's group of proxies.

    Each class owns a group of M proxies. A sample's similarity to the class is a
    softmax(similarity / tau_assign)-weighted average over that class's proxies, so
    each proxy specialises to an intra-class mode (SoftTriple-style) instead of every
    same-class proxy pulling every same-class sample equally. Returns
    ``(effective_similarity[B], assignment[B, M])``; gradients flow through both.
    """
    member_mask = proxy_labels == int(class_label)
    class_proxies = normalized_proxies[member_mask]  # (M, d)
    class_similarities = normalized_embeddings @ class_proxies.T  # (B, M)
    assignment = torch_module.softmax(class_similarities / float(tau_assign), dim=1)
    effective = (assignment * class_similarities).sum(dim=1)  # (B,)
    return effective, assignment


def _proxy_anchor_group_loss(
    embeddings: Any,
    labels: Any,
    *,
    proxy_embeddings: Any | None,
    proxy_labels: Any | None,
    alpha: float,
    delta: float,
    tau_assign: float,
    torch_module: Any,
) -> Any:
    """Proxy Anchor over per-class proxy *groups* with soft-nearest assignment.

    Builds an effective (B, num_classes) similarity where each class column is the
    soft-nearest similarity to that class's group of proxies, then applies the exact
    Proxy Anchor hardness-weighted LogSumExp on that matrix. Reduces bit-for-bit to
    ``_proxy_anchor_loss`` when there is one proxy per class.
    """
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError(
            "the proxy_anchor_group objective requires class proxies (proxy_count_per_class > 0)"
        )
    normalized_embeddings = _normalize(embeddings, torch_module)
    normalized_proxies = _normalize(proxy_embeddings, torch_module)
    class_labels = torch_module.unique(proxy_labels)
    columns = []
    for class_label in class_labels.tolist():
        effective, _ = _group_soft_class_similarity(
            normalized_embeddings,
            normalized_proxies,
            proxy_labels,
            class_label=int(class_label),
            tau_assign=tau_assign,
            torch_module=torch_module,
        )
        columns.append(effective)
    similarities = torch_module.stack(columns, dim=1)  # (B, num_classes)
    positive_mask = labels[:, None].eq(class_labels[None, :])  # (B, num_classes)

    positive_term = embeddings.sum() * 0.0
    with_positive = positive_mask.any(dim=0)
    if bool(with_positive.any()):
        pos_logits = (-float(alpha) * (similarities - float(delta))).masked_fill(
            ~positive_mask,
            -1.0e9,
        )
        positive_term = torch_module.nn.functional.softplus(
            torch_module.logsumexp(pos_logits[:, with_positive], dim=0)
        ).mean()

    negative_term = embeddings.sum() * 0.0
    negative_mask = ~positive_mask
    with_negative = negative_mask.any(dim=0)
    if bool(with_negative.any()):
        neg_logits = (float(alpha) * (similarities + float(delta))).masked_fill(
            ~negative_mask,
            -1.0e9,
        )
        negative_term = torch_module.nn.functional.softplus(
            torch_module.logsumexp(neg_logits[:, with_negative], dim=0)
        ).mean()
    return positive_term + negative_term


def _class_representative_proxies(
    normalized_proxies: Any, proxy_labels: Any, torch_module: Any
) -> tuple[Any, Any]:
    """Collapse M proxies per class to one L2-normalized mean proxy per class."""
    class_labels = torch_module.unique(proxy_labels)
    rows = [
        normalized_proxies[proxy_labels == int(label)].mean(dim=0)
        for label in class_labels.tolist()
    ]
    return _normalize(torch_module.stack(rows, dim=0), torch_module), class_labels


def _sample_synthesis_class_pairs(
    present: Sequence[int],
    present_proxies: Any,
    *,
    mode: str,
    temperature: float,
    generator: Any | None,
    torch_module: Any,
) -> tuple[int, int]:
    """Pick a pair of present classes to synthesise a virtual class between.

    ``mode="random"`` draws a uniform distinct pair (vanilla Proxy Synthesis).
    ``mode="confusable"`` (novel: Confusion-Guided Proxy Synthesis) draws a distinct
    pair with probability proportional to ``softmax(cos(proxy_i, proxy_j) / temperature)``
    over unordered pairs, so virtual classes densify the decision boundaries between
    the classes that are hardest to tell apart — where boundary smoothing most helps
    zero-shot transfer. The confusion axes GSI could only measure (not exploit) here
    steer a mechanism that actually moves the metric.
    """
    count = len(present)
    if mode != "confusable" or count <= 2:
        pair = torch_module.randperm(count, generator=generator, device=present_proxies.device)[:2]
        return present[int(pair[0])], present[int(pair[1])]
    with torch_module.no_grad():
        similarities = present_proxies @ present_proxies.T  # (C, C)
        triu = torch_module.triu_indices(count, count, offset=1, device=present_proxies.device)
        pair_scores = similarities[triu[0], triu[1]]  # (num_pairs,)
        weights = torch_module.softmax(pair_scores / float(temperature), dim=0)
        chosen = int(torch_module.multinomial(weights, num_samples=1, generator=generator).item())
    return present[int(triu[0][chosen])], present[int(triu[1][chosen])]


def _proxy_synthesis_proxy_anchor_loss(
    embeddings: Any,
    labels: Any,
    *,
    proxy_embeddings: Any | None,
    proxy_labels: Any | None,
    alpha: float,
    delta: float,
    ratio: float,
    beta_alpha: float,
    generator: Any | None,
    group_mix: bool = False,
    pair_selection: str = "random",
    pair_temperature: float = 0.1,
    torch_module: Any,
) -> Any:
    """Proxy Anchor with Proxy-Synthesis virtual-class augmentation.

    Following Proxy Synthesis (Gu et al., AAAI 2021): each step synthesises virtual
    classes by convex-combining pairs of real class representatives (proxies) and
    real embeddings with a shared Beta-sampled coefficient, then runs Proxy Anchor
    on the real + virtual set. The virtual classes densify the embedding space
    between real classes, forcing smoother boundaries that transfer better to unseen
    classes. Virtual proxies/embeddings are differentiable mixtures, so gradients
    flow back to the real proxies and backbone.

    Novel twist (``group_mix=True``): a virtual class is generated from the mix of
    the two source classes' GROUP MEANS (set representatives) instead of individual
    embedding pairs — a sfora-native form of synthesis.

    Novel twist (``pair_selection="confusable"``): Confusion-Guided Proxy Synthesis —
    source class pairs are drawn toward the most confusable (nearest-proxy) pairs
    rather than uniformly, so virtual classes densify the hardest decision boundaries.
    Reduces exactly to plain Proxy Anchor over per-class mean proxies when ``ratio == 0``.
    """
    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError(
            "the proxy_anchor_synthesis objective requires class proxies "
            "(proxy_count_per_class > 0)"
        )
    normalized_embeddings = _normalize(embeddings, torch_module)
    normalized_proxies = _normalize(proxy_embeddings, torch_module)
    class_proxy, class_labels = _class_representative_proxies(
        normalized_proxies, proxy_labels, torch_module
    )

    all_emb = normalized_embeddings
    all_lab = labels
    all_prox = class_proxy
    all_prox_lab = class_labels

    present = [int(label) for label in torch_module.unique(labels).tolist()]
    if ratio > 0.0 and len(present) >= 2:
        n_virtual = max(1, int(math.ceil(ratio * len(present))))
        proxy_row = {int(label): index for index, label in enumerate(class_labels.tolist())}
        emb_index = {
            label: torch_module.nonzero(labels == label, as_tuple=False).flatten()
            for label in present
        }
        base_label = int(max(int(class_labels.max()), int(labels.max()))) + 1
        present_proxies = class_proxy[[proxy_row[label] for label in present]]
        virtual_emb_blocks: list[Any] = []
        virtual_emb_labels: list[int] = []
        virtual_prox_rows: list[Any] = []
        virtual_prox_labels: list[int] = []
        for offset in range(n_virtual):
            class_i, class_j = _sample_synthesis_class_pairs(
                present,
                present_proxies,
                mode=pair_selection,
                temperature=pair_temperature,
                generator=generator,
                torch_module=torch_module,
            )
            lam = float(
                torch_module.distributions.Beta(float(beta_alpha), float(beta_alpha)).sample()
            )
            virtual_label = base_label + offset
            proxy_i = class_proxy[proxy_row[class_i]]
            proxy_j = class_proxy[proxy_row[class_j]]
            virtual_prox_rows.append(
                _normalize(lam * proxy_i + (1.0 - lam) * proxy_j, torch_module)
            )
            virtual_prox_labels.append(virtual_label)
            idx_i = emb_index[class_i]
            idx_j = emb_index[class_j]
            if group_mix:
                mean_i = normalized_embeddings[idx_i].mean(dim=0)
                mean_j = normalized_embeddings[idx_j].mean(dim=0)
                mixed = _normalize(lam * mean_i + (1.0 - lam) * mean_j, torch_module)
                virtual_emb_blocks.append(mixed.unsqueeze(0))
                virtual_emb_labels.append(virtual_label)
            else:
                count = int(min(idx_i.shape[0], idx_j.shape[0]))
                mixed = _normalize(
                    lam * normalized_embeddings[idx_i[:count]]
                    + (1.0 - lam) * normalized_embeddings[idx_j[:count]],
                    torch_module,
                )
                virtual_emb_blocks.append(mixed)
                virtual_emb_labels.extend([virtual_label] * count)

        virtual_proxies = torch_module.stack(virtual_prox_rows, dim=0)
        virtual_proxy_labels = torch_module.tensor(
            virtual_prox_labels, dtype=labels.dtype, device=labels.device
        )
        virtual_embeddings = torch_module.cat(virtual_emb_blocks, dim=0)
        virtual_embedding_labels = torch_module.tensor(
            virtual_emb_labels, dtype=labels.dtype, device=labels.device
        )
        all_emb = torch_module.cat([normalized_embeddings, virtual_embeddings], dim=0)
        all_lab = torch_module.cat([labels, virtual_embedding_labels], dim=0)
        all_prox = torch_module.cat([class_proxy, virtual_proxies], dim=0)
        all_prox_lab = torch_module.cat([class_labels, virtual_proxy_labels], dim=0)

    return _proxy_anchor_loss(
        all_emb,
        all_lab,
        proxy_embeddings=all_prox,
        proxy_labels=all_prox_lab,
        alpha=alpha,
        delta=delta,
        torch_module=torch_module,
    )


def _confusion_axes(
    proxy_embeddings: Any,
    proxy_labels: Any,
    *,
    top_k: int,
    torch_module: Any,
) -> dict[int, tuple[Any, Any]]:
    """Per-class confusion axes toward the nearest proxies of confusable classes.

    For each class ``c`` the top-k most confusable classes ``c'`` are ranked by
    the maximum cosine similarity between any proxy of ``c`` and any proxy of
    ``c'``. Each axis is the unit direction from the nearest proxy of ``c`` to
    the nearest proxy of ``c'``; the per-axis weights are the softmax of the
    proxy cosine similarities (hardest confuser weighted most). Everything is
    computed under ``no_grad`` so GSI never propagates gradients into proxies.
    """
    axes_by_class: dict[int, tuple[Any, Any]] = {}
    with torch_module.no_grad():
        proxies = _normalize(proxy_embeddings, torch_module)
        unique_labels = [int(label) for label in torch_module.unique(proxy_labels).tolist()]
        confuser_count = min(top_k, len(unique_labels) - 1)
        if confuser_count <= 0:
            return axes_by_class
        similarities = proxies @ proxies.T
        indices_by_label = {
            label: torch_module.nonzero(proxy_labels == label, as_tuple=False).flatten()
            for label in unique_labels
        }
        for label in unique_labels:
            own_indices = indices_by_label[label]
            scores = []
            axes = []
            for other_label in unique_labels:
                if other_label == label:
                    continue
                other_indices = indices_by_label[other_label]
                cross = similarities[own_indices][:, other_indices]
                flat_index = int(torch_module.argmax(cross))
                own_best = own_indices[flat_index // cross.shape[1]]
                other_best = other_indices[flat_index % cross.shape[1]]
                scores.append(cross.reshape(-1)[flat_index])
                axes.append(_normalize(proxies[other_best] - proxies[own_best], torch_module))
            top = torch_module.topk(torch_module.stack(scores), k=confuser_count)
            top_axes = torch_module.stack([axes[int(index)] for index in top.indices])
            weights = torch_module.softmax(top.values, dim=0)
            axes_by_class[label] = (top_axes, weights)
    return axes_by_class


def _boundary_confusion_axes(
    embeddings: Any,
    labels: Any,
    *,
    top_k: int,
    temperature: float,
    torch_module: Any,
) -> dict[int, tuple[Any, Any]]:
    """Per-class boundary axes from batch class means toward confusable means."""
    axes_by_class: dict[int, tuple[Any, Any]] = {}
    with torch_module.no_grad():
        unique_labels = [int(label) for label in torch_module.unique(labels).tolist()]
        confuser_count = min(top_k, len(unique_labels) - 1)
        if confuser_count <= 0:
            return axes_by_class

        means = []
        for label in unique_labels:
            means.append(embeddings[labels == label].mean(dim=0))
        mean_tensor = torch_module.stack(means)
        normalized_means = _normalize(mean_tensor, torch_module)
        similarities = normalized_means @ normalized_means.T

        for index, label in enumerate(unique_labels):
            scores = []
            axes = []
            own_mean = mean_tensor[index]
            for other_index, other_label in enumerate(unique_labels):
                if other_label == label:
                    continue
                axes.append(_normalize(mean_tensor[other_index] - own_mean, torch_module))
                scores.append(similarities[index, other_index])
            top = torch_module.topk(torch_module.stack(scores), k=confuser_count)
            top_axes = torch_module.stack([axes[int(axis_index)] for axis_index in top.indices])
            weights = torch_module.softmax(top.values / float(temperature), dim=0)
            axes_by_class[label] = (top_axes.detach(), weights.detach())
    return axes_by_class


def _uniform_axis_weights(count: int, *, reference: Any, torch_module: Any) -> Any:
    return torch_module.full(
        (count,),
        1.0 / count,
        dtype=reference.dtype,
        device=reference.device,
    )


def _ready_bgsi_labels(
    ema_state: BGSIClassMeanState,
    *,
    min_axis_observations: int,
) -> list[int]:
    ready: list[int] = []
    for label, row in ema_state.label_to_index.items():
        if int(ema_state.counts[row].item()) >= min_axis_observations:
            ready.append(label)
    return sorted(ready)


def _bgsi_ema_ready_fraction(
    labels: Any,
    *,
    bgsi_state: BGSIClassMeanState | None,
    min_axis_observations: int,
    torch_module: Any,
) -> float:
    batch_labels = [int(label) for label in torch_module.unique(labels).tolist()]
    if not batch_labels or bgsi_state is None:
        return 0.0
    ready = 0
    for label in batch_labels:
        row = bgsi_state.label_to_index.get(label)
        if row is not None and int(bgsi_state.counts[row].item()) >= min_axis_observations:
            ready += 1
    return float(ready / len(batch_labels))


def _bgsi_axis_step_diagnostics(
    labels: Any,
    *,
    axes_by_class: dict[int, tuple[Any, Any]],
    axis_mode: str,
    bgsi_state: BGSIClassMeanState | None,
    min_axis_observations: int,
    use_axis_agreement_gate: bool,
    torch_module: Any,
) -> dict[str, float]:
    batch_labels = [int(label) for label in torch_module.unique(labels).tolist()]
    eligible_class_count = max(1, len(batch_labels))
    axis_counts = [float(entry[0].shape[0]) for entry in axes_by_class.values()]
    diagnostics = {
        "bgsi_axis_coverage": float(len(axes_by_class) / eligible_class_count),
        "bgsi_axis_count": float(np.mean(axis_counts)) if axis_counts else 0.0,
        "bgsi_ema_ready_fraction": _bgsi_ema_ready_fraction(
            labels,
            bgsi_state=bgsi_state,
            min_axis_observations=min_axis_observations,
            torch_module=torch_module,
        ),
    }
    if axis_mode == "ema_boundary" and use_axis_agreement_gate:
        ready_count = max(1, int(round(diagnostics["bgsi_ema_ready_fraction"] * len(batch_labels))))
        diagnostics["bgsi_axis_agreement_fraction"] = float(len(axes_by_class) / ready_count)
    if axis_mode == "permuted":
        diagnostics["bgsi_permuted_match_fraction"] = 0.0
    return diagnostics


def _ema_boundary_confusion_axes(
    embeddings: Any,
    labels: Any,
    *,
    ema_state: BGSIClassMeanState | None,
    top_k: int,
    temperature: float,
    min_axis_observations: int,
    use_axis_agreement_gate: bool,
    axis_agreement: float,
    torch_module: Any,
) -> dict[int, tuple[Any, Any]]:
    if ema_state is None:
        return {}
    axes_by_class: dict[int, tuple[Any, Any]] = {}
    with torch_module.no_grad():
        ready_labels = _ready_bgsi_labels(
            ema_state,
            min_axis_observations=min_axis_observations,
        )
        confuser_count = min(top_k, len(ready_labels) - 1)
        if confuser_count <= 0:
            return axes_by_class
        batch_axes = _boundary_confusion_axes(
            embeddings,
            labels,
            top_k=1,
            temperature=temperature,
            torch_module=torch_module,
        )
        batch_labels = [int(label) for label in torch_module.unique(labels).tolist()]
        for label in batch_labels:
            row = ema_state.label_to_index.get(label)
            if row is None or int(ema_state.counts[row].item()) < min_axis_observations:
                continue
            own_mean = ema_state.means[row]
            scored: list[tuple[Any, int, Any]] = []
            for other_label in ready_labels:
                if other_label == label:
                    continue
                other_row = ema_state.label_to_index[other_label]
                other_mean = ema_state.means[other_row]
                axis = _normalize(other_mean - own_mean, torch_module)
                score = own_mean @ other_mean
                scored.append((score, other_label, axis))
            if not scored:
                continue
            score_tensor = torch_module.stack([item[0] for item in scored])
            top = torch_module.topk(score_tensor, k=min(confuser_count, len(scored)))
            chosen = [scored[int(index)] for index in top.indices]
            best_axis = chosen[0][2]
            best_label = chosen[0][1]
            if use_axis_agreement_gate and label in batch_axes:
                batch_axis = batch_axes[label][0][0]
                batch_scores = []
                own_batch_mean = embeddings[labels == label].mean(dim=0)
                normalized_own_batch = _normalize(own_batch_mean, torch_module)
                for other_label in batch_labels:
                    if other_label == label:
                        continue
                    other_batch_mean = embeddings[labels == other_label].mean(dim=0)
                    score = normalized_own_batch @ _normalize(other_batch_mean, torch_module)
                    batch_scores.append((score, other_label))
                batch_best_label = max(batch_scores, key=lambda item: float(item[0]))[1]
                agreement = float((batch_axis @ best_axis).detach().cpu())
                if batch_best_label != best_label and agreement < axis_agreement:
                    continue
            axes = torch_module.stack([item[2] for item in chosen]).detach()
            weights = torch_module.softmax(top.values / float(temperature), dim=0).detach()
            axes_by_class[label] = (axes, weights)
    return axes_by_class


def _random_bgsi_axes(
    embeddings: Any,
    labels: Any,
    *,
    top_k: int,
    generator: Any | None,
    torch_module: Any,
) -> dict[int, tuple[Any, Any]]:
    axes_by_class: dict[int, tuple[Any, Any]] = {}
    with torch_module.no_grad():
        unique_labels = [int(label) for label in torch_module.unique(labels).tolist()]
        axis_count = min(top_k, len(unique_labels) - 1)
        if axis_count <= 0:
            return axes_by_class
        weights = _uniform_axis_weights(axis_count, reference=embeddings, torch_module=torch_module)
        for label in unique_labels:
            raw = torch_module.randn(
                axis_count,
                embeddings.shape[1],
                generator=generator,
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            axes_by_class[label] = (_normalize(raw, torch_module).detach(), weights.detach())
    return axes_by_class


def _permuted_bgsi_axes(
    axes_by_class: dict[int, tuple[Any, Any]],
) -> dict[int, tuple[Any, Any]]:
    labels = sorted(axes_by_class)
    if len(labels) <= 1:
        return {}
    permuted: dict[int, tuple[Any, Any]] = {}
    for index, label in enumerate(labels):
        source_label = labels[(index + 1) % len(labels)]
        axes, weights = axes_by_class[source_label]
        permuted[label] = (axes.detach(), weights.detach())
    return permuted


def _global_bgsi_axes(
    embeddings: Any,
    labels: Any,
    *,
    ema_state: BGSIClassMeanState | None,
    top_k: int,
    min_axis_observations: int,
    torch_module: Any,
) -> dict[int, tuple[Any, Any]]:
    if ema_state is None:
        return {}
    axes_by_class: dict[int, tuple[Any, Any]] = {}
    with torch_module.no_grad():
        ready_labels = _ready_bgsi_labels(
            ema_state,
            min_axis_observations=min_axis_observations,
        )
        axis_count = min(top_k, len(ready_labels) - 1, embeddings.shape[1])
        if axis_count <= 0:
            return axes_by_class
        rows = [ema_state.label_to_index[label] for label in ready_labels]
        mean_matrix = ema_state.means[rows]
        centered = mean_matrix - mean_matrix.mean(dim=0, keepdim=True)
        _, _, vh = torch_module.linalg.svd(centered, full_matrices=False)
        axes = vh[:axis_count].detach()
        weights = _uniform_axis_weights(axis_count, reference=embeddings, torch_module=torch_module)
        for label in torch_module.unique(labels).tolist():
            label_int = int(label)
            row = ema_state.label_to_index.get(label_int)
            if row is None or int(ema_state.counts[row].item()) < min_axis_observations:
                continue
            axes_by_class[label_int] = (axes, weights.detach())
    return axes_by_class


def _bgsi_axes_for_mode(
    embeddings: Any,
    labels: Any,
    *,
    axis_mode: str,
    top_k: int,
    temperature: float,
    generator: Any | None,
    ema_state: BGSIClassMeanState | None,
    min_axis_observations: int,
    use_axis_agreement_gate: bool,
    axis_agreement: float,
    torch_module: Any,
) -> dict[int, tuple[Any, Any]]:
    if axis_mode == "batch_boundary":
        return _boundary_confusion_axes(
            embeddings,
            labels,
            top_k=top_k,
            temperature=temperature,
            torch_module=torch_module,
        )
    if axis_mode in {"ema_boundary", "permuted", "global"} and ema_state is None:
        return {}
    if axis_mode == "ema_boundary":
        return _ema_boundary_confusion_axes(
            embeddings,
            labels,
            ema_state=ema_state,
            top_k=top_k,
            temperature=temperature,
            min_axis_observations=min_axis_observations,
            use_axis_agreement_gate=use_axis_agreement_gate,
            axis_agreement=axis_agreement,
            torch_module=torch_module,
        )
    if axis_mode == "random":
        return _random_bgsi_axes(
            embeddings,
            labels,
            top_k=top_k,
            generator=generator,
            torch_module=torch_module,
        )
    if axis_mode == "permuted":
        base_axes = _ema_boundary_confusion_axes(
            embeddings,
            labels,
            ema_state=ema_state,
            top_k=top_k,
            temperature=temperature,
            min_axis_observations=min_axis_observations,
            use_axis_agreement_gate=False,
            axis_agreement=axis_agreement,
            torch_module=torch_module,
        )
        return _permuted_bgsi_axes(base_axes)
    if axis_mode == "global":
        return _global_bgsi_axes(
            embeddings,
            labels,
            ema_state=ema_state,
            top_k=top_k,
            min_axis_observations=min_axis_observations,
            torch_module=torch_module,
        )
    raise ValueError(f"unknown BGSI axis mode: {axis_mode}")


def _gsi_axes_for_mode(
    proxy_embeddings: Any,
    proxy_labels: Any,
    *,
    axis_mode: str,
    top_k: int,
    generator: Any | None,
    torch_module: Any,
) -> dict[int, tuple[Any, Any]]:
    """Resolve GSI axes per class for the configured axis mode.

    ``proxy`` uses per-class confusion axes; ``random`` draws fresh unit axes
    from ``generator`` on every call (ablation control); ``global`` shares the
    top-k principal components of the proxy matrix across all classes (second
    control). All axes are detached and ``top_k`` is clamped to the number of
    available classes minus one; ``global`` additionally clamps to the number
    of axes the SVD can return (at most the embedding dimension).
    """
    if axis_mode == "proxy":
        return _confusion_axes(
            proxy_embeddings,
            proxy_labels,
            top_k=top_k,
            torch_module=torch_module,
        )
    axes_by_class: dict[int, tuple[Any, Any]] = {}
    with torch_module.no_grad():
        unique_labels = [int(label) for label in torch_module.unique(proxy_labels).tolist()]
        axis_count = min(top_k, len(unique_labels) - 1)
        if axis_count <= 0:
            return axes_by_class

        def uniform_weights(count: int) -> Any:
            return torch_module.full(
                (count,),
                1.0 / count,
                dtype=proxy_embeddings.dtype,
                device=proxy_embeddings.device,
            )

        if axis_mode == "random":
            weights = uniform_weights(axis_count)
            for label in unique_labels:
                raw = torch_module.randn(
                    axis_count,
                    proxy_embeddings.shape[1],
                    generator=generator,
                    dtype=proxy_embeddings.dtype,
                    device=proxy_embeddings.device,
                )
                axes_by_class[label] = (_normalize(raw, torch_module), weights)
            return axes_by_class
        proxies = _normalize(proxy_embeddings, torch_module)
        centered = proxies - proxies.mean(dim=0, keepdim=True)
        _, _, right_singular = torch_module.linalg.svd(centered, full_matrices=False)
        shared_axes = right_singular[:axis_count]
        weights = uniform_weights(int(shared_axes.shape[0]))
        for label in unique_labels:
            axes_by_class[label] = (shared_axes, weights)
    return axes_by_class


def _gsi_interference_loss(
    embeddings: Any,
    labels: Any,
    *,
    axes_by_class: dict[int, tuple[Any, Any]],
    floor: float,
    variance_floor: float,
    min_group_size: int,
    torch_module: Any,
) -> Any:
    loss, _ = _gsi_interference_loss_with_diagnostics(
        embeddings,
        labels,
        axes_by_class=axes_by_class,
        floor=floor,
        variance_floor=variance_floor,
        min_group_size=min_group_size,
        torch_module=torch_module,
    )
    return loss


def _gsi_interference_loss_with_diagnostics(
    embeddings: Any,
    labels: Any,
    *,
    axes_by_class: dict[int, tuple[Any, Any]],
    floor: float,
    variance_floor: float,
    min_group_size: int,
    torch_module: Any,
) -> tuple[Any, dict[str, float] | None]:
    """Group Scatter-Interference: hinge on the scatter fraction along confusion axes.

    For each batch class with at least ``min_group_size`` members, the
    interference ratio ``rho`` is the fraction of the class's total scatter
    (batch statistics only) that lies along each confusion axis. The loss is
    the mean over (class, axis) pairs of ``weight * relu(rho - floor)``. The
    ratio is scale invariant while the total variance stays above
    ``variance_floor``; the clamp keeps per-member gradients bounded as
    classes compact. Axes must be detached (see ``_confusion_axes``).
    """
    terms = []
    ratio_terms = []
    for label in torch_module.unique(labels).tolist():
        class_embeddings = embeddings[labels == int(label)]
        if class_embeddings.shape[0] < min_group_size:
            continue
        entry = axes_by_class.get(int(label))
        if entry is None:
            continue
        axes, weights = entry
        centered = class_embeddings - class_embeddings.mean(dim=0, keepdim=True)
        total_variance = centered.pow(2).sum(dim=1).mean()
        parallel_variance = (centered @ axes.T).pow(2).mean(dim=0)
        ratios = parallel_variance / total_variance.clamp_min(variance_floor)
        ratio_terms.append(ratios.detach())
        terms.append(weights * torch_module.relu(ratios - floor))
    if not terms:
        return embeddings.sum() * 0.0, None
    loss = torch_module.cat(terms).mean()
    ratio_tensor = torch_module.cat(ratio_terms)
    diagnostics = {
        "unweighted_loss": float(loss.detach().cpu()),
        "active_fraction": float((ratio_tensor > floor).float().mean().detach().cpu()),
    }
    return loss, diagnostics


def _supervised_contrastive_loss(
    anchors: Any,
    anchor_labels: Any,
    *,
    contrast_embeddings: Any,
    contrast_labels: Any,
    temperature: float,
    torch_module: Any,
    exclude_self: bool,
) -> Any:
    logits = anchors @ contrast_embeddings.T / temperature
    positive_mask = anchor_labels[:, None].eq(contrast_labels[None, :])
    valid_mask = torch_module.ones_like(positive_mask, dtype=torch_module.bool)
    if exclude_self:
        diagonal_count = min(anchors.shape[0], contrast_embeddings.shape[0])
        diagonal = torch_module.arange(diagonal_count, device=positive_mask.device)
        valid_mask[diagonal, diagonal] = False
        positive_mask[diagonal, diagonal] = False
    logits = logits.masked_fill(~valid_mask, -1.0e9)
    positives_per_anchor = positive_mask.sum(dim=1)
    keep = positives_per_anchor > 0
    if not bool(keep.any()):
        return anchors.sum() * 0.0
    log_prob = logits - torch_module.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = (log_prob * positive_mask.float()).sum(
        dim=1
    ) / positives_per_anchor.clamp_min(1)
    return -positive_log_prob[keep].mean()


def _group_centroids(
    embeddings: Any,
    labels: Any,
    group_size: int,
    torch_module: Any,
) -> tuple[Any, Any]:
    centroids = []
    centroid_labels = []
    for label in torch_module.unique(labels).tolist():
        indices = torch_module.nonzero(labels == int(label), as_tuple=False).flatten()
        usable = (indices.shape[0] // group_size) * group_size
        if usable == 0:
            continue
        groups = embeddings[indices[:usable]].reshape(-1, group_size, embeddings.shape[1])
        centroids.append(_normalize(groups.mean(dim=1), torch_module))
        centroid_labels.append(
            torch_module.full(
                (groups.shape[0],),
                int(label),
                dtype=labels.dtype,
                device=labels.device,
            )
        )
    if not centroids:
        return embeddings[:0], labels[:0]
    return torch_module.cat(centroids, dim=0), torch_module.cat(centroid_labels, dim=0)


def _radius_penalty(embeddings: Any, labels: Any, *, target: float, torch_module: Any) -> Any:
    penalties = []
    for label in torch_module.unique(labels).tolist():
        class_embeddings = embeddings[labels == int(label)]
        if class_embeddings.shape[0] < 2:
            continue
        centroid = _normalize(class_embeddings.mean(dim=0, keepdim=True), torch_module)
        distances = 1.0 - (class_embeddings @ centroid.T).squeeze(1)
        penalties.append(torch_module.relu(distances.mean() - target) ** 2)
    if not penalties:
        return embeddings.sum() * 0.0
    return torch_module.stack(penalties).mean()


def _pairwise_similarity_preservation_loss(
    student_embeddings: Any,
    teacher_embeddings: Any,
    *,
    torch_module: Any,
) -> Any:
    student_similarities = student_embeddings @ student_embeddings.T
    teacher_similarities = teacher_embeddings @ teacher_embeddings.T
    return torch_module.nn.functional.mse_loss(student_similarities, teacher_similarities)


def _atomic_savez(path: Path, **arrays: NDArray[Any]) -> None:
    """Write an .npz atomically: save to a temp sibling, then os.replace into place.

    The best-epoch block rewrites the embeddings file every time retrieval improves,
    so an interruption mid-write must not leave a truncated/corrupt .npz behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends ".npz" if missing; write the temp file with that suffix so the
    # final os.replace target matches what the caller expects.
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}.npz")
    try:
        # numpy's savez stub can't distinguish **arrays from its allow_pickle kwarg.
        np.savez(tmp_path, **arrays)  # type: ignore[arg-type]
        os.replace(tmp_path, path if path.suffix == ".npz" else path.with_suffix(".npz"))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _encode_model(
    model: TorchImageModel,
    loader: Any,
    device: Any,
    torch_module: Any,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    embeddings = []
    labels = []
    model.eval()
    with torch_module.no_grad():
        for images, batch_labels in loader:
            batch_embeddings = _normalize(model(images.to(device, non_blocking=True)), torch_module)
            embeddings.append(batch_embeddings.detach().cpu().numpy().astype(np.float64))
            labels.append(batch_labels.detach().cpu().numpy().astype(np.int64))
    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


def _interference_diagnostics(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    top_k: int = 3,
) -> dict[str, float] | None:
    """Compute GSI interference diagnostics from test-class means.

    Test classes have no trainable proxies, so the diagnostic uses each class
    mean and the mean-difference axes toward its nearest test classes. The
    resulting ratios mirror GSI's ``rho`` mechanism: the fraction of a class's
    scatter that lies along a confusable-class axis.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or labels.shape[0] != embeddings.shape[0]:
        return None
    if top_k <= 0:
        return None

    unique_labels = np.asarray(sorted(int(label) for label in np.unique(labels)), dtype=np.int64)
    if unique_labels.shape[0] < 2:
        return None

    class_means = np.stack(
        [embeddings[labels == label].mean(axis=0) for label in unique_labels],
        axis=0,
    )
    # Pairwise class-mean distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, so we
    # never materialise the (C, C, D) difference tensor — that is 489 GiB at SOP's
    # ~11.3k classes. The (C, C) distance matrix is ~1 GiB and the per-class loop
    # below recomputes the axis differences it actually needs on the fly.
    sq_norms = np.sum(class_means * class_means, axis=1)
    distances = np.sqrt(
        np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2.0 * (class_means @ class_means.T), 0.0)
    )
    np.fill_diagonal(distances, np.inf)
    neighbor_count = min(top_k, unique_labels.shape[0] - 1)

    ratios: list[float] = []
    for class_index, label in enumerate(unique_labels):
        class_embeddings = embeddings[labels == label]
        if class_embeddings.shape[0] < 2:
            continue
        centered = class_embeddings - class_embeddings.mean(axis=0, keepdims=True)
        total_variance = float(np.mean(np.sum(centered * centered, axis=1)))
        nearest_indices = np.argsort(distances[class_index])[:neighbor_count]
        for neighbor_index in nearest_indices:
            axis = class_means[int(neighbor_index)] - class_means[class_index]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1.0e-12 or total_variance <= 1.0e-12:
                ratios.append(0.0)
                continue
            unit_axis = axis / axis_norm
            parallel_projection = centered @ unit_axis
            parallel_variance = float(np.mean(parallel_projection * parallel_projection))
            ratios.append(max(0.0, parallel_variance / total_variance))

    if not ratios:
        return None

    rho_values = np.asarray(ratios, dtype=np.float64)
    return {
        "rho_mean": float(np.mean(rho_values)),
        "rho_p90": float(np.percentile(rho_values, 90)),
        "rho_max": float(np.max(rho_values)),
        "fraction_above_floor_002": float(np.mean(rho_values > 0.02)),
        "fraction_above_floor_005": float(np.mean(rho_values > 0.05)),
    }


def _normalize_numpy(values: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, np.maximum(norms, 1.0e-12))


def _proxy_axis_interference_diagnostics(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    proxy_embeddings: NDArray[np.float64],
    proxy_labels: NDArray[np.int64],
    top_k: int = 3,
    floor: float = 0.02,
) -> dict[str, float] | None:
    """Compute train-class interference along the proxy-derived GSI axes.

    Unlike ``_interference_diagnostics``, this uses the same proxy-neighbor
    axis construction as the training loss. It is therefore a direct binding
    diagnostic for proxy-axis GSI on train classes.
    """
    if embeddings.ndim != 2 or proxy_embeddings.ndim != 2:
        return None
    if embeddings.shape[0] == 0 or labels.shape[0] != embeddings.shape[0]:
        return None
    if proxy_embeddings.shape[0] == 0 or proxy_labels.shape[0] != proxy_embeddings.shape[0]:
        return None
    if embeddings.shape[1] != proxy_embeddings.shape[1] or top_k <= 0:
        return None

    proxies = _normalize_numpy(proxy_embeddings)
    unique_proxy_labels = np.asarray(
        sorted(int(label) for label in np.unique(proxy_labels)),
        dtype=np.int64,
    )
    if unique_proxy_labels.shape[0] < 2:
        return None

    similarities = proxies @ proxies.T
    proxy_indices_by_label = {
        int(label): np.flatnonzero(proxy_labels == label) for label in unique_proxy_labels
    }
    neighbor_count = min(top_k, int(unique_proxy_labels.shape[0]) - 1)
    ratios: list[float] = []
    for label in unique_proxy_labels:
        label_int = int(label)
        class_embeddings = embeddings[labels == label_int]
        if class_embeddings.shape[0] < 2:
            continue
        own_indices = proxy_indices_by_label[label_int]
        if own_indices.shape[0] == 0:
            continue
        centered = class_embeddings - class_embeddings.mean(axis=0, keepdims=True)
        total_variance = float(np.mean(np.sum(centered * centered, axis=1)))

        scored_axes: list[tuple[float, NDArray[np.float64]]] = []
        for other_label in unique_proxy_labels:
            other_int = int(other_label)
            if other_int == label_int:
                continue
            other_indices = proxy_indices_by_label[other_int]
            if other_indices.shape[0] == 0:
                continue
            cross = similarities[np.ix_(own_indices, other_indices)]
            flat_index = int(np.argmax(cross))
            own_best = int(own_indices[flat_index // cross.shape[1]])
            other_best = int(other_indices[flat_index % cross.shape[1]])
            scored_axes.append(
                (
                    float(cross.reshape(-1)[flat_index]),
                    proxies[other_best] - proxies[own_best],
                )
            )

        for _, axis in sorted(scored_axes, key=lambda item: item[0], reverse=True)[:neighbor_count]:
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1.0e-12 or total_variance <= 1.0e-12:
                ratios.append(0.0)
                continue
            unit_axis = axis / axis_norm
            parallel_projection = centered @ unit_axis
            parallel_variance = float(np.mean(parallel_projection * parallel_projection))
            ratios.append(max(0.0, parallel_variance / total_variance))

    if not ratios:
        return None

    rho_values = np.asarray(ratios, dtype=np.float64)
    return {
        "proxy_axis_rho_mean": float(np.mean(rho_values)),
        "proxy_axis_rho_p90": float(np.percentile(rho_values, 90)),
        "proxy_axis_rho_max": float(np.max(rho_values)),
        "proxy_axis_fraction_above_floor": float(np.mean(rho_values > floor)),
    }


def _boundary_axis_interference_diagnostics(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    top_k: int = 3,
    floor: float = 0.0,
    temperature: float = 0.1,
) -> dict[str, float] | None:
    """Compute train-class interference along BGSI batch-mean boundary axes."""
    del temperature
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or labels.shape[0] != embeddings.shape[0]:
        return None
    if top_k <= 0:
        return None

    unique_labels = np.asarray(sorted(int(label) for label in np.unique(labels)), dtype=np.int64)
    if unique_labels.shape[0] < 2:
        return None

    class_means = np.stack(
        [embeddings[labels == label].mean(axis=0) for label in unique_labels],
        axis=0,
    )
    normalized_means = _normalize_numpy(class_means)
    similarities = normalized_means @ normalized_means.T
    np.fill_diagonal(similarities, -np.inf)
    neighbor_count = min(top_k, int(unique_labels.shape[0]) - 1)

    ratios: list[float] = []
    for class_index, label in enumerate(unique_labels):
        class_embeddings = embeddings[labels == label]
        if class_embeddings.shape[0] < 2:
            continue
        centered = class_embeddings - class_embeddings.mean(axis=0, keepdims=True)
        total_variance = float(np.mean(np.sum(centered * centered, axis=1)))
        nearest_indices = np.argsort(similarities[class_index])[-neighbor_count:][::-1]
        for neighbor_index in nearest_indices:
            axis = class_means[int(neighbor_index)] - class_means[class_index]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1.0e-12 or total_variance <= 1.0e-12:
                ratios.append(0.0)
                continue
            unit_axis = axis / axis_norm
            parallel_projection = centered @ unit_axis
            parallel_variance = float(np.mean(parallel_projection * parallel_projection))
            ratios.append(max(0.0, parallel_variance / total_variance))

    if not ratios:
        return None

    rho_values = np.asarray(ratios, dtype=np.float64)
    return {
        "boundary_axis_rho_mean": float(np.mean(rho_values)),
        "boundary_axis_rho_p90": float(np.percentile(rho_values, 90)),
        "boundary_axis_rho_max": float(np.max(rho_values)),
        "boundary_axis_fraction_above_floor": float(np.mean(rho_values > floor)),
    }


def _summarize_gsi_training_diagnostics(
    step_diagnostics: Sequence[dict[str, float]],
    *,
    proxy_axis_diagnostics: dict[str, float] | None = None,
    boundary_axis_diagnostics: dict[str, float] | None = None,
) -> dict[str, float] | None:
    if (
        not step_diagnostics
        and proxy_axis_diagnostics is None
        and boundary_axis_diagnostics is None
    ):
        return None

    summary: dict[str, float] = {}
    if step_diagnostics:
        unweighted_losses = np.asarray(
            [diagnostic["unweighted_loss"] for diagnostic in step_diagnostics],
            dtype=np.float64,
        )
        active_fractions = np.asarray(
            [diagnostic["active_fraction"] for diagnostic in step_diagnostics],
            dtype=np.float64,
        )
        summary.update(
            {
                "active_steps": float(unweighted_losses.shape[0]),
                "unweighted_loss_mean": float(np.mean(unweighted_losses)),
                "unweighted_loss_p90": float(np.percentile(unweighted_losses, 90)),
                "unweighted_loss_max": float(np.max(unweighted_losses)),
                "active_fraction_mean": float(np.mean(active_fractions)),
            }
        )
        for source_key, summary_key in (
            ("bgsi_axis_coverage", "bgsi_axis_coverage_mean"),
            ("bgsi_axis_count", "bgsi_axis_count_mean"),
            ("bgsi_ema_ready_fraction", "bgsi_ema_ready_fraction_mean"),
            ("bgsi_axis_agreement_fraction", "bgsi_axis_agreement_fraction_mean"),
            ("bgsi_permuted_match_fraction", "bgsi_permuted_match_fraction_mean"),
        ):
            values = [
                diagnostic[source_key]
                for diagnostic in step_diagnostics
                if source_key in diagnostic
            ]
            if values:
                summary[summary_key] = float(np.mean(np.asarray(values, dtype=np.float64)))
    else:
        summary.update(
            {
                "active_steps": 0.0,
                "unweighted_loss_mean": 0.0,
                "unweighted_loss_p90": 0.0,
                "unweighted_loss_max": 0.0,
                "active_fraction_mean": 0.0,
            }
        )
    if proxy_axis_diagnostics is not None:
        summary.update(proxy_axis_diagnostics)
    if boundary_axis_diagnostics is not None:
        summary.update(boundary_axis_diagnostics)
    return summary


def _checkpoint_selection_score(
    model: TorchImageModel,
    loader: Any,
    device: Any,
    torch_module: Any,
    *,
    config: ImageEndToEndConfig,
) -> float:
    embeddings, labels = _encode_model(model, loader, device, torch_module)
    retrieval = image_self_retrieval_score(
        embeddings,
        labels,
        query_limit=config.checkpoint_selection_query_limit,
        random_state=config.seed,
    )
    if config.checkpoint_selection_metric == "recall_at_1":
        return retrieval.recall_at_1
    return retrieval.map_at_r


def _normalize(tensor: Any, torch_module: Any) -> Any:
    return torch_module.nn.functional.normalize(tensor, p=2, dim=-1)


def _torchvision_model_factory(
    config: ImageEndToEndConfig,
    *,
    use_embedding_head: bool = True,
) -> TorchImageModel:
    try:
        import torch
        from torchvision import models
    except ImportError as error:
        raise RuntimeError(
            "Install the research extra to run end-to-end image benchmarks: "
            "uv sync --group dev --extra research"
        ) from error
    if config.backbone_name != "resnet50":
        raise ValueError("Only resnet50 is currently supported for end-to-end paper protocol runs")
    weights = (
        models.ResNet50_Weights.IMAGENET1K_V1
        if config.pretrained_weights == "v1"
        else models.ResNet50_Weights.IMAGENET1K_V2
    )
    model = models.resnet50(weights=weights)
    _set_resnet_output_layer(
        model,
        config,
        use_embedding_head=use_embedding_head,
        torch_module=torch,
    )
    return cast(TorchImageModel, model)


def _set_resnet_output_layer(
    model: Any,
    config: ImageEndToEndConfig,
    *,
    use_embedding_head: bool,
    torch_module: Any,
) -> None:
    if config.head_pooling == "avg_max" and hasattr(model, "avgpool"):
        model.avgpool = _avg_max_pooling_layer(torch_module)
    if use_embedding_head:
        in_features = int(model.fc.in_features)
        head = torch_module.nn.Linear(in_features, config.embedding_dimensions)
        if config.embedding_head_init == "kaiming_normal":
            torch_module.nn.init.kaiming_normal_(head.weight, mode="fan_out")
            if head.bias is not None:
                torch_module.nn.init.zeros_(head.bias)
        if config.embedding_layer_norm:
            # Reference `is_norm`: LayerNorm with no affine params, applied after the
            # embedding Linear (see ljin0429/HIST net/resnet.py Resnet50.forward).
            layer_norm = torch_module.nn.LayerNorm(
                config.embedding_dimensions, elementwise_affine=False
            )
            model.fc = torch_module.nn.Sequential(head, layer_norm)
        else:
            model.fc = head
    else:
        model.fc = torch_module.nn.Identity()


def _avg_max_pooling_layer(torch_module: Any) -> Any:
    class AvgMaxPool2d(torch_module.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self._avg = torch_module.nn.AdaptiveAvgPool2d((1, 1))
            self._max = torch_module.nn.AdaptiveMaxPool2d((1, 1))

        def forward(self, tensor: Any) -> Any:
            return self._avg(tensor) + self._max(tensor)

    return AvgMaxPool2d()


def _optimizer_parameter_groups(
    model: TorchImageModel,
    config: ImageEndToEndConfig,
) -> list[dict[str, Any]]:
    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if config.backbone_learning_rate is None or not named_parameters:
        parameters = [parameter for _, parameter in named_parameters] or list(model.parameters())
        return _adamw_parameter_groups(
            model,
            named_parameters,
            [{"params": parameters}],
            config,
        )

    proxy_parameters = [
        parameter for name, parameter in named_parameters if name == "metric_proxies"
    ]
    backbone_parameters = [
        parameter
        for name, parameter in named_parameters
        if not name.startswith("fc.")
        and not name.startswith("hist_module.")
        and name != "metric_proxies"
    ]
    head_parameters = [parameter for name, parameter in named_parameters if name.startswith("fc.")]
    # HIST trains its class distributions and HGNN at their own, much larger LRs.
    hist_distribution_parameters = [
        parameter
        for name, parameter in named_parameters
        if name in ("hist_module.means", "hist_module.log_vars")
    ]
    hist_hgnn_parameters = [
        parameter
        for name, parameter in named_parameters
        if name.startswith("hist_module.")
        and name not in ("hist_module.means", "hist_module.log_vars")
    ]
    groups: list[dict[str, Any]] = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": config.backbone_learning_rate})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": config.learning_rate})
    if hist_distribution_parameters:
        groups.append({"params": hist_distribution_parameters, "lr": config.hist_lr_ds})
    if hist_hgnn_parameters:
        groups.append(
            {
                "params": hist_hgnn_parameters,
                "lr": config.backbone_learning_rate * config.hist_lr_hgnn_factor,
            }
        )
    if proxy_parameters:
        groups.append(
            {
                "params": proxy_parameters,
                "lr": config.learning_rate * config.proxy_learning_rate_multiplier,
            }
        )
    return _adamw_parameter_groups(
        model,
        named_parameters,
        groups or [{"params": [parameter for _, parameter in named_parameters]}],
        config,
    )


def _adamw_parameter_groups(
    model: TorchImageModel,
    named_parameters: Sequence[tuple[str, Any]],
    groups: Sequence[dict[str, Any]],
    config: ImageEndToEndConfig,
) -> list[dict[str, Any]]:
    if config.optimizer != "adamw":
        return [dict(group) for group in groups]

    no_decay_parameter_ids = {
        id(parameter)
        for name, parameter in named_parameters
        if name == "bias" or name.endswith(".bias") or name == "metric_proxies"
    }
    for module in cast(Any, model).modules():
        if module.__class__.__name__.startswith("BatchNorm"):
            no_decay_parameter_ids.update(
                id(parameter) for parameter in module.parameters(recurse=False)
            )

    split_groups: list[dict[str, Any]] = []
    for group in groups:
        decay_parameters = []
        no_decay_parameters = []
        for parameter in group["params"]:
            if id(parameter) in no_decay_parameter_ids:
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)
        base_group = {key: value for key, value in group.items() if key != "params"}
        if decay_parameters:
            split_groups.append(
                {
                    **base_group,
                    "params": decay_parameters,
                    "weight_decay": config.weight_decay,
                }
            )
        if no_decay_parameters:
            split_groups.append(
                {
                    **base_group,
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                }
            )
    return split_groups


def _default_transform_factory(
    config: ImageEndToEndConfig,
    train: bool,
) -> Callable[[object], Any]:
    try:
        import torch
        from torchvision import transforms
    except ImportError as error:
        raise RuntimeError(
            "Install the research extra to run end-to-end image benchmarks: "
            "uv sync --group dev --extra research"
        ) from error
    if train and config.mead_weight > 0.0:
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        global_crop = transforms.RandomResizedCrop(
            224,
            scale=(config.mead_global_scale_min, 1.0),
        )
        local_crop = transforms.RandomResizedCrop(
            config.mead_local_size,
            scale=(0.14, config.mead_local_scale_max),
        )
        horizontal_flip = transforms.RandomHorizontalFlip()
        global_transform = transforms.Compose(
            [
                global_crop,
                horizontal_flip,
                transforms.ToTensor(),
                normalize,
            ]
        )
        local_transform = transforms.Compose(
            [
                local_crop,
                horizontal_flip,
                transforms.ToTensor(),
                normalize,
            ]
        )
        tensor_global_transform = transforms.Compose([global_crop, horizontal_flip, normalize])
        tensor_local_transform = transforms.Compose([local_crop, horizontal_flip, normalize])

        def apply_multicrop(image: object) -> tuple[torch.Tensor, torch.Tensor]:
            if torch.is_tensor(image):
                tensor_image = image.float()
                global_crops = torch.stack(
                    [tensor_global_transform(tensor_image) for _ in range(2)],
                    dim=0,
                )
                if config.mead_local_crops == 0:
                    local_crops = global_crops.new_empty(
                        (0, 3, config.mead_local_size, config.mead_local_size)
                    )
                else:
                    local_crops = torch.stack(
                        [
                            tensor_local_transform(tensor_image)
                            for _ in range(config.mead_local_crops)
                        ],
                        dim=0,
                    )
                return global_crops, local_crops

            source_image = image.convert("RGB") if hasattr(image, "convert") else image
            global_crops = torch.stack([global_transform(source_image) for _ in range(2)], dim=0)
            if config.mead_local_crops == 0:
                local_crops = global_crops.new_empty(
                    (0, 3, config.mead_local_size, config.mead_local_size)
                )
            else:
                local_crops = torch.stack(
                    [local_transform(source_image) for _ in range(config.mead_local_crops)],
                    dim=0,
                )
            return global_crops, local_crops

        return apply_multicrop

    if train and config.train_augmentation == "full_res_crop":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(config.input_size, scale=(0.16, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    elif train and config.train_augmentation == "standard":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(config.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def apply(image: object) -> Any:
        if torch.is_tensor(image):
            return image.float()
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        return transform(image)

    return apply


def _to_payload(result: ImageEndToEndResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "dataset_name": result.dataset_name,
        "protocol": result.protocol,
        "config": result.config.model_dump(mode="json"),
        "train_examples": result.train_examples,
        "test_examples": result.test_examples,
        "methods": {
            key: {
                "model_name": metrics.model_name,
                "objective": metrics.objective,
                "display_name": metrics.display_name,
                "dimensions": metrics.dimensions,
                "retrieval": asdict(metrics.retrieval),
                "precision_at_1": metrics.precision_at_1,
                "recall_at_1": metrics.recall_at_1,
                "recall_at_2": metrics.recall_at_2,
                "recall_at_4": metrics.recall_at_4,
                "recall_at_8": metrics.recall_at_8,
                "map_at_r": metrics.map_at_r,
                "loss_history": metrics.loss_history,
                "interference": metrics.interference,
                "train_interference": metrics.train_interference,
                "gsi_diagnostics": metrics.gsi_diagnostics,
                "selected_step": metrics.selected_step,
                "selection_metric": metrics.selection_metric,
                "selection_score": metrics.selection_score,
                "best_test_recall_at_1": metrics.best_test_recall_at_1,
                "best_test_epoch": metrics.best_test_epoch,
                "test_recall_history": metrics.test_recall_history,
            }
            for key, metrics in result.methods.items()
        },
    }
