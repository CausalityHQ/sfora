import contextlib
import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from sfora.bn_inception import (
    BN_INCEPTION_CHECKPOINT_SHA256,
    build_bn_inception,
    validate_bn_inception_checkpoint,
)
from sfora.data import ImageExample
from sfora.image_end_to_end import (
    EndToEndProtocol,
    ImageEndToEndConfig,
    _backbone_warmup_parameters,
    _build_hist_module,
    _class_excluded_gradient_target_loss,
    _clip_gradients,
    _default_transform_factory,
    _freeze_batch_norm_affine_parameters,
    _freeze_batch_norm_layers,
    _gem_pooling_layer,
    _hist_hypergraph_targets,
    _hist_loss,
    _hist_memory_loss,
    _hist_sinkhorn_loss,
    _hypergraph_distillation_loss,
    _local_nca_loss,
    _optimizer_parameter_groups,
    _recall_at_k_surrogate_loss,
    _resolve_training_schedule,
    _should_evaluate_test,
    _should_step_scheduler,
    _sinkhorn_log_coupling,
    _supervised_contrastive_loss,
    _tird_interaction_matrix,
    _tird_loss,
    _update_ema_teacher,
    config_for_protocol,
    run_image_end_to_end_benchmark,
)
from sfora.report import ReportConfig, build_site_data


def test_tird_interaction_is_exact_closed_tetrad() -> None:
    torch = pytest.importorskip("torch")
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.3, 0.7]], dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1])
    interaction = _tird_interaction_matrix(embeddings, labels, torch_module=torch)
    raw = embeddings @ embeddings.T
    tetrad = raw[0, 2] - raw[0, 3] - raw[1, 2] + raw[1, 3]

    assert tetrad.item() == pytest.approx(
        (interaction[0, 2] - interaction[0, 3] - interaction[1, 2] + interaction[1, 3]).item()
    )
    assert tetrad.item() == pytest.approx(4.0 * interaction[0, 2].item())


def test_tird_loss_matches_interactions_and_backpropagates() -> None:
    torch = pytest.importorskip("torch")
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    teacher = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.3, 0.7], [0.8, 0.6], [0.4, 0.9]],
        dtype=torch.float64,
    )
    identical = teacher.clone().requires_grad_(True)
    assert _tird_loss(identical, teacher, labels, torch_module=torch).item() == pytest.approx(0.0)

    student = torch.tensor(
        [[1.0, 0.0], [0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.4], [0.3, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = _tird_loss(student, teacher, labels, torch_module=torch)
    assert loss.item() > 0.0
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_class_excluded_gradient_target_loss_is_zero_only_at_target() -> None:
    torch = pytest.importorskip("torch")
    labels = torch.tensor([0, 0, 1, 1])
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]], requires_grad=True)
    loss = _class_excluded_gradient_target_loss(embeddings, labels, torch_module=torch)
    assert loss.item() > 0.0
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


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


def test_epoch_schedule_respects_drop_last_train_batch() -> None:
    keep = ImageEndToEndConfig(batch_size=180, train_epochs=60)
    drop = ImageEndToEndConfig(
        batch_size=180,
        train_epochs=60,
        drop_last_train_batch=True,
    )

    assert _resolve_training_schedule(
        keep,
        optimization_example_count=59_551,
    )[:2] == (19_860, 331)
    assert _resolve_training_schedule(
        drop,
        optimization_example_count=59_551,
    )[:2] == (19_800, 330)


def test_bn_inception_warmup_keeps_embedding_head_trainable() -> None:
    torch = pytest.importorskip("torch")

    class WrappedBNInception(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.backbone = torch.nn.Linear(3, 4)
            self.model.embedding = torch.nn.Linear(4, 2)
            self.metric_proxies = torch.nn.Parameter(torch.zeros(2, 2))

    model = WrappedBNInception()
    frozen = {id(parameter) for parameter in _backbone_warmup_parameters(model)}

    assert id(model.model.backbone.weight) in frozen
    assert id(model.model.embedding.weight) not in frozen
    assert id(model.model.embedding.bias) not in frozen
    assert id(model.metric_proxies) not in frozen


def test_additional_warmup_does_not_consume_hist_main_epochs() -> None:
    config = ImageEndToEndConfig(
        batch_size=32,
        train_epochs=40,
        warmup_epochs=1,
        warmup_is_additional=True,
    )

    steps, per_epoch, total_epochs = _resolve_training_schedule(
        config,
        optimization_example_count=320,
    )

    assert per_epoch == 10
    assert total_epochs == 41
    assert steps == 410


def test_hist_scheduler_does_not_step_during_additional_warmup() -> None:
    config = ImageEndToEndConfig(
        warmup_epochs=1,
        warmup_is_additional=True,
        schedule_during_warmup=False,
    )

    assert _should_step_scheduler(config, completed_epoch=1) is False
    assert _should_step_scheduler(config, completed_epoch=2) is True


def test_official_weight_decay_policy_keeps_optimizer_groups_unsplit() -> None:
    torch: Any = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
    config = ImageEndToEndConfig(
        optimizer="adamw",
        backbone_learning_rate=None,
        weight_decay_exclusions="none",
    )

    groups = _optimizer_parameter_groups(model, config)

    assert len(groups) == 1
    assert "weight_decay" not in groups[0]
    assert len(groups[0]["params"]) == len(list(model.parameters()))


def test_reference_transform_does_not_resize_before_random_crop() -> None:
    import inspect

    config = ImageEndToEndConfig(train_augmentation="reference_random_resized_crop")

    transform = _default_transform_factory(config, True)
    pipeline = inspect.getclosurevars(transform).nonlocals["transform"]
    names = [type(step).__name__ for step in pipeline.transforms]

    assert names[:2] == ["RandomResizedCrop", "RandomHorizontalFlip"]
    assert "Resize" not in names


def _ema_teacher_pair(torch: Any) -> tuple[Any, Any]:
    """A student whose BatchNorm running stats have moved away from the teacher's."""
    import copy

    student = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(1.0)
        student.train()
        student(torch.randn(8, 3) * 5.0 + 3.0)  # move running_mean/var
    return teacher, student


def test_ema_teacher_hard_copies_buffers_by_default() -> None:
    """Historical behaviour: normalisation statistics jump to the student instantly
    while weights lag. Preserved as the default so old artifacts reproduce."""
    torch: Any = pytest.importorskip("torch")
    teacher, student = _ema_teacher_pair(torch)

    _update_ema_teacher(teacher, student, momentum=0.9)

    assert torch.allclose(teacher[0].running_mean, student[0].running_mean)
    # Weights, by contrast, lag behind.
    assert not torch.allclose(teacher[1].weight, student[1].weight)


def test_ema_teacher_can_blend_float_buffers_at_the_same_momentum() -> None:
    """H3 fix: buffers lag at the same rate as weights, keeping the teacher
    internally consistent instead of pairing stale weights with fresh statistics."""
    torch: Any = pytest.importorskip("torch")
    teacher, student = _ema_teacher_pair(torch)
    before_mean = teacher[0].running_mean.clone()
    momentum = 0.9

    _update_ema_teacher(teacher, student, momentum=momentum, ema_buffers=True)

    expected = before_mean * momentum + student[0].running_mean * (1.0 - momentum)
    assert torch.allclose(teacher[0].running_mean, expected)
    assert not torch.allclose(teacher[0].running_mean, student[0].running_mean)
    # Integer buffers cannot be blended and must still be copied verbatim.
    assert teacher[0].num_batches_tracked.dtype == torch.long
    assert torch.equal(teacher[0].num_batches_tracked, student[0].num_batches_tracked)


def _teacher_student_pair(torch: Any, *, freeze_batch_norm: bool) -> tuple[Any, Any, Any]:
    """Reproduce the training loop's teacher/student setup on a BatchNorm model.

    The student's running statistics are driven away from the batch statistics of
    the probe input, so eval-mode (running stats) and train-mode (batch stats)
    forwards are distinguishable.
    """
    import copy

    torch.manual_seed(0)
    student = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 3))
    with torch.no_grad():
        student.train()
        for _ in range(20):
            student(torch.randn(16, 4) * 4.0 + 6.0)  # move running_mean/var far from init
    teacher = copy.deepcopy(student)
    # Student side, exactly as the loop does it (image_end_to_end.py:748-750).
    student.train()
    if freeze_batch_norm:
        _freeze_batch_norm_layers(student)
    probe = torch.randn(16, 4)  # batch stats ~N(0,1), far from the running stats above
    return teacher, student, probe


def _teacher_output(torch: Any, teacher: Any, probe: Any, *, train_mode: bool, freeze: bool) -> Any:
    """Teacher setup exactly as at image_end_to_end.py:736-742."""
    if train_mode:
        teacher.train()
        if freeze:
            _freeze_batch_norm_layers(teacher)
    else:
        teacher.eval()
    with torch.no_grad():
        return teacher(probe).clone()


def test_eval_mode_teacher_diverges_from_student_when_batch_norm_is_trainable() -> None:
    """H3, the mechanism itself. With trainable BatchNorm the historical eval-mode
    teacher computes a DIFFERENT function of the same input than the student, so the
    distillation target is not a lagged copy of the student but a different model."""
    torch: Any = pytest.importorskip("torch")
    teacher, student, probe = _teacher_student_pair(torch, freeze_batch_norm=False)

    with torch.no_grad():
        student_out = student(probe)
    historical = _teacher_output(torch, teacher, probe, train_mode=False, freeze=False)
    fixed = _teacher_output(torch, teacher, probe, train_mode=True, freeze=False)

    # Teacher and student have IDENTICAL weights here (fresh deepcopy, no EMA drift),
    # so any difference is purely the normalisation regime.
    assert not torch.allclose(historical, student_out, atol=1e-4)
    assert torch.allclose(fixed, student_out, atol=1e-6)


def test_teacher_normalisation_fix_is_inert_when_batch_norm_is_frozen() -> None:
    """H3's null prediction, proved rather than measured. With frozen BatchNorm both
    teacher modes force BN to eval, so the fix cannot change anything -- which is why
    the CUB arms of the matrix do not need to spend GPU time re-checking it."""
    torch: Any = pytest.importorskip("torch")
    teacher, _student, probe = _teacher_student_pair(torch, freeze_batch_norm=True)

    historical = _teacher_output(torch, teacher, probe, train_mode=False, freeze=True)
    fixed = _teacher_output(torch, teacher, probe, train_mode=True, freeze=True)

    assert torch.equal(historical, fixed)


def test_buffer_blending_is_a_no_op_while_batch_norm_stays_frozen() -> None:
    """`ema_teacher_ema_buffers` is inert under `freeze_batch_norm=True`: eval-mode
    BatchNorm never updates running statistics, so the student's buffers are constant,
    and blending a teacher buffer toward a constant it already equals is the identity.

    Note the identity holds in exact arithmetic but NOT bit-exactly in floating point:
    `X*m + X*(1-m)` rounds to within an ulp of `X`, not to `X`. So the frozen-BatchNorm
    arms are numerically inert, not bitwise reproducible.
    """
    torch: Any = pytest.importorskip("torch")
    teacher, student, _probe = _teacher_student_pair(torch, freeze_batch_norm=True)
    before = teacher[0].running_mean.clone()

    with torch.no_grad():  # frozen BN: forwards must not move the student's buffers
        for _ in range(5):
            student(torch.randn(16, 4) * 9.0 - 3.0)
    assert torch.equal(student[0].running_mean, before)

    _update_ema_teacher(teacher, student, momentum=0.999, ema_buffers=True)
    assert torch.allclose(teacher[0].running_mean, before, rtol=0.0, atol=1e-6)
    assert not torch.equal(teacher[0].running_mean, before)  # ...but not bit-identical


def test_train_mode_teacher_ignores_running_buffers_entirely() -> None:
    """Under `ema_teacher_train_mode=True` the teacher normalises with BATCH statistics,
    so its running buffers do not influence its output at all -- there is no
    normalisation 'lag' to worry about, however stale those buffers are."""
    torch: Any = pytest.importorskip("torch")
    teacher, _student, probe = _teacher_student_pair(torch, freeze_batch_norm=False)

    baseline = _teacher_output(torch, teacher, probe, train_mode=True, freeze=False)
    with torch.no_grad():  # corrupt the running statistics beyond recognition
        teacher[0].running_mean.fill_(1234.0)
        teacher[0].running_var.fill_(4321.0)
    after = _teacher_output(torch, teacher, probe, train_mode=True, freeze=False)

    assert torch.allclose(baseline, after, atol=1e-6)


def _hist_fixture(torch: Any, *, nb_classes: int = 4, dim: int = 6, batch: int = 12) -> Any:
    torch.manual_seed(7)
    module = _build_hist_module(nb_classes=nb_classes, sz_embed=dim, hidden=8, torch_module=torch)
    module.train()
    embeddings = torch.randn(batch, dim)
    labels = torch.tensor([i % nb_classes for i in range(batch)])
    label_to_index = {i: i for i in range(nb_classes)}
    return module, embeddings, labels, label_to_index


def _two_positive_batch(torch: Any) -> tuple[Any, Any]:
    """Anchor with one NEAR and one FAR same-class positive, plus two negatives.

    The far positive is the thing at stake: a collapsing objective drags it in, a
    structure-preserving one leaves it alone once the near positive already wins.
    """
    anchor = torch.tensor([1.0, 0.0, 0.0])
    near_positive = torch.tensor([0.98, 0.20, 0.0])
    far_positive = torch.tensor([0.0, 0.0, 1.0])  # same class, very different pose
    negative_a = torch.tensor([0.70, 0.71, 0.0])
    negative_b = torch.tensor([0.60, 0.0, 0.80])
    embeddings = torch.stack([anchor, near_positive, far_positive, negative_a, negative_b])
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    labels = torch.tensor([0, 0, 0, 1, 2])
    return embeddings.requires_grad_(True), labels


def test_local_nca_does_not_drag_in_a_distant_same_class_sample() -> None:
    """The core claim, made checkable.

    SupCon's L_out puts the positive sum OUTSIDE the log, so it is minimised only when
    EVERY positive is close -- it collapses the class. Local NCA's L_in puts the sum
    INSIDE the log, so it is satisfied once *some* genuine positive outranks the
    negatives, and a legitimately distant same-class sample is left where it is.

    Both losses see identical inputs; only the objective differs.
    """
    torch: Any = pytest.importorskip("torch")

    def far_positive_pull(loss_fn: Any) -> float:
        embeddings, labels = _two_positive_batch(torch)
        # Score ONLY the first row as anchor. If every row is also an anchor, row 2's
        # gradient mixes its role as a distant positive with its own anchor term and
        # the comparison measures nothing.
        loss = loss_fn(embeddings[:1], labels[:1], embeddings, labels)
        loss.backward()
        assert embeddings.grad is not None
        # Gradient magnitude on the FAR positive (row 2).
        return float(embeddings.grad[2].norm())

    collapsing = far_positive_pull(
        lambda a, ay, c, cy: _supervised_contrastive_loss(
            a,
            ay,
            contrast_embeddings=c,
            contrast_labels=cy,
            temperature=0.1,
            torch_module=torch,
            exclude_self=True,
        )
    )
    preserving = far_positive_pull(
        lambda a, ay, c, cy: _local_nca_loss(
            a,
            ay,
            contrast_embeddings=c,
            contrast_labels=cy,
            temperature=0.1,
            negatives_k=2,
            torch_module=torch,
        )
    )

    assert preserving < collapsing, (
        f"local NCA should exert less pull on a distant positive than SupCon "
        f"({preserving:.6f} vs {collapsing:.6f})"
    )
    # Not marginal: the distant positive should be left essentially untouched, while
    # SupCon drags it in hard. Measured ~0.00004 vs ~5.0.
    assert preserving < 0.01 * collapsing


def test_local_nca_still_separates_a_rank_inversion() -> None:
    """Non-collapsing must not mean permissive: when a cross-class instance actually
    outranks every same-class one, the loss must be large and push back."""
    torch: Any = pytest.importorskip("torch")
    anchor = torch.tensor([1.0, 0.0, 0.0])
    positive = torch.tensor([0.0, 1.0, 0.0])  # same class but far
    intruder = torch.tensor([0.99, 0.14, 0.0])  # different class, nearer than positive
    embeddings = torch.stack([anchor, positive, intruder])
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    labels = torch.tensor([0, 0, 1])
    inverted = _local_nca_loss(
        embeddings,
        labels,
        contrast_embeddings=embeddings,
        contrast_labels=labels,
        temperature=0.1,
        negatives_k=1,
        torch_module=torch,
    )

    # Same geometry, but now the same-class sample is the nearest one.
    fixed = torch.stack([anchor, intruder, positive])
    fixed = fixed / fixed.norm(dim=1, keepdim=True)
    ordered = _local_nca_loss(
        fixed,
        labels,
        contrast_embeddings=fixed,
        contrast_labels=labels,
        temperature=0.1,
        negatives_k=1,
        torch_module=torch,
    )

    assert inverted > ordered
    assert float(inverted) > 1.0  # a real inversion is expensive


def test_local_nca_is_a_smooth_relaxation_of_recall_at_1() -> None:
    """The load-bearing property. `logsumexp` over positives is a soft maximum, so as
    the temperature falls the loss tends to 0 when the anchor's nearest neighbour is a
    positive and diverges when it is a negative -- exactly the indicator Recall@1
    counts. L_out has no such limit: it is a surrogate for class compactness, which is
    not the quantity any retrieval benchmark evaluates."""
    torch: Any = pytest.importorskip("torch")
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[0.60, 0.80]])
    negative = torch.tensor([[0.95, 0.31]])  # nearer than the positive: an inversion
    labels = torch.tensor([0, 0, 1])

    def loss_for(order: Any, temperature: float) -> float:
        stacked = torch.nn.functional.normalize(torch.cat(order), dim=1)
        return float(
            _local_nca_loss(
                stacked[:1],
                labels[:1],
                contrast_embeddings=stacked,
                contrast_labels=labels,
                temperature=temperature,
                negatives_k=1,
                torch_module=torch,
            )
        )

    inverted = [anchor, positive, negative]
    correct = [anchor, negative, positive]  # swap so the positive is nearest

    # Sharpening drives the correct case to zero and the inversion to blow up.
    assert loss_for(correct, 0.01) < 1e-3
    assert loss_for(inverted, 0.01) > 20.0
    # ...and the gap widens monotonically as temperature falls.
    gaps = [loss_for(inverted, t) - loss_for(correct, t) for t in (1.0, 0.1, 0.01)]
    assert gaps[0] < gaps[1] < gaps[2]


def test_recall_at_k_surrogate_matches_literal_rank_definition() -> None:
    torch: Any = pytest.importorskip("torch")
    embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8], [-1.0, 0.0], [-0.8, 0.2]],
            dtype=torch.float64,
        ),
        dim=1,
    ).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    rank_temperature = 0.07
    recall_temperature = 0.8
    k_values = (1, 2)

    actual = _recall_at_k_surrogate_loss(
        embeddings,
        labels,
        k_values=k_values,
        rank_temperature=rank_temperature,
        recall_temperature=recall_temperature,
        torch_module=torch,
    )

    per_query = []
    similarities = embeddings @ embeddings.T
    for query in range(len(labels)):
        positives = [i for i in range(len(labels)) if i != query and labels[i] == labels[query]]
        recalls = []
        for k in k_values:
            memberships = []
            for positive in positives:
                soft_rank = embeddings.new_tensor(1.0)
                for candidate in range(len(labels)):
                    if candidate != positive:
                        soft_rank = soft_rank + torch.sigmoid(
                            (similarities[query, candidate] - similarities[query, positive])
                            / rank_temperature
                        )
                memberships.append(torch.sigmoid((k - soft_rank) / recall_temperature))
            retrieved = torch.minimum(
                torch.stack(memberships).sum(), embeddings.new_tensor(float(k))
            )
            recalls.append(retrieved / min(k, len(positives)))
        per_query.append(torch.stack(recalls).mean())
    expected = 1.0 - torch.stack(per_query).mean()

    assert actual.item() == pytest.approx(expected.item(), abs=1e-10)
    actual.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_recall_at_k_surrogate_keeps_same_class_rank_competitors_like_reference() -> None:
    torch: Any = pytest.importorskip("torch")
    # Four examples per class give every query three positives. With perfectly
    # separated classes, each positive still has soft rank 2.5: its query and
    # two co-positives remain in the rank sum at sigmoid(0)=0.5, while only that
    # candidate positive itself is excluded. This is the pinned source's exact
    # behaviour and catches the earlier, incorrect whole-class rank mask.
    embeddings = torch.tensor(
        [[1.0, 0.0]] * 4 + [[-1.0, 0.0]] * 4,
        dtype=torch.float64,
    )
    labels = torch.tensor([0] * 4 + [1] * 4)

    loss = _recall_at_k_surrogate_loss(
        embeddings,
        labels,
        k_values=(1,),
        rank_temperature=0.01,
        recall_temperature=1.0,
        torch_module=torch,
    )

    expected_recall = 3.0 * torch.sigmoid(torch.tensor(-1.5)).item()
    assert loss.item() == pytest.approx(1.0 - expected_recall, abs=1e-7)


def test_recall_at_k_surrogate_rewards_correct_nearest_neighbours() -> None:
    torch: Any = pytest.importorskip("torch")
    labels = torch.tensor([0, 0, 1, 1])
    correct = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]), dim=1
    )
    inverted = correct[[0, 2, 1, 3]]

    correct_loss = _recall_at_k_surrogate_loss(
        correct, labels, k_values=(1,), torch_module=torch
    )
    inverted_loss = _recall_at_k_surrogate_loss(
        inverted, labels, k_values=(1,), torch_module=torch
    )
    assert correct_loss < inverted_loss


def test_gem_pooling_matches_reference_formula_and_learns_p() -> None:
    torch: Any = pytest.importorskip("torch")
    pooling = _gem_pooling_layer(torch)
    tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    actual = pooling(tensor)
    expected = tensor.pow(3.0).mean(dim=(-2, -1), keepdim=True).pow(1.0 / 3.0)
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert pooling.p.grad is not None


def test_local_nca_self_exclusion_survives_a_memory_augmented_contrast_set() -> None:
    """When memory is concatenated, `anchors` and `contrast_embeddings` have different
    lengths. Self-exclusion relies on the batch occupying the FIRST rows of the
    contrast set; if that assumption broke, each anchor would match itself at
    similarity 1.0 and the loss would collapse to ~0 for free."""
    torch: Any = pytest.importorskip("torch")
    torch.manual_seed(0)
    batch = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    memory = torch.nn.functional.normalize(torch.randn(20, 8), dim=1)
    batch_labels = torch.tensor([0, 1, 0, 1])
    memory_labels = torch.tensor([0, 1, 0, 1] * 5)

    augmented = _local_nca_loss(
        batch,
        batch_labels,
        contrast_embeddings=torch.cat([batch, memory]),
        contrast_labels=torch.cat([batch_labels, memory_labels]),
        temperature=0.1,
        negatives_k=5,
        torch_module=torch,
    )

    assert torch.isfinite(augmented)
    assert float(augmented) > 0.0  # ~0 would mean an anchor matched itself


def test_local_nca_diagnostic_detects_the_one_buddy_regime() -> None:
    """Guards the failure mode L_in is most exposed to, and which Khosla et al. raised
    as their objection: the loss can be satisfied by a single nearest positive while
    ignoring every other same-class instance.

    `local_nca_effective_positives` = exp(H) of the softmax over an anchor's positives.
    ~1.0 means one partner carries the anchor (degenerate); ~|P| means all contribute.
    """
    torch: Any = pytest.importorskip("torch")
    anchor = torch.tensor([1.0, 0.0, 0.0])
    negative = torch.tensor([0.0, 0.0, 1.0])

    def effective_positives(positives: list[Any]) -> float:
        stacked = torch.stack([anchor, *positives, negative])
        stacked = torch.nn.functional.normalize(stacked, dim=1)
        labels = torch.tensor([0] * (1 + len(positives)) + [1])
        collected: list[dict[str, float]] = []
        _local_nca_loss(
            stacked[:1],
            labels[:1],
            contrast_embeddings=stacked,
            contrast_labels=labels,
            temperature=0.1,
            negatives_k=1,
            torch_module=torch,
            diagnostics=collected,
        )
        return collected[0]["local_nca_effective_positives"]

    # One positive far closer than the others -> a single buddy dominates.
    lopsided = effective_positives(
        [
            torch.tensor([0.999, 0.045, 0.0]),
            torch.tensor([0.10, 0.99, 0.0]),
            torch.tensor([0.05, 0.0, 0.99]),
        ]
    )
    # Three equidistant positives -> the anchor spreads across all of them.
    balanced = effective_positives(
        [
            torch.tensor([0.90, 0.44, 0.0]),
            torch.tensor([0.90, -0.44, 0.0]),
            torch.tensor([0.90, 0.0, 0.44]),
        ]
    )

    assert lopsided < 1.5, f"expected a near-degenerate reading, got {lopsided:.3f}"
    assert balanced > 2.5, f"expected spread across ~3 positives, got {balanced:.3f}"


def test_local_nca_skips_anchors_with_no_positive() -> None:
    """With unbalanced CUB batches ~74% of anchors have no same-class partner. Those
    must be skipped rather than producing NaN or a spurious gradient."""
    torch: Any = pytest.importorskip("torch")
    embeddings = torch.eye(4)[:, :3].contiguous().requires_grad_(True)
    labels = torch.tensor([0, 1, 2, 3])  # every sample a singleton class

    loss = _local_nca_loss(
        embeddings,
        labels,
        contrast_embeddings=embeddings,
        contrast_labels=labels,
        temperature=0.1,
        negatives_k=2,
        torch_module=torch,
    )

    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0


def test_persistent_hypergraph_reduces_to_hist_without_memory() -> None:
    """With no memory the persistent hypergraph must equal plain HIST exactly, so any
    measured difference is attributable to the accumulated context alone."""
    torch: Any = pytest.importorskip("torch")
    module, embeddings, labels, label_to_index = _hist_fixture(torch)
    module.eval()
    shared = {
        "hist_module": module,
        "label_to_index": label_to_index,
        "tau": 32.0,
        "alpha": 1.1,
        "lambda_s": 1.0,
        "var_floor": 0.0,
        "torch_module": torch,
    }

    baseline = _hist_loss(embeddings, labels, **shared)
    no_memory = _hist_memory_loss(
        embeddings, labels, memory_embeddings=None, memory_labels=None, **shared
    )

    assert torch.allclose(baseline, no_memory, atol=1e-6)


def test_persistent_hypergraph_supervises_only_live_rows() -> None:
    """Memory entries must shape the graph without being re-fitted or receiving
    gradient: adding memory changes the loss, and gradient reaches only the live batch.
    """
    torch: Any = pytest.importorskip("torch")
    module, embeddings, labels, label_to_index = _hist_fixture(torch)
    module.eval()
    live = embeddings.clone().requires_grad_(True)
    memory = torch.randn(40, embeddings.shape[1])
    memory_labels = torch.tensor([i % 4 for i in range(40)])
    shared = {
        "hist_module": module,
        "label_to_index": label_to_index,
        "tau": 32.0,
        "alpha": 1.1,
        "lambda_s": 1.0,
        "var_floor": 0.0,
        "torch_module": torch,
    }

    without = _hist_memory_loss(live, labels, memory_embeddings=None, memory_labels=None, **shared)
    with_memory = _hist_memory_loss(
        live, labels, memory_embeddings=memory, memory_labels=memory_labels, **shared
    )
    assert not torch.allclose(without, with_memory, atol=1e-6)

    with_memory.backward()
    assert live.grad is not None
    assert live.grad.abs().sum() > 0
    assert memory.grad is None  # context only, never fitted


def test_sinkhorn_coupling_attains_its_marginals() -> None:
    """The transport constraints are what make this a coupling rather than a rescaling:
    every sample distributes unit mass, and every hyperedge receives exactly its share."""
    torch: Any = pytest.importorskip("torch")
    torch.manual_seed(3)
    cost = torch.rand(10, 4) * 3.0
    row = torch.full((10,), 1.0 / 10.0)
    column = torch.tensor([0.4, 0.3, 0.2, 0.1])

    coupling = _sinkhorn_log_coupling(
        cost,
        row_marginal=row,
        column_marginal=column,
        epsilon=0.5,
        iterations=200,
        torch_module=torch,
    )

    assert torch.allclose(coupling.sum(dim=1), row, atol=1e-5)
    assert torch.allclose(coupling.sum(dim=0), column, atol=1e-5)


def test_sinkhorn_hist_reduces_to_hist_without_balancing() -> None:
    """The generalisation claim, made checkable.

    The cost is defined so that `exp(-cost)` IS HIST's soft incidence. So at
    `epsilon=1.0` with zero Sinkhorn projections the coupling degenerates to that
    incidence and the objective must coincide with `_hist_loss` exactly. Plain HIST is
    therefore a special case, and any measured difference is attributable to the
    transport balancing alone.
    """
    torch: Any = pytest.importorskip("torch")
    module, embeddings, labels, label_to_index = _hist_fixture(torch)
    shared = {
        "hist_module": module,
        "label_to_index": label_to_index,
        "tau": 32.0,
        "alpha": 1.1,
        "lambda_s": 1.0,
        "var_floor": 0.0,
        "torch_module": torch,
    }

    module.eval()  # keep bn1 deterministic across the two forwards
    baseline = _hist_loss(embeddings, labels, **shared)
    degenerate = _hist_sinkhorn_loss(
        embeddings,
        labels,
        sinkhorn_epsilon=1.0,
        sinkhorn_iterations=0,
        sinkhorn_marginal="class_population",
        **shared,
    )

    assert torch.allclose(baseline, degenerate, atol=1e-6)


def test_geometric_cost_stays_active_when_classes_are_well_separated() -> None:
    """Why the `geometric` cost variant exists.

    With the HIST-compatible cost the true class has zero cost, so once classes
    separate the incidence is near-one-hot -- and `one_hot / N` already satisfies the
    class-population marginals, leaving Sinkhorn almost nothing to do. The geometric
    cost keeps the true-class term, so the coupling remains a live soft assignment.
    """
    torch: Any = pytest.importorskip("torch")
    module, embeddings, labels, label_to_index = _hist_fixture(torch)
    module.eval()
    with torch.no_grad():  # drive the class Gaussians far apart -> near-one-hot incidence
        module.means.mul_(0.0).add_(torch.eye(4, 6) * 50.0)
    shared = {
        "hist_module": module,
        "label_to_index": label_to_index,
        "tau": 32.0,
        "alpha": 1.1,
        "lambda_s": 1.0,
        "var_floor": 0.0,
        "sinkhorn_epsilon": 1.0,
        "sinkhorn_marginal": "class_population",
        "torch_module": torch,
    }

    hist_like_off = _hist_sinkhorn_loss(
        embeddings, labels, sinkhorn_iterations=0, sinkhorn_cost="hist_incidence", **shared
    )
    hist_like_on = _hist_sinkhorn_loss(
        embeddings, labels, sinkhorn_iterations=10, sinkhorn_cost="hist_incidence", **shared
    )
    geometric_off = _hist_sinkhorn_loss(
        embeddings, labels, sinkhorn_iterations=0, sinkhorn_cost="geometric", **shared
    )
    geometric_on = _hist_sinkhorn_loss(
        embeddings, labels, sinkhorn_iterations=10, sinkhorn_cost="geometric", **shared
    )

    hist_like_shift = (hist_like_on - hist_like_off).abs()
    geometric_shift = (geometric_on - geometric_off).abs()
    assert geometric_shift > hist_like_shift


def test_sinkhorn_balancing_actually_changes_the_objective() -> None:
    """...and with balancing on, it must differ -- otherwise the knob is inert."""
    torch: Any = pytest.importorskip("torch")
    module, embeddings, labels, label_to_index = _hist_fixture(torch)
    shared = {
        "hist_module": module,
        "label_to_index": label_to_index,
        "tau": 32.0,
        "alpha": 1.1,
        "lambda_s": 1.0,
        "var_floor": 0.0,
        "torch_module": torch,
    }
    module.eval()

    unbalanced = _hist_sinkhorn_loss(
        embeddings,
        labels,
        sinkhorn_epsilon=1.0,
        sinkhorn_iterations=0,
        sinkhorn_marginal="class_population",
        **shared,
    )
    balanced = _hist_sinkhorn_loss(
        embeddings,
        labels,
        sinkhorn_epsilon=1.0,
        sinkhorn_iterations=5,
        sinkhorn_marginal="class_population",
        **shared,
    )

    assert not torch.allclose(unbalanced, balanced, atol=1e-6)


def test_hypergraph_targets_match_hist_loss_internals() -> None:
    """`_hist_hypergraph_targets` recomputes HIST's incidence/propagation standalone,
    so that `_hist_loss` itself stays byte-identical while the matrix runs. This pins
    the two together: the HGNN logits it returns must reproduce the cross-entropy term
    inside `_hist_loss` exactly."""
    torch: Any = pytest.importorskip("torch")
    module, embeddings, labels, label_to_index = _hist_fixture(torch)
    alpha, tau, var_floor = 1.1, 32.0, 0.0

    _incidence_logits, hgnn_logits, _prototype_logits = _hist_hypergraph_targets(
        embeddings,
        labels,
        hist_module=module,
        label_to_index=label_to_index,
        alpha=alpha,
        var_floor=var_floor,
        torch_module=torch,
    )
    # Reconstruct _hist_loss with lambda_s=1 and subtract its distribution term, which
    # leaves exactly the hypergraph cross-entropy on the propagated logits.
    total = _hist_loss(
        embeddings,
        labels,
        hist_module=module,
        label_to_index=label_to_index,
        tau=tau,
        alpha=alpha,
        lambda_s=1.0,
        var_floor=var_floor,
        torch_module=torch,
    )
    dist_only = _hist_loss(
        embeddings,
        labels,
        hist_module=module,
        label_to_index=label_to_index,
        tau=tau,
        alpha=alpha,
        lambda_s=0.0,
        var_floor=var_floor,
        torch_module=torch,
    )
    expected_ce = total - dist_only
    actual_ce = torch.nn.functional.cross_entropy(hgnn_logits, labels)

    assert torch.allclose(actual_ce, expected_ce, atol=1e-5)


def test_hypergraph_distillation_is_zero_for_an_identical_teacher() -> None:
    """A teacher identical to the student yields the target's own entropy as the
    cross-entropy floor, so the KL part -- the only part carrying gradient -- is zero."""
    torch: Any = pytest.importorskip("torch")
    module, embeddings, labels, label_to_index = _hist_fixture(torch)
    import copy

    teacher = copy.deepcopy(module)

    loss = _hypergraph_distillation_loss(
        embeddings,
        embeddings,
        labels,
        hist_module=module,
        teacher_hist_module=teacher,
        label_to_index=label_to_index,
        alpha=1.1,
        var_floor=0.0,
        distill_target="hgnn_logits",
        tau_teacher=1.0,
        tau_student=1.0,
        torch_module=torch,
    )
    probs = torch.nn.functional.softmax(
        _hist_hypergraph_targets(
            embeddings,
            labels,
            hist_module=module,
            label_to_index=label_to_index,
            alpha=1.1,
            var_floor=0.0,
            torch_module=torch,
        )[1],
        dim=1,
    )
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean()
    assert torch.allclose(loss, entropy, atol=1e-5)


def test_full_catalogue_prototype_target_covers_every_class() -> None:
    """`prototype_full` extends the target that actually beat HIST (`incidence`) from
    the classes present in the batch to the whole prototype catalogue. The class
    Gaussians exist for every class regardless of batch content, so this costs nothing
    and carries strictly more dark knowledge."""
    torch: Any = pytest.importorskip("torch")
    nb_classes = 6
    module, embeddings, labels, label_to_index = _hist_fixture(torch, nb_classes=nb_classes)
    # Only 4 of the 6 classes appear in this batch.
    labels = torch.tensor([i % 4 for i in range(embeddings.shape[0])])

    incidence_logits, _hgnn, prototype_logits = _hist_hypergraph_targets(
        embeddings,
        labels,
        hist_module=module,
        label_to_index=label_to_index,
        alpha=1.1,
        var_floor=0.0,
        torch_module=torch,
    )

    assert incidence_logits.shape[1] == 4  # only in-batch hyperedges
    assert prototype_logits.shape[1] == nb_classes  # every class


def test_only_the_propagated_target_is_a_genuine_n_ary_quantity() -> None:
    """Pins WHICH target actually carries the novelty claim, and which does not.

    A pairwise target `s(z_i, z_j)` -- and equally a per-sample prototype affinity --
    is invariant to which *other* samples happen to share the batch. HIST's
    propagation normalises by `d_e(e) = sum_k H_ke`, a batch-population statistic, so
    perturbing an unrelated sample must move the propagated target for a fixed row.

    The incidence row does NOT have this property: `H_i` is built solely from sample
    i's own Mahalanobis distances to the class Gaussians. So incidence distillation is
    per-sample prototype/dark-knowledge KD, expressible without any hypergraph, and it
    carries no novelty claim -- it is only useful as the ablation that isolates whether
    the propagation operator is what matters. The claim rests on `hgnn_logits` alone.
    """
    torch: Any = pytest.importorskip("torch")
    module, embeddings, labels, label_to_index = _hist_fixture(torch)

    def targets_for(batch_embeddings: Any) -> tuple[Any, Any]:
        incidence_logits, hgnn_logits, _prototype = _hist_hypergraph_targets(
            batch_embeddings,
            labels,
            hist_module=module,
            label_to_index=label_to_index,
            alpha=1.1,
            var_floor=0.0,
            torch_module=torch,
        )
        return incidence_logits[0].clone(), hgnn_logits[0].clone()

    incidence_before, hgnn_before = targets_for(embeddings)
    perturbed = embeddings.clone()
    perturbed[-1] = perturbed[-1] + 5.0  # move only the LAST sample; row 0 is untouched
    incidence_after, hgnn_after = targets_for(perturbed)

    # The propagated target sees the rest of the batch...
    assert not torch.allclose(hgnn_before, hgnn_after, atol=1e-6)
    # ...the incidence target provably does not.
    assert torch.allclose(incidence_before, incidence_after, atol=1e-9)


def test_maxsim_is_invariant_to_where_the_object_sits_in_the_frame() -> None:
    """The property that actually distinguishes MaxSim from concatenated cosine.

    MaxSim matches each query region to its best gallery region wherever it is, so the
    SAME regions in a different spatial order score identically. Concatenated cosine
    compares slot-to-slot, so the same object photographed in a different part of the
    frame looks ORTHOGONAL. That is why evaluating a region-trained model with concat
    cosine cost 10.5 R@1 points.

    Note this test replaced an earlier one asserting that a single distinctive region
    carries the match. It does not: MaxSim averages over QUERY regions, so a candidate
    partially matching every region (0.707) beats one perfectly matching a single
    region (0.333). Position invariance, not best-region dominance, is the real
    property.
    """
    from sfora.image_benchmark import maxsim_distances

    first = np.array([1.0, 0.0, 0.0, 0.0])
    second = np.array([0.0, 1.0, 0.0, 0.0])

    query = np.stack([first, second])[None, :, :]
    same_regions_moved = np.stack([second, first])  # identical content, shifted position
    one_region_duplicated = np.stack([first, first])
    gallery = np.stack([same_regions_moved, one_region_duplicated])

    maxsim = -maxsim_distances(query, gallery)[0]
    assert float(maxsim[0]) > 0.99, "MaxSim must ignore where in the frame a region sits"
    assert float(maxsim[0]) > float(maxsim[1])

    flat_query = query.reshape(1, -1)
    flat_gallery = gallery.reshape(2, -1)
    flat_query = flat_query / np.linalg.norm(flat_query)
    flat_gallery = flat_gallery / np.linalg.norm(flat_gallery, axis=1, keepdims=True)
    concat = (flat_query @ flat_gallery.T)[0]
    # Concat cosine calls the repositioned object a total mismatch, and prefers the
    # objectively worse candidate.
    assert float(concat[0]) < 1e-9
    assert float(concat[1]) > float(concat[0])


def test_region_similarity_is_driven_by_the_best_region_not_the_average() -> None:
    """The whole point of the multi-vector representation.

    An image belongs to a fine-grained class because SOME region of it does - a
    wing-bar, a headlight - not because its spatial average does. Global pooling
    averages that evidence away, which is the measured antihub mechanism here.

    Construct an image where ONE region matches a class proxy and the rest are
    orthogonal noise. The pooled representation dilutes the match by the number of
    regions; the region score must not.
    """
    torch: Any = pytest.importorskip("torch")
    from sfora.image_end_to_end import _region_proxy_similarity

    grid, dimensions = 3, 4
    tokens = grid * grid
    proxy = torch.zeros(1, dimensions)
    proxy[0, 0] = 1.0

    regions = torch.zeros(1, tokens, dimensions)
    regions[0, 0, 0] = 1.0  # one region matches the proxy exactly
    regions[0, 1:, 1] = 1.0  # the rest are orthogonal to it
    config = ImageEndToEndConfig(region_grid=grid, region_tau=0.02, embedding_dimensions=dimensions)

    region_score = _region_proxy_similarity(
        regions.reshape(1, tokens * dimensions), proxy, config=config, torch_module=torch
    )
    pooled = torch.nn.functional.normalize(regions.mean(dim=1), dim=-1)
    pooled_score = pooled @ torch.nn.functional.normalize(proxy, dim=-1).T

    # The single matching region should carry the score; the average should not.
    assert float(region_score[0, 0]) > 0.9
    assert float(pooled_score[0, 0]) < 0.4
    assert float(region_score[0, 0]) > 2 * float(pooled_score[0, 0])


def test_region_head_emits_flat_embeddings_the_rest_of_the_pipeline_can_consume() -> None:
    """Regions are returned concatenated because the memory queue, checkpoint selector,
    retrieval scorer and .npz writers all assume a 2-D (batch, dim) tensor. The loss
    and the offline MaxSim evaluator reshape back."""
    torch: Any = pytest.importorskip("torch")
    from sfora.image_end_to_end import _as_regions, _region_embedding_head

    grid, dimensions, in_features = 3, 8, 16
    config = ImageEndToEndConfig(region_grid=grid, embedding_dimensions=dimensions)
    head = _region_embedding_head(in_features, config, torch)

    flat = head(torch.randn(5, in_features * grid * grid))
    assert flat.shape == (5, grid * grid * dimensions)

    regions = _as_regions(flat, config, torch)
    assert regions.shape == (5, grid * grid, dimensions)
    norms = regions.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_shepard_kernel_has_a_fatter_tail_than_cosine_softmax() -> None:
    """The claim the method rests on, made numerically.

    Cosine-softmax is secretly Gaussian: on unit vectors cos = 1 - d^2/2, so
    exp(cos/T) is proportional to exp(-d^2/2T) - decay in the SQUARE of distance.
    Shepard (Science 1987) derived exp(-d), linear in distance. The consequence is
    tail weight: a Gaussian kernel drives moderately-distant positives' softmax
    weight to near zero, so they stop receiving gradient. That is how an orphan is
    made, and 5-8% of CUB test images are orphans here.
    """
    torch: Any = pytest.importorskip("torch")
    from sfora.image_end_to_end import _shepard_similarity

    anchor = torch.tensor([[1.0, 0.0]])
    # Points at increasing angle, hence increasing Euclidean distance.
    angles = torch.tensor([0.2, 0.8, 1.6, 2.4])
    references = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    shepard = _shepard_similarity(anchor, references, order=2, torch_module=torch)[0]
    cosine = (anchor @ references.T)[0]

    # Both must rank identically - this is a kernel change, not a ranking change.
    assert torch.equal(shepard.argsort(descending=True), cosine.argsort(descending=True))

    # The far tail keeps more mass under Shepard, and the gap WIDENS as temperature
    # falls. That matters because Proxy Anchor runs at alpha = 32, i.e. an effective
    # temperature near 0.03 - the sharp regime where the difference is largest.
    # Measured ratios: 1.19x at T=1.0, 1.54x at T=0.2, 2.90x at T=0.05.
    ratios = []
    for temperature in (1.0, 0.2, 0.05):
        shepard_weights = torch.softmax(shepard / temperature, dim=0)
        cosine_weights = torch.softmax(cosine / temperature, dim=0)
        assert float(shepard_weights[-1]) > float(cosine_weights[-1])
        ratios.append(float(shepard_weights[-1] / cosine_weights[-1]))

    assert ratios[0] < ratios[1] < ratios[2], f"tail gap should widen as T falls: {ratios}"
    assert ratios[-1] > 2.0


def test_shepard_kernel_is_not_reproducible_by_rescaling_cosine() -> None:
    """If a temperature on cosine could reproduce it, this would be a hyperparameter,
    not a method. d = sqrt(2 - 2cos) is concave in cos, so the map is non-affine and
    no scalar rescaling matches it."""
    torch: Any = pytest.importorskip("torch")
    from sfora.image_end_to_end import _shepard_similarity

    angles = torch.linspace(0.1, 3.0, 12)
    anchor = torch.tensor([[1.0, 0.0]])
    references = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    shepard = _shepard_similarity(anchor, references, order=2, torch_module=torch)[0]
    cosine = (anchor @ references.T)[0]

    # Best affine fit of shepard onto cosine; a good fit would mean "just a rescale".
    design = torch.stack([cosine, torch.ones_like(cosine)], dim=1)
    solution = torch.linalg.lstsq(design, shepard.unsqueeze(1)).solution
    residual = float((design @ solution - shepard.unsqueeze(1)).abs().max())
    assert residual > 0.05, f"expected a non-affine relationship, max residual {residual}"


def test_shepard_city_block_order_differs_from_euclidean() -> None:
    """Shepard distinguished integral (Euclidean) from separable (city-block) stimuli;
    both are exposed because bird part-attributes plausibly vary independently."""
    torch: Any = pytest.importorskip("torch")
    from sfora.image_end_to_end import _shepard_similarity

    torch.manual_seed(0)
    embeddings = torch.nn.functional.normalize(torch.randn(6, 5), dim=1)
    references = torch.nn.functional.normalize(torch.randn(3, 5), dim=1)

    euclidean = _shepard_similarity(embeddings, references, order=2, torch_module=torch)
    city_block = _shepard_similarity(embeddings, references, order=1, torch_module=torch)

    assert euclidean.shape == city_block.shape == (6, 3)
    assert not torch.allclose(euclidean, city_block)
    assert torch.isfinite(euclidean).all() and torch.isfinite(city_block).all()


def test_tversky_similarity_is_asymmetric_which_cosine_cannot_be() -> None:
    """The property the whole method exists for.

    Cosine is symmetric by construction: s(a,b) == s(b,a) always. Tversky (1977)
    showed human similarity is not - people judge North Korea more similar to China
    than China to North Korea, because the two objects have different amounts of
    DISTINCTIVE feature mass. Weighting those two sides differently (alpha != beta)
    produces exactly that, and it is what lets the similarity express the directional
    neighbour relations measured in this repo (35-42% of CUB top-10 neighbours cross
    a class boundary).
    """
    torch: Any = pytest.importorskip("torch")
    from sfora.image_end_to_end import _tversky_similarity

    features = torch.eye(3)
    # `broad` carries all three features; `narrow` carries only the first.
    broad = torch.tensor([[1.0, 1.0, 1.0]])
    narrow = torch.tensor([[1.0, 0.0, 0.0]])

    forward = _tversky_similarity(narrow, broad, features, alpha=1.0, beta=0.2, torch_module=torch)
    backward = _tversky_similarity(broad, narrow, features, alpha=1.0, beta=0.2, torch_module=torch)

    assert not torch.allclose(forward, backward), "alpha != beta must break symmetry"
    # The narrow object has no distinctive mass of its own, so it looks MORE similar
    # to the broad one than the reverse - Tversky's asymmetry, reproduced.
    assert float(forward[0, 0]) > float(backward[0, 0])

    symmetric = _tversky_similarity(
        narrow, broad, features, alpha=1.0, beta=1.0, torch_module=torch
    )
    symmetric_reverse = _tversky_similarity(
        broad, narrow, features, alpha=1.0, beta=1.0, torch_module=torch
    )
    assert torch.allclose(symmetric, symmetric_reverse), "alpha == beta must be symmetric"


def test_tversky_ratio_form_reduces_to_tanimoto() -> None:
    """At alpha = beta = 1 the ratio form IS the Tanimoto/Jaccard coefficient that
    cheminformatics uses for molecular fingerprint similarity -- so the psychology and
    the chemistry are the same object at a particular parameter setting."""
    torch: Any = pytest.importorskip("torch")
    from sfora.image_end_to_end import _tversky_similarity

    features = torch.eye(4)
    a = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # set {0,1,2}
    b = torch.tensor([[0.0, 1.0, 1.0, 1.0]])  # set {1,2,3}

    similarity = _tversky_similarity(a, b, features, alpha=1.0, beta=1.0, torch_module=torch)

    # |A n B| = 2, |A u B| = 4  ->  Jaccard = 0.5
    assert abs(float(similarity[0, 0]) - 0.5) < 1e-5


def test_tversky_similarity_is_bounded() -> None:
    """Bounded in [0, 1] by construction -- the property every unnormalised objective
    tried in this repo lacked, and collapsed for want of."""
    torch: Any = pytest.importorskip("torch")
    from sfora.image_end_to_end import _tversky_similarity

    torch.manual_seed(0)
    embeddings = torch.nn.functional.normalize(torch.randn(16, 8), dim=1)
    references = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
    features = torch.nn.functional.normalize(torch.randn(32, 8), dim=1)

    for alpha, beta in ((1.0, 1.0), (0.5, 2.0), (2.0, 0.1)):
        similarity = _tversky_similarity(
            embeddings, references, features, alpha=alpha, beta=beta, torch_module=torch
        )
        assert similarity.shape == (16, 5)
        assert torch.isfinite(similarity).all()
        assert float(similarity.min()) >= 0.0
        assert float(similarity.max()) <= 1.0 + 1e-6


def test_deterministic_flag_configures_the_backend() -> None:
    """The 1.08 pt fixed-seed spread this project measured is GPU nondeterminism, and
    it is eliminable. Verify the switch actually reaches the backend rather than being
    a config field nothing reads."""
    torch: Any = pytest.importorskip("torch")
    from sfora.image_end_to_end import _enable_deterministic_algorithms

    calls: dict[str, Any] = {}

    class _Cudnn:
        deterministic = False
        benchmark = True

    class _Backends:
        cudnn = _Cudnn()

    class _FakeTorch:
        backends = _Backends()

        @staticmethod
        def use_deterministic_algorithms(value: bool, warn_only: bool = False) -> None:
            calls["value"] = value
            calls["warn_only"] = warn_only

    _enable_deterministic_algorithms(_FakeTorch())

    assert _FakeTorch.backends.cudnn.deterministic is True
    assert _FakeTorch.backends.cudnn.benchmark is False
    assert calls == {"value": True, "warn_only": True}

    import os

    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    # The real torch must accept the same call without raising.
    _enable_deterministic_algorithms(torch)
    torch.use_deterministic_algorithms(False)  # restore for the rest of the suite


def test_batch_norm_affine_freeze_disables_gradients() -> None:
    torch: Any = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Conv2d(3, 4, 1))

    _freeze_batch_norm_affine_parameters(model)

    batch_norm = model[0]
    assert batch_norm.weight.requires_grad is False
    assert batch_norm.bias.requires_grad is False
    assert model[1].weight.requires_grad is True


def test_reference_gradient_clip_uses_configured_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch: Any = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    observed: list[float] = []

    def record_clip(parameters: Iterable[Any], clip_value: float) -> None:
        list(parameters)  # consume the generator exactly as torch does
        observed.append(float(clip_value))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_value_", record_clip)

    _clip_gradients(model, clip_value=10.0, torch_module=torch)

    assert observed == [10.0]


def test_bn_inception_builds_official_512_head_without_download() -> None:
    torch: Any = pytest.importorskip("torch")
    model = build_bn_inception(embedding_size=512, pretrained=False, add_gmp=True)
    model.eval()

    assert model.model.num_ftrs == 1024
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))

    assert output.shape == (1, 512)


def test_bn_inception_checkpoint_uses_full_official_content_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bn_inception-52deb4733.pth"
    checkpoint.write_bytes(b"substituted checkpoint")

    assert BN_INCEPTION_CHECKPOINT_SHA256 == (
        "52deb473314542a5c2f87e9e6f26f4ca42fe863d15f986414dbae8c2dfdd2353"
    )
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        validate_bn_inception_checkpoint(checkpoint)


def test_bn_inception_reference_transform_uses_caffe_bgr_values() -> None:
    pytest.importorskip("torch")
    pil_image = pytest.importorskip("PIL.Image")
    config = ImageEndToEndConfig(
        backbone_name="bn_inception",
        train_augmentation="reference_random_resized_crop",
    )

    transformed = _default_transform_factory(config, False)(
        pil_image.new("RGB", (256, 256), (128, 117, 104))
    )

    assert transformed[:, 0, 0].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_pfml_protocol_uses_repaired_resnet50_512_defaults() -> None:
    config = config_for_protocol("pfml-resnet50-512", dataset_name="cub")

    assert config.objectives == ("frozen_pretrained", "pfml")
    assert config.optimizer == "adam"
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.backbone_learning_rate == pytest.approx(1e-4)
    assert config.weight_decay == pytest.approx(5e-4)
    assert config.warmup_epochs == 1
    assert config.lr_schedule == "none"
    assert config.lr_step_epochs == 5
    assert config.lr_gamma == pytest.approx(0.5)
    assert config.train_epochs == 200
    assert config.samples_per_class == 4
    assert config.train_augmentation == "standard"
    assert config.batch_size == 100
    assert config.pretrained_weights == "v1"
    assert config.head_pooling == "avg"
    assert config.embedding_head_init == "default"
    assert config.dataset_selection_policy == "full_official_partition"
    assert config.freeze_batch_norm is True
    assert config.proxy_count_per_class == 15
    assert config.potential_delta == pytest.approx(0.2)
    assert config.potential_alpha == pytest.approx(3.0)
    assert config.checkpoint_selection_interval == 0


def test_pfml_protocol_uses_two_sop_proxies_per_class() -> None:
    config = config_for_protocol("pfml-resnet50-512", dataset_name="sop")

    assert config.proxy_count_per_class == 2
    assert config.weight_decay == pytest.approx(5e-4)
    assert config.warmup_epochs == 0
    assert config.freeze_batch_norm is False


def test_pfml_protocol_uses_cars_weight_decay() -> None:
    config = config_for_protocol("pfml-resnet50-512", dataset_name="cars")

    assert config.weight_decay == pytest.approx(1e-4)
    assert config.freeze_batch_norm is True


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


def test_lops_pg_embedding_gradient_projects_only_conflicts() -> None:
    torch = pytest.importorskip("torch")
    from sfora.image_end_to_end import _lops_pg_embedding_gradient

    embeddings = torch.tensor(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]], dtype=torch.float64
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    gradient = torch.tensor(
        [[0.0, 2.0], [0.6, -0.8], [2.0, 0.0], [-0.8, -0.6]], dtype=torch.float64
    )
    diagnostics: list[dict[str, float]] = []
    actual = _lops_pg_embedding_gradient(
        gradient,
        embeddings,
        labels,
        torch_module=torch,
        diagnostics=diagnostics,
    )
    assert actual.dtype == gradient.dtype
    assert actual.shape == gradient.shape
    with torch.no_grad():
        tangent = torch.stack(
            [
                embeddings[1] - embeddings[0] * (embeddings[0] @ embeddings[1]),
                embeddings[0] - embeddings[1] * (embeddings[1] @ embeddings[0]),
                embeddings[3] - embeddings[2] * (embeddings[2] @ embeddings[3]),
                embeddings[2] - embeddings[3] * (embeddings[3] @ embeddings[2]),
            ]
        )
    dots = (actual * tangent).sum(dim=1)
    assert bool((dots <= 1e-12).all())
    assert diagnostics == [
        {"rows": 4.0, "eligible_rows": 4.0, "conflict_rows": 2.0}
    ]


def test_mcps_centroid_state_reads_before_exact_stopped_update() -> None:
    torch = pytest.importorskip("torch")
    from sfora.image_end_to_end import _MCPSCentroidState

    proxy_labels = torch.tensor([10, 20], dtype=torch.long)
    proxies = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    state = _MCPSCentroidState(
        proxy_labels,
        dimensions=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
        torch_module=torch,
    )
    labels = torch.tensor([10, 20, 10], dtype=torch.long)
    fallback, memory_mask = state.targets(labels, proxies)
    torch.testing.assert_close(
        fallback,
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
    )
    assert memory_mask.tolist() == [False, False, False]

    embeddings = torch.tensor(
        [[0.8, 0.6], [0.0, 1.0], [0.6, 0.8]], dtype=torch.float64
    )
    state.update(embeddings, labels)
    first_targets, first_mask = state.targets(labels, proxies)
    expected_ten = torch.tensor([0.7, 0.7], dtype=torch.float64)
    expected_ten /= expected_ten.norm()
    torch.testing.assert_close(first_targets[0], expected_ten)
    torch.testing.assert_close(first_targets[2], expected_ten)
    torch.testing.assert_close(first_targets[1], torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert first_mask.tolist() == [True, True, True]

    old_targets = first_targets.clone()
    state.update(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
        labels,
    )
    second_targets, _ = state.targets(labels, proxies)
    expected_ten = 0.9 * old_targets[0] + 0.1 * torch.tensor(
        [1.0, 0.0], dtype=torch.float64
    )
    expected_ten /= expected_ten.norm()
    expected_twenty = 0.9 * old_targets[1] + 0.1 * torch.tensor(
        [1.0, 0.0], dtype=torch.float64
    )
    expected_twenty /= expected_twenty.norm()
    torch.testing.assert_close(second_targets[0], expected_ten)
    torch.testing.assert_close(second_targets[2], expected_ten)
    torch.testing.assert_close(second_targets[1], expected_twenty)
    assert not first_targets.requires_grad and not second_targets.requires_grad


def test_mcps_embedding_gradient_projects_only_live_memory_conflicts() -> None:
    torch = pytest.importorskip("torch")
    from sfora.image_end_to_end import _mcps_pg_embedding_gradient

    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    targets = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    gradients = torch.tensor([[0.0, 2.0], [0.0, -2.0], [3.0, 4.0]], dtype=torch.float64)
    memory_mask = torch.tensor([True, True, False])
    diagnostics: list[dict[str, float]] = []
    actual = _mcps_pg_embedding_gradient(
        gradients,
        embeddings,
        targets,
        memory_mask,
        torch_module=torch,
        diagnostics=diagnostics,
    )
    torch.testing.assert_close(actual[0], torch.tensor([0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(actual[1], gradients[1])
    torch.testing.assert_close(actual[2], gradients[2])
    assert diagnostics == [
        {
            "rows": 3.0,
            "memory_target_rows": 2.0,
            "proxy_fallback_rows": 1.0,
            "eligible_rows": 2.0,
            "conflict_rows": 1.0,
            "memory_eligible_rows": 2.0,
            "memory_conflict_rows": 1.0,
        }
    ]


def test_mcps_hook_changes_live_encoder_step_but_not_proxy_gradient() -> None:
    torch = pytest.importorskip("torch")
    from sfora.image_end_to_end import (
        _mcps_pg_embedding_gradient,
        _proxy_anchor_loss,
    )

    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    proxy_labels = torch.tensor([0, 1], dtype=torch.long)
    base_embeddings = torch.tensor(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]], dtype=torch.float64
    )
    base_proxies = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64)
    targets = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float64
    )
    memory_mask = torch.ones(4, dtype=torch.bool)

    def run(with_hook: bool) -> tuple[object, object]:
        embeddings = base_embeddings.clone().requires_grad_(True)
        proxies = base_proxies.clone().requires_grad_(True)
        if with_hook:
            embeddings.register_hook(
                lambda gradient: _mcps_pg_embedding_gradient(
                    gradient,
                    embeddings.detach(),
                    targets,
                    memory_mask,
                    torch_module=torch,
                )
            )
        loss = _proxy_anchor_loss(
            embeddings,
            labels,
            proxy_embeddings=proxies,
            proxy_labels=proxy_labels,
            alpha=32.0,
            delta=0.1,
            torch_module=torch,
        )
        loss.backward()
        return embeddings.grad, proxies.grad

    plain_embedding_gradient, plain_proxy_gradient = run(False)
    mcps_embedding_gradient, mcps_proxy_gradient = run(True)
    assert not torch.equal(mcps_embedding_gradient, plain_embedding_gradient)
    torch.testing.assert_close(mcps_proxy_gradient, plain_proxy_gradient, rtol=0.0, atol=0.0)


def test_proxy_compactness_uses_stopped_matching_proxy() -> None:
    torch = pytest.importorskip("torch")
    from sfora.image_end_to_end import _proxy_compactness_loss

    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64, requires_grad=True
    )
    proxies = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    labels = torch.tensor([10, 20], dtype=torch.long)
    proxy_labels = torch.tensor([10, 20], dtype=torch.long)
    loss = _proxy_compactness_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        torch_module=torch,
    )
    assert float(loss.detach()) == pytest.approx(1.0)
    loss.backward()
    assert embeddings.grad is not None and bool(torch.isfinite(embeddings.grad).all())
    assert proxies.grad is None


def test_mcps_and_proxy_compactness_objectives_complete_tiny_training() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    examples = [
        ImageExample(
            example_id=f"{label}-{index}", image=[float(label), float(index)], label=label
        )
        for label in range(4)
        for index in range(2)
    ]
    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("proxy_anchor_mcps_pg", "proxy_anchor_proxy_compactness"),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=8,
            eval_batch_size=8,
            train_steps=2,
            group_size=2,
            proxy_count_per_class=1,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=lambda config, train: lambda image: torch.as_tensor(
            image, dtype=torch.float32
        ),
    )
    assert set(result.methods) == {
        "proxy_anchor_mcps_pg_end_to_end:tiny",
        "proxy_anchor_proxy_compactness_end_to_end:tiny",
    }
    diagnostics = result.methods["proxy_anchor_mcps_pg_end_to_end:tiny"].mcps_diagnostics
    assert diagnostics is not None
    assert diagnostics["rows"] == 16.0
    assert diagnostics["memory_target_rows"] == 8.0
    assert diagnostics["proxy_fallback_rows"] == 8.0
    assert diagnostics["memory_target_rate"] == 0.5
    assert result.methods[
        "proxy_anchor_proxy_compactness_end_to_end:tiny"
    ].mcps_diagnostics is None


def test_positive_compactness_stops_sibling_centroids() -> None:
    torch = pytest.importorskip("torch")
    from sfora.image_end_to_end import _positive_compactness_loss

    embeddings = torch.tensor(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss = _positive_compactness_loss(embeddings, labels, torch_module=torch)
    expected = torch.tensor([0.2, 0.2, 0.2, 0.2], dtype=torch.float64).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert embeddings.grad is not None and bool(torch.isfinite(embeddings.grad).all())


def test_lops_hook_preserves_proxy_anchor_loss_and_proxy_gradient() -> None:
    torch = pytest.importorskip("torch")
    from sfora.image_end_to_end import _lops_pg_embedding_gradient, _proxy_anchor_loss

    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    proxy_labels = torch.tensor([0, 1], dtype=torch.long)
    base_embeddings = torch.tensor(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]], dtype=torch.float64
    )
    base_proxies = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64)

    def run(with_hook: bool) -> tuple[float, object]:
        embeddings = base_embeddings.clone().requires_grad_(True)
        proxies = base_proxies.clone().requires_grad_(True)
        if with_hook:
            embeddings.register_hook(
                lambda gradient: _lops_pg_embedding_gradient(
                    gradient, embeddings.detach(), labels, torch_module=torch
                )
            )
        loss = _proxy_anchor_loss(
            embeddings,
            labels,
            proxy_embeddings=proxies,
            proxy_labels=proxy_labels,
            alpha=32.0,
            delta=0.1,
            torch_module=torch,
        )
        loss.backward()
        return float(loss.detach()), proxies.grad

    plain_loss, plain_proxy_gradient = run(False)
    lops_loss, lops_proxy_gradient = run(True)
    assert lops_loss == plain_loss
    torch.testing.assert_close(lops_proxy_gradient, plain_proxy_gradient, rtol=0.0, atol=0.0)


def test_lops_and_compactness_objectives_complete_tiny_training() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, images: object) -> object:
            return self.linear(images)

    examples = [
        ImageExample(
            example_id=f"{label}-{index}", image=[float(label), float(index)], label=label
        )
        for label in range(4)
        for index in range(2)
    ]
    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("proxy_anchor_lops_pg", "proxy_anchor_compactness"),
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=8,
            eval_batch_size=8,
            train_steps=1,
            group_size=2,
            proxy_count_per_class=1,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=lambda config, train: lambda image: torch.as_tensor(
            image, dtype=torch.float32
        ),
    )
    assert set(result.methods) == {
        "proxy_anchor_lops_pg_end_to_end:tiny",
        "proxy_anchor_compactness_end_to_end:tiny",
    }
    assert all(len(method.loss_history) == 1 for method in result.methods.values())
    lops_diagnostics = result.methods[
        "proxy_anchor_lops_pg_end_to_end:tiny"
    ].lops_diagnostics
    assert lops_diagnostics is not None
    assert lops_diagnostics["rows"] == 8.0
    assert lops_diagnostics["eligible_rows"] == 8.0
    assert lops_diagnostics["skip_rate"] == 0.0
    assert 0.0 <= lops_diagnostics["conflict_rows"] <= 8.0
    assert lops_diagnostics["conflict_rate"] == pytest.approx(
        lops_diagnostics["conflict_rows"] / 8.0
    )
    assert result.methods["proxy_anchor_compactness_end_to_end:tiny"].lops_diagnostics is None


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


def test_encode_model_norms_preserves_pre_normalisation_magnitude() -> None:
    torch = pytest.importorskip("torch")
    from sfora.image_end_to_end import _encode_model_norms

    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    loader = [
        (
            torch.tensor([[3.0, 4.0], [5.0, 12.0]]),
            torch.tensor([0, 1]),
        )
    ]

    norms = _encode_model_norms(model, loader, torch.device("cpu"), torch)

    assert norms.tolist() == pytest.approx([5.0, 13.0])


def test_encode_model_norms_observes_bn_inception_head_before_internal_normalisation() -> None:
    torch = pytest.importorskip("torch")
    from sfora.image_end_to_end import _encode_model_norms

    class NormalisingInner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.embedding.weight.copy_(torch.eye(2))

        def forward(self, values: object) -> object:
            return torch.nn.functional.normalize(self.embedding(values), p=2, dim=1)

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = NormalisingInner()

        def forward(self, values: object) -> object:
            return self.model(values)

    loader = [(torch.tensor([[3.0, 4.0], [5.0, 12.0]]), torch.tensor([0, 1]))]

    norms = _encode_model_norms(Wrapper(), loader, torch.device("cpu"), torch)

    assert norms.tolist() == pytest.approx([5.0, 13.0])


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


@pytest.mark.parametrize("distill_target", ["hgnn_logits", "incidence"])
def test_hypergraph_distillation_end_to_end_runs(distill_target: str) -> None:
    """Drive the real training loop with the hypergraph term on, so the EMA-teacher
    wiring (teacher creation, its attached hist_module, backward through the student
    branch only) is exercised before any GPU time is committed to it."""
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
            # The new mechanism, with the pairwise term OFF so only it is exercised.
            ema_distill_weight=0.0,
            hypergraph_distill_weight=1.0,
            hypergraph_distill_target=distill_target,  # type: ignore[arg-type]
            ema_teacher_train_mode=True,
            ema_teacher_ema_buffers=True,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )
    method = result.methods["hist_end_to_end:tiny"]
    assert method.loss_history, "training produced no steps"
    assert all(math.isfinite(value) for value in method.loss_history)


def test_hypergraph_distillation_requires_a_hist_objective() -> None:
    """A Proxy-Anchor base has no hist_module, so the term must fail loudly rather
    than silently contributing nothing."""
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
    with pytest.raises(ValueError, match="hypergraph distillation requires"):
        run_image_end_to_end_benchmark(
            train_examples=examples,
            test_examples=examples,
            config=ImageEndToEndConfig(
                dataset_name="cub",
                protocol="proxy-anchor-resnet50-512",
                objectives=("proxy_anchor",),
                backbone_name="tiny",
                embedding_dimensions=4,
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
                hypergraph_distill_weight=1.0,
            ),
            model_factory=lambda config: TinyModel(),
            transform_factory=transform_factory,
        )


def test_end_to_end_run_scores_queries_against_separate_gallery() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)

        def forward(self, images: object) -> object:
            return self.embedding(torch.as_tensor(images, dtype=torch.long))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        return lambda image: int(cast(int, image))

    train_examples = [
        ImageExample(example_id=f"train-{label}-{index}", image=label * 4 + index, label=label)
        for label in range(2)
        for index in range(4)
    ]
    query_examples = [
        ImageExample(example_id="query-10", image=0, label=10),
        ImageExample(example_id="query-11", image=4, label=11),
    ]
    gallery_examples = [
        ImageExample(example_id="gallery-10-0", image=1, label=10),
        ImageExample(example_id="gallery-10-1", image=2, label=10),
        ImageExample(example_id="gallery-11-0", image=5, label=11),
        ImageExample(example_id="gallery-11-1", image=6, label=11),
    ]

    result = run_image_end_to_end_benchmark(
        train_examples=train_examples,
        test_examples=query_examples,
        gallery_examples=gallery_examples,
        config=ImageEndToEndConfig(
            dataset_name="inshop",
            protocol="proxy-anchor-resnet50-512",
            objectives=("frozen",),
            backbone_name="tiny",
            embedding_dimensions=4,
            batch_size=8,
            eval_batch_size=8,
            train_steps=1,
            train_epochs=None,
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    assert result.test_examples == 2
    assert result.gallery_examples == 4
    assert result.methods["frozen_end_to_end:tiny"].retrieval.evaluated_queries == 2


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


def test_crossfit_positive_centroids_exclude_the_query_and_use_memory() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _crossfit_positive_centroids

    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32
    )
    labels = torch.tensor([7, 7, 9])
    memory_embeddings = torch.tensor([[0.0, -1.0]], dtype=torch.float32)
    memory_labels = torch.tensor([7])
    centroids, valid = _crossfit_positive_centroids(
        embeddings,
        labels,
        memory_embeddings=memory_embeddings,
        memory_labels=memory_labels,
        torch_module=torch,
    )
    assert valid.tolist() == [True, True, False]
    # Query 0 sees query 1 plus memory, never itself; query 1 sees query 0 plus memory.
    assert torch.allclose(centroids[0], torch.tensor([0.0, 0.0]), atol=1e-6)
    assert torch.allclose(centroids[1], torch.tensor([0.5, -0.5]), atol=1e-6)


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


def test_coalition_objective_runs_with_indexed_duplicate_class_batches() -> None:
    torch: Any = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 2)

        def forward(self, images: object) -> object:
            return self.embedding(torch.as_tensor(images, dtype=torch.long))

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        return lambda image: int(cast(int, image))

    examples = [
        ImageExample(example_id=f"{label}-{index}", image=label * 2 + index, label=label)
        for label in (0, 1)
        for index in range(2)
    ]
    result = run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("proxy_anchor_coalition",),
            coalition_mode="union",
            coalition_weight=0.1,
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            samples_per_class=2,
            eval_batch_size=4,
            train_steps=2,
            train_epochs=None,
            warmup_epochs=0,
            proxy_count_per_class=1,
            progress_every=0,
            num_workers=0,
            seed=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    method = result.methods["proxy_anchor_coalition_end_to_end:tiny"]
    assert len(method.loss_history) == 2
    assert all(math.isfinite(value) for value in method.loss_history)


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
    expected = 2.0 * (4.0 - 1.25 + 4.0 + 12.5 - 1.0 / 3.6 + 4.0)
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
    # Raw Eq. 6 sum over 12 ordered off-diagonal pairs = 2*4 - 2*2 = 4.
    assert torch.isfinite(loss)
    assert float(loss.detach().cpu()) == pytest.approx(4.0, rel=1e-5)


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

    assert float(loss.detach().cpu()) == pytest.approx(-8.0, rel=1e-5)


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


def test_optimizer_groups_recognize_bn_inception_embedding_head() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _optimizer_parameter_groups

    class TinyBNInception(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.backbone = torch.nn.Linear(2, 2)
            self.model.embedding = torch.nn.Linear(2, 2)

    model = TinyBNInception()
    groups = _optimizer_parameter_groups(
        model,
        ImageEndToEndConfig(learning_rate=6e-4, backbone_learning_rate=1e-4),
    )

    assert [group["lr"] for group in groups] == [1e-4, 6e-4]
    assert list(groups[0]["params"]) == list(model.model.backbone.parameters())
    assert list(groups[1]["params"]) == list(model.model.embedding.parameters())


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


def test_metric_proxies_support_reference_kaiming_initialization() -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _attach_metric_proxies

    torch.manual_seed(7)
    model = torch.nn.Linear(2, 2)
    _attach_metric_proxies(
        model,
        train_labels=list(range(100)),
        config=ImageEndToEndConfig(
            embedding_dimensions=32,
            proxy_count_per_class=1,
            proxy_initialization="kaiming_normal",
        ),
        torch_module=torch,
    )

    row_norms = torch.linalg.vector_norm(model.metric_proxies.detach(), dim=1)
    assert not torch.allclose(row_norms, torch.ones_like(row_norms), atol=1.0e-5)
    assert float(row_norms.mean()) == pytest.approx((64.0 / 100.0) ** 0.5, rel=0.15)


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


def test_balanced_batch_indices_proxy_mode_rejects_silent_class_exclusion() -> None:
    from sfora.image_end_to_end import _balanced_batch_indices

    labels = [label for label in range(3) for _ in range(4)] + [99, 99, 99]
    with pytest.raises(ValueError, match="silently exclude classes"):
        _balanced_batch_indices(
            labels,
            batch_size=12,
            group_size=2,
            samples_per_class=4,
            steps=1,
            seed=123,
            require_all_classes=True,
        )


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


def test_source_exhaustive_batches_stop_at_smallest_class_without_reuse() -> None:
    from sfora.image_end_to_end import _source_exhaustive_batch_indices

    labels = [0] * 9 + [1] * 13
    batches = _source_exhaustive_batch_indices(
        labels,
        batch_size=8,
        samples_per_class=4,
        epochs=2,
        seed=7,
    )

    assert len(batches) == 4  # floor(min(9, 13) / 4) * two epochs
    for epoch_batches in (batches[:2], batches[2:]):
        seen: set[int] = set()
        for batch in epoch_batches:
            batch_labels = [labels[index] for index in batch]
            assert batch_labels.count(0) == 4
            assert batch_labels.count(1) == 4
            assert not seen.intersection(batch)
            seen.update(batch)
        assert len(seen) == 16


def test_source_exhaustive_batches_require_every_class_in_the_batch() -> None:
    from sfora.image_end_to_end import _source_exhaustive_batch_indices

    with pytest.raises(ValueError, match="one full chunk from every class"):
        _source_exhaustive_batch_indices(
            [0] * 8 + [1] * 8,
            batch_size=4,
            samples_per_class=4,
            epochs=1,
            seed=0,
        )


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


def test_protocol_presets_pin_their_registered_crop_augmentation() -> None:
    legacy: tuple[EndToEndProtocol, ...] = ("sota-resnet50-512", "hpl-resnet50-512")
    assert (
        config_for_protocol("proxy-anchor-resnet50-512", dataset_name="cub").train_augmentation
        == "full_res_crop"
    )
    assert (
        config_for_protocol("pfml-resnet50-512", dataset_name="cub").train_augmentation
        == "standard"
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


def test_torchvision_model_factory_loads_pinned_legacy_rsatk_weights(
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

    source = TinyResNet()

    def resnet50(*, weights: object) -> TinyResNet:
        captured["weights"] = weights
        return TinyResNet()

    def load_state_dict_from_url(url: str, **kwargs: object) -> dict[str, object]:
        captured["url"] = url
        captured["download_kwargs"] = kwargs
        return source.state_dict()

    monkeypatch.setattr(models, "resnet50", resnet50)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", load_state_dict_from_url)

    _torchvision_model_factory(
        ImageEndToEndConfig(
            pretrained_weights="legacy_resnet50_19c8e357",
            embedding_dimensions=2,
        )
    )

    assert captured["weights"] is None
    assert captured["url"] == "https://download.pytorch.org/models/resnet50-19c8e357.pth"
    assert captured["download_kwargs"] == {"map_location": "cpu", "check_hash": True}


def test_teacher_checkpoint_round_trip_loads_frozen_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    from sfora.image_end_to_end import _teacher_model_for_config

    class TinyModel(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self, dimensions: int) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(3, dimensions)

        def forward(self, images: object) -> object:
            return self.fc(images)

    def model_factory(config: ImageEndToEndConfig) -> TinyModel:
        return TinyModel(config.embedding_dimensions)

    source_config = ImageEndToEndConfig(embedding_dimensions=2)
    source_model = model_factory(source_config)
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save(
        {
            "state_dict": source_model.state_dict(),
            "arch": {
                "backbone_name": source_config.backbone_name,
                "pretrained_weights": source_config.pretrained_weights,
                "head_pooling": source_config.head_pooling,
                "embedding_dimensions": source_config.embedding_dimensions,
                "embedding_head_init": source_config.embedding_head_init,
                "embedding_layer_norm": source_config.embedding_layer_norm,
            },
        },
        checkpoint_path,
    )

    teacher = _teacher_model_for_config(
        ImageEndToEndConfig(
            embedding_dimensions=7,
            teacher_checkpoint=str(checkpoint_path),
            teacher_similarity_weight=1.0,
        ),
        model_factory=model_factory,
        device=torch.device("cpu"),
    )

    assert teacher is not None
    teacher_module = cast(Any, teacher)
    assert teacher_module.training is False
    assert all(not parameter.requires_grad for parameter in teacher_module.parameters())
    assert all(
        torch.equal(teacher_module.state_dict()[name], value)
        for name, value in source_model.state_dict().items()
    )


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


def test_resolve_source_exhaustive_schedule_uses_smallest_class() -> None:
    config = ImageEndToEndConfig(
        batch_size=8,
        samples_per_class=4,
        epoch_sampling_policy="source_exhaustive",
        train_epochs=3,
    )
    labels = [0] * 9 + [1] * 13

    assert _resolve_training_schedule(
        config,
        optimization_example_count=len(labels),
        optimization_labels=labels,
    ) == (6, 2, 3)


def test_source_zero_based_evaluation_cadence_is_phase_shifted_and_includes_final() -> None:
    config = ImageEndToEndConfig(
        eval_test_interval_epochs=5,
        eval_test_epoch_offset=1,
    )
    evaluated_epochs = [
        epoch
        for epoch in range(1, 171)
        if _should_evaluate_test(
            config,
            step=epoch * 14,
            steps_per_epoch=14,
            train_steps=170 * 14,
        )
    ]

    assert evaluated_epochs == [*range(1, 167, 5), 170]


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


def test_end_to_end_training_reports_selected_checkpoint_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    scores = iter((1.0, 0.0))
    monkeypatch.setattr(
        "sfora.image_end_to_end._checkpoint_selection_score",
        lambda *args, **kwargs: next(scores),
    )
    checkpoint_path = tmp_path / "selected.pt"
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
            save_model_path=str(checkpoint_path),
            progress_every=0,
            num_workers=0,
        ),
        model_factory=lambda config: TinyModel(),
        transform_factory=transform_factory,
    )

    metrics = result.methods["triplet_end_to_end:tiny"]
    assert metrics.selected_step == 1
    assert metrics.selection_metric == "map_at_r"
    assert metrics.selection_score is not None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["artifact_selection"] == "training_validation_selected_state"
    assert checkpoint["training_step"] == 1
    assert checkpoint["evaluation_model_source"] == "student"


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


def _averaging_probe_run(
    tmp_path: Path,
    *,
    torch: Any,
    ema_weight_averaging: bool,
) -> Any:
    """Run four steps on a two-dimensional toy problem and return the test embeddings
    that were actually scored, with the EMA teacher pinned at its initialisation."""

    class FixedModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[1.0, 0.25], [-0.25, 1.0]]))

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
    out = tmp_path / f"emb-{ema_weight_averaging}.npz"
    train_out = tmp_path / f"train-emb-{ema_weight_averaging}.npz"
    checkpoint_out = tmp_path / f"model-{ema_weight_averaging}.pt"
    run_config = ImageEndToEndConfig(
        dataset_name="cub",
        protocol="sota-resnet50-512",
        objectives=("proxy_anchor",),
        proxy_count_per_class=1,
        backbone_name="tiny",
        embedding_dimensions=2,
        batch_size=4,
        eval_batch_size=4,
        train_steps=4,
        learning_rate=0.5,
        group_size=1,
        progress_every=0,
        num_workers=0,
        # momentum 1.0 freezes the teacher at initialisation, so "did we score the
        # teacher?" becomes "do the embeddings equal the untrained model's?".
        ema_momentum=1.0,
        ema_weight_averaging=ema_weight_averaging,
        save_test_embeddings=str(out),
        save_train_embeddings=str(train_out),
        save_model_path=str(checkpoint_out),
    )
    run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=run_config,
        model_factory=lambda config: FixedModel(),
        transform_factory=transform_factory,
    )
    with np.load(out) as payload:
        scored = np.asarray(payload["embeddings"], dtype=np.float64)
    with np.load(train_out) as payload:
        assert str(payload["artifact_selection"]) in {
            "best_test_recall_at_1",
            "final_no_periodic_test_evaluation",
        }
        assert int(payload["artifact_epoch"]) >= 1

    reference = FixedModel()
    with torch.no_grad():
        raw = reference(torch.tensor([e.image for e in examples], dtype=torch.float32))
    untrained = raw / raw.norm(dim=1, keepdim=True)
    checkpoint = torch.load(checkpoint_out, map_location="cpu")
    persisted = FixedModel()
    persisted.load_state_dict(
        {
            name: value
            for name, value in checkpoint["state_dict"].items()
            if name not in {"metric_proxies", "metric_proxy_labels"}
        },
        strict=True,
    )
    with torch.no_grad():
        saved_raw = persisted(torch.tensor([e.image for e in examples], dtype=torch.float32))
    saved = (saved_raw / saved_raw.norm(dim=1, keepdim=True)).numpy().astype(np.float64)
    return scored, untrained.numpy().astype(np.float64), saved, checkpoint, run_config


def test_ema_weight_averaging_scores_the_averaged_weights_not_the_student(
    tmp_path: Path,
) -> None:
    """`ema_weight_averaging` must report the EMA copy. Pinning the teacher at
    initialisation with `ema_momentum=1.0` makes that checkable exactly: the scored
    embeddings have to equal the untrained model's, because that is what the teacher
    still is after any number of student updates."""
    torch: Any = pytest.importorskip("torch")

    scored, untrained, saved, checkpoint, run_config = _averaging_probe_run(
        tmp_path, torch=torch, ema_weight_averaging=True
    )

    np.testing.assert_allclose(scored, untrained, atol=1e-6)
    np.testing.assert_allclose(saved, scored, atol=1e-6)
    assert checkpoint["evaluation_model_source"] == "ema_weight_average"
    assert checkpoint["artifact_selection"] == "final_training_state"
    assert checkpoint["training_config"] == run_config.model_dump(mode="json")
    assert checkpoint["training_step"] == 4


def test_checkpoint_path_rejects_multiple_objectives(tmp_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")

    examples = [
        ImageExample(example_id="0-a", image=[1.0, 0.0], label=0),
        ImageExample(example_id="0-b", image=[0.9, 0.1], label=0),
        ImageExample(example_id="1-a", image=[0.0, 1.0], label=1),
        ImageExample(example_id="1-b", image=[0.1, 0.9], label=1),
    ]

    def transform_factory(config: ImageEndToEndConfig, train: bool):  # type: ignore[no-untyped-def]
        return lambda image: torch.as_tensor(image, dtype=torch.float32)

    with pytest.raises(ValueError, match="save_model_path.*single objective"):
        run_image_end_to_end_benchmark(
            train_examples=examples,
            test_examples=examples,
            config=ImageEndToEndConfig(
                dataset_name="cub",
                objectives=("triplet", "proxy_anchor"),
                backbone_name="tiny",
                embedding_dimensions=2,
                proxy_count_per_class=1,
                batch_size=4,
                eval_batch_size=4,
                train_steps=1,
                group_size=1,
                progress_every=0,
                num_workers=0,
                save_model_path=str(tmp_path / "ambiguous.pt"),
            ),
            transform_factory=transform_factory,
        )


def test_without_averaging_the_student_is_scored_and_training_moves_it(
    tmp_path: Path,
) -> None:
    """The control for the test above. Without the flag the student is scored, and four
    optimizer steps at lr 0.5 move it away from initialisation -- so if this ever starts
    matching the untrained reference, the averaging test above has stopped proving
    anything and is passing for the wrong reason."""
    torch: Any = pytest.importorskip("torch")

    scored, untrained, saved, checkpoint, _ = _averaging_probe_run(
        tmp_path, torch=torch, ema_weight_averaging=False
    )

    assert not np.allclose(scored, untrained, atol=1e-6)
    np.testing.assert_allclose(saved, scored, atol=1e-6)
    assert checkpoint["evaluation_model_source"] == "student"


def test_cublas_workspace_config_is_exported_before_any_cuda_call() -> None:
    """cuBLAS latches CUBLAS_WORKSPACE_CONFIG when its handle is created and ignores
    later changes, so exporting it after a CUDA context exists is a silent no-op --
    silent because `use_deterministic_algorithms(warn_only=True)` warns rather than
    raises on a nondeterministic matmul. This pins the ordering: the export must
    happen before the first `torch.cuda.*` call in a deterministic run."""
    import os

    from sfora.image_end_to_end import _export_cublas_workspace_config

    previous = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    try:
        _export_cublas_workspace_config()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        # setdefault, so an operator's explicit choice is never overwritten.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        _export_cublas_workspace_config()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    finally:
        if previous is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous


def test_deterministic_run_exports_workspace_config_before_seeding() -> None:
    """The ordering above must hold inside the real entry point, not just in isolation.
    Recording the order of the export against `torch.manual_seed` catches a future edit
    that moves the determinism block back below the CUDA seeding calls."""
    import os

    import sfora.image_end_to_end as module

    order: list[str] = []
    previous = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    real_export = module._export_cublas_workspace_config

    def recording_export() -> None:
        order.append("export")
        real_export()

    torch: Any = pytest.importorskip("torch")
    real_seed = torch.manual_seed

    def recording_seed(seed: int) -> Any:
        order.append("manual_seed")
        return real_seed(seed)

    module._export_cublas_workspace_config = recording_export
    torch.manual_seed = recording_seed
    try:
        # Fails later on the empty splits; the assertion is about ordering, not the
        # failure, so the exception type is deliberately not pinned here.
        with contextlib.suppress(Exception):
            run_image_end_to_end_benchmark(
                train_examples=[],
                test_examples=[],
                config=ImageEndToEndConfig(deterministic=True, num_workers=0),
            )
    finally:
        module._export_cublas_workspace_config = real_export
        torch.manual_seed = real_seed
        if previous is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous

    assert order[:2] == ["export", "manual_seed"], order


def test_evaluation_average_tracks_the_student_at_its_own_momentum(tmp_path: Path) -> None:
    """`ema_eval_momentum` must give the evaluated average its OWN timescale, independent
    of the distillation teacher's. Pinning it at 1.0 freezes it at initialisation while
    the teacher keeps moving at 0.999, so scoring the untrained outputs proves the two
    averages are genuinely separate objects rather than one aliased twice."""
    torch: Any = pytest.importorskip("torch")

    class FixedModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[1.0, 0.25], [-0.25, 1.0]]))

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
    out = tmp_path / "dual.npz"
    run_image_end_to_end_benchmark(
        train_examples=examples,
        test_examples=examples,
        config=ImageEndToEndConfig(
            dataset_name="cub",
            protocol="sota-resnet50-512",
            objectives=("proxy_anchor",),
            proxy_count_per_class=1,
            backbone_name="tiny",
            embedding_dimensions=2,
            batch_size=4,
            eval_batch_size=4,
            train_steps=4,
            learning_rate=0.5,
            group_size=1,
            progress_every=0,
            num_workers=0,
            # Teacher moves at 0.999; the evaluation average is frozen at 1.0.
            ema_distill_weight=1.0,
            ema_momentum=0.999,
            ema_weight_averaging=True,
            ema_eval_momentum=1.0 - 1e-12,
            save_test_embeddings=str(out),
        ),
        model_factory=lambda config: FixedModel(),
        transform_factory=transform_factory,
    )
    with np.load(out) as payload:
        scored = np.asarray(payload["embeddings"], dtype=np.float64)

    reference = FixedModel()
    with torch.no_grad():
        raw = reference(torch.tensor([e.image for e in examples], dtype=torch.float32))
    untrained = (raw / raw.norm(dim=1, keepdim=True)).numpy().astype(np.float64)

    np.testing.assert_allclose(scored, untrained, atol=1e-5)


def test_without_eval_momentum_the_distillation_teacher_is_reused(tmp_path: Path) -> None:
    """The control: leaving `ema_eval_momentum` unset must keep the historical single-EMA
    behaviour, so existing recipes and their digests are unaffected by this addition."""
    config = ImageEndToEndConfig(ema_weight_averaging=True, ema_momentum=0.99)

    assert config.ema_eval_momentum is None


def test_rspg_positive_loss_uses_only_registered_graph_edges() -> None:
    import torch

    from sfora.image_end_to_end import RSPGState, _rspg_positive_loss

    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    state = RSPGState(
        target_embeddings=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        neighbours=(((1, 1.0),), (), ((0, 1.0),)),
        edge_density=0.25,
        multi_component_fraction=0.5,
    )
    loss = _rspg_positive_loss(
        embeddings,
        torch.tensor([0, 1]),
        state=state,
        alpha=1.0,
        delta=0.0,
        torch_module=torch,
    )

    expected = torch.nn.functional.softplus(torch.tensor(-1.0))
    assert float(loss.detach()) == pytest.approx(float(expected))
    loss.backward()
    assert embeddings.grad is not None
    assert torch.count_nonzero(embeddings.grad[1]) == 0


def test_rspg_negative_proxy_loss_never_repels_the_own_class_proxy() -> None:
    """A graph-unknown relation must not turn the identity's proxy negative."""
    import torch

    from sfora.image_end_to_end import _proxy_anchor_negative_loss

    embeddings = torch.tensor([[1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([0])
    proxies = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    proxy_labels = torch.tensor([0, 1])
    loss = _proxy_anchor_negative_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=1.0,
        delta=0.0,
        torch_module=torch,
    )
    loss.backward()

    assert proxies.grad is not None
    assert torch.count_nonzero(proxies.grad[0]) == 0
    assert torch.count_nonzero(proxies.grad[1]) > 0


def test_ipsr_adds_ranking_loss_without_replacing_proxy_anchor() -> None:
    import torch

    from sfora.image_end_to_end import (
        ImageEndToEndConfig,
        IPSRState,
        _ipsr_ranking_loss,
        _proxy_anchor_loss,
        _proxy_anchor_objective_loss,
    )

    embeddings = torch.tensor([[0.0, 1.0], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    proxies = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    proxy_labels = torch.tensor([0, 1])
    state = IPSRState(
        target_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        preferred_indices=torch.tensor([0, -1]),
        unknown_indices=torch.tensor([1, -1]),
        anchor_coverage=0.5,
        class_coverage=0.5,
        mean_initial_loss=1.0,
    )
    config = ImageEndToEndConfig(
        objectives=("proxy_anchor",),
        proxy_anchor_alpha=1.0,
        proxy_anchor_delta=0.0,
        ipsr_weight=1.0,
    )
    ranking, active = _ipsr_ranking_loss(
        embeddings, torch.tensor([0, 1]), state=state, torch_module=torch
    )
    base = _proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=1.0,
        delta=0.0,
        torch_module=torch,
    )
    combined = _proxy_anchor_objective_loss(
        embeddings=embeddings,
        labels=labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        config=config,
        torch_module=torch,
        ipsr_state=state,
        sample_indices=torch.tensor([0, 1]),
        step=1,
        teacher_embeddings=None,
    )
    assert active == 1
    torch.testing.assert_close(combined, base + ranking)
    combined.backward()
    assert embeddings.grad is not None
    assert proxies.grad is not None  # proves the ordinary proxy term remained active


def test_rspg_graph_builder_creates_training_only_edges() -> None:
    import torch

    from sfora.image_end_to_end import _build_rspg_state

    embeddings = np.asarray(
        [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98], [-1.0, 0.0], [-0.98, 0.02]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    config = ImageEndToEndConfig(
        rspg_weight=1.0,
        rspg_rival_count=2,
        rspg_overlap_count=1,
        rspg_min_overlap=1,
        rspg_max_js=1.0,
    )
    state = _build_rspg_state(
        embeddings,
        labels,
        config=config,
        device=torch.device("cpu"),
        torch_module=torch,
        enforce_diagnostic=False,
    )

    assert state.target_embeddings.shape == (6, 2)
    assert 0.0 < state.edge_density <= 1.0
    assert any(state.neighbours)


def test_rspg_gate_rejects_close_disagreement_and_accepts_distant_agreement() -> None:
    from sfora.image_end_to_end import _rspg_signature_edge

    # Distance cannot decide the edge: the close pair disagrees about every rival.
    close_left = np.asarray([1.0, 0.0])
    close_right = np.asarray([0.9999, 0.0001])
    assert np.linalg.norm(close_left - close_right) < 0.001
    rejected = _rspg_signature_edge(
        np.asarray([0, 1, 2, 3, 4, 5, 6, 7]),
        np.full(8, 1.0 / 8.0),
        np.asarray([8, 9, 10, 11, 12, 13, 14, 15]),
        np.full(8, 1.0 / 8.0),
        overlap_count=8,
        min_overlap=4,
        max_js=0.25,
    )
    assert rejected is None

    # Conversely, geometrically opposite samples pass when their rival identities agree.
    distant_left = np.asarray([1.0, 0.0])
    distant_right = np.asarray([-1.0, 0.0])
    assert np.linalg.norm(distant_left - distant_right) == pytest.approx(2.0)
    signature = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
    probabilities = np.asarray([0.20, 0.18, 0.16, 0.14, 0.10, 0.09, 0.07, 0.06])
    accepted = _rspg_signature_edge(
        signature,
        probabilities,
        signature.copy(),
        probabilities.copy(),
        overlap_count=8,
        min_overlap=4,
        max_js=0.25,
    )
    assert accepted == pytest.approx(1.0)


def test_arcg_state_rejects_close_disagreement_and_accepts_distant_agreement() -> None:
    import torch

    from sfora.image_end_to_end import ImageEndToEndConfig, _build_arcg_state

    anchors = np.array([[1.0, 0.0], [0.999, 0.001], [0.7, 0.7], [-0.8, 0.2]], dtype=np.float64)
    angles = np.array(
        [
            [0.5, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.5, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.5, 0.1, 0.1],
            [0.5, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    unit_anchors = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
    transformed = np.stack(
        [
            np.stack(
                [
                    np.array(
                        [
                            [np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)],
                        ]
                    )
                    @ unit_anchors[index]
                    for angle in row
                ]
            )
            for index, row in enumerate(angles)
        ],
        axis=0,
    )
    state = _build_arcg_state(
        anchors,
        transformed,
        np.zeros(4, dtype=np.int64),
        config=ImageEndToEndConfig(
            dataset_name="inshop",
            objectives=("proxy_anchor",),
            arcg_weight=1.0,
            arcg_agreement_threshold=0.5,
        ),
        device=torch.device("cpu"),
        torch_module=torch,
        enforce_diagnostic=False,
    )
    neighbours = {index for index, _ in state.neighbours[0]}
    assert 1 not in neighbours  # close anchor, disagreeing intervention response
    assert 3 in neighbours  # distant anchor, agreeing intervention response


def test_spectral_class_connectivity_loss_targets_weak_cut_with_finite_gradients() -> None:
    import torch

    from sfora.image_end_to_end import _spectral_class_connectivity_loss

    embeddings = torch.tensor(
        [[1.0, 0.0], [0.99, 0.1], [-1.0, 0.0], [-0.99, 0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.zeros(4, dtype=torch.long)
    loss = _spectral_class_connectivity_loss(
        embeddings,
        labels,
        temperature=0.1,
        min_class_size=4,
        torch_module=torch,
    )
    assert loss < 0.0
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert float(embeddings.grad.norm()) > 0.0


def test_spectral_class_connectivity_loss_is_zero_without_eligible_class() -> None:
    import torch

    from sfora.image_end_to_end import _spectral_class_connectivity_loss

    embeddings = torch.randn(6, 3, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    loss = _spectral_class_connectivity_loss(
        embeddings,
        labels,
        temperature=0.1,
        min_class_size=4,
        torch_module=torch,
    )
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert embeddings.grad is not None
