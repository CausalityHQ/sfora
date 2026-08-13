from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from sfora.data import ImageExample
from sfora.foundation_finetune import (
    CosFaceHead,
    IdentityBalancedBatchSampler,
    IdentityNeck,
    TokenResidualGate,
    TrainableParameterEMA,
    batch_hard_soft_triplet,
    configure_vit_trainable_layers,
    identity_disjoint_train_validation,
    normalized_feature_anchor,
    paired_retrieval_statistics,
    query_gallery_from_identities,
    retrieval_query_values,
    select_query_gallery_identity_subset,
    split_cls_patch_tokens,
    warmup_cosine_learning_rate_factor,
)


class TinyVit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(2, 2)
        self.blocks = nn.Sequential(*(nn.Linear(2, 2) for _ in range(6)))
        self.norm = nn.LayerNorm(2)


def test_configure_vit_trainable_layers_unfreezes_only_tail_and_norm() -> None:
    model = TinyVit()

    names = configure_vit_trainable_layers(model, trainable_blocks=2)

    assert names == ("blocks.4", "blocks.5", "norm")
    assert not any(parameter.requires_grad for parameter in model.stem.parameters())
    assert not any(
        parameter.requires_grad for block in model.blocks[:4] for parameter in block.parameters()
    )
    assert all(
        parameter.requires_grad for block in model.blocks[4:] for parameter in block.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.norm.parameters())

    assert configure_vit_trainable_layers(model, trainable_blocks=0) == ()
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_identity_disjoint_split_and_query_gallery_are_deterministic() -> None:
    rows = [
        ImageExample(f"{label}-{index}", Path(f"{label}-{index}.jpg"), label)
        for label in range(10)
        for index in range(3)
    ] + [ImageExample("singleton", Path("singleton.jpg"), 10)]

    train, validation = identity_disjoint_train_validation(rows, seed=1, validation_fraction=0.2)
    query, gallery = query_gallery_from_identities(validation, seed=1)

    assert {row.label for row in train}.isdisjoint({row.label for row in validation})
    assert len({row.label for row in validation}) == 2
    assert len(query) == 2
    assert len(gallery) == 4
    assert {row.label for row in query} == {row.label for row in gallery}
    assert any(row.example_id == "singleton" for row in train)


def test_identity_neck_starts_as_exact_deployed_identity() -> None:
    neck = IdentityNeck(3)
    inputs = torch.tensor([[1.0, -2.0, 3.0]])

    torch.testing.assert_close(neck(inputs), inputs)
    assert all(parameter.requires_grad for parameter in neck.parameters())


def test_query_gallery_subset_selects_complete_fixed_identities() -> None:
    query = [ImageExample(f"q-{label}", Path(f"q-{label}.jpg"), label) for label in range(10)]
    gallery = [
        ImageExample(f"g-{label}-{index}", Path(f"g-{label}-{index}.jpg"), label)
        for label in range(10)
        for index in range(2)
    ]

    screen_query, screen_gallery = select_query_gallery_identity_subset(
        query, gallery, seed=17, fraction=0.2, complement=False
    )
    holdout_query, holdout_gallery = select_query_gallery_identity_subset(
        query, gallery, seed=17, fraction=0.2, complement=True
    )

    assert len(screen_query) == 2
    assert len(screen_gallery) == 4
    assert {row.label for row in screen_query} == {row.label for row in screen_gallery}
    assert {row.label for row in screen_query}.isdisjoint({row.label for row in holdout_query})
    assert len(holdout_query) == 8
    assert len(holdout_gallery) == 16


def test_cosface_head_applies_margin_only_to_target_logit() -> None:
    head = CosFaceHead(embedding_dim=2, class_count=2, margin=0.2, scale=10.0)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2))

    logits = head(torch.tensor([[1.0, 0.0]], dtype=torch.float32), torch.tensor([0]))

    torch.testing.assert_close(logits, torch.tensor([[8.0, 0.0]]))


def test_token_residual_gate_starts_at_the_normalized_cls_descriptor() -> None:
    gate = TokenResidualGate(3)
    cls = torch.tensor([[3.0, 4.0, 0.0], [0.0, -5.0, 12.0]], dtype=torch.float32)
    patches = torch.tensor([[4.0, 0.0, 3.0], [5.0, 12.0, 0.0]], dtype=torch.float32)

    output = gate(cls, patches)

    torch.testing.assert_close(output, torch.nn.functional.normalize(cls, dim=1))
    assert gate.gate.numel() == 3
    assert gate.gate.requires_grad
    with torch.no_grad():
        gate.gate.copy_(torch.tensor([0.5, -0.25, 0.75]))
    expected = torch.nn.functional.normalize(
        torch.nn.functional.normalize(cls, dim=1)
        + torch.tanh(gate.gate) * torch.nn.functional.normalize(patches, dim=1),
        dim=1,
    )
    torch.testing.assert_close(gate(cls, patches), expected)


def test_batch_hard_soft_triplet_uses_farthest_positive_and_nearest_negative() -> None:
    embeddings = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [6.0, 0.0], [0.0, 10.0], [2.0, 10.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)

    loss = batch_hard_soft_triplet(embeddings, labels)

    distances = torch.cdist(embeddings, embeddings)
    expected = torch.stack(
        [
            torch.nn.functional.softplus(distances[0, 2] - distances[0, 3]),
            torch.nn.functional.softplus(distances[1, 2] - distances[1, 4]),
            torch.nn.functional.softplus(distances[2, 0] - distances[2, 4]),
            torch.nn.functional.softplus(distances[3, 4] - distances[3, 0]),
            torch.nn.functional.softplus(distances[4, 3] - distances[4, 1]),
        ]
    ).mean()
    torch.testing.assert_close(loss, expected)


def test_identity_balanced_sampler_is_epoch_deterministic_and_preserves_pk_batches() -> None:
    labels = [0, 0, 1, 1, 1, 2, 2, 3, 3]
    first = IdentityBalancedBatchSampler(labels, labels_per_batch=3, instances_per_label=2, seed=7)
    second = IdentityBalancedBatchSampler(labels, labels_per_batch=3, instances_per_label=2, seed=7)

    batch = next(iter(first))

    assert batch == next(iter(second))
    assert len(batch) == 6
    observed = [labels[index] for index in batch]
    assert sorted(observed.count(label) for label in set(observed)) == [2, 2, 2]
    first.set_epoch(1)
    assert next(iter(first)) != batch
    assert set(labels) == {labels[index] for rows in first for index in rows}
    other_seed = IdentityBalancedBatchSampler(
        labels, labels_per_batch=3, instances_per_label=2, seed=8
    )
    assert list(first) != list(other_seed)


def test_identity_balanced_sampler_maximizes_distinct_examples_when_tiling() -> None:
    labels = [0, 0, 1, 1, 1]
    sampler = IdentityBalancedBatchSampler(
        labels, labels_per_batch=2, instances_per_label=4, seed=3
    )

    batch = next(iter(sampler))

    for label in (0, 1):
        indexes = [index for index in batch if labels[index] == label]
        assert len(indexes) == 4
        expected_distinct = len({i for i, observed in enumerate(labels) if observed == label})
        assert len(set(indexes)) == expected_distinct


def test_identity_balanced_sampler_padding_keeps_distinct_labels_in_final_batch() -> None:
    labels = [label for label in range(7) for _ in range(2)]
    sampler = IdentityBalancedBatchSampler(
        labels, labels_per_batch=3, instances_per_label=2, seed=0
    )

    final_batch = list(sampler)[-1]

    assert len({labels[index] for index in final_batch}) == 3


def test_split_cls_patch_tokens_excludes_cls_from_the_local_mean() -> None:
    tokens = torch.tensor(
        [[[3.0, 4.0], [1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]],
        dtype=torch.float32,
    )

    cls, patches = split_cls_patch_tokens(tokens)

    torch.testing.assert_close(cls, torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    torch.testing.assert_close(patches, torch.tensor([[2.0, 3.0], [8.0, 9.0]]))


def test_retrieval_query_values_and_paired_statistics_preserve_query_pairing() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    gallery = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=torch.float32)
    query_labels = torch.tensor([0, 1], dtype=torch.int64).numpy()
    gallery_labels = torch.tensor([0, 0, 1], dtype=torch.int64).numpy()

    hits, average_precision = retrieval_query_values(query, query_labels, gallery, gallery_labels)
    stats = paired_retrieval_statistics(
        initial_hits=torch.tensor([False, True, False, True]).numpy(),
        final_hits=torch.tensor([True, True, True, False]).numpy(),
        initial_ap=torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64).numpy(),
        final_ap=torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64).numpy(),
        seed=5,
        bootstrap_replicates=1000,
    )

    assert hits.tolist() == [True, True]
    assert average_precision.tolist() == [1.0, 1.0]
    assert stats["recall_lost"] == 1
    assert stats["recall_gained"] == 2
    assert stats["map_at_r_delta"] == 0.25
    assert stats["map_at_r_delta_ci95_lower"] <= 0.25
    assert stats["map_at_r_delta_ci95_upper"] >= 0.25


def test_normalized_feature_anchor_is_zero_at_teacher_and_penalizes_rotation() -> None:
    teacher = torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float32)
    rotated = torch.tensor([[4.0, -3.0], [2.0, 0.0]], dtype=torch.float32)

    torch.testing.assert_close(normalized_feature_anchor(teacher, teacher), torch.tensor(0.0))
    torch.testing.assert_close(normalized_feature_anchor(rotated, teacher), torch.tensor(1.0))
    student = teacher.clone().requires_grad_(True)
    normalized_feature_anchor(student, teacher).backward()
    assert student.grad is not None
    assert float(student.grad.abs().max()) < 1.0e-6


def test_trainable_parameter_ema_tracks_only_trainable_parameters_and_can_apply() -> None:
    module = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 1, bias=False))
    module[0].requires_grad_(False)
    with torch.no_grad():
        module[0].weight.fill_(7.0)
        module[1].weight.fill_(1.0)
    ema = TrainableParameterEMA(module, decay=0.5)
    with torch.no_grad():
        module[0].weight.fill_(9.0)
        module[1].weight.fill_(3.0)

    ema.update(module)
    ema.apply(module)

    torch.testing.assert_close(module[0].weight, torch.full((2, 2), 9.0))
    torch.testing.assert_close(module[1].weight, torch.full((1, 2), 2.0))


def test_warmup_cosine_schedule_uses_every_update_and_reaches_zero_after_training() -> None:
    observed = [
        warmup_cosine_learning_rate_factor(step, warmup_steps=2, total_steps=6) for step in range(7)
    ]

    assert observed == [0.5, 1.0, 1.0, 0.8535533905932737, 0.5, 0.14644660940672627, 0.0]
