from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from sfora.data import ImageExample
from sfora.foundation_finetune import (
    CosFaceHead,
    IdentityNeck,
    configure_vit_trainable_layers,
    identity_disjoint_train_validation,
    query_gallery_from_identities,
    select_query_gallery_identity_subset,
)


class TinyVit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(2, 2)
        self.blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(6))
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
