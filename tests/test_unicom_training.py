from __future__ import annotations

import torch
from torch.nn import functional as F

from sfora.unicom_inshop import InshopRecord
from sfora.unicom_training import (
    identity_holdout,
    padded_epoch_indices,
    sample_shard_masks,
    sharded_mask_arcface_loss,
)


def _manual_arcface(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    margin: float = 0.25,
    scale: float = 32.0,
) -> torch.Tensor:
    cosine = (
        F.normalize(embeddings[:, coordinates], dim=1)
        @ F.normalize(weights[:, coordinates], dim=1).T
    )
    rows = torch.arange(labels.numel())
    target = cosine[rows, labels].clamp(-1.0, 1.0)
    cosine = cosine.clone()
    cosine[rows, labels] = torch.cos(torch.acos(target) + margin)
    return F.cross_entropy(cosine * scale, labels)


def test_one_shard_matches_official_selected_subspace_arcface() -> None:
    embeddings = torch.tensor([[0.2, 0.4, -0.1, 0.7], [-0.3, 0.8, 0.5, 0.1]], dtype=torch.float64)
    weights = torch.tensor(
        [[0.7, -0.2, 0.4, 0.1], [0.3, 0.6, -0.5, 0.2], [-0.4, 0.1, 0.8, 0.5]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 2])
    mask = torch.tensor([[0, 2, 3]])

    actual = sharded_mask_arcface_loss(embeddings, weights, labels, mask)
    expected = _manual_arcface(embeddings, weights, labels, mask[0])

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_repeated_mask_shards_equal_one_global_mask() -> None:
    generator = torch.Generator().manual_seed(7)
    embeddings = torch.randn(5, 6, generator=generator, dtype=torch.float64)
    weights = torch.randn(8, 6, generator=generator, dtype=torch.float64)
    labels = torch.tensor([0, 2, 3, 5, 7])
    mask = torch.tensor([0, 2, 3, 5])

    sharded = sharded_mask_arcface_loss(embeddings, weights, labels, mask.repeat(4, 1))
    unsharded = _manual_arcface(embeddings, weights, labels, mask)

    torch.testing.assert_close(sharded, unsharded, rtol=1e-12, atol=1e-12)


def test_independent_shard_masks_route_each_class_through_its_coordinates() -> None:
    embeddings = torch.tensor([[2.0, 1.0, 3.0, -2.0]], dtype=torch.float64)
    weights = torch.tensor([[1.0, 4.0, 2.0, 0.5], [-3.0, 1.0, 0.25, 2.0]], dtype=torch.float64)
    labels = torch.tensor([1])
    masks = torch.tensor([[0, 1], [2, 3]])

    actual = sharded_mask_arcface_loss(embeddings, weights, labels, masks, margin=0.0, scale=1.0)
    logits = torch.stack(
        [
            F.cosine_similarity(embeddings[:, masks[0]], weights[0:1, masks[0]]),
            F.cosine_similarity(embeddings[:, masks[1]], weights[1:2, masks[1]]),
        ],
        dim=1,
    ).squeeze(-1)
    expected = F.cross_entropy(logits, labels)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_eight_shards_recover_gradient_coverage_lost_by_one_mask() -> None:
    embeddings = torch.randn(4, 8, generator=torch.Generator().manual_seed(9))
    weights = torch.randn(8, 8, generator=torch.Generator().manual_seed(10))
    labels = torch.tensor([0, 2, 5, 7])
    single = torch.tensor([[0, 1, 2, 3]])
    covering = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 2, 4, 6],
            [1, 3, 5, 7],
            [0, 1, 6, 7],
            [2, 3, 4, 5],
            [0, 3, 4, 7],
            [1, 2, 5, 6],
        ]
    )

    one = embeddings.clone().requires_grad_()
    sharded_mask_arcface_loss(one, weights, labels, single).backward()
    many = embeddings.clone().requires_grad_()
    sharded_mask_arcface_loss(many, weights, labels, covering).backward()

    assert torch.count_nonzero(one.grad[:, 4:]) == 0
    assert torch.count_nonzero(many.grad[:, 4:]) > 0


def test_padded_epoch_indices_match_distributed_sampler_global_union() -> None:
    actual = padded_epoch_indices(size=10, global_batch=8, epoch=3, seed=0)
    generator = torch.Generator().manual_seed(3)
    shuffled = torch.randperm(10, generator=generator).tolist()

    assert actual == tuple((shuffled * 2)[:16])
    assert len(actual) % 8 == 0


def test_padded_epoch_indices_do_not_alias_adjacent_experiment_seeds() -> None:
    seed_zero_epoch_one = padded_epoch_indices(size=97, global_batch=16, epoch=1, seed=0)
    seed_one_epoch_zero = padded_epoch_indices(size=97, global_batch=16, epoch=0, seed=1)

    assert seed_zero_epoch_one != seed_one_epoch_zero


def test_padded_epoch_indices_match_eight_rank_drop_last_tail() -> None:
    actual = padded_epoch_indices(size=25_882, global_batch=128, epoch=0, seed=0)

    assert len(actual) == 25_856
    assert len(set(actual)) == 25_856


def test_sample_shard_masks_use_one_official_noise_sort_per_shard() -> None:
    generator = torch.Generator().manual_seed(44)
    actual = sample_shard_masks(
        dimension=8, selected=4, shards=3, generator=generator, device=torch.device("cpu")
    )
    oracle = torch.Generator().manual_seed(44)
    expected = torch.stack([torch.argsort(torch.rand(8, generator=oracle))[:4] for _ in range(3)])

    assert torch.equal(actual, expected)


def test_identity_holdout_is_deterministic_complete_and_train_disjoint(tmp_path) -> None:
    records = tuple(
        InshopRecord(split="train", image_path=tmp_path / f"{label}-{index}.jpg", label=label)
        for label in ("a", "b", "c", "d", "e")
        for index in range(3)
    )

    first = identity_holdout(records, fraction=0.4, seed=19)
    second = identity_holdout(records, fraction=0.4, seed=19)

    assert first == second
    optimization, query, gallery, label_map = first
    optimization_labels = {row.label for row in optimization}
    heldout_labels = {row.label for row in query}
    assert optimization_labels.isdisjoint(heldout_labels)
    assert {row.label for row in gallery} == heldout_labels
    assert len(query) == 2
    assert len(gallery) == 4
    assert label_map == {label: index for index, label in enumerate(sorted(optimization_labels))}


def test_identity_holdout_keeps_singleton_identity_in_optimization(tmp_path) -> None:
    repeated = tuple(
        InshopRecord(
            split="train",
            image_path=tmp_path / f"{label}-{index}.jpg",
            label=label,
        )
        for label in ("a", "b")
        for index in range(2)
    )
    singleton = InshopRecord(
        split="train", image_path=tmp_path / "singleton.jpg", label="singleton"
    )

    optimization, query, gallery, label_map = identity_holdout(
        (*repeated, singleton), fraction=0.5, seed=3
    )

    assert singleton in optimization
    assert "singleton" not in {row.label for row in (*query, *gallery)}
    assert "singleton" in label_map
    assert len(query) == 1


def test_zero_holdout_fraction_returns_all_training_identities(tmp_path) -> None:
    records = tuple(
        InshopRecord(
            split="train",
            image_path=tmp_path / f"{label}-{index}.jpg",
            label=label,
        )
        for label in ("a", "b")
        for index in range(2)
    )

    optimization, query, gallery, label_map = identity_holdout(records, fraction=0.0, seed=99)

    assert optimization == records
    assert query == ()
    assert gallery == ()
    assert label_map == {"a": 0, "b": 1}
