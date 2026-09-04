from __future__ import annotations

from collections import Counter

import pytest
import torch

from sfora.unicom_rank_finish import identity_balanced_batches, smooth_ap_finish_loss


def _labels(identity_count: int = 40, images_per_identity: int = 5) -> tuple[str, ...]:
    return tuple(
        f"item-{identity:03d}"
        for identity in range(identity_count)
        for _ in range(images_per_identity)
    )


def test_identity_balanced_batches_are_exact_replay_and_epoch_separated() -> None:
    labels = _labels()
    arguments = {
        "batch_size": 128,
        "images_per_identity": 4,
        "seed": 7,
        "steps": 3,
    }

    first = identity_balanced_batches(labels, epoch=5, **arguments)
    replay = identity_balanced_batches(labels, epoch=5, **arguments)
    next_epoch = identity_balanced_batches(labels, epoch=6, **arguments)

    assert first == replay
    assert first != next_epoch
    assert len(first) == 3
    for batch in first:
        assert len(batch) == 128
        assert len(set(batch)) == 128
        counts = Counter(labels[index] for index in batch)
        assert len(counts) == 32
        assert set(counts.values()) == {4}


def test_identity_balanced_batches_cycle_only_after_identity_inventory() -> None:
    labels = _labels(images_per_identity=5)

    batches = identity_balanced_batches(
        labels,
        batch_size=128,
        images_per_identity=4,
        seed=11,
        epoch=5,
        steps=2,
    )

    by_identity: dict[str, list[int]] = {}
    for batch in batches:
        for index in batch:
            by_identity.setdefault(labels[index], []).append(index)
    assert all(
        len(indices) <= 5 or len(set(indices[:5])) == 5
        for indices in by_identity.values()
    )


def test_identity_balanced_batches_cycle_sparse_identity_images() -> None:
    labels = _labels(identity_count=32, images_per_identity=2)

    (batch,) = identity_balanced_batches(
        labels,
        batch_size=128,
        images_per_identity=4,
        seed=13,
        epoch=5,
        steps=1,
    )

    counts = Counter(labels[index] for index in batch)
    assert set(counts.values()) == {4}
    assert all(
        len(set(index for index in batch if labels[index] == label)) == 2
        for label in counts
    )


@pytest.mark.parametrize(
    ("labels", "kwargs"),
    (
        (("a", "a", "b", "b"), {"batch_size": 3, "images_per_identity": 2}),
        (("a", "a", "b", "b"), {"batch_size": 4, "images_per_identity": 1}),
        (("a", "a", "a", "a"), {"batch_size": 4, "images_per_identity": 2}),
        (("a", "", "b", "b"), {"batch_size": 4, "images_per_identity": 2}),
    ),
)
def test_identity_balanced_batches_reject_invalid_or_impossible_shapes(
    labels: tuple[str, ...], kwargs: dict[str, int]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        identity_balanced_batches(labels, seed=0, epoch=5, steps=1, **kwargs)


def _smooth_ap_scalar_oracle(
    embeddings: torch.Tensor, labels: tuple[int, ...], *, temperature: float
) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(embeddings.float(), dim=1)[:, :512]
    distances = torch.sum(
        (normalized[:, None, :] - normalized[None, :, :]) ** 2, dim=2
    )
    average_precisions = []
    for anchor, label in enumerate(labels):
        candidates = [index for index in range(len(labels)) if index != anchor]
        positives = [index for index in candidates if labels[index] == label]
        positive_precisions = []
        for positive in positives:
            competitors = [index for index in candidates if index != positive]
            rank = 1.0 + torch.stack(
                [
                    torch.sigmoid(
                        (distances[anchor, positive] - distances[anchor, other])
                        / temperature
                    )
                    for other in competitors
                ]
            ).sum()
            positive_competitors = [
                other for other in positives if other != positive
            ]
            positive_rank = 1.0 + (
                torch.stack(
                    [
                        torch.sigmoid(
                            (distances[anchor, positive] - distances[anchor, other])
                            / temperature
                        )
                        for other in positive_competitors
                    ]
                ).sum()
                if positive_competitors
                else 0.0
            )
            positive_precisions.append(positive_rank / rank)
        average_precisions.append(torch.stack(positive_precisions).mean())
    return 1.0 - torch.stack(average_precisions).mean()


def test_smooth_ap_finish_loss_matches_scalar_deployment_geometry_and_gradients() -> None:
    generator = torch.Generator().manual_seed(19)
    embeddings = torch.randn(8, 768, generator=generator, requires_grad=True)
    labels = (0, 0, 1, 1, 2, 2, 3, 3)

    observed = smooth_ap_finish_loss(
        embeddings, labels, dimensions=512, temperature=0.01
    )
    expected = _smooth_ap_scalar_oracle(embeddings, labels, temperature=0.01)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=1e-7)
    observed.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert float(embeddings.grad.abs().sum()) > 0.0


def test_smooth_ap_finish_loss_rewards_positive_ordering() -> None:
    perfect = torch.zeros(4, 768)
    perfect[0, 0] = perfect[1, 0] = 1.0
    perfect[2, 1] = perfect[3, 1] = 1.0
    reversed_order = perfect.clone()
    reversed_order[1] = torch.tensor([0.0, 1.0] + [0.0] * 766)
    reversed_order[2] = torch.tensor([1.0, 0.0] + [0.0] * 766)
    labels = (0, 0, 1, 1)

    assert smooth_ap_finish_loss(perfect, labels) < smooth_ap_finish_loss(
        reversed_order, labels
    )


@pytest.mark.parametrize(
    ("embeddings", "labels", "kwargs"),
    (
        (torch.ones(3, 768), (0, 0), {}),
        (torch.ones(4, 768), (0, 1, 2, 3), {}),
        (torch.ones(4, 511), (0, 0, 1, 1), {}),
        (torch.ones(4, 768), (0, 0, 1, 1), {"dimensions": 0}),
        (torch.ones(4, 768), (0, 0, 1, 1), {"temperature": 0.0}),
        (
            torch.full((4, 768), float("nan")),
            (0, 0, 1, 1),
            {},
        ),
    ),
)
def test_smooth_ap_finish_loss_rejects_invalid_inputs(
    embeddings: torch.Tensor, labels: tuple[int, ...], kwargs: dict[str, object]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        smooth_ap_finish_loss(embeddings, labels, **kwargs)
