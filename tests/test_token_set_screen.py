from __future__ import annotations

import pytest
import torch

from sfora.token_set_screen import (
    cross_label_token_permutation,
    leave_one_out_recall_at_one,
    validate_f1_class_partition,
)


def test_cross_label_permutation_is_seeded_bijective_and_never_keeps_a_label() -> None:
    labels = torch.arange(49).repeat_interleave(41)

    first = cross_label_token_permutation(labels, seed=17)
    repeated = cross_label_token_permutation(labels, seed=17)
    other = cross_label_token_permutation(labels, seed=18)

    assert torch.equal(first, repeated)
    assert sorted(first.tolist()) == list(range(labels.numel()))
    assert not bool((labels[first] == labels).any())
    assert float((first != other).float().mean()) > 0.5
    for label in labels.unique():
        target = labels == label
        assert labels[first[target]].unique().numel() >= 8


def test_cross_label_permutation_rejects_an_impossible_majority_class() -> None:
    with pytest.raises(ValueError, match="cross-label derangement"):
        cross_label_token_permutation(torch.tensor([0, 0, 0, 1]), seed=17)


def test_leave_one_out_recall_uses_exact_global_set_hybrid_score() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    global_embeddings = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1], [-0.9, 0.1]]),
        dim=-1,
    )
    token_embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    weights = torch.full((4, 2), 0.5)

    assert (
        leave_one_out_recall_at_one(
            global_embeddings,
            token_embeddings,
            weights,
            labels,
            set_weight=0.75,
            query_block=2,
        )
        == 1.0
    )


def test_leave_one_out_detaches_trained_outputs_before_kernel_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sfora.token_set_screen as screen

    observed: list[bool] = []

    def fake_kernel(
        query: torch.Tensor,
        gallery: torch.Tensor,
        *,
        query_weights: torch.Tensor,
        gallery_weights: torch.Tensor,
    ) -> torch.Tensor:
        observed.append(
            any(
                tensor.requires_grad
                for tensor in (query, gallery, query_weights, gallery_weights)
            )
        )
        return torch.zeros((query.shape[0], gallery.shape[0]))

    monkeypatch.setattr(screen, "fused_set_maxsim", fake_kernel)
    global_embeddings = torch.eye(4, requires_grad=True)
    token_embeddings = torch.eye(4).reshape(4, 1, 4).requires_grad_()
    token_weights = torch.ones((4, 1), requires_grad=True)

    screen.leave_one_out_recall_at_one(
        global_embeddings,
        token_embeddings,
        token_weights,
        torch.tensor([0, 0, 1, 1]),
        set_weight=0.25,
        query_block=2,
    )

    assert observed == [False, False]


def test_f1_partition_accepts_only_preregistered_disjoint_train_bands() -> None:
    validate_f1_class_partition(
        train_labels=torch.arange(49).repeat_interleave(2),
        validation_labels=torch.arange(49, 82).repeat_interleave(2),
    )

    with pytest.raises(ValueError, match="classes 0 through 48"):
        validate_f1_class_partition(
            train_labels=torch.arange(1, 49).repeat_interleave(2),
            validation_labels=torch.arange(49, 82).repeat_interleave(2),
        )
    with pytest.raises(ValueError, match="classes 49 through 81"):
        validate_f1_class_partition(
            train_labels=torch.arange(49).repeat_interleave(2),
            validation_labels=torch.arange(50, 82).repeat_interleave(2),
        )
