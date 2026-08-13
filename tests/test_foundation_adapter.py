from __future__ import annotations

import numpy as np
import pytest
import torch

from sfora.data import ImageExample
from sfora.foundation_adapter import (
    AdapterConfig,
    NestedLinearAdapter,
    NestedResidualAdapter,
    cosine_margin_loss,
    fit_ridge_stitch,
    hardened_retrieval_folds,
    identity_balanced_batches,
    initialize_residual_from_linear,
    nested_embeddings,
    retrieval_map_at_r,
    retrieval_recall_at_1,
    ridge_stitch,
)
from sfora.image_recipes import class_disjoint_recipe_selection_split


def test_registered_adapter_is_under_five_million_parameters() -> None:
    config = AdapterConfig(input_dim=768, class_count=3_997)
    model = NestedResidualAdapter(config)

    trainable = sum(parameter.numel() for parameter in model.parameters())

    assert config.prefixes == (64, 128, 256, 512)
    assert trainable == 3_753_473
    assert trainable <= 5_000_000


def test_nested_embeddings_are_independently_normalized() -> None:
    torch.manual_seed(7)
    model = NestedResidualAdapter(AdapterConfig(input_dim=12, hidden_dim=16, class_count=5))
    inputs = torch.randn(4, 12)

    full = model(inputs)
    prefixes = nested_embeddings(full, model.config.prefixes)

    assert tuple(prefixes) == model.config.prefixes
    for width, value in prefixes.items():
        assert value.shape == (4, width)
        torch.testing.assert_close(value.norm(dim=1), torch.ones(4), atol=1e-6, rtol=0)


def test_linear_control_has_no_hidden_path() -> None:
    config = AdapterConfig(input_dim=12, hidden_dim=16, class_count=5)
    model = NestedLinearAdapter(config)

    assert sum(parameter.numel() for parameter in model.parameters()) == 8_704
    assert model(torch.randn(3, 12)).shape == (3, 512)


def test_residual_adapter_can_start_exactly_from_linear_control() -> None:
    torch.manual_seed(3)
    config = AdapterConfig(
        input_dim=12, hidden_dim=16, output_dim=8, prefixes=(2, 4, 8), class_count=5
    )
    linear = NestedLinearAdapter(config)
    residual = NestedResidualAdapter(config)
    inputs = torch.randn(7, 12)

    initialize_residual_from_linear(residual, linear)

    torch.testing.assert_close(residual(inputs), linear(inputs), atol=0, rtol=0)
    torch.testing.assert_close(residual.class_proxies, linear.class_proxies, atol=0, rtol=0)
    assert residual.residual_scale.item() == 0.0


def test_cosine_margin_loss_uses_every_registered_prefix() -> None:
    config = AdapterConfig(
        input_dim=6,
        hidden_dim=8,
        output_dim=8,
        prefixes=(2, 4, 8),
        class_count=4,
        margin=0.2,
        scale=16.0,
    )
    model = NestedResidualAdapter(config)
    inputs = torch.randn(5, 6, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 0])

    loss, losses = cosine_margin_loss(model(inputs), labels, model.class_proxies, config)
    loss.backward()

    assert tuple(losses) == config.prefixes
    assert torch.isfinite(loss)
    assert model.class_proxies.grad is not None
    assert inputs.grad is not None
    assert all(torch.isfinite(value) for value in losses.values())


@pytest.mark.parametrize("bad_labels", [torch.tensor([0.0]), torch.tensor([4])])
def test_cosine_margin_loss_rejects_invalid_labels(bad_labels: torch.Tensor) -> None:
    config = AdapterConfig(
        input_dim=4, hidden_dim=8, output_dim=8, prefixes=(2, 4, 8), class_count=4
    )
    model = NestedResidualAdapter(config)

    with pytest.raises(ValueError, match="labels"):
        cosine_margin_loss(model(torch.randn(1, 4)), bad_labels, model.class_proxies, config)


def test_adapter_rejects_nonfinite_features() -> None:
    model = NestedResidualAdapter(
        AdapterConfig(input_dim=4, hidden_dim=8, output_dim=8, prefixes=(2, 4, 8))
    )
    inputs = torch.from_numpy(np.array([[1.0, 2.0, np.nan, 4.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="finite"):
        model(inputs)


def test_identity_balanced_batches_are_deterministic_and_balanced() -> None:
    labels = np.repeat(np.arange(6, dtype=np.int64), 3)

    first = identity_balanced_batches(
        labels, identities_per_batch=3, images_per_identity=2, seed=9
    )
    second = identity_balanced_batches(
        labels, identities_per_batch=3, images_per_identity=2, seed=9
    )

    assert len(first) == 2
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    for batch in first:
        values, counts = np.unique(labels[batch], return_counts=True)
        assert len(values) == 3
        assert counts.tolist() == [2, 2, 2]


def test_hardened_folds_hold_out_a_permanent_distractor_pool() -> None:
    labels = np.repeat(np.arange(30, dtype=np.int64), 4)

    folds = hardened_retrieval_folds(labels, seed=17)

    assert len(folds) == 4
    distractors = folds[0].distractor
    assert all(np.array_equal(fold.distractor, distractors) for fold in folds)
    assert len(set(labels[distractors])) >= 1
    assert set(labels[distractors]).isdisjoint(
        set(np.concatenate([labels[fold.query] for fold in folds]))
    )
    for fold in folds:
        assert set(fold.optimization).isdisjoint(
            np.concatenate((fold.query, fold.gallery, fold.distractor))
        )
        assert set(fold.query).isdisjoint(fold.gallery)
        assert set(labels[fold.query]) == set(labels[fold.gallery])
        assert set(labels[fold.optimization]).isdisjoint(
            set(labels[np.concatenate((fold.query, fold.gallery, fold.distractor))])
        )


def test_hardened_folds_use_only_the_original_identity_disjoint_holdout() -> None:
    labels = np.repeat(np.arange(30, dtype=np.int64), 4)
    examples = tuple(
        ImageExample(example_id=f"{index}", image=None, label=int(label))
        for index, label in enumerate(labels)
    )
    original = class_disjoint_recipe_selection_split(examples, fraction=0.2, seed=0)
    original_train = {int(row.label) for row in original.optimization}
    original_holdout = {int(row.label) for row in original.query + original.gallery}

    folds = hardened_retrieval_folds(labels, seed=0, heldout_fraction=0.2)

    assert {int(labels[index]) for index in folds[0].optimization} == original_train
    observed_holdout = {
        int(labels[index])
        for fold in folds
        for index in np.concatenate((fold.query, fold.gallery, fold.distractor))
    }
    assert observed_holdout == original_holdout


def test_chunked_recall_at_1_matches_exact_neighbors() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    gallery = torch.tensor([[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]], dtype=torch.float32)

    recall = retrieval_recall_at_1(
        query,
        np.array([7, 11]),
        gallery,
        np.array([7, 11, 99]),
        chunk_size=1,
    )

    assert recall == 1.0


def test_chunked_map_at_r_uses_each_query_relevant_count() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    gallery = torch.tensor(
        [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.7, 0.3]], dtype=torch.float32
    )
    query_labels = np.array([7, 11], dtype=np.int64)
    gallery_labels = np.array([7, 7, 11, 99], dtype=np.int64)

    score = retrieval_map_at_r(
        query, query_labels, gallery, gallery_labels, chunk_size=1
    )

    assert score == 1.0


def test_map_at_r_fractional_case_uses_only_first_r_ranks() -> None:
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    gallery = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=torch.float32)

    score = retrieval_map_at_r(
        query,
        np.array([7], dtype=np.int64),
        gallery,
        np.array([7, 99, 7], dtype=np.int64),
    )

    assert score == 0.5


def test_ridge_stitch_recovers_a_linear_target_from_optimization_rows() -> None:
    generator = torch.Generator().manual_seed(4)
    source = torch.randn(20, 5, generator=generator)
    transform = torch.randn(5, 3, generator=generator)
    target = source @ transform + torch.tensor([0.3, -0.2, 0.1])

    predicted = ridge_stitch(source, target, torch.arange(15), regularization=1e-8)

    torch.testing.assert_close(predicted, target, atol=2e-5, rtol=2e-5)

    model = fit_ridge_stitch(source, target, torch.arange(15), regularization=1e-8)
    torch.testing.assert_close(model.transform(source[15:]), target[15:], atol=2e-5, rtol=2e-5)
