from __future__ import annotations

import hashlib
import random

import pytest
import torch

from sfora import unicom_fepf as module
from sfora.unicom_training import experiment_stream_seed


def independent_sequential_fp64_means(
    features: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    rows = []
    for label in range(int(labels.max()) + 1):
        total = torch.zeros(features.shape[1], dtype=torch.float64)
        count = 0
        for feature, current in zip(features, labels, strict=True):
            if int(current) == label:
                total.add_((feature / torch.linalg.vector_norm(feature)).double())
                count += 1
        rows.append((total / count) / torch.linalg.vector_norm(total / count))
    return torch.stack(rows).float().mul_(0.01 * features.shape[1] ** 0.5)


def test_cache_preserves_partition_order_and_class_means_match_reference() -> None:
    """Fails if cache construction sorts rows or changes sequential FP64 means."""
    records = (("c2", "b.jpg"), ("c1", "z.jpg"), ("c2", "a.jpg"))
    embeddings = torch.tensor([[3.0, 0.0], [0.0, 2.0], [0.0, 4.0]], dtype=torch.float32)

    cache = module.build_fepf_cache(records, embeddings, {"c2": 0, "c1": 1})

    assert cache.record_inventory == records
    expected = independent_sequential_fp64_means(embeddings, torch.tensor([0, 1, 0]))
    assert torch.equal(module.canonical_class_means(cache, dimension=2), expected)
    assert cache.feature_sha256 == hashlib.sha256(embeddings.numpy().tobytes(order="C")).hexdigest()


@pytest.mark.parametrize(
    ("records", "features", "label_map"),
    (
        ((("c0", "a"),), torch.ones(1, 2, dtype=torch.float64), {"c0": 0}),
        ((("c0", "a"),), torch.ones(1, 2).t(), {"c0": 0}),
        ((("c0", "a"),), torch.tensor([[float("nan"), 1.0]]), {"c0": 0}),
        ((("c0", "a"),), torch.zeros(1, 2), {"c0": 0}),
        ((("c0", "a"),), torch.ones(1, 2), {"c0": 1}),
        ((("c0", "a"),), torch.ones(1, 2), {"c0": 0, "c1": 1}),
        ((("c0", "a"),), torch.ones(1, 2), {"c0": 0, "c1": 2}),
    ),
)
def test_cache_rejects_invalid_tensor_or_label_inventory(
    records: tuple[tuple[str, str], ...], features: torch.Tensor, label_map: dict[str, int]
) -> None:
    """Fails if invalid cache bytes can enter the canonical cache."""
    with pytest.raises(ValueError):
        module.build_fepf_cache(records, features, label_map)


def _fit_cache() -> module.FepfCache:
    labels = torch.arange(8, dtype=torch.int64).repeat_interleave(16)
    features = torch.zeros(128, 768, dtype=torch.float32)
    features[torch.arange(128), labels] = 1.0
    features[:, 32] = torch.linspace(0.01, 0.02, 128)
    records = tuple((f"c{int(label)}", f"{index}.jpg") for index, label in enumerate(labels))
    label_map = {f"c{index}": index for index in range(8)}
    return module.build_fepf_cache(records, features.contiguous(), label_map)


def test_prepare_fepf_start_head_projects_random_rows_and_copies_means() -> None:
    """Fails if random heads bypass finite/zero checks or mean heads are altered."""
    cache = _fit_cache()
    means = module.canonical_class_means(cache)
    random_head = torch.arange(1, means.numel() + 1, dtype=torch.float32).reshape_as(means)

    assert torch.equal(
        module.prepare_fepf_start_head(random_head, means, mode="fepf_mean"), means
    )
    prepared = module.prepare_fepf_start_head(random_head, means, mode="fepf_random")
    module.validate_projected_head(prepared)
    with pytest.raises(ValueError):
        module.prepare_fepf_start_head(torch.zeros_like(random_head), means, mode="fepf_random")


def test_fit_uses_registered_pseudoepochs_and_continuous_mask_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails if fitting restarts an epoch or mask stream before step 512."""
    cache = _fit_cache()
    start = module.canonical_class_means(cache)
    observed_epochs: list[int] = []
    batch_seed = experiment_stream_seed(7, 23_001)

    def recording_indices(**kwargs: int) -> tuple[int, ...]:
        if kwargs["seed"] == batch_seed:
            observed_epochs.append(kwargs["epoch"])
            return tuple(range(128)) * 161
        return tuple(range(128))

    def constant_loss(
        _features: torch.Tensor,
        head: torch.Tensor,
        _labels: torch.Tensor,
        _masks: torch.Tensor,
        **_kwargs: float,
    ) -> torch.Tensor:
        return head.sum() * 0.0

    monkeypatch.setattr(module, "padded_epoch_indices", recording_indices)
    monkeypatch.setattr(module, "sharded_mask_arcface_loss", constant_loss)
    result = module.fit_fepf_head(cache, start, training_seed=7, device=torch.device("cpu"))

    assert observed_epochs == [0, 1, 2, 3]
    assert result.completed_steps == 512
    assert result.batch_root_seed == experiment_stream_seed(7, 23_001)
    assert result.mask_root_seed == experiment_stream_seed(7, 23_002)
    assert result.mask_generator_initial_sha256 != result.mask_generator_final_sha256
    assert result.diagnostic_indices == tuple(range(128))
    module.validate_projected_head(result.head)


def test_fit_is_rng_neutral_and_rejects_an_invalid_start_head() -> None:
    """Fails if fitting consumes global state or accepts an unprojected start."""
    cache = _fit_cache()
    start = module.canonical_class_means(cache)
    python_before = random.getstate()
    torch_before = torch.random.get_rng_state().clone()

    result = module.fit_fepf_head(
        cache, start, training_seed=3, device=torch.device("cpu"), steps=1
    )

    assert result.completed_steps == 1
    assert random.getstate() == python_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    with pytest.raises(ValueError):
        module.fit_fepf_head(
            cache,
            torch.ones_like(start),
            training_seed=3,
            device=torch.device("cpu"),
            steps=1,
        )


def test_initialization_receipt_validates_all_mode_specific_relations() -> None:
    """Fails if the immutable receipt accepts changed mode, hash, or scalar fields."""
    cache = _fit_cache()
    head = module.canonical_class_means(cache)
    diagnostic = module.registered_diagnostic(cache.features, cache.labels, head, training_seed=5)
    audit = module.InitializationRngAudit(*(("a" * 64,) * 9), *(() for _ in range(3)))
    receipt = module.initialization_receipt_v2(
        mode="imprinted",
        training_seed=5,
        holdout_fraction=0.2,
        holdout_seed=11,
        source_sha256="1" * 64,
        checkpoint_sha256="2" * 64,
        config_sha256="3" * 64,
        schedule_sha256="4" * 64,
        official_random_head_sha256="5" * 64,
        prepared_start_head_sha256=module.tensor_sha256(head),
        final_head_sha256=module.tensor_sha256(head),
        initialization_seconds=1.0,
        cache=cache,
        classifier_shape=tuple(head.shape),
        diagnostic=diagnostic,
        rng_audit=audit,
    )

    module.validate_initialization_receipt_v2(receipt, device=torch.device("cpu"))
    changed = dict(receipt)
    changed["training_seed"] = True
    with pytest.raises(ValueError):
        module.validate_initialization_receipt_v2(changed, device=torch.device("cpu"))
