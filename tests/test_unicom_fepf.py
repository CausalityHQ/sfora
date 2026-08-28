from __future__ import annotations

import hashlib
import random
from dataclasses import replace

import numpy as np
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


def test_projected_head_accepts_only_the_registered_norm_tolerance() -> None:
    """Fails if the frozen norm tolerance is widened or a post-scale check is skipped."""
    target = 0.01 * 768**0.5
    allowed_delta = module.FEPF_ROW_NORM_ATOL + module.FEPF_ROW_NORM_RTOL * target
    inside = torch.zeros(2, 768, dtype=torch.float32)
    inside[0, 0] = target + allowed_delta * 0.9
    inside[1, 1] = target - allowed_delta * 0.9
    module.validate_projected_head(inside.contiguous())

    outside = inside.clone()
    outside[0, 0] = target + allowed_delta * 1.1
    with pytest.raises(ValueError):
        module.validate_projected_head(outside.contiguous())


def test_projection_rechecks_the_actual_post_scale_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails if projection trusts its formula instead of reading the scaled head again."""
    target = 0.01 * 768**0.5
    head = torch.zeros(2, 768, dtype=torch.float32)
    head[0, 0] = target * 2
    head[1, 1] = target * 2
    original = module.validate_projected_head

    def corrupt_after_scale(values: torch.Tensor) -> None:
        values.mul_(2.0)
        original(values)

    monkeypatch.setattr(module, "validate_projected_head", corrupt_after_scale)
    with pytest.raises(ValueError):
        module.project_and_validate_head_(head)


def test_fit_uses_registered_pseudoepochs_and_continuous_mask_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails if fitting restarts an epoch or mask stream before step 512."""
    cache = _fit_cache()
    start = module.canonical_class_means(cache)
    observed_epochs: list[int] = []
    producing_epochs: list[int | None] = []
    active_epoch: int | None = None
    batch_seed = experiment_stream_seed(7, 23_001)

    def recording_indices(**kwargs: int) -> tuple[int, ...]:
        nonlocal active_epoch
        if kwargs["seed"] == batch_seed:
            observed_epochs.append(kwargs["epoch"])
            active_epoch = kwargs["epoch"]
            return tuple(range(128)) * 161
        active_epoch = None
        return tuple(range(128))

    def constant_loss(
        _features: torch.Tensor,
        head: torch.Tensor,
        _labels: torch.Tensor,
        _masks: torch.Tensor,
        **_kwargs: float,
    ) -> torch.Tensor:
        producing_epochs.append(active_epoch)
        return head.sum() * 0.0

    monkeypatch.setattr(module, "padded_epoch_indices", recording_indices)
    monkeypatch.setattr(module, "sharded_mask_arcface_loss", constant_loss)
    result = module._fit_fepf_head_core(
        cache, start, training_seed=7, device=torch.device("cpu")
    )

    assert observed_epochs == [0, 1, 2, 3]
    assert producing_epochs == [None] + [0] * 161 + [1] * 161 + [2] * 161 + [3] * 29 + [None]
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
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    result = module._fit_fepf_head_core(
        cache, start, training_seed=3, device=torch.device("cpu"), steps=1
    )

    assert result.completed_steps == 1
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    with pytest.raises(ValueError):
        module._fit_fepf_head_core(
            cache,
            torch.ones_like(start),
            training_seed=3,
            device=torch.device("cpu"),
            steps=1,
        )


def test_registered_fit_rejects_cpu_and_noncanonical_step_count() -> None:
    """Fails if the public initializer exposes an unregistered fit schedule."""
    cache = _fit_cache()
    start = module.canonical_class_means(cache)

    with pytest.raises(ValueError):
        module.fit_fepf_head(
            cache, start, training_seed=3, device=torch.device("cpu"), steps=512
        )
    with pytest.raises(ValueError):
        module.fit_fepf_head(
            cache, start, training_seed=3, device=torch.device("cuda"), steps=1
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_registered_cuda_fit_matches_the_only_accepted_schedule() -> None:
    """Fails if the public CUDA path diverges from the frozen 512-step contract."""
    cache = _fit_cache()
    start = module.canonical_class_means(cache)

    result = module.fit_fepf_head(
        cache, start, training_seed=3, device=torch.device("cuda"), steps=512
    )

    assert result.completed_steps == 512
    assert result.head.device.type == "cuda"
    module.validate_projected_head(result.head)


def test_registered_receipt_rejects_the_private_cpu_test_path() -> None:
    """Fails if a CPU/short-step test artifact can become an official receipt."""
    cache = _fit_cache()
    means = module.canonical_class_means(cache)
    with pytest.raises(ValueError):
        module.initialization_receipt_v2(
            **_provenance(),
            official_random_head=torch.ones_like(means),
            prepared_start_head=means,
            initialization_seconds=1.0,
            cache=cache,
            rng_audit=_rng_audit(),
            fit=None,
            device=torch.device("cpu"),
        )


def _rng_audit() -> module.InitializationRngAudit:
    return module.InitializationRngAudit(
        "a" * 64,
        "a" * 64,
        "a" * 64,
        "b" * 64,
        "b" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "d" * 64,
        (),
        (),
        (),
    )


def _provenance() -> dict[str, object]:
    return {
        "mode": "imprinted",
        "training_seed": 5,
        "holdout_fraction": 0.2,
        "holdout_seed": 11,
        "source_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "schedule_sha256": "4" * 64,
    }


def _receipt_core(mode: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Build a cheap private fixture; the public path is CUDA-only."""
    cache = _fit_cache()
    means = module.canonical_class_means(cache)
    random_head = torch.arange(1, means.numel() + 1, dtype=torch.float32).reshape_as(means)
    prepared = means if mode != "fepf_random" else module.prepare_fepf_start_head(
        random_head, means, mode=mode
    )

    def constant_loss(
        _features: torch.Tensor,
        head: torch.Tensor,
        _labels: torch.Tensor,
        _masks: torch.Tensor,
        **_kwargs: float,
    ) -> torch.Tensor:
        return head.sum() * 0.0

    monkeypatch.setattr(module, "sharded_mask_arcface_loss", constant_loss)
    fit = (
        None
        if mode == "imprinted"
        else module._fit_fepf_head_core(
            cache, prepared, training_seed=5, device=torch.device("cpu"), steps=512
        )
    )
    provenance = _provenance()
    provenance["mode"] = mode
    return module._initialization_receipt_v2_core(
        **provenance,
        official_random_head=random_head,
        prepared_start_head=prepared,
        initialization_seconds=1.0,
        cache=cache,
        rng_audit=_rng_audit(),
        fit=fit,
        device=torch.device("cpu"),
        allow_test_device=True,
    )


@pytest.mark.parametrize("mode", ("imprinted", "fepf_mean", "fepf_random"))
def test_initialization_receipt_recomputes_canonical_mode_relations(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fails if a receipt can bind a head, diagnostic, or mask state not derived by FEPF."""
    receipt = _receipt_core(mode, monkeypatch)

    module._validate_initialization_receipt_v2_core(
        receipt, expected=dict(receipt), device=torch.device("cpu"), allow_test_device=True
    )
    changed = dict(receipt)
    changed["final_head_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        module._validate_initialization_receipt_v2_core(
            changed, expected=dict(receipt), device=torch.device("cpu"), allow_test_device=True
        )


def test_receipt_rejects_each_one_field_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails if any ordered receipt field can drift on resume without detection."""
    receipt = _receipt_core("imprinted", monkeypatch)
    for key, value in receipt.items():
        changed = dict(receipt)
        if type(value) is str:
            changed[key] = "f" * 64 if key.endswith("_sha256") else "other"
        elif type(value) is int:
            changed[key] = value + 1
        elif type(value) is float:
            changed[key] = value + 1.0
        elif type(value) is list:
            changed[key] = list(reversed(value)) if value else ["f" * 64]
        else:
            raise AssertionError(f"unhandled receipt field {key}")
        with pytest.raises(ValueError):
            module._validate_initialization_receipt_v2_core(
                changed,
                expected=dict(receipt),
                device=torch.device("cpu"),
                allow_test_device=True,
            )


def test_resume_validation_requires_requested_provenance_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails if a structurally valid receipt can self-authenticate another run."""
    receipt = _receipt_core("imprinted", monkeypatch)
    with pytest.raises(ValueError):
        module._validate_initialization_receipt_v2_core(
            receipt,
            device=torch.device("cpu"),
            allow_test_device=True,
            expected={**_provenance(), "training_seed": 6},
        )


def test_receipt_builder_rejects_noncanonical_imprinted_and_mean_starts() -> None:
    """Fails if canonical cache means are replaced by arbitrary projected rows."""
    cache = _fit_cache()
    means = module.canonical_class_means(cache)
    altered = means.clone()
    altered[0] = torch.roll(altered[0], shifts=1)
    random_head = torch.ones_like(means)
    common = {
        **_provenance(),
        "official_random_head": random_head,
        "initialization_seconds": 1.0,
        "cache": cache,
        "rng_audit": _rng_audit(),
        "fit": None,
        "device": torch.device("cpu"),
        "allow_test_device": True,
    }
    for mode in ("imprinted", "fepf_mean"):
        with pytest.raises(ValueError):
            module._initialization_receipt_v2_core(
                **{**common, "mode": mode, "prepared_start_head": altered}
            )


def test_rng_audit_requires_entry_draw_restore_relations() -> None:
    """Fails if non-Torch CPU global streams move during the official CPU draw."""
    audit = _rng_audit()
    module._validate_rng_audit(audit)
    with pytest.raises(ValueError):
        module._validate_rng_audit(replace(audit, numpy_rng_post_draw_sha256="e" * 64))


def test_private_fit_preserves_python_numpy_and_torch_rng_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails if an injected fit exception leaks global RNG changes."""
    cache = _fit_cache()
    start = module.canonical_class_means(cache)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    def fail_loss(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("injected failure")

    monkeypatch.setattr(module, "sharded_mask_arcface_loss", fail_loss)
    with pytest.raises(RuntimeError, match="injected failure"):
        module._fit_fepf_head_core(
            cache, start, training_seed=3, device=torch.device("cpu"), steps=1
        )

    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_private_fit_has_finite_head_gradients_without_backbone_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails if the real objective leaves a nonfinite head or differentiates frozen features."""
    cache = _fit_cache()
    start = module.canonical_class_means(cache)
    observed_gradients: list[torch.Tensor] = []
    original_step = torch.optim.AdamW.step

    def record_step(optimizer: torch.optim.AdamW, *args: object, **kwargs: object) -> object:
        parameter = optimizer.param_groups[0]["params"][0]
        assert isinstance(parameter, torch.Tensor)
        assert parameter.grad is not None
        observed_gradients.append(parameter.grad.detach().clone())
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)
    result = module._fit_fepf_head_core(
        cache, start, training_seed=3, device=torch.device("cpu"), steps=1
    )

    assert len(observed_gradients) == 1
    assert torch.isfinite(observed_gradients[0]).all()
    assert cache.features.requires_grad is False
    assert cache.features.grad is None
    module.validate_projected_head(result.head)
