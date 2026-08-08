from typing import Any

import pytest


def test_registry_dispatch_matches_hist_and_proxy_anchor_helpers() -> None:
    torch: Any = pytest.importorskip("torch")

    import sfora.image_end_to_end as image_end_to_end

    assert hasattr(image_end_to_end, "_OBJECTIVE_LOSSES")

    config = image_end_to_end.ImageEndToEndConfig()
    embeddings = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    hist_module = image_end_to_end._build_hist_module(
        nb_classes=3,
        sz_embed=4,
        hidden=8,
        torch_module=torch,
    )
    label_to_index = {0: 0, 1: 1, 2: 2}
    proxies = torch.randn(3, 4, generator=torch.Generator().manual_seed(1))
    proxy_labels = torch.tensor([0, 1, 2])

    dispatched_hist = image_end_to_end._loss_for_objective(
        "hist",
        embeddings,
        labels,
        step=0,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=None,
        proxy_labels=None,
        config=config,
        torch_module=torch,
        hist_module=hist_module,
        hist_label_to_index=label_to_index,
    )
    direct_hist = image_end_to_end._hist_loss(
        embeddings,
        labels,
        hist_module=hist_module,
        label_to_index=label_to_index,
        tau=config.hist_tau,
        alpha=config.hist_alpha,
        lambda_s=config.hist_lambda_s,
        var_floor=config.hist_var_floor,
        torch_module=torch,
    )

    dispatched_proxy_anchor = image_end_to_end._loss_for_objective(
        "proxy_anchor",
        embeddings,
        labels,
        step=0,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        config=config,
        torch_module=torch,
    )
    direct_proxy_anchor = image_end_to_end._proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        torch_module=torch,
    )

    assert torch.allclose(dispatched_hist, direct_hist)
    assert torch.allclose(dispatched_proxy_anchor, direct_proxy_anchor)


def test_coalition_proxy_loss_is_permutation_invariant_and_uses_union_target() -> None:
    torch: Any = pytest.importorskip("torch")

    import sfora.image_end_to_end as image_end_to_end

    config = image_end_to_end.ImageEndToEndConfig()
    embeddings = torch.randn(4, 8, generator=torch.Generator().manual_seed(3), requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    proxies = torch.randn(4, 8, generator=torch.Generator().manual_seed(4))
    proxy_labels = torch.tensor([0, 1, 2, 3])

    loss = image_end_to_end._loss_for_objective(
        "proxy_anchor_coalition",
        embeddings,
        labels,
        step=0,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        config=config,
        torch_module=torch,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(embeddings.grad).all()
    assert (embeddings.grad.norm(dim=1) > 0).all()

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_loss = image_end_to_end._loss_for_objective(
        "proxy_anchor_coalition",
        embeddings.detach()[permutation],
        labels[permutation],
        step=0,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        config=config,
        torch_module=torch,
    )
    assert torch.allclose(loss.detach(), permuted_loss)

    changed_labels = labels.clone()
    changed_labels[0] = 1
    changed_loss = image_end_to_end._loss_for_objective(
        "proxy_anchor_coalition",
        embeddings.detach(),
        changed_labels,
        step=0,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        config=config,
        torch_module=torch,
    )
    assert not torch.allclose(loss.detach(), changed_loss)


def test_coalition_controls_use_same_operator_and_differ_from_union() -> None:
    torch: Any = pytest.importorskip("torch")
    import sfora.image_end_to_end as image_end_to_end

    embeddings = torch.randn(4, 8, generator=torch.Generator().manual_seed(13))
    labels = torch.tensor([0, 1, 2, 3])
    proxies = torch.randn(4, 8, generator=torch.Generator().manual_seed(14))
    proxy_labels = torch.tensor([0, 1, 2, 3])
    kwargs = dict(
        embeddings=embeddings,
        labels=labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        torch_module=torch,
    )
    union = image_end_to_end._coalition_proxy_loss(**kwargs, mode="union")
    single = image_end_to_end._coalition_proxy_loss(**kwargs, mode="single")
    dropout = image_end_to_end._coalition_proxy_loss(**kwargs, mode="dropout")
    assert torch.isfinite(torch.stack((union, single, dropout))).all()
    assert not torch.allclose(union, single)
    assert not torch.allclose(union, dropout)
    one = image_end_to_end._coalition_proxy_loss(
        **{**kwargs, "embeddings": embeddings[:1], "labels": labels[:1]}, mode="single"
    )
    assert torch.isfinite(one)


def test_coalition_residual_uses_complementary_target_per_omitted_member() -> None:
    torch: Any = pytest.importorskip("torch")
    import sfora.image_end_to_end as image_end_to_end

    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], requires_grad=True
    )
    labels = torch.tensor([0, 1, 2])
    proxies = torch.eye(3, 2)
    proxy_labels = torch.tensor([0, 1, 2])
    kwargs = dict(
        embeddings=embeddings,
        labels=labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        torch_module=torch,
    )
    residual = image_end_to_end._coalition_proxy_loss(**kwargs, mode="residual")
    union = image_end_to_end._coalition_proxy_loss(**kwargs, mode="union")
    dropout = image_end_to_end._coalition_proxy_loss(**kwargs, mode="dropout")
    assert torch.isfinite(residual)
    assert not torch.allclose(residual, union)
    assert not torch.allclose(residual, dropout)
    residual.backward()
    assert torch.isfinite(embeddings.grad).all()


def test_coalition_single_complementary_is_distinct_and_finite() -> None:
    torch: Any = pytest.importorskip("torch")
    import sfora.image_end_to_end as image_end_to_end

    embeddings = torch.randn(3, 6, generator=torch.Generator().manual_seed(21))
    labels = torch.tensor([0, 1, 2])
    proxies = torch.randn(3, 6, generator=torch.Generator().manual_seed(22))
    proxy_labels = torch.tensor([0, 1, 2])
    kwargs = dict(
        embeddings=embeddings,
        labels=labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        torch_module=torch,
    )
    complementary = image_end_to_end._coalition_proxy_loss(
        **kwargs, mode="single_complementary"
    )
    single = image_end_to_end._coalition_proxy_loss(**kwargs, mode="single")
    assert torch.isfinite(complementary)
    assert not torch.allclose(complementary, single)


def test_src_operator_contains_both_union_and_leave_one_out_terms() -> None:
    torch: Any = pytest.importorskip("torch")
    import sfora.image_end_to_end as image_end_to_end

    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], requires_grad=True
    )
    labels = torch.tensor([0, 1, 2])
    proxies = torch.eye(3, 2)
    proxy_labels = torch.tensor([0, 1, 2])
    kwargs = dict(
        embeddings=embeddings,
        labels=labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        torch_module=torch,
    )
    union = image_end_to_end._coalition_proxy_loss(**kwargs, mode="union")
    residual = image_end_to_end._coalition_proxy_loss(**kwargs, mode="residual")
    src = image_end_to_end._stoichiometric_residual_coalition_loss(**kwargs)
    assert torch.allclose(src, union + residual)


def test_src_dispatcher_uses_union_plus_residual_operator() -> None:
    torch: Any = pytest.importorskip("torch")
    import sfora.image_end_to_end as image_end_to_end

    config = image_end_to_end.ImageEndToEndConfig(
        objectives=("proxy_anchor_coalition",),
        coalition_mode="residual",
        coalition_weight=0.25,
        proxy_anchor_alpha=8.0,
        proxy_anchor_delta=0.1,
    )
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], requires_grad=True
    )
    labels = torch.tensor([0, 1, 2])
    proxies = torch.eye(3, 2)
    proxy_labels = torch.tensor([0, 1, 2])
    base = image_end_to_end._proxy_anchor_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        alpha=config.proxy_anchor_alpha,
        delta=config.proxy_anchor_delta,
        torch_module=torch,
    )
    src = image_end_to_end._stoichiometric_residual_coalition_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        torch_module=torch,
    )
    dispatched = image_end_to_end._loss_for_objective(
        "proxy_anchor_coalition",
        embeddings,
        labels,
        step=0,
        steps_per_epoch=1,
        memory_embeddings=None,
        memory_labels=None,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        config=config,
        torch_module=torch,
    )
    assert torch.allclose(dispatched, base + config.coalition_weight * src)

