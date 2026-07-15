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
