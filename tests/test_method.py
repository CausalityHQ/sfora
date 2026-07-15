from sfora.image_end_to_end import ImageEndToEndConfig
from sfora.method import (
    HIST,
    Distill,
    FusedHistProxyAnchor,
    IsNorm,
    Objective,
    ProxyAnchor,
    build_config,
    herd,
    pa_distill,
)


def _base() -> ImageEndToEndConfig:
    return ImageEndToEndConfig(dataset_name="cub", train_epochs=60)


def test_base_bricks_set_their_objective_and_proxies() -> None:
    hist_cfg = HIST().configure(_base())
    assert hist_cfg.objectives == ("hist",)
    assert hist_cfg.proxy_count_per_class == 0

    pa_cfg = ProxyAnchor().configure(_base())
    assert pa_cfg.objectives == ("proxy_anchor",)
    assert pa_cfg.proxy_count_per_class == 1


def test_distill_modifier_adds_ema_teacher_on_any_base() -> None:
    cfg = Distill(ProxyAnchor(), weight=1.0, momentum=0.999, tau=0.1).configure(_base())
    # base preserved, distillation layered on top
    assert cfg.objectives == ("proxy_anchor",)
    assert cfg.proxy_count_per_class == 1
    assert cfg.ema_distill_weight == 1.0
    assert cfg.ema_momentum == 0.999
    assert cfg.ema_distill_tau == 0.1


def test_isnorm_modifier_adds_the_head() -> None:
    assert IsNorm(HIST()).configure(_base()).embedding_layer_norm is True
    assert HIST().configure(_base()).embedding_layer_norm is False


def test_herd_is_isnorm_distill_hist() -> None:
    cfg = build_config(herd(), _base())
    assert cfg.objectives == ("hist",)
    assert cfg.embedding_layer_norm is True
    assert cfg.ema_distill_weight == 1.0
    assert cfg.proxy_count_per_class == 0


def test_pa_distill_is_distill_proxy_anchor() -> None:
    cfg = build_config(pa_distill(), _base())
    assert cfg.objectives == ("proxy_anchor",)
    assert cfg.proxy_count_per_class == 1
    assert cfg.ema_distill_weight == 1.0
    assert cfg.embedding_layer_norm is False


def test_fused_brick_maps_to_fused_objective() -> None:
    cfg = FusedHistProxyAnchor(fusion_weight=0.5).configure(_base())
    assert cfg.objectives == ("hist_proxy_anchor",)
    assert cfg.proxy_fusion_weight == 0.5
    assert cfg.proxy_count_per_class == 1


def test_names_reflect_composition() -> None:
    assert herd().name == "IsNorm(Distill(HIST))"
    assert pa_distill().name == "Distill(ProxyAnchor)"


def test_bricks_satisfy_the_objective_protocol() -> None:
    for brick in (HIST(), ProxyAnchor(), Distill(HIST()), IsNorm(HIST()), herd(), pa_distill()):
        assert isinstance(brick, Objective)


def test_configure_is_pure_no_mutation() -> None:
    base = _base()
    herd().configure(base)
    # base unchanged
    assert base.objectives == ImageEndToEndConfig().objectives
    assert base.embedding_layer_norm is False


def test_build_config_revalidates_out_of_range_brick_values() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        build_config(HIST(tau=-1.0), _base())  # hist_tau must be > 0


def test_base_brick_resets_inherited_modifier_fields() -> None:
    # A base loss must be self-contained: it resets a Distill/IsNorm left on the base.
    dirty = ImageEndToEndConfig(ema_distill_weight=1.0, embedding_layer_norm=True)
    cfg = build_config(HIST(), dirty)
    assert cfg.ema_distill_weight == 0.0
    assert cfg.embedding_layer_norm is False
    # but HERD (which re-adds them) keeps them on
    herd_cfg = build_config(herd(), dirty)
    assert herd_cfg.ema_distill_weight == 1.0
    assert herd_cfg.embedding_layer_norm is True
