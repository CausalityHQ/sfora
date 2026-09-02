"""Shared faithful PFML potential energy."""

from __future__ import annotations

from typing import Any


def pfml_potential_loss(
    embeddings: Any,
    labels: Any,
    *,
    proxy_embeddings: Any | None,
    proxy_labels: Any | None,
    delta: float,
    alpha: float,
    torch_module: Any,
) -> Any:
    """Return the raw PFML total energy over samples and normalized proxies.

    This is the faithful Eq. 1--6 ordered-pair energy used by the Sfora PFML
    trainer. Attraction saturates inside ``delta`` and repulsion saturates
    outside it. Self interactions are excluded because they are constant.
    """

    if proxy_embeddings is None or proxy_labels is None:
        raise ValueError("the pfml objective requires class proxies (proxy_count_per_class > 0)")
    if (
        getattr(embeddings, "ndim", None) != 2
        or getattr(labels, "ndim", None) != 1
        or getattr(proxy_embeddings, "ndim", None) != 2
        or getattr(proxy_labels, "ndim", None) != 1
        or embeddings.shape[0] != labels.shape[0]
        or proxy_embeddings.shape[0] != proxy_labels.shape[0]
        or embeddings.shape[1] != proxy_embeddings.shape[1]
        or not bool(torch_module.isfinite(embeddings).all())
        or not bool(torch_module.isfinite(proxy_embeddings).all())
        or type(delta) is not float
        or not delta > 0.0
        or type(alpha) is not float
        or alpha < 0.0
    ):
        raise ValueError("PFML tensor authority differs")
    normalized_proxies = torch_module.nn.functional.normalize(proxy_embeddings, dim=-1)
    points = torch_module.cat([embeddings, normalized_proxies], dim=0)
    point_labels = torch_module.cat([labels, proxy_labels], dim=0)
    if points.shape[0] < 2:
        return embeddings.sum() * 0.0
    distances = torch_module.cdist(points, points, p=2).clamp_min(1.0e-4)
    same_label = point_labels[:, None].eq(point_labels[None, :])
    off_diagonal = ~torch_module.eye(
        points.shape[0], dtype=torch_module.bool, device=points.device
    )
    inside_margin = distances < delta
    inverse_power = distances.pow(-alpha)
    saturation = torch_module.full_like(distances, delta**-alpha)
    attraction = -torch_module.where(inside_margin, saturation, inverse_power)
    repulsion = torch_module.where(inside_margin, inverse_power, saturation)
    potentials = torch_module.where(same_label, attraction, repulsion)
    return potentials[off_diagonal].sum()
