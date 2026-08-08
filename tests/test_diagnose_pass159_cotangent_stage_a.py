"""Focused tests for the preregistered Pass159 Stage-A diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnose_pass159_cotangent_stage_a.py"
)
_SPEC = importlib.util.spec_from_file_location("diagnose_pass159_cotangent_stage_a", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_angular_proxy_anchor_cotangent_matches_singleton_autograd() -> None:
    rng = np.random.default_rng(159)
    z_np = _unit(rng.normal(size=(1, 7)))[0]
    proxies_np = _unit(rng.normal(size=(5, 7)))
    proxy_labels = np.asarray([10, 20, 30, 40, 50], dtype=np.int64)

    analytic = _MODULE.angular_proxy_anchor_cotangent(
        z_np,
        30,
        proxies_np,
        proxy_labels,
        alpha=32.0,
        delta=0.1,
    )

    z = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)
    proxies = torch.tensor(proxies_np, dtype=torch.float64)
    similarities = z @ proxies.T
    own = similarities[2]
    positive = torch.nn.functional.softplus(32.0 * (0.1 - own))
    foreign = torch.cat((similarities[:2], similarities[3:]))
    negative = torch.nn.functional.softplus(32.0 * (foreign + 0.1)).mean()
    ambient = torch.autograd.grad(positive + negative, z)[0]
    expected = ambient - torch.dot(ambient, z) * z

    np.testing.assert_allclose(analytic, expected.detach().numpy(), atol=1e-11, rtol=1e-11)
    assert float(np.dot(analytic, z_np)) == pytest.approx(0.0, abs=1e-11)


def test_parallel_transport_preserves_norm_and_target_tangency() -> None:
    origin = np.asarray([1.0, 0.0, 0.0])
    target = np.asarray([0.0, 1.0, 0.0])
    tangent = np.asarray([0.0, 0.6, 0.8])

    transported = _MODULE.parallel_transport(tangent, origin, target)

    assert float(np.dot(transported, target)) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(transported) == pytest.approx(np.linalg.norm(tangent), abs=1e-12)


def test_parallel_transport_rejects_antipodal_or_zero_inputs() -> None:
    with pytest.raises(ValueError, match="antipodal"):
        _MODULE.parallel_transport(
            np.asarray([0.0, 1.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([-1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="unit"):
        _MODULE.parallel_transport(
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 0.0]),
        )


def test_smooth_margin_gradient_matches_autograd_with_frozen_foreign_set() -> None:
    receiver = _unit(np.asarray([[1.0, 0.3, -0.2]]))[0]
    positives = _unit(np.asarray([[0.9, 0.2, 0.1], [0.7, 0.4, -0.1]]))
    foreign = _unit(
        np.asarray(
            [
                [0.2, 0.9, 0.1],
                [-0.1, 0.8, 0.3],
                [0.4, -0.2, 0.9],
            ]
        )
    )

    analytic = _MODULE.smooth_margin_gradient(receiver, positives, foreign, tau=0.05)

    z = torch.tensor(receiver, dtype=torch.float64, requires_grad=True)
    pos = torch.tensor(positives, dtype=torch.float64)
    neg = torch.tensor(foreign, dtype=torch.float64)
    margin = 0.05 * torch.logsumexp((z @ pos.T) / 0.05, dim=0)
    margin -= 0.05 * torch.logsumexp((z @ neg.T) / 0.05, dim=0)
    ambient = torch.autograd.grad(margin, z)[0]
    expected = ambient - torch.dot(ambient, z) * z

    np.testing.assert_allclose(analytic, expected.detach().numpy(), atol=1e-11, rtol=1e-11)


def test_partition_identity_is_input_order_invariant_and_disjoint() -> None:
    ids = np.asarray(["img-c", "img-a", "img-e", "img-b", "img-d", "img-f"])
    support, controllers = _MODULE.partition_identity(ids)
    support_ids = set(ids[support])
    controller_ids = set(ids[controllers])

    permutation = np.asarray([4, 2, 0, 5, 1, 3])
    shuffled = ids[permutation]
    support_2, controllers_2 = _MODULE.partition_identity(shuffled)

    assert support_ids == set(shuffled[support_2])
    assert controller_ids == set(shuffled[controllers_2])
    assert len(support_ids) == 2
    assert support_ids.isdisjoint(controller_ids)


def test_partition_identity_requires_two_supports_and_three_controllers() -> None:
    with pytest.raises(ValueError, match="at least five"):
        _MODULE.partition_identity(np.asarray(["a", "b", "c", "d"]))
