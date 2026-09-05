"""Language correspondence must affect image gradients, never text targets."""

import copy
import math
from typing import cast

import pytest
import torch

from sfora.siglip_language_guidance import language_centroid_cross_entropy, standardized_text_gram


@pytest.mark.parametrize("microbatch", [1, 2, 6])
@pytest.mark.parametrize("mode", ["base", "correct", "permuted", "relational"])
def test_full_batch_replay_matches_direct_gradients_and_one_optimizer_step(
    microbatch: int, mode: str
) -> None:
    from sfora.siglip_depth_recovery import relational_cross_entropy
    from sfora.siglip_language_guidance import recomputed_language_backward
    from sfora.siglip_proxy_control import PooledProxyAnchorModel
    from sfora.token_set_proxy_anchor import proxy_anchor_loss

    torch.manual_seed(17)
    direct = PooledProxyAnchorModel(
        tower=torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh()),
        input_dimensions=5,
        embedding_dimensions=3,
        class_count=4,
    )
    replay = copy.deepcopy(direct)
    x = torch.randn(6, 4)
    labels = torch.tensor([0, 0, 2, 2, 3, 3])
    text = torch.nn.functional.normalize(torch.randn(4, 5), dim=1)
    target = None if mode == "base" else standardized_text_gram(text)
    if mode == "permuted":
        order = torch.tensor([2, 0, 3, 1])
        assert target is not None
        target = target[order][:, order]
    teacher = (
        torch.nn.functional.normalize(torch.randn(6, 3), dim=1) if mode == "relational" else None
    )
    direct_opt = torch.optim.AdamW(direct.parameters(), lr=0.001, foreach=False)
    replay_opt = torch.optim.AdamW(replay.parameters(), lr=0.001, foreach=False)
    z = direct.encode(x)
    scores = z @ torch.nn.functional.normalize(direct.proxies, dim=1).T
    objective = proxy_anchor_loss(scores, labels, alpha=32.0, delta=0.1)
    if target is not None:
        direct_language = language_centroid_cross_entropy(z, labels, target)
        objective = objective + direct_language
    else:
        direct_language = z.new_zeros(())
    if teacher is not None:
        objective = objective + relational_cross_entropy(z, teacher)
    torch.autograd.backward(objective)
    evidence = recomputed_language_backward(
        replay,
        x,
        labels,
        text_gram=target,
        teacher_descriptors=teacher,
        microbatch_size=microbatch,
    )
    assert float(evidence.loss) == pytest.approx(float(objective.detach()), abs=2e-5)
    assert evidence.maximum_descriptor_disagreement <= 2e-5
    assert float(evidence.language_loss) == pytest.approx(float(direct_language.detach()), abs=2e-5)
    assert evidence.language_loss.requires_grad is False
    if mode == "base":
        assert float(evidence.language_loss) == 0.0
    for (dn, dp), (rn, rp) in zip(
        direct.named_parameters(), replay.named_parameters(), strict=True
    ):
        assert dn == rn and dp.grad is not None and rp.grad is not None
        assert torch.allclose(dp.grad, rp.grad, atol=2e-5, rtol=2e-5), dn
    for model, optimizer in ((direct, direct_opt), (replay, replay_opt)):
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 10.0, error_if_nonfinite=True, foreach=False
        )
        optimizer.step()
    for dp, rp in zip(direct.parameters(), replay.parameters(), strict=True):
        assert torch.allclose(dp, rp, atol=2e-6, rtol=2e-6)


def test_text_standardization_uses_all_off_diagonals_and_detaches_target() -> None:
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], requires_grad=True)
    gram = standardized_text_gram(text)
    # Off-diagonal values0, .6, .8 each occur twice: population mean7/15,
    # variance26/225. Diagonal is excluded rather than contributing ones.
    expected = torch.tensor(
        [
            [0.0, -7 / math.sqrt(26), 2 / math.sqrt(26)],
            [-7 / math.sqrt(26), 0.0, 5 / math.sqrt(26)],
            [2 / math.sqrt(26), 5 / math.sqrt(26), 0.0],
        ]
    )
    assert torch.allclose(gram, expected, atol=2e-6, rtol=0)
    assert gram.requires_grad is False and text.grad is None
    with pytest.raises(ValueError):
        standardized_text_gram(torch.eye(3))
    with pytest.raises(ValueError):
        standardized_text_gram(torch.tensor([[1.0, 0.0], [float("nan"), 1.0]]))


def test_language_loss_matches_independent_centroid_gradient_and_correspondence() -> None:
    descriptors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.6],
            [0.0, 0.0, 1.0],
            [0.8, 0.0, 0.6],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    target = torch.tensor([[0.0, -1.0, 1.0], [-1.0, 0.0, 0.5], [1.0, 0.5, 0.0]], requires_grad=True)
    loss = language_centroid_cross_entropy(descriptors, labels, target)
    torch.autograd.backward(loss)
    # Independent analytical gradient includes BOTH similarity endpoints and
    # normalization of each two-image centroid; no loss helper is reused.
    means = descriptors.detach().reshape(3, 2, 3).mean(dim=1)
    norms = means.norm(dim=1, keepdim=True)
    centers = means / norms
    mask = ~torch.eye(3, dtype=torch.bool)
    ql = target.detach()[mask].reshape(3, 2).softmax(dim=1)
    qs = ((centers @ centers.T)[mask].reshape(3, 2) / 0.1).softmax(dim=1)
    residual = torch.zeros(3, 3)
    residual[mask] = ((qs - ql) / 0.3).reshape(-1)
    dm = (residual + residual.T) @ centers
    dmean = (dm - centers * (dm * centers).sum(dim=1, keepdim=True)) / norms
    expected = dmean.repeat_interleave(2, dim=0) / 2
    assert descriptors.grad is not None
    assert torch.allclose(descriptors.grad, expected, atol=2e-6, rtol=2e-6)
    assert target.grad is None
    order = torch.tensor([2, 0, 1])
    wrong = language_centroid_cross_entropy(descriptors.detach(), labels, target[order][:, order])
    assert not torch.isclose(loss.detach(), wrong, atol=1e-4, rtol=0)


def test_language_loss_rejects_unbalanced_batches_and_zero_centroids() -> None:
    z = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        language_centroid_cross_entropy(z, torch.tensor([0, 0, 1, 1]), targets)
    with pytest.raises(ValueError):
        language_centroid_cross_entropy(z, torch.tensor([0, 1, 1, 1]), targets)
    with pytest.raises(ValueError):
        language_centroid_cross_entropy(z, torch.tensor([0, 0, 1, 2]), targets)


def test_two_class_language_softmax_is_rejected_instead_of_silent_zero_signal() -> None:
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        language_centroid_cross_entropy(
            z, torch.tensor([0, 0, 1, 1]), torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        )


def test_three_class_language_batch_rejects_unequal_counts_and_zero_centroid() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2])
    z = torch.eye(3)[labels]
    target = torch.tensor([[0.0, 1.0, -1.0], [1.0, 0.0, 0.5], [-1.0, 0.5, 0.0]])
    with pytest.raises(ValueError):
        language_centroid_cross_entropy(z, labels, target)
    balanced = z[:6].clone()
    balanced[1] *= -1
    with pytest.raises(ValueError):
        language_centroid_cross_entropy(balanced, labels[:6], target)


@pytest.mark.parametrize(
    "mutation",
    [
        "nan-input",
        "microbatch",
        "label-range",
        "target-shape",
        "dirty-grad",
        "frozen",
        "non-fp32",
        "zero-proxy",
        "dropout",
        "target-nan",
    ],
)
def test_replay_rejects_invalid_training_state(mutation: str) -> None:
    from sfora.siglip_language_guidance import recomputed_language_backward
    from sfora.siglip_proxy_control import PooledProxyAnchorModel

    torch.manual_seed(7)
    model = PooledProxyAnchorModel(
        tower=torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh()),
        input_dimensions=5,
        embedding_dimensions=3,
        class_count=4,
    )
    x = torch.randn(6, 4)
    labels = torch.tensor([0, 0, 2, 2, 3, 3])
    target = standardized_text_gram(torch.nn.functional.normalize(torch.randn(4, 5), dim=1))
    microbatch = 2
    if mutation == "nan-input":
        x[0, 0] = float("nan")
    elif mutation == "microbatch":
        microbatch = 4
    elif mutation == "label-range":
        labels[0] = 4
    elif mutation == "target-shape":
        target = target[:3, :3]
    elif mutation == "dirty-grad":
        model.proxies.grad = torch.zeros_like(model.proxies)
    elif mutation == "frozen":
        model.proxies.requires_grad_(False)
    elif mutation == "non-fp32":
        model.double()
    elif mutation == "zero-proxy":
        with torch.no_grad():
            model.proxies.zero_()
    elif mutation == "dropout":
        model.tower.add_module("stochastic", torch.nn.Dropout(0.5))
    elif mutation == "target-nan":
        target[0, 1] = float("nan")
    with pytest.raises(ValueError):
        recomputed_language_backward(model, x, labels, text_gram=target, microbatch_size=microbatch)


def test_replay_detects_real_stateful_forward_disagreement() -> None:
    from sfora.siglip_language_guidance import recomputed_language_backward
    from sfora.siglip_proxy_control import PooledProxyAnchorModel

    class DriftingTower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.linear = torch.nn.Linear(4, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            y = self.linear(x)
            y[:, 0] = y[:, 0] + self.calls
            return cast(torch.Tensor, y)

    torch.manual_seed(7)
    model = PooledProxyAnchorModel(
        tower=DriftingTower(),
        input_dimensions=5,
        embedding_dimensions=3,
        class_count=4,
    )
    target = standardized_text_gram(torch.nn.functional.normalize(torch.randn(4, 5), dim=1))
    with pytest.raises(RuntimeError, match="replay disagreement"):
        recomputed_language_backward(
            model,
            torch.randn(6, 4),
            torch.tensor([0, 0, 2, 2, 3, 3]),
            text_gram=target,
            microbatch_size=2,
        )
