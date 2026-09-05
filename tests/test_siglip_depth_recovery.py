"""Fixed surgery and full-batch gradient contracts for depth recovery."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import SiglipVisionConfig, SiglipVisionModel

from sfora.siglip_depth_recovery import (
    prune_siglip_student,
    recomputed_recovery_backward,
    recovery_multiplier,
    relational_cross_entropy,
    speed_gate,
)
from sfora.siglip_proxy_control import PooledProxyAnchorModel

_SPEC = importlib.util.spec_from_file_location(
    "run_siglip_proxy_control", Path(__file__).parents[1] / "scripts/run_siglip_proxy_control.py"
)
assert _SPEC and _SPEC.loader
control = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = control
_SPEC.loader.exec_module(control)


def _tiny_vision():
    torch.manual_seed(17)
    config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=27,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return PooledProxyAnchorModel(
        tower=control.SiglipPooledTower(SiglipVisionModel(config)),
        input_dimensions=16,
        embedding_dimensions=8,
        class_count=3,
    ).eval()


def test_surgery_executes_only_selected_blocks_and_independent_serialized_state(tmp_path):
    full = _tiny_vision()
    original = {k: v.clone() for k, v in full.state_dict().items()}
    student = prune_siglip_student(full)
    source = full.tower.vision_model.encoder.layers
    layers = student.tower.vision_model.encoder.layers
    retained = (0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26)
    assert len(source) == 27 and len(layers) == 18
    assert full.tower.vision_model.config.num_hidden_layers == 27
    assert student.tower.vision_model.config.num_hidden_layers == 18
    calls = []
    handles = []
    for index, source_index in enumerate(retained):
        for name, value in layers[index].state_dict().items():
            assert torch.equal(value, source[source_index].state_dict()[name])
        handles.append(
            layers[index].register_forward_hook(
                lambda module, inputs, output, n=source_index: calls.append(n)
            )
        )
    assert not (
        {p.data_ptr() for p in student.parameters()} & {p.data_ptr() for p in full.parameters()}
    )
    assert sum(p.numel() for p in student.parameters()) < sum(p.numel() for p in full.parameters())
    inputs = torch.randn(2, 3, 28, 28)
    with torch.no_grad():
        expected = student.encode(inputs)
    assert calls == list(retained)
    for handle in handles:
        handle.remove()
    for name, value in full.state_dict().items():
        assert torch.equal(value, original[name])
    path = tmp_path / "student.pt"
    torch.save(student.state_dict(), path)
    restored = prune_siglip_student(_tiny_vision())
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad():
        assert torch.equal(restored.encode(inputs), expected)
    assert not any("encoder.layers.18." in key for key in restored.state_dict())


def test_surgery_rejects_already_pruned_or_wrong_depth_without_mutation():
    model = _tiny_vision()
    model.tower.vision_model.config.num_hidden_layers = 26
    with pytest.raises(ValueError):
        prune_siglip_student(model)
    assert len(model.tower.vision_model.encoder.layers) == 27


def _loop_ce(z, teacher):
    rows = []
    for i in range(len(z)):
        js = [j for j in range(len(z)) if j != i]
        teacher_scores = torch.stack([torch.dot(teacher[i], teacher[j]) / 0.1 for j in js])
        student_scores = torch.stack([torch.dot(z[i], z[j]) / 0.1 for j in js])
        rows.append(-(teacher_scores.detach().softmax(0) * student_scores.log_softmax(0)).sum())
    return torch.stack(rows).mean()


def _loop_pa(z, raw_proxies, labels):
    p = F.normalize(raw_proxies, dim=1)
    scores = z @ p.T
    pos, neg = [], []
    for c in range(len(p)):
        positive = scores[labels == c, c]
        if len(positive):
            pos.append(torch.log1p(torch.exp(-32 * (positive - 0.1)).sum()))
        negative = scores[labels != c, c]
        neg.append(torch.log1p(torch.exp(32 * (negative + 0.1)).sum()))
    return torch.stack(pos).mean() + torch.stack(neg).mean()


def test_relational_objective_matches_independent_loop_and_both_neighbor_gradients():
    torch.manual_seed(6)
    z = F.normalize(torch.randn(5, 4), dim=1).requires_grad_()
    teacher = F.normalize(torch.randn(5, 4), dim=1).requires_grad_()
    actual = relational_cross_entropy(z, teacher)
    expected = _loop_ce(z, teacher)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    ga = torch.autograd.grad(actual, (z, teacher), allow_unused=True)
    ge = torch.autograd.grad(expected, z)[0]
    assert torch.allclose(ga[0], ge, atol=2e-6, rtol=2e-6)
    assert ga[1] is None


@pytest.mark.parametrize(
    "bad", [torch.zeros(4, 3), torch.full((4, 3), float("nan")), torch.ones(1, 3)]
)
def test_relational_objective_rejects_bad_descriptors(bad):
    with pytest.raises(ValueError):
        relational_cross_entropy(bad, bad)


def _dense_model():
    torch.manual_seed(4)
    return PooledProxyAnchorModel(
        tower=nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 5)),
        input_dimensions=5,
        embedding_dimensions=4,
        class_count=3,
    )


@pytest.mark.parametrize("microbatch", [1, 2, 4])
@pytest.mark.parametrize("distill", [False, True])
def test_replay_matches_full_batch_loss_all_gradients_and_adamw_update(microbatch, distill):
    full = _dense_model()
    replay = copy.deepcopy(full)
    inputs = torch.randn(8, 4)
    labels = torch.tensor([0, 1, 2, 0, 2, 1, 0, 1])
    teacher = F.normalize(torch.randn(8, 4), dim=1).requires_grad_() if distill else None
    fo = torch.optim.AdamW(full.parameters(), lr=1e-4, foreach=False)
    ro = torch.optim.AdamW(replay.parameters(), lr=1e-4, foreach=False)
    z = full.encode(inputs)
    expected = _loop_pa(z, full.proxies, labels)
    if teacher is not None:
        expected = expected + _loop_ce(z, teacher)
    expected.backward()
    evidence = recomputed_recovery_backward(
        replay, inputs, labels, teacher_descriptors=teacher, microbatch_size=microbatch
    )
    assert torch.allclose(evidence.loss, expected.detach(), atol=2e-5, rtol=2e-6)
    assert evidence.maximum_descriptor_disagreement <= 2e-5
    for (name, parameter), (other_name, other) in zip(
        full.named_parameters(), replay.named_parameters(), strict=True
    ):
        assert name == other_name
        assert parameter.grad is not None and other.grad is not None
        assert torch.allclose(parameter.grad, other.grad, atol=2e-5, rtol=2e-5), name
    assert teacher is None or teacher.grad is None
    fo.step()
    ro.step()
    for parameter, other in zip(full.parameters(), replay.parameters(), strict=True):
        assert torch.allclose(parameter, other, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    "bad", ["dropout", "batchnorm", "labels", "prior-grad", "nan-input", "nan-tolerance"]
)
def test_replay_rejects_nonreproducible_or_invalid_state(bad):
    model = _dense_model()
    inputs, labels = torch.randn(4, 4), torch.tensor([0, 1, 2, 0])
    if bad == "dropout":
        model.tower.append(nn.Dropout(0.2))
    elif bad == "batchnorm":
        model.tower.append(nn.BatchNorm1d(5))
    elif bad == "labels":
        labels[0] = 3
    elif bad == "prior-grad":
        model.proxies.grad = torch.ones_like(model.proxies)
    elif bad == "nan-input":
        inputs[0, 0] = float("nan")
    tolerance = float("nan") if bad == "nan-tolerance" else 2e-5
    with pytest.raises((ValueError, RuntimeError)):
        recomputed_recovery_backward(
            model, inputs, labels, microbatch_size=2, descriptor_tolerance=tolerance
        )


def test_recovery_schedule_never_makes_a_zero_update():
    assert recovery_multiplier(1) == 0.1
    assert recovery_multiplier(10) == 1.0
    assert recovery_multiplier(198) == 0.1
    assert all(0 < recovery_multiplier(k) <= 1 for k in range(1, 199))
    for invalid in (0, 199, True, 1.0):
        with pytest.raises(ValueError):
            recovery_multiplier(invalid)


def _windows():
    return [
        {scope: {"full": [400] * 100, "student": [300] * 100} for scope in ("pipeline", "encoder")}
        for _ in range(3)
    ]


def test_speed_gate_requires_each_paired_window_scope_and_exact_p95():
    windows = _windows()
    assert speed_gate(windows)
    windows[2]["encoder"]["student"][-6:] = [301] * 6
    assert not speed_gate(windows)
    windows[2]["encoder"]["student"][-6] = 300
    assert speed_gate(windows)  # Five tail samples do not change nearest-rank p95.


@pytest.mark.parametrize("bad", ["window", "sample", "bool", "zero", "scope"])
def test_speed_gate_rejects_incomplete_or_nonconcrete_measurements(bad):
    windows = _windows()
    if bad == "window":
        windows.pop()
    elif bad == "sample":
        windows[0]["encoder"]["full"].pop()
    elif bad == "bool":
        windows[0]["encoder"]["full"][0] = True
    elif bad == "zero":
        windows[0]["encoder"]["full"][0] = 0
    else:
        windows[0]["extra"] = windows[0]["pipeline"]
    with pytest.raises(ValueError):
        speed_gate(windows)
