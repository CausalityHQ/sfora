"""Real optimizer/crop and fail-closed budget contracts for recovery smoke."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from sfora.data import ImageExample
from sfora.siglip_proxy_control import PooledProxyAnchorModel

_SPEC = importlib.util.spec_from_file_location(
    "run_siglip_recovery_smoke",
    Path(__file__).parents[1] / "scripts/run_siglip_recovery_smoke.py",
)
assert _SPEC and _SPEC.loader
smoke = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = smoke
_SPEC.loader.exec_module(smoke)


def _model():
    torch.manual_seed(17)
    return (
        PooledProxyAnchorModel(
            tower=nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Tanh()),
            input_dimensions=8,
            embedding_dimensions=5,
            class_count=3,
        )
        .float()
        .train()
    )


def _oracle_optimizer(model):
    # Independent literal parameter inventory for this actual fixture.
    return torch.optim.AdamW(
        [
            {"params": [model.tower[0].weight], "lr": 1e-6, "weight_decay": 1e-4},
            {
                "params": [model.tower[0].bias, model.tower[1].weight, model.tower[1].bias],
                "lr": 1e-6,
                "weight_decay": 0.0,
            },
            {"params": [model.projection.weight], "lr": 1e-5, "weight_decay": 1e-4},
            {"params": [model.proxies], "lr": 1e-3, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )


def _objective(model, inputs, labels, teacher):
    z = model.encode(inputs)
    scores = z @ F.normalize(model.proxies, dim=1).T
    positive, negative = [], []
    for c in range(3):
        pos = scores[labels == c, c]
        if len(pos):
            positive.append(torch.log1p(torch.exp(-32 * (pos - 0.1)).sum()))
        negative.append(torch.log1p(torch.exp(32 * (scores[labels != c, c] + 0.1)).sum()))
    loss = torch.stack(positive).mean() + torch.stack(negative).mean()
    if teacher is not None:
        with torch.no_grad():
            t = teacher.encode(inputs)
        rows = []
        for i in range(len(z)):
            js = [j for j in range(len(z)) if j != i]
            ts = torch.stack([torch.dot(t[i], t[j]) / 0.1 for j in js])
            zs = torch.stack([torch.dot(z[i], z[j]) / 0.1 for j in js])
            rows.append(-(ts.softmax(0) * zs.log_softmax(0)).sum())
        loss = loss + torch.stack(rows).mean()
    return loss


@pytest.mark.parametrize("distill", [False, True])
def test_real_update_matches_independent_objective_clipping_and_adamw(distill):
    student = _model()
    reference = copy.deepcopy(student)
    teacher = copy.deepcopy(student).eval().requires_grad_(False) if distill else None
    teacher_before = copy.deepcopy(teacher.state_dict()) if teacher else None
    inputs, labels = torch.randn(8, 4), torch.tensor([0, 0, 1, 1, 2, 2, 0, 2])
    before = copy.deepcopy(student.state_dict())
    optimizer = smoke.new_recovery_optimizer(student)
    assert not optimizer.state
    expected_optimizer = _oracle_optimizer(reference)
    expected = _objective(reference, inputs, labels, teacher)
    expected.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 10.0)
    assert expected_norm > 10  # This fixture detects removal of clipping.
    expected_optimizer.step()
    result = smoke.recovery_update(
        student,
        optimizer,
        inputs,
        labels,
        update=1,
        teacher=teacher,
        microbatch_size=4,
    )
    assert result["loss"] == pytest.approx(float(expected.detach()), abs=2e-5)
    assert result["gradient_norm"] == pytest.approx(float(expected_norm), rel=2e-5)
    for name, p in student.named_parameters():
        assert torch.allclose(p, dict(reference.named_parameters())[name], atol=2e-7, rtol=1e-6)
        assert not torch.equal(p, before[name]), name
        assert optimizer.state[p]["step"] == 1
    assert not smoke.new_recovery_optimizer(student).state  # Never inherit warm state.
    if teacher:
        assert all(p.grad is None for p in teacher.parameters())
        for name, p in teacher.state_dict().items():
            assert torch.equal(p, teacher_before[name])


@pytest.mark.parametrize("mutation", ["training", "trainable", "alias"])
def test_bad_teacher_rejected_before_student_update(mutation):
    student = _model()
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    if mutation == "training":
        teacher.train()
    elif mutation == "trainable":
        teacher.projection.weight.requires_grad_(True)
    else:
        teacher.projection.weight = student.projection.weight
        teacher.projection.weight.requires_grad_(False)
    before = copy.deepcopy(student.state_dict())
    with pytest.raises(ValueError):
        smoke.recovery_update(
            student,
            smoke.new_recovery_optimizer(student),
            torch.randn(4, 4),
            torch.tensor([0, 0, 1, 1]),
            update=1,
            teacher=teacher,
            microbatch_size=4,
        )
    for name, p in student.state_dict().items():
        assert torch.equal(p, before[name])


def test_pair_crops_are_reproducible_and_do_not_consume_caller_rng():
    examples = tuple(
        ImageExample(str(i), Image.new("RGB", (30, 40), (i, 3, 4)), i) for i in range(4)
    )

    def transform(image):
        return torch.rand(3, 2, 2) + image.getpixel((0, 0))[0]

    torch.manual_seed(303)
    rng = torch.get_rng_state().clone()
    a, labels = smoke.paired_training_batch(examples, (0, 1, 2, 3), transform, update=1)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.rand(100)
    b, labels_b = smoke.paired_training_batch(examples, (0, 1, 2, 3), transform, update=1)
    c, _ = smoke.paired_training_batch(examples, (0, 1, 2, 3), transform, update=2)
    assert torch.equal(a, b) and torch.equal(labels, labels_b)
    assert labels.tolist() == [0, 1, 2, 3]
    assert not torch.equal(a, c)


def test_budget_uses_both_arm_maxima_headroom_elapsed_and_eval_allowance():
    times = {"pa": [10.0] * 9 + [20.0], "relational": [15.0] * 9 + [30.0]}
    result = smoke.project_recovery_budget(times, elapsed_seconds=1000.0, startup_seconds=100.0)
    # Spent1000 + futurestartup100 + updates12375 + eval1800 + saveallowance300.
    assert result["projected_total_seconds"] == 15575.0
    assert result["within_six_hours"] is True
    assert (
        smoke.project_recovery_budget(times, elapsed_seconds=8000.0, startup_seconds=100.0)[
            "within_six_hours"
        ]
        is False
    )
    for invalid in (
        {"pa": [1.0] * 10},
        {"pa": [1.0] * 9, "relational": [1.0] * 10},
        {"pa": [float("nan")] * 10, "relational": [1.0] * 10},
    ):
        with pytest.raises(ValueError):
            smoke.project_recovery_budget(invalid, elapsed_seconds=1000.0, startup_seconds=100.0)
    for startup in (-1.0, float("nan"), True):
        with pytest.raises(ValueError):
            smoke.project_recovery_budget(times, elapsed_seconds=1000.0, startup_seconds=startup)


def test_registered_preflights_bind_raw_speed_and_input_proof(tmp_path, monkeypatch):
    def raw(value):
        return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

    windows = [
        {scope: {"full": [400] * 100, "student": [300] * 100} for scope in ("pipeline", "encoder")}
        for _ in range(3)
    ]
    speed = {"timing": {"windows": windows}, "original_rgb_sha256": "a" * 64}
    speed_bytes = raw(speed)
    proof = {
        "prior_speed_sha256": hashlib.sha256(speed_bytes).hexdigest(),
        "original_rgb_sha256": "a" * 64,
        "matches_prior_speed_inputs": True,
        "only_optimization_image_rows_fetched": True,
        "selected_count": 128,
        "pixel_access_count": 128,
        "quality_measured": False,
        "claim_eligible": False,
    }
    speed_path, proof_path = tmp_path / "speed.json", tmp_path / "proof.json"
    speed_path.write_bytes(speed_bytes)
    proof_path.write_bytes(raw(proof))
    monkeypatch.setattr(smoke, "SPEED_SHA256", hashlib.sha256(speed_bytes).hexdigest())
    monkeypatch.setattr(smoke, "INPUT_PROOF_SHA256", hashlib.sha256(raw(proof)).hexdigest())
    smoke.authenticate_preflights(speed_path, proof_path)
    speed["timing"]["windows"][0]["encoder"]["student"] = [301] * 100
    changed = raw(speed)
    speed_path.write_bytes(changed)
    monkeypatch.setattr(smoke, "SPEED_SHA256", hashlib.sha256(changed).hexdigest())
    proof["prior_speed_sha256"] = hashlib.sha256(changed).hexdigest()
    proof_path.write_bytes(raw(proof))
    monkeypatch.setattr(smoke, "INPUT_PROOF_SHA256", hashlib.sha256(raw(proof)).hexdigest())
    with pytest.raises(ValueError):
        smoke.authenticate_preflights(speed_path, proof_path)


def test_cli_has_only_fixed_smoke_authorities_and_explicit_execution():
    args = [
        "--control-root",
        "/fixture/control",
        "--speed-result",
        "/fixture/speed",
        "--input-proof",
        "/fixture/proof",
        "--output",
        "/fixture/output",
        "--execute-recovery-smoke",
    ]
    parsed = smoke.parse_args(args)
    assert parsed.control_root == Path("/fixture/control")
    assert parsed.execute_recovery_smoke is True
    with pytest.raises(SystemExit):
        smoke.parse_args(args[:-1])
    for flag in ("--steps", "--epochs", "--depth", "--seed", "--eval", "--microbatch"):
        with pytest.raises(SystemExit):
            smoke.parse_args(args + [flag, "1"])


def test_smoke_runs_two_fresh_students_and_same_ten_batches():
    teacher = _model().eval().requires_grad_(False)
    initial = copy.deepcopy(teacher.state_dict())
    calls = []

    def factory():
        value = copy.deepcopy(teacher).train().requires_grad_(True)
        calls.append(value)
        return value

    def batch(update):
        g = torch.Generator().manual_seed(update)
        return torch.randn(8, 4, generator=g), torch.tensor([0, 0, 1, 1, 2, 2, 0, 2])

    ticks = iter(range(100000))
    result = smoke.measure_smoke_pair(
        teacher,
        factory,
        batch,
        device=torch.device("cpu"),
        microbatch_size=4,
        synchronize=lambda: None,
        clock=lambda: next(ticks) * 1_000_000_000,
        progress=lambda event: None,
    )
    assert len(calls) == 2
    assert (
        result["arms"]["pa"]["initial_state_sha256"]
        == result["arms"]["relational"]["initial_state_sha256"]
    )
    assert result["arms"]["pa"]["input_sha256"] == result["arms"]["relational"]["input_sha256"]
    for arm in result["arms"].values():
        assert len(arm["steps"]) == 10 and arm["optimizer_steps"] == 10
        assert arm["state_changed"] is True
    for name, p in teacher.state_dict().items():
        assert torch.equal(p, initial[name])
    assert result["teacher_unchanged"] is True


def test_inconsistent_pair_inputs_fail_before_second_arm_update():
    teacher = _model().eval().requires_grad_(False)
    count = 0
    students = []

    def factory():
        value = copy.deepcopy(teacher).train().requires_grad_(True)
        students.append(value)
        return value

    def batch(update):
        nonlocal count
        count += 1
        return torch.full((4, 4), float(count)), torch.tensor([0, 0, 1, 1])

    with pytest.raises(ValueError, match="paired input"):
        smoke.measure_smoke_pair(
            teacher,
            factory,
            batch,
            device=torch.device("cpu"),
            microbatch_size=4,
            synchronize=lambda: None,
            progress=lambda event: None,
        )
    assert len(students) == 2
    for name, p in students[1].state_dict().items():
        assert torch.equal(p, teacher.state_dict()[name])


def test_main_refuses_invalid_authorities_without_dataset_model_or_output(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid bytes must be rejected before expensive/scientific access")

    monkeypatch.setattr(smoke.control, "load_siglip_control_components", forbidden)
    output = tmp_path / "result.json"
    args = [
        "--control-root",
        str(tmp_path),
        "--speed-result",
        str(tmp_path / "absent"),
        "--input-proof",
        str(tmp_path / "absent-proof"),
        "--output",
        str(output),
        "--execute-recovery-smoke",
    ]
    with pytest.raises((ValueError, FileNotFoundError)):
        smoke.main(args)
    assert not output.exists()


def test_recovery_batch_schedule_covers_fixed_six_epochs_without_class_leakage():
    examples = tuple(
        ImageExample(f"{label:02}-{i:02}", None, label) for label in range(49) for i in range(8)
    )
    batches = smoke.recovery_batches(examples)
    assert len(batches) == 198
    assert batches == smoke.recovery_batches(examples)
    for positions in batches:
        assert len(positions) == len(set(positions)) == 120
        labels = [examples[p].label for p in positions]
        assert len(set(labels)) == 30
        assert all(labels.count(label) == 4 for label in set(labels))
    invalid = list(examples)
    invalid[0] = ImageExample("outside", None, 49)
    with pytest.raises(ValueError):
        smoke.recovery_batches(tuple(invalid))


def test_real_siglip_checkpointed_student_replay_matches_direct_one_update():
    from transformers import SiglipVisionConfig, SiglipVisionModel

    from sfora.siglip_depth_recovery import prune_siglip_student, relational_cross_entropy
    from sfora.token_set_proxy_anchor import proxy_anchor_loss

    torch.manual_seed(17)
    config = SiglipVisionConfig(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=27,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    vision = SiglipVisionModel(config)
    vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    teacher = (
        PooledProxyAnchorModel(
            tower=smoke.control.SiglipPooledTower(vision),
            input_dimensions=8,
            embedding_dimensions=5,
            class_count=3,
        )
        .float()
        .eval()
        .requires_grad_(False)
    )
    student = prune_siglip_student(teacher).train().requires_grad_(True)
    reference = copy.deepcopy(student)
    optimizer = smoke.new_recovery_optimizer(student)
    expected_optimizer = smoke.new_recovery_optimizer(reference)
    for group in expected_optimizer.param_groups:
        group["lr"] *= 0.1
    inputs, labels = torch.randn(4, 3, 28, 28), torch.tensor([0, 0, 1, 2])
    # This test isolates replay/checkpointing, not objective reduction order.
    # Independent loop objective is covered above. Mathematically equivalent
    # loop reductions perturb the near-zero softmax-invariant key-bias gradients;
    # AdamW then amplifies those tiny differences around its1e-8 epsilon.
    z = reference.encode(inputs)
    with torch.no_grad():
        target = teacher.encode(inputs)
    expected = proxy_anchor_loss(
        z @ F.normalize(reference.proxies, dim=1).T,
        labels,
        alpha=32.0,
        delta=0.1,
    ) + relational_cross_entropy(z, target)
    expected.backward()
    torch.nn.utils.clip_grad_norm_(
        reference.parameters(), 10, error_if_nonfinite=True, foreach=False
    )
    expected_optimizer.step()
    result = smoke.recovery_update(
        student, optimizer, inputs, labels, update=1, teacher=teacher, microbatch_size=4
    )
    assert result["loss"] == pytest.approx(float(expected.detach()), abs=2e-5)
    for name, p in student.named_parameters():
        other = dict(reference.named_parameters())[name]
        assert torch.equal(p.grad, other.grad), name
        assert torch.equal(p, other), name
