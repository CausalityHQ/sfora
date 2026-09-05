"""Final-only paired recovery engine and exclusive checkpoint authority."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from sfora.siglip_proxy_control import PooledProxyAnchorModel

_SPEC = importlib.util.spec_from_file_location(
    "run_siglip_recovery_pair",
    Path(__file__).parents[1] / "scripts/run_siglip_recovery_pair.py",
)
assert _SPEC and _SPEC.loader
pair = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pair
_SPEC.loader.exec_module(pair)


def _model():
    torch.manual_seed(17)
    return PooledProxyAnchorModel(
        tower=nn.Linear(4, 5), input_dimensions=5, embedding_dimensions=4, class_count=3
    ).float()


def _batch(update):
    return torch.randn(4, 4, generator=torch.Generator().manual_seed(update)), torch.tensor(
        [0, 0, 1, 2]
    )


def test_exact_pair_executes198_fresh_steps_and_no_pa_teacher_forward():
    teacher = _model().eval().requires_grad_(False)
    initial = copy.deepcopy(teacher.state_dict())
    teacher_calls = []
    hook = teacher.register_forward_hook(lambda *args: teacher_calls.append(True))
    # encode invokes the tower directly, so observe the actual tower boundary.
    tower_hook = teacher.tower.register_forward_hook(lambda *args: teacher_calls.append(True))
    student_a = copy.deepcopy(teacher).train().requires_grad_(True)
    # Clear copied hook: it is a test observation, not model behavior.
    student_a._forward_hooks.clear()
    student_a.tower._forward_hooks.clear()
    events = []
    a = pair.train_recovery_arm(
        student_a,
        teacher,
        _batch,
        arm="pa",
        expected_input_hashes=None,
        microbatch_size=4,
        progress=events.append,
        synchronize=lambda: None,
    )
    assert teacher_calls == []
    assert len(events) == 198 and [e["update"] for e in events] == list(range(1, 199))
    assert a["completed_updates"] == 198 and a["steps"][-1]["lr_multiplier"] == pytest.approx(0.1)
    assert len(a["input_sha256"]) == 198
    student_b = _model().train()
    b = pair.train_recovery_arm(
        student_b,
        teacher,
        _batch,
        arm="relational",
        expected_input_hashes=a["input_sha256"],
        microbatch_size=4,
        progress=lambda event: None,
        synchronize=lambda: None,
    )
    assert len(teacher_calls) == 198
    assert a["initial_state_sha256"] == b["initial_state_sha256"]
    assert a["input_sha256"] == b["input_sha256"]
    for name, tensor in teacher.state_dict().items():
        assert torch.equal(tensor, initial[name])
    hook.remove()
    tower_hook.remove()


@pytest.mark.parametrize(
    "arm,hashes",
    [
        ("wrong", None),
        ("relational", None),
        ("relational", ["a" * 64] * 197),
        ("pa", ["a" * 64] * 198),
    ],
)
def test_invalid_arm_or_pair_inventory_never_updates(arm, hashes):
    student, teacher = _model().train(), _model().eval().requires_grad_(False)
    before = copy.deepcopy(student.state_dict())
    with pytest.raises(ValueError):
        pair.train_recovery_arm(
            student,
            teacher,
            _batch,
            arm=arm,
            expected_input_hashes=hashes,
            microbatch_size=4,
            progress=lambda event: None,
            synchronize=lambda: None,
        )
    for name, tensor in student.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_pair_mismatch_stops_before_optimizer_update():
    student, teacher = _model().train(), _model().eval().requires_grad_(False)
    before = copy.deepcopy(student.state_dict())
    with pytest.raises(ValueError, match="paired input"):
        pair.train_recovery_arm(
            student,
            teacher,
            _batch,
            arm="relational",
            expected_input_hashes=["a" * 64] * 198,
            microbatch_size=4,
            progress=lambda event: None,
            synchronize=lambda: None,
        )
    for name, tensor in student.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_terminal_checkpoint_seals_only_final_student_and_never_overwrites(tmp_path):
    from transformers import SiglipVisionConfig, SiglipVisionModel

    from sfora.siglip_depth_recovery import prune_siglip_student

    config = SiglipVisionConfig(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=27,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
    )
    full = PooledProxyAnchorModel(
        tower=pair.smoke.control.SiglipPooledTower(SiglipVisionModel(config)),
        input_dimensions=8,
        embedding_dimensions=512,
        class_count=49,
    )
    student = prune_siglip_student(full)
    evidence = {
        "arm": "pa",
        "completed_updates": 198,
        "initial_state_sha256": "a" * 64,
        "final_state_sha256": pair.smoke.control._model_state_sha256(student),
        "input_sha256": ["b" * 64] * 198,
        "steps": [{"update": i} for i in range(1, 199)],
    }
    path = tmp_path / "pa-final.pt"
    seal = pair.write_terminal_student(path, student, evidence)
    assert seal["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert seal["bytes"] == path.stat().st_size
    payload = torch.load(path, weights_only=True)
    assert payload["claim_eligible"] is False and payload["completed_updates"] == 198
    assert "optimizer_state" not in payload and "teacher_state" not in payload
    assert payload["retained_one_indexed_blocks"] == [
        1,
        3,
        4,
        6,
        7,
        9,
        10,
        12,
        13,
        15,
        16,
        18,
        19,
        21,
        22,
        24,
        25,
        27,
    ]
    assert all("encoder.layers.18." not in name for name in payload["model_state"])
    for name, value in student.state_dict().items():
        assert torch.equal(value, payload["model_state"][name])
    with pytest.raises(FileExistsError):
        pair.write_terminal_student(path, student, evidence)
    for key, value in (
        ("completed_updates", 197),
        ("steps", evidence["steps"][:-1]),
        ("final_state_sha256", "c" * 64),
        ("input_sha256", ["b" * 64] * 197),
    ):
        with pytest.raises(ValueError):
            pair.write_terminal_student(
                tmp_path / f"bad-{key}.pt", student, {**evidence, key: value}
            )
        assert not (tmp_path / f"bad-{key}.pt").exists()


def test_smoke_authority_recomputes_feasibility_and_rejects_failed_receipts(tmp_path):
    arms = {}
    for arm in ("pa", "relational"):
        arms[arm] = {
            "initial_state_sha256": "a" * 64,
            "final_state_sha256": "b" * 64,
            "input_sha256": ["c" * 64] * 10,
            "optimizer_steps": 10,
            "state_changed": True,
            "steps": [
                {
                    "update": i,
                    "elapsed_ns": 20_000_000_000,
                    "loss": 1.0,
                    "proxy_loss": 1.0,
                    "relational_loss": 0.0,
                    "gradient_norm": 1.0,
                    "maximum_descriptor_disagreement": 0.0,
                    "lr_multiplier": i / 10,
                }
                for i in range(1, 11)
            ],
        }
    value = {
        "schema": "sfora-siglip-depth-recovery-smoke-v1",
        "claim_eligible": False,
        "quality_measured": False,
        "trained_checkpoint_retained": False,
        "seed": 17,
        "logical_batch": 120,
        "microbatch": 120,
        "checkpoint_sha256": pair.probe.CHECKPOINT_SHA256,
        "speed_sha256": pair.smoke.SPEED_SHA256,
        "input_proof_sha256": pair.smoke.INPUT_PROOF_SHA256,
        "source_sha256": {
            "runner": "4182deddf5be7af0fd538bb0fe197914e18f2f015e59fd2eb238dc64196690e7"
        },
        "arms": arms,
        "teacher_unchanged": True,
        "teacher_state_sha256": "d" * 64,
        "resources": {"elapsed_seconds": 500.0, "prior_gpu_seconds": 761.4893234689953},
        "budget": {
            "projected_total_seconds": 13361.489323468995,
            "within_six_hours": True,
            "future_startup_seconds": 100.0,
            "checkpoint_allowance_seconds": 300,
            "evaluation_allowance_seconds": 1800,
        },
    }

    # 500+761.4893 spent +100future overhead +40*198*1.25 +2100.
    def write(item):
        raw = (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode()
        path = tmp_path / "smoke.json"
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    path, digest = write(value)
    assert pair.read_smoke_authority(path, digest)["teacher_unchanged"] is True
    with pytest.raises(ValueError):
        pair.read_smoke_authority(path, "0" * 64)
    mutations = []
    bad = copy.deepcopy(value)
    bad["budget"]["projected_total_seconds"] -= 1
    mutations.append(bad)
    bad = copy.deepcopy(value)
    bad["arms"]["relational"]["input_sha256"][0] = "f" * 64
    mutations.append(bad)
    bad = copy.deepcopy(value)
    bad["arms"]["pa"]["steps"][0]["maximum_descriptor_disagreement"] = 0.01
    mutations.append(bad)
    bad = copy.deepcopy(value)
    bad["teacher_unchanged"] = False
    mutations.append(bad)
    bad = copy.deepcopy(value)
    bad["arms"]["pa"]["steps"][0]["gradient_norm"] = float("nan")
    mutations.append(bad)
    for bad in mutations:
        path, digest = write(bad)
        with pytest.raises(ValueError):
            pair.read_smoke_authority(path, digest)


def test_training_cli_is_fixed_and_invalid_receipt_precedes_output_creation(tmp_path):
    args = [
        "--control-root",
        str(tmp_path),
        "--smoke-result",
        str(tmp_path / "absent.json"),
        "--smoke-sha256",
        "0" * 64,
        "--output-dir",
        str(tmp_path / "output"),
        "--execute-recovery-pair",
    ]
    parsed = pair.parse_args(args)
    assert parsed.output_dir == tmp_path / "output"
    with pytest.raises(SystemExit):
        pair.parse_args(args[:-1])
    for flag in ("--epochs", "--updates", "--seed", "--microbatch", "--evaluation"):
        with pytest.raises(SystemExit):
            pair.parse_args(args + [flag, "1"])
    with pytest.raises((ValueError, FileNotFoundError)):
        pair.main(args)
    assert not (tmp_path / "output").exists()


def test_pair_orchestration_seals_both_after_fresh_initialization(tmp_path):
    teacher = _model().eval().requires_grad_(False)
    initial = pair.control._model_state_sha256(teacher)
    events = []
    saved = {}

    def factory():
        return copy.deepcopy(teacher).train().requires_grad_(True)

    def writer(path, model, evidence):
        # Real dense model cannot meet scientific18-layer checkpoint format;
        # substitute only the file format boundary, retaining actual training.
        torch.save(model.state_dict(), path)
        saved[path.name] = copy.deepcopy(model.state_dict())
        return {
            "basename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "arm": evidence["arm"],
            "completed_updates": 198,
        }

    result = pair.execute_pair(
        teacher,
        factory,
        _batch,
        output_dir=tmp_path,
        initial_sha256=initial,
        teacher_sha256=initial,
        microbatch_size=4,
        progress=events.append,
        synchronize=lambda: None,
        writer=writer,
    )
    assert set(saved) == {"pa-final.pt", "relational-final.pt"}
    assert result["teacher_unchanged"] is True
    assert len(events) == 396
    assert result["arms"]["pa"]["input_sha256"] == result["arms"]["relational"]["input_sha256"]
    assert (
        result["arms"]["pa"]["initial_state_sha256"]
        == result["arms"]["relational"]["initial_state_sha256"]
        == initial
    )
    assert result["checkpoints"]["pa"]["completed_updates"] == 198


def test_training_budget_reserves_evaluation_and_checkpoint_time():
    assert pair.remaining_training_seconds(1355.0) == 18145.0
    for prior in (19500.0, float("nan"), -1.0, True):
        with pytest.raises(ValueError):
            pair.remaining_training_seconds(prior)


def test_late_budget_failure_preserves_terminal_receipt_before_nonzero_exit(tmp_path, monkeypatch):
    output = tmp_path / "pair"
    output.mkdir()
    result = {
        "claim_eligible": False,
        "status": "completed-outside-budget",
        "resources": {"within_campaign_cap": False},
    }
    monkeypatch.setattr(pair, "run_pair", lambda args: result)
    with pytest.raises(RuntimeError, match="budget"):
        pair.main(
            [
                "--control-root",
                str(tmp_path),
                "--smoke-result",
                str(tmp_path / "smoke"),
                "--smoke-sha256",
                "0" * 64,
                "--output-dir",
                str(output),
                "--execute-recovery-pair",
            ]
        )
    assert json.loads((output / "pair-complete.json").read_bytes()) == result
