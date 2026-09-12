import hashlib
import importlib.util
import json
import struct
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_sop_teacher_anchored_distillation.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_sop_teacher_anchored_distillation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUBJECT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUBJECT
SPEC.loader.exec_module(SUBJECT)


def _evaluation_arguments(tmp_path: Path) -> list[str]:
    paths: dict[str, Path] = {}
    for name in (
        "panel-receipt",
        "source-checkpoint",
        "source-snapshot",
        "teacher-snapshot",
        "ceiling-receipt",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path
    checkout = tmp_path / "unicom"
    checkout.mkdir(exist_ok=True)
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    output = tmp_path / "evaluation.json"
    values = {
        "panel-receipt-sha256": "1" * 64,
        "source-checkpoint-sha256": "2" * 64,
        "source-snapshot-sha256": "3" * 64,
        "teacher-snapshot-sha256": "4" * 64,
        "image-tree-sha256": "5" * 64,
        "ceiling-receipt-sha256": "6" * 64,
        "teacher-pca-sha256": "7" * 64,
        "source-revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        "seed": "17",
    }
    arguments: list[str] = []
    for name, path in paths.items():
        arguments.extend((f"--{name}", str(path.resolve())))
    arguments.extend(("--unicom-checkout", str(checkout.resolve())))
    arguments.extend(("--image-root", str(images.resolve())))
    arguments.extend(("--output", str(output.resolve())))
    for name, value in values.items():
        arguments.extend((f"--{name}", value))
    arguments.append("--execute-teacher-anchored-evaluation")
    return arguments


def test_evaluator_cli_is_strict_local_only_and_refuses_official_test(tmp_path: Path) -> None:
    parsed = SUBJECT.parse_teacher_anchored_evaluation_args(_evaluation_arguments(tmp_path))

    assert parsed.seed == 17
    assert parsed.output == (tmp_path / "evaluation.json").resolve()
    assert parsed.panel_receipt == (tmp_path / "panel-receipt").resolve()
    for forbidden in ("--official-test", "--bucket", "--s3-uri", "--arm"):
        with pytest.raises(ValueError, match="unsupported argument"):
            SUBJECT.parse_teacher_anchored_evaluation_args(
                [*_evaluation_arguments(tmp_path), forbidden, "forbidden"]
            )


def _completed_panel_arm_results() -> dict[str, dict[str, object]]:
    arms = ("head-only", "base", "anchor", "symmetric", "complete")
    common_inputs = {
        "ceiling_receipt": "1" * 64,
        "image_tree": "2" * 64,
        "schedule": "3" * 64,
        "source_checkpoint": "4" * 64,
        "source_snapshot": "5" * 64,
        "teacher_checkpoint": "6" * 64,
        "teacher_snapshot": "7" * 64,
    }
    common_authority = {
        "anchor_schedule_sha256": "8" * 64,
        "batch_schedule_sha256": "9" * 64,
        "fitting_probe_sha256": "a" * 64,
        "ridge_sha256": "b" * 64,
        "runtime_sha256": "c" * 64,
        "seed": 17,
        "snapshot_replay_sha256": "d" * 64,
        "source_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        "split_sha256": "f" * 64,
        "teacher_pca_sha256": "0" * 64,
    }
    return {
        arm: {
            "arm": arm,
            "authority": {
                **common_authority,
                "arm": arm,
                "head_replay_sha256": chr(ord("a") + index) * 64,
                "inputs_sha256": {
                    **common_inputs,
                    "launch_receipt": str(index + 1) * 64,
                },
                "model_mode_sha256": chr(ord("f") - index) * 64,
                "objective_sha256": str(5 + index) * 64,
                "trainable_inventory_sha256": str(9 - index) * 64,
            },
            "attempted_updates": 3720,
            "final_encoder_sha256": "a" * 64 if arm == "head-only" else str(index) * 64,
            "initial_encoder_sha256": "a" * 64,
            "initial_frozen_sha256": "b" * 64,
            "initial_head_sha256": "c" * 64,
            "schedule_sha256": "9" * 64,
            "successful_updates": 3720,
        }
        for index, arm in enumerate(arms)
    }


def test_completed_panel_rejects_cross_arm_schedule_split_input_and_state_drift() -> None:
    baseline = _completed_panel_arm_results()
    SUBJECT.validate_teacher_anchored_cross_arm_authority(baseline)

    mutations = (
        ("base", "schedule_sha256", "0" * 64),
        ("anchor", "successful_updates", 3719),
        ("symmetric", "initial_encoder_sha256", "1" * 64),
    )
    for arm, key, value in mutations:
        changed = deepcopy(baseline)
        changed[arm][key] = value
        with pytest.raises(ValueError, match="cross-arm authority"):
            SUBJECT.validate_teacher_anchored_cross_arm_authority(changed)

    for key in ("batch_schedule_sha256", "split_sha256", "ridge_sha256", "runtime_sha256"):
        changed = deepcopy(baseline)
        changed["complete"]["authority"][key] = "1" * 64
        with pytest.raises(ValueError, match="cross-arm authority"):
            SUBJECT.validate_teacher_anchored_cross_arm_authority(changed)

    changed = deepcopy(baseline)
    changed["base"]["authority"]["inputs_sha256"]["source_snapshot"] = "1" * 64
    with pytest.raises(ValueError, match="cross-arm authority"):
        SUBJECT.validate_teacher_anchored_cross_arm_authority(changed)

    changed = deepcopy(baseline)
    changed["head-only"]["final_encoder_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="cross-arm authority"):
        SUBJECT.validate_teacher_anchored_cross_arm_authority(changed)


def _write_canonical_json(path: Path, value: dict[str, object]) -> bytes:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(payload)
    return payload


def _completed_panel_fixture(
    tmp_path: Path,
) -> tuple[object, dict[str, dict[str, object]]]:
    results = _completed_panel_arm_results()
    panel_inputs = {
        key: value
        for key, value in results["head-only"]["authority"]["inputs_sha256"].items()
        if key != "launch_receipt"
    }
    panel_arms: dict[str, dict[str, str]] = {}
    for arm, result in results.items():
        result_path = (tmp_path / f"{arm}.result.json").resolve()
        checkpoint_path = result_path.with_suffix(".pt")
        checkpoint_bytes = f"{arm}-checkpoint".encode()
        checkpoint_path.write_bytes(checkpoint_bytes)
        checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        launch = {
            "arm": arm,
            "claim_eligible": False,
            "inputs_sha256": panel_inputs,
            "output": str(result_path),
            "schema": "sfora-teacher-anchored-launch-v1",
            "seed": 17,
            "source_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        }
        launch_payload = _write_canonical_json(tmp_path / f"{arm}.launch.json", launch)
        launch_sha256 = hashlib.sha256(launch_payload).hexdigest()
        result["authority"]["inputs_sha256"]["launch_receipt"] = launch_sha256
        result.update(
            {
                "candidate_epoch": 10,
                "checkpoint_path": checkpoint_path.name,
                "checkpoint_sha256": checkpoint_sha256,
                "claim_eligible": False,
                "completed_epochs": list(range(1, 11)),
                "diagnostics": [{} for _ in range(11)],
                "final_frozen_sha256": "d" * 64,
                "final_head_sha256": "e" * 64,
                "optimizer_reset_epochs": [1, 2],
                "schema": "sfora-teacher-anchored-arm-v1",
                "stopped_reason": None,
            }
        )
        result_payload = _write_canonical_json(result_path, result)
        panel_arms[arm] = {
            "launch_receipt_sha256": launch_sha256,
            "result_sha256": hashlib.sha256(result_payload).hexdigest(),
        }
    panel = {
        "arms": panel_arms,
        "claim_eligible": False,
        "inputs_sha256": panel_inputs,
        "schema": "sfora-teacher-anchored-panel-v1",
        "seed": 17,
        "source_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
    }
    panel_path = (tmp_path / "panel.result.json").resolve()
    panel_payload = _write_canonical_json(panel_path, panel)
    source_checkpoint = (tmp_path / "source.pt").resolve()
    source_snapshot = (tmp_path / "source.npz").resolve()
    teacher_snapshot = (tmp_path / "teacher.npz").resolve()
    ceiling_receipt = (tmp_path / "ceiling.json").resolve()
    for path in (source_checkpoint, source_snapshot, teacher_snapshot, ceiling_receipt):
        path.write_bytes(path.name.encode())
    checkout = (tmp_path / "unicom").resolve()
    image_root = (tmp_path / "images").resolve()
    checkout.mkdir()
    image_root.mkdir()
    arguments = SUBJECT.TeacherAnchoredEvaluationArguments(
        panel_receipt=panel_path,
        panel_receipt_sha256=hashlib.sha256(panel_payload).hexdigest(),
        source_checkpoint=source_checkpoint,
        source_checkpoint_sha256=panel_inputs["source_checkpoint"],
        source_snapshot=source_snapshot,
        source_snapshot_sha256=panel_inputs["source_snapshot"],
        teacher_snapshot=teacher_snapshot,
        teacher_snapshot_sha256=panel_inputs["teacher_snapshot"],
        unicom_checkout=checkout,
        image_root=image_root,
        image_tree_sha256=panel_inputs["image_tree"],
        ceiling_receipt=ceiling_receipt,
        ceiling_receipt_sha256=panel_inputs["ceiling_receipt"],
        teacher_pca_sha256="0" * 64,
        source_revision="d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        seed=17,
        output=(tmp_path / "evaluation.json").resolve(),
        execute_teacher_anchored_evaluation=True,
    )
    return arguments, results


def test_completed_panel_loader_authenticates_panel_launch_result_and_checkpoint_chain(
    tmp_path: Path,
) -> None:
    arguments, expected = _completed_panel_fixture(tmp_path)

    loaded = SUBJECT.load_teacher_anchored_completed_panel(arguments)

    assert loaded == expected
    (tmp_path / "base.result.pt").write_bytes(b"foreign checkpoint")
    with pytest.raises(ValueError, match="panel authority"):
        SUBJECT.load_teacher_anchored_completed_panel(arguments)


class DropPath(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class ServingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4))
        self.drop_path = DropPath()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.projection(value))


class BatchSensitiveEncoder(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + value.mean(dim=0, keepdim=True)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_serving_codes(
    encoder: nn.Module,
    head: nn.Linear,
    batches: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    encoder.eval()
    head.eval()
    with torch.inference_mode():
        return torch.cat(
            [
                F.normalize(
                    head(F.normalize(encoder(batch).float(), dim=1)),
                    dim=1,
                ).cpu()
                for batch in batches
            ]
        ).contiguous()


def test_evaluator_uses_library_packer_and_exact_probe_scorer() -> None:
    assert SUBJECT.pack_int8_unit_embeddings is pack_int8_unit_embeddings
    scorer, path = SUBJECT.load_teacher_anchored_evaluation_scorer()

    assert callable(scorer)
    assert path.name == "probe_sop_relational_linear.py"


def test_float_and_packed_evidence_uses_exact_width_self_exclusion_and_ties() -> None:
    root = 2**-0.5
    codes = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [root, root],
            [root, root],
        ],
        dtype=torch.float32,
    ).contiguous()
    labels = (1, 1, 2, 2, 3, 3)

    evidence = SUBJECT.score_teacher_anchored_evaluation(
        codes,
        labels,
        device=torch.device("cpu"),
    )

    assert evidence.candidate_width == 1
    assert evidence.float_map_at_r == 1.0
    assert evidence.float_r1 == 1.0
    assert evidence.packed_map_at_r == 1.0
    assert evidence.packed_r1 == 1.0
    assert evidence.float_per_query_ap == (1.0,) * 6
    assert evidence.packed_per_query_ap == (1.0,) * 6
    assert evidence.packed_per_query_r1 == (1.0,) * 6


def test_evaluation_rejects_singletons_bad_norms_and_non_cpu_codes() -> None:
    valid = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    cases = (
        (valid, (1, 1, 2, 3)),
        (valid.mul(2.0), (1, 1, 2, 2)),
        (valid.double(), (1, 1, 2, 2)),
        (valid, (1, 1, 2**63, 2**63)),
    )
    for codes, labels in cases:
        try:
            SUBJECT.score_teacher_anchored_evaluation(
                codes,
                labels,
                device=torch.device("cpu"),
            )
        except ValueError as error:
            assert "evaluation authority" in str(error)
        else:
            raise AssertionError("invalid evaluation authority was accepted")


def _candidate(
    arm: str,
    map_at_r: float,
    r1: float,
    *,
    stopped_reason: str | None = None,
) -> object:
    per_query_ap = None if stopped_reason is not None else (map_at_r,) * 4
    return SUBJECT.TeacherAnchoredCandidateEvidence(
        arm=arm,
        representation="float32" if arm == "source" else "symmetric-int8",
        candidate_epoch=(
            None
            if stopped_reason is not None or arm in ("source", "teacher-pca")
            else 0
            if arm == "step-zero"
            else 10
        ),
        stopped_reason=stopped_reason,
        map_at_r=None if stopped_reason is not None else map_at_r,
        r1=None if stopped_reason is not None else r1,
        per_query_ap=per_query_ap,
    )


def test_advancement_requires_every_absolute_relative_and_paired_gate() -> None:
    result = SUBJECT.classify_teacher_anchored_advancement(
        source=_candidate("source", 0.550, 0.800),
        teacher_pca=_candidate("teacher-pca", 0.570, 0.820),
        step_zero=_candidate("step-zero", 0.540, 0.790),
        base=_candidate("base", 0.555, 0.798),
        complete=_candidate("complete", 0.560, 0.799),
        labels=(1, 1, 2, 2),
    )

    assert result.outcome == "stability-warranted"
    assert result.complete_step_zero_map_delta == 0.020000000000000018
    assert result.complete_base_map_delta == 0.0050000000000000044
    assert result.complete_step_zero_bootstrap_lower > 0.0
    assert result.complete_base_bootstrap_lower > 0.0
    assert result.gates == (True, True, True, True, True, True, True)


def test_advancement_preserves_control_and_endpoint_epoch_provenance() -> None:
    result = SUBJECT.classify_teacher_anchored_advancement(
        source=_candidate("source", 0.550, 0.800)._replace(candidate_epoch=None),
        teacher_pca=_candidate("teacher-pca", 0.570, 0.820)._replace(candidate_epoch=None),
        step_zero=_candidate("step-zero", 0.540, 0.790)._replace(candidate_epoch=0),
        base=_candidate("base", 0.555, 0.798),
        complete=_candidate("complete", 0.560, 0.799),
        labels=(1, 1, 2, 2),
    )

    assert result.outcome == "stability-warranted"


def test_advancement_outcomes_are_exhaustive_and_precedence_safe() -> None:
    source = _candidate("source", 0.550, 0.800)
    teacher = _candidate("teacher-pca", 0.570, 0.820)
    step = _candidate("step-zero", 0.540, 0.790)
    passing_base = _candidate("base", 0.555, 0.798)
    passing_complete = _candidate("complete", 0.560, 0.799)

    stopped = SUBJECT.classify_teacher_anchored_advancement(
        source=source,
        teacher_pca=teacher,
        step_zero=step,
        base=passing_base,
        complete=_candidate(
            "complete", 0.0, 0.0, stopped_reason="epoch-one-fitting-map-regression"
        ),
        labels=(1, 1, 2, 2),
    )
    inconclusive = SUBJECT.classify_teacher_anchored_advancement(
        source=source,
        teacher_pca=teacher,
        step_zero=step,
        base=_candidate("base", 0.0, 0.0, stopped_reason="epoch-one-effective-rank-collapse"),
        complete=passing_complete,
        labels=(1, 1, 2, 2),
    )
    generic = SUBJECT.classify_teacher_anchored_advancement(
        source=source,
        teacher_pca=teacher,
        step_zero=step,
        base=_candidate("base", 0.558, 0.798),
        complete=passing_complete,
        labels=(1, 1, 2, 2),
    )
    rejected = SUBJECT.classify_teacher_anchored_advancement(
        source=source,
        teacher_pca=teacher,
        step_zero=_candidate("step-zero", 0.547, 0.790),
        base=passing_base,
        complete=passing_complete,
        labels=(1, 1, 2, 2),
    )

    assert stopped.outcome == "stopped"
    assert inconclusive.outcome == "inconclusive"
    assert generic.outcome == "generic-anchored-adaptation"
    assert rejected.outcome == "quality-rejected"

    missing_base = SUBJECT.classify_teacher_anchored_advancement(
        source=source,
        teacher_pca=teacher,
        step_zero=step,
        base=None,
        complete=passing_complete,
        labels=(1, 1, 2, 2),
    )
    missing_complete = SUBJECT.classify_teacher_anchored_advancement(
        source=source,
        teacher_pca=teacher,
        step_zero=step,
        base=passing_base,
        complete=None,
        labels=(1, 1, 2, 2),
    )

    assert missing_base.outcome == "inconclusive"
    assert missing_complete.outcome == "implementation-error"


def test_advancement_rejects_wrong_roles_and_incoherent_per_query_evidence() -> None:
    source = _candidate("source", 0.550, 0.800)
    teacher = _candidate("teacher-pca", 0.570, 0.820)
    step = _candidate("step-zero", 0.540, 0.790)
    base = _candidate("base", 0.555, 0.798)
    complete = _candidate("complete", 0.560, 0.799)

    for mutation in (
        complete._replace(arm="base"),
        complete._replace(per_query_ap=(0.56, 0.56, 0.56)),
        complete._replace(per_query_ap=(0.56, 0.56, 0.56, float("nan"))),
        complete._replace(per_query_ap=(0.50,) * 4),
        complete._replace(candidate_epoch=None, stopped_reason="unknown-stop"),
        complete._replace(representation="float32"),
    ):
        try:
            SUBJECT.classify_teacher_anchored_advancement(
                source=source,
                teacher_pca=teacher,
                step_zero=step,
                base=base,
                complete=mutation,
                labels=(1, 1, 2, 2),
            )
        except ValueError as error:
            assert "advancement authority" in str(error)
        else:
            raise AssertionError("invalid advancement evidence was accepted")

    with pytest.raises(ValueError, match="advancement authority"):
        SUBJECT.classify_teacher_anchored_advancement(
            source=source,
            teacher_pca=teacher,
            step_zero=step,
            base=base,
            complete=complete,
            labels=(1, 1, 2**63, 2**63),
        )


def test_advancement_validates_present_evidence_before_missing_arm_outcome() -> None:
    with pytest.raises(ValueError, match="advancement authority"):
        SUBJECT.classify_teacher_anchored_advancement(
            source=_candidate("source", 0.550, 0.800),
            teacher_pca=_candidate("teacher-pca", 0.570, 0.820),
            step_zero=_candidate("step-zero", 0.540, 0.790),
            base=None,
            complete=_candidate("complete", 0.560, 0.799)._replace(representation="float32"),
            labels=(1, 1, 2, 2),
        )


@pytest.mark.parametrize(
    ("mutation", "outcome"),
    [
        ({"step_zero": _candidate("step-zero", 0.545001, 0.790)}, "quality-rejected"),
        (
            {
                "step_zero": _candidate("step-zero", 0.540, 0.790)._replace(
                    per_query_ap=(0.58, 0.58, 0.50, 0.50)
                ),
                "complete": _candidate("complete", 0.560, 0.799)._replace(
                    per_query_ap=(0.52, 0.52, 0.60, 0.60)
                ),
            },
            "quality-rejected",
        ),
        ({"source": _candidate("source", 0.560001, 0.800)}, "quality-rejected"),
        ({"teacher_pca": _candidate("teacher-pca", 0.575001, 0.820)}, "quality-rejected"),
        ({"source": _candidate("source", 0.550, 0.801001)}, "quality-rejected"),
        ({"base": _candidate("base", 0.557001, 0.798)}, "generic-anchored-adaptation"),
        (
            {
                "base": _candidate("base", 0.5549999999999999, 0.798)._replace(
                    per_query_ap=(0.61, 0.61, 0.50, 0.50)
                ),
            },
            "generic-anchored-adaptation",
        ),
    ],
)
def test_each_advancement_gate_is_independently_binding(
    mutation: dict[str, object], outcome: str
) -> None:
    candidates = {
        "source": _candidate("source", 0.550, 0.800),
        "teacher_pca": _candidate("teacher-pca", 0.570, 0.820),
        "step_zero": _candidate("step-zero", 0.540, 0.790),
        "base": _candidate("base", 0.555, 0.798),
        "complete": _candidate("complete", 0.560, 0.799),
    }
    candidates.update(mutation)

    result = SUBJECT.classify_teacher_anchored_advancement(
        **candidates,
        labels=(1, 1, 2, 2),
    )

    assert result.outcome == outcome
    assert not all(result.gates)


def test_serving_reconstruction_loads_only_merged_state_and_emits_unit_128d_codes(
    tmp_path: Path,
) -> None:
    source_encoder = ServingEncoder()
    source_head = nn.Linear(4, 128)
    checkpoint = tmp_path / "complete.pt"
    torch.save(
        {
            **{
                f"encoder.{name}": value.detach().clone()
                for name, value in source_encoder.state_dict().items()
            },
            **{
                f"head.{name}": value.detach().clone()
                for name, value in source_head.state_dict().items()
            },
        },
        checkpoint,
    )
    encoder = ServingEncoder()
    head = nn.Linear(4, 128)
    encoder.train()
    head.train()
    batches = (
        torch.eye(4, dtype=torch.float32).contiguous(),
        torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32).contiguous(),
    )
    expected_codes = _expected_serving_codes(source_encoder, source_head, batches)

    reconstructed = SUBJECT.reconstruct_teacher_anchored_serving(
        encoder,
        head,
        checkpoint,
        batches,
        expected_checkpoint_sha256=_checkpoint_sha256(checkpoint),
        expected_codes=expected_codes,
        expected_input_shape=(4,),
        retained_batch_rows=(4, 1),
        device=torch.device("cpu"),
    )

    assert reconstructed.rows == 5
    assert reconstructed.dimensions == 128
    assert reconstructed.batch_rows == (4, 1)
    assert reconstructed.codes.shape == (5, 128)
    assert torch.equal(reconstructed.codes, expected_codes)
    assert reconstructed.maximum_batch_shape_error <= 0.002
    assert reconstructed.minimum_batch_shape_cosine >= 1.0 - 1e-5
    torch.testing.assert_close(
        torch.linalg.vector_norm(reconstructed.codes.double(), dim=1),
        torch.ones(5, dtype=torch.float64),
        rtol=0.0,
        atol=2e-7,
    )
    assert not encoder.training
    assert not head.training
    assert not any(
        module.training
        for module in encoder.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        or type(module).__name__ == "DropPath"
    )
    assert all(
        torch.equal(value, source_encoder.state_dict()[name])
        for name, value in encoder.state_dict().items()
    )


def test_serving_reconstruction_discards_registered_padding_rows(tmp_path: Path) -> None:
    source_encoder = ServingEncoder()
    source_head = nn.Linear(4, 128)
    checkpoint = tmp_path / "padded.pt"
    torch.save(
        {
            **{f"encoder.{name}": value for name, value in source_encoder.state_dict().items()},
            **{f"head.{name}": value for name, value in source_head.state_dict().items()},
        },
        checkpoint,
    )
    batches = (
        torch.eye(4, dtype=torch.float32).contiguous(),
        torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ).contiguous(),
    )
    all_codes = _expected_serving_codes(source_encoder, source_head, batches)
    expected = torch.cat((all_codes[:4], all_codes[4:6])).contiguous()

    reconstructed = SUBJECT.reconstruct_teacher_anchored_serving(
        ServingEncoder(),
        nn.Linear(4, 128),
        checkpoint,
        batches,
        expected_checkpoint_sha256=_checkpoint_sha256(checkpoint),
        expected_codes=expected,
        expected_input_shape=(4,),
        retained_batch_rows=(4, 2),
        device=torch.device("cpu"),
    )

    assert reconstructed.rows == 6
    assert reconstructed.batch_rows == (4, 4)
    assert reconstructed.retained_batch_rows == (4, 2)
    assert torch.equal(reconstructed.codes, expected)


def test_serving_reconstruction_deserializes_only_authenticated_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = ServingEncoder()
    head = nn.Linear(4, 128)
    checkpoint = tmp_path / "complete.pt"
    torch.save(
        {
            **{f"encoder.{name}": value for name, value in encoder.state_dict().items()},
            **{f"head.{name}": value for name, value in head.state_dict().items()},
        },
        checkpoint,
    )
    batches = (torch.eye(4, dtype=torch.float32).contiguous(),)
    expected = _expected_serving_codes(encoder, head, batches)
    original_load = torch.load
    loaded_from: list[object] = []

    def recording_load(source: object, **kwargs: object) -> object:
        loaded_from.append(source)
        return original_load(source, **kwargs)

    monkeypatch.setattr(SUBJECT.torch, "load", recording_load)
    SUBJECT.reconstruct_teacher_anchored_serving(
        ServingEncoder(),
        nn.Linear(4, 128),
        checkpoint,
        batches,
        expected_checkpoint_sha256=_checkpoint_sha256(checkpoint),
        expected_codes=expected,
        expected_input_shape=(4,),
        retained_batch_rows=(4,),
        device=torch.device("cpu"),
    )

    assert len(loaded_from) == 1
    assert not isinstance(loaded_from[0], (str, Path))


def test_serving_reconstruction_rejects_extra_roles_and_batch_shape_drift(
    tmp_path: Path,
) -> None:
    encoder = ServingEncoder()
    head = nn.Linear(4, 128)
    baseline = {
        **{f"encoder.{name}": value for name, value in encoder.state_dict().items()},
        **{f"head.{name}": value for name, value in head.state_dict().items()},
    }
    checkpoints = []
    for index, state in enumerate(
        (
            {**baseline, "teacher.weight": torch.zeros(1)},
            {key: value for key, value in baseline.items() if key != "head.bias"},
        )
    ):
        path = tmp_path / f"invalid-{index}.pt"
        torch.save(state, path)
        checkpoints.append(path)
    valid_path = tmp_path / "valid.pt"
    torch.save(baseline, valid_path)

    cases = (
        (checkpoints[0], (torch.ones((2, 4), dtype=torch.float32),)),
        (checkpoints[1], (torch.ones((2, 4), dtype=torch.float32),)),
        (valid_path, (torch.ones((0, 4), dtype=torch.float32),)),
        (valid_path, (torch.ones((257, 4), dtype=torch.float32),)),
        (valid_path, (torch.ones((2, 5), dtype=torch.float32),)),
    )
    for checkpoint, batches in cases:
        try:
            SUBJECT.reconstruct_teacher_anchored_serving(
                ServingEncoder(),
                nn.Linear(4, 128),
                checkpoint,
                batches,
                expected_checkpoint_sha256=_checkpoint_sha256(checkpoint),
                expected_codes=torch.zeros(
                    (sum(len(batch) for batch in batches), 128), dtype=torch.float32
                ),
                expected_input_shape=(4,),
                retained_batch_rows=tuple(len(batch) for batch in batches),
                device=torch.device("cpu"),
            )
        except ValueError as error:
            assert "serving authority" in str(error)
        else:
            raise AssertionError("invalid serving reconstruction was accepted")


def test_serving_reconstruction_rejects_digest_replay_and_batch_sensitivity(
    tmp_path: Path,
) -> None:
    batches = (torch.eye(4, dtype=torch.float32).contiguous(),)
    encoder = ServingEncoder()
    head = nn.Linear(4, 128)
    checkpoint = tmp_path / "valid.pt"
    torch.save(
        {
            **{f"encoder.{name}": value for name, value in encoder.state_dict().items()},
            **{f"head.{name}": value for name, value in head.state_dict().items()},
        },
        checkpoint,
    )
    expected = _expected_serving_codes(encoder, head, batches)
    replay_drift = expected.clone()
    replay_drift[0] = torch.roll(replay_drift[0], 1)

    for expected_digest, expected_codes in (
        ("0" * 64, expected),
        (_checkpoint_sha256(checkpoint), replay_drift),
    ):
        with pytest.raises(ValueError, match="serving authority"):
            SUBJECT.reconstruct_teacher_anchored_serving(
                ServingEncoder(),
                nn.Linear(4, 128),
                checkpoint,
                batches,
                expected_checkpoint_sha256=expected_digest,
                expected_codes=expected_codes,
                expected_input_shape=(4,),
                retained_batch_rows=(4,),
                device=torch.device("cpu"),
            )

    sensitive_encoder = BatchSensitiveEncoder()
    sensitive_head = nn.Linear(4, 128)
    sensitive_checkpoint = tmp_path / "batch-sensitive.pt"
    torch.save(
        {
            **{f"encoder.{name}": value for name, value in sensitive_encoder.state_dict().items()},
            **{f"head.{name}": value for name, value in sensitive_head.state_dict().items()},
        },
        sensitive_checkpoint,
    )
    with pytest.raises(ValueError, match="serving authority"):
        SUBJECT.reconstruct_teacher_anchored_serving(
            BatchSensitiveEncoder(),
            nn.Linear(4, 128),
            sensitive_checkpoint,
            batches,
            expected_checkpoint_sha256=_checkpoint_sha256(sensitive_checkpoint),
            expected_codes=_expected_serving_codes(sensitive_encoder, sensitive_head, batches),
            expected_input_shape=(4,),
            retained_batch_rows=(4,),
            device=torch.device("cpu"),
        )


def test_serving_reconstruction_rejects_non_tensor_batch_as_authority_error(
    tmp_path: Path,
) -> None:
    encoder = ServingEncoder()
    head = nn.Linear(4, 128)
    checkpoint = tmp_path / "valid.pt"
    torch.save(
        {
            **{f"encoder.{name}": value for name, value in encoder.state_dict().items()},
            **{f"head.{name}": value for name, value in head.state_dict().items()},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="serving authority"):
        SUBJECT.reconstruct_teacher_anchored_serving(
            ServingEncoder(),
            nn.Linear(4, 128),
            checkpoint,
            (object(),),
            expected_checkpoint_sha256=_checkpoint_sha256(checkpoint),
            expected_codes=torch.zeros((1, 128), dtype=torch.float32),
            expected_input_shape=(4,),
            retained_batch_rows=(1,),
            device=torch.device("cpu"),
        )


def _serving_evidence() -> object:
    codes = torch.zeros((4, 128), dtype=torch.float32)
    codes[:2, 0] = 1.0
    codes[2:, 1] = 1.0
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", 2))
    digest.update(struct.pack("<2Q", 4, 128))
    digest.update(codes.numpy().astype("<f4", copy=False).tobytes(order="C"))
    return SUBJECT.TeacherAnchoredServingReconstruction(
        codes=codes,
        rows=4,
        dimensions=128,
        batch_rows=(2, 2),
        retained_batch_rows=(2, 2),
        scoring_device="cpu",
        checkpoint_sha256="a" * 64,
        codes_sha256=digest.hexdigest(),
        expected_codes_sha256=digest.hexdigest(),
        maximum_batch_shape_error=0.0,
        minimum_batch_shape_cosine=1.0,
    )


def test_canonical_result_recomputes_gates_and_excludes_serving_only_inputs() -> None:
    raw = SUBJECT.canonical_teacher_anchored_evaluation_bytes(
        seed=17,
        source=_candidate("source", 0.980, 1.000),
        teacher_pca=_candidate("teacher-pca", 1.000, 1.000),
        step_zero=_candidate("step-zero", 0.980, 1.000),
        base=_candidate("base", 0.995, 1.000),
        complete=_candidate("complete", 1.000, 1.000),
        labels=(1, 1, 2, 2),
        serving=_serving_evidence(),
    )
    value = json.loads(raw)

    assert raw == json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert value["schema"] == "sfora-teacher-anchored-evaluation-v1"
    assert value["claim_eligible"] is False
    assert value["partition"] == "class-disjoint-training-validation"
    assert value["advancement"]["outcome"] == "stability-warranted"
    assert value["arms"]["source"]["candidate_epoch"] is None
    assert value["arms"]["teacher-pca"]["candidate_epoch"] is None
    assert value["arms"]["step-zero"]["candidate_epoch"] == 0
    assert value["arms"]["base"]["candidate_epoch"] == 10
    assert value["arms"]["complete"]["map_at_r"] == 1.0
    assert value["serving"] == {
        "batch_rows": [2, 2],
        "checkpoint_sha256": "a" * 64,
        "codes_sha256": _serving_evidence().codes_sha256,
        "dimensions": 128,
        "expected_codes_sha256": _serving_evidence().expected_codes_sha256,
        "maximum_batch_shape_error": 0.0,
        "minimum_batch_shape_cosine": 1.0,
        "retained_batch_rows": [2, 2],
        "rows": 4,
        "scoring_device": "cpu",
    }
    assert "teacher" not in value["serving"]
    assert "anchors" not in value["serving"]
    assert "labels" not in value["serving"]
    assert "codes" not in value["serving"]


def test_canonical_result_rejects_serving_digest_and_seed_drift() -> None:
    common = {
        "source": _candidate("source", 0.980, 1.000),
        "teacher_pca": _candidate("teacher-pca", 1.000, 1.000),
        "step_zero": _candidate("step-zero", 0.980, 1.000),
        "base": _candidate("base", 0.995, 1.000),
        "complete": _candidate("complete", 1.000, 1.000),
        "labels": (1, 1, 2, 2),
    }
    for seed, serving in (
        (18, _serving_evidence()),
        (17, _serving_evidence()._replace(codes_sha256="0" * 64)),
        (17, _serving_evidence()._replace(rows=3)),
    ):
        try:
            SUBJECT.canonical_teacher_anchored_evaluation_bytes(
                seed=seed,
                serving=serving,
                **common,
            )
        except ValueError as error:
            assert "result authority" in str(error)
        else:
            raise AssertionError("invalid result authority was accepted")


def test_canonical_result_binds_complete_metrics_and_validation_rows_to_serving() -> None:
    common = {
        "seed": 17,
        "source": _candidate("source", 0.980, 1.000),
        "teacher_pca": _candidate("teacher-pca", 1.000, 1.000),
        "step_zero": _candidate("step-zero", 0.980, 1.000),
        "base": _candidate("base", 0.995, 1.000),
        "labels": (1, 1, 2, 2),
    }
    two_codes = torch.zeros((2, 128), dtype=torch.float32)
    two_codes[:, 0] = 1.0
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", 2))
    digest.update(struct.pack("<2Q", 2, 128))
    digest.update(two_codes.numpy().astype("<f4", copy=False).tobytes(order="C"))
    two_row_serving = SUBJECT.TeacherAnchoredServingReconstruction(
        codes=two_codes,
        rows=2,
        dimensions=128,
        batch_rows=(2,),
        retained_batch_rows=(2,),
        scoring_device="cpu",
        checkpoint_sha256="a" * 64,
        codes_sha256=digest.hexdigest(),
        expected_codes_sha256=digest.hexdigest(),
        maximum_batch_shape_error=0.0,
        minimum_batch_shape_cosine=1.0,
    )

    for complete, serving in (
        (_candidate("complete", 0.990, 1.000), _serving_evidence()),
        (_candidate("complete", 1.000, 1.000), two_row_serving),
    ):
        with pytest.raises(ValueError, match="result authority"):
            SUBJECT.canonical_teacher_anchored_evaluation_bytes(
                complete=complete,
                serving=serving,
                **common,
            )


def test_canonical_stopped_result_requires_no_serving_candidate() -> None:
    raw = SUBJECT.canonical_teacher_anchored_evaluation_bytes(
        seed=17,
        source=_candidate("source", 0.550, 0.800),
        teacher_pca=_candidate("teacher-pca", 0.570, 0.820),
        step_zero=_candidate("step-zero", 0.540, 0.790),
        base=_candidate("base", 0.555, 0.798),
        complete=_candidate(
            "complete",
            0.0,
            0.0,
            stopped_reason="epoch-one-leading-eigenvalue-collapse",
        ),
        labels=(1, 1, 2, 2),
        serving=None,
    )
    value = json.loads(raw)

    assert value["advancement"]["outcome"] == "stopped"
    assert value["serving"] is None

    try:
        SUBJECT.canonical_teacher_anchored_evaluation_bytes(
            seed=17,
            source=_candidate("source", 0.550, 0.800),
            teacher_pca=_candidate("teacher-pca", 0.570, 0.820),
            step_zero=_candidate("step-zero", 0.540, 0.790),
            base=_candidate("base", 0.555, 0.798),
            complete=_candidate(
                "complete",
                0.0,
                0.0,
                stopped_reason="epoch-one-leading-eigenvalue-collapse",
            ),
            labels=(1, 1, 2, 2),
            serving=_serving_evidence(),
        )
    except ValueError as error:
        assert "result authority" in str(error)
    else:
        raise AssertionError("stopped result accepted a serving candidate")


def test_canonical_result_records_missing_comparators_without_serving() -> None:
    common = {
        "seed": 17,
        "source": _candidate("source", 0.550, 0.800),
        "teacher_pca": _candidate("teacher-pca", 0.570, 0.820),
        "step_zero": _candidate("step-zero", 0.540, 0.790),
        "labels": (1, 1, 2, 2),
        "serving": None,
    }

    missing_base = json.loads(
        SUBJECT.canonical_teacher_anchored_evaluation_bytes(
            base=None,
            complete=_candidate("complete", 0.560, 0.799),
            **common,
        )
    )
    missing_complete = json.loads(
        SUBJECT.canonical_teacher_anchored_evaluation_bytes(
            base=_candidate("base", 0.555, 0.798),
            complete=None,
            **common,
        )
    )

    assert missing_base["advancement"]["outcome"] == "inconclusive"
    assert missing_base["arms"]["base"] is None
    assert missing_base["serving"] is None
    assert missing_complete["advancement"]["outcome"] == "implementation-error"
    assert missing_complete["arms"]["complete"] is None
    assert missing_complete["serving"] is None
