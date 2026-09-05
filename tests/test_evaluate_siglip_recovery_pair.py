"""Post-sealing evaluation must reject incomplete or cross-bound training evidence."""

import copy
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
import torch

import sfora.siglip_asymmetric_recovery as asymmetric_core

SPEC = importlib.util.spec_from_file_location(
    "evaluate_siglip_recovery_pair",
    Path(__file__).parents[1] / "scripts/evaluate_siglip_recovery_pair.py",
)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def _authority() -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(198)]
    smoke: dict[str, Any] = {
        "source_sha256": {"runner": "e" * 64},
        "teacher_state_sha256": "f" * 64,
        "arms": {
            a: {"initial_state_sha256": "a" * 64, "input_sha256": inputs[:10]}
            for a in ("pa", "relational")
        },
        "resources": {"elapsed_seconds": 589.932864409},
    }
    prior = 1351.4221878779953
    receipt: dict[str, Any] = {
        "schema": "sfora-siglip-depth-recovery-pair-v1",
        "claim_eligible": False,
        "quality_measured": False,
        "seed": 17,
        "updates_per_arm": 198,
        "status": "complete",
        "smoke_sha256": subject.SMOKE_SHA256,
        "teacher_checkpoint_sha256": subject.pair.probe.CHECKPOINT_SHA256,
        "runner_sha256": subject.PAIR_RUNNER_SHA256,
        "dependencies": smoke["source_sha256"],
        "retained_one_indexed_blocks": [
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
        ],
        "teacher_unchanged": True,
        "teacher_state_sha256": "f" * 64,
        "resources": {
            "within_campaign_cap": True,
            "elapsed_seconds": 12000.0,
            "prior_gpu_seconds": prior,
            "remaining_campaign_seconds": 21600 - prior - 12000,
        },
        "arms": {},
        "checkpoints": {},
    }
    for arm in ("pa", "relational"):
        receipt["arms"][arm] = {
            "arm": arm,
            "completed_updates": 198,
            "initial_state_sha256": "a" * 64,
            "final_state_sha256": ("b" if arm == "pa" else "c") * 64,
            "input_sha256": inputs,
            "steps": [
                {
                    "arm": arm,
                    "update": i,
                    "elapsed_ns": 25_000_000_000,
                    "loss": 1.0,
                    "proxy_loss": 1.0,
                    "relational_loss": 0.0,
                    "gradient_norm": 1.0,
                    "maximum_descriptor_disagreement": 0.0,
                    "lr_multiplier": i / 10
                    if i <= 10
                    else 0.1 + 0.45 * (1 + math.cos(math.pi * (i - 10) / 188)),
                }
                for i in range(1, 199)
            ],
        }
        receipt["checkpoints"][arm] = {
            "basename": f"{arm}-final.pt",
            "bytes": 100,
            "sha256": "d" * 64,
            "arm": arm,
            "completed_updates": 198,
        }
    return receipt, smoke


def test_pair_requires_both_final_198_step_arms_and_budget_before_evaluation() -> None:
    value, smoke = _authority()
    subject.validate_pair_receipt(value, smoke)
    mutations: list[Callable[[dict[str, Any]], Any]] = [
        lambda x: x.update(status="completed-outside-budget"),
        lambda x: x.update(quality_measured=True),
        lambda x: x.update(seed=True),
        lambda x: x["resources"].update(within_campaign_cap=False),
        lambda x: x["resources"].update(remaining_campaign_seconds=99999.0),
        lambda x: x["arms"]["pa"].update(completed_updates=197),
        lambda x: x["arms"]["relational"]["steps"].pop(),
        lambda x: x["arms"]["pa"]["steps"][3].update(gradient_norm=float("nan")),
        lambda x: x["arms"]["pa"]["steps"][4].update(lr_multiplier=99.0),
        lambda x: x["arms"]["relational"].update(input_sha256=["9" * 64] * 198),
        lambda x: x["checkpoints"]["pa"].update(basename="../pa-final.pt"),
        lambda x: x.update(teacher_state_sha256="9" * 64),
    ]
    for mutate in mutations:
        bad = copy.deepcopy(value)
        mutate(bad)
        with pytest.raises(ValueError):
            subject.validate_pair_receipt(bad, smoke)


def test_authenticate_both_checkpoint_bytes_before_any_tensor_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _ = _authority()
    for arm in ("pa", "relational"):
        path = tmp_path / f"{arm}-final.pt"
        path.write_bytes(b"not a tensor archive")
        receipt["checkpoints"][arm].update(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    (tmp_path / "relational-final.pt").write_bytes(b"changed")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("no tensor parser before both byte identities pass")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="checkpoint"):
        subject.authenticate_checkpoint_files(tmp_path, receipt)


def test_terminal_payload_bindings_and_actual_state_hash(tmp_path: Path) -> None:
    receipt, _ = _authority()
    state = {"weight": torch.tensor([[1.0, 2.0]])}
    model = torch.nn.Linear(2, 1, bias=False)
    model.load_state_dict(state)
    receipt["arms"]["pa"]["final_state_sha256"] = subject.pair.control._model_state_sha256(model)
    payload = {
        "schema": "sfora-siglip-depth-recovery-student-v1",
        "claim_eligible": False,
        "seed": 17,
        "completed_updates": 198,
        "arm": "pa",
        "teacher_checkpoint_sha256": receipt["teacher_checkpoint_sha256"],
        "retained_one_indexed_blocks": receipt["retained_one_indexed_blocks"],
        "input_dimensions": 1152,
        "embedding_dimensions": 512,
        "initial_state_sha256": "a" * 64,
        "final_state_sha256": receipt["arms"]["pa"]["final_state_sha256"],
        "input_sha256": receipt["arms"]["pa"]["input_sha256"],
        "model_state": state,
    }
    subject.validate_student_payload(payload, receipt, "pa")
    for key, val in (
        ("arm", "relational"),
        ("completed_updates", 197),
        ("embedding_dimensions", 4096),
        ("input_dimensions", 1024),
        ("model_state", {"weight": torch.tensor([[1.0, float("nan")]])}),
        ("final_state_sha256", "9" * 64),
    ):
        with pytest.raises(ValueError):
            subject.validate_student_payload({**payload, key: val}, receipt, "pa")


def test_direct_cli_refuses_unsealed_missing_inputs_without_output(tmp_path: Path) -> None:
    output = tmp_path / "evaluation.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        subject.main(
            [
                "--pair-directory",
                str(tmp_path),
                "--pair-sha256",
                "a" * 64,
                "--smoke-result",
                str(tmp_path / "smoke.json"),
                "--audit-result",
                str(tmp_path / "audit.json"),
                "--pair-monitor",
                str(tmp_path / "monitor.json"),
                "--monitor-sha256",
                "b" * 64,
                "--control-root",
                str(tmp_path / "control"),
                "--output",
                str(output),
                "--execute-recovery-evaluation",
            ]
        )
    assert not output.exists()
    with pytest.raises(SystemExit):
        subject.parse_args(["--updates", "1"])


def test_retrieval_reproduces_per_query_teacher_not_only_aggregate_and_retains_discordances() -> (
    None
):
    from sfora.qwen_geometry_control import geometry_retrieval_evidence

    vectors = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = (49, 49, 50, 50)
    cells = subject.retrieval_cells(
        {name: vectors for name in ("teacher", "pa", "relational")}, labels
    )
    assert cells["teacher"]["correct"] == 4 and cells["teacher"]["map_at_r"] == 1.0
    assert cells["teacher"]["retrieval"]["nearest_ordinals"] == (1, 0, 3, 2)
    baseline = asdict(geometry_retrieval_evidence(vectors, labels=labels, ordinals=(0, 1, 2, 3)))
    # Convert through real JSON: the historic receipt uses arrays, not tuples.
    import json

    baseline = json.loads(json.dumps(baseline))
    subject.require_teacher_reproduction(cells["teacher"], baseline)
    bad = copy.deepcopy(baseline)
    bad["nearest_ordinals"][0] = 3
    with pytest.raises(ValueError):
        subject.require_teacher_reproduction(cells["teacher"], bad)
    current = copy.deepcopy(cells)
    current["pa"]["retrieval"]["correct"] = (False, False, True, True)
    discordance = subject.paired_discordances(current["teacher"], current["pa"])
    assert discordance == {"both_correct": 2, "teacher_only": 2, "student_only": 0, "both_wrong": 0}


def test_asymmetric_retrieval_excludes_matching_id_and_is_gallery_order_invariant() -> None:
    query_ids = ("a", "b", "c", "d")
    query_labels = (49, 49, 50, 50)
    queries = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    gallery_order = (2, 0, 3, 1)
    galleries = queries[list(gallery_order)]
    gallery_ids = tuple(query_ids[i] for i in gallery_order)
    gallery_labels = tuple(query_labels[i] for i in gallery_order)

    checks: list[bool] = []
    evidence = asymmetric_core.asymmetric_retrieval_evidence(
        queries,
        galleries,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        check_time=lambda: checks.append(True),
    )

    assert checks == [True]
    assert evidence.nearest_ordinals == (1, 0, 3, 2)
    assert evidence.correct == (True, True, True, True)
    assert evidence.map_at_r == 1.0
    subject.validate_geometry_retrieval_evidence(evidence)

    with pytest.raises(ValueError, match="identity"):
        asymmetric_core.asymmetric_retrieval_evidence(
            queries,
            galleries,
            query_ids=query_ids,
            gallery_ids=("x", *gallery_ids[1:]),
            query_labels=query_labels,
            gallery_labels=gallery_labels,
        )
    zero = queries.clone()
    zero[0] = 0
    with pytest.raises(ValueError, match="nonzero"):
        asymmetric_core.asymmetric_retrieval_evidence(
            zero,
            galleries,
            query_ids=query_ids,
            gallery_ids=gallery_ids,
            query_labels=query_labels,
            gallery_labels=gallery_labels,
        )


def test_asymmetric_cross_map_gate_is_total_and_preregistered() -> None:
    cells = {
        "pa": {"queries": 2746, "map_at_r": 0.7000000000000001},
        "relational": {"queries": 2746, "map_at_r": 0.6},
    }
    result = asymmetric_core.asymmetric_recovery_decision(cells)
    assert result["selected_arm"] == "pa"
    assert result["arms"]["pa"]["classification"] == "alive"
    assert result["arms"]["relational"]["classification"] == "inconclusive-not-alive"
    assert result["claim_eligible"] is False

    cells["pa"]["map_at_r"] = 0.462
    result = asymmetric_core.asymmetric_recovery_decision(cells)
    assert result["selected_arm"] is None
    assert result["arms"]["pa"]["classification"] == "dead"
    assert result["arms"]["relational"]["classification"] == "inconclusive-not-alive"

    for bad in (float("nan"), True, "0.7"):
        mutated = copy.deepcopy(cells)
        mutated["pa"]["map_at_r"] = bad
        with pytest.raises(ValueError):
            asymmetric_core.asymmetric_recovery_decision(mutated)


def test_teacher_ranking_tie_drift_is_recorded_but_aggregate_quality_drift_fails() -> None:
    from sfora.qwen_geometry_control import geometry_retrieval_evidence

    vectors = torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    labels = (49, 49, 49, 50, 50, 50)
    cell = subject.retrieval_cells({k: vectors for k in ("teacher", "pa", "relational")}, labels)[
        "teacher"
    ]
    baseline = json.loads(
        json.dumps(
            asdict(geometry_retrieval_evidence(vectors, labels=labels, ordinals=tuple(range(6))))
        )
    )
    baseline["nearest_ordinals"][0] = 2
    baseline["top_r_ordinals"][0] = [2, 1]
    result = subject.require_teacher_reproduction(cell, baseline)
    assert result == {
        "aggregate_reproduced": True,
        "per_query_bitwise_reproduced": False,
        "first_differing_ordinal": 0,
    }
    wrong = copy.deepcopy(cell)
    wrong["correct"] = 5
    with pytest.raises(ValueError):
        subject.require_teacher_reproduction(wrong, baseline)


def test_quality_batch_size_bound_to_original_receipt_and_chunk_partition() -> None:
    assert subject.quality_batch_size({"environment": {"evaluation_batch_size": 32}}) == 32
    for value in (8, 48, True, 32.0):
        with pytest.raises(ValueError):
            subject.quality_batch_size({"environment": {"evaluation_batch_size": value}})


def test_decoded_pixel_hash_uses_native_audit_order_not_common_sorted_order() -> None:
    from PIL import Image

    from sfora.data import ImageExample

    examples = (
        ImageExample("b", Image.new("RGB", (1, 1), (4, 5, 6)), 50),
        ImageExample("a", Image.new("RGB", (1, 1), (1, 2, 3)), 49),
    )
    assert (
        subject.decoded_native_digest(examples, [1, 0])
        == "465db90f00314bd7239095f52c0ad8508fbce559d7eb60f200855b473c64277a"
    )
    assert (
        subject.decoded_native_digest(examples, [0, 1])
        != "465db90f00314bd7239095f52c0ad8508fbce559d7eb60f200855b473c64277a"
    )
    with pytest.raises(ValueError):
        subject.decoded_native_digest(examples, [0, 0])


def test_remaining_evaluation_time_includes_whole_training_monitor_not_only_script() -> None:
    receipt, _ = _authority()
    monitor = {
        "schema": "sfora-recovery-pair-monitor-v1",
        "claim_eligible": False,
        "exit_code": 0,
        "stop_reason": None,
        "prior_seconds": 1355,
        "elapsed_s": 12050.0,
        "result_sha256": "a" * 64,
    }
    assert subject.evaluation_budget_seconds(receipt, monitor, "a" * 64) == 8195.0
    for key, value in (
        ("exit_code", 125),
        ("stop_reason", "psi-cap"),
        ("elapsed_s", 21000.0),
        ("result_sha256", "b" * 64),
    ):
        with pytest.raises(ValueError):
            subject.evaluation_budget_seconds(receipt, {**monitor, key: value}, "a" * 64)


def test_inference_source_gate_distinguishes_control_runner_from_control_core() -> None:
    root = Path(__file__).parents[1]
    roles = {
        "runner": "scripts/run_siglip_recovery_smoke.py",
        "probe": "scripts/probe_siglip_depth_recovery.py",
        "control_runner": "scripts/run_siglip_proxy_control.py",
        "control_core": "src/sfora/siglip_proxy_control.py",
        "depth_core": "src/sfora/siglip_depth_recovery.py",
        "input_core": "src/sfora/siglip_recovery_inputs.py",
    }
    dependencies = {
        k: hashlib.sha256((root / p).read_bytes()).hexdigest() for k, p in roles.items()
    }
    subject.verify_inference_dependencies({"dependencies": dependencies})
    swapped = {**dependencies, "control_runner": dependencies["control_core"]}
    with pytest.raises(ValueError):
        subject.verify_inference_dependencies({"dependencies": swapped})


def test_evaluation_cuda_budget_fails_at_equality_not_only_after_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 96 * 1024**3 - 1)
    subject.check_cuda_evaluation_budget(torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 96 * 1024**3)
    with pytest.raises(RuntimeError, match="CUDA"):
        subject.check_cuda_evaluation_budget(torch.device("cuda"))
    subject.check_cuda_evaluation_budget(torch.device("cpu"))


def test_embedding_uses_real_processor_and_model_in_bounded_ordered_chunks() -> None:
    import numpy as np
    from PIL import Image

    from sfora.data import ImageExample
    from sfora.siglip_proxy_control import PooledProxyAnchorModel

    model = PooledProxyAnchorModel(
        tower=torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((2, 2)), torch.nn.Flatten(), torch.nn.Linear(12, 5)
        ),
        input_dimensions=5,
        embedding_dimensions=512,
        class_count=49,
    ).eval()
    examples = tuple(
        ImageExample(str(i), Image.new("RGB", (2, 2), (i % 255, 10, 20)), 49) for i in range(129)
    )
    seen: list[Any] = []
    batch_sizes: list[int] = []

    def processor(*, images: list[Image.Image], return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        batch_sizes.append(len(images))
        seen.extend(cast(tuple[int, ...], image.getpixel((0, 0)))[0] for image in images)
        pixels = (
            torch.from_numpy(np.stack([np.asarray(x) for x in images])).permute(0, 3, 1, 2).float()
            / 255
        )
        return {
            "pixel_values": torch.nn.functional.interpolate(pixels, size=(384, 384), mode="nearest")
        }

    checks = []
    result = subject.embed_recovery_model(
        model, examples, processor, torch.device("cpu"), lambda: checks.append(True)
    )
    assert seen == list(range(129)) and checks == [True, True]
    # Original authenticated seed17 quality extraction used32, unlike speed's8.
    assert batch_sizes == [32, 32, 32, 32, 1]
    assert result.shape == (129, 512) and result.dtype == torch.float32
    assert torch.allclose(
        torch.linalg.vector_norm(result, dim=1), torch.ones(129), atol=1e-6, rtol=0
    )


def test_full_final_evaluation_authenticates_and_restores_pair_before_measuring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from sfora.data import ImageExample
    from sfora.qwen_geometry_control import geometry_retrieval_evidence
    from sfora.siglip_proxy_control import PooledProxyAnchorModel

    class TinyTower(torch.nn.Module):
        vision_model: Any

        def __init__(self) -> None:
            super().__init__()
            self.vision_model = torch.nn.Module()
            self.vision_model.encoder = torch.nn.Module()
            self.vision_model.encoder.layers = torch.nn.ModuleList(
                torch.nn.Linear(2, 2) for _ in range(27)
            )
            self.vision_model.config = SimpleNamespace(num_hidden_layers=27)
            self.output = torch.nn.Linear(2, 1152)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.vision_model.encoder.layers:
                x = layer(x)
            return cast(torch.Tensor, self.output(x))

    torch.manual_seed(17)
    teacher = (
        PooledProxyAnchorModel(
            tower=TinyTower(), input_dimensions=1152, embedding_dimensions=512, class_count=49
        )
        .eval()
        .requires_grad_(False)
    )
    receipt, smoke = _authority()
    teacher_sha = subject.pair.control._model_state_sha256(teacher)
    receipt["teacher_state_sha256"] = smoke["teacher_state_sha256"] = teacher_sha
    for arm in ("pa", "relational"):
        student = subject.pair.prune_siglip_student(teacher)
        initial = subject.pair.control._model_state_sha256(student)
        with torch.no_grad():
            student.projection.weight.add_(0.001 if arm == "pa" else 0.002)
        ev = receipt["arms"][arm]
        ev["initial_state_sha256"] = smoke["arms"][arm]["initial_state_sha256"] = initial
        ev["final_state_sha256"] = subject.pair.control._model_state_sha256(student)
        receipt["checkpoints"][arm] = subject.pair.write_terminal_student(
            tmp_path / f"{arm}-final.pt", student, ev
        )

    def write(name: str, value: Any) -> tuple[Path, str]:
        path = tmp_path / name
        path.write_bytes(
            (
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode()
        )
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    pair_path, pair_sha = write("pair-complete.json", receipt)
    smoke_path, _ = write("smoke.json", smoke)
    monitor_path, monitor_sha = write(
        "monitor.json",
        {
            "schema": "sfora-recovery-pair-monitor-v1",
            "claim_eligible": False,
            "exit_code": 0,
            "stop_reason": None,
            "prior_seconds": 1355,
            "elapsed_s": 12050.0,
            "result_sha256": pair_sha,
        },
    )
    # Hand-designed150 errors: last150 rows point at class49, whose lower ordinal
    # representatives win the cosine ties. Every other row has a same-class peer.
    labels = tuple(49 + (i * 33 // 2746) for i in range(2746))
    vectors = torch.zeros(2746, 512)
    for i, label in enumerate(labels):
        vectors[i, label - 49] = 1
    vectors[-150:] = 0
    vectors[-150:, 0] = 1
    baseline = geometry_retrieval_evidence(vectors, labels=labels, ordinals=tuple(range(2746)))
    assert sum(baseline.correct) == 2596
    from PIL import Image

    examples = tuple(
        ImageExample(f"fixture-{i:04d}", Image.new("RGB", (1, 1), (i % 256, label, 0)), label)
        for i, label in enumerate(labels)
    )
    native_order = list(reversed(range(2746)))
    pixel_hash = hashlib.sha256()
    for native, common in enumerate(reversed(range(2746))):
        e = examples[common]
        pixel_hash.update(
            (
                json.dumps(
                    {
                        "ordinal": native,
                        "label": e.label,
                        "example_id": e.example_id,
                        "size": [1, 1],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        )
        pixel_hash.update(bytes((common % 256, e.label, 0)))
    audit_path, audit_sha = write(
        "audit.json",
        {
            "common_image_ids": [e.example_id for e in examples],
            "common_to_native": native_order,
            "decoded_native_sha256": pixel_hash.hexdigest(),
            "reproductions": {"siglip-projected-17": {"retrieval": asdict(baseline)}},
        },
    )
    monkeypatch.setattr(subject, "AUDIT_SHA256", audit_sha)
    # Substitute only independently tested immutable smoke parsing and external
    # model/cache/GPU/data/inference acquisition; checkpoint loading/ranking/gates
    # and canonical final output remain real.
    monkeypatch.setattr(subject.pair, "read_smoke_authority", lambda *args: smoke)
    monkeypatch.setattr(subject, "verify_inference_dependencies", lambda *args: None)
    monkeypatch.setattr(subject, "evaluation_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(subject, "load_teacher_and_processor", lambda root: (teacher, None))
    image_reads = []

    def images() -> tuple[ImageExample, ...]:
        image_reads.append(True)
        return examples

    monkeypatch.setattr(subject, "load_recovery_evaluation_images", images)
    measured = []

    def embed(
        model: Any,
        examples: Any,
        processor: Any,
        device: torch.device,
        check_time: Callable[[], None],
    ) -> torch.Tensor:
        assert model.projection.out_features == 512
        measured.append(len(model.tower.vision_model.encoder.layers))
        check_time()
        return vectors.clone()

    monkeypatch.setattr(subject, "embed_recovery_model", embed)
    output = tmp_path / "evaluation.json"
    arguments = [
        "--pair-directory",
        str(tmp_path),
        "--pair-sha256",
        pair_sha,
        "--smoke-result",
        str(smoke_path),
        "--audit-result",
        str(audit_path),
        "--pair-monitor",
        str(monitor_path),
        "--monitor-sha256",
        monitor_sha,
        "--control-root",
        str(tmp_path / "control"),
        "--output",
        str(output),
        "--execute-recovery-evaluation",
    ]
    subject.main(arguments)
    result = json.loads(output.read_bytes())
    assert image_reads == [True] and measured == [27, 18, 18]
    assert result["cells"]["teacher"]["correct"] == 2596
    assert result["cells"]["pa"]["map_at_r"] == baseline.map_at_r
    assert result["claim_eligible"] is False and result["quality_measured"] is True
    assert result["paired_discordances"]["pa"] == {
        "both_correct": 2596,
        "teacher_only": 0,
        "student_only": 0,
        "both_wrong": 150,
    }
    assert result["decision"]["arms"]["pa"]["gates"]["recall"] is True
    assert result["decision"]["arms"]["pa"]["gates"]["map"] is True
    # CPU timing stays real and noisy: do not assert either measured speed gate.
    assert result["pair_sha256"] == pair_sha and result["monitor_sha256"] == monitor_sha
    assert (
        result["source_sha256"]["retrieval_core"]
        == hashlib.sha256(
            (Path(__file__).parents[1] / "src/sfora/qwen_geometry_control.py").read_bytes()
        ).hexdigest()
    )
    assert (
        result["source_sha256"]["evaluation_core"]
        == hashlib.sha256(
            (Path(__file__).parents[1] / "src/sfora/siglip_recovery_evaluation.py").read_bytes()
        ).hexdigest()
    )
    assert output.read_bytes().endswith(b"\n") and not output.read_bytes().endswith(b"\n\n")
