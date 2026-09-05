"""Self-contained retrieval evaluation for the fixed SigLIP language pilot."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import probe_siglip_depth_recovery as probe
import run_siglip_proxy_control as control
import run_siglip_recovery_pair as pair
import run_siglip_recovery_smoke as smoke_runner
import torch
import transformers
from PIL import Image
from torch.nn import functional as F

from sfora.siglip_depth_recovery import RETAINED_BLOCKS, recovery_multiplier
from sfora.siglip_proxy_control import SiglipProxyControlConfig

AUDIT_SHA256 = "4ad592f0514bbb77515fe92d8b207c06d14c16271fd1c1bc0286d190718976cf"
SMOKE_SHA256 = "0481b835f594cbc9f910c40259a5d40c1958f236f51bbea49104cfdcaffd0344"
PAIR_RUNNER_SHA256 = "7b630dc3f15fec64729114e9a4a5edf70e570eb6ba4e21ae01951eaaf10fe6bb"
QUALITY_BATCH_SIZE = 32


@dataclass(frozen=True)
class RetrievalEvidence:
    """Per-image self-retrieval evidence in authenticated ordinal order."""

    ordinals: tuple[int, ...]
    labels: tuple[int, ...]
    nearest_ordinals: tuple[int, ...]
    top_r_ordinals: tuple[tuple[int, ...], ...]
    correct: tuple[bool, ...]
    average_precisions: tuple[float, ...]

    @property
    def recall_at_one(self) -> float:
        return sum(self.correct) / len(self.correct)

    @property
    def map_at_r(self) -> float:
        return math.fsum(self.average_precisions) / len(self.average_precisions)


def _exact(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def _digest(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def evaluation_budget_seconds(
    receipt: dict[str, Any], monitor: dict[str, Any], pair_sha: str
) -> float:
    """Use whole-process time, never refund setup or serialization."""
    fixed = {
        "schema": "sfora-recovery-pair-monitor-v1",
        "claim_eligible": False,
        "exit_code": 0,
        "stop_reason": None,
        "prior_seconds": 1355,
        "result_sha256": pair_sha,
    }
    if any(not _exact(monitor.get(key), value) for key, value in fixed.items()):
        raise ValueError("language recovery monitor is not the successful original")
    elapsed = monitor["elapsed_s"]
    if (
        type(elapsed) is not float
        or not math.isfinite(elapsed)
        or elapsed < receipt["resources"]["elapsed_seconds"]
        or monitor["prior_seconds"] < receipt["resources"]["prior_gpu_seconds"]
        or 1355 + elapsed >= 21600
    ):
        raise ValueError("language recovery monitor elapsed/budget differs")
    return 21600 - 1355 - elapsed


def validate_pair_receipt(value: dict[str, Any], smoke: dict[str, Any]) -> None:
    """Recompute final-only recovery ordering, identities, and budget evidence."""
    expected = {
        "schema": "sfora-siglip-depth-recovery-pair-v1",
        "claim_eligible": False,
        "quality_measured": False,
        "seed": 17,
        "updates_per_arm": 198,
        "status": "complete",
        "smoke_sha256": SMOKE_SHA256,
        "teacher_checkpoint_sha256": probe.CHECKPOINT_SHA256,
        "runner_sha256": PAIR_RUNNER_SHA256,
        "dependencies": smoke["source_sha256"],
        "retained_one_indexed_blocks": list(RETAINED_BLOCKS),
        "teacher_unchanged": True,
        "teacher_state_sha256": smoke["teacher_state_sha256"],
    }
    try:
        if any(
            not _exact(value.get(key), expected_value) for key, expected_value in expected.items()
        ):
            raise ValueError("language recovery fixed authority differs")
        if set(value["arms"]) != {"pa", "relational"} or set(value["checkpoints"]) != {
            "pa",
            "relational",
        }:
            raise ValueError("language recovery is not two sealed final arms")
        total_steps_ns = 0
        for name in ("pa", "relational"):
            arm, seal = value["arms"][name], value["checkpoints"][name]
            if (
                not _exact(arm["completed_updates"], 198)
                or arm["arm"] != name
                or not _exact(seal["completed_updates"], 198)
                or seal["arm"] != name
                or seal["basename"] != f"{name}-final.pt"
                or type(seal["bytes"]) is not int
                or seal["bytes"] <= 0
                or not _digest(seal["sha256"])
                or arm["initial_state_sha256"] != smoke["arms"][name]["initial_state_sha256"]
                or not _digest(arm["final_state_sha256"])
                or arm["final_state_sha256"] == arm["initial_state_sha256"]
                or type(arm["steps"]) is not list
                or len(arm["steps"]) != 198
                or type(arm["input_sha256"]) is not list
                or len(arm["input_sha256"]) != 198
                or not all(_digest(item) for item in arm["input_sha256"])
                or arm["input_sha256"][:10] != smoke["arms"][name]["input_sha256"]
            ):
                raise ValueError("language recovery final arm/checkpoint authority differs")
            for index, step in enumerate(arm["steps"], 1):
                numeric = (
                    "loss",
                    "proxy_loss",
                    "relational_loss",
                    "gradient_norm",
                    "maximum_descriptor_disagreement",
                    "lr_multiplier",
                )
                if (
                    not _exact(step["update"], index)
                    or step["arm"] != name
                    or type(step["elapsed_ns"]) is not int
                    or step["elapsed_ns"] <= 0
                    or any(
                        type(step[key]) is not float or not math.isfinite(step[key])
                        for key in numeric
                    )
                    or step["gradient_norm"] <= 0
                    or not 0 <= step["maximum_descriptor_disagreement"] <= 2e-5
                    or step["lr_multiplier"] != recovery_multiplier(index)
                    or (name == "pa" and step["relational_loss"] != 0.0)
                ):
                    raise ValueError("language recovery update numerical authority differs")
                total_steps_ns += step["elapsed_ns"]
        pa, relational = value["arms"]["pa"], value["arms"]["relational"]
        if (
            pa["input_sha256"] != relational["input_sha256"]
            or pa["initial_state_sha256"] != relational["initial_state_sha256"]
        ):
            raise ValueError("language recovery inputs or initial states differ")
        resources = value["resources"]
        prior = smoke["resources"]["elapsed_seconds"] + smoke_runner.PREFLIGHT_SECONDS
        for key in ("elapsed_seconds", "prior_gpu_seconds", "remaining_campaign_seconds"):
            if type(resources[key]) is not float or not math.isfinite(resources[key]):
                raise ValueError("language recovery budget numbers differ")
        if (
            resources["within_campaign_cap"] is not True
            or resources["elapsed_seconds"] < total_steps_ns / 1e9
            or resources["prior_gpu_seconds"] != prior
            or resources["remaining_campaign_seconds"]
            != 21600 - prior - resources["elapsed_seconds"]
            or resources["remaining_campaign_seconds"] <= 0
        ):
            raise ValueError("language recovery campaign budget differs")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("language recovery authority is incomplete") from error


def authenticate_checkpoint_files(directory: Path, receipt: dict[str, Any]) -> dict[str, Path]:
    """Authenticate both recovery checkpoint byte streams before tensor parsing."""
    paths = {}
    for arm in ("pa", "relational"):
        seal = receipt["checkpoints"][arm]
        path = directory / f"{arm}-final.pt"
        if (
            seal["basename"] != path.name
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != seal["bytes"]
            or probe._file_sha(path) != seal["sha256"]
        ):
            raise ValueError("language recovery checkpoint byte identity differs")
        paths[arm] = path
    return paths


def recovery_dependency_sha256() -> dict[str, str]:
    """Return the complete live recovery dependency identity used by the pilot."""
    import sfora.siglip_depth_recovery as depth_core
    import sfora.siglip_proxy_control as control_core
    import sfora.siglip_recovery_inputs as input_core

    modules = {
        "runner": smoke_runner,
        "depth_core": depth_core,
        "input_core": input_core,
        "probe": probe,
        "control_runner": control,
        "control_core": control_core,
    }
    return {role: probe._file_sha(Path(str(module.__file__))) for role, module in modules.items()}


def verify_recovery_dependencies(expected: Mapping[str, Any]) -> None:
    """Bind every live recovery implementation dependency to sealed training evidence."""
    observed = recovery_dependency_sha256()
    if (
        type(expected) is not dict
        or set(expected) != set(observed)
        or any(not _digest(value) for value in expected.values())
        or probe._file_sha(Path(pair.__file__)) != PAIR_RUNNER_SHA256
        or observed != expected
    ):
        raise ValueError("language recovery live dependencies differ")


def validate_student_payload(payload: Mapping[str, Any], receipt: dict[str, Any], arm: str) -> None:
    """Bind a finite FP32 state to its exact recovery terminal arm."""
    evidence = receipt["arms"][arm]
    expected = {
        "schema": "sfora-siglip-depth-recovery-student-v1",
        "claim_eligible": False,
        "seed": 17,
        "completed_updates": 198,
        "arm": arm,
        "teacher_checkpoint_sha256": receipt["teacher_checkpoint_sha256"],
        "retained_one_indexed_blocks": list(RETAINED_BLOCKS),
        "input_dimensions": 1152,
        "embedding_dimensions": 512,
        "initial_state_sha256": evidence["initial_state_sha256"],
        "final_state_sha256": evidence["final_state_sha256"],
        "input_sha256": evidence["input_sha256"],
    }
    if set(payload) != {*expected, "model_state"} or any(
        not _exact(payload.get(key), value) for key, value in expected.items()
    ):
        raise ValueError("language recovery student payload bindings differ")
    state = payload["model_state"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("language recovery student state absent")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if (
            type(name) is not str
            or not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("language recovery student state must be finite FP32")
        meta = control._canonical_bytes(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)}
        )
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(meta).to_bytes(8, "little"))
        digest.update(meta)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    if digest.hexdigest() != evidence["final_state_sha256"]:
        raise ValueError("language recovery student actual state digest differs")


def _retrieval_evidence(
    descriptors: torch.Tensor,
    *,
    labels: tuple[int, ...],
) -> RetrievalEvidence:
    """Compute exact blocked CPU FP32 cosine rankings with stable ordinal ties."""
    ordinals = tuple(range(len(labels)))
    count = len(ordinals)
    if (
        count < 2
        or any(type(label) is not int or label < 0 for label in labels)
        or descriptors.ndim != 2
        or descriptors.shape[0] != count
        or descriptors.shape[1] == 0
        or not bool(torch.isfinite(descriptors).all())
    ):
        raise ValueError("language retrieval inputs differ")
    vectors = descriptors.detach().to(device="cpu", dtype=torch.float32)
    norms = torch.linalg.vector_norm(vectors, dim=1)
    if not bool(torch.isfinite(norms).all()) or not bool((norms > 0).all()):
        raise ValueError("language descriptors require finite nonzero norms")
    class_counts = {label: labels.count(label) for label in set(labels)}
    if min(class_counts.values()) < 2:
        raise ValueError("language retrieval classes require pairs")
    vectors = F.normalize(vectors, dim=1)
    label_tensor = torch.tensor(labels)
    nearest: list[int] = []
    correct: list[bool] = []
    average_precisions: list[float] = []
    top_r: list[tuple[int, ...]] = []
    for start in range(0, count, 128):
        scores = vectors[start : start + 128] @ vectors.T
        for local, row in enumerate(range(start, min(start + 128, count))):
            scores[local, row] = -torch.inf
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)
        for local, row in enumerate(range(start, min(start + 128, count))):
            first = int(ranked[local, 0])
            nearest.append(first)
            correct.append(labels[first] == labels[row])
            relevant_count = class_counts[labels[row]] - 1
            retained = ranked[local, :relevant_count]
            top_r.append(tuple(int(index) for index in retained))
            relevant = (label_tensor[retained] == labels[row]).tolist()
            hits = 0
            terms = []
            for rank, hit in enumerate(relevant, 1):
                if hit:
                    hits += 1
                    terms.append(hits / rank)
            average_precisions.append(math.fsum(terms) / relevant_count)
    value = RetrievalEvidence(
        ordinals=ordinals,
        labels=labels,
        nearest_ordinals=tuple(nearest),
        top_r_ordinals=tuple(top_r),
        correct=tuple(correct),
        average_precisions=tuple(average_precisions),
    )
    _validate_retrieval_evidence(value)
    return value


def _validate_retrieval_evidence(value: RetrievalEvidence) -> None:
    """Recompute hit and AP evidence from retained rankings and labels."""
    count = len(value.ordinals)
    if (
        count < 2
        or value.ordinals != tuple(range(count))
        or any(type(item) is not int for item in value.labels)
        or any(
            len(item) != count
            for item in (
                value.labels,
                value.nearest_ordinals,
                value.top_r_ordinals,
                value.correct,
                value.average_precisions,
            )
        )
    ):
        raise ValueError("language retrieval evidence shape differs")
    class_counts = {label: value.labels.count(label) for label in set(value.labels)}
    for ordinal, label in zip(value.ordinals, value.labels, strict=True):
        ranks = value.top_r_ordinals[ordinal]
        relevant_count = class_counts[label] - 1
        if (
            relevant_count < 1
            or len(ranks) != relevant_count
            or len(set(ranks)) != relevant_count
            or ordinal in ranks
            or any(type(item) is not int or not 0 <= item < count for item in ranks)
            or value.nearest_ordinals[ordinal] != ranks[0]
        ):
            raise ValueError("language retained ranking differs")
        expected_correct = value.labels[ranks[0]] == label
        hits = 0
        terms = []
        for rank, item in enumerate(ranks, 1):
            if value.labels[item] == label:
                hits += 1
                terms.append(hits / rank)
        expected_ap = math.fsum(terms) / relevant_count
        if (
            type(value.correct[ordinal]) is not bool
            or value.correct[ordinal] != expected_correct
            or type(value.average_precisions[ordinal]) is not float
            or value.average_precisions[ordinal] != expected_ap
        ):
            raise ValueError("language metrics differ from retained rankings")


def _retrieval_cell(vectors: torch.Tensor, labels: tuple[int, ...]) -> dict[str, Any]:
    evidence = _retrieval_evidence(vectors, labels=labels)
    retrieval = {
        "ordinals": list(evidence.ordinals),
        "labels": list(evidence.labels),
        "nearest_ordinals": list(evidence.nearest_ordinals),
        "top_r_ordinals": [list(row) for row in evidence.top_r_ordinals],
        "correct": list(evidence.correct),
        "average_precisions": list(evidence.average_precisions),
    }
    return {
        "queries": len(labels),
        "correct": sum(evidence.correct),
        "recall_at_one": evidence.recall_at_one,
        "map_at_r": evidence.map_at_r,
        "descriptor_bytes": vectors.numel() * vectors.element_size(),
        "retrieval": retrieval,
    }


def _evidence_from_mapping(item: dict[str, Any]) -> RetrievalEvidence:
    if set(item) != {
        "ordinals",
        "labels",
        "nearest_ordinals",
        "top_r_ordinals",
        "correct",
        "average_precisions",
    }:
        raise ValueError("language teacher retrieval schema differs")
    value = RetrievalEvidence(
        ordinals=tuple(item["ordinals"]),
        labels=tuple(item["labels"]),
        nearest_ordinals=tuple(item["nearest_ordinals"]),
        top_r_ordinals=tuple(tuple(row) for row in item["top_r_ordinals"]),
        correct=tuple(item["correct"]),
        average_precisions=tuple(item["average_precisions"]),
    )
    _validate_retrieval_evidence(value)
    return value


def require_teacher_reproduction(cell: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Require aggregate/ordinal reproduction and disclose any ranking-level drift."""
    actual = _evidence_from_mapping(cell["retrieval"])
    previous = _evidence_from_mapping(baseline)
    if (
        actual.ordinals != previous.ordinals
        or actual.labels != previous.labels
        or cell["correct"] != sum(actual.correct)
        or cell["map_at_r"] != actual.map_at_r
        or sum(actual.correct) != sum(previous.correct)
        or actual.map_at_r != previous.map_at_r
    ):
        raise ValueError("language teacher aggregate or ordinal reproduction differs")
    exact = control._canonical_bytes(cell["retrieval"]) == control._canonical_bytes(baseline)
    first = next(
        (
            ordinal
            for ordinal in actual.ordinals
            if any(
                getattr(actual, key)[ordinal] != getattr(previous, key)[ordinal]
                for key in (
                    "nearest_ordinals",
                    "top_r_ordinals",
                    "correct",
                    "average_precisions",
                )
            )
        ),
        None,
    )
    return {
        "aggregate_reproduced": True,
        "per_query_bitwise_reproduced": exact,
        "first_differing_ordinal": first,
    }


def paired_discordances(first: dict[str, Any], second: dict[str, Any]) -> dict[str, int]:
    """Count paired correctness outcomes with positional legacy field names."""
    left, right = first["retrieval"]["correct"], second["retrieval"]["correct"]
    if len(left) != len(right) or any(type(value) is not bool for value in (*left, *right)):
        raise ValueError("language paired correctness evidence differs")
    return {
        "both_correct": sum(a and b for a, b in zip(left, right, strict=True)),
        "teacher_only": sum(a and not b for a, b in zip(left, right, strict=True)),
        "student_only": sum(not a and b for a, b in zip(left, right, strict=True)),
        "both_wrong": sum(not a and not b for a, b in zip(left, right, strict=True)),
    }


def decoded_native_digest(examples: tuple[Any, ...], common_to_native: list[int]) -> str:
    """Reproduce the audit RGB digest in native rather than common order."""
    count = len(examples)
    if (
        len(common_to_native) != count
        or any(type(index) is not int for index in common_to_native)
        or sorted(common_to_native) != list(range(count))
    ):
        raise ValueError("language native/common pixel permutation differs")
    common_order = sorted(range(count), key=lambda index: common_to_native[index])
    digest = hashlib.sha256()
    for ordinal, common in enumerate(common_order):
        example = examples[common]
        if not isinstance(example.image, Image.Image):
            raise ValueError("language evaluation pixel evidence is not decoded RGB")
        rgb = example.image.convert("RGB")
        digest.update(
            control._canonical_bytes(
                {
                    "ordinal": ordinal,
                    "label": example.label,
                    "example_id": example.example_id,
                    "size": list(rgb.size),
                }
            )
        )
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def evaluation_device() -> torch.device:
    """Require the pinned deterministic CUDA evaluation environment."""
    if not torch.cuda.is_available() or transformers.__version__ != "5.12.1":
        raise RuntimeError("scientific language evaluation requires pinned CUDA")
    device = torch.device("cuda")
    control.require_control_determinism(device)
    return device


def load_teacher_and_processor(root: Path) -> tuple[Any, Any]:
    """Restore the authenticated teacher and its local-only image processor."""
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor

    receipt_path = root / "seed-017.receipt.json"
    if probe._file_sha(receipt_path) != probe.SEED_RECEIPT_SHA256:
        raise ValueError("language teacher receipt SHA differs")
    receipt = json.loads(receipt_path.read_bytes())
    if (
        control._canonical_bytes(receipt) != receipt_path.read_bytes()
        or receipt["environment"]["evaluation_batch_size"] != QUALITY_BATCH_SIZE
    ):
        raise ValueError("language teacher evaluation batch authority differs")
    teacher = pair.load_teacher(root)
    config = SiglipProxyControlConfig()
    snapshot = Path(
        snapshot_download(config.model_name, revision=config.model_revision, local_files_only=True)
    ).resolve(strict=True)
    if snapshot.name != config.model_revision:
        raise ValueError("language evaluation processor revision differs")
    processor_class: Any = AutoImageProcessor
    processor = processor_class.from_pretrained(str(snapshot), local_files_only=True)
    return teacher, processor


def embed_recovery_model(
    model: Any,
    examples: tuple[Any, ...],
    processor: Any,
    device: torch.device,
    check_time: Any,
) -> torch.Tensor:
    """Embed the complete gallery in bounded chunks with time/CUDA checks."""
    chunks = []
    for start in range(0, len(examples), 128):
        check_time()
        _check_cuda_budget(device)
        print(f"language-eval: embedding {start}/{len(examples)}", flush=True)
        _, projected, _ = control.embed_control_examples(
            model=model,
            examples=examples[start : start + 128],
            processor=processor,
            device=device,
            batch_size=QUALITY_BATCH_SIZE,
        )
        chunks.append(projected)
        _check_cuda_budget(device)
    return torch.cat(chunks)


def _check_cuda_budget(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.max_memory_reserved() >= 96 * 1024**3:
        raise RuntimeError("language evaluation CUDA memory limit")
