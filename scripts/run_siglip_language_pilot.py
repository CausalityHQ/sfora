#!/usr/bin/env python3
"""Local authenticated boundaries for the fixed class-language pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from transformers import AutoTokenizer, SiglipTextConfig, SiglipTextModel

import sfora.siglip_language_guidance as language_guidance
import sfora.siglip_language_protocol as language_protocol
from sfora.siglip_depth_recovery import recovery_multiplier
from sfora.siglip_language_guidance import recomputed_language_backward, standardized_text_gram
from sfora.siglip_language_protocol import fixed_language_permutation, pilot_training_projection
from sfora.siglip_proxy_control import PooledProxyAnchorModel
from sfora.siglip_recovery_evaluation import recovery_decision

EVALUATOR_SHA256 = "e60d8fa318a17b69985deb2c8f43339427b34576f34f48bf66a21cf683ab6985"
EVALUATION_SOURCE_SHA256 = {
    "runner": EVALUATOR_SHA256,
    "evaluation_core": "d8952c1ea5e6ea9c747379ee07e25330aab9632829ae90fc179ebde8ffad0568",
    "retrieval_core": "fa2f06d1fa78a8058b1a90f4103eb3966e90931504acd990fbb5440f61bee34c",
}

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import evaluate_siglip_recovery_pair as evaluator  # noqa: E402
import run_siglip_proxy_control as control  # noqa: E402
import run_siglip_recovery_pair as pair  # noqa: E402
import run_siglip_recovery_smoke as smoke  # noqa: E402


def _authenticate(path: Path, expected_sha256: str) -> None:
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ValueError("input digest authority invalid")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("input digest differs")


def _read_canonical_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("input must be a regular file")
    _authenticate(path, expected_sha256)
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or control._canonical_bytes(value) != raw:
        raise ValueError("input is not finite canonical JSON")
    return value


def load_text_state(path: Path, expected_sha256: str, model: torch.nn.Module) -> None:
    """Authenticate the full container, then strictly restore only its text state."""
    _authenticate(path, expected_sha256)
    expected = model.state_dict()
    with safe_open(path, framework="pt", device="cpu") as container:
        container_keys = container.keys()
        keys = {k for k in container_keys if k.startswith("text_model.")}
        if keys != {f"text_model.{k}" for k in expected}:
            raise ValueError("text container key authority differs")
        state = {}
        for key in sorted(keys):
            name = key.removeprefix("text_model.")
            value = container.get_tensor(key)
            reference = expected[name]
            if (
                value.shape != reference.shape
                or value.dtype != torch.float32
                or reference.dtype != torch.float32
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("text container tensor authority differs")
            state[name] = value
    model.load_state_dict(state, strict=True)


def validate_text_header(path: Path, expected_sha256: str, model: torch.nn.Module) -> None:
    """Compare text tensor names, shapes, and dtypes without reading their payloads."""
    _authenticate(path, expected_sha256)
    expected = model.state_dict()
    with safe_open(path, framework="pt", device="cpu") as container:
        container_keys = container.keys()
        keys = {key for key in container_keys if key.startswith("text_model.")}
        if keys != {f"text_model.{key}" for key in expected}:
            raise ValueError("text container header authority differs")
        for key in sorted(keys):
            reference = expected[key.removeprefix("text_model.")]
            tensor_slice = container.get_slice(key)
            if (
                reference.dtype != torch.float32
                or tensor_slice.get_dtype() != "F32"
                or tuple(tensor_slice.get_shape()) != tuple(reference.shape)
            ):
                raise ValueError("text container header authority differs")


def official_optimization_prompts(path: Path, expected_sha256: str) -> tuple[str, ...]:
    """Bind the official name mapping but construct prompts for IDs0..48 only."""
    _authenticate(path, expected_sha256)
    value = json.loads(path.read_bytes())
    try:
        label = value["features"]["label"]
        names = label["names"]
        valid = (
            label["_type"] == "ClassLabel"
            and type(names) is list
            and len(names) == 196
            and all(type(name) is str and name.strip() for name in names)
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError("official class-name metadata differs")
    return tuple(f"a photo of a {names[i]}." for i in range(49))


def encode_frozen_text(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    expected_tokens_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one authenticated49x64 batch without a synthesized attention mask."""
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if (
        input_ids.shape != (49, 64)
        or input_ids.dtype != torch.int64
        or type(vocab_size) is not int
        or vocab_size <= 0
        or bool((input_ids < 0).any())
        or bool((input_ids >= vocab_size).any())
        or hashlib.sha256(input_ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        != expected_tokens_sha256
    ):
        raise ValueError("text token authority differs")
    model.eval().requires_grad_(False)
    with torch.no_grad():
        pooled = model(input_ids=input_ids).pooler_output.float()
        norms = pooled.norm(dim=1, keepdim=True)
        if not bool(torch.isfinite(pooled).all()) or bool((norms <= 0).any()):
            raise ValueError("invalid frozen text descriptors")
        vectors = pooled / norms
        gram = standardized_text_gram(vectors)
    return vectors, gram


def prepare_text_targets(
    args: Any,
    device: torch.device,
    progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Authenticate the frozen text inputs, encode IDs0..48, and seal both controls."""
    started_ns = perf_counter_ns()
    expected_snapshot = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
    roles = (
        "model",
        "config",
        "tokenizer",
        "tokenizer_config",
        "spiece",
        "special_tokens",
        "dataset_info",
    )
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for role in roles:
        path = Path(getattr(args, role))
        digest = getattr(args, f"{role}_sha256")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{role} source must be a regular file")
        _authenticate(path, digest)
        paths[role], digests[role] = path, digest
    snapshot = Path(args.snapshot)
    if snapshot.name != expected_snapshot or any(
        paths[role].resolve().parent != snapshot.resolve()
        for role in roles
        if role != "dataset_info"
    ):
        raise ValueError("text snapshot authority differs")
    progress({"stage": "text-targets-authenticated", "input_sha256": digests})

    config = SiglipTextConfig.from_json_file(str(paths["config"]))
    if (
        type(config.hidden_size) is not int
        or config.hidden_size != 1152
        or type(config.num_hidden_layers) is not int
        or config.num_hidden_layers != 27
        or type(config.vocab_size) is not int
        or config.vocab_size != 32000
    ):
        raise ValueError("text configuration authority differs")
    with torch.device("meta"):
        header_model = SiglipTextModel(config)
    validate_text_header(paths["model"], digests["model"], header_model)
    del header_model
    model = SiglipTextModel(config)
    load_text_state(paths["model"], digests["model"], model)
    prompts = official_optimization_prompts(
        paths["dataset_info"],
        digests["dataset_info"],
    )
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    encoded = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    if type(encoded) is not dict or set(encoded) != {"input_ids"}:
        raise ValueError("text tokenizer output authority differs")
    input_ids = encoded["input_ids"]
    torch.nn.Module.to(model, device)
    vectors, _ = encode_frozen_text(model, input_ids.to(device), args.expected_tokens_sha256)
    source = {
        **digests,
        "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "guidance": hashlib.sha256(Path(language_guidance.__file__).read_bytes()).hexdigest(),
        "protocol": hashlib.sha256(Path(language_protocol.__file__).read_bytes()).hexdigest(),
    }
    elapsed_seconds = (perf_counter_ns() - started_ns) / 1e9
    receipt = seal_language_targets(
        Path(args.output_dir),
        vectors.cpu(),
        input_ids.cpu(),
        prompts,
        source,
        expected_tokens_sha256=args.expected_tokens_sha256,
        elapsed_seconds=float(elapsed_seconds),
    )
    progress({"stage": "text-targets-sealed", "receipt": receipt})
    return receipt


def _gram_digest(gram: torch.Tensor) -> str:
    framing = control._canonical_bytes({"dtype": str(gram.dtype), "shape": list(gram.shape)})
    return hashlib.sha256(
        b"sfora-language-gram-v1\0" + framing + gram.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _target_components(
    vectors: torch.Tensor,
    input_ids: torch.Tensor,
    prompts: tuple[str, ...],
    input_sha256: dict[str, str],
    expected_tokens_sha256: str,
    elapsed_seconds: float,
) -> dict[str, torch.Tensor]:
    roles = {
        "model",
        "config",
        "tokenizer",
        "tokenizer_config",
        "spiece",
        "special_tokens",
        "dataset_info",
        "runner",
        "guidance",
        "protocol",
    }
    if (
        vectors.shape != (49, 1152)
        or vectors.dtype != torch.float32
        or vectors.requires_grad
        or not bool(torch.isfinite(vectors).all())
        or not torch.allclose(
            vectors.norm(dim=1), torch.ones(49, device=vectors.device), atol=2e-5, rtol=0
        )
        or input_ids.shape != (49, 64)
        or input_ids.dtype != torch.int64
        or bool((input_ids < 0).any())
        or bool((input_ids >= 32000).any())
        or hashlib.sha256(input_ids.cpu().contiguous().numpy().tobytes()).hexdigest()
        != expected_tokens_sha256
        or type(prompts) is not tuple
        or len(prompts) != 49
        or any(
            type(p) is not str or not p.startswith("a photo of a ") or not p.endswith(".")
            for p in prompts
        )
        or len(set(prompts)) != 49
        or type(input_sha256) is not dict
        or set(input_sha256) != roles
        or any(
            type(h) is not str or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
            for h in input_sha256.values()
        )
        or type(elapsed_seconds) is not float
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0
    ):
        raise ValueError("language target inputs differ")
    # CPU is the fixed target-matrix authority; training copies these exact bytes.
    vectors = vectors.detach().cpu().contiguous()
    correct = standardized_text_gram(vectors)
    permutation = torch.tensor(fixed_language_permutation(), dtype=torch.int64)
    permuted = correct[permutation][:, permutation].contiguous()
    if torch.equal(correct, permuted):
        raise ValueError("language control has identical correct/permuted targets")
    return {
        "vectors": vectors,
        "input_ids": input_ids.cpu().contiguous(),
        "correct": correct,
        "permuted": permuted,
    }


def seal_language_targets(
    directory: Path,
    vectors: torch.Tensor,
    input_ids: torch.Tensor,
    prompts: tuple[str, ...],
    input_sha256: dict[str, str],
    *,
    expected_tokens_sha256: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Seal the fixed49-class bundle; callers first authenticate source bytes and names."""
    path, blob = directory / "language-targets.json", directory / "language-targets.safetensors"
    if any(p.exists() or p.is_symlink() for p in (path, blob)):
        raise FileExistsError("language targets already exist")
    tensors = _target_components(
        vectors,
        input_ids,
        prompts,
        input_sha256,
        expected_tokens_sha256,
        elapsed_seconds,
    )
    raw = save_safetensors(tensors)
    receipt = {
        "schema": "sfora-siglip-language-targets-v1",
        "claim_eligible": False,
        "quality_measured": False,
        "seed": 17,
        "class_ids": list(range(49)),
        "prompts": list(prompts),
        "permutation": list(fixed_language_permutation()),
        "input_sha256": input_sha256,
        "tokens_sha256": expected_tokens_sha256,
        "tensors_basename": blob.name,
        "tensors_bytes": len(raw),
        "tensors_sha256": hashlib.sha256(raw).hexdigest(),
        "gram_sha256": {arm: _gram_digest(tensors[arm]) for arm in ("correct", "permuted")},
        "elapsed_seconds": elapsed_seconds,
        "torch_version": str(torch.__version__),
        "gram_backend": "cpu-float32",
    }
    control._write_new(blob, raw)
    control._write_new(path, control._canonical_bytes(receipt))
    return receipt


def load_language_targets(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Authenticate complete bundle before parsing tensors and deriving either control."""
    _authenticate(path, expected_sha256)
    raw = path.read_bytes()
    receipt = json.loads(raw)
    keys = {
        "schema",
        "claim_eligible",
        "quality_measured",
        "seed",
        "class_ids",
        "prompts",
        "permutation",
        "input_sha256",
        "tokens_sha256",
        "tensors_basename",
        "tensors_bytes",
        "tensors_sha256",
        "gram_sha256",
        "elapsed_seconds",
        "torch_version",
        "gram_backend",
    }
    if (
        type(receipt) is not dict
        or set(receipt) != keys
        or control._canonical_bytes(receipt) != raw
        or receipt["schema"] != "sfora-siglip-language-targets-v1"
        or receipt["claim_eligible"] is not False
        or receipt["quality_measured"] is not False
        or type(receipt["seed"]) is not int
        or receipt["seed"] != 17
        or receipt["class_ids"] != list(range(49))
        or receipt["permutation"] != list(fixed_language_permutation())
        or any(type(i) is not int for i in receipt["class_ids"] + receipt["permutation"])
        or type(receipt["prompts"]) is not list
        or receipt["tensors_basename"] != "language-targets.safetensors"
        or type(receipt["tensors_bytes"]) is not int
        or not 0 < receipt["tensors_bytes"] < 1000000
        or receipt["gram_backend"] != "cpu-float32"
        or receipt["torch_version"] != str(torch.__version__)
    ):
        raise ValueError("language target receipt differs")
    blob = path.parent / receipt["tensors_basename"]
    if blob.stat().st_size != receipt["tensors_bytes"]:
        raise ValueError("language target tensor length differs")
    _authenticate(blob, receipt["tensors_sha256"])
    tensors = load_safetensors(blob.read_bytes())
    if set(tensors) != {"vectors", "input_ids", "correct", "permuted"}:
        raise ValueError("language target tensor keys differ")
    expected = _target_components(
        tensors["vectors"],
        tensors["input_ids"],
        tuple(receipt["prompts"]),
        receipt["input_sha256"],
        receipt["tokens_sha256"],
        receipt["elapsed_seconds"],
    )
    if any(not torch.equal(tensors[k], expected[k]) for k in expected) or receipt[
        "gram_sha256"
    ] != {arm: _gram_digest(tensors[arm]) for arm in ("correct", "permuted")}:
        raise ValueError("language target control correspondence differs")
    return {"receipt": receipt, "tensors": tensors, "receipt_sha256": expected_sha256}


def recovery_initialization_choice(
    evaluation: dict[str, Any],
    monitor: dict[str, Any],
    *,
    pair_sha256: str,
    pair_monitor_sha256: str,
    evaluation_sha256: str,
) -> dict[str, Any]:
    """Recompute selection from authenticated result views, not copied decisions.

    The caller must first authenticate full files and the pair/checkpoint chain.
    This function validates the distinct evaluation monitor and measured gates;
    no remaining time from the recovery campaign becomes pilot time.
    """
    expected_evaluation = {
        "schema": "sfora-siglip-depth-recovery-evaluation-v1",
        "claim_eligible": False,
        "quality_measured": True,
        "seed": 17,
        "surface": "exploratory-reuse-49..81",
        "pair_sha256": pair_sha256,
        "monitor_sha256": pair_monitor_sha256,
        "runner_sha256": EVALUATOR_SHA256,
        "source_sha256": EVALUATION_SOURCE_SHA256,
    }
    expected_monitor = {
        "schema": "sfora-recovery-evaluation-monitor-v1",
        "claim_eligible": False,
        "exit_code": 0,
        "stop_reason": None,
        "result_sha256": evaluation_sha256,
        "pair_monitor_sha256": pair_monitor_sha256,
        "monitor_sha256": "2b542c65ab9ec644693559fafddcfdccbddff1aedf297307fe2abb2d22792987",
        "script_sha256": "140a8160dc81c3585247ff5e7227e87a99f8ffff6839fb0f6ba11db601e3072d",
    }
    for actual, expected in ((evaluation, expected_evaluation), (monitor, expected_monitor)):
        for key, wanted in expected.items():
            got = actual.get(key)
            if type(got) is not type(wanted) or got != wanted:
                raise ValueError(f"recovery evaluation {key} authority differs")
    try:
        elapsed = evaluation["resources"]["elapsed_seconds"]
        monitor_elapsed = monitor["elapsed_s"]
        if any(
            type(x) is not float or not math.isfinite(x) or x <= 0
            for x in (elapsed, monitor_elapsed)
        ):
            raise ValueError("recovery evaluation elapsed evidence invalid")
        projection = 2 * max(elapsed, monitor_elapsed) + 300
        if not math.isfinite(projection) or projection > 1800:
            raise ValueError("four-gallery evaluation reserve is insufficient")
        cells = evaluation["cells"]
        if set(cells) != {"teacher", "pa", "relational"}:
            raise ValueError("recovery evaluation cell authority differs")
        teacher = cells["teacher"]
        if teacher["map_at_r"] != 0.7913744556922272:
            raise ValueError("recovery teacher MAP reproduction differs")
        decision = recovery_decision(
            teacher,
            {k: cells[k] for k in ("pa", "relational")},
            evaluation["search_profile"]["samples_ns"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("recovery evaluation evidence incomplete") from error
    selected = decision["selected_arm"]
    return {
        "initialization": selected or "teacher",
        "layers": 18 if selected else 27,
        "relational_base": selected == "relational",
        "recomputed_decision": decision,
        "evaluation_projection_seconds": projection,
        "pilot_total_seconds": 7200,
    }


def read_initialization(args: Any) -> dict[str, Any]:
    """Authenticate the completed recovery chain without loading quality images."""
    pair_path = args.pair_directory / "pair-complete.json"
    pair_receipt = _read_canonical_json(pair_path, args.pair_sha256)
    smoke_authority = pair.read_smoke_authority(args.smoke_result, args.smoke_sha256)
    evaluator.validate_pair_receipt(pair_receipt, smoke_authority)
    pair_monitor = _read_canonical_json(args.pair_monitor, args.pair_monitor_sha256)
    # Validation only; the recovery campaign remainder never becomes pilot time.
    evaluator.evaluation_budget_seconds(pair_receipt, pair_monitor, args.pair_sha256)
    checkpoints = evaluator.authenticate_checkpoint_files(args.pair_directory, pair_receipt)
    for arm in ("pa", "relational"):
        payload = torch.load(checkpoints[arm], map_location="cpu", weights_only=True, mmap=True)
        evaluator.validate_student_payload(payload, pair_receipt, arm)
        del payload
    evaluation = _read_canonical_json(args.evaluation, args.evaluation_sha256)
    evaluation_monitor = _read_canonical_json(
        args.evaluation_monitor,
        args.evaluation_monitor_sha256,
    )
    choice = recovery_initialization_choice(
        evaluation,
        evaluation_monitor,
        pair_sha256=args.pair_sha256,
        pair_monitor_sha256=args.pair_monitor_sha256,
        evaluation_sha256=args.evaluation_sha256,
    )
    selected = choice["initialization"]
    return {
        "selected_arm": selected,
        "initialization_path": checkpoints.get(selected, args.teacher_checkpoint),
        "choice": choice,
        "pair_sha256": args.pair_sha256,
        "pair_monitor_sha256": args.pair_monitor_sha256,
        "evaluation_sha256": args.evaluation_sha256,
        "evaluation_monitor_sha256": args.evaluation_monitor_sha256,
    }


def train_language_arm(
    model: PooledProxyAnchorModel,
    teacher: PooledProxyAnchorModel | None,
    batch: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    *,
    arm: str,
    gram: torch.Tensor | None,
    expected_input_hashes: list[str] | None,
    microbatch_size: int,
    progress: Callable[[dict[str, Any]], None],
    synchronize: Callable[[], None],
) -> dict[str, Any]:
    """Run exactly20 fresh-optimizer updates; the caller supplies the arm's Gram.

    Scientific30x4/input/target authority is checked by the phase runner. This
    model-level kernel also supports reduced CPU differential fixtures.
    """
    return _train_language_steps(
        model,
        teacher,
        batch,
        arm=arm,
        gram=gram,
        expected_input_hashes=expected_input_hashes,
        microbatch_size=microbatch_size,
        progress=progress,
        synchronize=synchronize,
        updates=20,
    )


def _train_language_steps(
    model: PooledProxyAnchorModel,
    teacher: PooledProxyAnchorModel | None,
    batch: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    *,
    arm: str,
    gram: torch.Tensor | None,
    expected_input_hashes: list[str] | None,
    microbatch_size: int,
    progress: Callable[[dict[str, Any]], None],
    synchronize: Callable[[], None],
    updates: int,
) -> dict[str, Any]:
    if type(updates) is not int or updates not in (3, 20):
        raise ValueError("only disposable3 or final20 language updates exist")
    if arm not in ("base", "correct", "permuted") or (gram is None) != (arm == "base"):
        raise ValueError("language arm/target differs")
    consumed_gram_sha256 = None
    if gram is not None:
        # Own the actual consumed target; callbacks cannot mutate caller storage mid-run.
        gram = gram.detach().clone().contiguous()
        consumed_gram_sha256 = _gram_digest(gram)
    if arm == "base":
        if expected_input_hashes is not None:
            raise ValueError("base arm must establish paired input authority")
    elif (
        type(expected_input_hashes) is not list
        or len(expected_input_hashes) != updates
        or any(
            type(h) is not str or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
            for h in expected_input_hashes
        )
    ):
        raise ValueError("language arm paired input hash count differs")
    if teacher is not None:
        if any(m.training for m in teacher.modules()) or any(
            p.requires_grad or p.grad is not None for p in teacher.parameters()
        ):
            raise ValueError("language teacher must be frozen/eval")
        if {p.data_ptr() for p in teacher.parameters()} & {
            p.data_ptr() for p in model.parameters()
        }:
            raise ValueError("language teacher/student storage overlaps")
    initial = control._model_state_sha256(model)
    teacher_initial = None if teacher is None else control._model_state_sha256(teacher)
    optimizer = smoke.new_recovery_optimizer(model)
    device = next(model.parameters()).device
    steps, hashes = [], []
    for update in range(1, updates + 1):
        synchronize()
        began = perf_counter_ns()
        pixels, labels = batch(update)
        input_sha = smoke._batch_sha(pixels, labels)
        if expected_input_hashes is not None and input_sha != expected_input_hashes[update - 1]:
            raise ValueError("paired language input bytes differ before update")
        hashes.append(input_sha)
        pixels, labels = pixels.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        multiplier = recovery_multiplier(update)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier
        target = None
        if teacher is not None:
            with torch.no_grad():
                target = torch.cat(
                    [
                        teacher.encode(pixels[start : start + microbatch_size])
                        for start in range(0, len(pixels), microbatch_size)
                    ]
                )
        evidence = recomputed_language_backward(
            model,
            pixels,
            labels,
            text_gram=gram,
            teacher_descriptors=target,
            microbatch_size=microbatch_size,
        )
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 10.0, error_if_nonfinite=True, foreach=False
        )
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError("nonfinite language parameter after AdamW")
        synchronize()
        elapsed_ns = perf_counter_ns() - began
        if elapsed_ns <= 0:
            raise RuntimeError("language training clock did not advance")
        if device.type == "cuda" and torch.cuda.max_memory_reserved() >= 96 * 1024**3:
            raise RuntimeError("language CUDA memory limit")
        event = {
            "arm": arm,
            "update": update,
            "elapsed_ns": elapsed_ns,
            "input_sha256": input_sha,
            "loss": float(evidence.loss),
            "proxy_loss": float(evidence.proxy_loss),
            "relational_loss": float(evidence.relational_loss),
            "language_loss": float(evidence.language_loss),
            "gradient_norm": float(norm),
            "maximum_descriptor_disagreement": evidence.maximum_descriptor_disagreement,
            "lr_multiplier": multiplier,
        }
        steps.append(event)
        progress(event)
    if any(int(optimizer.state[p]["step"]) != updates for p in model.parameters()):
        raise RuntimeError("not every language parameter completed the fixed updates")
    final = control._model_state_sha256(model)
    if final == initial:
        raise RuntimeError("language training did not change model state")
    if teacher is not None and control._model_state_sha256(teacher) != teacher_initial:
        raise RuntimeError("language teacher state changed")
    optimizer.zero_grad(set_to_none=True)
    return {
        "arm": arm,
        "completed_updates": updates,
        "initial_state_sha256": initial,
        "final_state_sha256": final,
        "steps": steps,
        "input_sha256": hashes,
        "teacher_state_sha256": teacher_initial,
        "teacher_unchanged": True,
        "consumed_gram_sha256": consumed_gram_sha256,
    }


def measure_language_preflight(
    factory: Callable[[], PooledProxyAnchorModel],
    teacher: PooledProxyAnchorModel | None,
    batch: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    *,
    gram: torch.Tensor,
    microbatch_size: int,
    progress: Callable[[dict[str, Any]], None],
    synchronize: Callable[[], None],
) -> dict[str, Any]:
    """Measure3 base+3 language updates on disposable equal initializations."""
    arms: dict[str, Any] = {}
    initial: str | None = None
    inputs: list[str] | None = None
    seconds: list[float] = []
    for arm in ("base", "correct"):
        model = factory()
        observed = control._model_state_sha256(model)
        if initial is not None and initial != observed:
            raise ValueError("preflight initial model states differ")
        initial = observed
        evidence = _train_language_steps(
            model,
            teacher,
            batch,
            arm=arm,
            gram=None if arm == "base" else gram,
            expected_input_hashes=inputs,
            microbatch_size=microbatch_size,
            progress=progress,
            synchronize=synchronize,
            updates=3,
        )
        arms[arm] = evidence
        inputs = evidence["input_sha256"]
        seconds.extend(step["elapsed_ns"] / 1e9 for step in evidence["steps"])
        device = next(model.parameters()).device
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"arms": arms, "update_seconds": seconds, "quality_measured": False}


def execute_language_training(
    factory: Callable[[], PooledProxyAnchorModel],
    teacher: PooledProxyAnchorModel | None,
    batch: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    *,
    output_dir: Path,
    target_sha256: str,
    grams: dict[str, torch.Tensor],
    gram_sha256: dict[str, str],
    spent_seconds: float,
    microbatch_size: int,
    progress: Callable[[dict[str, Any]], None],
    synchronize: Callable[[], None],
) -> dict[str, Any]:
    """Time disposable copies, then train and seal three fresh matched arms."""
    if (
        set(grams) != {"correct", "permuted"}
        or set(gram_sha256) != set(grams)
        or any(
            gram.dtype != torch.float32
            or gram.ndim != 2
            or gram.shape[0] != gram.shape[1]
            or not bool(torch.isfinite(gram).all())
            or _gram_digest(gram) != gram_sha256[arm]
            for arm, gram in grams.items()
        )
        or gram_sha256["correct"] == gram_sha256["permuted"]
        or torch.equal(grams["correct"], grams["permuted"])
    ):
        raise ValueError("language controls must be finite, authenticated, and distinct")
    if output_dir.is_symlink() or not output_dir.is_dir() or any(output_dir.iterdir()):
        raise ValueError("language training output directory must be empty")
    teacher_initial = None if teacher is None else control._model_state_sha256(teacher)
    preflight = measure_language_preflight(
        factory,
        teacher,
        batch,
        gram=grams["correct"],
        microbatch_size=microbatch_size,
        progress=progress,
        synchronize=synchronize,
    )
    projection = pilot_training_projection(spent_seconds, preflight["update_seconds"])
    if projection > 7200:
        raise RuntimeError("language pilot projected wall cap")

    arms: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    initial_sha256: str | None = None
    input_sha256: list[str] | None = None
    for arm in ("base", "correct", "permuted"):
        model = factory()
        observed_initial = control._model_state_sha256(model)
        if initial_sha256 is not None and observed_initial != initial_sha256:
            raise ValueError("language scientific initial states differ")
        initial_sha256 = observed_initial
        gram = None if arm == "base" else grams[arm]
        evidence = train_language_arm(
            model,
            teacher,
            batch,
            arm=arm,
            gram=gram,
            expected_input_hashes=input_sha256,
            microbatch_size=microbatch_size,
            progress=progress,
            synchronize=synchronize,
        )
        if input_sha256 is None:
            input_sha256 = evidence["input_sha256"]
        checkpoints[arm] = seal_language_checkpoint(
            output_dir / f"{arm}-final.pt",
            model,
            evidence,
            target_sha256=target_sha256,
            expected_gram_sha256=None if arm == "base" else gram_sha256[arm],
        )
        arms[arm] = evidence
        device = next(model.parameters()).device
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if teacher is not None and control._model_state_sha256(teacher) != teacher_initial:
        raise RuntimeError("language teacher changed during training")
    return {
        "arms": arms,
        "checkpoints": checkpoints,
        "preflight": preflight,
        "projection_seconds": projection,
        "initial_state_sha256": initial_sha256,
        "input_sha256": input_sha256,
        "teacher_state_sha256": teacher_initial,
        "teacher_unchanged": True,
    }


def seal_language_checkpoint(
    path: Path,
    model: PooledProxyAnchorModel,
    evidence: dict[str, Any],
    *,
    target_sha256: str,
    expected_gram_sha256: str | None = None,
) -> dict[str, Any]:
    """Seal only a complete20-update state; scientific topology is checked by the runner."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    try:
        arm = evidence["arm"]
        consumed_gram = evidence["consumed_gram_sha256"]
        if (
            consumed_gram != expected_gram_sha256
            or (consumed_gram is None) != (arm == "base")
            or (
                consumed_gram is not None
                and (
                    type(consumed_gram) is not str
                    or len(consumed_gram) != 64
                    or any(c not in "0123456789abcdef" for c in consumed_gram)
                )
            )
        ):
            raise ValueError("language checkpoint consumed Gram authority differs")
        if (
            arm not in ("base", "correct", "permuted")
            or path.name != f"{arm}-final.pt"
            or type(evidence["completed_updates"]) is not int
            or evidence["completed_updates"] != 20
            or type(evidence["steps"]) is not list
            or len(evidence["steps"]) != 20
            or type(evidence["input_sha256"]) is not list
            or len(evidence["input_sha256"]) != 20
        ):
            raise ValueError("language checkpoint requires complete20-update evidence")
        digests = [
            target_sha256,
            evidence["initial_state_sha256"],
            evidence["final_state_sha256"],
            *evidence["input_sha256"],
        ]
        if any(
            type(h) is not str or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
            for h in digests
        ):
            raise ValueError("language checkpoint digest authority differs")
        for index, step in enumerate(evidence["steps"], 1):
            if (
                type(step["update"]) is not int
                or step["update"] != index
                or step["arm"] != arm
                or step["input_sha256"] != evidence["input_sha256"][index - 1]
                or type(step["elapsed_ns"]) is not int
                or step["elapsed_ns"] <= 0
                or step["lr_multiplier"] != recovery_multiplier(index)
                or any(
                    type(step[k]) is not float or not math.isfinite(step[k])
                    for k in (
                        "loss",
                        "proxy_loss",
                        "relational_loss",
                        "language_loss",
                        "gradient_norm",
                        "maximum_descriptor_disagreement",
                        "lr_multiplier",
                    )
                )
                or not 0 <= step["maximum_descriptor_disagreement"] <= 2e-5
                or step["gradient_norm"] < 0
            ):
                raise ValueError("language checkpoint update evidence differs")
        if control._model_state_sha256(model) != evidence["final_state_sha256"]:
            raise ValueError("language checkpoint final state differs")
        state = model.state_dict()
        if any(
            t.dtype != torch.float32 or not bool(torch.isfinite(t).all()) for t in state.values()
        ):
            raise ValueError("language checkpoint state must be finiteFP32")
    except (KeyError, TypeError) as error:
        raise ValueError("language checkpoint evidence incomplete") from error
    payload = {
        "schema": "sfora-siglip-language-final-v1",
        "claim_eligible": False,
        "seed": 17,
        "arm": arm,
        "completed_updates": 20,
        "target_sha256": target_sha256,
        "consumed_gram_sha256": consumed_gram,
        "initial_state_sha256": evidence["initial_state_sha256"],
        "final_state_sha256": evidence["final_state_sha256"],
        "input_sha256": evidence["input_sha256"],
        "input_dimensions": model.projection.in_features,
        "embedding_dimensions": model.projection.out_features,
        "class_count": model.class_count,
        "model_state": {k: v.detach().cpu() for k, v in state.items()},
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    control._fsync_directory(path.parent)
    digest, size = control._sha256_file(path)
    return {
        "basename": path.name,
        "sha256": digest,
        "bytes": size,
        "arm": arm,
        "completed_updates": 20,
    }
