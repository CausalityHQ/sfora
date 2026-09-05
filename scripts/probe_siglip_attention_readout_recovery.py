"""Local-only SigLIP attention-readout recovery probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

_DEPTHS = (6, 10, 14, 18, 22, 25, 27)
_DECISIVE_DEPTH = 18
_BATCH_SIZE = 32
_CUDA_MEMORY_CAP_BYTES = 96 * 1024**3
_EVALUATION_LABELS = frozenset(range(49, 82))
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _enforce_cuda_memory_cap(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.memory_reserved(device) > _CUDA_MEMORY_CAP_BYTES:
        raise RuntimeError("attention readout CUDA memory cap exceeded")


def _image_basename(example_id: str) -> str:
    if type(example_id) is not str or not example_id:
        raise ValueError("attention readout evaluation example identity differs")
    digest = hashlib.sha256(
        b"rsta-siglip-a-v1|image-path|\0" + example_id.encode("utf-8")
    ).hexdigest()
    return f"{digest}.image"


def _read_regular(path: Path, *, role: str) -> bytes:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{role} must be one regular file") from error
    if resolved != path or path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} must be one regular file")
    return path.read_bytes()


def load_local_evaluation_manifest(
    path: Path,
    expected_sha256: str,
    image_root: Path,
    *,
    dataset_id: str,
    dataset_revision: str,
    expected_count: int = 2_746,
    expected_labels: frozenset[int] = _EVALUATION_LABELS,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[Path, ...]]:
    """Authenticate a complete local evaluation manifest and flat pixel namespace."""

    raw = _read_regular(path, role="attention readout evaluation manifest")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("attention readout evaluation manifest digest differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("attention readout evaluation manifest is not JSON") from error
    if (
        type(value) is not dict
        or set(value) != {"schema", "claim_eligible", "dataset_id", "dataset_revision", "examples"}
        or raw != _canonical_json(value)
        or value["schema"] != "sfora-attention-readout-evaluation-v1"
        or value["claim_eligible"] is not False
        or value["dataset_id"] != dataset_id
        or value["dataset_revision"] != dataset_revision
        or type(value["examples"]) is not list
        or len(value["examples"]) != expected_count
        or type(expected_labels) is not frozenset
        or not expected_labels
    ):
        raise ValueError("attention readout evaluation manifest authority differs")
    try:
        resolved_root = image_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("attention readout evaluation root differs") from error
    if resolved_root != image_root or image_root.is_symlink() or not image_root.is_dir():
        raise ValueError("attention readout evaluation root differs")
    ids: list[str] = []
    labels: list[int] = []
    paths: list[Path] = []
    expected_names: set[str] = set()
    for row in value["examples"]:
        if (
            type(row) is not dict
            or set(row) != {"example_id", "label"}
            or type(row["example_id"]) is not str
            or not row["example_id"]
            or type(row["label"]) is not int
            or row["label"] not in expected_labels
        ):
            raise ValueError("attention readout evaluation row differs")
        basename = _image_basename(row["example_id"])
        image = image_root / basename
        try:
            resolved_image = image.resolve(strict=True)
        except OSError as error:
            raise ValueError("attention readout evaluation image differs") from error
        if (
            resolved_image != image
            or image.is_symlink()
            or not image.is_file()
            or not resolved_image.is_relative_to(resolved_root)
        ):
            raise ValueError("attention readout evaluation image differs")
        ids.append(row["example_id"])
        labels.append(row["label"])
        paths.append(image)
        expected_names.add(basename)
    if (
        tuple(sorted(ids)) != tuple(ids)
        or len(set(ids)) != expected_count
        or set(labels) != set(expected_labels)
        or {entry.name for entry in image_root.iterdir()} != expected_names
    ):
        raise ValueError("attention readout evaluation namespace differs")
    return tuple(ids), tuple(labels), tuple(paths)


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be 64 lowercase hexadecimal characters")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the strict local-only attention-readout command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-binding", type=_absolute_path, required=True)
    parser.add_argument("--control-binding-sha256", type=_lower_sha256, required=True)
    parser.add_argument("--checkpoint-seed17", type=_absolute_path, required=True)
    parser.add_argument("--optimization-manifest", type=_absolute_path, required=True)
    parser.add_argument("--optimization-manifest-sha256", type=_lower_sha256, required=True)
    parser.add_argument("--optimization-image-root", type=_absolute_path, required=True)
    parser.add_argument("--evaluation-manifest", type=_absolute_path, required=True)
    parser.add_argument("--evaluation-manifest-sha256", type=_lower_sha256, required=True)
    parser.add_argument("--evaluation-image-root", type=_absolute_path, required=True)
    parser.add_argument("--readout-artifact", type=_absolute_path, required=True)
    parser.add_argument("--result", type=_absolute_path, required=True)
    parser.add_argument("--execute-attention-readout", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


@dataclass(frozen=True, slots=True)
class StreamedAttentionReadoutInputs:
    """CPU-resident control planes, decisive tokens, and teacher targets."""

    control_planes: tuple[torch.Tensor, ...]
    decisive_tokens: torch.Tensor
    teacher_outputs: torch.Tensor
    teacher_targets: torch.Tensor
    cache_bytes: int


class LearnedAttentionReadout(nn.Module):
    """Teacher-initialized LayerNorm, MAP head, and projection."""

    def __init__(self, layernorm: nn.Module, head: nn.Module, projection: nn.Linear) -> None:
        super().__init__()
        self.layernorm = copy.deepcopy(layernorm)
        self.head = copy.deepcopy(head)
        self.projection = copy.deepcopy(projection)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = self.head(self.layernorm(tokens))
        return F.normalize(self.projection(pooled), dim=1)


@dataclass(frozen=True, slots=True)
class LearnedAttentionReadoutEvidence:
    """Final-only learned attention seal and its fixed-budget losses."""

    readout: LearnedAttentionReadout
    initial_loss: float
    final_loss: float
    final_200_losses: tuple[float, ...]


def _attention_loss(
    readout: LearnedAttentionReadout,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return (1.0 - torch.sum(readout(tokens) * targets, dim=1)).mean()


def _mean_attention_loss(
    readout: LearnedAttentionReadout,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    weighted_losses = []
    for start in range(0, tokens.shape[0], 256):
        stop = min(start + 256, tokens.shape[0])
        weighted_losses.append(
            float(_attention_loss(readout, tokens[start:stop], targets[start:stop]))
            * (stop - start)
        )
    return math.fsum(weighted_losses) / tokens.shape[0]


def fit_learned_attention_readout(
    tokens: torch.Tensor,
    targets: torch.Tensor,
    layernorm: nn.Module,
    head: nn.Module,
    projection: nn.Linear,
    *,
    device: torch.device | None = None,
) -> LearnedAttentionReadoutEvidence:
    """Fit only the teacher-initialized attention readout for 2,000 fixed updates."""

    if device is None:
        device = torch.device("cpu")
    if (
        type(tokens) is not torch.Tensor
        or tokens.device.type != "cpu"
        or tokens.dtype != torch.float16
        or tokens.ndim != 3
        or tokens.shape[0] < 2
        or tokens.shape[1] < 1
        or not tokens.is_contiguous()
        or not bool(torch.isfinite(tokens).all())
        or type(targets) is not torch.Tensor
        or targets.device.type != "cpu"
        or targets.dtype != torch.float32
        or targets.ndim != 2
        or targets.shape != (tokens.shape[0], projection.out_features)
        or not targets.is_contiguous()
        or not bool(torch.isfinite(targets).all())
        or not torch.allclose(
            torch.linalg.vector_norm(targets, dim=1),
            torch.ones(targets.shape[0]),
            atol=1.0e-6,
            rtol=0.0,
        )
        or not isinstance(layernorm, nn.Module)
        or layernorm.training
        or not isinstance(head, nn.Module)
        or head.training
        or not isinstance(projection, nn.Linear)
        or projection.training
        or projection.bias is not None
        or projection.in_features != tokens.shape[2]
        or type(device) is not torch.device
        or device.type not in {"cpu", "cuda"}
    ):
        raise ValueError("learned attention authority differs")
    readout = LearnedAttentionReadout(layernorm, head, projection).to(device).train()
    parameters = tuple(readout.parameters())
    if not parameters:
        raise ValueError("learned attention authority differs")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=1.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
        foreach=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    pending = torch.empty(0, dtype=torch.int64)
    device_tokens = tokens.to(device=device, dtype=torch.float32)
    device_targets = targets.to(device)
    _enforce_cuda_memory_cap(device)
    with torch.no_grad():
        initial_loss = _mean_attention_loss(readout, device_tokens, device_targets)
    losses: list[float] = []
    for update in range(1, 2_001):
        while pending.numel() < 256:
            pending = torch.cat((pending, torch.randperm(tokens.shape[0], generator=generator)))
        indexes, pending = pending[:256], pending[256:]
        if update <= 50:
            learning_rate = 1.0e-4 * update / 50.0
        else:
            progress = (update - 50) / 1_950
            learning_rate = 1.0e-5 + 0.5 * (1.0e-4 - 1.0e-5) * (1.0 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss = _attention_loss(
            readout,
            device_tokens[indexes.to(device)],
            device_targets[indexes.to(device)],
        )
        if not bool(torch.isfinite(loss)):
            raise ValueError("learned attention loss is nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        _enforce_cuda_memory_cap(device)
        if update > 1_800:
            losses.append(float(loss.detach()))
    with torch.no_grad():
        final_loss = _mean_attention_loss(readout, device_tokens, device_targets)
    if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
        raise ValueError("learned attention loss is nonfinite")
    return LearnedAttentionReadoutEvidence(
        readout=readout.eval().cpu(),
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_200_losses=tuple(losses),
    )


def apply_learned_attention_readout(
    readout: LearnedAttentionReadout,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Apply one sealed learned readout in bounded batches."""

    if (
        type(readout) is not LearnedAttentionReadout
        or readout.training
        or type(tokens) is not torch.Tensor
        or tokens.device.type != "cpu"
        or tokens.dtype != torch.float16
        or tokens.ndim != 3
        or tokens.shape[0] < 2
        or tokens.shape[2] != readout.projection.in_features
        or not tokens.is_contiguous()
        or not bool(torch.isfinite(tokens).all())
        or type(device) is not torch.device
        or device.type not in {"cpu", "cuda"}
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise ValueError("learned attention application authority differs")
    model = readout.to(device)
    batches = []
    with torch.inference_mode():
        for start in range(0, tokens.shape[0], batch_size):
            descriptors = model(tokens[start : start + batch_size].to(device).float())
            if not bool(torch.isfinite(descriptors).all()):
                raise ValueError("learned attention descriptor is nonfinite")
            batches.append(descriptors.float().cpu())
            _enforce_cuda_memory_cap(device)
    return torch.cat(batches).contiguous()


def _module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256(b"sfora-attention-readout-state-v1\0")
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(len(value.shape).to_bytes(4, "big"))
        for dimension in value.shape:
            digest.update(dimension.to_bytes(8, "big"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _weight_sha256(weight: torch.Tensor) -> str:
    holder = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    holder.weight.data.copy_(weight)
    return _module_sha256(holder)


def _write_readout_artifact(
    path: Path,
    linear_cells: list[tuple[int, str, torch.Tensor, bool]],
    learned: LearnedAttentionReadout,
) -> str:
    """Persist and reload every sealed readout before evaluation becomes accessible."""

    from safetensors.torch import load_file, save_file

    if not isinstance(path, Path):
        raise ValueError("attention readout artifact authority differs")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if (
        type(linear_cells) is not list
        or not linear_cells
        or type(learned) is not LearnedAttentionReadout
        or learned.training
        or any(parameter.device.type != "cpu" for parameter in learned.parameters())
    ):
        raise ValueError("attention readout artifact authority differs")
    tensors: dict[str, torch.Tensor] = {}
    for depth, fit, weight, optimization_limited in linear_cells:
        if (
            type(depth) is not int
            or type(fit) is not str
            or fit not in {"ridge", "refined"}
            or type(weight) is not torch.Tensor
            or weight.device.type != "cpu"
            or weight.dtype != torch.float32
            or weight.ndim != 2
            or not weight.is_contiguous()
            or not bool(torch.isfinite(weight).all())
            or type(optimization_limited) is not bool
        ):
            raise ValueError("attention readout artifact authority differs")
        key = f"linear.depth-{depth:02d}.{fit}.weight"
        if key in tensors:
            raise ValueError("attention readout artifact authority differs")
        tensors[key] = weight.detach().clone()
    for name, value in sorted(learned.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError("attention readout artifact authority differs")
        tensors[f"learned.{name}"] = tensor
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        dict(sorted(tensors.items())),
        str(partial),
        metadata={"schema": "sfora-siglip-attention-readouts-v1"},
    )
    restored = load_file(str(partial))
    if set(restored) != set(tensors) or any(
        not torch.equal(restored[name], value) for name, value in tensors.items()
    ):
        partial.unlink(missing_ok=True)
        raise ValueError("attention readout artifact replay differs")
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    partial.replace(path)
    return digest


def _retrieval_payload(evidence: object) -> dict[str, object]:
    correct = getattr(evidence, "correct", None)
    average_precisions = getattr(evidence, "average_precisions", None)
    if type(correct) is not tuple or type(average_precisions) is not tuple:
        raise ValueError("attention readout retrieval evidence differs")
    return {
        "hits": list(correct),
        "average_precision": list(average_precisions),
    }


def _optimization_limited(losses: tuple[float, ...]) -> bool:
    return len(losses) == 200 and math.fsum(losses[:20]) / 20 > math.fsum(losses[-20:]) / 20


def main(argv: list[str] | None = None) -> int:
    """Fit locally, seal every cell, then acquire and score evaluation pixels once."""

    from diagnose_siglip_rsta_stage_a import (
        _load_model_state_checkpoint,
        _load_optimization_manifest,
        _parse_control_binding,
        _stage_a_transforms,
        configure_stage_a_determinism,
        load_stage_a_checkpoint_model,
        load_stage_a_siglip_runtime,
    )
    from diagnose_siglip_rsta_stage_a import (
        _read_regular as read_rsta_regular,
    )
    from PIL import Image

    from sfora.siglip_attention_readout_recovery import (
        apply_readout,
        attention_retrieval_evidence,
        build_attention_readout_result,
        fit_ridge_readout,
        refine_directional_readout,
    )

    arguments = parse_args(argv)
    for output in (arguments.result, arguments.readout_artifact):
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    binding_raw = read_rsta_regular(arguments.control_binding, role="attention control binding")
    if hashlib.sha256(binding_raw).hexdigest() != arguments.control_binding_sha256:
        raise ValueError("attention control binding digest differs")
    binding = _parse_control_binding(binding_raw)
    if (
        binding.control_complete is not True
        or binding.optimization_manifest_sha256 != arguments.optimization_manifest_sha256
        or tuple(checkpoint.seed for checkpoint in binding.checkpoints) != (17, 29, 43)
    ):
        raise ValueError("attention control authority differs")
    seed17 = binding.checkpoints[0]
    checkpoint = _load_model_state_checkpoint(arguments.checkpoint_seed17, seed17, binding)
    _optimization_ids, _optimization_labels, optimization_paths = _load_optimization_manifest(
        arguments.optimization_manifest,
        arguments.optimization_manifest_sha256,
        binding,
        arguments.optimization_image_root,
    )
    configure_stage_a_determinism()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("attention readout requires CUDA bf16")
    device = torch.device("cuda")
    runtime = load_stage_a_siglip_runtime()
    model = load_stage_a_checkpoint_model(
        checkpoint,
        model_factory=runtime.model_factory,
        device=device,
    )
    runtime.disable_checkpointing(model)
    model.eval()
    vision_model = model.tower.vision_model
    projection = model.projection
    _graph_transform, evaluation_transform = _stage_a_transforms(runtime.processor)

    def pixel_batches(paths: tuple[Path, ...]) -> Iterable[torch.Tensor]:
        for start in range(0, len(paths), _BATCH_SIZE):
            tensors = []
            for path in paths[start : start + _BATCH_SIZE]:
                with Image.open(path) as image:
                    tensor = evaluation_transform(image)
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError("attention image transform differs")
                tensors.append(tensor)
            yield torch.stack(tensors)

    optimization = stream_attention_readout_inputs(
        vision_model,
        projection,
        pixel_batches(optimization_paths),
        depths=_DEPTHS,
        decisive_depth=_DECISIVE_DEPTH,
        device=device,
    )
    teacher_weight = projection.weight.detach().float().cpu().contiguous()
    linear_cells: list[tuple[int, str, torch.Tensor, bool]] = []
    for depth, features in zip(_DEPTHS, optimization.control_planes, strict=True):
        ridge = fit_ridge_readout(features, optimization.teacher_outputs, teacher_weight)
        linear_cells.append((depth, "ridge", ridge, False))
        refined = refine_directional_readout(features, optimization.teacher_targets, ridge)
        linear_cells.append(
            (
                depth,
                "refined",
                refined.weight,
                _optimization_limited(refined.final_200_losses),
            )
        )
    learned = fit_learned_attention_readout(
        optimization.decisive_tokens,
        optimization.teacher_targets,
        vision_model.post_layernorm,
        vision_model.head,
        projection,
        device=device,
    )
    learned_sha256 = _module_sha256(learned.readout)
    readout_artifact_sha256 = _write_readout_artifact(
        arguments.readout_artifact,
        linear_cells,
        learned.readout,
    )

    evaluation_ids, evaluation_labels, evaluation_paths = load_local_evaluation_manifest(
        arguments.evaluation_manifest,
        arguments.evaluation_manifest_sha256,
        arguments.evaluation_image_root,
        dataset_id=binding.dataset_id,
        dataset_revision=binding.dataset_revision,
    )
    evaluation = stream_attention_readout_inputs(
        vision_model,
        projection,
        pixel_batches(evaluation_paths),
        depths=_DEPTHS,
        decisive_depth=_DECISIVE_DEPTH,
        device=device,
    )
    teacher = evaluation.teacher_targets
    teacher_retrieval = attention_retrieval_evidence(
        teacher,
        teacher,
        query_ids=evaluation_ids,
        gallery_ids=evaluation_ids,
        query_labels=evaluation_labels,
        gallery_labels=evaluation_labels,
    )
    if sum(teacher_retrieval.correct) != 2_596 or teacher_retrieval.map_at_r != 0.7913744556922272:
        raise ValueError("attention readout teacher reproduction differs")
    controls_by_depth = dict(zip(_DEPTHS, evaluation.control_planes, strict=True))
    cells: list[dict[str, object]] = []
    for depth, fit, weight, optimization_limited in linear_cells:
        descriptors = apply_readout(controls_by_depth[depth], weight)
        self_evidence = attention_retrieval_evidence(
            descriptors,
            descriptors,
            query_ids=evaluation_ids,
            gallery_ids=evaluation_ids,
            query_labels=evaluation_labels,
            gallery_labels=evaluation_labels,
        )
        cross_evidence = attention_retrieval_evidence(
            descriptors,
            teacher,
            query_ids=evaluation_ids,
            gallery_ids=evaluation_ids,
            query_labels=evaluation_labels,
            gallery_labels=evaluation_labels,
        )
        cells.append(
            {
                "depth": depth,
                "fit": fit,
                "optimization_limited": optimization_limited,
                "weight_sha256": _weight_sha256(weight),
                "self": _retrieval_payload(self_evidence),
                "cross": _retrieval_payload(cross_evidence),
            }
        )
        if depth == _DECISIVE_DEPTH and fit == "refined":
            descriptors = apply_learned_attention_readout(
                learned.readout,
                evaluation.decisive_tokens,
                device=device,
            )
            cells.append(
                {
                    "depth": _DECISIVE_DEPTH,
                    "fit": "learned-attention",
                    "optimization_limited": _optimization_limited(learned.final_200_losses),
                    "weight_sha256": learned_sha256,
                    "self": _retrieval_payload(
                        attention_retrieval_evidence(
                            descriptors,
                            descriptors,
                            query_ids=evaluation_ids,
                            gallery_ids=evaluation_ids,
                            query_labels=evaluation_labels,
                            gallery_labels=evaluation_labels,
                        )
                    ),
                    "cross": _retrieval_payload(
                        attention_retrieval_evidence(
                            descriptors,
                            teacher,
                            query_ids=evaluation_ids,
                            gallery_ids=evaluation_ids,
                            query_labels=evaluation_labels,
                            gallery_labels=evaluation_labels,
                        )
                    ),
                }
            )
    raw = build_attention_readout_result(
        cells,
        checkpoint_sha256=seed17.sha256,
        optimization_manifest_sha256=arguments.optimization_manifest_sha256,
        evaluation_manifest_sha256=arguments.evaluation_manifest_sha256,
        readout_artifact_sha256=readout_artifact_sha256,
        depth_27_identity=True,
        learned_attention_optimization={
            "initial_loss": learned.initial_loss,
            "final_loss": learned.final_loss,
            "final_200_losses": list(learned.final_200_losses),
        },
    )
    value = json.loads(raw)
    if not isinstance(value, Mapping) or value.get("claim_eligible") is not False:
        raise ValueError("attention readout result binding differs")
    partial = arguments.result.with_name(f"{arguments.result.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(raw)
    partial.replace(arguments.result)
    return 0


def stream_attention_readout_inputs(
    vision_model: nn.Module,
    projection: nn.Linear,
    pixel_batches: Iterable[torch.Tensor],
    *,
    depths: tuple[int, ...],
    decisive_depth: int,
    device: torch.device,
) -> StreamedAttentionReadoutInputs:
    """Run the frozen teacher once per batch and retain exact registered planes."""

    post_layernorm = getattr(vision_model, "post_layernorm", None)
    head = getattr(vision_model, "head", None)
    if (
        not isinstance(vision_model, nn.Module)
        or vision_model.training
        or not isinstance(post_layernorm, nn.Module)
        or post_layernorm.training
        or not isinstance(head, nn.Module)
        or head.training
        or not isinstance(projection, nn.Linear)
        or projection.training
        or projection.bias is not None
        or type(device) is not torch.device
        or device.type not in {"cpu", "cuda"}
    ):
        raise ValueError("attention readout model authority differs")
    if (
        type(depths) is not tuple
        or not depths
        or any(type(depth) is not int or depth < 1 for depth in depths)
        or tuple(sorted(set(depths))) != depths
        or type(decisive_depth) is not int
        or decisive_depth not in depths
    ):
        raise ValueError("attention readout depth authority differs")

    control_batches: tuple[list[torch.Tensor], ...] = tuple([] for _ in depths)
    decisive_batches: list[torch.Tensor] = []
    output_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for pixels in pixel_batches:
            if (
                type(pixels) is not torch.Tensor
                or pixels.ndim != 4
                or pixels.shape[0] < 1
                or not pixels.is_floating_point()
                or not bool(torch.isfinite(pixels).all())
            ):
                raise ValueError("attention readout pixel authority differs")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = vision_model(
                    pixel_values=pixels.to(device),
                    output_hidden_states=True,
                    return_dict=True,
                )
                _enforce_cuda_memory_cap(device)
                if (
                    type(getattr(output, "hidden_states", None)) not in {tuple, list}
                    or len(output.hidden_states) <= depths[-1]
                    or type(getattr(output, "pooler_output", None)) is not torch.Tensor
                    or output.pooler_output.ndim != 2
                    or output.pooler_output.shape[0] != pixels.shape[0]
                    or output.pooler_output.shape[1] != projection.in_features
                    or not bool(torch.isfinite(output.pooler_output).all())
                ):
                    raise ValueError("attention readout hidden-state authority differs")
                hidden_states = output.hidden_states
                pooler_output = output.pooler_output
                for ordinal, depth in enumerate(depths):
                    hidden = hidden_states[depth]
                    if (
                        type(hidden) is not torch.Tensor
                        or hidden.ndim != 3
                        or hidden.shape[0] != pixels.shape[0]
                        or hidden.shape[2] != projection.in_features
                        or not bool(torch.isfinite(hidden).all())
                    ):
                        raise ValueError("attention readout hidden-state authority differs")
                    pooled = head(post_layernorm(hidden))
                    if (
                        type(pooled) is not torch.Tensor
                        or pooled.shape != pooler_output.shape
                        or not bool(torch.isfinite(pooled).all())
                    ):
                        raise ValueError("attention readout pooled authority differs")
                    control_batches[ordinal].append(pooled.float().cpu().contiguous())
                    if depth == decisive_depth:
                        decisive_batches.append(hidden.half().cpu().contiguous())
                    if depth == depths[-1] == len(hidden_states) - 1 and not torch.equal(
                        pooled.float(), pooler_output.float()
                    ):
                        raise ValueError("attention readout final-depth identity differs")
            teacher_outputs = projection(pooler_output.float()).float()
            teacher_targets = F.normalize(teacher_outputs, dim=1)
            if not bool(torch.isfinite(teacher_targets).all()):
                raise ValueError("attention readout teacher target authority differs")
            output_batches.append(teacher_outputs.cpu().contiguous())
            target_batches.append(teacher_targets.cpu().contiguous())
    if not output_batches or any(not values for values in control_batches) or not decisive_batches:
        raise ValueError("attention readout stream is empty")
    controls = tuple(torch.cat(values).contiguous() for values in control_batches)
    decisive_tokens = torch.cat(decisive_batches).contiguous()
    teacher_outputs = torch.cat(output_batches).contiguous()
    teacher_targets = torch.cat(target_batches).contiguous()
    cached = (*controls, decisive_tokens, teacher_outputs, teacher_targets)
    cache_bytes = sum(value.numel() * value.element_size() for value in cached)
    return StreamedAttentionReadoutInputs(
        control_planes=controls,
        decisive_tokens=decisive_tokens,
        teacher_outputs=teacher_outputs,
        teacher_targets=teacher_targets,
        cache_bytes=cache_bytes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
