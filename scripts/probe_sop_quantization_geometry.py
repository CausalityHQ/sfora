#!/usr/bin/env python3
"""Select an int4 output geometry using only the already-burned SOP split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

import torch
from probe_cub_relational_int4 import (
    fixed_random_rotation,
    score_int4_symmetric,
    verify_source_commit,
)
from probe_inshop_relational_linear import (
    _encode_floating,
    _familywise_lower_bound,
    _summary,
)
from probe_sop_relational_linear import (
    PairedArchives,
    _lexicographic_candidates,
    load_paired_archives,
    score_symmetric,
)

from sfora.joint_relational_compaction import (
    RelationalLinearEncoder,
    RelationalLinearTrainingConfig,
    _fit_uncentered_covariance_basis,
    fit_relational_linear_encoder,
)
from sfora.packed_int4 import PackedInt4Embeddings, pack_int4_unit_embeddings

DIMENSIONS = 128
BATCH = 1024
EPOCHS = 20
UPDATES = 1_000
LEARNING_RATE = 1e-4
TEMPERATURE = 0.05
WEIGHT_DECAY = 1e-4
SEED = 17
CANDIDATE_WIDTH = 256
CLIPPING_RATIOS = (0.6, 0.7, 0.8, 0.9, 1.0)
CLIPPING_PAIR_SAMPLES = 131_072
ASYMMETRIC_GAIN_GATE = 0.003
ROTATION_RECOVERY_FRACTION = 0.5
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_COMPARISONS = 2
SEALED_PCA_FLOAT_MAP_AT_R = 0.39334904153199973
SEALED_PCA_TOLERANCE = 1e-6


class GeometryScore(TypedDict):
    """Exact aggregate and per-query symmetric retrieval evidence."""

    map_at_r: float
    per_query_ap: tuple[float, ...]
    per_query_r1: tuple[float, ...]
    r1: float


def _unit_rows(value: torch.Tensor) -> bool:
    if not bool(torch.isfinite(value).all()):
        return False
    norms = torch.linalg.vector_norm(value.detach().double(), dim=1)
    return bool((torch.abs(norms - 1.0) <= 2e-5).all())


def pack_clipped_int4_unit_embeddings(
    value: torch.Tensor, *, scale_ratio: float
) -> PackedInt4Embeddings:
    """Pack unit rows with a fixed clipped fraction of each row maximum."""

    if (
        type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] < 2
        or value.shape[1] % 2 != 0
        or not value.is_contiguous()
        or not _unit_rows(value)
        or type(scale_ratio) is not float
        or not math.isfinite(scale_ratio)
        or not 0.0 < scale_ratio <= 1.0
    ):
        raise ValueError("SOP int4 clipping authority differs")
    if scale_ratio == 1.0:
        return pack_int4_unit_embeddings(value)
    scale = torch.amax(torch.abs(value), dim=1, keepdim=True) * scale_ratio
    codes = torch.round(value / scale * 7.0).clamp(-7, 7).to(torch.int8).contiguous()
    low = codes[:, 0::2].to(torch.int16) & 0x0F
    high = (codes[:, 1::2].to(torch.int16) & 0x0F) << 4
    packed_codes = (low | high).to(torch.uint8).contiguous()
    inverse_norms = (
        torch.linalg.vector_norm(codes.float(), dim=1).reciprocal().to(torch.float16).contiguous()
    )
    return PackedInt4Embeddings(
        packed_codes=packed_codes,
        inverse_norms=inverse_norms,
        dimensions=value.shape[1],
    )


def select_clipping_ratio(
    train_embeddings: torch.Tensor,
    *,
    ratios: tuple[float, ...] = CLIPPING_RATIOS,
) -> tuple[float, dict[str, float]]:
    """Choose the fixed ratio by label-free train pairwise Gram fidelity."""

    if (
        type(ratios) is not tuple
        or not ratios
        or len(set(ratios)) != len(ratios)
        or any(type(value) is not float or not 0.0 < value <= 1.0 for value in ratios)
    ):
        raise ValueError("SOP int4 clipping ratio authority differs")
    generator = torch.Generator().manual_seed(SEED)
    pairs = torch.randint(
        train_embeddings.shape[0],
        (2, CLIPPING_PAIR_SAMPLES),
        generator=generator,
    )
    reference = torch.sum(train_embeddings[pairs[0]] * train_embeddings[pairs[1]], dim=1)
    evidence: dict[str, float] = {}
    for ratio in ratios:
        restored = pack_clipped_int4_unit_embeddings(train_embeddings, scale_ratio=ratio).restore()
        candidate = torch.sum(restored[pairs[0]] * restored[pairs[1]], dim=1)
        evidence[str(ratio)] = float(torch.mean(torch.square(candidate - reference)).item())
    winner = min(ratios, key=lambda value: (evidence[str(value)], value))
    return winner, evidence


def score_asymmetric_int4(
    queries: torch.Tensor,
    gallery: PackedInt4Embeddings,
    labels: tuple[int, ...],
    *,
    candidate_width: int,
    device: torch.device,
) -> GeometryScore:
    """Score float queries against a persistent int4 gallery, excluding self."""

    row_count = gallery.packed_codes.shape[0]
    if (
        type(queries) is not torch.Tensor
        or queries.device.type != "cpu"
        or queries.dtype != torch.float32
        or queries.shape != (row_count, gallery.dimensions)
        or not queries.is_contiguous()
        or not _unit_rows(queries)
        or type(labels) is not tuple
        or len(labels) != row_count
        or any(type(label) is not int or label < 1 for label in labels)
        or type(candidate_width) is not int
        or not 0 < candidate_width < row_count
        or type(device) is not torch.device
    ):
        raise ValueError("SOP asymmetric int4 authority differs")
    counts = Counter(labels)
    if any(count < 2 or count - 1 > candidate_width for count in counts.values()):
        raise ValueError("SOP asymmetric class authority differs")
    integer_gallery = gallery.signed_codes().to(device=device, dtype=torch.float32)
    inverse_norms = gallery.inverse_norms.to(device=device, dtype=torch.float32).unsqueeze(0)
    rankings: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, row_count, 256):
            stop = min(start + 256, row_count)
            scores = (queries[start:stop].to(device) @ integer_gallery.T) * inverse_norms
            scores[
                torch.arange(stop - start, device=device),
                torch.arange(start, stop, device=device),
            ] = -torch.inf
            rankings.append(_lexicographic_candidates(scores, candidate_width).cpu())
    ranked = torch.cat(rankings)
    aps: list[float] = []
    r1: list[float] = []
    for ranking, label in zip(ranked.tolist(), labels, strict=True):
        positive_count = counts[label] - 1
        found = 0
        terms: list[float] = []
        for rank, index in enumerate(ranking[:positive_count], start=1):
            if labels[index] == label:
                found += 1
                terms.append(found / rank)
        aps.append(math.fsum(terms) / positive_count)
        r1.append(float(labels[ranking[0]] == label))
    return GeometryScore(
        map_at_r=math.fsum(aps) / row_count,
        per_query_ap=tuple(aps),
        per_query_r1=tuple(r1),
        r1=math.fsum(r1) / row_count,
    )


def fold_output_rotation(
    model: RelationalLinearEncoder, rotation: torch.Tensor
) -> RelationalLinearEncoder:
    """Fold one orthogonal output rotation into the linear projection."""

    if type(model) is not RelationalLinearEncoder:
        raise ValueError("SOP folded rotation authority differs")
    weight = model.projection.weight.detach().cpu().contiguous()
    if (
        type(rotation) is not torch.Tensor
        or rotation.device.type != "cpu"
        or rotation.dtype != torch.float32
        or rotation.shape != (weight.shape[0], weight.shape[0])
        or not rotation.is_contiguous()
        or not bool(torch.isfinite(rotation).all())
        or not torch.allclose(
            rotation @ rotation.T,
            torch.eye(rotation.shape[0], dtype=torch.float32),
            atol=2e-5,
            rtol=2e-5,
        )
    ):
        raise ValueError("SOP folded rotation authority differs")
    return RelationalLinearEncoder((rotation @ weight).contiguous()).eval()


def quantization_decision(
    *,
    baseline_symmetric_map_at_r: float,
    baseline_asymmetric_map_at_r: float,
    rotated_clipped_asymmetric_map_at_r: float,
    asymmetric_map_lower_bound: float,
    composed_map_lower_bound: float,
    pca_float_to_int4_loss: float,
    rotated_recovery: float,
) -> dict[str, object]:
    """Apply the preregistered sequential burned-SOP decision gates."""

    if any(
        type(value) is not float or not math.isfinite(value)
        for value in (
            baseline_symmetric_map_at_r,
            baseline_asymmetric_map_at_r,
            rotated_clipped_asymmetric_map_at_r,
            asymmetric_map_lower_bound,
            composed_map_lower_bound,
            pca_float_to_int4_loss,
            rotated_recovery,
        )
    ):
        raise ValueError("SOP quantization decision authority differs")
    asymmetric_gain = baseline_asymmetric_map_at_r - baseline_symmetric_map_at_r
    composed_gain = rotated_clipped_asymmetric_map_at_r - baseline_asymmetric_map_at_r
    asymmetric = asymmetric_gain >= ASYMMETRIC_GAIN_GATE and asymmetric_map_lower_bound > 0.0
    rotation_clipping = (
        asymmetric
        and pca_float_to_int4_loss > 0.0
        and rotated_recovery >= ROTATION_RECOVERY_FRACTION * pca_float_to_int4_loss
        and composed_gain >= 0.0
        and composed_map_lower_bound > 0.0
    )
    if rotation_clipping:
        winner = "rotated-clipped-asymmetric-int4"
    elif asymmetric:
        winner = "asymmetric-int4"
    else:
        winner = "baseline-symmetric-int4"
    return {
        "asymmetric": asymmetric,
        "asymmetric_gain": asymmetric_gain,
        "asymmetric_map_lower_bound": asymmetric_map_lower_bound,
        "composed_gain": composed_gain,
        "composed_map_lower_bound": composed_map_lower_bound,
        "rotation_clipping": rotation_clipping,
        "winner": winner,
    }


def validate_sealed_pca_reproduction(*, float_map_at_r: float) -> None:
    """Require the frozen probe to reproduce the prior burned-SOP controls."""

    if (
        type(float_map_at_r) is not float
        or not math.isfinite(float_map_at_r)
        or abs(float_map_at_r - SEALED_PCA_FLOAT_MAP_AT_R) > SEALED_PCA_TOLERANCE
    ):
        raise ValueError("SOP sealed PCA reproduction differs")


def _score_family(
    train: torch.Tensor,
    test: torch.Tensor,
    rotated_train: torch.Tensor,
    rotated_test: torch.Tensor,
    labels: tuple[int, ...],
    *,
    device: torch.device,
    factorial: bool,
) -> dict[str, object]:
    ratio, ratio_evidence = select_clipping_ratio(train)
    variants = {
        "baseline": pack_clipped_int4_unit_embeddings(test, scale_ratio=1.0),
        "rotated_clipped": pack_clipped_int4_unit_embeddings(rotated_test, scale_ratio=ratio),
    }
    if factorial:
        variants["clipped_only"] = pack_clipped_int4_unit_embeddings(test, scale_ratio=ratio)
        variants["rotation_only"] = pack_clipped_int4_unit_embeddings(rotated_test, scale_ratio=1.0)
    float_score = score_symmetric(test, labels, candidate_width=CANDIDATE_WIDTH, device=device)
    query_variants = {
        "baseline": test,
        "clipped_only": test,
        "rotation_only": rotated_test,
        "rotated_clipped": rotated_test,
    }
    scores: dict[str, object] = {
        "clipping_ratio": ratio,
        "clipping_train_pairwise_gram_mse": ratio_evidence,
        "float": _summary(float_score),
        "persistent_bytes_per_item": variants["baseline"].bytes_per_vector,
        "query_bytes_per_item": test.shape[1] * 4,
    }
    per_query: dict[str, object] = {}
    for name, packed in variants.items():
        symmetric = score_int4_symmetric(
            packed, labels, candidate_width=CANDIDATE_WIDTH, device=device
        )
        scores[f"{name}_symmetric"] = _summary(symmetric)
        if factorial:
            asymmetric = score_asymmetric_int4(
                query_variants[name],
                packed,
                labels,
                candidate_width=CANDIDATE_WIDTH,
                device=device,
            )
            scores[f"{name}_asymmetric"] = _summary(asymmetric)
            per_query[f"{name}_asymmetric"] = {
                "ap": asymmetric["per_query_ap"],
                "r1": asymmetric["per_query_r1"],
            }
        if factorial:
            per_query[f"{name}_symmetric"] = {
                "ap": symmetric["per_query_ap"],
                "r1": symmetric["per_query_r1"],
            }
    if factorial:
        scores["per_query_evidence"] = per_query
    return scores


def run_comparable_sop_128_authority(
    pair: PairedArchives,
) -> tuple[dict[str, object], bytes]:
    """Train the comparable 128D arm and return its receipt plus model bytes."""

    source_train = pair["source_train"]
    teacher_train = pair["teacher_train"]
    source_test = pair["source_test"]
    labels = pair["test_labels"]
    basis = _fit_uncentered_covariance_basis(source_train, dimensions=DIMENSIONS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pca = RelationalLinearEncoder(basis)
    rotation = fixed_random_rotation(DIMENSIONS, seed=SEED)
    rotated_pca = fold_output_rotation(pca, rotation)
    pca_train = _encode_floating(pca, source_train, device=device)
    pca_test = _encode_floating(pca, source_test, device=device)
    pca_rotated_train = _encode_floating(rotated_pca, source_train, device=device)
    pca_rotated_test = _encode_floating(rotated_pca, source_test, device=device)
    pca_scores = _score_family(
        pca_train,
        pca_test,
        pca_rotated_train,
        pca_rotated_test,
        labels,
        device=device,
        factorial=False,
    )
    pca_float = cast(dict[str, float], pca_scores["float"])
    validate_sealed_pca_reproduction(float_map_at_r=pca_float["map_at_r"])
    relational, losses = fit_relational_linear_encoder(
        source_train,
        teacher_train,
        basis,
        config=RelationalLinearTrainingConfig(
            batch_size=BATCH,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            seed=SEED,
            temperature=TEMPERATURE,
            updates=UPDATES,
            weight_decay=WEIGHT_DECAY,
        ),
        device=device,
    )
    rotated_relational = fold_output_rotation(relational, rotation)
    relational_train = _encode_floating(relational, source_train, device=device)
    relational_test = _encode_floating(relational, source_test, device=device)
    relational_rotated_train = _encode_floating(rotated_relational, source_train, device=device)
    relational_rotated_test = _encode_floating(rotated_relational, source_test, device=device)
    relational_scores = _score_family(
        relational_train,
        relational_test,
        relational_rotated_train,
        relational_rotated_test,
        labels,
        device=device,
        factorial=True,
    )
    rel_base = cast(dict[str, float], relational_scores["baseline_symmetric"])
    rel_asym = cast(dict[str, float], relational_scores["baseline_asymmetric"])
    pca_base = cast(dict[str, float], pca_scores["baseline_symmetric"])
    pca_rotated = cast(dict[str, float], pca_scores["rotated_clipped_symmetric"])
    rel_rotated_asym = cast(dict[str, float], relational_scores["rotated_clipped_asymmetric"])
    rel_per_query = cast(
        dict[str, dict[str, tuple[float, ...]]], relational_scores["per_query_evidence"]
    )
    clusters = torch.tensor(labels, dtype=torch.int64)
    asymmetric_lower = _familywise_lower_bound(
        rel_per_query["baseline_asymmetric"]["ap"],
        rel_per_query["baseline_symmetric"]["ap"],
        clusters,
        seed=SEED,
        samples=BOOTSTRAP_SAMPLES,
        comparisons=BOOTSTRAP_COMPARISONS,
    )
    composed_lower = _familywise_lower_bound(
        rel_per_query["rotated_clipped_asymmetric"]["ap"],
        rel_per_query["baseline_asymmetric"]["ap"],
        clusters,
        seed=SEED + 1,
        samples=BOOTSTRAP_SAMPLES,
        comparisons=BOOTSTRAP_COMPARISONS,
    )
    decision = quantization_decision(
        baseline_symmetric_map_at_r=rel_base["map_at_r"],
        baseline_asymmetric_map_at_r=rel_asym["map_at_r"],
        rotated_clipped_asymmetric_map_at_r=rel_rotated_asym["map_at_r"],
        asymmetric_map_lower_bound=asymmetric_lower,
        composed_map_lower_bound=composed_lower,
        pca_float_to_int4_loss=pca_float["map_at_r"] - pca_base["map_at_r"],
        rotated_recovery=pca_rotated["map_at_r"] - pca_base["map_at_r"],
    )
    selected_model = rotated_relational if decision["rotation_clipping"] else relational
    relational_model_wire = relational.to_bytes()
    steps_per_epoch = len(source_train) // BATCH
    complete_epochs, final_epoch_updates = divmod(UPDATES, steps_per_epoch)
    training_epoch_update_counts = [steps_per_epoch] * complete_epochs
    if final_epoch_updates:
        training_epoch_update_counts.append(final_epoch_updates)
    if len(training_epoch_update_counts) != len(losses):
        raise ValueError("comparable SOP training evidence differs")
    environment = {
        "cuda_device": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
    }
    return {
        "claim_eligible": False,
        "bootstrap_comparisons": BOOTSTRAP_COMPARISONS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "decision": decision,
        "dimensions": DIMENSIONS,
        "environment": environment,
        "gates": {
            "asymmetric_gain": ASYMMETRIC_GAIN_GATE,
            "rotation_recovery_fraction": ROTATION_RECOVERY_FRACTION,
        },
        "pca": pca_scores,
        "clipping_pair_samples": CLIPPING_PAIR_SAMPLES,
        "clipping_ratios": CLIPPING_RATIOS,
        "relational": relational_scores,
        "model_output_arm": "unrotated-relational",
        "relational_model_bytes": len(relational_model_wire),
        "relational_model_sha256": hashlib.sha256(relational_model_wire).hexdigest(),
        "rotation": {"kind": "seed-fixed-orthogonal", "seed": SEED},
        "rotated_relational_model_sha256": hashlib.sha256(
            rotated_relational.to_bytes()
        ).hexdigest(),
        "schema": "sfora-sop-quantization-geometry-v2",
        "selected_model_sha256": hashlib.sha256(selected_model.to_bytes()).hexdigest(),
        "seed": SEED,
        "test_classes": len(set(labels)),
        "test_rows": len(source_test),
        "train_rows": len(source_train),
        "training_epoch_update_counts": training_epoch_update_counts,
        "training_losses": losses,
        "training_recipe": {
            "batch_size": BATCH,
            "comparable_to": {
                "axis": "optimizer-updates",
                "optimizer_updates": 1_000,
                "receipt": "sfora-cub-relational-int4-evaluation-v1",
            },
            "epochs_capacity": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "steps_per_epoch": steps_per_epoch,
            "temperature": TEMPERATURE,
            "updates": UPDATES,
            "weight_decay": WEIGHT_DECAY,
        },
        "training_updates": UPDATES,
    }, relational_model_wire


def _comparable_sop_output_paths(output: Path, model_output: Path) -> tuple[Path, Path]:
    if not isinstance(output, Path) or not isinstance(model_output, Path) or output == model_output:
        raise ValueError("comparable SOP authority differs")
    receipt_partial = output.with_name(output.name + ".partial")
    model_partial = model_output.with_name(model_output.name + ".partial")
    paths = (output, model_output, receipt_partial, model_partial)
    if len(set(paths)) != len(paths):
        raise ValueError("comparable SOP authority differs")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError("comparable SOP output exists")
    return receipt_partial, model_partial


def _reserve_comparable_sop_outputs(output: Path, model_output: Path) -> tuple[Path, Path]:
    receipt_partial, model_partial = _comparable_sop_output_paths(output, model_output)
    created: list[Path] = []
    try:
        for path in (model_partial, receipt_partial):
            with path.open("xb"):
                pass
            created.append(path)
    except BaseException:
        for path in created:
            if path.is_file() and not path.is_symlink():
                path.unlink()
        raise
    return receipt_partial, model_partial


def _publish_reserved_comparable_sop_authority(
    *,
    receipt: dict[str, object],
    model_wire: bytes,
    output: Path,
    model_output: Path,
    receipt_partial: Path,
    model_partial: Path,
) -> None:
    if (
        type(receipt) is not dict
        or type(model_wire) is not bytes
        or not model_wire
        or receipt.get("relational_model_sha256") != hashlib.sha256(model_wire).hexdigest()
    ):
        raise ValueError("comparable SOP authority differs")
    raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    published: list[Path] = []
    try:
        for path, payload in ((model_partial, model_wire), (receipt_partial, raw)):
            with path.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        os.link(model_partial, model_output)
        published.append(model_output)
        os.link(receipt_partial, output)
        published.append(output)
    except BaseException:
        for path in published:
            if path.is_file() and not path.is_symlink():
                path.unlink()
        raise


def publish_comparable_sop_128_authority(
    *,
    receipt: dict[str, object],
    model_wire: bytes,
    output: Path,
    model_output: Path,
) -> None:
    """Atomically publish a canonical receipt and its bound relational model."""

    receipt_partial, model_partial = _reserve_comparable_sop_outputs(output, model_output)
    try:
        _publish_reserved_comparable_sop_authority(
            receipt=receipt,
            model_wire=model_wire,
            output=output,
            model_output=model_output,
            receipt_partial=receipt_partial,
            model_partial=model_partial,
        )
    finally:
        for path in (model_partial, receipt_partial):
            if path.is_file() and not path.is_symlink():
                path.unlink()


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _digest(value: str) -> str:
    if len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise argparse.ArgumentTypeError("digest must be lowercase SHA-256")
    return value


def _commit(value: str) -> str:
    if len(value) != 40 or set(value) - set("0123456789abcdef"):
        raise argparse.ArgumentTypeError("commit must be lowercase hexadecimal")
    return value


def verify_quantization_source_commit(source_commit: str) -> None:
    """Bind both shared dependencies and this experiment to one commit."""

    verify_source_commit(source_commit)
    root = Path(__file__).resolve().parents[1]
    relative = Path(__file__).resolve().relative_to(root).as_posix()
    committed = subprocess.run(
        ["git", "-C", str(root), "show", f"{source_commit}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout
    if committed != Path(__file__).read_bytes():
        raise ValueError("SOP quantization source differs from registered commit")


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the local-only burned-SOP experimental surface."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-embeddings", required=True, type=_absolute_path)
    parser.add_argument("--source-embeddings-sha256", required=True, type=_digest)
    parser.add_argument("--teacher-embeddings", required=True, type=_absolute_path)
    parser.add_argument("--teacher-embeddings-sha256", required=True, type=_digest)
    parser.add_argument("--source-commit", required=True, type=_commit)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--model-output", required=True, type=_absolute_path)
    parser.add_argument("--execute-sop-quantization", action="store_true", required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Authenticate, evaluate once, and publish one canonical decision receipt."""

    args = parse_args(arguments)
    receipt_partial, model_partial = _reserve_comparable_sop_outputs(args.output, args.model_output)
    try:
        verify_quantization_source_commit(args.source_commit)
        pair = load_paired_archives(
            args.source_embeddings,
            args.source_embeddings_sha256,
            args.teacher_embeddings,
            args.teacher_embeddings_sha256,
        )
        result, model_wire = run_comparable_sop_128_authority(pair)
        result["source_commit"] = args.source_commit
        result["source_embeddings_sha256"] = args.source_embeddings_sha256
        result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["teacher_embeddings_sha256"] = args.teacher_embeddings_sha256
        _publish_reserved_comparable_sop_authority(
            receipt=result,
            model_wire=model_wire,
            output=args.output,
            model_output=args.model_output,
            receipt_partial=receipt_partial,
            model_partial=model_partial,
        )
    finally:
        for path in (model_partial, receipt_partial):
            if path.is_file() and not path.is_symlink():
                path.unlink()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
