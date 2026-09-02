#!/usr/bin/env python3
"""Evaluate fixed cross-seed candidate towers against the burned local authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from scripts.build_cross_seed_denoising import _load_inputs
from scripts.diagnose_weight_space_transfer import LoadedBurnedInputs, load_burned_inputs
from scripts.run_siglip_proxy_control import (
    embed_control_examples,
    load_siglip_control_components,
    require_control_determinism,
)
from sfora.cross_seed_denoising import (
    CandidateEvaluation,
    HeadSwapEvaluation,
    ProjectedEvaluation,
    canonical_denoising_result_bytes,
    classify_denoising_result,
    read_tensor_artifact,
)
from sfora.data import ImageExample
from sfora.siglip_proxy_control import (
    PooledProxyAnchorModel,
    SiglipProxyControlConfig,
    nearest_class_margins,
)
from sfora.substrate_screen import score_frozen_substrate_evidence
from sfora.token_set_screen import F1_TRAIN_CLASSES
from sfora.weight_space_transfer import (
    AlphaEvaluation,
    SeedInterpolationCurve,
    canonical_interpolation_result_bytes,
    classify_interpolation_curves,
    model_state_sha256,
)

_SEEDS = (17, 29, 43)
_ROLES = ("tower-soup", "wiener-denoise", "spectral-denoise")


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be lowercase SHA-256")
    return value


def _positive(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("byte length must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("byte length must be positive")
    return parsed


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the local-only candidate evaluation capability."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True, type=_absolute_path)
    parser.add_argument("--prepared-manifest", required=True, type=_absolute_path)
    parser.add_argument("--prepared-manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--prepared-manifest-bytes", required=True, type=_positive)
    parser.add_argument("--candidate-root", required=True, type=_absolute_path)
    parser.add_argument("--candidate-receipt", required=True, type=_absolute_path)
    parser.add_argument("--candidate-receipt-sha256", required=True, type=_sha256)
    parser.add_argument("--candidate-receipt-bytes", required=True, type=_positive)
    parser.add_argument("--scalar-result", required=True, type=_absolute_path)
    parser.add_argument("--scalar-result-sha256", required=True, type=_sha256)
    parser.add_argument("--scalar-result-bytes", required=True, type=_positive)
    parser.add_argument("--burned-manifest", required=True, type=_absolute_path)
    parser.add_argument("--burned-manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--burned-manifest-bytes", required=True, type=_positive)
    parser.add_argument("--burned-image-root", required=True, type=_absolute_path)
    parser.add_argument("--source-manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument(
        "--execute-cross-seed-evaluation", required=True, action="store_true"
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class BandEvaluation:
    """One exact 1,345-query plane evaluation returned by the model callback."""

    correctness: tuple[bool, ...]
    mean_nearest_positive_cosine: float
    mean_nearest_negative_cosine: float
    mean_margin: float
    wall_time_ns: int
    peak_cuda_bytes: int
    peak_rss_bytes: int
    determinism_replay: bool

    def __post_init__(self) -> None:
        if (
            type(self.correctness) is not tuple
            or len(self.correctness) != 1345
            or any(type(value) is not bool for value in self.correctness)
        ):
            raise ValueError("band correctness evidence differs")
        means = (
            self.mean_nearest_positive_cosine,
            self.mean_nearest_negative_cosine,
            self.mean_margin,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in means):
            raise ValueError("band means must be concrete finite floats")
        if (
            type(self.wall_time_ns) is not int
            or self.wall_time_ns <= 0
            or type(self.peak_cuda_bytes) is not int
            or self.peak_cuda_bytes < 0
            or type(self.peak_rss_bytes) is not int
            or self.peak_rss_bytes <= 0
        ):
            raise ValueError("band resource evidence differs")
        if type(self.determinism_replay) is not bool or not self.determinism_replay:
            raise ValueError("band determinism replay differs")


def _state_mapping(value: object, *, role: str) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(value, (OrderedDict, Mapping)) or not value:
        raise ValueError(f"{role} state differs")
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in sorted(cast(Mapping[object, object], value).items()):
        if type(name) is not str or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{role} state differs")
        if tensor.layout != torch.strided or (
            tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"{role} tensor differs")
        result[name] = tensor.detach().cpu().contiguous().clone()
    return result


def _seed_states(value: object, *, role: str) -> dict[int, OrderedDict[str, torch.Tensor]]:
    if type(value) is not dict or set(value) != set(_SEEDS):
        raise ValueError(f"{role} must contain exactly the registered seeds")
    return {
        seed: _state_mapping(cast(dict[int, object], value)[seed], role=f"{role} {seed}")
        for seed in _SEEDS
    }


def _candidate_states(value: object) -> dict[str, OrderedDict[str, torch.Tensor]]:
    if type(value) is not dict or tuple(value) != _ROLES:
        raise ValueError("candidate tower order differs")
    return {
        role: _state_mapping(cast(dict[str, object], value)[role], role=role)
        for role in _ROLES
    }


def _folded_digest(
    tower: OrderedDict[str, torch.Tensor], head: OrderedDict[str, torch.Tensor]
) -> str:
    if any(not name.startswith("tower.") for name in tower):
        raise ValueError("tower state contains head tensors")
    if tuple(head) != ("projection.weight", "proxies"):
        raise ValueError("head state differs")
    return model_state_sha256(OrderedDict((*tower.items(), *head.items())))


def _require_embedding_replay(
    first: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    replay: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for left, right in zip(first, replay, strict=True):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(
                left.detach().contiguous().view(torch.uint8),
                right.detach().contiguous().view(torch.uint8),
            )
        ):
            raise ValueError("model forward determinism replay differs")


def evaluate_cross_seed_denoising(
    *,
    candidate_towers: object,
    trained_towers: object,
    trained_heads: object,
    scalar_curves: object,
    candidate_state_sha256: object,
    construction_evidence_sha256: str,
    evaluate_raw: Callable[[OrderedDict[str, torch.Tensor]], BandEvaluation],
    evaluate_projected: Callable[
        [OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]], BandEvaluation
    ],
    failure: str | None = None,
) -> bytes:
    """Evaluate exactly three raw candidates, nine candidate heads, and six swaps."""

    candidates = _candidate_states(candidate_towers)
    towers = _seed_states(trained_towers, role="trained towers")
    heads = _seed_states(trained_heads, role="trained heads")
    if (
        type(candidate_state_sha256) is not dict
        or tuple(candidate_state_sha256) != _ROLES
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in candidate_state_sha256.values()
        )
    ):
        raise ValueError("candidate state digest authority differs")
    if (
        type(construction_evidence_sha256) is not str
        or len(construction_evidence_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in construction_evidence_sha256
        )
    ):
        raise ValueError("construction evidence digest differs")
    if not callable(evaluate_raw) or not callable(evaluate_projected):
        raise ValueError("evaluation callback differs")

    candidate_rows: list[CandidateEvaluation] = []
    for role in _ROLES:
        tower = candidates[role]
        raw = evaluate_raw(tower)
        if type(raw) is not BandEvaluation:
            raise ValueError("raw evaluator returned the wrong type")
        projected_rows: list[ProjectedEvaluation] = []
        for seed in _SEEDS:
            projected = evaluate_projected(tower, heads[seed])
            if type(projected) is not BandEvaluation:
                raise ValueError("projected evaluator returned the wrong type")
            projected_rows.append(
                ProjectedEvaluation(
                    seed=seed,
                    correctness=projected.correctness,
                    mean_nearest_positive_cosine=projected.mean_nearest_positive_cosine,
                    mean_nearest_negative_cosine=projected.mean_nearest_negative_cosine,
                    mean_margin=projected.mean_margin,
                    folded_state_sha256=_folded_digest(tower, heads[seed]),
                    wall_time_ns=projected.wall_time_ns,
                    peak_cuda_bytes=projected.peak_cuda_bytes,
                    peak_rss_bytes=projected.peak_rss_bytes,
                    determinism_replay=projected.determinism_replay,
                )
            )
        candidate_rows.append(
            CandidateEvaluation(
                role=role,
                raw_correctness=raw.correctness,
                raw_mean_nearest_positive_cosine=raw.mean_nearest_positive_cosine,
                raw_mean_nearest_negative_cosine=raw.mean_nearest_negative_cosine,
                raw_mean_margin=raw.mean_margin,
                raw_wall_time_ns=raw.wall_time_ns,
                raw_peak_cuda_bytes=raw.peak_cuda_bytes,
                raw_peak_rss_bytes=raw.peak_rss_bytes,
                raw_determinism_replay=raw.determinism_replay,
                projected=tuple(projected_rows),
                tower_state_sha256=cast(dict[str, str], candidate_state_sha256)[role],
                construction_evidence_sha256=construction_evidence_sha256,
            )
        )

    swap_rows: list[HeadSwapEvaluation] = []
    own_rows = {
        seed: evaluate_projected(towers[seed], heads[seed]) for seed in _SEEDS
    }
    if any(type(row) is not BandEvaluation for row in own_rows.values()):
        raise ValueError("head swap evaluator returned the wrong type")
    if (
        type(scalar_curves) is not tuple
        or len(scalar_curves) != 3
        or any(type(curve) is not SeedInterpolationCurve for curve in scalar_curves)
    ):
        raise ValueError("scalar endpoint replay authority differs")
    typed_curves = cast(tuple[SeedInterpolationCurve, ...], scalar_curves)
    for curve in typed_curves:
        endpoint = curve.rows[-1]
        own = own_rows[curve.seed]
        if (
            endpoint.alpha != 1.0
            or own.correctness != endpoint.correctness
            or own.mean_nearest_positive_cosine
            != endpoint.mean_nearest_positive_cosine
            or own.mean_nearest_negative_cosine
            != endpoint.mean_nearest_negative_cosine
            or own.mean_margin != endpoint.mean_margin
        ):
            raise ValueError("trained endpoint replay differs")
    for source in _SEEDS:
        for target in _SEEDS:
            if source == target:
                continue
            own = own_rows[source]
            swapped = evaluate_projected(towers[source], heads[target])
            if type(own) is not BandEvaluation or type(swapped) is not BandEvaluation:
                raise ValueError("head swap evaluator returned the wrong type")
            swap_rows.append(
                HeadSwapEvaluation(
                    source_seed=source,
                    target_seed=target,
                    own_correctness=own.correctness,
                    swapped_correctness=swapped.correctness,
                    own_mean_margin=own.mean_margin,
                    swapped_mean_margin=swapped.mean_margin,
                )
            )
    decision = classify_denoising_result(
        scalar_curves, tuple(candidate_rows), tuple(swap_rows), failure=failure
    )
    return canonical_denoising_result_bytes(
        scalar_curves,
        tuple(candidate_rows),
        tuple(swap_rows),
        decision,
        failure=failure,
    )


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _read_bound(path: Path, sha256: str, byte_count: int, *, role: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} must be a regular file")
    raw = path.read_bytes()
    if len(raw) != byte_count or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"{role} identity differs")
    return raw


def _load_scalar_curves(
    raw: bytes,
    *,
    prepared_bindings: object,
    burned_manifest_sha256: str,
) -> tuple[SeedInterpolationCurve, ...]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("scalar result is not valid JSON") from exc
    if type(value) is not dict or _canonical(value) != raw:
        raise ValueError("scalar result is not canonical")
    campaign_keys = {
        "capabilities",
        "capabilities_sha256",
        "child_result_bytes",
        "child_result_sha256",
        "claim_eligible",
        "result",
        "schema",
    }
    if (
        set(value) != campaign_keys
        or value["schema"] != "sfora-weight-space-transfer-campaign-result-v1"
        or value["claim_eligible"] is not False
        or type(value["capabilities"]) is not dict
        or type(value["result"]) is not dict
    ):
        raise ValueError("scalar campaign result schema differs")
    capabilities = cast(dict[str, object], value["capabilities"])
    capability_keys = {
        "claim_eligible",
        "controller_source_commit",
        "roles",
        "schema",
        "source_commit",
        "source_manifest_sha256",
        "source_tree_digest",
        "spec_bytes",
        "spec_sha256",
    }
    capability_raw = _canonical(capabilities)
    expected_roles = ("burned-manifest",) + tuple(
        role
        for seed in _SEEDS
        for role in (f"seed-{seed:03d}-result", f"seed-{seed:03d}-checkpoint")
    )
    roles = capabilities.get("roles")
    if (
        set(capabilities) != capability_keys
        or capabilities.get("schema") != "sfora-weight-space-transfer-capabilities-v1"
        or capabilities.get("claim_eligible") is not False
        or type(roles) is not list
        or tuple(
            row.get("role") if type(row) is dict else None
            for row in cast(list[object], roles)
        )
        != expected_roles
        or type(value["capabilities_sha256"]) is not str
        or value["capabilities_sha256"] != hashlib.sha256(capability_raw).hexdigest()
    ):
        raise ValueError("scalar campaign capabilities differ")
    if type(prepared_bindings) is not dict:
        raise ValueError("scalar campaign prepared bindings differ")
    bindings = cast(dict[str, object], prepared_bindings)
    for role in cast(list[object], roles):
        if (
            type(role) is not dict
            or set(role) != {"bytes", "role", "sha256"}
            or type(role["bytes"]) is not int
            or role["bytes"] <= 0
            or type(role["sha256"]) is not str
            or len(role["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in role["sha256"])
        ):
            raise ValueError("scalar campaign role authority differs")
    role_sha256 = {
        cast(str, cast(dict[str, object], role)["role"]): cast(
            str, cast(dict[str, object], role)["sha256"]
        )
        for role in cast(list[object], roles)
    }
    if (
        capabilities["source_commit"] != bindings.get("source_commit")
        or capabilities["source_tree_digest"] != bindings.get("source_tree_digest")
        or capabilities["source_manifest_sha256"]
        != bindings.get("dataset_manifest_sha256")
        or role_sha256["burned-manifest"] != burned_manifest_sha256
        or any(
            role_sha256[f"seed-{seed:03d}-result"]
            != bindings.get(f"seed_{seed}_result_sha256")
            or role_sha256[f"seed-{seed:03d}-checkpoint"]
            != bindings.get(f"seed_{seed}_checkpoint_sha256")
            for seed in _SEEDS
        )
    ):
        raise ValueError("scalar campaign binding differs")
    child_raw = _canonical(value["result"])
    if (
        type(value["child_result_bytes"]) is not int
        or value["child_result_bytes"] != len(child_raw)
        or type(value["child_result_sha256"]) is not str
        or value["child_result_sha256"] != hashlib.sha256(child_raw).hexdigest()
    ):
        raise ValueError("scalar campaign child result differs")
    value = cast(dict[str, object], value["result"])
    if (
        set(value) != {"claim_eligible", "curves", "decision", "schema"}
        or value["schema"] != "sfora-weight-space-transfer-result-v1"
        or value["claim_eligible"] is not False
        or type(value["curves"]) is not list
    ):
        raise ValueError("scalar result schema differs")
    curves: list[SeedInterpolationCurve] = []
    for raw_curve in cast(list[object], value["curves"]):
        if type(raw_curve) is not dict or set(raw_curve) != {"rows", "seed"}:
            raise ValueError("scalar curve schema differs")
        curve = cast(dict[str, object], raw_curve)
        if type(curve["rows"]) is not list:
            raise ValueError("scalar curve rows differ")
        rows: list[AlphaEvaluation] = []
        for raw_row in cast(list[object], curve["rows"]):
            if type(raw_row) is not dict:
                raise ValueError("scalar row schema differs")
            row = cast(dict[str, object], raw_row)
            bits = row.get("correctness_bits")
            if type(bits) is not str:
                raise ValueError("scalar correctness bits differ")
            try:
                correctness_raw = bytes.fromhex(bits)
            except ValueError as exc:
                raise ValueError("scalar correctness bits differ") from exc
            if len(correctness_raw) != 1345 or any(item not in (0, 1) for item in correctness_raw):
                raise ValueError("scalar correctness bits differ")
            correctness = tuple(bool(item) for item in correctness_raw)
            rows.append(
                AlphaEvaluation(
                    seed=cast(int, curve["seed"]),
                    alpha=cast(float, row["alpha"]),
                    correct=cast(int, row["correct"]),
                    queries=cast(int, row["queries"]),
                    recall_ppm=cast(int, row["recall_ppm"]),
                    mean_nearest_positive_cosine=cast(
                        float, row["mean_nearest_positive_cosine"]
                    ),
                    mean_nearest_negative_cosine=cast(
                        float, row["mean_nearest_negative_cosine"]
                    ),
                    mean_margin=cast(float, row["mean_margin"]),
                    correctness=correctness,
                    folded_state_sha256=cast(str, row["folded_state_sha256"]),
                    tower_squared_displacement=cast(
                        float, row["tower_squared_displacement"]
                    ),
                    wall_time_ns=cast(int, row["wall_time_ns"]),
                    peak_cuda_bytes=cast(int, row["peak_cuda_bytes"]),
                    peak_rss_bytes=cast(int, row["peak_rss_bytes"]),
                )
            )
        curves.append(
            SeedInterpolationCurve(seed=cast(int, curve["seed"]), rows=tuple(rows))
        )
    result = tuple(curves)
    if tuple(curve.seed for curve in result) != _SEEDS:
        raise ValueError("scalar curve seed order differs")
    decision = classify_interpolation_curves(result)
    if canonical_interpolation_result_bytes(result, decision) != child_raw:
        raise ValueError("scalar result authority differs")
    return result


def _load_prepared_states(
    root: Path, manifest_raw: bytes
) -> tuple[
    dict[int, OrderedDict[str, torch.Tensor]],
    dict[int, OrderedDict[str, torch.Tensor]],
]:
    _initial, towers = _load_inputs(root, manifest_raw, include_heads=True)
    value = json.loads(manifest_raw)
    heads: dict[int, OrderedDict[str, torch.Tensor]] = {}
    for raw_row, seed in zip(value["seeds"], _SEEDS, strict=True):
        row = cast(dict[str, object], raw_row)
        directory = row["head_directory"]
        if type(directory) is not str:
            raise ValueError("prepared head path differs")
        head_root = root / directory
        manifest_path = head_root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("prepared head manifest differs")
        raw = manifest_path.read_bytes()
        if (
            row["head_manifest_bytes"] != len(raw)
            or row["head_manifest_sha256"] != hashlib.sha256(raw).hexdigest()
            or json.loads(raw).get("state_sha256") != row["head_state_sha256"]
        ):
            raise ValueError("prepared head identity differs")
        heads[seed] = read_tensor_artifact(head_root, raw, role="trained-head")
    return towers, heads


def _load_candidates(
    root: Path, receipt_raw: bytes, prepared_sha256: str
) -> tuple[
    dict[str, OrderedDict[str, torch.Tensor]],
    dict[str, str],
    str,
]:
    try:
        value = json.loads(receipt_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("candidate receipt is not valid JSON") from exc
    if type(value) is not dict or _canonical(value) != receipt_raw:
        raise ValueError("candidate receipt is not canonical")
    keys = {
        "candidates",
        "claim_eligible",
        "construction_evidence",
        "aggregate_retained_energy_ratio",
        "determinism_replay",
        "prepared_manifest_bytes",
        "prepared_manifest_sha256",
        "projected_peak_rss_bytes",
        "schema",
    }
    if (
        set(value) != keys
        or value["schema"] != "sfora-cross-seed-candidate-receipt-v1"
        or value["claim_eligible"] is not False
        or value["determinism_replay"] is not True
        or value["prepared_manifest_sha256"] != prepared_sha256
        or type(value["candidates"]) is not list
        or len(value["candidates"]) != 3
    ):
        raise ValueError("candidate receipt authority differs")
    construction = value["construction_evidence"]
    if type(construction) is not dict or type(construction.get("spectral")) is not list:
        raise ValueError("candidate construction evidence differs")
    spectral = cast(list[object], construction["spectral"])
    if any(type(row) is not dict for row in spectral):
        raise ValueError("candidate construction evidence differs")
    for raw_row in spectral:
        row = cast(dict[str, object], raw_row)
        if (
            type(row.get("retained_energy")) is not float
            or not math.isfinite(cast(float, row["retained_energy"]))
            or cast(float, row["retained_energy"]) < 0.0
            or type(row.get("total_energy")) is not float
            or not math.isfinite(cast(float, row["total_energy"]))
            or cast(float, row["total_energy"]) < 0.0
        ):
            raise ValueError("candidate construction evidence differs")
    retained = sum(
        cast(float, cast(dict[str, object], row)["retained_energy"]) for row in spectral
    )
    total = sum(
        cast(float, cast(dict[str, object], row)["total_energy"]) for row in spectral
    )
    expected_ratio = 0.0 if total == 0.0 else retained / total
    if (
        type(value["aggregate_retained_energy_ratio"]) is not float
        or value["aggregate_retained_energy_ratio"] != expected_ratio
    ):
        raise ValueError("candidate aggregate retained energy differs")
    states: dict[str, OrderedDict[str, torch.Tensor]] = {}
    digests: dict[str, str] = {}
    expected_namespace = {"receipt.json"}
    for raw_row, role in zip(value["candidates"], _ROLES, strict=True):
        if type(raw_row) is not dict or raw_row.get("role") != role:
            raise ValueError("candidate receipt order differs")
        row = cast(dict[str, object], raw_row)
        directory = row.get("directory")
        if type(directory) is not str or directory != role:
            raise ValueError("candidate path differs")
        expected_namespace.add(directory)
        candidate_root = root / directory
        manifest_path = candidate_root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("candidate manifest differs")
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        if (
            row.get("manifest_bytes") != len(raw)
            or row.get("manifest_sha256") != hashlib.sha256(raw).hexdigest()
            or row.get("state_sha256") != manifest.get("state_sha256")
        ):
            raise ValueError("candidate manifest identity differs")
        states[role] = read_tensor_artifact(candidate_root, raw, role=role)
        digests[role] = cast(str, row["state_sha256"])
    if {path.name for path in root.iterdir()} != expected_namespace:
        raise ValueError("candidate namespace differs")
    construction_sha256 = hashlib.sha256(_canonical(value["construction_evidence"])).hexdigest()
    return states, digests, construction_sha256


class _CudaBandEvaluator:
    def __init__(
        self,
        burned: LoadedBurnedInputs,
        *,
        evaluation_batch_size: int,
        query_block: int,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("cross-seed evaluation requires CUDA")
        self.device = torch.device("cuda")
        require_control_determinism(self.device)
        self.config = SiglipProxyControlConfig()
        tower, self.processor = load_siglip_control_components(config=self.config)
        self.model = PooledProxyAnchorModel(
            tower=tower,
            input_dimensions=self.config.input_dimensions,
            embedding_dimensions=self.config.embedding_dimensions,
            class_count=len(F1_TRAIN_CLASSES),
            projection_initialization=self.config.projection_initialization,
            proxy_initialization=self.config.proxy_initialization,
        ).to(self.device)
        vision = self.model.tower.vision_model
        disable = getattr(vision, "gradient_checkpointing_disable", None)
        if not callable(disable):
            raise TypeError("evaluation tower lacks checkpointing control")
        disable()
        self.model.eval()
        self.burned = burned
        if evaluation_batch_size <= 0 or query_block <= 0:
            raise ValueError("evaluation protocol differs")
        self.evaluation_batch_size = evaluation_batch_size
        self.query_block = query_block

    def evaluate(
        self,
        tower: OrderedDict[str, torch.Tensor],
        head: OrderedDict[str, torch.Tensor],
        *,
        plane: str,
    ) -> BandEvaluation:
        state = OrderedDict((*tower.items(), *head.items()))
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError("evaluation model state differs") from exc
        examples = tuple(
            ImageExample(example_id=row.example_id, image=row.path, label=row.label)
            for row in self.burned.rows
        )
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.monotonic_ns()
        with torch.inference_mode():
            raw, projected, labels = embed_control_examples(
                model=self.model,
                examples=examples,
                processor=self.processor,
                device=self.device,
                batch_size=self.evaluation_batch_size,
            )
            replayed = embed_control_examples(
                model=self.model,
                examples=examples,
                processor=self.processor,
                device=self.device,
                batch_size=self.evaluation_batch_size,
            )
        _require_embedding_replay((raw, projected, labels), replayed)
        embeddings = raw if plane == "raw" else projected
        if plane not in ("raw", "projected"):
            raise ValueError("evaluation plane differs")
        evidence = score_frozen_substrate_evidence(
            embeddings, labels, query_block=self.query_block
        )
        replay = score_frozen_substrate_evidence(
            embeddings, labels, query_block=self.query_block
        )
        margins = nearest_class_margins(
            embeddings, labels, query_block=self.query_block
        )
        if evidence != replay:
            raise ValueError("retrieval determinism replay differs")
        correctness = [True] * len(self.burned.rows)
        for error in evidence.errors:
            if not 0 <= error.query_position < len(correctness) or not correctness[
                error.query_position
            ]:
                raise ValueError("retrieval error identity differs")
            correctness[error.query_position] = False
        if sum(correctness) != evidence.metrics.correct:
            raise ValueError("retrieval correctness arithmetic differs")
        torch.cuda.synchronize(self.device)
        return BandEvaluation(
            correctness=tuple(correctness),
            mean_nearest_positive_cosine=margins.mean_nearest_positive_cosine,
            mean_nearest_negative_cosine=margins.mean_nearest_negative_cosine,
            mean_margin=margins.mean_margin,
            wall_time_ns=time.monotonic_ns() - started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(self.device),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            determinism_replay=True,
        )


def _publish_new(path: Path, raw: bytes) -> None:
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if path.exists() or path.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError(path)
    try:
        with partial.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        partial.rename(path)
    finally:
        partial.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Authenticate all local inputs, run one serialized GPU evaluation, and publish."""

    arguments = parse_arguments(argv)
    prepared_raw = _read_bound(
        arguments.prepared_manifest,
        arguments.prepared_manifest_sha256,
        arguments.prepared_manifest_bytes,
        role="prepared manifest",
    )
    if arguments.prepared_manifest.parent != arguments.prepared_root:
        raise ValueError("prepared manifest path differs")
    candidate_raw = _read_bound(
        arguments.candidate_receipt,
        arguments.candidate_receipt_sha256,
        arguments.candidate_receipt_bytes,
        role="candidate receipt",
    )
    if arguments.candidate_receipt.parent != arguments.candidate_root:
        raise ValueError("candidate receipt path differs")
    scalar_raw = _read_bound(
        arguments.scalar_result,
        arguments.scalar_result_sha256,
        arguments.scalar_result_bytes,
        role="scalar result",
    )
    prepared_value = json.loads(prepared_raw)
    prepared_bindings = prepared_value.get("bindings")
    if type(prepared_bindings) is not dict:
        raise ValueError("prepared evaluation bindings differ")
    scalar_curves = _load_scalar_curves(
        scalar_raw,
        prepared_bindings=prepared_bindings,
        burned_manifest_sha256=arguments.burned_manifest_sha256,
    )
    burned = load_burned_inputs(
        manifest_path=arguments.burned_manifest,
        expected_sha256=arguments.burned_manifest_sha256,
        expected_bytes=arguments.burned_manifest_bytes,
        expected_source_manifest_sha256=arguments.source_manifest_sha256,
        image_root=arguments.burned_image_root,
    )
    towers, heads = _load_prepared_states(arguments.prepared_root, prepared_raw)
    candidates, candidate_digests, construction_digest = _load_candidates(
        arguments.candidate_root,
        candidate_raw,
        arguments.prepared_manifest_sha256,
    )
    try:
        evaluation_batch_size = int(prepared_bindings["evaluation_batch_size"])
        query_block = int(prepared_bindings["query_block"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("prepared evaluation protocol differs") from exc
    evaluator = _CudaBandEvaluator(
        burned,
        evaluation_batch_size=evaluation_batch_size,
        query_block=query_block,
    )
    result = evaluate_cross_seed_denoising(
        candidate_towers=candidates,
        trained_towers=towers,
        trained_heads=heads,
        scalar_curves=scalar_curves,
        candidate_state_sha256=candidate_digests,
        construction_evidence_sha256=construction_digest,
        evaluate_raw=lambda tower: evaluator.evaluate(tower, heads[17], plane="raw"),
        evaluate_projected=lambda tower, head: evaluator.evaluate(
            tower, head, plane="projected"
        ),
    )
    _publish_new(arguments.output, result)
    sys.stdout.buffer.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
