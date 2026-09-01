"""Complete-cut, candidate-marginal gradient evidence for ASG-CV."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from sfora.asgcv_protocol import (
    AsgcvCompletionGroup,
    AsgcvMarginalSchedule,
    AsgcvPairSchedule,
)

ASGCV_MARGINAL_GRADIENT_SAMPLE_SCHEMA = "sfora-asgcv-marginal-gradient-sample-v1"
ASGCV_VISION_CUT_SCHEMA = "sfora-asgcv-vision-cut-v1"
ASGCV_VISION_BOUNDARIES = ("merger", "deepstack-0", "deepstack-1", "deepstack-2")
ASGCV_MARGINAL_ARRAY_DOMAIN = b"sfora-asgcv-marginal-gradient-array-v1\0"
ASGCV_REPLAY_BRANCH_COUNT = 8


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ASG-CV marginal {name} differs")
    return value


def _commit(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ASG-CV marginal {name} differs")
    return value


@dataclass(frozen=True, slots=True)
class AsgcvVisionCutAuthority:
    """Exact ordered cut covering every Qwen3-VL vision-to-language path."""

    boundary_names: tuple[str, ...]
    images: int
    patches_per_boundary: int
    channel_dimensions: int

    def validated(self) -> AsgcvVisionCutAuthority:
        if (
            type(self.boundary_names) is not tuple
            or self.boundary_names != ASGCV_VISION_BOUNDARIES
            or type(self.images) is not int
            or self.images != 2
            or type(self.patches_per_boundary) is not int
            or self.patches_per_boundary <= 0
            or type(self.channel_dimensions) is not int
            or self.channel_dimensions <= 0
        ):
            raise ValueError("ASG-CV vision cut authority differs")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_VISION_CUT_SCHEMA,
            "boundary_names": list(self.boundary_names),
            "images": self.images,
            "patches_per_boundary": self.patches_per_boundary,
            "channel_dimensions": self.channel_dimensions,
            "flattened_patches_per_image": (len(self.boundary_names) * self.patches_per_boundary),
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvVisionCutAuthority:
        expected = {
            "schema",
            "boundary_names",
            "images",
            "patches_per_boundary",
            "channel_dimensions",
            "flattened_patches_per_image",
        }
        if (
            type(value) is not dict
            or set(value) != expected
            or value["schema"] != ASGCV_VISION_CUT_SCHEMA
            or type(value["boundary_names"]) is not list
        ):
            raise ValueError("ASG-CV vision cut schema differs")
        authority = cls(
            boundary_names=tuple(value["boundary_names"]),
            images=value["images"],
            patches_per_boundary=value["patches_per_boundary"],
            channel_dimensions=value["channel_dimensions"],
        ).validated()
        if (
            type(value["flattened_patches_per_image"]) is not int
            or value["flattened_patches_per_image"]
            != len(authority.boundary_names) * authority.patches_per_boundary
        ):
            raise ValueError("ASG-CV vision cut flattened shape differs")
        return authority


def _array(value: object, *, role: str, cut: AsgcvVisionCutAuthority) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float32)
        or value.shape
        != (
            cut.images,
            len(cut.boundary_names) * cut.patches_per_boundary,
            cut.channel_dimensions,
        )
        or not bool(np.isfinite(value).all())
    ):
        raise ValueError(f"ASG-CV marginal {role} array differs")
    return np.ascontiguousarray(value)


def _array_authority(value: np.ndarray, *, role: str) -> dict[str, object]:
    frame = bytearray(ASGCV_MARGINAL_ARRAY_DOMAIN)
    encoded_role = role.encode("ascii")
    frame.extend(len(encoded_role).to_bytes(8, "big"))
    frame.extend(encoded_role)
    frame.extend(value.ndim.to_bytes(8, "big"))
    for size in value.shape:
        frame.extend(int(size).to_bytes(8, "big"))
    frame.extend(value.astype(np.dtype("<f4"), copy=False).tobytes(order="C"))
    return {
        "dtype": "float32-le",
        "shape": list(value.shape),
        "sha256": hashlib.sha256(frame).hexdigest(),
    }


def canonical_marginal_gradient_sample_bytes(
    *,
    source_commit: object,
    model_revision: object,
    fixture_sha256: object,
    completion_group_sha256: object,
    completion_protocol_sha256: object,
    marginal_schedule_sha256: object,
    pooler_state_sha256: object,
    candidate_pair_ordinal: object,
    pair_ordinals: object,
    relation_sign: object,
    zero_semantic_target: object,
    grpo_loss: object,
    attention_kl: object,
    generated_tokens: object,
    vision_cut_authority: object,
    patch_tokens: object,
    exact_gradient: object,
) -> bytes:
    """Seal one candidate-ordered complete-cut semantic target."""

    if type(vision_cut_authority) is not AsgcvVisionCutAuthority:
        raise ValueError("ASG-CV marginal vision cut differs")
    cut = vision_cut_authority.validated()
    tokens = _array(patch_tokens, role="patch-token", cut=cut)
    gradient = _array(exact_gradient, role="exact-gradient", cut=cut)
    if type(zero_semantic_target) is not bool:
        raise ValueError("ASG-CV marginal zero-target flag differs")
    if zero_semantic_target is not bool(np.count_nonzero(gradient) == 0):
        raise ValueError("ASG-CV marginal zero-target relation differs")
    if type(candidate_pair_ordinal) is not int or candidate_pair_ordinal < 0:
        raise ValueError("ASG-CV marginal candidate ordinal differs")
    if (
        type(pair_ordinals) is not tuple
        or len(pair_ordinals) != 2
        or any(type(value) is not int or value < 0 for value in pair_ordinals)
        or pair_ordinals[0] == pair_ordinals[1]
    ):
        raise ValueError("ASG-CV marginal pair ordinals differ")
    if type(relation_sign) is not int or relation_sign not in {-1, 1}:
        raise ValueError("ASG-CV marginal relation sign differs")
    if (
        type(grpo_loss) is not float
        or not math.isfinite(grpo_loss)
        or type(attention_kl) is not float
        or not math.isfinite(attention_kl)
        or attention_kl < 0.0
        or type(generated_tokens) is not int
    ):
        raise ValueError("ASG-CV marginal replay evidence differs")
    if zero_semantic_target:
        if grpo_loss != 0.0 or attention_kl != 0.0 or generated_tokens != 0:
            raise ValueError("ASG-CV marginal zero-target evidence differs")
    elif generated_tokens <= 0:
        raise ValueError("ASG-CV marginal generated-token evidence differs")
    payload: dict[str, object] = {
        "schema": ASGCV_MARGINAL_GRADIENT_SAMPLE_SCHEMA,
        "claim_eligible": False,
        "source_commit": _commit(source_commit, name="source commit"),
        "model_revision": _commit(model_revision, name="model revision"),
        "fixture_sha256": _sha256(fixture_sha256, name="fixture digest"),
        "completion_group_sha256": _sha256(completion_group_sha256, name="completion group digest"),
        "completion_protocol_sha256": _sha256(
            completion_protocol_sha256, name="completion protocol digest"
        ),
        "marginal_schedule_sha256": _sha256(
            marginal_schedule_sha256, name="marginal schedule digest"
        ),
        "pooler_state_sha256": _sha256(pooler_state_sha256, name="pooler-state digest"),
        "candidate_pair_ordinal": candidate_pair_ordinal,
        "pair_ordinals": list(pair_ordinals),
        "relation_sign": relation_sign,
        "zero_semantic_target": zero_semantic_target,
        "replay_branch_count": 0 if zero_semantic_target else ASGCV_REPLAY_BRANCH_COUNT,
        "losses": {
            "grpo": grpo_loss,
            "attention_kl": attention_kl,
            "semantic": grpo_loss + attention_kl,
        },
        "generated_tokens": generated_tokens,
        "vision_cut_authority": cut.to_mapping(),
        "arrays": {
            "patch_tokens": _array_authority(tokens, role="patch-tokens"),
            "exact_gradient": _array_authority(gradient, role="exact-gradient"),
        },
    }
    payload["sample_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return _canonical_json_bytes(payload)


def validate_marginal_gradient_sample_bytes(raw: bytes) -> dict[str, object]:
    """Validate canonical marginal receipt bytes and all scalar relations."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV marginal sample is not canonical JSON") from error
    expected = {
        "schema",
        "claim_eligible",
        "source_commit",
        "model_revision",
        "fixture_sha256",
        "completion_group_sha256",
        "completion_protocol_sha256",
        "marginal_schedule_sha256",
        "pooler_state_sha256",
        "candidate_pair_ordinal",
        "pair_ordinals",
        "relation_sign",
        "zero_semantic_target",
        "replay_branch_count",
        "losses",
        "generated_tokens",
        "vision_cut_authority",
        "arrays",
        "sample_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected
        or _canonical_json_bytes(value) != raw
        or value["schema"] != ASGCV_MARGINAL_GRADIENT_SAMPLE_SCHEMA
        or value["claim_eligible"] is not False
    ):
        raise ValueError("ASG-CV marginal sample authority differs")
    _commit(value["source_commit"], name="source commit")
    _commit(value["model_revision"], name="model revision")
    for name in (
        "fixture_sha256",
        "completion_group_sha256",
        "completion_protocol_sha256",
        "marginal_schedule_sha256",
        "pooler_state_sha256",
    ):
        _sha256(value[name], name=name)
    if type(value["candidate_pair_ordinal"]) is not int or value["candidate_pair_ordinal"] < 0:
        raise ValueError("ASG-CV marginal candidate ordinal differs")
    pair_ordinals = value["pair_ordinals"]
    if (
        type(pair_ordinals) is not list
        or len(pair_ordinals) != 2
        or any(type(ordinal) is not int or ordinal < 0 for ordinal in pair_ordinals)
        or pair_ordinals[0] == pair_ordinals[1]
        or type(value["relation_sign"]) is not int
        or value["relation_sign"] not in {-1, 1}
        or type(value["zero_semantic_target"]) is not bool
    ):
        raise ValueError("ASG-CV marginal sample relation differs")
    zero = value["zero_semantic_target"]
    if type(value["replay_branch_count"]) is not int or value["replay_branch_count"] != (
        0 if zero else ASGCV_REPLAY_BRANCH_COUNT
    ):
        raise ValueError("ASG-CV marginal replay count differs")
    losses = value["losses"]
    if type(losses) is not dict or set(losses) != {"grpo", "attention_kl", "semantic"}:
        raise ValueError("ASG-CV marginal loss schema differs")
    grpo = losses["grpo"]
    attention = losses["attention_kl"]
    semantic = losses["semantic"]
    generated = value["generated_tokens"]
    if (
        type(grpo) is not float
        or not math.isfinite(grpo)
        or type(attention) is not float
        or not math.isfinite(attention)
        or attention < 0.0
        or type(semantic) is not float
        or not math.isfinite(semantic)
        or semantic != grpo + attention
        or type(generated) is not int
        or (zero and (grpo != 0.0 or attention != 0.0 or generated != 0))
        or (not zero and generated <= 0)
    ):
        raise ValueError("ASG-CV marginal replay evidence differs")
    cut = AsgcvVisionCutAuthority.from_mapping(value["vision_cut_authority"])
    arrays = value["arrays"]
    if type(arrays) is not dict or set(arrays) != {"patch_tokens", "exact_gradient"}:
        raise ValueError("ASG-CV marginal sample array schema differs")
    expected_shape = [
        cut.images,
        len(cut.boundary_names) * cut.patches_per_boundary,
        cut.channel_dimensions,
    ]
    for authority in arrays.values():
        if (
            type(authority) is not dict
            or set(authority) != {"dtype", "shape", "sha256"}
            or authority["dtype"] != "float32-le"
            or authority["shape"] != expected_shape
        ):
            raise ValueError("ASG-CV marginal sample array authority differs")
        _sha256(authority["sha256"], name="array digest")
    digest = _sha256(value["sample_sha256"], name="sample digest")
    unsigned = dict(value)
    del unsigned["sample_sha256"]
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != digest:
        raise ValueError("ASG-CV marginal sample digest differs")
    return value


def validate_marginal_gradient_sample_inputs(
    raw: bytes,
    *,
    patch_tokens: object,
    exact_gradient: object,
) -> dict[str, object]:
    """Reopen dense evidence and require byte-identical reconstruction."""

    value = validate_marginal_gradient_sample_bytes(raw)
    losses = value["losses"]
    pair_ordinals = value["pair_ordinals"]
    if type(losses) is not dict or type(pair_ordinals) is not list:
        raise ValueError("ASG-CV marginal sample row schema differs")
    rebuilt = canonical_marginal_gradient_sample_bytes(
        source_commit=value["source_commit"],
        model_revision=value["model_revision"],
        fixture_sha256=value["fixture_sha256"],
        completion_group_sha256=value["completion_group_sha256"],
        completion_protocol_sha256=value["completion_protocol_sha256"],
        marginal_schedule_sha256=value["marginal_schedule_sha256"],
        pooler_state_sha256=value["pooler_state_sha256"],
        candidate_pair_ordinal=value["candidate_pair_ordinal"],
        pair_ordinals=tuple(pair_ordinals),
        relation_sign=value["relation_sign"],
        zero_semantic_target=value["zero_semantic_target"],
        grpo_loss=losses["grpo"],
        attention_kl=losses["attention_kl"],
        generated_tokens=value["generated_tokens"],
        vision_cut_authority=AsgcvVisionCutAuthority.from_mapping(value["vision_cut_authority"]),
        patch_tokens=patch_tokens,
        exact_gradient=exact_gradient,
    )
    if rebuilt != raw:
        raise ValueError("ASG-CV marginal sample reopened inputs differ")
    return value


def validate_marginal_gradient_sample_context(
    raw: bytes,
    *,
    marginal_schedule: AsgcvMarginalSchedule,
    candidate_schedule: AsgcvPairSchedule,
    completion_groups: tuple[AsgcvCompletionGroup, ...],
) -> dict[str, object]:
    """Cross-bind one marginal sample to its candidate pair and completion outcome."""

    value = validate_marginal_gradient_sample_bytes(raw)
    if (
        type(marginal_schedule) is not AsgcvMarginalSchedule
        or type(candidate_schedule) is not AsgcvPairSchedule
        or type(completion_groups) is not tuple
    ):
        raise ValueError("ASG-CV marginal sample context differs")
    marginal_schedule.validated()
    candidate_schedule.validated()
    if (
        marginal_schedule.candidate_schedule_sha256 != candidate_schedule.sha256()
        or value["marginal_schedule_sha256"] != marginal_schedule.sha256()
        or len(completion_groups) != candidate_schedule.pair_count
    ):
        raise ValueError("ASG-CV marginal sample context differs")
    candidate_ordinal = value["candidate_pair_ordinal"]
    if (
        type(candidate_ordinal) is not int
        or not 0 <= candidate_ordinal < marginal_schedule.target_pair_count
        or marginal_schedule.candidate_ordinals[candidate_ordinal] != candidate_ordinal
    ):
        raise ValueError("ASG-CV marginal sample candidate context differs")
    pair = candidate_schedule.pairs[candidate_ordinal]
    group = completion_groups[candidate_ordinal]
    if type(group) is not AsgcvCompletionGroup:
        raise ValueError("ASG-CV marginal sample completion context differs")
    group.validated()
    if (
        group.candidate_pair_ordinal != candidate_ordinal
        or group.expected_relation_sign != pair.relation_sign
        or value["completion_group_sha256"] != group.sha256()
        or value["completion_protocol_sha256"] != group.protocol_sha256
        or value["pair_ordinals"] != [pair.left_index, pair.right_index]
        or value["relation_sign"] != pair.relation_sign
        or value["zero_semantic_target"]
        is not marginal_schedule.zero_target_flags[candidate_ordinal]
        or value["zero_semantic_target"] is group.nonzero_reward_variance
    ):
        raise ValueError("ASG-CV marginal sample context differs")
    return value
