from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from sfora.asgcv import canonical_gradient_sample_bytes
from sfora.asgcv_protocol import (
    AsgcvCompletionProtocol,
    AsgcvRolloutAuthority,
    assemble_asgcv_eligible_schedule,
    build_asgcv_pair_schedule,
    classify_asgcv_completion_group,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_asgcv_e0.py"
_SPEC = importlib.util.spec_from_file_location("run_asgcv_e0_subject", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _sample(ordinal: int) -> tuple[bytes, np.ndarray, np.ndarray]:
    patch = np.full((2, 49, 4), ordinal + 1, dtype=np.float32)
    gradient = patch * np.float32(0.25)
    receipt = canonical_gradient_sample_bytes(
        source_commit="1" * 40,
        model_revision="2" * 40,
        fixture_sha256="3" * 64,
        completion_group_sha256=f"{ordinal + 1:064x}",
        completion_protocol_sha256="4" * 64,
        eligible_schedule_sha256="5" * 64,
        pooler_state_sha256="6" * 64,
        eligible_pair_ordinal=ordinal,
        candidate_pair_ordinal=ordinal,
        pair_ordinals=(ordinal * 2, ordinal * 2 + 1),
        relation_sign=1 if ordinal % 2 == 0 else -1,
        grpo_loss=0.0,
        attention_kl=0.0,
        generated_tokens=8,
        patch_tokens=patch,
        exact_gradient=gradient,
    )
    return receipt, patch, gradient


def test_capture_triples_are_atomic_idempotent_and_resume_from_first_absent(tmp_path: Path) -> None:
    for ordinal in range(2):
        receipt, patch, gradient = _sample(ordinal)
        assert _MODULE.write_capture_triple(
            tmp_path,
            ordinal=ordinal,
            receipt=receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
        ) == ("written" if ordinal == 0 else "written")
        assert not tuple(tmp_path.glob("*.partial"))
    assert _MODULE.validated_capture_prefix(tmp_path, expected_count=4) == 2

    receipt, patch, gradient = _sample(1)
    assert (
        _MODULE.write_capture_triple(
            tmp_path,
            ordinal=1,
            receipt=receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
        )
        == "reused"
    )


def test_capture_resume_rejects_partial_gap_corruption_and_shape_drift(tmp_path: Path) -> None:
    receipt, patch, gradient = _sample(0)
    _MODULE.write_capture_triple(
        tmp_path,
        ordinal=0,
        receipt=receipt,
        patch_tokens=patch,
        exact_gradient=gradient,
    )
    np.save(tmp_path / "patch-000001.npy", patch, allow_pickle=False)
    with pytest.raises(ValueError, match="partial"):
        _MODULE.validated_capture_prefix(tmp_path, expected_count=4)

    (tmp_path / "patch-000001.npy").unlink()
    np.save(tmp_path / "gradient-000000.npy", gradient + np.float32(1.0), allow_pickle=False)
    with pytest.raises(ValueError):
        _MODULE.validated_capture_prefix(tmp_path, expected_count=4)

    other = tmp_path / "other"
    other.mkdir()
    bad_patch = np.zeros((2, 48, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        _MODULE.write_capture_triple(
            other,
            ordinal=0,
            receipt=receipt,
            patch_tokens=bad_patch,
            exact_gradient=np.zeros_like(bad_patch),
        )

    receipt_one, patch_one, gradient_one = _sample(1)
    with pytest.raises(ValueError, match="ordinal"):
        _MODULE.write_capture_triple(
            other,
            ordinal=0,
            receipt=receipt_one,
            patch_tokens=patch_one,
            exact_gradient=gradient_one,
        )
    assert not tuple(other.iterdir())


def _protocol_bundle() -> tuple[object, ...]:
    protocol = AsgcvCompletionProtocol(
        same_prefix_ids=(11, 12),
        different_prefix_ids=(21, 22),
        terminal_token_ids=(99,),
    ).validated()
    rollout = AsgcvRolloutAuthority(
        master_seed_sha256="8" * 64,
        model_revision="2" * 40,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=128,
    ).validated()
    example_ids = tuple(f"cars-{index:02d}" for index in range(32))
    labels = tuple(index // 4 for index in range(32))
    candidates = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256="9" * 64,
        pair_count=16,
    )
    groups = tuple(
        classify_asgcv_completion_group(
            tuple(
                (
                    *((11, 12) if pair.relation_sign == 1 else (21, 22)),
                    30 + rollout_ordinal,
                    99,
                )
                if rollout_ordinal < 4
                else (
                    *((21, 22) if pair.relation_sign == 1 else (11, 12)),
                    30 + rollout_ordinal,
                    99,
                )
                for rollout_ordinal in range(8)
            ),
            pair.relation_sign,
            protocol,
            rollout_authority=rollout,
            candidate_pair_ordinal=pair.ordinal,
        )
        for pair in candidates.pairs
    )
    eligible = assemble_asgcv_eligible_schedule(candidates, groups, target_pair_count=8)
    return protocol, rollout, example_ids, labels, candidates, groups, eligible


def test_capture_schedule_validates_context_and_skips_authenticated_prefix(tmp_path: Path) -> None:
    protocol, rollout, example_ids, labels, candidates, groups, eligible = _protocol_bundle()
    calls: list[int] = []

    def capture_one(
        eligible_ordinal: int, candidate_ordinal: int
    ) -> tuple[bytes, np.ndarray, np.ndarray]:
        calls.append(eligible_ordinal)
        pair = candidates.pairs[candidate_ordinal]
        group = groups[candidate_ordinal]
        patch = np.full((2, 49, 4), eligible_ordinal + 1, dtype=np.float32)
        gradient = patch * np.float32(0.25)
        receipt = canonical_gradient_sample_bytes(
            source_commit="1" * 40,
            model_revision="2" * 40,
            fixture_sha256="3" * 64,
            completion_group_sha256=group.sha256(),
            completion_protocol_sha256=protocol.sha256(),
            eligible_schedule_sha256=eligible.sha256(),
            pooler_state_sha256="6" * 64,
            eligible_pair_ordinal=eligible_ordinal,
            candidate_pair_ordinal=candidate_ordinal,
            pair_ordinals=(pair.left_index, pair.right_index),
            relation_sign=pair.relation_sign,
            grpo_loss=0.0,
            attention_kl=0.0,
            generated_tokens=8,
            patch_tokens=patch,
            exact_gradient=gradient,
        )
        return receipt, patch, gradient

    assert (
        _MODULE.capture_schedule(
            tmp_path,
            protocol=protocol,
            rollout_authority=rollout,
            candidate_schedule=candidates,
            completion_groups=groups,
            eligible_schedule=eligible,
            example_ids=example_ids,
            labels=labels,
            capture_one=capture_one,
        )
        == 8
    )
    assert calls == list(range(8))
    assert (
        _MODULE.capture_schedule(
            tmp_path,
            protocol=protocol,
            rollout_authority=rollout,
            candidate_schedule=candidates,
            completion_groups=groups,
            eligible_schedule=eligible,
            example_ids=example_ids,
            labels=labels,
            capture_one=lambda *_: pytest.fail("authenticated prefix must not recapture"),
        )
        == 8
    )
