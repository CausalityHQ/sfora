"""Tests for the prospective SigLIP RSTA Stage-A authority."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sfora.siglip_proxy_control import PooledProxyAnchorModel
from sfora.siglip_rsta_stage_a import (
    RstaCheckpointBinding,
    RstaControlBinding,
    RstaReceiverEvidence,
    RstaRolePanel,
    RstaStageAConfig,
    contextual_rsta_direction,
    rsta_control_binding_bytes,
    rsta_stage_a_result_bytes,
    select_rsta_roles,
    summarize_rsta_stage_a,
)
from sfora.token_set_proxy_anchor import proxy_anchor_loss

PRIMARY_CLASS_ORDER = (
    43,
    45,
    29,
    7,
    46,
    8,
    44,
    13,
    26,
    25,
    24,
    0,
    27,
    40,
    38,
    36,
    4,
    17,
    11,
    3,
    42,
    30,
    41,
    32,
    20,
    34,
    47,
    5,
    18,
    14,
    39,
    31,
    37,
    48,
    21,
    1,
    9,
    2,
    10,
    6,
    19,
    12,
    23,
    16,
    28,
    33,
    35,
    15,
    22,
)

ALTERNATE_CLASS_ORDER = (
    18,
    32,
    20,
    7,
    36,
    17,
    12,
    38,
    46,
    48,
    34,
    33,
    45,
    15,
    39,
    10,
    2,
    9,
    5,
    31,
    43,
    27,
    6,
    41,
    26,
    23,
    25,
    35,
    16,
    0,
    44,
    21,
    37,
    14,
    19,
    29,
    4,
    3,
    24,
    40,
    30,
    11,
    28,
    22,
    47,
    13,
    8,
    42,
    1,
)

CLASS_ZERO_ROLE_ORDER = (
    "class-00-row-01",
    "class-00-row-04",
    "class-00-row-11",
    "class-00-row-03",
    "class-00-row-02",
    "class-00-row-00",
    "class-00-row-10",
    "class-00-row-07",
    "class-00-row-12",
    "class-00-row-06",
    "class-00-row-08",
    "class-00-row-05",
    "class-00-row-14",
    "class-00-row-13",
    "class-00-row-09",
)


def _examples(rows_per_class: int = 15) -> list[tuple[str, int]]:
    return [
        (f"class-{label:02d}-row-{row:02d}", label)
        for label in range(49)
        for row in range(rows_per_class)
    ]


class TestRstaRoles:
    def test_selects_exact_disjoint_primary_and_alternate_panels(self) -> None:
        panel = select_rsta_roles(_examples())

        assert isinstance(panel, RstaRolePanel)
        assert panel.primary_class_order == PRIMARY_CLASS_ORDER
        assert panel.alternate_class_order == ALTERNATE_CLASS_ORDER
        assert panel.ranked_ids_by_label[0] == CLASS_ZERO_ROLE_ORDER
        assert tuple(len(batch.rows) for batch in panel.primary_batches) == (120, 120)
        assert tuple(len(batch.rows) for batch in panel.alternate_batches) == (120, 120)
        assert len(panel.receivers) == 147

        receiver_ids = {receiver.example_id for receiver in panel.receivers}
        primary_ids = {
            row.example_id for batch in panel.primary_batches for row in batch.rows
        }
        alternate_ids = {
            row.example_id for batch in panel.alternate_batches for row in batch.rows
        }
        support_ids = {
            example_id
            for support_pair in panel.support_ids_by_label
            for example_id in support_pair
        }

        assert len(primary_ids) == 240
        assert len(alternate_ids) == 240
        assert primary_ids & alternate_ids == receiver_ids
        assert not support_ids & (primary_ids | alternate_ids)
        assert all(
            receiver.primary_peer_id != receiver.alternate_peer_id
            for receiver in panel.receivers
        )
        assert all(
            receiver.primary_foreign_labels != receiver.alternate_foreign_labels
            and len(
                set(receiver.primary_foreign_labels)
                & set(receiver.alternate_foreign_labels)
            )
            <= 22
            for receiver in panel.receivers
        )

    def test_selection_is_independent_of_manifest_input_order(self) -> None:
        examples = _examples()
        shuffled = examples.copy()
        random.Random(991).shuffle(shuffled)

        assert select_rsta_roles(shuffled) == select_rsta_roles(examples)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda rows: rows[:-1], "at least 15"),
            (lambda rows: rows + [rows[0]], "duplicate example_id"),
            (
                lambda rows: [item for item in rows if item[1] != 48],
                "exact optimization labels",
            ),
            (
                lambda rows: rows + [("forbidden", 49)],
                "exact optimization labels",
            ),
        ],
    )
    def test_rejects_role_authority_drift(self, mutate, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            select_rsta_roles(mutate(_examples()))


def _binding() -> RstaControlBinding:
    return RstaControlBinding(
        schema="rsta-control-binding-v1",
        claim_eligible=False,
        control_complete=True,
        source_commit="a" * 40,
        config_sha256="b" * 64,
        run_authority_sha256="c" * 64,
        dataset_id="tanganke/stanford_cars",
        dataset_revision="d" * 40,
        environment_sha256="e" * 64,
        optimization_manifest_sha256="f" * 64,
        checkpoints=tuple(
            RstaCheckpointBinding(seed=seed, sha256=str(index) * 64, byte_length=1_000)
            for index, seed in enumerate((17, 29, 43), start=1)
        ),
    )


class TestRstaControlBinding:
    def test_serializes_only_outcome_blind_control_authority(self) -> None:
        encoded = rsta_control_binding_bytes(_binding())

        assert encoded.endswith(b"\n") and not encoded.endswith(b"\n\n")
        assert b'"schema":"rsta-control-binding-v1"' in encoded
        assert b'"claim_eligible":false' in encoded
        assert b'"control_complete":true' in encoded
        assert b'"optimization_manifest_sha256":"' in encoded
        for forbidden in (b"accuracy", b"loss", b"threshold", b"verdict"):
            assert forbidden not in encoded

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda value: replace(value, schema="rsta-control-binding-v2"),
            lambda value: replace(value, claim_eligible=0),
            lambda value: replace(value, control_complete=1),
            lambda value: replace(value, source_commit="a" * 39),
            lambda value: replace(value, optimization_manifest_sha256="f" * 63),
            lambda value: replace(value, checkpoints=value.checkpoints[:2]),
            lambda value: replace(
                value,
                checkpoints=(
                    replace(value.checkpoints[0], seed=18),
                    *value.checkpoints[1:],
                ),
            ),
            lambda value: replace(
                value,
                checkpoints=(
                    replace(value.checkpoints[0], byte_length=True),
                    *value.checkpoints[1:],
                ),
            ),
        ],
    )
    def test_rejects_binding_schema_type_and_identity_drift(self, mutate) -> None:
        with pytest.raises(ValueError, match="control binding"):
            rsta_control_binding_bytes(mutate(_binding()))


def _evidence(
    *,
    seed_deltas: tuple[float, float, float] = (0.05, 0.05, 0.05),
    self_minus_desc: float = 0.02,
    rho: float = 0.30,
    abs_log_ratio: float = 0.20,
    deranged_delta: float = 0.0,
    alternate_deltas: tuple[float, float, float] = (0.02, 0.02, 0.02),
) -> tuple[RstaReceiverEvidence, ...]:
    rows = []
    for seed_index, seed in enumerate((17, 29, 43)):
        for label in range(49):
            for receiver_rank in (2, 3, 4):
                rows.append(
                    RstaReceiverEvidence(
                        seed=seed,
                        label=label,
                        receiver_id=f"s{seed}-c{label}-r{receiver_rank}",
                        delta=seed_deltas[seed_index],
                        self_minus_desc=self_minus_desc,
                        rho=rho,
                        abs_log_ratio=abs_log_ratio,
                        deranged_delta=deranged_delta,
                        alternate_delta=alternate_deltas[seed_index],
                    )
                )
    return tuple(rows)


class TestRstaGates:
    def test_passes_only_when_every_registered_gate_passes(self) -> None:
        result = summarize_rsta_stage_a(_evidence(), RstaStageAConfig())

        assert result.verdict == "PASS_ONWARD"
        assert result.first_decisive_clause == "all-pass-gates"
        assert result.pooled_delta == pytest.approx(0.05)
        assert result.bootstrap_delta_95_lower > 0.0
        assert result.bootstrap_self_minus_desc_95_lower > 0.0
        assert result.seed_deltas == (0.05, 0.05, 0.05)
        assert result.alternate_seed_deltas == (0.02, 0.02, 0.02)
        assert result.median_rho == pytest.approx(0.30)
        assert result.median_abs_log_ratio == pytest.approx(0.20)

    @pytest.mark.parametrize(
        ("rows", "verdict", "clause"),
        [
            (
                _evidence(seed_deltas=(0.05, 0.05, 0.019)),
                "UNRESOLVED",
                "no-decisive-clause",
            ),
            (
                _evidence(seed_deltas=(-0.01, -0.01, 0.05)),
                "UNRESOLVED",
                "no-decisive-clause",
            ),
            (
                _evidence(seed_deltas=(-0.01, -0.01, -0.01)),
                "FAIL",
                "pooled-delta-nonpositive",
            ),
            (_evidence(rho=0.05), "FAIL", "median-rho-below-fail-floor"),
            (
                _evidence(alternate_deltas=(0.0, 0.0, 0.0)),
                "FAIL",
                "alternate-pooled-delta-nonpositive",
            ),
            (
                _evidence(self_minus_desc=-0.01),
                "UNRESOLVED",
                "no-decisive-clause",
            ),
        ],
    )
    def test_applies_frozen_pass_fail_precedence(
        self,
        rows: tuple[RstaReceiverEvidence, ...],
        verdict: str,
        clause: str,
    ) -> None:
        result = summarize_rsta_stage_a(rows, RstaStageAConfig())

        assert result.verdict == verdict
        assert result.first_decisive_clause == clause

    def test_result_bytes_are_canonical_claim_ineligible_and_recomputed(self) -> None:
        result = summarize_rsta_stage_a(_evidence(), RstaStageAConfig())

        encoded = rsta_stage_a_result_bytes(result)

        assert encoded.endswith(b"\n") and not encoded.endswith(b"\n\n")
        assert b'"claim_eligible":false' in encoded
        assert b'"schema":"siglip-rsta-stage-a-result-v1"' in encoded
        assert b'"verdict":"PASS_ONWARD"' in encoded

        with pytest.raises(ValueError, match="recomputed evidence"):
            rsta_stage_a_result_bytes(replace(result, verdict="FAIL"))

    @pytest.mark.parametrize(
        "mutation",
        [
            {"seeds": (17, 29, 44)},
            {"bootstrap_replicates": 9_999},
            {"bootstrap_seed": 201},
            {"pass_delta": 0.02},
            {"pass_seed_delta": 0.019},
            {"pass_rho": 0.19},
            {"fail_rho": 0.11},
            {"pass_abs_log_ratio": 0.10},
            {"max_abs_deranged_delta": 0.02},
        ],
    )
    def test_config_rejects_gate_drift(self, mutation: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="frozen contract"):
            RstaStageAConfig(**mutation)

    def test_rejects_nonfinite_or_incomplete_receiver_evidence(self) -> None:
        rows = list(_evidence())
        rows[0] = RstaReceiverEvidence(**{**rows[0].__dict__, "rho": float("nan")})
        with pytest.raises(ValueError, match="finite"):
            summarize_rsta_stage_a(tuple(rows), RstaStageAConfig())

        with pytest.raises(ValueError, match="147 receiver rows per seed"):
            summarize_rsta_stage_a(_evidence()[:-1], RstaStageAConfig())


class TestContextualRstaDirection:
    def test_matches_direct_descriptor_cotangent_and_unclipped_gradient(self) -> None:
        torch.manual_seed(31)
        model = PooledProxyAnchorModel(
            tower=nn.Linear(3, 4, bias=False),
            input_dimensions=4,
            embedding_dimensions=2,
            class_count=3,
        ).double()
        inputs = torch.tensor(
            [
                [0.2, -0.3, 0.5],
                [0.7, 0.1, -0.2],
                [-0.4, 0.9, 0.3],
                [0.6, -0.8, 0.2],
                [-0.1, 0.4, 0.8],
                [0.3, 0.5, -0.7],
            ],
            dtype=torch.float64,
        )
        labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64)

        descriptors = model.encode(inputs)
        normalized_proxies = F.normalize(model.proxies.float(), dim=1)
        scores = descriptors @ normalized_proxies.T
        loss = proxy_anchor_loss(scores, labels, alpha=32.0, delta=0.1)
        expected_dbar = -torch.autograd.grad(loss, descriptors, retain_graph=True)[0]
        selected = tuple(
            sorted(
                (
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if not name.startswith("proxies")
                ),
                key=lambda item: item[0],
            )
        )
        expected_gradients = torch.autograd.grad(
            loss,
            tuple(parameter for _, parameter in selected),
        )
        model.zero_grad(set_to_none=True)

        evidence = contextual_rsta_direction(
            model,
            inputs,
            labels,
            microbatch_size=2,
            alpha=32.0,
            delta=0.1,
            score_tolerance=1.0e-12,
        )

        torch.testing.assert_close(evidence.dbar, expected_dbar)
        assert evidence.parameter_names == tuple(name for name, _ in selected)
        for actual, gradient in zip(
            evidence.parameter_direction, expected_gradients, strict=True
        ):
            torch.testing.assert_close(actual, -gradient)
        assert all(parameter.grad is None for parameter in model.parameters())
        assert "proxies" not in " ".join(evidence.parameter_names)
        assert evidence.maximum_score_disagreement <= 1.0e-12

    @pytest.mark.parametrize(
        ("alpha", "delta"),
        [(31.0, 0.1), (32.0, 0.2), (32, 0.1), (32.0, True)],
    )
    def test_rejects_proxy_anchor_operator_drift(
        self,
        alpha: object,
        delta: object,
    ) -> None:
        model = PooledProxyAnchorModel(
            tower=nn.Linear(2, 3, bias=False),
            input_dimensions=3,
            embedding_dimensions=2,
            class_count=2,
        )
        with pytest.raises(ValueError, match="frozen Proxy Anchor operator"):
            contextual_rsta_direction(
                model,
                torch.ones((2, 2)),
                torch.tensor([0, 1]),
                microbatch_size=1,
                alpha=alpha,
                delta=delta,
                score_tolerance=0.0,
            )
