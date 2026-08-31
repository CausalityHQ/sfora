"""Tests for the prospective SigLIP RSTA Stage-A authority."""

from __future__ import annotations

import math
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
    RstaReceiverFields,
    RstaReceiverScore,
    RstaRolePanel,
    RstaStageAConfig,
    contextual_rsta_direction,
    preflight_rsta_jvp_backend,
    proxy_free_margin_direction,
    receiver_rsta_fields,
    rsta_control_binding_bytes,
    rsta_control_directions,
    rsta_stage_a_result_bytes,
    score_rsta_receiver,
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
        primary_ids = {row.example_id for batch in panel.primary_batches for row in batch.rows}
        alternate_ids = {row.example_id for batch in panel.alternate_batches for row in batch.rows}
        support_ids = {
            example_id for support_pair in panel.support_ids_by_label for example_id in support_pair
        }

        assert len(primary_ids) == 240
        assert len(alternate_ids) == 240
        assert primary_ids & alternate_ids == receiver_ids
        assert not support_ids & (primary_ids | alternate_ids)
        assert all(
            receiver.primary_peer_id != receiver.alternate_peer_id for receiver in panel.receivers
        )
        assert all(
            receiver.primary_foreign_labels != receiver.alternate_foreign_labels
            and len(set(receiver.primary_foreign_labels) & set(receiver.alternate_foreign_labels))
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
        selected_microbatch_size=120,
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
        assert b'"selected_microbatch_size":120' in encoded
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
            lambda value: replace(value, selected_microbatch_size=True),
            lambda value: replace(value, selected_microbatch_size=7),
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
    def score(
        *, delta: float, self_desc: float, rho_value: float, log_ratio: float, deranged: float
    ) -> RstaReceiverScore:
        a_self = delta
        a_batch = 0.0
        a_desc = a_self - self_desc
        cos_batch_self = math.sqrt(max(0.0, 1.0 - rho_value * rho_value))
        norm_b = math.exp(log_ratio)
        cross_norm = math.sqrt(norm_b**2 + 1.0 - 2.0 * norm_b * cos_batch_self)
        cross_contribution = (a_batch * norm_b - a_self) / cross_norm
        return RstaReceiverScore(
            a_self=a_self,
            a_batch=a_batch,
            delta=delta,
            a_desc=a_desc,
            self_minus_desc=self_desc,
            cos_batch_self=cos_batch_self,
            rho=rho_value,
            log_ratio=log_ratio,
            cross_contribution=cross_contribution,
            random_a_self=0.0,
            random_a_batch=0.0,
            random_delta=0.0,
            deranged_a_self=deranged,
            deranged_a_batch=0.0,
            deranged_delta=deranged,
            norm_z=1.0,
            norm_dbar=1.0,
            norm_b=norm_b,
            norm_s=1.0,
            norm_q=1.0,
            norm_random_target=1.0,
            norm_deranged_target=1.0,
            batch_radial_fraction=0.0,
            self_radial_fraction=0.0,
            dbar_radial_fraction=0.0,
        )

    rows = []
    for seed_index, seed in enumerate((17, 29, 43)):
        for label in range(49):
            for receiver_rank in (2, 3, 4):
                rows.append(
                    RstaReceiverEvidence(
                        seed=seed,
                        label=label,
                        receiver_id=f"s{seed}-c{label}-r{receiver_rank}",
                        primary=score(
                            delta=seed_deltas[seed_index],
                            self_desc=self_minus_desc,
                            rho_value=rho,
                            log_ratio=abs_log_ratio,
                            deranged=deranged_delta,
                        ),
                        alternate=score(
                            delta=alternate_deltas[seed_index],
                            self_desc=0.0,
                            rho_value=rho,
                            log_ratio=abs_log_ratio,
                            deranged=0.0,
                        ),
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
        rows[0] = replace(rows[0], primary=replace(rows[0].primary, cos_batch_self=float("nan")))
        with pytest.raises(ValueError, match="primitive"):
            summarize_rsta_stage_a(tuple(rows), RstaStageAConfig())

        with pytest.raises(ValueError, match="147 receiver rows per seed"):
            summarize_rsta_stage_a(_evidence()[:-1], RstaStageAConfig())

    def test_rejects_receiver_derived_statistic_drift(self) -> None:
        rows = list(_evidence())
        rows[0] = replace(
            rows[0], primary=replace(rows[0].primary, delta=rows[0].primary.delta + 0.1)
        )

        with pytest.raises(ValueError, match="primitive"):
            summarize_rsta_stage_a(tuple(rows), RstaStageAConfig())

    def test_accepts_cotangent_radial_roundoff_above_one(self) -> None:
        rows = list(_evidence())
        rows[0] = replace(
            rows[0],
            primary=replace(rows[0].primary, dbar_radial_fraction=1.0 + 5.0e-13),
        )

        summarize_rsta_stage_a(tuple(rows), RstaStageAConfig())

    def test_rejects_cross_contribution_drift_from_primitive_geometry(self) -> None:
        rows = list(_evidence())
        rows[0] = replace(
            rows[0],
            primary=replace(
                rows[0].primary,
                cross_contribution=rows[0].primary.cross_contribution + 1.0e-3,
            ),
        )

        with pytest.raises(ValueError, match="primitive"):
            summarize_rsta_stage_a(tuple(rows), RstaStageAConfig())


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
            binding=replace(_binding(), selected_microbatch_size=2),
            alpha=32.0,
            delta=0.1,
        )

        torch.testing.assert_close(evidence.dbar, expected_dbar)
        assert evidence.parameter_names == tuple(name for name, _ in selected)
        for actual, gradient in zip(evidence.parameter_direction, expected_gradients, strict=True):
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
                binding=replace(_binding(), selected_microbatch_size=1),
                alpha=alpha,
                delta=delta,
            )

    def test_replay_execution_uses_only_bound_microbatch_and_frozen_tolerance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sfora.siglip_rsta_stage_a as subject

        model = PooledProxyAnchorModel(
            tower=nn.Linear(2, 3, bias=False),
            input_dimensions=3,
            embedding_dimensions=2,
            class_count=2,
        )
        observed: dict[str, object] = {}
        original = subject.recomputed_proxy_anchor_backward

        def observe(*args, **kwargs):
            observed.update(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(subject, "recomputed_proxy_anchor_backward", observe)
        contextual_rsta_direction(
            model,
            torch.tensor([[0.2, 0.8], [0.8, 0.2]]),
            torch.tensor([0, 1]),
            binding=replace(_binding(), selected_microbatch_size=1),
            alpha=32.0,
            delta=0.1,
        )

        assert observed["microbatch_size"] == 1
        assert observed["score_tolerance"] == 2.0e-5


class TestReceiverRstaFields:
    @pytest.mark.parametrize("backend", ["forward-mode", "double-backward"])
    def test_matrix_free_fields_match_explicit_dense_jacobian(self, backend: str) -> None:
        torch.manual_seed(47)
        model = PooledProxyAnchorModel(
            tower=nn.Sequential(nn.Linear(2, 3, bias=False), nn.Tanh()),
            input_dimensions=3,
            embedding_dimensions=2,
            class_count=2,
        ).float()
        receiver = torch.tensor([[0.4, -0.7]], dtype=torch.float32)
        dbar = torch.tensor([0.3, -0.5], dtype=torch.float32)
        selected = tuple(
            sorted(
                (
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if name.startswith("tower.") or name.startswith("projection.")
                ),
                key=lambda item: item[0],
            )
        )
        direction = tuple(
            torch.linspace(-0.2, 0.3, parameter.numel()).reshape_as(parameter)
            for _, parameter in selected
        )
        flat = torch.cat([parameter.detach().reshape(-1) for _, parameter in selected])
        flat_direction = torch.cat([value.reshape(-1) for value in direction])
        projection_count = model.projection.weight.numel()

        def manual_encode(values: torch.Tensor) -> torch.Tensor:
            projection = values[:projection_count].reshape_as(model.projection.weight)
            tower = values[projection_count:].reshape_as(model.tower[0].weight)
            pooled = torch.tanh(F.linear(receiver.squeeze(0), tower))
            return F.normalize(F.linear(pooled, projection), dim=0)

        jacobian = torch.autograd.functional.jacobian(manual_encode, flat)
        expected_batch = jacobian @ flat_direction
        expected_self = jacobian @ jacobian.T @ dbar

        fields = receiver_rsta_fields(
            model,
            receiver,
            dbar,
            parameter_names=tuple(name for name, _ in selected),
            parameter_direction=direction,
            backend=backend,
        )

        torch.testing.assert_close(fields.descriptor, manual_encode(flat))
        torch.testing.assert_close(fields.batch_motion, expected_batch)
        torch.testing.assert_close(fields.self_motion, expected_self)
        assert abs(float(fields.descriptor @ fields.batch_motion)) <= 2.0e-6
        assert abs(float(fields.descriptor @ fields.self_motion)) <= 2.0e-6
        assert fields.batch_radial_fraction <= 1.0e-3
        assert fields.self_radial_fraction <= 1.0e-3
        assert fields.backend == backend

    def test_rejects_jvp_radial_residual_before_projection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = PooledProxyAnchorModel(
            tower=nn.Linear(2, 3, bias=False),
            input_dimensions=3,
            embedding_dimensions=2,
            class_count=2,
        )
        selected = tuple(
            sorted(
                (
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if name.startswith("tower.") or name.startswith("projection.")
                ),
                key=lambda item: item[0],
            )
        )
        original_jvp = torch.func.jvp

        def radial_jvp(*args, **kwargs):
            output, tangent = original_jvp(*args, **kwargs)
            return output, tangent + 0.01 * output

        monkeypatch.setattr(torch.func, "jvp", radial_jvp)
        with pytest.raises(ValueError, match="radial fraction"):
            receiver_rsta_fields(
                model,
                torch.tensor([[0.2, 0.8]]),
                torch.tensor([0.1, -0.3]),
                parameter_names=tuple(name for name, _ in selected),
                parameter_direction=tuple(torch.ones_like(parameter) for _, parameter in selected),
                backend="forward-mode",
            )

    def test_rejects_projection_only_selected_parameter_tuple(self) -> None:
        model = PooledProxyAnchorModel(
            tower=nn.Identity(),
            input_dimensions=2,
            embedding_dimensions=2,
            class_count=2,
        )
        names = ("projection.weight",)
        direction = (torch.ones_like(model.projection.weight),)

        with pytest.raises(ValueError, match="complete trainable tower"):
            receiver_rsta_fields(
                model,
                torch.tensor([[0.2, 0.8]]),
                torch.tensor([0.1, -0.3]),
                parameter_names=names,
                parameter_direction=direction,
                backend="forward-mode",
            )

    def test_preflight_selects_forward_mode_after_dense_agreement(self) -> None:
        model = PooledProxyAnchorModel(
            tower=nn.Linear(2, 3, bias=False),
            input_dimensions=3,
            embedding_dimensions=2,
            class_count=2,
        )
        selected = tuple(
            sorted(
                (
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if name.startswith("tower.") or name.startswith("projection.")
                ),
                key=lambda item: item[0],
            )
        )
        evidence = preflight_rsta_jvp_backend(
            model,
            torch.tensor([[0.2, 0.8]]),
            torch.tensor([0.1, -0.3]),
            parameter_names=tuple(name for name, _ in selected),
            parameter_direction=tuple(torch.ones_like(parameter) for _, parameter in selected),
        )

        assert evidence.backend == "forward-mode"
        assert evidence.comparison_available is True
        assert evidence.maximum_relative_disagreement <= 1.0e-5

    def test_preflight_selects_registered_fallback_only_on_coverage_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = PooledProxyAnchorModel(
            tower=nn.Linear(2, 3, bias=False),
            input_dimensions=3,
            embedding_dimensions=2,
            class_count=2,
        )
        selected = tuple(
            sorted(
                (
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if name.startswith("tower.") or name.startswith("projection.")
                ),
                key=lambda item: item[0],
            )
        )

        def unsupported(*args, **kwargs):
            raise NotImplementedError("fixture forward AD coverage failure")

        monkeypatch.setattr(torch.func, "jvp", unsupported)
        evidence = preflight_rsta_jvp_backend(
            model,
            torch.tensor([[0.2, 0.8]]),
            torch.tensor([0.1, -0.3]),
            parameter_names=tuple(name for name, _ in selected),
            parameter_direction=tuple(torch.ones_like(parameter) for _, parameter in selected),
        )

        assert evidence.backend == "double-backward"
        assert evidence.comparison_available is False
        assert evidence.forward_error == "NotImplementedError"

    def test_preflight_accepts_runtime_forward_ad_coverage_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = PooledProxyAnchorModel(
            tower=nn.Linear(2, 3, bias=False),
            input_dimensions=3,
            embedding_dimensions=2,
            class_count=2,
        )
        selected = tuple(
            sorted(
                (
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if name.startswith("tower.") or name.startswith("projection.")
                ),
                key=lambda item: item[0],
            )
        )

        def unsupported(*args, **kwargs):
            raise RuntimeError("forward-mode AD not implemented for fixture operator")

        monkeypatch.setattr(torch.func, "jvp", unsupported)
        evidence = preflight_rsta_jvp_backend(
            model,
            torch.tensor([[0.2, 0.8]]),
            torch.tensor([0.1, -0.3]),
            parameter_names=tuple(name for name, _ in selected),
            parameter_direction=tuple(torch.ones_like(parameter) for _, parameter in selected),
        )

        assert evidence.backend == "double-backward"
        assert evidence.forward_error == "RuntimeError"

    def test_receiver_fields_allow_registered_descriptor_transform_tolerance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = PooledProxyAnchorModel(
            tower=nn.Linear(2, 3, bias=False),
            input_dimensions=3,
            embedding_dimensions=2,
            class_count=2,
        )
        selected = tuple(
            sorted(
                (
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if name.startswith("tower.") or name.startswith("projection.")
                ),
                key=lambda item: item[0],
            )
        )
        original_vjp = torch.func.vjp

        def within_tolerance(*args, **kwargs):
            descriptor, pullback = original_vjp(*args, **kwargs)
            return descriptor + torch.tensor([0.0, 1.0e-7]), pullback

        monkeypatch.setattr(torch.func, "vjp", within_tolerance)
        fields = receiver_rsta_fields(
            model,
            torch.tensor([[0.2, 0.8]]),
            torch.tensor([0.1, -0.3]),
            parameter_names=tuple(name for name, _ in selected),
            parameter_direction=tuple(torch.ones_like(parameter) for _, parameter in selected),
            backend="forward-mode",
        )

        assert fields.backend == "forward-mode"

    def test_preflight_rejects_backend_disagreement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = PooledProxyAnchorModel(
            tower=nn.Linear(2, 3, bias=False),
            input_dimensions=3,
            embedding_dimensions=2,
            class_count=2,
        )
        selected = tuple(
            sorted(
                (
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if name.startswith("tower.") or name.startswith("projection.")
                ),
                key=lambda item: item[0],
            )
        )
        original_jvp = torch.func.jvp

        def distorted(*args, **kwargs):
            output, tangent = original_jvp(*args, **kwargs)
            perpendicular = torch.stack((-output[1], output[0]))
            return output, tangent + 0.1 * perpendicular

        monkeypatch.setattr(torch.func, "jvp", distorted)
        with pytest.raises(ValueError, match="backends disagree"):
            preflight_rsta_jvp_backend(
                model,
                torch.tensor([[0.2, 0.8]]),
                torch.tensor([0.1, -0.3]),
                parameter_names=tuple(name for name, _ in selected),
                parameter_direction=tuple(torch.ones_like(parameter) for _, parameter in selected),
            )


class TestRstaOutcomeDirection:
    def test_top32_proxy_free_margin_direction_matches_autograd(self) -> None:
        descriptor = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        positives = F.normalize(
            torch.tensor([[0.9, 0.3, 0.0], [0.8, -0.2, 0.1]], dtype=torch.float64),
            dim=1,
        )
        cosines = torch.linspace(-0.8, 0.8, 40, dtype=torch.float64)
        foreign = torch.stack(
            (
                cosines,
                torch.sqrt(1.0 - cosines.square()),
                torch.zeros_like(cosines),
            ),
            dim=1,
        )
        example_ids = tuple(f"foreign-{index:02d}" for index in range(40))
        role_digests = tuple(f"{index:064x}" for index in range(40))

        evidence = proxy_free_margin_direction(
            descriptor,
            positives,
            foreign,
            foreign_example_ids=example_ids,
            foreign_role_digests=role_digests,
        )

        expected_indices = tuple(range(39, 7, -1))
        variable = descriptor.detach().clone().requires_grad_(True)
        positive_scores = positives @ variable
        negative_scores = foreign[list(expected_indices)] @ variable
        expected_margin = 0.05 * (
            torch.logsumexp(positive_scores / 0.05, dim=0) - math.log(2.0)
        ) - 0.05 * (torch.logsumexp(negative_scores / 0.05, dim=0) - math.log(32.0))
        (gradient,) = torch.autograd.grad(expected_margin, variable)
        expected_direction = gradient - descriptor * torch.dot(descriptor, gradient)
        expected_direction = F.normalize(expected_direction, dim=0)

        assert evidence.selected_foreign_indices == expected_indices
        assert evidence.selected_foreign_ids == tuple(
            example_ids[index] for index in expected_indices
        )
        assert evidence.margin == pytest.approx(float(expected_margin.detach()))
        torch.testing.assert_close(evidence.direction, expected_direction)
        assert abs(float(descriptor @ evidence.direction)) <= 1.0e-12

    def test_receiver_scoring_matches_hand_derived_metrics(self) -> None:
        descriptor = torch.tensor([1.0, 0.0, 0.0])
        batch = torch.tensor([0.0, 3.0, 4.0])
        self_motion = torch.tensor([0.0, 4.0, 3.0])
        dbar = torch.tensor([0.0, 1.0, 1.0])
        outcome = F.normalize(torch.tensor([0.0, 2.0, 1.0]), dim=0)
        random_target = F.normalize(torch.tensor([0.0, -1.0, 2.0]), dim=0)
        deranged = F.normalize(torch.tensor([0.0, 1.0, -2.0]), dim=0)

        fields = RstaReceiverFields(
            descriptor=descriptor,
            batch_motion=batch,
            self_motion=self_motion,
            batch_radial_fraction=2.0e-4,
            self_radial_fraction=3.0e-4,
            backend="forward-mode",
        )
        score = score_rsta_receiver(
            fields=fields,
            dbar=dbar,
            outcome_direction=outcome,
            random_target=random_target,
            deranged_direction=deranged,
        )

        def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
            return float(F.cosine_similarity(left, right, dim=0))

        expected_self = cosine(self_motion, outcome)
        expected_batch = cosine(batch, outcome)
        expected_desc = cosine(dbar, outcome)
        expected_bs = cosine(batch, self_motion)
        assert score.a_self == pytest.approx(expected_self)
        assert score.a_batch == pytest.approx(expected_batch)
        assert score.delta == pytest.approx(expected_self - expected_batch, abs=3.0e-7)
        assert score.a_desc == pytest.approx(expected_desc)
        assert score.self_minus_desc == pytest.approx(expected_self - expected_desc, abs=3.0e-7)
        assert score.cos_batch_self == pytest.approx(expected_bs)
        assert score.rho == pytest.approx(math.sqrt(max(0.0, 1.0 - expected_bs**2)))
        assert score.log_ratio == pytest.approx(0.0)
        assert score.cross_contribution == pytest.approx(cosine(batch - self_motion, outcome))
        assert score.random_delta == pytest.approx(
            cosine(self_motion, random_target) - cosine(batch, random_target)
        )
        assert score.random_a_self == pytest.approx(cosine(self_motion, random_target))
        assert score.random_a_batch == pytest.approx(cosine(batch, random_target))
        assert score.deranged_delta == pytest.approx(
            cosine(self_motion, deranged) - cosine(batch, deranged)
        )
        assert score.deranged_a_self == pytest.approx(cosine(self_motion, deranged))
        assert score.deranged_a_batch == pytest.approx(cosine(batch, deranged))
        assert score.norm_z == pytest.approx(1.0)
        assert score.norm_dbar == pytest.approx(math.sqrt(2.0))
        assert score.norm_b == pytest.approx(5.0)
        assert score.norm_s == pytest.approx(5.0)
        assert score.norm_q == pytest.approx(1.0)
        assert score.norm_random_target == pytest.approx(1.0)
        assert score.norm_deranged_target == pytest.approx(1.0)
        assert score.batch_radial_fraction == pytest.approx(2.0e-4)
        assert score.self_radial_fraction == pytest.approx(3.0e-4)
        assert score.dbar_radial_fraction == pytest.approx(0.0)

    def test_foreign_ties_use_role_digest_then_example_id(self) -> None:
        descriptor = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        positives = F.normalize(
            torch.tensor([[0.8, 0.6, 0.0], [0.8, 0.0, 0.6]], dtype=torch.float64),
            dim=1,
        )
        foreign = F.normalize(torch.tensor([[0.5, -0.8, 0.2]], dtype=torch.float64), dim=1).repeat(
            34, 1
        )
        example_ids = tuple(f"tied-{index:02d}" for index in range(34))
        role_digests = tuple(f"{33 - index:064x}" for index in range(34))

        evidence = proxy_free_margin_direction(
            descriptor,
            positives,
            foreign,
            foreign_example_ids=example_ids,
            foreign_role_digests=role_digests,
        )

        assert evidence.selected_foreign_indices == tuple(range(33, 1, -1))

    def test_rejects_rotation_residual_and_zero_projected_outcome(self) -> None:
        descriptor = torch.tensor([1.0, 0.0, 0.0])
        valid_fields = RstaReceiverFields(
            descriptor=descriptor,
            batch_motion=torch.tensor([0.0, 2.0, 0.0]),
            self_motion=torch.tensor([0.0, 0.0, 1.0]),
            batch_radial_fraction=1.0e-4,
            self_radial_fraction=2.0e-4,
            backend="forward-mode",
        )
        with pytest.raises(ValueError, match="tangent residual"):
            score_rsta_receiver(
                fields=replace(valid_fields, batch_motion=torch.tensor([0.01, 2.0, 0.0])),
                dbar=torch.tensor([4.0, 1.0, 1.0]),
                outcome_direction=F.normalize(torch.tensor([0.0, 1.0, 1.0]), dim=0),
                random_target=torch.tensor([0.0, 1.0, 0.0]),
                deranged_direction=torch.tensor([0.0, 0.0, 1.0]),
            )

        foreign = torch.tensor([[0.0, 1.0, 0.0]]).repeat(32, 1)
        with pytest.raises(ValueError, match="zero projected norm"):
            proxy_free_margin_direction(
                descriptor,
                foreign[:2],
                foreign,
                foreign_example_ids=tuple(f"foreign-{index}" for index in range(32)),
                foreign_role_digests=tuple(f"{index:064x}" for index in range(32)),
            )

    def test_scoring_records_expected_dbar_radial_and_rejects_zero_cross(self) -> None:
        descriptor = torch.tensor([1.0, 0.0, 0.0])
        fields = RstaReceiverFields(
            descriptor=descriptor,
            batch_motion=torch.tensor([0.0, 2.0, 0.0]),
            self_motion=torch.tensor([0.0, 0.0, 1.0]),
            batch_radial_fraction=1.0e-4,
            self_radial_fraction=2.0e-4,
            backend="forward-mode",
        )
        score = score_rsta_receiver(
            fields=fields,
            dbar=torch.tensor([4.0, 1.0, 1.0]),
            outcome_direction=F.normalize(torch.tensor([0.0, 1.0, 1.0]), dim=0),
            random_target=torch.tensor([0.0, 1.0, 0.0]),
            deranged_direction=torch.tensor([0.0, 0.0, 1.0]),
        )

        assert score.dbar_radial_fraction == pytest.approx(4.0 / math.sqrt(18.0))
        assert score.a_desc == pytest.approx(1.0)
        assert score.self_minus_desc == pytest.approx(1.0 / math.sqrt(2.0) - 1.0)
        assert score.log_ratio == pytest.approx(math.log(2.0))

        identical = replace(fields, self_motion=fields.batch_motion)
        with pytest.raises(ValueError, match="zero norm"):
            score_rsta_receiver(
                fields=identical,
                dbar=torch.tensor([4.0, 1.0, 1.0]),
                outcome_direction=F.normalize(torch.tensor([0.0, 1.0, 1.0]), dim=0),
                random_target=torch.tensor([0.0, 1.0, 0.0]),
                deranged_direction=torch.tensor([0.0, 0.0, 1.0]),
            )

    def test_control_directions_are_id_seeded_tangent_and_cyclic(self) -> None:
        descriptors = torch.eye(4, dtype=torch.float32)[:3]
        outcomes = F.normalize(
            torch.tensor(
                [
                    [0.0, 1.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0],
                ]
            ),
            dim=1,
        )

        controls = rsta_control_directions(
            descriptors,
            outcomes,
            receiver_ids=("r0", "r1", "r2"),
        )

        torch.testing.assert_close(
            controls.random_targets[0],
            torch.tensor([0.0, 0.57018137, -0.4013722, 0.71679395]),
        )
        assert torch.allclose(
            torch.linalg.vector_norm(controls.random_targets, dim=1),
            torch.ones(3),
        )
        assert torch.allclose((descriptors * controls.random_targets).sum(dim=1), torch.zeros(3))
        expected_first_deranged = outcomes[1] - descriptors[0] * torch.dot(
            descriptors[0], outcomes[1]
        )
        expected_first_deranged = F.normalize(expected_first_deranged, dim=0)
        torch.testing.assert_close(controls.deranged_directions[0], expected_first_deranged)
        expected_last_deranged = outcomes[0] - descriptors[-1] * torch.dot(
            descriptors[-1], outcomes[0]
        )
        expected_last_deranged = F.normalize(expected_last_deranged, dim=0)
        torch.testing.assert_close(controls.deranged_directions[-1], expected_last_deranged)
        replayed = rsta_control_directions(
            descriptors,
            outcomes,
            receiver_ids=("r0", "r1", "r2"),
        )
        assert torch.equal(replayed.random_targets, controls.random_targets)
        assert torch.equal(replayed.deranged_directions, controls.deranged_directions)

        with pytest.raises(ValueError, match="outcome directions"):
            rsta_control_directions(
                descriptors,
                torch.cat((descriptors[:1], outcomes[1:])),
                receiver_ids=("r0", "r1", "r2"),
            )
