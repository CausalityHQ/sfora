# Pass 204 — corrected CIS closeout

## Verdict

The configured full Coalition Interference Supervision (CIS) arm produced a
notable one-seed In-Shop observation, but **no effect or coalition mechanism is
established**. The original screen is formally **non-adjudicable**: it registered
a “selection-corrected” metric four days after that estimator had already been
retracted. Raw analogues cannot silently replace that metric. The raw full-CIS value and its raw deltas
against the two registered executed arms clear their numerical floors, while the
post hoc closest configured control is only `0.091433` point lower.

This is not evidence that the measurement is useless. It is evidence that the
current experiment cannot attribute the observation to cross-image coalition
supervision because the intended per-image full-union control is absent. The
configured experiment is therefore **closed and invalid for a CIS method,
novelty, or SOTA claim**. The prospectively frozen Pass201 equal-update-norm
operator diagnostic remains the only authorized hill-climb; no additional CIS training
arm, weight sweep, Cars/CUB escalation, novelty claim, or SOTA claim follows
from Pass204.

## Bound artifacts and executed contract

All four corrected arms executed `8,580` steps at seed 0 through objective
`proxy_anchor_coalition`, coalition weight `0.1`, with the intended modes. The
full report and checkpoint are bound by:

- report SHA-256: `fc1f950a23cdd11571571ad102fb68cf6ba1866190eb47a967dc8d47cc069346`;
- checkpoint SHA-256: `1f7068a06309432680e74f4e8702f6d42d79ac321669509d9bb04e7108da81ea`.

Control report SHA-256 values are:

- atomic one-hot: `afb698c4341764da35fc55bed06422ba31a6ba607f653e549740d0ecebc65d1f`;
- atomic complementary: `4483d64d4924431e26d9a5921aa7b205ccaf6b9784a9d421696ebdab2f057e0b`;
- summed dropout: `c84070995acab474b1d53c4b636f19f0858d11eb8228cdcf55d44215778ec23a`.

The earlier derived-objective no-op artifact remains excluded under
`docs/pass120_noop_invalidation_2026-08-08.md`.

## Results

| Arm | Mode | Raw best R@1 | Best epoch | Immutable final R@1 | Local-neighbour trend | Local peak gap |
|---|---|---:|---:|---:|---:|---:|
| atomic one-hot | `single` | 0.916725278 | 53 | 0.913278942 | 0.914492193 | 0.223308 pt |
| atomic complementary | `single_complementary` | 0.918413279 | 41 | 0.915037277 | 0.914474610 | 0.393867 pt |
| summed dropout | `dropout` | 0.916514278 | 55 | 0.914755943 | 0.915441694 | 0.107258 pt |
| full CIS | `union` | **0.919327613** | 55 | **0.917428612** | **0.916619778** | 0.270784 pt |

The local-neighbour values come from `scripts/measure_selection_bias.py` with
the frozen two-neighbour window. They are descriptive curve diagnostics only.
The estimator was retracted as a selection-bias correction in
`docs/selection_bias_estimator_retraction_254_2026-08-03.md`; therefore this
document does **not** report or imply a “selection-corrected” benchmark score.

Raw-best full-CIS deltas are:

- versus atomic one-hot: `+0.260234` point;
- versus summed dropout: `+0.281334` point;
- versus atomic complementary: `+0.091433` point.

Immutable-final deltas are `+0.414967`, `+0.267267`, and `+0.239133` point,
respectively. Final-epoch sensitivity is useful context but was not the frozen
primary control criterion.

## Decision after estimator retraction and source audit

The raw full-CIS value clears the numerical analogue of the `0.9180` headline
floor and lands only `0.017239` point below the registered `0.9195` prediction.
Its raw deltas versus the two registered executed arms—atomic one-hot and summed
dropout—are `+0.260234` and `+0.281334` point, both above `+0.10`. Atomic
complementary is the post hoc closest configured control; its `+0.091433` delta
is descriptive and cannot retrospectively fail the registered threshold.

Pass120 nevertheless registered **selection-corrected** best R@1 after the
estimator had already been retracted. This is a preregistration/spec-process
defect, so the formal numerical screen can be neither passed nor failed. Raw
best, immutable final, and local-neighbour trend
are reported as distinct observables, not substitutions for the invalid metric.

Independently of that non-adjudicable metric, the experiment cannot clear its
mechanism claim:

1. The required per-image full-union/no-coalition control was never implemented.
   `single` is one-hot and `single_complementary` uses `U minus {y_i}`; neither
   applies the full union target to each image independently.
2. Summed modes have an exact `sqrt(m)` auxiliary-gradient coefficient advantage
   over atomic modes under the executed mean reductions. Realized gradients
   need not differ by exactly that factor, but the configured comparison is not
   update-norm matched.
3. `single_complementary` contains no cross-image embedding interaction. Its
   higher raw trajectory can arise from complementary-label/proxy regularization
   without supporting the coalition mechanism.
4. The proxy table is learnable in the auxiliary, so a gain may be proxy-table
   regularization rather than a network-mediated cross-image effect.

These are source-level identities frozen before the full result in
`docs/pass201_cis_operator_contract_audit_2026-08-09.md`. They are not
post-result excuses and they prevent a positive headline from being promoted
to a mechanism or novelty claim.

## Next measured hill-climb

The source-bound fresh ordinary-PA seed-0 control was frozen before launch in
`docs/pass201_pa_source_prelaunch_manifest.json` and was then launched. After it
lands, Pass201 may compare ordinary PA, atomic one-hot, atomic complementary,
the missing per-image full-union control, summed union, and summed dropout on
disjoint-image train contexts at configured-loss and equal-update norms.

Only if summed union retains a prospectively material advantage over per-image
full-union after update-norm matching, with positive held-out owner-margin
action and aligned shared-confuser excess, may a new norm-calibrated method be
preregistered. Otherwise the entire coalition line closes and the next
candidate must be conditioned on the exact Pass200 RSTA or Pass201 failure
phenotype rather than invented in advance.
