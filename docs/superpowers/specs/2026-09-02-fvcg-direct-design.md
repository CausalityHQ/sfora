# FVCG-Direct Design

Date: 2026-09-02  
Status: **PREREGISTERED DESIGN — no FVCG quality run has started**

## Objective

Forced-Verbalizer Collapsed Gradient Direct (FVCG-Direct) replaces sampled
free-form semantic rollouts with the exact gradient of a deterministic two-prefix
vision-language objective.  It trains the Qwen vision tower and retrieval pooler
directly; it does not predict, cache, or reconstruct a dense teacher-gradient
field.  The language tower is frozen and is absent from retrieval serving.

The first objective is a bounded combined-step falsifier.  Only a passing step
may start a three-epoch train-only capacity pilot.  Only a positive capacity
pilot may start a full retrieval run.

## Evidence boundary

The design follows three terminal observations.

1. The sampled eight-completion pilot produced seed-invariant verdicts and no
   mixed-verdict groups.  On the same model and prompt, deterministic forced
   likelihoods for the exact `SAME` and `DIFFERENT` prefixes reached 87.5% balanced
   train-pair accuracy and 0.992188 ROC AUC.  The result is recorded in
   `docs/asgcv_forced_p32_result_2026-09-02.md`.
2. The rank-16 dense-gradient student failed on the disjoint E1 band: median
   cosine -50 ppm and positive-cosine rate 468,750 ppm.  It also failed on its
   training rows.  Individual and shared-subspace diagnostics show that the
   fields are compressible, but their coefficients are not recovered from the
   registered patch-token input by either the nonlinear student or a closed-form
   probe.  That student family is closed.
3. The authenticated GB10 feasibility result at source
   `c5e74aee4136788dab35ac7353a122d2acbc58b5` is `FITS`.  Its canonical file is
   3,249 bytes with SHA-256
   `623c6d541e43ce18c45ff46510ddedc33f7146a92f09b31ce68943c55285edb3`
   and internal result digest
   `26c10627c6bf4f38e2e0d9c81776a525fe86c8ac2d73f836b55355ad49203a87`.
   The 64-image DML backward took 3.020389293 seconds and peaked at
   57,818,480,640 CUDA-reserved bytes.  The rollout-based projected best-case
   step was 467.112409925 seconds.  Separately, the forced two-prefix replay took
   0.614647193 seconds median and 0.618729075 seconds p90 over 32 pairs, with
   22,873,636,864 CUDA-reserved bytes.

All observations are claim ineligible and use Cars training identities only.
The Cars development bands E0 and E1 are burned.  They may diagnose or screen
FVCG, but they cannot independently qualify a final method.

## Scientific claim boundary

FVCG is a new deterministic surrogate, not a claim of byte- or distributional
equivalence to SAGA's sampled continuation policy.  Actual free generation did
not behave as a two-action categorical sampler calibrated by the forced-prefix
scores.  Therefore the paper may say:

- the binary forced objective analytically marginalizes a registered hypothetical
  eight-draw categorical verdict group;
- it is generation free, finite, conservative, and lower variance for that
  objective; and
- it empirically replaces a degenerate sampled semantic path.

It may not say that it is the exact conditional expectation of the unpublished
SAGA implementation.  SAGA remains a capacity-matched empirical comparator.

## Forced-verbalizer objective

For one ordered image pair `X`, relation sign `y in {-1,+1}`, and fixed prefix
token sequences `SAME` and `DIFFERENT`, the frozen language tower returns mean
teacher-forced log-likelihood scores `s_same(X)` and `s_different(X)`.  Define the
correct and incorrect scores by the ground-truth relation:

```text
s_correct   = s_same       if y = +1 else s_different
s_incorrect = s_different  if y = +1 else s_same
gap         = s_correct - s_incorrect
p           = sigmoid(gap)
c(p)        = E[sqrt(M(8-M))/8], M ~ Binomial(8,p)
L_fvcg      = -stop_gradient(c(p)) * gap
```

This is exactly the existing finite scalar authority in
`src/sfora/asgcv_verdict_marginal.py`.  Because `c` is a scalar function of the
gap, the local field is conservative: an antiderivative of `-c(sigmoid(gap))`
defines a scalar potential.  The implementation differentiates `L_fvcg`
directly through both forced branches into the vision tower.  It never
materializes the `[2,256,4096]` field during training.

Every step records both branch scores, `p`, `c(p)`, loss, nonzero finite vision
gradient count, zero language-gradient count, semantic elapsed nanoseconds,
CUDA-reserved peak, process RSS peak, and exact source/model/protocol identities.

## Unbiased pair subsampling

A registered class-balanced DML minibatch defines ordered strata of eight pairs.
The stratum construction and selection seed are frozen before training.  Exactly
one pair `j` is selected uniformly in each stratum and receives `L_fvcg`.  With
`g_i = grad L_fvcg(X_i)`, the selected semantic estimator is `g_j`, and

```text
E_j[g_j | X_0,...,X_7] = (1/8) * sum_i g_i.
```

No inverse-probability factor is added.  The ordinary DML objective is exact for
every image.  Selection is independent of labels beyond the already-fixed
class-balanced stratum, forced scores, gradients, losses, and model state.

## Combined optimizer step

The trainable state is the Qwen vision tower, the existing single-query retrieval
pooler, and the registered PFML proxies.  The language model and LM head are
frozen and must have no gradients.  One step is:

1. run the exact 64-image DML forward and registered PFML loss;
2. backward the DML loss without stepping;
3. run one selected pair through the two forced prefixes;
4. backward `semantic_weight * L_fvcg`, accumulating into vision gradients;
5. validate finite gradients, apply the registered global clip once to the
   combined field, and take one optimizer step;
6. prove that vision, pooler, and proxy state changed while language state did not.

The first implementation uses one selected pair per optimizer step.  More pairs,
attention KL, a student, cached gradient fields, and custom kernels are out of
scope until this exact path produces positive retrieval evidence.

## Phase A — combined-step falsifier

Phase A reuses the authenticated Qwen snapshot, synthetic 64-image fixture, and
completion protocol.  It performs a warm-up followed by three measured combined
steps from identical restored initial state.  A canonical result passes only if:

- all three steps finish with finite nonzero vision, pooler, and proxy gradients;
- language-gradient count is exactly zero and language-state bytes do not change;
- optimizer state and all trainable roles reopen exactly;
- peak CUDA reserved is at most 96 GiB and process RSS at most 96 GiB;
- memory PSI full avg10 remains below the registered stop threshold;
- measured p90 combined-step wall time is at most 15 seconds;
- semantic p90 is at most 2 seconds and produces zero generated tokens;
- repeated restored-state step 0 reproduces branch-score bits, coefficient ppm,
  selected pair, loss bits, gradient digest, and updated-state digest; and
- the direct scalar backward agrees with an independently captured boundary-field
  VJP on every trainable vision parameter within the registered fp32 tolerance.

Any failure closes the direct GB10 implementation before a dataset run.

## Phase B — three-epoch train-only capacity pilot

Phase B uses the burned Cars development surface only and remains
`claim_eligible=false`.  From one byte-identical initialization and batch order it
runs:

1. PFML-only Qwen vision baseline;
2. FVCG-Direct with one selected semantic pair per step; and
3. a compute-matched PFML-only continuation with the same measured wall budget.

Hyperparameters, epoch count, selection stream, evaluation cadence, and semantic
weight are frozen before the first receipt.  The semantic weight is set by a
pre-run gradient-norm ratio on the synthetic fixture, not by retrieval quality:
the median unclipped FVCG vision-gradient norm is scaled to 0.25 times the median
unclipped PFML vision-gradient norm.  The resulting scalar is sealed and unchanged.

The pilot passes only if FVCG:

- improves development Recall@1 by at least 0.5 percentage point over both
  controls at the common three-epoch boundary;
- reduces neither the median per-class Recall@1 nor MAP@R by more than 0.2
  percentage point relative to PFML-only;
- retains a nonzero finite semantic field on at least 95% of steps;
- changes the pre-clip combined gradient direction by a median of at least
  10,000 ppm cosine distance from PFML alone;
- increases clip activation by no more than five percentage points; and
- takes no more than 1.5 times PFML-only wall time per completed step.

No hyperparameter is changed after this result.  Failure stops FVCG.

## Phase C — retrieval evidence

Only a passing Phase B may run one complete Cars seed.  Cars is an exploratory
capacity result because its method-development surfaces are burned.  Go to three
seeds only if the one-seed fixed-epoch result is at least 97.4% Recall@1 and at
least +0.5 point over both matched controls.  Report paired per-query discordance,
fixed-epoch and best-epoch metrics, MAP@R, and complete runtime/memory evidence.

Before opening the one-seed Cars result, freeze a second-dataset protocol.  CUB is
the supported method-specific corroborating dataset, not an untouched repository
holdout, with a dataset-specific prompt asking whether the images show the same
bird species.  CUB must repeat the matched three-arm, three-seed comparison.  A
publishable method claim requires positive mean gain on both datasets and the
repository's normal multiple-seed uncertainty reporting.

## Prior-art boundary

Language Guidance for DML (CVPR 2022) already shows that language semantics can
improve visual metric learning, and continuous/structured supervision predates it.
FVCG's candidate novelty is narrower: query-conditioned, image-pair-specific
forced-verbalizer gradients from a frozen VLM, generation-free semantic training,
and measured replacement of a degenerate rollout path.  General language-guided
DML, verbalizer classification, Rao-Blackwellization, GRPO, or direct gradient-space
objectives are not claimed as new.  A dedicated primary-source collision audit is
required before any novelty statement.

## Artifacts and implementation boundary

- `src/sfora/fvcg_direct.py`: pure authorities, schedules, result arithmetic, and
  canonical receipts.
- `scripts/run_fvcg_direct.py`: local-only Qwen adapter, Phase A/B execution, and
  explicit CLI.  No network capability.
- `tests/test_fvcg_direct.py`: pure contract and mutation tests.
- `tests/test_run_fvcg_direct.py`: fake-model gradient roles, direct/VJP agreement,
  optimizer step, CLI refusal, resume, and result tests.

The implementation reuses the authenticated model loader and forced-verdict math,
but does not add FVCG behavior to `image_end_to_end.py` until Phase A passes.
Phase B integration may then reuse the existing PFML loss through one narrow public
wrapper rather than importing private internals from the 300-KiB training module.

## Stops and ETA

- Spec, TDD plan, and Phase-A implementation: 8--16 hours.
- Phase-A DGX result: less than one hour after a verified commit.
- Phase-B three-arm result: 12--36 hours, bounded by measured Phase-A throughput.
- One complete Cars seed after a pass: 1--3 days.
- Three seeds plus CUB corroboration: 4--10 days after the first positive seed.

These are execution estimates, not promised quality dates.  Authority, memory,
determinism, throughput, or quality failure terminates the branch instead of
starting another unregistered sweep.
