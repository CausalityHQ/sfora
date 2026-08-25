# UniCOM ProxyMuon F0 Design

## Status and purpose

This document preregisters the cheapest decisive test of **ProxyMuon**: a
Muon-style orthogonalized optimizer applied only to UniCOM's class-proxy
matrix. It is a training-time candidate intended to compose with the already
confirmed class-proxy imprinting result. It does not change the encoder,
descriptor, retrieval rule, inference path, or deployment bytes.

The scientific question is narrow:

> After class-proxy imprinting, does matrix-wise orthogonalization move the
> proxy matrix to the same fixed training-only objective substantially faster
> than a same-budget, learning-rate-selected AdamW control?

F0 is a cached-feature falsifier, not a benchmark claim. It must complete
before any full training integration or query/gallery evaluation is written or
run.

## Frozen evidence and routing

The candidate starts from Git commit
`496064ae33e2b94b0fbb5d7b66bb8ff431371b64`, whose final quality--efficiency
report is `reports/similarity_quality_efficiency_final_2026-08-25.md`, SHA-256
`972255e3f447065c83da2c41529e13e08449a0765f5a24bb69b2bfce5ea3cf0d`.

The load-bearing prior evidence is:

- Class-proxy imprinting improves the matched random-proxy UniCOM baseline by
  `+0.026484` mAP@R and `+0.014039` Recall@1 over five prospective paired
  seeds, reaches baseline epoch-16 quality by epoch 8 on all five, and leaves
  inference and deployment storage unchanged.
- The spherical-probe artifact
  `reports/generated/unicom-spherical-probe-ed2e789.json`, SHA-256
  `d1a52703849acb96f359c2c7f209942fcbf6fa770eeaa0ed41d947780d714ddf`,
  shows that the AdamW-trained proxy head remains materially short of its own
  cached-feature objective: mean paired loss deltas are approximately
  `0.2005`, `0.2086`, and `0.2049`, validation accuracy improves by about
  `0.026`, and every one of 64 masks improves for every fit seed.
- That spherical screen closed only its conservative direction constraint:
  fitted heads moved to mean cosine about `0.9475`, below the frozen `0.95`
  floor. Its result explicitly routes future work to a distinct training-time
  mechanism rather than a smaller-step retune of the fitted head.
- CAP F0 is permanently closed as scientifically unresolved after two used
  attempts. Its attempt-2 failure receipt is
  `reports/generated/unicom-cap-f0-515e8cd-attempt2-failure.json`, SHA-256
  `8b636848cd81c8e8cb3dc6f84aff06dd3758437e7c887acc082f5707127660b1`.
  ProxyMuon never imports CAP, never reads a CAP value, and is not a CAP retry.
- The profiled fusible non-backbone fraction is about `0.046%`. A custom CUDA
  kernel is ineligible because even an infinite speedup of that fraction cannot
  materially change end-to-end time-to-quality.

The scientific parent fixes the following tensor hashes from the spherical
screen:

```text
class mean:
d183c0d26d451cc5184f4da0a2112766fb5b32d206ea711011f573b3b4aa9613

AdamW fitted targets, fit seeds 0, 1, 2:
bfabb3159677577cf8e6489a40b4765c4510c07a0c18e9094443a01de4cf244b
a56392a806fcf028876a0d1933c0095a7e20aad46cbb8f84f8c8d96d8468e8cd
c1fe4cb49668e9b02796ca2fe48432518174cb3495cb1970d7e26ee3a187fd8f
```

## Why this candidate

The production trainer currently updates the pretrained backbone and the
single `3200 x 768` class-proxy tensor with one two-group AdamW optimizer. The
proxy rows receive sparse class-specific supervision and independently masked
512-of-768 coordinate gradients. AdamW therefore estimates a separate second
moment from an anisotropic, mostly non-target stream. The spherical result
shows that this is not merely theoretical: the final proxy matrix has a large,
consistent optimization residual.

Muon accumulates momentum and replaces its matrix update with an approximate
polar factor computed by five Newton--Schulz iterations. It is insensitive to
the coordinate-wise second-moment history that motivates the candidate. The
intervention is complementary to imprinting: imprinting chooses the initial
proxy geometry; ProxyMuon changes how that same matrix moves afterward.

This is deliberately contrarian. The original Muon guidance sends classifier
heads to AdamW. Here, however, the classifier rows are the metric proxies
themselves, and the confirmed imprinting result proves their geometry is
load-bearing. The experiment tests this exception; it does not assume it.

## Alternatives rejected or deferred

1. **Longer/full UniCOM training.** Required later for an external 96.7 R@1
   comparison, but it is a scaling/reproduction study rather than a new method
   and costs hundreds of GPU-hours. It is fallback F2, after the cheap
   optimizer question is settled.
2. **Classifier learning-rate tuning alone.** This is the matched fallback F1.
   It receives exactly the same search budget in F0. If selected AdamW matches
   ProxyMuon, the optimizer mechanism closes and the honest result is an LR
   correction for the imprinted initializer.
3. **A stronger backbone or pretraining corpus.** This changes the model/data
   axis and cannot establish a same-protocol method gain.
4. **Hyperbolic or other non-Euclidean geometry.** Earlier passes found no
   Euclidean-inexpressible residual, and the current defect is directly in the
   training dynamics of an otherwise strong Euclidean proxy head.
5. **A custom kernel.** Closed by the measured Amdahl fraction. Kernel work is
   reconsidered only if a later profile places at least 10% of end-to-end
   time-to-quality in one fusible candidate-specific operation.
6. **Initializing from the fitted spherical head.** Ineligible post-result
   threshold relaxation: it directly reuses a head chosen after observing the
   closed direction screen.

## Prior-art boundary

ProxyMuon is not claimed as a novel optimizer primitive. Muon and matrix
orthogonalization are prior art. The contribution under test is the targeted
use of pinned vanilla Muon on a metric-learning class-proxy matrix, composed
with prospectively validated proxy imprinting, under a same-budget Pareto
protocol.

The preregistration must disclose two adverse priors:

- `docs/opus_pass70_none_2026-08-06.md` closed
  Shampoo/K-FAC/Muon as a novelty family.
- `docs/pass79_proposal_gcnp_2026-08-07.md` warned that an optimizer swap may
  be small and disappear against a matched hyperparameter search.

The second warning becomes a decisive kill rule: if same-budget selected AdamW
matches or beats ProxyMuon, the ProxyMuon lane closes without retuning.

Primary implementation references are the official Muon repository and the
pinned PyTorch `torch.optim.Muon` implementation. The runtime's exact
constructor is:

```text
Muon(params, lr=0.001, weight_decay=0.1, momentum=0.95,
     nesterov=True, ns_coefficients=(3.4445, -4.775, 2.0315),
     eps=1e-7, ns_steps=5, adjust_lr_fn=None)
```

## Scope

F0 creates only:

- a small pure decision/optimizer-construction module;
- an optimizer-injectable cached-head fit path that preserves the legacy
  AdamW default byte-for-byte;
- a standalone screen with strict input checks and atomic result publication;
- tests and one Git-bound run configuration.

F0 does **not** modify `scripts/train_unicom_inshop.py`, checkpoints, the
official evaluator, query/gallery data, or deployment code. Full trainer
integration is authorized only by `PROCEED_TRAINING`.

## Fixed scientific inputs

The runner reconstructs the same training-only features and split as the
spherical parent:

- UniCOM revision:
  `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`;
- checkpoint SHA-256:
  `3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`;
- In-Shop partition SHA-256:
  `cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c`;
- holdout fraction `0.2`, holdout seed `0`;
- optimization identities `3200`, optimization images `20650`;
- probe split seed `23000`;
- fitting images `14330`, validation images/classes `3188`;
- FP32 normalized 768-dimensional features;
- ArcFace margin `0.25`, scale `32.0`;
- eight independently sampled 512-of-768 masks;
- batch size `128`, batch stream seed `23001`, mask stream seed `23002`;
- evaluation mask seed `23003` with exactly `64` mask sets;
- diagnostic-panel seed `23004` and parent gradient-audit seed `23005`;
- LR-selection fit seeds exactly `(0, 1, 2)` and fresh Phase-2 gating fit
  seeds exactly `(3, 4, 5)`;
- row norm exactly the reconstructed class-mean row norm, approximately
  `0.27712812921102037`.

The script must reconstruct and compare the four parent tensor hashes before
constructing any F0 cell. Query and gallery records must be rejected if passed
to an F0 loader and must never be opened.

## Initializers

F0 evaluates two initial proxy matrices:

1. `imprinted`: the authenticated class-mean tensor above;
2. `random`: `torch.empty(3200, 768, dtype=torch.float32)` followed by
   `torch.nn.init.normal_(std=0.01, generator=generator)` from a dedicated CPU
   generator seeded by `experiment_stream_seed(fit_seed, 24000)`, followed by
   per-row normalization and scaling to the exact class-mean row norm.

The random arm is mechanism discrimination, not the primary gate. If Muon
helps random initialization but fails the imprinted gate, it is redundant with
the already promoted initializer and the candidate closes.

Each `(initializer, optimizer, learning_rate, fit_seed)` cell starts from a
fresh byte-identical initializer and fresh registered batch/mask streams. No
optimizer state or mutated head may cross a cell boundary.

## Optimizers and common projection

### AdamW control

```python
torch.optim.AdamW(
    [head],
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
)
```

### ProxyMuon candidate

```python
torch.optim.Muon(
    [head],
    lr=learning_rate,
    momentum=0.95,
    nesterov=True,
    ns_coefficients=(3.4445, -4.775, 2.0315),
    eps=1e-7,
    ns_steps=5,
    adjust_lr_fn="match_rms_adamw",
    weight_decay=0.0,
)
```

`match_rms_adamw` is the pinned PyTorch rule
`0.2 * sqrt(max(rows, cols))`; no local reinterpretation is permitted. It does
not prove equal effective step sizes: AdamW's update RMS is data-dependent,
whereas this rule gives the `3200 x 768` Muon matrix an LR multiplier of
`11.313708498984761`. Fairness therefore comes from the same registered grid,
an explicit interior-selection predicate, and the validated AdamW anchor below,
not from an asserted equality of optimizer scales.

Pinned PyTorch `2.12.1` casts Muon's effective momentum update to `bfloat16`
before its five Newton--Schulz iterations and returns that BF16 polar-factor
approximation to the FP32 parameter update. F0 records this asymmetry rather
than calling the entire protocol FP32. At every retained Phase-2 step, a
read-only pre-step trace invokes the exact pinned private Newton--Schulz helper
on the same effective momentum update and records:

```text
update_dtype = "torch.bfloat16"
orthogonalization_residual = ||U.float().T @ U.float() - I_768||_F
```

The trace for retained step zero is `null`; every later retained head records
the update from the optimizer step that produced that head. The trace must not
mutate optimizer state or parameters, and tests must prove the traced and
untraced built-in Muon steps are byte-identical.

F0 also runs a diagnostic `proxy_muon_fp32` sensitivity in Phase 2. It uses the
same selected ProxyMuon LR, momentum, Nesterov rule, coefficients, five
iterations, scaling, and row projection, but performs only the Newton--Schulz
calculation in FP32. It is not eligible to authorize training. It can only
prevent a BF16-attributable result from permanently closing the broader
orthogonalization mechanism and route a new preregistration.

After every optimizer step, both arms apply the parent's exact common row-norm
projection. Nonfinite loss, gradient, momentum, update, parameter, or zero row
norm invalidates the run structurally. Optimizer state must contain exactly one
momentum tensor for Muon and the registered AdamW state for the control.

## Same-budget successive-halving screen

The exact shared LR grid is:

```text
(0.000025, 0.00005, 0.0001, 0.0002, 0.0004)
```

The screen uses the following fixed nested-loop order, with `fit_seed` as the
innermost dimension:

```text
initializer: imprinted, random
optimizer:   adamw, proxy_muon
lr:          ascending registered grid
fit_seed:    0, 1, 2
```

### Phase 1: 64-step selection

Run all `2 x 2 x 5 x 3 = 60` cells for exactly 64 optimizer steps using only
selection seeds `(0, 1, 2)`. For each cell, compute the fixed diagnostic-panel
fitting loss at steps `(0, 64)`. The panel is the arithmetic mean of the exact
Cartesian product of:

- the first four 128-image batches from `padded_epoch_indices` at epoch zero,
  seeded by `experiment_stream_seed(fit_seed, 23004)`; and
- four consecutively sampled eight-shard 512-of-768 mask sets from a dedicated
  generator with the same seed.

Thus every panel contains 16 deterministic batch/mask losses, its first cell is
the parent's original diagnostic, and no optimization-stream state is consumed.
For each `(initializer, optimizer)`, select the LR with the smallest arithmetic
mean step-64 diagnostic fitting loss over seeds `(0, 1, 2)`. Ties are resolved
by the smaller numeric LR. Validation metrics are not computed or inspected in
Phase 1.

Persist `selected_lr_interior` for every `(initializer, optimizer)`; it is true
exactly when the selected LR is neither `0.000025` nor `0.0004`. The random-arm
values are descriptive because no random row enters Phase 2 or any primary
decision. Both imprinted-arm interior predicates are decision-bearing.

### Phase 2: independent 512-step imprinted reruns

Use only fresh gating seeds `(3, 4, 5)` and the imprinted initializer. For each
seed, independently run from initial bytes:

1. the Phase-1-selected AdamW LR;
2. mandatory AdamW `lr=0.0001` (deduplicated when it is already selected);
3. the Phase-1-selected ProxyMuon LR; and
4. the FP32 Newton--Schulz sensitivity at that same ProxyMuon LR.

This is at most `4 x 3 = 12` independent fits and never reuses a Phase-1
selection seed for a decision-bearing comparison. Phase-2 rows are ordered by
gating seed `(3, 4, 5)`, then variant
`(adamw_selected, adamw_anchor_if_distinct, proxy_muon,
proxy_muon_fp32)`.
Retain heads only at steps `(0, 64, 128, 192, 256, 307, 384, 435, 512)`.
Evaluate the fixed 16-cell diagnostic-panel fitting loss at every retained head.
Evaluate the fixed train-identity validation accuracy only at steps
`(307, 435, 512)`.

The AdamW reference for each seed is whichever of the selected AdamW cell and
the mandatory `0.0001` anchor has the lower step-512 diagnostic-panel fitting
loss; an exact tie chooses the smaller numeric LR. Its step-512 validation
accuracy is the corresponding reference accuracy, never independently selected
by accuracy. For ProxyMuon, define
`reach_step` as the first of `(307, 435, 512)` whose diagnostic fitting loss is
less than or equal to that seed's AdamW step-512 loss. If none qualifies,
`reach_step` is the string `">512"`.

Validation noninferiority at a qualifying step means:

```text
proxy_muon_accuracy_at_reach - adamw_accuracy_at_512 >= -0.002
```

No query/gallery metric participates in LR selection, gating, or reporting.

## Decision

The validator evaluates the following exact cascade in order. A structural
failure is never allowed to fall through to a scientific route.

### `STRUCTURAL_FAILURE`

Any nonfinite or malformed tensor/state/metric, runtime mismatch, failed input
or parent-tensor check, elapsed time above `0.75` GPU-hour, peak allocation above
8 GiB, publication failure, or incomplete registered row set. The reduced
failure receipt records the completed-cell count and no scientific status is
inferred from partial rows.

### `UNRESOLVED_LR_BOUNDARY`

No structural predicate failed, but either decision-bearing imprinted
`selected_lr_interior` value is false. This authorizes only a new, reviewed
scale-expansion preregistration; it is neither a ProxyMuon pass nor a permanent
close. A random-arm boundary is reported but cannot trigger this status.

### `PROCEED_TRAINING`

All conditions hold:

- for imprinted initialization, all three ProxyMuon seeds reach by step `307`;
- for those three seeds, validation is noninferior at the first qualifying
  step;
- for imprinted initialization, the selected ProxyMuon step-512 validation
  accuracy is not below the per-seed AdamW reference by more than `0.002` on
  any gating seed;
- no structural/runtime predicate fails.

This is the only status that authorizes full trainer integration and the
preregistered paired confirmation.

### `ROUTE_FP32_ORTHOGONALIZATION`

`PROCEED_TRAINING` does not hold, but the FP32 Newton--Schulz sensitivity meets
the exact imprinted `PROCEED_TRAINING` reach and accuracy predicates against the
same per-seed AdamW references. This does not authorize training. It closes only
the pinned BF16 candidate and requires a new preregistration with a separately
selected FP32 LR before any FP32 optimizer claim or full-training run.

### `ROUTE_MATCHED_LR`

All conditions hold:

- `PROCEED_TRAINING` does not hold;
- `ROUTE_FP32_ORTHOGONALIZATION` does not hold;
- for imprinted initialization, all three ProxyMuon seeds reach by step `435`;
- all three are validation-noninferior at their first qualifying step;
- for imprinted initialization, selected ProxyMuon is not worse than the
  per-seed AdamW reference at step 512 by more than `0.002` validation accuracy
  on any gating seed;
- no structural/runtime predicate fails and both decision-bearing selected LRs
  are interior.

This closes ProxyMuon as the mechanism and routes to F1: a matched-budget
classifier-LR study using AdamW only.

### `CLOSE_PROXY_MUON`

Any other structurally valid, interior-LR result. This includes:

- any imprinted seed with `reach_step=">512"`;
- any validation-accuracy loss greater than `0.002`;
- improvement only under random initialization;
- same-budget selected AdamW matching or beating the candidate under the
  registered reach/noninferiority rule.

No threshold, LR, momentum, Newton--Schulz coefficient, step count, or
initializer may be changed after observing the result. This status permanently
closes the exact pinned PyTorch-2.12.1 BF16 ProxyMuon candidate. It does not
claim that every possible FP32 orthogonalization method has been tested.

## Result contract

The runner publishes either a complete scientific result or a reduced
structural-failure receipt; both are canonical UTF-8 JSON and are disjoint
strict schemas. The complete scientific result has exact ordered top-level
keys:

```text
schema_version
status
authority
runtime
protocol
initializers
phase1
selected_learning_rates
phase2
comparisons
predicates
process
```

Required properties include:

- `schema_version="unicom-proxy-muon-f0-v1"`;
- exact Git source commit and SHA-256 values for the runner, decision module,
  probe module, training primitives, and In-Shop loader;
- exact hashes for the final report, spherical parent artifact, CAP closure
  receipt, checkpoint, partition, and four parent tensors;
- observed Python, PyTorch, NumPy, CUDA, GPU, deterministic flags, and
  `torch.optim.Muon` constructor defaults, plus the observed BF16 update dtype;
- all 60 Phase-1 rows in registered order;
- the four selected LRs, all four interior predicates, and complete tie-break
  evidence;
- all 9--12 Phase-2 rows in registered order, with head hashes, 16 component
  diagnostic losses, their recomputed mean, update dtype, and
  orthogonalization residual for every retained step;
- comparisons, predicates, and status recomputed from rows by the validator;
- elapsed seconds and peak allocated/reserved GPU bytes;
- no NaN, infinity, candidate omission, or arbitrary extra key.

The reduced failure receipt has schema version
`"unicom-proxy-muon-f0-failure-v1"`, status `"STRUCTURAL_FAILURE"`, exact
authority/runtime/protocol fields available before the failure, completed-cell
count and hashes only (never decision metrics from an incomplete row set), the
first error class/message, process evidence, and no complete-result-only keys.
The validator rejects either schema if it contains keys from the other branch.

Validation must recursively recompute the diagnostic-panel membership and
means, LR selection, AdamW anchor choice, reach steps, noninferiority,
predicates, and status from persisted cell rows. Tests must
mutate every decision-bearing scalar while recomputing dependent summaries and
prove the independent rows still reject the artifact.

Publication uses create-exclusive temporary bytes, flush, `fsync`, atomic
rename, directory `fsync`, strict reload, byte equality, and a second full
validation. Existing output or temporary paths are never overwritten.

## Operational model

Git is the code authority. There is no separate handoff-authentication
subsystem.

1. Commit and review the design.
2. Commit and review source/tests.
3. Commit one run config that names the exact source commit and input hashes.
4. `git push`, create a clean detached checkout of the config commit on the
   GPU host, and use `rsync` only for the checkpoint/dataset inputs that are
   outside Git.
5. Verify `git rev-parse HEAD`, `git diff --quiet`, input SHA-256 values, absent
   output/temp, runtime, and idle GPU.
6. Launch one foreground process tree with exact environment and command.
7. Observe that original PID at intervals no longer than 55 seconds, reporting
   phase transitions, liveness, errors, pressure, and terminal status. Never
   start a duplicate process because output is quiet.
8. Copy the canonical result back, validate it independently, write the result
   report, commit, and push.

A failure before any F0 optimizer cell completes may be repaired without
scientific interpretation and rerun from a new reviewed source/config commit.
A failure after any F0 cell completes is outcome-bearing: publish an
outcome-blind failure receipt and do not retry this F0 without a new
preregistration.

## Expected cost and impact

Phase 1 is `60 x 64 = 3,840` optimizer steps. Phase 2 is at most
`12 x 512 = 6,144` steps. The total remains at most 9,984 cached-head steps,
equivalent to 19.5 parent 512-step fits, plus registered panel evaluations. The parent
three-fit screen completed in 239.53 seconds; the budget is therefore capped
at **0.75 GPU-hour** and 8 GiB peak allocated memory. Exceeding either cap is a
structural close, not permission to optimize kernels.

If F0 passes, the predicted full-training effect is `+0.003` to `+0.010`
mAP@R over the imprinted control, baseline epoch-16 quality by epoch 12 or
earlier, step-time and peak-memory ratios at most `1.02`, identical deployment
bytes, and smaller optimizer-state storage. These are predictions, not current
results.

## Post-F0 routing

### Training confirmation after `PROCEED_TRAINING`

Write a separate preregistration before implementation. It will use five
prospective gating seeds plus one sensitivity seed, both arms imprinted, a
same-budget selected classifier LR, official query-weighted mAP@R as primary,
paired Student-t 95% lower bound above zero, at least 4/5 positive gating
seeds, baseline epoch-16 mAP@R reached by epoch 12 on at least 4/5, A-B-B-A
step-time ratio at most `1.02`, peak-memory ratio at most `1.02`, and identical
deployment bytes. Checkpoint bytes are not required to match because Muon and
AdamW have different optimizer state.

### F1 after `ROUTE_MATCHED_LR`

Run an AdamW-only matched classifier-LR confirmation. The claim becomes that
the inherited UniCOM classifier LR is misspecified for imprinted proxies, not
that Muon is better.

### Precision or LR-scale routes

`UNRESOLVED_LR_BOUNDARY` requires a new grid-expansion preregistration.
`ROUTE_FP32_ORTHOGONALIZATION` requires a new FP32-specific LR selection and
falsifier. Neither status authorizes full training or permits alteration of the
completed F0 result.

### F2 external-anchor reproduction

Only after F0/F1 selects the training rule, reproduce the official full-
identity, 128-epoch UniCOM evaluation geometry and compare random versus
imprinted initialization over at least four paired seeds. This is the route to
a defensible 96.7 external-anchor comparison; the current 16-epoch 3,200-
identity protocol is not directly comparable.

### Closure

On `CLOSE_PROXY_MUON`, do not retune Muon. Continue to the already designed
full-width objective or cross-dataset imprinting replication. Hyperbolic
geometry, CAP, and custom kernels remain closed unless new measured evidence
invalidates their specific prior kill conditions.

## Acceptance criteria before execution

- The design and implementation plan have independent Claude review with no
  Critical or Important findings.
- The AdamW default path reproduces the four registered parent tensor hashes.
- Unit tests cover the pinned Muon call, state isolation, row projection,
  BF16 trace noninterference, FP32 sensitivity isolation, diagnostic-panel
  construction, Phase-1 selection, AdamW anchor selection, tie breaks, reach
  logic, all decision statuses, strict
  recursive schema, mutation matrix, atomic no-clobber, and query/gallery
  sentinels.
- Focused tests, repository-wide tests, Ruff, pycompile, and diff checks pass.
- The source and run-config commits are pushed and the GPU checkout is clean.
- One original process is actively monitored to a terminal result.

## References

- Keller Jordan, [Muon: An optimizer for hidden layers in neural
  networks](https://github.com/KellerJordan/muon).
- PyTorch, [`torch.optim.Muon`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html).
- An et al., [UniCOM: Universal and Compact Representation Learning for Image
  Retrieval](https://arxiv.org/abs/2304.05884).
