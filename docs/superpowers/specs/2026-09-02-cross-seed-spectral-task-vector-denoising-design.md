# Cross-seed Wiener and spectral task-vector denoising

Date: 2026-09-02
Status: prospective design; Cars evidence is claim-ineligible

## Decision

After the scalar weight-space transfer diagnostic, evaluate two fixed,
outcome-blind estimators of the transferable tower update. Both use the three
independently trained seed checkpoints as stochastic replicates. The primary
method applies a closed-form tensorwise Wiener gain derived from cross-seed task
vector agreement. The secondary keeps only mean-task singular modes that rise
above the measured cross-seed contrast edge. Neither has a tuned scalar.

This specification is frozen before the scalar alpha curves exist. It therefore
prospectively supersedes the earlier design's outcome-dependent funding clause:
the cross-seed diagnostic runs after the scalar control whether or not the
scalar control finds an interior alpha. The scalar outcome remains the primary
comparator and cannot change this method's equations, controls, or gates.

The scientific authority requires exactly the completed seed order
`(17, 29, 43)`. There is deliberately no two-seed estimator: three replicates
are required to estimate a mean update and two independent noise directions.
If any seed, checkpoint, or final seed receipt is absent or invalid, the run
terminates `authority-failure`; it never substitutes a provisional two-seed
formula.

The deployed object is an ordinary folded model state. It adds no inference
branch, parameters, memory, or latency.

## Existing evidence and target

The fixed control is 1242/1345, or 92.3420%, on the burned Cars band. The two
completed trained seeds score 1258/1345 and 1248/1345, a 93.1599% mean. These
are trained-endpoint results, not scalar-interpolation results. The learned
512-dimensional projection does not change which queries are correct relative
to the raw 1152-dimensional tower on the completed endpoints. Training improves
transfer, but optimization gains on classes 0--48 are much larger than gains on
classes 82--97 and vary materially by seed.

The immediate engineering target is a stable improvement over the scalar
control, not a declaration that weight denoising alone will reach 95%. Reaching
95% requires 1278/1345 correct, twenty more than the current best completed
seed. Fourteen known cross-label exact-duplicate queries impose about a
1.04-point deterministic pixel ambiguity floor, so 95% is not ruled out by the
data audit. It is nevertheless an ambitious weight-only target.

## Why cross-seed denoising

All three runs begin from the same pretrained vision tower and use the same
data, objective, schedule, and architecture. Their random projection/proxy
initializations and sampling order perturb the gradients. A tower update that
appears consistently across all three runs is therefore a stronger candidate
for transferable structure than an equally large update that changes with the
seed-specific head gauge or sampler path.

All endpoints fine-tune the same named tensors from a byte-identical initial
tower without architecture surgery, reinitialization, or permutation. This
shared coordinate system is the load-bearing premise for subtracting and
averaging task vectors; the preparer and builder authenticate it explicitly.

This differs from scalar interpolation. Scalar interpolation assumes every
tower coordinate has the same transfer coefficient and shrinks shared signal
and seed noise at the same rate. The primary estimator assigns one
geometry-derived gain independently to every named floating tensor. The
secondary can keep a shared singular mode at full
strength while deleting modes that do not clear the measured noise edge.

It also differs from ordinary model soup. A tower soup is retained as an exact
control. The novel method must improve over both the scalar comparator and the
unfiltered cross-seed mean.

## Alternatives not selected

### Final-Adam-moment shrinkage

Every checkpoint retains AdamW `exp_avg` and `exp_avg_sq`. A coordinate or mode
could be weighted by a ratio such as `m^2 / (v + eps)`. This is attractive and
query-independent, but the final exponential moments estimate recent gradient
statistics, not the uncertainty of the complete accumulated parameter update.
Learning-rate decay, clipping, weight decay, and adaptive normalization break a
direct Wiener interpretation. The moments may be reported later as mechanism
evidence, but they do not determine the candidate state in this experiment.

### Layerwise trust-ratio shrinkage

A rule based on `||delta_l|| / ||W0_l||` is deterministic and cheap. It gives
one coefficient to an entire attention or MLP tensor, however, and cannot
separate a stable low-rank update from seed-specific directions in the same
tensor. It is too close to a coarse layerwise alpha grid.

### Coordinate sign consensus

Keeping only coordinates with three-seed sign agreement is a tuning-free TIES
style control, but three observations make coordinate-level decisions unstable
near zero and the rule ignores matrix structure. The selected methods pool
evidence over complete tensors or singular modes instead.

## Inputs and capability separation

An authority preparer authenticates the complete checkpoint/result graph and
publishes one normalized, model-only tower artifact and one head-only artifact
per seed. A tower artifact contains sorted trained tower tensors plus
checkpoint/configuration/run digests; it contains no endpoint metric,
correctness bit, image, label, optimizer state, projection, or proxy. A
head-only artifact contains only the exact trained `projection.weight` and
`proxies`, their state-name/shape/dtype authority, and the same binding digests.
The normalized manifest additionally carries each seed's independently
reconstructed tower-only initial digest. It carries no initial or endpoint
metric. All three initial tower digests must equal the builder's reconstructed
pretrained tower digest.
The state-construction phase receives only:

- the three authenticated normalized tower artifacts and their receipts;
- a normalized authority manifest binding those artifacts to the complete
  checkpoint/result graph without carrying any outcome value;
- the pinned local pretrained tower and exact reconstruction environment;
- this specification digest and the fixed seed order `(17, 29, 43)`.

It receives no raw checkpoint, optimizer moment, image, label, embedding,
nearest-neighbour result, scalar alpha outcome, or Cars band artifact. It
reconstructs the initial tower, validates
that its complete tower state is byte-identical across seeds, constructs the
three fixed tower states, publishes their content digests, and exits.

Only after all candidate digests are sealed may a separate evaluation phase
receive the three sealed candidate tower states and their published content
digests, the three authenticated normalized trained tower artifacts, the three
head-only artifacts, a normalized scalar-comparator manifest, and the exact
content-addressed burned artifact for classes 82--97 from the scalar diagnostic.
The comparator manifest carries the
full authenticated five-row scalar curves and their complete cross-object
binding; it cannot change candidate construction. The evaluator
combines each common candidate tower with each seed's own trained head. Raw
checkpoints and seed-result files are unavailable. Classes 49--81 and their
outcomes are unavailable. All results remain claim-ineligible because the
development band has already influenced the research program.

The evaluation child inherits the scalar diagnostic's leakage and capability
boundary verbatim: it receives only the registered burned manifest and
content-addressed image root, cannot import or load the source dataset, has no
network capability, runs in one named user unit with `@network-io` denied, is
observed through that unit's cgroup, and uses file-backed stdout/stderr. The
controller rejects any dataset path, clean-band artifact, classes 49--81,
official test capability, storage client, or unregistered file.

## Mathematical authority

Let `S = 3`, let `W0[p]` be one floating pretrained tower tensor, and let

```text
D_s[p] = W_s[p] - W0[p]
M[p]   = (D_17[p] + D_29[p] + D_43[p]) / 3.
```

All subtraction and estimation use CPU float64 authority values. Published
states are rounded once to contiguous CPU fp32.

### Primary tensorwise Wiener estimator

Every floating tower tensor, including every one-dimensional or scalar tensor,
is its own group. No group crosses a state-name boundary, so LayerNorm weights,
biases, attention biases, and matrix updates cannot dominate one another by
concatenation scale. For each group `G`, compute the three pairwise cosine
similarities

```text
c_st[G] = <D_s[G], D_t[G]> / (||D_s[G]||_2 ||D_t[G]||_2).
```

A zero-norm member makes `rho[G] = 0`. Otherwise define

```text
rho[G]  = clip(mean(c_17,29[G], c_17,43[G], c_29,43[G]), 0, 1)
beta[G] = 3 * rho[G] / (1 + 2 * rho[G])
D_wiener[G] = beta[G] * M[G].
```

Under the explicit replicate model `D_s = mu + epsilon_s` with independent,
isotropic, zero-mean seed perturbations, `rho` estimates the shared-signal
fraction and `beta` is the closed-form three-replicate Wiener/James--Stein gain
for the mean update. This is an estimator, not a claim that SGD noise is exactly
Gaussian or isotropic. The receipt also records the positive-part
variance-energy gain

```text
N[G] = [1 / (S * (S - 1))] * sum_s ||D_s[G] - M[G]||_2^2
P[G] = max(||M[G]||_2^2 - N[G], 0)
g_js[G] = P[G] / (P[G] + N[G]), or 0 when the denominator is 0.
```

`beta` is the state authority; `g_js` is an independently derived reported
statistic and cannot gate or change the candidate. Its disagreement with
`beta` is descriptive evidence about anisotropy/model mismatch, not a numerical
failure.

### Secondary symmetric contrast-edge spectral estimator

Every tensor with at least two dimensions is reshaped to
`[shape[0], product(shape[1:])]`. Define the three symmetric pairwise seed
contrasts

```text
C_17,29 = (D_17 - D_29) / sqrt(2)
C_17,43 = (D_17 - D_43) / sqrt(2)
C_29,43 = (D_29 - D_43) / sqrt(2).
```

For `M = U diag(sigma) V^T`, let

```text
edge = max(||C_17,29||_2, ||C_17,43||_2, ||C_29,43||_2) / sqrt(3)
D_spectral = U diag(sigma_i * 1[sigma_i > edge]) V^T.
```

The matrix norm is the largest singular value. The symmetric maximum makes the
candidate invariant to permuting seed labels while retaining the single-seed
noise scale; division by `sqrt(3)` maps that scale to mean-update noise.

Let

```text
tol = 64 * eps_float64 * max(rows, columns) * max(sigma_1, edge, 1).
```

When `sigma_1 == 0` and `edge == 0` exactly, the spectral update is the exact
zero matrix and construction skips the cluster/edge test for that tensor.
Adjacent singular values belong to one numerical cluster when their absolute
difference is at most `tol`. Every member of a cluster receives one decision.
If any singular value lies within `tol` of `edge`, construction terminates
`numerical-failure` instead of publishing a discontinuous rank choice.
Otherwise a cluster is retained exactly when all its members are above the
edge. This makes the reconstructed matrix invariant to SVD sign and basis
choices within numerically repeated subspaces. There is one empirical edge, no
Marchenko--Pastur assumption, rank ladder, energy ladder, or outcome-selected
threshold. Vector/scalar tensors use their primary Wiener estimate in the
spectral state.

### Folded states

The three fixed states are:

```text
tower-soup:       W0 + M
wiener-denoise:   W0 + D_wiener
spectral-denoise: W0 + D_spectral.
```

Every seed evaluation uses the same candidate tower and that seed's exact
trained projection and proxy tensors; only the projection participates in the
leave-one-out retrieval score, while proxies remain folded-state authority.
Non-floating tower tensors must be
byte-identical across all endpoints and are copied from the pretrained state.
All names, shapes, dtypes, finite values, and complete-state digests are strict.

## Numerical and determinism contract

The authority implementation processes one tensor at a time in sorted state
name order and releases all decomposition workspaces before advancing. BLAS,
LAPACK, torch, CPU model, thread count, and floating-point environment are
bound in the receipt. The builder runs twice from independently reloaded inputs;
all folded fp32 state digests, per-group cosines/gains, per-matrix edges,
numerical tolerances, singular-value clusters, kept ranks, and retained energies
must agree exactly.

A GPU or custom-kernel implementation is optional acceleration, never a second
scientific method. Before it can replace the authority path it must match every
published fp32 tensor byte-for-byte on synthetic random, exact and near-repeated
singular values, values inside/outside the registered edge tolerance, zero,
subnormal, rectangular, convolution-shaped, and real sampled tensors. No custom
kernel is planned: construction is offline, while the folded model preserves
the existing inference kernel and latency exactly.

## Controls and evaluation

The evaluation phase first replays every trained endpoint against its
authenticated result and validates the complete normalized five-row
scalar-comparator manifest inside the retained
`sfora-weight-space-transfer-campaign-result-v1` envelope. It recomputes the
embedded child-result and capability digests and cross-binds the source,
dataset manifest, burned manifest, seed-result, and checkpoint identities to
the prepared three-seed authority. A scalar `authority-failure` or `resource-failure`
blocks this run. The comparator is always the best observed common scalar row
over the complete three-seed grid by `(aggregate correct count, mean margin,
alpha)`, independent of whether the scalar funding gate passed. No scalar
outcome changes construction of any cross-seed state.

Before evaluating common towers, run all six ordered head-swap controls. For
each source seed `s` and different target seed `t`, evaluate seed `s`'s trained
tower using seed `t`'s `projection.weight`, and compare it with the same tower
using seed `s`'s own projection. `proxies` remain authenticated in the folded
state but are inert in leave-one-out retrieval. The receipt records every swap
correctness vector and margin. `head-coadaptation-observed` is true exactly when
the six swaps have fewer aggregate correct queries than their six own-head
controls. This control cannot select or change a candidate; it only separates a
negative end-to-end result with observed projection/tower coadaptation from one
without it.

For each of `tower-soup`, `wiener-denoise`, and `spectral-denoise`, evaluate the
full burned band once in the raw 1152-dimensional tower plane. The raw result
must be byte-identical in the builder/evaluator determinism replay. Then, for
each seed and each candidate, evaluate the full burned band with the existing
leave-one-out Recall@1 implementation and
`(similarity, source ordinal)` tie rule in the trained 512-dimensional
projection plane using that seed's `projection.weight`. The projection plane is
the quality authority; the raw plane isolates the common tower geometry and is
not recomputed three times. Record in a new candidate-specific row type rather
than the scalar alpha-row type:

- exact raw correctness bits once per candidate and exact projected correctness
  bits per candidate/seed, with count, denominator, and recall ppm;
- nearest-positive, nearest-negative, and mean margin for both planes;
- paired candidate-only and comparator-only counts and exact McNemar evidence
  separately for each seed's 1345 projected pairs; no pooled McNemar or
  independence claim is permitted;
- tower-state, complete folded-state, input, and result digests;
- per-group cosine/Wiener gain and reported `g_js` statistics, plus per-matrix
  spectral edge, kept rank, retained energy, and aggregate retained-energy ratio;
- wall time, peak CUDA allocation, peak RSS, and determinism replay.

## Gates and outcomes

Using the three projected-plane quality rows, `wiener-denoise` passes only if:

1. it gains at least nine correct queries over the scalar comparator, which is
   more than 0.20 percentage points of the 4035 operational row total;
2. it gains at least five correct queries over `tower-soup`, isolating value
   beyond ordinary averaging;
3. no seed loses more than one query relative to the scalar comparator;
4. its mean projected margin exceeds both the scalar comparator and
   `tower-soup`;
5. all authority, determinism, finite-value, memory, and replay gates pass.

The 4035 total is an operational funding count over three head-conditioned
rows, not 4035 independent observations and not a significance claim. Exact
McNemar evidence is reported only within each 1345-query seed row.

`spectral-denoise` passes only if it gains at least nine aggregate correct
queries over the scalar comparator, at least five over `tower-soup`, and at
least five over `wiener-denoise`; no seed loses more than one query relative to
the scalar comparator; its mean projected margin exceeds the scalar,
`tower-soup`, and Wiener margins; and every common gate passes. This isolates
mode selection beyond both ordinary averaging and the cheaper tensorwise
estimator. The candidate priority is fixed as spectral, Wiener, then soup; no
per-seed or post-result coefficient is permitted.

`tower-soup` independently passes its control gate only if it gains at least
nine aggregate correct queries over the scalar comparator, no seed loses more
than one query, its mean margin exceeds the scalar comparator, and all common
authority/resource gates pass.

The receipt also reports whether any method reaches 1278/1345 on every seed,
but 95% is a target crossing, not a selection rule.

The terminal class is exactly one of:

- `authority-failure`;
- `numerical-failure`;
- `resource-failure`;
- `spectral-denoise-benefit` exactly when spectral clears every spectral gate;
- `wiener-denoise-benefit` when Wiener clears its complete gate but spectral
  does not clear all spectral gates;
- `tower-soup-only-benefit` when soup clears its complete control gate but the
  two denoisers do not clear their complete gates;
- `no-cross-seed-benefit-with-head-coadaptation` when no candidate clears its
  complete gate and `head-coadaptation-observed` is true;
- `no-cross-seed-benefit`.

No result authorizes a publication claim or another external benchmark. A
positive result funds integration and a fresh benchmark plan. A negative result
rejects these fixed three-seed estimators, not all spectral task vectors,
optimizer-aware shrinkage, or retraining objectives.

## Resources and execution order

The builder and evaluator are serialized behind the active DGX seed and the
scalar diagnostic. They never overlap another scientific GPU process. The
three approximately 5.2 GB checkpoint objects already reside on the DGX. The
CPU-only builder may retain at most the reconstructed `W0`, three fp32 task
vectors, three accumulating fp32 candidate states, and one tensor's float64
decomposition workspace. Its hard stop is 110 GiB RSS plus registered memory
pressure, swap growth, or five minutes without per-tensor progress. The
evaluator additionally stops at 96 GiB combined GPU memory. A one-tensor builder
preflight and one projected evaluation preflight must project the complete
builder plus 3 raw candidate, 9 projected candidate, 6 projected swap-control,
and required endpoint/scalar replay work below six hours before the full run
starts.

Scratch contains only authenticated inputs, one tensor workspace, and partial
candidate outputs. It is removed after process exit. Only complete canonical
candidate manifests, result bytes, and their SHA-256 digests survive.

## Prior-art boundary and limitations

Model soups establish that averaging fine-tuned weights can improve robustness
without inference overhead. Model Stock derives layerwise center-seeking from a
few fine-tuned models. Task Singular Vectors and subsequent common versus
task-specific subspace work establish that task matrices contain useful
low-rank structure and that singular-vector alignment matters in merging. This
design uses neither validation-selected soup membership nor multi-task merging.
It treats same-task stochastic seeds as noisy replicate measurements and fixes
Wiener and empirical-noise-edge rules from their disagreement.

Three seeds provide only two variance degrees of freedom. The trained heads may
be co-adapted to their individual towers, so even a better common tower can
perform worse when paired with a seed-specific projection. The burned band is
already selection-contaminated. For these reasons a pass is feasibility evidence
and a failure is narrow. If the method improves by less than the fixed gate, the
next credible route is a prospectively specified training objective or fresh
data split, not a larger post-hoc shrinkage grid.
