# Pass 205 — Receiver-Diagonal Gain Calibration (RDGC)

## Status and independence

**LIVE-NARROW; prospective; no training authorized.** RDGC is a new candidate,
not an amendment, ablation, continuation, or favorable reinterpretation of
Receiver-Self Tangent Alignment (RSTA). The original RSTA artifact is
authenticated `VALID` as immutable scientific evidence and its scientific
decision is `UNRESOLVED`. RDGC never changes either status.

The old RSTA result is hypothesis-generation evidence only. Its authenticated
aggregates were:

```text
pooled Delta = 0.26607445
Delta bootstrap lower bound = 0.23252317
Delta seed means >= 0.26 = 4/4
pooled A_self - A_desc = -0.05681438
A_self - A_desc bootstrap lower bound = -0.11418958
A_self - A_desc seed means negative = 4/4
pooled median rho = 0.916114
pooled median absolute log norm ratio = 3.254504
deranged control pooled Delta = -0.00949555
random control pooled Delta = 0.00242748
alternate-context pooled Delta = 0.313704
alternate-context seed means positive = 4/4
```

RSTA's load-bearing criterion 2 required positive pooled
`A_self-A_desc` and a positive bootstrap lower bound. Both were negative, so
the angular receiver-self target did not separate from the occupied raw
cotangent. None of RSTA's frozen FAIL conditions fired; its exact result was
therefore `UNRESOLVED`. The positive Delta, context, rho, norm-ratio, and
negative-control results do not count toward any RDGC threshold. No old RSTA
receiver, row, bootstrap replicate, or candidate value may be loaded by the
RDGC implementation.

RDGC asks only whether a receiver-specific diagonal **scalar gain** is useful.
It contains no angular receiver-self target. A fresh preliminary falsifier must
first show that the scalar is receiver-specific, context-stable, and sensitive
to contributor count rather than a global rescaling. Only then may the same
one-shot process run a fresh no-training virtual-update panel.

## Exact upstream authority and outcome boundary

The following current branch chain is frozen:

```text
original RSTA candidate path:
  docs/pass200_rsta_candidate_2026-08-09.md
original RSTA candidate SHA-256:
  a35cd3469d5561ce59202030dd3c3050e018dbfc537cb0ee0401a1d0340f5857
original RSTA candidate commit:
  4b33076f7d7fd8da78987eb4d04664bde14452c6

RSTA Gate-2 audit path:
  docs/pass200_rsta_gate2_primary_audit_2026-08-09.md
RSTA Gate-2 audit SHA-256:
  3efad753b1328c1a23188dfb1422cf86fa1376e625434a6ea419b24dfc0caf0b
RSTA Gate-2 audit commit:
  9d0cc9646607e1637f593457a507dce547d7d4b8

RSTA scientific producer source S:
  15234a529a181c39c1c8b6477ad7eb7823fd0798
RSTA scientific producer handoff H:
  c04574e2bb751c3229bce673408577cfedc00a88
RSTA immutable artifact path:
  reports/generated/pass200_rsta_receipt/c04574e2bb751c3229bce673408577cfedc00a88-stage-a.json
RSTA immutable artifact SHA-256:
  e9bcd77c6e372e9c3bab4a420b97ff56f8ea164cbca56f53ec9c99a3b3c527ae
RSTA producer PID / exit:
  1002393 / 0

roundtrip-verifier source V_R:
  3c368713e0890c0ffc63308f07d8d4ee5b19db1c
roundtrip-verifier handoff HV_R:
  e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae
HV_R manifest SHA-256:
  fb089cf5905cea32a9d22563b50160af5fc8643efb657c49cb519d6d0c0da80b
roundtrip validation receipt path:
  reports/generated/pass200_rsta_receipt/e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae-scientific-artifact-roundtrip-validation.json
roundtrip validation status:
  VALID
RSTA scientific decision:
  UNRESOLVED
RSTA first decisive clause:
  no_pass_or_fail_rule
```

The RDGC source activation manifest must bind the exact validation-receipt
bytes and SHA-256 after those provenance-only bytes are transferred to the DGX.
The receipt SHA is a future manifest-derived authority, not a free CLI value.
It must validate exact `V_R/HV_R`, the immutable artifact path/SHA, and
`status="VALID"`. The RDGC process reads the provenance-only validation receipt
but never opens or parses the old RSTA scientific artifact.

This document and its implementation plan are committed before any RDGC source,
test, manifest, result, or GPU execution. A later reviewed source commit `V_G`
may add only the exact RDGC diagnostic and test. A direct-child manifest-only
handoff `HV_G` creates the new RDGC manifest and binds `V_G`. The current RSTA
manifest at `HV_R` is never modified.

## Narrow primary-literature and audit tuple

Before implementation, a fresh read-only Gate-2 reviewer must bind this exact
tuple and return `LIVE-NARROW` for the scalar object below:

1. [DoCL / Curriculum Learning by Optimizing Learning Dynamics, AISTATS
   2021](https://proceedings.mlr.press/v130/zhou21a.html): raw residual or
   cotangent aligned with full functional motion and used for curriculum
   weighting; no receiver-diagonal scalar-gain target.
2. [Model Gradient Similarity, NeurIPS
   2022](https://papers.nips.cc/paper_files/paper/2022/hash/67b0579a7298d9cf39c59404d867bdd7-Abstract-Conference.html): explicit diagonal
   and cross-example tangent ingredients with global kernel summaries; no
   receiver-specific full/diagonal gain-ratio penalty.
3. [NINT v2](https://arxiv.org/html/2511.15487v2): intended full functional
   motion norm for coordinate selection; no diagonal scalar reference or
   differentiated receiver-gain ratio. The existing RSTA audit's implementation
   caveat remains binding.
4. [Charpiat et al., NeurIPS
   2019](https://proceedings.neurips.cc/paper/2019/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html): differentiable tangent-field
   shaping without the contextual full/diagonal scalar gain object.
5. [GradNorm, ICML
   2018](https://proceedings.mlr.press/v80/chen18a.html): parameter-gradient
   magnitude balancing across multitask losses, not a named receiver's
   output-space total/diagonal empirical-tangent gain.
6. [K-FAC, ICML
   2015](https://proceedings.mlr.press/v37/martens15.html): a Fisher-curvature
   optimizer/preconditioner, not a loss-induced receiver-specific scalar target.
7. The exact RSTA Gate-2 audit path/SHA/commit above, especially its
   load-bearing non-collapse section.

The only defensible proposition is: **RDGC differentiates the logarithmic gain
error between a named receiver's full contextual empirical-tangent response and
the stopped scalar norm of its diagonal receiver path.** It leaves sample
eligibility, sample weights, proxies, optimizer metric, and inference unchanged.

A primary source that already applies this exact receiver-specific
full/diagonal gain-ratio penalty closes RDGC before source work. Claims about a
new NTK regularizer, norm balancing generally, functional motion generally, or
preconditioning are forbidden.

## Exact candidate operator

For an ordinary full B=180 Proxy Anchor batch graph, retain the original RSTA
definitions:

```text
z_i       = normalized descriptor
J_i       = partial z_i / partial theta for every trainable encoder parameter
dbar_i    = -partial L_PA(B) / partial z_i
b_i       = J_i sum_j J_j^T dbar_j
s_i       = J_i J_i^T dbar_i
epsilon   = 1e-8
```

For the one registered receiver `r`, define exactly:

```text
e_r      = log((||b_r||_2 + epsilon)
               / (stopgrad(||s_r||_2) + epsilon))
R_RDGC   = 0.5 * e_r^2
```

`b_r` and its exact contextual cotangents remain differentiable. The complete
scalar `||s_r||_2` is detached before addition of epsilon. RDGC never constructs
`s_hat`, never computes `cos(b,s)`, never sends a vector toward `s`, and never
uses a raw-cotangent angular target. The future training form, which is not
authorized here, would be a newly preregistered `L_PA + lambda_G R_RDGC`.
Neither `lambda_G` nor a benchmark schedule is selected by this document.

RDGC differs from ordinary gradient norm balancing because it compares two
output-space actions of the same contextual loss at one named receiver after
mapping through `J_r`, and its reference is the diagonal path `J_rJ_r^T`.
It differs from preconditioning because it neither estimates nor inverts a
parameter metric and does not modify the ordinary PA update. It differs from
generic full-motion damping because it uses a receiver-specific stopped
diagonal scalar reference. The preliminary and full controls must falsify each
of these claimed distinctions empirically.

The Euclidean parameter tangent kernel remains non-invariant to arbitrary
hidden-layer rescaling and AdamW preconditioning. RDGC claims only the frozen
BN-Inception/Proxy-Anchor parameterization and the registered common descriptor
rotation. This is a material scope limitation, not a footnote.

## Candidate-free bindings and fresh selection

Use only the same corrected final In-Shop PA seeds `0,1,2,3`, checkpoints,
training rows, configuration, exact source exports, and immutable artifact
bindings already authenticated by the current RSTA manifest. Query/gallery
arrays remain binding-only and are released before any RDGC quantity.

Before any candidate graph or tensor exists, a fresh isolated process must:

1. authenticate `HV_G^ == V_G`, the manifest-only `HV_G` scope, candidate
   authority, exact source order and every Git-blob/worktree digest;
2. authenticate the provenance-only RSTA validation receipt and the complete
   frozen chain above without opening the old scientific artifact;
3. validate all four seed checkpoints, reports, final packs, train IDs/labels,
   source-image membership, configurations, and source exports;
4. run the dense Jacobian, BN, repeatability, exact normwise adjoint,
   sign-control, rotation, atomic-writer, and no-candidate-reachability fixtures;
5. create and validate the complete fresh preliminary and panel identity,
   support, receiver, distractor, context, transform, and tensor-hash plans for
   all four seeds; and
6. release every binding/query/gallery object not needed by scientific work.

All four seed integrity records must exist and pass before the first RDGC,
control, contributor, margin, or bootstrap state is constructed.

Recompute the original 64 RSTA primary labels deterministically from the
authenticated training IDs and the old literal domains; do not read them from
the old result. Exclude every row of those identities from RDGC receiver and
support roles. Old distractors may not be used in the fresh preliminary or
panel contexts; derive their IDs from the old deterministic selection recipe,
again without opening the old result.

Use SHA-256 domain hashing with NUL separator and unsigned canonical decimal
integers. Eligible identities require at least three corrected-train images.
Order all remaining eligible identities by
`(H("rdgc-stage-b-v1|identity|", label), label)`.

- Preliminary identities are positions `0..7`.
- Full-panel identities are positions `8..39`, exactly 32 identities.
- Within each identity, rows ordered by
  `H("rdgc-stage-b-v1|role|", example_id)` have rank 0 and 1 as clean
  eval-mode supports and rank 2 as the receiver.
- No support enters a PA graph.

For each eight-receiver group and context `A` or `B`, exclude every row of all
40 RDGC identities, all old RSTA identities, every old or RDGC support, every
receiver, and every distractor already assigned to another RDGC context. Order
remaining rows by
`H("rdgc-stage-b-v1|distractor|<phase>|<group>|<context>|", example_id)`
and take the first 172. Final batch order uses
`H("rdgc-stage-b-v1|batch-order|<phase>|<group>|<context>|", example_id)`.
Receiver tensors are identical across contexts and seeds; distractors differ.
Use the exact independently seeded official transform/cache protocol from RSTA
with new domain prefix `rdgc-stage-b-v1`.

## Cheapest preliminary falsifier

The preliminary uses only the first eight fresh identities: one B=180 group,
two contexts, and four seeds. It computes no auxiliary parameter correction,
proxy-free margin, or bootstrap.

For each receiver/context, obtain `dbar` once from the exact full B=180
contextual PA graph. Order the 179 nonreceiver rows by
`H("rdgc-stage-b-v1|contributor|<seed>|<receiver_id>|<context>|", example_id)`.
For contributor counts `n = 1, 8, 32, 180`, define `C_n` as the receiver plus
the first `n-1` ordered rows and compute:

```text
g_r,n = sum_{j in C_n} J_j^T dbar_j
b_r,n = J_r g_r,n
b_r,1 must equal s_r under the exact normwise equality gate
kappa_r = (||s_r||_2 + 1e-8) / (||dbar_r||_2 + 1e-8)
E_r,n = abs(log((||b_r,n||_2 + 1e-8) / (||s_r||_2 + 1e-8)))
C_r = E_r,180 - E_r,8
```

For each seed/context define the global scalar
`log_kappa_global` as the median of its eight `log(kappa_r)` values and define
each receiver's global-scalar relative error as
`abs(exp(log(kappa_r)-log_kappa_global)-1)`.

The preliminary **SURVIVES** and automatically enters the full panel only if
all are true:

1. in each context, pooled median `C_r >= log(1.10)`, and at least three of four
   seed means are positive;
2. pooled median across seeds of the eight-receiver Spearman correlation between
   context-A and context-B `log(kappa_r)` is `>=0.50`, and at least three seed
   correlations are `>=0.30`;
3. in each context, pooled IQR of `log(kappa_r) >= log(1.10)`;
4. in each context, pooled median global-scalar relative error is `>=0.05`; and
5. in each context, pooled median `E_r,180 >= log(1.25)`, with at least three
   seed medians `>=log(1.10)`.

Preliminary **CLOSE** takes precedence if any is true:

1. either context has pooled median `C_r <=0` and at least three nonpositive
   seed means;
2. the pooled median context correlation is `<=0`, or at least three seed
   correlations are `<=0`;
3. either context has pooled IQR `log(kappa_r) <= log(1.02)`;
4. either context has pooled median global-scalar relative error `<=0.02`; or
5. either context has pooled median `E_r,180 <= log(1.05)` and at least three
   seed medians at or below that threshold.

Any integrity/schema/norm failure is `INVALID`. Every other preliminary result
is `UNRESOLVED`. `CLOSE` or `UNRESOLVED` stops the process before any full-panel
candidate or control correction is constructed.

## Full no-training virtual-update panel

If and only if the preliminary SURVIVES, process the 32 full-panel identities
as four groups of eight in both contexts and every seed. Context A constructs
all parameter corrections. Context B tests transfer of those exact detached
directions; it never reconstructs or retunes a correction.

Let `p=-partial L_PA/partial theta` be the exact ordinary PA descent direction
over the complete named encoder parameter tuple. For each receiver define the
following exact operator order:

```text
rdgc
raw_cotangent
full_motion
batch_global_gain
scalar_diagonal_raw
per_example_gradient_normalized
```

Their penalties are:

```text
R_rdgc = 0.5 * log((||b_r||+eps)/(stopgrad(||s_r||)+eps))^2

R_raw_cotangent = 1 - cos(b_r, stopgrad(dbar_r))

R_full_motion = 0.5 * ||b_r||^2
                / stopgrad(||b_r||^2 + eps^2)

T_batch = exp(mean_{k in 8 receivers} log(||s_k||+eps))
R_batch_global_gain = 0.5 * log((||b_r||+eps)/stopgrad(T_batch))^2

kappa_batch = exp(mean_{k in 8 receivers}
                   log((||s_k||+eps)/(||dbar_k||+eps)))
T_scalar_raw,r = kappa_batch * (||dbar_r||+eps)
R_scalar_diagonal_raw =
  0.5 * log((||b_r||+eps)/stopgrad(T_scalar_raw,r))^2
```

For the last control, form each differentiable per-example parameter direction
`g_j=J_j^T dbar_j` in exact batch order. With FP64 cast-before-reduction norms,
define the detached geometric mean
`nu=exp(mean_j log(||g_j||+1e-12))`, detached coefficients
`a_j=stopgrad(nu/(||g_j||+1e-12))`,
`g_PGN=sum_j a_j g_j`, and `b_PGN,r=J_r g_PGN`. Then:

```text
R_per_example_gradient_normalized =
  0.5 * log((||b_PGN,r||+eps)/(stopgrad(||s_r||)+eps))^2
```

This control tests whether a few large per-example parameter gradients explain
the apparent gain. It is a diagnostic control, not a training proposal.

For `X` in the exact operator order, compute `c_X=-partial R_X/partial theta`.
All corrections and updates use FP64 cast-before-product reductions in the
exact named parameter order. Let `n=||p||`. Require every norm `>1e-12`, then:

```text
c_hat_X = n * c_X / ||c_X||
v_X = p + 0.10 * c_hat_X
u_X = n * v_X / ||v_X||
u_PA = p
```

Require the recomputed FP64 norm of every `u_X` to equal `n` within relative
error `5e-7`. No optimizer, moment, clipping, decay, proxy update, BN-buffer
update, or parameter mutation occurs.

Using fresh functional graphs, compute `w_X,A=J_r,A u_X` and
`w_X,B=J_r,B u_X`, including `X=PA`. Construct the same class-disjoint
proxy-free margin ascent `q_r` as RSTA from the two fresh supports and frozen 32
foreign rank-0 supports, separately for each context. Record:

```text
A_X,c = cos(w_X,c, q_r,c)
M_X,c = <w_X,c, q_r,c> / (||p|| * ||q_r,c||)
```

All direction construction, JVPs, and scientific scalars are FP32 except the
named FP64 reductions, bootstrap, and exact hashes. A norm `<=1e-12`, nonfinite
value, missing gradient, or failed update-norm equality is `INVALID`.

## Aggregation, bootstrap, and exact decision

Aggregate equally by receiver within seed and equally by seed. Bootstrap the
32 complete identity labels jointly across all four seeds with exactly 10,000
NumPy `PCG64(201)` paired resamples. One replicate draws 32 indices once and
applies that same vector to every seed, context, operator, `A`, and `M`
difference. Record every float64 C-order distribution and SHA-256.

Full-panel **PASS** requires all:

1. primary-context pooled mean `A_rdgc-A_PA >=0.02`, bootstrap lower bound
   `>0`, and at least three seed means `>=0.01`;
2. primary-context pooled mean `M_rdgc-M_PA >0`, lower bound `>0`, and at least
   three positive seed means;
3. for each of the five controls, primary pooled means and lower bounds of both
   `A_rdgc-A_control` and `M_rdgc-M_control` are `>0`, with at least three
   positive seed means for each `A` difference;
4. context-B pooled mean `A_rdgc-A_PA >0`, lower bound `>0`, and at least three
   positive seed means;
5. in context B, every pooled `A_rdgc-A_control >0`;
6. pooled median absolute parameter-direction cosine
   `abs(cos(c_rdgc,c_control)) <0.95` for every control; and
7. every row, seed, context, operator, hash, integrity gate, and bootstrap
   distribution is present and valid.

Full-panel **CLOSE** takes precedence if any:

1. primary or context-B pooled `A_rdgc-A_PA <=0`, with at least three
   nonpositive seed means;
2. primary pooled `M_rdgc-M_PA <=0`, with at least three nonpositive seed means;
3. any control has primary pooled `A_rdgc-A_control <=0` and at least three
   nonpositive seed differences;
4. the batch-global, scalar-diagonal/raw, or per-example-normalized control has
   primary pooled `M_rdgc-M_control <=0` and at least three nonpositive seed
   differences; or
5. pooled median absolute correction cosine with any control is `>=0.99`, with
   at least three seed medians `>=0.99`.

Any binding, integrity, schema, finite, norm, identity, graph, equality,
determinism, or publication failure is `INVALID`. Every complete scientific
outcome satisfying neither all PASS predicates nor a CLOSE predicate is
`UNRESOLVED`. CLOSE precedence is evaluated in the literal order above; PASS is
evaluated only after no CLOSE condition fires.

PASS authorizes only a separately reviewed and preregistered training study.
It does not authorize implementation in `src/`, optimizer integration, a
benchmark run, hyperparameter tuning, or publication. CLOSE ends RDGC.
UNRESOLVED permits no GPU follow-up until another prospective authority exists.

## Process isolation, graph lifetime, and one attempt

The exact future CLI is:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
.venv/bin/python -I -B scripts/diagnose_pass205_rdgc_stage_b.py
  --manifest docs/pass205_rdgc_stage_b_manifest.json
  --output reports/generated/pass205_rdgc_stage_b/<HV_G>-rdgc-stage-b.json
  --scientific-once
```

It runs in one fresh detached clean checkout at `HV_G`. PyTorch is not imported
until the candidate-free prefix has authenticated CLI paths, Git/manifest,
source bytes, validation receipt, output absence, environment, and process
isolation. After PyTorch import, all four seed integrity prefixes complete and
release their graphs before preliminary state is constructed.

The scientific process runs the preliminary and, only on exact SURVIVES,
automatically continues to the full panel without exiting, editing authority,
or receiving operator input. There is one process, one output, and one attempt.
Opening the first checkpoint begins attempt 1. Any exit, exception, signal,
timeout, CUDA fault, preliminary result, panel result, or publication failure
consumes it. No retry or second output is authorized.

Use one full B=180 CUDA graph at peak. Process seeds, groups, receivers,
contributor counts, operators, and contexts in registered order. Within one
graph, compute only the actions needed for the current registered unit, reduce
and hash them, detach only JSON scalars/booleans/hashes and the one required CPU
parameter direction, then delete every CUDA tensor, closure, gradient,
functional state, model reference, and graph before the next graph.

Store `p` and the current receiver's `c_rdgc` as detached CPU tensors only while
needed for update normalization and correction-cosine controls. Process each
other correction serially, compute its FP64 CPU dot/cosine with `c_rdgc`, build
and consume its one virtual JVP direction, then delete it before the next
control. Per-example gradients are accumulated serially in exact row order;
never retain 180 live CUDA gradient trees. No candidate or control action from
two graphs may coexist on CUDA.

On preliminary CLOSE/UNRESOLVED, the full-panel builder is unreachable. On any
INVALID condition, no later candidate or control call occurs. Result publication
is exclusive-create, flush/fsync, hard-link no-replace, directory-fsync, and
strict reload. The immutable RSTA artifact, RSTA receipt, manifests, source,
checkpoints, reports, and packs are never modified.

## Exact output schemas

The scientific output has exactly these top-level keys in order:

```text
schema_version
diagnostic
mode
status
phase_reached
candidate_values_computed
training_performed
benchmark_authorized
scope_limitation
authority
source
execution
environment
binding
integrity
selection
preliminary
panel
bootstrap
decision
```

Fixed scalars are:

```text
schema_version = 1
diagnostic = "pass205_rdgc_stage_b"
mode = "scientific_no_training_virtual_update"
status = exactly "PASS", "CLOSE", "UNRESOLVED", or "INVALID"
phase_reached = exactly "integrity", "preliminary", or "full_panel"
training_performed = false
benchmark_authorized = false
scope_limitation = "Euclidean BN-Inception/Proxy-Anchor parameter tangent; invariant only to the registered common descriptor rotation, not hidden-layer rescaling or AdamW preconditioning"
```

`candidate_values_computed` is false only for `INVALID` before preliminary
science. A preliminary CLOSE/UNRESOLVED has `phase_reached="preliminary"`,
`panel=null`, and `bootstrap=null`. A full-panel PASS/CLOSE/UNRESOLVED has both
objects present. An INVALID result is reduced and contains no partial candidate
row, aggregate, or bootstrap value.

`authority` binds this candidate path/SHA/commit, the exact literature/audit
tuple, and the implementation plan path/SHA/commit. `source` binds `V_G/HV_G`,
the manifest path/SHA, exact ordered source paths/hashes, clean detached
worktree, and ancestry. `execution` binds one attempt, command, cwd, PID,
Python/CUDA process settings, start/end status, and output path. `environment`
is the exact deterministic runtime audit. `binding` contains the complete RSTA
chain above, the validation-receipt SHA/status, and all four seed artifact
digests. `integrity` contains complete all-seed fixture/action/hash audits.

`selection` contains exact preliminary/panel identity, role, support, receiver,
distractor, batch, transform, and tensor hashes. `preliminary.rows` are ordered
by seed, context `A,B`, then receiver identity order and have exactly:

```text
seed
context
receiver_label
receiver_id
dbar_norm
self_norm
kappa
log_kappa
b_norms_by_contributor_count
absolute_log_gain_errors_by_contributor_count
count_gain
```

The two contributor mappings have exact string keys `"1","8","32","180"`.
`preliminary` also contains exact per-seed/context aggregates, pooled
aggregates, all threshold booleans, and its first decisive clause.

Full `panel.rows` are ordered by seed, group, receiver, context, and have exact
identity/batch hashes, parameter metadata, `p_norm`, every correction norm,
every matched update norm, every correction cosine, and operator records in
order `pa`, `rdgc`, `raw_cotangent`, `full_motion`, `batch_global_gain`,
`scalar_diagonal_raw`, `per_example_gradient_normalized`. Each operator record
contains exactly `motion_sha256`, `motion_norm`, `margin_alignment`, and
`margin_slope`. `panel` contains every within-seed and pooled aggregate and
threshold Boolean. `bootstrap` contains PCG64 version/seed/replicates, every
paired distribution and hash, lower bounds, and complete-label order.

`decision` contains exactly `close_precedence`, ordered predicate records,
`status`, `first_decisive_clause`, and `authorized_action`. Authorized actions
are exactly `stop_invalid`, `stop_close`, `stop_unresolved`, or
`new_training_preregistration_only`.

Every mapping uses exact insertion order and concrete JSON types. Strict
validators reject every missing, extra, reordered, mistyped, nonfinite, alias,
inconsistent, selectively omitted, or post-hoc canonicalized field. Aggregates
without all registered rows and hashes are invalid.

## Source, manifest, review, and DGX sequence

1. Commit this candidate alone and obtain its SHA-256/commit.
2. Commit the bound executable TDD/compute plan alone.
3. Obtain a fresh primary-source/Gate-2 and specification review. Repair docs
   only and rebind before source work if any Critical or Important finding exists.
4. Write tests RED first, then implement only
   `scripts/diagnose_pass205_rdgc_stage_b.py` and
   `tests/test_diagnose_pass205_rdgc_stage_b.py`.
5. Run all synthetic CPU/tiny-CUDA-if-available tests, lint, compilation, scope,
   and source review. Commit the exact two-file source/test revision as `V_G`.
6. Create only `docs/pass205_rdgc_stage_b_manifest.json`, binding the candidate,
   plan, exact upstream chain, validation receipt, artifacts, `V_G`, and the
   current 32 scientific source paths plus the new diagnostic in frozen order.
   Commit it alone as direct-child `HV_G`, then obtain a fresh manifest review.
7. Transfer exact `HV_G`, manifest, validation receipt, and frozen artifacts to
   a fresh DGX checkout. Run candidate-free preflight. If and only if green, run
   the exact scientific command once.
8. Preserve the one atomic result and stop for every status. No training or
   benchmark command follows.
