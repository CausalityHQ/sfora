# Pass 212 — Amortized Semantic Gradient Control Variate

Date: 2026-08-31
Status: **HYPOTHESIS AND FALSIFIER DESIGN — no retrieval-quality result.**

## Objective

SAGA reports `97.0 +/- 0.3` Cars Recall@1, while the SFORA release bar is a
clean three-seed mean of at least `97.4%` and a paired gain of at least `0.5`
point over a capacity-matched implementation. SAGA's principal quality gain
comes from its frozen-MLLM GRPO gradient, but every contributing pair pays for
eight stochastic rollouts and differentiable replay through the language
backbone.

The proposed method, **Amortized Semantic Gradient Control Variate (ASG-CV)**,
does not replace that gradient with an uncorrected student prediction. It learns
a cheap pair-conditioned gradient predictor and adds an independently sampled,
inverse-probability-weighted exact residual. The resulting minibatch estimator
has the same expectation as the registered SAGA semantic gradient *before the
nonlinear global-gradient clip and optimizer transform*. If the predictor
explains most of its conditional variation, ASG-CV can spend the same compute on
more distinct pairs with lower gradient variance without relying on a biased
surrogate.

This is a prospective mechanism, not a SOTA claim. The exact SAGA feasibility
receipt and a primary-source collision audit remain prerequisites for training.

## Inputs and gradient boundary

For pair `i` at registered optimizer step `t`, let:

```text
X_it      = stopped pair patch tokens before the trainable vision update
r_it      = source-bound rollout seed block
g_it      = d L_semantic(X_it, r_it) / d X_it
q_phi(X)  = detached prediction of E[g | X]
```

`L_semantic` contains only the expensive GRPO and attribute-attention terms.
The ordinary metric-learning loss remains exact for every image. Rollout seeds,
pair order, sampler strata, prompt bytes, relation tokens, tokenizer, processor,
and model revisions are immutable authority.

The predictor emits a rank-`R` patch/channel field rather than a dense unconstrained
`P x D` tensor:

```text
q_phi(X) = A_phi(X) B_phi(X)^T,
A_phi in R^(P x R), B_phi in R^(D x R).
```

`R=16` is fixed before gradient evidence is opened. A source-bound signed SRHT
sketch provides a second, independently reproducible target for predictor
training and detects a low-rank model that matches norms while rotating the
teacher field.

For the registered Qwen3-VL implementation, the complete vision-gradient cut is
the ordered tuple `(merger, deepstack-0, deepstack-1, deepstack-2)`. Qwen3-VL
injects the three DeepStack merger outputs into early language layers, so the
final merger alone is not a complete vision-gradient interface. The eight
sealed completions execute eight distinct autograd branches through all four
deterministic modules. E0 retains every branch output, requires byte-identical
stopped values within each boundary and the registered `4 x 2 x 49 x D` shape,
derives the detached layer-26 teacher map only from registered correct
completions, applies the pooler attention KL to the first live final-merger
tensor, and performs one backward through the sum of mean GRPO and KL. It then
sums all eight gradients separately at each boundary. A hard one-pair
equivalence gate recomputes the vision graph and applies the captured four-field
VJP: every trainable vision-parameter gradient must match direct combined-loss
backward within the sealed fp32 tolerance. Missing a DeepStack boundary is an
authority failure. A separately recomputed attention-only feature forward is
not a valid substitute because it is outside the combined replay graph.

Completion parsing is fixed at the tokenizer-ID level before generation. The
registered prompt requires either one exact `SAME` prefix or one exact
`DIFFERENT` prefix followed by at least one attribute token. Their token-ID
sequences, the sorted terminal-token set, and the exact tokenizer revision form
a content-addressed protocol. Classification never decodes text or applies a
regex: it strips only registered trailing terminal IDs, matches exactly one
non-overlapping prefix, and takes the remaining half-open token interval as the
attribute span. A valid verdict earns binary reward one exactly when it matches
the pair relation; wrong or malformed completions earn zero, and only correct
valid completions contribute detached attention teachers. Every captured sample
binds the completion-protocol digest as well as the completion-group digest.
The group classifier operates on exactly eight completions and derives rewards,
correct-rollout flags, and teacher spans together. A group has nonzero reward
variance only when it contains at least one correct and one incorrect verdict.
All-correct and all-incorrect groups have the exact semantic target `g=0`,
matching DAPO; they remain in candidate-schedule order rather than being
discarded or refilled. Duplicate completion token sequences are valid evidence
when their independently derived generation seeds differ.

The scalar predictor consumes stopped fp32 patch tokens with shape
`batch x 2 x patches x channels` plus an exact `-1/+1` relation sign. It uses
one shared token LayerNorm and one shared local rank projection. For each image,
the context is the ordered tuple of its mean token, the other image's mean
token, their signed difference, their Hadamard product, and the relation sign.
A shared context projection emits a patch-factor bias and a bounded modulation
of one shared channel basis. Swapping the two images therefore swaps the two
predicted gradient fields without changing their values, and every per-image
field remains a product of `P x 16` and `D x 16` factors. The predictor refuses
tokens that still require vision autograd; student injection uses an explicitly
detached output. Every named fp32 state tensor is framed by name and shape and
sealed with SHA-256 before E0 evaluation. Initialization uses a domain-separated
SHA-256 seed through an isolated CPU generator scope, producing byte-identical
state without consuming or reseeding the retrieval trainer's CPU or CUDA RNG
streams. Captured-gradient receipts bind the independently sealed nondegenerate
attention-pooler state but deliberately contain no predictor-state identity:
capture precedes fitting and has no predictor capability. The later E0 result
binds the sealed predictor-state digest and predicted field, preventing E1
replay targets from being evaluated against a different learned control variate
or KL query without introducing a circular pre-fit authority.

E0 additionally emits one custody artifact over every validation row in each of
the two exact rollout-seed blocks. Each block requires strictly increasing
unique candidate ordinals, unique completion groups, globally disjoint
source-pair ordinals, and exactly four positive plus four negative relations in
every eight-row stratum. Corresponding rows must bind the same candidate,
source pair, and relation, while the two blocks must bind disjoint completion
group digests. Numeric distance is not used as proof of independent capture:
distinct registered rollout seeds may legitimately produce equal or nearly
equal gradients. All sample receipts share the same source commit, model
revision, fixture, pooler, completion protocol, and eligible schedule.
Corresponding rows must bind the same patch-token authority because rollout sampling cannot change
the frozen image-pair representation. The
custody artifact binds both exact fp32 gradient digests and both patch-token
digests for every row, requires both E0 fp64 exact fields to be lossless fp32
widenings, and obtains the predictor identity only from the E0 result. The
custody receipt co-binds that sealed identity, the captured patch-token
authorities, and the content-addressed predicted field; it does not claim to
replay the predictor from those inputs.

Predictor fitting uses one fixed, dimensionless objective with no learned or
retrieval-tuned weights:

```text
L_predictor = ||q - g||^2 / ||g||^2
            + ||S(q) - S(g)||^2 / ||S(g)||^2,
```

where `S` is the sealed SRHT operator. Both terms have unit coefficient. The
teacher field and patch tokens are stopped; only predictor state receives this
loss. The differentiable torch SRHT reference uses the same signs, selected
rows, butterfly order, and normalization as the fp64 scalar authority. Zero
teacher or projected energy and every non-finite intermediate fail closed.

Before any captured gradient is opened, the capture manifest also freezes the
entire fit authority: SRHT dimensions/rows/seed, rank, training-row count,
minibatch size, epoch and derived update counts, optimizer algorithm and exact
rational hyperparameters, initialization seed, and sample-order seed. The E0
campaign uses 512 training rows, batches of 4, and 32 complete epochs (4,096
updates), with single-tensor fp32 AdamW, learning rate `3/10,000`, weight decay
`1/10,000`, betas `9/10` and `999/1,000`, and epsilon `1/100,000,000`. There is
no tail batch. Epoch order is a deterministic SHA-256 ordering derived from the
sealed sample-order seed, epoch, and candidate ordinal. The full campaign uses
the manifest-bound `cuda-deterministic-v1` execution backend; reduced CPU
controls use `cpu-one-thread-deterministic-v1`, and neither may silently fall
back to the other. The initialization and
ordering seed domains must differ. Optimizer settings cannot be changed after
capture or in response to E0 validation evidence; a different coherent
manifest is a different claim-ineligible experiment, not a retry.

## Unbiased stratified estimator

Each class-balanced minibatch is divided deterministically into strata of eight
pairs. Exactly one pair `j_s` per stratum is selected from a separate registered
sampler stream. Only that pair receives the eight-rollout exact semantic replay.
For a stratum `s`, use

```text
g_hat_s = (1 / 8) sum_i q_phi(X_i)
          + (g_j_s - q_phi(X_j_s)).
```

Conditioned on the eight pairs and the registered rollout seed blocks,

```text
E_j[g_hat_s] = (1 / 8) sum_i g_i.
```

The correction is therefore unbiased without clipping, fitted scalar weights,
or retrieval-outcome feedback. The exact residual is computed in fp32. The
predictor output and its training optimizer are detached from the retrieval
optimizer before gradient injection. Predictor training uses only previously
sealed exact `(X, g)` pairs from the training partition.

The update order is load-bearing. At step `t`, freeze `phi_t` before pair
stratification and before revealing the source-bound selected index. Compute the
student estimator entirely with `q_phi_t`, apply the student update, and only
then allow the sealed exact pair to update the predictor for step `t+1`. Updating
the predictor with the selected residual before forming `g_hat_s` would correlate
the control variate with the selection event and is forbidden. Strata and the
selection schedule are committed before any exact semantic gradient is opened.

If the selected pair has zero reward variance, its exact residual is the
registered zero field; the optimizer step is not conditioned on an eligibility
outcome and is never discarded. A missing replay, non-finite value, or
overflowing correction is a fail-closed scientific terminal.

SAGA's declared global gradient clipping at `1.0` is applied only after the
complete DML and semantic gradient is assembled. Because clipping is nonlinear,
the expected *clipped update* is not generally unbiased even when `g_hat_s` is.
E0 derives the pre-clip semantic-field estimator norms and clip activation from
each bound exact seed block, predicted fields, and the source-bound selection
seed, then forms one combined two-block norm distribution. E0 does not accept
self-reported norm arrays. E1 must likewise derive these values
at its registered optimizer boundary. ASG-CV is ineligible if its clip-activation rate is
more than `5` percentage points above the matched exact estimator or if its
pre-clip norm p99 exceeds `2.0x` the exact p99. No residual clipping or
winsorization is allowed to repair this gate.

The E0 semantic patch-field proxy threshold is the higher-method p90 norm of the
combined two-block exact-seed stratum means. A relative fp64 comparison tolerance prevents exact
ties from changing sides under uniform rescaling. This makes the `5`-percentage-point comparison invariant
to uniform rescaling of the cut field; it is not the assembled parameter-gradient
clip itself. Its registered role is to reject an estimator whose semantic
correction has a materially worse tail before E1;
E1 must measure the actual assembled DML-plus-semantic parameter gradient at
the optimizer boundary and cannot inherit the proxy as product evidence.

## Why quality could improve

The estimator does not promise quality merely by being cheaper. It has three
testable advantages at fixed wall-clock budget:

1. it replaces generation and replay for seven of eight candidate pairs with a
   learned eligibility-marginal conditional mean while preserving the exact
   expected gradient;
2. it permits more distinct class-balanced pairs per optimizer step, increasing
   attribute coverage rather than resampling eight descriptions of one pair;
3. when the predictor explains conditional gradient structure, the exact
   residual becomes a lower-variance correction, analogous to a learned control
   variate rather than a biased distillation target.

The relevant adjacent ideas are SAGA's semantic attribute gradient
([arXiv:2606.15134](https://arxiv.org/abs/2606.15134)), conditional neural
control variates
([arXiv:2602.21357](https://arxiv.org/abs/2602.21357)), compute-aware reuse for
stochastic teacher gradients
([arXiv:2605.21489](https://arxiv.org/abs/2605.21489)), Sobolev training with
first-order targets
([arXiv:2604.19011](https://arxiv.org/abs/2604.19011)), and indirect input-gradient
distillation
([arXiv:2312.03286](https://arxiv.org/abs/2312.03286)). Synthetic-gradient
interfaces already predict downstream error signals
([arXiv:1608.05343](https://arxiv.org/abs/1608.05343)), while Q-Prop, LAX, and
REBAR already establish learned or relaxed control variates for unbiased policy
and discrete-variable gradients
([arXiv:1611.02247](https://arxiv.org/abs/1611.02247),
[arXiv:1711.00123](https://arxiv.org/abs/1711.00123),
[arXiv:1703.07370](https://arxiv.org/abs/1703.07370)). vOPD applies a detached
control-variate baseline specifically to on-policy language-model distillation
([arXiv:2605.07865](https://arxiv.org/abs/2605.07865)). Therefore neither gradient
prediction, unbiased learned control variates, nor their use around language-model
policy gradients are claimed as new. ASG-CV's proposed load-bearing combination
is the exact stratified residual correction around an amortized frozen-MLLM
semantic *patch-gradient field* for deep metric learning, with that corrected
field injected through a trainable vision encoder. That narrower novelty claim
remains provisional until the collision audit is complete.

## Falsifier E0 — predictor and variance viability

E0 uses frozen training pairs only and never optimizes or evaluates a retrieval
model. Before opening exact gradients, seal the predictor architecture, rank,
pair partition, stratum sampler, rollout seeds, SRHT rows, optimizer, update
count, and thresholds.

Split sealed exact-gradient pairs by source-image identity so no image appears
in both predictor training and validation. On the validation partition require:

The candidate-pair schedule is created before any completion or gradient is
opened. It consumes only the authenticated ordered example-ID/label manifest
and a sealed SHA-256 seed. Each candidate image appears once, candidate blocks
contain exactly four same-class and four different-class pairs, and all example,
pair, orientation, and block ordering uses domain-separated hashes with
deterministic ID/index ties. Positive pairs are formed within labels; negative
pairs are formed across labels. The complete candidate schedule and source
manifest are content-addressed.

The E0 capture phase generates and classifies candidate completions in schedule
order, without predictor or selection-stream access. Under the collapsed route,
groups with at least one valid completion of each verdict receive exact
four-boundary two-branch targets using the lowest seed ordinal of each verdict;
all other groups receive the exact zero target. A canonical nonzero collapsed
receipt therefore has replay branch count `2`, zero generated tokens, and the
two branch completion ordinals. Exact eight-branch receipts are a different
capture mode and cannot be mixed into the same marginal schedule. All rows
remain in their original four-positive/four-negative
candidate strata. The complete candidate ordinals and candidate-schedule digest
are sealed before exact-gradient replay and before the ASG-CV selection schedule
is revealed; no outcome-conditioned refill exists. Predictor training and
validation use disjoint Cars training class bands, which also guarantees
disjoint image identities; official Cars test classes remain inaccessible. One
content-addressed partition authority binds the source example manifest,
partition seed, and exact sorted predictor-training, E0-validation, and
E1-optimization class-ID bands. Bundle validation rejects any class overlap,
source-image ID reused across phases, row outside its registered band, or
authority that exposes the official test partition.

Generation is bound to one content-addressed rollout authority containing the
model revision, master seed, temperature, top-p, maximum generated tokens, and
the fixed eight-rollout count. Each candidate ordinal deterministically derives
eight distinct 64-bit seeds from that authority. Completion-group receipts bind
the authority digest, candidate ordinal, and exact seed block; bundle validation
reconstructs all three before accepting rewards or eligibility.

Before any capacity pilot or gradient corpus is opened, a sealed training-only
P32 pilot uses exactly `32` candidate pairs from the predictor-training class
band, balanced `16/16` by relation sign, with eight registered rollouts per
pair. P32 uses a domain-separated schedule that cannot become an E0 row, reads
no retrieval test data, persists no dense gradient arrays, and is always
`claim_eligible=false`. It measures the assumptions introduced by the analytic
two-branch collapsed-verdict field rather than treating that field as an
implementation-equivalent replacement for eight-branch replay.

For every candidate P32 seals all completion validity flags, verdict signs,
rewards, attribute spans, generated-token counts, teacher-forced completion
scores, and synchronized phase timings. When both verdicts are present it
computes a lowest-seed-ordinal correct/incorrect branch pair. When at least two
byte-distinct valid completions of each verdict are present it also computes the
corresponding highest-seed-ordinal pair and records normalized branch-exchange
energy. Duplicate lowest/highest completions are ineligible rather than
zero-energy evidence. The
first four schedule-ordered nonzero-variance candidates additionally run the
registered eight-branch exact replay for a latency baseline and an ungated
field diagnostic. No candidate is chosen for favorable agreement or runtime.
The independently replayed branch scores must agree with the selection pass to
an absolute tolerance of `1e-6`, and the derived backend coefficient to `2` ppm;
this admits bounded floating-point replay drift without allowing branch changes.

P32 passes only if all of the following preregistered gates pass:

- median normalized branch-exchange energy is at most `350,000` ppm and is
  evaluable on at least `8/32` candidates;
- median and p90 within-verdict score-dispersion ratios are at most `250,000`
  and `500,000` ppm;
- both-verdict branch yield is at least `500,000` ppm overall and `375,000`
  ppm within each relation sign;
- median collapsed coefficient is at least `200,000` ppm and median absolute
  teacher-score-probability versus empirical correct-verdict probability,
  conditioned only on protocol-valid completions, is at most `250,000` ppm;
- at least `750,000` ppm of the `256` completions are protocol-valid;
- the p90-projected collapsed semantic-step wall ratio is at most `250,000`
  ppm, peak CUDA reserved memory is at most `96 GiB`, and candidate p90 wall
  time is at most `300` seconds.

The projected ratio uses measured synchronized medians and p90s for pair
preparation, generation, eight-completion scoring, two-branch replay,
eight-branch replay, and predictor forward. The full `1024`-row capture estimate
separately multiplies replay work by measured branch and variance yields.
For a stratum of eight pairs, the collapsed step ratio is
`(t_prepare + t_generate + t_score + t_two_branch + 8*t_predictor) /
(8*(t_prepare + t_generate + t_eight_branch))`; the factor eight is the
registered SAGA baseline's eight exact semantic pairs, not a timing-unit
conversion. Both median and p90 ratios are published, and the p90 ratio owns
the gate. The result also publishes the exact candidate counts contributing to
the coefficient, calibration, dispersion, collapsed-timing, and exact-timing
populations so differently conditioned summaries cannot be mistaken for one
common sample.
Collapsed-versus-exact cosine on four rows is reported but is not a gate: one
eight-rollout draw cannot separate Rao-Blackwell shrinkage from field error.

Every candidate and terminal binds the launch-authority digest and predictor
initialization seed. An execution or result-assembly failure writes a canonical
failure receipt and exits nonzero; a completed scientific result, including a
valid `passed=false` result, exits zero. P32 failure is terminal for the
implicated route. Branch-exchange or dispersion
failure rejects the two-branch collapse and returns to measured eight-branch
replay. Yield, coefficient, calibration, or validity failure rejects the frozen
model's ASG-CV semantic-signal premise. Runtime or resource failure rejects GB10
execution. Thresholds cannot move after the first candidate receipt is written.

Only after P32 passes, a sealed capacity pilot uses `64` candidate pairs and
two independently derived rollout-seed blocks per pair. Three lower bounds are
computed from the exact `[pair, image, patch, channel]` gradient fields:

- the paired-seed conditional-variance floor
  `sum ||g-g'||^2 / sum (||g||^2+||g'||^2)`;
- the residual energy outside the best fixed rank-16 channel subspace over all
  captured fields;
- the residual energy outside the best rank-16 matrix approximation of each
  captured pair field, aggregated by gradient energy.

The noise floor remains a capacity gate at `0.35`. After fitting, predictor
error is evaluated conservatively on both protocol-separated validation
blocks: the registered residual is the larger of the two per-block normalized
residual energies; every higher-is-better statistic records the smaller block
value, every remaining lower-is-better statistic records the larger block value,
and p99/clip statistics use the combined two-block norm distribution. Auxiliary
randomization audits likewise record the smaller p-value and larger magnitude
across blocks. Block order therefore cannot select a favorable gate, and no
negative point estimate is treated as a scientific terminal. Exact-block array
authority has a deterministic content order, with receipt identity breaking the
tie when numeric blocks are equal. Both exact blocks and the predicted block are
content-bound into the E0 result, and both exact blocks are reopened row by row
against disjoint captured-gradient receipt sets. A failure closes this predictor family before retrieval; the rank,
sample count, and seed blocks are not adapted after observing validation.

- worst-block median dense gradient cosine at least `0.85` and worst-block
  median SRHT-projected gradient cosine at least `0.90`;
- patch-salience Spearman correlation at least `0.80`;
- worst-block normalized residual energy at most `0.35`;
- empirical variance of the registered one-of-eight estimator at most `0.60`
  times the variance of one exact pair gradient at equal stratum scale;
- the source-seed-derived, one-sided `95%` stratum-bootstrap upper bound on the
  realized estimator mean error, normalized by RMS exact-stratum-mean energy,
  at most `0.15` (`150,000` ppm), using exactly `10,000` deterministic draws;
- pre-clip semantic-field estimator p99 at most `2.0x` the exact estimator and clip
  activation no more than `5` percentage points higher;
- auxiliary selection-independence evidence from a source-bound signed
  patch/channel sketch, reported as an exact `10,000`-draw randomization
  p-value and a diagonal-free U-statistic z score. These two scale-invariant
  quantities audit accidental alignment between the registered selection
  stream and the predictor errors; they are not substitutes for the powered
  mean-agreement gate and are not adaptive pass thresholds;
- no systematic class or relation-sign bias, evaluated only after exact
  receipt custody joins every eligible row to its sealed class/relation
  authority. The relation audit binds the complete custody digest, reports
  separate source-derived `10,000`-draw within-sign selection p-values for the
  realized positive and negative subsets, and reports the normalized projected
  residual-mean contrast between signs. These are claim-ineligible diagnostics,
  not post-hoc pass thresholds; class-stratified evidence additionally requires
  the sealed eligible schedule join;
- measured semantic wall time, including generation, vision forwarding, replay,
  and synchronization on both arms, at most `0.35` times the measured SAGA
  semantic wall time, with peak memory inside the GB10 envelope.

Failure closes ASG-CV without a retrieval run. Thresholds are not retuned on E0.

## Falsifier E1 — matched one-seed retrieval

Only if E0 passes, compare the following on a clean optimization band with the
same backbone, metric loss, retrieval evaluation protocol, and sealed total
semantic wall-clock budget. Pair count is an outcome of that budget rather than
an equality constraint, because increased candidate-pair coverage is the
method's quality mechanism:

1. the capacity-matched SAGA reproduction;
2. ASG-CV with the frozen one-of-eight estimator;
3. predictor-only gradient, as a bias control;
4. exact one-pair semantic replay without the predictor, as a compute-matched
   variance control.

ASG-CV must beat both compute-matched controls, retain the E0 residual-energy
gate during training, and improve Recall@1 by at least `0.5` point over the
matched SAGA seed before three-seed evaluation. Predictor-only success without
the exact residual does not support the control-variate mechanism.

The Cars test classes remain sealed until the method, compute budget, and all
hyperparameters are frozen. Because Cars influenced method selection, the final
claim also requires one untouched corroborating dataset chosen before training.

## Kernel plan after scalar proof

Three custom kernels are justified only by measured profiling and exact
reference agreement:

1. `fused_semantic_gradient_srht`: accumulate the signed FWHT sketch, channel
   second moments, and patch salience without materializing a second dense
   teacher-gradient plane;
2. `fused_low_rank_gradient_inject`: form the low-rank predictor field and add
   the selected exact residual directly to patch-token gradients, with fp32
   accumulation and scalar-reference parity;
3. `packed_rollout_replay`: share prompt/vision prefix state across eight sealed
   completions while preserving exact token log-probabilities, attention rows,
   and gradient roles.

The CPU/fp32 implementation defines scientific semantics. Triton or CUDA kernels
must match finite outputs, gradient digests, and tie behavior before latency is
considered. A speedup cannot rescue a failed E0 estimator.

## Stop conditions and next action

Stop on authority drift, predictor leakage across image identities, estimator
mean bias, residual-energy failure, non-finite correction, GB10 memory failure,
or failure to clear the wall-time projection. No adaptive rank, sample rate, or
architecture ladder is selected on retrieval accuracy.

The next action is a read-only primary-source collision and estimator review,
followed—only after the current DGX control and SAGA feasibility terminal—by a
small exact-gradient E0 implementation. ASG-CV does not delay or alter the
already running three-seed control.
