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

For the registered Qwen3-VL implementation, `X` is the output of the single
vision `merger` module used by the full language replay, before it is split into
the two image-token spans. The eight sealed completions execute eight distinct
autograd branches through that same deterministic module. E0 retains each
branch output, requires byte-identical stopped patch values and the registered
`2 x 49 x D` shape, derives the detached layer-26 teacher map only from
registered correct completions, applies the pooler attention KL to the first
live branch's identical patch tensor, and performs one backward through the sum
of mean GRPO and KL. It then sums all eight branch-output gradients; the first
contains both GRPO and KL contributions and the remaining branches contain their
GRPO contributions. For deterministic vision replay, this sum is the exact
field whose Jacobian product reproduces the semantic contribution to the
vision-parameter gradient. A separately recomputed attention-only feature
forward is not a valid substitute because it is outside the combined replay
graph.

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
correct-rollout flags, and teacher spans together. A group is semantically
eligible only when it contains at least one correct and one incorrect verdict;
all-correct and all-incorrect groups have zero reward variance and contribute
neither GRPO nor attention KL, matching the registered DAPO refill rule.

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
streams.

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

If a stratum cannot produce its exact residual, the optimizer step is discarded;
it never silently falls back to the predictor-only gradient. A non-finite or
overflowing correction is a fail-closed scientific terminal.

SAGA's declared global gradient clipping at `1.0` is applied only after the
complete DML and semantic gradient is assembled. Because clipping is nonlinear,
the expected *clipped update* is not generally unbiased even when `g_hat_s` is.
E0 derives the pre-clip semantic-field estimator norms and clip activation from
the bound exact and predicted fields plus the source-bound selection seed; it
does not accept self-reported norm arrays. E1 must likewise derive these values
at its registered optimizer boundary. ASG-CV is ineligible if its clip-activation rate is
more than `5` percentage points above the matched exact estimator or if its
pre-clip norm p99 exceeds `2.0x` the exact p99. No residual clipping or
winsorization is allowed to repair this gate.

## Why quality could improve

The estimator does not promise quality merely by being cheaper. It has three
testable advantages at fixed wall-clock budget:

1. it replaces seven of eight expensive pair replays with a learned conditional
   mean while preserving the exact expected gradient;
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

DAPO eligibility is resolved in a separate forward-only phase. Candidate
completions are generated and classified in schedule order, without backward or
predictor access. The first four nonzero-reward-variance candidates of each
relation form each final eight-pair stratum; zero-variance candidates remain
sealed negative evidence but are skipped. The resulting candidate ordinals and
their candidate-schedule digest are sealed before exact-gradient replay and
before the ASG-CV selection schedule is revealed. Insufficient eligible capacity
is a terminal rather than an adaptive resample. Predictor training and
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

Before predictor fitting, a sealed capacity pilot uses `64` eligible pairs and
two independently derived rollout-seed blocks per pair. Three lower bounds are
computed from the exact `[pair, image, patch, channel]` gradient fields:

- the paired-seed conditional-variance floor
  `sum ||g-g'||^2 / sum (||g||^2+||g'||^2)`;
- the residual energy outside the best fixed rank-16 channel subspace over all
  captured fields;
- the residual energy outside the best rank-16 matrix approximation of each
  captured pair field, aggregated by gradient energy.

Each lower bound must be at most `0.35`. A failure closes this predictor family
before fitting or retrieval; the rank, sample count, and seed blocks are not
adapted after observing the pilot.

- median dense gradient cosine at least `0.85`;
- median SRHT-projected gradient cosine at least `0.90`;
- patch-salience Spearman correlation at least `0.80`;
- normalized residual energy
  `sum ||g-q||^2 / sum ||g||^2 <= 0.35`;
- empirical variance of the registered one-of-eight estimator at most `0.60`
  times the variance of one exact pair gradient at equal stratum scale;
- pre-clip semantic-field estimator p99 at most `2.0x` the exact estimator and clip
  activation no more than `5` percentage points higher;
- exact empirical mean agreement within a preregistered bootstrap confidence
  interval and no systematic class or relation-sign bias;
- projected semantic wall time at most `0.35` times the measured SAGA semantic
  wall time, with peak memory inside the GB10 envelope.

Failure closes ASG-CV without a retrieval run. Thresholds are not retuned on E0.

## Falsifier E1 — matched one-seed retrieval

Only if E0 passes, compare the following on a clean optimization band with the
same backbone, metric loss, pair schedule, optimizer-step count, and retrieval
evaluation protocol:

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
