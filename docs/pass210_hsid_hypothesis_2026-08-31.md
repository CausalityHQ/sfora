# Pass 210 — Hadamard Semantic Influence Distillation hypothesis

Date: 2026-08-31
Status: **REVISE at primary-source audit — no source, model acquisition, or GPU
execution authorized.** The subsequent audit in
`docs/pass210_hsid_primary_audit_2026-08-31.md` closes the broad causal-distillation
novelty claim and retains only a claim-ineligible amortization diagnostic.

## Why this lane exists

The current SigLIP-so400m pooled control is `1,242 / 1,345` (`0.923420`)
on the burned Cars train band, below the registered `0.94` headroom gate. RSTA
Stage A is still pending and can decide whether receiver-self tangent alignment
is a defensible correction for that controlled lane. It cannot by itself justify
forecasting the roughly five-point increase needed to clear the global Cars
horizon.

The new external observation is SAGA (arXiv:2606.15134), which reports
`97.0 +/- 0.3` Cars R@1 with a Qwen3-VL-8B vision tower. Its main ablation says
the frozen-MLLM GRPO term supplies most of the gain, while attention alignment
alone is much smaller. This supports a supervision-information hypothesis:
fine-grained transfer improves when the training signal identifies *which visual
evidence* resolves a relation, rather than applying one scalar class relation to
every feature direction.

SAGA is not a local implementation baseline. It uses eight stochastic rollouts
per pair, differentiable replay through the language backbone, dataset-specific
attribute vocabularies, and an H200. Reproducing it blindly on the GB10 would be
expensive and would not establish a distinct method. HSID asks whether the same
load-bearing information can be extracted as deterministic, cached causal patch
evidence and transferred through a cheaper retrieval objective.

## Hypothesis

For a frozen MLLM relation supervisor, let

```text
v(X_a, X_b, r)
```

be the teacher log-odds margin of the correct versus incorrect relation token `r`
for image patch tokens `X_a, X_b`. The prompt, tokenizer, relation tokens, model
revision, image processor, and pair order are immutable authority. No generated
description or free-form parse participates in the score.

One deterministic VJP through the frozen language backbone gives the teacher
influence field for image `a`:

```text
G_a = partial v / partial X_a,             G_a in R^(P x D)
w_ai = <G_ai, X_ai>.
```

The scalar `w_ai` is the first-order multiplicative influence of patch `i`. A
source-bound signed subsampled randomized Hadamard transform `R in R^(K x D)`
compresses the feature-direction evidence for storage:

```text
S_ai = R G_ai,                             S_ai in R^K.
```

This is a compressed local gradient, not a Shapley value and not the full MLLM
Jacobian. On a fixed H0 audit subset, central differences along registered
Hadamard patch-amplitude directions must agree with the VJP directional
derivatives. The finite differences audit the local causal interpretation; they
are not run for every cached pair.

The fixed pair schedule assigns each image to two disjoint partner blocks. These
supply a selection-independent stability measurement. Scale each patch influence
and sketch coordinate by the square root of its cross-partner second moment,
clipped before normalization, to stop high-variance teacher responses from
dominating:

```text
u_hat_i = w_i / sqrt(F_i + epsilon_F).
```

This diagonal score-metric normalization is an estimator-stability device. It is
not advertised as a novel natural-gradient optimizer.

Each training image participates in a fixed number of same-class and hard
different-class pairs chosen without evaluation outcomes. Aggregate its
pair-conditioned `u_hat` maps and projected sketches with coordinate-wise
median-of-means over fixed partner blocks. Clip the robust scalar map to `[-3,3]`
and apply a unit-temperature softmax over patches; the result `alpha_x` is a
nonnegative unit-mass patch target. The MLLM and all pair evaluations are then
removed from the training process.

The student has a trainable attention pooler with patch weights `beta_x` and one
ordinary normalized descriptor `z_x`. Its initial candidate objective is

```text
L = L_DML
  + lambda_att KL(stopgrad(alpha_x) || beta_x)
  + lambda_ot  L_signed_transport.
```

`L_DML` is the same-backbone controlled metric loss. The transport term uses a
small entropic plan between the highest-mass teacher patches of a registered
pair. Its fixed cost combines student-token distance and cosine distance between
stopped robust teacher sketches. For same-class pairs it minimizes transported
student-token distance; for hard different-class pairs it applies a margin only
to edges whose teacher sketches identify a shared comparison axis. Marginals are
the stopped teacher weights. Transport never balances global sample popularity
and is never used at retrieval time. The deployed representation remains one
descriptor and cosine nearest-neighbour search.

This objective is deliberately provisional. A primary-source audit must decide
whether causal token intervention plus cached robust influence aggregation and
signed local transport is distinct from SAGA, attention transfer, perturbation
attribution distillation, DIML/DeepEMD, and language-guided local metric learning.
An exact collision kills or narrows HSID before implementation.

## Relationship to RSTA

HSID does not reinterpret the pending RSTA result:

- `PASS_ONWARD`: retain HSID as the high-capacity semantic lane and allow one
  separate RSTA interaction ablation after HSID independently clears its clean
  paired gate. RSTA is not folded into the primary HSID arm.
- `FAIL_CLOSED`: omit every receiver-self term. HSID survives only as the
  independently motivated semantic-information lane.
- `UNRESOLVED` or invalid execution: omit RSTA from HSID and repair or close the
  diagnostic separately.

Thus a favorable tangent result cannot manufacture semantic-supervision evidence,
and a semantic result cannot retroactively validate RSTA.

## Smallest falsifiers

### H0 — supervisor and intervention viability

Use only a frozen optimization band. Before opening images, seal teacher/model
authority, pairs, Hadamard rows, `epsilon`, relation tokens, and thresholds.
Measure:

1. deterministic teacher relation accuracy and log-odds margin;
2. cosine/Spearman stability of `u_hat` across the two disjoint partner blocks;
3. localization contrast against permuted influence maps and partner-deranged
   controls;
4. fraction of pairs with finite, non-flat influence; and
5. VJP-versus-central-difference directional agreement on the registered audit
   subset; and
6. measured teacher forwards/backwards, wall time, and peak memory.

Close HSID if the teacher relation signal is weak, influence stability does not
beat both controls, or the projected preprocessing budget is infeasible. H0 does
not train a retrieval model and cannot select a final hyperparameter.

### H1 — cached-target pooler falsifier

Freeze one same-backbone student, one `lambda_att`, and one training budget. Train
only the attention-distillation arm against a matched DML control. Evaluate on a
clean class band unavailable to H0. Require a positive paired gain and prove that
the learned `beta` is closer to authentic `alpha` than to the two controls. A
quality gain without authentic-target selectivity is not mechanism support.

### H2 — signed transport contribution

Only if H1 passes, add the single frozen transport arm and three controls:

- attention-only;
- pair labels with uniform patch mass; and
- authentic patch mass with deranged pair relations.

The transport arm must beat attention-only and every control under the same
training budget. Otherwise retain the simpler attention-distillation method or
close the transport claim.

### H3 — SOTA evaluation

Seal the full method before any final evaluation. Report paired same-backbone
controls and capacity-unrestricted comparisons separately. Cars requires a
three-seed mean at least `97.4` and a paired gain at least `0.5` point. Because
Cars outcomes have influenced this search, a publication-level claim additionally
requires untouched CUB, SOP, In-Shop, Aircraft, or iNat evidence selected before
training and evaluated once. No test-time fitting, reranking, gallery adaptation,
or multi-view descriptor is permitted.

## Compute and custom-kernel opportunity

The expensive phase is outcome-blind teacher preprocessing. It needs one
deterministic relation forward and one VJP per pair, plus central-difference
forwards only on the small registered audit subset. It performs no autoregressive
rollouts. H0 must measure whether batching and cached preprocessing make this
cheaper than SAGA's eight-rollout plus differentiable-replay schedule; this is a
falsifiable engineering claim, not an assumption.

Two kernels are plausible only after scalar correctness:

1. `fused_srht_gradient_sketch`: apply the signed FWHT feature projection and
   accumulate patch influence/second moments without materializing the full
   cached `P x D` gradient plane; and
2. `fused_sparse_signed_ot`: perform log-domain Sinkhorn and the signed local
   margin reduction over the registered top-mass patches without materializing a
   full batch-pair transport tensor.

CPU/fp32 reference implementations define the scientific result. Triton/CUDA must
match finite outputs and selected patch/edge ties before timing. Kernel speedups do
not rescue a failed scientific gate.

## Decision now

HSID is a prospective hypothesis, not the next authorized GPU job. The immediate
critical path remains: complete and authenticate the three-seed SigLIP control,
run the already committed RSTA Stage-A falsifier once, and preserve its result.
In parallel, HSID may receive a read-only primary-source collision audit and an
operation-count review. No implementation or data acquisition should begin until
both are complete and the DGX control campaign is terminal.
