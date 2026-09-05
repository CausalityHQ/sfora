# SigLIP Attention-Readout Recovery Diagnostic

Date: 2026-09-05

## Status and purpose

This is a Sfora-only, claim-ineligible causal diagnostic. It asks whether a
contiguous prefix of the intact 27-block SigLIP teacher already contains the
fine-grained retrieval information needed for a cheaper query encoder, and
whether a small deterministic readout can express that information in the
teacher's deployed 512-dimensional coordinate system.

The question is motivated by authenticated failures, not by a new quality
claim. The intact teacher obtains 2,596/2,746 Recall@1 hits and mAP@R
0.7913744556922272. Both fixed 18-block recovery students pass the <=0.75
speed ratios but fall to 2,428/2,746 and 2,438/2,746 hits, with mAP@R near
0.462. Their cross-teacher-gallery results are lower still because neither
training objective pins the teacher coordinate frame. The existing sealed
intermediate screen also completed: post-LN mean pooling rises monotonically
from 20.8680% Recall@1 at depth 1 to 87.3832% at depth 18 and 96.7953% at
depth 27, while the deployed attention-pooler context obtains 98.4607% on the
optimization band. That result rejects mean pooling as the primary readout;
it does not test the teacher attention pooler at intermediate depths or a
learned teacher-coordinate map.

## Immutable roles

Reuse the exact seed-17 epoch-60 teacher checkpoint, model revision,
preprocessing, dataset manifest, optimization classes 0..48, and exposed
diagnostic classes 49..81 already authenticated by the depth-recovery
campaign. No official-test classes, network input, arbitrary checkpoint,
class-name text, or student checkpoint is an input capability. The process
must bind all source files and input artifacts by SHA-256 before model
execution and emit `claim_eligible=false`.

Depths are fixed before execution as `{6, 10, 14, 18, 22, 25, 27}`. The
optimization band may fit and select regularization. The 49..81 band is read
once only after every readout is sealed; it may score the sealed cells but may
not change depth, regularization, loss, or gates.

## Exact feature path

One intact frozen teacher forward requests all 28 hidden states. For one-based
depth `k`, use `hidden_states[k]`, then apply the teacher's frozen
`post_layernorm` and frozen multi-head attention-pooling `head`. This produces
`h_k` with width 1,152. The zero-train descriptor is
`normalize(W_teacher h_k)`, where `W_teacher` is the authenticated bias-free
1,152-to-512 projection.

This definition is intentionally different from the earlier mean-pooled
screen. At depth 27 it is the deployed teacher path. Under the same batch
partition, BF16 autocast, and FP32 projection, depth-27 zero-train descriptors
must exactly reproduce the separately obtained teacher descriptor digest,
2,596 hits, and mAP@R 0.7913744556922272. Any mismatch is an invalid execution,
not scientific evidence. A repeated extraction must disagree by at most 2e-5
per pooled feature and must reproduce identical normalized descriptors.

Persist only CPU FP32 pooled matrices and targets. Seven 1,152-dimensional
pooled vectors plus the unnormalized and normalized 512-dimensional teacher
targets require 36,352 bytes/image, or 243,885,568 bytes for 6,709 images.
The implementation must derive and record the exact observed byte count rather
than trust this projection. No patch-token cache is retained.

## Deterministic teacher-coordinate readout

For every `k`, fit a bias-free linear map `W_k: 1152 -> 512` from optimization
features `X_k` to the teacher's unnormalized projected outputs `Y`. Let
`B=W_k^T`, `B0=W_teacher^T`, `N` be the optimization row count, and
`s_k^2=||X_k||_F^2/(1152*N)`. Solve exactly

`min_B ||X_k B - Y||_F^2/N + 1e-3*s_k^2*||B-B0||_F^2`.

Use FP64 Cholesky without an explicit inverse, bias, centering, coefficient
sweep, or depth-specific hyperparameter. The equivalent normal equations are
`(X^T X + N*lambda*I)B = X^T Y + N*lambda*B0`, with
`lambda=1e-3*s_k^2`. Record ridge as its own sealed cell.

From ridge, run one fixed directional refinement that optimizes only
`mean(1-cos(normalize(W_k h_i), t_i))`: AdamW with weight decay zero,
betas `(0.9,0.999)`, epsilon `1e-8`, 2,000 updates, batch 256 cached rows,
seed 17 shared across depths, 50-update warmup to `1e-4`, cosine decay to
`1e-5`, FP32 arithmetic, global gradient norm 1, and final update only. Record
the full optimization loss before/after and over the final 200 updates. If the
loss is still improving at the cap, classify the cell as optimization-limited
rather than treating it as a definitive linear-capacity failure. Ridge and
refined cells are both evaluated after sealing; no evaluation outcome selects
between them.

Controls are:

1. zero-train teacher projection at every registered depth;
2. the depth-27 exact identity control;
3. a fixed seed-17 Gaussian 1,152-to-512 random projection with entry variance
   `1/1152`;
4. a fixed seed-17 derangement of optimization image-to-target assignments,
   fit with the same ridge and refinement budget.

The last two controls are reported but cannot be selected. They are not
required to reach chance because pretrained features and teacher initialization
remain informative; their purpose is to measure incremental improvement and
detect target-identity shortcuts.

## Retrieval evidence and decisions

For each sealed depth on classes 49..81, compute both:

- self retrieval: readout queries against readout gallery;
- teacher compatibility: readout queries against the exact teacher gallery.

Use identical example order, same-ID exclusion, stable lowest-ordinal ties,
float64 cosine scoring, and the existing independently validated Recall@1 and
mAP@R authority. Recompute all aggregate metrics from per-query evidence.

Decision precedence is:

1. `invalid`: any authority, identity, topology, finiteness, replay, depth-27,
   control, or result-recomputation gate fails;
2. `deployment-grade`: some depth <=18 has both self and cross-gallery hits
   >=2,591 and both mAP@R values >=0.7893744556922272;
3. `compression-promising`: some depth <=18 has both self and cross-gallery
   hits >=2,555 and both mAP@R values >=0.7713744556922272;
4. `nonlinear-alignment-needed`: some depth <=18 meets the promising bounds in
   self retrieval but none meets them against the teacher gallery;
5. `linear-readout-rejected`: none of the above.

Select the smallest qualifying depth, then higher cross mAP@R, then higher
cross hits. The promising tier authorizes only a structured compression
experiment; it is not quality success. Final product advancement still
requires the deployment-grade gate, <=0.75 measured inference time, matched
baselines, multiple seeds, and untouched official evaluation.

## Follow-on architecture

If depth <=18 is deployment-grade or promising, build a contiguous-prefix
student rather than the failed interleaved deletion. Initialize it from teacher
blocks 1..k, retain the frozen teacher attention pooler, initialize the
teacher-coordinate projection from `W_k`, and train with explicit pointwise
teacher-coordinate cosine loss plus positive-before-negative teacher margin
constraints across the full logical batch. Proxy Anchor remains a secondary
class objective. Unlike Gram-only relational distillation, the coordinate loss
removes global rotation freedom. Benchmark the complete prefix path because
the existing interleaved speed result is supportive but not identical.

If linear readout is rejected, do not repeat depth deletion with another loss.
The next branch is token/width compression or adaptive early exit, preserving
all 27 transformations while reducing per-block work. Class-name semantics,
hyperbolic geometry, optimal transport, and diffusion are not introduced at
this boundary: class-language guidance is established prior art, may collapse
fine make/model/year distinctions, and does not itself reduce inference cost.

## Files and verification

Create a sibling module and runner; do not mutate the sealed intermediate or
recovery result formats:

- `src/sfora/siglip_attention_readout_recovery.py`: ridge fitting, metrics,
  decisions, and canonical validation;
- `scripts/probe_siglip_attention_readout_recovery.py`: strict local artifact
  boundary, streamed teacher extraction, sealing, and one evaluation read;
- matching focused tests for exact attention-pooler identity, fold isolation,
  regression/control arithmetic, retrieval evidence, schema mutations, and
  forbidden capabilities.

Use TDD. Before DGX execution, require focused tests, dependency-complete
Python discovery, Ruff, formatting, mypy, py_compile, diff-check, and two
independent Astra/Fable reviews of the concrete implementation. Run one
pressure-monitored original process and preserve its sole terminal result.
