# SigLIP Attention-Readout Recovery Diagnostic

Date: 2026-09-05

## Status and purpose

This is a Sfora-only, claim-ineligible causal diagnostic. It asks whether a
contiguous prefix of the intact 27-block SigLIP teacher already contains the
fine-grained retrieval information needed for a cheaper query encoder, and
whether a teacher-initialized attention readout can express that information
in the teacher's deployed 512-dimensional coordinate system.

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
it does not test a trainable attention pooler at an intermediate depth or a
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

Persist CPU FP32 pooled matrices for the seven linear controls and one FP16
depth-18 token plane for the branch-deciding attention readout, plus the
unnormalized and normalized 512-dimensional teacher targets. The token cache
must preserve the exact token count and width reported by the authenticated
model, bind example order, dtype, shape, and bytes by SHA-256, and remain a
local training-only artifact. At the observed 729 tokens, 1,152 width, and
6,709 total cached images, that token plane projects to 11,268,543,744 bytes
(10.49 GiB). The implementation must derive and record exact observed bytes
rather than trust this projection.

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

Controls are the sealed ridge and refined linear readouts at every registered
depth plus the exact depth-27 teacher identity. They are reported but cannot be
selected. Random projection and target-derangement sweeps are deliberately
omitted: neither changes the branch decision, while the latter would duplicate
the dominant optimization cost. The canonical result instead records the
learned cell's initial loss, final loss, and all final 200 update losses so
convergence is observable rather than represented by an unevaluated boolean.

## Branch-deciding attention readout

The linear cells are diagnostic controls, not a capacity ceiling. A linear
map after a frozen attention head can fail even when the depth-18 token state
contains enough information for a small learned readout.

The decisive cell therefore freezes the entire encoder and trains only a
fresh LayerNorm, a copy of the teacher's native MAP attention-pooling head, and
the teacher's bias-free 1,152-to-512 projection. Initialize all three from the
teacher and apply them directly to cached depth-18 tokens. Optimize pure
teacher-descriptor recovery, `mean(1-cos(z_student,z_teacher))`, with no class
labels or metric loss. Use the same fixed seed-17 schedule, final-only sealing,
and optimization-limited evidence as the linear refinement. Evaluation data
remains inaccessible until the readout is sealed.

This cell adds training-time capacity only; inference remains the contiguous
18-block prefix followed by the same LayerNorm, MAP head, and 512-dimensional
projection topology used by the teacher. Requalify the complete path against
the <=0.75 latency requirement before advancing it.

## Retrieval evidence and decisions

For every sealed linear cell and the depth-18 learned-attention cell on classes
49..81, compute both:

- self retrieval: readout queries against readout gallery;
- teacher compatibility: readout queries against the exact teacher gallery.

Use identical example order, same-ID exclusion, stable lowest-ordinal ties,
float64 cosine scoring, and the existing independently validated Recall@1 and
mAP@R authority. Recompute all aggregate metrics from per-query evidence.

Decision precedence is:

1. `invalid`: any authority, identity, topology, finiteness, replay, depth-27,
   control, or result-recomputation gate fails;
2. `quality-qualified`: the depth-18 learned-attention cell has both self and
   cross-gallery hits
   >=2,591 and both mAP@R values >=0.7893744556922272;
3. `compression-promising`: the depth-18 learned-attention cell has both self
   and cross-gallery
   hits >=2,555 and both mAP@R values >=0.7713744556922272;
4. `coordinate-alignment-needed`: the depth-18 learned-attention cell meets
   the promising bounds in self retrieval but not against the teacher gallery;
5. `depth-18-recovery-rejected`: none of the above.

The linear cells cannot select or reject the architecture; they quantify how
much recovery requires nonlinear attention adaptation. The promising tier
authorizes only a structured compression experiment; it is not quality
success. Final product advancement still requires the quality gate plus a
separately authenticated complete-path latency ratio <=0.75, matched baselines, multiple seeds, and
untouched official evaluation.

## Follow-on architecture

If depth 18 is quality-qualified or promising, build a contiguous-prefix
student rather than the failed interleaved deletion. Initialize it from teacher
blocks 1..k, retain the frozen teacher attention pooler, initialize the
teacher-coordinate readout from the sealed cell, and train first with explicit
pointwise teacher-coordinate cosine loss. Add listwise teacher-rank preservation
and then a separately ablated margin-gated supervised-deviation term: preserve
teacher-correct positive-before-negative margins while spending deviation on
teacher-error or low-margin triplets. This is the only proposed mechanism for
beating rather than merely matching the teacher, and it must be compared with
ordinary descriptor distillation plus the same supervised mining. Unlike
Gram-only relational distillation, the coordinate loss removes global rotation
freedom. Benchmark the complete prefix path because the existing interleaved
speed result is supportive but not identical.

If depth-18 recovery is rejected after converged attention-readout fitting, do
not repeat depth deletion with another loss. The next branch is delayed token
merging while preserving all 27 transformations and the native attention head.
Calibrate token count on latency only, then test teacher-descriptor recovery.
Class-name semantics,
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
