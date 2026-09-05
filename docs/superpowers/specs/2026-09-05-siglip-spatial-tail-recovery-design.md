# SigLIP Spatial Tail Recovery Design

## Purpose

Determine whether a latency-bounded module with cross-token interaction can
approximate the transferable computation of SigLIP vision blocks 19 through 27
from the frozen depth-18 token field. This is a causal development experiment,
not a final benchmark or SOTA claim.

The preceding learned native attention readout converged on optimization
classes (descriptor cosine loss 0.7566115475501071 to
0.008516397059279621) but transferred poorly to disjoint classes: self-space
2,463/2,746 Recall@1 and 0.5144812425459536 mAP@R; student-query to
teacher-gallery 1,030/2,746 and 0.22595743238755656. The teacher is
2,596/2,746 and 0.7913744556922272. Canonical evidence SHA-256 is
`93e43e6bfd9984a0b6789ef8a90b4a8580b72859a795f82fe238eb3897a9dd8e`.
This rejects a global attention readout as the complete repair. It does not
show that the depth-18 token field lacks the information required by the
teacher, because the teacher tail is itself a deterministic function of that
field.

## Scientific question

Does inexpensive cross-token computation recover class-transferable teacher
geometry beyond an equally supervised tokenwise transformation?

Two arms receive the same depth-18 inputs, depth-27 spatial targets, teacher
descriptors, sample order, optimizer, and update budget:

1. `tokenwise-control`: two residual tokenwise MLP updates. Tokens never
   exchange information.
2. `latent-interaction`: eight width-128 latent variables read the spatial
   field, interact once, and write a residual back to every spatial token.

Both predicted token fields pass through frozen copies of the teacher's final
LayerNorm, native attention pooler, and projection. The treatment-control
difference therefore isolates cross-token interaction rather than supervision,
readout, or output coordinates.

## Development isolation

Use only the existing optimization labels 0 through 48. Assign a class to the
development fold when the first eight bytes of
`SHA256("sfora-spatial-tail-dev-v1\0" || decimal_label)` interpreted as an
unsigned big-endian integer are among the ten smallest class hashes. The other
39 classes form the fitting fold. This exact split is computed before feature
extraction and is immutable.

Fitting-fold pixels may produce optimizer inputs and fitting-only residual RMS
statistics. Development-fold pixels may be decoded only after both final arm
artifacts are sealed. They may never affect fitting, normalization statistics,
checkpoint selection, or hyperparameters. The external labels 49 through 81
remain inaccessible in this experiment. A development pass can justify a later
frozen evaluation, but cannot itself establish publication quality.

## Token authority

One frozen teacher forward supplies hidden states 18 and 27 for every fitting
image. Targets are the final hidden state immediately before the teacher's
`post_layernorm`. The frozen teacher descriptor is
`normalize(projection(head(post_layernorm(H27))))`. The runner must mutation-
lock hidden-state indexing and exact depth-27 identity against the model's
native `pooler_output` under the same BF16 autocast boundary.

Cache fitting tensors as contiguous CPU FP16 token fields and CPU FP32
normalized descriptors. The cache is ephemeral and is never a released model
artifact. Development tensors are streamed after sealing rather than retained
during training.

## Models

All trainable projections include biases unless stated otherwise. Every arm
accepts `[batch,tokens,1152]` and returns the same shape.

`tokenwise-control` applies input LayerNorm, then two pre-norm residual MLP
updates. Each MLP is `Linear(1152,128)`, GELU, `Linear(128,1152)`. The final
output is `H18 + delta1 + delta2`.

`latent-interaction` applies input LayerNorm and `Linear(1152,128)` to spatial
tokens. Eight learned width-128 latent queries perform four-head cross-attention
over the spatial tokens, then one pre-norm four-head latent self-attention plus
a width-256 GELU MLP, then spatial queries perform four-head cross-attention
over the updated latents. `Linear(128,1152)` writes the result as a residual on
H18. Attention dropout and all other dropout are zero.

Trainable weights use seed 20260905. Neither arm initializes from blocks 19
through 27; the experiment measures supervision and interaction, not direct
tail copying. Frozen readout weights are not trainable.

## Objective and optimizer

For each channel, compute the fitting-only RMS of `H27 - H18` in FP64 fixed
sample order. Floor every channel scale at `1e-3 * median(nonzero RMS)`. Reject
nonfinite values or a zero median.

For predicted field `P`, target `H27`, predicted descriptor `s`, and teacher
descriptor `t`, use:

`L = mean(((P-H27)/scale)^2) + (1 - dot(s,t)).mean()`.

Use AdamW, learning rate `3e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, weight
decay `0.01`, gradient-norm cap 1.0, batch 64, 4,000 updates, 200-update linear
warmup, then cosine decay to `3e-5`. Generate the same deterministic shuffled
index stream for both arms. Train each arm from its initial seed once. No class
loss, ranking loss, augmentation change, language feature, or checkpoint
selection is allowed.

## Evidence and decisions

Before development access, persist both arms in one safetensors artifact and
reload it. Bind its SHA-256 into the canonical result.

On the development fold, record per-query hits and average precision for:

- the frozen depth-18 native-pooler baseline;
- the tokenwise control;
- the latent-interaction treatment;
- the teacher;
- each student query against the teacher gallery.

Independently recompute counts and mAP@R. Let `B`, `C`, `S`, and `T` be the
baseline, tokenwise control, treatment, and teacher aggregate values. For each
of Recall@1 and mAP@R, define treatment gap closure `(S-B)/(T-B)` and reject a
nonpositive teacher gap.

Classification precedence is:

1. `invalid` for authority, schema, nonfinite, isolation, artifact, or identity
   failure;
2. `interaction-rejected` unless treatment exceeds control in both metrics and
   closes at least 0.5 of the teacher gap in both metrics;
3. `latency-pending` when the quality gate passes. No external evaluation is
   authorized by this result alone.

If quality passes, benchmark the complete 18-block encoder plus treatment and
frozen readout using the existing paired latency protocol. The final deployed
path must remain at or below 0.75 of teacher p95 in every registered window.
Latency is not inferred from block counts.

## Resource and safety envelope

Run on one DGX GPU with a 90-minute wall cap, 96-GiB in-process CUDA reserved-
memory cap, 110-GiB process-group RSS cap, immediate stop at memory PSI full
avg10 0.79, sustained stop at 0.50 for three samples, and stop on swap growth.
Never run two scientific copies. Preserve the original terminal and canonical
receipt. Explicit cleanup occurs only after PID clearance.

## Publication interpretation

A pass would support the claim that late-block contextual computation is
compressible and that token-field supervision transfers better than descriptor
regression. It would still require replication, measured full-path latency,
one frozen external evaluation, and matched public benchmark comparisons.
Language semantics may later choose hard negatives but must not define visual
equivalence. Hyperbolic geometry is out of scope because it cannot manufacture
missing local visual evidence.

