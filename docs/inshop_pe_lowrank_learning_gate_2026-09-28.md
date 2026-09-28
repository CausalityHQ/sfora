# Fixed rank32 PE adaptation gate

Decision: conditional GO for native CPU qualification, one17-step GPU
mechanics attempt, then one100-update TRAIN feasibility pilot if mechanics
and cost pass. This is a new complete training procedure, not approximation
of the stopped dense delta or an extension of the old PE100 recipe. No rank,
LR, prefix, budget, precision or epoch rescue after an outcome.

## Review reconciliation

Fable05f2cad4e3ad478f completed exit0 in515s and recommended STOP. Its
claim that monotone improvement implies low-rank learning can help only by
over-adaptation is unsupported. The saved two endpoints do not measure a
trajectory; counterfactual heads are not checkpoints along time. The
[closed-form premise check](../scripts/check_lowrank_learning_stop_premise.py)
has both dense and constrained held errors improving at every analytic step
while the constrained error is lower. This is a toy counterexample to that
inference, not a PE/LoRA quality prediction or factor-Adam test.

[Hu2021](https://arxiv.org/abs/2106.09685) reports parity or better on named
language models; [Biderman2024](https://arxiv.org/abs/2405.09673) reports
code/math underperformance and better retention elsewhere. Neither establishes
a near-certain PE/In-Shop failure. Only these primary abstracts were checked
here. Opus7dc2f985311643fd and Astra b6a7737af9f6464d, dual28efdc35b5464386,
both completed normally (126/117s) with conditional GO. Their separately
labelled [full reviews](evidence/compact_metric/sop-siglip2-substrate-v1/lowrank-learning-research-v1/dual-result.json)
are retained. Adopt Astra's stricter quality intervals below. Do not adopt
Opus's rough function-space step estimate or assertion that unchanged serving
architecture preserves the old training-time ratio. Cost must be measured.

## Frozen method

Use pinned PE-Core-B16-224, original FP32 source, upper blocks6–11 only:
24 fused-QKV/output/MLP matrices listed in the authenticated
[inventory](evidence/compact_metric/sop-siglip2-substrate-v1/lowrank-learning-research-v1/inventory.json).
Freeze the native prefix first; register `W0 + BA` through torch parametrize
afterward and explicitly freeze originals. Rank32/alpha32 gives scale1;
A is Kaiming-uniform with `a=sqrt(5)` (bound1/sqrt(input width)), B zero.
Lexical matrix order, isolated CPU factor RNG seed179034, restore caller RNG.
Materialize BA and W0+BA in autocast-disabled FP32; then use the unchanged
native forward under FP16 autocast. No new dependency or serving layer.

Factors LR1e-4; remaining native dense norms/biases/pool/projection LR1e-5;
head/proxies LR1e-4. AdamW decay.05, global clip1, scaler128. Parameterization,
optimizer coordinates, decay placement, initialization and clip norm jointly
define the method; do not claim an isolated rank effect or calibrated LR.
Preserve exact head/proxy/bank initialization, seed179032 first100 augmented
batches, RGB/native-pixel hashes, singleton handling and original objectives.
Use TRAIN-fit13283images/2004products and TRAIN-held6354q/6245g/1993products,
with the original authenticated role hashes. No official scoring or selection.

## Qualification and ordered execution

1. CPU120s/8GiB/CUDA-hidden: authenticate all unchanged source/environment,
   checkpoint/config, executing module origins, initializers/schedule/images
   and archived controls; pin intentional new files separately. Check actual
   native zero-B source-forward parity on fit-only inputs, exact24/48 matrix/
   factor inventory, optimizer identity coverage and freeze/foreign-rope
   state. A's first data gradient is exactly zero at B=0; require connected
   nonzero B gradients and A differentiation after a nonzero B fixture.
   Nonzero FP32 materialization, merge, native strict key/shape set and
   save/reload must qualify actual native outputs. No optimizer update or
   held-quality read in this CPU prerequisite.
2. One17-step DGX GPU mechanics attempt,120s/8GiB host/<10GB CUDA,
   shared lifetime lock, no held quality. Same authenticated first17 batches,
   no skipped steps, complete finite data-gradients/parameter/state audits,
   original/prefix/rotary unchanged, updated merged-checkpoint parity. Require
   median guarded update steps3–17 <=0.71769696s (0.8× archived Large median).
   Stop on any integrity, cost or resource failure. Discard its trained state.
3. Restart from authenticated source/initializers for one100-update pilot,
  300s whole process including initialization/serialization/evaluation,
  8GiB host/<10GB CUDA, same shared lock and exclusive attempt reservation.
   Evaluate the strict-loaded merged checkpoint, comparing its actual vectors
   and packed hits/AP to in-process updated-model inference (cosine>=.999999,
   per-query hit/AP difference<=1e-6). Old-model golden parity is insufficient.
   Preserve raw per-query outcomes, gradients, input digests and cost.

The small bare-parameter/Linear CPU fixture has passed zero-B gradients,
autocast-disabled FP32 materialization, nonzero merge and strict save/reload.
Its initial RED was missing helper module; GREEN on DGX exit0. Actual native
CPU qualification now passed (invocation2515165e7457443c8cacfd45cc3bf676,
exit0): 9.50s process wall, 1,953,412KiB maximum RSS. Both TRAIN-fit-only
inputs reproduced source outputs bitwise at B=0; all24 sites/48 factors
qualified connected B gradients, zero initial A gradients and later nonzero
A data gradients. Frozen prefix/foreign rotary/originals stayed exact. A
synthetic nonzero fixture merged with exact native output, restored all163
native keys/shapes, and reproduced updated output bitwise after strict save
and fresh native reload. No optimizer, CUDA or held images were used.
[Raw native receipt](evidence/compact_metric/sop-siglip2-substrate-v1/lowrank-learning-research-v1/native-cpu-v1.json).
Isolated code root: `/home/riomus/runs/sfora-pe-lowrank-v1`.
Concrete17-step implementation review f9ed1be2d0904b09 completed: Opus
8b51c8e6d74a4463 conditional GO; Astra58a1d25a642e4bb4 required resource fixes.
Verified and corrected mandatory new-file/CPU-receipt hash binding, retained
qualification hash in attempt, canonical plus legacy lifetime locks, post-merge
and strict-loaded GPU peak checks, whole-attempt119s runtime+1s shutdown,
8GiB RAM/no swap. The current executable additionally verifies nonzero dense
data gradients, group movement, post-load/final executing authority, and actual
updated-checkpoint GPU strict reload rather than only in-memory merge. Trained
state is discarded. All method/quality/cost floors remain frozen. CPU actual
startup passed before CUDA. The sole17-update attempt then passed; see
[measured native result](inshop_pe_lowrank_mechanics_result_2026-09-28.md).
No100-update pilot has been launched.

## Frozen TRAIN100 advance rule

ALL required: merged packed R@1>=95.1720176%, mAP@R>=77.6237120%, and
median guarded steps3–100<=0.71769696s. Additionally, paired product-bootstrap
95% lower bounds for LoRA−archived dense PE must be strictly positive for
BOTH R@1 and mAP@R, using the inherited5000 resamples/seed/role estimator.
Any resource or authority failure stops. Reuse archived controls only when
all unchanged source/environment/schedule/initializer/RGB/evaluation authority
matches; disclose historical cross-process nondeterminism. Exposure is matched;
compute is measured and not assumed equal. No gain or peak-memory forecast
from factor counts. Repeated TRAIN-held use makes this exploratory; bootstrap
does not cover seed variation or method-selection multiplicity.

A pass supports this procedure's feasibility and permits separate updated
checkpoint serving qualification, rather than production promotion. A failure
closes this configuration without sweeps, retries, prefix extension, additional
epochs or official evaluation. The complete SOP/In-Shop quality-and-speed goal
remains active and unmet; protected Rust work remains untouched.
