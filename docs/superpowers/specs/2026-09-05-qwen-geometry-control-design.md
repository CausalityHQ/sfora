# Qwen Geometry Control Design

Date: 2026-09-05
Status: prospectively frozen development experiment; no official-test access

## Decision

The next experiment is a paired, language-free Qwen vision control. It tests whether the
largest unresolved SAGA confound—mean pooling for its baselines versus learned attention
pooling for SAGA—explains useful retrieval quality without rollout, replay, generation, or
language-model gradients. It is not an exact SAGA reproduction and neither arm is a novel
method claim.

The experiment has exactly two arms:

- `mean`: mean over Qwen vision patch tokens followed by one bias-free projection;
- `attention`: one learned query, one bias-free key projection, softmax attention over
  patches, and one bias-free output projection.

Both produce one normalized 4096-dimensional descriptor. Pooling is the only arm-specific
operation; the vision state, proxy state, example order, augmentation seeds, loss,
optimizer schedule, number of updates, evaluation code, and checkpoint boundary are
paired.

## Frozen authority

- model: `Qwen/Qwen3-VL-8B-Instruct`, revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`;
- input: RGB Cars images resized to 224 by 224 under one source-bound augmentation plan;
- optimization classes: `0..48` only;
- clean terminal development classes: `49..81`, read once after both arms and all three
  seeds are terminal;
- burned diagnostics: `82..97`, never used to select a cell;
- official Cars test classes: inaccessible;
- seeds: `17`, `29`, `43`;
- three epochs, 61 updates per epoch, exactly 183 optimizer updates per arm and seed;
- logical batch 64: 16 classes by four distinct images;
- Proxy Anchor: alpha 32, delta 0.1, 49 normalized proxies;
- AdamW: tower learning rate `2e-5`, pooler/projection `1e-4`, proxies `1e-2`, betas
  `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4` on non-bias/non-normalization
  model parameters and zero decay on proxies, bias, and normalization parameters;
- ten-update linear warm-up followed by cosine decay to zero;
- one global gradient clip at norm 1.0 per successful update.

The trainable visual tower, pooler, proxies, gradients, and AdamW moments are stored in
FP32. Only the visual forward executes under CUDA BF16 autocast; pooling, Proxy Anchor,
backward accumulation, clipping, and optimizer updates remain FP32. AdamW uses
`foreach=False`. The unused DeepStack mergers are the only frozen visual parameters and
their exact names are bound into the role manifest. This precision contract replaces the
initial BF16-weight smoke, where only 22 of 1,005 fixed parameter samples changed per
step under warm-up learning rates; those receipts remain implementation evidence but are
ineligible for the campaign.

The paired initialization manifest contains the shared vision and proxy tensors plus the
source random stream used to initialize each arm's projection. Because the poolers have
different shapes, their raw state hashes need not match; their initialization algorithm,
seed material, input width, output width, dtype, and parameter-role manifest must match.

## Execution boundary

The scientific process accepts only an authenticated local snapshot, an authenticated
local Cars manifest, a sealed protocol, checkpoint/output paths, an arm, a seed, and an
explicit execution flag. It has no Hugging Face Hub, HTTP, generation, tokenizer,
language-forward, official-test, or arbitrary dataset interface. A controller owns
process limits and publication.

The runner exposes separate `smoke`, `train`, and `aggregate` commands:

1. `smoke` performs three optimization-only updates, then restores the initial state and
   repeats them. It records input, loss, gradient, update, and terminal state digests.
2. `train` executes one arm/seed, resumes only from a fully authenticated epoch checkpoint,
   and publishes no clean metric.
3. `aggregate` authenticates all six terminal checkpoints, opens the clean development
   classes once, evaluates both arms identically, profiles inference, and emits the paired
   result.

The mean arm must never instantiate or call the attention pooler. Neither arm may load or
call language modules. Exactly the intended vision, pooling/projection, and proxy roles
receive finite gradients and optimizer updates.

## Replay and determinism

The 64-row logical Proxy Anchor loss may use microbatch replay to fit memory. A synthetic
test must prove full-batch and replay paths agree for loss, score cotangent, parameter
gradients, clipped gradients, AdamW state, and updated model/proxy state within tolerances
frozen before the DGX smoke. Scheduler advancement occurs only after a successful optimizer
step. A skipped or nonfinite update terminates the attempt.

The three-update smoke must reproduce exact input/example/augmentation digests and exact
state digests after restore. Floating evidence that cannot be byte-identical must remain
inside a predeclared mixed absolute/relative envelope; the envelope is not changed after
the first real smoke.

## Gates

Engineering admission requires both arms to complete the paired smoke with correct roles,
finite evidence, restored-state replay, no pressure stop, and projected runtime no more
than four hours per arm.

Each repaired smoke loads initialization twice and reproduces the same three updates. It
also saves after update two and requires a fresh restore to reproduce uninterrupted update
three. Full FP32 tower comparisons must show cumulative relative L2 displacement of at
least `1e-6`, at least 90% of trainable transformer blocks must individually reach `1e-6`,
and a fixed optimization image's pre-pooler visual tokens must move by more than
`max(1e-6, 10 * unchanged-repeat-discrepancy)`. Pooler or proxy movement cannot satisfy
the tower gate.

The development campaign passes only if all conditions hold:

- attention-minus-mean clean Recall@1 is at least 0.5 percentage point in every seed;
- no attention seed loses more than 0.2 point MAP@R versus its paired mean arm;
- attention training wall time is at most 1.2 times mean training wall time;
- attention inference p95 is at most 1.1 times mean inference p95 under identical hardware,
  batch sizes, dtype, warm-up, and sample count;
- all six attempts have exactly 183 successful optimizer and scheduler updates;
- all result identities, per-query outcomes, and checkpoint bytes authenticate.

The result is claim-ineligible. A pass identifies a credible efficiency hypothesis but
does not establish novelty or SOTA. Publication requires a new, prospectively untouched
confirmation surface, a capacity/pooling/budget-matched baseline, at least six paired
seeds under a predeclared power rule, class-clustered uncertainty, and complete training,
teacher, inference, and storage accounting.

## Failure actions

- Authority, replay, role, or numeric failure: repair only that execution defect and rerun
  the unchanged smoke.
- Resource projection over four hours or a pressure stop: close this GB10 configuration;
  do not silently shrink the logical batch.
- Both arms fail to improve optimization geometry: diagnose gradient coverage and stop
  before transfer interpretation.
- Optimization improves but clean retrieval does not: record transfer failure and close
  the recipe.
- Attention gain below the quality gate: retain the matched baseline and reject pooling as
  the explanation.
- Quality passes but cost fails: report the trade-off; do not claim simultaneous quality
  and performance.

## Files

- `src/sfora/qwen_geometry_control.py`: frozen protocol, pooling modules, paired evidence,
  decisions, and canonical receipts.
- `scripts/run_qwen_geometry_control.py`: local-only model/data adapter and smoke/train/
  aggregate CLI.
- `tests/test_qwen_geometry_control.py`: pure protocol, pooling, evidence, and decision
  tests.
- `tests/test_run_qwen_geometry_control.py`: replay, role, checkpoint, resume, and CLI
  tests with fake vision tokens.

## ETA

Implementation and review are estimated at one to two working days. Deployment and smoke
are one to two hours. Six serial development attempts are estimated at six to twenty-four
DGX hours, revised only from measured smoke throughput. The first decision is therefore
expected in two to four days. A passing result still needs roughly three to eight weeks
for fresh confirmation and publication work.
