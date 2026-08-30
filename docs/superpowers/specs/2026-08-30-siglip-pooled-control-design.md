# SigLIP-so400m pooled Proxy Anchor control design

## Purpose

SFORA cannot fund another representation method until it has a same-backbone
trainable pooled control. The control is not a method claim. It measures whether
fine-tuning the pinned SigLIP-so400m vision representation with ordinary Proxy
Anchor transfers across unseen Cars classes, and it supplies the paired baseline
for the next method.

## Evidence roles

- Optimization: Cars train classes `0..48` only.
- Clean development comparison: classes `49..81`, once before optimization and
  once after the final epoch only; never at an intermediate epoch.
- Hypothesis-generating diagnostics: already-burned classes `82..97` only.
- Official test classes `98..195`: never loaded by this subsystem.

The pinned dataset revision is
`9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40`. The pinned model is
`google/siglip-so400m-patch14-384` at revision
`9fdffc58afc957d1a03a25b10dba0329ab15c2a3`. Seeds are `17`, `29`, and `43`.

## Representation and objective

Load only the SigLIP vision tower. Its attention-pooler output is mapped by a
bias-free `1152 -> 512` projection and unit-normalized. There is one normalized
512-dimensional proxy per optimization class. Class scores are cosine
similarities and the objective is Proxy Anchor with `alpha=32`, `delta=0.1`.
Deployment remains one 512-dimensional pooled descriptor, one image view, and
cosine retrieval with no reranking.

The optimizer is AdamW with three learning-rate families: vision tower `1e-5`,
pooled projection `1e-4`, proxies `1e-2`; weight decay is `1e-4` except for
proxies and normalization/bias parameters. The tower rate is deliberately 10x
below SFORA's ResNet-50 Proxy Anchor recipe because this control fine-tunes a
428M-parameter pretrained transformer; it is frozen prospectively and is not
called an official Proxy Anchor hyperparameter. A parameter-group constructor
must prove that every trainable parameter occurs exactly once and that every
decay exclusion is name- and type-justified.

The fixed schedule is 60 total epochs with warmup included. For `N` optimization
examples, the resolved steps per epoch are exactly `S=floor(N/120)`; the
floor defines the conventional epoch compute budget rather than a partition of
examples into one exhaustive pass. Learning rates rise linearly from `1/(5*S)` to `1` times
their group base rates over the first `5*S` optimizer steps, where `S` is the
resolved steps per epoch. Decays by `0.5`
occur at the starts of epochs 10, 20, 30, 40, and 50; warmup does not shift the
milestones. These values are frozen before the smoke and are not selected on
validation results.

Training augmentation is one torchvision
`RandomResizedCrop(384, scale=(0.16,1.0), interpolation=BICUBIC)` followed by
`RandomHorizontalFlip(0.5)`, tensor conversion, and normalization with the
authenticated SigLIP mean/std `(0.5,0.5,0.5)`. Evaluation uses the pinned
processor's deterministic 384x384 resize and the same normalization.

## Exact logical-batch recomputation

Proxy Anchor is not separable across ordinary microbatches: its classwise
`logsumexp` depends on all examples in the logical batch. Gradient accumulation
over independent microbatch losses is therefore forbidden.

For each deterministic logical batch of 120 images:

1. Materialize the augmented `120 x 3 x 384 x 384` tensor once. Run fixed
   microbatches under `no_grad`, collecting the complete `120 x 49` fp32 score
   matrix without retaining transformer activations.
2. Treat the score matrix as a leaf, evaluate the exact Proxy Anchor loss, and
   obtain `dL/dscores` with autograd.
3. Replay the identical augmented image tensors in the identical microbatch
   order with gradients enabled.
4. Recompute each score slice in fp32 and require maximum absolute disagreement
   from pass 1 at most `2e-5`. Backpropagate it using the frozen `dL/dscores`
   slice. Accumulated parameter gradients are exactly the chain rule for the full
   logical batch within this registered replay tolerance.
5. Clip the global gradient norm at 10 and take one optimizer step.

Input tensors are materialized once per logical batch, so augmentation is not
resampled between passes. Both passes invoke the same forward function with the
same fixed eager-attention implementation and deterministic CUDA settings;
gradient checkpointing must use `use_reentrant=False`. The `2e-5` threshold is
a determinism guard between two evaluations of the same bf16 arithmetic, not a
bf16-versus-fp32 approximation allowance. If the registered real-model smoke
cannot satisfy it, the campaign stops without relaxing the threshold or
switching precision. The vision tower has no active stochastic dropout; the
implementation refuses training batch normalization, `nn.Dropout(p>0)`, and
nonzero float attributes named `dropout`, `attention_dropout`, or `drop_path`
anywhere in the replayed module. Every trainable tower parameter must receive a
finite non-`None` gradient and the aggregate tower gradient norm must be
positive. A scalar one-pass oracle mutation-locks loss, all parameter gradients,
and one optimizer update against the replay operator. The DGX smoke additionally
checks the real bf16-autocast, nonreentrant-checkpointed path against a full fp32
micro-fixture.

## GB10 smoke

Before scientific training, run exactly three optimizer steps on optimization
classes only. GB10 names the Grace Blackwell processor, not a 10 GiB limit. The
smoke chooses only the first passing divisor of 120 in the fixed descending
ladder `120,60,40,30,24,20,15,12,10,8,6,5,4,3,2,1`. A rung passes only when
the conservative sum of process RSS and CUDA peak reserved memory stays below
96 GiB, memory PSI and swap do not grow, and measured throughput projects the
resolved `60*S` steps to no more than 24 hours per seed. It may not inspect
validation or burned examples. Each rung starts
from a newly loaded model/optimizer, resets CUDA peak statistics, and destroys
the model, gradients, optimizer state, and allocator cache before the next rung.
It records peak allocated/reserved CUDA memory, process RSS, examples/second,
logical and microbatch sizes, finite loss, complete tower-gradient coverage, the
chosen microbatch, and projected per-seed wall time. Failure at microbatch one
or failure of every rung stops the campaign. The smoke
publishes a receipt, destroys its mutated model, and scientific seeds reload the
pinned pretrained bytes independently.

Each rung executes in a child process. A child terminated by `SIGKILL` publishes
no child receipt; the parent records that rung as `process-sigkill`, treats it as
nonpassing, and continues to the next registered smaller microbatch. Any other
nonzero child exit is an authority or implementation failure and stops the smoke.

## Training and evaluation

Each seed starts from identical pinned pretrained weights and seed-specific
projection/proxy initialization. The partition authority imports
`F1_TRAIN_CLASSES` and `F1_VALIDATION_CLASSES` from `sfora.token_set_screen` and
`SUBSTRATE_F0_CLASSES` from `sfora.substrate_screen`; it does not duplicate
those band literals. Every optimization class must contain at least four
examples and every clean/burned class at least two.

Training uses exactly `S=floor(N/120)` logical batches per epoch. For each
`(seed, epoch, step)`, a stateless SHA-256-derived CPU generator independently
permutes the sorted 49 optimization class IDs; the first 30 are the batch's
distinct classes in permutation order, so classes may recur across steps in an
epoch but never within one step. Each selected class contributes four examples.
Example selection uses a persistent per-class stream: concatenate successive
permutations of that class's sorted example IDs, with permutation-cycle seeds
derived from `(seed, class ID, cycle)`. Consume four entries when at least four
remain; otherwise discard that permutation's one-to-three-entry tail and consume
the first four of the next cycle. Thus one image never appears twice in a
logical batch and there is no replacement within a permutation. Cursors continue
across epochs and are checkpointed. The receipt mutation-locks every class and
example position. The floor-defined `S` is only an epoch compute budget; there is
no dataset partition remainder, 49-class tail, or refill rule.

Before training, embed optimization, clean-validation, and burned-diagnostic
bands through both (a) the normalized raw 1152-dimensional pinned pooler output
and (b) the seed-specific initialized 512-dimensional projection. The projected
initial representation is the sole initial reference for projected-to-final
margin and Recall@1 changes; the raw representation is reported separately as
the same-backbone frozen reference. Initial and final passes both use `eval()`,
`inference_mode`, identical deterministic preprocessing, and fp32 output
normalization/scoring. After epoch 60 only, embed the three projected bands
again. Report exact leave-one-out cosine Recall@1 separately within `49..81` and
within `82..97`; neither gallery contains the other band. Call
`sfora.substrate_screen.score_frozen_substrate` directly as the sole Recall@1
metric authority. No parallel Recall@1 implementation, intermediate validation
embedding, or checkpoint selection is permitted.

For every query, compute nearest-positive and nearest-negative cosine and their
margin. Per seed and band, report the initial/final means and changes for all
three quantities. The explicitly named memorization-to-transfer ratio is
`burned_margin_change / train_margin_change` when `train_margin_change > 0`;
otherwise it is undefined and the receipt cannot support a transfer-mechanism
conclusion. A finite per-seed ratio remains descriptive and never sets a
mechanism-conclusion support flag; no unregistered numerical threshold is
inferred from a small positive denominator. Clean validation also reports
initial-to-final Recall@1 change so an
accidentally degraded baseline cannot become an artificially easy denominator.

## Artifact contract

The canonical newline-terminated JSON receipt binds source revision/tree digest,
dataset/model identities, exact ordered example manifest, initial model digest,
config, seed, smoke receipt, initial/final validation Recall@1, full margin
summaries, memorization-to-transfer ratio, final objective, checkpoint digest,
peak memory, resolved optimizer-step count, wall time, examples/second, torch and
transformers versions, CUDA runtime, device identity, evaluation batch size, and
query block. It has
`claim_eligible=false`. The aggregate receipt requires exactly three seeds and
requires byte-identical shared source, dataset, model, config, smoke, training,
and environment authority across them before reporting their mean
clean-validation Recall@1/change and the three descriptive per-seed transfer
ratios. It never averages those potentially unstable ratios, emits the aggregate
mean as null, and keeps mechanism-conclusion support false. It carries the
shared source revision/tree, manifest, config, and smoke digests. It makes no
`97.4` claim.

Each completed epoch first publishes a create-new checkpoint containing model,
optimizer, scheduler, sampler, and CPU/CUDA RNG state plus an authority digest.
After the new checkpoint is fsynced, authenticated, and hard-linked under its
epoch name, the immediately preceding non-final checkpoint is deleted. Thus at
most the newest complete checkpoint and its in-progress successor coexist; only
the epoch-60 checkpoint is retained after successful completion. Before every
write, a free-space preflight requires room for the current checkpoint, one
successor of the registered maximum size, and 20% headroom. An interrupted seed
may resume only from its latest authenticated complete-epoch checkpoint under
identical source/config/environment authority. Final checkpoint and receipt
publication use partial-file plus hard-link publication. Authority,
nonfinite, split, CUDA OOM, or determinism failures publish no scientific receipt
and do not authorize an adaptive configuration change.
