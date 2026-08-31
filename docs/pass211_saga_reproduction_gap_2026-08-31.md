# Pass 211 — SAGA reproduction and GB10 feasibility gap

Date: 2026-08-31
Status: **READ-ONLY GAP MAP — no model acquisition, baseline training, or new GPU
execution authorized by this document.**

## Purpose

The current SFORA Cars control is `1,242 / 1,345` (`92.342%`) on a burned
training band. That number is useful for evidence-conditioned search but is not
comparable to an official zero-shot Cars-196 result. The current external target
is SAGA's reported three-seed Cars R@1 of `97.0 +/- 0.3`; the SFORA success bar
remains a clean three-seed mean of at least `97.4%` and a paired gain of at least
`0.5` point over a capacity-matched reproduction.

This document separates what the [SAGA primary paper](https://arxiv.org/abs/2606.15134)
and [author project page](https://shubhangb97.github.io/saga/) actually disclose
from what must be authenticated or measured before SFORA spends a high-capacity
training run. It is not a SAGA result and does not modify the pending pooled
control or RSTA Stage-A protocol.

## Disclosed primary-source contract

The paper discloses the following load-bearing choices:

- Qwen3-VL-8B supplies both the trainable vision tower and frozen language
  supervisor; all compared DML baselines use the same vision tower.
- Images are resized to `224 x 224`; no bounding-box crop is used.
- The retrieval pooler is one learned query with a linear key projection,
  softmax cross-attention over patch tokens, a linear projection to `4096`
  dimensions, and L2 normalization.
- The objective is Potential Field DML plus GRPO plus attention KL, with all
  three top-level weights equal to `1`.
- The encoder and pooler use AdamW learning rates `2e-5` and `1e-4`, cosine
  annealing for three epochs, linear warm-up over the first `5%` of steps, bf16
  mixed precision, and global gradient clipping at `1.0`.
- Each class-balanced micro-batch contains `64` images. Candidate same/different
  pairs are sampled from the batch. Each pair receives `G=8` completions at
  temperature `0.7`, top-p `0.95`, and at most `1,024` generated tokens.
- DAPO-style refill continues until `K=8` pairs with non-zero reward variance
  contribute to one optimizer step. Zero-variance pairs still receive DML
  gradients but no GRPO or KL gradient.
- GRPO uses binary verdict reward, group-normalized advantages, token-level loss
  normalization, no reference policy, and an update-time importance ratio of
  one. The KL target is head-averaged layer-26 MLLM attention, renormalized over
  each image's patches and restricted to attribute-description tokens from
  correct rollouts.
- The main evaluation uses canonical class-disjoint halves for CUB and Cars,
  an alphabetical `50/50` variant split for pooled FGVC-Aircraft images, and a
  lexicographic `743/743` species split of iNaturalist-2021 `train_mini` Aves.
- The reported means use three random seeds. The main SAGA R@1 values are
  `87.9 +/- 0.3` CUB, `97.0 +/- 0.3` Cars, `83.5 +/- 0.4` Aircraft, and
  `60.1 +/- 0.4` iNat-Aves.
- The loss ablation identifies GRPO as load-bearing: PF is `81.6/77.4` on
  CUB/Aircraft, PF+GRPO is `87.0/82.3`, PF+KL is `82.1/78.1`, and the full method
  is `87.9/83.5`.
- The authors report one H200 141-GB GPU. The paper explicitly notes that every
  contributing pair pays for eight rollout forwards and a differentiable replay
  through the frozen language backbone.

## Missing reproduction authority

Neither the v1 paper nor the public project page links an implementation
repository. The following values are therefore not reproducibly bound:

1. exact Qwen repository variant (`Instruct`, `Thinking`, or another release),
   model revision, tokenizer revision, processor revision, and trust-remote-code
   state;
2. Python, PyTorch, Transformers, attention-kernel, generation-engine, and CUDA
   versions, including whether rollout and replay share one implementation;
3. the three numeric random seeds and the complete deterministic/non-deterministic
   runtime policy;
4. AdamW betas, epsilon, weight decay, parameter grouping, scheduler endpoint,
   proxy learning rate, and whether proxies share the pooler group;
5. training augmentation beyond the stated resize, interpolation and color
   conversion, normalization, and train/evaluation processor equality;
6. exact Potential Field implementation, its hyperparameters, number of proxies
   per class, proxy initialization, and any mining or stabilization rules;
7. the class-balanced sampler, candidate-pair enumeration, same/different ratio,
   pair order, duplicate policy, refill exhaustion behavior, and the precise
   definition of an epoch when DAPO consumes variable numbers of micro-batches;
8. exact system/chat template, image-token order, tokenizer serialization,
   generation seed schedule, stop-token policy, JSON parser, relation-token
   interpretation, and behavior for malformed or truncated completions;
9. attribute-span extraction, subword-to-attribute mapping, attention head set,
   attention backend, patch-token indexing, and handling of attributes omitted
   or repeated by a completion;
10. gradient-accumulation scaling across DML, buffered pairs, successful
    rollouts, and variable generated-token counts;
11. checkpoint selection and evaluation cadence, k-means implementation and
    seeds for NMI, nearest-neighbor self-match exclusion, and numerical tie
    policy;
12. dataset file identities, image decoding authority, exact Cars canonical split
    files, and the curated iNat-Aves manifest.

These are not cosmetic omissions. Several directly change the number and order
of optimizer updates, the stochastic reward distribution, or which patch
attention is distilled. A local implementation must label substitutions as
SFORA choices rather than silently attributing them to SAGA.

## GB10 feasibility protocol after the current control terminal

The GB10 has no authenticated Qwen3-VL-8B snapshot cached. Acquisition and any
GPU preflight remain fenced until the sole three-seed pooled control is terminal,
its receipts are authenticated, the committed RSTA Stage-A gate is resolved, and
the repository assurance gate is green.

The first high-capacity action is a bounded, no-quality throughput preflight:

1. **Seal authority.** Select and digest one exact Qwen variant/revision,
   tokenizer, processor, environment lock, dataset-free synthetic image pair,
   prompt bytes, generation configuration, and binary/source identity.
2. **Inference floor.** Measure peak allocated/reserved memory and tokens/second
   for eight bf16 rollouts from one pair. Record completion lengths and prohibit
   evaluation images or labels.
3. **Replay floor.** Replay the sealed completions with the language backbone
   frozen and gradients flowing only to vision tokens. Measure peak memory,
   forward/backward wall time, and whether exact layer-26 attention is available
   without changing the attention implementation.
4. **One-step floor.** Add the declared 64-image vision/pooler DML forward and
   eight contributing pairs without taking an optimizer step. Project a complete
   step only from measured component counts; do not extrapolate from model FLOPs.
5. **Stop/go receipt.** Emit `FITS`, `MEMORY_FAIL`, `ATTENTION_UNAVAILABLE`, or
   `TIME_BUDGET_FAIL`. No result contains retrieval quality. A full run is not
   scheduled unless the projected three-epoch wall time and scratch footprint
   fit a separately frozen resource envelope.

The preflight must test the exact unfused reference path first. Custom kernels
are justified only by a measured bottleneck and must retain scalar/reference
agreement. Likely candidates are packed rollout log-prob replay, sparse
attribute-attention reduction, and fused pooler/KL normalization; none is
authorized before profiling.

## Reproduction and method sequence

If the GB10 preflight fits, the shortest defensible sequence is:

1. reproduce the same-backbone PF baseline on one development seed and verify
   data, sampler, update-count, and evaluation invariants;
2. reproduce the disclosed SAGA objective on the same seed, explicitly recording
   every SFORA substitution for missing authority;
3. run three frozen seeds only after the one-seed reproduction clears an
   outcome-blind reproducibility tolerance;
4. compare a new method against that matched SAGA implementation with paired
   seeds, not against the current SigLIP number;
5. evaluate Cars test classes once at the end, then require corroboration on CUB
   and Aircraft before any broad SOTA claim.

The first new-method candidate is not HSID as originally proposed; the primary
audit showed that its broad teacher-gradient story collides with SAGA and other
distillation work. A future candidate must change a load-bearing mechanism and
survive primary-source review before training. The most plausible remaining
direction is an amortized supervisor that preserves the *measured* SAGA gradient
ordering while reducing rollout/replay cost, but its estimator agreement must be
shown before student optimization and cannot be selected on retrieval accuracy.

## ETA and current decision

- Pooled control terminal: approximately `1-2` days from this checkpoint, based
  on the observed first-seed runtime; this is an estimate, not a deadline.
- RSTA Stage A: bounded by its registered one-DGX-hour projection after control
  authentication and launch preconditions.
- Qwen feasibility: unknown until the exact inference/replay preflight measures
  GB10 throughput and memory. The H200 disclosure is not enough to infer a GB10
  training ETA.
- Better-than-SOTA result: no honest calendar ETA exists yet. The next decisive
  ETA is the measured three-epoch projection emitted by the preflight.

Current decision: finish the sole control, execute the already committed RSTA
Stage-A boundary, then run the bounded SAGA feasibility preflight. Do not overlap
another GPU workload, download an unbound model, or treat the current `92.342%`
development control as an official SOTA comparison.
