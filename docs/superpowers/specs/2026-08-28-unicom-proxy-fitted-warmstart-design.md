# UniCOM proxy-fitted warm-start and runtime substrate design

**Date:** 2026-08-28  
**Status:** preregistration draft for adversarial review  
**Working name:** Frozen-Embedding Proxy Fit (FEPF)

## Decision

Do not launch the reviewed full-width recovery campaign. Its expected official
effect is about `+0.16` Recall@1 points, the recovery retains the historically
worst seed while rerunning the others, and corrected cost is roughly 41--42
GPU-hours. That is too much compute for a marginal, regression-to-the-mean
sensitive result.

The next quality candidate is a proxy-fitted classifier warm-start. It replaces
the one-step normalized class means used by the verified imprinting recipe with
a classifier fitted for exactly 512 steps on frozen pretrained embeddings under
the same masked ArcFace objective used during fine-tuning. A separate runtime
smoke tests maintained PyTorch compilation, fused AdamW, and removal of the
scientifically closed EMA shadow. The runtime smoke and quality experiment have
separate decisions so a speed change cannot be mistaken for a quality mechanism.

The operator has already authorized autonomous candidate selection, early
closure, and continuation without routine approval. Official test outcomes are
not used for selecting this candidate or any of its settings.

## Existing evidence

The strongest verified result remains class-proxy imprinting on pretrained
UniCOM ViT-L/14-336. Across five prospective paired seeds on official In-Shop:

- mean mAP@R gain: `+0.026484`, paired-t 95% CI
  `[+0.024766, +0.028202]`;
- mean Recall@1 gain: `+0.014039` (1.404 points);
- every seed is positive;
- the imprinted arm reaches the random arm's epoch-16 quality by epoch 8 or
  earlier in all five seeds, a 2--4x epoch speedup;
- matched-quality profiled compute is 30.8--73.1% lower;
- full 16-epoch compute is 1.7--2.2% higher because of one frozen feature pass;
- inference graph, descriptor storage, and deployed checkpoint are unchanged.

The resulting mean official Recall@1 is 95.15%, below the published UniCOM
96.7% anchor. The comparison is not protocol symmetric: the published anchor
uses a longer eight-GPU recipe and reports a best checkpoint over repeated test
reads, while this project uses a fixed short recipe and terminal decisions.
No SOTA claim is currently supported.

The full-width objective adds only `+0.00311279` holdout mAP@R (95% CI
`[+0.00140579, +0.00481979]`) and no measurable slowdown. It is positive but
too small to be the next campaign.

The frozen-proxy screen already supplies a stronger mechanistic signal. A
512-step AdamW fit moves classifier rows away from the class mean (mean row
cosine about `0.9475`) while improving the registered validation loss from
about `3.2930` to `3.084--3.093` and accuracy from `0.8396` to about
`0.8655--0.8660`. The previous spherical candidate closed because it required
the fit to remain near the mean. FEPF tests the opposite proposition: the
repeatable residual direction is useful as an initialization for end-to-end
fine-tuning.

Measured step time is 5.2955 seconds and the previously profiled custom-kernel
candidate covers only 0.047% of a step, against a frozen 10% eligibility gate.
No custom CUDA kernel is authorized for this design. Non-Euclidean scoring is
also closed: under unit queries the tested Lorentz form reduces to a gallery
scale and bias, and published hyperbolic In-Shop results are below the current
95.15% operating point.

## Approaches considered

1. **Recommended: proxy-fitted warm-start plus an isolated runtime smoke.** It
   follows the strongest measured mechanism, has an early falsifier, and can
   improve quality and wall-clock efficiency without changing deployment.
2. **Full-data/long-schedule scaling.** It may close the remaining absolute
   gap, but a five-seed 128-epoch program costs roughly 190 GPU-hours before
   runtime optimization and has poor information gain. It is authorized only
   after a cheaper method survives.
3. **Matryoshka or deployment-width training.** It could improve descriptor
   storage and retrieval speed, but it does not explain the current quality
   gap and is established prior art. It remains a later orthogonal efficiency
   extension, not this experiment.

## Quality mechanism

For each training seed, construct the same class-mean head used by imprinting:

`W_mean[c] = 0.01 * sqrt(768) * normalize(mean(normalize(z_i)))`

where `z_i` are frozen pretrained UniCOM embeddings from every optimization
image of class `c`, using only the identity-disjoint training partition.

Starting from `W_mean`, fit only `W` for exactly 512 optimizer steps:

- model is the unmodified pretrained UniCOM checkpoint in evaluation mode;
- inputs are all 20,650 optimization images and never the holdout query/gallery;
- cached embeddings are contiguous FP32 and their tensor hash is recorded;
- loss is the existing eight-shard masked ArcFace loss;
- batch size 128, selected width 512, margin 0.25, scale 32;
- AdamW learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, zero
  weight decay;
- batch, mask, and diagnostic streams use the existing experiment-stream seed
  derivation with the training seed as `fit_seed`;
- after each optimizer step, every row is projected back to the fixed norm
  `0.01 * sqrt(768)`;
- there is no early stopping, grid search, or official-test read.

The fitted head then initializes the unchanged end-to-end 16-epoch recipe.
The trainer must first consume the official random `normal_(std=0.01)` stream
and overwrite it, exactly as the imprinted arm does. Python, NumPy, Torch CPU,
and every CUDA RNG state after initialization must match the imprinted control.
Pre-optimization backbone tensors and embeddings must match exactly.

This is called *proxy-fitted*, not *converged*, because 512 steps are frozen by
the prior screen rather than selected by a new convergence study.

## Runtime substrate smoke

Before the quality pair, run a non-scientific 220-step seed-0 smoke with four
cells from identical initial bytes and identical data/mask streams:

1. current runtime;
2. EMA shadow disabled only;
3. `torch.compile(mode="default")` plus fused AdamW, EMA retained;
4. the composed runtime: default compile, fused AdamW, EMA disabled.

The first 20 steps are warm-up. The decision uses steps 21--220. The composed
runtime passes only if all of the following hold:

- median synchronized step-wall ratio versus current runtime is `<= 0.869565`
  (at least 1.15x faster);
- the same conclusion is reproduced by one A-B-B-A ordering;
- no non-finite loss or gradient occurs;
- mean loss over the measured window differs by no more than the registered
  within-recipe smoke tolerance;
- final raw model and classifier shapes, dtypes, and parameter inventories are
  identical;
- peak allocated and reserved memory do not exceed the current arm by 2%.

If it fails, quality experiments use the unchanged runtime. If it passes, both
quality arms use the composed runtime. Runtime selection is therefore symmetric
and happens before any FEPF quality value is observed. The smoke supports only
a throughput decision; it does not support quality neutrality or a paper claim.

## Seed-0 quality falsifier

Run a fresh matched pair at seed 0 on the chosen runtime:

- control: class-mean imprinting;
- candidate: proxy-fitted warm-start;
- identical checkpoint, data, identity holdout, batch order, masks, optimizer,
  scheduler, augmentation, objective width, evaluation width, and checkpoints;
- evaluate only the identity-disjoint training holdout at epochs 4, 8, 12, 16.

At epoch 4, immediately close FEPF if candidate minus control mAP@R is below
`+0.003`. If the early gate passes, continue both arms to epoch 16. Promotion to
multi-seed confirmation requires all of:

- epoch-16 mAP@R delta `>= +0.010` (more than three times the full-width mean);
- epoch-16 Recall@1 delta is positive;
- no more than two holdout queries lose top-1 rank for every ten that gain it;
- candidate reaches the control's epoch-16 mAP@R no later than the control;
- full candidate initialization plus training wall time to matched quality is
  no greater than the control's;
- no structural, RNG, tensor, or runtime predicate fails.

An endpoint gain between 0 and `+0.010` is recorded as positive-but-marginal and
does not consume a five-seed campaign under the present SOTA objective.

## Multi-seed confirmation

If seed 0 promotes, run five fresh paired seeds selected before any further
outcome read. Keep the entire method and runtime fixed. The primary estimand is
the per-seed difference in identity-disjoint holdout mAP@R at epoch 16.

Promotion to a terminal external test requires:

- mean paired mAP@R delta `>= +0.010`;
- all 5/5 seed deltas are positive (one-sided exact sign probability 0.03125
  under a symmetric null);
- paired Student-t 95% lower bound is positive;
- median delta `>= +0.008`;
- every leave-one-seed-out mean is `>= +0.008`;
- identity-cluster bootstrap 95% lower bound is positive for the pooled
  holdout readout; query rows are never resampled independently;
- mean initialization-plus-training time to the control endpoint is no worse;
- mean steady-state step-time and peak-memory ratios have one-sided 95% upper
  confidence bounds no greater than 1.02;
- inference latency, deployed checkpoint bytes, and descriptor bytes are exact
  ties, because the method changes training initialization only.

The t interval is reported as a sensitivity analysis rather than the only
uncertainty model. Raw seed deltas, sign evidence, median, leave-one-out means,
and identity-cluster bootstrap are all published.

## Terminal evaluation and publication boundary

Development, selection, and all mechanism decisions use only the
identity-disjoint training holdout. After confirmation, freeze one full-data
recipe and evaluate it once on each terminal benchmark. In-Shop official
results are reported with an explicit warning that this project has previously
inspected that split and therefore cannot treat it as a pristine test set. A
paper-quality generalization claim requires at least one separate benchmark
whose terminal test split was not used for candidate selection.

The published contribution, if successful, is a reproducible training recipe:
fitting masked ArcFace proxies on frozen pretrained embeddings before
end-to-end open-set retrieval fine-tuning. Weight imprinting and LP-FT are prior
art; novelty must not be claimed for those primitives. The potentially new
empirical result is the controlled increment of an exact masked-proxy fit over
class means, together with quality/time-to-quality/cost evidence.

## Automatic decisions

- Runtime smoke fails: keep current runtime and continue FEPF.
- FEPF epoch-4 delta below `+0.003`: close FEPF and return to the ranked slate.
- FEPF epoch-16 delta below `+0.010`: record marginal result and do not expand.
- Five-seed confirmation fails any quality/resource predicate: close the broad
  claim and report the exact surviving narrower claim.
- Five-seed confirmation passes: freeze the full-data recipe, run terminal
  evaluation and an external replication, then prepare the paper package.

No stage asks the operator for routine decisions. Paid execution is serial,
actively observed, and stopped immediately on structural failure or a frozen
scientific kill condition.
