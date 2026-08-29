# UniCOM proxy-fitted warm-start and runtime substrate design

**Date:** 2026-08-28
**Status:** post-hoc hypothesis, prospectively frozen on fresh outer splits
**Working name:** Frozen-Embedding Proxy Fit (FEPF)

## Decision and claim boundary

Do not launch the reviewed full-width recovery campaign. Its expected official
effect is about `+0.16` Recall@1 points, its recovery rule retains the
historically worst seed while rerunning the others, and corrected cost is about
41--42 GPU-hours. That is a poor allocation for a marginal result.

The next quality candidate is a proxy-fitted classifier warm-start. It replaces
one-step normalized class means with a classifier fitted for exactly 512 steps
on frozen pretrained embeddings under the same masked ArcFace objective used in
fine-tuning. A separate A-B-B-A runtime smoke tests maintained PyTorch
compilation, fused AdamW, and removal of the scientifically closed EMA shadow.
The runtime and quality decisions are separate.

FEPF is explicitly **post-hoc hypothesis generation**, not a clean continuation
of the closed spherical-probe claim. The earlier screen observed a useful fitted
head but closed because its mean cosine to the class mean was below the frozen
0.95 floor. That result prohibited relaxing the cosine threshold on the same
data. This design does not relax or reinterpret it: the old split supplies only
motivation and an exploratory kill test. Confirmatory evidence comes from five
new, frozen outer identity splits that have not been evaluated for FEPF.

The project had already inspected official In-Shop and Cars196 outcomes before
this design. Neither can be described as a pristine test set. Any result on
those benchmarks must disclose that limitation. The candidate settings and
decisions below never adapt to a new official-test result.

## Existing evidence

Class-proxy imprinting on pretrained UniCOM ViT-L/14-336 is the strongest
verified result. Across five prospective paired seeds on official In-Shop:

- mean mAP@R gain `+0.026484`, paired-t 95% CI
  `[+0.024766, +0.028202]`;
- mean Recall@1 gain `+0.014039` (1.404 points), positive in 5/5 seeds;
- 2--4x fewer epochs to the matched control endpoint;
- 30.8--73.1% less profiled compute to matched quality;
- 1.7--2.2% more full 16-epoch compute because of one frozen feature pass;
- identical inference graph, descriptor storage, and deployed architecture.

Its mean official Recall@1 is 95.15%, below the published UniCOM 96.7% anchor.
The anchor uses a longer eight-GPU recipe and a best checkpoint over repeated
test reads, so the comparison is not protocol symmetric. No SOTA claim is
currently supported.

The full-width objective adds only `+0.00311279` holdout mAP@R (95% CI
`[+0.00140579, +0.00481979]`) with no measurable slowdown. It is positive but
too small for the next campaign.

The old frozen-proxy screen fitted 14,330 rows and validated on 3,188 rows. Its
512-step fits moved outside the conservative neighborhood (mean row cosine
about `0.9475`) while improving frozen-head validation loss from about `3.2930`
to `3.084--3.093` and accuracy from `0.8396` to about `0.8655--0.8660`.
Those quantities do **not** predict an open-set retrieval effect or calibrate
the `+0.010` gate below. They only motivate testing the fitted head on fresh
outer splits. The new all-optimization-row initializer has new tensor bytes and
must not inherit the old screen's hashes or performance values.

Measured step time is 5.2955 seconds. Previously profiled custom-kernel work is
only 0.047% of a step against a frozen 10% eligibility gate, so no custom CUDA
kernel is authorized. The tested non-Euclidean scoring family is also closed:
under unit queries its Lorentz score reduces to gallery scale and bias, while
published hyperbolic In-Shop results are below the present operating point.

## Approaches considered

1. **Selected: proxy-fitted warm-start plus an isolated runtime smoke.** It
   follows the strongest measured signal and can be killed after epoch 4.
2. **Full-data/long-schedule scaling.** Potentially useful only after a cheaper
   method survives; a five-seed 128-epoch program is roughly 190 GPU-hours.
3. **Matryoshka/deployment-width training.** Orthogonal storage/search work,
   established prior art, and not an explanation for the quality gap.

## Canonical initializer

Every arm first consumes the official CPU FP32
`normal_(std=0.01)` classifier initialization. This keeps the downstream global
RNG stream balanced. Arms then behave as follows:

- `imprinted`: overwrite with canonical class means;
- `fepf_mean`: overwrite with canonical class means, then fit the head;
- `fepf_random`: retain the random head, fail if any row norm is zero/nonfinite,
  scale every row to exactly `0.01 * sqrt(768)`, then fit it.

`fepf_random` is a seed-0 descriptive mechanism control only. A single
seed/split cannot establish equivalence or a class-mean-specific mechanism, so
no positive class-mean-residual claim is authorized regardless of its outcome.
The arm may show whether a larger mechanism study is worth preregistering.

For a given outer split, preserve the optimization tuple's exact authenticated
partition-file order used by the verified imprinted control; do not sort it.
Bind the separately constructed label map and every ordered `(label, path)` row.
Encode every row once with the authenticated released UniCOM model in evaluation
mode and its deterministic evaluation transform. Build two CPU tensors in that
exact order:

- contiguous FP32 `[N, 768]` embeddings;
- contiguous int64 `[N]` labels.

Hash exact C-order bytes with SHA-256 and bind the ordered `(label, path)`
inventory. The fit device is CUDA and the uploaded tensors remain FP32. The
canonical class mean is computed from this cache only: normalize each embedding
in FP32, accumulate each class sequentially in CPU FP64, divide by the exact
count, normalize in FP64, cast once to FP32, and multiply by
`0.01 * sqrt(768)`. Imprinting and `fepf_mean` must start from byte-identical
class-mean tensors.

Fail closed before every normalization if an embedding or class mean is
nonfinite or has zero norm; no epsilon substitution is permitted. Require every
class count positive and every final class-mean row finite with the exact target
norm under `torch.allclose(row_norms, full_like(row_norms, target_norm),
rtol=2e-6, atol=2e-7)`. The same literal comparison gates prepared random-head
rows and every post-update projected head.

Fit only the classifier for exactly 512 steps:

- all optimization rows; no outer holdout query/gallery row;
- existing eight-shard masked ArcFace loss;
- batch size 128, selected width 512, margin 0.25, scale 32;
- AdamW LR `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay;
- batch root `experiment_stream_seed(training_seed, 23_001)`; start pseudo-epoch
  `e=0`, call `padded_epoch_indices(size=N, global_batch=128, epoch=e,
  seed=batch_root, shards=8)`, consume that complete returned tuple in consecutive
  nonoverlapping 128-index slices, then increment `e` by one and repeat; stop
  immediately after step 512, truncating the final pseudo-epoch after its
  producing batch;
- one continuous CUDA mask generator seeded with
  `experiment_stream_seed(training_seed, 23_002)` and advanced once per step;
- after every update, project every row to `0.01 * sqrt(768)`;
- no early stopping, grid search, or official-test read.

The diagnostic batch is the first 128 indices from
`padded_epoch_indices(size=N, global_batch=128, epoch=0,
seed=experiment_stream_seed(training_seed, 23_004), shards=8)`, preserving that
order. Its masks come from a fresh CUDA generator with the same
`experiment_stream_seed(training_seed, 23_004)` seed and exactly one
`sample_shard_masks(dimension=768, selected=512, shards=8, ...)` call. Bind its
ordered indices, feature/label bytes, masks, and initial/final loss.

Save and restore Python, NumPy, Torch CPU, and every CUDA RNG state around cache
construction and head fitting, on success or failure. Restore the model's prior
training mode. The post-initialization RNG hashes for all three arms must match.
Pre-optimization backbone tensors and the diagnostic batch above must match
exactly.

Synchronize CUDA, then start `initialization_seconds` immediately before the
official CPU random-head allocation and `normal_` draw consumed by every arm.
Stop it only after that draw, cache encoding, class-mean construction or random
row norm matching, host-to-device upload, all 512 fit/projection steps
when applicable, final diagnostic evaluation, and a final CUDA synchronization.
Thus the imprinted control is charged its complete feature/imprint construction,
and each FEPF arm is charged that same work plus its complete head fit.

An initialization-v2 receipt binds mode, outer split, training seed, source,
ordered record inventory, feature/label/start/final-head hashes, fit schedule,
initial/final diagnostic loss, fit duration, and all RNG hashes. It is emitted
for every registered seed including seed 0. The fresh initializer runs once.
Resume loads and validates the receipt and epoch checkpoint; it never re-encodes
features or reruns the 512 steps. Initialization cost is charged once from the
original receipt. The implementation replaces the historical seeds-2-through-6
and imprinted-only receipt gate: v2 receipts are mandatory for every registered
seed and all three modes.

The primitive is named *proxy-fitted*, not *converged*: 512 steps are a frozen
post-hoc free choice motivated by the hypothesis-generating screen. Because the
new all-row cache changes passes per row, no old schedule equivalence is claimed.

## Frozen fine-tuning and evaluation recipe

All In-Shop quality arms bind the exact Git blobs implementing this section and
the following literal protocol before execution:

- released UniCOM revision `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`
  and ViT-L/14-336 checkpoint SHA-256
  `3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`;
- partition SHA-256
  `cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c`;
- 16 epochs, batch size 128, four workers, backbone LR `1e-5`, classifier LR
  `1e-4`, AdamW betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay;
- OneCycleLR over exactly `len(loader) * 16` steps, `pct_start=0.1`, with max
  LRs `(1e-5, 1e-4)`; no per-epoch step cap;
- margin `0.25`, scale `32`, `official-eight-mask`, selected width 512,
  evaluation width 512, FP32 training (`bf16=false`), evaluations and
  checkpoints at epochs `(4, 8, 12, 16)`;
- `build_train_transform(336)` with color jitter `0.4`,
  `rand-m9-mstd0.5-inc1`, bicubic interpolation, random-erasing probability
  `0.25`, pixel mode/count 1, and the released UniCOM mean/std;
- the deterministic evaluation transform returned by the authenticated official
  model loader; exact query/gallery partition order;
- raw-backbone evaluation: encode FP32, L2-normalize all 768 coordinates, then
  take coordinates `0..511` with no second normalization, rank by ascending
  squared Euclidean distance (`normalize_before=true`), and compute the repository's exact
  Recall@K and per-query AP@R/mAP@R definitions. Classifier and EMA tensors are
  never used for retrieval.

Every evaluation emits a bounded receipt plus a separately published canonical
ranked-prefix JSON artifact in exact query-partition order containing query
path/label, relevant-gallery count, AP@R, and the exact ranked prefix through
`min(max(30, relevant_gallery_count), gallery_rows)` with gallery indices,
paths, labels, scores, and correctness flags. It also binds hashes of the
normalized query plus the complete ordered gallery scores/indices. The strict
reload authenticates the ranked artifact from the receipt, recomputes per-query
Recall@1/10/20/30 and AP@R from the stored prefix,
then recomputes aggregate Recall@K and mAP@R. Gain/loss and sensitivity
predicates consume these bound records rather than aggregate-only history rows.
The ranked-prefix publication budget is derived from the complete worst-case
canonical per-query envelope—including query path/label, relevant count, AP,
both hashes, JSON framing/indentation, and every bounded ranked entry—not only
the repeated gallery rows. Evaluators strict-load this separate path/hash/byte
authority; they do not expect the large rows inline in the bounded receipt.

The runtime smoke may change only compile mode, fused-AdamW selection, and
whether the unused EMA shadow/hook exists. Every other literal above is fixed.
For inference equality, compare the raw backbone state-dict parameter and buffer
inventory in sorted key order: exact names, shapes, dtypes, and
`sum(numel * element_size)` bytes. Classifier, optimizer, scheduler, scaler, and
EMA are excluded because none is exported. The bound inference forward function,
full-vector normalization, 512-coordinate slice, squared-Euclidean ranking, and
retrieval operation inventory must be identical between arms; no
serializer-dependent file-size predicate is used.

## Runtime substrate smoke

Use one already-authenticated seed-2 imprinted epoch-16 checkpoint as the sole
starting state. It binds raw model, classifier, AdamW state, scheduler, scaler,
EMA state, data epoch, batch stream, and mask stream. Extend the existing step
profiler with explicit runtime overrides; do not use trainer `--max-steps`.

Run eight **fresh processes** in exact A-B-B-A-A-B-B-A order, treated as two
independent cycles:

- A: current runtime (`compile=false`, `fused=false`, EMA hook active);
- B: composed runtime (`torch.compile(mode="reduce-overhead")`, fused AdamW, no EMA
  allocation or hook).

Each process reloads the same checkpoint bytes, performs the existing 20 warm-up
steps, 50 unprofiled measured steps, and 10 profiler steps. Reset CUDA peak
statistics after warm-up and before measurement; do not call `empty_cache`.
Record every measured loss, GradScaler step/skip decision, synchronized wall and
CUDA-event times, peak allocated/reserved bytes, and parameter inventories.
After unscaling and before the optimizer step, every gradient must be finite;
zero scaler skips are allowed.

Pool the four A measured sequences (200 observations) and the four B measured
sequences (200 observations) in execution order. The point ratio is
`median(B step_wall) / median(A step_wall)`. Consecutive steps are descriptive
observations, not independent replicates. Pair each B process with its nearest A
process inside the same A-B-B-A cycle. The composed runtime passes only if:

- point ratio `<= 0.8695652173913043` (at least 1.15x faster);
- all four paired process-median B/A ratios are `<= 0.90`;
- for each nearest A/B process pair, step `k` is aligned with step `k`, and
  `||loss_B-loss_A||_2 / max(||loss_A||_2, 1e-12) <= 2e-4` while the maximum
  absolute aligned difference is `<= 2e-3`;
- no non-finite value or scaler skip occurs;
- final raw model/classifier shapes, dtypes, names, and optimizer state schema
  match control;
- `max(B peak allocated)/max(A peak allocated) <= 1.02` and likewise for
  reserved bytes.

The smoke costs 560 optimizer steps plus 80 objective-only profiler calls and
compilation overhead, roughly 1 GPU-hour at the measured rate. If it fails,
every quality arm uses the current runtime. If it passes, every quality arm uses
the composed runtime.
The selection happens before any new FEPF retrieval value is observed. The
smoke supports a runtime choice only, not endpoint quality neutrality.

## Quality-arm profiling

For the exploratory pair and each fresh confirmation pair, after both epoch-16
checkpoints exist, profile four fresh processes ordered
control-candidate-candidate-control. Every process reloads its bound epoch-16
checkpoint and runs 20 warm-up plus 50 measured optimizer steps under the
already-selected runtime, with no objective-only profiler phase. Reset CUDA peak
statistics after warm-up, never call `empty_cache`, and record synchronized
wall/CUDA times and peak allocated/reserved bytes. Every process must have 50
finite losses, finite unscaled gradients, zero scaler skips, the expected step
count, and the registered final parameter/optimizer schemas. For each arm,
`profiled_step_wall` is the median of its pooled 100 step-wall observations;
allocated and reserved peaks are each the maximum across its two processes.
These arm-specific profiles, not the seed-2 runtime smoke, supply every quality
resource and matched-quality computation.

## Exploratory seed-0 gate

The old outer split (`holdout_fraction=0.2`, `holdout_seed=0`) is explicitly
exploratory. Run fresh matched `imprinted` and `fepf_mean` arms with training
seed 0 on the selected runtime. Evaluate only the identity-disjoint training
holdout at epochs 4, 8, 12, and 16.

Add `--stop-after-epoch` as a controller boundary distinct from `--epochs`.
Both first-stage commands use `--epochs 16 --stop-after-epoch 4`, so OneCycleLR
is constructed for the full 16-epoch schedule but execution stops after the
epoch-4 checkpoint. Each stage writes to a fresh, absent attempt directory. A
continuation command uses `--epochs 16 --resume <bound-epoch-4-checkpoint>
--stop-after-epoch 16` and a new absent continuation directory. Run-receipt v2
explicitly permits this resume form, binds the parent checkpoint and original
initialization receipt, restores all mutable state/history, and bypasses every
initializer/cache code path. It never reuses or overwrites the first-stage
output or receipt.

Here “absent” is the transfer/first-launch contract for a stage destination,
not a condition that may be reapplied while validating an authenticated
campaign resume. The campaign root is created by whichever registered
controller/canary entrypoint runs first; subsequent entrypoints validate and
reuse its immutable budget and terminal evidence.
Canary acceptance independently recomputes deterministic fitted science once;
wall-clock fit/initialization durations and peak-memory values are validated as
finite nonnegative observations but are excluded from rerun equality. The
terminal is published only after that validation. Controller restarts memoize
the terminal/manifest digest during one invocation, and every post-canary child
receives the canonical cuBLAS workspace before its first Torch import.

The four-argument campaign builder executes in a clean DGX checkout of the
reviewed source parent, where the registered historical/checkpoint/partition
preimages exist. Its canonical config is committed as the sole child delta,
then consumed from a separate detached execution checkout.

After both epoch-4 checkpoints and evaluations exist, pause the controller.
Close FEPF if candidate-minus-control mAP@R is below `+0.003`. If it passes,
resume both existing checkpoints without recomputing initialization. Promotion
to fresh-split confirmation requires all of:

- epoch-16 mAP@R delta `>= +0.010` (over three times the full-width mean);
- epoch-16 Recall@1 delta positive;
- among queries whose top-1 correctness differs, `losses <= floor(gains/5)`;
- candidate first reaches the control's epoch-16 mAP@R at an evaluated epoch no
  later than the control;
- profiled compute to that first attainment is at most `1.02` times control;
- every structural/RNG/tensor/runtime predicate passes.

First attainment is the earliest of epochs `(4, 8, 12, 16)` at or above the
control epoch-16 target; there is no interpolation. Failure to attain is right
censored and cannot pass. Profiled compute is
`initialization_seconds + epoch * optimizer_steps_per_epoch * profiled_step_wall`.
Raw run wall time is disclosed but never gates a claim.

If `fepf_mean` passes, run `fepf_random` under the same seed/split/runtime to
epoch 16. This arm cannot replace the candidate or change any threshold. It
is descriptive evidence for deciding whether a separate mechanism study is
worth preregistering; it cannot authorize a class-mean-specific claim here.

An endpoint gain in `(0, +0.010)` is positive-but-marginal and does not consume
the confirmation campaign.

## Fresh-split confirmation

Freeze these five paired `(training_seed, holdout_seed)` draws now:

1. `(7, 20_260_828)`
2. `(8, 271_828)`
3. `(9, 314_159)`
4. `(10, 1_618_033)`
5. `(11, 57_721)`

Each draw creates a fresh `identity_holdout(fraction=0.2)` from official train
identities. Within a draw, `imprinted` and `fepf_mean` use identical identities,
bytes, batch/mask streams, optimizer, scheduler, augmentation, runtime, and
checkpoint/evaluation schedule. No outcome from one draw changes another.

The primary estimand is the distribution of paired epoch-16 mAP@R differences
under this joint draw of training randomness and outer identity split. It does
not separately estimate arbitrary-dataset generalization. Confirmation needs:

- mean paired mAP@R delta `>= +0.010`;
- mean paired Recall@1 delta `>= +0.005` and all 5/5 Recall@1 deltas positive;
- all 5/5 deltas positive (one-sided exact sign probability 0.03125);
- one-sided paired Student-t 95% lower bound
  `mean(delta) - 2.131846786326649 * sample_std(delta, ddof=1) / sqrt(5) > 0`;
- median delta `>= +0.008`;
- every leave-one-pair-out mean `>= +0.008`;
- candidate profiled compute to the paired control endpoint is at most `1.02`
  times control in 5/5;
- mean log step-time, peak-allocated, and peak-reserved candidate/control ratios
  each have one-sided 95% upper bounds no greater than `log(1.02)`, using
  `mean(log_ratio) + 2.131846786326649 * sample_std(log_ratio, ddof=1) / sqrt(5)`;
- bound inference operation inventory, canonical raw-backbone state-dict
  inventory/byte count, and descriptor bytes are exact ties.

As supportive query sensitivity, use each held-out query's paired AP@R
difference against the fixed original gallery. Because `identity_holdout`
selects exactly one query per held-out identity, do not claim an additional
within-identity level. For 10,000 PCG64 replicates (seed `20_260_829`), resample
the five paired draws, then resample paired query differences within each
selected draw, and average. Report the 2.5th/97.5th percentiles using
`numpy.quantile(..., method="linear")`. Shared-gallery dependence remains, so
this is not a promotion predicate; raw pair deltas, sign evidence, median,
leave-one-out, and t sensitivity remain visible.

Expected maximum cost is about 14 GPU-hours for the exploratory three-arm panel
plus its initializer/evaluation/profiles, and about 42 GPU-hours for five
confirmation pairs including initializers, evaluation, and profiles. The
runtime smoke adds about 1 GPU-hour.
The confirmation spend occurs only after a seed-0 effect at least `+0.010`.
A passing 1.15x runtime substrate reduces but does not erase this cost.

## External replication boundary

This design does not authorize a Cars196 run. The repository has no frozen,
authenticated Cars196 adapter and end-to-end recipe equivalent to the In-Shop
contract above, so inventing one after seeing confirmation would create an
under-specified claim-bearing stage. If fresh-split confirmation passes, write
and independently approve a separate Cars196 preregistration that freezes its
dataset bytes, adapter, transforms, optimizer/scheduler, evaluator, five paired
seeds, effect-size and uncertainty gates before reading a new Cars result. The
project's previous Cars196 inspection must still be disclosed.

Only after such a replication may one full-data In-Shop recipe be frozen and
read once for a descriptive comparison to 96.7. The paper must disclose repeated
historical access to In-Shop and the anchor's best-checkpoint selection.

## Novelty boundary

Weight imprinting, LP-FT, and normalized classifiers are prior art. FEPF must not
claim those primitives as novel. A successful practical contribution is a
reproducible masked-ArcFace frozen-proxy fitting recipe for open-set retrieval
with joint quality/cost evidence. The seed-0 random-fit arm is descriptive; this
preregistration never authorizes a class-mean-residual mechanism claim.

## Automatic decisions

- Runtime smoke fails: retain current runtime and continue.
- Exploratory epoch-4 delta below `+0.003`: close FEPF immediately.
- Exploratory endpoint delta below `+0.010`: record marginal/negative and stop.
- Exploratory endpoint delta at least `+0.010` but any Recall@1, query
  gain/loss, first-attainment, `1.02` compute, or structural predicate fails:
  close as `CLOSE_NONPARETO` and report the exact quality-only result.
- Fresh-split confirmation fails any primary predicate: close the broad claim
  and report the exact surviving narrower result.
- Confirmation passes: freeze and independently approve the separate Cars196
  replication protocol before any Cars run, then proceed toward the publication
  package under that new gate.

Paid execution is serial, actively observed, and stopped on structural failure
or a frozen scientific kill. No stage asks the operator for routine decisions.
