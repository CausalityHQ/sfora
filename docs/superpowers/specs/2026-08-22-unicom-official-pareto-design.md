# UniCOM Official-Readout Gate and Pareto Escalation

**Date:** 2026-08-22  
**Status:** prospective; no retained checkpoint has been evaluated on the official
In-Shop query/gallery split.

## Objective and claim boundary

Measure whether class-proxy imprinting's internally replicated convergence and mAP@R
advantage transfers to the untouched official DeepFashion In-Shop test identities.
The first experiment reuses the already frozen paired checkpoints. It is an official
**evaluation-split** readout, not an official training-protocol reproduction: every
checkpoint was trained for 16 epochs on the same fixed 80% subset of the official
training identities.

The intervention is established prior art (weight imprinting/class-mean classifier
initialization). This program can support a paired engineering measurement under
UniCOM's masked-shard margin-softmax training, not a novelty claim. The published
UniCOM 96.7% Recall@1 result remains a contextual anchor only; it cannot be subtracted
from or compared as a matched result.

Primary sources:

- DeepFashion protocol: <https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf>
- UniCOM: <https://openreview.net/pdf?id=3YFDsSRSxB->
- Imprinted Weights: <https://arxiv.org/abs/1712.07136>
- A Good Start Matters: <https://arxiv.org/abs/2503.06385>
- VPTSP-G: <https://arxiv.org/abs/2402.02340>
- Proxy Anchor: <https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf>

## Evidence entering this gate

The selection run used seed 0. Training seeds 1--6 were then frozen before execution,
but the official gate uses the five homogeneous, fully authenticated pairs produced by
the final trainer, seeds 2--6. Their internal mAP@R deltas are +0.019491, +0.017514,
+0.016551, +0.021711, and +0.015771: mean +0.018207, sample SD +0.002397, paired
Student-t (df=4) 95% CI [+0.015231,+0.021184], and two-sided sign p=0.0625. These
statistics cover training randomness conditional on one fixed internal holdout; they
do not cover holdout-sampling variance.

Seed 1 was produced by an earlier trainer and lacks the later initialization/RNG
receipt. Its already-known internal delta (+0.014433) is the weakest of the six, so
excluding it after observation moves the estimate favorably. To disclose and bound
that choice, Gate 1 also evaluates seed 1's eight checkpoints in the same process as a
prespecified non-gating sensitivity row. It never enters the five-seed decisions. No
replacement seed is allowed.

The retained checkpoints are the exact `random_raw` and `imprinted_raw` arms at epochs
4, 8, 12, and 16. The official query/gallery identities were unreachable during
training and during intervention selection.

## Gate 1: one-shot official evaluation of retained checkpoints

### Inputs and order

- Dataset: exact official In-Shop query (14,218 images) and gallery (12,612 images),
  3,985 test identities.
- Gating seeds: exactly 2, 3, 4, 5, 6. Sensitivity-only seed: exactly 1.
- Arms: `random_raw`, then `imprinted_raw`.
- Epochs: 4, 8, 12, 16.
- Evaluation order: gating seeds 2--6 followed by sensitivity seed 1; within seed,
  `random_raw` then `imprinted_raw`; within arm, epochs 4, 8, 12, 16. This is 48 rows.
- Model state: the exact raw `checkpoint["model"]` state used by the completed
  `random_raw`/`imprinted_raw` comparison; classifier and EMA-shadow state are excluded.
- BatchNorm: assert that the authenticated ViT contains no BatchNorm/SyncBatchNorm
  modules; do not introduce a train-data recalibration pass into this readout.
- Primary geometry: use all 768 dimensions, normalize each complete descriptor after
  extraction, then exact Euclidean query-to-gallery ranking. This is the `full_unit`
  view that reproduces the released UniCOM score in
  `src/sfora/unicom_retrieval_audit.py`; the legacy normalize-full-then-prefix-512 view
  is not the published-score reproducing view.
- Secondary geometry: the legacy normalize-full-then-prefix-512 Euclidean view. A
  disagreement in the sign of the five-seed epoch-16 mean mAP@R delta blocks the
  quality decision. Persist each row's mean squared prefix energy to expose the known
  truncation confound.
- Metrics in both views: query-weighted mAP@R and Recall@1/10/20/30/40/50. Persist
  per-query AP@R and top-1 correctness; higher-k correctness is aggregate-only.
- Like-for-like secondary: identity-uniform mAP@R using the lexicographically first
  official query path for each of the 3,985 test identities. The selection rule is
  deterministic and has no random seed. Internal mAP@R used one query per held-out
  identity, while official primary mAP@R weights all 14,218 queries.
- Evaluation transform: the exact deterministic 336-pixel evaluation transform
  returned by the authenticated `train_unicom_inshop._load_official_model`; persist its
  complete `repr` and canonical-text SHA-256. Labels remain the exact nonempty strings
  from `parse_inshop_partition` in file order; no train-label map or numeric recoding is
  permitted. Retrieval metrics are computed by the audited
  `sfora.unicom_retrieval_audit` implementation extended only for registered k=40/50.

No metric, partial aggregate, or arm comparison may be printed or published before all
48 checkpoint rows have been scored and validated. One process writes one atomic,
no-clobber, strict-reloaded result. The result binds checkpoint bytes, dataset manifest,
source revision, environment, order, every per-query evidence array, raw timings, and
peak allocation. It must be published regardless of outcome.

Run inference in FP32 with autocast disabled, TF32 disabled, deterministic algorithms
enabled, cuDNN benchmark disabled/deterministic enabled, fixed nonshuffled file order,
and fixed batch size/workers. Both arms of each seed execute in the same process under
the same settings. Persist a byte hash of every complete embedding matrix.

### Frozen decisions

`official_transfer_quality_supported` is true only if all are true:

- the two-sided paired Student-t 95% lower bound for the mean epoch-16 primary mAP@R
  delta exceeds the smallest effect of interest +0.002 (0.2 mAP points); with n=5,
  df=4, and t=2.7764451052 this requires `mean > 0.002 + 1.241663*sample_sd`;
- at least four of five epoch-16 mAP@R deltas are positive;
- both the seed-level paired-t 95% lower bound and a 10,000-replicate paired-query
  bootstrap lower bound for Recall@1 exceed the practical noninferiority margin -0.001;
- no seed's epoch-16 Recall@1 delta is below -0.003; and
- the primary and legacy-view five-seed mean mAP@R deltas have the same sign.

The +0.002 threshold is a prospectively chosen practical floor, about one ninth of the
authenticated internal point estimate; it is not inferred from published mAP@R because
no comparable published official mAP@R exists. A mean primary delta above +0.0305 (the
approximate upper 99% prediction bound from seeds 2--6) is a non-blocking anomaly flag:
publish it, then re-audit geometry, labels, and hashes before making a claim.

For each seed/arm, first attainment is the first registered epoch whose primary mAP@R
reaches that same arm's epoch-16 value. `official_transfer_trajectory_supported` is
true only if the imprinted first-attainment epoch is no later than the paired random
first-attainment epoch in all five seeds and is at most epoch 8 in at least three. A
missing attainment fails that seed; a tie has speedup 1.0. Report the exact ratio
`random_first_attainment / imprinted_first_attainment`, compare grid-native epochs
without interpolation, and never retrofit epochs 2/6 into the retained run.

`retained_checkpoint_gate_passed` requires both transfer decisions. It never means
SOTA, official-training reproduction, or a novel method.

## Gate 2: fairness controls before a full-training claim

Gate 2 runs only if Gate 1 passes. It uses training-internal selection only and never
reopens the official result while choosing settings.

1. Head-LR factorial at 16 epochs on a fresh prospectively frozen seed:
   `{random, imprinted} x {1e-3, 3e-4, 1e-4, 3e-5, 1e-5}`, plus a random-head arm
   with a frozen-backbone head warmup before joint training.
2. Shuffled-imprint nuisance control on five prospectively frozen paired seeds. It preserves
   proxy norms, spectrum, and manifold placement while breaking class assignment.
3. A 32-epoch single-seed budget probe to falsify a short-schedule-only effect.

For a publishable matched-quality claim, imprinting must beat the best random head-LR/
warmup configuration; the paired Student-t 95% upper bound of the shuffled-imprint gain
must be below half the Gate-1 correctly assigned mean gain; and the 32-epoch delta must
be nonnegative. Shuffled proxies use the same norm-match scalar and consume the same RNG
stream. Failure of head-LR fairness closes the quality method. Failure of the shuffled
control reclassifies the mechanism as generic proxy conditioning. Failure at 32 epochs
limits any claim to short schedules.

## Gate 3: full official-training escalation

Only Gates 1 and 2 can authorize this expensive phase. Train on all 25,882 official
training images / 3,997 identities with `holdout_fraction=0.0`, no query/gallery
access, and prospectively frozen paired seeds. Random and imprinted arms differ only in
the initial proxy tensor and restore identical Python, NumPy, Torch CPU/CUDA, sampler,
worker, and feature-mask RNG states before optimizer step 1.

Mixed precision, compile, and fused-optimizer settings must be benchmarked and frozen
before the program starts; they may not change mid-program. A custom kernel remains
ineligible because profiling found only about 0.046% fusible non-backbone time.

The full-training result is evaluated once after every run completes. Even a result at
or above 96.7% Recall@1 is described as a single-device recipe port unless the exact
published 128-epoch, multi-GPU pipeline is reproduced. No global SOTA wording is
allowed from this chain alone.

## Continuation after Gate 1

- Quality and trajectory pass: run Gate 2, then the full-training escalation if all
  fairness controls pass.
- Quality fails but trajectory passes: retain only an iso-quality convergence claim;
  test SOP, where the classes-per-image mechanism predicts a larger effect.
- Both fail: close static class-mean initialization for this setting. Do not tune on
  the official outcomes.
- After a surviving In-Shop chain, run prospectively frozen SOP and Cars196 tests to
  test whether effect size follows classes per image. A monotone SOP > In-Shop > Cars
  pattern is the mechanism claim; an inverted pattern falsifies it.

Zero-cost mechanism evidence from the retained checkpoints should also measure
backbone parameter drift and CKA versus the released initialization. Equal drift
withdraws the LP-FT distortion explanation.

## Monitoring, reproducibility, and storage

Freeze source, tests, input inventory, gates, and output schema in Git before opening
the official split. Run from a clean detached DGX checkout with exact hashes. The
retained footprint is 58,283,916,296 bytes per arm/seed: 542.8 GiB for gating seeds
2--6 and 651.4 GiB including sensitivity seed 1. The process streams about 651 GiB of
checkpoint reads. Per-query evidence is limited to FP64 AP@R and one top-1 bit per
query/view/row; higher-k values are aggregates. The canonical result must be at most
64 MiB, and preflight requires at least 1 GiB free plus process/runtime reserve.
Expected runtime is 4--6 GPU-hours; the hard timeout is 12 hours. No retained
checkpoint is deleted until the result is validated, copied off-host, and the
continuation branch is fixed. No new training may start until capacity is resolved.

Progress logs have the exact metric-free fields `{row_index,seed,arm,epoch,
checkpoint_sha256,elapsed_seconds,peak_gpu_mib}` and no other floats or metric names.
All 48 rows remain in process memory and one exclusive temporary is written only after
complete validation, fsynced, strict-reloaded, and no-replace linked to the destination.
Nonzero exit leaves no partial result.

The controlling session owns the original PID from launch through validation. It polls
at intervals no longer than 55 seconds, checks liveness, GPU utilization, memory,
bounded metric-free log growth, atomic destination/temp state, and disk headroom, and
immediately classifies exit or failure. Completion triggers strict offline validation
and the next authorized branch in the same working session. A watcher may never
terminate at a terminal marker without advancing or explicitly closing the branch.

If the sole evaluator fails after opening official inputs, preserve the failure and
stop. At most one replacement attempt is allowed, and only after an independent review
demonstrates a structural implementation/infrastructure defect, confirms that no metric
or partial aggregate was exposed, and freezes any required source fix before the
replacement. It reruns all 48 rows from index 0. Attempt number and every prior exit
status enter the final result. A second structural failure publishes the failed gate;
observed scientific values may never motivate a rerun or threshold change.

The final report includes all five gating paired rows plus the seed-1 sensitivity row,
confidence intervals, exact query-level evidence hashes, registered-epoch trajectories,
profiler and wall-clock measurements,
peak allocation, checkpoint/storage costs, commits, commands, environment, and an
explicit statement that the training subset differs from the official training recipe.
Independent adversarial review must have no Critical or Important findings before any
Pareto language is used.
