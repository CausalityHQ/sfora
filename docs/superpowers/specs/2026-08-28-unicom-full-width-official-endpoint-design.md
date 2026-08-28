# UniCOM Full-Width Official Endpoint Gate

**Date:** 2026-08-28
**Status:** prospective; no full-width checkpoint has been evaluated on the official
DeepFashion In-Shop query/gallery split.

## Objective and claim boundary

Decide whether the verified advantage of full-width ArcFace fine-tuning transfers
from the fixed training-identity holdout to the untouched official In-Shop test
identities. The experiment compares five paired epoch-16 checkpoint bundles:
official sampled-512 feature training versus full-768 feature training, with the same
768-dimensional deployed descriptor in both arms. The seed-2 pair is retained from
the historical campaign. The seed-3--6 checkpoint files were missing at the
2026-08-28 preflight and must be prospectively reproduced under the historical
training recipe before this gate can run.

This is an official **evaluation-split transfer gate**, not yet an official training
reproduction. Both arms were trained on the same fixed identity-disjoint 80% of the
official training identities (about 3,200 of about 3,997 ArcFace classes and,
incidentally, about 80% of the 25,882 training images). The result can support a
paired generalization claim and authorize full-data training; it cannot by itself
support a state-of-the-art claim.

The strongest verified directly comparable raw-descriptor anchor found in the
primary literature is UniCOM ViT-L/14-336 at 96.7% In-Shop Recall@1. UniCOM reports
96.0% for ViT-L/14 and 96.7% for ViT-L/14-336. Recent LoCoRE results are a separate
local-descriptor re-ranking system and do not replace the raw-descriptor anchor.

Primary sources:

- UniCOM, ICLR 2023: <https://arxiv.org/abs/2304.05884>
- DeepFashion, CVPR 2016:
  <https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf>
- LoCoRE, CVPR 2025: <https://arxiv.org/abs/2503.21772>

## Evidence entering the gate

The candidate is `full_768`; the paired baseline is `sampled_512`. Seeds are exactly
`(2, 3, 4, 5, 6)`. Both arms use the same initial UniCOM ViT-L/14-336 checkpoint,
class-mean classifier initialization, image bytes, training-identity split, batch
order, optimizer, OneCycle schedule, ArcFace margin/scale, 16 epochs, and full-768
holdout readout. They differ only in whether each classifier shard uses a random
512-of-768 coordinate subset or all 768 coordinates.

The historical controlled holdout mAP@R deltas are
`(+0.0012584412, +0.0045961954, +0.0022570035, +0.0041930814,
+0.0032592434)`. Their mean is `+0.0031127930`, paired Student-t 95% interval is
`[+0.0014057929, +0.0048197931]`, all five are positive, no seed loses a top-1
query, and four of five candidates reach the matched control endpoint by epoch 12.
Seed-0 A-B-B-A step wall-time ratio is `0.9946993577`, with 95% interval
`[0.9913006311, 0.9979290843]`; this is evidence of no measurable slowdown, not a
speedup claim. Checkpoint file sizes and deployed inference graphs are identical.

Those historical rows remain immutable evidence but are not automatically the rows
used by this endpoint. For every seed-3--6 checkpoint pair whose reproduced file hash
differs, the recovery run prospectively recomputes the same frozen holdout evaluation
and v4 substitutes that generation-matched row before applying the unchanged
confirmation decision. Any reported endpoint or paper must disclose that its panel
contains two retained historical artifacts and eight prospectively reproduced
artifacts unless an exact archived-byte restoration is actually observed.

Full-width also wins under the registered 512-coordinate diagnostic readout, so
train/deploy-width alignment is already falsified. The surviving explanation is that
random coordinate masking reduces useful optimization per coordinate. This gate tests
generalization, not that mechanism.

## Why operational completion and the endpoint are evaluated first

The prior official evaluator measured about 7.2 minutes per checkpoint while computing
two retrieval views. The new evaluator still encodes each checkpoint once, but computes
20 views: primary, two renormalized prefixes, 16 random masks, and the registered
normalize-before-prefix diagnostic. The prior official and holdout timings imply about
5.5 minutes of encoding plus 50--75 seconds per official retrieval view, so ten
epoch-16 checkpoints cost about 3.5--5 hours rather than 72 minutes. All 40 checkpoints
with the primary and registered diagnostic only cost about 4.8--5.3 hours, while ten
fresh paired 16-epoch training runs cost about 40 hours. Because eight seed-3--6
checkpoint files are missing, the prerequisite recovery is eight fresh 16-epoch runs,
about 32 GPU-hours. The five missing A-B-B-A quartets cost about 2.3 GPU-hours at the
measured 5.3 seconds per step. Recovery, operational completion, and endpoint evaluation
are therefore roughly 38--40 hours of GPU work plus four serialized 108.56-GiB
offload transfers. A same-route throughput probe before launch determines their
additional wall time and blocks the run if it is too slow for safe retention. This
remains the smallest evidence-based path to finish the current Pareto decision: it
restores a completed five-seed training comparison and then tests official transfer
without adding a tuned candidate.

Before any official query/gallery embedding is computed, run one A-B-B-A profile
quartet for each seed in `(2, 3, 4, 5, 6)` and the empirical deployment comparison
already required by the reviewed confirmation decision. Within a seed, reduce the two
candidate positions and the two baseline positions by their arithmetic means, then
form candidate divided by baseline. Across seeds, use the arithmetic mean of those
five ratios. A mean step-wall-time, peak-allocated-memory, or peak-reserved-memory
ratio above `1.02`, or any seed with unequal paired checkpoint file sizes, closes the
candidate under the existing registered rule.

The deployment comparison uses the two seed-2 epoch-16 checkpoints and identical
ordered synthetic inputs: a binary32 tensor of shape `(2, 3, 336, 336)` filled by
`torch.linspace(-1, 1, numel, dtype=torch.float32)` and reshaped without randomness.
It requires exact equality of model parameter names, shapes, dtypes, element counts,
and parameter byte counts computed as `numel * element_size`;
exact equality of the sorted CPU-profiler signature `(aten:: key, call count, input
shapes)` from one inference-mode forward after one warmup; exact equality of output
shape, dtype, and byte count; and exact equality of checkpoint and exported-state file
sizes in bytes. The output tensors must be finite but their bytes are expected to
differ because the
learned weights differ; values are not an equivalence predicate. Inference latency is
not re-benchmarked because the deployed graphs and tensor shapes are identical and the
registered missing-evidence contract asks for inference operations, not a new timing
gate. Any failed equality closes the candidate; only an all-pass comparison publishes
the completed operational decision and continues automatically to the endpoint.

Only epoch 16 is read in the official gate. Epochs `(4, 8, 12)` remain unopened until
a passing endpoint authorizes the official trajectory readout.

## Frozen inputs and order

- Dataset root: `/home/riomus/datasets/inshop_official_standard`.
- Partition: `Eval/list_eval_partition.txt`, exact SHA-256 already bound by the
  reviewed full-width run configuration.
- Official query: 14,218 images; gallery: 12,612 images; test identities: 3,985.
- Runtime NumPy: exactly `2.5.0`; the result also records the observed Python,
  PyTorch, CUDA, and NumPy versions.
- Seeds: `(2, 3, 4, 5, 6)` in that order.
- Within each seed: `sampled_512`, then `full_768`.
- Checkpoint: epoch 16 raw `checkpoint["model"]` only.
- Exactly ten distinct checkpoint byte hashes. Seed 2 must match its existing
  immutable training receipts and pair inventory. Seeds 3--6 use the historical
  hashes only if archived bytes are restored exactly; otherwise they use the
  prospective recovery inventory published under the recovery configuration.
- Before either stage launches, every checkpoint must be a regular non-symlink file
  whose size and SHA-256 match that inventory. A missing checkpoint blocks execution;
  it does not change the scientific decision. Recovery may restore archived bytes or
  prospectively rerun a config-only handoff whose source commit contains trainer and
  pair-evaluator Git blobs byte-identical to historical source
  `f76cd832e84c06b64c63a4ac728017123928b96c`, under the registered recipe. The
  historical trainer does not enforce
  deterministic CUDA algorithms, so a rerun is treated as a prospective reproduction;
  exact restoration is recognized only if the complete bytes happen to match. A
  reproduction whose SHA-256 differs cannot silently replace the paired checkpoint;
  it requires a prospective re-evaluation of its holdout pair before official use.
  The reproduced checkpoint bytes, new receipt, and new paired holdout row form one
  indivisible seed bundle. A v4 decision must substitute that new row for the
  historical row, and `SUPPORTED_HOLDOUT` remains a hard prerequisite: any other
  recomputed holdout decision publishes CLOSE and forbids opening the official split.
- One process, one launch, one result, and no in-place retry. A structural failure is
  not a scientific outcome, but a replacement launch requires a reviewed corrective
  Git commit and a newly committed run configuration before any metric-bearing result
  has been published.

## Evaluation

Load the authenticated UniCOM ViT-L/14-336 model once in FP32 on CUDA. For each
checkpoint, strict-load the raw model state, set evaluation mode, and encode the same
ordered query and gallery records using the existing deterministic 336-pixel
transform. Use all 768 coordinates, L2-normalize each complete descriptor, and rank
by exact Euclidean distance.

Persist, per seed and arm:

- Recall@`(1, 10, 20, 30, 40, 50)` and integer hit counts;
- query-weighted mAP@R and all per-query AP@R values;
- all per-query top-1 correctness values;
- identity-uniform mAP@R using the lexicographically first query per identity;
- ordered query/gallery path and label digests;
- complete query/gallery embedding byte digests;
- checkpoint path, hash, byte count, elapsed seconds, and peak allocated GPU bytes.

The following prespecified width panel is computed from each persisted 768-D embedding
without another model forward pass:

- primary full-768, unit-normalized;
- renormalized-prefix-512 and renormalized-prefix-256, each renormalized after slicing;
- 16 random 512-coordinate subsets, each renormalized after slicing. Construct them by
  creating `numpy.random.Generator(numpy.random.PCG64(76851216))`, calling
  `choice(768, size=512, replace=False)` exactly 16 times, sorting each returned index
  vector increasingly, and sharing those exact vectors across arms and seeds. Persist
  the 16 exact vectors and their canonical JSON SHA-256 in the Git-tracked run
  configuration, and require regeneration under the bound NumPy version to match; and
- registered normalize-full-then-prefix-512 without a second normalization.

Only full-768 and the registered normalize-full-then-prefix-512 view control candidate
promotion. The renormalized width-panel views test the published compactness rationale,
scope the claim, and never gate promotion. An expectation/logit-scale matching arm is
forbidden because the implementation already renormalizes both masked embeddings and
masked classifier weights before every cosine; there is no missing norm expectation to
match.

## Statistics and decision

The independent unit is the training seed. For each seed compute candidate minus
baseline deltas at epoch 16.

`PRACTICAL_OFFICIAL_TRANSFER` requires every predicate below:

1. mean primary mAP@R delta is at least `+0.003`;
2. the two-sided paired Student-t 95% lower bound for the five primary mAP@R deltas
   is above zero, using `t(4)=2.7764451052`;
3. at least four of five primary mAP@R deltas are positive;
4. the paired-seed Recall@1 95% lower bound is above `-0.001`;
5. a paired-query bootstrap Recall@1 lower bound is above `-0.001`, using one shared
   query resample across the five seeds, PCG64 seed `768`, and 10,000 replicates;
6. no seed's Recall@1 delta is below `-0.003`; and
7. the registered `normalize_before=True`, coordinates `0..511` retrieval view (the
   normalize-full-then-prefix-512 panel item without a second normalization) has mean
   mAP@R delta strictly above zero.

The seed-level interval controls the training-randomness claim; the query bootstrap is
supportive and cannot replace it. The `+0.003` practical floor is unchanged from the
controlled-holdout campaign, whose observed mean clears it by only 3.7%.

If predicates 2--7 pass and the mean primary mAP@R delta is in `[+0.002, +0.003)`,
status is `DIRECTIONAL_OFFICIAL_TRANSFER`: the direction is statistically supported
but the recipe-change floor is not. It may support a correction section and authorizes
only the extended sampled control plus one-mask mechanism control. If the mean is below
`+0.002` or any of predicates 2--7 fails, status is `CLOSE_FULL_WIDTH_OFFICIAL`.
There is no discretionary branch and no threshold change after observing the result.

For each renormalized diagnostic width, report the same paired seed interval. Also
compute baseline-minus-candidate deltas at that width. If their mean is at least
`+0.003` and their paired-t 95% lower bound is above zero at any width, set
`compact_readout_limitation=true`; this does not reverse the primary result but forbids
a width-general or compactness-improvement claim.

The result also reports each arm's absolute Recall@1 and its distance from the 96.7%
published UniCOM ViT-L/14-336 anchor. Because this experiment uses only 80% of the
training identities (about 3,200 of about 3,997 ArcFace classes), crossing or missing
96.7% is descriptive and never changes the gate.

All arithmetic is binary64. The paired Student-t interval is
`mean(delta) ± 2.7764451052 * sample_std(delta, ddof=1) / sqrt(5)`. Each bootstrap
replicate samples 14,218 query ordinals with replacement once and applies that same
ordinal vector to every seed and arm before recomputing Recall@1, subtracting within
seed, and taking the arithmetic mean of the five seed deltas; its interval is the NumPy
linear 2.5th and 97.5th percentiles of those 10,000 replicate means. Equality
at a threshold passes an `at least` or `no greater than` predicate; every nonfinite
value is structural failure.

## Implementation boundary

Operational completion adds exactly two focused producers while preserving the v3
confirmation result immutably: a deployment comparator for the structural measurements
above, and a v4 confirmation producer that supervises and validates the five newly
generated A-B-B-A comparison artifacts and calls the already-tested
`evaluate_unicom_full_width_objective.confirmation_decision` with five
generation-matched quality rows (historical only for an exact retained/restored pair,
otherwise the prospective recovery row), the arithmetic means of the five A-B-B-A
comparisons, and the deployment predicates. The v4 producer owns the completed
operational schema, recomputation, decision, and exclusive publication; it does not
rewrite v3. A v4 result other than `SUPPORTED_HOLDOUT` closes before official data is
loaded.

The official endpoint adds one focused evaluator that reuses the existing official
evaluator's tested partition parsing, model loading, embedding extraction, retrieval
metrics, deterministic runtime configuration, and exclusive JSON publication. The new
code owns only the ten-row inventory, additional width views, endpoint statistics,
exact result schema, and CLI. None of these producers copies the old 48-row evaluator
or adds another provenance subsystem.

The run configuration is ordinary Git-tracked JSON. Source and configuration are
committed normally, pushed on the research branch, checked out on the DGX, and the
ignored result is returned with `rsync`. Reproducibility bindings are the Git commit,
the configuration bytes, the ten checkpoint hashes, the dataset partition hash, and
the exact NumPy version plus the complete runtime record.

## Publication and failure behavior

The evaluator prints only value-free progress while running. It writes no partial
metric result. The final JSON rejects duplicate keys and nonfinite values, validates
all recomputable rows/statistics/decisions, publishes once with no-clobber semantics,
strict-reloads the persisted bytes, and exits nonzero on structural failure.

The same original PID is monitored at intervals no longer than 55 seconds. Each poll
records liveness, bounded log tail, GPU utilization/memory, process RSS, disk headroom,
and output/temp state. No replacement process starts while the original is alive.

## Automatic continuation

- `PRACTICAL_OFFICIAL_TRANSFER`: use the preserved seed-3--6 four-epoch recovery
  bundles and prospectively rerun both seed-2 arms under a separately committed
  trajectory configuration. Use all four epochs, including epoch 16, from that one
  new seed-2 pair in the trajectory analysis; never splice its early epochs with the
  endpoint's retained historical epoch 16. Disclose that the five-seed trajectory
  uses a prospective seed-2 bundle distinct from the endpoint bundle and costs about
  8 additional GPU-hours, then measure official time-to-quality on the resulting
  generation-matched 40-checkpoint panel. Afterward run three-seed In-Shop
  mechanism/fairness controls in this
  order: sampled-512 for 24 epochs, `official-one-mask` for 16 epochs, `prefix-512` for
  16 epochs, and a one-seed three-point sampled-classifier-learning-rate probe only if
  it improves its own frozen baseline by at least `0.003`. The 24-epoch arm matches
  the full-width arm's per-coordinate update opportunity; catching up changes the
  claim to faster time-to-quality, while remaining below strengthens the quality claim.
- `DIRECTIONAL_OFFICIAL_TRANSFER`: run only the 24-epoch sampled and 16-epoch
  one-mask controls. Do not promote the default recipe unless a subsequent fresh
  experiment clears its own preregistered practical floor.
- `CLOSE_FULL_WIDTH_OFFICIAL`: close fixed full-width ArcFace as the paper candidate.
  Retain it as a negative result and move to the next evidence-based similarity-
  learning candidate without tuning against official outcomes.

After the mechanism controls, require a five-paired-seed replication on one additional
dataset or backbone. CUB is first because it is already available and cheap; SOP is a
3+2 group-sequential follow-up with a stop-for-futility look after three seeds and a
claim only from all five. A Matryoshka/multi-prefix constructive arm is eligible only
after the endpoint, controls, and one replication survive. This ordering keeps the
current full-width null intervention as a reproducibility correction instead of
retroactively presenting it as a novel method.

A standalone paper requires operational completion, practical or directional official
transfer, the 24-epoch fairness result, a mechanism control, and one five-seed external
replication. Official transfer without replication is a correction section in the
broader reproducible-recipe paper. The full-data program must reproduce or exceed the
96.7% anchor before any SOTA wording.

No custom CUDA kernel is authorized by this gate. The measured objective is too small
a share of step time; kernel work becomes eligible only if a fresh profile identifies
at least 10% fusible training time and an exact-output prototype improves measured
time-to-quality.
