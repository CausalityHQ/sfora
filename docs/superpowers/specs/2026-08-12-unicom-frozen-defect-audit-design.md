# UNICOM Frozen-Defect Audit design

Date: 2026-08-12

## Decision

Build and run a no-training audit before inventing another UNICOM loss. The
audit measures two source-confirmed but outcome-unmeasured defects:

1. deployment ranks an unnormalized 512-D prefix cut from a normalized 768-D
   vector; and
2. four class shards use different random feature masks inside one distributed
   softmax.

This is an evidence-acquisition tool, not a learning method and not a SOTA
claim. It does not resurrect DCSR, GCBSS, or PE-HTGC. Those proposals remain
closed. A later mechanism is allowed only if this audit exposes a material
failure that a simpler evaluation or full-768 control does not already fix.

The operator has instructed the project to proceed autonomously with the
recommended option and to avoid further approval pauses. That standing choice
is applied here.

## Alternatives considered

### A. Deployment geometry only

Export frozen embeddings and compare the official prefix, a renormalized
prefix, and random 512-D subspaces. This is the fastest option and directly
tests the evaluation mismatch, but it leaves the four-rank training objective
unmeasured.

### B. Deployment geometry plus four-shard emulation — selected

Run the same frozen-embedding comparison and emulate the exact class-sharded
softmax on one machine. This adds modest CPU work and separates class-shard
placement sensitivity from feature-mask effects. It gives both known defects a
chance to be falsified before expensive training.

### C. Start a one-GPU supervised UNICOM port immediately

This could approach the published 95.5 operating point, but any result would
mix gradient accumulation, mask coherence, evaluator changes, and one-versus-
four-GPU optimization. It is expensive and lower-information than B until the
two known defects are measured. It is deferred.

## Scope

The first implementation has three isolated components:

1. a pure NumPy audit library for retrieval metrics, paired bootstrap, energy
   bias diagnostics, and four-shard objective emulation;
2. a small exporter that uses the pinned official UNICOM checkout and released
   ViT-B/16 weights to write one frozen FP32 embedding bundle; and
3. a command that validates the bundle, runs the audit, and atomically writes a
   JSON report.

The library must not import training code or require Torch. The exporter is the
only Torch-dependent component. No component modifies weights, performs
backpropagation through the backbone, reranks a gallery, or fits on query or
gallery identities.

## Frozen inputs

The released UNICOM ViT-B/16 checkpoint is evaluated on the official
DeepFashion In-Shop split with the official transform and one-image inference
path, without test-time augmentation.

One embedding bundle contains, in this exact logical order:

- train embeddings and labels: 25,882 rows;
- query embeddings and labels: 14,218 rows;
- gallery embeddings and labels: 12,612 rows;
- each embedding is a finite, C-contiguous FP32 array with 768 columns;
- labels preserve the exact dataset identity strings; and
- model identifier, official UNICOM Git revision, checkpoint SHA-256, dataset
  root-relative image-list SHA-256, transform description, and array SHA-256
  values.

The exporter sorts rows by the official dataset order, not filesystem order.
It writes to a same-directory temporary path, fsyncs, and publishes without
overwriting an existing bundle.

## Experiment E1: deployment geometry

For a full embedding `z`, define `u = z / ||z||_2` and prefix projector `P`
selecting coordinates `[0, 512)`. Compute all retrieval views from the same
frozen arrays:

1. `official_512`: `P u`, squared Euclidean ranking;
2. `prefix_unit_512`: normalize `P z`, squared Euclidean ranking;
3. `full_unit_768`: `u`, squared Euclidean ranking; and
4. `random_unit_512[j]`, `j=0..31`: draw 512 coordinates without replacement
   from `Generator(PCG64(j))`, sort the indices, select, normalize, and rank.

Every view reports Recall@1/10/20/30 and mAP@R as fractions. Exact distance
ties use original gallery order. Retrieval is chunked without changing order or
floating-point expressions.

For each query, persist top-1 correctness and top-1 gallery index for the
official and corrected prefix. For every query where those views select
different galleries, define

```text
energy_gap(q) = ||P u_official_top1(q)||^2
              - ||P u_prefix_unit_top1(q)||^2.
```

The source-derived gallery-energy bias predicts a negative gap. Report the
mean, median, negative fraction, and a 10,000-resample paired percentile
interval using `PCG64(205)`. Also report the point-biserial association between
official top-1 error and the selected gallery's prefix energy. These are
diagnostics, not decision thresholds.

Define

```text
delta_norm = R1(prefix_unit_512) - R1(official_512)
delta_mask = median_j R1(random_unit_512[j]) - R1(prefix_unit_512)
mask_wins  = count_j[R1(random_unit_512[j]) > R1(prefix_unit_512)]
disagree   = median_j mean_q[top1_random_j(q) != top1_prefix(q)]
```

The released checkpoint must first reproduce documented `official_512` R@1 to
within 0.002 absolute of 0.746. If it does not, the report is a reproduction
failure and no scientific decision is made.

After that gate, compute two independent flags:

1. `coordinate_nonexchangeability` is true if `delta_mask >= 0.002`,
   `mask_wins >= 24`, and `disagree >= 0.10`; and
2. `evaluator_repair` is true if `delta_norm >= 0.002` and the 95% paired
   bootstrap lower bound for `delta_norm` is positive.

The primary decision is `EVALUATOR_REPAIR` whenever `evaluator_repair` is true,
otherwise `COORDINATE_NONEXCHANGEABILITY` when its flag is true, and
`GEOMETRY_NULL` otherwise. Both flags are always persisted, so a coordinate
finding is retained as a diagnostic when evaluator repair has priority.
Thresholds cannot be changed after the embedding bundle is opened.

## Experiment E2: four-shard objective emulation

E2 uses only frozen training embeddings. It selects a tractable but frozen real
panel rather than pretending to reproduce all 3,997 classes. From sorted
identities having at least four images, `PCG64(205)` selects 64 identities
without replacement, then restores identity order. For each identity, its
prototype is the FP64 mean of all its FP32 training embeddings, cast to FP32
and L2-normalized; the batch contains the first four examples in official
dataset order. The resulting panel has 64 prototypes and 256 examples.

The 64 sorted panel classes are split into four contiguous 16-class shards,
matching the official `PartialFC_V2` partition rule on the reduced panel. The
emulation computes the official global loss:

- each shard draws its own sorted 512-of-768 mask from an independent
  `PCG64(1000 + trial * 4 + rank)` stream;
- each shard normalizes the batch and its local prototypes in its own selected
  subspace;
- local logits are concatenated into one global softmax;
- the target logit receives the official additive angular margin `m=0.25` and
  all logits use scale `s=32`; and
- the straight-through gradient convention in the official implementation is
  emulated exactly for gradients with respect to the selected normalized
  embeddings.

Run 32 trials. In each trial use the same four masks under 16 deterministic
class-to-shard permutations from `PCG64(3000 + trial)`. Compare three paths:

1. `independent_masks`: the source-faithful four-mask objective;
2. `coherent_mask`: all four shards use rank zero's mask; and
3. `full_768`: all shards use all dimensions.

Persist loss, per-example loss, and embedding-gradient arrays only long enough
to reduce them to scalar statistics. Report:

- standard deviation and range of loss over class-shard permutations;
- mean gradient MSE and cosine distance versus `full_768`;
- the same values for `coherent_mask`;
- invariance error when class and prototype order are permuted consistently;
- mask union coverage; and
- finite/count checks.

E2 is `SHARD_SENSITIVE` only if all hold:

1. independent-mask permutation loss range is at least `1e-3`;
2. its mean gradient MSE versus full-768 is at least 25% larger than the
   coherent-mask MSE;
3. coherent-mask consistent-permutation invariance error is at most `1e-6`;
4. independent-mask class-shard placement changes at least 10% of per-example
   top-1 prototype predictions; and
5. all values are finite.

Otherwise E2 is `SHARD_NULL`. This gate measures sensitivity; it does not prove
that synchronized masks improve trained retrieval.

## Output and failure handling

The JSON report contains version, input hashes, exact constants, E1 metrics and
per-query bootstrap summaries, E2 aggregate statistics, decisions, runtime,
and warnings. It contains no candidate training outcome.

Malformed arrays, nonfinite values, duplicate output paths, split-count drift,
label mismatches, or a failed zero-shot reproduction gate produce a nonzero
exit and either no report or a report explicitly marked `REPRODUCTION_FAILED`.
The command never overwrites an existing report. JSON is strict (no NaN or
Infinity) and is reloaded and validated after publication.

## Testing

Tests are written before implementation and cover:

- exact normalization order for every view;
- stable tie breaking and Recall@K/mAP@R on hand-computed fixtures;
- deterministic masks and bootstrap indices;
- a synthetic energy-bias example where official and corrected rankings differ;
- the three E1 decisions and reproduction failure;
- exact four-shard sizes, mask assignment, joint-softmax ArcFace values, and
  straight-through gradients against a small enumerated oracle;
- invariance under coherent masks and violation under frozen adversarial
  independent masks;
- all E2 thresholds at pass/fail boundaries;
- strict bundle/report schemas, nonfinite and shape mutations;
- atomic no-clobber publication and persisted reload validation; and
- exporter dataset-order preservation using a tiny fake model/dataset without
  requiring Torch in the unit suite.

## Next experiment boundary

No new UNICOM training starts from this design alone. After the active
PA→MCPS→compactness sequence completes, export may use the free GPU. A
training mechanism is considered only if E1 or E2 exposes a material failure
that survives its simpler control. If both are null, the next high-value step is
a one-GPU supervised UNICOM B/16 port comparing the official sampled-subspace
objective directly with full-768, not another mask or consistency loss.

Any top-10 or SOTA claim still requires a locally reproduced comparable UNICOM
point, paired multi-seed evidence, unchanged backbone/input/descriptor/data/
pretraining/inference lane, no best-test-epoch selection, and replication on at
least two of CUB, Cars196, and SOP. In-Shop results using web-scale pretraining
must disclose possible train/test contamination.
