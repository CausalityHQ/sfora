# Teacher-Anchored Neighborhood Distillation Design

## Objective

Build a generic compact similarity method that transfers local retrieval geometry
from a strong offline teacher into a cheaper encoder. The deployed model emits one
normalized 128-dimensional vector for Sfora's existing symmetric-int8 storage and
search path. Teacher inference, labels, class names, and dataset-specific logic are
absent from the training loss and from serving.

Stanford Online Products (SOP) is the first development screen. The reusable
library accepts tensors and immutable sample identities, not SOP records. Class
IDs are used only to make and score class-disjoint development splits and to
define the fitting-only diagnostic subset.

## Evidence and hypothesis

The only numerical authority for this design is
`docs/evidence/representation_ceiling/sop-representation-ceiling-v1.json`,
SHA-256 `a89a09f73661fd64acc666b84732c411cb74b215103dbf7d818ba47c71624f3e`.
It is a claim-ineligible probe on three deterministic 80/20 class-disjoint splits
of the official SOP training classes, not the official SOP test partition.

On seed 17 validation, full-width source and teacher mAP@R are respectively
`0.5183644663` and `0.5748696112`; teacher PCA128 reaches `0.5525252015`.
Across seeds 17, 1729, and 65537, teacher PCA128 mAP@R is approximately `0.5525`,
`0.5649`, and `0.5673`. The sealed ridge arm scoring `0.5040401316` is a
different estimator: it first regresses to the full teacher and then fits PCA to
normalized ridge outputs. It is not the initialization specified below.

Separate operator-run diagnostics motivated abandoning random proxies and large
learned heads, but they are not repository-sealed evidence and therefore supply
no numeric gate in this design. The testable hypothesis is that a cheap encoder
can learn the teacher's neighborhood relationships from images while remaining
close to its pretrained geometry. Neither another frozen-feature regressor nor
randomly initialized class proxies tests that hypothesis.

The serving target is explicitly ViT-B/16 plus a 768-by-128 affine head and
normalization. The ViT-L/14@336 teacher is training-only. Encoder and retrieval
latency are measured as separate evidence; no unsealed latency number is an
acceptance premise.

## Split-local initialization

For each seed, partition official SOP training classes before fitting anything.
Fit the centered teacher PCA128 transform and the source-to-teacher-PCA ridge map
using that seed's fitting rows only. Validation rows cannot affect PCA, ridge,
penalty selection, anchors, batches, or checkpoint selection.

The ridge penalty is fixed at the sealed inner-selection value `1e-6`; it is not
selected on the new validation results. The exact
map first unit-normalizes every raw snapshot row with the ceiling probe's exact
semantics: compute row norms in float64, reject nonfinite or norms at most
`1e-12`, divide in float64, then return contiguous float32. It then calls Sfora's
`fit_teacher_guided_projection(source_fit, teacher_fit,
dimensions=128, penalty=1e-6)`, including its centering, teacher PCA basis, and
normalization semantics, where `source_fit` and `teacher_fit` are those normalized
matrices. Recompute the split's teacher-PCA state digest and require exact equality
to the sealed `transform_sha256.teacher-pca` before any gradient update. The
digest recipe is exactly the ceiling probe's `_parameter_sha256`: for the mean
and then components, append little-endian `u32 ndim`, little-endian `u64` shape
entries, and C-contiguous little-endian float32 bytes. Then
evaluate and seal this exact
map by applying the float32 runtime head to the sealed source snapshot features
for the seed-17 validation rows on the registered CUDA device in fixed chunks of
256 rows. The receipt binds device name and architecture, chunking,
fitting/validation identity digests, source and teacher arrays, PCA state, ridge
state, penalty, float32 outputs, and mAP@R/Recall@1. Step zero is required to
replay that new snapshot-feature receipt exactly at the state, identity, output,
and metric level only on that registered device and chunking. For portable
validation, float32 outputs are moved to CPU and compared with the library's CPU
float64-fit map using maximum absolute code error `1e-5` and float64-computed
cosine at least `1 - 1e-7`; the tolerance is
mutation-locked rather than called bitwise equality. The historical
`0.5040401316` arm is not its comparator.

This recomputation is a pre-scientific DGX authority check. Its receipt records
`torch.get_num_threads()` and Torch's parallel backend, and authenticates the
B/16 and L/14 snapshot `train_embeddings` arrays as
`d88e9d35f8419c7a661bd1358c901ecb2c64d4111ecd6ec311229d0d7c76dd74`
and `5a8629deee1adff92ac941a0f78f4fd55f1d4cb0db43b1459a0d6c92f84b46fc`,
respectively. Digest mismatch stops before science.

Let frozen teacher PCA produce normalized target `t_i` in 128 dimensions. Let the
original source encoder produce the normalized 768D feature `h_i^0`. The student
is

`s_i = normalize(W h_i + b)`.

`W,b` are initialized from the split-local ridge map. The head is one affine map:
no MLP, BatchNorm, proxies, or learned residual gate.

## Model modes and trainable state

Keep two source-model instances during the sampled snapshot replay preflight. The
original reference model is permanently in `eval()` and runs under `no_grad`;
after the replay check it is not used by training because authoritative `H0`
comes from the authenticated snapshot. The student is put in training mode only for
the trainable modules, while every `BatchNorm1d/2d/3d` and `SyncBatchNorm` in the
complete UNICOM graph is forced to `eval()` on every epoch and after every mode
transition. The frozen source feature projection, all blocks except the final
two, and their parameters and buffers remain byte-identical to initialization.

The authoritative `h_i` is the normalized output of the complete UNICOM feature
path, including its projection BatchNorm layers, not a pre-projection transformer
token. Authenticate the exact module inventory: the pinned ViT-B/16 has real
`DropPath` modules in blocks 1--11 with rates derived from `drop_path_rate=0.1`.
Set every `DropPath` to evaluation mode after every mode transition so stochastic
depth is disabled without rewriting the graph. The epoch-two trainable inventory
is exactly `blocks.10`, `blocks.11`, `norm`, and the new head. Snapshot hashes
cover all frozen parameters and buffers before and after training.

For epoch one train only the affine head. For epochs two through ten train the
head, final two transformer blocks, and final LayerNorm. Master parameters remain
float32. The saved artifact is a complete merged encoder-plus-head state dict,
not merely a patch. A fresh serving reconstruction in `eval()` must reproduce the
saved model bitwise for the same registered batch composition. A pristine-model
preflight seals batch-shape sensitivity before training; single-image versus
batched normalized codes must have maximum absolute difference at most `0.002`
and cosine at least `1 - 1e-5`. Before optimization, replay a registered sample
of image paths through the pristine encoder and compare its normalized outputs
with the authenticated source snapshot under those same bounds. Both checks run
inside the trainer, not only in a later evaluator. The same bounds apply before
evaluation.

The head-only causal control is separate: its complete encoder stays frozen and
byte-identical through all ten epochs while the affine head uses the complete
objective. It consumes the identical batches and update count, constructs fresh
head-only AdamW state at epoch two, and follows the registered head LR schedule.

## Fixed teacher anchors

For each fitting row construct 512 distinct anchors from fitting rows only:

- nearest 64 by cosine similarity of normalized teacher PCA128 codes;
- 64 sampled without replacement from teacher ranks 65--512;
- 384 uniformly sampled from fitting rows after excluding self and every row at
  teacher rank 1--512.

Exclude self. Break ranking ties by immutable sample identity: integers use
ascending numeric order, strings use Unicode lexical order, and mixed types use
integers before strings. Digest serialization is separate from this ordering.
Sampling uses
SHA-256-domain-separated PCG64 streams derived from experiment seed and query
identity. Retained anchor identities and source digests are evidence.

At temperatures `0.05` and `0.20`, form teacher distributions from `t_i dot t_j`
and asymmetric student distributions from `s_i dot t_j` over the same anchors.
`L_anchor` is mean forward KL across queries and temperatures.

## Symmetric deployment geometry and guards

Within each complete 256-row batch, exclude self and form teacher and student
pair distributions at the same temperatures. `L_symmetric` is mean forward KL
from teacher to student distributions. `L_point = mean(1 - dot(s_i,t_i))`.
The code-space terms share `s=t` as a global optimum, while the feature-drift
term is minimized by `H=H0`; the neighborhood controls therefore test gradient
reweighting and geometry preservation, not distinct code targets.

All loss reductions and diagnostics below run in float32 outside autocast. With
centered, normalized batch matrices, define covariance using denominator
`batch_size - 1`. For coordinate standard deviations use
`sigma = sqrt(diag(C).clamp_min(0) + 1e-4)`. Then

`L_cov = ||C_s-C_t||_F^2 / (||C_t||_F^2 + 1e-12)`

plus the coordinate mean of
`relu(0.5 - sigma_s/(sigma_t+1e-4))^2`. Constant-code, zero-variance,
nonfinite-input, and backward-gradient fixtures must remain finite and stable.

Let `H` and `H0` be adapted and original normalized 768D features. The original
features come from the authenticated source snapshot made with the exact same
canonical UNICOM view as training; a sampled image-path replay must match the
snapshot before training under the separately sealed pristine batch-shape bounds
of maximum absolute error `0.002` and cosine at least `1 - 1e-5`. `L_drift` is
the mean squared difference between the complete within-batch pairwise cosine
matrices of `H` and `H0`.

The complete fixed objective is

`L = L_anchor + 0.1 L_point + 0.5 L_symmetric + 0.05 L_drift + 0.01 L_cov`.

Each term is returned as an independently finite scalar. Coefficients do not
change after development results.

## Optimization and determinism

Each epoch uses a PCG64 permutation of fitting identities derived from the seed
and epoch. Form each 256-row batch from 128 distinct seed rows followed, in seed
order, by one distinct nearest neighbor under teacher-PCA128 cosine for each seed.
Within the current batch, partners cannot be seeds or previously chosen partners;
ties use immutable identity. This exclusion does not extend across batches: a row
may be a partner in one batch and a seed in another during the same epoch. If 128
partners cannot be selected for a batch, fail closed. Record per epoch the number
of identities appearing in more than one batch. This query-independent,
label-free schedule supplies real local pairs to the symmetric loss. Drop unused
tail seed rows, record them, and execute exactly
`floor(fitting_rows/128)` complete updates per epoch. All arms consume identical
batches. Compute and authenticate each row's exact first 256 non-self neighbors
once with blocked matrix products, then reuse that immutable ranking across every
epoch and arm. Because at most 254 other rows are excluded when the last partner
is chosen, this bounded prefix is exactly equivalent to an exhaustive search.
Build and seal all ten epoch schedules once per seed before any arm launches;
every arm consumes those same bytes. The preflight must complete the ranking and
all schedules within 15 minutes or stop before science.

The canonical view has no augmentation and data loading uses zero workers. Set
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, enable Torch deterministic algorithms, set
cuDNN deterministic and benchmark off, and pin the math SDPA backend. Bind
Python, NumPy, Torch, CUDA, UNICOM, sampler, and bootstrap seeds. Bitwise equality
across GPU architectures is not claimed.

Disable TF32 for both CUDA matrix multiplication and cuDNN, set float32 matrix
multiplication precision to `highest`, and bind those states into every receipt.
This is required for the step-zero head tolerance and deterministic patch-embedding
convolution; a backend that cannot honor the settings fails closed.

Do not add an outer autocast context: pinned UNICOM already enters float16
autocast inside attention and transformer blocks, while final norm, feature
projection, head, normalization, and all losses must remain float32, matching the
official exporter. Retain GradScaler because block backward traverses float16.

Epoch one uses head-only AdamW with LR `1e-4`, betas `(0.9,0.999)`, epsilon
`1e-8`, weight decay `0.01` except bias, constant LR, clipping `1.0`, and no
scheduler. Epoch two constructs fresh AdamW state over the registered inventory
with the same betas, epsilon, decay exclusions, and clipping; head LR is `1e-4`
and backbone LR `1e-6`. For zero-based update `u` across the remaining `N`
updates, the first 100 factors are `(u+1)/100`; afterward the factor is
`0.5*(1+cos(pi*(u-99)/(N-100)))`. Unscale before clipping. GradScaler starts at
`1024`, has growth interval `2^31-1`, and any skipped/nonfinite update
invalidates the arm rather than silently changing its update count. Record
attempted and successful updates. Endpoint is epoch 10; no best-epoch selection
or interpolation sweep exists.

## Evaluation and causal controls

Use the committed seed-specific 80/20 class-disjoint split. Every validation
image queries every other validation image; self is excluded and ties use source
order. The official SOP test partition is burned prior evidence and cannot be
used for training, selection, or a new claim.

Evaluate float32 and symmetric int8-by-int8 self retrieval using the library's
`pack_int8_unit_embeddings` and `score_symmetric` loaded from
`scripts/probe_sop_relational_linear.py` by the repository's tested
`importlib.util.spec_from_file_location` pattern, with candidate width
equal to the validation split's maximum class count minus one, for:

The loader first places the repository `scripts` directory on `sys.path`, because
the probe imports sibling scripts. The int8 packer itself is imported from its
library module; the script loader supplies only the scorer and probe-specific
evaluation behavior. The trainer uses this same loader for the epoch-one
fitting-probe stop and every per-epoch diagnostic.

1. original full-width source;
2. teacher full width;
3. teacher PCA128;
4. exact ridge-initialized snapshot-feature student at step zero, plus a separately
   named live-encoder step-zero pass under the registered evaluation batching;
5. head-only training with the complete objective;
6. adapted base control `B = 0.1 L_point + 0.05 L_drift + 0.01 L_cov`;
7. adapted anchor control `B + L_anchor`;
8. adapted symmetric control `B + 0.5 L_symmetric`;
9. final-two-block training with the complete objective.

Arms 6--8 are nested objective controls; no global gradient-scale matching is
claimed because AdamW largely cancels such scaling. They isolate the two
neighborhood mechanisms while preserving the same guards. All arms share rows,
batches, update count, endpoint, and optimizer schedule.

Primary gates use student-student symmetric retrieval. Student-query versus
teacher-gallery retrieval is diagnostic only. Report per-query AP and Recall@1,
teacher-neighbor overlap, validation-to-fitting-anchor KL, effective rank
`exp(-sum(p*log(p)))` for normalized covariance eigenvalues `p`, leading and
top-eight eigenvalue shares, gradient norms, and fitting/validation losses.

## Stop and advancement gates

Step zero must replay its newly sealed exact-initializer receipt. A fitting-only
free-code optimization fixture must approach its teacher-code solution; failure
is an implementation error.

Before training, order fitting class identities by SHA-256 of the 8-byte
little-endian unsigned experiment seed followed by the 8-byte little-endian
signed class ID, with ties broken by class ID. Select the first 512 classes in
that seed's global ordering and seal all rows of those classes as the fitting-only
diagnostic subset. This preserves complete
positive sets; its candidate width is its maximum class count minus one. The
fitting-probe step-zero comparator is the live encoder-plus-head under the same
registered batching used for later fitting probes; it is reported alongside the
snapshot-feature receipt and must satisfy the sealed batch-shape bound. An arm
stops after epoch one only if fitting-probe mAP@R is more than `0.002` below this
live step zero, fitting-probe effective rank is below 70% of initialization, or its leading
eigenvalue share exceeds twice initialization. Per-epoch validation is recorded
but non-binding; validation affects only the epoch-10 arm decision. A stopped arm
has no epoch-10 candidate.

The authoritative candidate is epoch 10. On seed 17, the complete arm advances
to within-population stability checks only if all are true:

- packed-int8 mAP@R improves over the live packed step-zero pass by at least `0.015`;
- packed-int8 mAP@R is no worse than that split's float32 full-width source;
- packed-int8 mAP@R is within `0.015` of that split's packed teacher PCA128;
- packed-int8 Recall@1 loses at most `0.002` versus float32 split-local source;
- packed complete minus the packed base control is at least
  `0.003` mAP@R and has a positive one-sided paired bootstrap lower bound.

The float32-to-symmetric-int8 delta is always reported but is not a separate
absolute gate: cross-dataset sealed evidence shows dataset-dependent quantization
loss, while every advancement comparison above already uses the deployed packed
representation.

The paired bootstrap reuses the sealed ceiling probe's exact estimator: per-query
AP differences clustered by validation class, query-weighted ratio-of-sums,
10,000 `torch.Generator` draws from seed 17 in blocks of 128, and the 5th
percentile with lower interpolation. A scalar replay fixture pools the sealed
ceiling receipt's three validation splits and must reproduce
`-0.0003904282392322747`. It authenticates the training-label array with SHA-256
`d785d2eca417d91257195bf2c16d87b7805178984db82c0d0a0f5ca8546a216f`, maps labels
through each receipt split's validation row indexes, and fails closed if either
authority is absent or differs. The digest is over the C-contiguous raw bytes of
the little-endian int64 `train_labels` member, which must agree in both
authenticated B/16 and L/14 snapshot archives. The replay treatment is per-query AP from
`ridge-source-teacher-full`, the baseline is per-query AP from `source-full`,
splits are concatenated in receipt order, and cluster identity is the original
class ID across splits. The seed-17 advancement gate applies the same estimator
to seed-17 validation rows only; it does not pool later stability splits. This
historical Torch estimator is deliberately distinct from PCG64 streams used for
anchors and batch permutations. It compares epoch-10 complete both to the live
packed step-zero pass and to the base control. Both contrasts are binding: the
complete-minus-step-zero contrast must have a positive one-sided lower bound in
addition to the point improvement of at least `0.015`, and the
complete-minus-base contrast must satisfy its `0.003` point gate and have a
positive one-sided lower bound. Failure of only the neighborhood contrast classifies
the result as generic anchored adaptation, not evidence for neighborhood
distillation.

If the base control stops or otherwise lacks an epoch-10 result, the causal result
is inconclusive and neighborhood advancement is forbidden; never substitute an
earlier score or treat stopping as superiority. The same rule holds on the
stability seeds.

If complete advances, repeat it and the same base control as within-population
stability checks on seeds
1729 and 65537 without changing settings. Every stability gate is relative to
that seed's source and teacher-PCA ceiling. The bootstrap generator seed remains
17. Both stability seeds must pass every advancement gate against their own
ceilings and live step-zero/base controls; any failed gate or missing base result
classifies the recipe as unstable, with no partial stability claim. `0.567` mAP@R is an aspirational
engineering marker, not a preregistered gate. A publication-quality claim also
requires the frozen recipe on at least two additional retrieval datasets and a
fresh evaluation protocol; SOP alone cannot establish generality.

## Scientific execution envelope

The DGX launcher owns one process group and polls it every 30 seconds. The
seed-17 five-arm panel and each later stability launch have an 18-hour wall cap.
Before science, a 256-row synthetic forward/backward at the epoch-two trainable
inventory records summed `/proc/<pid>/status` `VmRSS` across the process group
and peak CUDA allocation; summed RSS must remain below 16 GiB, half the 32-GiB
run cap. A real-image throughput preflight measures training updates plus
validation/fitting-probe encodes and requires the conservative projected panel
time below 12 hours, leaving one-third of the 18-hour cap as margin. The head-only
arm consumes authenticated snapshot features because its encoder and feature
drift are frozen. The environment is pinned to the sealed DGX authority: Python
3.13.9, Torch 2.12.1+cu130, and the exact imported `timm` version. Stop without
resume or altered retry if summed
process-group RSS exceeds 32 GiB, host memory PSI `full avg10` is
at least `0.79` once or at least `0.50` for three consecutive samples, swap usage
grows by more than 2 GiB from the prelaunch sample, or no authenticated stage or
successful-update progress event appears for 15 minutes. Send `TERM` to the
process group, then `KILL` after 30 seconds if needed. A stop emits an
outcome-blind canonical receipt binding the last completed arm/epoch/update, all
pressure samples, and the reason; it is never a partial quality result. Unit
tests mutation-lock threshold equality, sustained counters, progress reset,
process-group termination, and canonical stop bytes before science launches.
The trainer appends one canonical JSON line to a dedicated progress file for
every authenticated stage transition and successful optimizer update. Each line
binds the launch-receipt digest, a strictly monotone sequence number,
arm/epoch/update, and the SHA-256 of the preceding line. The launcher resets the
15-minute timer only after accepting an exact chain continuation; a duplicate,
foreign run, broken chain, or truncated tail cannot reset it. An authority or
quality terminal is never retried. An external resource stop, whether before or
during science, permits exactly one unchanged relaunch of arms lacking epoch-10
results after the recorded external cause is corrected. Discard the partial arm,
never resume it, and bind both attempts in the receipt chain. A second resource
stop is terminal and requires a new specification.

## Artifacts and serving contract

The trainer accepts only authenticated local checkpoints, fitting snapshots,
image root, source revision, and explicit execution flag. It writes no official
test artifact. Receipts bind files, identities, split, PCA/ridge states, anchor
lists, model modes, frozen-state hashes, trainable inventory, optimizer schedule,
all coefficients, per-epoch diagnostics, endpoint or
stop reason, and model digest. JSON is canonical and newline-terminated;
`claim_eligible=false`.

Deployment uses only the merged ViT-B/16, affine 128D head, normalization, and
existing symmetric-int8 codec. Teacher, anchor tables, labels, and PCA fitting
data are training-only. Latency evidence separately measures batch-one encoder,
batch throughput, and retrieval on the registered hardware.

## Non-goals

- No class-name or text-semantic supervision.
- No SOP-specific behavior in the reusable library.
- No random proxy table or Proxy Anchor retry.
- No quantizer change before symmetric-int8 loss is material.
- No SOTA or broad-transfer claim from this development screen.
