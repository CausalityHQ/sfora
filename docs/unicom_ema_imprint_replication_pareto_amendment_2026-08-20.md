# UniCOM imprint replication Pareto amendment

## Scope and chronology

This prospective amendment repairs only the six-seed confirmation evidence and final
claim. It does not change either training arm, seeds `1..6`, registered epochs
`[4, 8, 12, 16]`, selected cell `imprinted_raw`, data order, objective, quality
thresholds, or any already-produced checkpoint or paired report.

It is written after the immutable seed-1 paired report was observed and before any
seed-2 training. The seed-1 report remains unchanged at
`reports/generated/unicom_ema_imprint_replication_c83cd96_seed1.json`, SHA-256
`0cfb888fdbc0e409048943a8d6e47635e571dec5830b86260e0730d80ebf4ab8`.
It strictly validates and gives epoch-16 deltas `+0.01443298839283702` mAP@R and
`+0.008782936010037656` Recall@1. Its per-query bootstrap interval
`[0.007050684568871645, 0.02209061680646096]` is descriptive within-run evidence,
not a training-seed confidence interval.

Independent review `2946f2385b914ac9` found that the quality result is sound but
the v1 summary incorrectly treated contaminated wall time as a decision gate and
misreported the non-monotone seed-1 time-to-quality ratio as `2.0`. Review
`42524fa58bb04504` then rejected an initial repair because it combined epoch-16
quality with first-to-target cost and called dominance what the original
preregistration required only to be non-domination. This document replaces that
rejected repair. It keeps the operating points separate and discloses every
historical limitation.

## Frozen selection authority

The v2 summary embeds and validates this exact ordered object:

```json
{
  "path": "reports/generated/unicom_ema_imprint_factorial_88604a4_seed0.json",
  "sha256": "c0666a68e70990115d80e8dc06a9f94efe83156a3fddd50f36bdbf2b3b8cd217",
  "recording_commit": "81f3f48c374d14b5a91bbeba7a1fec2fb0a4a2d6",
  "selected_cell": "imprinted_raw",
  "decision": "PROMOTE"
}
```

`path` is relative to the repository root containing the executing summarizer.
The CLI requires that exact regular non-symlink path, hashes its exact bytes, runs
the existing strict factorial validator, requires gate fields
`selected_cell="imprinted_raw"`, `decision="PROMOTE"`, and `promoted=true`, requires
the recording commit to be an ancestor of the executing revision, and requires
`git show 81f3f48c...:<path>` to equal the worktree bytes. The recording commit is
the commit that added the immutable result, not the producing-source revision
encoded in its filename. `validate_summary` compares the embedded object to frozen
module constants; it never trusts an authority copied from the object being
validated.

## Two operating points, never one blended claim

### Fixed-budget quality

The quality gate remains exactly the preregistered epoch-16 all-six paired training-
seed gate. It reports the paired mAP@R and Recall@1 deltas and makes no assertion
that epoch-16 imprinted training uses no more compute. Fixed-epoch compute is reported
beside the quality result.

### Iso-quality time to the control endpoint

For each seed, the target is that seed's `random_raw` epoch-16 mAP@R. For each arm,
the first-quality epoch is the earliest registered epoch whose mAP@R is at least
that target. The epoch speedup is:

```text
random_first_quality_epoch / imprinted_first_quality_epoch
```

This is the sole confirmation estimator. It handles non-monotone controls. Seed 1
is exactly `{random_raw: 12, imprinted_raw: 8, speedup: 1.5}`. The earlier
`16/8=2.0` shorthand is withdrawn. `1.5` is also the preregistered imprint-closure
threshold and, on the `[4, 8, 12, 16]` grid, `12/8` is its finest attainable
boundary. The seed-0 factorial's stored `4.0` used `16/first_epoch`; its control was
monotone and the canonical estimator also yields `16/4=4.0`, so the stored value is
retained as historical selection evidence but that factorial formula is withdrawn
for future use.

## Cost evidence and Pareto semantics

The original preregistration required training time, time-to-quality, latency, and
storage to be **reported** and the candidate to be Pareto-nondominated. The v1
summarizer accidentally strengthened raw training wall time into a claim gate.
Removing that gate restores the preregistration rather than making the claim easier.

Seed-1 raw wall time is `17,629.0` seconds for random and
`14,252.320465842` seconds for imprinted. Seed 1 did not persist the optimization
image count or loader length. Its retrospective `161` steps per epoch is therefore
descriptive only: the trainer's exact sampler formula
`ceil(optimization_images / 8 shards) // (128 / 8 local batch)` equals `161` for
every optimization-image count in `[20_601, 20_728]`, covering the discarded
historical estimate. It cannot gate a profiled-compute claim. The matched 16-epoch
step models at that retrospective count are
`13,641.277351625264` and `13,642.094341494143` seconds before initialization cost,
leaving non-step residuals of `3,987.7226483747363` seconds (`29.2%` of the
random step model) and `610.226124347857` seconds (`4.5%` of the imprinted
step model). The
asymmetry is about `6.5x` larger in the control and therefore flatters the candidate.
Raw wall time remains mandatory descriptive evidence but decides no claim.

For each arm with prospective initialization evidence, report both:

```text
fixed_epoch_profiled_compute_seconds =
    16 * optimizer_steps_per_epoch * step_wall_seconds
    + initialization_seconds

iso_quality_profiled_compute_seconds =
    first_quality_epoch * optimizer_steps_per_epoch * step_wall_seconds
    + initialization_seconds
```

`optimizer_steps_per_epoch` and `initialization_seconds` come from the run's frozen
initialization receipt; they are not literals inferred by the summary. Initialization
timing synchronizes CUDA immediately before and after the complete initializer. For
imprinting it includes loader construction/iteration, decode, transform, host-to-
device transfer, forward passes, device-to-host transfer, and class-mean reduction.
The architecture inference benchmark remains a separate deployment measurement and
is never substituted for initialization time.

Seed 1 has no recoverable initialization duration or epoch-0 classifier bytes. Its
summary row must therefore contain exact status
`"historical_initialization_receipt_unavailable"`, null fixed/iso compute totals,
and the raw wall time, profiler values, epoch estimator, memory, latency, and storage.
The rejected illustrative charge `25_882 * inference_latency` is withdrawn even from
quantitative chronology: its denominator was not persisted, and it excluded
preprocessing and transfer, so it was not a conservative time bound. It gates nothing.

These quantities are profiled compute proxies, not measurements of complete run wall
time: each multiplies a measured steady-state step latency by the registered step
count and adds the directly measured initialization duration. They omit the disclosed
non-step residuals. For seeds 2..6, iso-quality profiled-compute non-inferiority is
evaluated per seed. Fixed-epoch profiled compute and its signed imprinted-minus-random
overhead are always reported and may be higher; a higher fixed-epoch proxy cost cannot
be hidden behind the iso-quality result.

Peak GPU memory and checkpoint storage must be no greater for imprinted than random
within **all six seeds**, including the already observed seed 1. Deployment storage
must be exactly equal within all six seeds. Seed 1's immutable report already passes
these three resource gates (`87,167 <= 87,187` MiB and exact checkpoint/deployment
byte equality), but it remains part of the final conjunction. Architecture inference
latency is shared by the paired final backbone and is reported, not attributed to
initialization.

`fixed_epoch_pareto_nondominated` means the random epoch-16 point does not dominate
the imprinted epoch-16 point: random must not be both at least as good in quality and
no worse in every available fixed-budget registered cost: profiled compute when
prospectively available, peak GPU memory, checkpoint storage, and deployment storage.
Raw wall time remains descriptive and shared inference latency cannot distinguish the
arms. The predicate is recomputed seed by seed; a lower-quality imprinted point remains
nondominated when it improves at least one registered cost. Because the unchanged
quality gate requires a strictly positive mAP@R delta in every seed, non-domination is
still implied whenever the quality gate passes and carries no independent claim
evidence. Exact ties use conservative weak domination: when quality and every available
cost are equal, random weakly dominates imprinted. The signed per-seed
`fixed_epoch_profiled_compute_overhead_seconds` rows provide the explicit fixed-budget
cost disclosure instead.

`all_future_iso_quality_profiled_compute_noninferior` means the imprinted arm reaches
the random epoch-16 target with no greater profiled compute proxy in every future seed
2..6. `all_first_quality_epochs_noninferior` means the imprinted arm reaches that
target no later than random in every seed 1..6. The historical `1.5` threshold was the
seed-0 factorial's imprint-closure criterion; it is reported for context and is not a
six-seed confirmation threshold. `per_seed_resource_noninferior` means every seed
1..6 passes peak-memory and checkpoint non-inferiority plus exact deployment-storage
equality. The final
`claim_supported` is an explicit trajectory-frontier claim: the unchanged six-seed
quality gate passes, every seed passes first-quality epoch non-inferiority, every
prospectively measured seed 2..6 passes the iso-quality profiled-compute proxy gate,
and all six seeds pass the resource gate. Checkpoint non-inferiority and deployment
equality are architectural integrity guards for the shared backbone/classifier shape;
only peak memory provides live resource-efficiency evidence. The reported
fixed-epoch Pareto predicate is a derived consequence, not a separate claim pillar.
Seed 1 contributes quality, epoch-to-quality, and resource evidence but cannot
contribute to the prospective profiled-compute conjunction.

The A-B-B-A profiler kernel gate remains `0.1`; seed 1's maximum registered ratio is
`0.0004655412645827709`, so custom-kernel work remains closed.

## Forward-only initialization receipts

Before seed-2 training, the trainer adds an atomic no-clobber
`initialization-receipt.json`. Its exact ordered schema is:

```text
schema_version
seed
classifier_init
trainer_sha256
algorithm
classifier_tensor_sha256
classifier_shape
classifier_dtype
optimizer_steps_per_epoch
initialization_seconds
post_initialization_rng
```

`schema_version` is `unicom-classifier-initialization-v1`; `seed` is an exact int in
`2..6`; `classifier_init` is `random` or `imprinted`; `trainer_sha256` is lowercase
64-hex; `algorithm` is respectively
`torch-normal-std-0.01-rng-balanced` or
`normalized-class-means-norm-matched-rng-restored`; the classifier hash covers exact
contiguous CPU FP32 bytes in row-major order. The first dimension is derived from
`len(identity_holdout(authenticated_train_records, fraction=0.2, seed=0)[3])` before
model load and then cross-checked against the registered literal. The immutable query
evidence has exactly `797` held-out identities; independently, the live optimization
label mapping must have exactly `3200` contiguous rows, yielding registered
`[3200, 768]` in both arms and all five future seeds. This registered-count gate applies
only to the prospective seed-2..6 receipt path; other documented trainer modes derive
their row count without inheriting the confirmation-specific gate. The JSON
dtype value is the exact built-in string `"torch.float32"`; steps is a positive exact
int; seconds is a positive finite Python float. `post_initialization_rng` has exact ordered keys
`python_sha256`, `numpy_sha256`, `torch_cpu_sha256`, and
`torch_cuda_sha256_by_device`; the first three are lowercase 64-hex strings and the
last is an ordered nonempty list of lowercase 64-hex strings.

Receipt construction is observational: it cannot mutate classifier/model tensors,
RNG states, data order, model mode, or BatchNorm buffers. It is published after the
loader length is known and initialization is complete, but before optimizer
construction or the first batch. Random initialization consumes the same random-
initializer stream in both arms; imprinting saves and restores Python, NumPy, Torch
CPU, and all Torch CUDA states. Consequently, the random and imprinted receipts for
each future seed must have identical `post_initialization_rng` objects. A mismatch
invalidates the pair before evaluation.

On resume the receipt must already exist at the resumed run's exact
`<output-dir>/initialization-receipt.json`; a checkpoint from any other directory does
not relocate or waive that requirement. The registered initializer is rerun
deterministically before optimizer construction so its tensor hash and post-init RNG
can be reauthenticated, but it is not timed again. The expected 11-field object is
rebuilt with the existing receipt's validated `initialization_seconds` and must match
exactly; the receipt is never replaced. Checkpoint restore then overwrites that verified
initial tensor and restores the saved training state.

Future external training measurement receipts use exact schema
`unicom-training-measurement-v2`: the existing v1 ordered fields followed by
`optimizer_steps_per_epoch`, `initialization_seconds`,
`initialization_receipt_sha256`, and `post_initialization_rng_sha256`. The evaluator
loads the named initialization receipt, authenticates all fields and its exact bytes,
and requires the combined canonical RNG-object digest
`sha256(json.dumps(post_initialization_rng, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())`.
Both arms and all five future seeds must have one identical
`optimizer_steps_per_epoch`; this prevents receipt drift from changing the gating
iso-quality compute denominator. It continues to accept v1 only for immutable seed 1.
Future paired reports use
`unicom-ema-imprint-replication-pair-v2`, adding those four initialization fields to
each arm after `profile`; the evaluator cannot emit v1 for seeds 2..6. Pair-v2 keeps
the pair-v1 top-level order exactly. Its arm order is therefore the nine pair-v1 arm
keys followed by `optimizer_steps_per_epoch`, `initialization_seconds`,
`initialization_receipt_sha256`, and `post_initialization_rng_sha256`. The v2 summary
accepts exactly seed-1 pair-v1 followed by seed-2..6 pair-v2 and derives the explicit
historical row. The seed-1 CLI argument must be the exact repository path
`reports/generated/unicom_ema_imprint_replication_c83cd96_seed1.json`, a regular
non-symlink whose bytes equal SHA-256
`0cfb888fdbc0e409048943a8d6e47635e571dec5830b86260e0730d80ebf4ab8` and the Git
blob recorded by commit `5a08b95266c7d40d57ec7fd747999969147a04e6`; a copied
byte-identical alias is not authority. Direct summary construction and validation also
require the seed-1 semantic object to equal that authenticated payload exactly.
Cross-seed recipe comparison removes `seed`, `classifier_init`, and `trainer_sha256`
before equality: the first two are arm/replicate variables and the last is versioned
execution provenance that necessarily differs between immutable pair-v1 and prospective
pair-v2. The summary separately requires seeds 2 through 6 to share one exact
`trainer_sha256`; only immutable historical seed 1 may differ. Each report preserves its
exact per-arm trainer digest, and the paired evaluator requires both arms within a seed
to match the executing trainer bytes.

A fresh seed run is no-clobber: if its output directory already contains
`initialization-receipt.json`, launching again without the exact registered `--resume`
path is a structural failure. Operators must resume the same run or choose a new,
previously absent run directory; they must never delete or replace a valid receipt.

The summary schema becomes `unicom-ema-imprint-replication-summary-v2`. Its exact
top-level key order is:

```text
schema_version
training_seeds
selected_cell
selection_authority
reports
initialization_evidence
map_deltas
mean_map_delta
map_delta_sample_standard_deviation
student_t_critical_two_sided_95_df5
map_delta_paired_student_t_95_interval
exact_two_sided_sign_p_value
recall_at_1_deltas
recall_at_1_delta_guard
all_map_deltas_positive
nondegenerate_training_seed_variation
all_recall_at_1_deltas_above_guard
quality_claim_supported
first_quality_epochs
costs
fixed_epoch_pareto_nondominated
all_first_quality_epochs_noninferior
all_future_iso_quality_profiled_compute_noninferior
per_seed_resource_noninferior
claim_supported
```

Each `initialization_evidence` row has exact keys `seed`, `status`, `random_raw`,
`imprinted_raw`, `post_initialization_rng_equal`. Seed 1 uses status
`historical_initialization_receipt_unavailable`, two null arms, and a null equality.
Seeds 2..6 use status `prospective_authenticated`, equality `true`, and each arm has
exact keys `initialization_receipt_sha256`, `optimizer_steps_per_epoch`,
`initialization_seconds`, `post_initialization_rng_sha256`.

`costs` has exact ordered keys `training_seconds`, `first_quality_epochs`,
`fixed_epoch_profiled_compute_seconds`,
`fixed_epoch_profiled_compute_overhead_seconds`,
`iso_quality_profiled_compute_seconds`, `peak_gpu_mib`,
`inference_latency_protocol`, `inference_latency_ms_per_image`,
`checkpoint_storage_bytes`, `deployment_storage_bytes`,
`profile_fusible_non_backbone_fraction`, `kernel_profile_threshold`,
`kernel_eligible`, and `historical_cost_limitations`. Training, fixed/iso profiled
compute, peak, checkpoint, deployment, and profile rows use exact order `seed`,
`random_raw`, `imprinted_raw`. Latency rows use exact order `seed`,
`milliseconds_per_image`. `first_quality_epochs` rows use exact order `seed`,
`random_raw`, `imprinted_raw`, `speedup`. Fixed-epoch overhead rows use exact order
`seed`, `imprinted_minus_random`; seed 1 is null and seeds 2..6 are the difference of
the two profiled proxies. `inference_latency_protocol` is exactly
`{"warmup_repetitions": 10, "measured_repetitions": 50, "batch_size": 128}` in
that key order. `historical_cost_limitations` is exactly:

```json
[
  {
    "seed": 1,
    "status": "historical_initialization_receipt_unavailable"
  }
]
```

The four decision fields are recomputed, and
`claim_supported` is derived only from `quality_claim_supported`,
`all_first_quality_epochs_noninferior`,
`all_future_iso_quality_profiled_compute_noninferior`, and
`per_seed_resource_noninferior`. `fixed_epoch_pareto_nondominated` is reported but is
not a fourth independent conjunct.

## Decision and claim boundary

The Student-t interval generalizes over training seeds on the fixed holdout. The exact
sign-test p-value is the mathematical consequence of all six nonzero deltas sharing
one sign, not independent evidence. Query bootstrap remains within-run descriptive
evidence.

No official-protocol or global-SOTA claim is authorized by this summary. An exact
one-shot official In-Shop test readout requires a separate prospective protocol.
