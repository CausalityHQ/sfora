# Teacher-Anchored Neighborhood Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and execute this plan task-by-task. Preserve every focused RED before writing its corresponding implementation.

**Goal:** Build and evaluate a generic teacher-anchored neighborhood distillation method that emits one normalized 128-dimensional embedding for Sfora's symmetric-int8 retrieval path.

**Architecture:** A dataset-agnostic library module owns deterministic anchor/batch schedules, differentiable geometry losses, and diagnostics. Strict SOP-only trainer and evaluator scripts authenticate local images, snapshots, checkpoints, split identities, model modes, state transitions, and canonical receipts. The first DGX screen uses only official SOP training classes with class-disjoint fitting/validation rows; the official test partition remains inaccessible.

**Tech Stack:** Python 3.13.9, PyTorch 2.12.1+cu130, NumPy, UNICOM revision `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-08-teacher-anchored-neighborhood-distillation-design.md`

## Global constraints

- Keep labels, paths, SOP parsing, metric scoring, checkpoints, and receipts outside `src/sfora`.
- Fit PCA, ridge state, anchors, batches, and every training decision from fitting rows only.
- Never open or accept an official SOP test input in this campaign.
- Treat all validation measurements before epoch 10 as non-binding diagnostics.
- Emit complete merged encoder-plus-head state, canonical claim-ineligible receipts, and exact input/state digests.
- Run only one monitored DGX science process at a time; retain every terminal result and never retry a failed scientific arm under changed settings.

---

### Task 1: Generic configuration, anchor schedule, and structured batches

**Files:**
- Create: `src/sfora/teacher_anchored_distillation.py`
- Modify: `src/sfora/__init__.py`
- Create: `tests/test_teacher_anchored_distillation.py`

**Interfaces:**
- `TeacherAnchoredConfig`
- `teacher_anchor_schedule(teacher_codes, sample_ids, *, seed)`
- `teacher_neighbor_batches(teacher_codes, sample_ids, *, seed, epoch)`

- [ ] **Step 1: Write strict configuration and anchor REDs**

  Mutation-lock concrete types and the exact dimensions, temperatures, loss weights,
  anchor group widths, seed derivation, self exclusion, teacher-rank boundaries,
  without-replacement behavior, natural numeric/string immutable-ID tie breaks,
  separate digest serialization, and input immutability.

- [ ] **Step 2: Preserve focused RED**

  Run: `.venv/bin/pytest -q tests/test_teacher_anchored_distillation.py -k 'config or anchor'`

  Expected: import failure for only the absent module or symbols.

- [ ] **Step 3: Implement anchors with bounded blocked similarity scans**

  Do not allocate the full fitting-row similarity matrix. Rank each query in fixed
  blocks, retain only the required first 512 rows, and derive the middle/uniform
  samples from SHA-256-domain-separated PCG64 streams. Return immutable int64 row
  indexes and an exact schedule digest.

- [ ] **Step 4: Write structured-batch REDs**

  Cover exact 128-seed plus 128-partner order, current-batch-only exclusions,
  cross-batch reuse, collision repair, insufficient partner inventory, dropped
  tail rows, per-epoch repeated-identity counts, seed/epoch sensitivity, and one
  authenticated first-256 neighbor ranking reused across every epoch and arm.
  Build and seal all ten schedules once per seed before any arm starts and
  mutation-lock a 15-minute real-width construction preflight bound.

- [ ] **Step 5: Implement batches and run focused GREEN**

  Run the complete new library test file. Require deterministic byte-identical
  schedules and no dependency on `scripts` or SOP types.

### Task 2: Differentiable neighborhood objective and stability diagnostics

**Files:**
- Modify: `src/sfora/teacher_anchored_distillation.py`
- Modify: `tests/test_teacher_anchored_distillation.py`

**Interfaces:**
- `teacher_anchored_loss(student, teacher, anchors, adapted_features, original_features, config)`
- `embedding_geometry_diagnostics(codes)`

- [ ] **Step 1: Write scalar-reference REDs for all five terms**

  Hand-derive dual-temperature asymmetric anchor KL, symmetric batch KL, point
  cosine loss, pairwise feature-drift loss, and covariance/variance guard. Require
  every term and the registered weighted sum to match the scalar reference.

- [ ] **Step 2: Mutation-lock finite behavior and gradients**

  Cover constant codes, zero variance, zero norms, nonfinite tensors, invalid
  anchors, shape/device/dtype drift, detached teacher/original inputs, and finite
  nonzero gradients for the student/head/adapted-feature paths.

- [ ] **Step 3: Implement the minimal objective and diagnostics**

  Run reductions in float32 outside autocast. Return named component scalars plus
  total loss; never accept labels or dataset records.

- [ ] **Step 4: Prove the free-code optimization fixture**

  Optimize free student codes against fixed teacher codes and require convergence
  toward the teacher solution without collapse. Treat failure as implementation
  error, not scientific evidence.

### Task 3: Deterministic model-mode, initialization, and optimizer authority

**Files:**
- Create: `scripts/train_sop_teacher_anchored_distillation.py`
- Create: `tests/test_train_sop_teacher_anchored_distillation.py`

**Interfaces:**
- strict CLI and authenticated local input loaders
- `initialize_teacher_anchored_student`
- `configure_teacher_anchored_epoch`
- `build_teacher_anchored_optimizer`

- [ ] **Step 1: Write CLI/capability and input-authority REDs**

  Require absolute local paths, exact SHA-256/revisions, seed 17/1729/65537,
  registered arm names, no-clobber output, and `--execute-teacher-anchored`.
  Reject official-test, network, URL, S3, class-name, sweep, resume, and arbitrary
  hyperparameter flags.

- [ ] **Step 2: Write exact initializer REDs**

  Reuse `fit_teacher_guided_projection(..., dimensions=128, penalty=1e-6)` on
  fitting rows only after the exact float64-norm/float32-output unit normalization
  used by the sealed ceiling probe. Require the sealed teacher-PCA state digest,
  a 768-to-128 affine head, snapshot-feature receipt,
  exact registered-CUDA replay in fixed 256-row chunks, portable CPU comparison
  of moved float32 outputs to the float64 map within `1e-5` max error and
  `1-1e-7` cosine, and the separately named live-encoder step-zero pass.
  Mutation-lock the teacher-PCA digest as mean then components, each encoded by
  little-endian `u32 ndim`, little-endian `u64` shape entries, and contiguous
  little-endian float32 bytes exactly as the sealed ceiling probe.

- [ ] **Step 3: Write model-mode and frozen-state REDs**

  On a faithful synthetic UNICOM-shaped graph, require every BatchNorm and real
  DropPath to remain in eval mode after every transition. Epoch one trains only
  the head. Epochs two-ten train exactly blocks 10/11, final norm, and head. Hash
  all frozen parameters and buffers before/after; reject inventory drift.
  Also replay registered image paths against authenticated snapshot codes and
  compare pristine single-image versus registered-batch outputs at `0.002`
  maximum absolute error and cosine at least `1-1e-5`.

- [ ] **Step 4: Write precision and optimizer REDs**

  Require no outer autocast, internal UNICOM autocast compatibility, float32
  head/norm/loss, disabled TF32, highest float32 matmul precision, pinned math
  SDPA, deterministic algorithms, exact two-stage AdamW state, LR formula,
  exclusions, clipping, GradScaler, and fail-closed skipped updates.

- [ ] **Step 5: Implement only the tested authority boundary**

  Keep data/model construction injectable so CPU fixtures exercise all decisions.
  Do not begin a real training loop until all focused tests pass.

### Task 4: Strict training loop, controls, stops, and artifacts

**Files:**
- Modify: `scripts/train_sop_teacher_anchored_distillation.py`
- Modify: `tests/test_train_sop_teacher_anchored_distillation.py`
- Create: `scripts/run_sop_teacher_anchored_panel.py`
- Create: `tests/test_run_sop_teacher_anchored_panel.py`

- [ ] **Step 1: Write synthetic end-to-end training RED**

  Exercise step zero, epoch-one head-only training, epoch-two optimizer reset,
  epochs two-ten final-block adaptation, a ten-epoch complete-objective head-only
  control whose encoder stays byte-identical, identical schedules across all five
  training arms, fitting-only stop rules, and epoch-10-only candidacy.
  Load the probe scorer with `importlib.util.spec_from_file_location` after
  inserting the repository scripts directory on `sys.path`; import the int8
  packer directly from its library module. Exercise this exact scorer in the
  epoch-one fitting stop and every per-epoch diagnostic.

- [ ] **Step 2: Write fitting diagnostic and bootstrap replay REDs**

  Select the first 512 fitting classes by the registered class hash and retain all
  their rows, using the exact little-endian seed/class encoding and tie rule.
  Test the estimator and absent/wrong authority rejection locally with a synthetic
  three-split fixture and a hand-computed lower bound. Keep the exact historical
  `ridge-source-teacher-full` minus `source-full` replay as a mandatory DGX
  preflight using the authenticated little-endian int64 label member and receipt
  row indexes. Reject singleton-producing row subsampling, PCG64 substitution,
  or interpolation drift.

- [ ] **Step 3: Implement canonical arm receipts and merged checkpoints**

  Bind inputs, split/PCA/ridge state, anchors, batches, modes, trainable/frozen
  inventories, optimizer schedule, attempted/successful updates, diagnostics,
  stop/endpoint, and full state-dict digest. Publish atomically with no overwrite.
  Append canonical authenticated progress JSON lines on every stage transition
  and successful optimizer update. Bind launch-receipt digest, strictly monotone
  sequence, arm/epoch/update, and previous-line SHA-256; the launcher resets its
  timer only on exact chain continuations. Mutation-test duplicate sequence,
  broken chain, foreign launch, truncated tail, and valid progress resets.
  Implement the process-group watchdog as a separate launcher with pure,
  mutation-tested stop classification and canonical receipt helpers.

- [ ] **Step 4: Run focused trainer GREEN and static checks**

  Run the trainer, launcher, and library test files, Ruff, strict mypy for the new
  production module and scripts, pycompile, and `git diff --check`.

### Task 5: Packed evaluator, causal gates, and serving reconstruction

**Files:**
- Create: `scripts/evaluate_sop_teacher_anchored_distillation.py`
- Create: `tests/test_evaluate_sop_teacher_anchored_distillation.py`

- [ ] **Step 1: Write scorer/codec/evidence REDs**

  Require `pack_int8_unit_embeddings` imported from its library module and
  `score_symmetric` loaded from `scripts/probe_sop_relational_linear.py` with the
  tested `importlib.util.spec_from_file_location` pattern after inserting the
  repository scripts directory on `sys.path`, exact candidate width,
  self exclusion, source-order ties, per-query AP/R@1, float/int8 arms, and no
  student-to-teacher diagnostic leakage into primary gates.

- [ ] **Step 2: Write advancement and causal-classification REDs**

  Mutation-lock every absolute/relative gate, the complete-versus-base paired
  historical bootstrap, the binding positive lower bound for complete versus
  live step zero, missing/stopped comparator handling, and exhaustive
  `implementation-error`, `stopped`, `inconclusive`, `generic-anchored-adaptation`,
  `quality-rejected`, and `stability-warranted` outcomes.

- [ ] **Step 3: Write serving reconstruction REDs**

  Load the complete merged state into a fresh eval model, reapply BatchNorm and
  DropPath mode authority, reproduce registered batches, enforce pristine/final
  batch-shape bounds, emit normalized 128D codes, and prove the serving artifact
  contains no teacher, anchors, labels, or PCA fitting rows.

- [ ] **Step 4: Implement evaluator and run focused GREEN**

  Emit one canonical newline-terminated claim-ineligible receipt and no official
  SOP test artifact.

### Task 6: Repository assurance and reviewed delivery

**Files:**
- Modify only files introduced by Tasks 1-5 and the public export list.

- [ ] **Step 1: Run focused tests serially**

  Run the four new test files plus `tests/test_representation_ceiling.py` and
  `tests/test_probe_sop_relational_linear.py`.

- [ ] **Step 2: Run static assurance**

  Run `.venv/bin/ruff check src scripts tests`, strict mypy on the new production
  files, pycompile on all three scripts, and `git diff --check`.

- [ ] **Step 3: Obtain independent cold Astra and Fable code review**

  Give each reviewer the exact committed spec/plan, diff, focused terminal
  evidence, and no proposed fixes. Repair only verified Critical/Important issues
  using focused RED/GREEN cycles.

- [ ] **Step 4: Run the full repository gate once**

  Run `.venv/bin/pytest -q`. After any failure, repair and rerun the failing layer
  first, then execute one final full gate.

- [ ] **Step 5: Commit, push, and verify authority**

  Commit with configured operator identity and no attribution trailers. Push
  `HEAD:master`, then require local HEAD, `origin/master`, and remote
  `refs/heads/master` equality with a clean worktree.

### Task 7: One monitored DGX seed-17 causal screen

**Files:**
- Add after authentication: one canonical result under `docs/evidence/teacher_anchored_distillation/`

- [ ] **Step 1: Preflight exact DGX inputs and environment**

  Authenticate the Sfora commit, SOP image tree, paired B16/L14 train snapshots,
  both UNICOM checkpoints, checkout revision, split rows, output absence, TF32/
  SDPA/determinism controls, disk, memory pressure, and GPU health.
  Require Python 3.13.9 and Torch 2.12.1+cu130. Run one non-scientific 256-row
  synthetic forward/backward with the real epoch-two trainable inventory,
  record summed process-group `/proc/<pid>/status` `VmRSS` and peak CUDA
  allocation, and require RSS below 16 GiB. Bind `torch.get_num_threads()`, the
  Torch parallel backend, and exact `timm` version. Authenticate both snapshot
  `train_embeddings` digests and recompute teacher-PCA inside this preflight.
  Time real-image updates and validation/fitting-probe encodes, project the
  complete panel conservatively below 12 hours, and keep the 18-hour hard cap.
  Reproduce the historical pooled bootstrap lower bound exactly from the two
  authenticated snapshot label members and sealed ceiling receipt before the
  trainer receives any input.

- [ ] **Step 1a: Verify the watchdog with no scientific model**

  Mutation-lock a 30-second poll, 18-hour wall cap, 32-GiB summed process-group
  RSS limit, PSI full-avg10 immediate `0.79` and sustained `0.50` for three
  samples, 2-GiB swap-growth cap, 15-minute authenticated-progress timeout,
  group TERM/30-second KILL, and outcome-blind canonical stop receipt.

- [ ] **Step 2: Execute one original seed-17 causal panel**

  Run the ten-epoch head-only, base, anchor, symmetric, and complete arms with
  identical schedules. Prove the head-only encoder remains byte-identical.
  Enforce registered wall/RSS/PSI/swap/progress stops, retain every terminal arm,
  and never launch a duplicate or altered retry. An authority or quality terminal
  is final. Any external resource stop permits one unchanged relaunch of only
  arms lacking epoch-10 results after the recorded cause is repaired. Discard,
  never resume, the partial arm and cross-bind both receipts; a second resource
  stop is terminal and requires a new spec.

- [ ] **Step 3: Authenticate and interpret the epoch-10 result**

  Recompute canonical bytes, input/model hashes, metrics, bootstrap, gates, and
  causal outcome. A negative result remains evidence; do not inspect official test
  rows or tune settings from validation.

- [ ] **Step 4: Run fixed within-population stability only after advancement**

  Run complete and base on seeds 1729 and 65537 with bootstrap generator seed 17.
  Both seeds must pass every gate against their own source, teacher-PCA, live
  step-zero, and base controls; any failure or missing base is unstable with no
  partial stability claim. Each stability launch has the same explicit 18-hour
  wall cap and registered pressure/progress stops as seed 17.

- [ ] **Step 5: Commit the result and follow only its authorized branch**

  Authenticate and commit the already completed stability results; never relaunch
  a completed arm. If stability is not warranted, classify the mechanism failure
  and return to a new train-only design, not an adaptive retry. No publication or
  SOTA claim is authorized by SOP alone.

### Task 8: Cross-dataset qualification after a successful SOP screen

- [ ] Freeze the unchanged generic recipe on at least two additional retrieval datasets.
- [ ] Use fresh, dataset-appropriate training/validation partitions and untouched final tests.
- [ ] Re-measure encoder batch-one latency, batch throughput, symmetric-int8 retrieval, model bytes, and peak memory on registered hardware.
- [ ] Require consistent quality gains and causal contrast before describing the method as broadly effective or publication-ready.
