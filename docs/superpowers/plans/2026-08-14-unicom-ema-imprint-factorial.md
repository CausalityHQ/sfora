# UniCOM EMA × Imprinted-Head Factorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run the prospectively frozen two-run UniCOM seed-0 factorial that tests step-EMA and class-mean classifier initialization for final quality and time-to-quality.

**Architecture:** Extend the existing trainer with two isolated mechanisms: an FP32 EMA state updated only after successful optimizer steps, and a deterministic class-mean initializer. Persist raw and EMA states together so resume is exact. Add a dedicated evaluator that independently recalibrates/evaluates every registered arm and emits one strict atomic decision report.

**Tech Stack:** Python 3.12, PyTorch 2.12, NumPy 2.5, pytest, Ruff, existing UniCOM/In-Shop loaders and hardened retrieval evaluator.

**Spec:** `docs/unicom_ema_imprint_factorial_2026-08-14.md`

## Global Constraints

- Preserve the official 16-epoch seed-0 training recipe, data order, objective, optimizer, schedule, batch size, FP32 mode, and epochs 4/8/12/16.
- EMA decay is exactly `0.999`; imprint row norm is exactly `0.01 * sqrt(768)`; neither is tunable.
- The random arm remains the existing seeded `normal_(std=0.01)` control.
- Evaluation uses the train-identity holdout only, independent full BatchNorm recalibration per arm, and hardened normalize-then-prefix retrieval with query chunks of 256.
- Use TDD for every production change and preserve RED/GREEN evidence.
- Publish outputs atomically without overwriting an existing destination.
- Do not start seed 1–6 confirmation unless the strict seed-0 report promotes one exact cell.

---

### Task 1: Successful-step FP32 EMA

**Files:**
- Modify: `scripts/train_unicom_inshop.py`
- Modify: `tests/test_train_unicom_inshop.py`

**Interfaces:**
- Produces: `StepEMA(backbone, classifier, decay=0.999)`, `StepEMA.update()`,
  `StepEMA.register_step_hook(optimizer)`, `StepEMA.state_dict()`, and
  `StepEMA.load_state_dict(...)`.

- [ ] **Step 1: Write failing EMA arithmetic and isolation tests**

Test initialization is an exact detached same-device FP32 copy, one update applies
`0.999 * old + 0.001 * current`, source mutation cannot mutate the shadow,
all buffers are excluded from averaging, and invalid decay/state shapes or keys
are rejected.

- [ ] **Step 2: Run the focused RED tests**

Run: `pytest -q tests/test_train_unicom_inshop.py -k 'step_ema'`

Expected: failure because `StepEMA` is absent.

- [ ] **Step 3: Implement the minimal EMA state object**

Keep ordered same-device FP32 tensors for every trainable backbone parameter and
the classifier. Update with explicit in-place
`mul_(0.999).add_(current, alpha=0.001)` after type/key/shape/device checks.
Materialization combines averaged parameters with every current raw buffer;
only checkpoint serialization copies the shadow to CPU.

- [ ] **Step 4: Write failing successful-step tests**

Exercise ordinary optimizer steps and a fake GradScaler that skips one step.
Assert the registered optimizer post-step hook updates EMA exactly when
`optimizer.step()` executes, not for batches, scheduler steps, scaler updates,
or skipped steps. Assert registration rejects a second live hook.

- [ ] **Step 5: Integrate update timing and make tests GREEN**

Register EMA through `optimizer.register_step_post_hook` before the first batch.
GradScaler bypasses `optimizer.step()` on overflow, so the hook does not run.
Do not read the loss scale or synchronize the device per step. Remove the hook
when training exits and persist the exact EMA update count.

- [ ] **Step 6: Run focused tests and lint**

Run: `pytest -q tests/test_train_unicom_inshop.py -k 'step_ema or training_epoch'`

Run: `ruff check scripts/train_unicom_inshop.py tests/test_train_unicom_inshop.py`

---

### Task 2: Deterministic class-mean initialization

**Files:**
- Modify: `scripts/train_unicom_inshop.py`
- Modify: `tests/test_train_unicom_inshop.py`

**Interfaces:**
- Produces: `imprinted_classifier_values(model, records, labels, transform, *, device, batch_size, workers) -> torch.Tensor` with shape `(len(labels), 768)`, CPU FP32.
- Consumes: optimization records and the existing official evaluation transform before training.

- [ ] **Step 1: Write failing formula/order tests**

Use a tiny deterministic model and non-grouped records. Independently compute
per-image FP32 normalization, dataset-order FP64 class sums, normalized means,
and the exact row norm. Assert label-map order, bytes, source-model state, BN
buffers, training mode, Python/NumPy/Torch CPU/Torch CUDA RNG states, and the
subsequent dedicated training-loader and mask streams are preserved.
Assert both run modes first execute the same seeded random initialization and
the imprinted mode only overwrites its values afterward.

- [ ] **Step 2: Write failing rejection tests**

Cover missing classes, duplicate/non-contiguous label indices, zero/nonfinite
embeddings, zero class means, wrong dimensionality, and output type/shape.

- [ ] **Step 3: Run the focused RED tests**

Run: `pytest -q tests/test_train_unicom_inshop.py -k 'imprinted_classifier'`

- [ ] **Step 4: Implement the deterministic initializer**

Build a non-shuffled evaluation loader, temporarily set `model.eval()`, restore
the original mode in `finally`, accumulate normalized embeddings on CPU in FP64
in input order, normalize/cast, and multiply by `0.01 * math.sqrt(768)`. Give
the loader a dedicated generator and restore all parent RNG states in `finally`.

- [ ] **Step 5: Add the frozen CLI choice**

Add `--classifier-init` with choices `random` and `imprinted`, default `random`.
Record the choice and fixed imprint algorithm in `training_protocol`; do not add
a norm flag.

- [ ] **Step 6: Run focused tests and lint**

Run: `pytest -q tests/test_train_unicom_inshop.py -k 'imprinted_classifier or cli_defaults'`

---

### Task 3: Exact EMA checkpoint and resume state

**Files:**
- Modify: `scripts/train_unicom_inshop.py`
- Modify: `tests/test_train_unicom_inshop.py`

**Interfaces:**
- Extends the training checkpoint with ordered key `ema`, whose exact nested
  state contains `decay`, `updates`, `backbone`, and `classifier`.
- `save_training_checkpoint(..., step_ema)` and `restore_training_checkpoint(..., step_ema)` round-trip the exact shadow and update count.

- [ ] **Step 1: Write failing checkpoint schema tests**

Assert checkpoints contain raw model/classifier plus EMA backbone/classifier and
the exact update count. Mutate every EMA key, tensor dtype/shape/value, decay,
and count; strict restore must reject each mutation.

- [ ] **Step 2: Write failing resume-equivalence test**

Train a tiny CPU model continuously for two epochs and through a save/restore
boundary. Assert exact raw parameters, classifier, EMA tensors, update count,
optimizer/scheduler/scaler state, RNG state, and history.

- [ ] **Step 3: Run RED**

Run: `pytest -q tests/test_train_unicom_inshop.py -k 'ema_checkpoint or ema_resume'`

- [ ] **Step 4: Implement strict save/restore**

Persist the EMA state only in the new protocol and require it on restore. Keep
atomic temporary creation/publication behavior unchanged. Reject resume when
the initialization choice or EMA configuration differs.

- [ ] **Step 5: Run the complete trainer test file**

Run: `pytest -q tests/test_train_unicom_inshop.py`

- [ ] **Step 6: Commit the trainer layer**

Run: `git add scripts/train_unicom_inshop.py tests/test_train_unicom_inshop.py`

Run: `git commit -m 'add UNICOM EMA imprint training arms'`

---

### Task 4: Factorial metrics, selection, and report validator

**Files:**
- Create: `scripts/evaluate_unicom_ema_imprint_factorial.py`
- Create: `tests/test_evaluate_unicom_ema_imprint_factorial.py`

**Interfaces:**
- Produces: `select_candidate(cells)`, `paired_map_bootstrap_interval(...)`, `time_to_quality(...)`, `factorial_gate(...)`, and `validate_factorial_report(report)`.
- Consumes four registered cells at epochs `(4, 8, 12, 16)` with paired query evidence and checkpoint hashes.

- [ ] **Step 1: Write failing pure decision tests**

Cover candidate tie order, exact promotion boundary, Recall@1 guard, strictly
positive bootstrap lower bound, EMA/imprint closure predicates, target reached
at 4/8/12/16, target not reached, and the instrument-reproduction stop.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_evaluate_unicom_ema_imprint_factorial.py -k 'select or gate or time_to_quality or bootstrap'`

- [ ] **Step 3: Implement pure calculations**

Use builtin finite floats, `math.fsum` where order matters, NumPy PCG64 seed 0,
10,000 paired query bootstrap replicates, and exact thresholds from the spec.

- [ ] **Step 4: Write strict recursive schema/mutation tests**

Start from one internally consistent report. Exhaustively remove/add/reorder
mapping keys, mutate fixed values and builtin types, inject NaN/infinities, and
make each recomputable metric, hash, delta, interval, closure flag, winner, or
decision inconsistent.

- [ ] **Step 5: Implement strict report validation**

Require exact top-level/nested order and types, exact four-cell/four-epoch order,
disjoint run checkpoint paths, SHA-256 syntax, paired evidence lengths, metrics
recomputed from evidence, gate recomputation, and no test-query/gallery fields.

- [ ] **Step 6: Run focused tests and lint**

Run: `pytest -q tests/test_evaluate_unicom_ema_imprint_factorial.py`

---

### Task 5: Hardened arm evaluation and atomic CLI

**Files:**
- Modify: `scripts/evaluate_unicom_ema_imprint_factorial.py`
- Modify: `tests/test_evaluate_unicom_ema_imprint_factorial.py`

**Interfaces:**
- CLI consumes `--random-run`, `--imprinted-run`, `--unicom-checkout`, `--initial-checkpoint`, `--dataset-root`, and `--output`.
- Reuses hardened score/evidence and BatchNorm recalibration behavior from the existing evaluators without changing their registered logic.

- [ ] **Step 1: Write failing checkpoint-binding tests**

Require each run to contain epochs 4/8/12/16, one training protocol within the
run, the expected init choice, EMA decay 0.999, exact EMA state keys, distinct
checkpoint bytes across runs, and matching data/objective/schedule fields.

- [ ] **Step 2: Write failing arm-order and recalibration tests**

With a tiny BN model, assert exact order
`random_raw, random_ema, imprinted_raw, imprinted_ema` within each epoch, one
fresh cumulative optimization-loader traversal per arm, no state leakage, and
normalize-then-prefix scores with query chunks of 256.

- [ ] **Step 3: Implement arm loading/evaluation**

Materialize raw or EMA floating state, preserve current non-floating buffers,
perform fresh BatchNorm recalibration, encode the fixed holdout, and record
metrics plus per-query top1/AP evidence immediately before releasing the model
state for the next arm.

- [ ] **Step 4: Write failing publication tests**

Cover existing destination, pre-existing temp, publication race, write/fsync/
link/directory-fsync failures, owned-temp cleanup, foreign-path preservation,
mode 0600, and strict JSON reload followed by a second validation.

- [ ] **Step 5: Implement one-shot CLI and atomic publication**

Authenticate inputs before loading CUDA state, build one report, validate in
memory, publish by exclusive same-directory temp plus no-replace link, strict
reload and validate, then return 0 for promoted and 1 for valid closed. Ordinary
structural exceptions return 2 without a partial destination.

- [ ] **Step 6: Run evaluator tests and static gates**

Run: `pytest -q tests/test_evaluate_unicom_ema_imprint_factorial.py`

Run: `ruff check scripts/evaluate_unicom_ema_imprint_factorial.py tests/test_evaluate_unicom_ema_imprint_factorial.py`

Run: `python -m py_compile scripts/evaluate_unicom_ema_imprint_factorial.py tests/test_evaluate_unicom_ema_imprint_factorial.py`

- [ ] **Step 7: Commit the evaluator layer**

Run: `git add scripts/evaluate_unicom_ema_imprint_factorial.py tests/test_evaluate_unicom_ema_imprint_factorial.py`

Run: `git commit -m 'add UNICOM EMA imprint factorial evaluator'`

---

### Task 6: Independent review and local assurance

**Files:**
- Modify only files required by confirmed review findings.

- [ ] **Step 1: Request adversarial cross-provider review**

Ask Claude/Opus to review the spec, plan, source, tests, checkpoint compatibility,
AMP skipped-step semantics, evaluator leakage, statistical gates, and whether
tests can pass while production violates the experiment. The consultation is
read-only and uses ordered fallback models `opus`, `gpt-5.6-sol`.

- [ ] **Step 2: Reproduce each confirmed finding with a failing test**

Do not change production before the focused RED identifies the defect.

- [ ] **Step 3: Apply minimal fixes and rerun focused tests**

Preserve separate commits for review fixes.

- [ ] **Step 4: Run affected and repository assurance once**

Run the two focused test files, Ruff, py_compile, and `git diff --check`, then
the repository's full pytest command once after coordinating the shared local
test lane. Do not overlap heavy suites.

---

### Task 7: Seed-0 factorial execution

**Files:**
- Create through program execution: two run directories and one factorial JSON.
- Commit: the validated compact report and a concise Markdown interpretation.

- [ ] **Step 1: Deploy the reviewed Git commit**

Create a Git bundle, copy it to a fresh detached GPU checkout, verify the commit
and clean status, then rsync only registered dataset/checkpoint inputs if absent.

- [ ] **Step 2: Run the random control with EMA**

Launch exactly one process. Capture command, exit, wall time, peak GPU memory,
checkpoint sizes, and logs. Do not launch the imprinted run until raw epoch 16
passes the instrument-reproduction tolerance under the hardened evaluator.

- [ ] **Step 3: Run the imprinted arm**

Use the identical recipe and seed, changing only `--classifier-init imprinted`.

- [ ] **Step 4: Evaluate and validate the four-cell report**

Run the one-shot evaluator, strict-load it offline, confirm no test split was
used, rsync it back, validate its SHA and predicates locally, and commit it.

- [ ] **Step 5: Follow the frozen decision**

If closed, document the exact failed predicates and advance to the next
evidence-based candidate. If promoted, freeze the winning cell and proceed to
Task 8 without changing constants.

---

### Task 8: Six-seed confirmation and Pareto report

**Files:**
- Create: `scripts/summarize_unicom_ema_imprint_replication.py`
- Create: `tests/test_summarize_unicom_ema_imprint_replication.py`
- Create: `reports/unicom_ema_imprint_replication_2026-08-14.md`

- [ ] **Step 1: TDD the six-seed summary**

Require seeds exactly 1–6, disjoint checkpoint digests, the frozen winning cell,
all six positive mAP deltas, nonzero sample SD, exact Student-t lower bound,
sign-test `p=0.03125`, Recall@1 guards, and complete cost fields.

- [ ] **Step 2: Run paired control/candidate training for seeds 1–6**

Schedule serially on the registered GPU unless independent GPUs are explicitly
available. Preserve every valid failure/result; do not extend or replace seeds.

- [ ] **Step 3: Measure cost**

Report training wall time, first registered epoch reaching the control endpoint,
peak GPU memory, inference latency with warmup/repetitions, checkpoint storage,
and final deployment storage. Profile step components; only consider a custom
kernel if the frozen 10% fusible non-backbone threshold is met.

- [ ] **Step 4: Produce and independently review the final report**

State the exact verified baseline, protocol limits, paired effects and intervals,
costs, all negative evidence, and whether the method is Pareto-superior. Make no
global-SOTA claim without exact official-protocol evidence. Resolve every
Critical/Important review finding before completion.
