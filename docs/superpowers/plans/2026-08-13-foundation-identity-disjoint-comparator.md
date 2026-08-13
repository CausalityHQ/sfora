# Foundation Identity-Disjoint Comparator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one final-state Proxy Anchor comparator without held-out identity exposure, bind it prospectively, and use it for one valid train-only In-Shop F1 decision.

**Architecture:** Add an identity-disjoint comparator-training surface beside the existing foundation screen. The training process consumes the exact outer optimization role and atomically publishes a checkpoint plus strict receipt; a second authority commit binds the observed checkpoint before the existing screen may load it. Extend the screen with a non-deciding contaminated-control role and explicit split-power/boundary outcomes.

**Tech Stack:** Python 3.12, PyTorch 2.12, NumPy 2.5, Typer, pytest, Pydantic, Git, JSON/SHA-256 authorities, GB10 CUDA.

**Spec:** `docs/superpowers/specs/2026-08-13-foundation-identity-disjoint-comparator-amendment.md`

## Global Constraints

- No official In-Shop query/gallery or SOP official split may be loaded by comparator training or train-only F1.
- Comparator training uses exact recipe `proxy_anchor.inshop.official-51db570`, outer fraction `0.2`, outer seed `0`, training seed `2`, and `60` epochs.
- Persist final training state only; no held-out checkpoint selection, early stopping, or outcome-based retry.
- Require `CUBLAS_WORKSPACE_CONFIG=:4096:8` before the first CUDA call and exact deterministic algorithms.
- Training checkpoint and receipt publication are no-clobber, atomic, strict-reloaded, and one-shot.
- Training ceiling is 2.5 GB10 wall-clock hours; complete repair plus F0/F1 ceiling is 4.0 hours.
- Existing official-test capability remains sealed unless a later reviewed addendum authorizes it.

---

### Task 1: Comparator request, receipt, and strict schema

**Files:**
- Modify: `src/sfora/foundation_pareto.py`
- Modify: `tests/test_foundation_pareto.py`

**Interfaces:**
- Produces: `IdentityDisjointComparatorRequest`, `validate_identity_disjoint_comparator_receipt(value: object, *, request: IdentityDisjointComparatorRequest) -> dict[str, object]`, and `identity_disjoint_role_digests(split: RecipeSelectionSplit) -> dict[str, object]`.
- Consumes: `class_disjoint_recipe_selection_split`, `ImageExample`, existing strict JSON/hash helpers.

- [ ] **Step 1: Write failing split-binding and receipt-schema tests**

Add tests that construct noncontiguous labels and assert exact optimization/query/gallery order, counts, ordered example-ID SHA-256 values, disjoint label sets, outer `split_sha256`, and exact recursive receipt keys/types. Generate isolated mutations for every key removal/addition/order/type, every digest, every count, label overlap, wrong seed/fraction/epochs, non-final selection, official-read flag, NaN, and infinity.

- [ ] **Step 2: Run the focused RED selector**

Run:
`pytest -q tests/test_foundation_pareto.py -k 'identity_disjoint_role or comparator_receipt_schema'`

Expected: failures for missing request/digest/validator interfaces.

- [ ] **Step 3: Implement immutable request and strict validator**

Freeze request fields in this order: `schema_version`, `dataset`, `dataset_root`, `source_commit`, `recipe_id`, `recipe_digest`, `outer_seed`, `outer_fraction`, `training_seed`, `epochs`, `checkpoint_path`, `receipt_path`, `pretrained_backbone_path`, `wall_clock_ceiling_seconds`. Require values `foundation-identity-disjoint-comparator-request-v1`, `inshop`, outer seed `0`, fraction `0.2`, training seed `2`, epochs `60`, ceiling `9000`, and absolute normalized paths.

Freeze receipt top-level order: `schema_version`, `status`, `request`, `source`, `recipe`, `split`, `training`, `environment`, `checkpoint`, `diagnostic`, `official_test`, `process`. Require complete builtin types, finite values, exact hashes/relations, `artifact_selection=final_training_state`, `official_test.consumed=false`, and status `VALID` only when every predicate is true.

- [ ] **Step 4: Run GREEN and mutation coverage**

Run the Step-2 selector. Expected: all pass and the generated mutation set reports no accepted mutation.

- [ ] **Step 5: Commit Task 1**

Stage only `src/sfora/foundation_pareto.py` and `tests/test_foundation_pareto.py`; commit `add foundation comparator receipt schema`.

### Task 2: One-shot identity-disjoint training runner

**Files:**
- Modify: `src/sfora/foundation_pareto.py`
- Modify: `tests/test_foundation_pareto.py`
- Modify: `src/sfora/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `run_identity_disjoint_comparator_training(request: IdentityDisjointComparatorRequest) -> Path` and CLI command `sfora foundation-comparator-train`.
- Consumes: Task-1 request/validator, `reference_recipe`, `config_for_recipe`, `run_image_end_to_end_benchmark`, existing no-clobber JSON publisher.

- [ ] **Step 1: Write failing runner tests**

Assert that the runner loads only In-Shop `train`, constructs the exact outer split once, passes only `split.optimization` to training, passes `split.query/gallery` only as final diagnostics, and resolves a config with checkpoint selection interval `0`, periodic test evaluation `0`, seed `2`, 60 recomputed epochs, exact reference hyperparameters, a PID-owned temporary checkpoint path, and no official loader reachability.

- [ ] **Step 2: Write failing atomic/failure tests**

Cover successful hard-link publication, existing checkpoint/receipt, foreign temporary collision, race at checkpoint link, race at receipt link, training exception, checkpoint validation failure, receipt validation failure, strict-reload failure, and cleanup after every publication phase. Assert unchanged foreign bytes and no owned temp after each failure.

- [ ] **Step 3: Run runner RED**

Run:
`pytest -q tests/test_foundation_pareto.py tests/test_cli.py -k 'foundation_comparator_train or identity_disjoint_training or comparator_publication'`

Expected: failures for missing runner/CLI.

- [ ] **Step 4: Implement minimal one-shot runner**

Authenticate source commit and recipe before importing Torch-dependent training code. Build the exact split, derive role digests, resolve the reference config with only the amendment-authorized changes, and call `run_image_end_to_end_benchmark` once. Write the model to a same-directory exclusive PID temp; require the saved checkpoint root/architecture/config/proxy tensors/final step/final-state marker; hard-link checkpoint without replacement; then build, validate, atomically publish, strict-reload, and revalidate the receipt. Fsync the directory after publication and cleanup.

- [ ] **Step 5: Implement exact CLI and exit contract**

The CLI accepts only absolute dataset/checkpoint/receipt/backbone paths plus the frozen scalar arguments. Structural failure returns `2`; a valid receipt returns `0`; BaseException is not swallowed. Require isolated invocation through the ledger's exact interpreter and leave no output on pre-authority failure.

- [ ] **Step 6: Run GREEN**

Run the Step-3 selector. Expected: all pass.

- [ ] **Step 7: Commit Task 2**

Stage the four files; commit `add identity-disjoint comparator training`.

### Task 3: Contaminated control and valid F1 decisions

**Files:**
- Modify: `src/sfora/foundation_pareto.py`
- Modify: `tests/test_foundation_pareto.py`

**Interfaces:**
- Produces: `FoundationScreenArmSpec.role` including `contaminated_control`; report outcomes `INVALID_SPLIT_POWER` and `BOUNDARY_REPLICATION_REQUIRED`.
- Consumes: existing `decide_f1`, report validator, registered/execution arm order.

- [ ] **Step 1: Write failing authority/order tests**

Freeze exact roles `[candidate, comparator, contaminated_control]`, require one of each, require local comparator/control, and assert execution order `[comparator, contaminated_control, candidate]` while persisted rows remain registered order.

- [ ] **Step 2: Write failing decision-isolation tests**

Mutate the control probe from 0 to 100 points, its profile, fixture, cache, and availability; assert the candidate/comparator gap and ordinary decision remain byte-identical. Assert control failure cannot rescue or close F1, but structural control-row contradictions are rejected.

- [ ] **Step 3: Write failing split-power and boundary tests**

Assert comparator `99.499999999` uses the ordinary gate, `99.5` produces `INVALID_SPLIT_POWER`, and candidate-minus-comparator gaps `-1.500000001`, `-1.5`, `-0.5`, `-0.499999999` route respectively to ordinary, boundary, boundary, ordinary outcomes. Boundary and invalid outcomes must prohibit official-read code paths.

- [ ] **Step 4: Run RED**

Run:
`pytest -q tests/test_foundation_pareto.py -k 'contaminated_control or split_power or boundary_replication'`

Expected: role/schema/decision failures.

- [ ] **Step 5: Implement minimal role and outcomes**

Extend exact schemas without changing candidate/comparator arithmetic. Run comparator fully before other arms; run control descriptively; derive split-power before F1 and boundary after finite ordinary gap calculation. Require no official rows for every non-CONTINUE outcome.

- [ ] **Step 6: Run GREEN and affected suite**

Run the Step-4 selector, then:
`pytest -q tests/test_foundation_pareto.py tests/test_image_benchmark.py tests/test_cli.py`

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

Stage source/test only; commit `gate foundation transfer on disjoint comparator`.

### Task 4: Source assurance and independent review

**Files:**
- Modify only if a confirmed review finding requires it: the four Task-2 files.

**Interfaces:**
- Produces: one reviewed source commit `S_D` with exact file SHA-256 values.

- [ ] **Step 1: Run source assurance**

Run affected tests, Ruff, format check, mypy on modified source, py_compile, and `git diff --check`. Confirm no docs authority or generated result is staged with source.

- [ ] **Step 2: Request cross-provider review**

Use explicit models `['opus','gpt-5.6-sol']`. Ask for adversarial checks of identity exposure, checkpoint selection, training-set hash binding, atomic publication, official-read reachability, control isolation, boundary arithmetic, and actual runner compatibility.

- [ ] **Step 3: TDD each confirmed finding**

For each finding, add a focused failing test, verify RED, implement the minimal fix, verify GREEN, rerun affected assurance, and commit review fixes separately.

- [ ] **Step 4: Freeze `S_D`**

Record exact HEAD and SHA-256 for all changed source/test files. Require clean tracked status except the pre-existing plan/ledger edits.

### Task 5: Prospective training ledger

**Files:**
- Create: `docs/foundation_identity_disjoint_comparator_training_ledger.json`
- Modify: `docs/superpowers/plans/2026-08-12-foundation-f0-f1.md`
- Modify: `docs/foundation_f0_f1_execution_ledger.json`

**Interfaces:**
- Produces: reviewed training-authority commit `H_D` binding `S_D`, exact DGX paths, exact command, and absent outputs.

- [ ] **Step 1: Freeze exact ledger**

Bind source commit/file hashes, amendment commit/hash, In-Shop dataset root, BN-Inception backbone path/hash, output checkpoint/receipt paths keyed by `S_D`, Python environment, the exact isolated command beginning with the registered interpreter followed by `-I -B -m sfora.cli foundation-comparator-train` and every literal argument, CUBLAS value, CUDA device 0, 9000-second ceiling, preflight predicates, single-attempt rule, receipt schema, and stop conditions. Use only absolute target-host authority paths.

- [ ] **Step 2: Update the parent plan/ledger chronology**

Preserve the failed old-comparator review and state that it authorized no execution. Replace the obsolete two-dataset conjunction with the amendment's In-Shop-only decision. Keep the already-added absolute authority paths, receipt-root mode check, and per-report source-commit check.

- [ ] **Step 3: Validate and commit docs only**

Strict-parse JSON, scan placeholders, run `git diff --check`, stage only the three docs, and commit `bind disjoint comparator training authority`.

- [ ] **Step 4: Independently review `H_D`**

Require exact source/ancestry/path/hash/command/budget/output-absence bindings and no Critical/Important finding before DGX transfer.

### Task 6: One comparator training process

**Files:**
- Create remotely: frozen checkpoint and receipt at exact ledger paths.
- Create after validation: `docs/foundation_identity_disjoint_comparator_seed2_receipt.json` containing exact receipt bytes.

**Interfaces:**
- Produces: immutable checkpoint SHA/config SHA and committed receipt commit `R_D`.

- [ ] **Step 1: Build and transfer a fresh Git bundle**

Bundle through `H_D`, verify bundle SHA, create a fresh detached checkout on the GB10 host, authenticate amendment/source/ledger/worktree blobs, environment, dataset/backbone, clean status, GPU/process absence, and exact output/temp absence.

- [ ] **Step 2: Launch exactly one process**

Run the exact ledger command once, retaining the original PID/session. Monitor wall clock, process identity, GPU memory/utilization, and output absence without starting a second process. Terminate at 9000 seconds and mark `INVALID_BUDGET`.

- [ ] **Step 3: Strict offline validation**

After exit, require code `0`, checkpoint and receipt regular non-symlink files, exact modes, no temp, strict production receipt validation, independent split/hash/config/final-state checks, no official capability use, and observed runtime within ceiling.

- [ ] **Step 4: Commit immutable receipt only**

Copy exact receipt bytes opaquely, verify SHA and validator result, force-add only its registered ignored path if needed, and commit `record disjoint comparator seed2 receipt`. Do not commit the large checkpoint.

### Task 7: Freeze the observed checkpoint into F1 authority

**Files:**
- Modify: `docs/foundation_model_specs.json`
- Modify: `docs/foundation_native_fixtures.json`
- Modify: `docs/foundation_metric_tolerances.json`
- Modify: `docs/foundation_published_metric_register.json`
- Modify: `docs/foundation_test_read_register.json`
- Create: `docs/foundation_native_inputs/inshop-pa-bninception-disjoint-seed2__embedding_cosine.json`
- Create: `docs/foundation_native_sources/inshop-pa-bninception-disjoint-seed2__embedding_cosine.py`
- Modify: `tests/test_foundation_pareto.py`

**Interfaces:**
- Produces: reviewed F1 handoff `H_F1` with three exact arms and observed comparator checkpoint/config hashes.

- [ ] **Step 1: Generate the new comparator fixture once**

From the authenticated checkpoint, encode only `assets/sfora-logo.png`, persist the exact FP32 embedding, and bind input/source/checkpoint/config hashes. Mark the independent native cross-check unavailable with a nonempty reason; require repository output exact equality with tolerance `0.0`.

- [ ] **Step 2: Update the ordered train-only authorities and keep official capability empty**

Freeze candidate, disjoint comparator, and contaminated control in exact order across the model, fixture, and tolerance authorities. The disjoint comparator paths/hashes come only from the validated receipt; the contaminated control retains prior bytes and is marked descriptive/non-deciding. Refreeze the published-metric register with exact `records=[]` and the test-read register with exact `records=[]` while preserving its registered receipt root. Those two empty registers are intentionally not three-arm authorities: the contaminated control may never receive official capability, and this amendment authorizes no candidate or comparator official read either. Only a later separately reviewed official-read addendum may populate them, and that addendum may bind candidate and disjoint comparator only.

Update `test_repository_fidelity_authorities_are_frozen_and_complete` and its
mutation coverage prospectively: model/fixture/tolerance coverage remains the
exact three-arm product, while published-metric and test-read coverage is
deliberately decoupled from registered arms and must equal exact empty tuples at
this handoff. Do not weaken the register loaders or generic nonempty-register
cross-product checks. Add a focused assertion that any later populated official
register can contain only the candidate and disjoint comparator, never the
contaminated control.

- [ ] **Step 3: Run authority tests and full affected assurance**

Run exact authority loaders/mutation tests, the three affected test files, Ruff,
mypy, py_compile, JSON parse, and diff check. Require the two checkpoint-producing
production hashes, `src/sfora/foundation_pareto.py` and `src/sfora/cli.py`, still
match `S_D`. The prospectively authorized
`tests/test_foundation_pareto.py` authority-coverage edit advances its test hash
at `H_F1`; it does not refreeze or alter either production source hash.

- [ ] **Step 4: Commit authority only and review**

Commit `refreeze foundation F1 disjoint comparator`. Independent review must authenticate receipt ancestry, checkpoint/config/fixture relations, exact arm order/roles, official capability remaining sealed, and no Critical/Important finding.

### Task 8: One train-only In-Shop F1 screen

**Files:**
- Create remotely: exact train-only report and caches named by `H_F1`.
- Create after validation: one immutable report record if repository policy requires it.

**Interfaces:**
- Produces: authenticated `CONTINUE`, `CLOSE_FOUNDATION_TRANSFER`, `INVALID_SPLIT_POWER`, or `BOUNDARY_REPLICATION_REQUIRED` result.

- [ ] **Step 1: Fresh detached preflight**

Authenticate `H_F1`, all authorities/source/worktree blobs, checkpoint and backbone SHA values, clean status, environment/CUBLAS/GPU, empty cache/output paths, and absence of official receipts/processes.

- [ ] **Step 2: Launch exactly one train-only screen**

Run In-Shop only without `--allow-registered-test-read`. Monitor the original process; do not launch SOP or a second seed.

- [ ] **Step 3: Validate before interpretation**

Strict-reload the report and independently recompute arm order, cache/fixture/probe bindings, control isolation, candidate/comparator gap, cost predicates, split-power threshold, boundary interval, overall status, source commit, and absence of official rows/receipts.

- [ ] **Step 4: Route without discretion**

- `CONTINUE`: commit the report/hash and write a separate official-read addendum; do not read official pixels yet.
- `BOUNDARY_REPLICATION_REQUIRED`: stop and prospectively authorize only comparator seed 0.
- `INVALID_SPLIT_POWER`: close this train-only gate and redesign before any official read.
- `CLOSE_FOUNDATION_TRANSFER`: close the frozen-foundation lane and continue the broader research goal with the next evidence-based candidate.

No result in this task is itself a SOTA claim.
