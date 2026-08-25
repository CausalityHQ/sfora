# UniCOM ProxyMuon F0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build, validate, and execute the preregistered cached-feature ProxyMuon falsifier against a same-budget selected AdamW control.

**Architecture:** Put optimizer math and scientific decisions in a small pure module, extend the existing cached-probe fitter through an optimizer-injectable internal path while preserving public AdamW behavior, and keep runtime/input checks plus strict recursive result validation in one standalone screen. Review and commit source/tests before a config-only Git handoff; execute the sole GPU run from a clean detached checkout.

**Tech Stack:** Python 3.13.9 on DGX, PyTorch 2.12.1+cu130, NumPy 2.5.0, scikit-learn 1.9.0, pytest, Ruff, canonical UTF-8 JSON, Git, rsync.

**Spec:** docs/superpowers/specs/2026-08-25-unicom-proxy-muon-f0-design.md at commit 591e9d3118811b2bc762683fcfa9915e144e571f, SHA-256 fbcc30286ed5dd55585fa498a2845791e283621f9d26427565a223e81a7016fa.

## Global Constraints

- F0 is cached-feature-only. It may not modify or invoke the full trainer, query/gallery evaluator, checkpoints, or deployment code.
- LR grid: (0.000025,0.00005,0.0001,0.0002,0.0004); Phase 1 seeds (0,1,2); Phase 2 fresh seeds (3,4,5).
- Phase 1 is exactly 30 cells/1,920 steps. Phase 2 is exactly 9 or 12 rows and at most 6,144 steps.
- AdamW and Muon constructors, row projection, diagnostic panel, retained steps, reach thresholds, decision cascade, 0.75 GPU-hour limit, and 8 GiB peak-allocation limit are exactly those in the spec.
- Every cell starts from fresh byte-identical imprinted head bytes, optimizer state, and registered streams.
- Query/gallery records are rejected and never opened.
- Git is code authority; do not build a separate handoff-authentication subsystem.
- Result/failure JSON schemas are disjoint and exact. Publication is create-exclusive, fsync-backed, atomic, strict-reloaded, byte-compared, revalidated, and no-clobber.
- A failure after any optimizer cell completes is outcome-bearing and cannot be retried without a new preregistration.

## File Structure

- Create src/sfora/unicom_proxy_muon.py: protocol constants, optimizer factories, precision adapter, trace, selection, comparison, decisions.
- Modify src/sfora/unicom_probe.py: optimizer-injectable cached-head trajectory and deterministic 16-cell diagnostic panel; unchanged public AdamW path.
- Create scripts/screen_unicom_proxy_muon_f0.py: config/input/runtime checks, orchestration, validators, failure reduction, atomic writer, CLI.
- Create tests/test_unicom_proxy_muon.py: pure optimizer/precision/decision tests.
- Modify tests/test_unicom_probe.py: legacy equivalence, injected optimizer, panel, projection, lifetime tests.
- Create tests/test_screen_unicom_proxy_muon_f0.py: orchestration, schema, mutation, publication, sentinel, real-CPU integration.
- Create tests/test_unicom_proxy_muon_f0_run_config.py: config-only handoff tests.
- Create docs/unicom_proxy_muon_f0_run_config.json only after source review.

---

### Task 1: Freeze the Pure Protocol and Decisions

**Files:**
- Create: src/sfora/unicom_proxy_muon.py
- Create: tests/test_unicom_proxy_muon.py

**Interfaces:**
- Produces: LR_GRID, PHASE1_SEEDS, PHASE2_SEEDS, RETAINED_STEPS, VALIDATION_STEPS, select_learning_rate(rows, optimizer), select_adamw_reference(rows), compute_reach_step(losses, reference_loss), decide_proxy_muon_f0(evidence).

- [ ] **Step 1: Write the missing-module RED**

~~~python
def test_frozen_protocol_constants() -> None:
    assert MODULE.LR_GRID == (2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4)
    assert MODULE.PHASE1_SEEDS == (0, 1, 2)
    assert MODULE.PHASE2_SEEDS == (3, 4, 5)
    assert MODULE.RETAINED_STEPS == (0, 64, 128, 192, 256, 307, 384, 435, 512)
    assert MODULE.VALIDATION_STEPS == (307, 435, 512)
~~~

Run: .venv/bin/pytest -q tests/test_unicom_proxy_muon.py

Expected: collection ERROR because the module is absent.

- [ ] **Step 2: Add constants and exact concrete-type validation**

Reject bool-as-int, numeric subclasses, nonfinite values, missing/extra/reordered seeds, duplicate cell keys, and LRs outside the registered grid.

- [ ] **Step 3: Write selection/anchor/reach/status REDs**

~~~python
def test_selection_uses_mean_step64_and_smaller_lr_tie() -> None:
    selected = select_learning_rate(tied_phase1_rows(5e-5, 1e-4), "adamw")
    assert selected.learning_rate == 5e-5
    assert selected.interior is True

def test_reach_uses_first_registered_qualifying_step() -> None:
    assert compute_reach_step({307: 1.1, 435: 0.9, 512: 0.8}, 1.0) == 435
    assert compute_reach_step({307: 1.1, 435: 1.01, 512: 1.001}, 1.0) == ">512"
~~~

Parameterize all six statuses and exact boundaries at step 307, step 435, accuracy delta -0.002, and both LR endpoints. Prove structural failure cannot fall through.

- [ ] **Step 4: Implement minimal pure logic**

Use math.fsum, smaller-numeric-LR ties, loss-only AdamW anchor selection, and exact cascade: structural → LR boundary → proceed → FP32 route → matched-LR route → close.

- [ ] **Step 5: Verify and commit**

Run: .venv/bin/pytest -q tests/test_unicom_proxy_muon.py

Expected: PASS.

~~~bash
git add src/sfora/unicom_proxy_muon.py tests/test_unicom_proxy_muon.py
git commit -m "implement ProxyMuon protocol decisions"
~~~

### Task 2: Implement Pinned Muon, Trace, and Precision Adapter

**Files:**
- Modify: src/sfora/unicom_proxy_muon.py
- Modify: tests/test_unicom_proxy_muon.py

**Interfaces:**
- Produces: build_head_optimizer(head, optimizer_name, learning_rate), MuonTrace, trace_builtin_muon_step(head, optimizer), and PrecisionMuon(head, lr, ns_dtype).

- [ ] **Step 1: Write constructor/state REDs**

Assert exact AdamW and Muon kwargs, one parameter only, fresh state per cell, one Muon momentum tensor, exact AdamW state keys, and rejection of unknown names or wrong types.

- [ ] **Step 2: Write BF16 fidelity REDs**

Use seeded eight-step FP32 gradient sequences for tall (32,16) and wide (16,32) tensors. After every step, require byte-identical FP32 parameter and momentum-buffer bytes between built-in torch.optim.Muon and PrecisionMuon(ns_dtype=torch.bfloat16).

- [ ] **Step 3: Write trace identity/noninterference REDs**

Reconstruct the Nesterov effective update out of place from cloned gradient/prior momentum, call the pinned private helper, require exact helper bytes, update_dtype=="torch.bfloat16", finite descriptive polar_factor_residual, and byte-identical traced/untraced step results and state.

- [ ] **Step 4: Implement the source-locked single-parameter adapter**

Mirror pinned PyTorch 2.12.1 momentum, Nesterov, five-step Newton--Schulz, 0.2*sqrt(max(rows,cols)) adjustment, and FP32 parameter addition. The sole switch changes Newton--Schulz dtype from BF16 to FP32.

- [ ] **Step 5: Add nonfinite/state mutation tests and verify**

Cover NaN/Inf gradient/momentum/update/parameter, zero row, sparse/non-matrix/multiple parameters, coefficient/step/momentum/Nesterov/scaling drift.

Run: .venv/bin/pytest -q tests/test_unicom_proxy_muon.py

Expected: PASS.

~~~bash
git add src/sfora/unicom_proxy_muon.py tests/test_unicom_proxy_muon.py
git commit -m "add pinned ProxyMuon precision adapter"
~~~

### Task 3: Add an Optimizer-Injectable Cached-Head Fitter

**Files:**
- Modify: src/sfora/unicom_probe.py
- Modify: tests/test_unicom_probe.py
- Modify: src/sfora/unicom_proxy_muon.py

**Interfaces:**
- Produces: _fit_probe_trajectory_with_optimizer(..., optimizer_factory, trace_factory), diagnostic_panel_losses(...), fit_proxy_muon_trajectory(...); existing public probe signatures remain unchanged.

- [ ] **Step 1: Freeze existing AdamW bytes**

On one deterministic fixture, compare the old public trajectory with an injected equivalent AdamW factory. Require exact head/snapshot bytes, exact ProbeFit, and unchanged Python/NumPy/Torch RNG states.

- [ ] **Step 2: Refactor behind unchanged wrappers**

Move only the existing epoch/batch/mask loop to the injected internal function. Public wrappers pass the former exact AdamW constructor. Preserve seed derivation, snapshot timing, diagnostic ordering, and row projection.

- [ ] **Step 3: Write and implement 16-cell panel tests**

Independently generate the first four epoch-zero 128-image batches and four consecutive eight-shard mask sets from experiment_stream_seed(fit_seed,23004). Require batch-major/mask-minor Cartesian order, exact first-cell equality with the parent diagnostic, and math.fsum(components)/16.

- [ ] **Step 4: Integrate ProxyMuon trajectories**

Record trace null at step zero and producing-step trace later; reduce retained heads to CPU bytes/scalars; reject nonfinite/zero-row/state drift. Weak-reference tests prove optimizer, gradient, masks, and unretained heads die after each cell.

- [ ] **Step 5: Verify and commit**

Run: .venv/bin/pytest -q tests/test_unicom_probe.py

Expected: all legacy and new tests PASS.

~~~bash
git add src/sfora/unicom_probe.py src/sfora/unicom_proxy_muon.py tests/test_unicom_probe.py
git commit -m "add optimizer-injectable UniCOM probe fitting"
~~~

### Task 4: Build Exact Schemas and Atomic Publication

**Files:**
- Create: scripts/screen_unicom_proxy_muon_f0.py
- Create: tests/test_screen_unicom_proxy_muon_f0.py

**Interfaces:**
- Produces: strict_json_object(bytes), validate_scientific_result(payload), validate_failure_receipt(payload), canonical_json_bytes(payload), publish_result_exclusive(path,payload,validator).

- [ ] **Step 1: Write an absent-script RED with independent fixtures**

Hand-write one minimal golden Phase-1 row, one golden retained-step row for each optimizer variant, and the complete reduced failure receipt. Build the exact 30-row/9-row and 30-row/12-row scientific fixtures with a test-local builder whose literal inputs and independent arithmetic do not call any production assembler, selector, summarizer, or decision helper.

- [ ] **Step 2: Implement exact recursive schemas**

Use ordered key tuples at every level and concrete built-in JSON types. Reject bool-as-int, nonfinite numbers, bad digests, negative counts/times/bytes, missing/extra/reordered keys, and keys exclusive to the other branch.

Every retained step has both trace keys present. AdamW selected/anchor steps use `update_dtype: null` and `polar_factor_residual: null`; ProxyMuon step zero also uses both nulls; later BF16 ProxyMuon steps use `"torch.bfloat16"` plus a finite float residual; later FP32 sensitivity steps use `"torch.float32"` plus a finite float residual.

- [ ] **Step 3: Recompute all relations**

Rebuild exact row orders, 16-cell means, selected LRs/interior flags, anchor choices, retained steps, reach, accuracy deltas/noninferiority, FP32 predicates, elapsed/peak limits, head hashes, 9-or-12 count, and final status from independent rows.

- [ ] **Step 4: Add dependent-summary mutation tests**

Mutate each decision scalar and recompute all dependent summaries/status while keeping independent evidence unchanged. Cover panel component/mean, LR/tie/selection/interior, anchor, retained step, reach, accuracy, trace dtype/residual, head hash, row order/count, elapsed, peak, status, and branch-exclusive keys.

- [ ] **Step 5: Implement atomic no-clobber publication**

Create the PID temp with O_CREAT|O_EXCL and mode 0600; write, flush, file-fsync, atomic rename without destination overwrite, directory-fsync, strict reload, byte equality, and validation of the distinct reload. Inject failures at every publication operation.

- [ ] **Step 6: Verify and commit**

Run: .venv/bin/pytest -q tests/test_screen_unicom_proxy_muon_f0.py

Expected: PASS.

~~~bash
git add scripts/screen_unicom_proxy_muon_f0.py tests/test_screen_unicom_proxy_muon_f0.py
git commit -m "add strict ProxyMuon result contract"
~~~

### Task 5: Bind CLI, Git, Inputs, and Runtime

**Files:**
- Modify: scripts/screen_unicom_proxy_muon_f0.py
- Modify: tests/test_screen_unicom_proxy_muon_f0.py
- Create: tests/test_unicom_proxy_muon_f0_run_config.py

**Interfaces:**
- Produces: load_spherical_feature_module(repo_root, expected_git_revision, expected_sha256), parse_args(argv), load_run_config(path), authenticate_source_and_inputs(config,repo_root), observe_runtime(), load_training_only_records(partition,dataset_root).

- [ ] **Step 1: Freeze exact CLI/config REDs**

Exact command:

~~~text
.venv/bin/python -I -B scripts/screen_unicom_proxy_muon_f0.py --config docs/unicom_proxy_muon_f0_run_config.json
~~~

Reject unknown flags, wrong Git HEAD/source bytes, dirty tracked files, symlinks/aliases, preexisting output/temp, wrong runtime, and wrong detached-clean state.

- [ ] **Step 2: Implement simple Git/file checks**

Use ordinary git rev-parse, git diff --quiet, git diff --cached --quiet, git ls-files, and SHA-256. Require config commit parent/source binding, exact UniCOM checkout path and revision `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`, and exact checkpoint/partition hashes before model construction. Do not add another authority framework.

- [ ] **Step 3: Reuse the exact authenticated parent feature implementation**

Do not move, copy, wrap, or edit `_EvaluationDataset`, `_load_official_model(checkout: Path, checkpoint: Path)`, or `_encode_feature_sets(args, fitting, validation)`. Keep `scripts/screen_unicom_spherical_probe.py` and its frozen five-key result/source schema byte-unchanged. `load_spherical_feature_module` must resolve exactly `repo_root/scripts/screen_unicom_spherical_probe.py`, reject symlinks/non-regular files and a preexisting private module name, recheck both the worktree SHA-256 and `git show expected_git_revision:scripts/screen_unicom_spherical_probe.py` SHA-256 against `expected_sha256`, and only then execute that file with `importlib.util.spec_from_file_location` under the fixed private name `_sfora_proxy_muon_parent_features`. Require the resolved `__file__` and both feature callables' `__module__` values to match that authenticated module. Call its existing `_encode_feature_sets` directly, preserving its hard-coded official ViT-L/14@336px loader, CUDA placement, record order, dtype/shape checks, and cleanup semantics. Remove the private module and all `unicom` package entries from `sys.modules` in a `finally` block after encoding.

Write a subprocess test using `.venv/bin/python -I -B` that loads the authenticated spherical file by absolute path without adding `scripts/` to `sys.path`, proves the returned feature callable is defined by that exact module/file, and exercises `_load_official_model(checkout, checkpoint)` against the existing tiny stub package with explicit `sys.modules` cleanup. A one-byte spherical-script mutation, a different resolved file, a copied function, or an import before source authentication must fail. Run the complete spherical screen test file unchanged to prove the parent path and result bytes remain intact. Full CUDA encoding is reserved for the sole DGX run because the production body requires the real ViT-L checkpoint and `.cuda()`; no CPU test may claim byte-equivalence for that path.

- [ ] **Step 4: Add query/gallery sentinels**

A partition fixture includes train/query/gallery records and an open poison for query/gallery images. The training-only loader must never touch them and must reject explicit query/gallery inputs.

- [ ] **Step 5: Bind observed runtime/Muon defaults**

After non-Torch authority, require exact configured Python, Torch, NumPy, sklearn, CUDA, GPU, deterministic flags, CUDA availability/idle device, and observed Muon signature/defaults. Runtime failure before cells yields a zero-cell failure receipt; pre-authority failure yields no receipt.

- [ ] **Step 6: Verify and commit**

Run: .venv/bin/pytest -q tests/test_screen_unicom_spherical_probe.py tests/test_screen_unicom_proxy_muon_f0.py tests/test_unicom_proxy_muon_f0_run_config.py

Expected: PASS.

~~~bash
git add scripts/screen_unicom_proxy_muon_f0.py tests/test_screen_unicom_proxy_muon_f0.py tests/test_unicom_proxy_muon_f0_run_config.py
git commit -m "bind ProxyMuon screen inputs and runtime"
~~~

### Task 6: Implement the Scientific Orchestrator

**Files:**
- Modify: scripts/screen_unicom_proxy_muon_f0.py
- Modify: tests/test_screen_unicom_proxy_muon_f0.py

**Interfaces:**
- Produces: run_proxy_muon_f0(config), assemble_scientific_result(...), assemble_failure_receipt(...), main(argv=None)->int.

- [ ] **Step 1: Write order/fail-fast REDs**

Record callback events. Assert Phase 1 order (optimizer, lr, seed) with seed innermost; Phase 2 order (seed, selected AdamW, optional anchor, ProxyMuon, FP32); fresh state per cell; no Phase-1 validation; and no Phase 2 before complete Phase 1/interior selection. Inject failure at each boundary and require exact completed-prefix count/hash and no later callbacks.

- [ ] **Step 2: Reconstruct parent evidence before candidates**

Load the already-authenticated `scripts/screen_unicom_spherical_probe.py` by its absolute file path under isolated Python, invoke that module's exact `_encode_feature_sets` implementation, reconstruct the exact spherical-parent training-only features/split, and require exact class-mean plus three fitted-target hashes. The spherical path is an explicit source-bound dependency of the ProxyMuon result; it remains byte-unchanged and is never imported by package or sibling name. A one-byte script/feature/head mutation must fail before optimizer construction.

- [ ] **Step 3: Implement both phases**

Phase 1 retains step-0/64 panel scalars and hashes. Phase 2 fits selected AdamW, optional anchor, BF16 ProxyMuon, and FP32 sensitivity independently; retains registered heads; evaluates 16-cell diagnostics at every retained step and train-identity accuracy only at 307/435/512. Release GPU tensors after each row and check elapsed/peak bounds between steps/cells.

- [ ] **Step 4: Add a genuine small real-CPU integration**

Use deterministic 32-class/16-dimension tensors with real loss, AdamW, built-in Muon, FP32 adapter, row projection, trace, both phase orders, full assembly, validation, atomic reload/no-clobber, and weakref death. Patch only sizes/step counts via explicit test parameters; do not mock optimizer/loss/fitter/selector/decision/validator/writer.

- [ ] **Step 5: Wire main/failure receipts**

Authenticate once, run once, validate/publish, return 0. Ordinary post-authority exceptions publish only reduced completed-prefix evidence and return 2. Publication I/O failure returns 2 without claiming a receipt. Restore RNG/thread/determinism/device state.

- [ ] **Step 6: Verify and commit**

Run: .venv/bin/pytest -q tests/test_unicom_proxy_muon.py tests/test_unicom_probe.py tests/test_screen_unicom_spherical_probe.py tests/test_screen_unicom_proxy_muon_f0.py tests/test_unicom_proxy_muon_f0_run_config.py

Expected: PASS.

~~~bash
git add src/sfora/unicom_proxy_muon.py src/sfora/unicom_probe.py scripts/screen_unicom_proxy_muon_f0.py tests/test_unicom_proxy_muon.py tests/test_unicom_probe.py tests/test_screen_unicom_proxy_muon_f0.py tests/test_unicom_proxy_muon_f0_run_config.py
git commit -m "implement UniCOM ProxyMuon cached-feature screen"
~~~

### Task 7: Review Source and Freeze the Config-Only Handoff

**Files:**
- Modify only Task 1--6 files for verified review fixes.
- Create after source freeze: docs/unicom_proxy_muon_f0_run_config.json.

**Interfaces:**
- Produces: reviewed source commit V and direct-child config-only commit H.

- [ ] **Step 1: Run affected/static gates**

~~~bash
.venv/bin/pytest -q tests/test_unicom_proxy_muon.py tests/test_unicom_probe.py tests/test_screen_unicom_spherical_probe.py tests/test_screen_unicom_proxy_muon_f0.py tests/test_unicom_proxy_muon_f0_run_config.py
.venv/bin/ruff check src/sfora/unicom_proxy_muon.py src/sfora/unicom_probe.py scripts/screen_unicom_proxy_muon_f0.py tests/test_unicom_proxy_muon.py tests/test_unicom_probe.py tests/test_screen_unicom_spherical_probe.py tests/test_screen_unicom_proxy_muon_f0.py tests/test_unicom_proxy_muon_f0_run_config.py
.venv/bin/python -m py_compile src/sfora/unicom_proxy_muon.py src/sfora/unicom_probe.py scripts/screen_unicom_proxy_muon_f0.py tests/test_unicom_proxy_muon.py tests/test_unicom_probe.py tests/test_screen_unicom_spherical_probe.py tests/test_screen_unicom_proxy_muon_f0.py tests/test_unicom_proxy_muon_f0_run_config.py
git diff --check
~~~

- [ ] **Step 2: Run one monitored repository-wide pytest**

Coordinate the shared lane; preserve the original PID/session; stop under the registered pressure/wedge criterion; do not replace it. Classify CUDA skip separately.

- [ ] **Step 3: Obtain independent source review**

Use one read-only consultation with provider="other", models=["opus","gpt-5.6-sol"], exact commit/base/paths/digests/spec/plan and test evidence. Repair each verified Critical/Important finding test-first, rerun affected gates, then one final full gate after the diff stabilizes. Repeat review until READY; record V.

- [ ] **Step 4: Write/test the exact config**

Bind spec, source.commit=V, all source hashes at V including the byte-unchanged spherical screen used as the feature implementation, frozen DGX runtime, exact UniCOM checkout path/revision, checkpoint/partition paths/hashes, protocol, and distinct absent result/temp/failure paths. Run the full config mutation suite.

- [ ] **Step 5: Commit config alone and review**

~~~bash
git add docs/unicom_proxy_muon_f0_run_config.json
git commit -m "freeze ProxyMuon F0 run configuration"
test "$(git rev-parse HEAD^)" = "$V"
test "$(git diff-tree --no-commit-id --name-only -r HEAD)" = "docs/unicom_proxy_muon_f0_run_config.json"
git diff --check HEAD^
~~~

Get a config-only Claude review. If source changes, create a new source/config chain rather than amending beneath the binding. Push and require remote ref equals reviewed H.

### Task 8: Execute, Observe, Validate, and Route the Sole Run

**Files:**
- Create after result: reports/generated/unicom-proxy-muon-f0-<source-short>.json
- Create after validation: reports/unicom_proxy_muon_f0_result_2026-08-25.md

**Interfaces:**
- Consumes: pushed H, external checkpoint/dataset, idle DGX.
- Produces: one canonical result/failure receipt, independent validation, next frozen route.

- [ ] **Step 1: Prepare clean detached DGX checkout**

Fetch H, create a fresh detached checkout, verify exact HEAD/clean tree, rsync only non-Git checkpoint/dataset inputs, and verify input hashes, UniCOM revision, runtime, idle GPU, absent output/temp/failure, and memory headroom.

- [ ] **Step 2: Launch exactly one process tree**

Run the exact config command with one retained PID/session and no pipe/retry/duplicate. Record launch time, PID, HEAD, command, GPU identity, and initial path absence.

- [ ] **Step 3: Observe at intervals no longer than 55 seconds**

Record liveness, elapsed, phase/cell progress, GPU utilization/memory, host pressure, output/temp/failure metadata, and bounded new log output. React immediately to terminal error, cap breach, nonfinite signal, or publication. Silence is not a wedge.

- [ ] **Step 4: Independently validate terminal bytes**

Copy once, record SHA-256, reject duplicate/nonfinite JSON, invoke production validation from a clean checkout, independently recompute selections/anchors/reach/predicates/status, verify canonical bytes/mode/link/temp absence, and prove no second process/artifact exists.

- [ ] **Step 5: Report, review, and route**

Report seed losses/accuracies/reach, selected LRs/interior flags, BF16/FP32 evidence, elapsed/peak cost, and prior confirmed quality/efficiency. Obtain independent no-Critical/Important review before publication/SOTA language. Route automatically: PROCEED_TRAINING → five-seed confirmation; FP32 route → fresh FP32 LR falsifier; matched-LR route → AdamW-only F1; LR boundary → scale expansion; close → full-width objective/cross-dataset replication; structural failure → retry only if zero cells completed and the repaired chain is reviewed.
