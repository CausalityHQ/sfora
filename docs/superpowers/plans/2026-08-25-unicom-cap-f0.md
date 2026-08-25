# UniCOM CAP F0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, authenticate, and execute the preregistered no-training Covariance-Adjusted Prototype screen without observing candidate metrics before the reviewed source and run handoff are frozen.

**Architecture:** Keep reusable tensor math in a focused `sfora.unicom_cap` module, add only a trajectory primitive to the existing probe module without changing its legacy outputs, and place provenance, recursive validation, orchestration, and atomic publication in a standalone script that imports Torch only after source and artifact authentication. Freeze a source commit first, then add one config-only handoff commit and run exactly once from a clean detached checkout.

**Tech Stack:** Python 3.13.9, PyTorch 2.12.1+cu130, NumPy 2.5.0, scikit-learn 1.9.0, pytest, Ruff, Git, CUDA 13.0.

**Spec:** `docs/superpowers/specs/2026-08-25-unicom-cap-f0-design.md` at commit `87d26fc433362b31d255bfc9931319b8f12e2eba`, SHA-256 `4daafda9ed31218a77cbe3fe2017f48a90f8717555261774270771933ec40583`.

## Global Constraints

- Authenticate the parent result SHA-256 `d1a52703849acb96f359c2c7f209942fcbf6fa770eeaa0ed41d947780d714ddf`, parent source `ed2e7893b05d3b5105ff992691efccc5b13ad5a0`, UniCOM revision `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`, checkpoint SHA-256 `3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`, and partition SHA-256 `cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c` before importing Torch or computing candidate values.
- Reuse the exact parent fitting/validation split, seeds, masks, ArcFace constants, FP32 feature path, and FP64 reduction rules.
- Evaluate seed-invariant `cap_centered` and `cap_uncentered` exactly once, then evaluate each seed's `fitted_target`; never use validation values to construct or tune either CAP head.
- Use one 512-step fitted-head trajectory per fit seed with snapshots at `(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)`.
- Publish one strict canonical JSON artifact atomically and never overwrite, retry, or read query/gallery test data.
- Freeze this exact ordered reviewed-source list in the config: `scripts/screen_unicom_cap_f0.py`, `src/sfora/unicom_cap.py`, `src/sfora/unicom_probe.py`, `src/sfora/unicom_training.py`, `src/sfora/unicom_inshop.py`, `tests/test_screen_unicom_cap_f0.py`, `tests/test_unicom_cap.py`, `tests/test_unicom_probe.py`, `tests/test_unicom_cap_f0_run_config.py`.

---

### Task 0: Prove the parent CUDA replay before source freeze

**Files:**
- Read only: parent source at `ed2e7893b05d3b5105ff992691efccc5b13ad5a0`
- Create only in a dedicated DGX temporary directory: two throwaway parent replay JSON files
- Create locally: `docs/unicom_cap_f0_parent_replay_readiness_2026-08-25.md`

**Interfaces:**
- Consumes: the exact parent script, checkpoint, partition, runtime, and registered class-mean/three fitted-target hashes.
- Produces: a candidate-free replay-readiness record; it never imports or computes CAP.

This task is a mandatory gate before Task 5 may freeze `V_CAP`. Run it before Task 1 if the GPU lane is idle. If the existing registered Cars/UniCOM queue owns the GPU, Tasks 1--4 may proceed on CPU while this task remains queued, but Task 5 cannot complete until it passes.

- [ ] **Step 1: Authenticate an unmodified parent checkout**

Create a clean detached checkout at `ed2e7893b05d3b5105ff992691efccc5b13ad5a0`. Verify the parent script/worktree bytes equal their Git blobs, the parent result SHA-256 is `d1a52703849acb96f359c2c7f209942fcbf6fa770eeaa0ed41d947780d714ddf`, and the registered checkpoint, partition, runtime, and UniCOM revision match the spec. Use a fresh mode-0700 temporary directory on the DGX for outputs.

- [ ] **Step 2: Run two fresh candidate-free parent replays**

Run the unmodified parent screen twice, sequentially, using two distinct absent throwaway output paths. Do not change deterministic flags, source, runtime, or constants. Require both results to contain class-mean SHA `d183c0d26d451cc5184f4da0a2112766fb5b32d206ea711011f573b3b4aa9613` and fitted-target SHA values `bfabb3159677577cf8e6489a40b4765c4510c07a0c18e9094443a01de4cf244b`, `a56392a806fcf028876a0d1933c0095a7e20aad46cbb8f84f8c8d96d8468e8cd`, and `c1fe4cb49668e9b02796ca2fe48432518174cb3495cb1970d7e26ee3a187fd8f` in seed order.

- [ ] **Step 3: Record readiness and remove only the throwaway directory**

Record both output SHA-256 values, exact commands, runtimes, and their four embedded head hashes in `docs/unicom_cap_f0_parent_replay_readiness_2026-08-25.md`; commit only that report. Remove only the exact dedicated temporary directory after verification. Any mismatch closes CAP structurally before source freeze; do not retry or alter determinism settings.

---

### Task 1: Add trajectory snapshots without changing legacy bytes

**Files:**
- Modify: `src/sfora/unicom_probe.py`
- Modify: `tests/test_unicom_probe.py`

**Interfaces:**
- Produces: `fit_spherical_probe_trajectory(features: torch.Tensor, labels: torch.Tensor, initial: torch.Tensor, *, snapshot_steps: tuple[int, ...], steps: int = PROBE_STEPS, batch_size: int = PROBE_BATCH_SIZE, batch_seed: int = PROBE_BATCH_SEED, mask_seed: int = PROBE_MASK_SEED, diagnostic_seed: int = PROBE_DIAGNOSTIC_SEED, fit_seed: int = 0) -> tuple[ProbeFit, dict[int, torch.Tensor]]`.
- Preserves: `fit_spherical_probe` and `evaluate_probe_heads` behavior and output values exactly; the latter remains restricted to `("class_mean", "spherical_probe")`.

- [ ] **Step 1: Write trajectory RED tests**

Add tests that snapshot the initial head at step zero, snapshot each requested update exactly once, prove the trajectory final head is byte-identical to the existing `fit_spherical_probe` result, and require one live final tensor to serve as both the returned fit head and final snapshot:

```python
def test_probe_trajectory_reuses_one_optimizer_and_matches_legacy_final() -> None:
    features, labels, initial = _separable_probe_fixture()
    fit, snapshots = fit_spherical_probe_trajectory(
        features,
        labels,
        initial,
        steps=8,
        snapshot_steps=(0, 1, 2, 4, 8),
        batch_size=8,
    )
    legacy = fit_spherical_probe(features, labels, initial, steps=8, batch_size=8)
    assert tuple(snapshots) == (0, 1, 2, 4, 8)
    assert torch.equal(snapshots[0], initial)
    assert snapshots[8] is fit.head
    assert torch.equal(fit.head, legacy.head)
```

- [ ] **Step 2: Run the exact RED selector**

Run:

```bash
.venv/bin/pytest -q tests/test_unicom_probe.py -k 'probe_trajectory'
```

Expected: an attribute failure for the missing trajectory interface.

- [ ] **Step 3: Extract one shared fit loop**

Implement a private fit worker that records detached contiguous clones only at registered steps. Have both public fit functions call it. Preserve the evaluator implementation untouched. Ensure the final registered snapshot is the returned `ProbeFit.head` object rather than a second clone.

- [ ] **Step 4: Prove legacy equivalence and new behavior GREEN**

Run:

```bash
.venv/bin/pytest -q tests/test_unicom_probe.py
.venv/bin/ruff check src/sfora/unicom_probe.py tests/test_unicom_probe.py
```

Expected: all tests pass and Ruff exits zero.

- [ ] **Step 5: Commit the independently testable primitive change**

```bash
git add src/sfora/unicom_probe.py tests/test_unicom_probe.py
git diff --cached --check
git commit -m "extend UniCOM probe trajectory primitives"
```

### Task 2: Implement CAP construction and decision logic

**Files:**
- Create: `src/sfora/unicom_cap.py`
- Create: `tests/test_unicom_cap.py`

**Interfaces:**
- Consumes: Task 1 trajectory primitive and the unchanged two-head evaluator.
- Produces: `build_cap_heads(features, labels, *, row_norm, ledoit_wolf_fn=sklearn.covariance.ledoit_wolf) -> CapConstruction`.
- Produces: `CapConstruction(sample_count, feature_count, shrinkage, covariance_trace, cholesky_diagonal_min, cholesky_diagonal_max, covariance_sha256, condition_number, effective_rank, covariance, class_means, global_mean, heads)` with concrete Python scalar metadata and ordered heads.
- Produces: `cap_step_equivalence(cap_loss, trajectory_losses) -> int | str`.
- Produces: `covariance_mask_mismatch(construction, *, seed=23006, mask_sets=8) -> dict[str, object]`.
- Produces: `cap_decision(class_mean, cap_metrics, target_heads, trajectories) -> CapDecision`.

- [ ] **Step 1: Write analytic CAP formula RED tests**

Use a small FP64 analytic residual matrix and an injected `ledoit_wolf_fn`. Assert exact call arguments `assume_centered=True, block_size=1000`, FP32 normalization before FP64 accumulation, exact centered and uncentered right-hand sides, Cholesky solves, row norm, variant order, sample/feature counts, concrete float shrinkage, covariance trace, minimum/maximum Cholesky diagonal, covariance digest, and rejection of nonfinite/asymmetric/non-positive-definite inputs. Add an unbalanced-class fixture proving `mu_c` is the per-class mean of normalized fitting rows and `mu` is the row-weighted global mean over all fitting rows, not the mean of class means.

```python
def test_build_cap_heads_uses_registered_ledoit_wolf_solve() -> None:
    calls: list[tuple[np.ndarray, bool, int]] = []
    covariance = np.diag([2.0, 4.0])
    def fake(values, *, assume_centered, block_size):
        calls.append((values.copy(), assume_centered, block_size))
        return covariance.copy(), 0.25
    result = build_cap_heads(
        FEATURES,
        LABELS,
        row_norm=0.5,
        ledoit_wolf_fn=fake,
    )
    assert calls[0][1:] == (True, 1000)
    assert tuple(result.heads) == ("cap_centered", "cap_uncentered")
    assert torch.equal(
        torch.linalg.vector_norm(result.heads["cap_centered"], dim=1),
        torch.full((2,), 0.5, dtype=torch.float32),
    )
```

- [ ] **Step 2: Write exact threshold and decision RED tests**

Cover both sides of `0.0501203852609845`, `0.006380126646800488`, cosine `0.95`, both paired lower bounds at zero, `60/64`, unrepresented equality, `64`, `">512"`, centered tie break, exact `selected_variant=null` under close, and statuses `PROCEED_STAGE_A`, `ROUTE_STAGE_B`, `CLOSE_CAP`. Require builtin scalar types and reject reordered/missing seed data.

- [ ] **Step 3: Write masked-covariance diagnostic RED tests**

On a diagonal covariance require restricted-full and principal-submatrix directions to have cosine one. On an off-diagonal covariance require the independently computed mismatch. Assert seed `23006`, exactly 8 x 8 ordered masks, little-endian int64 mask hashes, eigenvalue condition number, Shannon effective rank, and `method="linear"` p05/median.

- [ ] **Step 4: Run CAP RED**

```bash
.venv/bin/pytest -q tests/test_unicom_cap.py
```

Expected: collection failure because `sfora.unicom_cap` does not exist.

- [ ] **Step 5: Implement the minimal pure CAP module**

Use frozen dataclasses with concrete Python scalar fields. Compute `z = F.normalize(features_fp32, dim=1).double()`. Recompute `mu_c` inside this function as `index_add(z) / counts`, exactly matching the pre-normalize/pre-scale means in `class_mean_head`; compute `mu = z.mean(dim=0)` over all fitting rows, never the unweighted mean of class means. The separately reconstructed class-mean head is authenticated by the runner and is not an input to CAP residuals or right-hand sides. Convert the normalized rows and means to contiguous FP64 CPU arrays. Compute residuals in fitting-row order, call scikit-learn once, validate symmetry with `np.allclose(rtol=1e-12, atol=1e-14)`, use `np.linalg.cholesky`, then solve exactly with `np.linalg.solve(cholesky, rhs.T)` followed by `np.linalg.solve(cholesky.T, first_solution)`; do not add an unpinned SciPy dependency. Normalize every nonzero row, scale to the registered norm, and cast once to contiguous FP32 on the input device. Persist every `CapConstruction` field named above. Compute the exact eight-set mismatch diagnostic and implement paired statistics, predicates, and decisions as pure functions that never import data or inspect files.

- [ ] **Step 6: Run CAP GREEN and static checks**

```bash
.venv/bin/pytest -q tests/test_unicom_cap.py
.venv/bin/ruff check src/sfora/unicom_cap.py tests/test_unicom_cap.py
.venv/bin/python -m py_compile src/sfora/unicom_cap.py tests/test_unicom_cap.py
```

- [ ] **Step 7: Commit CAP math**

```bash
git add src/sfora/unicom_cap.py tests/test_unicom_cap.py
git diff --cached --check
git commit -m "implement UniCOM covariance-adjusted prototypes"
```

### Task 3: Build strict candidate-free provenance and result validation

**Files:**
- Create: `scripts/screen_unicom_cap_f0.py`
- Create: `tests/test_screen_unicom_cap_f0.py`
- Create: `tests/test_unicom_cap_f0_run_config.py`

**Interfaces:**
- Consumes: the frozen spec, parent artifact, and Task 1/2 modules.
- Produces: `strict_json_object`, `validate_run_config`, `authenticate_run`, `validate_result`, `write_result_atomic`, `run_parent_replay_preflight`, `parse_args`, and `main`.

- [ ] **Step 1: Write provenance and parser RED tests**

Freeze exact CLI flags `--config`, `--unicom-checkout`, `--checkpoint`, `--dataset-root`, `--parent-result`, `--output`, and optional literal `--parent-replay-only`. Construct a temporary linear Git fixture and require the config bytes to equal the Git blob at detached `HEAD`, `HEAD` to change only the config path relative to `HEAD^`, `config["source"]["commit"] == HEAD^`, every ordered source worktree digest to equal both the configured literal and its `HEAD^` Git blob, and every path flag to resolve exactly to its config-derived path. Also require parent artifact bytes/SHA, source commit ancestry, UniCOM revision, checkpoint, partition, clean detached checkout, real non-symlink inputs, real output parent, and absent destination/PID-temp before importing Torch. Wrong config ancestry, drifted source digest, or flag/config disagreement exits `2` with no output.

For `--parent-replay-only`, require candidate-import sentinels to remain untouched, reconstruct only class mean and all three fitted targets, emit exactly one canonical stdout JSON line with keys `class_mean_sha256`, `target_sha256_by_seed` (ordered `0,1,2`), `candidate_values_computed=false`, publish no file, and exit zero only on all four registered hashes. Run two fresh subprocesses in the test and require byte-identical stdout. A mismatch exits `2` and cannot be retried by the CLI itself.

- [ ] **Step 2: Write exact recursive schema RED tests**

Build one valid synthetic result in the exact eleven-key top-level order. Parameterize mutations for every nested key order, concrete type, aggregate, covariance sample/feature counts, trace, minimum/maximum Cholesky diagonal, digest, condition number, effective rank, mismatch summary, construction-mask order, the stored-once class-mean head and metric replay, seed-invariant CAP metric duplication/drift, comparator order, seed order, 64-mask and 3188-image lengths, both paired lower bounds, 3200-row cosine vectors, 11-step trajectory, step equivalence, predicate, nullable selected variant, decision status, `candidate_values_computed`, NaN/infinity, duplicate key, and unknown key.

- [ ] **Step 3: Write atomic publication RED tests**

Require mode 0600 temporary bytes, completed partial writes, file and directory `fsync`, strict reload, byte comparison, exact hard-link publication, no-clobber behavior, owned rollback after post-link failure, and preservation of a foreign race winner.

- [ ] **Step 4: Write future handoff-schema RED tests**

In `tests/test_unicom_cap_f0_run_config.py`, call the production `validate_run_config` on a synthetic future config with exact top-level order `schema_version`, `spec`, `parent`, `environment`, `inputs`, `protocol`, `source`, `handoff`, `result`. Freeze nested key order for spec path/SHA/commit; parent path/SHA/source commit; environment versions/device/dtypes; input checkout/checkpoint/dataset paths and hashes; every protocol constant; source commit and ordered `{path,sha256}` rows; handoff parent/sole-path/detached-clean requirements; and result relative path/schema. Require a direct config-only handoff child of a synthetic reviewed source and reject reordered/extra keys, wrong ancestry, source drift, flag/config path disagreement, an existing result, and an existing PID-temp.

- [ ] **Step 5: Run strict-runner RED**

```bash
.venv/bin/pytest -q tests/test_screen_unicom_cap_f0.py tests/test_unicom_cap_f0_run_config.py
```

Expected: collection failure for the absent script.

- [ ] **Step 6: Implement non-Torch authority and schema functions**

Keep module-scope imports limited to the standard library and NumPy. Implement `validate_run_config` in this production script, not only in its test. Before importing `sfora.unicom_cap`, `sfora.unicom_probe`, Torch, or scikit-learn, authenticate the config-only `HEAD` edge, bind `config["source"]["commit"]` to `HEAD^`, authenticate every literal source digest against both the `HEAD^` Git blob and worktree bytes, and require all CLI path arguments to match the config. Implement type/order-aware result validation that recomputes all derived fields from primitive persisted rows rather than comparing a payload to itself.

- [ ] **Step 7: Implement publication and structural CLI behavior**

Return exit `2` with no output on every ordinary exception. Refuse an existing output before authority checks. Publish only a fully validated result, strict-reload it, compare canonical bytes, and never retry.

- [ ] **Step 8: Run runner GREEN**

```bash
.venv/bin/pytest -q tests/test_screen_unicom_cap_f0.py
.venv/bin/pytest -q tests/test_unicom_cap_f0_run_config.py
.venv/bin/ruff check scripts/screen_unicom_cap_f0.py tests/test_screen_unicom_cap_f0.py tests/test_unicom_cap_f0_run_config.py
.venv/bin/python -m py_compile scripts/screen_unicom_cap_f0.py tests/test_screen_unicom_cap_f0.py tests/test_unicom_cap_f0_run_config.py
```

- [ ] **Step 9: Commit the strict runner boundary**

```bash
git add scripts/screen_unicom_cap_f0.py tests/test_screen_unicom_cap_f0.py tests/test_unicom_cap_f0_run_config.py
git diff --cached --check
git commit -m "add strict UniCOM CAP screen runner"
```

### Task 4: Wire the authenticated real scientific path

**Files:**
- Modify: `scripts/screen_unicom_cap_f0.py`
- Modify: `tests/test_screen_unicom_cap_f0.py`

**Interfaces:**
- Consumes: authenticated inventory, Task 1 primitives, and Task 2 CAP functions.
- Produces: `execute_screen(args, inventory) -> dict[str, object]` and a complete one-process `run` path.

- [ ] **Step 1: Write the real tiny-model RED**

Use a deterministic CPU feature encoder whose outputs exercise class means, CAP covariance, two seed-invariant CAP evaluator calls, three target trajectories/evaluator calls, 64 fixed masks, exact CAP-to-target row cosines, paired uncertainty, step equivalence, decision validation, atomic reload, and weak-reference release. Target `execute_screen`, whose inputs are already authenticated; patch only the feature source/device boundary and do not patch CAP math, fit loop, evaluator, validator, or writer.

- [ ] **Step 2: Write parent-reproduction and no-leakage RED tests**

Require the real runner to reproduce the parent class-mean and three fitted-target primitive metrics before recording CAP values. Install query/gallery and candidate-output open sentinels and assert they remain untouched. Inject a mismatch into each parent primitive and require structural exit `2` with no result.

- [ ] **Step 3: Run scientific-path RED**

```bash
.venv/bin/pytest -q tests/test_screen_unicom_cap_f0.py -k 'real_cpu or parent_reproduction or no_leakage'
```

Expected: failures because `execute_screen` is not implemented.

- [ ] **Step 4: Implement exact one-pass execution**

Load and encode the exact fitting/validation rows once, build class mean and CAP once, verify the parent class-mean SHA, evaluate class mean once, persist its exact head and primitive validation evidence once, and evaluate `cap_centered` and `cap_uncentered` once. Then for seeds `0,1,2` run exactly one 512-step target trajectory. Require each final trajectory SHA to equal the registered parent seed hash and use that same live tensor as `fitted_target`. Evaluate each `fitted_target` and each of the eleven snapshots. Compute/persist all 3200 clamped FP64 CAP-to-target row cosines per variant/seed. Store CAP validation metrics and predicates 2--6 once at top level; store only target/trajectory/cosine evidence and predicates 1/7 in seed rows. Reduce/detach each metric immediately, release snapshot tensors after each seed, build the exact primitive JSON, call `validate_result`, and return it.

- [ ] **Step 5: Restore all RNG states and bind runtime**

Save/restore Python, NumPy, CPU Torch, and all CUDA RNG states around the scientific call. Pin CPU BLAS/OpenMP threads to one. Put exact Python/Torch/NumPy/scikit-learn/CUDA/GB10 validation in the authenticated public `run` boundary outside `execute_screen`; add a focused unit test using an injected observed-version/device mapping. The real CPU tiny-model test calls `execute_screen` and therefore does not claim to emulate the DGX runtime gate. Record elapsed seconds and peak allocated GPU MiB only after synchronization.

- [ ] **Step 6: Run the complete affected gate**

```bash
.venv/bin/pytest -q tests/test_unicom_probe.py tests/test_unicom_cap.py tests/test_screen_unicom_spherical_probe.py tests/test_screen_unicom_cap_f0.py
.venv/bin/ruff check src/sfora/unicom_probe.py src/sfora/unicom_cap.py scripts/screen_unicom_cap_f0.py tests/test_unicom_probe.py tests/test_unicom_cap.py tests/test_screen_unicom_cap_f0.py
.venv/bin/python -m py_compile src/sfora/unicom_probe.py src/sfora/unicom_cap.py scripts/screen_unicom_cap_f0.py
git diff --check
```

- [ ] **Step 7: Commit the integrated scientific path**

```bash
git add scripts/screen_unicom_cap_f0.py tests/test_screen_unicom_cap_f0.py
git diff --cached --check
git commit -m "integrate UniCOM CAP frozen-feature screen"
```

### Task 5: Review and freeze the exact source commit

**Files:**
- Modify only files identified by independently reproduced review findings.

**Interfaces:**
- Produces: reviewed source commit `V_CAP` and exact source digests.

- [ ] **Step 1: Run one repository assurance gate serially**

Coordinate the shared lane, then run one original process only:

```bash
.venv/bin/pytest -q
```

Stop rather than restart if cgroup pressure reaches the registered local safety boundary. Preserve the terminal result from the original process.

- [ ] **Step 2: Request adversarial cross-provider source review**

Ask Claude with ordered fallback `models=["opus", "gpt-5.6-sol"]` to inspect the cumulative diff and execute narrow tests, focusing on leakage, parent reproduction, formula fidelity, source authentication, recursive validation, tensor lifetime, and atomic publication. Do not compute candidate values.

- [ ] **Step 3: Repair every reproduced Critical/Important finding with focused RED→GREEN**

For each accepted finding, add a failing mutation/behavior test first, implement the smallest repair, rerun the affected selector, and commit the repair separately. Repeat the same reviewer until READY.

- [ ] **Step 4: Run one final post-review repository assurance gate**

After the reviewer returns READY and no further production/test bytes change, coordinate the shared lane and run exactly one original `.venv/bin/pytest -q` process. Preserve its terminal result; do not overlap or replace it. Run Ruff, py_compile, and `git diff --check` over every changed source/test file afterward.

- [ ] **Step 5: Record the reviewed source identity**

```bash
git rev-parse HEAD
sha256sum src/sfora/unicom_probe.py src/sfora/unicom_cap.py scripts/screen_unicom_cap_f0.py
git status --short
```

Require a clean tracked worktree. Name this reviewed commit `V_CAP` in the run handoff.

### Task 6: Freeze a config-only run handoff

**Files:**
- Create: `docs/unicom_cap_f0_run_config.json`
- Use: `tests/test_unicom_cap_f0_run_config.py`

**Interfaces:**
- Consumes: reviewed `V_CAP`, exact source digests, model/data authorities, and spec authority.
- Produces: one config-only direct child `H_CAP` that binds the scientific output path.

- [ ] **Step 1: Bind the reviewed source into the frozen schema**

Populate the already-reviewed production `validate_run_config` schema with `V_CAP`, its ordered source digests, and output path `reports/generated/unicom-cap-f0-<V_CAP[:7]>.json`. The exact top-level order is `schema_version`, `spec`, `parent`, `environment`, `inputs`, `protocol`, `source`, `handoff`, `result`, with the nested orders frozen in Task 3 Step 4. Do not edit the validator after source review.

- [ ] **Step 2: Make the exact real config GREEN**

Populate hashes from Git blobs at `V_CAP`; do not open query/gallery data or run Torch. Run:

```bash
.venv/bin/pytest -q tests/test_unicom_cap_f0_run_config.py
.venv/bin/ruff check tests/test_unicom_cap_f0_run_config.py
git diff --check
```

- [ ] **Step 3: Commit only the handoff and independently review it**

```bash
git add docs/unicom_cap_f0_run_config.json
git diff --cached --name-status
git commit -m "freeze UniCOM CAP screen handoff"
```

Require the commit edge to contain only the config file, then obtain a READY review of exact `V_CAP -> H_CAP` parentage, source hashes, authorities, config schema, and absent output.

### Task 7: Execute and adjudicate exactly one DGX attempt

**Files:**
- Create remotely and later register: `reports/generated/unicom-cap-f0-<V_CAP[:7]>.json`
- Create after validation: `docs/unicom_cap_f0_result_2026-08-25.md`

**Interfaces:**
- Consumes: clean detached `H_CAP`, frozen config, exact environment, and absent destination/temp.
- Produces: one authenticated result and one evidence-based route.

- [ ] **Step 1: Prepare a clean detached DGX checkout without interrupting registered work**

Transfer the exact Git bundle, verify its SHA-256, detach at `H_CAP`, authenticate `H_CAP^ == V_CAP`, every source/config Git blob and worktree byte, runtime, UniCOM checkout/checkpoint, partition, clean status, absent result/temp, and an idle GPU lane. Do not start until the Cars queue and any registered UniCOM process release the GPU.

- [ ] **Step 2: Run exactly two candidate-free parent replay preflights**

Launch two fresh sequential preflight processes under the frozen checkout/runtime. Each reconstructs only the class-mean head and all three fitted-target trajectories, computes no CAP value, and must reproduce class-mean SHA `d183c0d26d451cc5184f4da0a2112766fb5b32d206ea711011f573b3b4aa9613` plus target SHA values `bfabb3159677577cf8e6489a40b4765c4510c07a0c18e9094443a01de4cf244b`, `a56392a806fcf028876a0d1933c0095a7e20aad46cbb8f84f8c8d96d8468e8cd`, and `c1fe4cb49668e9b02796ca2fe48432518174cb3495cb1970d7e26ee3a187fd8f` for seeds `0,1,2`. Stop without a scientific attempt if either differs. Do not enable deterministic-algorithm flags or retry a failed preflight.

- [ ] **Step 3: Launch one process and monitor the original PID**

Run the frozen CLI once with the authenticated `--config` and exact absolute paths that must equal the config values. Retain its PID/session, poll that same process at intervals no longer than 55 seconds, report liveness and errors, and never launch a replacement.

- [ ] **Step 4: Validate offline before reading the decision**

Require exit zero, no temp, exact SHA-256, strict production validation, independent recomputation of covariance/formulas/metrics/trajectory/predicates/decision, no query/gallery access, and unchanged authorities. Stop on any structural discrepancy; do not rerun.

- [ ] **Step 5: Record the decision and route**

Write a concise result document with primitive values, per-seed predicates, selected variant, status, elapsed/peak cost, artifact path/SHA, exact commands, and review outcome. `PROCEED_STAGE_A` opens a separate CAP initialization training preregistration; `ROUTE_STAGE_B` opens only a tracked-head design; `CLOSE_CAP` closes CAP and returns to the Muon time-to-quality candidate.

- [ ] **Step 6: Transfer and register the immutable result**

Copy the exact DGX artifact back without changing its bytes, recompute SHA-256 locally and require equality with the DGX value, then re-run strict production validation locally. Because `reports/generated/` is intentionally ignored, first stage the result document normally, then force-stage only the literal registered artifact path with `git add -f -- reports/generated/unicom-cap-f0-<V_CAP[:7]>.json`; no wildcard or ignore-file edit is authorized. Verify cached scope is exactly those two paths and the cached artifact digest equals the DGX value, then commit with subject `record UniCOM CAP F0 result`. Never amend source/config commits or overwrite the artifact.
