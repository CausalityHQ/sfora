# UniCOM Full-Width Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and prospectively evaluate an imprinted UniCOM full-768 ArcFace training control against the sampled-512 recipe with identical full-768 holdout evaluation and five fresh paired confirmation seeds.

**Architecture:** Keep the registered eight-shard ArcFace implementation and vary only its selected training width. Split training width from evaluation width in the trainer, add a strict paired checkpoint evaluator and decision layer, then freeze a source-addressed run configuration before any DGX outcome is observed. A seed-0 pair gates the five fresh confirmation pairs.

**Tech Stack:** Python 3.12, PyTorch 2.12, NumPy 2.5, pytest, Ruff, UniCOM ViT-L/14-336, strict JSON artifacts, Git source binding.

**Spec:** `docs/unicom_full_width_objective_2026-08-23.md`

## Global Constraints

- Training arms are exactly `sampled_512=(official-eight-mask,512)` and `full_768=(official-eight-mask,768)`.
- `evaluation_features` is exactly 768 for every gating trainer evaluation; prefix-512 is diagnostic only in the paired evaluator.
- The mask generator performs eight 768-element random draws per optimizer step in both arms; tests bind the resulting generator state.
- Seed 0 is selection-only. Confirmation uses five fresh paired controls and candidates at seeds 2 through 6.
- Arm order is control-first for seeds 0, 2, 4, and 6 and candidate-first for seeds 3 and 5.
- No official query/gallery split is opened by this plan.
- Every scientific JSON rejects duplicate keys and nonfinite values, strict-reloads after publication, and refuses an existing destination.
- No custom kernel, mixed-width loss, threshold adjustment, or rerun after a finite failed gate is authorized.

---

### Task 1: Decouple Training Width From Evaluation Width

**Files:**
- Modify: `scripts/train_unicom_inshop.py`
- Modify: `tests/test_train_unicom_inshop.py`

**Interfaces:**
- Consumes: existing `objective_masks`, `run_training_epoch`, `evaluate_holdout`, `training_protocol` checkpoint binding.
- Produces: CLI field `--evaluation-features`; `resolve_evaluation_features(selected_features: int, evaluation_features: int | None) -> int`; checkpoint protocol key `evaluation_features` immediately after `selected_features`.

- [ ] **Step 1: Write failing CLI and resolution tests**

Add tests proving the legacy omitted flag resolves to the selected training width, the prospective explicit value resolves to 768, and bool/zero/>768 values fail:

```python
def test_evaluation_width_is_independent_and_legacy_default_is_preserved() -> None:
    module = _load_script()
    assert module.resolve_evaluation_features(512, None) == 512
    assert module.resolve_evaluation_features(512, 768) == 768
    assert module.resolve_evaluation_features(768, 768) == 768
    for value in (True, 0, 769):
        with pytest.raises((TypeError, ValueError)):
            module.resolve_evaluation_features(512, value)
```

- [ ] **Step 2: Run the focused RED**

Run: `.venv/bin/pytest -q tests/test_train_unicom_inshop.py -k evaluation_width`

Expected: FAIL because `resolve_evaluation_features` and `--evaluation-features` do not exist.

- [ ] **Step 3: Implement the minimal independent width**

Add:

```python
def resolve_evaluation_features(
    selected_features: int, evaluation_features: int | None
) -> int:
    if type(selected_features) is not int or not 0 < selected_features <= 768:
        raise ValueError("selected feature width differs")
    if evaluation_features is None:
        return selected_features
    if type(evaluation_features) is not int:
        raise TypeError("evaluation feature width must be a builtin integer")
    if not 0 < evaluation_features <= 768:
        raise ValueError("evaluation feature width differs")
    return evaluation_features
```

Parse `--evaluation-features` as `int` with default `None`, resolve it once in `run`, record it immediately after `selected_features` in `training_protocol`, and pass the resolved value—not `selected_features`—to `evaluate_holdout`.

- [ ] **Step 4: Add the loss/evaluator noninterference test**

Monkeypatch `run_training_epoch` and `evaluate_holdout`, call the smallest `fit_model`/`run` fixture, and assert loss receives 512 while every checkpoint evaluation receives 768. Repeat with loss width 768 and assert the evaluator call bytes/order are identical.

- [ ] **Step 5: Bind protocol and resume behavior**

Extend checkpoint tests so old checkpoints without `evaluation_features` are accepted only when the CLI omits the new flag and then resolve to their `selected_features`; prospective checkpoints must contain the exact builtin-int key/value and reject a changed evaluation width on resume.

- [ ] **Step 6: Run the trainer layer GREEN**

Run: `.venv/bin/pytest -q tests/test_train_unicom_inshop.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add scripts/train_unicom_inshop.py tests/test_train_unicom_inshop.py
git commit -m "decouple UniCOM evaluation width"
```

---

### Task 2: Freeze Mask-State and Full-Width Loss Equivalence

**Files:**
- Modify: `tests/test_train_unicom_inshop.py`
- Modify only if a test exposes a defect: `scripts/train_unicom_inshop.py`

**Interfaces:**
- Consumes: `objective_masks`, `sharded_mask_arcface_logits`.
- Produces: executable proof that sampled and full arms consume identical mask RNG shapes/counts and that full masks use all 768 coordinates.

- [ ] **Step 1: Write the exact generator-state test**

For two generators with the same seed, call `objective_masks("official-eight-mask", dimension=768, selected=512, generator=control_generator, device=torch.device("cpu"))` and the same function with `selected=768` and `candidate_generator`; assert both shapes, eight unique-coordinate rows, every full row sorted equals `torch.arange(768)`, and the final generator states are byte-equal.

- [ ] **Step 2: Write the full-width algebra test**

Use nontrivial FP32 embeddings, weights, labels, and the actual full masks. Compare logits against a direct per-shard reference that indexes the same permutation, normalizes, applies `F.linear`, concatenates in exact class order, applies ArcFace target mutation, and scales. Require `torch.equal`, not approximate equality.

- [ ] **Step 3: Run the focused tests**

Run: `.venv/bin/pytest -q tests/test_train_unicom_inshop.py -k 'mask_state or full_width_loss'`

Expected: PASS on the existing objective; if not, stop and repair the proven implementation defect without changing the registered objective.

- [ ] **Step 4: Commit Task 2**

```bash
git add tests/test_train_unicom_inshop.py scripts/train_unicom_inshop.py
git commit -m "bind UniCOM full-width objective semantics"
```

---

### Task 3: Build Strict Paired Checkpoint Evaluation

**Files:**
- Create: `scripts/evaluate_unicom_full_width_objective.py`
- Create: `tests/test_evaluate_unicom_full_width_objective.py`
- Reuse without modification: `scripts/evaluate_unicom_ema_imprint_official.py`

**Interfaces:**
- Consumes: two arms × epochs `(4,8,12,16)`, exact query/gallery records, raw-model checkpoints, full-768 and legacy-prefix retrieval functions.
- Produces: `evaluate_pair(config, load_checkpoint, encode) -> dict[str, object]`; `selection_decision(rows, costs) -> dict[str, object]`; `confirmation_decision(rows, costs) -> dict[str, object]`; strict `validate_result(value, expected_config)`.

- [ ] **Step 1: Write the missing-module RED**

Create the test loader for `scripts/evaluate_unicom_full_width_objective.py` and tests for constants:

```python
ARMS = ("sampled_512", "full_768")
EPOCHS = (4, 8, 12, 16)
SELECTION_SEEDS = (0,)
CONFIRMATION_SEEDS = (2, 3, 4, 5, 6)
PRIMARY_COORDINATES = tuple(range(768))
LEGACY_COORDINATES = tuple(range(512))
```

Run: `.venv/bin/pytest -q tests/test_evaluate_unicom_full_width_objective.py`

Expected: collection FAIL because the evaluator is absent.

- [ ] **Step 2: Implement pure paired statistics with TDD**

Write RED tests then implement:

Use the exact interfaces `paired_t_interval(values: tuple[float, ...], critical: float) -> tuple[float, float]`, `paired_query_bootstrap(control: tuple[float, ...], candidate: tuple[float, ...], *, seed: int, samples: int) -> tuple[float, float]`, and `first_epoch_reaching(values: dict[int, float], target: float) -> int | None`. The t interval uses `statistics.fmean`, `statistics.stdev`, and `critical / sqrt(n)`; the bootstrap indexes both paired arrays with one PCG64 integer matrix; the trajectory scans `EPOCHS` in literal order and returns the first qualifying epoch or `None`.

Require builtin finite floats, exact five-seed t critical `2.7764451052`, PCG64 seed `768`, exactly 10,000 paired-query replicates, and exact epoch order.

- [ ] **Step 3: Implement the selection decision**

The exact ordered predicates are:

```python
(
    ("primary_map_delta_at_least_0_003", delta >= 0.003),
    ("top1_query_loss_at_most_1", candidate_top1 >= control_top1 - 1),
    ("time_to_quality_no_later", candidate_epoch <= control_epoch),
    ("training_time_ratio_at_most_1_02", training_ratio <= 1.02),
    ("peak_allocated_ratio_at_most_1_02", allocated_ratio <= 1.02),
    ("peak_reserved_ratio_at_most_1_02", reserved_ratio <= 1.02),
    ("checkpoint_bytes_ratio_at_most_1_02", bytes_ratio <= 1.02),
)
```

`decision` is `PROMOTE_CONFIRMATION` only if every predicate is true, otherwise `CLOSE_FULL_WIDTH`.

- [ ] **Step 4: Implement the confirmation decision**

Recompute exact five paired seed deltas and require mean ≥0.003, paired-t lower >0, ≥4 positive, aggregate top-1 loss ≤5, per-seed top-1 loss ≤2, time-to-quality no later in ≥4 seeds, mean training/allocated/reserved/checkpoint ratios ≤1.02, and inference/storage ratios ≤1.02. Return only `SUPPORTED_HOLDOUT` or `CLOSE_FULL_WIDTH`; never emit a SOTA claim.

- [ ] **Step 5: Implement identical-view checkpoint evaluation**

For each registered seed/arm/epoch row, strict-load `checkpoint["model"]`, encode the same ordered query/gallery paths, and compute:

- primary: normalize all 768 coordinates, exact Euclidean retrieval;
- legacy: normalize all 768 coordinates, select prefix `[0,512)`, no second normalization.

Persist per-query AP@R, top-1 booleans, embedding hashes, ordered ID/path hashes, aggregate metrics, elapsed evaluation seconds, and peak allocated memory. Test that swapping arms cannot change coordinates, labels, or path order and that trainer history metrics are not consumed as scientific metrics.

- [ ] **Step 6: Implement exhaustive recursive validation**

Build a valid minimal payload fixture and generated mutation coverage that removes/adds/reorders every mapping key, changes every fixed value/type, injects NaN/±infinity at every numeric path, mutates every hash, per-query value, aggregate, ratio, predicate, and top-level decision. Assert generated path coverage equals traversed schema paths.

- [ ] **Step 7: Implement exclusive atomic publication**

Use same-directory mode-0600 `xb` temporary creation, fsync, strict reload/validation, `os.link` no-replace publication, directory fsync, inode-owned cleanup, and no rename fallback. Test success, preexisting destination/temp, publication race, and failure at open/write/fsync/reload/link/directory-fsync cleanup boundaries.

- [ ] **Step 8: Run evaluator GREEN**

Run: `.venv/bin/pytest -q tests/test_evaluate_unicom_full_width_objective.py`

Expected: all tests pass.

- [ ] **Step 9: Commit Task 3**

```bash
git add scripts/evaluate_unicom_full_width_objective.py tests/test_evaluate_unicom_full_width_objective.py
git commit -m "add paired UniCOM full-width evaluation"
```

---

### Task 4: Record Training Cost and Compare A-B-B-A Profiles

**Files:**
- Modify: `scripts/train_unicom_inshop.py`
- Modify: `tests/test_train_unicom_inshop.py`
- Create: `scripts/compare_unicom_full_width_profiles.py`
- Create: `tests/test_compare_unicom_full_width_profiles.py`

**Interfaces:**
- Consumes: one completed trainer run and four existing `profile_unicom_training_step.py` artifacts ordered control/candidate/candidate/control.
- Produces: atomic `run-receipt.json` per training arm and one strict A-B-B-A comparison artifact.

- [ ] **Step 1: Write trainer receipt RED tests**

Require exact fields for source/config hashes, seed, arm, resolved widths, command, start/finish timestamps, elapsed seconds, peak allocated/reserved bytes, checkpoint sizes/hashes at four epochs, history hash, exit status, and runtime. Test structural failures publish no receipt and finite completed runs publish once.

- [ ] **Step 2: Implement optional `--run-receipt` publication**

Reset CUDA peak stats immediately before `fit_model`, measure with `perf_counter_ns`, validate all four checkpoints, and publish the receipt only after `history.json` is closed and hashed. Keep omitted-flag legacy behavior unchanged.

- [ ] **Step 3: Write A-B-B-A comparator RED tests**

Require the exact arm order, equal checkpoint epoch/runtime/source, 50 timing and 10 profiler samples per artifact, and no duplicated artifact hashes. Compute paired position-adjusted candidate/control wall, CUDA, objective, peak-memory, and kernel-gate ratios.

- [ ] **Step 4: Implement strict profile comparison and publication**

The comparator authenticates each input with `validate_profile`, reports ratios and percentile intervals, and never promotes the scientific candidate. Kernel work remains eligible only if the existing lower-95% fusible fraction is ≥0.10 and an exact-output prototype later improves time-to-quality.

- [ ] **Step 5: Run the cost layer GREEN and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_train_unicom_inshop.py tests/test_compare_unicom_full_width_profiles.py
.venv/bin/ruff check scripts/train_unicom_inshop.py scripts/compare_unicom_full_width_profiles.py tests/test_train_unicom_inshop.py tests/test_compare_unicom_full_width_profiles.py
```

Commit:

```bash
git add scripts/train_unicom_inshop.py tests/test_train_unicom_inshop.py scripts/compare_unicom_full_width_profiles.py tests/test_compare_unicom_full_width_profiles.py
git commit -m "measure UniCOM full-width training cost"
```

---

### Task 5: Freeze Source and Run Configuration

**Files:**
- Create: `docs/unicom_full_width_objective_run_config.json`
- Create: `tests/test_unicom_full_width_objective_run_config.py`
- Modify only before source freeze: files from Tasks 1–4.

**Interfaces:**
- Consumes: independently reviewed source commit `S`.
- Produces: config-only handoff commit `H` binding exact commands, paths, hashes, seeds, arm order, attempts, and output names.

- [ ] **Step 1: Run complete source verification once**

Run the affected tests, then one serial repository-wide suite, Ruff on all changed Python files, `py_compile`, and `git diff --check`. Record exact counts and the one authorized CUDA-unavailable skip.

- [ ] **Step 2: Request independent Claude source review**

Use one read-only consultation with explicit models `['opus','gpt-5.6-sol']`. Require no Critical/Important findings across evaluator leakage, RNG state, checkpoint loading, statistics, publication, CLI execution, and forbidden official-split access. Repair via focused RED→GREEN and repeat the full gate only after code changes stabilize.

- [ ] **Step 3: Commit the reviewed source**

Commit only source and tests as `S`; record every changed Git blob SHA-256.

- [ ] **Step 4: Write the exact run configuration**

Bind `S`, dataset partition SHA, initial checkpoint SHA, UniCOM checkout revision, Python/Torch/NumPy/CUDA versions, seed/arm order, absolute DGX paths, trainer/evaluator/profiler commands, thresholds, one attempt per arm, and all output/temp names. Include `evaluation_features=768` in every arm.

- [ ] **Step 5: Test config mutation and candidate isolation**

Fresh-process tests authenticate every source path/hash, exact command token order, exact seed/arm order, output absence, and reject official query/gallery paths or candidate outcome fields before launch.

- [ ] **Step 6: Commit config-only handoff `H`**

Require `H^ == S`, sole commit path `docs/unicom_full_width_objective_run_config.json`, clean detached checkout, and all source Git/worktree hashes equal the config.

---

### Task 6: Run the Monitored Seed-0 Pair

**Files:**
- Create after execution: registered control/candidate trainer receipts and paired evaluator result under `reports/generated/`
- Create after validation: `docs/unicom_full_width_objective_seed0_result_2026-08-23.md`

**Interfaces:**
- Consumes: detached handoff `H`, idle DGX, absent registered destinations.
- Produces: one immutable seed-0 decision and cost evidence.

- [ ] **Step 1: Authenticate DGX preflight**

Verify detached `H`, clean status, all source/config/data/checkpoint hashes, exact environment, output/temp absence, no competing process/container/service/queue, and idle GPU.

- [ ] **Step 2: Launch exactly one control process**

Run the frozen sampled-512 command with `evaluation_features=768`. Retain the original PID/session and poll it every 45–55 seconds, reporting epoch/checkpoint progress, GPU memory/utilization, and failure immediately. Do not launch another process while it is alive.

- [ ] **Step 3: Validate control before candidate**

Require exit 0, four strict-load checkpoints, finite full-768 trainer metrics, exact receipt/schema/provenance, no temp, and idle GPU.

- [ ] **Step 4: Launch and monitor exactly one candidate process**

Run the frozen full-768 command and apply the same original-PID monitoring and validation. No rerun.

- [ ] **Step 5: Run A-B-B-A profiles and paired evaluation**

Only after both arms validate, run four serial profiler processes in exact control/candidate/candidate/control order, then one paired evaluator process. Strict-validate all artifacts offline.

- [ ] **Step 6: Apply and record the frozen decision**

If every seed-0 predicate passes, record `PROMOTE_CONFIRMATION`; otherwise record `CLOSE_FULL_WIDTH`. Commit and push the immutable artifacts and concise result without changing thresholds.

---

### Task 7: Run Five Fresh Paired Confirmation Seeds

**Files:**
- Create after execution: ten trainer receipts, paired evaluator result, and `docs/unicom_full_width_objective_result_2026-08-23.md`

**Interfaces:**
- Consumes: seed-0 `PROMOTE_CONFIRMATION` only.
- Produces: a five-seed holdout-supported or closed result with quality, training, inference, and storage costs.

- [ ] **Step 1: Reauthenticate before each seed pair**

Check source/config/data hashes, exact arm order for that seed, destination/temp absence, no old process, and idle GPU.

- [ ] **Step 2: Run ten jobs serially with active observation**

For seeds `(2,4,6)` run control then candidate; for `(3,5)` run candidate then control. Poll each original PID every 45–55 seconds, strict-validate before continuing, and stop all later work on any structural failure.

- [ ] **Step 3: Run the confirmation evaluator once**

Authenticate all 40 checkpoints and ten receipts, recompute both retrieval views from checkpoint bytes, compute the five seed-level deltas and costs, and apply the frozen confirmation decision.

- [ ] **Step 4: Measure inference and storage**

Use the identical full-768 deployment path for both arms, balanced A-B-B-A latency order on the same ordered tensors, exact checkpoint/model byte counts, and bootstrap intervals reported without overriding the 2% gates.

- [ ] **Step 5: Independently review and publish the final report**

Ask Claude for read-only artifact/schema/statistical review with `['opus','gpt-5.6-sol']`. Resolve only implementation/authentication defects; never rerun or retune a finite scientific result. Commit code/configs/artifacts/report, push, and state plainly whether the method is holdout-supported, closed, or structurally invalid. A later official readout is a separate prospective task.
