# UniCOM Retained-Checkpoint Official Readout Implementation Plan

> Execute test-first. No official query/gallery metric may be produced until the
> design, implementation, and input inventory are committed and independently reviewed.

**Goal:** Evaluate five fully authenticated random/imprinted checkpoint pairs plus one
non-gating sensitivity pair on the untouched official In-Shop query/gallery split and
atomically decide the frozen transfer-quality and trajectory gates.

**Architecture:** Add one standalone evaluator with pure inventory, schema, statistics,
and decision functions around a narrow authenticated model-encoding adapter. The CLI
authenticates all 48 checkpoint bytes and all non-outcome inputs before loading the
official split, keeps metric-bearing state in memory, emits progress without values,
and exclusively publishes one strict-reloaded result after all rows validate.

**Stack:** Python 3.12, PyTorch/torchvision, NumPy, SciPy, existing UniCOM trainer and
`sfora.unicom_inshop`/retrieval helpers, pytest, Ruff.

---

## Task 1: Freeze the prospective authority

**Files:**

- Commit: `docs/superpowers/specs/2026-08-22-unicom-official-pareto-design.md`

1. Incorporate every Critical/Important independent-review finding.
2. Verify no official result/output exists and the DGX checkpoint inventory is still
   complete.
3. Run `git diff --check` and commit only the design on the research branch.
4. Obtain a clean independent re-review of the exact commit. Stop on any unresolved
   Critical/Important issue.

## Task 2: RED then GREEN for exact inventory and statistics

**Files:**

- Create: `scripts/evaluate_unicom_ema_imprint_official.py`
- Create: `tests/test_evaluate_unicom_ema_imprint_official.py`

1. RED tests freeze exact seeds `(2,3,4,5,6)`, arms, epochs, checkpoint basename/order,
   sensitivity seed 1, summary authority SHA, raw-model choice, official partition
   counts/hash, and unique
   registered checkpoint hashes.
2. RED tests freeze exact builtin types, finite behavior, paired Student-t interval,
   five per-seed deltas, the +0.002 lower-bound rule, seed/query R@1 intervals, the
   non-gating seed-1 row, the +0.0305 anomaly flag, sign agreement, and the registered
   grid-native trajectory rule. Include boundary and mutation cases.
3. Implement only pure inventory/statistics/decision functions; rerun the focused tests
   to GREEN.

## Task 3: RED then GREEN for query-level evaluation and result validation

**Files:** same two files.

1. RED tests use a tiny real CPU model and deterministic records to require
   primary full-768 unit-normalized Euclidean ranking, secondary normalize-full then
   prefix-512 ranking, exact R@1/10/20/30/40/50 and query-/identity-weighted mAP@R,
   per-query AP@R/top-1 evidence, prefix-energy evidence, raw checkpoint state, and the
   exact eval-only FP32 BatchNorm projection state frozen by the structural erratum.
   Bind direct `retrieval_view` calls and prove the k=40/50 extension leaves existing
   mAP@R and Recall@1/10/20/30 bytes unchanged.
2. RED recursive mutation tests remove/add/reorder every object key, change fixed
   values/types, inject NaN/infinity, alter checkpoint/evidence hashes, perturb any
   recomputable metric/statistic/decision, and reorder seed/arm/epoch rows.
3. Implement the encoder adapter and strict recursive validator; rerun focused tests to
   GREEN.

## Task 4: RED then GREEN for one-shot CLI and atomic publication

**Files:** same two files.

1. RED fresh-process tests require isolated execution, detached clean checkout, exact
   source/blob identity, literal dataset/checkpoint roots, all 48 hashes checked before
   official records are loaded, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, strict FP32 raw-model
   loading and deterministic backend flags, and no candidate/result import reachability
   elsewhere.
2. RED sentinels require exactly 48 evaluations in frozen order, no metric-bearing
   stdout/stderr, the exact value-free progress schema, no partial result on any row
   failure, a 64-MiB output cap, maximum two structurally authorized attempts, and one
   final payload only.
3. RED publication tests cover pre-existing destination/temp, create/write/fsync/reload/
   link/directory-fsync failures, publication races, mode 0600, inode-owned rollback,
   64-MiB cap, attempt-1/authorized-attempt-2 history, and no-clobber behavior.
4. Implement the minimal CLI and exclusive strict-reloaded publisher. GREEN all focused
   tests.

## Task 5: Verify, review, and freeze source

1. Run the complete new test file, all affected UniCOM tests, Ruff, py_compile, and
   `git diff --check`.
2. Run the repository-wide serial suite once after the diff is stable.
3. Commit exactly evaluator + tests + the reviewed structural spec/plan/erratum changes,
   with no result or manifest.
4. Obtain independent source review. Reproduce every finding with a focused RED,
   minimally fix, rerun affected/full verification once, and repeat review until READY.

## Task 6: Freeze the executable run configuration

1. Add one ordinary Git-tracked JSON run configuration containing the reviewed source
   commit, dataset partition hash, runtime requirements, 48 ordered checkpoint path/hash
   records, and one output path.
2. Validate the configuration recursively, commit it normally, push the research
   branch, and transfer that exact commit to the DGX with Git plus `rsync` only for
   ignored outputs. No separate provenance framework or handoff-only schema is needed.
3. Re-run the focused configuration/source tests and obtain a short independent review.

## Task 7: Preflight and launch exactly once

1. In a fresh detached DGX checkout, authenticate Git/source/manifest/runtime/dataset,
   hash all 48 retained checkpoints, confirm output/temp absence, disk headroom, idle
   process/GPU state, and value-free smoke behavior without opening official metrics.
2. Launch one isolated evaluator process. Retain its original PID/session and command.
3. Poll that same process at intervals no longer than 55 seconds. At every poll record
   liveness, GPU utilization/allocation, RSS, disk headroom, bounded log tail, and
   output/temp state. Do not start another process.
4. On exit, immediately capture exit status, stop on structural failure, or validate
   the one result offline with both production and independent recomputation.

## Task 8: Report and continue automatically

1. Commit the immutable result separately after exact validation and independent review.
2. Write the concise result report with all five gating paired rows plus seed-1
   sensitivity, intervals, trajectories,
   resource evidence, and explicit non-SOTA/non-official-training boundary.
3. If both gates pass, immediately freeze and execute the fairness-control plan. If
   only trajectory passes, freeze the SOP iso-quality replication. If both fail, close
   static imprinting without tuning on the official outcome.
