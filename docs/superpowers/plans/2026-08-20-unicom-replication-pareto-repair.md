# UniCOM Replication Pareto Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the six-seed UniCOM imprint summary so its time-to-quality, cost, and selection-lineage claims remain valid under the observed seed-1 evidence.

**Architecture:** Preserve every existing paired result. Extend only the summary authority and derived cost model, then add forward-only initialization receipts for seeds 2..6. Raw wall time remains visible but cannot decide the claim.

**Tech Stack:** Python 3.12+, PyTorch 2.12, NumPy 2.5, pytest, Ruff, strict atomic JSON publication.

**Spec:** `docs/unicom_ema_imprint_replication_pareto_amendment_2026-08-20.md`

## Global Constraints

- Do not alter seed-1 report bytes or replace any valid run.
- Do not change training constants, data order, selected cell, seeds, epochs, or quality thresholds.
- Use RED before production changes and preserve exact failure evidence.
- Keep raw wall time descriptive and use the frozen conservative compute proxy for the Pareto gate.
- Authenticate the exact seed-0 selection report before summary construction.
- Initialization receipts add evidence only and must not perturb RNG or training state.

---

### Task 1: Canonical time-to-quality and Pareto cost summary

**Files:**
- Modify: `scripts/summarize_unicom_ema_imprint_replication.py`
- Modify: `tests/test_summarize_unicom_ema_imprint_replication.py`

**Interfaces:**
- Produces: `conservative_compute_proxy_seconds(...) -> dict[str, float]`.
- Extends: `summarize_replications(reports, *, selection_authority)`.
- Consumes: exact seed-0 selection authority and six v1 paired reports.

- [ ] **Step 1: Write the non-monotone control RED**

Construct seed 1 with control epoch-12 mAP above its epoch-16 target and candidate first
reaching that target at epoch 8. Assert `{random_raw: 12, imprinted_raw: 8, speedup: 1.5}`
and reject any fixed-16-derived `2.0` claim.

- [ ] **Step 2: Write the wall-time contamination RED**

Make imprinted raw wall time arbitrarily larger while profiler step time, first-quality
epochs, peak memory, and storage remain non-inferior. Assert wall time is retained in
`costs.training_seconds` but does not flip the Pareto decision. Then make the conservative
compute proxy or deployment cost inferior and assert the gate fails.

- [ ] **Step 3: Write the selection-authority RED**

Require the exact five-key authority from the amendment. Mutate path, digest, commit,
selected cell, and decision independently and require rejection.

- [ ] **Step 4: Run focused RED**

Run: `.venv/bin/pytest -q tests/test_summarize_unicom_ema_imprint_replication.py -k 'non_monotone or contaminated_wall or selection_authority'`

Expected: failures because the summary has no authority input and raw wall time still gates.

- [ ] **Step 5: Implement the minimal summary repair**

Add literal constants `STEPS_PER_EPOCH = 161` and
`TRAIN_SPLIT_IMAGE_UPPER_BOUND = 25_882`. Compute the exact proxy from the amendment,
embed the validated selection authority, remove raw wall time from the decision
conjunction, and preserve every existing descriptive cost row.

- [ ] **Step 6: Run focused and complete summary tests**

Run: `.venv/bin/pytest -q tests/test_summarize_unicom_ema_imprint_replication.py`

Run: `.venv/bin/ruff check scripts/summarize_unicom_ema_imprint_replication.py tests/test_summarize_unicom_ema_imprint_replication.py`

### Task 2: Forward-only initialization receipts

**Files:**
- Modify: `scripts/train_unicom_inshop.py`
- Modify: `tests/test_train_unicom_inshop.py`
- Modify: `scripts/evaluate_unicom_ema_imprint_replication.py`
- Modify: `tests/test_evaluate_unicom_ema_imprint_replication.py`

**Interfaces:**
- Produces: `classifier_initialization_receipt(...) -> dict[str, object]`.
- Persists: `initialization-receipt.json` atomically before the first optimizer step.
- Extends future measurement receipt binding without changing seed-1 report bytes.

- [ ] **Step 1: Write receipt purity RED**

Assert exact classifier tensor bytes/shape/dtype, trainer digest, seed/arm, algorithm name,
and Python/NumPy/Torch CPU/Torch CUDA RNG-state hashes. Assert receipt construction cannot
mutate tensors, RNG state, data order, model mode, or BatchNorm buffers.

- [ ] **Step 2: Write atomic publication and evaluator-binding RED**

Cover no-clobber publication, strict reload, wrong tensor digest, wrong seed/arm, and a
receipt copied across runs. Preserve the v1 seed-1 validation path with an explicit
`historical_initialization_receipt_unavailable` disclosure.

- [ ] **Step 3: Run focused RED**

Run: `.venv/bin/pytest -q tests/test_train_unicom_inshop.py tests/test_evaluate_unicom_ema_imprint_replication.py -k 'initialization_receipt'`

- [ ] **Step 4: Implement the minimal forward-only evidence path**

Publish the receipt after deterministic initialization and before optimizer construction
or the first batch. Bind it in future measurement evidence. Do not add the digest to the
training objective, initializer inputs, or RNG streams.

- [ ] **Step 5: Run affected verification**

Run: `.venv/bin/pytest -q tests/test_train_unicom_inshop.py tests/test_evaluate_unicom_ema_imprint_replication.py tests/test_summarize_unicom_ema_imprint_replication.py`

Run: `.venv/bin/ruff check scripts/train_unicom_inshop.py scripts/evaluate_unicom_ema_imprint_replication.py scripts/summarize_unicom_ema_imprint_replication.py tests/test_train_unicom_inshop.py tests/test_evaluate_unicom_ema_imprint_replication.py tests/test_summarize_unicom_ema_imprint_replication.py`

Run: `.venv/bin/python -m py_compile scripts/train_unicom_inshop.py scripts/evaluate_unicom_ema_imprint_replication.py scripts/summarize_unicom_ema_imprint_replication.py`

Run: `git diff --check`

### Task 3: Review, freeze, and resume confirmation

**Files:**
- Modify only files required by confirmed review findings.

**Interfaces:**
- Produces: reviewed source commit and fresh detached GPU checkout.
- Consumes: immutable seed-1 report and exact selection authority.

- [ ] **Step 1: Request adversarial review**

Use an explicit ordered review chain `models=["opus", "gpt-5.6-sol"]`. Require no
Critical/Important findings on estimator semantics, cost proxy, v1 compatibility,
initialization receipt purity, and tests.

- [ ] **Step 2: Resolve confirmed findings with focused RED/GREEN**

Do not change training constants or seed-1 evidence. Rerun the complete affected gate.

- [ ] **Step 3: Commit and push exact scope**

Commit the amendment/plan first, then source/tests in a separate commit. Verify each
commit exists with `git log`, push `devbox/similarity-ghc`, and create a fresh detached
GPU checkout from the reviewed commit.

- [ ] **Step 4: Resume seeds 2..6 serially**

For each seed, require idle GPU and absent destination; run random then imprinted with
identical recipe except classifier initialization, build strict receipts, run the hardened
paired evaluator once, and preserve every valid result. Do not replace a failed seed.
