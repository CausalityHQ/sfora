# UniCOM Finish Causal Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a matched A/B/C continuation panel that attributes the observed UniCOM finish gain.

**Architecture:** Add a new protocol module and runner rather than modifying the frozen historical trainer or screen. Reuse the authenticated legacy trainer functions through an adapter, publish complete paired evidence, and evaluate the two causal contrasts only after all three arms finish.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, canonical JSON and atomic no-replace publication.

**Spec:** `docs/superpowers/specs/2026-09-04-unicom-finish-causal-panel-design.md`

## Global Constraints

- Work only in Sfora; do not modify Borsuk.
- Never modify `scripts/train_unicom_inshop.py`, because its digest is parent authority.
- Preserve untracked `img`.
- Use TDD for every behavior change.
- Run one DGX GPU process at a time and retain the original process terminal.

---

### Task 1: Protocol and schedule authority

**Files:**
- Create: `src/sfora/unicom_finish_protocol.py`
- Test: `tests/test_unicom_finish_protocol.py`

**Interfaces:**
- Produces: `FinishArm`, `validate_finish_config`, `build_finish_batches`, `capture_rng_state`, `restore_rng_state`, and canonical schedule digests.

- [ ] Write failing tests for exact A/B/C inventory, seed 3, epochs 5--8, 161 steps, A sampler parity, B/C byte-identical schedules, sparse identities, and RNG save/restore.
- [ ] Run `uv run --locked pytest -q tests/test_unicom_finish_protocol.py` and require missing-interface failures.
- [ ] Implement only the validated protocol and deterministic schedule helpers.
- [ ] Rerun the focused tests and Ruff; commit the protocol slice.

### Task 2: Paired evaluation evidence

**Files:**
- Create: `src/sfora/unicom_finish_evidence.py`
- Test: `tests/test_unicom_finish_evidence.py`

**Interfaces:**
- Produces: canonical evaluation bundles and aggregate recomputation from ordered per-query rows.

- [ ] Write failing tests for query/gallery identities, descriptor hashes, AP@R/recall reconstruction, shuffled/duplicate/missing rows, nonfinite descriptors, and geometry drift.
- [ ] Run the exact focused RED.
- [ ] Implement canonical no-replace bundle writing and strict reload/recomputation.
- [ ] Rerun focused tests and Ruff; commit the evidence slice.

### Task 3: Controlled runner

**Files:**
- Create: `scripts/run_unicom_finish_ablation.py`
- Test: `tests/test_run_unicom_finish_ablation.py`

**Interfaces:**
- Consumes: Task-1 config/schedules and Task-2 evidence writer.
- Produces: `unicom-finish-ablation-result-v1` plus raw/EMA model artifacts.

- [ ] Write failing tests for strict CLI authority, parent bindings, restored optimizer/scheduler/scaler/EMA state, A/B/C loss paths, classifier-gradient semantics, update counters, evaluation RNG isolation, and no-overwrite publication.
- [ ] Run the focused RED and confirm failures are missing runner behavior.
- [ ] Implement the minimal legacy-trainer adapter without changing the legacy trainer.
- [ ] Run focused tests, Ruff, bytecode compilation, and `git diff --check`; commit.

### Task 4: Causal evaluator

**Files:**
- Create: `scripts/evaluate_unicom_finish_ablation.py`
- Test: `tests/test_evaluate_unicom_finish_ablation.py`

**Interfaces:**
- Consumes: three arm results and paired bundles.
- Produces: canonical contrasts, bootstrap intervals, win/tie/loss, discordant recall counts, and GO/CLOSE decision.

- [ ] Write failing tests for all authority mutations, exact `C-A`/`C-B` deltas, identity-clustered resampling seed `20260904`, aggregate reconstruction, and gate precedence.
- [ ] Run the focused RED.
- [ ] Implement validation and evaluation with no adaptive thresholds.
- [ ] Run focused tests and static checks; commit.

### Task 5: Zero-GPU preregistration and assurance

**Files:**
- Create: `scripts/build_unicom_finish_ablation_config.py`
- Test: `tests/test_build_unicom_finish_ablation_config.py`
- Modify: causal-panel design and result ledger only for authenticated diagnostics.

- [ ] Build the config from the exact parent, partition and source identities; bind A/B/C schedule hashes and the source-image census.
- [ ] Verify the census against 20,650 optimization images, 3,200 identities and 644 steps.
- [ ] Run focused tests, `uv run --locked pytest -q`, Ruff, `py_compile`, and `git diff --check` once.
- [ ] Commit and push the clean implementation/config; deploy that exact commit to DGX.

### Task 6: Serial DGX execution

**Files:**
- No repository edits during execution.

- [ ] Run one replay canary and require restored-state and first-step evidence.
- [ ] Run seed-3 arms A, B, C serially under the registered pressure/progress envelope.
- [ ] Authenticate every result and evidence artifact immediately after its original process exits.
- [ ] Run the causal evaluator only after all three terminal outcomes exist.
- [ ] If status is CLOSE, record the negative causal result and stop. If GO, preregister the second-phase soft-triplet/full-width panel before any further GPU work.

### Task 7: Final evidence and performance gate

**Files:**
- Create: `docs/unicom_finish_causal_panel_result_2026-09-04.md`

- [ ] Record exact sources, hashes, paired results, resource observations, limitations and decision.
- [ ] If GO, implement performance candidates one at a time behind exact replay RED/GREEN tests.
- [ ] Run the dependency-complete test and static gates, commit every verified repair, push `master`, and verify local/remote SHA equality.
