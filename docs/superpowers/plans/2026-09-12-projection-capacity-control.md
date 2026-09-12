# Projection Capacity Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the observed direct 768-to-128 projection gain comes from retained
teacher information or merely from a larger optimization parameterization.

**Architecture:** Add a generic foldable two-factor affine module whose 128-to-384-to-128
parameterization has exactly 98,432 trainable scalars, matching the direct 768-to-128 affine head.
Extend the authenticated projection diagnostic to compare restricted, parameter-matched
factorized, and direct arms from the same effective initial map; publish only single-affine
deployment checkpoints.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, Ruff, mypy.

**Spec:** `docs/sop_projection_parameterization_preregistration_2026-09-12.md`, amended by the
post-result capacity-confound note in
`docs/positive_coverage_adaptation_result_2026-09-12.md`.

## Global Constraints

- Work only in the Sfora repository.
- Keep the encoder features, mean-logit objective, positives, frozen negative identities,
  schedule, update count, clipping, evaluator, and seeds unchanged.
- Initialize every arm to the same effective affine map and validate code equality.
- The factorized arm must see only the frozen raw 128-dimensional base affine output, before
  normalization, so its learned bias remains exactly foldable into the deployed base head.
- Its trainable parameter count must exactly equal the direct affine arm: `98,432`.
- Fold every trained arm to one authenticated 768-to-128 affine deployment head before scoring.
- Keep all results `claim_eligible=false`; SOP official test remains untouched.

---

### Task 1: Generic foldable affine factorization

**Files:**
- Create: `src/sfora/foldable_linear.py`
- Create: `tests/test_foldable_linear.py`
- Modify: `src/sfora/__init__.py`

**Interfaces:**
- Produces: `FoldableLinear(input_dim: int, hidden_dim: int, output_dim: int)`.
- Produces: `initialize_repeated_identity()` for the exact square repeated-frame initialization.
- Produces: `fold() -> tuple[torch.Tensor, torch.Tensor]` returning one affine weight and bias.

- [ ] **Step 1: Write failing construction, initialization, forward/fold, gradient, shape, and
  parameter-count tests.** The 128/384/128 instance must contain `98,432` trainable scalars and
  its initialized output must match its input exactly within the registered float tolerance.

- [ ] **Step 2: Run `pytest -q tests/test_foldable_linear.py` and preserve the missing-module RED.**

- [ ] **Step 3: Implement the minimal validated module.** Initialize both factors as three repeated
  identity frames scaled by `1/sqrt(3)`, initialize output bias to zero, and compute folded weight
  as `second.weight @ first.weight` with the second bias unchanged.

- [ ] **Step 4: Export the type from `sfora.__init__` and rerun the focused test to GREEN.**

### Task 2: Three-arm authenticated diagnostic

**Files:**
- Modify: `scripts/run_sop_projection_parameterizations.py`
- Modify: `scripts/positive_coverage_artifacts.py`
- Modify: `tests/test_run_sop_projection_parameterizations.py`
- Modify: `tests/test_positive_coverage_artifacts.py`

**Interfaces:**
- Consumes: `FoldableLinear` from Task 1.
- Produces: v2 arms `restricted_adapter`, `factorized_adapter`, and `direct_projection`.
- Produces: three self-contained deployed affine checkpoints plus a canonical complete receipt.

- [ ] **Step 1: Add failing tests.** Lock exact three-arm initialization equality, exact parameter
  counts, factorized input authority, one-affine folding, role-specific state validation,
  checkpoint no-clobber behavior, and a decision that reports how much of the direct-minus-linear
  gain the factorized arm closes.

- [ ] **Step 2: Run the two focused test files and preserve the intended RED.**

- [ ] **Step 3: Implement the minimal v2 path.** Use the same learning rate as the direct arm,
  train against only raw base affine outputs, fold factorization then base head, score actual deployment codes,
  and record factorization width, trainable count, displacement, effective-head digest, and
  closure fraction. Do not alter mining or the objective.

- [ ] **Step 4: Run focused tests to GREEN.**

### Task 3: Verification, evidence, and DGX screen

**Files:**
- Modify after terminal evidence: `docs/positive_coverage_adaptation_result_2026-09-12.md`

**Interfaces:**
- Consumes: the v2 driver and exact authenticated SOP inputs.
- Produces: five seed receipts and a causal classification.

- [ ] **Step 1: Run Ruff format/check, py_compile, mypy, and `git diff --check`.** Distinguish known
  imported legacy mypy findings from new errors.

- [ ] **Step 2: Run the full Python suite once and require zero failures.**

- [ ] **Step 3: Commit and push the exact Sfora slice to canonical `master`; verify local, tracking,
  and remote refs plus a clean worktree.**

- [ ] **Step 4: Deploy that exact commit to the DGX, run focused tests against checkout `src`, then
  run seed 0 with no-clobber artifacts.**

- [ ] **Step 5: If valid, run frozen seeds 1 through 4 unchanged.** Classify information as the
  leading cause only if the factorized control closes less than 60% of the direct-minus-linear
  mAP@R gap and direct retains at least `0.005` mAP@R advantage over factorized with a positive
  class-cluster lower bound. Otherwise classify optimization parameterization as sufficient or
  mixed. No outcome authorizes SOP official-test reuse.

- [ ] **Step 6: Authenticate receipts/checkpoints, update the evidence document, verify the doc
  diff, commit, push, and notify the operator.**
