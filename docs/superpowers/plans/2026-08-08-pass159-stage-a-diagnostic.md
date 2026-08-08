# Pass159 Stage-A Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the preregistered, artifact-bound four-seed Stage-A causal screen for norm-ranked cotangent transplant.

**Architecture:** A single diagnostic module exposes pure NumPy geometry, selection, bootstrap, and verdict functions that tiny tests can exercise. Its CLI loads the frozen JSON manifest, fails closed on every digest/config/order/retrieval mismatch, computes one identity row at a time without an all-pairs matrix, and atomically writes an auditable JSON result. All tests and the full diagnostic run on the DGX; the devbox runs only syntax/static checks.

**Tech Stack:** Python 3.11, NumPy, PyTorch checkpoint loading/autograd test oracle, pytest, existing Sfora artifact schemas.

## Global Constraints

- The immutable mechanism, constants, hashes, controls, and thresholds are those committed in `docs/pass159_gradient_transplant_search_2026-08-08.md` and `docs/pass159_stage_a_manifest.json`.
- Official In-Shop query/gallery data may be used only for artifact binding, never candidate selection or scoring.
- No wall-clock time is a scientific outcome; memory is bounded by chunking and no `N x N` matrix.
- Stage A can return `PASS_ONWARD`, `FAIL`, or `UNRESOLVED`; it cannot clear Gate 1 or authorize benchmark training.
- No full-dataset execution or Torch-heavy test runs occur on the memory-limited devbox.

---

### Task 1: Pure geometry and deterministic selection

**Files:**
- Create: `scripts/diagnose_pass159_cotangent_stage_a.py`
- Create: `tests/test_diagnose_pass159_cotangent_stage_a.py`

**Interfaces:**
- Produces: `angular_proxy_anchor_cotangent`, `parallel_transport`, `smooth_margin_gradient`, `partition_identity`, `select_controls`.

- [x] Write tiny tests proving the analytic cotangent equals an autograd oracle and is tangent; parallel transport preserves tangency/norm and rejects antipodes; smooth margin/top-32 matches a dense fixture; and selection is input-order invariant with no support leakage.
- [x] Run only `tests/test_diagnose_pass159_cotangent_stage_a.py` on DGX and verify RED because the module/functions do not exist.
- [x] Implement the minimal pure functions, including every frozen hash/tie/zero guard.
- [x] Rerun the same DGX test and verify GREEN.
- [x] Commit the pure diagnostic core and tests.

### Task 2: Artifact binding and seed computation

**Files:**
- Modify: `scripts/diagnose_pass159_cotangent_stage_a.py`
- Modify: `tests/test_diagnose_pass159_cotangent_stage_a.py`

**Interfaces:**
- Produces: `load_bound_seed(manifest_entry, seed)` and `compute_seed_rows(bound_seed)`.

- [x] Add synthetic artifact tests where the valid bundle passes, then independently corrupt a digest, label order, embedded digest, config, checkpoint step, reconstructed embedding, and official R@1 and require each to fail closed.
- [x] Run the focused DGX test and verify the new cases fail for missing behavior.
- [x] Implement SHA-256 validation, schema/config invariants, head reconstruction, exact row-order bridge, chunked official R@1 binding, and training-only row computation.
- [x] Rerun the focused test and verify GREEN.
- [x] Commit artifact binding and per-seed computation.

### Task 3: Cluster bootstrap, frozen verdict, and CLI

**Files:**
- Modify: `scripts/diagnose_pass159_cotangent_stage_a.py`
- Modify: `tests/test_diagnose_pass159_cotangent_stage_a.py`

**Interfaces:**
- Produces: `clustered_verdict(identity_rows, seed=159, replicates=1000)` and CLI arguments `--manifest`, `--output`, `--top-k`, `--bootstrap-replicates`.

- [x] Add boundary tests for pooled strongest-control selection, joint identity resampling across seeds, simultaneous max-control bootstrap, noncollapse thresholds, and all three decision states.
- [x] Run the focused DGX test and verify RED.
- [x] Implement deterministic bootstrap/verdict, auditable per-identity/per-seed/pooled JSON, and atomic output.
- [x] Rerun focused tests plus `python -m py_compile` on DGX and verify GREEN.
- [x] Commit and push the complete diagnostic.

### Task 4: Frozen four-seed execution and verdict

**Files:**
- Create after execution: `docs/pass159_stage_a_result_2026-08-08.md`
- Modify after execution: `docs/method_search_verdict.md`

**Interfaces:**
- Consumes: the committed manifest and diagnostic.
- Produces: remote JSON artifact plus a mechanism-level durable verdict.

- [x] Sync the exact committed script/test/manifest to `/home/riomus/group-learning` without touching the active GPU run.
- [x] Run the diagnostic on DGX CPU with bounded BLAS/OpenMP threads and retain the original process/log until exit.
- [x] Check every artifact-binding field, exclusion count, control mean, bootstrap interval, seed delta, and frozen criterion before accepting the decision.
- [x] If `FAIL`, record why and spend no GPU. If `UNRESOLVED`, freeze the cheapest discriminating follow-up before running it. If `PASS_ONWARD`, implement only the preregistered bounded Stage-B VJP test.
- [x] Commit and push the result/verdict documents.
