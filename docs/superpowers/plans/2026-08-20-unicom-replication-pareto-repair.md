# UniCOM Replication Pareto Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a strict v2 six-seed summary that separates fixed-budget quality, iso-quality compute, and Pareto non-domination while adding trajectory-neutral initialization evidence for seeds 2..6.

**Architecture:** Preserve immutable seed-1 pair-v1 bytes. Add atomic initialization receipts and pair-v2 evidence prospectively, then make summary-v2 accept exactly one historical v1 pair plus five future v2 pairs. Every decision is recomputed from authenticated evidence; raw wall time and seed-1 proxy limitations remain explicit and non-gating.

**Tech Stack:** Python 3.12+, PyTorch 2.12, NumPy 2.5, pytest, Ruff, strict atomic JSON publication.

**Spec:** `docs/unicom_ema_imprint_replication_pareto_amendment_2026-08-20.md`

## Global Constraints

- Do not alter seed-1 report bytes or replace any valid run.
- Do not change training constants, data order, selected cell, seeds, epochs, objective, or quality thresholds.
- Use RED before every production behavior change and preserve exact failure evidence.
- Keep fixed-epoch quality, fixed-epoch compute, and iso-quality compute as separate fields and predicates.
- Raw wall time is descriptive only, as in the original preregistration.
- Initialization receipts must not perturb tensors, RNG, data order, model mode, or BatchNorm buffers.
- No seed-2 GPU run starts until source review is READY and a fresh commit is deployed.

---

### Task 1: Summary-v2 authority and operating-point semantics

**Files:**
- Modify: `scripts/summarize_unicom_ema_imprint_replication.py`
- Modify: `tests/test_summarize_unicom_ema_imprint_replication.py`

**Interfaces:**
- Produces: `load_selection_authority(path: Path) -> dict[str, str]`.
- Produces: `compute_cost_rows(report: Mapping[str, object]) -> dict[str, object]`.
- Extends: `summarize_replications(reports, *, selection_authority)`.
- Validates: exact summary-v2 constants rather than self-supplied authority.

- [ ] **Step 1: Write selection-authority REDs**

Add a literal five-key authority fixture. Assert direct and JSON-round-tripped summaries
reject each mutated key. Exercise the public CLI with the real seed-0 report and assert
that wrong path, SHA, gate, recording Git blob, non-ancestor commit, symlink, and a
different current working directory fail before publication.

```python
authority = {
    "path": "reports/generated/unicom_ema_imprint_factorial_88604a4_seed0.json",
    "sha256": "c0666a68e70990115d80e8dc06a9f94efe83156a3fddd50f36bdbf2b3b8cd217",
    "recording_commit": "81f3f48c374d14b5a91bbeba7a1fec2fb0a4a2d6",
    "selected_cell": "imprinted_raw",
    "decision": "PROMOTE",
}
summary = module.summarize_replications(reports, selection_authority=authority)
mutated = copy.deepcopy(summary)
mutated["selection_authority"]["decision"] = "CLOSE"
with pytest.raises(ValueError, match="selection authority"):
    module.validate_summary(mutated)
```

- [ ] **Step 2: Run the authority RED**

Run: `.venv/bin/pytest -q tests/test_summarize_unicom_ema_imprint_replication.py -k 'selection_authority or summary_v2'`

Expected: fail because summary-v1 has no frozen authority and the CLI has no selection input.

- [ ] **Step 3: Implement frozen authority and summary-v2 schema**

Add literal path/SHA/recording-commit/cell/decision constants. Resolve the path from
`Path(__file__).resolve().parents[1]`, authenticate exact bytes against both worktree and
`git show`, run the strict factorial validator, require the three gate values, and embed
the literal object. Bump `SUMMARY_SCHEMA` to
`unicom-ema-imprint-replication-summary-v2`; update every existing summary test call to
pass the test-owned literal authority.

- [ ] **Step 4: Write operating-point REDs**

Use a hand-derived non-monotone seed-1 row and assert exact first epochs `12` and `8`
and speedup `1.5`. Assert raw wall-time drift changes only descriptive rows. Assert
fixed-epoch and iso-quality compute are distinct. Assert seed-1 compute totals are null
with exact historical status. Assert each future seed independently rejects higher
iso-quality compute, peak memory, or checkpoint storage, and rejects either larger or
smaller deployment storage. Assert fixed-epoch higher compute is reported without being
mislabelled dominance.

```python
summary = module.summarize_replications(reports, selection_authority=authority)
assert summary["first_quality_epochs"][0] == {
    "seed": 1, "random_raw": 12, "imprinted_raw": 8, "speedup": 1.5
}
assert summary["costs"]["fixed_epoch_compute_seconds"][0] == {
    "seed": 1, "random_raw": None, "imprinted_raw": None
}
reports[1]["imprinted_raw"]["training_seconds"] = 1_000_000.0
assert module.summarize_replications(
    reports, selection_authority=authority
)["claim_supported"] is True
```

- [ ] **Step 5: Run the operating-point RED**

Run: `.venv/bin/pytest -q tests/test_summarize_unicom_ema_imprint_replication.py -k 'non_monotone or operating_point or contaminated_wall or per_seed_cost or historical_initialization'`

Expected: fail because summary-v1 sums costs and has no mixed evidence or separated predicates.

- [ ] **Step 6: Implement minimal derived semantics**

Accept exactly pair-v1 at seed 1 and pair-v2 at seeds 2..6. For v2 rows compute the two
registered formulas from arm evidence. Keep training wall time, latency, profiler,
memory, checkpoint, and deployment rows. Recompute per-seed gates, fixed-epoch
non-domination, future iso-quality conjunction, and the final trajectory-frontier
claim. Add exact initialization-evidence rows, including seed 1's null historical row.

- [ ] **Step 7: Verify Task 1**

Run: `.venv/bin/pytest -q tests/test_summarize_unicom_ema_imprint_replication.py`

Run: `.venv/bin/ruff check scripts/summarize_unicom_ema_imprint_replication.py tests/test_summarize_unicom_ema_imprint_replication.py`

Run: `.venv/bin/python -m py_compile scripts/summarize_unicom_ema_imprint_replication.py tests/test_summarize_unicom_ema_imprint_replication.py`

### Task 2: Pure initialization receipt construction and atomic publication

**Files:**
- Modify: `scripts/train_unicom_inshop.py`
- Modify: `tests/test_train_unicom_inshop.py`

**Interfaces:**
- Produces: `classifier_initialization_receipt(...) -> dict[str, object]`.
- Produces: `rng_state_hashes() -> dict[str, object]`.
- Persists: `<output-dir>/initialization-receipt.json` using no-clobber publication.

- [ ] **Step 1: Write receipt-purity REDs**

With independently literal expected hashes, assert the exact 11-key schema, FP32
row-major classifier bytes/shape/dtype, trainer digest, seed/arm/algorithm, exact loader
length, positive finite synchronized duration, and four RNG domains. Snapshot classifier,
model parameters, BatchNorm buffers, model mode, Python/NumPy/Torch CPU/all CUDA RNG,
and sampler/data-generator seeds before receipt construction and require byte equality
afterward. Mutate every key/type/hash/order and require rejection.

```python
before = snapshot_process_and_model(model, classifier)
receipt = module.classifier_initialization_receipt(
    seed=2,
    classifier_init="random",
    classifier=classifier,
    optimizer_steps_per_epoch=161,
    initialization_seconds=1.25,
    trainer_sha256="a" * 64,
)
assert tuple(receipt) == EXPECTED_INITIALIZATION_KEYS
assert receipt["classifier_tensor_sha256"] == EXPECTED_CLASSIFIER_SHA256
assert snapshot_process_and_model(model, classifier) == before
```

- [ ] **Step 2: Run receipt RED**

Run: `.venv/bin/pytest -q tests/test_train_unicom_inshop.py -k 'initialization_receipt or rng_state_hashes'`

Expected: fail because the receipt API does not exist.

- [ ] **Step 3: Implement pure receipt helpers**

Hash domain-separated deterministic encodings of Python state, NumPy MT state, Torch
CPU state, and each CUDA-device state. Hash contiguous CPU classifier bytes. Validate
exact built-in types and order. Do not consume RNG or alter device/model state.

- [ ] **Step 4: Write atomic lifecycle REDs**

Cover unnamed-inode write, strict pre/post-publication reload, mode `0600`, directory
fsync, destination race, rollback, pre-existing foreign temp, no-clobber, and resume
loading. Assert fresh publication occurs after initialization/loader-length discovery
and before optimizer construction; inject a constructor sentinel that fails if the
receipt is absent.

```python
module.write_initialization_receipt_atomic(receipt, output)
module.validate_initialization_receipt(
    module.strict_json_object(output.read_bytes())
)
assert stat.S_IMODE(output.stat().st_mode) == 0o600
with pytest.raises(FileExistsError):
    module.write_initialization_receipt_atomic(receipt, output)
```

- [ ] **Step 5: Implement trainer integration**

Synchronize CUDA around the complete initializer, construct the loader before the
optimizer without iterating it, create/publish the receipt, then construct optimizer
and continue the unchanged trajectory. On resume, authenticate the existing receipt
instead of replacing it.

- [ ] **Step 6: Verify Task 2**

Run: `.venv/bin/pytest -q tests/test_train_unicom_inshop.py -k 'initialization or classifier or rng or checkpoint'`

Run: `.venv/bin/ruff check scripts/train_unicom_inshop.py tests/test_train_unicom_inshop.py`

### Task 3: Measurement-v2 and pair-v2 binding

**Files:**
- Modify: `scripts/evaluate_unicom_ema_imprint_replication.py`
- Modify: `tests/test_evaluate_unicom_ema_imprint_replication.py`

**Interfaces:**
- Accepts: immutable measurement-v1 only for seed 1.
- Accepts: measurement-v2 plus authenticated initialization receipt for seeds 2..6.
- Produces: pair-v2 for seeds 2..6 with four added initialization fields per arm.

- [ ] **Step 1: Write exact dual-schema REDs**

Build complete literal v1 and v2 receipts. Require v1 only at seed 1 and v2 only at
seeds 2..6. Mutate initialization path/digest, classifier digest, steps, duration,
canonical RNG digest, seed, arm, and trainer digest. Require copied cross-run receipts
to fail. Require random/imprinted post-init RNG objects to be exactly equal within a
future seed.

```python
pair = module.build_replication_pair(
    **future_inputs,
    random_initialization_receipt=random_receipt,
    imprinted_initialization_receipt=imprinted_receipt,
)
assert pair["schema_version"] == "unicom-ema-imprint-replication-pair-v2"
assert pair["random_raw"]["post_initialization_rng_sha256"] == pair[
    "imprinted_raw"
]["post_initialization_rng_sha256"]
mutated = copy.deepcopy(imprinted_receipt)
mutated["post_initialization_rng"]["torch_cpu_sha256"] = "f" * 64
with pytest.raises(ValueError, match="post-initialization RNG"):
    module.build_replication_pair(
        **future_inputs,
        random_initialization_receipt=random_receipt,
        imprinted_initialization_receipt=mutated,
    )
```

- [ ] **Step 2: Run measurement RED**

Run: `.venv/bin/pytest -q tests/test_evaluate_unicom_ema_imprint_replication.py -k 'measurement_v2 or initialization_binding or post_initialization_rng'`

Expected: fail because only measurement-v1 and pair-v1 exist.

- [ ] **Step 3: Implement dual loader and pair-v2**

Load and strictly validate the initialization receipts passed by new random/imprinted
CLI arguments. Validate v2 measurement evidence and its transitive bindings. Preserve
the existing v1 validation code byte-for-byte where practical. Build pair-v2 only for
future seeds, append exact arm fields `optimizer_steps_per_epoch`,
`initialization_seconds`, `initialization_receipt_sha256`, and
`post_initialization_rng_sha256`, and strict-reload atomic output.

- [ ] **Step 4: Verify Task 3**

Run: `.venv/bin/pytest -q tests/test_evaluate_unicom_ema_imprint_replication.py`

Run: `.venv/bin/ruff check scripts/evaluate_unicom_ema_imprint_replication.py tests/test_evaluate_unicom_ema_imprint_replication.py`

### Task 4: Cumulative assurance and independent review

**Files:**
- Modify only files required by confirmed review findings.

**Interfaces:**
- Produces: reviewed source commit suitable for a detached GPU checkout.

- [ ] **Step 1: Run affected assurance**

Run: `.venv/bin/pytest -q tests/test_train_unicom_inshop.py tests/test_evaluate_unicom_ema_imprint_replication.py tests/test_summarize_unicom_ema_imprint_replication.py`

Run: `.venv/bin/ruff check scripts/train_unicom_inshop.py scripts/evaluate_unicom_ema_imprint_replication.py scripts/summarize_unicom_ema_imprint_replication.py tests/test_train_unicom_inshop.py tests/test_evaluate_unicom_ema_imprint_replication.py tests/test_summarize_unicom_ema_imprint_replication.py`

Run: `.venv/bin/python -m py_compile scripts/train_unicom_inshop.py scripts/evaluate_unicom_ema_imprint_replication.py scripts/summarize_unicom_ema_imprint_replication.py`

Run: `git diff --check`

- [ ] **Step 2: Run one repository-wide gate**

Run one serial `.venv/bin/pytest -q`. Do not overlap it with another test/build process.

- [ ] **Step 3: Request adversarial review**

Start exactly one review consultation with `models=["opus", "gpt-5.6-sol"]`. Require
no Critical/Important findings on operating-point semantics, strict schemas, historical
honesty, authority, timing/RNG purity, v1/v2 routing, and test mutation power.

- [ ] **Step 4: Resolve findings with focused RED/GREEN**

Verify each finding against source, add a failing regression, make the minimal repair,
rerun the focused selector, then rerun the affected gate once after the diff stabilizes.

### Task 5: Freeze, deploy, and resume confirmation

**Files:**
- Add only immutable result artifacts after each successful run.

**Interfaces:**
- Produces: exact source commit, fresh detached DGX checkout, seed-2..6 pair-v2 reports,
  and final summary-v2.

- [ ] **Step 1: Commit and push exact scope**

Commit the reviewed docs follow-up, source/tests, and each immutable result separately.
Verify commits with `git log`, push `devbox/similarity-ghc`, and deploy the reviewed
source commit to a fresh detached checkout.

- [ ] **Step 2: Resume seeds 2..6 serially**

For each seed require idle GPU and absent destinations; run random then imprinted with
identical recipe except classifier initialization; authenticate initialization and
measurement receipts; run the paired evaluator once; validate the pair-v2 JSON offline;
preserve every valid outcome. Never replace a failed seed.

- [ ] **Step 3: Build the final summary once**

After all six reports exist, run summary-v2 once with the exact seed-0 selection path.
Strict-reload it, report fixed-budget quality, epoch speedups, fixed-epoch compute,
iso-quality compute, raw time, memory, inference, and storage separately, and state the
exact trajectory-frontier decision without an official-protocol or SOTA claim.
