# Pass 205 RDGC Authority Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair RDGC's historical authority, literature-audit binding, and contaminated preliminary decision surface without changing its scalar operator, controls, panel, or no-training execution contract.

**Architecture:** Keep the existing import-safe RDGC diagnostic and strict result validator, but replace caller-shaped historical records with values derived through the authenticated Pass 200 production validators. Bind the reviewed audit/design/amendment chain literally, remove only the direct full-gain decision predicates while retaining every descriptive endpoint metric and the fresh paired count-gain contrast, then create a direct-child manifest-only handoff and run one DGX process only after independent review.

**Tech Stack:** Python 3.12.3, PyTorch `torch.func`, NumPy PCG64(201), strict insertion-ordered JSON, SHA-256/Git blob authentication, pytest, Ruff, `py_compile`, atomic hard-link publication.

## Global Constraints

- Original candidate authority is `docs/pass205_rdgc_candidate_2026-08-10.md` at `30d533e532d0f22c8b1e474987001685a4aa3488`, SHA-256 `2a86f11f8d6a4563610b0585db74c372903bdbf7deabd580fa929114fda2af0f`.
- Original implementation plan is `docs/superpowers/plans/2026-08-10-pass205-rdgc-stage-b.md` at `c1e49b13c08f853ae17d5b8b48be1aa7b8a4bc11`, SHA-256 `20915982228bd4a17f1260952fe184d9e09b27b9b28165b5931bad843872c7ed`.
- Repair design authority is `docs/superpowers/specs/2026-08-11-pass205-rdgc-authority-repair-design.md` at `2f2ea249a754a1fb4186ba55939d95c85de747a8`, SHA-256 `63be862bd099703eb3189d7317e766eb7900fe6a855d130409a32775d1008144`.
- Literature audit authority is `docs/pass205_rdgc_gate2_primary_audit_2026-08-11.md` at `9ae137f3af0558728554c6af865fe96d6bf10060`, SHA-256 `6f99134b905213049f0506b19b1acbcc7e5760b8412a9dc790e2c085b4f8573b`, verdict `LIVE-NARROW`.
- Scientific/authority amendment is `docs/pass205_rdgc_authority_amendment_2026-08-11.md` at `c7fae7683533e740660d7e860bd313be07a41014`, SHA-256 `41b53d1955c4a800bfd2f901e35167e4d483e175c949353984c0ae75a69c7228`.
- Reopened source baseline is `291ccbfbe322565c71c1e08317ca6e5c914b74a9`; its diagnostic/test SHA-256 values are `d3f41a169d2b1b69cefbd8a4677c2820328d5ebf590e2714240089f3377a1d94` and `e2ac85be9dbb3381ef4bf661f1c8c9b43b639de4b67ff53e1519e6c17183d694`.
- RDGC is the magnitude-only ablation of RSTA's registered composite penalty. Never describe it as wholly independent or as continuation of a validated directional mechanism.
- Keep `E_r,n` for `[1,8,32,180]`. Direct `E_r,180` endpoint levels and summaries are descriptive only. Remove exactly `survives_full_gain` and `close_full_gain`; retain exact paired `C_r = E_r,180 - E_r,8` through `survives_count_gain` and `close_count_gain`.
- Do not change the scalar formula, epsilon, controls, selection, graph schedule, panel thresholds, bootstrap, CLOSE-before-PASS precedence, no-training boundary, inference behavior, exact 33-path source order, command, or atomic writer mechanics.
- Never open or parse the old RSTA scientific artifact. The provenance-only validation receipt may be transferred and read only through its authenticated verifier.
- Before Torch/model/artifact access, invoke Pass 200 production `validate_scientific_execution_source` and `validate_historical_binding_receipt`; independently derive every seed record from their validated outputs.
- Source aggregate scope after this plan commit is exactly `scripts/diagnose_pass205_rdgc_stage_b.py` and `tests/test_diagnose_pass205_rdgc_stage_b.py`. The handoff direct child changes only `docs/pass205_rdgc_stage_b_manifest.json`.
- Protect `HANDOFF_BRIEF.md`, `RSPG_SPECDEFECT.md`, and `RSPG_TASK.md`.
- No RDGC GPU process, result, manifest handoff, or candidate value exists yet. Stop on any unresolved Critical/Important review finding or authority mismatch.

The commit containing this plan is `P_G2`; because a document cannot self-bind,
its exact commit and SHA-256 are derived immediately after this file is committed
alone and are written as literals into production before the first source GREEN.
The final independently reviewed source descendant is `V_G2`. Its direct-child,
manifest-only handoff is `HV_G2`. These are derived names for exact Git objects,
not caller inputs or values that may remain symbolic in production.

## File Structure

- Modify `scripts/diagnose_pass205_rdgc_stage_b.py`: literal repair authorities, repaired preliminary predicates, derived historical seed records, authenticated audit/RSTA receipt relations, result/future-manifest validation, and new source-chain gate.
- Modify `tests/test_diagnose_pass205_rdgc_stage_b.py`: focused RED/GREEN tests, exhaustive recursive mutations, production-validator sentinels, unchanged scientific-surface regression, and future-manifest fixtures.
- Create later, alone: `docs/pass205_rdgc_stage_b_manifest.json`.
- Create at most once after handoff review: the result path formed from the exact
  40-hex `HV_G2` plus suffix `-rdgc-stage-b.json` under
  `reports/generated/pass205_rdgc_stage_b/`.

---

### Task 1: Freeze the Reviewed Repair Authorities

**Files:**
- Modify: `scripts/diagnose_pass205_rdgc_stage_b.py:79-116`
- Modify: `tests/test_diagnose_pass205_rdgc_stage_b.py`

**Interfaces:**
- Consumes: the exact candidate, original plan, repair design, audit, and amendment bindings in Global Constraints.
- Produces: literal `ORIGINAL_PLAN_*`, `REPAIR_DESIGN_*`, `LITERATURE_AUDIT_*`, `AUTHORITY_AMENDMENT_*`, and repair `PLAN_*` constants used by all later provenance validation.

- [ ] **Step 1: Add a RED literal-authority test**

```python
def test_repair_authority_literals_and_chain_are_exact() -> None:
    assert (_MODULE.ORIGINAL_PLAN_PATH, _MODULE.ORIGINAL_PLAN_COMMIT,
            _MODULE.ORIGINAL_PLAN_SHA256) == (
        "docs/superpowers/plans/2026-08-10-pass205-rdgc-stage-b.md",
        "c1e49b13c08f853ae17d5b8b48be1aa7b8a4bc11",
        "20915982228bd4a17f1260952fe184d9e09b27b9b28165b5931bad843872c7ed",
    )
    assert _MODULE.REPAIR_DESIGN_COMMIT == "2f2ea249a754a1fb4186ba55939d95c85de747a8"
    assert _MODULE.LITERATURE_AUDIT_COMMIT == "9ae137f3af0558728554c6af865fe96d6bf10060"
    assert _MODULE.AUTHORITY_AMENDMENT_COMMIT == "c7fae7683533e740660d7e860bd313be07a41014"
```

- [ ] **Step 2: Run the RED test**

Run: `.venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k repair_authority_literals_and_chain`

Expected: FAIL because the repair authority constants do not exist and `PLAN_*` still binds the obsolete plan.

- [ ] **Step 3: Add exact constants and the literal chronology**

```python
ORIGINAL_PLAN_PATH = "docs/superpowers/plans/2026-08-10-pass205-rdgc-stage-b.md"
ORIGINAL_PLAN_COMMIT = "c1e49b13c08f853ae17d5b8b48be1aa7b8a4bc11"
ORIGINAL_PLAN_SHA256 = "20915982228bd4a17f1260952fe184d9e09b27b9b28165b5931bad843872c7ed"
REPAIR_DESIGN_PATH = "docs/superpowers/specs/2026-08-11-pass205-rdgc-authority-repair-design.md"
REPAIR_DESIGN_COMMIT = "2f2ea249a754a1fb4186ba55939d95c85de747a8"
REPAIR_DESIGN_SHA256 = "63be862bd099703eb3189d7317e766eb7900fe6a855d130409a32775d1008144"
LITERATURE_AUDIT_PATH = "docs/pass205_rdgc_gate2_primary_audit_2026-08-11.md"
LITERATURE_AUDIT_COMMIT = "9ae137f3af0558728554c6af865fe96d6bf10060"
LITERATURE_AUDIT_SHA256 = "6f99134b905213049f0506b19b1acbcc7e5760b8412a9dc790e2c085b4f8573b"
AUTHORITY_AMENDMENT_PATH = "docs/pass205_rdgc_authority_amendment_2026-08-11.md"
AUTHORITY_AMENDMENT_COMMIT = "c7fae7683533e740660d7e860bd313be07a41014"
AUTHORITY_AMENDMENT_SHA256 = "41b53d1955c4a800bfd2f901e35167e4d483e175c949353984c0ae75a69c7228"
```

Immediately after this plan-only commit, run `git rev-parse HEAD` and `sha256sum
docs/superpowers/plans/2026-08-11-pass205-rdgc-authority-repair.md`. During this
task, set `PLAN_PATH`, `PLAN_COMMIT`, and `PLAN_SHA256` to that literal path and
those exact observed values; do not leave a symbolic value in production.

- [ ] **Step 4: Run the focused test and commit only source/test later with Task 5**

Run: `.venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k repair_authority_literals_and_chain`

Expected: PASS.

---

### Task 2: Repair the Preliminary Decision Without Erasing Descriptive Evidence

**Files:**
- Modify: `scripts/diagnose_pass205_rdgc_stage_b.py:353-467,784-820,2035-2070,2999-3060`
- Modify: `tests/test_diagnose_pass205_rdgc_stage_b.py:450-590`

**Interfaces:**
- Consumes: existing `summarize_preliminary_rows`, `decide_preliminary`, and persisted row/aggregate schemas.
- Produces: exact eight-key predicate object, four-key CLOSE order, one-key internal close evidence, unchanged descriptive full-gain aggregates.

- [ ] **Step 1: Write RED tests for removal and retention**

```python
def test_repair_removes_direct_full_gain_decisions_but_retains_descriptive_metrics() -> None:
    rows = _preliminary_decision_rows(full_gain_seed_values=(0.0,) * 4)
    summary = _MODULE.summarize_preliminary_rows(rows)
    assert tuple(summary["predicates"]) == (
        "survives_count_gain", "survives_context_stability",
        "survives_receiver_heterogeneity", "survives_global_scalar",
        "close_count_gain", "close_context_stability",
        "close_receiver_heterogeneity", "close_global_scalar",
    )
    assert "full_gain_error_median_A" in summary["pooled_aggregates"]
    assert "full_gain_error_median_B" in summary["pooled_aggregates"]
    assert summary["decision"]["first_decisive_clause"] != "close_full_gain"

def test_count_gain_remains_the_only_decision_path_using_full_endpoint() -> None:
    summary = _MODULE.summarize_preliminary_rows(
        _preliminary_decision_rows(count_gain=-1.0, full_gain_seed_values=(0.0,) * 4)
    )
    assert summary["predicates"]["close_count_gain"] is True
    assert summary["decision"]["first_decisive_clause"] == "close_count_gain"
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'repair_removes_direct_full_gain or count_gain_remains'`

Expected: FAIL because `survives_full_gain` and `close_full_gain` still exist.

- [ ] **Step 3: Implement the exact predicate/decision repair**

Delete only the two direct endpoint predicates. Change `_preliminary_close_evidence` to return only `context_spearman_nonpositive_seed_count`. Iterate CLOSE in exact order `close_count_gain`, `close_context_stability`, `close_receiver_heterogeneity`, `close_global_scalar`; compute SURVIVES from the first four predicate values. Keep every `full_gain_error_*` aggregate and validation relation.

- [ ] **Step 4: Add strict mutation tests**

For each removed key, insert it with both Boolean values and assert `validate_scientific_payload` raises. For each descriptive full-gain aggregate, remove it or drift it by `0.125` and assert rejection. Recompute the decision after each mutation so the validator proves it rejects the forbidden schema rather than merely stale dependent fields.

- [ ] **Step 5: Run GREEN**

Run: `.venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'preliminary or full_gain or count_gain'`

Expected: all selected tests PASS.

---

### Task 3: Derive the Exact Six-Key Historical Seed Records

**Files:**
- Modify: `scripts/diagnose_pass205_rdgc_stage_b.py:1183-1365,1766-1785,2550-2680,4027-4075`
- Modify: `tests/test_diagnose_pass205_rdgc_stage_b.py:1900-2025,2290-2335`

**Interfaces:**
- Consumes: authenticated Pass 200 manifest and `ValidatedBindingReceipt` returned by production validators.
- Produces: `derive_rdgc_seed_artifacts(manifest, receipt) -> list[dict[str, object]]` and exact six-key validation.

- [ ] **Step 1: Write RED schema and derivation tests**

```python
def test_repaired_seed_schema_is_derived_from_validated_pass200_authorities() -> None:
    records = _MODULE.derive_rdgc_seed_artifacts(validated_manifest, validated_receipt)
    assert tuple(records[0]) == (
        "seed", "checkpoint", "training_report", "retrieval_report",
        "train_final_pack", "train_source_export_sha256",
    )
    assert records[0]["checkpoint"] == validated_manifest["seeds"]["0"]["checkpoint_pt"]
    assert records[0]["retrieval_report"] == validated_manifest["seeds"]["0"]["retrieval_json"]
    assert records[0]["train_source_export_sha256"] == (
        validated_receipt.seeds[0].train_source_export_sha256
    )
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k repaired_seed_schema`

Expected: FAIL because the old schema contains `configuration_sha256` and lacks retrieval report.

- [ ] **Step 3: Implement exact derivation**

```python
def derive_rdgc_seed_artifacts(manifest, receipt):
    return [
        {
            "seed": seed,
            "checkpoint": dict(manifest["seeds"][str(seed)]["checkpoint_pt"]),
            "training_report": dict(manifest["seeds"][str(seed)]["report_json"]),
            "retrieval_report": dict(manifest["seeds"][str(seed)]["retrieval_json"]),
            "train_final_pack": dict(manifest["seeds"][str(seed)]["train_npz"]),
            "train_source_export_sha256": receipt.seeds[seed].train_source_export_sha256,
        }
        for seed in range(4)
    ]
```

Require builtin `int` for `seed`, exact ordered mappings, regular nonempty paths, lowercase 64-hex digests, and recursive exact equality against independently derived records.

- [ ] **Step 4: Add exhaustive mutation coverage**

Generate mutations for every dict key removal/addition/reorder, every list reorder/type, every leaf type/value, valid-looking cross-seed swaps, valid-looking artifact swaps, and reintroduction of `configuration_sha256`. Assert generated path coverage equals the set of all recursive paths before running the matrix.

- [ ] **Step 5: Run GREEN**

Run: `.venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'seed_artifacts or historical or repaired_seed'`

Expected: all selected tests PASS.

---

### Task 4: Authenticate Audit, Upstream RSTA, Receipt, and Repair History

**Files:**
- Modify: `scripts/diagnose_pass205_rdgc_stage_b.py:1183-1365,2310-2515,2550-2680,3930-3970`
- Modify: `tests/test_diagnose_pass205_rdgc_stage_b.py:1230-1305,1940-2025,2290-2335`

**Interfaces:**
- Consumes: exact repair authorities, exact Pass 200 production module, fixed receipt path, future manifest receipt record.
- Produces: `authenticate_authority` that rejects shape-only audit/upstream records before Torch or artifact access.

- [ ] **Step 1: Add RED audit/history tests**

Construct a real temporary linear Git chain with the exact authority blobs. Add mutants for wrong audit bytes, wrong audit commit, reordered source IDs, self-reference, wrong reviewed-candidate digest, skipped chronology commit, merge commit, and old plan as source-chain base. Each must raise before any Torch import sentinel or artifact-open sentinel is reached.

- [ ] **Step 2: Add RED production-validator sentinels**

Wrap the authenticated Pass 200 module's `validate_scientific_execution_source` and `validate_historical_binding_receipt`. Assert each is called exactly once before `derive_rdgc_seed_artifacts`; return values intentionally different from the caller manifest and assert the derived records follow validator output.

- [ ] **Step 3: Add RED receipt tests**

Use a verifier-valid ten-key roundtrip receipt fixture. Compute its digest from serialized bytes, place that digest in the future manifest, and assert authentication succeeds. Mutate nested artifact SHA, `V_R`, `HV_R`, status, `outcome_disclosed`, or manifest digest and assert rejection. Make the forbidden artifact path a FIFO/sentinel and prove it is never opened.

- [ ] **Step 4: Implement exact authentication**

Authenticate the audit Git/worktree bytes and literal 14-ID order; walk the full merge-free document chronology; validate the Pass 200 manifest and receipt with production validators; validate the roundtrip receipt with the authenticated verifier; require its SHA to equal the manifest-derived record; reuse that record in result binding. Do not embed a receipt digest constant.

- [ ] **Step 5: Run GREEN**

Run: `.venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'authority or literature or upstream_rsta or roundtrip_receipt or repair_chain'`

Expected: all selected tests PASS with no Torch/artifact sentinel access on failures.

---

### Task 5: Freeze Future Manifest, Result Relations, and Source Chain

**Files:**
- Modify: `scripts/diagnose_pass205_rdgc_stage_b.py:1580-2710,3916-4230`
- Modify: `tests/test_diagnose_pass205_rdgc_stage_b.py:1200-1285,1740-1830,2160-2335`

**Interfaces:**
- Consumes: repaired predicates, six-key seed records, authenticated audit/receipt records.
- Produces: exact future manifest and scientific result validators plus repair-plan-to-source history validation.

- [ ] **Step 1: Add RED future-manifest tests**

Update `_future_manifest()` to use the repaired six-key seeds, literal audit object, original candidate, this repair plan, and manifest-derived receipt. Assert exact top-level order remains:

```python
(
    "schema_version", "candidate", "implementation_plan", "upstream_rsta",
    "literature_audit", "validation_receipt", "historical",
    "current_scientific_source", "artifact_schema", "seeds",
)
```

Add independent mutations for old plan binding, old seed schema, fabricated audit, changed 33-path order, receipt-record drift, and old ten-key predicate array.

- [ ] **Step 2: Add RED result-binding tests**

Build complete preliminary-only and full-panel synthetic receipts. Assert `binding.seeds` equals derived manifest seeds, `binding.validation_receipt` reuses the exact manifest record, descriptive full-gain aggregates are required/recomputed, and decisions use only the repaired eight predicates.

- [ ] **Step 3: Implement strict recursive validation**

Update `_validate_seed_artifacts`, preliminary schema/order, result binding, `validate_future_manifest`, and `authenticate_authority`. Set production `PLAN_PATH`, `PLAN_COMMIT`, and `PLAN_SHA256` to this committed plan. Walk every nonempty single-parent source commit back to that exact plan and require aggregate scope exactly the two source/test paths.

- [ ] **Step 4: Run focused GREEN**

Run: `.venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'future_manifest or scientific_payload or source_history or result_binding'`

Expected: all selected tests PASS.

- [ ] **Step 5: Commit the implementation**

```bash
git add -- scripts/diagnose_pass205_rdgc_stage_b.py tests/test_diagnose_pass205_rdgc_stage_b.py
git diff --cached --check
git diff --cached --name-only
git commit -m "repair RDGC scientific authority"
```

Require the cached name list to contain exactly those two files.

---

### Task 6: Run Full Assurance and Obtain Fresh Source Review

**Files:**
- Verify: `scripts/diagnose_pass205_rdgc_stage_b.py`
- Verify: `tests/test_diagnose_pass205_rdgc_stage_b.py`

**Interfaces:**
- Consumes: final two-file source commit `V_G2`.
- Produces: test/lint/compile/scope evidence and independent READY source review.

- [ ] **Step 1: Run the complete RDGC suite once**

Run: `.venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py`

Expected: PASS; report exact pass/skip/warning counts.

- [ ] **Step 2: Run static gates**

```bash
.venv/bin/ruff check scripts/diagnose_pass205_rdgc_stage_b.py tests/test_diagnose_pass205_rdgc_stage_b.py
.venv/bin/python -m py_compile scripts/diagnose_pass205_rdgc_stage_b.py tests/test_diagnose_pass205_rdgc_stage_b.py
git diff --check
git diff --name-only HEAD^..HEAD
```

Expected: all exit 0; the final command lists exactly the two planned files.

- [ ] **Step 3: Obtain fresh Claude source review**

Start one read-only consultation with explicit review models `['opus','gpt-5.6-sol']`. Give it the candidate, audit, design, amendment, repair plan, `V_G2`, full test output, and prohibit artifact/GPU access. Require no Critical/Important finding. Fix findings under focused RED/GREEN in separate commits, rerun one final full suite, then re-review the changed source.

---

### Task 7: Build and Review the Direct-Child Manifest Handoff

**Files:**
- Create: `docs/pass205_rdgc_stage_b_manifest.json`
- Read allowed: `/home/rb/pass200-rsta-roundtrip-e73e9d4/reports/generated/pass200_rsta_receipt/e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae-scientific-artifact-roundtrip-validation.json`

**Interfaces:**
- Consumes: independently READY `V_G2`, exact provenance-only receipt bytes, reviewed source hashes.
- Produces: direct-child manifest-only `HV_G2`.

- [ ] **Step 1: Transfer the receipt opaquely to the literal ignored path**

Require the source to be a regular non-symlink mode-0600 file and the destination absent. Copy only that receipt. Do not copy or open the old scientific artifact.

- [ ] **Step 2: Validate and derive the receipt digest**

Run the authenticated roundtrip verifier's strict receipt validator on the transferred bytes. Require exact nested artifact path/SHA, `V_R`, `HV_R`, `status="VALID"`, and `outcome_disclosed=false`; compute SHA-256 from those bytes only. Use that derived digest in `validation_receipt` and nowhere as a CLI input.

- [ ] **Step 3: Construct the exact future manifest**

Bind original candidate, this repair plan, literal audit, derived receipt record, repaired historical seeds, exact 33-source order at `V_G2`, unchanged artifact schema except repaired eight-key predicate projection, and seeds `[0,1,2,3]`. Run production future-manifest and source-auth validators before commit.

- [ ] **Step 4: Commit only the manifest**

```bash
git add -- docs/pass205_rdgc_stage_b_manifest.json
git diff --cached --check
git diff --cached --name-only
git commit -m "refreeze RDGC authority handoff"
```

Require `HV_G2^ == V_G2`, one parent, and the cached/committed path list to contain only the manifest.

- [ ] **Step 5: Obtain separate manifest/provenance review**

Review exact manifest bytes, direct-child scope, every authority hash/commit, all 33 `V_G2` source blobs/worktree bytes, derived receipt bytes/digest, repaired seed derivation, and absence of self-reference. Require no Critical/Important finding.

---

### Task 8: Execute Exactly One DGX Process After All Gates

**Files:**
- Read: `docs/pass205_rdgc_stage_b_manifest.json`
- Create at most once: the exact `HV_G2`-named result under
  `reports/generated/pass205_rdgc_stage_b/`.

**Interfaces:**
- Consumes: reviewed `HV_G2`, transferred provenance-only receipt, all registered historical artifacts.
- Produces: one atomic RDGC scientific receipt or a durable structural/INVALID stop.

- [ ] **Step 1: Authenticate fresh detached DGX checkout**

Verify `HV_G2`, its sole parent `V_G2`, manifest SHA, exact authority blobs, all 33 source blobs/worktree bytes, validated receipt digest, all historical artifacts, clean status, pinned Python/Torch/NumPy/CUDA runtime, idle queue/GPU, and absent output/temp.

- [ ] **Step 2: Launch one exact isolated process**

```bash
handoff_commit=$(git rev-parse HEAD)
source_commit=$(git rev-parse HEAD^)
test "$(git show -s --format=%P HEAD | wc -w)" -eq 1
test "$(git diff-tree --no-commit-id --name-only -r HEAD)" = \
  "docs/pass205_rdgc_stage_b_manifest.json"
output_path="reports/generated/pass205_rdgc_stage_b/${handoff_commit}-rdgc-stage-b.json"
.venv/bin/python -I -B scripts/diagnose_pass205_rdgc_stage_b.py \
  --manifest docs/pass205_rdgc_stage_b_manifest.json \
  --output "$output_path" \
  --scientific-once
```

Retain the original PID/session. Do not tee, pipe, retry, or launch a second process.

- [ ] **Step 3: Validate once and stop**

After exit, require atomic temp absence, hash the output, strict-load persisted JSON, run the production recursive validator, independently recompute registered relations, and report the frozen decision. Do not start training or another scientific stage. Any structural failure, INVALID, CLOSE, UNRESOLVED, or failed review stops the chain exactly as registered.

## Final Self-Review Checklist

- [ ] Every requirement in the design, audit, and amendment maps to a task above.
- [ ] No `TBD`, `TODO`, ellipsis placeholder, symbolic commit, or caller-selected digest remains in production instructions.
- [ ] All function names, key orders, paths, hashes, and source scopes are consistent across tasks.
- [ ] The old RSTA artifact is never opened; only the provenance receipt is transferred/validated.
- [ ] No direct full-gain decision survives; descriptive endpoint metrics and paired count gain remain.
- [ ] DGX execution remains last, single-attempt, and conditional on both reviews.
