# Pass 201 Source-v5 Immutable-Output Mode Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`, `superpowers:test-driven-development`,
> `superpowers:receiving-code-review`, and
> `superpowers:verification-before-completion` task-by-task.

**Goal:** Accept the exact immutable `0100444` source-output evidence already
authenticated by the completed-run receipt, refreeze it under a new
manifest-only handoff, and resume the registered CPU diagnostic without
rerunning training.

**Architecture:** Add one non-mutating descriptor-based existing-output
verifier to the shared contract and route activation's four required files
through it. A new source-v6 authority binds H5/V5, the failed activation, this
repair authority, reviewed V6 bytes, and unchanged historical output evidence.
H6 is the sole-manifest child of V6; only then may a fresh CPU activation run.

**Tech Stack:** Python 3.12/3.13, pytest, strict JSON, Git object
authentication, POSIX `open`/`fstat`/`lstat`, SHA-256.

## Authority

```text
amendment path: docs/pass201_pa_source_v5_output_mode_repair_amendment_2026-08-11.md
amendment commit A6: c575ffc1a040530eeeb3e439d67d1d4b24fe92ac
amendment SHA-256: 32133627314017b0ff62a7f4ad3da100619828f7bca08fd171df790e16e93c0d
initial amendment commit: 622e145b2dfeafdf6a202c7012ad92813a2932c2
first amendment fix: a8119bcf4b97de6a7a948d75ffd393e64c406b10
initial plan: 651920e80126d2c8f31b2acce6d04438fe0c12a8
V5: 656b5f2069f76ee6d8c5079bee8ae6a371a89f69
H5: 18b225f33b61dd221d6878cf8b14eb75a0037323
H5 manifest SHA-256: 2cf3b9a1c5cb41304f8d653e839d5372fa9570c4f442d4948ecdec4256c0de20
H5 preservation ref: pass201-source-v5-handoff-18b225f
H5 bundle SHA-256: 838b3f65435b374e172220caa1612910fc3ca73fd24560a0ab7affec6e7ceb75
historical receipt SHA-256: a494ead85167539670f8c5d1481f8d9eabc274776727df06d7362e99e9b7cdf9
historical output mode: decimal 33060 / octal 0100444
```

## Global constraints

- Never edit, stage, move, or remove `HANDOFF_BRIEF.md`, `RSPG_SPECDEFECT.md`,
  or `RSPG_TASK.md`.
- Never rerun Proxy Anchor training or launch a GPU process for this repair.
- Never chmod, rewrite, normalize, deserialize before authentication, or
  regenerate the six historical outputs.
- Preserve the baseline Recall@1 `0.9174989449992966` as the reproducible
  comparison anchor.
- The only P6-to-V6 source/test paths are the diagnostic, shared contract, and
  their two tests named by the amendment.
- Use one Opus-to-Sol consultation per completed docs/source/H6 boundary;
  explicit ordered models are `opus,gpt-5.6-sol`. Do not replay failed jobs.
- H6 contains no candidate or scientific values. Activation, binding-only,
  smoke, and science remain separate processes and fail closed in that order.

---

### Task 1: Review and freeze the repair design

**Files:**

- Review: `docs/pass201_pa_source_v5_output_mode_repair_amendment_2026-08-11.md`
- Review: this plan

**Interfaces:**

- Consumes: immutable H5 manifest, H4 receipt, four exact output records.
- Produces: reviewed A6/P6 authority for source work.

- [ ] **Step 1: Verify the failure boundary read-only**

  On `riomus@spark-2751`, confirm PID `1061572` is absent;
  activation/source-manifest/smoke/science and owned temporary paths are
  absent; no GPU compute process remains. Confirm all four copied files are
  regular non-symlinks with mode `0444`, exact bytes, and exact receipt
  SHA-256. Locally, verify the registered bundle SHA, preservation ref, H5^=V5,
  sole manifest edge, and H5 manifest Git-byte SHA.

- [ ] **Step 2: Verify producer intent from source**

  Read `hash_open_regular` and `publish_new_file` completely. Confirm the
  producer deliberately records mode `0100444` after chmod and re-hash.

- [ ] **Step 3: Obtain a read-only independent review**

  Review the exact receipt key order, complete-mode invariant, TOCTOU defenses,
  no-mutation rule, Git topology, and retry boundary. Repair docs-only before
  source work if any Critical/Important finding exists.

- [ ] **Step 4: Commit this plan alone**

  Require A6 as the exact parent, sole path modification, mode `100644`, and
  clean `git diff --check`. Record the resulting P6 commit/SHA later in V6
  production constants/tests; P6 cannot self-bind its own commit.

### Task 2: Establish exact immutable-output RED tests

**Files:**

- Modify: `tests/test_diagnose_pass201_cis_operator.py`
- Protect: both production source files

**Interfaces:**

- Consumes: `_validate_source_v3_output(root, path, evidence, relative)`.
- Produces: behavior-level RED proving valid `0444` receipt evidence is
  currently rejected only by the hard-coded `0644` literal.

- [ ] **Step 1: Write the valid read-only file regression**

  Create a real repository-relative regular file, write independent bytes,
  chmod `0444`, compute SHA-256, and pass canonical receipt-shaped evidence:

  ```python
  evidence = {
      "bytes": len(data),
      "file_type": "regular",
      "mode": 0o100444,
      "path": relative,
      "sha256": hashlib.sha256(data).hexdigest(),
  }
  module._validate_source_v3_output(root, path, evidence, relative)
  ```

- [ ] **Step 2: Run RED and verify the precise cause**

  Run only the new test. Expected: `ValueError("source-v3 output evidence
  differs")` at the old `evidence["mode"] == 0o100644` predicate.

- [ ] **Step 3: Add independent mutation REDs**

  Cover receipt mode `0100644`, `0100400`, bool, float, string, negative;
  live chmod drift; symlink; directory; FIFO where supported; path/size/hash
  drift; and replacement during validation. Assert no parser, checkpoint,
  model, candidate, or publisher sentinel is reached.

### Task 3: Implement the non-mutating verifier GREEN

**Files:**

- Modify: `scripts/pass201_pa_source_v2_contract.py`
- Modify: `scripts/diagnose_pass201_cis_operator.py`
- Modify: `tests/test_diagnose_pass201_cis_operator.py`
- Modify: `tests/test_pass201_pa_source_v2_contract.py`

**Interfaces:**

- Produces the exact interface
  `verify_existing_regular_file(path: Path, *, expected_mode: int,
  expected_bytes: int, expected_sha256: str) -> OutputEvidence`.

  The function is read-only and never calls `chmod`, `fchmod`, publication, or
  a producer helper that mutates mode.

- [ ] **Step 1: Implement descriptor-bound verification**

  Open the parent directory and named file with no-follow semantics, require
  regular type and exact `st_mode`, stream SHA-256 and byte count, compare
  pre/post descriptor identity, compare the named entry and parent identity,
  and return observed absolute path, complete mode, size, and digest. The helper
  is mode-generic; it must not contain the activation-specific literal 33060.

- [ ] **Step 2: Replace the hard-coded activation predicate**

  Rely on prior canonical-receipt byte authentication for key order. Require
  the exact evidence key set and concrete types; require the activation-only
  historical literal mode `0o100444`; compare the absolute live path separately
  to `root / evidence["path"]`; call `verify_existing_regular_file`; compare
  only returned complete mode/size/hash to receipt evidence.

- [ ] **Step 3: Run focused GREEN**

  Run every Task 2 case and the existing source-binding/receipt tests. Confirm
  the valid `0444` case passes and every mutation fails before parsing.

- [ ] **Step 4: Prove producer behavior is unchanged**

  Run existing `hash_open_regular`, `publish_new_file`, no-clobber, rollback,
  and complete-receipt tests. Assert their function source and defaults did not
  change.

### Task 4: Add source-v6 authority and freezer

**Files:** same four files.

**Interfaces:**

- Produces: `pass201-pa-source-v6-authorization-v1` at
  `docs/pass201_pa_source_v6_authorization_manifest.json`.
- Consumes: exact H5 Git manifest blob/SHA, V5, A6, P6, historical H4/S4 and
  receipt/output records.

- [ ] **Step 1: Add future-manifest RED fixtures**

  Build a real merge-free Git history with separate H5 and linear
  V5→initial-amendment→A6→P6→V6→H6 branches. Require H5^=V5 and H6^=V6;
  each handoff has one exact manifest edge. Mutate every parent, path, mode,
  blob, source row, authority digest, and retained domain.

- [ ] **Step 2: Freeze the exact v6 shape**

  Preserve all v5 domains recursively and add one ordered
  `output_mode_repair` object binding A6/P6, H5/V5, PID/exit/error/no-output
  chronology, and literal mode `33060`. Update only current source revision,
  changed source hashes, and the new authority. Forbid result values.

- [ ] **Step 3: Implement candidate-free freezer and dispatch**

  Freeze only from detached clean V6, authenticate H5/V5 and all historical
  domains, require result/activation absence, publish mode `0600` with
  hard-link no-replace, strict reload, and exact byte equality. Route public
  activation only through the literal v6 path; retain v5 solely as the prior
  failed executor authority.

- [ ] **Step 4: Run exhaustive v6 schema/provenance GREEN**

  Cover recursive removal/addition/order/type/value mutations, source-chain
  merges/empty/out-of-scope commits, manifest cycles, and all output-mode
  mutations.

### Task 5: Full assurance and independent source review

**Files:** exact four-file source/test scope.

- [ ] **Step 1: Run the complete affected suite once**

  Run the full diagnostic test file plus shared-contract tests. Then run Ruff,
  `py_compile`, and `git diff --check`.

- [ ] **Step 2: Review the cumulative P6-to-V6 diff**

  Ask Claude with explicit `models=["opus","gpt-5.6-sol"]` to find real
  Critical/Important defects. Require explicit review of no mutation, mode
  exactness, symlink/race handling, v6/H6 topology, source scope, and unchanged
  scientific functions.

- [ ] **Step 3: Apply review feedback with fresh RED→GREEN cycles**

  Commit each source repair separately, rerun focused tests, then one final
  affected suite only after the diff stabilizes.

- [ ] **Step 4: Commit reviewed V6**

  Require aggregate P6..V6 scope exactly the four registered files, record
  V6 and all source digests, and leave only protected untracked files.

### Task 6: Freeze and review manifest-only H6

**Files:**

- Create: `docs/pass201_pa_source_v6_authorization_manifest.json`

- [ ] **Step 1: Freeze once in a fresh detached V6 checkout**

  Authenticate every predecessor Git blob and authority, confirm all protected
  output/result paths absent, use one prospectively recorded UTC literal, and
  publish the candidate manifest once.

- [ ] **Step 2: Commit exactly the manifest**

  H6^ must equal V6 and its sole edge must be `A 100644` at the exact v6 path.
  Verify every declared V6 source blob/worktree digest and every retained
  historical output digest.

- [ ] **Step 3: Independently review H6**

  Require production validators plus a read-only Opus-to-Sol review of
  manifest bytes/order, authorities, topology, source rows, no cycle, no
  candidate/scientific values, and exact `33060` mode authority.

### Task 7: Resume CPU gates once

**Files:** no repository edits.

- [ ] **Step 1: Prepare a fresh remote detached H6 checkout**

  On `riomus@spark-2751`, copy the six historical outputs byte-for-byte,
  authenticate exact sizes,
  SHA-256, modes, non-symlink status, receipt, dataset root, runtime, clean
  checkout, output/temp absence, process absence, and empty GPU compute list.

- [ ] **Step 2: Launch one fresh activation**

  Use `CUDA_VISIBLE_DEVICES=` and the exact registered environment. Preserve
  PID, start/finish, exit, output SHA, and absence evidence. Never reuse failed
  process state.

- [ ] **Step 3: Continue only through green gates**

  If activation validates offline, run binding-only, then fresh seed-0 smoke,
  then exactly one scientific CPU process. Stop immediately on any structural,
  integrity, provenance, or no-clobber failure.

- [ ] **Step 4: Compare only authenticated results**

  Compare the authenticated diagnostic against the reproducible Proxy Anchor
  baseline, not against an unreproduced headline SOTA. Use any higher published
  result only as a labeled external target until its full protocol is
  reproduced.
