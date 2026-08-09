# Pass 200 RSTA Binding-Receipt Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the prospectively registered RSTA integrity smoke and Stage-A diagnostic using a content-addressed historical binding receipt while keeping all scientific model arithmetic in a fresh TF32-disabled process.

**Architecture:** The immutable historical receipt proves current-source equivalence to all frozen descriptor packs under the arithmetic that created those packs. A separate receipt validator authenticates that proof without loading arrays. A training-only loader then consumes only the final train pack and checkpoint; smoke/scientific processes configure deterministic TF32-off arithmetic before any torch action and can never reach descriptor export or query/gallery/prehead arrays.

**Tech Stack:** Python 3.11, strict JSON and SHA-256, NumPy NPZ, PyTorch, pytest, Ruff, Git blob verification, atomic JSON output.

## Global Constraints

- Authoritative base preregistration: `docs/pass200_rsta_candidate_2026-08-09.md`, exact SHA-256 `a35cd3469d5561ce59202030dd3c3050e018dbfc537cb0ee0401a1d0340f5857`.
- Authoritative prospective amendment: `docs/pass200_rsta_binding_receipt_amendment_2026-08-09.md`, exact SHA-256 `691d786942c33cf8a943159280287bea08570114242854cdb7111795dc79e019`, at full commit `d1aeed63ade0e15d5f5a44be5981a4312e9a8df2`; implement it literally without changing any candidate statistic, threshold, role, support, control, or decision.
- Immutable receipt: `docs/pass200_rsta_binding_receipt_d6270a9.json`, exact SHA-256 `e75944aed5af0fbe53af9febbc9a9a5d30045357eb6b1f086c4ba61e10f82300`.
- Historical producer commit is full `d6270a94f14f5e0b4f4a3eeaa23f3f66d9bfaa54`; historical manifest SHA-256 is `aafab355a06667a9ca513cddeceb2a0129ea8ee09ce3dec0a19b6839fe15ffb1`.
- The receipt is binding-only: `candidate_values_computed=false`, verdict `NOT_COMPUTED`, and `uses_test_data=artifact_binding_only`. No RSTA value has been observed.
- Exact registered membership is train `25,882` rows / `3,997` identities, query `14,218` / `3,985`, and gallery `12,612` / `3,985` for every seed.
- Smoke/scientific processes export `CUBLAS_WORKSPACE_CONFIG=:4096:8` before start, configure deterministic algorithms and both TF32 flags off as their first torch action, and never toggle TF32 afterward.
- Smoke/scientific never call `_export_current_source`, `load_bound_seed`, `_load_digest_bound_packs`, or any equivalent that materializes query/gallery/prehead arrays.
- Query/gallery/prehead files are stream-hashed only. Candidate inputs and outcomes use the In-Shop training split exclusively.
- Receipt, manifest, artifact, source, or deterministic mismatch is `INVALID` before scoring; no fallback receipt, selective seed, tolerance widening, or output on pre-model failure.
- Local tests are tiny/mocked and serial. Full data, BN-Inception, B=180, smoke, and scientific execution run only on DGX `spark-2751` in an isolated clean checkout.
- Use strict TDD: focused missing-behavior RED, minimal GREEN, fresh verification. Do not overlap tests.
- Authorized tracked changes are only `scripts/diagnose_pass200_rsta_stage_a.py`, `tests/test_diagnose_pass200_rsta_stage_a.py`, `docs/pass200_rsta_receipt_stage_a_manifest.json`, and the final `docs/method_search_verdict.md` result entry. Authorized DGX outputs are uniquely addressed files under `reports/generated/pass200_rsta_receipt/`; they are never source inputs. Never edit trainer/recipes or protected untracked root files.

## Frozen amended-manifest topology

`docs/pass200_rsta_receipt_stage_a_manifest.json` has exactly these top-level keys:
`schema_version`, `base_preregistration`, `amendment`, `binding_receipt`,
`historical`, `current_scientific_source`, `artifact_schema`, and `seeds`.
`schema_version` is exactly `pass200-rsta-receipt-manifest-v1`.

- `base_preregistration`, `binding_receipt`, and `artifact_schema` each have exactly
  `path` and `sha256`.
- `amendment` has exactly `path`, `sha256`, and `commit` and must equal the literal
  amendment constants above.
- `historical` has exactly `producer_commit`, `manifest`, and `source`.
  `producer_commit` is the literal d6270a9 full commit. `manifest` has exact `path`
  and `sha256`; its blob is read from that producer commit. `source` has exact
  `git_revision` and `files`, matching the historical receipt and validating every
  file against that historical revision, never against the executing file.
- `current_scientific_source` has exactly `git_revision` and `files`. Its revision
  is the reviewed source commit `S`; `files` covers the diagnostic and every imported
  model/data/loss helper with exact lowercase SHA-256 values.
- `seeds` is the unchanged four-entry artifact list from the historical manifest,
  in seed order `0,1,2,3`, with exact nested key sets validated recursively.

Every validator accepts `manifest_path: Path`, derives the repository root from the
resolved manifest location, and resolves repo-relative paths from that root. The
production receipt wrapper requires the literal receipt path and SHA from Global
Constraints *before* independently requiring the manifest entry to equal them; a
substituted manifest cannot nominate a different receipt.

---

### Task 1: Strict Historical Receipt and Dual-Provenance Validation

**Files:**
- Modify: `scripts/diagnose_pass200_rsta_stage_a.py`
- Modify: `tests/test_diagnose_pass200_rsta_stage_a.py`

**Interfaces:**
- Produces: `load_strict_json(path: Path) -> dict[str, Any]`, `validate_historical_binding_receipt(manifest_path: Path, receipt_path: Path) -> ValidatedBindingReceipt`, and `validate_scientific_execution_source(manifest_path: Path) -> dict[str, Any]`.
- `ValidatedBindingReceipt` is immutable and contains only hashes, scalar bindings, per-seed artifact metadata, per-split row/count/order/source-export records, and producer provenance; it contains no checkpoint, tensor, or NPZ array.

- [ ] **Step 1: Write strict receipt parser RED tests**

  Test two layers independently. For the private strict parser/schema validator, build synthetic trusted JSON bytes and independently inject duplicate keys, `NaN`, `true` where integer `1` is required, `128.0` where integer `128` is required, and missing/extra keys at every nested object including `execution_audit`, `manifest.source`, artifact records, `prehead_reconstruction`, and each split record. For the production wrapper, mutate only receipt bytes/path or manifest receipt entry and require the immutable literal path/SHA gate to fail before schema parsing, checkpoint, `np.load`, model factory, or torch sentinel.

- [ ] **Step 2: Run parser tests and capture RED**

  Run: `.venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'historical_receipt or strict_receipt_json'`

  Expected: failures name missing strict receipt interfaces; all semantic-access sentinels remain zero.

- [ ] **Step 3: Implement exact receipt validation**

  First require the literal receipt repo-relative path and SHA from Global Constraints,
  then independently require the manifest receipt entry to equal both. Parse with
  `json.loads(..., parse_constant=reject, object_pairs_hook=reject_duplicate_keys)`,
  reject nonfinite values recursively, require exact JSON scalar types (booleans are
  not integers), and compare every recursive key set. Require the exact schema/status
  fields, four seeds `0..3` once each, batch `128`, both tolerances `2e-5`, zero
  recorded max descriptor differences, frozen counts, common train ID/label/source-
  order hashes, exact artifact maps, and exact report/retrieval scalars. Verify
  `d6270a9:docs/pass200_rsta_stage_a_manifest.json`, the base preregistration at
  d6270a9, the receipt diagnostic at d6270a9, and every historical source file at
  `0146f2d`, without requiring the old diagnostic to equal the executing file.

- [ ] **Step 4: Write independent current-source provenance RED tests**

  In a tiny real Git repository, prove current scientific source drift fails even when historical receipt provenance is intact, and historical blob drift fails even when current source is intact. Require the amended manifest to bind the base preregistration, amendment, receipt, historical source domain, and current scientific source domain with no self-cycle.

- [ ] **Step 5: Implement dual-provenance validation and verify GREEN**

  Retain current `validate_execution_audit` semantics for the executing script/current scientific sources. Add a separate historical validator. Run the focused receipt/source tests, then the full local file, Ruff, `py_compile`, and `git diff --check`.

- [ ] **Step 6: Commit Task 1**

  Commit only the script/test with message `validate historical RSTA binding receipt` and write the ignored Task-1 report with exact RED/GREEN evidence.

---

### Task 2: Receipt-Backed Training-Only Loader

**Files:**
- Modify: `scripts/diagnose_pass200_rsta_stage_a.py`
- Modify: `tests/test_diagnose_pass200_rsta_stage_a.py`

**Interfaces:**
- Produces: `load_training_only_seed(entry: dict[str, Any], receipt_seed: ReceiptSeed, *, artifact_hasher: Callable[..., str] = sha256_file, checkpoint_loader: Callable[..., Any] | None = None) -> TrainingOnlySeedInput`. Resolve `torch.load` only inside the call, after deterministic configuration has succeeded.
- Removes `source_exporter` and `load_and_bind_seed` from smoke/scientific data flow. The amended operational CLI removes `--binding-only` entirely; the immutable receipt already exists, so no receipt-creation controller is reachable.

- [ ] **Step 1: Write training-only access-boundary RED tests**

  Build a tiny seven-artifact seed bundle. Record every file-open and `np.load`. Require all seven files to be stream-hashed, report/retrieval JSON scalars to be read, and only `train_npz` to reach `np.load`. Patch `_export_current_source`, `_load_digest_bound_packs`, `load_and_bind_seed`, query/gallery/prehead `np.load`, and model factory to raise; receipt validation and training-only loading must leave every forbidden sentinel untouched.

- [ ] **Step 2: Write fail-closed semantic RED tests**

  Independently mutate train-pack bytes, embedded report/checkpoint digests, shape, finite/unit rows, label/ID/source-path order, canonical row indices, checkpoint final-state/evaluation-model source/config/proxy labels/count, report/retrieval scalar binding, and receipt train hashes. Assert the first mismatch fails without returning a partial object.

- [ ] **Step 3: Implement the training-only loader**

  Stream-hash all immutable artifacts first. For semantically consumed report,
  retrieval, checkpoint, and train artifacts, load from the exact bytes that were
  hashed: strict JSON from captured bytes, `torch.load(io.BytesIO(checkpoint_bytes))`,
  and `np.load(io.BytesIO(train_bytes))`. Query/gallery/prehead are stream-hashed and
  never captured or loaded. Validate the final train NPZ literally, synthesize
  canonical row indices, recompute train order/source-export hashes, and return a
  frozen training-only object containing the in-memory checkpoint state and its
  expected digest plus receipt provenance. Model construction consumes that in-memory
  state and never reloads its path. Explicitly release bytes/objects not required by
  the returned object.

- [ ] **Step 4: Add seed-set and immutability tests**

  Smoke must load seed 0 only after global four-seed receipt validation. Scientific must load seeds `0..3` exactly and require identical cross-seed train ID/label/source order. Mutate checkpoint/train files between manifest validation and byte capture and prove digest failure; mutate paths after the training object returns and prove model construction uses only captured bound state. The returned type must expose no query/gallery/prehead field.

- [ ] **Step 5: Verify and commit Task 2**

  Run focused access/loader tests, the full local file once, Ruff, `py_compile`, and `git diff --check`. Commit only script/test with message `load receipt-bound RSTA training inputs` and write the ignored report.

---

### Task 3: TF32-Off Smoke/Scientific CLI and Fail-Closed Ordering

**Files:**
- Modify: `scripts/diagnose_pass200_rsta_stage_a.py`
- Modify: `tests/test_diagnose_pass200_rsta_stage_a.py`

**Interfaces:**
- Produces CLI `--manifest PATH --binding-receipt PATH --output PATH (--smoke-only | --scientific)`; both modes consume `ValidatedBindingReceipt` and `load_training_only_seed`.
- `--binding-only` is rejected/absent in the amended CLI; receipt creation is completed historical provenance, not an operational mode.

- [ ] **Step 1: Write process-order and unreachable-path RED tests**

  In fresh subprocess fixtures, assert separate exact traces. Smoke is `configure ->
  receipt/manifest validation -> seed-0 training-only load -> cache -> model ->
  integrity -> atomic smoke output`; its candidate-scoring sentinel remains zero.
  Scientific is `configure -> receipt/manifest validation -> four-seed training-only
  load -> cache -> model -> integrity -> score -> atomic scientific output`. Assert
  `torch` is absent from `sys.modules` before configuration and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` exists before torch import. Deterministic
  algorithms are fail-closed, benchmark/autocast/both TF32 flags are false at model
  factory, forward, VJP, and JVP boundaries, and remain false on exception paths.
  Patch descriptor export, legacy full loaders, old binding CLI, and
  query/gallery/prehead loading; invalid receipt must touch none and create no output.

- [ ] **Step 2: Run process tests and capture RED**

  Run: `.venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'receipt_backed_smoke or receipt_backed_scientific or tf32_boundary or cli_receipt'`

- [ ] **Step 3: Refactor smoke and scientific entry points**

  Remove pre-configuration torch imports and torch-valued default arguments from
  runner/loader paths. Make deterministic configuration the first torch action, then
  validate manifest/receipt, then load training-only state. Remove `source_exporter`
  from smoke/scientific signatures and remove the operational `--binding-only` mode.
  Keep the registered first-batch integrity helper and scientific scorer unchanged
  after their input boundary.

- [ ] **Step 4: Add frozen-science characterization tests**

  Pin literal tiny-fixture outputs for role/batch selection, `A_self/A_batch/Delta`,
  all named controls, equal-seed aggregation, the 10,000-replicate bootstrap digest,
  PASS/FAIL/UNRESOLVED precedence, and scope-limitation fields. These must remain
  unchanged across the boundary refactor. Review the task diff and reject edits to
  field mathematics, role construction, statistics, controls, or thresholds.

- [ ] **Step 5: Persist the receipt audit without candidate leakage**

  Smoke output records receipt SHA, producer commit, historical manifest SHA, seed-0 train-pack/source-export hashes, current execution audit, and integrity residuals only. Scientific output records the same global provenance plus all four per-seed training bindings. No query/gallery array, path-derived candidate statistic, alternative receipt, or binding output is accepted.

- [ ] **Step 6: Add atomic-output and CLI exclusivity tests**

  Require `--binding-receipt` for smoke/scientific and reject mode combinations,
  unpinned paths, removed `--binding-only`, and any pre-existing destination. Publish
  to a unique run-addressed, initially absent path by writing/fsyncing a sibling temp,
  atomically hard-linking it to the absent destination (no clobber), fsyncing the
  parent directory, and unlinking the temp. Test serialization, file-fsync,
  publication, and directory-fsync failures; each cleans its temp and never replaces
  or leaves a stale requested result.

- [ ] **Step 7: Verify and commit Task 3**

  Run the complete focused local file serially, Ruff, `py_compile`, and `git diff --check`. Commit only script/test with message `separate RSTA binding and scientific processes` and write the ignored report.

---

### Task 4: Independent Review, Manifest Freeze, and DGX Execution

**Files:**
- Create: `docs/pass200_rsta_receipt_stage_a_manifest.json`
- Modify only if review requires: `scripts/diagnose_pass200_rsta_stage_a.py`, `tests/test_diagnose_pass200_rsta_stage_a.py`
- Produce on DGX: `reports/generated/pass200_rsta_receipt/<H>-smoke-seed0.json`
- Produce conditionally on DGX: `reports/generated/pass200_rsta_receipt/<H>-stage-a.json`
- Modify after adjudication: `docs/method_search_verdict.md`

**Interfaces:**
- Consumes reviewed Tasks 1--3 and immutable receipt.
- Produces one committed dual-provenance manifest, one integrity-smoke record containing the complete receipt audit, and only if smoke is green one scientific Stage-A result. There is no separate validation-only output.

- [ ] **Step 1: Run independent whole-repair review**

  Review the exact Tasks 1--3 diff against the amendment. Fix every Critical/Important finding with RED→GREEN and scoped re-review. Confirm receipt parsing, query/gallery non-materialization, provenance domains, first-torch-action ordering, TF32 invariants, loader reachability, output schema, and original RSTA math/statistics unchanged.

- [ ] **Step 2: Freeze the final source manifest**

  Use an explicit two-commit freeze. First commit all reviewed source/test fixes as
  source commit `S`. Create the exact frozen-topology manifest with
  `current_scientific_source.git_revision=S`, source Git-blob hashes for the
  diagnostic plus every imported model/data/loss helper, and the exact historical/
  amendment/receipt/artifact/seed domains. Commit only that manifest as handoff
  commit `H`, with no source change. Verify the worktree manifest equals `H:path`,
  `S` is the declared ancestor/source revision, and every current source file equals
  both its declared digest and `S:path`. Independently review `S..H`; any manifest
  fix creates a new `H` and repeats that scoped review.

- [ ] **Step 3: Prepare isolated DGX checkout**

  Bundle the independently approved handoff commit `H`, verify bundle SHA after
  transfer, create a new detached clean checkout exactly at `H`, verify the manifest
  equals `H:path` and every declared source equals `S:path`, validate artifact/data
  paths and Git objects, and reuse only the pinned DGX environment. Do not pull into
  the dirty group-learning checkout. Confirm no active compute process before launch;
  CPU/GPU wall time is operational only, never scientific evidence.

- [ ] **Step 4: Run binding-receipt validation and integrity smoke**

  Start one fresh process with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, exact amended manifest/receipt, and `--smoke-only`. Verify exit code, atomic output SHA, no candidate fields, exact receipt/current-source audits, seed-0-only training load, TF32-off audit, repeatability/adjoint/rotation/BN fixture gates, and no query/gallery/prehead array access.

- [ ] **Step 5: Apply the preregistered controller decision**

  If smoke is not completely green, record `INVALID`, run no scientific values, diagnose the named integrity failure prospectively, and do not widen any boundary. If smoke is green, immediately start the exact `--scientific` four-seed process with no code/config change.

- [ ] **Step 6: Validate and report Stage A**

  Collect the original process exit and output. Re-run the offline strict validator, verify all 64 primary and 16 alternate identities across four seeds, receipt/source/artifact hashes, exact 10,000-replicate bootstrap digests, control/rotation/repeatability gates, and the original PASS/FAIL/UNRESOLVED predicates. Append the outcome and mechanism to `docs/method_search_verdict.md`, commit, push, and authorize only the separately preregistered Stage B if and only if Stage A says PASS ONWARD.
