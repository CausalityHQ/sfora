# Pass 200 RSTA Immutable Scientific-Artifact Roundtrip Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Canonicalize future RSTA scientific JSON support-label keys and validate the one immutable legacy artifact exactly once, offline and outcome-blind, without rewriting it or re-running candidate-free or scientific execution.

**Architecture:** Commit one reviewed four-file source/test revision `V`: the live producer uses canonical ordered string keys, while a separate verifier authenticates `V/HV`, creates an isolated old-`H` checkout, adapts only the legacy support-map key types in memory, and invokes the authenticated old `scientific_payload`. A later manifest-only `HV` binds the amendment and exact 32 source paths; only then may one CPU-only verifier process publish an atomic provenance-only `VALID`/`INVALID` receipt.

**Tech Stack:** Python 3.12.3, strict insertion-ordered JSON, SHA-256, Git blob/ancestry authentication, `importlib` source loading, local isolated Git checkout, file-descriptor passing, NumPy through the authenticated legacy validator, pytest, Ruff, `py_compile`, atomic hard-link publication.

## Global Constraints

- Implement `docs/pass200_rsta_scientific_artifact_roundtrip_recovery_amendment_2026-08-10.md` literally as committed at `043121f8a414b91d7fb2e3d6a1635a6bd585676a`, SHA-256 `6e1767e802295fcfbf29e7151ac05991a016994ca92b99bf2e2cbcd46e4e9591`.
- Do not open or inspect the real artifact during Tasks 1–7. Use only synthetic fixtures. The only permitted real-artifact open is the single Task 8 attempt.
- The immutable artifact path is exactly `reports/generated/pass200_rsta_receipt/c04574e2bb751c3229bce673408577cfedc00a88-stage-a.json`; its SHA-256 is exactly `e9bcd77c6e372e9c3bab4a420b97ff56f8ea164cbca56f53ec9c99a3b3c527ae`.
- Preserve disclosed producer metadata exactly: PID `1002393`, exit `0`, `H=c04574e2bb751c3229bce673408577cfedc00a88`, `S=15234a529a181c39c1c8b6477ad7eb7823fd0798`, old manifest SHA-256 `9260329a0f9ad45257f51292d40c3a6d70c9494ea3e8fd185afcf8484f9378fe`, and old diagnostic SHA-256 `85958a940c5a4c9f0ae27f3342e436a8a37e49d94fe9515b22db0340d597ef6e`.
- The old persisted scientific manifest projection is exactly the ordered ten
  keys `path`, `sha256`, `base_preregistration`, `amendment`,
  `deterministic_pool_amendment`, `zero_jacobian_classifier_amendment`,
  `binding_receipt`, `historical`, `artifact_schema`, `source`, independently
  derived from H. Producer S omitted the five later amendment authorities.
- The registered verifier interpreter is the live repository's exact
  `.venv/bin/python`; observed Python is exactly `3.12.3`. Parent, child, and
  persisted `environment.numpy_version` must agree exactly before legacy
  recomputation, and the receipt records only authenticated observed runtime
  values.
- The inadvertently seen header `UNRESOLVED` is chronology only. Do not read, print, log, branch on, interpret, or persist it or any scientific content.
- Never rewrite, rename, touch, chmod, normalize, migrate, re-indent, or replace the artifact.
- Never run a GPU command, model/data load, candidate-free audit, scientific producer, scoring path, field path, receiver serialization, new aggregation/bootstrap, or decision path.
- `V` is the final independently reviewed source commit. Its aggregate diff
  from this plan commit is confined to the exact four source/test files below;
  its aggregate diff from this plan's parent contains only this plan plus those
  four files. `HV` is the later independently reviewed manifest-only commit
  with parent `V`.
- The real manifest and all result paths are protected until Task 7. Task 7 edits only the manifest. Task 8 writes only the one receipt path derived from `HV`.
- The previous 31 scientific source paths retain relative order. Insert the verifier as exact path 4, after `scripts/rsta_normwise_adjoint.py` and before `src/sfora/__init__.py`.
- Tests must prove RED before implementation GREEN. Existing uncommitted producer diagnosis is re-proven RED in an isolated checkout without discarding or rewriting root changes.
- One authenticated artifact open begins attempt `1`. Every outcome consumes it. Never invoke the verifier CLI a second time.

## Amendment binding

```text
path: docs/pass200_rsta_scientific_artifact_roundtrip_recovery_amendment_2026-08-10.md
sha256: 6e1767e802295fcfbf29e7151ac05991a016994ca92b99bf2e2cbcd46e4e9591
commit: 043121f8a414b91d7fb2e3d6a1635a6bd585676a
```

## Exact implementation file structure

- `scripts/diagnose_pass200_rsta_stage_a.py`: canonical live string-key producer/validator repair plus future recovery authority, projection, and exact 32-path manifest schema.
- `tests/test_diagnose_pass200_rsta_stage_a.py`: producer roundtrip/mutation REDs and future authority/source/projection REDs.
- `scripts/verify_pass200_rsta_scientific_artifact.py`: separate parent/child offline legacy verifier and atomic provenance-only receipt writer.
- `tests/test_verify_pass200_rsta_scientific_artifact.py`: synthetic artifact, provenance, isolation, mutant, receipt, no-clobber, and no-science tests.
- `docs/pass200_rsta_receipt_stage_a_manifest.json`: protected until the manifest-only `HV` task.
- `receipt_path(HV)`: the exact receipt-path function defined by the amendment; the implementation derives `HV` from authenticated `HEAD` and never accepts it as free text.

The exact future source tuple is:

```python
ROUNDTRIP_SOURCE_ORDER = (
    "scripts/diagnose_pass159_cotangent_stage_a.py",
    "scripts/diagnose_pass200_rsta_stage_a.py",
    "scripts/rsta_normwise_adjoint.py",
    "scripts/verify_pass200_rsta_scientific_artifact.py",
    "src/sfora/__init__.py",
    "src/sfora/ablation.py",
    "src/sfora/api.py",
    "src/sfora/arcg.py",
    "src/sfora/benchmark.py",
    "src/sfora/bn_inception.py",
    "src/sfora/catalog.py",
    "src/sfora/cea.py",
    "src/sfora/cem.py",
    "src/sfora/cli.py",
    "src/sfora/compose.py",
    "src/sfora/data.py",
    "src/sfora/encoder_ablation.py",
    "src/sfora/encoder_training.py",
    "src/sfora/evaluation.py",
    "src/sfora/experiments.py",
    "src/sfora/image_benchmark.py",
    "src/sfora/image_end_to_end.py",
    "src/sfora/image_recipes.py",
    "src/sfora/ipsr.py",
    "src/sfora/losses.py",
    "src/sfora/method.py",
    "src/sfora/oapf.py",
    "src/sfora/publication.py",
    "src/sfora/remote.py",
    "src/sfora/report.py",
    "src/sfora/text_baselines.py",
    "src/sfora/training.py",
)
```

---

### Task 1: Independently Review and Freeze the Recovery Authority

**Files:**
- Review: `docs/pass200_rsta_scientific_artifact_roundtrip_recovery_amendment_2026-08-10.md`
- Review: `docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md`
- Review: producer source at Git commit `S`; do not open the artifact
- Modify only if defective: the recovery amendment and then this plan binding

**Interfaces:**
- Consumes: exact amendment path/SHA/commit and disclosed metadata.
- Produces: clean independent authority approval before source commitment or validation execution.

- [ ] **Step 1: Authenticate the amendment and docs-only chronology**

  Run:

  ```bash
  test "$(git rev-parse 043121f8a414b91d7fb2e3d6a1635a6bd585676a^{commit})" = 043121f8a414b91d7fb2e3d6a1635a6bd585676a
  test "$(git show 043121f8a414b91d7fb2e3d6a1635a6bd585676a:docs/pass200_rsta_scientific_artifact_roundtrip_recovery_amendment_2026-08-10.md | sha256sum | cut -d ' ' -f 1)" = 6e1767e802295fcfbf29e7151ac05991a016994ca92b99bf2e2cbcd46e4e9591
  test "$(sha256sum docs/pass200_rsta_scientific_artifact_roundtrip_recovery_amendment_2026-08-10.md | cut -d ' ' -f 1)" = 6e1767e802295fcfbf29e7151ac05991a016994ca92b99bf2e2cbcd46e4e9591
  test "$(git diff-tree --no-commit-id --name-only -r 043121f8a414b91d7fb2e3d6a1635a6bd585676a)" = docs/pass200_rsta_scientific_artifact_roundtrip_recovery_amendment_2026-08-10.md
  test "$(git rev-parse 043121f8a414b91d7fb2e3d6a1635a6bd585676a^)" = a2fbde70730409c66561b759866d69b4802cfb9e
  test "$(git rev-parse a2fbde70730409c66561b759866d69b4802cfb9e^)" = 444d82278b2f81f5d0fe429791e078137165abdc
  test "$(git rev-parse 444d82278b2f81f5d0fe429791e078137165abdc^)" = c04574e2bb751c3229bce673408577cfedc00a88
  git diff --check 043121f8a414b91d7fb2e3d6a1635a6bd585676a^ 043121f8a414b91d7fb2e3d6a1635a6bd585676a
  ```

  Expected: every command exits `0`; the reviewed original amendment, original
  plan, and amendment-fix chronology is exact, and no source, test, manifest,
  result, or artifact entered the amendment-fix commit.

- [ ] **Step 2: Obtain a read-only adversarial review without artifact access**

  Run:

  ```bash
  devbox-ask claude --model opus --effort max "In this repository, read docs/pass200_rsta_scientific_artifact_roundtrip_recovery_amendment_2026-08-10.md, docs/superpowers/plans/2026-08-10-pass200-rsta-scientific-artifact-roundtrip-recovery.md, the old manifest only through git show at H=c04574e2bb751c3229bce673408577cfedc00a88, and producer/test source without opening any reports/generated scientific artifact. Do not edit. Review outcome blindness, canonical ordered string support keys, exact live recursive/signed-zero/byte roundtrip, strict raw loading, the exact ten-key persisted manifest projection derived from H with later authorities omitted by S, the single in-memory integer-key legacy adapter, complete old-H scientific_payload recomputation, exact recursive order/type equality, byte-identical producer serialization, exact sys.executable/Python 3.12.3 and parent/child/persisted NumPy runtime binding, H/S and V/HV provenance, child __file__/cwd/import binding, exact 32 source paths, receipt schema/no-clobber, one-attempt stop, and no GPU/rerun boundary. Report only Critical/Important/Minor findings with exact lines; say CLEAN if none."
  ```

  Expected: a complete read-only report that contains no artifact content.

- [ ] **Step 3: Apply the authority stop rule**

  If any finding affects a predicate, provenance edge, outcome boundary, schema,
  or process rule, stop source work. Repair only the amendment, commit it, obtain
  its new digest/commit, replace every old authority occurrence in this plan,
  commit only the plan, and repeat Steps 1–2 until clean. A stale plan binding is
  structural.

---

### Task 2: Re-prove the Existing Canonical Producer RED, Then Preserve Its GREEN

**Files:**
- Modify/test: `scripts/diagnose_pass200_rsta_stage_a.py`
- Modify/test: `tests/test_diagnose_pass200_rsta_stage_a.py`
- Protect: the new verifier paths, manifest, artifact, and results

**Interfaces:**
- Produces: canonical string-key live producer and validator behavior without changing scientific arithmetic or execution.
- Preserves: every pre-existing scientific payload field, row, metric, predicate, order, and atomic writer.

- [ ] **Step 1: Reproduce RED against old source in an isolated checkout**

  The root worktree already contains the uncommitted test-first diagnosis. Do
  not revert, reset, or overwrite it. Create a temporary checkout at the current
  docs-only plan commit, copy only the producer test file into it, and run the
  exact new nodes against the unmodified old producer:

  ```bash
  recovery_root=$(pwd)
  plan_commit=$(git rev-parse HEAD)
  recovery_red_root=$(mktemp -d)
  git worktree add --detach "$recovery_red_root/repo" "$plan_commit"
  cp "$recovery_root/tests/test_diagnose_pass200_rsta_stage_a.py" "$recovery_red_root/repo/tests/test_diagnose_pass200_rsta_stage_a.py"
  set +e
  (
    cd "$recovery_red_root/repo"
    PYTHONDONTWRITEBYTECODE=1 "$recovery_root/.venv/bin/python" -B -m pytest -q -p no:cacheprovider \
      tests/test_diagnose_pass200_rsta_stage_a.py::test_scientific_payload_requires_canonical_persisted_support_label_keys
  )
  recovery_red_exit=$?
  set -e
  test "$recovery_red_exit" -ne 0
  git worktree remove "$recovery_red_root/repo"
  rmdir "$recovery_red_root"
  ```

  Expected: FAIL because old source validates integer keys and cannot accept
  the canonical string-key mapping.
  Record the exact failing nodes without artifact access, then remove only this
  named temporary worktree and directory.

- [ ] **Step 2: Require the exact canonical producer implementation**

  The source change must be exactly equivalent to:

  ```python
  eligible_labels = primary.get("eligible_labels")
  support_ids_by_label = primary.get("support_ids_by_label")
  if (
      not isinstance(eligible_labels, list)
      or any(type(label) is not int or label < 0 for label in eligible_labels)
      or len(set(eligible_labels)) != len(eligible_labels)
      or type(support_ids_by_label) is not dict
      or list(support_ids_by_label) != [str(label) for label in eligible_labels]
  ):
      raise ValueError("registered primary support label keys differ")
  ```

  `_validate_registered_rows` must use
  `support_ids_by_label[str(row["label"])]`. Before the one live
  `scientific_payload` call, `run_scientific_diagnostic` constructs:

  ```python
  canonical_primary = {
      **primary,
      "support_ids_by_label": {
          str(label): ids
          for label, ids in primary["support_ids_by_label"].items()
      },
  }
  ```

  and places `canonical_primary` under `panel_binding["primary"]`. Do not alter
  any support value, eligible-label order, row, arithmetic, criterion, or writer.

- [ ] **Step 3: Run the focused producer GREEN**

  Run:

  ```bash
  .venv/bin/pytest -q \
    tests/test_diagnose_pass200_rsta_stage_a.py::test_scientific_payload_requires_canonical_persisted_support_label_keys \
    tests/test_diagnose_pass200_rsta_stage_a.py::test_scientific_cli_executes_exact_four_seed_pipeline_and_writes_atomic_rows
  ```

  Expected: PASS using synthetic data only. The exact live roundtrip/byte RED
  is added after the comparator interface is frozen in Task 3. No real CLI or
  artifact path is touched.

---

### Task 3: Write the Complete Offline Verifier RED Suite

**Files:**
- Create: `tests/test_verify_pass200_rsta_scientific_artifact.py`
- Do not create yet: `scripts/verify_pass200_rsta_scientific_artifact.py`
- Modify: `tests/test_diagnose_pass200_rsta_stage_a.py` for future manifest REDs
- Do not further modify yet: `scripts/diagnose_pass200_rsta_stage_a.py`

**Interfaces:**
- Requires future verifier functions listed in Step 1.
- Produces: focused failures for every artifact, adapter, legacy, provenance, receipt, and isolation contract before verifier implementation.

- [ ] **Step 1: Freeze the verifier interfaces in tests**

  Tests import and require these exact interfaces:

  ```python
  strict_json_object(data: bytes, *, name: str) -> dict[str, object]
  exact_ordered_equal(left: object, right: object) -> bool
  adapt_legacy_support_keys(raw: dict[str, object]) -> tuple[dict[str, object], tuple[tuple[str, ...], ...]]
  legacy_scientific_payload_arguments(adapted: dict[str, object]) -> dict[str, object]
  legacy_manifest_projection(repository: Path, manifest: dict[str, object]) -> dict[str, object]
  validate_legacy_roundtrip(raw_bytes: bytes, legacy_module: ModuleType) -> None
  validate_roundtrip_receipt(value: dict[str, object]) -> None
  receipt_path(repository: Path, handoff_commit: str) -> Path
  authenticate_runtime(repository: Path) -> dict[str, str]
  authenticate_legacy_provenance(repository: Path) -> dict[str, object]
  authenticate_verifier_provenance(repository: Path, manifest_path: Path) -> dict[str, object]
  run_isolated_legacy_child(repository: Path, artifact_fd: int, *, verifier_source_commit: str, verifier_handoff_commit: str, python_executable: Path, expected_numpy_version: str) -> tuple[int, int]
  write_validation_receipt_atomic(path: Path, value: dict[str, object]) -> None
  main(argv: Sequence[str] | None = None) -> int
  ```

- [ ] **Step 2: Add strict JSON, canonical-key, and adapter mutation REDs**

  Add tests named exactly:

  ```text
  test_strict_json_object_rejects_duplicate_nonfinite_and_nonobject
  test_legacy_adapter_requires_canonical_ordered_string_keys
  test_legacy_adapter_changes_only_support_key_types
  test_exact_ordered_equal_rejects_key_order_scalar_type_and_signed_zero_drift
  ```

  Use synthetic labels `[0, 7, 42]` and unrelated support strings. Mutate every
  key to integer, boolean, `"00"`, `"+7"`, `"042"`, missing, extra, reordered,
  and alias collision. The mutation ledger must equal exactly:

  ```python
  (
      ("panel_binding", "primary", "support_ids_by_label", "0", "str", "int"),
      ("panel_binding", "primary", "support_ids_by_label", "7", "str", "int"),
      ("panel_binding", "primary", "support_ids_by_label", "42", "str", "int"),
  )
  ```

  Recursively prove every value and every path outside that mapping remains
  exact. `exact_ordered_equal` must distinguish `True` from `1`, mapping order,
  and the IEEE bytes of `0.0` from `-0.0`.

  Replace the provisional ordinary-equality producer test with
  `test_roundtrip_recovery_live_scientific_payload_is_exact_and_byte_identical`.
  It dynamically loads the future verifier only inside this test and uses this
  exact predicate/serialization shape:

  ```python
  arguments = _valid_scientific_payload_arguments(tmp_path, monkeypatch)
  arguments["environment"]["roundtrip_signed_zero_probe"] = -0.0
  first = _MODULE.scientific_payload(**arguments)
  first_path = tmp_path / "first-live-roundtrip.json"
  _MODULE.write_json_atomic(first_path, first, sort_keys=False)
  first_bytes = first_path.read_bytes()
  assert first_bytes == (
      json.dumps(first, indent=2, sort_keys=False, allow_nan=False) + "\n"
  ).encode("utf-8")
  persisted = verifier.strict_json_object(first_bytes, name="live roundtrip")
  second = _MODULE.scientific_payload(
      manifest_audit=persisted["manifest"],
      execution_audit=persisted["execution_audit"],
      environment=persisted["environment"],
      seed_audits=persisted["seed_audits"],
      primary_rows=persisted["rows"]["primary"],
      alternate_rows=persisted["rows"]["alternate"],
      integrity=persisted["integrity"],
      aggregation=persisted["aggregation"],
      bootstrap=persisted["bootstrap"],
      panel_binding=persisted["panel_binding"],
  )
  assert verifier.exact_ordered_equal(second, persisted)
  second_path = tmp_path / "second-live-roundtrip.json"
  _MODULE.write_json_atomic(second_path, second, sort_keys=False)
  assert second_path.read_bytes() == first_bytes
  signed_zero_mutant = deepcopy(second)
  signed_zero_mutant["environment"]["roundtrip_signed_zero_probe"] = 0.0
  assert not verifier.exact_ordered_equal(signed_zero_mutant, persisted)
  ```

  No `first == persisted`, `second == persisted`, or other ordinary mapping
  equality may substitute for either live exact predicate.

- [ ] **Step 3: Add full legacy-call and mutant REDs**

  Add tests named:

  ```text
  test_legacy_roundtrip_calls_old_scientific_payload_with_exact_components
  test_legacy_roundtrip_requires_full_ordered_equality_and_exact_writer_bytes
  test_legacy_roundtrip_rejects_selected_field_current_source_and_canonicalizing_mutants
  test_legacy_manifest_projection_is_exact_ordered_ten_keys_derived_from_h
  test_legacy_manifest_projection_rejects_later_authority_and_current_projection_mutants
  test_real_h_scientific_payload_roundtrips_a_synthetic_artifact_in_isolated_child
  ```

  The synthetic legacy callable records one call. Require exact keyword order:

  ```python
  (
      "manifest_audit", "execution_audit", "environment", "seed_audits",
      "primary_rows", "alternate_rows", "integrity", "aggregation",
      "bootstrap", "panel_binding",
  )
  ```

  Require the adapted integer-key map only in `panel_binding`; require the raw
  object unchanged after the call. Mutants that validate selected fields, call
  a current module, use ordinary `dict ==`, sort keys, normalize signed zero,
  or omit byte equality must fail their recording sentinels.

  Freeze the projection's literal key tuple as:

  ```python
  LEGACY_SCIENTIFIC_MANIFEST_ORDER = (
      "path", "sha256", "base_preregistration", "amendment",
      "deterministic_pool_amendment", "zero_jacobian_classifier_amendment",
      "binding_receipt", "historical", "artifact_schema", "source",
  )
  ```

  Build its values only from authenticated H: fixed path and old manifest
  digest, same-named H values through `artifact_schema`, and
  `source = H["current_scientific_source"]`. Add, one at a time, each of
  `adjoint_integrity_amendment`, `normwise_adjoint_calibration_protocol`,
  `normwise_adjoint_calibration_result`, `normwise_adjoint_amendment`, and
  `normwise_adjoint_sign_control_amendment`; each must fail. Replacing the
  ten-key projection with either H's full top level or the current producer's
  larger candidate-free projection must fail.

- [ ] **Step 4: Add old/new provenance and process-isolation REDs**

  Add tests named:

  ```text
  test_legacy_provenance_binds_h_s_old_manifest_and_all_31_blobs
  test_legacy_child_uses_old_h_cwd_diagnostic_file_and_callable
  test_verifier_provenance_binds_v_hv_manifest_file_and_all_32_blobs
  test_isolated_child_uses_exact_command_fd_environment_tokens_and_timeout
  test_verifier_rejects_wrong_parent_import_path_dirty_checkout_and_blob
  test_runtime_authentication_rejects_wrong_sys_executable_python_and_numpy
  test_legacy_child_requires_parent_child_persisted_numpy_before_call
  test_roundtrip_receipt_runtime_fields_are_observed_authenticated_values
  ```

  Mock subprocess calls to require exact order, then include one local temporary
  Git repository integration. Assert `H^ == S`, old manifest/digest, diagnostic
  `__file__`, `S:path == H:path`, `HV^ == V`, exact verifier `__file__`, and
  exact source path order. Require child command `sys.executable, -I, -B`, cwd
  old H, `close_fds=True`, exactly one passed artifact FD, `start_new_session=True`,
  timeout `600`, empty CUDA visibility, closed stdin, and 64-byte output limits.

  Authenticate `sys.executable` as the exact absolute live-repository
  `.venv/bin/python`, its resolved regular-file target, and observed
  `sys.version_info[:3] == (3, 12, 3)`. Mutate the invocation path, resolved
  target, tuple, formatted version, parent NumPy module identity/version, child
  old-diagnostic `np` identity/version, and persisted
  `environment.numpy_version` independently. Assert the old
  `scientific_payload` recording sentinel has zero calls for every drift. A
  persisted-only mismatch produces the fixed invalid token; interpreter or
  runtime-module drift produces the fixed structural token. Require the exact
  receipt process order to include `python_executable`, `python_version`, then
  `numpy_version`, populated from the authenticated observations rather than
  caller literals.

- [ ] **Step 5: Add receipt, atomic, and no-science REDs**

  Add tests named:

  ```text
  test_roundtrip_receipt_exact_schema_predicates_and_every_nested_mutation
  test_roundtrip_receipt_contains_no_scientific_content
  test_receipt_path_is_derived_only_from_authenticated_hv
  test_atomic_receipt_never_replaces_or_follows_a_path
  test_cli_consumes_one_attempt_and_never_reaches_science_or_gpu
  ```

  Recursively remove, add, reorder, and mistype every receipt field. Require
  `VALID <=> child exit 0/exact valid token` and
  `INVALID <=> child exit 1/exact invalid token`. Traverse all keys and reject
  the forbidden scientific names from the amendment. Require runtime values
  to be exactly the authenticated observed `.venv/bin/python`, `3.12.3`, and
  parent/child/persisted NumPy version; reject hard-coded or caller-supplied
  substitutes. Patch dataset, model,
  torch/CUDA, candidate-free, producer, scoring, field, row, aggregation,
  bootstrap, and decision entry points to raise if reached.

- [ ] **Step 6: Add future manifest/source REDs**

  Add
  `test_roundtrip_recovery_manifest_authority_order_and_32_source_paths` and
  `test_roundtrip_recovery_projection_rejects_every_nested_mutation` to the
  producer test file. The future manifest fixture inserts this exact authority:

  ```python
  "scientific_artifact_roundtrip_recovery_amendment": {
      "path": "docs/pass200_rsta_scientific_artifact_roundtrip_recovery_amendment_2026-08-10.md",
      "sha256": "6e1767e802295fcfbf29e7151ac05991a016994ca92b99bf2e2cbcd46e4e9591",
      "commit": "043121f8a414b91d7fb2e3d6a1635a6bd585676a",
  },
  ```

  Require exact top-level/projection order, `ROUNDTRIP_SOURCE_ORDER`, every
  nested authority mutation, every source insertion/removal/reorder/hash
  mutation, old authority/domain equality, and ancestry
  `043121f8a414b91d7fb2e3d6a1635a6bd585676a -> plan -> V -> HV`
  through mocked exact commits.

- [ ] **Step 7: Run all new nodes and prove RED**

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_verify_pass200_rsta_scientific_artifact.py
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'roundtrip_recovery_live or roundtrip_recovery_manifest or roundtrip_recovery_projection'
  ```

  Expected: FAIL because the verifier file and recovery manifest authority are absent.

---

### Task 4: Implement the Minimal Verifier and Manifest-Validator GREEN

**Files:**
- Create: `scripts/verify_pass200_rsta_scientific_artifact.py`
- Modify: `scripts/diagnose_pass200_rsta_stage_a.py`
- Test: both exact test files

**Interfaces:**
- Produces: the Task 3 functions and exact parent/child CLI modes.
- Preserves: live scientific execution beyond canonical representation and all old artifacts.

- [ ] **Step 1: Add fixed authorities and strict primitives**

  Define exact constants for the amendment, artifact, `H`, `S`, manifest,
  diagnostic, source order, child tokens, timeout `600`, and output cap `64`.
  Implement strict JSON with duplicate/nonfinite rejection and exact equality:

  ```python
  def exact_ordered_equal(left: object, right: object) -> bool:
      if type(left) is not type(right):
          return False
      if type(left) is dict:
          return list(left) == list(right) and all(
              exact_ordered_equal(left[key], right[key]) for key in left
          )
      if type(left) is list:
          return len(left) == len(right) and all(
              exact_ordered_equal(a, b) for a, b in zip(left, right, strict=True)
          )
      if type(left) is float:
          return math.isfinite(left) and struct.pack(">d", left) == struct.pack(">d", right)
      if left is None:
          return True
      if type(left) in (bool, int, str):
          return left == right
      return False
  ```

  Reject tuples, mappings other than concrete dicts, and scalar types outside
  JSON `None/bool/int/float/str`. `strict_json_object` must decode UTF-8
  strictly, use an `object_pairs_hook` that rejects a repeated key before
  constructing each concrete `dict`, use `parse_constant` to reject every
  non-finite token, reject trailing content, require an exact top-level `dict`,
  and recursively reject non-string object keys or any value outside those
  concrete JSON types. It must never normalize key order or scalar values.

  Implement `authenticate_runtime` before any artifact-open helper. It derives
  `registered = (repository / ".venv/bin/python").absolute()`, requires
  `Path(sys.executable) == registered`, requires both strict resolutions to be
  the same regular executable with execute permission, and requires
  `sys.version_info[:3] == (3, 12, 3)`. It imports NumPy, requires
  `sys.modules["numpy"] is numpy`, and requires a concrete nonempty
  `str(numpy.__version__)`. It returns exactly, in order:

  ```python
  {
      "python_executable": ".venv/bin/python",
      "python_version": ".".join(str(value) for value in sys.version_info[:3]),
      "numpy_version": str(numpy.__version__),
  }
  ```

- [ ] **Step 2: Implement the single-mutation adapter and full legacy call**

  Require canonical keys by positional `key == str(label)`, not `int(key)`
  parsing. Deep-copy once, replace only the copied support mapping with integer
  keys paired to labels, and return the exact internal ledger. Implement:

  ```python
  def legacy_scientific_payload_arguments(adapted: dict[str, object]) -> dict[str, object]:
      return {
          "manifest_audit": adapted["manifest"],
          "execution_audit": adapted["execution_audit"],
          "environment": adapted["environment"],
          "seed_audits": adapted["seed_audits"],
          "primary_rows": adapted["rows"]["primary"],
          "alternate_rows": adapted["rows"]["alternate"],
          "integrity": adapted["integrity"],
          "aggregation": adapted["aggregation"],
          "bootstrap": adapted["bootstrap"],
          "panel_binding": adapted["panel_binding"],
      }
  ```

  `validate_legacy_roundtrip` calls only
  `legacy_module.scientific_payload(**arguments)`, requires exact ordered/type
  equality against raw, and requires exact UTF-8 writer bytes:

  ```python
  encoded = (
      json.dumps(recomputed, indent=2, sort_keys=False, allow_nan=False) + "\n"
  ).encode("utf-8")
  if encoded != raw_bytes:
      raise ArtifactInvalid
  ```

- [ ] **Step 3: Implement H/S and V/HV authentication**

  Use argument-vector `subprocess.run` only; never `shell=True`. Authenticate
  exact commits, parents, commit path scopes, old/new manifest blobs,
  authority bytes, source order, and every worktree/Git-blob digest. Build the
  exact old manifest projection and compare it recursively to the artifact's
  raw `manifest` object before the adapter call. The implementation literal is:

  ```python
  LEGACY_SCIENTIFIC_MANIFEST_ORDER = (
      "path", "sha256", "base_preregistration", "amendment",
      "deterministic_pool_amendment", "zero_jacobian_classifier_amendment",
      "binding_receipt", "historical", "artifact_schema", "source",
  )

  projection = {
      "path": "docs/pass200_rsta_receipt_stage_a_manifest.json",
      "sha256": OLD_MANIFEST_SHA256,
      "base_preregistration": manifest["base_preregistration"],
      "amendment": manifest["amendment"],
      "deterministic_pool_amendment": manifest["deterministic_pool_amendment"],
      "zero_jacobian_classifier_amendment": manifest["zero_jacobian_classifier_amendment"],
      "binding_receipt": manifest["binding_receipt"],
      "historical": manifest["historical"],
      "artifact_schema": manifest["artifact_schema"],
      "source": manifest["current_scientific_source"],
  }
  ```

  Require exact key order and compare with `exact_ordered_equal`; never copy
  any of H's five later authority keys and never invoke a current projection
  helper.

  The verifier source commit is derived from
  `manifest["current_scientific_source"]["git_revision"]`; handoff is derived
  from clean detached `HEAD`. Require `HEAD^ == source_commit`, manifest-only
  `HEAD`, and exact verifier `__file__` digest. No caller-supplied V/HV is
  authoritative.

- [ ] **Step 4: Implement isolated old-H child execution**

  Create a `TemporaryDirectory`, local clone with `--no-hardlinks --no-checkout`,
  detached checkout `H`, and clean-status checks. Run the authenticated absolute
  verifier path in hidden child mode with the exact process contract. The child
  loads the old diagnostic with `spec_from_file_location` only after file/blob
  authentication, verifies its module `__file__`, invokes exactly its
  `scientific_payload`, removes its temporary module entry, and restores
  `sys.path` after use.

  Child outcomes are exactly:

  ```python
  VALID = (b"RSTA_LEGACY_VALID\n", b"", 0)
  INVALID = (b"RSTA_LEGACY_INVALID\n", b"", 1)
  STRUCTURAL = (b"RSTA_LEGACY_STRUCTURAL\n", b"", 2)
  ```

  Data/schema/equality/byte failures map to `INVALID`; provenance, import,
  process, or environment failures map to `STRUCTURAL`.

  The public parser accepts exactly the four amendment-frozen flags. The
  private child invocation is an internal mutually exclusive parser mode with
  exact arguments `--legacy-child`, `--live-repository`, `--old-checkout`,
  `--artifact-fd`, `--verifier-source-commit`,
  `--verifier-handoff-commit`, and `--expected-numpy-version`; none is accepted
  in public mode. The parent supplies its already authenticated `V/HV` and
  observed NumPy version, while the child independently
  reauthenticates those values and the absolute executing verifier bytes before
  reading the descriptor. `run_isolated_legacy_child` returns exactly
  `(child_pid, child_exit_code)` after validating its token and empty stderr;
  raw stdout is never returned to receipt construction.

  Before adapter construction and before the recording sentinel can observe a
  call, require `type(raw["environment"]) is dict`, exact concrete nonempty
  string `raw["environment"]["numpy_version"]`,
  `legacy_module.np is sys.modules["numpy"]`, and exact equality among the
  expected parent version, `str(legacy_module.np.__version__)`, and the raw
  persisted version. A persisted-only mismatch maps to `INVALID`; interpreter,
  parent/child version, or module-identity drift maps to `STRUCTURAL`.

  Public mode must first authenticate `V/HV`, exact CLI paths, output absence,
  and the output parent. Only then open the exact artifact path with
  `os.O_RDONLY | os.O_NOFOLLOW`, require a regular file by `fstat`, record
  `(st_dev, st_ino, st_size)`, and pass that sole descriptor to the child. The
  child rereads only that descriptor, requires the registered SHA-256 before
  strict parsing, and `fstat`s again; any identity or size change is invalid. No
  process opens the artifact by a second path.

- [ ] **Step 5: Implement exact receipt validation and publication**

  Construct fields in the amendment's literal order, including process order
  `parent_pid`, `child_pid`, `child_exit_code`, `python_executable`,
  `python_version`, `numpy_version`, `isolated`, `child_head_commit`,
  `cuda_visible_devices`. Populate the three runtime fields only from the exact
  `authenticate_runtime` return after child agreement. Validate every exact type,
  digest, commit, path, status relation, token relation, and forbidden key
  recursively before writing. Derive output using:

  ```python
  def receipt_path(repository: Path, handoff_commit: str) -> Path:
      require_lowercase_hex(handoff_commit, length=40)
      return repository / "reports/generated/pass200_rsta_receipt" / (
          f"{handoff_commit}-scientific-artifact-roundtrip-validation.json"
      )
  ```

  Use exclusive temporary creation, flush/fsync, `os.link` no-replace,
  directory fsync, temporary unlink, strict reload, and exact final validation.
  Exit `0` only for published VALID, `1` only for published INVALID, and `2`
  for structural/preflight/publication failure.

- [ ] **Step 6: Extend producer future manifest validation**

  Add exact amendment constants and insert the authority after the sign-control
  amendment in full manifest and candidate-free projection order. Change the
  source path tuple to exact `ROUNDTRIP_SOURCE_ORDER`. Preserve every prior
  authority/domain validator and candidate-free/scientific prefix; do not run
  either execution mode.

- [ ] **Step 7: Run all focused GREEN nodes**

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_verify_pass200_rsta_scientific_artifact.py
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'roundtrip_recovery_live or canonical_persisted_support or roundtrip_recovery_manifest or roundtrip_recovery_projection'
  ```

  Expected: PASS, entirely synthetic and CPU-only.

---

### Task 5: Full Assurance, Exact Four-File Commit V, and Independent Review

**Files:**
- Commit exactly the four source/test files
- Protect every doc manifest and result path

**Interfaces:**
- Produces: final independently reviewed source commit `V` and four digests.

- [ ] **Step 1: Run the complete affected assurance gate once**

  Run serially:

  ```bash
  .venv/bin/pytest -q tests/test_verify_pass200_rsta_scientific_artifact.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/ruff check scripts/verify_pass200_rsta_scientific_artifact.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_verify_pass200_rsta_scientific_artifact.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/python -m py_compile scripts/verify_pass200_rsta_scientific_artifact.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_verify_pass200_rsta_scientific_artifact.py tests/test_diagnose_pass200_rsta_stage_a.py
  git diff --check
  ```

  Expected: PASS. Do not run a real verifier, candidate-free mode, scientific mode, or GPU command.

- [ ] **Step 2: Verify protected paths and commit exact scope**

  Run:

  ```bash
  test -z "$(git diff --name-only -- docs/pass200_rsta_receipt_stage_a_manifest.json reports/generated)"
  git add -- scripts/diagnose_pass200_rsta_stage_a.py scripts/verify_pass200_rsta_scientific_artifact.py tests/test_diagnose_pass200_rsta_stage_a.py tests/test_verify_pass200_rsta_scientific_artifact.py
  test "$(git diff --cached --name-only | sort)" = "$(printf '%s\n' scripts/diagnose_pass200_rsta_stage_a.py scripts/verify_pass200_rsta_scientific_artifact.py tests/test_diagnose_pass200_rsta_stage_a.py tests/test_verify_pass200_rsta_scientific_artifact.py | sort)"
  git diff --cached --check
  git commit -m "implement RSTA artifact roundtrip recovery"
  ```

  Expected: one commit containing exactly those four paths.

- [ ] **Step 3: Obtain a fresh full-source review and repair test-first**

  Run a read-only review of the complete amendment, final source files, tests,
  and history diff without artifact access. Require explicit coverage of the
  canonical producer's exact live roundtrip and bytes, exact ten-key old-H
  projection, exact old-H loader/callable, V/HV self-authentication,
  adapter-only mutation, full legacy validation, equality/bytes, authenticated
  interpreter and parent/child/persisted NumPy runtime, receipt, process
  isolation, source order, atomicity, and unreachable science/GPU.

  For each Critical or Important finding: add one focused failing synthetic
  test, run RED, implement the smallest change within the same four paths, run
  GREEN and the full gate, commit with subject
  `fix RSTA artifact verifier review findings`, and repeat review until clean.

- [ ] **Step 4: Freeze V only after the clean review loop**

  Run:

  ```bash
  verifier_source_commit=$(git rev-parse HEAD)
  test "$(git diff --name-only 043121f8a414b91d7fb2e3d6a1635a6bd585676a "$verifier_source_commit" -- | sort)" = "$(printf '%s\n' docs/superpowers/plans/2026-08-10-pass200-rsta-scientific-artifact-roundtrip-recovery.md scripts/diagnose_pass200_rsta_stage_a.py scripts/verify_pass200_rsta_scientific_artifact.py tests/test_diagnose_pass200_rsta_stage_a.py tests/test_verify_pass200_rsta_scientific_artifact.py | sort)"
  sha256sum scripts/diagnose_pass200_rsta_stage_a.py scripts/verify_pass200_rsta_scientific_artifact.py tests/test_diagnose_pass200_rsta_stage_a.py tests/test_verify_pass200_rsta_scientific_artifact.py
  printf '%s\n' "$verifier_source_commit"
  ```

  The comparison includes this already-committed plan; the source/test portion
  remains exactly four paths. Record `V` and all four digests. Do not edit the
  manifest or open the artifact.

---

### Task 6: Re-prove Future Manifest RED/Green Before the Real Refreeze

**Files:**
- Test: `tests/test_diagnose_pass200_rsta_stage_a.py`
- Verify: both source files
- Protect: real manifest and results

**Interfaces:**
- Consumes: clean reviewed `V`.
- Produces: already-GREEN validators for the exact future manifest and receipt provenance.

- [ ] **Step 1: Run focused future-manifest validators at V**

  Run:

  ```bash
  verifier_source_commit=$(git rev-parse HEAD)
  .venv/bin/pytest -q \
    tests/test_diagnose_pass200_rsta_stage_a.py::test_roundtrip_recovery_manifest_authority_order_and_32_source_paths \
    tests/test_diagnose_pass200_rsta_stage_a.py::test_roundtrip_recovery_projection_rejects_every_nested_mutation \
    tests/test_verify_pass200_rsta_scientific_artifact.py::test_verifier_provenance_binds_v_hv_manifest_file_and_all_32_blobs
  ```

  Expected: PASS using future/synthetic manifests. The real manifest remains old.

- [ ] **Step 2: Confirm no result or manifest path changed**

  Run:

  ```bash
  verifier_source_commit=$(git rev-parse HEAD)
  git diff --quiet -- docs/pass200_rsta_receipt_stage_a_manifest.json
  test "$(git show c04574e2bb751c3229bce673408577cfedc00a88:docs/pass200_rsta_receipt_stage_a_manifest.json | sha256sum | cut -d ' ' -f 1)" = 9260329a0f9ad45257f51292d40c3a6d70c9494ea3e8fd185afcf8484f9378fe
  test ! -e "reports/generated/pass200_rsta_receipt/${verifier_source_commit}-scientific-artifact-roundtrip-validation.json"
  ```

  Expected: old manifest blob exact and no validation receipt.

---

### Task 7: Manifest-Only Refreeze, Independent Review, and Freeze HV

**Files:**
- Modify only: `docs/pass200_rsta_receipt_stage_a_manifest.json`
- Protect: all source, tests, artifacts, and results

**Interfaces:**
- Consumes: reviewed `V`, amendment authority, exact source hashes.
- Produces: reviewed manifest-only handoff `HV` with parent `V`.

- [ ] **Step 1: Apply the exact manifest transition**

  Insert the recovery authority after
  `normwise_adjoint_sign_control_amendment`:

  ```json
  "scientific_artifact_roundtrip_recovery_amendment": {
    "path": "docs/pass200_rsta_scientific_artifact_roundtrip_recovery_amendment_2026-08-10.md",
    "sha256": "6e1767e802295fcfbf29e7151ac05991a016994ca92b99bf2e2cbcd46e4e9591",
    "commit": "043121f8a414b91d7fb2e3d6a1635a6bd585676a"
  }
  ```

  Set `current_scientific_source.git_revision` to exact `V`. Insert
  `scripts/verify_pass200_rsta_scientific_artifact.py` as exact path 4. Refresh
  only the producer/verifier hashes; retain all unchanged 30 hashes and every
  prior manifest domain byte-semantically.

- [ ] **Step 2: Run already-GREEN validation against the real manifest**

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_verify_pass200_rsta_scientific_artifact.py -k 'provenance or manifest or receipt_path'
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'roundtrip_recovery_manifest or roundtrip_recovery_projection'
  git diff --check -- docs/pass200_rsta_receipt_stage_a_manifest.json
  ```

  Expected: PASS without opening the artifact.

- [ ] **Step 3: Commit only the manifest**

  Run:

  ```bash
  verifier_source_commit=$(git rev-parse HEAD)
  test "$(git diff --name-only)" = docs/pass200_rsta_receipt_stage_a_manifest.json
  git add -- docs/pass200_rsta_receipt_stage_a_manifest.json
  test "$(git diff --cached --name-only)" = docs/pass200_rsta_receipt_stage_a_manifest.json
  git diff --cached --check
  git commit -m "refreeze RSTA artifact verifier handoff"
  verifier_handoff_commit=$(git rev-parse HEAD)
  verifier_source_commit=$(git rev-parse HEAD^)
  test "$(git rev-parse "$verifier_handoff_commit^")" = "$verifier_source_commit"
  test "$(git diff-tree --no-commit-id --name-only -r "$verifier_handoff_commit")" = docs/pass200_rsta_receipt_stage_a_manifest.json
  ```

  Expected: manifest-only `HV` with exact parent `V`.

- [ ] **Step 4: Obtain a full manifest/provenance review before execution**

  Review the complete `HV` manifest, `V` source, amendment, plan, and commit
  ancestry without artifact access. Require exact authority bytes/blob/order,
  all 32 paths/digests, unchanged prior domains, `HV^ == V`, self-cycle
  avoidance, receipt derivation, and verifier import/child bindings.

  If any Critical/Important finding exists, run no verifier and do not stack a
  repair commit on the rejected handoff. Create a new isolated branch/worktree
  directly at exact `V`, apply only the repaired manifest there, rerun the
  already-GREEN validators, and commit a replacement manifest-only handoff whose
  parent is still exactly `V`. Update `HV`, archive the rejected handoff as
  non-authoritative review evidence, and repeat until a direct-child `HV` is
  clean. Never rewrite or execute from a rejected handoff.

- [ ] **Step 5: Record exact HV and derived receipt path**

  Run:

  ```bash
  verifier_handoff_commit=$(git rev-parse HEAD)
  verifier_source_commit=$(git rev-parse HEAD^)
  manifest_sha256=$(sha256sum docs/pass200_rsta_receipt_stage_a_manifest.json | cut -d ' ' -f 1)
  receipt_output="reports/generated/pass200_rsta_receipt/${verifier_handoff_commit}-scientific-artifact-roundtrip-validation.json"
  test ! -e "$receipt_output"
  test ! -L "$receipt_output"
  printf 'V=%s\nHV=%s\nmanifest_sha256=%s\nreceipt=%s\n' "$verifier_source_commit" "$verifier_handoff_commit" "$manifest_sha256" "$receipt_output"
  ```

  These shell variables are derivations, not authority placeholders. The
  verifier independently rederives them.

---

### Task 8: Execute Exactly One Outcome-Blind Offline Validation Attempt

**Files:**
- Read once: exact immutable artifact
- Create at most once: receipt path derived from `HV`
- Never modify: artifact, manifest, source, tests, or other results

**Interfaces:**
- Consumes: independently reviewed clean `V/HV` and immutable artifact bytes.
- Produces: exactly one atomic provenance-only `VALID` or `INVALID` receipt, or structural stop with no status claim.

- [ ] **Step 1: Create a fresh detached clean checkout at HV without artifact access**

  Transfer exact `HV` plus the immutable artifact to the registered offline
  host. Create a new checkout and authenticate:

  ```bash
  verifier_handoff_commit=$(git rev-parse HEAD)
  test "$(git rev-parse --abbrev-ref HEAD)" = HEAD
  test -z "$(git status --porcelain --untracked-files=no)"
  test "$(git rev-parse "$verifier_handoff_commit^")" = "$(jq -r .current_scientific_source.git_revision docs/pass200_rsta_receipt_stage_a_manifest.json)"
  test "$(git diff-tree --no-commit-id --name-only -r "$verifier_handoff_commit")" = docs/pass200_rsta_receipt_stage_a_manifest.json
  ```

  Do not run `nvidia-smi`; GPU access is neither needed nor authorized.

- [ ] **Step 2: Authenticate output absence before the artifact is opened**

  Run:

  ```bash
  verifier_handoff_commit=$(git rev-parse HEAD)
  receipt_output="reports/generated/pass200_rsta_receipt/${verifier_handoff_commit}-scientific-artifact-roundtrip-validation.json"
  test ! -e "$receipt_output"
  test ! -L "$receipt_output"
  test -d reports/generated/pass200_rsta_receipt
  ```

  Expected: output is absent. Stop if not.

- [ ] **Step 3: Run the verifier command once and only once**

  Run exactly this one process:

  ```bash
  verifier_handoff_commit=$(git rev-parse HEAD)
  receipt_output="reports/generated/pass200_rsta_receipt/${verifier_handoff_commit}-scientific-artifact-roundtrip-validation.json"
  set +e
  CUDA_VISIBLE_DEVICES='' .venv/bin/python -I -B \
    scripts/verify_pass200_rsta_scientific_artifact.py \
    --manifest docs/pass200_rsta_receipt_stage_a_manifest.json \
    --artifact reports/generated/pass200_rsta_receipt/c04574e2bb751c3229bce673408577cfedc00a88-stage-a.json \
    --output "$receipt_output" \
    --validate-immutable-artifact-once
  validation_exit=$?
  set -e
  printf 'exit=%s\n' "$validation_exit"
  ```

  Do not pipe, tee, inspect, summarize, or rerun. Record only the process exit.
  Any exit or interruption consumes attempt `1`.

- [ ] **Step 4: Validate only the provenance receipt and stop**

  If the receipt exists, run this exact receipt-only import. Module import must
  be side-effect free and cannot open the artifact:

  ```bash
  verifier_handoff_commit=$(git rev-parse HEAD)
  receipt_output="reports/generated/pass200_rsta_receipt/${verifier_handoff_commit}-scientific-artifact-roundtrip-validation.json"
  RECEIPT_OUTPUT="$receipt_output" .venv/bin/python -I -B - <<'PY'
  import importlib.util
  import os
  import sys
  from pathlib import Path

  repository = Path.cwd().resolve()
  verifier_path = repository / "scripts/verify_pass200_rsta_scientific_artifact.py"
  spec = importlib.util.spec_from_file_location("_rsta_receipt_validator", verifier_path)
  if spec is None or spec.loader is None:
      raise SystemExit(2)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  try:
      spec.loader.exec_module(module)
      receipt = module.strict_json_object(
          Path(os.environ["RECEIPT_OUTPUT"]).read_bytes(), name="validation receipt"
      )
      module.validate_roundtrip_receipt(receipt)
      print(f"status={receipt['status']}")
      print(f"V={receipt['verifier_provenance']['source_commit']}")
      print(f"HV={receipt['verifier_provenance']['handoff_commit']}")
  finally:
      sys.modules.pop(spec.name, None)
  PY
  sha256sum "$receipt_output"
  ```

  Record only receipt SHA-256, `status`, `V`, and `HV`. Never print another
  artifact-derived field.

  - Exit `0` requires a receipt with `status="VALID"`; preserve it and STOP.
  - Exit `1` requires a receipt with `status="INVALID"`; preserve it and STOP.
  - Exit `2`, signal, timeout, missing receipt, mismatched receipt, or atomic
    publication failure is structural; preserve all evidence and STOP.

  Under every branch: no second verifier, candidate-free audit, scientific
  producer, GPU command, artifact rewrite, result interpretation, or scientific
  publication is authorized.

---

## Final self-review checklist

- [ ] Amendment commit/SHA/path occur consistently in every binding and manifest object.
- [ ] Artifact path/SHA/PID/exit, `H`, `S`, old manifest, and diagnostic digests are exact.
- [ ] Canonical keys are ordered `str(label)` and aliases/types/order are rejected.
- [ ] Live producer roundtrip uses `exact_ordered_equal`, preserves signed zero,
      and reserializes to byte-identical producer bytes without ordinary dict equality.
- [ ] Old persisted manifest is exactly the H-derived ordered ten-key projection;
      all five later authorities omitted by S remain absent.
- [ ] Adapter mutates only copied key types and invokes complete old `scientific_payload`.
- [ ] Recursive comparison is type/order/signed-zero exact; producer serialization is byte-identical.
- [ ] Old child uses old-H cwd/import/manifest/HEAD; executing verifier bytes remain V/HV-authenticated.
- [ ] Exact `.venv/bin/python`, observed Python `3.12.3`, NumPy module identity,
      and parent/child/persisted version agreement precede recomputation and bind receipt runtime.
- [ ] Source scope is exactly four files; manifest scope is exactly one later file.
- [ ] Exact prior 31 relative order plus verifier at position 4 yields 32 paths.
- [ ] Receipt is exact, atomic, no-clobber, provenance-only, and contains no science.
- [ ] V precedes HV; receipt follows HV; no self-cycle or unresolved authority placeholder exists.
- [ ] The real artifact remains unopened until the one Task 8 attempt.
- [ ] No GPU, candidate-free, scientific rerun, rewrite, or second validation attempt appears anywhere.
