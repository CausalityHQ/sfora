# Pass 205 RDGC Falsifier and No-Training Virtual-Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the prospective Receiver-Diagonal Gain Calibration (RDGC) one-attempt diagnostic that closes cheaply on a fresh scalar-gain falsifier or, only after exact survival, runs a fresh no-training norm-matched virtual-update panel.

**Architecture:** One import-safe diagnostic owns strict authority/provenance validation, pure selection/formula/decision functions, a candidate-free all-four-seed integrity prefix, the serial preliminary, the automatically gated virtual panel, and one atomic result. Source and tests are reviewed as final commit `V_G`; a new manifest is then created alone in direct-child `HV_G`; one clean DGX process executes both phases without training or opening the old RSTA scientific artifact.

**Tech Stack:** Python 3.12.3, PyTorch functional VJP/JVP, NumPy PCG64(201), strict insertion-ordered JSON, SHA-256/Git blob authentication, pytest, Ruff, `py_compile`, FP32 action arithmetic, named FP64 reductions, atomic hard-link publication.

## Global Constraints

- Implement `docs/pass205_rdgc_candidate_2026-08-10.md` literally at commit `2599bb8c3f8238bb70f0f8935c0960cadda0dfb6`, SHA-256 `a2d9103e8856f6edf8a1fbaddfe3b0b18a0a63017710e4fb0a85476978dda6f7`.
- RDGC is independent. Never change, relabel, rerun, or reinterpret RSTA's authenticated `VALID` artifact and `UNRESOLVED/no_pass_or_fail_rule` decision.
- The old criterion-2 failure and disclosed aggregates are hypothesis generation only. Tests use unrelated synthetic values. Source may bind their documented provenance but may not encode them as expected RDGC results.
- Never open the old RSTA scientific artifact. Authenticate only its provenance-only roundtrip validation receipt and manifest chain.
- Candidate formula is exactly `e=log((||b||+1e-8)/(stopgrad(||s||)+1e-8))`, `R=0.5*e^2`. No `s_hat`, `cos(b,s)`, angular receiver-self target, or vector self target may be reachable.
- The preliminary must finish and satisfy every SURVIVES predicate before any full-panel correction, margin, control, or bootstrap object is constructed.
- All four seed bindings and integrity audits pass before any preliminary candidate state or call.
- One B=180 CUDA graph at peak. No live CUDA tensor/action/closure from two graphs may coexist. Only detached JSON data and the exact current CPU parameter directions cross graphs.
- Exact operator order is `pa`, `rdgc`, `raw_cotangent`, `full_motion`, `batch_global_gain`, `scalar_diagonal_raw`, `per_example_gradient_normalized`.
- Action/JVP arithmetic is FP32. Norms, dot products, cosine denominators, aggregate reductions, bootstrap arrays, and hashes use cast-before-operation FP64 in exact named order.
- Run one fresh DGX command once. Preliminary survival continues automatically in-process. Every outcome or interruption consumes attempt 1.
- Training, optimizer state, parameter mutation, benchmark execution, query/gallery scoring, hyperparameter tuning, and `src/` changes are forbidden.
- Protect the untracked root files `HANDOFF_BRIEF.md`, `RSPG_SPECDEFECT.md`, and `RSPG_TASK.md`.
- Protect every existing source, test, manifest, artifact, report, and result. Implementation aggregate scope from this plan commit is exactly the new diagnostic and test; handoff scope is exactly the new manifest.

## Candidate and current-chain binding

```text
candidate path: docs/pass205_rdgc_candidate_2026-08-10.md
candidate SHA-256: a2d9103e8856f6edf8a1fbaddfe3b0b18a0a63017710e4fb0a85476978dda6f7
candidate commit C_G: 2599bb8c3f8238bb70f0f8935c0960cadda0dfb6
candidate parent / current RSTA handoff HV_R: e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae
RSTA verifier source V_R: 3c368713e0890c0ffc63308f07d8d4ee5b19db1c
HV_R manifest SHA-256: fb089cf5905cea32a9d22563b50160af5fc8643efb657c49cb519d6d0c0da80b
RSTA producer H/S: c04574e2bb751c3229bce673408577cfedc00a88 / 15234a529a181c39c1c8b6477ad7eb7823fd0798
RSTA artifact SHA-256: e9bcd77c6e372e9c3bab4a420b97ff56f8ea164cbca56f53ec9c99a3b3c527ae
RSTA outcome: VALID scientific bytes; UNRESOLVED/no_pass_or_fail_rule decision
```

The plan commit `P_G` is the commit containing this file after its independent
review. `V_G` is the final reviewed descendant whose aggregate diff from `P_G`
is confined to the exact two source/test paths below. `HV_G` is a direct child
of `V_G` whose only changed path is the new RDGC manifest. These symbols are
derived commits, not caller placeholders.

## Exact file structure

- Create `scripts/diagnose_pass205_rdgc_stage_b.py`: import-safe strict schema, provenance, selection, fields, preliminary, controls, update/JVP panel, aggregation, one-shot CLI, and atomic writer.
- Create `tests/test_diagnose_pass205_rdgc_stage_b.py`: synthetic RED/GREEN coverage for every formula, boundary, provenance, graph, lifetime, schema, and process contract.
- Later create only `docs/pass205_rdgc_stage_b_manifest.json`: new immutable execution authority; never modify the Pass 200 manifest.
- Later create at most once `reports/generated/pass205_rdgc_stage_b/<HV_G>-rdgc-stage-b.json` on the DGX.

The future scientific source order is exactly:

```python
RDGC_SOURCE_ORDER = (
    "scripts/diagnose_pass159_cotangent_stage_a.py",
    "scripts/diagnose_pass200_rsta_stage_a.py",
    "scripts/diagnose_pass205_rdgc_stage_b.py",
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

The prior 32 paths retain relative order; the new diagnostic is exact path 3.
Tests and plans are excluded from scientific source files.

---

### Task 1: Authenticate and Independently Review the Prospective Authority

**Files:**
- Review: `docs/pass205_rdgc_candidate_2026-08-10.md`
- Review: `docs/pass200_rsta_candidate_2026-08-09.md`
- Review: `docs/pass200_rsta_gate2_primary_audit_2026-08-09.md`
- Review: this plan
- Do not read: old RSTA scientific artifact or any row-level result

**Interfaces:**
- Produces: clean candidate/specification approval and recorded `P_G` before source work.

- [ ] **Step 1: Authenticate candidate chronology and protected scope**

  Run:

  ```bash
  test "$(git rev-parse 2599bb8c3f8238bb70f0f8935c0960cadda0dfb6^{commit})" = 2599bb8c3f8238bb70f0f8935c0960cadda0dfb6
  test "$(git rev-parse 2599bb8c3f8238bb70f0f8935c0960cadda0dfb6^)" = e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae
  test "$(git show 2599bb8c3f8238bb70f0f8935c0960cadda0dfb6:docs/pass205_rdgc_candidate_2026-08-10.md | sha256sum | cut -d ' ' -f 1)" = a2d9103e8856f6edf8a1fbaddfe3b0b18a0a63017710e4fb0a85476978dda6f7
  test "$(git diff-tree --no-commit-id --name-only -r 2599bb8c3f8238bb70f0f8935c0960cadda0dfb6)" = docs/pass205_rdgc_candidate_2026-08-10.md
  test "$(git rev-parse e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae^)" = 3c368713e0890c0ffc63308f07d8d4ee5b19db1c
  test "$(git diff-tree --no-commit-id --name-only -r e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae)" = docs/pass200_rsta_receipt_stage_a_manifest.json
  test "$(git show e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae:docs/pass200_rsta_receipt_stage_a_manifest.json | sha256sum | cut -d ' ' -f 1)" = fb089cf5905cea32a9d22563b50160af5fc8643efb657c49cb519d6d0c0da80b
  git diff --check e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae 2599bb8c3f8238bb70f0f8935c0960cadda0dfb6
  ```

  Expected: all pass; no source/test/manifest/result/artifact changed.

- [ ] **Step 2: Obtain the mandatory primary-source/Gate-2 review**

  Review the exact seven-item literature/audit tuple in the candidate. Require
  explicit adjudication of receiver-specific full/diagonal scalar gain versus
  DoCL, MGS, NINT, Charpiat tangent shaping, GradNorm, K-FAC, generic functional
  damping, and the failed RSTA angular proposition. The reviewer must not read
  old row values or suggest repairing RSTA.

  Required verdict is `LIVE-NARROW` for the exact scalar formula. `DEAD` closes
  RDGC and authorizes no source. `UNRESOLVED` requires a prospective docs-only
  audit repair before source. Broad novelty language is Critical.

- [ ] **Step 3: Review the full candidate and plan contract**

  Require exact coverage of fresh disjoint selection, two contexts, nested
  contributor masks, global-scalar/heterogeneity gates, all six corrections,
  per-example gradient normalization, update norm matching, margins, paired
  bootstrap, decisions, graph lifetime, all-four integrity prefix, output,
  one-attempt behavior, source/manifest ancestry, and no training.

  Repair docs only and rebind this plan if any predicate or provenance finding
  exists. Do not proceed with stale candidate bytes.

---

### Task 2: Write Pure Formula, Control, Normalization, and Decision REDs

**Files:**
- Create: `tests/test_diagnose_pass205_rdgc_stage_b.py`
- Do not create yet: `scripts/diagnose_pass205_rdgc_stage_b.py`

**Interfaces:**
- Produces tests for the future pure functions below.

- [ ] **Step 1: Freeze exact pure interfaces**

  Tests require:

  ```python
  rdgc_error(torch_module, b, s, *, epsilon: float = 1e-8)
  rdgc_penalty(torch_module, b, s, *, epsilon: float = 1e-8)
  control_penalties(torch_module, *, b, s, dbar, receiver_fields, per_example_gradients)
  fp64_named_norm(torch_module, values: tuple[Tensor, ...])
  fp64_named_dot(torch_module, left: tuple[Tensor, ...], right: tuple[Tensor, ...])
  normalize_virtual_updates(torch_module, p, corrections, *, alpha: float = 0.10)
  preliminary_metrics(rows: list[dict[str, object]]) -> dict[str, object]
  decide_preliminary(aggregates: dict[str, object]) -> dict[str, object]
  paired_bootstrap(panel_rows: list[dict[str, object]]) -> dict[str, object]
  decide_panel(aggregates: dict[str, object], bootstrap: dict[str, object]) -> dict[str, object]
  validate_scientific_payload(value: dict[str, object]) -> None
  ```

- [ ] **Step 2: Add exact candidate-formula REDs**

  Add tests:

  ```text
  test_rdgc_is_exact_half_squared_log_gain_error
  test_rdgc_detaches_only_scalar_self_norm
  test_rdgc_has_no_angular_or_vector_self_target_reachability
  test_rdgc_requires_fp32_actions_and_finite_nonzero_norms
  ```

  Use a recording tensor double that distinguishes `detach(s).norm()` from the
  registered `s.norm().detach()`; only the latter passes. Require epsilon to be
  added after the numerator norm and after detached denominator norm. Perturb
  the direction of `s` at fixed norm and require identical penalty/gradient;
  perturb its norm and require the expected scalar change. Patch cosine,
  normalization, and `s / norm(s)` entry points to raise if called.

- [ ] **Step 3: Add every control-formula RED**

  Add:

  ```text
  test_control_order_and_formulas_are_literal
  test_batch_global_gain_uses_eight_receiver_geometric_mean
  test_scalar_diagonal_raw_uses_batch_gain_times_each_raw_norm
  test_per_example_gradient_normalization_uses_all_180_in_row_order
  test_full_motion_control_is_generic_normalized_damping
  ```

  Assert exact operator order. For PGN, use unequal synthetic gradient norms,
  require FP64 geometric mean, detached coefficients, exact ordered sum, and a
  recording sentinel proving no contributor is skipped, sorted, or reused.
  Reject replacing per-example normalization with clipping, mean gradients,
  scalar loss weights, or a preconditioner.

- [ ] **Step 4: Add update-normalization REDs**

  Add:

  ```text
  test_virtual_updates_match_pa_parameter_norm_in_named_order
  test_virtual_update_uses_alpha_point_one_before_final_normalization
  test_virtual_update_rejects_zero_nonfinite_wrong_dtype_and_reordered_trees
  ```

  Hand-compute in float64:

  ```python
  c_hat = p_norm * c / c_norm
  v = p + 0.10 * c_hat
  u = p_norm * v / fp64_named_norm(v)
  ```

  Require relative final-norm error `<=5e-7`, FP32 returned tensors, and exact
  original parameter-tree order.

- [ ] **Step 5: Add exhaustive decision-boundary REDs**

  Add:

  ```text
  test_preliminary_survival_requires_every_literal_predicate
  test_preliminary_close_precedence_and_exact_boundaries
  test_preliminary_middle_region_is_unresolved
  test_panel_pass_requires_every_pa_control_context_and_alias_predicate
  test_panel_close_precedence_and_exact_boundaries
  test_panel_middle_region_is_unresolved
  test_any_integrity_schema_or_nonfinite_fault_is_invalid
  ```

  Generate the Cartesian boundary cases at exact equality, one ULP below, and
  one ULP above for every threshold. Prove CLOSE precedence order, exact first
  decisive clauses, and authorized actions. Never use a truthy/falsy float or
  short-circuit away validation of later required inputs.

- [ ] **Step 6: Add paired-bootstrap REDs**

  Add `test_paired_bootstrap_uses_exact_32_labels_four_seeds_and_pcg64_201` and
  `test_bootstrap_rejects_unpaired_missing_reordered_or_partial_rows`.
  Independently hand-compute a tiny oracle, then require production shape
  `(10_000,)`, one shared index vector per replicate across all seeds/contexts/
  operators/metrics, ordinary 2.5th percentile, FP64 C-order bytes, hashes, and
  exact NumPy version.

- [ ] **Step 7: Run focused tests and record RED**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'rdgc or control or virtual_update or preliminary or panel or bootstrap'
  ```

  Expected: collection/import FAIL because source is absent. Preserve output;
  do not weaken tests.

---

### Task 3: Implement Pure Candidate, Controls, Decisions, and Schemas GREEN

**Files:**
- Create: `scripts/diagnose_pass205_rdgc_stage_b.py`
- Test: `tests/test_diagnose_pass205_rdgc_stage_b.py`

**Interfaces:**
- Produces all Task 2 interfaces without importing torch at module import.

- [ ] **Step 1: Create an import-safe module and frozen constants**

  Import only standard-library modules and NumPy at module import. Put all torch
  access behind explicit functions receiving `torch_module`. Define literal
  candidate/provenance paths, commits, hashes, thresholds, operator order,
  contributor counts `(1,8,32,180)`, `PCG64(201)`, and source order.

- [ ] **Step 2: Implement formula and FP64 reduction primitives**

  Use explicit `tensor.float()` action checks. `fp64_named_norm` must iterate
  the input tuple once in registered order and accumulate each
  `value.to(torch.float64).reshape(-1)` square product before the square root.
  `fp64_named_dot` uses the same cast-before-product/order. No concatenation,
  foreach reduction, reversed accumulation, or duplicate arithmetic is allowed.

- [ ] **Step 3: Implement exact controls and normalized updates**

  Keep the six control constructors separate and assemble them once through a
  literal ordered dispatcher. Return fresh tuples; validate identity, dtype,
  device, shape, finite values, names, and lengths before reductions.

- [ ] **Step 4: Implement pure aggregate, bootstrap, and decision functions**

  Validate exact row order and schemas before any mean/median. Use NumPy
  float64 arrays and stable insertion order. Evaluate all CLOSE predicates in
  literal order, then all PASS predicates, then UNRESOLVED. Return exact Boolean
  ledgers and first decisive clause.

- [ ] **Step 5: Implement exact full and reduced output validators**

  Freeze the candidate's top-level and nested key order. Require exact concrete
  JSON types, finite floats including signed-zero-aware comparisons where
  equality matters, fixed scalars, status/phase/null relations, all row counts,
  all hashes, and forbidden scientific content in pre-science INVALID output.

- [ ] **Step 6: Run Task 2 tests GREEN**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'rdgc or control or virtual_update or preliminary or panel or bootstrap or schema'
  ```

  Expected: PASS, CPU-only.

---

### Task 4: Write Fresh Selection, Provenance, and Preliminary REDs

**Files:**
- Modify: `tests/test_diagnose_pass205_rdgc_stage_b.py`
- Test only: source from Task 3

**Interfaces:**
- Requires future functions:

  ```python
  authenticate_authority(repository: Path, manifest_path: Path, receipt_path: Path) -> dict[str, object]
  load_authenticated_rsta_module(repository: Path, source: dict[str, object]) -> ModuleType
  build_rdgc_selection(training_ids, labels, *, old_selection) -> dict[str, object]
  nested_contributor_masks(batch_ids, receiver_id, *, seed: int, context: str) -> tuple[tuple[bool, ...], ...]
  run_all_seed_integrity_prefix(bindings, *, adapters) -> list[dict[str, object]]
  run_preliminary(selection, bindings, *, adapters) -> dict[str, object]
  ```

- [ ] **Step 1: Add authority/provenance REDs**

  Add:

  ```text
  test_authority_binds_candidate_plan_vg_hvg_and_exact_33_sources
  test_authority_binds_vr_hvr_manifest_and_validation_receipt
  test_authority_never_opens_old_scientific_artifact
  test_authenticated_rsta_loader_uses_exact_file_blob_and_cleans_sys_modules
  test_authority_rejects_wrong_ancestry_scope_dirty_worktree_and_every_digest
  ```

  Use local temporary Git repositories. Deny access to the old artifact path.
  Require receipt exact path, manifest-bound digest, strict schema, `VALID`,
  exact old artifact path/SHA and `V_R/HV_R`, without reading artifact bytes.
  Load the reused RSTA module only by authenticated absolute file path after
  blob/worktree verification; require exact `__file__`, remove its `sys.modules`
  entry, and restore `sys.path` even on exception.

- [ ] **Step 2: Add fresh-selection/disjointness REDs**

  Add:

  ```text
  test_selection_recomputes_old_64_labels_without_old_result_access
  test_selection_freezes_8_preliminary_then_32_panel_identities
  test_selection_roles_supports_receivers_and_contexts_are_disjoint
  test_selection_uses_exact_domains_orders_and_two_172_distractor_sets
  test_selection_rejects_duplicate_missing_overlap_and_insufficient_rows
  ```

  Patch all old result/report candidate paths to raise. Require exact synthetic
  hash oracle, identical receivers across contexts/seeds, mutually disjoint
  distractors, supports outside graphs, and exact B=180 final order.

- [ ] **Step 3: Add nested contributor and preliminary-metric REDs**

  Add:

  ```text
  test_nested_masks_are_receiver_plus_prefix_counts_1_8_32_180
  test_preliminary_b1_equals_self_action_before_reduction
  test_preliminary_metrics_include_context_stability_count_and_global_scalar
  test_preliminary_close_or_unresolved_never_constructs_panel
  test_preliminary_survival_constructs_panel_once_without_operator_input
  ```

  Require direct live `torch.equal` and equal action hashes for `b_1` and `s`
  before detach/copy. Mutation tests reorder contributors, recompute dbar on a
  subset, use different receivers across contexts, pool before per-seed
  statistics, use Pearson instead of Spearman, or fit the global scalar on the
  other context; all fail.

- [ ] **Step 4: Add all-four integrity-prefix REDs**

  Add `test_all_four_seed_integrity_passes_before_any_candidate_state_or_call`
  and `test_later_seed_integrity_failure_has_zero_preliminary_and_panel_calls`.
  Recording adapters must prove exact seed order `0,1,2,3`, no alternate/full
  cache, RDGC tensor, contributor mask, margin, correction, or bootstrap before
  all four audits pass, and immediate fail-fast on every seed.

- [ ] **Step 5: Run new nodes RED**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'authority or selection or nested or preliminary or integrity_prefix'
  ```

  Expected: FAIL on absent orchestration functions.

---

### Task 5: Implement Provenance, Selection, and Cheapest Preliminary GREEN

**Files:**
- Modify: `scripts/diagnose_pass205_rdgc_stage_b.py`
- Test: `tests/test_diagnose_pass205_rdgc_stage_b.py`

**Interfaces:**
- Produces all Task 4 functions and preliminary result objects.

- [ ] **Step 1: Implement strict Git/manifest/receipt authentication**

  Use argument-vector Git calls only, exact timeouts/output caps, no shell, no
  environment `PYTHONPATH`, and exact SHA-256. Validate candidate and plan
  authority, `HV_G^==V_G`, manifest-only handoff, 33 source blobs/worktree
  files, `V_R/HV_R`, old manifest, receipt, and receipt relations before any
  torch import or checkpoint open.

- [ ] **Step 2: Implement exact authenticated source loading**

  Source-load Pass 200 helpers under one private name after authentication.
  Reject a preexisting module name, mismatched `__file__`, loader substitution,
  changed bytes between hash and execution, or current-directory import. Clean
  up on success/failure. Do not call its scientific CLI or open its artifact.

- [ ] **Step 3: Implement fresh deterministic selection**

  Reuse only pure authenticated training-ID rules. Build immutable selection
  records for all seeds before candidate work. Validate exact disjointness and
  transform/tensor hashes, then release builders and binding-only arrays.

- [ ] **Step 4: Implement all-four integrity and preliminary schedule**

  Complete all seed integrity records first. For preliminary science process
  seed, context, receiver, count in literal order. Each fresh graph computes
  exact full contextual `dbar` plus only its registered mask action; directly
  compare `b_1/s`, reduce/hash, detach JSON evidence, then delete graph/model/
  CUDA state before the next count. Do not accumulate full-panel state.

- [ ] **Step 5: Implement automatic preliminary branch**

  Validate complete 64 preliminary rows (`4*2*8`), aggregate, and decide.
  CLOSE/UNRESOLVED returns immediately with `panel=null`, `bootstrap=null`.
  INVALID discards all partial scientific rows and returns the reduced schema.
  Only exact SURVIVES calls the panel builder once.

- [ ] **Step 6: Run Task 4 nodes GREEN**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'authority or selection or nested or preliminary or integrity_prefix'
  ```

  Expected: PASS with synthetic/tiny fixtures and no real artifact or GPU.

---

### Task 6: Write Virtual-Panel, Lifetime, Atomic, and Process REDs

**Files:**
- Modify: `tests/test_diagnose_pass205_rdgc_stage_b.py`

**Interfaces:**
- Requires future functions:

  ```python
  compute_parameter_correction(field, operator: str, *, torch_module) -> tuple[Tensor, ...]
  evaluate_virtual_direction(context, direction, q, *, torch_module) -> dict[str, object]
  run_full_panel(selection, bindings, preliminary, *, adapters) -> dict[str, object]
  write_json_atomic(path: Path, payload: dict[str, object]) -> None
  main(argv: Sequence[str] | None = None) -> int
  ```

- [ ] **Step 1: Add exact virtual-panel schedule REDs**

  Add:

  ```text
  test_panel_uses_exact_seed_group_receiver_operator_context_order
  test_context_a_builds_each_correction_once_and_context_b_never_rebuilds_it
  test_virtual_jvp_uses_exact_detached_cpu_direction_and_fresh_graph
  test_panel_persists_every_32_by_4_by_2_row_and_operator_record
  test_panel_fail_fast_stops_all_later_operator_and_context_calls
  ```

  Recording adapters require candidate first, then controls in frozen order;
  context A then B for each detached direction; exact 256 context rows
  (`32*4*2`); and no hidden alternate ordering or omitted negative control.

- [ ] **Step 2: Add control and alias-sentinel REDs**

  Add:

  ```text
  test_raw_control_is_dbar_angle_not_self_angle
  test_batch_global_and_scalar_raw_controls_cannot_reuse_receiver_target
  test_pgn_control_has_180_serial_contributors_and_no_live_gradient_list
  test_full_motion_control_has_no_diagonal_or_raw_target
  test_correction_cosines_use_original_cpu_named_order
  ```

  Use identity-sensitive tensor doubles and mutation sentinels. Prohibit a
  shared helper that silently substitutes candidate arithmetic into controls.

- [ ] **Step 3: Add graph/CPU lifetime and peak-memory REDs**

  Add:

  ```text
  test_only_one_cuda_graph_and_one_operator_action_live_at_peak
  test_action_tensors_are_reduced_hashed_detached_and_deleted_before_next_graph
  test_only_p_and_current_rdgc_cpu_directions_cross_control_graphs
  test_weakrefs_die_after_every_preliminary_count_operator_and_context
  test_invalid_or_close_releases_models_caches_closures_and_cuda_references
  ```

  Exercise CUDA reachability when available and a strict fake otherwise.
  Weakrefs cover outputs, tangents, VJPs, per-example gradients, functional
  parameters, images, proxies, q, `p`, candidate correction, control correction,
  and JVP direction. Tests must inspect CPU action lifetime, not CUDA only.

- [ ] **Step 4: Add output/atomic/process REDs**

  Add:

  ```text
  test_full_preliminary_and_reduced_invalid_schemas_reject_every_nested_mutation
  test_atomic_writer_never_replaces_follows_or_leaves_temporary
  test_cli_is_exact_fresh_process_one_attempt_and_no_retry
  test_cli_preliminary_close_never_imports_panel_builder
  test_cli_survival_continues_in_same_pid_without_second_command
  test_training_benchmark_optimizer_and_old_artifact_paths_are_unreachable
  test_hidden_torch_import_and_pre_prefix_candidate_construction_fail
  ```

  Recursively remove/add/reorder/mistype every field. Patch training, optimizer,
  benchmark, publication, query/gallery, old artifact, scientific RSTA CLI, and
  source `src/` imports to raise. Require exact command, environment, cwd,
  output derivation from `HV_G`, absent output/temp, PID consistency, one
  checkpoint-open transition, and exact exit/status relations.

- [ ] **Step 5: Run new nodes RED**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'panel or control or lifetime or weakref or atomic or cli or unreachable'
  ```

  Expected: FAIL on absent panel/CLI implementation.

---

### Task 7: Implement Virtual Panel, Atomic Output, and One-Shot CLI GREEN

**Files:**
- Modify: `scripts/diagnose_pass205_rdgc_stage_b.py`
- Test: `tests/test_diagnose_pass205_rdgc_stage_b.py`

**Interfaces:**
- Produces complete scientific payload or reduced INVALID payload exactly once.

- [ ] **Step 1: Implement one-operator correction graphs**

  For each operator, construct a fresh exact B=180 field and compute only that
  correction. Reduce its norm and CPU direction before graph deletion. Candidate
  direction may persist on CPU while controls are serially compared; no other
  control direction persists after its two context JVPs and cosine.

- [ ] **Step 2: Implement serial PGN without a live 180-gradient list**

  First pass computes/hashes the 180 FP64 norms serially and obtains detached
  `nu`. Rebuild one fresh graph, traverse rows in exact order, scale each live
  `g_j`, add immediately into the one FP32 accumulator, and delete `g_j` before
  the next. Validate row-call count/order and accumulator provenance.

- [ ] **Step 3: Implement norm-matched directions and context JVPs**

  Normalize on CPU in named order, validate `5e-7`, move one direction to the
  registered GPU, compute context-A JVP/action/reductions, delete graph; then do
  context B from the same CPU bytes. Hash direction bytes and require the same
  hash in both context records.

- [ ] **Step 4: Implement complete panel aggregation/decision**

  Require all 256 rows and every operator. Aggregate/paired-bootstrap only after
  all rows exist. Apply CLOSE precedence then PASS then UNRESOLVED. Preserve all
  raw rows and distribution hashes; no selected aggregate may substitute.

- [ ] **Step 5: Implement atomic output and CLI**

  Parse exactly `--manifest`, `--output`, `--scientific-once`; reject aliases,
  extra flags, symlinks, alternate paths, existing output/temp, or wrong cwd.
  Authenticate before torch import; checkpoint open consumes attempt 1. Use
  exclusive temp, flush/fsync, hard-link no-replace, directory fsync, temp
  unlink, strict reload. Exit `0` for PASS/CLOSE/UNRESOLVED published output and
  `2` for published INVALID or structural failure; never retry.

- [ ] **Step 6: Run all focused tests GREEN**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py
  ```

  Expected: PASS with synthetic/tiny fixtures; no real GPU/artifact/result.

---

### Task 8: Full Assurance, Review, and Freeze Source Commit V_G

**Files:**
- Commit aggregate exactly:
  - `scripts/diagnose_pass205_rdgc_stage_b.py`
  - `tests/test_diagnose_pass205_rdgc_stage_b.py`
- Protect: all docs, manifests, artifacts, reports, results, root untracked files

**Interfaces:**
- Produces final reviewed `V_G` and source/test digests.

- [ ] **Step 1: Run complete local assurance once**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py
  .venv/bin/ruff check scripts/diagnose_pass205_rdgc_stage_b.py tests/test_diagnose_pass205_rdgc_stage_b.py
  .venv/bin/python -m py_compile scripts/diagnose_pass205_rdgc_stage_b.py tests/test_diagnose_pass205_rdgc_stage_b.py
  git diff --check
  ```

  Run no production CLI, checkpoint/model load, GPU command, or existing RSTA
  test suite unless a changed dependency requires a narrowly named test.

- [ ] **Step 2: Verify exact aggregate source scope and commit**

  ```bash
  plan_commit=$(git rev-parse HEAD)
  test -z "$(git diff --name-only -- docs reports/generated src)"
  git add -- scripts/diagnose_pass205_rdgc_stage_b.py tests/test_diagnose_pass205_rdgc_stage_b.py
  test "$(git diff --cached --name-only | sort)" = "$(printf '%s\n' scripts/diagnose_pass205_rdgc_stage_b.py tests/test_diagnose_pass205_rdgc_stage_b.py | sort)"
  git diff --cached --check
  git commit -m "implement RDGC no-training diagnostic"
  ```

  Expected: source/test-only commit; no protected path staged.

- [ ] **Step 3: Obtain fresh complete source/spec review**

  Review candidate, plan, aggregate diff from `P_G`, tests, final source, and
  ancestry. Require explicit adjudication of formula detach order, no angular
  path, preliminary gating, fresh selection, every control, update math,
  bootstrap/decisions, authenticated helper loading, all-four prefix, graph/CPU
  lifetime, atomic/attempt behavior, schema, and training unreachability.

  Repair every Critical/Important finding test-first within only the two paths.
  Repeat full assurance/review until clean.

- [ ] **Step 4: Freeze V_G and exact digests**

  ```bash
  verifier_source_commit=$(git rev-parse HEAD)
  plan_commit=$(git log -1 --format=%H "$verifier_source_commit" -- docs/superpowers/plans/2026-08-10-pass205-rdgc-stage-b.md)
  test -n "$plan_commit"
  git merge-base --is-ancestor "$plan_commit" "$verifier_source_commit"
  test "$(git diff --name-only "$plan_commit" "$verifier_source_commit" -- | sort)" = "$(printf '%s\n' scripts/diagnose_pass205_rdgc_stage_b.py tests/test_diagnose_pass205_rdgc_stage_b.py | sort)"
  sha256sum scripts/diagnose_pass205_rdgc_stage_b.py tests/test_diagnose_pass205_rdgc_stage_b.py
  printf 'V_G=%s\n' "$verifier_source_commit"
  ```

  Record exact `V_G` and both digests. The path-local `git log` result must equal
  the reviewed plan commit; if it does not, stop and bind the final reviewed one
  explicitly rather than choosing by convenience.

---

### Task 9: Create and Review the New Manifest-Only Handoff HV_G

**Files:**
- Create only: `docs/pass205_rdgc_stage_b_manifest.json`
- Protect: Pass 200 manifest, source, tests, results, artifacts, root files

**Interfaces:**
- Consumes final `V_G`, candidate/plan authorities, validation receipt, seed artifacts.
- Produces direct-child manifest-only `HV_G`.

- [ ] **Step 1: Build the exact new manifest**

  Exact top-level order is:

  ```text
  schema_version
  candidate
  implementation_plan
  upstream_rsta
  literature_audit
  validation_receipt
  historical
  current_scientific_source
  artifact_schema
  seeds
  ```

  `candidate` binds exact path/SHA/commit. `implementation_plan` binds final
  reviewed path/SHA/commit. `upstream_rsta` contains the literal chain in this
  plan. `literature_audit` binds the fresh review. `validation_receipt` binds
  exact path/SHA and validated relations. `historical` copies complete seed
  artifact bindings without candidate values. `current_scientific_source`
  binds `V_G` and exact `RDGC_SOURCE_ORDER` hashes. `artifact_schema` binds the
  candidate's full/reduced schemas and exact result path function. `seeds` is
  exactly `[0,1,2,3]`.

- [ ] **Step 2: Run already-GREEN manifest/provenance tests**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'authority or manifest or source or receipt or schema'
  git diff --check -- docs/pass205_rdgc_stage_b_manifest.json
  ```

  Expected: PASS without opening checkpoint or old artifact.

- [ ] **Step 3: Commit only the new manifest**

  ```bash
  verifier_source_commit=$(git rev-parse HEAD)
  test "$(git diff --name-only)" = docs/pass205_rdgc_stage_b_manifest.json
  git add -- docs/pass205_rdgc_stage_b_manifest.json
  test "$(git diff --cached --name-only)" = docs/pass205_rdgc_stage_b_manifest.json
  git diff --cached --check
  git commit -m "freeze RDGC diagnostic handoff"
  verifier_handoff_commit=$(git rev-parse HEAD)
  test "$(git rev-parse HEAD^)" = "$verifier_source_commit"
  test "$(git diff-tree --no-commit-id --name-only -r HEAD)" = docs/pass205_rdgc_stage_b_manifest.json
  ```

- [ ] **Step 4: Obtain independent manifest/provenance review**

  Require exact authority bytes/order, literature audit, receipt, all artifacts,
  `HV_G^==V_G`, source order/hashes, result path, unchanged RSTA manifest, no
  self-cycle, and no result input. If rejected, do not stack a repair on the
  rejected handoff: build a replacement manifest-only direct child of exact
  `V_G`, rerun tests, and review again.

- [ ] **Step 5: Freeze reviewed HV_G and result path**

  ```bash
  verifier_handoff_commit=$(git rev-parse HEAD)
  verifier_source_commit=$(git rev-parse HEAD^)
  manifest_sha256=$(sha256sum docs/pass205_rdgc_stage_b_manifest.json | cut -d ' ' -f 1)
  output="reports/generated/pass205_rdgc_stage_b/${verifier_handoff_commit}-rdgc-stage-b.json"
  test ! -e "$output"
  test ! -L "$output"
  printf 'V_G=%s\nHV_G=%s\nmanifest_sha256=%s\noutput=%s\n' "$verifier_source_commit" "$verifier_handoff_commit" "$manifest_sha256" "$output"
  ```

---

### Task 10: Run Candidate-Free DGX Preflight and the One Scientific Attempt

**Files:**
- Read: reviewed new manifest, provenance receipt, frozen seed artifacts/source
- Create at most once: exact result path
- Never read: old RSTA scientific artifact

**Interfaces:**
- Produces one atomic PASS/CLOSE/UNRESOLVED/INVALID result or structural stop.

- [ ] **Step 1: Create a fresh detached clean DGX checkout at HV_G**

  Authenticate detached `HEAD==HV_G`, `HV_G^==V_G`, manifest-only handoff,
  clean tracked status, exact source/manifest blobs, `.venv/bin/python` 3.12.3,
  registered NumPy/PyTorch/CUDA versions, `CUDA_VISIBLE_DEVICES=0`, output
  parent, absent output/temp, and validation receipt. Run no `nvidia-smi` or
  exploratory model command.

- [ ] **Step 2: Run the candidate-free CLI preflight tests**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'authority or manifest or receipt or cli or unreachable'
  ```

  Expected: PASS without checkpoint/model open or candidate state. This is test
  execution, not the scientific attempt.

- [ ] **Step 3: Execute the exact scientific command once**

  ```bash
  verifier_handoff_commit=$(git rev-parse HEAD)
  output="reports/generated/pass205_rdgc_stage_b/${verifier_handoff_commit}-rdgc-stage-b.json"
  set +e
  CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python -I -B scripts/diagnose_pass205_rdgc_stage_b.py \
      --manifest docs/pass205_rdgc_stage_b_manifest.json \
      --output "$output" \
      --scientific-once
  rdgc_exit=$?
  set -e
  printf 'exit=%s\n' "$rdgc_exit"
  ```

  Run this process exactly once. Do not pipe, tee, tail, inspect partial values,
  interrupt for a preliminary decision, or invoke a second command. Preliminary
  survival automatically continues inside this PID.

- [ ] **Step 4: Validate the one final result and stop**

  Run the source module's import-safe strict result validator in a separate
  CPU-only process. Record only result SHA-256, exact status, phase reached,
  `V_G`, and `HV_G` in operational logs. Do not summarize row values before
  independent review.

  - PASS: preserve, independently review, and authorize only a new training preregistration.
  - CLOSE: preserve and close RDGC.
  - UNRESOLVED: preserve and stop; no follow-up without new authority.
  - INVALID, structural exit, signal, timeout, missing/mismatched output, or publication failure: preserve evidence and stop.

  Under every branch: no retry, training, benchmark, optimizer integration,
  old-artifact read, or result rewrite.

---

## Final self-review checklist

- [ ] Candidate commit/SHA/path and current `V_R/HV_R` chain are exact.
- [ ] Old artifact is never opened; only the provenance receipt is validated.
- [ ] Old values are hypothesis generation only and never test expectations.
- [ ] RDGC uses only detached scalar self norm and no angular/vector self target.
- [ ] Preliminary uses fresh 8 identities, two contexts, and counts 1/8/32/180.
- [ ] Full panel uses fresh 32 identities and all six controls in exact order.
- [ ] All updates have exact matched PA norm; context B reuses context-A CPU bytes.
- [ ] PGN consumes 180 contributors serially with no live gradient list.
- [ ] All four seed integrity audits precede all candidate state/calls.
- [ ] Graph/CUDA/CPU action lifetime and fail-fast paths have weakref coverage.
- [ ] Paired PCG64(201) bootstrap and every decision boundary are exact.
- [ ] Full/reduced schemas, atomic no-clobber, and one-attempt semantics are exact.
- [ ] Aggregate source scope is exactly two new files; handoff is one new manifest.
- [ ] Current RSTA manifest, `src/`, artifacts, reports, results, and root untracked files remain untouched.
- [ ] PASS authorizes only a separate training preregistration; no training or benchmark occurs.
