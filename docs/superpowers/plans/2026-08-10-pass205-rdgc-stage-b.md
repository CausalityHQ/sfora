# Pass 205 RDGC Falsifier and No-Training Virtual-Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the prospective Receiver-Diagonal Gain Calibration (RDGC) one-attempt diagnostic that closes cheaply on a fresh scalar-gain falsifier or, only after exact survival, runs a fresh no-training norm-matched virtual-update panel.

**Architecture:** One import-safe diagnostic owns strict authority/provenance validation, pure selection/formula/decision functions, a candidate-free all-four-seed integrity prefix, the serial preliminary, the automatically gated virtual panel, and one atomic result. Source and tests are reviewed as final commit `V_G`; a new manifest is then created alone in direct-child `HV_G`; one clean DGX process executes both phases without training or opening the old RSTA scientific artifact.

**Tech Stack:** Python 3.12.3, PyTorch functional VJP/JVP, NumPy PCG64(201), strict insertion-ordered JSON, SHA-256/Git blob authentication, pytest, Ruff, `py_compile`, FP32 action arithmetic, named FP64 reductions, atomic hard-link publication.

## Global Constraints

- Implement `docs/pass205_rdgc_candidate_2026-08-10.md` literally at reviewed
  schema-fix commit `30d533e532d0f22c8b1e474987001685a4aa3488`, SHA-256
  `2a86f11f8d6a4563610b0585db74c372903bdbf7deabd580fa929114fda2af0f`.
- RDGC is independent. Never change, relabel, rerun, or reinterpret RSTA's authenticated `VALID` artifact and `UNRESOLVED/no_pass_or_fail_rule` decision.
- The old criterion-2 failure and disclosed aggregates are hypothesis generation only. Tests use unrelated synthetic values. Source may bind their documented provenance but may not encode them as expected RDGC results.
- Never open the old RSTA scientific artifact. Authenticate only its provenance-only roundtrip validation receipt and manifest chain.
- Candidate formula is exactly `e=log((||b||+1e-8)/(stopgrad(||s||)+1e-8))`, `R=0.5*e^2`. No `s_hat`, `cos(b,s)`, angular receiver-self target, or vector self target may be reachable.
- The preliminary must finish and satisfy every SURVIVES predicate before any full-panel correction, margin, control, or bootstrap object is constructed.
- All four seed bindings and integrity audits pass before any preliminary candidate state or call.
- Every example ID is the exact nonempty Pass 200 Bound-ID JSON string. Never
  coerce, trim, parse, Unicode-normalize, case-fold, or regenerate an ID; hashes
  consume its original UTF-8 bytes.
- A pre-Torch INVALID has `phase_reached="pre_import"`, an authenticated
  non-Torch `pre_import` environment and `torch_runtime=null`. Every post-import
  outcome has `environment.phase="post_import"` and the complete observed Torch
  runtime. Never import/probe Torch or fabricate runtime fields to build an
  early receipt.
- One B=180 CUDA graph at peak. No live CUDA tensor/action/closure from two graphs may coexist. Only detached JSON data and the exact current CPU parameter directions cross graphs.
- Exact operator order is `pa`, `rdgc`, `raw_cotangent`, `full_motion`,
  `batch_global_gain`, `scalar_diagonal_raw`,
  `per_example_gradient_normalized`, `layerwise_trust_ratio`.
- Action/JVP arithmetic is FP32. Norms, dot products, cosine denominators, aggregate reductions, bootstrap arrays, and hashes use cast-before-operation FP64 in exact named order.
- Run one fresh DGX command once. Preliminary survival continues automatically in-process. Every outcome or interruption consumes attempt 1.
- Training, optimizer state, parameter mutation, benchmark execution, query/gallery scoring, hyperparameter tuning, and `src/` changes are forbidden.
- Protect the untracked root files `HANDOFF_BRIEF.md`, `RSPG_SPECDEFECT.md`, and `RSPG_TASK.md`.
- Protect every existing source, test, manifest, artifact, report, and result. Implementation aggregate scope from this plan commit is exactly the new diagnostic and test; handoff scope is exactly the new manifest.

## Candidate and current-chain binding

```text
candidate path: docs/pass205_rdgc_candidate_2026-08-10.md
candidate SHA-256: 2a86f11f8d6a4563610b0585db74c372903bdbf7deabd580fa929114fda2af0f
candidate reviewed schema-fix commit C_G: 30d533e532d0f22c8b1e474987001685a4aa3488
candidate prior review-fix commit C_G1: b2a82f79836515714b4c8b57eb9596730fa3ed55
candidate original commit C_G0: 2599bb8c3f8238bb70f0f8935c0960cadda0dfb6
candidate original parent / current RSTA handoff HV_R: e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae
prior plan commits P_G0/P_G1: 5438e63dc925efd02857d52295b05a49304f71ed / 992f387bdc2d237cb51a2bb2a2e5c28329fa56e5
RSTA verifier source V_R: 3c368713e0890c0ffc63308f07d8d4ee5b19db1c
HV_R manifest SHA-256: fb089cf5905cea32a9d22563b50160af5fc8643efb657c49cb519d6d0c0da80b
RSTA producer H/S: c04574e2bb751c3229bce673408577cfedc00a88 / 15234a529a181c39c1c8b6477ad7eb7823fd0798
RSTA artifact SHA-256: e9bcd77c6e372e9c3bab4a420b97ff56f8ea164cbca56f53ec9c99a3b3c527ae
RSTA outcome: VALID scientific bytes; UNRESOLVED/no_pass_or_fail_rule decision
```

The plan commit `P_G` is the commit containing this repaired file after its independent
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
  test "$(git rev-parse 30d533e532d0f22c8b1e474987001685a4aa3488^{commit})" = 30d533e532d0f22c8b1e474987001685a4aa3488
  test "$(git rev-parse 30d533e532d0f22c8b1e474987001685a4aa3488^)" = 992f387bdc2d237cb51a2bb2a2e5c28329fa56e5
  test "$(git rev-parse 992f387bdc2d237cb51a2bb2a2e5c28329fa56e5^)" = b2a82f79836515714b4c8b57eb9596730fa3ed55
  test "$(git rev-parse b2a82f79836515714b4c8b57eb9596730fa3ed55^)" = 5438e63dc925efd02857d52295b05a49304f71ed
  test "$(git rev-parse 5438e63dc925efd02857d52295b05a49304f71ed^)" = 2599bb8c3f8238bb70f0f8935c0960cadda0dfb6
  test "$(git rev-parse 2599bb8c3f8238bb70f0f8935c0960cadda0dfb6^)" = e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae
  test "$(git show 30d533e532d0f22c8b1e474987001685a4aa3488:docs/pass205_rdgc_candidate_2026-08-10.md | sha256sum | cut -d ' ' -f 1)" = 2a86f11f8d6a4563610b0585db74c372903bdbf7deabd580fa929114fda2af0f
  test "$(git diff-tree --no-commit-id --name-only -r 2599bb8c3f8238bb70f0f8935c0960cadda0dfb6)" = docs/pass205_rdgc_candidate_2026-08-10.md
  test "$(git diff-tree --no-commit-id --name-only -r b2a82f79836515714b4c8b57eb9596730fa3ed55)" = docs/pass205_rdgc_candidate_2026-08-10.md
  test "$(git diff-tree --no-commit-id --name-only -r 30d533e532d0f22c8b1e474987001685a4aa3488)" = docs/pass205_rdgc_candidate_2026-08-10.md
  test "$(git rev-parse e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae^)" = 3c368713e0890c0ffc63308f07d8d4ee5b19db1c
  test "$(git diff-tree --no-commit-id --name-only -r e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae)" = docs/pass200_rsta_receipt_stage_a_manifest.json
  test "$(git show e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae:docs/pass200_rsta_receipt_stage_a_manifest.json | sha256sum | cut -d ' ' -f 1)" = fb089cf5905cea32a9d22563b50160af5fc8643efb657c49cb519d6d0c0da80b
  git diff --check e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae 30d533e532d0f22c8b1e474987001685a4aa3488
  ```

  Expected: all pass; no source/test/manifest/result/artifact changed.

- [ ] **Step 2: Obtain the mandatory primary-source/Gate-2 review**

  Review the exact thirteen-item literature/audit tuple (fourteen primary
  source identifiers plus the RSTA audit) in the candidate. Require
  explicit adjudication of receiver-specific full/diagonal scalar gain versus
  DoCL, MGS, NINT, Charpiat tangent shaping, GradNorm, K-FAC, NTKMTL, RelatIF,
  Automatic Clipping, Fishr, AdaFace, MagFace, AdaCos, LARS, generic functional
  damping, and the failed RSTA angular proposition. The reviewer must not read
  old row values or suggest repairing RSTA.

  Required verdict is `LIVE-NARROW` for the exact scalar formula. `DEAD` closes
  RDGC and authorizes no source. `UNRESOLVED` requires a prospective docs-only
  audit repair before source. Broad novelty language is Critical.

- [ ] **Step 3: Review the full candidate and plan contract**

  Require exact coverage of fresh disjoint selection, two contexts, nested
  contributor masks, global-scalar/heterogeneity gates, all seven corrections,
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
  control_penalties(torch_module, *, b, s, dbar, receiver_fields, pgn_motion)
  pgn_detached_coefficients(torch_module, first_order_gradient_norms)
  pgn_weighted_cotangent(torch_module, dbar, coefficients)
  layerwise_trust_ratio_direction(torch_module, named_parameters, p)
  fp64_named_norm(torch_module, values: tuple[Tensor, ...])
  fp64_named_dot(torch_module, left: tuple[Tensor, ...], right: tuple[Tensor, ...])
  normalize_virtual_updates(torch_module, p, corrections, *, alpha: float = 0.10)
  preliminary_metrics(rows: list[dict[str, object]]) -> dict[str, object]
  decide_preliminary(aggregates: dict[str, object]) -> dict[str, object]
  paired_bootstrap(panel_rows: list[dict[str, object]]) -> dict[str, object]
  decide_panel(aggregates: dict[str, object], bootstrap: dict[str, object]) -> dict[str, object]
  build_pre_import_environment(authenticated_non_torch_fields) -> dict[str, object]
  attach_observed_torch_runtime(pre_import_environment, torch_module) -> dict[str, object]
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
  test_pgn_coefficients_are_detached_before_one_weighted_global_vjp
  test_pgn_one_global_vjp_is_algebraically_equal_to_tiny_explicit_sum
  test_full_motion_control_is_generic_normalized_damping
  test_layerwise_trust_ratio_uses_exact_registered_groups_and_formula
  test_layerwise_trust_ratio_is_distinct_from_batch_global_gain
  ```

  Assert exact eight-operator order. For PGN, use unequal synthetic gradient
  norms, require FP64 geometric mean, detached coefficients, exact row order,
  a single weighted cotangent/global VJP, and a tiny real autograd equality
  oracle against the explicit mathematical sum. A recording sentinel proves no
  contributor is skipped, sorted, or reused and no collection of 180
  differentiable gradient trees exists. Reject replacing per-example
  normalization with clipping, mean gradients, scalar loss weights, or a
  preconditioner. For the trust-ratio control, group exact names by
  `name.rsplit(".",1)[0]`, require first-occurrence group order, FP64 named norms,
  detached `tau_l=||theta_l||/(||p_l||+1e-12)`, FP32 multiplication, and no
  optimizer/decay/moment state.

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
  test_panel_pass_requires_every_pa_six_control_context_and_alias_predicate
  test_context_b_each_control_requires_alignment_and_slope_pooled_lb_and_three_seeds
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

  Keep RDGC and the five penalty-control constructors separate, keep the
  PGN two-pass constructor separate, and keep the non-penalty trust-ratio
  direction separate. Assemble them once through the literal eight-operator
  dispatcher. Return fresh tuples; validate identity, dtype, device, shape,
  finite values, names, and lengths before reductions.

- [ ] **Step 4: Implement pure aggregate, bootstrap, and decision functions**

  Validate exact row order and schemas before any mean/median. Use NumPy
  float64 arrays and stable insertion order. Evaluate all CLOSE predicates in
  literal order, then all PASS predicates, then UNRESOLVED. Context-B
  superiority over each of six controls independently requires pooled
  `A_rdgc-A_control>0`, pooled `M_rdgc-M_control>0`, both paired-bootstrap lower
  bounds `>0`, and at least three positive seed means for each metric. Return
  exact Boolean ledgers and first decisive clause.

- [ ] **Step 5: Implement exact full and reduced output validators**

  Implement the candidate's literal nested schema registry, including the
  future-manifest schema, without a permissive fallback. Require exact concrete
  JSON types, finite floats including signed-zero-aware comparisons where
  equality matters, fixed scalars, status/phase/null relations, all row counts,
  all hashes, and forbidden scientific content in pre-science INVALID output.
  Recursively enumerate every path occurrence and prove remove/add/reorder/
  mistype/nonfinite/signed-zero/relational/hash mutations are rejected; ordinary
  key-set or `dict` equality is forbidden.

  Implement the exact result union before any Torch-dependent validator:
  pre-import INVALID requires `phase_reached="pre_import"`, complete ordered
  non-Torch environment, `torch_runtime=null`, and the five registered null
  fields; post-import INVALID and every scientific status require
  `environment.phase="post_import"`, the complete observed Torch runtime, and
  the registered four null fields where applicable. Cross-branch
  phase/status/null combinations fail. The pre-import builder accepts only
  already-authenticated Python/NumPy/environment/source/manifest values and is
  statically unreachable from Torch imports or CUDA probes.

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
  test_every_batch_receiver_support_distractor_and_contributor_id_is_exact_nonempty_bound_string
  test_bound_ids_reject_numeric_bool_empty_trimmed_and_unicode_normalized_mutations
  test_selection_uses_exact_domains_orders_and_two_172_distractor_sets
  test_selection_rejects_duplicate_missing_overlap_and_insufficient_rows
  ```

  Patch all old result/report candidate paths to raise. Require exact synthetic
  hash oracle, identical receivers across contexts/seeds, mutually disjoint
  distractors, supports outside graphs, and exact B=180 final order. The oracle
  uses original Bound-ID UTF-8 bytes; even after dependent hashes are
  recomputed, numeric-looking coercion, `str()` conversion, empty strings,
  whitespace trimming, and NFC/NFD normalization fail at their exact nested
  occurrence.

- [ ] **Step 3: Add nested contributor and preliminary-metric REDs**

  Add:

  ```text
  test_nested_masks_are_receiver_plus_prefix_counts_1_8_32_180
  test_preliminary_one_shared_graph_and_dbar_serves_all_receivers_and_counts
  test_preliminary_exact_one_forward_eight_diagonal_and_32_masked_action_calls
  test_preliminary_b1_equals_self_action_before_reduction
  test_preliminary_metrics_include_context_stability_count_and_global_scalar
  test_preliminary_close_or_unresolved_never_constructs_panel
  test_preliminary_survival_constructs_panel_once_without_operator_input
  ```

  Per seed/context require one object-identical live functional graph and
  `dbar` shared by all eight receivers/counts, exactly one forward/loss/dbar,
  eight independently constructed diagonal VJP/JVPs, and 32 masked VJP/JVPs in
  receiver then `1,8,32,180` order. Require direct live `torch.equal` on
  distinct `b_1` and `s` tensor identities and equal action hashes before
  detach/copy. Mutation tests reorder contributors, recompute forward/BN/dbar
  for a receiver or mask, reuse one tensor as both equality operands, use
  different receivers across contexts, pool before per-seed statistics, use
  Pearson instead of Spearman, or fit the global scalar on the other context;
  all fail. Weakrefs prove each mask/VJP/JVP dies before the next action and the
  one shared graph dies before the next seed/context.

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
  seed then context. Build one exact full B=180 graph and `dbar`; within it
  process receiver order and, for each receiver, an independent diagonal action
  followed by masked counts `1,8,32,180`. Directly compare the distinct live
  `b_1/s`, reduce/hash each action, and delete each action/mask immediately.
  Retain only the one shared graph until all 32 masks finish, then delete graph,
  model, functional state, inputs, BN/context state, and CUDA references before
  the next seed/context. Do not recompute forward/loss/dbar or accumulate
  full-panel state.

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

  Recording adapters require candidate first, then six controls in frozen order;
  context A then B for each detached direction; exact 256 context rows
  (`32*4*2`), eight operator records per row, 28 exact paired-bootstrap
  distributions; and no hidden alternate ordering or omitted negative control.

- [ ] **Step 2: Add control and alias-sentinel REDs**

  Add:

  ```text
  test_raw_control_is_dbar_angle_not_self_angle
  test_batch_global_and_scalar_raw_controls_cannot_reuse_receiver_target
  test_pgn_coefficient_pass_has_180_serial_first_order_norms_and_no_live_gradient_list
  test_pgn_correction_pass_has_exactly_one_weighted_global_vjp
  test_full_motion_control_has_no_diagonal_or_raw_target
  test_layerwise_trust_ratio_has_no_candidate_or_batch_global_gain_target
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
  test_pgn_coefficient_graph_dies_before_weighted_correction_graph
  test_pgn_only_detached_scalar_norms_cross_two_pass_boundary
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
  test_full_preliminary_and_reduced_invalid_schemas_reject_every_recursive_occurrence_mutation
  test_future_manifest_schema_rejects_every_recursive_occurrence_mutation
  test_pre_import_invalid_has_authenticated_non_torch_environment_and_null_torch_runtime
  test_pre_import_invalid_never_imports_torch_or_fabricates_runtime_fields
  test_post_import_invalid_and_scientific_receipts_require_complete_observed_torch_runtime
  test_environment_union_rejects_every_cross_phase_status_null_and_runtime_mutation
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

  Run the early-receipt tests in a fresh subprocess with an import sentinel that
  raises on `torch`, `torch.*`, CUDA libraries, and the authenticated helper.
  Require exact `source_files_sha256` recomputation and
  `environment=(phase,pre_import,torch_runtime)` insertion order. Separately
  exercise post-import INVALID and each PASS/CLOSE/UNRESOLVED fixture with a
  real observed runtime object. Mutate each union arm, null-field array, phase,
  status, source digest, and Torch field independently; no short-circuit or
  generic environment default may accept them.

- [ ] **Step 5: Add the mandatory unmocked real-CPU derivative/receipt RED**

  Add exactly
  `test_real_cpu_torch_func_end_to_end_authenticated_pass200_helper_no_adapters`.
  It source-loads the real Pass 200 helper only after authenticating its
  registered Git blob/worktree bytes and `__file__`, then uses real CPU
  `torch.func` on a small deterministic functional model and real tensors. In
  one end-to-end call it constructs the contextual field, the real higher-order
  RDGC correction, the two-pass/one-global-VJP PGN correction, the real
  layerwise trust-ratio direction, all normalized updates, distinct contexts A
  and B, JVP motions, margins, a complete scientific payload, strict recursive
  validation, and receipt serialization/reload. No fake tensor/module,
  recording adapter, monkeypatched VJP/JVP/grad, fake helper, or schema adapter
  is allowed in this test. It asserts numerical formulas and exact action/call
  order plus weakref death of both context graphs, PGN coefficient/correction
  graphs, functional states, actions, corrections, and output tensors after
  only JSON evidence remains. A test collection hook fails if this named node
  is skipped, xfailed, parametrically empty, or CUDA-gated.

- [ ] **Step 6: Run new nodes RED**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'panel or control or lifetime or weakref or atomic or cli or unreachable or real_cpu_torch_func'
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

  For RDGC and each penalty-based control, construct a fresh exact B=180 field
  and compute only that correction. The layerwise trust-ratio direction is
  constructed directly from the authenticated named parameters and ordinary PA
  direction, without a loss or candidate graph. Reduce each norm and CPU
  direction before graph deletion. Candidate direction may persist on CPU while
  controls are serially compared; no other control direction persists after its
  two context JVPs and cosine.

- [ ] **Step 2: Implement viable two-pass PGN with one weighted global VJP**

  The coefficient graph computes full contextual `dbar` once, then serial
  first-order `g_j^(0)=J_j^T stopgrad(dbar_j)` with no higher-order graph,
  detaches each FP64 norm, and deletes each gradient before the next. Compute
  detached `nu` and 180 detached coefficients, release the entire graph, and
  assert only scalar coefficient evidence survives. Rebuild one hash-identical
  higher-order context graph, form the weighted cotangent
  `dbar_PGN,j=a_j*dbar_j`, and call exactly one global VJP to obtain the
  differentiable `g_PGN=J_all^T dbar_PGN`. Build `b_PGN,r`, its penalty, and
  correction from that one tree. Validate exact call order/count, VJP-linearity
  algebra against the tiny test oracle, graph weakrefs, and no 180-tree path.

- [ ] **Step 3: Implement norm-matched directions and context JVPs**

  Normalize on CPU in named order, validate `5e-7`, move one direction to the
  registered GPU, compute context-A JVP/action/reductions, delete graph; then do
  context B from the same CPU bytes. Hash direction bytes and require the same
  hash in both context records.

- [ ] **Step 4: Implement complete panel aggregation/decision**

  Require all 256 rows and all eight operators. Aggregate/paired-bootstrap only
  after all rows exist. Materialize exactly 28 distributions in context,
  comparator, metric order. Require the context-B pooled/LB/three-seed
  alignment **and** slope predicates for every control. Apply CLOSE precedence
  then PASS then UNRESOLVED. Preserve all raw rows and distribution hashes; no
  selected aggregate may substitute.

- [ ] **Step 5: Implement atomic output and CLI**

  Parse exactly `--manifest`, `--output`, `--scientific-once`; reject aliases,
  extra flags, symlinks, alternate paths, existing output/temp, or wrong cwd.
  Authenticate the complete non-Torch publication authority before Torch
  import. A later pre-import failure publishes only the exact pre-import
  INVALID union with `torch_runtime=null`; a failure before publication
  authority is complete exits structurally with no receipt. Only after all
  pre-import work succeeds may Torch be imported and observed runtime fields be
  attached. Checkpoint open consumes attempt 1. Use
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

  Instantiate the candidate's **Literal future manifest schema** exactly; this
  is normative, not an example. The nested insertion orders are exactly:

  ```text
  candidate = path,sha256,commit
  implementation_plan = path,sha256,commit
  upstream_rsta = candidate,gate2_audit,producer_source_commit,
    producer_handoff_commit,producer_artifact,producer_pid,producer_exit_code,
    verifier_source_commit,verifier_handoff_commit,verifier_manifest,
    scientific_status,scientific_decision,first_decisive_clause
  literature_audit = path,sha256,commit,verdict,reviewed_candidate_sha256,
    primary_source_ids
  validation_receipt = path,sha256,status,verifier_source_commit,
    verifier_handoff_commit,artifact_path,artifact_sha256
  historical = manifest_path,manifest_sha256,seeds
  current_scientific_source = git_revision,files
  artifact_schema = result_path_template,schema_version,diagnostic,mode,
    statuses,phases,top_level_keys,pre_import_invalid_null_fields,
    post_import_invalid_null_fields,operator_order,contributor_counts
  ```

  Every `ref`, `artifact_ref`, `seed_artifacts`, and `source_file` has the exact
  key order/type/semantics in that registry. `primary_source_ids` has exact
  length 14 and reviewed order; source files have exact length 33; status,
  phase (including `pre_import`), top-key, both phase-specific null-field,
  eight-operator, contributor-count, and four-seed arrays are literal. Every
  example ID inside all historical/selection/result schema relations remains an
  exact nonempty Bound-ID string; the manifest author may not canonicalize it.
  `candidate` binds the reviewed schema-fix bytes above;
  `implementation_plan` binds this final reviewed commit. No ordinary dict
  equality, key sorting, inferred default, or descriptive extra field is
  accepted.

- [ ] **Step 2: Run already-GREEN manifest/provenance tests**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py -k 'authority or manifest or source or receipt or schema or recursive_occurrence'
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
- [ ] Each preliminary seed/context uses one shared graph/dbar, eight diagonal
      and 32 masked VJP/JVP actions, with no BN/context recomputation.
- [ ] Full panel uses fresh 32 identities and all six controls in exact order.
- [ ] All updates have exact matched PA norm; context B reuses context-A CPU bytes.
- [ ] PGN detaches 180 coefficient norms, releases that graph, then uses one
      weighted global higher-order VJP; no 180-tree path exists.
- [ ] The trust-ratio control uses exact named-layer groups/formula and is not
      aliased to batch-global gain.
- [ ] All four seed integrity audits precede all candidate state/calls.
- [ ] Graph/CUDA/CPU action lifetime and fail-fast paths have weakref coverage.
- [ ] Paired PCG64(201) bootstrap and every decision boundary are exact.
- [ ] Full/reduced schemas, atomic no-clobber, and one-attempt semantics are exact.
- [ ] Every nested result/manifest occurrence has recursive mutation coverage.
- [ ] The named unmocked real-CPU torch.func/helper end-to-end test passes
      without fakes, adapters, derivative monkeypatches, skips, or xfails.
- [ ] Aggregate source scope is exactly two new files; handoff is one new manifest.
- [ ] Current RSTA manifest, `src/`, artifacts, reports, results, and root untracked files remain untouched.
- [ ] PASS authorizes only a separate training preregistration; no training or benchmark occurs.
