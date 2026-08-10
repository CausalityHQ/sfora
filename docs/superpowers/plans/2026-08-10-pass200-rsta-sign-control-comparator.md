# Pass 200 RSTA Sign-Control Comparator Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the prospective production RSTA sign controls' raw-negated-hash comparator with same-graph direct `torch.equal` target/reference evidence while preserving every prior scientific domain and keeping all candidate work blocked through a fresh green candidate-free DGX audit.

**Architecture:** Keep the baseline, rebuild, and reversed-order trials unchanged. Add one strict reusable live-tensor sign comparator to the authenticated normwise helper, and make each production sign graph execute its target and ephemeral baseline-reference pairs through one VJP closure in a fixed order; persist target-only metrics plus target/reference hashes and exact booleans. Extend the production validators and future manifest projection without changing the published calibration result schema or the real manifest until the final manifest-only refreeze.

**Tech Stack:** Python 3.12.3, PyTorch 2.12.1 `torch.func.jvp`/`torch.func.vjp` and `torch.equal`, NumPy 2.5.0, SHA-256, strict insertion-ordered JSON, pytest, weak references/GC, Ruff, py_compile, Git, deterministic CUDA DGX audit.

## Global Constraints

- Implement `docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md` literally as committed at `a27dd7b3c8ff089c7cb80821c43658b975985a34`, SHA-256 `b87830197b162e6e3ce9ed20a3d631e138a1054d7af382054358c82867441259`.
- The amendment is prospective and non-scientific. H8 remains failed with no candidate; the synthetic calibration remains passed; no prior artifact is reinterpreted.
- Do not run a DGX command, real-data audit, candidate field, receiver row, score, decision, bootstrap, or scientific payload before Tasks 1–6 are complete and the manifest-only handoff is authenticated.
- Preserve the original baseline, rebuild, and reversed-action-order trials byte-semantically, including their graph construction, action order, metrics, hashes, schemas, predicates, and serialization order.
- For `parameter_sign`, use target `(-v,u)` and reference `(v,u)`. For `output_sign`, use target `(v,-u)` and reference `(v,u)`.
- Each sign graph constructs exactly one VJP closure and calls target JVP, target VJP, reference JVP, reference VJP in that literal order. It therefore has two JVP action calls and two calls to the same VJP closure.
- Compute the exact relation while target/reference tensors are live using direct `torch.equal`: parameter target JVP equals negated reference JVP and target VJP equals reference VJP; output target JVP equals reference JVP and target VJP equals negated reference VJP.
- Signed zero is equal under `torch.equal`. Do not derive `exact_relation` from raw hashes, `allclose`, a tolerance, scalar reductions, zero canonicalization, or a separately negated baseline hash.
- Persist only target JVP/VJP hashes, reference JVP/VJP hashes, target `beta_norm`, `reference_exact_action_hash_match`, `exact_relation`, and `passed` for each sign control. Compute and persist no reference metric.
- Require both reference hashes to equal the original baseline hashes. Target/reference sign consistency must not mask reference drift.
- Release all graph objects, live actions, target/reference tensors, detached CPU tensor trees, VJP closure, and temporary negations before the next graph. Only JSON scalars, exact Python booleans, and hashes survive. One derivative graph is the peak.
- Each production sign control has exact key order `jvp_sha256`, `vjp_sha256`, `reference_jvp_sha256`, `reference_vjp_sha256`, `beta_norm`, `reference_exact_action_hash_match`, `exact_relation`, `passed`.
- Sign `passed` is true exactly when the reference match is exact `True`, the direct relation is exact `True`, `type(beta_norm) is float`, and `beta_norm <= 0.0005`. `integrity_passed` uses the extended sign `passed` values plus the unchanged normwise/rebuild/reversed predicates.
- The published calibration artifact and its original five-key correct-fixture sign controls remain byte-semantically unchanged and continue to use the registered `6.25e-5` calibration ceiling.
- Add future manifest authority `normwise_adjoint_sign_control_amendment` with exact nested order `path`, `sha256`, `commit`, immediately after `normwise_adjoint_amendment` and before `binding_receipt`; insert it at the same position in the candidate-free manifest projection.
- Keep the exact existing 31 `current_scientific_source.files` paths and order. Only the reviewed source revision/hashes change at the later manifest refreeze.
- Source/test implementation is limited to the four files below. The real manifest, result files, GPU state, and unrelated paths are protected until Task 7.

## Amendment binding

```text
path: docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md
sha256: b87830197b162e6e3ce9ed20a3d631e138a1054d7af382054358c82867441259
commit: a27dd7b3c8ff089c7cb80821c43658b975985a34
```

## File structure

- `scripts/rsta_normwise_adjoint.py`: add the strict reusable live-tensor sign-relation comparator only; leave calibration payload construction and validation schemas unchanged.
- `tests/test_rsta_normwise_adjoint.py`: prove signed-zero semantics, direct `torch.equal` use, exact tree topology/type checks, and unchanged published calibration validation.
- `scripts/diagnose_pass200_rsta_stage_a.py`: run the two same-graph target/reference sign pairs, persist the extended production schema, validate its predicates, and authenticate the new future manifest authority/projection.
- `tests/test_diagnose_pass200_rsta_stage_a.py`: RED/GREEN coverage for dead ReLU signed zero, target-consistent reference drift, call order/count, target-only metrics, graph release, structural fail-fast, recursive schemas, provenance, candidate-free behavior, and prior-domain preservation.
- `docs/pass200_rsta_receipt_stage_a_manifest.json`: protected until Task 7; then the only modified path in the manifest refreeze commit.
- `reports/generated/pass200_rsta_receipt/${handoff_commit}-sign-control-comparator-integrity-all-seeds.json`: one absent candidate-free output created only on the authenticated DGX handoff.

---

### Task 1: Independently Review and Freeze the Amendment Before Code

**Files:**
- Review: `docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md`
- Review: `docs/pass200_rsta_normwise_adjoint_amendment_2026-08-09.md`
- Review: `docs/pass200_rsta_normwise_adjoint_calibration_protocol_2026-08-09.md`
- Modify only if review finds a defect: `docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md`
- Modify only if the amendment bytes change: `docs/superpowers/plans/2026-08-10-pass200-rsta-sign-control-comparator.md`

**Interfaces:**
- Consumes: the amendment binding above and the historical normwise/calibration authorities.
- Produces: an independently approved, internally consistent amendment and a plan whose literal path/SHA/commit binding matches it.

- [ ] **Step 1: Authenticate the authored amendment before review**

  Run:

  ```bash
  test "$(git rev-parse a27dd7b3c8ff089c7cb80821c43658b975985a34^{commit})" = a27dd7b3c8ff089c7cb80821c43658b975985a34
  test "$(git show a27dd7b3c8ff089c7cb80821c43658b975985a34:docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md | sha256sum | cut -d ' ' -f 1)" = b87830197b162e6e3ce9ed20a3d631e138a1054d7af382054358c82867441259
  test "$(sha256sum docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md | cut -d ' ' -f 1)" = b87830197b162e6e3ce9ed20a3d631e138a1054d7af382054358c82867441259
  git diff --check a27dd7b3c8ff089c7cb80821c43658b975985a34^ a27dd7b3c8ff089c7cb80821c43658b975985a34
  ```

  Expected: all four commands exit `0`; the amendment worktree bytes equal its committed Git blob.

- [ ] **Step 2: Obtain an independent adversarial amendment review**

  Run this read-only consultation from repository root:

  ```bash
  devbox-ask claude --model opus --effort max "Read docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md, docs/pass200_rsta_normwise_adjoint_amendment_2026-08-09.md, docs/pass200_rsta_normwise_adjoint_calibration_protocol_2026-08-09.md, scripts/rsta_normwise_adjoint.py, and scripts/diagnose_pass200_rsta_stage_a.py at repository HEAD. Do not edit. Adversarially review whether the prospective sign-control amendment exactly fixes signed-zero/dead-ReLU raw-negated-hash failure with same-graph target/reference direct torch.equal; preserves baseline/rebuild/reversed trials and the published calibration schema; freezes exact call order/count, target-only metrics, reference-baseline hash binding, graph lifetime, nested field order/predicates, manifest insertion/order, unchanged 31-path source membership, chronology, and no-candidate boundary. Report only concrete Critical, Important, or Minor findings with exact section evidence; say CLEAN if none."
  ```

  Expected: a durable read-only report that explicitly covers every listed authority boundary.

- [ ] **Step 3: Apply the amendment stop rule**

  If the review is `CLEAN`, proceed to Task 2 without changing the amendment. If it reports any concrete finding, stop all source/test work. Repair only the amendment, then run:

  ```bash
  git diff --check -- docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md
  git add -- docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md
  test "$(git diff --cached --name-only)" = docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md
  git commit -m "fix RSTA sign-control comparator amendment review"
  sha256sum docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md
  git rev-parse HEAD
  ```

  Replace every occurrence in this plan of the superseded amendment path,
  SHA-256, and commit, including the global constraint, binding block, review
  commands, authority constants/object, ancestry edge, manifest-refreeze
  object, and DGX authentication commands. Commit only this plan with subject
  `rebind RSTA sign-control comparator plan`, rerun Steps 1–2 against the new
  binding, and repeat until the independent report is `CLEAN`. A stale plan
  binding is a structural stop.

---

### Task 2: Write the Complete Comparator and Production RED Suite

**Files:**
- Modify: `tests/test_rsta_normwise_adjoint.py`
- Modify: `tests/test_diagnose_pass200_rsta_stage_a.py`
- Do not modify yet: `scripts/rsta_normwise_adjoint.py`
- Do not modify yet: `scripts/diagnose_pass200_rsta_stage_a.py`

**Interfaces:**
- Requires future helper: `exact_sign_control_relation(control_name: str, target_jvp: torch.Tensor, target_vjp: Mapping[str, torch.Tensor], reference_jvp: torch.Tensor, reference_vjp: Mapping[str, torch.Tensor], parameter_names: Sequence[str]) -> bool`.
- Requires future production evidence: each sign control's exact eight-key ordered mapping from the amendment.
- Produces: focused failing tests for every comparator, schedule, lifetime, schema, and failure requirement before implementation changes.

- [ ] **Step 1: Write signed-zero and direct-comparator helper REDs**

  Add these exact tests to `tests/test_rsta_normwise_adjoint.py`:

  ```python
  def test_exact_sign_control_relation_accepts_signed_zero_only_through_torch_equal(
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      reference_jvp = torch.tensor([0.0, -0.0], dtype=torch.float32)
      target_jvp = torch.tensor([0.0, 0.0], dtype=torch.float32)
      reference_vjp = {"weight": torch.tensor([0.0, -0.0], dtype=torch.float32)}
      target_vjp = {"weight": reference_vjp["weight"].clone()}
      assert normwise.tensor_sha256(target_jvp) != normwise.tensor_sha256(-reference_jvp)
      real_equal = torch.equal
      calls: list[tuple[torch.Tensor, torch.Tensor]] = []

      def sentinel(left: torch.Tensor, right: torch.Tensor) -> bool:
          calls.append((left, right))
          return real_equal(left, right)

      monkeypatch.setattr(torch, "equal", sentinel)
      assert normwise.exact_sign_control_relation(
          "parameter_sign",
          target_jvp,
          target_vjp,
          reference_jvp,
          reference_vjp,
          ("weight",),
      ) is True
      assert len(calls) == 2


  @pytest.mark.parametrize("control_name", ["parameter_sign", "output_sign"])
  def test_exact_sign_control_relation_rejects_tree_shape_dtype_and_order_drift(
      control_name: str,
  ) -> None:
      reference_jvp = torch.tensor([1.0, -2.0], dtype=torch.float32)
      target_jvp = -reference_jvp if control_name == "parameter_sign" else reference_jvp.clone()
      reference_vjp = {
          "weight": torch.tensor([3.0, -4.0], dtype=torch.float32),
          "bias": torch.tensor([5.0], dtype=torch.float32),
      }
      target_vjp = {
          name: (value.clone() if control_name == "parameter_sign" else -value)
          for name, value in reference_vjp.items()
      }
      invalid_calls = (
          (
              target_jvp,
              {"bias": target_vjp["bias"], "weight": target_vjp["weight"]},
              reference_jvp,
              reference_vjp,
          ),
          (target_jvp[:1], target_vjp, reference_jvp, reference_vjp),
          (target_jvp.double(), target_vjp, reference_jvp, reference_vjp),
          (
              target_jvp,
              target_vjp,
              reference_jvp,
              {"weight": reference_vjp["weight"].double(), "bias": reference_vjp["bias"]},
          ),
          (
              target_jvp,
              target_vjp,
              torch.empty(reference_jvp.shape, dtype=torch.float32, device="meta"),
              reference_vjp,
          ),
      )
      for bad_target_jvp, bad_target_vjp, bad_reference_jvp, bad_reference_vjp in invalid_calls:
          with pytest.raises(ValueError):
              normwise.exact_sign_control_relation(
                  control_name,
                  bad_target_jvp,
                  bad_target_vjp,
                  bad_reference_jvp,
                  bad_reference_vjp,
                  ("weight", "bias"),
              )
  ```

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py::test_exact_sign_control_relation_accepts_signed_zero_only_through_torch_equal tests/test_rsta_normwise_adjoint.py::test_exact_sign_control_relation_rejects_tree_shape_dtype_and_order_drift
  ```

  Expected: FAIL because `exact_sign_control_relation` is absent.

- [ ] **Step 2: Write the dead-ReLU, exact action-order, and target-only-metric REDs**

  Add tests named
  `test_normwise_sign_controls_accept_dead_relu_signed_zero_by_direct_equal`,
  `test_normwise_sign_controls_use_exact_target_reference_call_count_and_order`,
  and `test_normwise_sign_controls_compute_metrics_for_targets_only` to
  `tests/test_diagnose_pass200_rsta_stage_a.py`.

  The first test must use an actually dead FP32 ReLU functional graph whose
  action contains zero, prove the target hash differs from the raw hash of an
  explicitly negated reference tensor, and require both sign
  `exact_relation=True`. Monkeypatch `torch.equal` with a recording sentinel
  and require the helper to call it for the JVP plus every named VJP tensor.
  The core signed-zero assertions are exactly:

  ```python
  assert torch.equal(target_jvp, -reference_jvp)
  assert _sha256_tensor(target_jvp) != _sha256_tensor(-reference_jvp)
  assert audit["controls"]["parameter_sign"]["exact_relation"] is True
  assert audit["controls"]["output_sign"]["exact_relation"] is True
  ```

  The action-order test must wrap `torch.func.vjp`, the returned closure, and
  `torch.func.jvp`. Require this full per-seed event sequence:

  ```python
  [
      "baseline:vjp_construct", "baseline:jvp", "baseline:vjp",
      "rebuild:vjp_construct", "rebuild:jvp", "rebuild:vjp",
      "reversed:vjp_construct", "reversed:vjp", "reversed:jvp",
      "parameter_sign:vjp_construct", "parameter_sign:target_jvp",
      "parameter_sign:target_vjp", "parameter_sign:reference_jvp",
      "parameter_sign:reference_vjp",
      "output_sign:vjp_construct", "output_sign:target_jvp",
      "output_sign:target_vjp", "output_sign:reference_jvp",
      "output_sign:reference_vjp",
  ]
  ```

  Also require exactly five VJP closure constructions, seven JVP calls, and
  seven VJP-closure calls. The target-only-metric test must wrap
  `normwise_adjoint_metrics`, require exactly five calls total, and prove the
  two sign calls receive target `(-v,u)` and `(v,-u)` factors rather than either
  reference pair.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py::test_normwise_sign_controls_accept_dead_relu_signed_zero_by_direct_equal tests/test_diagnose_pass200_rsta_stage_a.py::test_normwise_sign_controls_use_exact_target_reference_call_count_and_order tests/test_diagnose_pass200_rsta_stage_a.py::test_normwise_sign_controls_compute_metrics_for_targets_only
  ```

  Expected: FAIL because production still executes one action pair per sign
  graph and derives the sign relation from raw-negated hashes.

- [ ] **Step 3: Write reference-drift and exact nested-schema REDs**

  Add tests named
  `test_normwise_sign_control_reference_drift_fails_despite_target_consistency`
  and `test_normwise_sign_control_schema_rejects_every_nested_mutation`.

  In the drift test, wrap the parameter-sign graph so its target and reference
  JVP/VJP pairs drift together and still satisfy the registered direct sign
  relation. Require `exact_relation is True`, both reference hashes to differ
  from baseline, `reference_exact_action_hash_match is False`, sign
  `passed is False`, and top-level `integrity_passed is False`.
  Use these literal outcome assertions:

  ```python
  sign = audit["controls"]["parameter_sign"]
  assert sign["exact_relation"] is True
  assert sign["reference_jvp_sha256"] != audit["jvp_sha256"]
  assert sign["reference_vjp_sha256"] != audit["vjp_sha256"]
  assert sign["reference_exact_action_hash_match"] is False
  assert sign["passed"] is False
  assert audit["integrity_passed"] is False
  ```

  In the schema test, start from one valid audit and recursively exercise every
  sign-control mutation below for both `parameter_sign` and `output_sign`:

  ```python
  SIGN_CONTROL_KEYS = (
      "jvp_sha256",
      "vjp_sha256",
      "reference_jvp_sha256",
      "reference_vjp_sha256",
      "beta_norm",
      "reference_exact_action_hash_match",
      "exact_relation",
      "passed",
  )
  ```

  Remove every key, append an extra key, move every key out of order, mutate
  each hash to valid wrong lowercase hex, replace each boolean with `0`, `1`,
  or `np.bool_`, use `beta_norm` at `0.0005`, immediately above `0.0005`, and
  string `"infinity"`, and forge each derived boolean independently. Require
  rejection for every inconsistent object and acceptance at the exact float
  boundary only when both booleans are exact `True`.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py::test_normwise_sign_control_reference_drift_fails_despite_target_consistency tests/test_diagnose_pass200_rsta_stage_a.py::test_normwise_sign_control_schema_rejects_every_nested_mutation
  ```

  Expected: FAIL because reference evidence and predicates are absent.

- [ ] **Step 4: Write graph-release, one-graph-peak, and structural fail-fast REDs**

  Extend the existing weak-reference coverage with tests named
  `test_normwise_sign_controls_release_target_reference_actions_before_next_graph`
  and `test_normwise_sign_control_reference_structure_fails_before_next_graph`.

  Track weak references to each functional encoder, parameter tree, closure,
  target JVP/VJP, reference JVP/VJP, detached CPU action tree, and temporary
  negation. At every next `_functional_encoder` call, run `gc.collect()` and
  require all previous weak references dead. Require `graph_count == 5`,
  `peak_live_graphs == 1`, and `metric_count == 5` after completion.
  The terminal assertions are exactly:

  ```python
  gc.collect()
  assert graph_count == 5
  assert metric_count == 5
  assert peak_live_graphs == 1
  assert all(reference() is None for reference in graph_refs)
  assert all(reference() is None for reference in target_reference_action_refs)
  ```

  For fail-fast, make the parameter-sign reference VJP tree omit the last
  parameter after the first three trials have completed. Require the audit to
  raise `ValueError` before metric/hash persistence for that malformed sign
  graph and before construction of the output-sign graph. Require exactly four
  graph constructions and zero candidate/scoring calls.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py::test_normwise_sign_controls_release_target_reference_actions_before_next_graph tests/test_diagnose_pass200_rsta_stage_a.py::test_normwise_sign_control_reference_structure_fails_before_next_graph
  ```

  Expected: FAIL because each current sign graph has no live reference pair to
  validate or release.

- [ ] **Step 5: Prove the source files are still untouched at the RED checkpoint**

  Run:

  ```bash
  test -z "$(git diff --name-only -- scripts/rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py)"
  git diff --check -- tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  ```

  Expected: both commands exit `0`; only the two test files contain the REDs.

---

### Task 3: Implement the Minimal Same-Graph Comparator GREEN

**Files:**
- Modify: `scripts/rsta_normwise_adjoint.py`
- Modify: `scripts/diagnose_pass200_rsta_stage_a.py`
- Test: `tests/test_rsta_normwise_adjoint.py`
- Test: `tests/test_diagnose_pass200_rsta_stage_a.py`

**Interfaces:**
- Produces: `exact_sign_control_relation(control_name: str, target_jvp: torch.Tensor, target_vjp: Mapping[str, torch.Tensor], reference_jvp: torch.Tensor, reference_vjp: Mapping[str, torch.Tensor], parameter_names: Sequence[str]) -> bool`.
- Extends: authenticated helper interface required by `_load_authenticated_normwise_adjoint_helper`.
- Extends: production `parameter_sign` and `output_sign` evidence to the amendment's exact eight-key ordered mappings.
- Preserves: `run_fixture_controls`, `validate_calibration_result`, the published calibration artifact schema, and rebuild/reversed production behavior.

- [ ] **Step 1: Add only the strict live-tensor helper**

  In `scripts/rsta_normwise_adjoint.py`, add a function with this exact shape:

  ```python
  def exact_sign_control_relation(
      control_name: str,
      target_jvp: torch.Tensor,
      target_vjp: Mapping[str, torch.Tensor],
      reference_jvp: torch.Tensor,
      reference_vjp: Mapping[str, torch.Tensor],
      parameter_names: Sequence[str],
  ) -> bool:
      names = tuple(parameter_names)
      if control_name not in ("parameter_sign", "output_sign"):
          raise ValueError("adjoint sign control name differs")
      # Require nonempty exact ordered topology and aligned finite FP32 tensors
      # before comparison. Use _tensor for tensor type/dtype/device/finite checks,
      # plus explicit shape/device equality across every target/reference pair.
      if control_name == "parameter_sign":
          return bool(
              torch.equal(target_jvp, -reference_jvp)
              and all(torch.equal(target_vjp[name], reference_vjp[name]) for name in names)
          )
      return bool(
          torch.equal(target_jvp, reference_jvp)
          and all(torch.equal(target_vjp[name], -reference_vjp[name]) for name in names)
      )
  ```

  Implement the stated checks directly; do not alter `_validate_control`,
  `_validate_entry`, `run_fixture_controls`, or any calibration result key.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py::test_exact_sign_control_relation_accepts_signed_zero_only_through_torch_equal tests/test_rsta_normwise_adjoint.py::test_exact_sign_control_relation_rejects_tree_shape_dtype_and_order_drift
  ```

  Expected: PASS.

- [ ] **Step 2: Preserve the three original trials and add one sign-trial executor**

  In `adjoint_integrity_audit`, leave the existing ordinary `run_trial` path for
  baseline, rebuild, and reversed order. Add a dedicated
  `run_sign_trial(control_name: str)` that:

  1. calls `_functional_encoder` once;
  2. constructs the target/reference directions exactly from `u` and `v`;
  3. constructs one VJP closure;
  4. calls target JVP, target VJP, reference JVP, reference VJP in order;
  5. calls `exact_sign_control_relation` before detaching or deleting an action;
  6. computes legacy and normwise metrics for the target only;
  7. hashes detached target and reference actions in exact named order;
  8. computes the reference-baseline predicate only after the unchanged
     baseline hashes exist; and
  9. deletes all graph/live/CPU tensor state before returning JSON evidence.

  The produced mapping must be constructed literally in this order:

  ```python
  {
      "jvp_sha256": target_jvp_sha256,
      "vjp_sha256": target_vjp_sha256,
      "reference_jvp_sha256": reference_jvp_sha256,
      "reference_vjp_sha256": reference_vjp_sha256,
      "beta_norm": target_beta_norm,
      "reference_exact_action_hash_match": reference_exact_action_hash_match,
      "exact_relation": exact_relation,
      "passed": (
          type(reference_exact_action_hash_match) is bool
          and reference_exact_action_hash_match is True
          and type(exact_relation) is bool
          and exact_relation is True
          and type(target_beta_norm) is float
          and target_beta_norm <= 5.0e-4
      ),
  }
  ```

  Delete the production-only baseline `negative_jvp_sha256` and
  `negative_vjp_sha256` evidence because no later comparator may use it. Keep
  baseline persisted metrics/hashes unchanged.

- [ ] **Step 3: Bind the new helper bytes and run every behavioral GREEN**

  Recompute the helper SHA-256 and set `_NORMWISE_ADJOINT_HELPER_SHA256` to that
  exact lowercase digest. Add `exact_sign_control_relation` to the authenticated
  helper's required callable tuple. Run:

  ```bash
  sha256sum scripts/rsta_normwise_adjoint.py
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py -k 'exact_sign_control_relation or calibration_schema'
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'normwise_sign_control or normwise_sign_controls'
  ```

  Expected: PASS, including the signed-zero, direct-comparator, drift, call
  schedule, target-only metrics, graph release, and structural fail-fast tests.

- [ ] **Step 4: Reauthenticate the immutable calibration result under the changed helper**

  Run:

  ```bash
  .venv/bin/python - <<'PY'
  import json
  from pathlib import Path
  import sys

  sys.path.insert(0, "scripts")
  import rsta_normwise_adjoint

  artifact = Path("reports/generated/pass200_rsta_receipt/0f5d1e2f626524f02c565a04f6fa0ae7127cd7e2-normwise-adjoint-calibration.json")
  payload = json.loads(artifact.read_text(encoding="utf-8"))
  rsta_normwise_adjoint.validate_calibration_result(payload)
  assert payload["all_passed"] is True
  assert list(payload["correct_fixtures"]["zero_corner"]["controls"]["parameter_sign"]) == [
      "jvp_sha256", "vjp_sha256", "beta_norm", "exact_relation", "passed"
  ]
  PY
  test "$(sha256sum reports/generated/pass200_rsta_receipt/0f5d1e2f626524f02c565a04f6fa0ae7127cd7e2-normwise-adjoint-calibration.json | cut -d ' ' -f 1)" = 5fcb09a1e3a6eedddd05ef49bd22bc9920656089aa401a5aae2c5704a9d9dc50
  ```

  Expected: validation succeeds and artifact bytes remain exact.

---

### Task 4: RED-to-GREEN the Production Schema, Manifest Authority, and Source Provenance

**Files:**
- Modify: `tests/test_diagnose_pass200_rsta_stage_a.py`
- Modify: `scripts/diagnose_pass200_rsta_stage_a.py`
- Verify unchanged semantics: `tests/test_rsta_normwise_adjoint.py`
- Verify authenticated helper: `scripts/rsta_normwise_adjoint.py`

**Interfaces:**
- Extends: `_validate_adjoint_integrity_audit` for exact sign reference hashes and predicates.
- Adds constants: `_NORMWISE_ADJOINT_SIGN_CONTROL_AMENDMENT_PATH`, `_NORMWISE_ADJOINT_SIGN_CONTROL_AMENDMENT_SHA256`, `_NORMWISE_ADJOINT_SIGN_CONTROL_AMENDMENT_COMMIT`.
- Extends: `_validate_amended_manifest_schema`, `validate_scientific_execution_source`, candidate-free manifest projection construction, and `validate_all_seed_adjoint_integrity_payload`.
- Preserves: `_CURRENT_SCIENTIFIC_SOURCE_FILES` exact 31-path membership/order and all prior manifest domains.

- [ ] **Step 1: Write and run the exact manifest/source-validator RED**

  Extend `_future_normwise_manifest()` in the test file with this literal object
  immediately after `normwise_adjoint_amendment`:

  ```python
  "normwise_adjoint_sign_control_amendment": {
      "path": "docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md",
      "sha256": "b87830197b162e6e3ce9ed20a3d631e138a1054d7af382054358c82867441259",
      "commit": "a27dd7b3c8ff089c7cb80821c43658b975985a34",
  },
  ```

  Add `test_sign_control_manifest_authority_order_provenance_and_prior_domains`.
  It must require the amendment's worktree bytes, Git blob, SHA, commit, and
  ancestry edge
  `a27dd7b3c8ff089c7cb80821c43658b975985a34 -> current_scientific_source.git_revision`;
  require exact future top-level and candidate-free projection order; mutate
  every nested authority leaf and order; and recursively compare every prior
  manifest domain before/after insertion. The only permitted differences are
  the new authority object and future source revision/hashes. Assert the exact
  source tuple still has length `31` and equals `_NORMWISE_SOURCE_ORDER`.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py::test_sign_control_manifest_authority_order_provenance_and_prior_domains
  ```

  Expected: FAIL because the new authority is not recognized or authenticated.

- [ ] **Step 2: Write and run candidate-free projection/schema REDs**

  Update the candidate-free exact-schema oracle so the projected manifest
  order is exactly:

  ```python
  [
      "path", "sha256", "base_preregistration", "amendment",
      "deterministic_pool_amendment", "zero_jacobian_classifier_amendment",
      "adjoint_integrity_amendment", "normwise_adjoint_calibration_protocol",
      "normwise_adjoint_calibration_result", "normwise_adjoint_amendment",
      "normwise_adjoint_sign_control_amendment", "binding_receipt",
      "historical", "artifact_schema", "source",
  ]
  ```

  Recursively remove, reorder, add, and mutate the projected authority, then
  require rejection before artifact/model/candidate access. Also require each
  seed's adjoint sign controls to use the exact eight-key schema and preserve
  the candidate-forbidden call set.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'sign_control and (manifest or source or projection or schema or candidate_free)'
  ```

  Expected: FAIL because the source and candidate-free validators still use the
  prior manifest authority set/order.

- [ ] **Step 3: Implement only exact schema and predicate validation**

  For `parameter_sign` and `output_sign`, require the exact eight keys/order.
  Recompute:

  ```python
  expected_reference_match = (
      record["reference_jvp_sha256"] == value["jvp_sha256"]
      and record["reference_vjp_sha256"] == value["vjp_sha256"]
  )
  expected_passed = (
      type(record["reference_exact_action_hash_match"]) is bool
      and record["reference_exact_action_hash_match"] is True
      and type(record["exact_relation"]) is bool
      and record["exact_relation"] is True
      and type(record["beta_norm"]) is float
      and record["beta_norm"] <= 5.0e-4
  )
  ```

  Require exact Python booleans and equality with both derived predicates. Do
  not compare target hashes to hashes of negated baseline/reference tensors.
  Recompute top-level `integrity_passed` from the complete control predicates.

- [ ] **Step 4: Implement only the new authority/order/provenance transition**

  Add the three exact amendment constants from Step 1. Insert the authority
  after `normwise_adjoint_amendment` and before `binding_receipt` in:

  - future manifest schema/order and exact-reference validation;
  - worktree/Git-blob authority authentication;
  - ancestry edges, after the normwise amendment and before reviewed source;
  - candidate-free projected manifest schema/order;
  - candidate-free projection production; and
  - recursive candidate-free payload validation.

  Leave `_CURRENT_SCIENTIFIC_SOURCE_FILES` byte-for-byte unchanged. Do not edit
  `docs/pass200_rsta_receipt_stage_a_manifest.json`.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py::test_sign_control_manifest_authority_order_provenance_and_prior_domains
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'sign_control and (manifest or source or projection or schema or candidate_free)'
  ```

  Expected: PASS.

- [ ] **Step 5: Confirm the protected manifest and result domains remain untouched**

  Run:

  ```bash
  test -z "$(git diff --name-only -- docs/pass200_rsta_receipt_stage_a_manifest.json reports/generated)"
  test "$(sha256sum reports/generated/pass200_rsta_receipt/0f5d1e2f626524f02c565a04f6fa0ae7127cd7e2-normwise-adjoint-calibration.json | cut -d ' ' -f 1)" = 5fcb09a1e3a6eedddd05ef49bd22bc9920656089aa401a5aae2c5704a9d9dc50
  ```

  Expected: both commands exit `0`.

---

### Task 5: Full Local Assurance and the Required Source/Test Commit

**Files:**
- Commit: `scripts/rsta_normwise_adjoint.py`
- Commit: `scripts/diagnose_pass200_rsta_stage_a.py`
- Commit: `tests/test_rsta_normwise_adjoint.py`
- Commit: `tests/test_diagnose_pass200_rsta_stage_a.py`
- Protect: every other tracked path

**Interfaces:**
- Consumes: all GREEN Tasks 2–4 behavior and validator gates.
- Produces: the source/test commit with exact subject `implement RSTA sign-control comparators`.

- [ ] **Step 1: Run the complete affected assurance gate once**

  Run serially:

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/ruff check scripts/rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/python -m py_compile scripts/rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  git diff --check
  ```

  Expected: every command exits `0`. Do not start overlapping copies of the
  full pytest gate.

- [ ] **Step 2: Verify exact scope and commit**

  Run:

  ```bash
  git diff --name-only | sort
  git add -- scripts/rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  test "$(git diff --cached --name-only | sort)" = "$(printf '%s\n' scripts/diagnose_pass200_rsta_stage_a.py scripts/rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py | sort)"
  git diff --cached --check
  git commit -m "implement RSTA sign-control comparators"
  git show --name-only --format= HEAD
  ```

  Expected: the commit has exactly the four paths above and the exact subject.

- [ ] **Step 3: Record the committed source identity**

  Run:

  ```bash
  git rev-parse HEAD
  sha256sum scripts/rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  ```

  Record the full commit and four digests in the review handoff. Do not update
  the real manifest yet.

---

### Task 6: Fresh Independent Full-Source Review and Repair Loop

**Files:**
- Review/fix if necessary: `scripts/rsta_normwise_adjoint.py`
- Review/fix if necessary: `scripts/diagnose_pass200_rsta_stage_a.py`
- Review/fix if necessary: `tests/test_rsta_normwise_adjoint.py`
- Review/fix if necessary: `tests/test_diagnose_pass200_rsta_stage_a.py`
- Protect: `docs/pass200_rsta_receipt_stage_a_manifest.json`

**Interfaces:**
- Consumes: the Task 5 source/test commit, amendment, and complete tests.
- Produces: final independently reviewed source revision and hashes for the manifest-only handoff.

- [ ] **Step 1: Obtain a fresh adversarial full-source review**

  Run:

  ```bash
  devbox-ask claude --model opus --effort max "Read docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md and the complete HEAD versions and history diff of scripts/rsta_normwise_adjoint.py, scripts/diagnose_pass200_rsta_stage_a.py, tests/test_rsta_normwise_adjoint.py, and tests/test_diagnose_pass200_rsta_stage_a.py. Do not edit. Review the implementation against every amendment requirement. Explicitly audit signed-zero/dead-ReLU behavior, direct live torch.equal, same-graph immutable target/reference pairs, literal closure/JVP/VJP call order and counts, target-only metrics, reference-baseline raw hashes, target-consistent reference drift, exact schemas/types/predicates/output order, calibration backward compatibility, structural fail-fast, weakref release/one-graph peak, candidate-free/scientific prefixes, authority Git bytes/ancestry/projection order, unchanged 31-path source membership, and protected manifest/results. Report concrete Critical, Important, or Minor findings with file/line evidence; say CLEAN if none."
  ```

  Expected: a fresh review of the full committed source, not only the latest
  patch fragment.

- [ ] **Step 2: Repair every Critical or Important finding test-first**

  For each finding, add one behavior-specific failing test, run its exact node
  to prove RED, implement the smallest source change, rerun the node to prove
  GREEN, then rerun:

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/ruff check scripts/rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/python -m py_compile scripts/rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  git diff --check
  git add -- scripts/rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  git commit -m "fix RSTA sign-control comparator review findings"
  ```

  Re-run Step 1 against the new HEAD. Repeat until no Critical or Important
  finding remains. Resolve any Minor finding that affects an amendment
  predicate, evidence authenticity, or protected scope before proceeding.

- [ ] **Step 3: Freeze final reviewed source only after the clean loop**

  Run:

  ```bash
  reviewed_source_commit=$(git rev-parse HEAD)
  test "$(git show -s --format=%s a27dd7b3c8ff089c7cb80821c43658b975985a34)" = "amend RSTA sign-control comparators"
  git diff --quiet -- docs/pass200_rsta_receipt_stage_a_manifest.json
  sha256sum scripts/rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  printf '%s\n' "$reviewed_source_commit"
  ```

  Expected: a full lowercase 40-hex reviewed commit, protected manifest still
  unchanged, and fresh hashes recorded.

---

### Task 7: Manifest-Only Refreeze and Candidate-Free DGX Audit

**Files:**
- Modify only: `docs/pass200_rsta_receipt_stage_a_manifest.json`
- Produce on DGX only: `reports/generated/pass200_rsta_receipt/${handoff_commit}-sign-control-comparator-integrity-all-seeds.json`
- Never produce in this plan: a scientific Stage A result

**Interfaces:**
- Consumes: independently reviewed source, the exact amendment authority, and already-GREEN validators.
- Produces: one manifest-only handoff and at most one candidate-free all-seed DGX audit.

- [ ] **Step 1: Refreeze only the manifest with already-GREEN validators**

  Insert this exact object immediately after `normwise_adjoint_amendment` and
  before `binding_receipt`:

  ```json
  "normwise_adjoint_sign_control_amendment": {
    "path": "docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md",
    "sha256": "b87830197b162e6e3ce9ed20a3d631e138a1054d7af382054358c82867441259",
    "commit": "a27dd7b3c8ff089c7cb80821c43658b975985a34"
  }
  ```

  Set `current_scientific_source.git_revision` to the exact
  `$reviewed_source_commit`. Refresh hashes only for paths whose Git blobs
  changed, while preserving the existing exact 31 paths/order. Preserve every
  prior manifest object byte-semantically.

  Run the already-GREEN validators against the real manifest:

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'sign_control and (manifest or source or projection or candidate_free)'
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/ruff check scripts/rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/python -m py_compile scripts/rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  git diff --check
  ```

  Expected: PASS. If a validator requires a source/test edit, do not weaken it
  in Task 7. Return to Task 6, create a new reviewed source commit, and restart
  the manifest-only refreeze.

- [ ] **Step 2: Commit and authenticate the manifest-only handoff**

  Run:

  ```bash
  test "$(git diff --name-only)" = docs/pass200_rsta_receipt_stage_a_manifest.json
  git add -- docs/pass200_rsta_receipt_stage_a_manifest.json
  test "$(git diff --cached --name-only)" = docs/pass200_rsta_receipt_stage_a_manifest.json
  git diff --cached --check
  git commit -m "refreeze RSTA sign-control comparator handoff"
  handoff_commit=$(git rev-parse HEAD)
  test "$(git diff-tree --no-commit-id --name-only -r "$handoff_commit")" = docs/pass200_rsta_receipt_stage_a_manifest.json
  git diff --exit-code "$handoff_commit^" "$handoff_commit" -- scripts tests reports
  sha256sum docs/pass200_rsta_receipt_stage_a_manifest.json
  printf '%s\n' "$handoff_commit"
  ```

  Expected: the handoff changes the manifest only.

- [ ] **Step 3: Prepare an isolated authenticated DGX checkout**

  Transfer the exact handoff commit to the registered DGX, create a new
  detached clean checkout at `$handoff_commit`, and run:

  ```bash
  test "$(git rev-parse HEAD)" = "$handoff_commit"
  test -z "$(git status --porcelain --untracked-files=all)"
  test "$(sha256sum docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md | cut -d ' ' -f 1)" = b87830197b162e6e3ce9ed20a3d631e138a1054d7af382054358c82867441259
  test "$(git show a27dd7b3c8ff089c7cb80821c43658b975985a34:docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md | sha256sum | cut -d ' ' -f 1)" = b87830197b162e6e3ce9ed20a3d631e138a1054d7af382054358c82867441259
  nvidia-smi
  ```

  Independently authenticate every manifest authority, source Git
  blob/worktree hash, receipt/artifact path and digest, CUDA environment, and
  absence of another Pass 200 RSTA process. Export
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts.

- [ ] **Step 4: Run exactly one candidate-free all-seed audit**

  Run:

  ```bash
  handoff_commit=$(git rev-parse HEAD)
  integrity_output="reports/generated/pass200_rsta_receipt/${handoff_commit}-sign-control-comparator-integrity-all-seeds.json"
  test ! -e "$integrity_output"
  test ! -L "$integrity_output"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python scripts/diagnose_pass200_rsta_stage_a.py \
    --manifest docs/pass200_rsta_receipt_stage_a_manifest.json \
    --binding-receipt docs/pass200_rsta_binding_receipt_d6270a9.json \
    --output "$integrity_output" \
    --integrity-all-seeds-only
  integrity_exit=$?
  sha256sum "$integrity_output"
  printf 'exit=%s\n' "$integrity_exit"
  ```

  Collect the original exit code and output hash from this one process. Do not
  rerun to seek a different result.

- [ ] **Step 5: Independently validate and stop before science**

  Validate exact execution/manifest/environment/binding schemas; exact seeds
  `0,1,2,3`; every unchanged legacy/normwise/rebuild/reversed value; each sign
  control's exact eight-key order; both reference hashes equal baseline;
  `reference_exact_action_hash_match`, direct relation evidence, `passed`, and
  `integrity_passed`; all graph/candidate-forbidden invariants; and global
  `all_passed`.

  If structural validation fails or `all_passed=false`, keep RSTA blocked and
  run no scientific command. If it is green, record only that it is
  candidate-free executability evidence. Scientific execution is not part of
  this plan and remains forbidden until the green artifact receives separate
  independent authentication and the existing authorization process explicitly
  permits science.
