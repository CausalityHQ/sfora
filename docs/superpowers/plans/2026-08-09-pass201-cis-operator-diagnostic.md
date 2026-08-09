# Pass201 CIS Operator Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the prospectively frozen, source-bound Pass201 diagnostic that distinguishes coalition-specific cross-image action from scalar-gradient and proxy-only explanations without reading any CIS result.

**Architecture:** One import-safe diagnostic script owns pure codecs, deterministic production-context construction, exact PyTorch operator mathematics, strict payload validation, and a three-process CLI controller. Tiny local tests use literal synthetic fixtures; current-code In-Shop source activation and all full-dataset/model execution occur only in an isolated DGX checkout after the source-bound PA controller exits successfully.

**Tech Stack:** Python 3.11, NumPy `PCG64`, PyTorch autograd/stateless functional calls, pytest, Ruff, SHA-256/canonical JSON.

## Global Constraints

- Authoritative specification: `docs/pass201_cis_operator_diagnostic_draft_2026-08-09.md` at commit `2e4f86d`; implement every exact field, formula, threshold, conditional payload, invalidation rule, and process order literally.
- The draft remains `PENDING_SOURCE`: no operator value, CIS artifact, Pass120 report/log/checkpoint/table, or benchmark metric may be read during implementation.
- The only admissible source is the fresh seed-0 ordinary Proxy Anchor artifact authenticated by `docs/pass201_pa_source_prelaunch_manifest.json` SHA-256 `37644551f99976a7982589c1574effa00a9c77aa4a690117b5a8cd84244cc803` and a successful post-run source replay.
- Query/gallery data are artifact-binding-only; scientific contexts and outcomes use the In-Shop training split exclusively.
- Exact constants: batch size 180, 32 context pairs, 256 null replicates, 20,000 bootstrap replicates, seeds 2010809/2010810/2010811/2010812, FP32, autocast/TF32 off, deterministic algorithms fail-closed.
- Preserve production train-mode BN semantics with disposable buffers, one shared 180-row graph per context, no microbatching, no persistent checkpoint mutation, and clean eval-mode `S_prime` outcomes.
- Exactly six operators, two parameter panels, two stateless regimes, panel-specific equal-norm references, and the frozen PASS/FAIL/UNRESOLVED predicates; no tuning or substitute controls.
- Exactly three fresh processes prepare the same 32 input contexts; integrity A and B score only context 0, scientific scores 0–31 sequentially.
- Local tests must be tiny/mocked and run serially. Full datasets, BN-Inception, 180-row contexts, integration smoke, and scientific computation run only on DGX `spark-2751`.
- Use strict TDD: each production behavior begins with a focused test that fails for the intended missing behavior, then minimal implementation, then fresh GREEN evidence. Do not overlap tests.
- Create only `scripts/diagnose_pass201_cis_operator.py` and `tests/test_diagnose_pass201_cis_operator.py` unless a reviewed task proves another file is necessary. Do not edit the trainer or recipes.
- Never touch untracked `HANDOFF_BRIEF.md`, `RSPG_SPECDEFECT.md`, or `RSPG_TASK.md`.

---

### Task 1: Deterministic Codecs, Context Construction, and Schema Validation

**Files:**
- Create: `scripts/diagnose_pass201_cis_operator.py`
- Create: `tests/test_diagnose_pass201_cis_operator.py`

**Interfaces:**
- Produces: `canonical_json_bytes(value) -> bytes`, `sha256_tensor_frame(array) -> str`, `sha256_named_tensors(named_tensors) -> str`, `build_input_context_digest(context) -> dict[str, Any]`, `construct_context_audit(...) -> tuple[list[dict], list[dict]]`, `bootstrap_indices() -> np.ndarray`, `summarize_metric(values, bootstrap_indices) -> dict[str, Any]`, `validate_construction_evidence(payload, raw_evidence) -> None`, and `validate_payload_structure(payload) -> None`.
- Consumes later: Task 2 fills complete context/operator records; Task 3 uses the validators and digests at every process boundary.

- [ ] **Step 1: Write failing codec and context tests**

  Add literal fixtures that prove name/shape/length framing prevents concatenation collisions, tensor dtype/shape changes alter the digest, `S_prime` preserves label order while excluding all `S` IDs, representative selection uses minimum stable index, reuse is causal-prefix-only, and a rejected batch remains in the partial audit.

  ```python
  def test_named_tensor_digest_frames_names_shapes_and_lengths():
      left = [("ab", np.array([1.0], dtype="<f8"))]
      right = [("a", np.array([98.0, 1.0], dtype="<f8"))]
      assert MODULE.sha256_named_tensors(left) != MODULE.sha256_named_tensors(right)

  def test_s_prime_is_disjoint_and_preserves_literal_label_sequence():
      context = MODULE.construct_one_context(
          rows=LITERAL_ROWS, train_manifest=LITERAL_MANIFEST, context_index=0
      )
      assert context["row_labels"] == [7, 3, 7, 5]
      assert context["s_prime_example_ids"] == ["7-b", "3-b", "7-c", "5-b"]
      assert set(context["row_example_ids"]).isdisjoint(context["s_prime_example_ids"])
  ```

- [ ] **Step 2: Run the focused tests and capture RED**

  Run: `pytest -q tests/test_diagnose_pass201_cis_operator.py -k 'digest or s_prime or representative or reuse or partial'`

  Expected: collection succeeds and fails because the new interfaces do not exist.

- [ ] **Step 3: Implement minimal pure codecs and context logic**

  Keep imports side-effect-free. Encode integer fields with `struct.pack('<I'/'<q'/'<Q')`, canonicalize JSON exactly with `sort_keys=True`, compact separators, UTF-8, `ensure_ascii=False`, and `allow_nan=False`. Build accepted/rejected audit records from literal row metadata without loading images or torch.

- [ ] **Step 4: Write failing summary and conditional-schema tests**

  Cover all three payload families: scored, early insufficient contexts, and BLOCKED/INVALID. Mutate every exact field family: extra key, missing key, wrong null, nonfinite float, status/decision mismatch, wrong process prefix, source-activation invalid count, component/reason-code mixing, malformed digest, and a context-0 hash that includes process metadata.

  ```python
  @pytest.mark.parametrize("mutation", CONDITIONAL_PAYLOAD_MUTATIONS)
  def test_payload_validator_fails_closed(mutation):
      payload = deepcopy(LITERAL_VALID_PAYLOAD)
      mutation(payload)
      with pytest.raises(ValueError):
          MODULE.validate_payload_structure(payload)

  def test_construction_validation_rejects_digest_raw_evidence_mismatch():
      payload, evidence = literal_valid_scored_payload_and_evidence()
      evidence["gradient_tensors"]["network_only"][0][1][0] += 1.0
      with pytest.raises(ValueError, match="gradient_sha256"):
          MODULE.validate_construction_evidence(payload, evidence)
  ```

- [ ] **Step 5: Run schema tests and capture RED**

  Run: `pytest -q tests/test_diagnose_pass201_cis_operator.py -k 'payload or summary or bootstrap'`

  Expected: failures name the missing validator/summary behavior rather than fixture errors.

- [ ] **Step 6: Implement summary, bootstrap, and strict validators**

  Use one `(20000,32)` little-endian int64 `PCG64(2010811)` index matrix for every metric. Store float64 replicate distributions in fixed replicate order. `validate_construction_evidence` receives the still-live raw gradient/update tensors, null vectors, bootstrap index matrix, and every bootstrap distribution and recomputes their digests before those raw values are discarded. `validate_payload_structure` never claims to reconstruct omitted evidence: it compares exact key sets, digest syntax, contained context/input hashes, component decisions, failure predicates, overall status, and authorized action.

  Add RED→GREEN tests proving all metrics use the identical resample-index matrix, its digest is the little-endian int64 C-order bytes, every distribution digest is little-endian float64 C-order bytes, and threshold values exactly equal to each LCB/UCB boundary obey the frozen inclusive/strict comparisons and failure precedence.

- [ ] **Step 7: Verify Task 1 GREEN and commit**

  Run serially:

  ```bash
  pytest -q tests/test_diagnose_pass201_cis_operator.py
  ruff check scripts/diagnose_pass201_cis_operator.py tests/test_diagnose_pass201_cis_operator.py
  python -m py_compile scripts/diagnose_pass201_cis_operator.py tests/test_diagnose_pass201_cis_operator.py
  git diff --check
  ```

  Commit only the two task files with message `implement Pass201 diagnostic foundations`.

---

### Task 2: Exact Operator Mathematics and Stateless Outcome Panels

**Files:**
- Modify: `scripts/diagnose_pass201_cis_operator.py`
- Modify: `tests/test_diagnose_pass201_cis_operator.py`

**Interfaces:**
- Consumes: Task 1 codecs and record validators.
- Produces: `coalition_losses(embeddings, labels, sample_indices, proxies, proxy_labels) -> dict[str, Tensor]`, `operator_gradients(...) -> dict[str, PanelGradient]`, `make_stateless_updates(...) -> dict[str, Any]`, `shared_confuser_statistic(...) -> dict[str, Any]`, `owner_outcomes(...) -> OutcomeFields`, and `score_context(...) -> dict[str, Any]`.

- [ ] **Step 1: Write failing literal-loss tests**

  Use a double-precision 4-row/3-proxy hand-computed fixture. Assert the exact six losses, minimum-index representative selection, full-union targets, complementary targets, deterministic dropout targets, default BCE mean, summed `1/sqrt(m)` scaling, and proxy inclusion/exclusion by panel.

  ```python
  def test_atomic_full_union_is_per_image_and_summed_union_is_cross_image():
      members = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
      proxies = torch.tensor(
          [[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]],
          dtype=torch.float64,
      )
      losses = MODULE.coalition_losses(
          embeddings=members,
          labels=torch.tensor([10, 20]),
          sample_indices=torch.tensor([4, 9]),
          proxies=proxies,
          proxy_labels=torch.tensor([10, 20, 30]),
      )
      assert losses["atomic_full_union"].item() == pytest.approx(
          0.7047830585784727, abs=1e-12
      )
      assert losses["summed_union"].item() == pytest.approx(
          0.7049762468198759, abs=1e-12
      )
  ```

- [ ] **Step 2: Run the loss tests and capture RED**

  Run: `pytest -q tests/test_diagnose_pass201_cis_operator.py -k 'operator_loss or full_union or complementary or dropout'`

  Expected: failures are missing operator functions.

- [ ] **Step 3: Implement the exact six loss graphs and panel gradients**

  Call the activated production `_proxy_anchor_loss` for ordinary PA. Implement diagnostic-only formulas in the new script without editing trainer/recipe code. Preserve all 180 embeddings in the shared graph, select one representative per class by minimum stable index, normalize embeddings/proxies through production paths, and use one autograd call per operator with retained graph.

- [ ] **Step 4: Write failing gradient/update/outcome tests**

  Use a tiny linear normalized model with a learnable proxy table. Compare each autograd gradient to finite differences, prove complementary has no other-member embedding derivative, prove summed union does, prove network-only zeroes proxy updates, prove joint includes them, and prove panel-specific equal-norm error is within `1e-10*max(rho,1e-12)`. For a literal clean support fixture, compare `R_F`, `Delta_M`, `D_F`, and `D_M` to hand-derived values and verify a small stateless step agrees in sign with the directional derivative.

  Add explicit RED cases for exact lexicographically ordered trainable parameter-name membership, one missing/disconnected required parameter, one unexpected trainable parameter, independent per-row null permutations from `PCG64(2010810)`, the exact frozen null stream order, and a null implementation that incorrectly reuses one permutation across rows.

- [ ] **Step 5: Run math tests and capture RED**

  Run: `pytest -q tests/test_diagnose_pass201_cis_operator.py -k 'gradient or equal_norm or outcome or shared_confuser'`

  Expected: each new test fails on its named missing behavior.

- [ ] **Step 6: Implement gradients, virtual updates, outcomes, and null**

  Apply activated LR and proxy multiplier literally. Keep configured-loss and equal-norm regimes separate. Use stateless functional calls against clean `S_prime`, preserve original BN buffers, stream the 256 `PCG64(2010810)` proxy-column permutations, and persist exact gradient/update/null hashes.

- [ ] **Step 7: Add the train-BN disposable-buffer regression**

  First write a failing test with a tiny train-mode BN model proving one shared forward supplies all operators, disposable buffers may change, persistent buffers and flags remain byte-identical, and an eval-mode substitution produces a different output. Then implement the minimal bufferless graph helper.

- [ ] **Step 8: Verify Task 2 GREEN and commit**

  Run the full focused file serially plus Ruff, `py_compile`, and `git diff --check`. Commit only the two task files with message `implement exact Pass201 operator panels`.

---

### Task 3: Source Binding, Three-Process Replay, and Atomic CLI

**Files:**
- Modify: `scripts/diagnose_pass201_cis_operator.py`
- Modify: `tests/test_diagnose_pass201_cis_operator.py`

**Interfaces:**
- Consumes: Tasks 1–2 pure/scientific functions.
- Produces: `activate_source(args) -> tuple[dict[str, Any], dict[str, Any]]`, `run_process_role(role, source_manifest, output_path) -> None`, `compare_integrity_records(a, b, scientific) -> None`, `run_controller(args) -> None`, and CLI modes `--activate-source`, `--process-role`, `--binding-only`, `--smoke-only`, and `--scientific`. Activation emits exactly `docs/pass201_cis_operator_activated_preregistration.json` followed by `docs/pass201_cis_operator_source_manifest.json`.

- [ ] **Step 1: Write failing source-binding tests**

  Construct a tiny temporary git repository and synthetic report/checkpoint/config/train manifest. Assert exact prelaunch manifest SHA, executing-source tree/revision, run-command, dataset tree, report/checkpoint embedded cross-digests, seed/config/objective/batch/BN/proxy invariants, and post-run source/data replay. Mutate every binding independently and require failure before importing torch/model code or calling any scorer.

- [ ] **Step 2: Run binding tests and capture RED**

  Run: `pytest -q tests/test_diagnose_pass201_cis_operator.py -k 'source_binding or activation'`

  Expected: missing activation functions fail; scorer sentinels remain untouched.

- [ ] **Step 3: Implement source activation and activated preregistration emission**

  Read no CIS paths. First emit `docs/pass201_cis_operator_activated_preregistration.json`. Its exact top-level keys are `schema_version`, `frozen_draft_path`, `frozen_draft_sha256`, `result_path`, `source`, `constants`, `thresholds`, and `authorized_action`; `schema_version` is `pass201-cis-activated-preregistration-v1`, `result_path` is exactly `reports/generated/pass201_cis_operator/pass201_inshop_seed0.json`, `authorized_action` is exactly `binding_and_integrity_smoke_then_scientific_if_green`, and `source` contains every authenticated source-dependent literal except any source-manifest or self digest. The action prospectively authorizes full scientific execution only when the exact three-process smoke in Task 4 returns zero, validates every binding/integrity predicate, and emits no INVALID/BLOCKED reason; no post-smoke discretion or second authorization artifact is allowed. Atomically write and validate this file, then hash its committed-intent bytes.

  Second emit `docs/pass201_cis_operator_source_manifest.json` with exactly the `pass201-source-v1` keys frozen in the draft. Replace every draft `null`, including `activated_preregistration_sha256`, with a literal: that field is the SHA-256 of the first file's exact bytes. The source manifest contains no digest of itself and the activated preregistration contains no digest of the later source manifest, so the graph is acyclic. Write each file by temporary sibling, file `fsync`, `os.replace`, and parent-directory `fsync`; validate both complete artifacts after both replacements. If either write or validation fails, remove neither prior committed file and authorize no binding/smoke. No third activation artifact is permitted.

- [ ] **Step 4: Write failing process/replay/CLI tests**

  In subprocesses with a tiny fake dataset/model, prove exactly three ordered roles, identical 32 input digest records, context-0 equality, A/B scoring only context 0, scientific scoring all 32 sequentially, sole process-start RNG reset, prefix-only reduced integrity on failures, atomic output, and no aggregate contribution from A/B. Patch scientific scoring to raise if binding or replay validation has not already completed.

  Add explicit RED cases proving `CUBLAS_WORKSPACE_CONFIG` is present before the child imports torch, deterministic flags are set before model creation, tensor/scalar replay residuals exactly on each tolerance pass while the next representable larger value fails, and process-start RNG initialization occurs once with no reset between preparation and scoring.

- [ ] **Step 5: Run process tests and capture RED**

  Run: `pytest -q tests/test_diagnose_pass201_cis_operator.py -k 'process_role or replay or cli or atomic'`

  Expected: failures identify missing orchestration, not unavailable DGX dependencies.

- [ ] **Step 6: Implement controller and child modes**

  Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the parent environment before child Python starts. Each child initializes the exact RNGs once, records environment/device/RNG hashes, prepares all 32 contexts, and follows its role. The controller validates child JSON before starting the next role and rejects any tolerance, hash, status, or schema mismatch before scientific aggregation.

- [ ] **Step 7: Verify Task 3 GREEN and commit**

  Run the full focused file once, then Ruff, `py_compile`, and `git diff --check`. Commit only the two task files with message `add bound Pass201 diagnostic controller`.

---

### Task 4: Independent Review, DGX Activation, Integrity Smoke, and Scientific Run

**Files:**
- Modify only if review finds a tested defect: `scripts/diagnose_pass201_cis_operator.py`, `tests/test_diagnose_pass201_cis_operator.py`
- Create after successful source binding, in this exact order: `docs/pass201_cis_operator_activated_preregistration.json` and `docs/pass201_cis_operator_source_manifest.json`; commit and push both before scientific values.
- Create after execution: the exact result JSON path named by the activated preregistration.

**Interfaces:**
- Consumes: reviewed Task 3 CLI, successful source-bound PA controller receipt, clean isolated DGX checkout.
- Produces: source-activation commit, binding-only receipt, integrity-smoke receipt, and—only if both pass—the frozen full scientific result.

- [ ] **Step 1: Run a broad no-values code review**

  Give the reviewer the frozen draft, plan, ledger, full diff package, and task reports. Require separate spec-compliance and code-quality verdicts. Fix Critical/Important findings through tested RED→GREEN rounds and scoped re-review; do not inspect candidate values.

- [ ] **Step 2: Verify the source-bound PA controller**

  On DGX, require the durable controller receipt for the exact command/source manifest to record exit status zero and validate its post-run source/data/output digests. PID 933996 may be recorded only as supporting operational evidence; PID identity or continued existence is neither necessary nor sufficient because PIDs can be reused. If the durable receipt is absent, mismatched, or failed, emit only a BLOCKED/INVALID reduced payload as specified and do not substitute another checkpoint.

- [ ] **Step 3: Activate source fields and commit before values**

  Run `--activate-source` and `--binding-only` in a clean detached DGX checkout created from the reviewed commit. Copy back only `docs/pass201_cis_operator_activated_preregistration.json` and `docs/pass201_cis_operator_source_manifest.json`. Locally validate both schemas, their literal file hashes, their cross-file preregistration hash, and equality to the durable controller receipt; do not claim to recompute remote checkpoint/data digests without copying those underlying artifacts. Commit and push both files together. Recreate the DGX checkout at that activation commit before smoke; the result path comes literally from the activated preregistration.

- [ ] **Step 4: Run DGX integrity smoke**

  Use exact B=180 and the real BN-Inception/source artifact. Smoke launches all three production preparation processes, requires each to construct the same 32 input-context digest records, and scores only context 0 in each process; it emits integrity/binding evidence only, never a candidate aggregate, decision component, or candidate value, and cannot call adjudication. Require deterministic settings, source binding, disposable-buffer invariants, exact parameter membership, operator finite guards, and replay tolerances to pass. A failure writes INVALID and stops.

- [ ] **Step 5: Run the full scientific diagnostic only after green smoke**

  Launch the original controller once and retain its PID/session. Do not duplicate it. Poll at most every 55 seconds, collect final exit status, validate the result JSON independently, and adjudicate only with the frozen thresholds. No timing measurement is scientific evidence.

- [ ] **Step 6: Persist the verdict and next hill-climb action**

  Commit/push the exact result and append the mechanism-level verdict to `docs/method_search_verdict.md`. PASS authorizes only a new, separate prospective GPU-training preregistration. FAIL/UNRESOLVED closes or refines the mechanism exactly as frozen; never rewrites Pass120.

---

## Self-Review Record

- Spec coverage: source activation, production contexts, exact transforms, three-process replay, train-BN buffer handling, all six operators, both panels/regimes, null, outcomes, bootstrap, thresholds, schemas, invalidation, and DGX-only heavy execution each map to a task.
- Placeholder scan: no forbidden placeholder phrase, illustrative expectation, or unspecified implementation/testing step remains.
- Type consistency: Task 1 codecs/validators feed Task 2 records; Task 2 `score_context` feeds Task 3 processes; Task 3 CLI/activated manifest feed Task 4 without another implementation path.
