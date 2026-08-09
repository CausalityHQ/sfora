# Pass 200 RSTA Normwise-Adjoint Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate the preregistered normwise adjoint metric on deterministic synthetic CPU fixtures, and only after a passing independently reviewed calibration prospectively amend and implement the real candidate-free RSTA integrity gate.

**Architecture:** A small candidate-independent numerical module owns FP64 diagnostic reductions and the frozen synthetic fixtures. A separate CPU-only CLI produces one strict calibration artifact; production RSTA code cannot change until that artifact passes and a later amendment binds it. Production integration then reuses the reviewed diagnostic helper while preserving the legacy metrics and all-seed candidate-free prefix.

**Tech Stack:** Python 3.11, PyTorch `torch.func.jvp`/`torch.func.vjp`, NumPy PCG64, pytest, strict JSON, SHA-256, Git, Ruff, CPU calibration, DGX deterministic CUDA audit.

## Global Constraints

- Implement `docs/pass200_rsta_normwise_adjoint_calibration_protocol_2026-08-09.md` literally.
- H8 is `cc7b0a102d938db0cf49e756fc4d18410186bf4d`; its candidate-free audit SHA-256 is `234cc3055b0209bcd095f2932867ff09252ca1c2d4b5e5080a988c77fcd5e74c`.
- Disclose all four H8 legacy errors exactly as recorded in the protocol. H8 remains failed and no candidate value has been observed.
- Freeze `beta_norm=2*abs(lhs-rhs)/(||u||||a||+||v||||b||)`, threshold `5e-4`, and correct-fixture ceiling `6.25e-5`. Do not tune any threshold, fixture, seed, dimension, scale, or fault after observing calibration output.
- Cast every FP32 factor to float64 before multiplication; perform all products, sums, sum-of-squares norms, per-parameter reductions, and the RHS stack sum in float64 exact named-parameter order.
- Retain the complete legacy adjoint object and metric in all future production evidence. This work may add evidence but must not erase H8 or rewrite it as passing.
- Calibration is CPU-only, single-threaded, synthetic-only, and candidate-free. It must not load manifests, checkpoints, data, or models and must not import or call any field, scoring, bootstrap, decision, payload, or receiver-row function.
- Calibration failure keeps RSTA blocked and ends this plan before amendment or production work. No calibration retry with changed constants is permitted.
- Commit and authenticate every complete structurally valid calibration artifact, including `all_passed=false`; a negative artifact stops before amendment but remains durable evidence.
- No DGX or real-data execution occurs before a passing calibration artifact, independent reviews, prospective amendment, reviewed production source, and manifest-only refreeze.

## File structure

- `scripts/rsta_normwise_adjoint.py`: pure diagnostic arithmetic, hashing, fixture definitions, controls, and strict result validation; no RSTA candidate dependency.
- `scripts/calibrate_pass200_rsta_normwise_adjoint.py`: CPU-only atomic calibration CLI with only `--output`.
- `tests/test_rsta_normwise_adjoint.py`: independent numerical references, frozen fixture/fault construction, strict-schema mutation tests, and forbidden-import/call assertions.
- `reports/generated/pass200_rsta_receipt/${calibration_source_commit}-normwise-adjoint-calibration.json`: complete candidate-free calibration result produced once from reviewed source.
- `docs/pass200_rsta_normwise_adjoint_amendment_2026-08-09.md`: authored only after a passing calibration; binds the protocol and result.
- `scripts/diagnose_pass200_rsta_stage_a.py`: later production integration of reviewed normwise evidence and gate.
- `tests/test_diagnose_pass200_rsta_stage_a.py`: later production schema, numerical, prefix, and candidate-free tests.
- `docs/pass200_rsta_receipt_stage_a_manifest.json`: final manifest-only refreeze after reviewed production source.

---

### Task 1: Normwise Arithmetic and Exact Schema

**Files:**
- Create: `scripts/rsta_normwise_adjoint.py`
- Create: `tests/test_rsta_normwise_adjoint.py`

**Interfaces:**
- Produces: `normwise_adjoint_metrics(u: Tensor, a: Tensor, parameter_direction: Mapping[str, Tensor], vjp_action: Mapping[str, Tensor], parameter_names: Sequence[str]) -> dict[str, object]`.
- Produces: `tensor_sha256(tensor: Tensor) -> str`, `parameter_tree_sha256(tree: Mapping[str, Tensor], names: Sequence[str]) -> str`, and `validate_calibration_result(value: object) -> None`.

- [ ] **Step 1: Write and run the cast-before-product RED**

  Add `test_normwise_metrics_cast_every_factor_before_product_and_norm`. Use FP32 vectors with alternating large products and small residual terms. Independently compute every expected value with NumPy float64 after conversion, then require exact `lhs`, `rhs`, absolute-product sums, four L2 norms, denominator, `eta_norm`, and `beta_norm`. Monkeypatch `Tensor.float` and FP32 `sum` paths to raise if the implementation reduces before casting.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py::test_normwise_metrics_cast_every_factor_before_product_and_norm
  ```

  Expected: FAIL because `scripts/rsta_normwise_adjoint.py` does not exist.

- [ ] **Step 2: Implement only the exact arithmetic and verify GREEN**

  Implement the formulas and corner conventions from the protocol. Flatten no parameter tensor for computation; reduce each named tensor in exact supplied order, stack its float64 scalar terms, and reduce that stack in float64. Reject wrong dtype, device, topology, shape, or nonfinite factors. Rerun Step 1 and require PASS.

- [ ] **Step 3: Write and run corner/schema/hash REDs**

  Add:

  ```python
  def test_normwise_zero_zero_corner_is_exact_zero(): ...
  def test_normwise_zero_positive_corner_is_infinity_and_fails(): ...
  def test_cancellation_factor_corner_contract(): ...
  def test_action_hashes_cover_actual_c_contiguous_fp32_bytes_in_named_order(): ...
  def test_calibration_schema_rejects_every_recursive_mutation(): ...
  ```

  The recursive mutation test removes and adds every key, changes every fixed
  value/type, substitutes NaN/infinity JSON numbers, changes fixture order, and
  makes each recomputable scalar, hash, threshold, or `passed` flag inconsistent.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py -k 'corner or hashes or schema'
  ```

  Expected: FAIL because corner serialization, hashing, and strict validation are absent.

- [ ] **Step 4: Implement only corner handling, hashes, and validation**

  Use JSON string `"infinity"` only in the two protocol-authorized derived corner cases. Hash actual C-contiguous FP32 bytes. Implement the exact top-level, environment, fixture-entry, and control schemas and recompute all derived fields during validation. Rerun Steps 1 and 3 and require PASS.

- [ ] **Step 5: Commit the arithmetic unit**

  ```bash
  git add scripts/rsta_normwise_adjoint.py tests/test_rsta_normwise_adjoint.py
  git commit -m "add RSTA normwise adjoint arithmetic"
  ```

---

### Task 2: Frozen Correct Fixtures

**Files:**
- Modify: `scripts/rsta_normwise_adjoint.py`
- Modify: `tests/test_rsta_normwise_adjoint.py`

**Interfaces:**
- Produces: `correct_fixture_specs() -> tuple[FixtureSpec, ...]` in exact protocol order.
- Produces: `run_correct_fixture(spec: FixtureSpec) -> dict[str, object]` using only `torch.func` actions.

- [ ] **Step 1: Write and run construction REDs before fixture code**

  Add one parameterized test over the exact six fixture keys
  `zero_corner`, the three affine scales, `smooth_parameter_tree`, and
  `paired_cancellation`. Independently reconstruct every PCG64 array and assert
  exact C-order FP32 bytes and recursive equality with every literal ordered
  `fixture_id`, `kind`, `seeds`, `dimensions`, and `scales` JSON object in the
  protocol. Also assert named-parameter order, pair duplication, cancellation
  diagonal, and final `1.0` direction entries. Reject numerically equivalent
  scale encodings, reordered keys, extra keys, wrong string/integer types, and
  any seed string not matching lowercase `0x` plus sixteen hex digits.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py::test_correct_fixture_construction_is_byte_exact
  ```

  Expected: FAIL because `correct_fixture_specs` does not exist.

- [ ] **Step 2: Implement only frozen fixture construction**

  Encode the literal seeds, shapes, scales, and formulas from the protocol. Use
  a fresh NumPy `Generator(PCG64(seed))` for each listed draw. Reject any CUDA
  device or non-FP32 action tensor. Rerun Step 1 and require PASS.

- [ ] **Step 3: Write and run torch.func and calibration-band REDs**

  Wrap `torch.func.jvp` and `torch.func.vjp` to count calls and capture action
  tensors. Require exactly one real JVP and VJP per baseline, no analytic action
  substitution, exact action hashes, and `beta_norm <= 6.25e-5` for every correct
  fixture. Independently require the paired-cancellation fixture's mathematical
  `lhs=rhs=2**-10` construction and a positive large absolute-product sum.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py -k 'correct_fixture_uses_torch_func or correct_fixture_calibration_band'
  ```

  Expected: FAIL because fixture execution is absent.

- [ ] **Step 4: Implement minimal correct-fixture execution**

  Obtain actions only from fresh `torch.func.jvp` and `torch.func.vjp` calls,
  pass the actual FP32 actions to Task 1, and preserve all observed values even
  on a finite threshold failure. Rerun all Task 2 tests. If a frozen correct
  fixture exceeds `6.25e-5`, stop the entire plan: do not change its constants.

- [ ] **Step 5: Commit the correct fixtures**

  ```bash
  git add scripts/rsta_normwise_adjoint.py tests/test_rsta_normwise_adjoint.py
  git commit -m "add frozen RSTA adjoint fixtures"
  ```

---

### Task 3: Reproducibility Controls and Guaranteed Faults

**Files:**
- Modify: `scripts/rsta_normwise_adjoint.py`
- Modify: `tests/test_rsta_normwise_adjoint.py`

**Interfaces:**
- Produces: `run_fixture_controls(spec: FixtureSpec) -> dict[str, object]`.
- Produces: `registered_fault_specs() -> tuple[FaultSpec, ...]` and `run_registered_fault(spec: FaultSpec) -> dict[str, object]`.

- [ ] **Step 1: Write and run rebuild/order/sign REDs**

  Test fresh reconstruction from every seed, baseline order, reversed VJP/JVP
  order, `-v`, and `-u` exactly as registered. Require byte-identical baseline,
  rebuild, and reverse-order hashes; exact `torch.equal` sign relations; and
  every sign-trial `beta_norm <= 6.25e-5`. Inject one altered rebuild tensor and
  one order-dependent fake action and require deterministic failure.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py -k 'rebuild or action_order or sign_control'
  ```

  Expected: FAIL because controls are absent.

- [ ] **Step 2: Implement only the reproducibility controls**

  Rebuild graphs and arrays rather than reusing actions. Persist every control
  hash and metric, and make any exact-reproducibility failure set fixture
  `passed=false`. Rerun Step 1 and require PASS.

- [ ] **Step 3: Write and run fault-construction REDs before fault code**

  Independently require recursive equality with all three literal ordered fault
  metadata objects, then construct all three fault action pairs. Assert the zero-map
  injection has mathematical `beta_norm=2`, the reverse scale fault has
  `beta_norm=2/511`, and the paired-sign fault has `beta_norm=1`. Require their
  unmodified controls to meet `6.25e-5`, each observed fault to be at least
  `5e-4`, and each separation to be at least `4.375e-4`. Mutate each registered
  amplitude, seed, dimension, or pair pattern and require schema/provenance
  rejection.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py -k 'registered_fault'
  ```

  Expected: FAIL because registered faults are absent.

- [ ] **Step 4: Implement only the exact fault matrix**

  Build passing zero/identity controls first, then substitute precisely the
  registered action tensor. Do not choose a fault magnitude from an observed
  metric. Persist substituted action hashes/norms and the unmodified control.
  Rerun every Task 3 test. A missed separation stops the plan without tuning.

- [ ] **Step 5: Commit controls and faults**

  ```bash
  git add scripts/rsta_normwise_adjoint.py tests/test_rsta_normwise_adjoint.py
  git commit -m "calibrate RSTA adjoint separation"
  ```

---

### Task 4: Candidate-Free Calibration CLI

**Files:**
- Create: `scripts/calibrate_pass200_rsta_normwise_adjoint.py`
- Modify: `tests/test_rsta_normwise_adjoint.py`

**Interfaces:**
- Produces: `calibration_payload(protocol_path: Path) -> dict[str, object]`.
- Produces CLI: `python scripts/calibrate_pass200_rsta_normwise_adjoint.py --output ABSENT_PATH`.

- [ ] **Step 1: Write and run isolation/schema/atomicity REDs**

  Import the CLI with raising sentinels for all forbidden RSTA functions and
  loaders. Require only `--output`, CPU/single-thread configuration before
  tensor work, exact fixture/fault order and metadata, exact recursive schema,
  protocol Git blob/worktree/SHA binding, and fixed candidate-free values.
  Require exact `execution_audit` and `source` keys: full commits, source commit
  ancestor-or-equal to executing HEAD, the exact helper/CLI/test path order and
  SHA-256 values, equality among worktree bytes and Git blobs at both the
  reviewed source and executing commits, and executing CLI equality with its
  source entry. Mutate every path, digest,
  revision, ancestry relation, worktree byte, and Git blob and require failure
  before fixture construction. Require the normalized destination to be exactly
  `reports/generated/pass200_rsta_receipt/${calibration_source_commit}-normwise-adjoint-calibration.json`
  below repository root and parse the reviewed source commit only from that
  40-hex basename prefix.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py -k 'cli or candidate_free or atomic'
  ```

  Expected: FAIL because the CLI and source authenticator do not exist.

- [ ] **Step 2: Write and run exact no-clobber publication REDs**

  Use the protocol's exact same-directory
  `.${destination.name}.tmp.${decimal_pid}` path. Test pre-existing destination,
  pre-existing exact temp, a destination race immediately before `os.link`,
  short write, file `fsync`, hard-link, both directory `fsync` points, and temp
  unlink failures. Require `xb`, mode `0o600`, no symlink following,
  hard-link/no-replace publication, inode-checked rollback, deletion of only a
  process-owned temp/link, unchanged foreign files, and no successful output
  unless its bytes equal the validated buffer plus one LF.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py -k 'no_clobber or publication_rollback'
  ```

  Expected: FAIL because exact hard-link publication is absent.

- [ ] **Step 3: Implement the minimal isolated CLI and validator**

  Configure CPU determinism; authenticate the protocol plus exact reviewed
  helper/CLI/test worktree and Git bytes; run the exact matrix once; validate a
  freshly reconstructed payload; serialize strict JSON in registered insertion
  order; and implement only the protocol's `xb`/file-fsync/hard-link/directory-
  fsync/unlink/directory-fsync protocol with owned-inode rollback. Never rename,
  replace, overwrite, or clean a foreign path. Rerun Steps 1 and 2 and require
  PASS.

- [ ] **Step 4: Run the complete local source gate and commit**

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py
  .venv/bin/ruff check scripts/rsta_normwise_adjoint.py scripts/calibrate_pass200_rsta_normwise_adjoint.py tests/test_rsta_normwise_adjoint.py
  .venv/bin/python -m py_compile scripts/rsta_normwise_adjoint.py scripts/calibrate_pass200_rsta_normwise_adjoint.py tests/test_rsta_normwise_adjoint.py
  git diff --check
  git add scripts/rsta_normwise_adjoint.py scripts/calibrate_pass200_rsta_normwise_adjoint.py tests/test_rsta_normwise_adjoint.py
  git commit -m "add candidate-free RSTA adjoint calibration"
  ```

---

### Task 5: Review and One Calibration Execution

**Files:**
- Produce: `reports/generated/pass200_rsta_receipt/${calibration_source_commit}-normwise-adjoint-calibration.json`

**Interfaces:**
- Consumes: reviewed Tasks 1–4 and the committed protocol.
- Produces: one complete calibration artifact and an authenticated pass/falsifier decision.

**Pre-fixture invocation erratum:** The first attempted Task-5 invocation used
reviewed calibration source
`0f5d1e2f626524f02c565a04f6fa0ae7127cd7e2` and the then-registered relative
plan construction:

```bash
calibration_source_commit=$(git rev-parse HEAD)
calibration_output="reports/generated/pass200_rsta_receipt/${calibration_source_commit}-normwise-adjoint-calibration.json"
test ! -e "$calibration_output"
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/calibrate_pass200_rsta_normwise_adjoint.py --output "$calibration_output"
```

It exited exactly `2` with `output must be absolute`. Source authentication
rejects a relative `--output` before `_configure_cpu`; therefore that attempt
performed no CPU configuration, fixture construction or execution, artifact or
temporary-file creation, and produced no calibration value. It does not consume
the one registered calibration execution. The defect was in this plan command,
not in the reviewed source or protocol. This erratum changes no protocol byte,
source byte, seed, dimension, scale, formula, threshold, fault, or result rule.

- [ ] **Step 1: Obtain independent adversarial source review**

  Give the reviewer the protocol and full calibration source/test diff. Require
  explicit findings on cast-before-product, exact norms, corner cases, factor
  two, threshold rationale, fixture bytes, cancellation construction, fault
  guarantees, action hashes, fresh graphs, forbidden candidate reachability,
  strict schema, and atomic output. Repair every Critical or Important finding
  test-first and repeat focused review before calibration.

- [ ] **Step 2: Freeze reviewed calibration source and run exactly once**

  Use a fresh clean checkout at the full executing commit that contains this
  plan erratum. Keep the reviewed calibration source fixed at
  `0f5d1e2f626524f02c565a04f6fa0ae7127cd7e2`. Authenticate that source as an
  ancestor and prove its exact helper, CLI, and test blobs equal both the
  executing-commit blobs and worktree bytes. Then derive the registered output
  from the absolute normalized repository root. `reports/generated` must
  already be a real non-symlink directory. Create only its
  `pass200_rsta_receipt` child if absent; if present, require that child to be a
  real non-symlink directory, and verify it again before destination and
  temporary-path checks. Run:

  ```bash
  repo_root=$(git rev-parse --show-toplevel)
  test "$(pwd -P)" = "$repo_root"
  test -z "$(git status --porcelain --untracked-files=all)"
  executing_plan_fix_commit=$(git rev-parse HEAD)
  calibration_source_commit=0f5d1e2f626524f02c565a04f6fa0ae7127cd7e2
  git merge-base --is-ancestor "$calibration_source_commit" "$executing_plan_fix_commit"
  for source_path in \
    scripts/rsta_normwise_adjoint.py \
    scripts/calibrate_pass200_rsta_normwise_adjoint.py \
    tests/test_rsta_normwise_adjoint.py
  do
    source_blob=$(git rev-parse "${calibration_source_commit}:${source_path}")
    test "$(git rev-parse "${executing_plan_fix_commit}:${source_path}")" = "$source_blob"
    test "$(git hash-object "${repo_root}/${source_path}")" = "$source_blob"
  done
  generated_directory="${repo_root}/reports/generated"
  test -d "$generated_directory" || exit 2
  test ! -L "$generated_directory" || exit 2
  test "$(cd "$generated_directory" && pwd -P)" = "$generated_directory" || exit 2
  calibration_directory="${generated_directory}/pass200_rsta_receipt"
  if test -e "$calibration_directory" || test -L "$calibration_directory"
  then
    test -d "$calibration_directory" || exit 2
    test ! -L "$calibration_directory" || exit 2
  else
    mkdir -- "$calibration_directory" || exit 2
  fi
  test -d "$calibration_directory" || exit 2
  test ! -L "$calibration_directory" || exit 2
  test "$(cd "$calibration_directory" && pwd -P)" = "$calibration_directory" || exit 2
  calibration_output="${repo_root}/reports/generated/pass200_rsta_receipt/${calibration_source_commit}-normwise-adjoint-calibration.json"
  test ! -e "$calibration_output"
  test ! -L "$calibration_output"
  (
    calibration_pid=$BASHPID
    calibration_temp="${repo_root}/reports/generated/pass200_rsta_receipt/.${calibration_source_commit}-normwise-adjoint-calibration.json.tmp.${calibration_pid}"
    test ! -e "$calibration_temp"
    test ! -L "$calibration_temp"
    exec env CUDA_VISIBLE_DEVICES='' "${repo_root}/.venv/bin/python" \
      "${repo_root}/scripts/calibrate_pass200_rsta_normwise_adjoint.py" \
      --output "$calibration_output"
  )
  calibration_exit=$?
  sha256sum "$calibration_output"
  ```

  Record `executing_plan_fix_commit`, the three equal blob IDs,
  `calibration_exit`, and the artifact SHA-256. The subshell `exec` preserves
  `BASHPID` as the Python PID, so the absent path checked immediately before
  `exec` is the exact protocol-registered temporary path. Do not rerun to seek
  a different value.

- [ ] **Step 3: Independently validate and apply the frozen decision**

  Validate protocol binding, exact execution/source provenance and ancestry,
  helper/CLI/test worktree/Git-blob hashes, exact schema, all hashes, six correct fixtures,
  every rebuild/order/sign control, three registered faults, thresholds, and
  recomputed derived values. If validation fails, preserve the external
  execution log, commit no invalid artifact, keep RSTA blocked, and stop. If the
  artifact is structurally valid with `all_passed=false`, retain it, declare the
  calibration falsified, and proceed only to its durable commit in Step 4. Do
  not change a constant.

- [ ] **Step 4: Commit every complete valid result, then branch only on its verdict**

  For either `all_passed=true` or `all_passed=false`, commit the one complete,
  independently validated artifact unchanged:

  ```bash
  git add "$calibration_output"
  git commit -m "record RSTA normwise adjoint calibration"
  ```

  Record the artifact SHA-256 and result commit. If `all_passed=false`, stop the
  plan before Task 6. Only `all_passed=true` permits Task 6.

---

### Task 6: Prospective Amendment After Calibration

**Files:**
- Create: `docs/pass200_rsta_normwise_adjoint_amendment_2026-08-09.md`

**Interfaces:**
- Consumes: this protocol and the passing Task 5 artifact, each by path, SHA-256, and commit.
- Produces: the only authority for later production changes.

- [ ] **Step 1: Author the amendment without production edits**

  Record the H8 disclosure verbatim, the calibration source/result provenance,
  every calibration value, independent validation, exact added production
  fields, retained legacy object, action hashes, fresh-graph/order/sign controls,
  `beta_norm <= 5e-4`, all-seed candidate-free prefix, and failure/no-output
  rules. State that the new gate is prospective and cannot repair H8.

- [ ] **Step 2: Self-review, independently review, and commit only the amendment**

  Scan for placeholders, inconsistent formulas, omitted calibration failures,
  threshold drift, candidate reachability, and ambiguous schemas. Obtain
  independent approval, repair findings, run `git diff --check`, and commit only
  the amendment with message `amend RSTA normwise adjoint integrity`.

---

### Task 7: Production TDD and Source Review

**Files:**
- Modify: `scripts/diagnose_pass200_rsta_stage_a.py`
- Modify: `tests/test_diagnose_pass200_rsta_stage_a.py`
- Reuse: `scripts/rsta_normwise_adjoint.py`

**Interfaces:**
- Extends: `adjoint_integrity_audit(...)` with the amendment's exact normwise/action evidence while retaining every legacy field.
- Preserves: frozen PCG64 directions, FP32 model/JVP/VJP, B=180, seed order `0,1,2,3`, provenance, and candidate-free/scientific integrity prefix.

- [ ] **Step 1: Write arithmetic/schema REDs before integration**

  Add independent FP64-reference tests for norms, factor two, corners,
  cancellation, exact action hashes, retained legacy fields, and every recursive
  payload mutation. Require the production helper to equal the reviewed
  calibration helper and reject any changed direction bytes/order.

- [ ] **Step 2: Implement only production metric composition and verify GREEN**

  Reuse the reviewed pure helper; do not duplicate formulas. Preserve legacy
  scalar computation and append only the exact amendment fields. Rerun Step 1.

- [ ] **Step 3: Write controller/reproducibility REDs before behavior changes**

  Require fresh rebuild and reversed-action-order hashes for all seeds before
  scoring. Parameterize later-seed zero-Jacobian, repeatability, adjoint,
  action-hash, sign, and rotation failures; require zero scoring, decision,
  bootstrap, payload, destination, or temp activity. Require complete four-seed
  candidate-free finite-failure evidence and no candidate calls.

- [ ] **Step 4: Implement the amended gate and verify all REDs GREEN**

  Gate each seed on exact structural/reproducibility controls and
  `beta_norm <= 5e-4`; retain legacy pass/fail only as evidence. Continue all
  four seeds only for finite candidate-free tolerance failures; preserve
  structural fail-fast and the scientific no-output prefix.

- [ ] **Step 5: Write manifest/source-validator REDs before changing validators**

  In `tests/test_diagnose_pass200_rsta_stage_a.py`, build an in-memory future
  manifest fixture containing exact protocol, passing calibration result, and
  normwise amendment objects with only `path`, `sha256`, and `commit`; the exact
  reviewed-source revision; and the complete exact source-file mapping including
  the shared helper. Recursively remove/add/mutate every new leaf, mutate the
  result verdict to false, alter Git blob/worktree bytes and ancestry, and
  require rejection before artifact/model access. Require byte-semantic equality
  of every H8 receipt, historical, artifact, seed, base, and prior-amendment
  domain.

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'normwise and (manifest or source or provenance)'
  ```

  Expected: FAIL because production validators do not recognize or authenticate
  the three new authorities or helper source file.

- [ ] **Step 6: Implement only manifest/source validation and verify GREEN**

  Add exact constants and strict recursive validators for the protocol, passing
  calibration artifact, amendment, reviewed source revision/files, Git
  blob/worktree equality, and ancestry. Do not edit the real manifest. Rerun
  Step 5 plus all Task 7 tests and require PASS.

- [ ] **Step 7: Run assurance, commit, and obtain independent review**

  ```bash
  .venv/bin/pytest -q tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/ruff check scripts/rsta_normwise_adjoint.py scripts/calibrate_pass200_rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/python -m py_compile scripts/rsta_normwise_adjoint.py scripts/calibrate_pass200_rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  git diff --check
  git add scripts/rsta_normwise_adjoint.py scripts/diagnose_pass200_rsta_stage_a.py tests/test_rsta_normwise_adjoint.py tests/test_diagnose_pass200_rsta_stage_a.py
  git commit -m "implement RSTA normwise adjoint integrity"
  ```

  Independently review the complete production diff. Repair every Critical or
  Important finding with a behavior-specific RED, minimal GREEN, full affected
  gate, and focused re-review. Define `reviewed_source_commit=$(git rev-parse HEAD)`
  only after approval.

---

### Task 8: Manifest Refreeze and Candidate-Free DGX Audit

**Files:**
- Modify: `docs/pass200_rsta_receipt_stage_a_manifest.json`
- Produce on DGX: `reports/generated/pass200_rsta_receipt/${handoff_commit}-normwise-adjoint-integrity-all-seeds.json`

**Interfaces:**
- Consumes: passing calibration, committed amendment, and reviewed Task 7 source.
- Produces: one manifest-only handoff and one candidate-free real-data audit.

- [ ] **Step 1: Modify only the manifest using already-GREEN validators**

  Modify only `docs/pass200_rsta_receipt_stage_a_manifest.json`: add the exact
  already-validated protocol, passing calibration result, and normwise amendment
  path/SHA/commit objects; set the exact reviewed source revision and file
  hashes; and preserve every H8 receipt, historical, artifact, seed, base, and
  prior-amendment domain byte-semantically. Run the already-GREEN Task 7
  production validators and focused provenance tests against the real manifest.
  If they fail, do not edit source or tests in Task 8: return to Task 7, create a
  newly reviewed source commit, and restart the manifest-only handoff.

- [ ] **Step 2: Commit the manifest-only handoff**

  Confirm `git diff --name-only` lists only the manifest. Run the already-GREEN
  focused provenance tests, the full two test files, Ruff, py_compile, and
  `git diff --check`. Commit only
  `docs/pass200_rsta_receipt_stage_a_manifest.json` with message
  `refreeze RSTA normwise adjoint handoff`, then set
  `handoff_commit=$(git rev-parse HEAD)`. Verify the handoff changes no source,
  test, protocol, amendment, calibration artifact, or result file.

- [ ] **Step 3: Prepare and authenticate an isolated DGX checkout**

  Transfer an exact bundle, verify its SHA at both ends, use a new detached clean
  checkout at `$handoff_commit`, authenticate all Git blobs/worktree bytes and
  bindings, confirm no duplicate process, and export
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization.

- [ ] **Step 4: Run only the candidate-free all-seed audit**

  Use a unique absent output and the amended candidate-free mode. Collect the
  original exit code and output SHA, then independently validate provenance,
  legacy and normwise scalars, action hashes, all controls, exact seeds
  `0,1,2,3`, and `all_passed`. Do not inspect or compute candidate fields.

  If any structural, reproducibility, or normwise condition fails, stop with
  RSTA blocked and run no scientific command. Scientific execution requires a
  separately confirmed green candidate-free audit under this exact handoff and
  the existing authorization process; it is not part of calibration.
