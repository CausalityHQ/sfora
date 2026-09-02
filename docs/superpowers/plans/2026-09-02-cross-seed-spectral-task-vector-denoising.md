# Cross-seed Wiener and spectral task-vector denoising implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Preserve
> every RED before its production implementation.

**Goal:** Build and run a capability-separated diagnostic that folds three
authenticated SigLIP towers into fixed soup, tensorwise Wiener, and symmetric
spectral-denoised states and evaluates them against the best scalar fold.

**Architecture:** A pure package module owns deterministic tensor artifacts,
estimator math, candidate/result schemas, and classification. A preparer
releases outcome-free tower/head artifacts. A CPU-only builder seals candidates
before a separate burned-only evaluator receives images or scalar outcomes. A
controller projects exact capabilities into named offline units.

**Tech Stack:** Python 3.12, PyTorch CPU float64/fp32 and CUDA inference,
canonical JSON, raw tensor bytes, SHA-256, systemd user units, pytest, unittest,
Ruff, and mypy.

**Spec:**
`docs/superpowers/specs/2026-09-02-cross-seed-spectral-task-vector-denoising-design.md`

## Global constraints

- Work only in Sfora; do not modify Borsuk.
- Require exactly seeds `(17, 29, 43)`; there is no two-seed mode.
- Construction receives no images, labels, metrics, scalar outcomes, optimizer
  state, clean band, dataset path, or network capability.
- Evaluation receives only the content-addressed classes 82--97 artifact.
- Estimator authority is CPU float64; published tensors round once to CPU fp32.
- Builder/evaluator never overlap another scientific GPU process.
- All results are `claim_eligible=false`.

## File structure

- `src/sfora/cross_seed_denoising.py`: artifact codec, estimators, evidence,
  classification, canonical serialization.
- `tests/test_cross_seed_denoising.py`: pure mutation tests.
- `scripts/prepare_cross_seed_denoising_inputs.py`: authenticated extraction.
- `scripts/build_cross_seed_denoising.py`: CPU-only candidate construction.
- `scripts/diagnose_cross_seed_denoising.py`: burned-only evaluation.
- `scripts/run_cross_seed_denoising.py`: capability projection and lifecycle.
- Matching `scripts/test_*.py` files test each executable boundary.

---

### Task 1: Deterministic tensor artifacts

**Files:** create `src/sfora/cross_seed_denoising.py` and
`tests/test_cross_seed_denoising.py`.

**Interfaces:** `TensorRecord`, `TensorArtifactManifest`,
`write_tensor_artifact(root, state, *, role, bindings) -> bytes`, and
`read_tensor_artifact(root, manifest_bytes, *, role) -> OrderedDict[str,
torch.Tensor]`. The wire format is canonical newline JSON plus one raw
little-endian file per sorted tensor.

- [ ] **Step 1: Write missing-interface and mutation REDs.**

```python
def test_tensor_artifact_round_trip_is_deterministic(tmp_path):
    state = OrderedDict((
        ("tower.a", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
        ("tower.count", torch.tensor([3], dtype=torch.int64)),
    ))
    left = write_tensor_artifact(tmp_path / "left", state, role="tower", bindings=BINDINGS)
    right = write_tensor_artifact(tmp_path / "right", state, role="tower", bindings=BINDINGS)
    assert left == right
    assert_state_equal(read_tensor_artifact(tmp_path / "left", left, role="tower"), state)
```

Cover reordered names, bool-as-int, unknown dtype, shape/length/digest drift,
missing/extra raw file, nonfinite float, symlink/path traversal, wrong role,
binding drift, and non-floating inequality.

- [ ] **Step 2: Run RED.**

Run: `.venv/bin/pytest -q tests/test_cross_seed_denoising.py -k tensor_artifact`

Expected: collection fails only on the missing new interface.

- [ ] **Step 3: Implement the explicit codec.** Use a closed dtype map and:

```python
raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
digest = hashlib.sha256(raw).hexdigest()
```

Authenticate manifest/schema before opening files; require exact bytes/digest;
reconstruct from owned `bytearray`, reshape, clone, and recompute state digest.

- [ ] **Step 4: Run GREEN and Ruff.**

```bash
.venv/bin/pytest -q tests/test_cross_seed_denoising.py -k tensor_artifact
.venv/bin/ruff check src/sfora/cross_seed_denoising.py tests/test_cross_seed_denoising.py
```

- [ ] **Step 5: Commit.**

```bash
git add src/sfora/cross_seed_denoising.py tests/test_cross_seed_denoising.py
git commit -m "Add deterministic cross-seed tensor artifacts"
```

### Task 2: Fixed denoising estimators

**Files:** modify the Task 1 module and test.

**Interfaces:** `GroupEvidence`, `SpectralEvidence`, `CandidateStates`, and
`build_cross_seed_candidates(initial, endpoints) -> CandidateStates`.

- [ ] **Step 1: Write estimator REDs.** Prove:

```python
assert wiener_gain(0.0) == 0.0
assert wiener_gain(0.5) == 0.75
assert wiener_gain(1.0) == 1.0
```

Add analytic soup/gain tensors, cosine clipping, zero norm, one group per named
tensor, report-only `g_js`, non-floating equality, and all six seed
permutations. Spectral cases cover three symmetric pairwise contrasts,
`sqrt(3)` scaling, vector fallback, convolution flattening, edge equality,
tolerance clusters, near-edge failure, subnormal, and rectangular matrices.

- [ ] **Step 2: Run RED.**

Run: `.venv/bin/pytest -q tests/test_cross_seed_denoising.py -k 'wiener or spectral or candidate'`

- [ ] **Step 3: Implement sorted one-tensor float64 authority.** The edge is:

```python
contrasts = ((d17-d29)/SQRT2, (d17-d43)/SQRT2, (d29-d43)/SQRT2)
edge = max(torch.linalg.matrix_norm(c, ord=2) for c in contrasts) / math.sqrt(3.0)
tol = 64 * torch.finfo(torch.float64).eps * max(rows, cols) * max(sigma1, edge, 1.0)
```

Reject singular values within `tol` of the edge; otherwise decide whole
adjacent tolerance clusters. Release decomposition workspaces per tensor.

- [ ] **Step 4: Run GREEN twice with one CPU thread and compare artifact bytes.**

- [ ] **Step 5: Commit.**

```bash
git add src/sfora/cross_seed_denoising.py tests/test_cross_seed_denoising.py
git commit -m "Add fixed cross-seed denoising estimators"
```

### Task 3: Result authority

**Files:** modify the Task 1 module and test.

**Interfaces:** `CandidateEvaluation`, `HeadSwapEvaluation`,
`DenoisingDecision`, `classify_denoising_result(...)`, and
`canonical_denoising_result_bytes(...) -> bytes`.

- [ ] **Step 1: Write REDs** for raw correctness once/candidate; projected
  correctness per seed; count/ppm recomputation; per-seed McNemar only; best
  scalar order; six swaps; exact coadaptation flag; all quality/margin gates;
  fixed priority; every terminal/precedence; and type, NaN, cardinality, order,
  digest, stored aggregate/class, and claim-eligibility mutations.

- [ ] **Step 2: Run RED.**

Run: `.venv/bin/pytest -q tests/test_cross_seed_denoising.py -k 'evaluation or decision or canonical'`

- [ ] **Step 3: Implement recompute-or-reject logic.** Never trust stored
  aggregates. Recompute rows, paired evidence, swaps, gates, and class; emit
  sorted compact JSON plus LF; validate emitted bytes through the public parser.

- [ ] **Step 4: Run full module GREEN and strict mypy.**

```bash
.venv/bin/pytest -q tests/test_cross_seed_denoising.py
.venv/bin/mypy --strict src/sfora/cross_seed_denoising.py
```

- [ ] **Step 5: Commit.**

```bash
git add src/sfora/cross_seed_denoising.py tests/test_cross_seed_denoising.py
git commit -m "Add cross-seed denoising result authority"
```

### Task 4: Outcome-free checkpoint preparer

**Files:** create `scripts/prepare_cross_seed_denoising_inputs.py` and its
focused unittest.

**Interfaces:** consume exact result/checkpoint path, digest, length, seed, and
source authority for `(17,29,43)`; produce trained tower artifacts, head
artifacts, independent initial-tower digests, and an outcome-free manifest.

- [ ] **Step 1: Write REDs** for endpoint/optimizer leakage, wrong seed,
  mismatched initial tower, projection in tower, tower in head, digest drift,
  symlink, partial/existing output, and source drift. Assert outputs/errors carry
  no metric, correctness, label, image, clean-band, or scalar-alpha value.

- [ ] **Step 2: Run RED.**

Run: `.venv/bin/python -m unittest scripts.test_prepare_cross_seed_denoising_inputs`

- [ ] **Step 3: Implement strict extraction.** Reuse epoch-60 authority and
  initial reconstruction; authenticate first; normalize only `tower.*`,
  `projection.weight`, and `proxies`; require equal initial tower-only digests;
  publish via Task 1 with atomic directory rename.

- [ ] **Step 4: Run focused GREEN, Ruff, and strict mypy.**

```bash
.venv/bin/python -m unittest scripts.test_prepare_cross_seed_denoising_inputs
.venv/bin/ruff check scripts/prepare_cross_seed_denoising_inputs.py scripts/test_prepare_cross_seed_denoising_inputs.py
.venv/bin/mypy --strict scripts/prepare_cross_seed_denoising_inputs.py
```

- [ ] **Step 5: Commit preparer.**

```bash
git add scripts/prepare_cross_seed_denoising_inputs.py scripts/test_prepare_cross_seed_denoising_inputs.py
git commit -m "Add outcome-free cross-seed input preparation"
```

### Task 5: CPU-only candidate builder

**Files:** create `scripts/build_cross_seed_denoising.py` and focused unittest.

**Interfaces:** accept only three tower artifacts, authority manifest,
pretrained tower/spec authorities, output root, and explicit execute flag;
produce three sealed candidates and one canonical receipt.

- [ ] **Step 1: Write REDs** for strict CLI; forbidden checkpoint/result/
  optimizer/dataset/image/label/scalar/head/network/storage/GPU flags; wrong
  initial digest; missing seed; output preexistence; edge ambiguity;
  interruption; cleanup; progress; and largest-tensor preflight.

- [ ] **Step 2: Run RED.**

Run: `.venv/bin/python -m unittest scripts.test_build_cross_seed_denoising`

- [ ] **Step 3: Implement builder.** Authenticate every byte, bind CPU
  environment, call the pure builder, publish/reload candidates, then independently
  replay from reloaded inputs and require identical artifacts/evidence.

- [ ] **Step 4: Run focused GREEN, Ruff, and strict mypy.**

```bash
.venv/bin/python -m unittest scripts.test_build_cross_seed_denoising
.venv/bin/ruff check scripts/build_cross_seed_denoising.py scripts/test_build_cross_seed_denoising.py
.venv/bin/mypy --strict scripts/build_cross_seed_denoising.py
```

- [ ] **Step 5: Commit builder.**

```bash
git add scripts/build_cross_seed_denoising.py scripts/test_build_cross_seed_denoising.py
git commit -m "Add deterministic cross-seed candidate builder"
```

### Task 6: Burned-only evaluator

**Files:** create `scripts/diagnose_cross_seed_denoising.py` and focused
unittest.

**Interfaces:** consume sealed candidates/receipt, three trained tower/head
artifacts, final scalar five-row manifest, and existing burned manifest/root;
produce one Task 3 canonical result.

- [ ] **Step 1: Write REDs** for endpoint replay, best scalar selection,
  candidate raw-once behavior, raw correctness retention, three projected rows,
  six projection swaps, proxy inertness, `(similarity, source ordinal)` ties,
  one resident model, eval/no-grad, finite embeddings, state digests, and result
  replay. Prove no dataset, clean band, classes 49--81, network, storage, or raw
  checkpoint capability is accepted.

- [ ] **Step 2: Run RED.**

Run: `.venv/bin/python -m unittest scripts.test_diagnose_cross_seed_denoising`

- [ ] **Step 3: Implement the evaluator.** Extend the local model-band evidence
  with `raw_correctness`. Load one candidate tower at a time, score raw once,
  then score each exact projection. Load trained towers only for endpoint replay
  and ordered swaps. Keep proxies in full-state digest authority, not kNN.

- [ ] **Step 4: Run focused GREEN, Ruff, and strict mypy.**

```bash
.venv/bin/python -m unittest scripts.test_diagnose_cross_seed_denoising
.venv/bin/ruff check scripts/diagnose_cross_seed_denoising.py scripts/test_diagnose_cross_seed_denoising.py
.venv/bin/mypy --strict scripts/diagnose_cross_seed_denoising.py
```

- [ ] **Step 5: Commit evaluator.**

```bash
git add scripts/diagnose_cross_seed_denoising.py scripts/test_diagnose_cross_seed_denoising.py
git commit -m "Add burned-only cross-seed evaluator"
```

### Task 7: Capability-separated controller

**Files:** create `scripts/run_cross_seed_denoising.py` and focused unittest.

**Interfaces:** explicit `prepare`, `build`, `evaluate`, and `complete` phases;
reuse named user units, network denial, cgroup monitoring, and file logs from
`scripts/run_weight_space_transfer.py`.

- [ ] **Step 1: Write REDs** for seed order; clean committed source; host/spec/
  script identity; phase allowlists; read-only mounts; builder blindness to
  burned/scalar/head inputs; evaluator blindness to checkpoint/result/dataset;
  one active unit; exact cleanup; existing output; 110 GiB RSS, 96 GiB CUDA,
  pressure, swap, five-minute progress, six-hour projection, interruption, and
  canonical stop receipts.

- [ ] **Step 2: Run RED.**

Run: `.venv/bin/python -m unittest scripts.test_run_cross_seed_denoising`

- [ ] **Step 3: Implement phase projection.** Create a fresh explicit scratch
  namespace per phase, capture logs to files, deny `@network-io`, monitor the
  child unit cgroup, and remove only registered paths after exit. Seal candidate
  digests before admitting burned/scalar capabilities. Never auto-restart.

- [ ] **Step 4: Run focused GREEN, Ruff, and strict mypy.**

```bash
.venv/bin/python -m unittest scripts.test_run_cross_seed_denoising
.venv/bin/ruff check scripts/run_cross_seed_denoising.py scripts/test_run_cross_seed_denoising.py
.venv/bin/mypy --strict scripts/run_cross_seed_denoising.py
```

- [ ] **Step 5: Commit controller.**

```bash
git add scripts/run_cross_seed_denoising.py scripts/test_run_cross_seed_denoising.py
git commit -m "Add cross-seed denoising execution controller"
```

### Task 8: Synthetic end-to-end falsifier

**Files:** modify `tests/test_cross_seed_denoising.py`,
`scripts/test_prepare_cross_seed_denoising_inputs.py`,
`scripts/test_build_cross_seed_denoising.py`,
`scripts/test_diagnose_cross_seed_denoising.py`, and
`scripts/test_run_cross_seed_denoising.py`.

- [ ] **Step 1: Add one independent reduced fixture.** Use three small towers,
  independent head matrices, and deterministic scored vectors with no production
  outcomes. Exercise prepare -> build -> seal -> evaluate -> classify for every
  benefit/negative/numerical/authority/resource terminal.

- [ ] **Step 2: Add leakage and lifecycle attacks.** Expose scalar data to the
  builder, source dataset to evaluator, network socket, symlink, undeclared
  file, partial candidate, stale result, initial mismatch, and post-seal
  mutation. Require pre-score failure and no partial output.

- [ ] **Step 3: Run grouped evidence twice.**

```bash
.venv/bin/pytest -q tests/test_cross_seed_denoising.py
.venv/bin/python -m unittest \
  scripts.test_prepare_cross_seed_denoising_inputs \
  scripts.test_build_cross_seed_denoising \
  scripts.test_diagnose_cross_seed_denoising \
  scripts.test_run_cross_seed_denoising
```

Expected: both runs pass with identical canonical bytes.

- [ ] **Step 4: Commit integration evidence.**

```bash
git add tests/test_cross_seed_denoising.py \
  scripts/test_prepare_cross_seed_denoising_inputs.py \
  scripts/test_build_cross_seed_denoising.py \
  scripts/test_diagnose_cross_seed_denoising.py \
  scripts/test_run_cross_seed_denoising.py
git commit -m "Test cross-seed denoising end to end"
```

### Task 9: Assurance, independent review, and delivery

**Files:** all new/changed Sfora files plus this spec and plan.

- [ ] **Step 1: Run dependency-complete tests once.**

```bash
.venv/bin/pytest -q -p no:cacheprovider
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

- [ ] **Step 2: Run static assurance serially.**

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict \
  src/sfora/cross_seed_denoising.py \
  scripts/prepare_cross_seed_denoising_inputs.py \
  scripts/build_cross_seed_denoising.py \
  scripts/diagnose_cross_seed_denoising.py \
  scripts/run_cross_seed_denoising.py
.venv/bin/python -m py_compile \
  src/sfora/cross_seed_denoising.py \
  scripts/prepare_cross_seed_denoising_inputs.py \
  scripts/build_cross_seed_denoising.py \
  scripts/diagnose_cross_seed_denoising.py \
  scripts/run_cross_seed_denoising.py
git diff --check
```

- [ ] **Step 3: Obtain a read-only cross-provider diff review.** Ask for exact
  estimator, capability, lifecycle, result-recomputation, and test defects.
  Apply only verified repairs, rerun the narrow layer, then the full chain once.

- [ ] **Step 4: Commit and push all verified files.**

```bash
git add docs/superpowers/specs/2026-09-02-cross-seed-spectral-task-vector-denoising-design.md \
  docs/superpowers/plans/2026-09-02-cross-seed-spectral-task-vector-denoising.md \
  src/sfora/cross_seed_denoising.py tests/test_cross_seed_denoising.py \
  scripts/prepare_cross_seed_denoising_inputs.py \
  scripts/test_prepare_cross_seed_denoising_inputs.py \
  scripts/build_cross_seed_denoising.py scripts/test_build_cross_seed_denoising.py \
  scripts/diagnose_cross_seed_denoising.py scripts/test_diagnose_cross_seed_denoising.py \
  scripts/run_cross_seed_denoising.py scripts/test_run_cross_seed_denoising.py
git commit -m "Add cross-seed task-vector denoising diagnostic"
git push origin HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse @{upstream})"
test -z "$(git status --porcelain)"
```

### Task 10: Separate serialized DGX execution

**Files:** no repository edits during science.

- [ ] **Step 1: Verify lane clearance.** Require seed43 terminal, no scientific
  GPU PID, clean reviewed source/upstream equality, and complete authenticated
  final three-seed scalar result. Do not request approval and do not overlap.

- [ ] **Step 2: Run registered preflights.** Time the largest-tensor builder
  operation and one projected seed/candidate evaluation. Stop if projected total
  exceeds six hours or any resource/progress gate.

- [ ] **Step 3: Run one original diagnostic.** Execute prepare, build, and
  evaluate once through the committed controller. Preserve canonical candidates,
  result/SHA, quality/margins, swaps, resources, and cleanup/PID evidence. Never
  restart after a terminal.

- [ ] **Step 4: Interpret without moving gates.** A pass funds a fresh benchmark
  plan. A negative ends these exact post-hoc estimators. If no candidate reaches
  95%, move to a prospective training-objective/data-breadth design instead of
  adding a post-result coefficient or threshold ladder.
