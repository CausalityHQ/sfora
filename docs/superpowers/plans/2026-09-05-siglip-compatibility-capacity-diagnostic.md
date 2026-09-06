# SigLIP Compatibility Capacity Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a claim-ineligible descriptor-only diagnostic that separates post-hoc mapping coverage failure from loss of teacher-compatible information.

**Architecture:** A focused library owns deterministic folds, affine/residual fitting, retrieval evidence, selection, and canonical validation. A thin probe reuses the existing authenticated SigLIP descriptor extractor, emits one sealed descriptor artifact, and runs all registered arms without external access. A new deployment wrapper reuses the proven Landlock/seccomp and pressure envelope.

**Tech Stack:** Python 3.13, PyTorch, safetensors, pytest, Ruff, mypy, Bash, Landlock/seccomp.

**Spec:** `docs/superpowers/specs/2026-09-05-siglip-compatibility-capacity-diagnostic-design.md`

## Global Constraints

- Never read labels, names, images, descriptors, or metadata for classes 49 through 81.
- Use the exact existing checkpoint, spatial artifact, manifest, preprocessing, IDs, and 39/10 class split.
- Development rows select no fitting-only arm or hyperparameter; the oracle is diagnostic-only and never deployable.
- Preserve both `97%`/`0.95` cross-direction gates, the `99%`/`0.96` self floors, and plain cosine as the promotion metric.
- Emit only finite canonical newline JSON with `claim_eligible=false`; bind every input and artifact SHA-256.
- Keep the existing 48 GiB RSS/GPU caps, PSI/swap/progress stops, offline sandbox, and one-process/no-restart discipline.

---

### Task 1: Deterministic folds and affine maps

**Files:**
- Create: `src/sfora/siglip_compatibility_capacity.py`
- Create: `tests/test_siglip_compatibility_capacity.py`

**Interfaces:**
- Produces: `compatibility_folds(labels: tuple[int, ...]) -> tuple[tuple[int, ...], ...]`
- Produces: `fit_centered_similarity(student: Tensor, teacher: Tensor) -> AffineMap`
- Produces: `fit_regularized_affine(student: Tensor, teacher: Tensor, regularization: float) -> AffineMap`
- Produces: frozen `AffineMap(weight: Tensor, bias: Tensor)` with `apply(descriptors: Tensor) -> Tensor`

- [ ] **Step 1: Write failing authority and numerical tests**

Test exact 13/13/13 SHA-ranked folds, label/type/cardinality rejection, FP64-only fitting, finite normalized FP32 output, deterministic centered Procrustes, the literal ridge set `{1e-4, 1e-2, 1}`, identity regularization, and singular/NaN rejection. Mutation-lock bias and normalization.

- [ ] **Step 2: Run the focused RED**

Run: `uv run pytest -q tests/test_siglip_compatibility_capacity.py -k 'fold or affine or similarity'`

Expected: import failure for the new library API.

- [ ] **Step 3: Implement the minimal scalar authority**

Use `torch.linalg.svd` for centered Procrustes and the augmented normal equation with `torch.linalg.solve` for ridge. Construct the regularizer with an identity target for `W` and zero target for `b`; normalize only after applying the map.

- [ ] **Step 4: Run focused GREEN and static checks**

Run: `uv run pytest -q tests/test_siglip_compatibility_capacity.py -k 'fold or affine or similarity' && uv run ruff check src/sfora/siglip_compatibility_capacity.py tests/test_siglip_compatibility_capacity.py && git diff --check`

- [ ] **Step 5: Commit the slice**

```bash
git add src/sfora/siglip_compatibility_capacity.py tests/test_siglip_compatibility_capacity.py
git commit -m "Add SigLIP compatibility affine authority"
```

### Task 2: Teacher-anchored residual fitting

**Files:**
- Modify: `src/sfora/siglip_compatibility_capacity.py`
- Modify: `tests/test_siglip_compatibility_capacity.py`

**Interfaces:**
- Produces: `CompatibilityResidual(torch.nn.Module)` with rank exactly 32 and zero-residual initialization
- Produces: `fit_teacher_anchored_residual(student, teacher, ids, *, relational: bool, seed: int) -> ResidualFit`
- `ResidualFit` contains the frozen state dictionary, four 2,000-value finite
  loss trajectories, training device, Torch version, and fixed AdamW
  hyperparameters.

- [ ] **Step 1: Write failing architecture, loss, and determinism tests**

Use a small synthetic bank. Assert exact parameter shapes/count, zero initial
residual, fixed seed, the SHA-ranked 256-anchor set, cyclic 256-row blocks,
2,000 updates, identical-ID masking, equal normalized loss weights, and
distinct paired-only versus relational objectives. Reject nonfinite input,
duplicate IDs, wrong dimensions, and missing trajectories.

- [ ] **Step 2: Run the focused RED**

Run: `uv run pytest -q tests/test_siglip_compatibility_capacity.py -k 'residual or teacher_anchored'`

Expected: missing residual API.

- [ ] **Step 3: Implement the fixed training loop**

Use FP64 loss computation, AdamW at `1e-3`, `weight_decay=0`, exactly 2,000
updates, all 256 fixed anchors per update, deterministic cyclic query blocks,
and no early stopping. Normalize every score loss by the number of unmasked
entries. Return CPU tensors only.

- [ ] **Step 4: Run focused GREEN and static checks**

Run: `uv run pytest -q tests/test_siglip_compatibility_capacity.py -k 'residual or teacher_anchored' && uv run ruff check src/sfora/siglip_compatibility_capacity.py tests/test_siglip_compatibility_capacity.py && git diff --check`

- [ ] **Step 5: Commit the slice**

```bash
git add src/sfora/siglip_compatibility_capacity.py tests/test_siglip_compatibility_capacity.py
git commit -m "Add teacher-anchored compatibility residual"
```

### Task 3: Retrieval evidence, CSLS, selection, and decisions

**Files:**
- Modify: `src/sfora/siglip_compatibility_capacity.py`
- Modify: `tests/test_siglip_compatibility_capacity.py`

**Interfaces:**
- Produces: `compatibility_retrieval_evidence(...) -> CompatibilityRetrievalEvidence`
- Produces: `csls_scores(query: Tensor, gallery: Tensor, *, neighbors: int = 10) -> Tensor`
- Produces: `select_compatibility_finalist(fold_results: Mapping[str, ...]) -> Finalist`
- Produces: `build_compatibility_capacity_result(...) -> bytes`
- Produces: `validate_compatibility_capacity_result_bytes(raw: bytes) -> dict[str, object]`

- [ ] **Step 1: Write failing evidence and decision mutation tables**

Cover exact query IDs/labels, micro and class-macro R@1/mAP@R, exact hit/AP
vectors, paired cosine, cross-score MSE, top-10 overlap, hubs, identical-ID-
excluded CSLS `k=10` with ordered IDs/labels/per-query hit evidence, three-fold
worst-direction selection, stronger-lambda and
affine tie breaks, development non-selection, disjoint-panel oracle half
reversal, all four terminal classes, independent `hubness-present`, concrete
types, nonfinite values, schema keys, complete provenance bindings, and
canonical bytes. Mutation-lock that every fold aggregate and control aggregate
is reconstructed from its serialized per-query evidence.

- [ ] **Step 2: Run the focused RED**

Run: `uv run pytest -q tests/test_siglip_compatibility_capacity.py -k 'retrieval or csls or finalist or result'`

Expected: missing evidence/result API.

- [ ] **Step 3: Implement exact recomputation**

Reuse identical-ID exclusion and stable score ordering from
`spatial_retrieval_evidence`, add label-group macro aggregation, and keep plain
cosine and CSLS cells separate. Score each oracle validation half only against
its own held-out gallery before combining evidence. Record non-selectable
centered-similarity and paired-only residual controls, fitting identity evidence,
the paired-cosine gap, all residual loss trajectories, and exact parameter
counts. The validator reconstructs every aggregate and decision from per-query
evidence. Before fitting, reject unless every fitting-fold training complement
and both oracle training halves independently contain at least 256 rows.

- [ ] **Step 4: Run complete library GREEN and static checks**

Run: `uv run pytest -q tests/test_siglip_compatibility_capacity.py && uv run ruff check src/sfora/siglip_compatibility_capacity.py tests/test_siglip_compatibility_capacity.py && python3 -m py_compile src/sfora/siglip_compatibility_capacity.py tests/test_siglip_compatibility_capacity.py && git diff --check`

- [ ] **Step 5: Commit the slice**

```bash
git add src/sfora/siglip_compatibility_capacity.py tests/test_siglip_compatibility_capacity.py
git commit -m "Add compatibility capacity evidence"
```

### Task 4: Authenticated probe and sealed descriptor artifact

**Files:**
- Create: `scripts/probe_siglip_compatibility_capacity.py`
- Create: `tests/test_probe_siglip_compatibility_capacity.py`

**Interfaces:**
- Consumes the existing control binding, optimization manifest, checkpoint, spatial artifact, and materialized registered image directory.
- Produces one descriptor safetensors artifact containing fitting/development
  student and teacher descriptors plus ID/label digests and exact checkpoint,
  control-binding, optimization-manifest, spatial-tail, ordered-image-byte, and
  preprocessing authority, and one canonical result that repeats and binds
  those authorities.

- [ ] **Step 1: Write failing parser, authority, and no-leak tests**

Require absolute local paths and exact SHA-256 flags, `--execute-capacity-diagnostic`, offline mode, exactly the manifest rows for labels 0..48, exact class split, one descriptor pass, atomic outputs, artifact reload equality, and rejection of dataset roots, network/storage flags, external labels/names, symlinks, extra files, and duplicate runs.

- [ ] **Step 2: Run the focused RED**

Run: `uv run pytest -q tests/test_probe_siglip_compatibility_capacity.py`

Expected: missing probe module.

- [ ] **Step 3: Implement the thin orchestration**

Reuse `stream_alignment_descriptor_pairs` and the existing frozen tail/readout loading. Seal descriptors before fitting, reload them, run all registered fitting folds and the diagnostic oracle, validate canonical bytes, and use `os.replace` for both outputs.

- [ ] **Step 4: Run probe and dependency GREEN**

Run: `uv run pytest -q tests/test_probe_siglip_compatibility_capacity.py tests/test_siglip_compatibility_capacity.py tests/test_probe_siglip_gallery_compatibility_alignment.py tests/test_siglip_gallery_compatibility_alignment.py`

- [ ] **Step 5: Commit the slice**

```bash
git add scripts/probe_siglip_compatibility_capacity.py tests/test_probe_siglip_compatibility_capacity.py
git commit -m "Add SigLIP compatibility capacity probe"
```

### Task 5: DGX sandbox deployment

**Files:**
- Create: `scripts/deploy_siglip_compatibility_capacity_v1.sh`
- Create: `tests/test_deploy_siglip_compatibility_capacity.py`
- Reuse: `scripts/landlock_exec.c`

**Interfaces:**
- Produces local `/tmp/sfora-compatibility-capacity-$REVISION.json` and `.safetensors` only after remote validation.

- [ ] **Step 1: Write failing deployment contract tests**

Require clean tracked Sfora sources, exact revision bundle, remote process/GPU clearance, authenticated inputs, registered-image-only materialization, ABI-4 Landlock, socket seccomp, private HOME/TMP, no external label/name path, 48 GiB RSS/GPU guards, PSI `0.79` immediate/`0.50` sustained, 256 MiB swap delta, 300-second progress, 7,200-second wall, invocation-owned clone cleanup on failure, output preservation on success, PID clearance, and exact result/artifact hashes.

- [ ] **Step 2: Run the focused RED**

Run: `uv run pytest -q tests/test_deploy_siglip_compatibility_capacity.py`

Expected: deployment file missing.

- [ ] **Step 3: Implement by narrowing the proven alignment wrapper**

Copy the security/resource structure from `deploy_siglip_gallery_compatibility_alignment_v1.sh`, change only source inventory, output names, and probe arguments, and retain the repaired ownership guards and 48 GiB cap.

- [ ] **Step 4: Run deployment/static GREEN**

Run: `uv run pytest -q tests/test_deploy_siglip_compatibility_capacity.py tests/test_landlock_exec.py && shellcheck scripts/deploy_siglip_compatibility_capacity_v1.sh && cc -std=c11 -Wall -Wextra -Werror -fsyntax-only scripts/landlock_exec.c && git diff --check`

- [ ] **Step 5: Commit the slice**

```bash
git add scripts/deploy_siglip_compatibility_capacity_v1.sh tests/test_deploy_siglip_compatibility_capacity.py
git commit -m "Add guarded compatibility capacity deployment"
```

### Task 6: Assurance, single execution, and evidence ledger

**Files:**
- Create after execution: `docs/siglip_compatibility_capacity_result_2026-09-05.md`

**Interfaces:**
- Consumes the committed probe/deployment and produces one verified claim-ineligible scientific result.

- [ ] **Step 1: Run complete local assurance once**

Run: `uv run pytest -q tests/test_*siglip*.py`

Then run Ruff on the four new Python files, pycompile them, shellcheck the deployment, compile `landlock_exec.c`, and run `git diff --check`. Stop at the first failure and rerun only that layer after repair; once repaired, run the complete assurance command one final time.

- [ ] **Step 2: Obtain two independent release reviews**

Start one cold Codex Astra review and one cold Fable review against the exact committed diff and spec. Require both to check leakage, selection, numerical authority, classification, sandboxing, and whether the oracle can influence the deployable finalist. Reconcile and locally verify every actionable finding.

- [ ] **Step 3: Commit verified repairs and assert a clean scientific source**

Commit only diagnostic files with configured operator identity and no attribution trailers. Assert `git diff --quiet HEAD -- src/sfora scripts` with unrelated Qwen exclusions and record the full revision.

- [ ] **Step 4: Execute exactly one monitored DGX run**

Run: `scripts/deploy_siglip_compatibility_capacity_v1.sh`

Retain the original session ID, poll it at no more than 55-second waits, send concise liveness updates, and never launch a duplicate. Stop without restart on any authority/resource terminal. On completion, independently validate the canonical result, artifact hashes, classification, process/GPU clearance, and remote scratch/source lifecycle.

- [ ] **Step 5: Record and commit the evidence**

Write the exact arm metrics, fold selection, oracle branch, CSLS effect, resources, hashes, limitations, and next branch to `docs/siglip_compatibility_capacity_result_2026-09-05.md`. Run `git diff --check`, commit only that document, and notify the operator through `devbox-tell` with the result and next action.
