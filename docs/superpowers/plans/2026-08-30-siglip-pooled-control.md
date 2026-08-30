# SigLIP-so400m Pooled Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe, GB10-bounded, full-backbone SigLIP-so400m pooled Proxy Anchor control whose exact logical-batch gradients are preserved by two-pass recomputation.

**Architecture:** A small library module owns frozen authority, split validation, the pooled head, the exact replay backward, retrieval metrics, and receipt aggregation. A standalone script owns model/dataset loading, the memory smoke, training, and create-new publication. Tests first mutation-lock the mathematical kernel and evidence boundary without requiring a GPU.

**Tech Stack:** Python 3.12, PyTorch, Transformers, Hugging Face Datasets, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-08-30-siglip-pooled-control-design.md`

## Global Constraints

- Operate only in `/home/rb/worktrees/sfora-emafactorial`.
- Never load Cars classes `98..195`.
- Optimization uses `0..48`; clean validation uses `49..81`; descriptive evidence uses `82..97`.
- Preserve one pooled 512-dimensional descriptor and exact Proxy Anchor `alpha=32`, `delta=0.1`.
- Ordinary microbatch loss accumulation is forbidden.
- Scientific execution requires a passing three-step GB10 smoke with the
  conservative sum of process RSS and CUDA peak reserved memory below 96 GiB,
  plus the registered pressure, swap, determinism, and throughput gates.
- All scientific receipts are canonical, newline-terminated, and `claim_eligible=false`.

---

### Task 1: Frozen control contract and split authority

**Files:**
- Create: `src/sfora/siglip_proxy_control.py`
- Create: `tests/test_siglip_proxy_control.py`

**Interfaces:**
- Produces: `SiglipProxyControlConfig`, `validate_control_partition`, and `ControlSplit`.

- [ ] **Step 1: Write failing tests** that construct literal label tensors and require the imported `F1_TRAIN_CLASSES`, `F1_VALIDATION_CLASSES`, and `SUBSTRATE_F0_CLASSES` roles; reject missing/overlapping/out-of-range classes, optimization classes with fewer than four examples, and clean/burned classes with fewer than two; and validate every frozen config field including `S=floor(N/120)`.
- [ ] **Step 2: Run** `uv run --offline --locked pytest tests/test_siglip_proxy_control.py -q` and confirm import failure for `sfora.siglip_proxy_control`.
- [ ] **Step 3: Implement** frozen dataclasses and strict concrete-type/value validation, including pinned model/dataset revisions and seeds `(17,29,43)`.
- [ ] **Step 4: Rerun the focused test** and require green.
- [ ] **Step 5: Commit** only the module and focused test.

### Task 2: Exact recomputed Proxy Anchor backward

**Files:**
- Modify: `src/sfora/siglip_proxy_control.py`
- Modify: `tests/test_siglip_proxy_control.py`

**Interfaces:**
- Produces: `PooledProxyAnchorModel` and `recomputed_proxy_anchor_backward`.
- Consumes: `proxy_anchor_loss` from `sfora.token_set_proxy_anchor`.

- [ ] **Step 1: Write a failing differential test** using a deterministic two-layer encoder, projection, and proxies. Derive one full logical-batch update and one replay update from cloned state; assert identical loss, score gradients, every parameter gradient, and updated tensors within `rtol=1e-6, atol=1e-7` for microbatches `1,2,4`.
- [ ] **Step 2: Run the single node** and confirm failure at the missing replay API.
- [ ] **Step 3: Implement the minimum two-pass operator**: materialize full detached fp32 scores, differentiate the exact full loss with respect to scores, replay chunks, require maximum score disagreement at most `2e-5`, and inject score-gradient slices. Reject empty/misaligned batches, active nonzero module or float-attribute dropout, training batch normalization, nonfinite values, microbatch sizes outside `1..logical_batch`, missing tower gradients, and a microbatch that does not divide the logical batch.
- [ ] **Step 4: Add mutation tests** proving independent microbatch losses, rescaled slices, reordered labels, and detached proxies fail the differential oracle.
- [ ] **Step 5: Run the focused nodes** and require green.
- [ ] **Step 6: Commit** the exact replay slice.

### Task 3: Retrieval and transfer evidence

**Files:**
- Modify: `src/sfora/siglip_proxy_control.py`
- Modify: `tests/test_siglip_proxy_control.py`

**Interfaces:**
- Produces: `nearest_class_margins` and `summarize_control_seeds`.
- Consumes: `score_frozen_substrate` from `sfora.substrate_screen` as the sole
  Recall@1 implementation.

- [ ] **Step 1: Write failing literal-matrix tests** for strict per-band galleries, nearest-positive/negative/margin values, initial-to-final clean Recall@1 change, exact three-seed cardinality, and undefined memorization-to-transfer ratios when training margin improvement is nonpositive. Patch/spy on `score_frozen_substrate` and require the production path to call it directly for Recall@1, preserving its lowest-position ties and self masking.
- [ ] **Step 2: Run the focused nodes** and verify missing-function failures.
- [ ] **Step 3: Implement only the blockwise fp32 positive/negative margin computation** without a full quadratic allocation and with strict finite/unit-normalization guards; delegate Recall@1 directly to `score_frozen_substrate`.
- [ ] **Step 4: Rerun the focused file** and require green.
- [ ] **Step 5: Commit** the evidence computations.

### Task 4: Authenticated local runner and smoke

**Files:**
- Create: `scripts/run_siglip_proxy_control.py`
- Create: `tests/test_run_siglip_proxy_control.py`

**Interfaces:**
- Consumes: all Task 1–3 interfaces.
- Produces: `run_memory_smoke`, `run_control_seed`, `_canonical_bytes`, `_write_new`, and CLI `main`.

- [ ] **Step 1: Write failing tests** around the real CLI parser and a tiny injected encoder/dataset boundary. Require no test-class load, create-new output, exact receipt schema/environment bindings, separate raw and seed-projected initial references, identical initial/final eval precision, no intermediate evaluation, fixed augmentation, `S=floor(N/120)`, exact stateless per-step 30-of-49 class permutations and persistent per-class example-cycle cursors across epochs/resume, schedule values at epochs 0/4/5/10/15/60, smoke ladder order and rung isolation, combined RSS+CUDA-reserved/pressure/throughput gates, rolling checkpoint retention/free-space preflight, exact optimizer-group partition, and no scientific output after authority/OOM/nonfinite failures.
- [ ] **Step 2: Run** `uv run --offline --locked pytest tests/test_run_siglip_proxy_control.py -q` and confirm failure at the absent script.
- [ ] **Step 3: Implement the runner** using `SiglipVisionModel` and the pinned `AutoImageProcessor`, nonreentrant gradient checkpointing with one fixed eager-attention path in both replay passes, bf16 CUDA tower execution with fp32 projection normalization/scores/loss, deterministic 30-class logical batches, the exact crop/flip/normalization policy, isolated three-step memory-smoke rungs over `120,60,40,30,24,20,15,12,10,8,6,5,4,3,2,1`, fully audited RSS/CUDA/pressure/24-hour projected-throughput gates, optimizer groups/schedule, raw plus projected initial and final-only embedding, and authenticated rolling latest/final epoch checkpoints/resume.
- [ ] **Step 4: Add receipt mutation tests** for model/dataset/source/split/config/smoke/seed/order/environment/step-count/checkpoint drift and canonical byte enforcement. Add a CUDA-marked test that requires complete non-`None` finite tower gradients with positive aggregate norm under the real bf16/nonreentrant/eager-attention path, requires replay score disagreement at most `2e-5`, and compares replay scores/gradients to an fp32 micro-fixture. A real-model drift failure is a terminal unsupported-control result; no precision or tolerance fallback is permitted.
- [ ] **Step 5: Run both focused test files** and require green.
- [ ] **Step 6: Commit** the runner slice.

### Task 5: Deployment wrapper and static assurance

**Files:**
- Create: `scripts/run_siglip_proxy_control_v1.sh`
- Create: `tests/test_run_siglip_proxy_control_v1.py`
- Modify: `docs/pass209_evidence_conditioned_search_2026-08-30.md`

**Interfaces:**
- Produces: one source-bound DGX command that runs smoke first and scientific seeds only after smoke success.

- [ ] **Step 1: Write a failing subprocess test** that executes the wrapper against temporary fake commands and asserts source revision/tree binding, offline model access, deterministic CUDA environment, smoke-before-clean-reload ordering, explicit output absence, authenticated complete-epoch resume only, and refusal to overwrite.
- [ ] **Step 2: Run the single test** and verify the missing wrapper failure.
- [ ] **Step 3: Implement the shell wrapper** with one original process, explicit PID/terminal handling, source manifest verification, and no adaptive restart.
- [ ] **Step 4: Update Pass209** with the exact frozen control authority and command, without claiming a result.
- [ ] **Step 5: Run focused tests, Ruff, strict mypy, `python3 -m py_compile`, and `git diff --check`**.
- [ ] **Step 6: Commit** the deployment slice.

### Task 6: Repository verification and delivery

**Files:**
- Modify only files required by verified repairs.

- [ ] **Step 1: Run focused control tests** and repair only demonstrated failures.
- [ ] **Step 2: Run dependency-complete Python discovery** with `uv run --offline --locked python -m unittest discover -s scripts -p 'test_*.py'` where applicable, followed by the repository's complete pytest gate once.
- [ ] **Step 3: Run Ruff and strict mypy** over every changed Python file, then `python3 -m py_compile` and `git diff --check`.
- [ ] **Step 4: Obtain a read-only cross-provider review** of the final diff and repair only verified Critical/Important findings through fresh RED/GREEN cycles.
- [ ] **Step 5: Commit any review repairs**, verify only intended tracked paths changed, and fast-forward push `devbox/emafactorial` without attribution trailers.
- [ ] **Step 6: Recheck DGX availability**. If online, run exactly the preregistered smoke; if offline, preserve the clean source authority and do not substitute another machine or recipe.
