# SigLIP Spatial Tail Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a two-arm causal probe of whether compact cross-token interaction can recover transferable SigLIP depth-27 geometry from frozen depth-18 tokens.

**Architecture:** A pure evidence module validates the fixed split, metrics, gap closure, and canonical decision. A local-only runner extracts authenticated token fields, trains a tokenwise control and latent-interaction treatment under identical supervision, seals/reloads both, and evaluates only the internal development fold. A deployment wrapper stages local inputs to DGX, monitors the sole process, and retains canonical evidence.

**Tech Stack:** Python 3.12, PyTorch, Transformers, safetensors, pytest, Ruff, mypy, ShellCheck.

**Spec:** `docs/superpowers/specs/2026-09-05-siglip-spatial-tail-recovery-design.md`

## Global Constraints

- Do not modify the sealed attention-readout result or its formats.
- Use labels 0..48 only; the external labels 49..81 are forbidden.
- Both arms use identical token/descriptor targets, index stream, update count, optimizer family, and frozen teacher readout.
- One original DGX process only; no restart after a scientific terminal without diagnosis and a separately recorded repair.
- Canonical results are claim-ineligible and cannot assert SOTA or deployed latency.
- Preserve unrelated Qwen worktree changes and stage only the exact files in this plan.

---

### Task 1: Split, evidence, and canonical decision

**Files:**
- Create: `src/sfora/siglip_spatial_tail_recovery.py`
- Create: `tests/test_siglip_spatial_tail_recovery.py`

**Interfaces:**
- Produces: `spatial_tail_class_split(labels: tuple[int,...]) -> tuple[frozenset[int],frozenset[int]]`.
- Produces: `retrieval_evidence(descriptors, labels) -> SpatialRetrievalEvidence`.
- Produces: `build_spatial_tail_result(...) -> bytes` and `validate_spatial_tail_result_bytes(raw) -> dict[str,object]`.

- [ ] **Step 1: Write split and mutation RED tests.** Lock all 49 labels, ten deterministic development classes, disjointness, and independence from input ordering.
- [ ] **Step 2: Run `pytest -q tests/test_siglip_spatial_tail_recovery.py` and require missing-interface RED.**
- [ ] **Step 3: Implement the exact SHA-256 class split and strict concrete types.**
- [ ] **Step 4: Write retrieval/result RED tests.** Cover per-query hit/AP recomputation, baseline/control/treatment/teacher/cross evidence, positive teacher gaps, exact gap-closure arithmetic, treatment-over-control, classification precedence, 64-hex artifact identities, canonical sorted JSON plus one LF, and every key/type/cardinality/nonfinite mutation.
- [ ] **Step 5: Implement minimal evidence and canonical validation.** Reuse the self-contained ranking semantics from `siglip_attention_readout_recovery.py` without importing its runner.
- [ ] **Step 6: Run the complete Task-1 test file GREEN.**
- [ ] **Step 7: Commit only the two Task-1 files.**

### Task 2: Token extraction and residual-scale authority

**Files:**
- Create: `scripts/probe_siglip_spatial_tail_recovery.py`
- Create: `tests/test_probe_siglip_spatial_tail_recovery.py`

**Interfaces:**
- Produces: `stream_spatial_tail_fit_inputs(vision_model, projection, pixel_batches, device) -> SpatialTailFitInputs` containing CPU FP16 `h18`, CPU FP16 `h27`, CPU FP32 descriptors, and byte accounting.
- Produces: `residual_channel_scale(h18, h27) -> torch.Tensor` with FP64 fixed-order accumulation and FP32 output.

- [ ] **Step 1: Write tiny-SigLIP extraction RED tests.** Assert one teacher forward per batch, exact hidden-state indexes 18/27 using a reduced configurable test depth, BF16 autocast enclosure, depth-27 native pooler identity, frozen/eval modules, contiguous dtypes, finite checks, and exact byte count.
- [ ] **Step 2: Run only the extraction nodes and preserve missing-interface RED.**
- [ ] **Step 3: Implement bounded streamed extraction without evaluation-fold access.**
- [ ] **Step 4: Write scale RED tests.** Compare against an explicit FP64 sample-order oracle; cover zero/nonfinite shapes and floor arithmetic.
- [ ] **Step 5: Implement residual scale and run extraction/scale GREEN.**
- [ ] **Step 6: Commit only the Task-2 runner and test.**

### Task 3: Two model arms and matched training

**Files:**
- Modify: `scripts/probe_siglip_spatial_tail_recovery.py`
- Modify: `tests/test_probe_siglip_spatial_tail_recovery.py`

**Interfaces:**
- Produces: `TokenwiseTailControl`, `LatentInteractionTail`, `FrozenTeacherReadout`, and `fit_spatial_tail_arm(...) -> SpatialTailFitEvidence`.

- [ ] **Step 1: Write architecture RED tests.** Lock input/output shape, width 128, eight latents, four heads, zero dropout, frozen readout, treatment cross-sample isolation, treatment cross-token dependence, and control token independence.
- [ ] **Step 2: Run architecture nodes for RED.**
- [ ] **Step 3: Implement both arms and the frozen teacher readout.**
- [ ] **Step 4: Write objective/training RED tests.** Lock normalized residual MSE plus descriptor cosine, 4,000-step constants through a reduced-step injection used only by tests, identical deterministic index streams, finite gradients, clipping, schedule endpoints, no frozen-readout gradients, final-200 losses, and seed reproducibility.
- [ ] **Step 5: Implement the matched trainer and run all Task-3 tests GREEN.**
- [ ] **Step 6: Commit only the Task-3 files.**

### Task 4: Artifact seal, development loader, and local CLI

**Files:**
- Modify: `scripts/probe_siglip_spatial_tail_recovery.py`
- Modify: `tests/test_probe_siglip_spatial_tail_recovery.py`

**Interfaces:**
- Produces: `_write_spatial_tail_artifact`, `_load_spatial_tail_artifact`, strict `parse_args`, and `main`.

- [ ] **Step 1: Write artifact RED tests.** Require both complete state dicts plus frozen readout identity, exclusive create, SHA-256 binding, reload equality, and mutation rejection.
- [ ] **Step 2: Implement the safetensors artifact boundary and run focused GREEN.**
- [ ] **Step 3: Write local CLI RED tests.** Require authenticated control binding/checkpoint/optimization manifest and image namespace, output artifact/result paths, and `--execute-spatial-tail`; reject evaluation paths, labels 49..81, URLs, AWS/storage flags, duplicates, relative paths, and unknown flags.
- [ ] **Step 4: Implement the CLI.** Load the existing optimization manifest metadata, compute the fixed class split before pixels, extract/train only fit classes, seal/reload, then decode and stream development classes for baseline/control/treatment/teacher/cross evidence.
- [ ] **Step 5: Mutation-lock the loader boundary and run the complete runner test file GREEN.**
- [ ] **Step 6: Commit only the Task-4 files.**

### Task 5: DGX deployment and monitoring

**Files:**
- Create: `scripts/deploy_siglip_spatial_tail_recovery_v1.sh`
- Create: `tests/test_deploy_siglip_spatial_tail_recovery.py`

**Interfaces:**
- Produces: one content-addressed, source-bundled, local-only DGX execution with `monitor.json`, artifact, and canonical result.

- [ ] **Step 1: Write static deployment RED tests.** Lock exact source revision, content hashes, seven named cleanup targets, PNG compression level zero, no external evaluation staging, 5-second liveness sampling, 90-minute timeout, 110-GiB RSS, in-process 96-GiB CUDA cap, PSI/swap stops, PID clearance, and no restart loop.
- [ ] **Step 2: Implement the wrapper and run deploy tests GREEN.** Follow `deploy_siglip_attention_readout_recovery_v1.sh` patterns but use a new remote namespace and optimization-only manifest.
- [ ] **Step 3: Run Ruff, mypy, py_compile, ShellCheck, bash syntax, focused tests, and `git diff --check`; repair only observed failures.**
- [ ] **Step 4: Commit only the deployment files and verified repairs.**

### Task 6: Assurance and one development science run

**Files:**
- Create after the terminal: `docs/siglip_spatial_tail_recovery_result_2026-09-05.md`

**Interfaces:**
- Consumes: all prior tasks and the authenticated optimization corpus.
- Produces: one canonical development result and a documented branch decision.

- [ ] **Step 1: Run complete focused Python tests, dependency-complete `python -m unittest discover`/pytest repository gate as configured by the project, Ruff, mypy, py_compile, ShellCheck, and diff check.** Preserve original terminals.
- [ ] **Step 2: Commit the exact implementation slice.** Verify the worktree's unrelated files are unstaged.
- [ ] **Step 3: Build a content-addressed source bundle and run exactly one monitored DGX development process.** Keep external evaluation absent by construction.
- [ ] **Step 4: Validate the canonical result independently.** Recompute every metric/decision, verify artifact/result/manifest hashes, monitor caps, cleanup, PID clearance, and source revision.
- [ ] **Step 5: If quality passes, run the existing paired full-path latency protocol on the sealed treatment only.** Stop if any p95 ratio exceeds 0.75. Do not access external evaluation.
- [ ] **Step 6: Write the result document with exact evidence, limitations, and next branch.**
- [ ] **Step 7: Commit only the result document and push the implementation dependency stack to canonical `master` when it can fast-forward without unrelated work.** If the dependency stack is not integrable, preserve the exact commits and report the blocker rather than pushing unrelated history.

