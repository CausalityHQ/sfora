# SigLIP Attention-Readout Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the intact 18-block SigLIP prefix can reproduce the teacher's retrieval geometry through a teacher-initialized learned attention readout while retaining the verified latency advantage.

**Architecture:** A new pure evidence module fits and validates deterministic ridge controls and retrieval decisions. A sibling local-only runner authenticates the existing teacher/data authority, streams seven attention-pooled control planes plus a depth-18 token plane, seals a teacher-initialized trainable LayerNorm/MAP-head/projection readout, and performs one exposed-band evaluation. Existing intermediate-readout and depth-recovery formats remain unchanged.

**Tech Stack:** Python 3.12, PyTorch, Transformers 5.12.1 on DGX, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-05-siglip-attention-readout-recovery-design.md`

## Global Constraints

- Sfora only; never modify or execute Borsuk code.
- Exact teacher path is hidden state -> frozen post-LN -> frozen MAP head -> bias-free readout -> L2 normalization.
- Depths are exactly 6, 10, 14, 18, 22, 25, and 27.
- Fit only on classes 0..48; seal every cell before reading classes 49..81; no official-test access.
- Depth-27 zero-train descriptors must reproduce 2,596/2,746 hits and mAP@R 0.7913744556922272.
- Final quality is not relaxed: deployment-grade requires at least 2,591 hits and mAP@R 0.7893744556922272 at depth <=18.
- Preserve unrelated dirty Qwen files and stage only named files.

---

### Task 1: Pure readout fitting and decision authority

**Files:**
- Create: `src/sfora/siglip_attention_readout_recovery.py`
- Create: `tests/test_siglip_attention_readout_recovery.py`

**Interfaces:**
- Produces `fit_ridge_readout(features, targets, teacher_weight) -> torch.Tensor`.
- Produces `apply_readout(features, weight) -> torch.Tensor`.
- Produces `refine_directional_readout(...)`, `build_attention_readout_result(...)`, and `validate_attention_readout_result_bytes(raw)`.
- Reuses `build_sfq_fold_schedule`, `asymmetric_retrieval_evidence`, and the existing retrieval result validators.

- [ ] **Step 1: Write ridge and normalization RED tests.** Use a hand-derived full-rank matrix to check the teacher-anchored FP64 normal equations, `1e-3` scale computation, bias absence, finite checks, deterministic weights, and unit FP32 outputs. Mutation-lock bool-as-number, rank, dtype, nonfinite, zero norm, row-order, and teacher-weight drift.
- [ ] **Step 2: Run the exact new test nodes.** Run `.venv/bin/pytest -q tests/test_siglip_attention_readout_recovery.py -k 'ridge or apply'`; require missing-module/interface failure.
- [ ] **Step 3: Implement the minimal deterministic ridge kernel.** Accumulate `H.T@H` and `H.T@P` in float64 on CPU, add `lambda*I`, solve with `torch.linalg.solve`, return contiguous FP32 `[512,1152]`, and independently verify the residual is finite.
- [ ] **Step 4: Rerun the exact nodes to GREEN.** Preserve the original RED and GREEN terminals.
- [ ] **Step 5: Write directional-refinement and control RED tests.** Prove the exact 2,000-update schedule, final-only result, deterministic cyclic shuffle, FP32 cosine loss, clipping, zero weight decay, fixed random control, and target derangement. Ridge and refined cells remain separately named.
- [ ] **Step 6: Implement refinement and controls, then rerun focused tests.** No evaluation labels, metrics, or features may enter fitting, stopping, or cell selection.
- [ ] **Step 7: Write result/decision RED tests.** Cover self/cross per-query evidence, exact recomputation, depth ordering, depth-27 identity, the required depth-18 `learned-attention` cell, two quality tiers, coordinate-alignment branch, invalid precedence, canonical newline bytes, and every schema/type/count/digest mutation. Linear cells are controls and cannot select the branch.
- [ ] **Step 8: Implement result construction and independent validation.** The validator must derive every aggregate and selected depth from per-query evidence rather than trust stored floats.
- [ ] **Step 9: Run the full new core file and commit the isolated core slice.** Stage only the two Task-1 files after pytest, Ruff, formatting, mypy, and diff-check pass.

### Task 2: Exact hidden-state extraction and learned attention readout

**Files:**
- Create: `scripts/probe_siglip_attention_readout_recovery.py`
- Create: `tests/test_probe_siglip_attention_readout_recovery.py`

**Interfaces:**
- Produces `stream_attention_readout_inputs(model, pixel_batches, depths, ...)` with seven CPU FP32 `[N,1152]` control planes, one depth-18 CPU FP16 `[N,tokens,1152]` plane, and teacher projected targets.
- CLI consumes the exact local control binding, checkpoint, optimization manifest/images, evaluation authority, and one exclusive result path.

- [ ] **Step 1: Write real tiny-SigLIP extraction RED tests.** Compare each registered control plane to direct `vision_model.head(vision_model.post_layernorm(hidden_states[k]))`; prove hidden-state indexing, one teacher forward per batch, no mean pooling, frozen encoder modules, exact depth-18 token caching, and depth-27 equality to `pooler_output`.
- [ ] **Step 2: Run the exact extraction tests and preserve the RED.** Expected failure is the missing runner/interface only.
- [ ] **Step 3: Implement streamed extraction.** Use the same BF16 autocast and batch partition as the authenticated evaluator, move only pooled control planes, the depth-18 FP16 token plane, and targets to CPU, hash each plane, and record derived cache bytes.
- [ ] **Step 4: Rerun extraction tests to GREEN.** Include replay tolerance and exact normalized descriptor equality tests.
- [ ] **Step 5: Write learned-attention RED tests.** Freeze encoder features; initialize LayerNorm, MAP head, and projection exactly from the teacher; optimize only teacher-descriptor cosine recovery with the fixed schedule; prove labels and evaluation examples cannot enter fitting, the final seal is deterministic, and explicit `optimization_limited` evidence is preserved.
- [ ] **Step 6: Implement the learned attention readout, then rerun focused tests.** Keep the seven linear cells as non-selectable controls and add exactly one selectable depth-18 `learned-attention` cell.
- [ ] **Step 7: Write strict CLI/authority RED tests.** Require explicit execution, absolute local paths, exact SHA-256 identities, exclusive output, optimization/evaluation role separation, and refuse network, arbitrary model/checkpoint, student, text/class-name, official-test, and tuning flags.
- [ ] **Step 8: Implement the local runner and two-phase data access.** Authenticate and fit optimization cells first; serialize and hash sealed weights; only then acquire the exposed evaluation band and score every fixed cell once.
- [ ] **Step 9: Run focused runner and affected retrieval tests.** Include direct-script execution and deliberate digest/role/capability mutations.
- [ ] **Step 10: Commit the runner slice.** Stage only the runner and its test after scoped static checks.

### Task 3: Synthetic end-to-end assurance and research review

**Files:**
- Modify: `tests/test_probe_siglip_attention_readout_recovery.py`
- Modify: `docs/superpowers/specs/2026-09-05-siglip-attention-readout-recovery-design.md` only if review finds a verified contract defect.
- Modify: `docs/superpowers/plans/2026-09-05-siglip-attention-readout-recovery.md` only if review changes execution steps.

**Interfaces:**
- Produces one synthetic complete canonical result without network/GPU.

- [ ] **Step 1: Add an end-to-end tiny 7-depth fixture.** It must exercise optimization fitting, weight sealing, delayed evaluation access, both retrieval modes, controls, decision precedence, and exclusive output.
- [ ] **Step 2: Add failure-injection tests.** Evaluation access before seal, hidden-state drift, permuted-control shortcut, partial output, replay mismatch, and a second evaluation attempt must fail closed.
- [ ] **Step 3: Run focused core+runner tests and static gates.** Run pytest on both new files, Ruff, formatter check, targeted mypy, py_compile, and git diff-check.
- [ ] **Step 4: Start one Astra and one Fable read-only review in parallel.** Give both the exact spec, plan, diff, prior scientific receipts, and test terminals; reconcile findings independently.
- [ ] **Step 5: Repair only verified findings with fresh RED/GREEN evidence.** Do not broaden the scientific surface or change thresholds without a new explicit spec amendment.
- [ ] **Step 6: Run repository assurance once.** Use dependency-complete Python test discovery and preserve the original terminal; after any failure, repair the narrow layer before one final assurance run.
- [ ] **Step 7: Commit and push only the verified Sfora slice.** Verify local HEAD, tracking ref, remote ref, and clean intended scope; unrelated user changes remain untouched.

### Task 4: One monitored DGX diagnostic

**Files:**
- Create after terminal: `docs/siglip_attention_readout_recovery_result_2026-09-05.md`

**Interfaces:**
- Produces one immutable canonical claim-ineligible result and monitor receipt.

- [ ] **Step 1: Deploy a hash-named source overlay.** Rehash every local/remote source and input role; verify DGX GPU/PID/PSI/swap clearance and exact Transformers 5.12.1 runtime.
- [ ] **Step 2: Launch one original monitored process.** Enforce a 90-minute wall cap, RSS 110 GiB, CUDA reserved 96 GiB, PSI full avg10 0.79 immediate or 0.50 for three 5-second samples, swap growth 256 MiB, and 300-second progress gap. Never duplicate or restart automatically.
- [ ] **Step 3: Poll the original process at <=55-second tool intervals.** Send Telegram only for a meaningful phase change, scientific terminal, or actionable blocker.
- [ ] **Step 4: Authenticate the sole terminal.** Recompute every metric, decision, digest, resource bound, cleanup, and PID clearance independently.
- [ ] **Step 5: Ask Astra and Fable for post-result ideation in parallel.** Give each only the verified objective, failures, result, and constraints; do not disclose an existing candidate plan. Require independent next instructions, reconcile them, and update the evidence ledger.
- [ ] **Step 6: Follow the frozen branch.** Positive/promising -> contiguous-prefix teacher-coordinate distillation, listwise rank preservation, margin-gated teacher-error correction, and a fresh matched speed gate. Coordinate mismatch -> diagnose alignment without changing the gallery contract. Rejected after convergence -> full-depth delayed token merging. No class-language or exotic geometry detour without separate evidence.
