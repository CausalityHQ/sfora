# PE image-training mechanics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify PE's native trainable/frozen routes and paired update cost before full-fit training.

**Architecture:** One explicit prefix-freezing helper for the two native architectures; one smoke script reuses existing objective, initialization and bank helpers. Keep the image trainer and deployed packages unchanged until learned-quality qualification.

**Tech Stack:** Existing Torch/transformers, isolated pinned PE source, NumPy and stdlib.

**Spec:** `docs/inshop_pe_training_smoke_gate_2026-09-28.md`.

## Global Constraints

- GPU/training only authenticated DGX Spark; inspect before launch; preserve original jobs.
- Original FP32 checkpoints; matched seed179032/16x64/native BF16 objective; no held/official score.
- Whole GPU process180s/8GiB host/allocatedCUDA Large<16,000,000,000B/PE<10,000,000,000B, shared lock; no resource or gate rescue.
- Protected Rust topk hash1ae7f50aa3d222dbc6c6489c4c344ce836d01b895e05341d368db35c6d0cb587 stays untouched.

## Review Focus

- PE pre-normalization or position/class parameters accidentally trainable: prefix inventory check.
- Native parameter aliasing or wrong depth: validate exact depths, name inventory and shared parameters.
- Cached FP16/native BF16 disagreement: both-arm first-four BF16/FP32 and cachedFP16/freshBF16 cosine>=.999 guards before optimizer.
- Duplicate batch ordinals: native last-occurrence refresh and one small fixture assertion.
- Frozen prefix drift despite no gradient: byte hashes before/after optimization.

---

### Task1: Native prefix authority

**Files:** Create `scripts/pe_core_training.py`; runnable `--self-test` CPU check.
**Interfaces:** `freeze_prefix(vision, arm)` consumes native loaded module and `arm in {'pe','large'}`; returns named trainable/frozen parameter inventories.

- [x] Add one fixture check requiring exact frozen PE stem/position/pre-LN/first6 and Large embeddings/first12 inventories; reject wrong depth.
- [x] Run CPU check on DGX, observe red before implementation.
- [x] Implement explicit native prefix freeze; reuse original strict loaders, no facade model class.
- [x] Commit passing faithful fixture and real-native PE prefix checks, including unregistered rotary state.

### Task2: Matched native image mechanics

**Files:** Create `scripts/probe_inshop_pe_training_smoke.py`; use existing initializer, sampler, ArcFace and member-bank helpers.
**Interfaces:** Consume exact source pilot NPZ/preflight/acquisition authorities; produce one raw receipt plus fixed named initializer, pixel, schedule and source manifests before optimizer.

- [x] Implement16-step matched source smoke exactly as spec, no quality evaluation.
- [x] Run one CPU metadata/objective/duplicate-refresh check; freeze executable before optimizer.
- [x] Obtain Opus/Astra critique of concrete gate/code prerequisites; independently reconcile findings.
- [ ] Launch original bounded DGX job once after all prerequisites pass; collect its terminal status and receipt.
- [ ] Replay route/hash/timing/cost criteria; publish result and honest next gate, commit/push master.

Ruling: CPU authority v3/v4 replaces v2 after real startup defect and framework
cleanup review; preserve previous raw versions. No GPU outcome/cap rescue.
Final review: three startup/provenance/workspace findings fixed before GPU;
TF32 and rotary-grid audit tightened. Resource/objective/schedule unchanged.
