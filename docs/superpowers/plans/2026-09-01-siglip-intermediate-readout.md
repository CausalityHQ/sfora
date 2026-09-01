# SigLIP Intermediate-Readout Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sealed optimization-only screen over all 27 SigLIP encoder depths using the existing post-LN, mean pooling, trained 512-D projection, and exact cosine retrieval.

**Architecture:** A pure evidence module validates per-depth descriptor planes, scores the existing four class folds, selects one depth, and emits canonical evidence. A guarded script restores the completed control checkpoint, streams hidden states, and exposes no evaluation or network capability.

**Tech Stack:** Python 3.12, PyTorch, Transformers, existing SFORA control/checkpoint/fold authorities.

**Spec:** `docs/superpowers/specs/2026-09-01-siglip-intermediate-readout-design.md`

## Global Constraints

- Exactly 27 depths, width 1152, output width 512, post-LN mean pooling, and the sealed seed-17 projection.
- Exactly four optimization-only SFQ folds; no clean, burned, or official-test capability.
- Select by aggregate integer hits descending then depth ascending.
- Require +10,000 ppm and wins on at least three folds relative to operator-matched depth 27.
- Do not start science before the active three-seed control is terminal.

---

### Task 1: Descriptor-plane authority and depth selection

**Files:**
- Create: `src/sfora/siglip_intermediate_readout.py`
- Create: `tests/test_siglip_intermediate_readout.py`

**Interfaces:**
- Produces `IntermediateReadoutDepthEvidence`, `score_intermediate_readout_depths(...)`, `validate_intermediate_readout_result_bytes(...)`.

- [ ] Write failing tests for exact 27-plane topology, normalization, finite values, four-fold integer scoring, lowest-depth ties, +10,000 ppm/three-fold gates, scalar replay, and every schema/count/gate mutation.
- [ ] Run the focused test file and preserve missing-interface RED.
- [ ] Implement only descriptor validation, existing SFQ fold reuse, exact leave-one-out scoring, selection, gates, and canonical validation.
- [ ] Rerun focused tests to GREEN and commit `Add intermediate readout authority`.

### Task 2: Streamed checkpoint diagnostic

**Files:**
- Create: `scripts/diagnose_siglip_intermediate_readout.py`
- Create: `tests/test_diagnose_siglip_intermediate_readout.py`

**Interfaces:**
- Consumes exact final control receipt/checkpoint and optimization manifest/images.
- Produces one canonical claim-ineligible result; no token cache.

- [ ] Write failing tests with a real tiny 3-block vision fixture for post-LN mean/projection arithmetic, one-forward hidden-state streaming, topology refusal, strict checkpoint binding, absent evaluation files, forbidden flags, and non-clobber output.
- [ ] Run the focused CLI tests and preserve missing-file/interface RED.
- [ ] Implement the strict loader and batch stream by reusing the existing control component/checkpoint authority; parameterize physical depth only for synthetic fixtures while the scientific CLI requires 27.
- [ ] Rerun library/CLI tests to GREEN and commit `Add streamed intermediate readout screen`.

### Task 3: Guarded deployment and assurance

**Files:**
- Create: `scripts/deploy_siglip_intermediate_readout_v1.sh`
- Create: `tests/test_deploy_siglip_intermediate_readout.py`

**Interfaces:**
- Produces one remotely authenticated result/terminal after the control terminal.

- [ ] Write failing process tests for source/checkpoint/receipt binding, sole-GPU-process fencing, timeout process-group cleanup, pressure stops, exact output preservation, and no restart.
- [ ] Implement deployment following the verified SFQ/RSTA guarded scripts, with a hard one-hour scientific cap and explicit scratch cleanup.
- [ ] Run focused process tests, Ruff, format, `py_compile`, shell syntax, `git diff --check`, dependency-complete Python tests, and independent review.
- [ ] Commit/push verified repairs and require local/remote SHA equality before any scientific execution.
