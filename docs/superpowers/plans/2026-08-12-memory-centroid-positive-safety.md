# MCPS-PG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add memory-centroid positive-safe encoder-gradient projection to the
existing Proxy Anchor trainer and kill or advance it using a one-epoch official
In-Shop smoke.

**Architecture:** A small device-resident class-centroid state supplies stopped
pre-batch targets. The existing PA loss receives an embedding hook that projects
only conflicting encoder cotangents; the state updates after the optimizer.
Diagnostics flow through the existing method metrics and JSON report.

**Tech Stack:** Python 3.12, PyTorch 2.12, Pydantic, Typer, pytest, Ruff, Git.

## Global Constraints

- Objective is exactly `proxy_anchor_mcps_pg`.
- Memory momentum is exactly `0.9`; update weight is exactly `0.1`.
- Current-batch embeddings cannot enter their own target.
- Proxy loss and proxy gradients are unchanged.
- Official In-Shop recipe and cosine inference are unchanged.
- Implementation uses ordinary commits; no provenance or authentication work.

---

### Task 1: Centroid state and projection

**Files:**
- Modify: `src/sfora/image_end_to_end.py`
- Modify: `tests/test_image_end_to_end.py`

**Interfaces:**
- Produces: `_MCPSCentroidState.targets(labels, proxies, proxy_labels)` and
  `.update(embeddings, labels)`; `_mcps_pg_embedding_gradient(...)`.

- [ ] Add RED tests using literal tensors for unseen proxy fallback, seen
  pre-update targets, exact normalized 0.9/0.1 update, per-label batch means,
  conflict projection, safe identity, and degenerate identity.
- [ ] Run only those tests and confirm missing-symbol failures.
- [ ] Implement the minimal FP32/FP64-compatible state and pure projection.
- [ ] Run the selector GREEN; run Ruff and `py_compile` for the touched files.
- [ ] Commit `add MCPS-PG centroid geometry`.

### Task 2: Trainer, control, diagnostics, and CLI

**Files:**
- Modify: `src/sfora/image_end_to_end.py`
- Modify: `src/sfora/cli.py`
- Modify: `tests/test_image_end_to_end.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces objectives `proxy_anchor_mcps_pg` and
  `proxy_anchor_proxy_compactness`; report field `mcps_diagnostics`.

- [ ] Add RED tests proving the PA scalar/proxy gradient is identical, a live
  conflicting hook changes encoder parameters, the state updates only after
  backward, proxy compactness is stopped, both recipe selectors preserve the
  requested objective, and report rates recompute from counts.
- [ ] Run the focused selectors and confirm behavior-specific failures.
- [ ] Register both objectives, install the MCPS hook, update state after the
  optimizer, add the 0.1 proxy compactness control, and persist diagnostics.
- [ ] Run focused GREEN tests, CLI regression tests, Ruff, `py_compile`, and
  `git diff --check`.
- [ ] Commit `add MCPS-PG training objective`.

### Task 3: Official one-epoch smoke

**Files:**
- Create remotely: `reports/generated/mcps_pg_smoke_1/<arm>/report.json`
- Create remotely: `reports/generated/mcps_pg_smoke_1/<arm>/checkpoint.pt`
- Create: `docs/inshop_mcps_pg_smoke_result_2026-08-12.md`

**Interfaces:**
- Consumes the completed PA seed-0 smoke and the exact official In-Shop data.
- Produces one MCPS and one proxy-compactness seed-0 run.

- [ ] Bundle the reviewed source commit into a fresh detached remote checkout;
  require an idle GPU and absent outputs.
- [ ] Run MCPS then proxy compactness sequentially with seed 0 and one epoch.
- [ ] Strict-load reports and verify 143 finite losses, checkpoints, Recall@1,
  config equality, and the four frozen MCPS smoke predicates.
- [ ] Record PASS/KILL with exact values and report hashes; do not reinterpret a
  failed threshold.
- [ ] Commit `record MCPS-PG training smoke`.

### Task 4: Conditional multi-seed comparison

**Files:**
- Create remotely: `reports/generated/mcps_pg_training/<arm>-seed<seed>/...`
- Create: `docs/inshop_mcps_pg_training_result_2026-08-12.md`

**Interfaces:**
- Runs only after Task 3 PASS.

- [ ] Run fresh PA and MCPS seeds 0, 1, and 2 sequentially for 60 epochs.
- [ ] If all MCPS runs complete, run proxy compactness seeds 0, 1, and 2.
- [ ] Gate only final R@1 using the five frozen predicates; report best R@1
  descriptively.
- [ ] Request read-only review with `models=["opus","gpt-5.6-sol"]`, verify
  concrete findings, and commit `record MCPS-PG training comparison`.

