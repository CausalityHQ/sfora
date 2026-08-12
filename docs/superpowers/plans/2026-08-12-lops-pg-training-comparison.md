# LOPS-PG Training Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LOPS-PG and a positive-compactness control to the existing image trainer, verify them locally, then run the frozen four-arm multi-seed In-Shop comparison.

**Architecture:** Implement one pure Torch cotangent projector and one compactness loss beside the existing Proxy Anchor loss. Register two objective names; the LOPS objective reuses the exact PA loss and installs an embedding hook before backward, while the compactness objective adds one scalar auxiliary term. Reuse the existing CLI, recipe resolver, checkpointing, and retrieval evaluator.

**Tech Stack:** Python 3.12/3.13, PyTorch 2.12, existing Sfora image trainer/CLI, pytest, Ruff, Git.

## Global Constraints

- Do not alter existing `proxy_anchor` or `batch_hard_triplet` behavior.
- LOPS changes only the encoder embedding cotangent; proxy gradients remain ordinary PA.
- Compactness weight is exactly `0.1` and centroids are stopped.
- Inference remains normalized embedding cosine.
- Seeds are exactly 0, 1, 2; full recipe is the existing official In-Shop PA recipe.

---

### Task 1: Torch operators and objective dispatch

**Files:**
- Modify: `src/sfora/image_end_to_end.py`
- Modify: `tests/test_image_end_to_end.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `_lops_pg_embedding_gradient(gradient, embeddings, labels, torch_module)`, `_positive_compactness_loss(embeddings, labels, torch_module)`, objectives `proxy_anchor_lops_pg` and `proxy_anchor_compactness`.

- [ ] Write RED tests with literal tensors proving conflict projection, safe and degenerate identity, stopped sibling centroids, unchanged PA scalar/proxy gradient, compactness value/gradient, objective parsing, and tiny trainer completion.
- [ ] Run the focused selector and verify failures are missing symbols/objectives.
- [ ] Implement the pure helpers. Use builtin Torch operations, finite FP32/FP64 tensors, `torch.no_grad()` for centroid construction, and no per-row CPU synchronization.
- [ ] Map both objectives to `_proxy_anchor_objective_loss`; add compactness in that handler only for its objective, and register an embedding hook only for LOPS after normalized embeddings are created and before loss construction.
- [ ] Run focused tests, Ruff, py_compile, and `git diff --check`; commit `add LOPS-PG training objectives`.

### Task 2: One-epoch local/remote smoke

**Files:**
- Create after execution: four smoke report/checkpoint pairs under `reports/generated/lops_pg_smoke/`.

**Interfaces:**
- Consumes: existing `sfora.cli image-end-to-end` and official recipe with train epochs overridden to one.
- Produces: finite reports for PA, LOPS-PG, compactness, and batch-hard triplet.

- [ ] Run the existing tiny image integration for all four objectives locally.
- [ ] On the registered GPU environment, run seed 0 for one epoch sequentially with the exact same recipe/data and distinct outputs.
- [ ] Require exit 0, finite loss, checkpoint presence, valid Recall@1, LOPS conflict coverage >=50%, and no inference/config difference beyond objective/compactness weight.
- [ ] Fix only demonstrated implementation defects under RED/GREEN tests; commit fixes separately.

### Task 3: Multi-seed comparison and decision

**Files:**
- Create after execution: `reports/generated/lops_pg_training/<arm>-seed<seed>/...`
- Create: `docs/inshop_lops_pg_training_result_2026-08-12.md`

**Interfaces:**
- Produces: 12 reports/checkpoints and paired seed Recall@1 decision.

- [ ] Run PA and LOPS seeds 0-2 sequentially with the official 60-epoch recipe.
- [ ] If at least two LOPS runs complete, run compactness and batch-hard-triplet seeds 0-2 sequentially.
- [ ] Validate every report, tabulate final and best Recall@1, paired LOPS-PA differences, means, conflict/skip diagnostics, and apply the six frozen predicates.
- [ ] Request read-only cross-provider review with `models=["opus","gpt-5.6-sol"]`; independently verify concrete findings.
- [ ] Commit the result doc and ordinary generated result files with subject `record LOPS-PG training comparison`.
