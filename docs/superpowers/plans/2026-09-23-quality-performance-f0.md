# Deployed-Code Rank Finish F0 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Decide whether a code-aware rank objective improves the class-disjoint In-Shop training holdout at the unchanged 130-byte serving wire.

**Architecture:** Refit the existing compact head on the rank-finished model's optimization identities. Continue two matched heads with SmoothAP on float-normalized and fake-quantized int8-128 vectors respectively. Score all three arms with the same exact packed cosine on the held-out identities.

**Tech Stack:** Python 3.13, PyTorch, NumPy, existing Sfora compact-metric and UNICOM holdout utilities, DGX GB10.

**Spec:** `docs/superpowers/specs/2026-09-23-quality-performance-continuation-design.md`

## Global Constraints

- Read only `train_embeddings` and `train_labels` from the authenticated archive; official query/gallery arrays are outside this screen.
- Bind the rank-finished seed-1 artifact to its authenticated parent run receipt (`holdout_fraction=0.2`, `holdout_seed=0`) before constructing the split.
- Preserve the identical 130-byte int8-128 plus f16 inverse-norm search representation.
- Do not compare a head trained on all official train identities with this training-only holdout.
- Launch only when the DGX has no active overlapping GPU job; retain the original session and exit status.
- Run once from a clean `git archive` of the pushed source commit at `/home/riomus/runs/sfora-quality-f0-<commit>/source`, with a 3,600-second wall cap and output `/home/riomus/runs/sfora-quality-f0-<commit>/result.json`.
- Report both mAP@R and Recall@1, with paired per-query evidence, input hashes and resource use.

## Review Focus

- Archive row order must equal the official train partition row order.
- Holdout query/gallery must be complete-class disjoint from optimization rows.
- Straight-through fake quantization must have the same forward code geometry as the packed serving scorer.
- Any rank ties must have deterministic ordinal resolution.
- A head-only negative is not evidence against full-backbone training.

## Task 1: Research screen

**Files:** Create `scripts/_scratch_deployed_code_rank_finish_f0.py`; add a focused test for code geometry and gradient flow.

- [ ] Bind the archive and partition hashes, verify train-label row order, and reconstruct the registered identity holdout.
- [ ] Fit the existing compact head only on optimization rows and score it on held-out query/gallery.
- [ ] Continue two heads for exactly two identity-balanced epochs with matched float and code-aware SmoothAP, fixed optimizer settings, and one shared schedule.
- [ ] Score both continuations with the same packed routine, report paired differences, net Recall@1 flips, gate and hashes.
- [ ] Run the focused test, then run the screen once on the idle DGX and collect its final exit status.

## Task 2: Decide next experiment

**Files:** Add the result receipt and a brief result document.

- [ ] Verify receipt/input hashes and paired metrics independently.
- [ ] If the frozen gate passes, write a separately frozen last-block joint probe; if it fails, close only the head continuation and identify the next distinct backbone experiment.
- [ ] Obtain cross-provider critique of any promotion decision, then commit and push the result to `master` with the configured operator identity.
