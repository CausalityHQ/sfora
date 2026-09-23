# Packed top-k Lean proof implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` for native execution. Steps use checkbox syntax for tracking.

**Goal:** Compile a conditional Lean proof of block top-k exactness, margin-based recall preservation, and symbolic cost bounds.

**Architecture:** A separate `formal/` Lake package models a finite gallery and a strict score/ordinal ranking. Its theorems describe the abstract selector and conditional quality/cost claims. A README records how the model relates to Rust/CUDA and where it does not.

**Tech Stack:** Lean 4 and Mathlib v4.33.0, Lake.

**Spec:** `docs/superpowers/specs/2026-09-23-packed-topk-lean-proof-design.md`

## Global constraints

- Do not change the RC3 production scorer, public API, or package metadata.
- Do not claim empirical recall, latency, or floating-point equivalence from a mathematical proof.
- Use no `sorry` or custom axioms in proof files.
- Keep the Lean dependency isolated under `formal/`.

## Review focus

- Equal scores must choose the lower ordinal; include a boundary example.
- Padded indices must not enter the gallery model; include a boundary example.
- A non-robust margin must not be presented as a recall guarantee; include a counterexample.
- Local winners must be enough for a global merge, even when a block is shorter than `k`.
- Cost assumptions must be named separately from derived arithmetic bounds.

---

### Task 1: Pin and verify the proof environment

**Files:** Create `formal/lean-toolchain`, `formal/lakefile.toml`, `formal/.gitignore`, `formal/SforaProofs.lean`.

- [x] Pin Lean and Mathlib to v4.33.0 and fetch the binary cache.
- [x] Add a first theorem declaration with an intentionally failing proof and run `lake build` to confirm the checker rejects it.
- [x] Replace the placeholder with a minimal checked theorem; run `lake build`.

### Task 2: Deterministic top-k

**Files:** Create `formal/SforaProofs/Core.lean` and import it from `formal/SforaProofs.lean`.

- [x] Define strict rank by score descending and ordinal ascending over a finite gallery.
- [x] Prove global winners belong to their block's local top `k`.
- [x] Prove selecting top `k` from all local winners equals the global top `k` when the blocks cover the gallery.
- [x] Check equal-score and short-block examples with `example` declarations.

### Task 3: Conditional recall

**Files:** Extend `formal/SforaProofs/Core.lean`.

- [x] Define pointwise `ε` score error, robust true winners, and recall cardinality.
- [x] Prove a `2ε` margin preserves robust winners in approximate top `k`.
- [x] Prove the robust-item fraction lower bound and full-margin recall-one corollary.
- [x] Check a boundary example where the margin premise is false.

### Task 4: Symbolic cost and claim limits

**Files:** Create `formal/SforaProofs/Cost.lean`,
`formal/SforaProofs/Partition.lean`, and `formal/README.md`.

- [x] Prove a block-candidate count upper bound and a conditional additive latency bound.
- [x] Document exact assumptions, the proof-to-Rust/CUDA gap, and the relation to measured RC4 receipts.
- [x] Run `lake build`, scan proof files for holes, and inspect the generated dependency lock.

Commit and push only the new proof artifact and design documents to `master`.
