# Packed top-k formal proof design

## Purpose

Add a checked Lean 4 model of the deterministic claims behind Sfora's packed
top-k search. The proof is supplementary to the RC4 measured decision at
`2d9ccfeb`; it does not change the scorer, package version, or release gate.

## Model and assumptions

The gallery is a finite set of unique ordinals. A score map assigns each
ordinal a mathematical score. Ranking is by descending score, then ascending
ordinal. The model excludes NaN and uses exact ordered arithmetic; mapping
CUDA `f32` operations to that model is a separate refinement obligation.
Invalid padded ordinals are absent from the finite gallery.

Each gallery block is a subset, blocks cover the gallery, and a local selector
keeps its top `k` ordinals. A merge selector keeps the top `k` of the union of
local winners. The proof must establish that every global top-`k` ordinal
survives local selection and that the merge returns exactly the global top
`k`, including the score-tie rule. A concrete positive-width range partition
and an arbitrary finite chain of valid merge levels instantiate the generic
claim. Kernel placeholder and duplicate lanes remain a source refinement
obligation.

For quality, exact and approximate score maps have pointwise absolute error
at most `ε ≥ 0`. A true top-`k` item is *robust* if its exact score exceeds
every true outsider by more than `2ε`. The proof must show that robust items
remain in approximate top `k`, and hence approximate recall against exact
top `k` is at least the robust-item fraction. A full-margin condition gives
recall one. This is a conditional rank guarantee. It is not a proof of
semantic Pet/In-Shop recall, descriptor training quality, or an empirical
error bound.

## Cost model

Define symbolic work and storage counts from gallery rows `N`, dimensions
`D`, block width `B`, and retained width `k`. Prove an upper bound on the
number of block candidates and an additive latency bound conditional on
specified upper bounds for scoring, selection, merge, transfer, and host
stages. The theorem cannot yield measured p50/p95/p99, GPU memory peaks, or
throughput without validating those premises on hardware.

## Proof artifact and verification

Pin Lean 4 and Mathlib to matching `v4.33.0` releases. Keep the Lean project
in `formal/` so Python and Rust builds are unaffected. Provide named theorems,
small boundary examples (ties, padded-tail exclusion, non-robust margin), a
README mapping each assumption to the Sfora code or measured evidence, and a
single `lake build` verification command. No `sorry`, custom axioms, or
unverified external oracle may appear in the proof files.

The accepted deliverable is a compiling proof artifact plus its explicit
proof-to-implementation and empirical-claim limits. It is not a source-level
verification of the Cutile kernel or a replacement for the authenticated
million-row measurements.
