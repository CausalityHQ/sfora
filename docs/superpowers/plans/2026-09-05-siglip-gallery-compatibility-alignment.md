# SigLIP Gallery-Compatibility Alignment Implementation Plan

**Goal:** Test a fitting-only orthogonal coordinate repair for the sealed fast
tokenwise tail without touching external classes.

**Architecture:** A pure module owns FP64 Procrustes fitting, application, and
canonical evidence. A local runner streams fitting/development descriptors from
the authenticated checkpoint and sealed tail. A guarded DGX wrapper stages only
optimization authority and retains a sealed matrix plus canonical result.

## Tasks

- [x] Add RED tests for FP64 orthogonal fitting, reflection support,
  deterministic replay, finite/type/shape rejection, and exact preservation of
  student cosine rankings.
- [x] Implement the minimal alignment core and make focused tests green.
- [x] Add RED tests for canonical four-direction retrieval evidence, fidelity,
  exact 97%/0.95 asymmetric and 99%/0.96 self gates, digest binding, and every
  concrete schema/relation mutation.
- [x] Implement canonical result construction/validation.
- [x] Add a local-only runner that authenticates the prior control and spatial
  artifact, streams descriptors from 39 fitting classes, seals the alignment,
  then opens only the ten burned development classes for evaluation.
- [x] Add a guarded source-bundled DGX wrapper with an optimization-row-only
  materializer, Landlock filesystem/TCP capability boundary, one
  process, RSS/CUDA/PSI/swap/progress/wall stops, artifact/result validation,
  and explicit partial cleanup.
- [x] Run focused and dependency-complete tests, Ruff, mypy, pycompile,
  ShellCheck, and diff-check; commit only this slice.
- [ ] Run exactly one DGX diagnostic, independently validate the terminal, and
  record exact evidence. Keep external labels 49..81 sealed.
