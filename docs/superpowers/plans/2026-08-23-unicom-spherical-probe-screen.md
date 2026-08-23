# UniCOM Spherical-Probe Causal Screen Implementation Plan

> Use test-driven development for every behavior change and request an independent
> cross-provider review before the DGX run.

**Goal:** Decide cheaply whether conservative classifier-direction optimization
changes the causal backbone learning signal enough to justify a fine-tuning screen.

**Spec:** `docs/superpowers/specs/2026-08-23-unicom-spherical-probe-screen-design.md`

**Status at plan refreeze:** reusable logits are committed at `9d493fd`; the initial
pure implementation is committed through `aa56d4a`; the first CLI test scaffold is
uncommitted. Claude review `c69025073e204d92` found three Critical and five Important
issues. The steps below supersede the original invalid split/LR/decision protocol.

## Frozen constants

- split stream 23,000; fitting batch 23,001; fitting masks 23,002;
  validation masks 23,003; independent fit diagnostic 23,004; gradient diagnostic
  23,005; all stochastic streams use `experiment_stream_seed`;
- fit seeds `(0, 1, 2)`, 512 steps, batch 128, AdamW LR `1e-4`, betas
  `(0.9, 0.999)`, epsilon `1e-8`, zero decay;
- 8 shards, 512 of 768 features, ArcFace margin 0.25 and scale 32;
- 64 validation masks; paired two-sided 95% t interval with df 63;
- minimum 48/64 positive mask deltas; head cosine minimum 0.80 and mean 0.95;
- gradient relative-L2 minimum 0.05 and median-cosine maximum 0.995.

## Task 1: Review-fix the pure split

**Files:** `src/sfora/unicom_probe.py`, `tests/test_unicom_probe.py`

1. RED: prove a singleton stays in fitting, has no validation row, and preserves its
   class row; prove seeded validation choice is byte deterministic and differs from
   lexicographic-last behavior; prove represented/unrepresented series strata.
2. GREEN: return a frozen `ProbeSplit` containing fitting/validation rows, aligned
   stratum flags, validation-class count, and singleton-class count.
3. Run the split selector and commit the isolated fix.

## Task 2: Review-fix fitting and head diagnostics

**Files:** same pure source/test pair.

1. RED: pin LR `1e-4`; prove optimizer and independent diagnostic streams do not
   reuse their first batch/mask; prove all stochastic mask seeds are namespaced;
   prove the fitted head retains minimum/mean cosine boundaries on the fixture.
2. GREEN: add exact constants, use stream 23,004 for fit diagnostics, and return
   rowwise cosine summary with structural norm evidence.
3. Run the fitting selector and commit.

## Task 3: Review-fix validation, uncertainty, and causal diagnostics

**Files:** same pure source/test pair.

1. RED: independently verify margin-0 accuracy, paired mask deltas/t lower bound,
   both acquisition strata, and three-seed symmetric decisions.
2. RED: build a real cached-feature gradient oracle and prove exact gradient norms,
   relative difference, per-sample cosine/zero behavior, and close boundaries.
3. GREEN: implement `evaluate_probe_heads`, `compare_probe_gradients`, and the new
   three-seed `probe_decision` without duplicating registered logits/loss.
4. Run the complete pure suite and commit.

## Task 4: Complete the strict CLI

**Files:** `scripts/screen_unicom_spherical_probe.py`,
`tests/test_screen_unicom_spherical_probe.py`

1. RED: replace the stale single-head fixture with the literal frozen protocol and
   complete exact schema: singleton/validation counts, three fits, cosine summaries,
   64 paired deltas and t bounds, two strata, three gradient diagnostics, structural
   norm checks, hashes, and recomputed decision.
2. RED: cover exact arguments/source/checkpoint/partition authentication, train-only
   access, deterministic one-pass feature extraction, RNG restoration, candidate-free
   progress, strict JSON, no-clobber atomic publication, cleanup/rollback, and strict
   persisted reload.
3. GREEN: implement `run`, result construction/validation, and publication. Structural
   failure exits 2 without a scientific result; valid close remains a valid result.
4. Run both focused files, Ruff, py_compile, and `git diff --check`; commit.

## Task 5: Independent source review and assurance

1. Start one read-only review with ordered models `['opus','gpt-5.6-sol']`, naming the
   exact commits and asking for Critical/Important findings only.
2. Reproduce and repair every valid finding RED-to-GREEN. Do not tune scientific
   constants after observing candidate values.
3. Coordinate the shared heavy lane and run one serial repository pytest, then final
   Ruff, py_compile, and diff-check.
4. Commit and push the reviewed source to `devbox/similarity-ghc`.

## Task 6: One monitored DGX screen and immediate routing

1. Sync the reviewed Git checkout to `riomus@spark-2751`; authenticate source/model/
   dataset, ensure the GPU and queue are idle, and require destination/temp absence.
2. Launch exactly one screen process. Retain its PID/session and poll the same process
   at intervals no longer than 55 seconds. Report liveness without candidate values.
3. On exit, strict-reload and independently validate the artifact before interpretation.
4. If `PROMOTE`, freeze the 2x2 continuation before launching it. If
   `CLOSE_DIRECTION`, record closure and select the next evidence-based training
   candidate. Commit the immutable artifact/report and push.
