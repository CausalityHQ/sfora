# Verbalizer-Margin Distillation F0 Design

Status: preregistered, claim-ineligible mechanism screen.

## Purpose

The forced-verdict Qwen teacher separates SAME from DIFFERENT pairs, while the
dense gradient student and FVCG-Norm do not transfer that signal reliably. F0
tests the narrower causal question before any more training: on each frozen
SigLIP retrieval error, does the teacher prefer a deterministically selected
same-class neighbor to the frozen wrong nearest neighbor?

This work is confined to Sfora. It does not modify or depend on Borsuk.

## Immutable inputs

- Cars train-band M2 error manifest:
  `reports/generated/pass209-m2-siglip-so400m-errors-2026-08-30.json`, SHA-256
  `64d491607d4dac144b31edac3a182130e6f94f994a272f612c195a7a72d55611`.
- Reproduced SigLIP-so400m M4 query evidence, cell `siglip-so400m`, file
  `selecting.queries.json`, SHA-256
  `b2fc9baf52feb3917554241b5aba205a7a10799ef6e3742e128e7aa173b33c67`.
  The table has 1,345 ordered rows and binds the legacy descriptor digest
  `4031dc2da90588dcc39005eab92c6c519f3058c581222421ca917501dd3df071`.
- Frozen Cars train examples and ordering already authenticated by both inputs:
  dataset revision `9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40` and examples digest
  `83a7800ee948a816e2fb9a2c9163027d9e90f167abc90052bf220619fa32240f`.
- The exact local Qwen snapshot, snapshot manifest, SAGA fixture, prompt,
  completion prefixes, and model revision previously authenticated by the
  forced P32 experiment.

For error row `q`, the true candidate is `historical_cuda_rows[q].best_same_position`.
The wrong candidate is the M2 row's `nearest_position`. The loader must prove
that query identities match, the wrong neighbor is reproduced by M4 historical
CUDA evidence, the true candidate has the query label, and all three positions
are distinct where required. No Qwen output participates in candidate choice.

## Teacher measurement

For each ordered image pair `(query, candidate)`, prepare the existing fixed
two-image prompt and teacher-force the registered SAME and DIFFERENT completion
prefixes without generation or gradients. Define:

`gap(query,candidate) = mean_logp(SAME) - mean_logp(DIFFERENT)`.

For each of the 103 frozen errors:

`preference_margin = gap(query,true) - gap(query,wrong)`.

A strict win requires a finite margin greater than zero. A zero margin is not a
win. Each observation records both pairs' branch scores, gaps, elapsed time,
and peak CUDA/RSS measurements. Ordinals 0 and 102 are replayed; all four
branch scores must be bit-identical.

## Frozen gates

The result passes only when all conditions hold:

- all 103 observations are finite, deterministic, and identity-bound;
- overall wins are at least 62/103 (`601941` ppm when integer-divided);
- the dominant Caliber 2012/2007 block wins at least 38/63;
- the remaining errors win at least 24/40;
- generated tokens are zero and language-model gradients are absent;
- peak CUDA reserved memory is at most 56 GiB, peak process RSS at most 32 GiB,
  and total scoring wall time at most 900 seconds.

The subgroup gates prevent the 63 correlated Caliber errors from deciding the
screen alone. These are engineering falsification thresholds, not independent
Bernoulli significance claims. The 103 errors and their labels are already
burned development evidence, so every F0 artifact is `claim_eligible=false`.

## Outcome and next action

- `teacher-target-supported`: every gate passes. Implement VMD using offline
  target probabilities; do not implement another gradient-space transport.
- `teacher-target-rejected`: any quality gate fails. Do not train VMD on this
  teacher signal; pivot to a non-teacher retrieval objective.
- authority, determinism, resource, or runtime failure: no scientific result;
  repair only the identified execution boundary and rerun the same protocol.

F0 passing authorizes one bounded Cars development pilot, not a paper claim.
The pilot compares baseline, wall-matched control, and VMD. A publication claim
requires a prospectively frozen dataset/holdout not used to design F0 or VMD.

