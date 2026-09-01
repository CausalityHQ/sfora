# Frozen SigLIP Band Audit Design

## Purpose

The current Cars control reaches high optimization recall but transfers poorly to
class-disjoint clean and burned bands. This diagnostic determines whether that
loss already exists in the untouched frozen SigLIP representation or is created
by training. It is causal triage, not a benchmark claim and not another head
search.

The audit is restricted to the pinned Stanford Cars training split and the
frozen `google/siglip-so400m-patch14-384` vision pooler at revision
`9fdffc58afc957d1a03a25b10dba0329ab15c2a3`. It never reads the official test
split, a trained checkpoint, clean/burned metrics, or prior method outputs.

## Frozen authority

The dataset revision is
`9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40`. The complete ordered 196-name Cars
label vocabulary, serialized as compact JSON plus one LF, has SHA-256
`9da9ec6333105a7a2f0d50d7a5a6afe18b1ec3ede7dd8f1df298e59eb859ce35`.
Every input example is bound by ordered example ID, integer label, and the source
tree digest already used by the frozen-substrate screen.

The three disjoint training-class bands are fixed:

- optimization: labels 0 through 48;
- clean: labels 49 through 81;
- burned: labels 82 through 97.

Every class must occur at least twice. The audit scores each band independently;
an example from one band can never be a gallery candidate for another band.

## Twin equivalence

Strict Recall@1 treats every model-year label as distinct. A second diagnostic
metric collapses only these preregistered visually near-identical label groups:

- optimization: `(7,8)`, `(9,10)`, `(16,17)`, `(20,21)`, `(22,23)`,
  `(26,27)`, `(28,29)`, `(41,42)`, `(44,45)`;
- clean: `(53,68,69,73,74)`, `(54,55,56)`, `(63,70)`, `(66,72)`;
- burned: `(82,83)`, `(85,86)`, `(89,90)`, `(93,94)`, `(95,96)`.

All other labels are singleton equivalence classes. Groups are disjoint and
contained within one registered band. Twin-collapsed recall is explanatory
only: it cannot pass a method, tune a model, or replace strict benchmark recall.

## Scoring contract

The runner encodes every Cars training image once using the frozen vision
pooler, converts output to contiguous float32, verifies finiteness and nonzero
norm, and L2-normalizes once. The scorer computes exact leave-one-out cosine
nearest neighbours within each band in fixed query blocks. The query diagonal is
set to negative infinity and `torch.argmax`'s lowest-row tie is authoritative.
Focused tests compare every selected neighbour against an independent scalar
oracle over random, tied, and nonfinite fixtures; the scientific run does not
duplicate the full quadratic score pass.

For each band the result records query count, strict hits and ppm, twin-collapsed
hits and ppm, twin-rescued error count, and ordered confusion counts. It also
records aggregate values over all three bands, but no aggregate can hide an
individual band.

## Interpretation

The audit has no pass flag. Its outcome selects the next causal branch:

1. If frozen strict recall is strong in clean/burned but the trained control is
   weak, training damages transferable geometry. Prioritize frozen-feature
   distillation, representation-preserving regularization, and transfer-aware
   stopping before changing the backbone.
2. If frozen strict recall is weak but twin collapse recovers most errors, the
   split is dominated by model-year ambiguity. Continue to report strict recall,
   but use manufacturer/twin-aware training objectives and hard-negative
   supervision.
3. If both frozen strict and twin-collapsed recall are weak, the 512-D frozen
   substrate lacks the necessary information. Stop tuning heads and move to a
   capacity-matched tower such as the Qwen/SAGA family.

No threshold is chosen from the observed audit. Any subsequent method keeps its
own preregistered clean gate and untouched official-test boundary.

## Artifacts and safety

`src/sfora/siglip_band_audit.py` owns pure validation, scoring, canonical result
construction, and independent validation. `scripts/audit_siglip_frozen_bands.py`
owns the strict local encoder and atomic output boundary. The runner has no
checkpoint, trained-head, official-test, network-write, or benchmark-publication
option. The canonical result uses schema `sfora-siglip-band-audit-v1`, sorted
compact JSON plus one LF, `claim_eligible=false`, and
`official_test_access=false`.

Descriptor, label, ordered-example-ID, class-name, source-tree, model revision,
and result-file digests are independently authenticated. Output is written via a
new partial file, fsynced, atomically renamed, directory-fsynced, read back, and
byte-compared. A failure publishes no result.

The implementation is TDD-first, receives focused scalar/vector and mutation
coverage, direct-script refusal tests, repository-wide Python assurance, and an
independent read-only review. Scientific execution remains serialized behind the
sole three-seed DGX control.
