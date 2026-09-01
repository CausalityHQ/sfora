# SigLIP Class-Disjoint Gradient-Agreement Diagnostic Design

Date: 2026-09-01

## Purpose

The pooled SigLIP Proxy Anchor control reaches 97.9813% Recall@1 on its
optimization classes but only 94.5375% on class-disjoint clean validation. The
existing cached-head screen fits every optimization class together, so it does
not test whether its learned projection transfers to unseen classes. This
diagnostic tests a narrower mechanism: updates induced by different groups of
seen classes conflict in the shared projection, and removing their conflicting
component improves leave-class-out retrieval.

The diagnostic is optimization-only, deterministic, cached-feature based, and
claim-ineligible. It cannot read clean-validation, burned-diagnostic, or
official-test feature files. A pass may authorize one separately frozen clean
evaluation; it is not a SOTA claim.

## Inputs and isolation

The CLI consumes the authenticated SigLIP cached-feature manifest and its
expected SHA-256, authenticates the manifest and registered source commit, and
opens only the `optimization-train` feature file. The library accepts one
finite CPU `float32` matrix, one CPU `int64` label vector, and a validated
`FeatureSplitAuthority` whose role is `optimization-train` and whose
`official_test_access` is false.

The executable has no band selector, network client, output-dimension knob, or
path for evaluation features. The result binds the cache/control manifests,
ordered examples, features, labels, schedule, seeds, projections, and integer
retrieval counts. Canonical result bytes are the external authority.

## Class-disjoint folds and batches

Reuse `build_sfq_fold_schedule(..., fold_count=4)`. It deterministically keeps
nearest class-mean pairs together and assigns every optimization class to
exactly one validation fold. For each fold, all fitting uses only the other
three folds; Recall@1 uses only the held classes.

Within a fit partition, sort labels and deterministically split them into two
pseudo-domains by alternating labels after a seed-derived cyclic rotation.
Each training step draws one registered class-balanced batch from each
pseudo-domain using the existing stateful sampler. Every example selected for a
batch must have a fit label; no validation label may enter an optimizer step.

## Comparator and CDGA update

Both arms start from the same deterministic uncentered 512-dimensional spectral
projection and the same class proxies. Both use the existing Proxy Anchor loss,
AdamW hyperparameters, step count, batches, and proxy updates. The comparator
sums the two pseudo-domain losses and performs the ordinary update.

CDGA changes only the shared projection gradient. Let `g_a` and `g_b` be the
flattened projection gradients from the two pseudo-domain losses. If
`dot(g_a,g_b) < 0`, replace them symmetrically with

`g'_a = g_a - dot(g_a,g_b) / max(||g_b||^2, eps) g_b`

`g'_b = g_b - dot(g_a,g_b) / max(||g_a||^2, eps) g_a`.

Otherwise retain both gradients. The installed projection gradient is
`(g'_a + g'_b) / 2`. Proxy parameters receive the ordinary mean of the two
domain-loss gradients; only the shared projection gradient is projected. The
fixed epsilon is `1e-12`. Nonfinite losses, gradients,
denominators, or parameters fail closed. The result records conflict count,
mean pre-projection cosine, and projection digests.

This is deliberately a first-order conflict-removal diagnostic, not MAML. It
tests whether a class-generic direction exists without adding a second-order
compute confounder.

## Evidence and decision

For every fold, score raw pooled, spectral initialization, ordinary comparator,
and CDGA projections on the held classes using exact lowest-row tie handling.
Recompute aggregate Recall@1 from integer hits.

The diagnostic is valid only if every fold completes, both arms reduce their
fit loss, comparator optimization Recall@1 is at least its spectral baseline,
all evidence is finite, and at least one conflicting projection-gradient pair
was observed. It passes only if valid and:

- CDGA aggregate held-class Recall@1 exceeds the ordinary comparator by at
  least 2,000 ppm;
- CDGA is not below the spectral arm; and
- no fold loses more than 10,000 ppm relative to the comparator.

A failure rejects this exact four-fold, symmetric gradient-projection method on
the cached pooled representation. It does not reject class-domain
generalization or tower-level meta-learning. A pass authorizes one clean
evaluation of the sealed projection recipe, after SFQ and without retuning.

## Files and verification

- `src/sfora/siglip_cdga.py`: authority, pseudo-domains, gradient projection,
  training, fold scoring, and canonical result validation.
- `scripts/diagnose_siglip_cdga.py`: strict optimization-only local CLI.
- `tests/test_siglip_cdga.py`: arithmetic, isolation, determinism, training,
  evidence, and mutation tests.
- `tests/test_diagnose_siglip_cdga.py`: loader and CLI capability tests.

Implementation follows focused RED/GREEN cycles, dependency-complete Python
tests, Ruff, format, `py_compile`, `git diff --check`, and independent read-only
review. No scientific run begins while the three-seed DGX control is active.
