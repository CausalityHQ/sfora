# Current evidence reliability boundary (audit 321)

Date: 2026-08-03.

## Decision

The method-search catalogue is not a uniformly trustworthy empirical dataset.
It contains experiments run on the wrong In-Shop pixel corpus, SOP artifacts
with the wrong benchmark split or best-test selection, a retracted
selection-bias estimator, and candidates affected by implementation defects.
Those numbers must not be pooled, ranked, or used as candidate provenance.
Prior-art-only deaths remain valid because they do not depend on the affected
measurements.

From candidate 321 onward, an empirical premise is eligible for Gate 1 only if
its artifact selection, dataset membership, model architecture, and metric have
been independently recomputed or prospectively functionally validated.  A
green unit test or a configuration label is not enough.

## Quarantined evidence

- Every In-Shop score produced from `img_highres` is not benchmark evidence.
- Historical SOP conclusions produced from the non-official split or from
  training embeddings saved at the best-test epoch are withdrawn.
- The leave-neighbour peak-gap statistic is not a selection correction and may
  not be reported as one.
- Any candidate result whose deciding mechanism was affected by an established
  objective, buffer, dispatcher, representation-stage, or evaluation bug is not
  reusable as a quantitative prior.  A separately established prior-art death
  may still stand.
- CUB/Cars and other historical arms are not automatically promoted to the
  verified tier merely because the two known dataset bugs were In-Shop/SOP.
  Their exact artifact and current-code path must be audited before reuse.

## Verified corrected In-Shop reference

The replacement 256-pixel official In-Shop corpus is functionally validated by
the Proxy Anchor authors' published BN-Inception checkpoint: prospectively
registered inference produced R@1 **0.9176396118**, within the locked
`[0.917, 0.921]` interval, with identical upstream, float64 Euclidean, cosine,
and exact-tie scorers.

The local seed-0 reference is bound to checkpoint SHA-256
`2b46a68a0364cd204e60068858198f1da699f043897fc93d0c22525b6f635546`.
Its state dict contains BN-Inception keys, its architecture declares
`bn_inception` and 512 dimensions, its proxy table is `3997 x 512`, and its
artifact selection is the final training state at step 8,580.  The report gives
raw best-over-training R@1 **0.9163032775** at epoch 41 and independently
exported final R@1 **0.9137009425**.  The final exporter verifies the exact
25,882/14,218/12,612 official partitions, train/test identity disjointness,
source-path and byte-content disjointness, and agreement of three independent
nearest-neighbour scorers.

The prospectively registered seed-1 reference is independently verified as
well. Its raw best-over-training R@1 is **0.9189056126** at epoch 46 and its
frozen-final R@1 is **0.9167956112** at step 8,580, both within their locked
intervals. The checkpoint SHA-256 is
`a25dc22691981e6ad7df899878f448d96d4ac41adbb8e346e10322e93883e580`.
Production, float64 Euclidean, float64 cosine and exact-tie scorers agree
exactly. Its official partition/content manifests equal seed 0, with zero path,
content or identity leakage. The two-seed raw/final means are **0.9176044451**
and **0.9152482768** respectively. This adds a second artifact, not a defensible
variance estimate.

The config field `protocol="proxy-anchor-resnet50-512"` is misleading legacy
schema metadata.  It is not the executed architecture: both the checkpoint and
result identify BN-Inception, and the published-checkpoint functional test
exercises the same vendored BN-Inception.  Future reports must use
`config.backbone_name`, checkpoint `arch`, recipe ID/digest, and state keys as
the architecture authority; the protocol-family string must never be used to
claim which backbone ran.

These are two local seeds. They remain paired references, not a variance
estimate and not evidence that a small one-seed gain generalizes.

## Independent recomputation of the surviving descriptive signal

An inline NumPy/Torch audit that did not import `sfora.cem` loaded the final
training pack and checkpoint, asserted the model/artifact invariants above, and
recomputed all nearest neighbours in float64 cosine chunks.  It reproduced:

- pack SHA-256
  `67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`;
- training leave-one-out error **0.0049841589**;
- nearest-foreign-image / nearest-foreign-proxy class agreement
  **0.1569044123** (4,061 / 25,882);
- error given agreement **0.0238857424**;
- error given disagreement **0.0014664772**.

This validates the diagnostic as a descriptive corrected-corpus measurement.
It does not validate CEM as a novel method.  CEM's gradient reduces to
hard-negative-class mining, so candidate 320 remains dead independently of all
performance numbers.

## Operational rule

The next Claude or human candidate critique receives only a compact verified
evidence packet derived under this boundary, not the entire historical verdict
as if every number were commensurate.  Before any new GPU run, Gate 0 must
include a second implementation or scorer for the deciding premise, followed by
the mechanism-level Gate-2 reduction.
