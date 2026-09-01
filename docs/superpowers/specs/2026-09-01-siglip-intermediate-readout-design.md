# SigLIP Intermediate-Readout Depth Diagnostic Design

Date: 2026-09-01

## Purpose

The frozen-substrate ladder tested SigLIP-so400m only at
`vision_pooler_output`. That does not establish that the final layer/readout is
the best fine-grained descriptor exposed by the pinned tower. This diagnostic
tests whether an earlier transformer block retains vehicle-detail evidence that
the final caption-oriented representation suppresses.

It changes neither supervision nor retrieval: every arm emits one
512-dimensional vector and uses exact cosine leave-one-out Recall@1. It is an
optimization-only, claim-ineligible readout screen over the sealed seed-17
control checkpoint. A pass authorizes one clean-band evaluation of one sealed
depth; it does not establish a new-method claim by itself.

## Authority and isolation

The command consumes the exact completed three-seed control receipt, seed-17
epoch-60 checkpoint, optimization-only manifest, and optimization-band images.
It refuses clean, burned, official-test, network, and arbitrary checkpoint
capabilities. The checkpoint is restored through the existing strict control
loader and must bind the pinned model/dataset/source identities.

The process validates exactly 27 encoder blocks and width 1152 before image
execution. It uses the checkpoint's trained bias-free 1152-to-512 projection
unchanged. No parameter, buffer, depth, normalization, or pooling operator is
fit from retrieval outcomes.

## Registered readouts

For each image, request encoder hidden states once. For depths 1 through 27:

1. take the complete patch-token matrix emitted by that block;
2. apply the vision transformer's existing final post-layer normalization;
3. average every token with uniform weight;
4. apply the sealed seed-17 projection; and
5. L2-normalize in float32.

Depth 27 under the same post-LN/mean/projection operator is the primary
comparator. The existing attention-pooler descriptor is recorded as context but
is not the operator-matched selection baseline. A secondary no-post-LN table is
recorded to expose normalization artifacts and cannot select a depth.

Hidden states are consumed batch-by-batch. The implementation stores only the
27 final 512-dimensional descriptors per optimization image, never the full
token cache. Peak persistent descriptor storage is below 0.5 GiB.

## Folds, selection, and gates

Reuse the exact four-fold `build_sfq_fold_schedule` on optimization classes.
For every depth and fold, compute integer Recall@1 among that fold's held
classes using the existing `score_frozen_substrate` authority with lowest-row
ties. Aggregate from summed hits.

Select the unique depth by `(aggregate_hits descending, depth ascending)`.
The diagnostic passes only when:

- the selected depth improves aggregate operator-matched depth-27 Recall@1 by
  at least 10,000 ppm (1.0 percentage point);
- it strictly beats depth 27 in at least three of four folds;
- all descriptor norms, dimensions, counts, and hidden-state topology are
  exact and finite; and
- a repeated scalar scoring pass reproduces every integer hit count.

If depth 27 wins or improvement is below 1.0 point, the earlier-readout
hypothesis is rejected. If an earlier depth passes, seal its depth, descriptor
digests, fold counts, checkpoint identity, and operator contract before one
clean read. No depth ladder may be revisited after clean evidence.

## Files and verification

- `src/sfora/siglip_intermediate_readout.py`: topology, descriptor scoring,
  selection, gates, and canonical result validation.
- `scripts/diagnose_siglip_intermediate_readout.py`: strict local scientific
  boundary and streamed hidden-state extraction.
- `tests/test_siglip_intermediate_readout.py`: selection, ties, gates,
  topology, mutation, and deterministic scoring tests.
- `tests/test_diagnose_siglip_intermediate_readout.py`: checkpoint/input
  capability and streamed integration tests.

Implementation is TDD-first. The scientific run remains fenced until the
active three-seed control reaches a terminal receipt and no other GPU process is
active. Expected execution is under one GPU-hour.
