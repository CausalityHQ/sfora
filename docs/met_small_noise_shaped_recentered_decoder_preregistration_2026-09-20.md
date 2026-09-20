# MET-small noise-shaped PQ conditional-mean decoder control

## Question

The native 64-byte noise-shaped PQ experiment changed roughly one third of the
hard assignments but decoded them with the original isotropic k-means centers.
That decoder is not the conditional mean under the changed assignment rule.
Does one closed-form M-step—replacing each center by the mean of the gallery
rows assigned to it while keeping every packed code fixed—recover the deployed
squared-L2 retrieval loss?

This is an exploratory causal control prompted by an independent review after
the normalization attribution had closed the original promotion path. It is
not a reopening of threshold, seed, block, payload, or scorer selection.

## Frozen protocol

- Reuse feature archive SHA-256
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`.
- Reuse seeds 50, 51, and 52; 20 isotropic k-means iterations; ScaNN-compatible
  threshold `0.2`; at most ten noise-shaping coordinate rounds; 128 width-6
  blocks; 16 centers; and the exact 64-byte low-nibble-first packed codes.
- For each seed and block independently, compute each replacement center as
  the deterministic float64 mean of all 35,257 gallery rows carrying that
  noise-shaped assignment. An empty cell retains its original center.
- Do not reassign a row after recentering. This isolates decoder calibration;
  another assignment or centroid round is forbidden.
- Score the unchanged packed codes with raw dot, squared L2, and normalized
  squared L2 against the same 3,050 powered pseudoqueries. The 129 official
  validation queries remain descriptive shift evidence only. The external
  official test remains sealed.
- Report conditional-mean codebook digests, empty-cell counts, reconstruction
  diagnostics, per-query metrics, all three seed deltas, and the paired-query
  bootstrap interval. No custom CUDA kernel is used.

The matched full-precision powered-proxy source is `0.590541` mMP@5 /
`0.523934` Recall@1. Conventional PQ128x4 seed 50 under squared L2 is
`0.570426 / 0.502951`, leaving approximately `0.0201 / 0.0210` absolute room.

## Frozen decision

`recentered_decoder_supported` is true only when recentered squared L2, relative
to the same seed's conventional PQ, has:

- mean powered-proxy mMP@5 delta at least `+0.002`;
- a paired-query bootstrap 95% lower endpoint above zero;
- positive mMP@5 delta for all three seeds;
- mean Recall@1 delta at least `-0.003`; and
- mean mMP@5 delta above the conventional-PQ seed spread.

Otherwise this decoder explanation is rejected and fixed-codebook
noise-shaped PQ stays closed. A pass is only exploratory evidence that
assignment and decoder must be co-estimated; it permits a separately frozen
prospective replication, not immediate public-codec promotion.
