# RP2K label-sparsification causality diagnostic

## Question

The fresh MET-small screen showed that the supervised compact projection is
competitive with OPQ but does not clearly exceed it.  This diagnostic asks
whether that cross-domain divergence is caused by sparse positive-label
coverage rather than by different visual geometry.  It reuses only the
already-observed RP2K fit and validation features; it makes no new test claim.

## Immutable authority

- RP2K feature archive SHA-256
  `46cfbd7dea5592de7fcbdfedd7a3b5bc3960160b853d4e484c46ac07b31abbc7`:
  15,266 fit rows from 1,074 classes and 17,185 class-disjoint validation
  rows from 120 classes.
- Candidate module SHA-256
  `d6a644476ee6b2b8486876e770eb103bdf478197234208da7e00fbf081850654`.
- Dense-label learned int8-64 reference: mMP@5 `0.9698040935685893`, R@1
  `0.9842304335176025`, from authenticated result SHA-256
  `f15a17eb26b809a00fded5fb8dde13d06ec2c1a42dfe8b07d88a2d5bb200bb5a`.

## Deterministic intervention

Within each original RP2K fit class, sorted source-row ordinals zero through
two retain the original label.  Every later row receives a distinct singleton
label derived from its source-row ordinal.  Existing classes with fewer than
three rows are unchanged.  This leaves 3,214 positive-eligible rows in 1,072
classes, or 21.053% of all fit rows, close to MET-small's measured 20.507%
eligible-row coverage.  Embeddings, validation rows and labels, PCA
initialization, negative bank, byte width, quantizer, scorer, and tie rule are
unchanged.

Two arms separate supervision coverage from optimizer exposure:

1. `sparse_native` uses the unchanged default schedule: 135 updates per cycle,
   four cycles.
2. `sparse_matched_updates` changes only `anchor_epochs_per_cycle` to
   `25.486350645408145`, producing 640 updates per cycle, equal to the dense
   reference schedule.

Both output 64 signed bytes per validation item and are scored with the exact
RP2K mMP@5 and R@1 implementation used by the dense reference.

## Frozen interpretation

Let mMP drop mean dense-reference mMP minus sparse-arm mMP.

- `optimizer_exposure_supported`: native drop is at least `0.003`, while the
  matched-update drop is at most `0.002`.
- `positive_coverage_supported`: both drops are at least `0.003`.
- `density_hypothesis_falsified`: both drops are at most `0.002`.
- all other outcomes are `mixed`.

R@1 deltas are reported but do not select the classification.  No threshold
or arm may be changed after execution.  This diagnostic cannot promote a
method or reveal any held-out test; it only determines the next generic
algorithmic boundary.
