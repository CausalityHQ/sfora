# Matched-Control Native Twin Probe Design

## Purpose

The deployed Cars196 result is 92.342%, with 63 of 103 errors concentrated in
the Dodge Caliber 2007/2012 pair. Training reaches roughly 98% on its
optimization band but only 94.54--94.61% on clean validation. Another loss on
the same resized descriptor cannot distinguish model failure from missing
pixels. This claim-ineligible probe asks the narrower causal question: **does
source resolution add pair-discriminating information after view geometry and
scoring are held fixed?**

The probe uses only the already burned class-82/83 diagnostic population. It
never reads the clean-validation or official-test bands.

## Three frozen planes

Each authenticated source image produces three descriptor planes through the
same authenticated checkpoint and processor:

1. `global-384`: the whole RGB image through the existing 384px processor. It
   is context only and never supplies the causal baseline.
2. `crop-control-256`: nine fixed, row-major, overlapping two-thirds crops.
   A crop above a 256px long edge is downsampled with Pillow bicubic
   interpolation to a 256px long edge, then restored to its original pixel
   dimensions with the same filter before the 384px processor. A crop already
   at or below 256px is copied byte-for-byte and contributes no artificial
   contrast.
3. `crop-native`: the identical nine crop boxes passed directly through the
   same processor.

Thus control and native planes have identical examples, crop geometry, input
dimensions, view count, checkpoint, processor, descriptor count, aggregation,
and scorer. Their intended contrast is the spatial frequency removed by the
registered downsample/restore ablation. No detector, label,
retrieval outcome, PRISM result, or evaluation metric may choose a crop.

## Scoring and paired inference

Every view is a finite unit-normalized fp32 descriptor. The nine descriptors
are concatenated in frozen row-major crop order and normalized once. Each plane
uses leave-one-out cosine nearest-neighbour retrieval. Self candidates are
excluded. Exact score ties use the candidate RGB SHA-256, not example ID, so a
class-sorted manifest cannot bias ties.

The causal inference is the exact paired retrieval outcome, not a fitted
high-dimensional classifier. A 10,000-draw, authority-seeded randomization
swaps the complete control/native descriptor assignment independently within
each image, recomputes both coupled leave-one-out retrieval sets from four
precomputed Gram planes, and compares the error improvement. This tests the
full coupled retrieval procedure without assuming query-wise independence.
Global-384 retrieval is recorded only as context.

## Authority

The authority binds source identity, checkpoint SHA-256, model revision,
ordered unique example IDs, ordered unique decoded-RGB SHA-256 values, labels,
and exact global/control/native descriptor SHA-256 values. It also binds the
probe revision/tree and every crop long edge, making no-op control crops
auditable. Descriptor digests frame rank, shape, and canonical little-endian
fp32 bytes. Validation recomputes all descriptor digests, predictions, paired
counts, randomization probability, and gates.

## Causal gate

For each row, a rescue is control-wrong/native-correct and a harm is
control-correct/native-wrong. The one-sided McNemar binomial tail is recorded as
a descriptive paired statistic, but it is not a gate because leave-one-out
queries share candidates. The authoritative gate uses the fully recomputed
within-image plane-swap randomization above. No unstable error-ratio bootstrap
is used.

`native-pixel-cue-pass` requires all of:

- control has at least one error;
- native errors are at least 25% lower than control errors;
- native balanced accuracy is at least 0.75;
- rescues exceed harms;
- one-sided recomputed plane-swap randomization `p <= 0.05`;
Global-384 performance is recorded but cannot make the causal gate pass.
Canonical output is sorted JSON with one trailing LF and
`claim_eligible=false`.

## Decision

Run the immutable cell on seed 17, then seed 29, then seed 43 after the existing
campaign terminates. A replicated pass warrants a confusion-gated native pair
verifier and only then custom crop/attention kernels. A failure rules out this
fixed 1.5x native-grid discriminator; it does not prove all native resolution,
taxonomy, or data improvements impossible. PRISM and a data audit remain
independent evidence paths.
