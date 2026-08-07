# Pass 84 proposal: Positive-Exchange Bilinear Head (PEBH)

## Gate 1 — repository measurement

Across four corrected BN-Inception/512-D Proxy Anchor seeds, exactly-three-peer
same-identity positives had unseen-minus-seen gaps
`-0.049281, -0.050450, -0.049061, -0.049930` (mean `-0.04968`). Only 44.7%
of learned within-class contraction transfers to unseen identities. Foreign
crowding disappears under the BN-buffer placebo, so the target is positive
transfer rather than negatives. Cross-seed error overlap is `0.76754` and
between:local error ratios are `3.05, 3.11, 3.22, 3.15`.

## Gate 2 — mechanism and prior-art distinction

For pooled trunk feature `h_i`, learn `a_i=A h_i`, `b_i=B h_i`, and
`z(i→j)=C(a_i ⊙ b_j)`. During training, each identity supplies a random pair
and Proxy Anchor is applied to both the self descriptor `z(i→i)` and the
positive-exchanged descriptor `z(j→i)`. At inference only `z(i→i)` is used,
so deployment is one ordinary 512-D vector and cosine retrieval.

Mechanism-level claim: PEBH supervises computational exchangeability—a
selector emitted by one view must gate evidence from another same-identity
view, while the same program folds onto one image at deployment. Bilinear CNNs
cover self-products; Dynamic Filter Networks cover input-conditioned filters;
CADC covers group-conditioned cross-image kernels; DMML covers support/query
episodes. The novelty kill condition is a pre-existing paper that combines
distinct same-label peer exchange during training with self-only open-set
descriptor deployment. The review found no such exact combination.

## Gate 3 — preregistration and CPU kill test

Use only the four corrected training packs, never query/gallery. Split identities
by `SHA256(label) mod 5`; fit a width-128 PEBH probe on four folds and evaluate
held-out identities. Compare with a parameter-, optimizer-, and batch-matched
self-only bilinear control. Require, across all four seeds, same-sign gain and:

* nearest-positive gain ≥ `+0.0050`;
* positive transfer-gap magnitude reduction ≥ `20%`;
* leave-one-out R@1 degradation ≤ `0.0010`;
* nearest-foreign increase ≤ `0.0020`.

Any failure kills PEBH before GPU. This probe tests exchangeability rather than
another positive-tail weighting.

## GPU prediction and falsifiers

Seed-0 prediction: frozen-final R@1 `0.9200` versus archived baseline
`0.9137009425`; positive gap `-0.0345` versus `-0.049281`. Kill if R@1 is
below `0.9132182`, paired gain is below `+0.0031`, the gap remains at or below
`-0.039425`, or a parameter-matched self-only bilinear head is within `0.001`
R@1 (capacity, not exchange, would explain the result).

If the CPU gate passes, run three matched In-Shop seed-0 arms: corrected PA
with the same two-per-class sampler, self-only bilinear control, and PEBH.
Preserve BN-Inception, 512-D output, optimizer, schedule, augmentation,
batching, and frozen-final reporting; report raw best-over-training and
selection-corrected values.
