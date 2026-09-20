# MET-small noise-shaped PQ decoded-norm attribution

## Question

The paired native PQ128x4 result at commit `825544da` shows a positive,
seed-stable dot-product effect and a significant negative squared-L2 effect on
the same packed codes. This one fixed diagnostic asks whether the decoded
reconstruction-norm term causes the reversal.

This is the only scorer-compatibility diagnostic licensed by the prior
preregistration. It does not tune the threshold, codebooks, block geometry,
seeds, payload, or queries.

## Frozen authority and intervention

Reuse every authority and construction rule from
`docs/met_small_native_noise_shaped_pq128x4_preregistration_2026-09-20.md`:

- feature SHA-256
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- 35,257-row reduced gallery, 3,050 powered pseudo-queries, 129 secondary
  official-validation queries;
- seeds `{50,51,52}`;
- conventional paired codebooks, 128 width-6 blocks, 16 centers, threshold
  `0.2`, ten coordinate rounds, and exact 64-byte low-nibble-first codes;
- dot and unnormalized squared-L2 scorers unchanged;
- updated diagnostic driver SHA-256
  `16ff376a645bd1eb156eaad3afdcb9f20acdf7b6bb7c641a0eed5df1bb6c2203`.

Add exactly one scorer, `normalized_squared_l2`: after unpacking and decoding,
L2-normalize every float32 gallery reconstruction, independently normalize each
query, then exhaustively rank by squared L2 with stable lowest-gallery-ordinal
ties. Do not refit or re-encode after normalization. For unit vectors this
removes the gallery norm term and is order-equivalent to cosine similarity.

Highest float32 matmul precision remains required and TF32 remains disabled.
The driver self-test must distinguish all three scoring geometries on an
unequal-reconstruction-norm fixture.

## Frozen decision

Apply the prior powered-proxy mechanism, bootstrap, Recall@1, seed-floor, and
descriptive official-validation rules independently to
`normalized_squared_l2`.

`normalization_resolves_scorer_mismatch=true` only if:

- dot product retains `promotion_supported=true`;
- unnormalized squared L2 retains `promotion_supported=false`; and
- normalized squared L2 has `promotion_supported=true`.

If true, attribute the prior reversal to decoded reconstruction norms and
promote a design decision about normalized/cosine scoring at the public 64-byte
codec boundary. If false, normalization does not explain the incompatibility;
close this noise-shaped-assignment route without a threshold, seed, geometry,
or scorer sweep. In either case `generic_supported=false`, and no production or
custom-kernel work is licensed by this diagnostic alone.
