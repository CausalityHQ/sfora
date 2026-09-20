# Compact Teacher-Error Census and Local-Scatter Result (2026-09-20)

## Outcome

The sealed error census showed that the signed-int8 128-D power codes are not at
the float-768 teacher's top-1 ceiling. The fraction of compact-code errors also
made by the teacher was 77.9% on Cars, 81.6% on Food101, 87.6% on InShop, and
78.1% on Pet. That leaves 46, 138, 95, and 14 compact-only errors respectively.
Wrong-target in-degree was not generally elevated: it was lower than the
correct-target in-degree on Cars, Food101, and Pet, and higher only on InShop.
This rejected hubness correction as the general next mechanism and licensed the
pre-registered local-scatter falsifier.

The fit-only local-scatter screen then failed. Its one common selected arm,
local-within plus local-between scatter, improved macro Recall@1 by only
0.001696 against the sealed global power incumbent, below the 0.003 gate. It
also lost mAP@R by 0.027593 on Cars, 0.013612 on Food101, and 0.010939 on Pet.
It improved Recall@1 on InShop by 0.004059 and Pet by 0.005294, but those gains
did not transfer without damaging ranking quality. No outer metric was read.

This closes the compact affine/local-scatter lane without a neighbour-count
sweep or per-dataset fallback. A subsequent experiment must add genuinely
nonlinear structure or new image information while retaining the 128-byte item
code and protecting both the best mAP@R and best Recall@1 incumbents.

## Authority

- Census source checkpoint: `fc05d45cc2cacb8c25ab2df4833d3b9bb24cef73`
- Census result SHA-256:
  `5bff90d6bbaa4b29710c2359c322cfbb42f32d5a40083ae854c54d00b5736adf`
- Local-scatter source checkpoint:
  `a3ebc881`
- Local-scatter result SHA-256:
  `696e450e4a9f651b6a2be6d7f57ab022bc26b67693cedf4da4650d89c1a766ef`
- Census runtime: 16.56 seconds on the DGX GB10.
- Local-scatter runtime: 14.08 seconds on the DGX GB10.
- Both results are development-only and `claim_eligible=false`.

Canonical result payloads are retained at
`docs/evidence/compact_metric/compact-teacher-error-census-v1/result.json` and
`docs/evidence/compact_metric/local-scatter-power-v1/result.json`.
