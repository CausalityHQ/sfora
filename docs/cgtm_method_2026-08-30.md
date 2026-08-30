# Correspondence-Gated Token-Maxima (CGTM)

Status: **candidate; F-1 error taxonomy required before implementation.**

## Provenance

The frozen-substrate ladder failed, so a head-only method is not authorized.
TSPA supplied a sharper causal measurement: on 1,345 burned Cars train images,
pooled SigLIP retrieval reached `0.927881`, while ungated cross-image token
MaxSim fell to `0.855019`. Local maxima are therefore noise-dominated, but this
does not show that local evidence is absent. It shows that the reader cannot
distinguish corresponding vehicle parts from accidental maxima.

## Proposed mechanism

CGTM fine-tunes SigLIP-so400m@384 with an occupied pooled control loss plus one
new training-only operator. For each same-class image pair, a stop-gradient EMA
teacher identifies mutual-nearest-neighbour token matches. A match is admitted
only when the token round trip is cyclically consistent and its spatial
displacement is within the registered tolerance. The student then:

1. pulls admitted corresponding token pairs together; and
2. places a margin between each admitted match and the largest unmatched
   cross-image token similarity.

The gate therefore decides which same-class local maxima are evidence and which
must be suppressed. Deployment remains one pooled descriptor; no tokens,
teacher, correspondence search, or reranking survive inference.

Closest occupied families to clear before implementation are FILIP/TokenFlow
and TGDT token alignment, DenseCL/DINO dense self-supervision, and part-aligned
ReID. The proposed distinct decision point is the cyclic-consistency gate that
selects which supervised cross-instance maxima are trained up versus pushed
down while retaining pooled-only deployment. If the Gate-2 review cannot defend
that distinction, CGTM dies before code.

## F-1: zero-GPU error taxonomy

Reproduce the exact 103 errors from the sealed SigLIP-so400m band receipt
(`ba2d0fc...`) and record query/nearest-neighbour image IDs and class names.
Classify every pair, without changing the model, as:

- same-make adjacent-year/body-trim twin;
- same-body-type different make;
- extreme viewpoint;
- occlusion/crop;
- plausible mislabel or visually indeterminate.

The classifier and counts are descriptive and claim-ineligible. Decision rule
frozen before inspection:

- at least 50 twin-or-same-make errors: CGTM proceeds to Gate 2;
- viewpoint is the plurality: prefer the separately reviewed pair-comparator
  candidate;
- at least 40 plausible-mislabel/indeterminate errors: stop and report that the
  97.4 target lacks evidence at this substrate.

## If F-1 and Gate 2 pass

The paired F0 trains on Cars classes `{1..81, 98}` only and evaluates classes
`82..97`; the Cars test split remains forbidden. Control is LP-FT pooled loss
with EMA under the exact same schedule. CGTM must beat control by at least 1.0
R@1 point, be positive in 6/6 paired seeds, close at least half of TSPA's 7.29
point token-MaxSim deficit, and beat a matched-compute random-correspondence
placebo by at least 2x. Below +0.5 point, the method is falsified.

Training overhead is `Theta(P T^2 d)` for `P` same-class pairs, `T` tokens, and
token dimension `d`; correspondence matrices are training-only. Evaluation
remains `Theta(ND)` storage and ordinary pooled cosine search.
