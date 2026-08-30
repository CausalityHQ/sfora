# Correspondence-Gated Token-Maxima (CGTM)

Status: **REJECTED at Gate 2; no implementation or training authorized.**

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
teacher identifies mutual-nearest-neighbour token matches. Mutual-nearest
matching is the two-image cycle test; it is one condition, not two tunable
conditions. A match is admitted only when it repeats under a second
geometry-preserving weak photometric view and its spatial displacement is within
the registered tolerance. The student then:

1. pulls admitted corresponding token pairs together; and
2. places a margin between each admitted match and the largest unmatched
   similarity sharing either endpoint's row or column.

The gate therefore decides which same-class local maxima are evidence and which
must be suppressed. Deployment remains one pooled descriptor; no tokens,
teacher, correspondence search, or reranking survive inference.

The nearest occupied retrieval methods are DIML and CFCD, not merely the broad
FILIP/TokenFlow, DenseCL/DINO, and part-aligned ReID families. DIML already lets
cross-image structural matching define a DML similarity. CFCD already uses
prominent local descriptors, reciprocal matching, and hard negatives to train a
compact single-stage global representation. More decisively, cross-view-stable
same-class pseudo-correspondences are established semantic-correspondence
supervision, while raising an admitted entry above unmatched row/column entries
is an established local-feature matching objective. Combining these occupied
operators does not create a new supervision object.

The repository had also already rejected Candidate 6, which used detached
same-class mutual-nearest patches as auxiliary positives under an unchanged
global Proxy Anchor objective, and separately rejected a cross-instance cycle
gate. CGTM changed filters and the treatment of unmatched entries but still let
cross-image correspondence determine positive DML similarity. It is therefore a
delta on an occupied and locally rejected candidate, not a defensible new
method. The full audit is in `docs/cgtm_gate2_dossier_2026-08-30.md`.

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

## Superseded training sketch

The following split correction is retained only to prevent the original label
bug from being copied into another candidate. It is not an authorization to run
CGTM.

Labels are zero-indexed in this repository. The paired F0 trains on Cars
classes `{0..81}` only and evaluates classes `{82..97}`; the Cars test split
`{98..195}` remains forbidden and must never be instantiated. Control is LP-FT pooled loss
with EMA under the exact same schedule. CGTM must beat control by at least 1.0
R@1 point, be positive in 6/6 paired seeds, close at least half of TSPA's 7.29
point token-MaxSim deficit *only after that deficit is remeasured with the same
SigLIP-so400m@384 full-token operator*, and beat a matched-compute
random-correspondence placebo by at least 2x. Below +0.5 point, the method is
falsified.

Training overhead would include the teacher backbone forwards as well as token
matrices; the earlier `Theta(P T^2 d)` statement omitted the dominant teacher
forward cost. No CGTM preflight, implementation, fine-tuning, or terminal test
is authorized.
