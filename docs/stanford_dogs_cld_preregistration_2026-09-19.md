# Stanford Dogs CLD Fresh Falsifier

This is a one-shot, claim-ineligible engineering falsifier frozen before the
Stanford Dogs archives are downloaded or inspected. It tests whether the fixed
candidate-local discriminant result transfers beyond SOP, In-Shop, and CUB.

## Data and split

- Authorities: the Stanford Dogs `images.tar` and `lists.tar` published at
  `vision.stanford.edu/aditya86/ImageNetDogs/`.
- Enumerate the 120 breed directory names from the official train and test
  lists, order them by `(sha256(name), name)`, and assign the first 60 to fit
  and the last 60 to evaluation.
- Use only official-train images of fit breeds for head/covariance fitting.
  Use only official-test images of evaluation breeds for same-set retrieval.
- The frozen UNICOM ViT-L/14@336 teacher and the existing default compact
  projection recipe produce 128-dimensional unit-normalized int8 codes.

## Frozen scorer

- Exact cosine top-128 candidates with deterministic ordinal ties.
- Normalized candidate-local DBA: five neighbours and 0.5 mixing.
- Positive mean: query and rank-one candidate.
- Local negatives: ranks 41 through 128 inclusive.
- Local covariance shrinkage target: trace-normalized within-class covariance
  from fit codes only. `lambda = trace(local_covariance) / 128`.
- Final score: equal sum of population-z-scored CLD and DBA scores.
- Reorder only inside baseline bands induced by cutoffs `{1, 10, 100}`.

## Fixed controls and gates

Controls are DBA, identity-prior CLD+DBA, positives-only, global-prior
mean-difference+DBA, and globally WCCN-whitened codes+DBA. The run passes only
if CLD+DBA improves mAP@R over DBA by at least `0.002`, beats every control,
and preserves the exact member set at R@1, R@10, and R@100 for every query.
Failure ends this scorer direction; no constant may be changed on this data.

All results remain claim-ineligible because Stanford Dogs is not a standard
DML reporting protocol and this run lacks independent custody. A pass supports
building a sealed recognised-benchmark or private-catalog confirmation; it is
not itself a SOTA or publication claim.
