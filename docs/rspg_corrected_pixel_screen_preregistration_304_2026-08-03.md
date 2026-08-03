# Corrected-pixel RSPG screen preregistration

Status: **REGISTERED BEFORE EITHER CORRECTED-PIXEL RSPG RETRIEVAL RUN.**

## Evidence boundary

All earlier In-Shop RSPG and Proxy Anchor retrieval scores used the wrong
`img_highres` parsing/segmentation corpus and are quarantined.  On the acquired
standard 256x256 official partition, plain Proxy Anchor seed 0 reached raw
best-over-training R@1 **0.91630328** (epoch 41) and independently exported
final-state R@1 **0.91370094**.  Float64 cosine, float64 Euclidean and exact-tie
expected scorers agree at the final state; no train/query/gallery source overlap
or cross-identity content-duplicate contamination was found.

The exact epoch-10 corrected-pixel operating pack then passed RSPG's unchanged
CPU gate: **12,791 / 153,115 edges**, density **0.0835**, and multi-component
class fraction **0.8866**.  This confirmation is contaminated as a decision path
because a favourable epoch-10 result on the former pixels was known before the
operating-point rerun was chosen.  It therefore authorizes two full seeds, not a
one-seed claim.  It does not itself predict retrieval quality.

## Numeric prediction and falsification

RSPG is predicted to reach mean raw best-over-training R@1 **0.9190** across
corrected-pixel In-Shop seeds 0 and 1.  This is a deliberately small but positive
prediction over both the corrected seed-0 reference and the authors' independently
validated 0.91764 checkpoint.

The candidate is falsified at Gate 4 if **either** corrected-pixel RSPG seed has
raw best R@1 below **0.9175**.  In that event no RSPG ablation, Cars, CUB, or extra
seed is allowed.  If both seeds clear 0.9175, the three already registered
mechanism controls become mandatory; the full method must strictly beat every
control or it is dead regardless of its advantage over Proxy Anchor.

For every completed arm report both raw best-over-training and independently
exported final-state R@1.  The historical leave-neighbour peak-gap estimator has
been retracted and must not be relabelled selection-corrected; raw-to-final is an
observed checkpoint-selection gap, not an unbiased correction.

## Locked execution

- Dataset root: `/home/riomus/datasets/inshop_official_standard`.
- Official full partition, BN-Inception, 512 dimensions, Proxy Anchor reference
  recipe, RSPG recipe delta and fixed epoch-10/40 graph schedule.
- Seeds: 0 and 1, serially.
- New corrected-specific report/checkpoint/final-retrieval names; historical
  digest-matching artifacts are forbidden because their pixel corpus was wrong.
- No test result may tune the graph thresholds, schedule, or decision threshold.

