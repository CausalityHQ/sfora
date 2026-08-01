# Ensemble all-miss rescue diagnostic

Date preregistered: 2026-08-01

## Question

The five-seed CUB concatenation reaches 0.7350 R@1 versus seed 0 at 0.6940,
while a train-fit map from seed 0 to the consensus worsens retrieval. Does the
pack create correct decisions even when every member's top-1 decision is wrong,
or is its gain only voting among individually correct members?

## Frozen computation

Use the five `herd_tt_seed{0..4}.test.npz` packs. For every query and seed,
compute the rank of the highest-ranked other image with the correct class.
Compute concatenation retrieval as the arithmetic mean of the five cosine-score
matrices (exactly equivalent to cosine on concatenated unit blocks). Report:

1. number and fraction of queries for which all five seeds miss at R@1 but the
   concatenation succeeds;
2. the median and 90th percentile, over those rescues, of the *worst* correct-
   class rank among the five seeds; and
3. the fraction of rescued queries whose correct class is within every seed's
   top 10.

## Prediction and falsification

Prediction: all-five-miss rescues are at least **0.5% of queries**, and at least
**50%** of those rescues have a correct-class image within every seed's top 10.
This would show consistent weak evidence combined across different distractors,
not ordinary majority voting.

The complementarity hypothesis is falsified if the all-miss rescue rate is
below **0.1%**, or if fewer than **25%** of rescues place the correct class in
every seed's top 10. Intermediate outcomes are descriptive and do not justify a
GPU arm. This is a CPU measurement, not a novelty claim.
