# Candidate 268: developmental-timing positive eligibility

**Verdict: DEAD at Gate 1. No implementation and no GPU.**

## Proposed mechanism

Inspired by developmental timing in biology, each training image would carry a short
trajectory of its true-class margin (or loss) across epochs. Two different images of
the same class would remain positive only when their normalized learning trajectories
had compatible onset, direction, and forgetting events; disagreement would turn the
pair unknown. This is not an embedding-distance or rival-identity signature. It asks
whether the model acquires the two examples through the same temporal learning regime.

## Why Gate 1 fails

The repository has no independently valid **per-example** margin/loss trajectories for
the corrected reference recipes. The corrected SOP report exposes global epoch metrics,
which cannot establish within-class temporal heterogeneity. RSPG's CUB/In-Shop density
split motivates seeking a non-rival signal, but it does not show that learning-time
compatibility exists or predicts retrieval correctness.

Obtaining the required evidence would mean instrumenting a training run or repeatedly
exporting sample-level predictions. Therefore the diagnostic is neither already present
nor CPU-only, and running it would reverse the protocol by spending compute to invent
the provenance of an unregistered candidate.

## Adjacent literature (not the deciding gate)

Training dynamics are an established sample signal: Swayamdipta et al.,
[Dataset Cartography](https://arxiv.org/abs/2009.10795), use confidence and variability
across epochs; forgetting-event and data-selection work similarly intervenes on samples.
The search did not establish an exact same-class positive-to-unknown gate based on two
loss trajectories, but novelty is irrelevant because the candidate already lacks the
required repository measurement.

## Residue

Do not add trajectory logging to a reference run for this idea. Reconsider the mechanism
only if a future, independently motivated experiment already produces per-example
trajectories and shows a preregisterable association with retrieval errors.
