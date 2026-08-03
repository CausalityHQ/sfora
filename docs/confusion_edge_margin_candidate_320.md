# Candidate 320: confusion-edge margin (CEM)

## Gate 1 — provenance

On corrected In-Shop seed-0 embeddings, nearest foreign image and nearest
foreign proxy classes agree for 15.69% of samples. These samples have 2.3886%
leave-one-out error versus 0.1466% when the signals disagree.

## Gate 2 — distinction

CEM constructs a directed class graph `c -> d` only when independent sample- and
proxy-level evidence agrees that `d` is the confusing foreign class. It adds a
class-level proxy relation enforcing a calibrated margin on `p_c - p_d` for
class-`c` samples. This is not a sample kNN graph, a diagnostic-only confusion
graph, or a scalar hard-negative weight: it adds a directed class-relation target
derived from two observables.

## Gate 3 preregistration

One corrected official In-Shop seed. Baseline final-state R@1 is 0.91370094.
Prediction: **0.9155** final-state R@1. Falsifier: **<0.9140** or failure to
beat the paired baseline. No GPU until a unit test verifies the registered
directed relation and unchanged unrelated gradients.
