# Pass 120 — Coalition Interference Supervision (CIS) preregistration

## Status

Gate 2 is **LIVE-NARROW**. This document freezes the Gate-3 prediction; no GPU
run is queued while the Pass 119 corrected ECTR controller is active.

## Provenance

The corrected CUB failure decomposition found **51.9%** of failures in the
between-class component. The candidate tests whether a training-only bundle of
distinct-class images can expose shared off-class interference that independent
image-to-own-proxy terms do not constrain.

## Exact operator

For a bundle (S) of (m) distinct labelled classes, compute normalized image
descriptors (f_i), form (b_S=m^{-1/2}sum_i f_i), and apply the existing
proxy classification objective to (b_S) with the multi-hot target containing
exactly the (m) member classes. The deployed path remains a single image to a
single 512-D descriptor. The ordinary Proxy Anchor term is retained. Bundle
sampling is class-balanced and detached from test identities.

## Pre-registered In-Shop screen

The paired corrected Proxy Anchor reference is **0.9163033** (raw and
selection-corrected values will both be reported). CIS predicts corrected best
R@1 **0.9195** (a +0.32-point effect) at seed 0. The screen is falsified if
corrected best R@1 is **<0.9180**, or if the coalition term fails to beat both
the single-image multi-label control and the class-dropout control by at least
0.10 point. A failed screen stops the candidate; no Cars/CUB confirmation is
authorized.

## Required CPU/operator checks

Before queueing: (1) permutation of a bundle must leave the scalar loss
unchanged; (2) replacing one member's class must change exactly the
corresponding multi-hot target; (3) all bundle members must receive finite
nonzero gradients; (4) a one-image bundle must be distinguishable from the
ordinary single-image term; and (5) class-dropout and single-image multi-label
controls must execute through the same code path. A failure is an implementation
mismatch, not benchmark evidence.

A standalone Torch sanity check of the proposed sum-plus-multi-hot operator
passed the first three properties (finite loss, permutation invariance, changed
target, and nonzero finite gradients). This is only an algebraic check: the
repository/remote trainer integration and the one-image/control checks remain
open and must pass before queueing.

The controls use the same `proxy_anchor_coalition` operator and differ only in
`coalition_mode`: `pa_coalition_single` applies the multi-label BCE to each
single image independently, while `pa_coalition_dropout` removes the final
member class from the summed union target. Both now have CPU coverage; the
remote trainer integration must still be verified in the isolated checkout.

## Reporting

For CIS and every control, report raw best-over-training and
selection-corrected values using `scripts/measure_selection_bias.py`. No
novelty or SOTA claim is permitted from the screen alone; an out-of-sample
confirmation and second dataset remain required.
