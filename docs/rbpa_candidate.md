# Candidate 43: rate-balanced proxy attraction (RBPA)

**Gate-2 death recorded 2026-07-31; no implementation or GPU run.**

## Gate 1: PASS

The exact epoch-10 Proxy Anchor gradient predicts that the positive-proxy term
increases same-session pair similarity at `7.77e-5`, versus `4.16e-5` across
sessions. The foreign-proxy negative term partially corrects this asymmetry.
Thus the measured 7.18× acquisition-gap amplification is specifically attributable
to unequal contraction induced by positive attraction.

Inspired by homeostatic control, RBPA would rescale each acquisition group's
positive-proxy contribution so the induced contraction rate is equal across a
class's session groups, without modifying the negative term.

## Gate 2: FAIL

RBPA changes the weight of existing image-to-own-proxy relations. It adds no
positive, negative, or unknown relation. General Pair Weighting and DML-ALA occupy
adaptive DML sample weighting, while cross-view center and camera-diversity losses
already increase the force assigned to under-aligned conditions. A controller that
equalizes measured gradient rates is a new weight estimator, not a new supervision
mechanism. It also falls directly into the project's repeatedly failed class of
adding regularization to a fitting base.

**Verdict: DEAD at Gate 2.** The decomposition explains the shortcut, but the
homeostatic action is occupied weighting and does not warrant GPU tuning.
