# IRH local Gate 1–2 audit (Pass 63)

## Gate 1 — provenance

IRH is motivated by the measured CUB split between local within-class failures
(48.1%) and between-class overlap (51.9%), and by ECT’s concrete failure: a
hard deletion/must-switch probe was unsatisfiable and area-confounded. IRH
removes deletion/area as a stimulus variable and targets dispersion of response
thresholds, so provenance is evidence-aware and Gate 1 provisionally passes.

## Gate 2 — prior-art hazards

The ingredients are adjacent to MMA Training’s adaptive per-example
perturbation, VAT/Π-model and Jacobian/Lipschitz consistency, elastic/TPS and
incompressible registration, and biological iso-response measurement. The
novelty claim must be narrowly limited to the training mechanism: a
measure-preserving, evidence-structured transport; a persistent per-example
Robbins–Monro response-level state; and a zero-sum objective that equalizes
finite-radius response thresholds rather than driving all responses to zero or
maximizing margins. A cold review must check whether this is merely adaptive
augmentation plus heteroscedastic/Jacobian regularization or an occupied
psychophysics-inspired loss.

Key specification risks before any GPU: the implicit threshold estimator and
rank surrogate may not implement the claimed gradient; the staircase state may
leak identity or act as an uncontrolled curriculum; divergence-free numerical
warps can still have interpolation artifacts; the response may be non-monotone
in transport magnitude; and the `sigma_psi` controller may hide an extra
validation loop. The probe-compute-only, iso-stimulus, sign-flip, divergent
field, fixed-threshold, and determinant-drift controls are mandatory.

**Decision:** Gate 1 passes provisionally; Gate 2 pending cold review. No GPU
candidate run is authorized yet.
