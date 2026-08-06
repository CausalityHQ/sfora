# KCR local Gate 1–2 audit (Pass 65)

## Gate 1 — provenance

KCR is motivated by the project’s measured local/between error split and by the
support-local closure: it changes the extrapolation operator itself, rather than
reweighting an on-support loss. The proposed causal variable (seen/unseen
residual contraction mismatch) is not yet measured in this repository, so Gate
1 is provisional pending the registered 5-epoch D0 diagnostic. Unlike previous
proposals, the gradient path is intrinsically differentiable through the
continuation coefficients, which is a concrete mechanism rather than detached
state.

## Gate 2 — prior-art hazards

Anderson/RRE/MPE/epsilon methods, DEQ solvers, Richardson extrapolation, and
recurrent logical-extrapolation networks are close. The distinction is only
defensible if KCR is a deployed finite-depth descriptor readout trained through
the coefficient solve in a deliberately non-convergent regime; if the learned
block converges, coefficients become an ordinary fixed-point solver or average.
The review must check whether this is simply Anderson acceleration inserted in a
network, whether negative coefficients cause unstable feature norms, and
whether the zero-initialized residual block merely learns a standard deeper
backbone.

Engineering risks: the proposal’s Gram ridge depends on transient scale; GroupNorm
must be used inside the block; horizon randomization and exact eval depth must
be fixed; and C1 (final iterate) is the decisive matched control. No GPU run is
authorized before the cold review and D0 contraction-ratio gate.
