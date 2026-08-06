# Pass 63 blind proposal: Iso-Response Homogenization (IRH)

IRH is a Lane-A (R50/512-D) train-time method using a label-free,
measure-preserving counterfactual transport. It adapts the *stimulus level* per
image until a fixed descriptor response is reached, then minimizes dispersion
of those per-image thresholds.

## Frozen mechanism

From the detached layer-4 evidence map `E=upsample(||F||_channel)`, min-max
normalize and mix with band-limited noise (`rho=.3`), smooth at bandwidth
`sigma_psi`, and form the divergence-free field
`v = (d_y psi, -d_x psi)`, RMS-normalized. Integrate a four-step RK2
semi-Lagrangian warp with one `grid_sample`; a clean identity-grid sample is
used as a matched interpolation control. The transport preserves pixel area
exactly under matched central-difference stencils and slides along evidence
contours rather than deleting or reweighting pixels.

For each image maintain a persistent log displacement state `ell_i` (initial
log 6px, clipped log .5–log 48). The probe level is `t_i=exp(ell_i)*LogU`
and response is `r_i=1-cos(d_i,d_i^t)`, criterion `r*=.15`. A Kesten-decayed
Robbins–Monro staircase updates `ell_i` by clipped `(r*-r_i)/r*`; an epoch
controller adapts `sigma_psi` to keep endpoint fractions in range. The train
loss is the zero-sum rank-weighted covariance surrogate
`L_iso = -mean(s_i r_i/r*)`, where `s_i=tanh((ell_i-median ell)/(1.4826 MAD))`
and is batch-centered. Total loss is the unchanged base MS/PA objective plus
`lambda_iso L_iso`, ramped from zero at epoch 10 to one by epoch 30; 25% of a
class-stratified batch is probed. Deployment remains the baseline descriptor.

## Causal claim and controls

IRH raises sensitivity for hypersensitive examples (predicted local
within-class failures) and lowers sensitivity for hyposensitive examples
(between-class collisions), addressing both halves of the measured CUB
48.1%/51.9% split without a sign-definite invariance penalty. Required controls
are probe-compute-only, iso-stimulus positive warp, pure random field,
sign-flipped dispersion, matched-RMS divergent field, fixed global threshold,
1.3x base compute, and clean-branch stop-gradient. Pre-flight F0 aborts if
baseline threshold spread sigma_ell < .25 nats; other falsifiers include
removed-mass/response correlation >=.15, sign-flip not hurting, out-of-range
controller fractions, iso-stimulus matching IRH, failure to reduce both error
halves, or Jacobian determinant drift >2%.

Frozen Lane-A forecasts (three seeds): CUB .745 (+1.1), Cars .933 (+.6), SOP
.832 (+.3), In-Shop +.4 versus reproduced baseline. Success requires >=+.7 CUB,
+.4 Cars, +.25 SOP. The proposer estimates 1.3x step time and zero deployment
overhead.

## Prior-art risks

Nearest works are MMA Training (adaptive adversarial perturbation), VAT/Π-model
consistency and Jacobian/Lipschitz regularization, elastic/incompressible image
warps, and biological iso-response measurement protocols. The claimed
distinction is the dual control: a finite-radius, area-preserving response shell
with persistent per-example staircase states and a zero-sum threshold-variance
objective, rather than driving every response to zero or maximizing margins.

This is a recovered operational freeze of the retained consultation stream;
Gate 2 review is required before any GPU run.
