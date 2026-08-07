# Pass 70 — blind Fable/Claude search: NONE

Fable had no credits and automatically fell back to Claude Opus. Two blind
passes converged on NONE. Candidate families killed included VIB/quantized
descriptors, LISTA/predictive coding, compressed-sensing heads,
Forward-Forward/target propagation, fast weights, reversible/Hamiltonian
trunks, hyperbolic/product manifolds, ETF/simplex heads, OT/Sinkhorn slots,
SAM/SGLD, Shampoo/K-FAC/Muon, and episodic holdout.

The useful new result is an **Owner-Tangent-Field relocation theorem**: if a
method's update is a sum of per-image forward Jacobian pullbacks, its field on
the L2 descriptor sphere is either integrable (there is an equivalent scalar
loss) or non-integrable (gradient surgery/reweighting). This does not prove all
losses are closed; it only relocates any surviving candidate into the loss
space, and must not be overclaimed. A separate update-rule dichotomy covers
isotropic preconditioners as learning-rate schedules and common anisotropic
second-moment preconditioners as already-closed Krylov/acceleration families,
with explicit caveats for non-scale-invariant BN parameters.

The proposer suggested a zero-training **head-ceiling diagnostic**: export
pre-head features from a corrected In-Shop checkpoint and ImageNet-init trunk,
fit a regularized 512-D LDA head on training IDs and (diagnostically only) on
test IDs, and measure the resulting fixed-descriptor R@1. It could reveal
whether remaining headroom lies in the closed head space or in the trunk, but
the forecast and provenance were intentionally absent under blind protocol.
No candidate or GPU training run was authorized in Pass 70.
