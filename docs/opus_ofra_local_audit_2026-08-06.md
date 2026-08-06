# OFRA local Gate 1–2 audit (Pass 57)

## Gate 1 — provenance

The proposal is motivated by the project’s repeated finding that isotropic
same-class attraction can erase transfer-relevant within-class structure, and
by RSPG’s measured dataset dependence. However, no local experiment directly
measured a directional attraction gradient or showed that augmentation tangent
directions coincide with nuisance axes. Thus provenance is a mechanistic
hypothesis, not a demonstrated repo measurement. Gate 1 is provisional pending
the cold review; the proposal’s forecast is not evidence.

## Gate 2 — prior-art hazards

The claimed primitive combines several occupied ideas: tangent-propagation and
manifold-tangent/contractive penalties, augmentation-equivariant or
augmentation-aware representation learning, local PCA/tangent estimation,
gradient projection/surgery, and anisotropic metric learning. OFRA’s proposed
distinction is the specific composition of (i) augmentation-pair tangent
estimation, (ii) a train-time sample-side frame, and (iii) fixed-magnitude
redirection of a supervised attractive representation gradient. The distinction
is not yet defensible: a reviewer must check whether prior work already projects
metric-learning gradients or masks attraction using augmentation tangents, and
whether the stop-gradient surrogate is merely an implementation of known
gradient surgery. The reconstructed PFML recipe also has many undisclosed knobs,
so the claimed frontier crossing is inadmissible without a matched reproduction.

**Decision:** do not implement or queue GPU work. Run the mandatory cold
independent prior-art/degeneracy review first. If it finds the composition
already present, record OFRA dead at Gate 2; if it survives, reconcile the
provenance and controls before any screen.
