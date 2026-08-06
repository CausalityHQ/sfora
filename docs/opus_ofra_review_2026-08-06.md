# Cold Opus review — OFRA (Pass 57)

**Decision: DEAD at Gate 2; Gate 1 also not cleared. No GPU.**

The review found no local measurement of directional attraction or evidence
that augmentation tangents are transfer-critical. It also noticed that the
forecast deltas (+1.6 CUB, +0.4 SOP) match the published non-isotropic proxy
DML deltas to the decimal, so they cannot serve as an independent falsifier.

Mechanism overlap is decisive. Non-Local Manifold Tangent Learning (Bengio &
Monperrus, NIPS 2004) already learns a pointwise tangent frame by projection
residual on local displacements. TangentProp (NIPS 1991), the Manifold Tangent
Classifier (NIPS 2011), and Dao et al. (ICML 2019) occupy the augmentation
tangent penalty. PCGrad, Gradient Projection Memory (ICLR 2021), CAGrad and
RI-PCGrad occupy projected/rescaled gradients. Xiao et al. (ICLR 2021), SIE
(ICML 2023), MIC, DiVA, self-supervised ranking, and non-isotropic proxy DML
occupy the invariance/equivariant and intra-class-preservation motivations.
The remaining difference—rotating a subspace split per sample—reduces to a
known subspace-split attraction with a learned tangent predictor, not a new
component-level mechanism.

The review also found fatal specification weaknesses: the invariance loss
drives augmentation displacement to zero while the scale-normalized frame loss
still appears healthy; fixed-norm projection amplifies frame noise by roughly
5.6–8x when the claimed hypothesis is strongest; no uniformly attenuated
attraction control is present; encoder rotation can make the frame equal the
attraction direction and silently recover PFML; and the projected-attraction
fallback is undefined at zero. The PFML reproduction has undisclosed knobs, so
cross-paper frontier arithmetic is inadmissible.

Primary sources cited by the review: Bengio & Monperrus NIPS 2004; Rifai et al.
NIPS 2011; Dao et al. ICML 2019; Saha et al. ICLR 2021; Liu et al. NeurIPS
2021; RI-PCGrad 2024; Xiao et al. ICLR 2021; Garrido et al. ICML 2023; MIC
ICCV 2019; DiVA ECCV 2020; Fu & Li AAAI 2021; Kirchhof et al. ECCV 2022;
Non-isotropy Regularization arXiv:2203.08547; Musgrave et al. ECCV 2020.
