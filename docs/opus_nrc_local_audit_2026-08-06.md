# Local Gate 1/2 audit: Nuisance-Rotor Coding (blind pass 48)

**Verdict: DEAD at Gates 1 and 2. No diagnostic, preregistration,
implementation, or candidate GPU run is authorized.** NRC prescribes an
augmentation-dependent phase and deploys the phase-invariant radii. That is a
clear executable object, but the closest corrected repository measurement is
adverse to its universal response law, augmentation-prediction/equivariance is
occupied prior art, and the written normalization, group, maximal-invariant,
and gradient-separation claims do not hold.

## Frozen provenance

The blind proposal is frozen verbatim in
`docs/opus_nrc_proposal_pass48_2026-08-06.md` at commit `caec9d2`. Its file
SHA-256 is
`d5e95dddde03b4d71473b64cc6f74432d16bf0e1a56c79daf2926c539541fe59`.
The durable proposal job was `8fabed3e4ede4d29`; Fable exhausted its credit and
the same durable job continued under Claude Opus. The worker received the
byte-identical blind prompt and used public Web sources, not repository outcome
documents.

The frozen method is **Nuisance-Rotor Coding (NRC)**. A 1024-D head is divided
into 512 two-dimensional planes. Each plane's radius feeds the normalized 512-D
retrieval descriptor, while recorded crop/color/flip augmentation parameters
prescribe phase rotations. Three augmented views and rotor, composition,
radius-invariance, and occupancy losses are intended to make phase equivariant
and radius invariant. The central claim is that radial task gradients and
tangential rotor gradients cannot interfere and that the radii are the maximal
invariant of the prescribed torus action.

## Gate 1: the measured premise is absent and the nearest measurement is adverse

No repository artifact shows that official-query errors arise because Proxy
Anchor deletes useful synthetic-augmentation state, that preserving that state
improves retrieval, or that crop/color parameters carry fine identity evidence.
The fixed 512-frequency rotor, the predicted dataset ordering, and all R@1
forecasts are prospective assumptions.

The closest corrected measurement points against the universal law. ARCG found
that augmentation-response compatibility is **image-specific and
heterogeneous** on In-Shop: its response graph retained only about **0.3631 to
0.3640** of same-class pairs, including the required close-pair rejection and
distant-pair acceptance. NRC instead assigns every image and class the same
deterministic response to a given augmentation parameter. ARCG's earlier
self-erasure result was retracted after a bug was found, so it cannot be used as
negative performance evidence; the corrected graph measurement remains adverse
to NRC's fixed universal response premise.

The IPSR correction likewise establishes only that an augmentation-response
relation exists, not that an auxiliary equivariance objective causally improves
retrieval. Candidate 147, the augmentation-response supervision family, the
tangent-metric line, cycle-equivariance candidate 307, and ARCG already explored
nearby ideas. The ARCG audit explicitly says that replacing a cross-instance
eligibility gate by an augmentation-prediction/equivariance loss or a
single-image auxiliary objective collapses into existing work. NRC does exactly
that. This is missing provenance plus adverse local evidence, not a license for
a GPU screen.

## Gate 2: the exact algebraic claims fail

### The executable direction is not purely tangential

For one plane, NRC defines `r=||u||` and `u_hat=u/(r+eps)`. This vector has norm
`r/(r+eps)`, not one, and is not zero-homogeneous. Along the radial direction,

`d [u/(r+eps)] / dr = eps * u/r / (r+eps)^2`,

which is nonzero for the written `eps>0`. Therefore a loss of `u_hat` can have a
radial gradient. The asserted exact orthogonality between radius losses and
`L_rot`/`L_comp` is false for the frozen forward pass.

Even if `eps` were removed, orthogonality in head-output space would not imply
orthogonality in shared parameters. For trunk Jacobian `J`, radial and
tangential output gradients `g_r` and `g_t` produce parameter gradients
`J^T g_r` and `J^T g_t`, whose inner product is
`g_r^T J J^T g_t`, not generally zero. Batch sums add cross-sample terms, and
AdamW does not restore this separation. The same invalid output-to-parameter
inference was identified in the NSRC audit.

Using the same final coordinate pairs also does not force the trunk to use the
same visual evidence. A network can compute identity amplitude `a_j(x)` and an
augmentation code `q_j(t)` in disjoint upstream channels and output
`u_j=a_j(x)[cos q_j(t), sin q_j(t)]`. It then satisfies the intended final-plane
factorization while the two tasks remain disjoint.

### The deployed descriptor is not a maximal invariant as written

Raw radii are a maximal invariant for independent nonzero `SO(2)` rotations of
the supplied planes. NRC does not deploy raw radii: it deploys
`normalize(W_g log(1+r)+b_g)`. Full rank of `W_g` is insufficient to preserve
injectivity after normalization. For example, with `W_g=I` and `b_g=0`, any two
valid radius vectors whose transformed values satisfy
`log(1+r_2)=c log(1+r_1)` for positive `c != 1` produce the same normalized
descriptor but are not on the same torus orbit. An affine bias does not in
general repair projective collisions. Thus the proposition applies to `r`, not
to the claimed deployed descriptor.

### The declared augmentation object is not the rendered action group

`A ~= R^5 x S^1 x Z_2` is not the action implemented by the augmentations.

- Random-resized crop composition depends on crop position/translation, which
  NRC omits. Scale and translation form a noncommutative affine action rather
  than independent additive scale/aspect coordinates.
- Brightness, contrast, saturation, and hue image operators are not generally
  commuting invertible translations; clipping and quantization further destroy
  inverses and exact composition.
- NRC forms `t13` by adding and clamping metadata, then renders
  `a_t13(x)` once. That is not the same object as rendering
  `a_t3(a_t1(x))`. Clamping itself violates the asserted homomorphism.

Consequently `L_comp` compares another prescribed parameter target; it does not
test composition of the actual image transformations. Holding out a parameter
cell in D4 tests interpolation of a known low-dimensional ledger, not group
closure. The network can infer augmentation parameters from ordinary rendering
cues without learning a physical group representation or improving identity
retrieval. A blank-content failure in D5 would expose one shortcut, but passing
it would not identify an identity-useful nuisance.

### The occupancy guard does not close scale collapse

`L_occ` depends on each plane radius divided by the sum of radii. Uniformly
scaling all radii toward zero leaves these ratios unchanged. The descriptor's
affine map and final normalization can also absorb broad scale changes. Because
`u_hat=u/(r+eps)` is radius-dependent, shrinking can change the rotor losses;
the proposal's claim that it buys exactly zero is internally inconsistent.

## Gate 2 prior art: the supervision object and action are occupied

NRC's narrow rotor parameterization may be uncommon, but its mechanism is a
single-image auxiliary objective that preserves prescribed augmentation
information while an invariant subrepresentation serves the downstream task.
That object and action are occupied:

- Lee et al., **AugSelf** (NeurIPS 2021), predicts differences in crop and color
  augmentation parameters as an auxiliary objective so representations retain
  augmentation-aware information, including in supervised transfer.
- Dangovski et al., **Equivariant Self-Supervised Learning** (ICLR 2022), adds
  transformation prediction to invariant contrastive learning.
- Devillers and Lefort, **EquiMod** (ICLR 2023), models augmentation-induced
  displacement in representation space.
- Garrido et al., **Learning and Leveraging World Models in Visual
  Representation Learning** / SIE (ICML 2023), explicitly splits invariant and
  equivariant representations for controlled transformations.
- Harmonic/group-equivariant networks and complex-modulus/scattering methods
  already occupy the fixed rotation plus invariant-radius operator.

NRC changes the coordinate wrapper and prescribes a multi-frequency phase, but
it neither creates a new supervision relation nor establishes that this wrapper
forces a new mechanism. Its claimed protection from task interference is false,
so the wrapper does not provide a substantive novelty distinction from the
occupied auxiliary-equivariance family.

Primary sources:

- Lee et al., *AugSelf*, NeurIPS 2021:
  <https://proceedings.neurips.cc/paper_files/paper/2021/hash/94130ea17023c4837f0dcdda95034b65-Abstract.html>
- Dangovski et al., *Equivariant Self-Supervised Learning*, ICLR 2022:
  <https://research.ibm.com/publications/equivariant-contrastive-learning>
- Devillers and Lefort, *EquiMod*, ICLR 2023:
  <https://arxiv.org/abs/2211.01244>
- Garrido et al., *SIE*, ICML 2023:
  <https://proceedings.mlr.press/v202/garrido23a.html>

## Protocol, recipe, and forecast failures

The proposal starts five-seed CUB, Cars, and SOP experiments instead of the
mandatory corrected paired In-Shop screen. It omits raw-best versus
independently selected/final reporting and out-of-sample confirmation. Its
forecast bypasses In-Shop entirely.

The advertised baseline is not reproduced verbatim: NRC fixes trainable Batch
Norm and a universal batch/schedule even though the audited Proxy Anchor recipes
freeze BN and use dataset-specific learning rates and batch sizes. Counting 50
images with three views as 150 baseline forward images also leaves exposure
unmatched: at fixed steps it sees one third as many distinct training images,
while preserving epochs requires three times as many steps. NRC+PFML is a
conditional carrier with no frozen faithful local reproduction. Its strongest
headline, 0.744 CUB against 0.734 PFML, is only a one-point forecast and not
robust benchmark evidence.

## What survives

Raw per-plane radii are a valid maximal invariant of a *supplied* independent
rotation action, and a fixed prescribed phase target cannot eliminate its loss
by learning different group parameters. The proposal also contributes useful
controls: identity action, random orthogonal action, separate-head carrier,
blank images, removal of composition, and explicit occupancy-only controls.
Those are valuable design checks, not authorization to run this method.

**Process lesson:** exact equivariance must be checked against the rendered image
operators, not merely against additive augmentation metadata. Orthogonal output
coordinates do not imply noninterfering shared-parameter updates, and a maximal
invariant remains maximal only while every downstream map is injective on the
orbit space.
