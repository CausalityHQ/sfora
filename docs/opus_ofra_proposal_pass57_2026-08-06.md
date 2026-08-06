# Pass 57 blind proposal (Opus fallback): OFRA

The proposer returned one method, **Orbit-Frame Restricted Attraction (OFRA)**,
in Lane A (ResNet-50, 512-D descriptor). The complete answer was emitted in the
retained consultation stream; this file preserves its operative specification
and forecasts for gate review.

## Mechanism

OFRA extends PFML by fitting a pointwise rank-k tangent frame to augmentation
displacements and redirecting the supervised attractive gradient into that
frame at unchanged magnitude. For an embedding `z`, an MLP produces an
orthonormal tangent frame `F(z)` and projectors `P_parallel=FF^T` and
`P_perp=(I-zz^T)-P_parallel`. A duplicate augmented view of an existing
same-class batch item supplies a certified nuisance displacement `u`.

The frame is trained only by the detached local-PCA loss
`||u-P_parallel u||^2/(||u||^2+eps)`. The encoder receives
`||P_perp u||^2 + eta ||P_parallel u||^2`, with the frame detached. For each
PFML attractive gradient `a_i`, OFRA uses the normalized interpolation of
`a_i` and `P_parallel a_i`, then rescales to `||a_i||`; repulsion and proxy
gradients remain unchanged. A stop-gradient linear surrogate injects exactly
the replacement gradient. Warmup is lambda=0 for five epochs, ramp to one by
epoch 25, then fixed. Frozen proposal values: k=16 (CUB/Cars), 8 (SOP),
eta=.1, beta=1, duplicate fraction .5/.5/.25, two-layer 512-wide GELU frame
MLP, frame lr 1e-3; deployed model is only the 512-D normalized descriptor.

## Claimed causal test and controls

The claim is that isotropic same-class attraction destroys transfer-critical
appearance axes. OFRA should preserve them by removing only the fitted nuisance
direction while retaining attraction magnitude. Proposed controls are PFML
reproduction, isotropic deadzone, random frame, frozen global frame, sampler-only,
invariance-only, k/eta sweep, and sample-vs-proxy redirection ablation. Key
degeneracy diagnostics are fitted-frame fraction, projected-attraction ratio,
class-mean participation ratio, and the fraction of pairs outside PFML's delta
ball.

## Frozen forecasts and falsifiers

Against the proposer’s reconstructed PFML control, forecast R@1 is CUB .729 ->
.745 (+1.6 pt), Cars .922 -> .933 (+1.1), SOP .826 -> .830 (+.4), five seeds.
The method is falsified if CUB gain is <=.5 pt, if it is no better than the
isotropic deadzone or random-frame controls, if the pointwise frame is no
better than a global frame, if frame fit is <.5, or if SOP gain exceeds CUB
gain (the causal story predicts little SOP benefit). The proposer explicitly
warned that PFML’s undisclosed sampler/batch/LR details make the published
cross-paper numbers secondary to a matched reproduction.

## Prior-art claims to audit before any GPU

The answer distinguished OFRA from PFML, Proxy Anchor, class-collapsing
positive selection, SoftTriple/Sub-center ArcFace, non-isotropic proxy models,
augmentation-equivariance/self-supervision, TangentProp/Manifold Tangent
Classifier, Lie-group discovery, PCGrad/continual-learning gradient surgery,
and AdvRF. The closest unresolved risks are augmentation-tangent invariance
methods and representation-space gradient projection; these require the
independent cold review. No GPU is authorized by this frozen proposal.

## Provenance note

This is a recovered operational summary of the retained Opus stream, not a
byte-for-byte provider transcript; the stream itself ended after emitting the
proposal. Exact source links and any wording omitted here must not be treated
as independently verified until Gate 2 review.
