# LRTS — local-response transport supervision

**Status: DEAD after Gate-2/identifiability audit on 2026-07-31; no diagnostic,
implementation, or GPU use.** Gate 1 was recorded before the audit below.

## Repository provenance

At the In-Shop operating point, augmentation-response agreement is highly
selective (8.63% RSPG pair retention), and ARCG found response/distance
inversions independent of base similarity. Yet treating response signatures as
equal/not-equal positive eligibility self-erased, and ordering images by response
agreement did not improve retrieval. DiDE's global equal-displacement idea was
also rejected as established equivariance.

These measurements support a narrower possibility: controlled response carries
local factor coordinates, but two images need not express those factors in the
same embedding directions. Scalar agreement and one global augmentation vector
both discard that local-frame structure.

## Cross-disciplinary mechanism

LRTS borrows parallel transport and sheaf consistency. For each training image,
the five registered intervention displacements form a small local response frame
`D_i`. For a same-class pair `(i,j)`, estimate a low-dimensional orthogonal
transition map `Q_ij` between their response frames, rather than requiring
`D_i = D_j`. The maps must satisfy two constraints:

1. transported response residual is small: `D_j ≈ Q_ij D_i` in factor
   coordinates;
2. transition maps around a same-class three-cycle have trivial holonomy:
   `Q_ki Q_jk Q_ij ≈ I`.

An auxiliary loss would train the embedding so same-identity local factor frames
admit cycle-consistent transport while ordinary Proxy Anchor remains unchanged.
The maps and extra views are training-only; inference remains one normalized
embedding and cosine similarity.

Unlike DiDE, no single displacement is forced to be equal across images. Unlike
ARCG/RSPG, response similarity does not decide pair relevance. The supervision
asserts that labelled same-class observations share a *consistent transformation
system* even when their local factor coordinates differ.

## Gate-2 and identifiability attacks required

Before any diagnostic, search primary sources for vector-bundle/sheaf metric
learning, local-frame alignment, transformation synchronization, cycle-consistent
equivariant representation learning, and gauge-equivariant contrastive losses.
LRTS is dead if prior work already aligns per-instance transformation frames
with cycle consistency for recognition or retrieval.

Even if literature-live, LRTS is dead on identifiability if five intervention
vectors cannot constrain the transition map without an arbitrary high-dimensional
rotation fitting every pair. Any diagnostic must preregister a fixed
low-dimensional frame construction and show lower transport residual for
same-class than matched different-class pairs before training.

## Audit result

The exact proposed transition is underidentified. Each image provides only five
displacement vectors in a 512-dimensional embedding. An unrestricted ambient
orthogonal map can fit every pair of five-dimensional spans. Restricting the map
to a 5x5 factor-coordinate rotation avoids that extreme but still estimates the
map from the same two frames whose residual it minimizes; without an independent
factor basis, Procrustes absorbs pair-specific disagreement and makes the
auxiliary target self-fitting.

The available ARCG export stores only five normalized response magnitudes, not
the displacement frames. Identifying and differentiating the proposed maps
would require repeated six-view inference (centre plus five interventions) or a
large stale memory, moving training far beyond the roughly-1x constraint and
reintroducing the stale-target problem. A one-transform stochastic version no
longer identifies a frame or its holonomy.

The remaining high-level operation is also established: temporal and cross-video
cycle-consistency learn representations from correspondence cycles; class-pose
and gauge-equivariant methods align local transformation frames; synchronization
literature explicitly imposes identity composition around cycles. LRTS has no
identified, efficient mechanism left between those precedents. Candidate 33 is
**DEAD before diagnostic**.
