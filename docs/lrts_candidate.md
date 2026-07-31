# LRTS — local-response transport supervision

**Gate 1 recorded 2026-07-31 before prior-art audit, diagnostic, implementation,
or GPU use.**

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

