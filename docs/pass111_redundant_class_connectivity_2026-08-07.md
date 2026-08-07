# Pass 111 — redundant class connectivity (RCC)

Status: **LIVE-NARROW at Gates 0–2; no GPU authorized yet.**

## Gate 0/1: corrected, training-only provenance

The earlier one-scale fragmentation result was not sufficient: on the epoch-10
In-Shop pack, fragmented classes had *better* leave-one-out retrieval after
class-size matching.  I therefore ran a new class-disjoint half-split diagnostic
on the corrected Proxy Anchor training pack (`inshop_trained_prehead_seed0.npz`):

1. Split each identity's training photographs into two fixed halves using RNG
   seed 123.
2. Build a support graph only on the first half, with cosine edges at 0.80 or
   the two-nearest-neighbour union (the threshold is fixed before the outcome).
3. Retrieve the held-out half against the pooled support halves; score only the
   held-out identity labels.
4. Record support connected-component count, edge density, and mean edge cosine.

Across **1,313 identities** with at least six photographs, support component
count predicts held-out same-class top-1 accuracy: **Spearman ρ = −0.13494,
p = 9.22×10⁻⁷**. The held-out class-balanced top-1 mean is 0.98678. The result
is training-only, uses no official query/gallery labels, and is distinct from
the previously retracted test-selected geometry claims. It motivates a
redundancy hypothesis: a class whose support is held together by a few bridge
images is less likely to transfer its identity geometry to another photograph.

This is provenance for a *graph-redundancy* intervention, not proof that forcing
connectivity helps. The diagnostic was run on the 1024-D trained pre-head pack;
the final 512-D operating representation must be rechecked before a deciding
run. If the same statistic is absent after the exact deployed projection, the
candidate dies at Gate 1.

### Final operating-point check (2026-08-07)

The same fixed RNG-123 half split and graph rule were rerun on the exported
512-D deployment embeddings from the epoch-10 In-Shop checkpoint. The result
is **1,313 identities**, held-out class-balanced top-1 **0.941186**, and
component count versus held-out accuracy **Spearman ρ = −0.180968,
p = 3.95×10⁻¹¹**. The signal survives the projection and is stronger than the
pre-head measurement; Gate 1 remains open.

## Candidate training object

For a balanced class block with normalized descriptors `z_1,…,z_n`, define a
soft within-class affinity `w_ij = sigmoid((cos(z_i,z_j)-τ)/s)` and its weighted
Laplacian `L`. The proposed auxiliary term is the negative log weighted
spanning-tree mass,

`L_RCC = - log det(L[1:n,1:n] + ε I)`.

By Kirchhoff's matrix-tree theorem this is the log partition function of all
spanning trees, not a selected positive edge. It rewards *many redundant
within-class paths* while preserving the ordinary Proxy Anchor term and all
foreign negatives. It is intentionally not “connect the class with an MST” and
not “maximize only the Fiedler eigenvalue.” A class already connected by one
fragile bridge receives a different gradient from a class with many independent
paths.

## Gate 2: adversarial prior-art boundary

The exact nearby DML collision is Xu et al., *Deep Asymmetric Metric Learning
via Rich Relationship Mining* (CVPR 2019), which builds a minimum-cost spanning
tree per class. That occupies **selected sparse tree supervision**, not a
differentiable log-partition over all trees.

Other nearby work covers topology-preserving metric learning (TCDesc), graph
consistency (CGML), TopNet, and the repository's own Fiedler candidate 136.
Those make connectivity or a persistence summary the target; none found in the
primary-source search optimizes the matrix-tree log-partition of a labelled
within-class image graph for a fixed single-vector zero-shot retrieval model.
This remains a qualified novelty distinction: if any cited method's objective
is a weighted spanning-tree partition or an equivalent determinant, RCC is
DEAD regardless of its score. The distinction is therefore `LIVE-NARROW`, not a
novelty claim.

## Gate 3 preregistration (before implementation)

The exact corrected In-Shop screen will use the current BN-Inception/512-D
Proxy Anchor recipe, seed 0, with the full official sampler (`samples_per_class=0`),
matching the corrected reference. The attempted `n=4` block screen was stopped
as a sampler confound: matched Proxy Anchor reached only 0.2278 at epoch 1 and
0.0494 at epoch 2, versus the corrected full-sampler reference near 0.91. The
RCC-memory variant therefore adds no sampling intervention: it augments each
current class block with at most four detached same-class descriptors from the
existing cross-batch memory. The baseline is the identical full-sampler Proxy
Anchor arm. This amendment was made before the new run, not after an RCC result.
Before implementation, the
coefficient was amended to a fixed **0.05**: the earlier “0.25× initial
gradient norm” target is not portable because the base gradient is unavailable
to a loss constructor before the first backward pass; calibrating it per batch
would introduce an unregistered adaptive method. Temperature, threshold,
coefficient, memory cap, and sampler are now frozen (no retrieval tuning).

Prediction: raw best-over-training R@1 **0.9190**, versus the matched baseline;
the method is falsified if it is below **0.9175** or fails to improve the
training-only held-out connectivity statistic by 10% relative. Report raw and
frozen-final/independently selected metrics; no selection-bias estimator will be
called a correction.

The candidate earns confirmation only if it strictly beats both controls:

1. ordinary Proxy Anchor with the same full sampler;
2. the already-occupied MST/edge-mining control.

If either control matches RCC, the mechanism is dead even if absolute R@1 rises.
No second seed or second dataset is authorized until that mechanism test passes.

## Falsifiers and risks

- final-projection Gate-1 correlation is non-positive;
- `det(L+εI)` is numerically dominated by ε or by class size;
- RCC matches MST or Fiedler connectivity, showing an occupied shortcut;
- the auxiliary term collapses classes or destroys the base gradient;
- the screen misses 0.9175.

Primary graph-theory basis: Kirchhoff's weighted matrix-tree theorem. The
theorem is classical; the claimed object is its use as a differentiable,
labelled positive-support partition in this deployment lane—not the theorem.
