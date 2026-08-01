# Candidate 136: spectral class connectivity

**Gate 1--3 record, committed before implementation or GPU screening —
2026-08-01.**

**Final verdict: DEAD AT GATE 1 after a stronger CPU diagnostic, before the
spectral arm started.**

## Gate 1: repository provenance and CPU diagnostic

The project repeatedly found that contracting every labelled positive is a bad
fit for multimodal classes: sub-centres did not help, RSPG/ARCG gates failed,
and the local structure that does reproduce across CUB runs is ordinal and
neighbourhood-scale rather than a stable global partition. In the exact
epoch-10 In-Shop Proxy Anchor pack, a CPU-only diagnostic built an undirected
within-class 1-nearest-neighbour graph from normalized embeddings. **1,603 / 3,975
eligible classes (40.33%) were disconnected.** Thus local same-class relations
often do not form a connected identity manifold at the operating point.

The proposed supervision object is not a selected positive edge. For each class
present with four samples in a batch, form a complete differentiable affinity
graph

`w_ij = exp((cos(z_i,z_j) - 1) / 0.1)`

and maximize its algebraic connectivity, the second-smallest eigenvalue of the
unnormalized graph Laplacian. This says only that no weak cut may split a labelled
class; it lets the model decide which edges provide the connecting paths.

The same pack was used for an algebraic-collapse check registered before reading
the result: kill if fewer than 10% of class 1-NN graphs fragment, or if the pair
with maximum Fiedler derivative equals the ordinary farthest positive in at
least 80% of classes. The results were **40.33%** and **74.52%**, respectively,
so the diagnostic passes narrowly. The caveat is important: 57.03% of Fiedler
cuts isolate one sample, so much of the signal remains close to hard-positive
mining.

## Gate 2: adversarial prior-art audit

The closest DML work is Xu et al., *Deep Asymmetric Metric Learning via Rich
Relationship Mining* (CVPR 2019): it replaces all positive pairs by a
visual-distance minimum spanning tree per class. The distinction is
mechanism-level but narrow. DAMLRRM fixes which labelled edges are positive;
this candidate differentiates a class-level spectral property and declares no
edge positive independently.

Fiedler regularization (Tam and Dunson, ICML 2020) regularizes the graph of neural
network weights, not the sample-similarity graph. Spectral clustering,
Laplacian representation learning, deep subspace clustering, and graph
connectivity objectives use the same mathematics in other problem objects. An
adversarial search found no primary source that maximizes algebraic connectivity
of a learned within-class image-similarity graph for zero-shot retrieval. This
is a qualified novelty claim, not proof of absence.

## Gate 3: preregistered In-Shop screen

The official Proxy Anchor sampler has `samples_per_class=0`, so it cannot form
class graphs reliably. Balanced sampling is itself a material intervention and
must not be hidden. Run two seed-0 In-Shop arms with identical IPC=4 sampling:

1. `pa_ipc4`: Proxy Anchor only;
2. `pa_fiedler`: Proxy Anchor plus the spectral class-connectivity term.

Use affinity temperature 0.1 and one fixed auxiliary coefficient selected only
to keep the initial auxiliary gradient norm at or below the Proxy Anchor gradient
norm; do not tune on retrieval. Prediction: `pa_fiedler` reaches raw best R@1 at
least **0.9085** and beats `pa_ipc4` by at least **+0.50 point**. Either condition
failing kills candidate 136. If both pass, run selection-bias correction, then
confirm on unseen seeds before any second dataset. Report both raw and corrected
numbers.

Estimated screen cost: two In-Shop runs, about **4.5 GPU-hours** total.

## Conditional mechanism controls

These controls are registered before either screening result is known. Run none
unless both Gate-4 conditions pass. A passing screen earns two additional
seed-0 In-Shop arms under the identical IPC=4 recipe:

1. a batch-hard positive term acting on the farthest same-class pair;
2. a DAMLRRM-style minimum-spanning-tree positive term.

The full spectral method must strictly beat both controls in raw R@1 and must not
reverse below either after selection correction. Otherwise its performance is
attributable to an established edge-mining/tree mechanism and candidate 136 is
dead regardless of its absolute score. Only after this mechanism test may unseen
confirmation seeds run. Conditional control cost: about **4.5 GPU-hours**.

## Stronger outcome-relevance diagnostic and termination

Before the matched control completed, the same epoch-10 pack was tested for the
missing premise: whether disconnected class graphs actually predict retrieval
failure. They predict the opposite. Class-balanced leave-one-out training R@1 is
**0.94813 for fragmented classes versus 0.93605 for connected classes**, a
**+1.208-point** advantage. After exact class-size stratification, fragmented
classes remain **+3.534 points** better; the common size-4 and size-5 strata are
respectively +3.165 and +3.181 points.

The 40.33% fragmentation measurement is real, but it is not a defect. It is
consistent with legitimate multimodality whose local components retrieve well.
Maximizing connectivity would erase structure associated with *better* retrieval,
so candidate 136 fails provenance rather than earning a GPU test. The in-progress
`pa_ipc4` control was terminated after epoch 5 and is excluded from evidence;
`pa_fiedler` never started and produced no artifact. No controls or confirmation
seeds may run.
