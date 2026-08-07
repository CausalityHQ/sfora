# Pass 113 — Barrier-energy positive supervision (BEP)

## Gate 1: corrected operating-point measurement

I used the retained corrected-corpus In-Shop Proxy Anchor final training pack
(`inshop_corrected_pa_seed0_train_final.npz`, 25,882 images, 512-D) and the
class-centroid table derived from the same training split. For a sampled image
`a`, its nearest same-class neighbour `p` was found by cosine. I then evaluated
the normalized linear path `gamma(t)` between `a` and `p` at nine fixed values
of `t`. At each point, the energy was the maximum cosine to a *foreign* class
centroid minus cosine to the anchor's class centroid. The barrier is the
maximum energy along the path.

On 4,999 eligible images (one singleton class was skipped), ordinary leave-one-
out training R@1 was 0.995599. Barrier versus correctness had Pearson
`r = -0.1637256`; the barrier median was -0.25756 (10th/90th percentiles
-0.32460/-0.15139). This is a descriptive measurement, not test-label
provenance: the foreign centroids and nearest neighbours use training labels
only. The sign predicts that same-class pairs whose interpolation crosses a
foreign-class energy ridge are less reliable than endpoint cosine alone.

## Candidate object

BEP keeps the ordinary Proxy Anchor positive and negative terms. For a sampled
same-class pair `(a,p)`, it adds a hinge on the differentiable path barrier:

`L_BEP(a,p) = softplus((max_t max_{c != y} cos(gamma(t), q_c)
                         - cos(gamma(t), q_y)) / T)`,

where `q_c` is the current normalized class proxy, `gamma` is the normalized
linear interpolation of the two live descriptors, and the max is smoothed by
log-sum-exp over the nine registered path points and foreign proxies. The term
does **not** select or delete positives, replace the class proxy, or change
inference: deployment remains one 512-D descriptor and cosine retrieval. Its
claim is that a low endpoint distance can still cross a foreign-class saddle,
and that explicitly lowering this path energy teaches a safer within-class
route.

## Gate 2: adversarial prior-art boundary

The audit checked geodesic/manifold retrieval, energy-based DML, graph/path
mining, and intra-batch connection methods. Relevant primary neighbours include
Energy Confused Adversarial Metric Learning (Chen & Deng, 2019), Learning
Intra-Batch Connections for DML (2021), GeoMM's graph shortest-path metric
(CVPR 2025), and recent geodesic semantic search. Those methods either add a
global energy regularizer, exchange messages/edges, or use graph shortest paths
at inference. None of the checked sources uses a **differentiable foreign-proxy
saddle along a same-class descriptor interpolation as a train-time positive
constraint while retaining Proxy Anchor**.

This is therefore **LIVE-NARROW**, not a broad novelty claim. If the
implementation reduces to ordinary pair distance, graph connectivity, or an
energy regularizer independent of the path endpoints, it is dead at Gate 2.
The mandatory controls are (i) endpoint positive margin with matched gradient
norm, and (ii) the same path barrier using only the nearest foreign proxy
(rather than the foreign-proxy soft maximum).

## Gate 3: preregistration

Use corrected In-Shop BN-Inception, official sampler (`samples_per_class=0`),
seed 0, full 8,640-step horizon, and fixed `T=0.10`, nine path points, and
`lambda=0.05` (chosen by initial auxiliary-gradient norm, not retrieval).
Prediction: raw best R@1 **0.9185** and independent final R@1 **at least
0.9165**; the candidate is falsified if raw best is below **0.9175**, final is
below **0.9155**, or BEP fails to beat both registered controls by 0.10 point
after independent export. Report raw and final metrics; no selection estimator
will be relabelled as a correction.

No GPU implementation is authorized until the algebraic CPU test confirms
finite gradients and the path term is nonzero on the exact operating pack.

## Implementation audit before deciding run

The first launch was stopped at step 400/8,580 after code review found that
the provisional Torch implementation averaged the path-point penalties,
whereas the frozen object above requires a smooth maximum over both path
positions and foreign proxies. Its partial R@1 values (0.2408 at epoch 1,
0.6822 at epoch 2) are excluded from evidence. The implementation now uses a
single log-sum-exp over the flattened path/foreign dimensions, the NumPy
reference and CPU tests agree, and the deciding run restarts from step zero.

The next launch exposed a second implementation issue in review: its temporary
flattening also coupled different same-class pairs in one batch. That partial
run reached roughly 200 steps and is likewise excluded. The final implementation
applies the smooth maximum over path points and foreign proxies separately for
each aligned pair, then averages pairs; an explicit Torch-vs-NumPy numerical test
now passes.
