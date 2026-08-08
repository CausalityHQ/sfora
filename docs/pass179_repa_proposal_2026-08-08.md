# Pass 179 — Random-Projection Erasure-Coded Proxy Anchor (REPA)

## Frozen blind proposal

Train `h_i=f_theta(x_i)` with normalized full descriptor `z_i` and ordinary
Proxy Anchor proxies. For each batch, draw a signed, permuted Walsh–Hadamard
rotation and split its rows into four complementary 128-D orthoprojectors
`A_k`. Form `z_i^k=normalize(A_k h_i)` and projected proxies
`p_c^k=normalize(A_k v_c)`. Apply Proxy Anchor independently in every view and
add a soft-worst aggregate:

`L_REPA = L_PA(full) + lambda*tau*log((1/4) sum_k exp(L_PA(A_k)/tau))`.

The frozen values are `alpha=32`, `delta=0.1`, `tau=0.25`, with `lambda`
ramped from zero to `0.5` over the first 10% of training. All projections and
proxies are discarded at deployment; the single normalized 512-D full
descriptor is unchanged.

## Gate 1 premise and CPU falsifier

The proposal targets the hypothesis that unseen-class errors are concentrated
in brittle descriptor subspaces. On saved Proxy Anchor embeddings, generate
256 signed-Hadamard four-way partitions and measure full-space versus projected
R@1 and top-10-neighbour Jaccard. The premise is supported only if ordinary PA
has projection drop `Delta_128 >= 5` points and mean Jaccard `<= 0.60`.
REPA's prediction is to halve that drop. If all checked datasets have
`Delta_128 < 2` and Jaccard `> 0.80`, the candidate is killed before GPU.

## Gate 2 primary-art audit

Gate 2 is **DEAD**. Opitz et al., *BIER: Boosting Independent Embeddings
Robustly* (ICCV 2017), explicitly partitions the final embedding into an
ensemble, trains the branches with differentiable diversity objectives, and
reports improved CUB/Cars/In-Shop retrieval without additional test-time
parameters. Xuan et al., *Deep Randomized Ensembles for Metric Learning*
(ECCV 2018), trains randomized embedding functions for the same retrieval
benchmarks. Yao et al., *High-dimensional Similarity Learning via Dual-sparse
Random Projection* (IJCAI 2018), directly uses random projections in
similarity-learning optimization. Dropout/DropBlock and random-subspace
regularization occupy the stochastic erasure mechanism.

Replacing learned or coordinate subspaces with signed Hadamard blocks and
using a soft maximum over Proxy Anchor branches does not change the supervised
object enough to escape these mechanisms: it is a randomized embedding
ensemble/subspace-robustness objective with branches discarded at inference.
The proposal is therefore closed before CPU or GPU work. The exact equations
and frozen response are retained for auditability, but REPA is not a novelty
candidate.

No implementation or GPU run is authorized by this proposal alone.
