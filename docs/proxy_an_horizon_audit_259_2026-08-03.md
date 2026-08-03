# Proxy-AN 2026 horizon audit (259)

Date: 2026-08-03. Primary article record and author code inspected before proposing
another proxy aggregation operator.

## Source and mechanism

Peng et al., *Proxy-AN Loss for Deep Metric Learning*, Neural Networks 195 (2026),
108254, DOI 10.1016/j.neunet.2025.108254, is a peer-reviewed current-horizon method.
The official code was pinned at commit
`b40dd5b139087b50f81dfd99f4a6f342489930b5`.

For positives, the code retains Proxy Anchor's proxy-centric aggregation: sum positive
sample exponentials by proxy/class and average over proxies represented in the batch.
For negatives, it changes the aggregation axis: for each sample, sum exponentials over
all negative proxies, apply log1p, then average over samples. In shorthand it combines

- positive: `mean_proxy log(1 + sum_same-class-samples exp(...))`;
- negative: `mean_sample log(1 + sum_other-class-proxies exp(...))`.

This is exactly the paper's claimed proxy-centric-positive/sample-centric-negative
hybrid. It changes which object receives the hard log-sum-exp aggregation, not the
labels or similarity itself.

## Relevance to this search

Proxy-AN closes any proposal whose residual claim is merely choosing the sample axis
for negatives and proxy axis for positives, or combining Proxy Anchor and Proxy-NCA
aggregation directions. Several repository proxy-ownership/interference observations
could otherwise tempt that rediscovery. A new proxy method must specify an operator
beyond this published axis hybrid, DADA distribution alignment, probabilistic proxies,
multi/sub-centres, and existing reciprocal-proxy controls.

The released trainer evaluates every epoch (every two on SOP), retains the maximum test
R@1, and writes no uncertainty evidence in the repository. That means small numerical
advantages still require a faithful multi-seed audit; it does not undo the prior-art
verdict. The code has only two commits and no released license file, so copying it is
also inappropriate.

## Verdict

**Occupied horizon, no implementation or GPU.** Proxy-AN is not a candidate from this
project, but it prevents a future renamed aggregation-axis proposal and sharpens the
remaining novelty boundary toward new supervision or an empirically identified
operator outside proxy/sample axis selection.

Primary sources:

- https://doi.org/10.1016/j.neunet.2025.108254
- https://github.com/QuhuiKe/Proxy-AN-Loss/tree/b40dd5b139087b50f81dfd99f4a6f342489930b5
