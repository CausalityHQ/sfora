# Sub-center research decision, 28 September 2026

**Reject the proposed three-arm cached training screen.** No classifier code,
training, encoder forward, official evaluation, or serving change was made.
The full joint quality and speed goal remains unmet and active.

Fable consultation `b347ee2c978b4eb8` completed normally in526s with no fallback
(USD6 attempt cap, not measured spend). It proposed three training proxies per
product, taking the maximum class cosine before ArcFace. Its causal premise was
that one proxy wastes encoder optimization on compressing multi-series products.
The [original recommendation](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-subcenter-research-v1/fable-result.json)
is conjecture, not a new quality measurement.

The independent stdlib
[census replay](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-subcenter-research-v1/verify_census.py)
pins the partition, original positive-tail receipt, and fit-row digest. Run it
with the partition replay path as its sole argument. It uses existing seed179026
1000-update float128 margins on13283 TRAIN fit images/2004 products;
13271 self-excluded eligible anchors. It does not rescore feature tensors.

| Existing fit evidence | Verified result | Interpretation |
|---|---|---|
| At least10 images and3 filename series |1982/4471 tail anchors44.3301%; median margin−0.001175 | Descriptive concentration; worst-positive order statistic depends on product size |
|9–10 images,2 series |193/988 tail anchors19.5344% | Corrects Fable's1057-anchor count |
| Fixed6-image products,1 vs2 series |11.1111% vs5.5556% tail | Does not support more-series harm |
| Fixed9-image products,2 vs3 series |20.2509% vs0% tail | Same limitation; observational strata, not a randomized causal test |
| Fixed13-image products,3 vs4 series |48.0769% vs7.6923% tail | Same limitation |
| Existing initial pooler-gradient screen |12/960 opposed rows1.25%; median cosine0.620746; weighted-rank/main norm ratio0.509859 | Its frozen25% conflict floor failed; this is initialization, not convergence |

All census strata are retained in
[census.json](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-subcenter-research-v1/census.json).
The gradient receipt SHA is
`7ba7a0349518b093de0e1d0324371b6406ff6679f15abf6e2fc2857e01356eb1`;
its raw960 cosines independently reproduce12 opposed rows and the median.
This supports rejecting the claimed causal signal, without proving product
size explains every tail or that later objective conflict is impossible.

Dual critique `5387e72d01a84e6a` completed normally in135s: Opus5.5
`898e6b975a2644fb`117s and Astra `5ba89206fa4f48f1`135s, neither falling back.
The [separate full reviews](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-subcenter-research-v1/dual-result.json)
agree that the census does not justify this training screen. Forced random
positive-proxy assignment changes the objective rather than supplying a matched
null; requiring its delta inside its own interval is vacuous. Occupancy above
random is a poor specialization gate. A fixed-feature head experiment cannot
identify the claimed upper-encoder gradient mechanism. Expanding shared classifier
weight rows without preserving class-to-shard assignment also risks comparing
centers under different coordinate masks. The native In-Shop arm uses one full
mask; it has no existing multi-shard defect.

Independently corrected two Opus statements: the13-image comparison also runs
opposite Fable's claim, and random64-of128 masking belongs only to the optional
`subspace` arm, not the reviewed native `freeze_emb` arm. Neither correction
changes the rejection. Fable's claim that width is nonbinding also exceeds the
failed fixed-recipe material gate. The July sub-center Proxy Anchor negative
remains scoped to its different substrate/loss, not an ArcFace impossibility.

The correct [ECCV2020 primary paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)
studies noisy face supervision and removing non-dominant centers; it does not
establish a fashion-series mechanism. Fable's arXiv2010.05350 citation is a
different landmark competition report.

Astra's proposed prerequisite is a static fit-only gradient read on matching
converged embeddings and classifier, with no optimizer updates. The tail probe
records the feature hash but does not save its tensor. Read-only DGX inspection
found only the receipt/code in the tail run and receipt/checkpoint in the matching
seed179026 run; a bounded filename search under runs found no fit-feature export.
This is an availability check, not proof about every possible remote file.
The checkpoint alone does not supply those embeddings. No new encoder export is
authorized by this research result. Opus's proposed held-series odds-ratio read
is deferred: held association cannot establish fit-gradient conflict or license
new post-hoc rescue thresholds. No reopening, extra seeds, or loss sweep follows.

A subsequent read-only recursive inventory under DGX `/home/riomus/runs`
read247 array headers across142 regular NPY/NPZ files below128MiB and found
no13283×128 array. The
[header inventory](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-subcenter-research-v1/retained-npy-header-inventory.json)
does not inspect PT/checkpoint tensors, larger files, or other directories.
No array payload, held outcome, encoder, or optimizer was evaluated. It narrows
the retained-export availability gap without asserting universal absence.
