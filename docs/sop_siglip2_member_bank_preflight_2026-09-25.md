# SOP SigLIP2 member-bank preflight (frozen before execution)

The two product-neighbour graph lanes failed their preregistered coverage gates.
The next candidate changes the **training candidate set**: the existing float
SmoothAP term would compare each of 64 current anchors to the 53,700 SOP TRAIN
fit members held in a detached memory bank. It would retain the ArcFace
control, encoder, 128-dimensional head, optimizer, 1,000-update schedule,
export, 130-byte packed format, and exact native top-10 search. Cross-batch
memory and SmoothAP are prior art; this is a causal experiment, not a novelty
claim or a quality result.

Before training, use only the frozen SOP TRAIN fit partition (seed 179019),
the authenticated zero-update PCA head and class proxies, and the 1,000-update
ArcFace control endpoint. The diagnostic set is all **4,296** fit leave-one-out
errors of that control, with each error's actual nearest wrong-product member
fixed by the fit census. The endpoint is diagnostic only; no trained weights
may initialize a fresh treatment.

For each trained error query, score its current 128-dimensional unit embedding
against two stale candidate banks, excluding all members of the query's own
product, and break equal-score ties by ascending fit ordinal. Bank A is the
zero-update 128-dimensional unit embedding of every fit member. Bank B is
`normalize(e0 - w0[class] + w1[class])`, using zero-update and endpoint unit
ArcFace proxies. Record the share whose fixed impostor member is among the
top 64 negative members. This tests whether an old bank can expose the actual
member-level training error; it does not test whether gradients will fix it.

The choice is fixed: A coverage **>=60%** permits only raw-bank training. If
A fails and B coverage **>=60%**, permit only reanchored-bank training. If both
fail, stop this lane before training. No search over K, temperature, coefficient,
or bank update rule follows a failure. A permitted implementation must then
pass a separate measured timing gate: its actual SmoothAP forward plus backward
for 64 anchors against 53,700 candidates, with positive-rank computation, must
take **<=0.06 s per step** on the DGX GB10, including bank gathering but not
encoder computation. This is necessary, not sufficient, for <=1.15x total
training wall relative to the 1,152.795 s ArcFace control. The existing
`smooth_ap_float_loss` builds a cubic batch comparison and must not be called
on the full bank; a correct bank loss needs O(anchors x positives x candidates)
work and independent numerical tests.

If either branch passes both gates, train one treatment and a same-source
ArcFace control replay, with identical data tickets, initial weights, head,
optimizer, 1,000 updates, and checkpoint rule. Compare on the untouched SOP
TRAIN product-disjoint holdout: 5,851 queries/1,132 products against all 59,551
TRAIN images, excluding self. Advancement requires >=+1.0 percentage point
paired product-bootstrap Recall@1 with a positive lower confidence bound,
mAP@R decline <=0.005, actual wall <=1.15x, and matching native top-10
semantics. The official SOP TEST and In-Shop panels remain closed to this
method-selection preflight. Neither a passing coverage nor timing gate is a
claim of a quality or performance gain.
