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
optimizer, 1,000 updates, and checkpoint rule. Compare on the SOP TRAIN
product-disjoint holdout, which was excluded from fitting but used in earlier
method-selection screens: 5,851 queries/1,132 products against all 59,551
TRAIN images, excluding self. Advancement requires >=+1.0 percentage point
paired product-bootstrap Recall@1 with a positive lower confidence bound,
mAP@R decline <=0.005, actual wall <=1.15x, and matching native top-10
semantics. The official SOP TEST and In-Shop panels remain closed to this
method-selection preflight. Neither a passing coverage nor timing gate is a
claim of a quality or performance gain.

## Executed gate evidence

The [fit-only coverage receipt](evidence/compact_metric/sop-siglip2-substrate-v1/member-bank-preflight-v1.json)
(SHA-256 `54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c`)
records **4,127/4,296 = 96.0661%** raw-bank coverage and
**4,131/4,296 = 96.1592%** reanchored coverage. The frozen branch rule selects
the raw bank. The completed DGX systemd unit used 6.400 s probe wall and
131,377,152 bytes peak CUDA allocation. This checks candidate availability
only; no new model was trained.

The [source-bound timing receipt](evidence/compact_metric/sop-siglip2-substrate-v1/member-bank-step-cost-v2.json)
(SHA-256 `8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9`)
records 20 timed worst-positive-count steps, median **0.013466 s** for the
loss-only forward, backward, and 64-row bank write on NVIDIA GB10, with
562,051,584 bytes peak CUDA allocation. It binds the exact loss source SHA-256
`a57a1b8cb4722a12dbc8fd255255632b05aff918c8c66e46e5708c2da9e4854a`.
The first timing job failed before import because the isolated staging package
lacked its imported modules; the second completed but did not bind the loss
source. Neither failed/insufficient receipt is used as gate evidence. This
third job passed the frozen **0.06 s** timing gate. Total training wall remains
unmeasured until the matched training pair completes.
