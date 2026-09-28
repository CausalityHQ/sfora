# Full-source MAIN classification, directly learned128 serving: one cheap gate

Fixed256→post-fit-SVD128 is closed on its280s budget; do not change its width,
compactor or deadline. Next distinct mechanism removes post-fit calibration:
classify in the full1024 source, train the128 serving head directly with the
existing bank ranking objective. This differs from the closed source-centroid
auxiliary: that used fixed proxies/CE64/margin0/weight.1 on top of128 main.
Here full-source **learnable** proxies carry the unchanged MAIN ArcFace.3/64;
128 classification is replaced, not augmented. Compact ranking stays8 and the
native trainable128 PCA initializer/bank are retained. No teacher or SVD.

Baseline: ArcFace(H128(normalize(z)),P128)+8bank128.
Proposal: ArcFace(normalize(normalize(z)−mu_fit),P1024)+8bank128.
P1024 initializes from normalized class sums of fit-centered unit source;
mu_fit is fixed from original fit source only. H128 receives bank gradients;
encoder receives both branches. Serving remains H128(normalize(z)) with existing
128byte codes/f16 inverse norms. Risks: full-source proxies may still saturate,
head loses its class gradient, clipping pressure changes, source gradients may
not reach useful encoder parameters. This gate tests only the first two risks.

Before encoder training or paid review, one≤120s CPU original fit-cache-only
screen:13283/2004 original fit products,source/partition/nativePCA/schedule
hashes pinned, same first17 batches/1088 rows, same singleton whole-batch skip,
same detached native128 bank. No optimizer or held/official image reads.
Use native ArcFace straight-through derivative and8 bank loss in both arms.

GO only if every frozen guard passes:

-1088 finite/nonzero total source gradients; native unused residual≤1e-5.
- Median new total source-gradient norm outside native128 span+radial axis
≥20% (norm fraction, not energy). This is connectivity, not quality evidence.
- Median full-source MAIN/total source-gradient norm fraction≥10%; reject
saturation before another GPU run. Median new/native total gradient norm ratio
in[0.25,4]. No scale/margin/coefficient tuning after a negative.
- On original15 rank-active batches, median new/native128 head-gradient norm
ratio≥0.25, all nonzero/finite. On singleton-inactive batches, head gradient is
expected absent by design; record them. Source classifier gradient finite and
nonzero in every batch. CPU≤120s and all authority guards pass.

Failure KILLs this fixed source-main configuration. A pass permits consequential
Opus/Astra design review and one bounded paired17-update actual encoder smoke,
then separately frozen100-update TRAIN gate only if that smoke passes. No new
source, compactor search, held evaluation, full training or latency yet. Keep
production128 default and all prior negative gates intact.

## CPU terminal result

Original cached probe exited0 in5.261354s, all frozen guards passed. Receipt
`evidence/compact_metric/sop-siglip2-substrate-v1/inshop-source-main-cache-v1.json`
SHA256 `f5bec2865b8aeb86f66003ada0dc5902616d98e043a3d3b8e1716ead95c41203`.
Median new total unused source **norm fraction**29.09358%; MAIN/total norm
fraction73.15933%; new/native total norm ratio0.9303555; bank-only128-head/native
head norm ratio0.4134365 on15 active batches. All1088 source gradients and17
learnable source-classifier gradients finite/nonzero; two inactive head
gradients absent as required. No optimizer, held read, trained encoder, quality
or serving-speed result. Decision is **GO_ENCODER_DESIGN_REVIEW** only.

One consequential review group `0056bc341ded4f98` launched: Opus
`7d3049869ed6414f`, Astra `e73924f282d643cc`. Prior wide-method reviewers are
terminal, GPU idle, no duplicate or encoder arm launched. Review must settle
source/classifier mask authority, sparse head optimizer steps, actual encoder
gradient routes, clipping, native128 checkpoint reload and bounded cost before
implementation/training. Default model and production serving stay unchanged.
