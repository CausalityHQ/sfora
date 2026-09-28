# Retained256 design review: conditional mechanics GO

Original dual7e10a8fc8fa24171 completed109seconds: Opus5.5
c51549b81b784643 and Astra06ea076aeae04510 separately support one bounded
real-encoder mechanics smoke. Neither authorizes native256/public release or
quality/SOTA claims. [Full independent reviews](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-deployed256-cached-v1/dual-review.json).
No new consultation is needed for this question.

Independent source inspection confirms both major implementation findings:
`training_width=256` is bound to the old calibration/folding authority, and
`export_all` also hardcodes128. Simply using that flag would implement the
wrong experiment. Reuse the training/export helpers with a narrow explicit
direct256 research authority; preserve the killed folded workflow unchanged.
No new general trainer or production loader bypass.

Accepted safeguards: externally bounded paired smoke, actual matched pixel/
target/schedule hashes, first128PCA weights ANDbias equality, all finite
parameters/optimizer/refreshedbank, exact frozen hashes and changed trainable
encoder, variance/effective-rank floor50% of initialization, saved checkpoint
and fixture hashes, **exact** same-profile packed codes ANDinverse norms after
save/reload. NativeFP16 parameters and FP32+autocast are distinct profiles;
use explicit nativeFP16 research inference atbatch32 with FP32 head and label
its reload equality as private256 mechanics, not public/native256 parity.
Keep saved FP32 training weights; compare exact loaded tensors before casting.
Select100seconds TOTAL for fresh128/256 pair,≤17updates each; no quality read,
calibration, downloads or tail timing. GPU code/gate is not yet implemented or
launched. The reviewer suggested1LSB tolerance is rejected: exactness is a
goal requirement. No timeout extension on failure.

For the subsequent real100update gate, training on the original13283fit images
requires evaluation on the original outer12599held images, fixed6354query/
6245gallery. The cached inner6514validation images are INSIDE originalfit;
using them after full-fit training would invalidate unseen-identity quality.
Thus "same panel" cannot mean the cached panel in this real-encoder stage.
Both arms must use the same outer TRAIN panel, labelled previously observed
and exploratory. Proposed floors, frozen before that future run: R@1≥+.5pp,
mAP@R≥+.01, both paired95% lower>0; nativeR1≥92.2%; trainwall/medianstep≤1.10×,
peakallocation≤native+1GiB;600sec total/280secchild including export. Save
checkpoints and held arrays for independent rescoring. These are new direct256
criteria; no change to old folded-budget KILL.

Before GPU implementation, accept one≤60sec CPU **step0 attribution** read of
PCA128/PCA256 and unit1024 on the SAME cached inner roles. Reuse the existing
probe/helpers; no additional training, schedule seeds, int4 format or new panel.
It separates initial width information from the observed100step contrast and
must not be described as independent validation. The proposed extra seeds/
int4 comparison are deferred: they add treatments before the real-encoder
mechanism survives, and actual cached terminal embeddings were not retained.

Corrections to reviewer reasoning: loss values from different sampled batches
are not a controlled loss slope and do not prove training failed; the stated
1e-2 parameter-movement bound is not verified. Integer256D int8 dot sums are
exactly representable inFP32, but multiplying inverse norms still uses the
declared float arithmetic, not exact real numbers. A width-initialization gain
does not prove a novel training algorithm. Opus's proposed kill condition
(trained delta≤initial delta AND initial delta<.5pp) cannot hold together when
the recorded trained delta is already+1.715pp; it is not a useful new selector.
Step0 is attribution, not a posthoc rescue/kill rule for the completed gate.

Production remains128. Native256 stable ties, full-call latency at matched
quality/hardware/batches/gallery, independent seeds, SOP/In-Shop/transfer and
official protocol qualification remain unproven. No official-gap forecast.
