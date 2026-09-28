# Frozen centroid-PCA paired100 TRAIN quality gate

Positive qualification: paired17 receipt SHA
`9ae62ed93a7546e4d71b11b12a145bdca23908ec781e60731340d44fd9744e1f`.
Freeze before either100 arm or any new held scores. Same seed179024,
Large256/freeze_emb/native128, all13,283 fit images/2,004 products,
normal ArcFace0.3/64 and bank8, batch64, LR1e-5/head+proxy1e-4,
AdamW0.05, BF16 vision and100 updates. Native then products, serial on the
sole DGX Spark GPU. Refit the same full-fit initializers as the qualified17
smoke. Only image versus equally weighted product-mean PCA basis+centre
changes; no scaling, new sources, proxies, covariance or loss search.

Both arms are fresh source-paired runs, with first17 pixels/losses matching
their corresponding qualified17 receipt (pixel hashes exact; loss absolute
1e-5, relative0). All100 pixel and schedule hashes match between arms.
Require finite100 steps/parameters/bank/optimizer, fixed lower vision state,
recoverable initializer/checkpoint/source authority and exact live-public
packed parity at two32-image fit batches. Save raw training evidence before
public parity/export; on child failure retain checkpoint/logs/last completed
result and close the fixed arm, without manufacturing missing quality.

Held official TRAIN only: 12,599 unseen-product images/1,993 products,
fixed6,354 query/6,245 gallery roles. Digests are the existing
`9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b`
held rows, query `89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68`,
gallery `e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3`.
Export at batch32 with matching FP16 encoder autocast/FP32 weights and
native128 packed signed-int8/f16-norm score arithmetic with ordinal tie
semantics. Reuse the packed oracle and5,000 product-cluster bootstrap,
seed179019. Report R@1 and mAP@R per arm, paired effects/95% intervals,
rescues/regressions and training/resources. This reused TRAIN panel is
exploratory; the intervals condition on the selected seed and panel.

Advance only if all gates pass:

- Packed treatment-minus-native mAP@R point >=**+0.005** (+0.5pp) and
  product-bootstrap95% lower **>0**.
- Packed R@1 point delta **>=0**; report its full interval.
- Training wall including bank initialization and median step2–100 each
  **<=1.05×** native; peak allocated CUDA **<=1.005×** native.
- Whole treatment arm wall **<=1.10×** native; total campaign **<=600s**,
  each child **<=280s**. Cost includes actual diagnostics as executed.
- Each arm retains >=50% initial compact variance and >=80% initial
  effective rank. All provenance, paired inputs and live parity guards pass.

If control packed R@1<92.2%, stop before treatment as invalid control
(existing runner floor); source/finite/parity/budget errors also stop early.
Otherwise complete both arms once, including negative quality, then KILL
on any failed acceptance rule. No extra seed, longer budget, initializer
retune or official read after a failure. A pass permits only a separately
frozen paired-seed quality gate; no product default or SOTA claim. Serving
latency/QPS is unmeasured for these checkpoints; no serving-cost projection.

## Terminal: GO to paired-seed design

Sole DGX unit `sfora-inshop-centroid-pca-100-v1`, invocation
`5176c5b8f4fc41799d26e7a6ce80806d`, exited0. Whole campaign **457.250924s**;
receipt SHA-256
`d6f0d09de37c9121f6358a95514fb312f04d56800d719045b6e09132c508c214`.
Both arms completed100 stable encoder updates, all100 input batches match,
and the qualified17 prefixes and initializer hashes replay. Exact saved/live
tensors and live terminal/public packed codes/norms pass at two32 fit batches.

| Official TRAIN held6,354q/6,245g, seed179024, same Large256/native128 | Image-PCA control | Product-mean PCA |
| --- | ---: | ---: |
| Packed R@1 | **95.247088%** | **95.687756%** |
| Packed mAP@R | **0.7798550375** | **0.7918794856** |
| Training wall excluding bank/init | 86.345588s | 86.045028s |
| Training wall including bank initialization | 87.667472s | 87.269351s |
| Images/s including bank initialization,6,400 samples | 73.0031 | 73.3362 |
| Peak allocated CUDA | 12,988,997,120 bytes | 12,988,997,120 bytes |
| Median step2–100 | 0.843707s | 0.841400s |
| Initializer wall, reported separately | 1.453019s | 0.478495s |
| Held feature export wall | 109.402426s | 108.949881s |
| Whole child arm | 224.117898s | 220.830331s |

Paired R@1 gain: **+0.440667pp**, conditional product95% interval
**[+0.192305,+0.694874]pp**. Paired mAP@R gain: **+1.202445pp**,
**[+0.995367,+1.415658]pp**. All nine acceptance rules pass. Independent
row-by-row generation verifies both5,000-draw product-bootstrap interval
endpoints exactly. The native arm's6,354 hit/AP vectors are identical to
the earlier source-MAIN native100 control. Source/checkpoint/role authority
and all100 finite steps are preserved in raw receipts and logs.

**Product decision:** keep the fixed product-mean initializer as experimental
training functionality; stop initialization/covariance/loss search. Freeze
one full-budget paired-seed confirmation before promotion. This result is
an exploratory single-seed100-update effect on previously observed TRAIN
identities; it does not establish full-budget, seed-independent, official,
SOP/CUB/Cars-transfer or SOTA quality. No image-to-top-k latency/QPS was
measured and export wall is not serving latency. All production defaults
remain native pending the joint quality/cost/public-serving gates.

Evidence is under
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/inshop-centroid-pca-100-v1/`.
Both raw checkpoints, full initializers and held feature arrays remain at
`/home/riomus/runs/sfora-inshop-centroid-pca-100-v1/result/`.
