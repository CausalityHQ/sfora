# Actual direct256 encoder mechanics: GO to frozen real quality gate

Original DGX Spark unit `sfora-inshop-direct-width-smoke-v1`, invocation
`f6901805c7c04e6f868627c9802bddc2`, finished exit0/inactive/MainPID0.
Both fresh arms completed17successful updates within **77.522167seconds total**.
Protocol: In-Shop official TRAIN original13283fit images/2004products,
seed179024,17×64augmented presentations per arm, SiglipLarge256,freeze_emb12,
BF16training. No held/official query/gallery quality was evaluated.

| Measured short mechanics training,GB10 | Native128 | Direct256 |
|---|---:|---:|
| Updates/presentations |17/1088|17/1088|
| Training wall,s |16.003651|16.121083|
| Training+bank initialization,s |16.859738|16.877093|
| Presentations/s,including bank initialization |64.5324|64.4661|
| Median step,s |0.835091|0.837717|
| Peak allocated CUDA memory,bytes |12,246,201,856|12,259,986,944|
| Packed row bytes |130|258|
| Same-profile packed checkpoint reload |Exact64fit images,batch32|Exact64fit images,batch32|
| R@1/mAP@R,full-call p50/p95/p99/QPS |Unmeasured|Unmeasured|

All frozen guards passed:17matched actual pixel/target hashes, identical
leading128PCAweights/bias, same fit/held/cache/model/source/schedule/recipe,
positive finite vision/head/classifier gradients and finite parameters/
optimizer/refreshedbank, exact frozen parameter preservation and changed
trainable encoder, terminal variance/effective rank above50%initial. Direct256
median step was within1.10× and peak allocation withinnative+1GiB.
No calibration, fold, held export or alternate recipe was invoked.

The private checker verified saved FP32 tensors before casting, then compared
nativeFP16vision/FP32head outputs after strict reload with identical processor,
TF32off and batch32. **All codes and FP16 inverse norms matched exactly**;
fixtures contain64fit-image pixels/codes/norms. This establishes private
checkpoint mechanics, not compatibility with the current128-only public
loader/native scorer. CUDA allocation is tensor allocation, not total device
or reserved memory; the short serial smoke is not a training-speed claim or
full-training cost estimate. No uncertainty interval for cost was measured.

Decision **GO_FREEZE_REAL_100_GATE**: separately freeze and implement one
paired100update TRAIN-quality gate on original outer12599held images,
6354query/6245gallery. The previous cached inner panel is inside full fit and
must not be used as unseen validation here. Save held tensors for independent
rescoring; retain new direct256 authority separately from the killed folded
method. No new consultation for the same design, native256 implementation,
public flag, official evaluation or long serving run is licensed yet.
Production128 remains unchanged, full joint goal remains unmet.

[Receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-direct-width-mechanics-v1/receipt.json),
[independent verification](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-direct-width-mechanics-v1/verification.json),
[journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-direct-width-mechanics-v1/journal.txt).
SHA256`5334209bf78c70b08e0dd20bd55572e57130f2b31ef5570ebdc2ce5c535b2333`.
Verification rebound all20loaded source files per arm and replayed histories,
cost, geometry, counts and decision; original GPU reload was not rerun.

Remote artifacts remain under
`/home/riomus/runs/sfora-inshop-direct-width-smoke-v1/result/{128,256}/`.
Checkpoint128SHA`731500afc2011a3101f47e60e94faa1b81f99a58fa9a29e174182835906eca9b`;
checkpoint256SHA`e06fb42ee9858a9b33f7ca9536f5a0ded9168a1466bc934aab57191c6541c787`.
Both rawFP32checkpoints and private fixtures are retained. GPU lock is free;
no CPU/GPU/consult/test/build is active at this terminal.
