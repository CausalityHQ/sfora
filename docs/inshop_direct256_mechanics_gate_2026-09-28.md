# Frozen direct256 paired mechanics gate

This follows cached quality GO and reconciled Opus5.5/Astra conditional review,
including step0 attribution. It is one actual encoder test, not another width
search, official quality read, public format change or reopening of folding.

Fresh128 then256 children use the existing TRAIN trainer with
`--direct-width-receipt` bound to cached receipt
`1ac62b0edda6dd93bc808a8d2bb81fef7c89cc1c4eaef816d276c4fecb2a555f`.
The authority allows ONLY freeze_emb,seed179024,17updates,first12blocks frozen,
1e-5visionLR,128/256, no other research arm or receipt. The old folded gate
and its authority remain distinct. No calibration or held export is called.

Pinned preflight
`f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034`,
original13283fit images/2004products, frozen pretrainedLarge2561024source cache,
BF16 live training, same17×64 augmented tensors/targets, native ArcFace.3/64
plus8detached bank, AdamW/clip/coverage/singleton handling unchanged. Existing
initializer/head/bank/helpers already accept256. Leading128weight ANDbias
hashes must match. Training-coordinate receipt metadata now correctly records
the actual width; historical folded receipts are not rewritten.

External service100seconds TOTAL covers runner startup, BOTHchildren, setup,
training,save,reload and receipts. Child subprocess timeout uses remaining
budget; SIGTERM writes a failure receipt and the service controls all child
processes. Host MemoryMax32G; shared DGX GPU lock. No retry, extra steps or
deadline extension after failure. Persist completed control and raw checkpoints
even on later-child timeout; regression verifies this failure path.

Require every17update successful, finite loss/gradients/parameters/optimizer/
refreshedbank, nonzero vision/head/classifier gradient norms; exact frozen
parameter hashes; changed trainable encoder; terminal normalized variance and
effective rank≥50%initial. Actual17pixel/target hashes, fit/held/cache/model/
source/preflight/schedule/recipe fields equal between arms.256medianstep≤1.10×
128; training peakCUDA≤128+1GiB. All source manifests bound and unchanged.

Private nativeFP16 reload profile: verify saved FP32 training tensors exactly,
then cast vision parameters toFP16; head staysFP32, TF32off, explicit disabled
autocast. Reuse pinned processor and existing normalized head/pack functions
atbatch32 on64TRAINfit images. Fresh strictly loaded model must reproduce ALL
packed codes ANDFP16 inverse norms exactly. Save pixels/codes/norms fixture,
checkpoint and image hashes; rows130/258bytes. This is **private reload parity**,
not public256 loader, native256 search, quality or serving certification.

KILL on any failed guard/deadline/cost limit. A pass licenses separately frozen
100update paired TRAIN quality only; no default/API/kernel promotion. Subsequent
quality uses original outer12599held identities (fixed6354q6245g), not the inner
panel included in full-fit training. Current128 production path is unchanged.

Before launch: authority/timeout-preservation regression passed; default19 and
direct20 source manifest paths checked; CLI imports and Python compilation
passed. These local checks do not prove actual GPU mechanics. Original DGX
jobs checked idle and2.3TiB disk headroom before staging; inspect again atlaunch.
