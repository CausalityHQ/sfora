# Fixed256 MAIN training to128 serving: mechanics smoke only

Dual group579a7c2e61c745dd completed: Opus6898634e564b4d5d and
Astra356ae7e7cdf2402d both conditionally GO for one mechanics smoke. No claim
of learned capacity or generalization follows from the CPU screen. Its34.91%
is a **norm fraction**, not energy; extra gradient directions largely follow
from extra orthogonal PCA rows. Folded prototype counts are unchanged, with
two individual flips. Whether training information survives a top-energy
compactor is the unresolved causal premise, and width changes the cosine
distribution, proxies, clipping pressure and bank temperature regime.

Verified correction to Opus: original `freeze_emb` preflight marks a whole
batch inactive when any singleton occurs, exactly as the CPU probe does.
The alternative `freeze_emb_rank` recovery arm is not used here. Preserve the
original policy, not the valid-anchor alternative. Existing768 helper support
also stays intact; only explicit256 training support is added.

## Frozen before any DGX launch

- Exactly native128 then wide256,17 updates each,seed179024, original13283/2004
  TRAIN fit, all17 pixel/target hashes matched. Large24blocks, frozen
  embeddings/first12, original BF16, same AdamW/groups/decay/LRs/global clip1,
  ArcFace0.3/64 plus8 detached bank and last-duplicate-view refresh. Wide
  initializer uses a forked RNG after native128 initialization; original
  native RNG stream is retained. No other arm/source/loss knobs allowed.
- Fresh native128 must replay all17 old control pixel hashes and losses
  within1e-5 absolute. Old control receiptSHA
  `83c977aa6d57fc5c0aff10824b8ffa0ac38c5d47cfb43c4e88679848e301063b`.
  Failure stops before wide training; no altered seed/order retry.
- Whole **paired** subprocess/setup/calibration/reload budget120s, external
  watchdog enforces it. Record per-arm diagnostic-inclusive training wall,
  synchronized step times, peak CUDA memory. No held/official images read.
- Finite losses, gradients, parameters, banks, optimizer states;17 successful
  updates; exact frozen-parameter digest and changed trainable-encoder digest.
  Record ArcFace/bank losses, vision/head/classifier gradient norms and global
  preclip norms. Initial/terminal wide/native encoder norms must be[0.25,4];
  median encoder gradient-share ratio[0.5,2]. Normalized variance/effective
  rank terminal must be≥50% of wide initial.
- Wide median diagnostic-inclusive step time≤1.05× fresh native; peak CUDA
  allocation≤native+1GiB. Ratios from17 steps are smoke cost guards, not
  production training-speed certification or a confidence interval.
- After wide training, calibrate once on first2048 original fit-order rows,
  no augmentation, trained encoder eval, existing pinned torchvision processor
  with established direct-processor parity, FP32 parameters/FP16 autocast,
  TF32 off (the public `fp32_autocast` profile). Deterministic float64
  uncentered SVD/sign convention,σ128>1e-10, orthogonality≤2e-5. Save spectrum,
  retained energy, compactor/fit-row/parent/checkpoint hashes. This subset is a
  mechanics fixture; any later quality gate requires once-only **all fit rows**
  calibration from trained encoder outputs before reading held images.
- Fold to1024→128, check actual normalized FP32 two-stage/fold error≤1e-5.
  Reload standard folded checkpoint with `Siglip2CompactEncoder.from_checkpoint`
  and encode first64 fixed fit images through public preprocessing. Require
  exact packed codes/inverse norms versus in-process GPU folded reference.
  Raw256 checkpoint must fail native loader on shape. Two-stage versus folded
  packing may differ; record flip rate and require every flip≤1LSB. No
  two-matmul bit-exact claim, no new serving width/processor/kernel.

Any failed guard KILLs this fixed arm, no full run/threshold change. Exceptions
and watchdog termination preserve child logs and mean failure, not a pass.
A pass only authorizes separately freezing a100-update packed TRAIN-held
comparison; no official evaluation, multi-seed/full training or latency yet.
Both reviewers' quality proposals remain prospective: freeze one rule before
that comparison, using the actual folded128 scorer and product uncertainty.
Production default/checkpoint stays unchanged throughout this smoke.

## Qualification wiring repair

Original unit `sfora-inshop-wide-head-smoke-v1`, invocation
`afacf43986434ead80b4f563462f41a0`, exited1 after both17-update arms and
fit-only calibration. Native128 replayed every original loss and input hash;
wide256 remained finite. The public encoder correctly rejected the harness's
64-image call because its maximum is32. This is invalid qualification wiring,
not a quality result or a completed smoke pass.

Repair only the harness: split the fixed64 images into two32-image public
calls, preserving image order and existing checkpoints/calibration. Add one
regression check for that bound. No training repetition, encoder/source/loss/
selection/compactor changes. Original service elapsed91.401745s, conservatively
charged92s. One qualification-only repair unit
`sfora-inshop-wide-head-qualify-v1`, invocation
`5ddcd438e8df4cea8a81de3f39f3da79`, is allowed the **remaining28s only**;
whole120s budget is unchanged. Timeout/failure stops escalation. Both original
logs/status and repaired qualification logs/status remain separate.

## Terminal and root cause

Qualification-only unit exited0 with scientific decision **KILL**: all guards
passed except `exact_native_reload`. ReceiptSHA256
`e23a5f2c646ea437885e56a049b7611c3bf3f56fefb73b070cb0840928274921`.
Conservative combined main wall106.200193s. This decision is preserved; it is
not proof that wide supervision loses quality because no retrieval evaluation
was done and the comparator's batch shape was wrong.

| In-Shop official TRAIN fit-only,17×64 smoke | Fresh native128 | Wide256 |
|---|---:|---:|
| Successful updates |17|17|
| Diagnostic-inclusive training wall,s |16.029435|15.997651|
| Training images/s |67.8751|68.0100|
| Median step wall,s |0.8426703|0.8391910|
| Peak training CUDA allocation,bytes |12246201856|12259986944|
| Terminal normalized variance |0.9028247|0.8990791|
| Terminal effective rank |23.54953|26.60400|

Initial/terminal encoder-gradient norm ratios0.982334/0.928073, median encoder
share ratio1.000673. Actual normalized fold error9.536744e-7; two-stage/folded
packing had zero differences on the saved64 poolers. There is no R@1/mAP@R,
end-to-end p50/p95/p99/QPS, quality uncertainty or production-speed result.

One bounded read-only root-cause unit
`sfora-inshop-wide-reload-diagnosis-v1`, invocation
`5eee62deaac042b8b10f9f402a133885`, exited0, main10.160368s. It did no optimizer
updates or recalibration and used the same saved head/compactor/checkpoint.
The reloaded batch64 poolers/folded codes/inverse norms exactly reproduce the
original in-process batch64 reference. Splitting only the head arithmetic
into32 also reproduces it. Splitting the **encoder** into32 changes poolers
by at most0.01953125, folded normalized outputs by0.000540078,77/8192 code bytes
and23/64 inverse norms. Public32 is exactly equal to in-process32. The failed
check therefore compared different encoder batch shapes, not a damaged
checkpoint or a packed-search kernel mismatch.

Repair future calibration/reference acquisition to batch32, collecting the
first two batches for the fixed64-image probe and folding each at32. Production
serving code stays unchanged; no cross-batch bit-invariance claim. The regression
checks the public32 limit and ordered64 concatenation. Fresh folded/serving
tests and an isolated wheel check pass. Original failed receipts/artifacts stay
unchanged, with no rerun of training or replacement compactor. Next required
gate is a correct matched-batch qualification using the existing artifacts;
freeze any100-update quality gate separately only after that check. No new
model search, held or official evaluation was launched. Spark and reviewers
are terminal/idle; no operator decision is needed.
