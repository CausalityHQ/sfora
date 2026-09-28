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
