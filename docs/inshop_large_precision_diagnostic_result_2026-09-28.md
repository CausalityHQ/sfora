# Large BF16 mismatch identified; FP16 mechanism review GO

One forward-only DGX diagnostic, invocation `5308d73c46804c24829e11df0a066cf6`,
completed successfully:15.85s whole process,2,508,016KiB peak host RSS,
1,938,665,472B peak allocated CUDA. No optimizer, retrieval quality or PE forward.
Original FP32 Large weight bytes stayed unchanged. Same four official In-Shop
TRAIN-fit rows0,1,2,5 and native256 pixels as the failed mechanics guard.

| Four-row feature comparison | Minimum cosine, verified | Fixed0.999 floor |
|---|---:|---|
| Fresh FP32 versus FP16 eval |0.9999095201|PASS|
| Fresh FP32 versus BF16 eval |0.9974756241|FAIL|
| Cached FP16 versus fresh FP16 eval |0.9999963045|PASS|
| Cached FP16 versus BF16 train |0.9971187115|FAIL|

BF16 train/eval output arrays were byte-identical, maximum absolute difference0.
This reproduces both BF16 guard deficits with saved vectors and excludes training
mode and gross source/input alignment as causes for these rows. It does not
backfill the exact unlogged values of the previous job. Full-source accuracy,
training stability and generalization remain unmeasured.

Saved NPZ SHA`143e38936b5ddf78d3e63048beb03537d16fcbf645c7d04ce734b1a98c3f133b`;
dependency-free float64 scalar cosine replay passes within5e-7 of GPU values.
Reproduce with `python3 docs/evidence/compact_metric/sop-siglip2-substrate-v1/large-precision-v1/verify.py`.
Raw JSON/vectors/process cost/launch in that directory. Frozen code `bc9aaf0e`.

The responsible layer is the numerical profile. Keep the BF16 configuration
closed and the0.999 floor unchanged. The separately declared successor uses
matched native FP16 plus existing GradScaler128 in both arms, original FP32
weights, the same head/bank/objective/pixels/schedule and resource limits.
[FP16 gate](inshop_pe_fp16_training_gate_2026-09-28.md) and its concrete dual
review require no scale/threshold/cap rescue and permit no automatic quality read.

The production joint quality-and-speed goal remains active and unproven.
