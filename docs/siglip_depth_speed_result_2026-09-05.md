# Fixed SigLIP depth preflight: speed passes, recovery quality unmeasured

The original fixed27-to18-layer seed17 preflight completed successfully on DGX
GB10. This is an engineering compression result, not a novel method or a SOTA
claim. The frozen384px input grid and512D output are unchanged.

| Window | Encoder full/student p95 ms | Pipeline full/student p95 ms |
| --- | --- | --- |
| 0 | 634.454 /425.751 | 674.660 /468.267 |
| 1 | 634.742 /425.890 | 685.114 /472.106 |
| 2 | 633.916 /425.847 | 683.225 /474.479 |

All six student/full ratios pass the preregistered<=0.75 gate. Pipeline p95
falls30.6–31.1%; encoder p95 falls32.8–32.9%. Each is batch8,100 measured
samples following10 warmups; order alternates within paired rounds. Raw timing
values independently recompute to the reported p95s and integer ratio gates.

## Authority and execution

- Canonical result `/home/riomus/siglip-depth-speed-21849eb5.json`, local copy
  `/tmp/siglip-depth-speed-21849eb5.json`, SHA256
  `90bc6846060e9df54ef344857eb7cc0d52433d4613865320cc53ea89abedadda`.
- Immutable deployment `/home/riomus/sfora-depth-speed-21849eb5-4331289b`,
  runner21849eb5/core4331289b, full hashes in the foundation plan and receipt.
- Exact original session66791/PGID1776443 completed exit0; wrapper761.489323s,
  script758.251373s. No restart; subsequent GPU/PID clearance confirmed.
- Peak allocated CUDA3,961,049,088B, reserved5,351,931,904B;
  whole-process peak RSS32,871,514,112B includes eager source decoding.
  Monitor PSI samples0, no swap-growth stop.
- Parameter counts428,840,512full and291,684,976student include49 training
  proxies; these are not deployment-file byte counts.
- `claim_eligible=false`, `quality_measured=false`, optimization steps0.

## Input-access deviation and prospective repair

The original runner reused an eager source loader. It decoded nonselected
images before choosing128 optimization examples. Only those128 entered either
model; no evaluation scores or updates occurred. Therefore this receipt proves
matched speed, but not an optimization-only pixel-access boundary.

The separate metadata-first loader validates all label/ID authority first and
then fetches only optimization image rows. Its independent real-data check
completed exit0, no GPU model,128selected/128pixel reads. IDs and original RGB
digest match the speed receipt exactly:
`2d0ab3bf1a901e20eaba032ca5ab4bcafacc1535622a152c6c87a6c9b692dd74`.

Input proof `/home/riomus/siglip-depth-inputs-e66f1d5a.json`, SHA256
`effeabc19451897d2cb5b75de1347b671c9071694aad7e23a8b92472d7197d45`.
Loader SHA256`e66f1d5ab99a87065d48cbb7203f34d5dde157dd6a0ed855a7f2c5146c6b1cd2`.
Internal time0.354420s, wrapper1.334707s, peak RSS435,531,776B. This fixes
future input access; it does not retroactively claim isolation for the old run.

## Next decisive boundary

Ten disposable recovery updates per arm must establish valid gradients and
time/memory feasibility. Then reload the same pruned initialization for fixed
198-update PA-only and PA+frozen-teacher relational arms. Neither smaller-model
quality nor six-hour feasibility is established by speed alone. Current
incumbent quality remains three-seed94.50% Recall@1 /79.07% MAP on exposed
development evaluation. Fresh matched qualification remains necessary for SOTA.
