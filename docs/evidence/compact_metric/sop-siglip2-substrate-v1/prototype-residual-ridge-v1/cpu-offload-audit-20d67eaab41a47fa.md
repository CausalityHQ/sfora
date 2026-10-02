**NO-PILOT for the current frozen evaluator.** Neither CPU qualification nor scoring is portable as a small bundle without changing admission gates. Leave session **16755** under root ownership; collect its v6 outcome before deciding the next run. All GPU work stays on DGX Spark.

| Slice | Evidence and implication |
|---|---|
| Admission | CPUv5 spent **96.982s** authenticating authority; scorev1 spent **111.753s**. The accepted CPU receipt has **15,174 file guards**, including **13,283 images**. These bytes must remain available for fresh exit hashing. |
| Strict CPU reload | Qualification reloads each complete fitted payload twice. Reconstruction authenticates warm state, canonical cache and the immutable encoder checkpoint. Exporting only residual coefficients cannot satisfy it. |
| Cache transfer | Original FIT array payload is **61,208,064 bytes**; each canonical/augmented TRAIN array is **29,283,840 bytes**, excluding file headers. Caches alone are insufficient. |
| Scoring | v6 combines TRAIN64 and full-panel witnesses into four independent reloads before candidate quality. The launcher exposes only `cpu` and `score`; no isolated metrics/bootstrap phase exists. |
| GPU | No cloud GPU is needed or justified. DGX-produced data dependencies remain mandatory even with CUDA hidden. |

Evidence: [evaluator](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:564), [strict reload](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:634), [CPUv5 receipt](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/evaluation-cpu-v5-receipt.json).

The concrete transfer blocker is **at least 3,423,153,584 bytes** across two required encoder files: `vision.safetensors` is **1,711,601,328 bytes**, and `fresh_vision.pt` contains **1,711,552,256 bytes** of FP32 vision tensors before serialization overhead. This excludes warm/fitted checkpoints, images, caches and runtime dependencies. The interpreter is pinned to **CPython 3.13.9 aarch64**, with authenticated package/native origins including NVIDIA libraries. A routine x86 or replacement CPU environment cannot preserve those checks. [Encoder provenance](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1/so400-vision-provenance-v1.json)

The [existing launcher](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/evaluation-source-v6/cpu-launch.sh) therefore cannot support a small EC2 slice unchanged. It also requires canonical paths, both locks, systemd resource enforcement, cold-cache setup and authenticated footers. Bootstrap arithmetic is separable mathematically, but candidate per-query results are unavailable and it is not an admitted standalone decision path.

**Bounded falsifier:** root may perform one **≤55s metadata-only inventory check** against CPUv5 guards and v6 authority: are all mandatory large files and the exact aarch64 runtime already present on the proposed host, requiring no large DGX transfer? Any missing dependency rejects the pilot before provisioning or copying. Passing only reopens feasibility review; it does not qualify execution.

For a future comparison, measure  
`provision + transfer-in + admission/reload/scoring/exit + transfer-out + root verification`.  
Transfer-in alone is bounded below by `3,423,153,584 / measured_bytes_per_second`. CPUv5 completed in **289.441s**; scorev1 consumed **300.258s without accepted candidate quality**. Neither establishes an EC2 speedup or a successful local time-to-decision baseline.

Current cloud spend: **$0**. Any later authorized falsifier should use one CPU On-Demand instance, no retries, unchanged **300s/8GiB/noSwap** phase limits, an external teardown deadline, and termination plus temporary-volume deletion on every exit. Root must set the dollar ceiling from an actual Causality quote; no price evidence supports a numeric estimate here.
