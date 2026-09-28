# L14 own full-fit initialization cache: source prerequisite PASS

The sole GPU acquisition passed, followed by saved-matrix/reference CPU
replay. Native L14 now has its own complete TRAIN-fit features for PCA,
compact head, proxies and bank initialization. No B16 feature/initializer
reuse, optimizer update, held decoding or quality read occurred. The actual
native training cost and retrieval quality decision remain open.

| Dataset / split | Relevant historical controls R@1 / mAP@R (%) | L14 R@1 / mAP@R | Training cost / acquisition | Public latency | Remaining gap / next decisive test |
|---|---|---|---|---|---|
| In-Shop TRAIN-fit13283images/2004products/12singletons; target TRAIN-held6354q/6245g/1993products excluded here | prior verified100update densePE95.0739692/76.3915922; Large95.6720176/78.6237120, not concurrent reruns | Unmeasured | Training unmeasured; verified fit export186.4063s/71.2583images/s; whole196.49s;1,590,978,048B allocatedCUDA;3,618,732KiB processRSS | Unmeasured | Quality/full-pipeline speed requirements remain unmet; own initialization+actual native half-prefix12 CPU qualification, then one17step mechanics and fresh100TRAIN pilot only on PASS |

Historical same-fit, B32 acquisitions exported Large in118.5661s at112.0303
images/s and B16 in53.3109s at249.1608images/s. L14 acquisition is slower.
These are prior source export measurements with their native preprocessing,
not simultaneous matched-quality training/serving comparisons. No public
latency or training-speed win follows from this acquisition.

Original FP32 checkpoint, pretrained336 config and fixed native224 RGB
interpolation were preserved. First4fit FP16-versus-FP32 cosines minimum
.9998070598 passed unchanged.999 floor. All13283×1024 descriptors are unit
FP32, maximum float64 norm error9.9377606e-8. Saved-cache first4 versus FP32
references minimum.9997935353 (independent float64 replay); versus FP16
.9999859034. Registered source state and foreign rotary/grid hashes stayed
exact. Both lifetime locks and300s whole/8GiB host/no swap/<10GB CUDA held.
No input, precision, floor, prefix or budget rescue occurred.

CPU preflight5.04s/976,236KiB RSS, GPU original session10124 exited0 in196.49s,
invocationf8d29092506e436880a8e63f134f57dc. CPU audit original5586 exited0
in4.01s/1,154,244KiB RSS. Report `/usr/bin/time` process RSS; systemd's tiny
CPU reports and2.6G GPU aggregate report are different accounting measures.
Stdlib execution/authority receipt replay also passed. DGX idle afterward;
no jobs, research or critique are active.

DGX matrix retained at `/home/riomus/runs/sfora-pe-l14-fit-v1/l14.fit.npy`, SHA
`2b88edda325368801fd5a6ec5a79672be1b90e18d7e36803a316595ade63974a`.
Receipt SHA `9f9a1c6e91740cbd8c80fd5e90c7f0e631e000e6a031eb1276e94ab11f8a870c`;
CPU audit SHA `c19ca161820bdda4c316b839a37b4d6daa9b229b2585c7382c2038e420309819`.
No acquisition/source blocker remains. Next qualify the actual native24block
half-prefix12 training inventory and own full-fit initializer, preserve
unchanged100 RGB schedule/loss/scaler/AdamW control, discard17mechanics state,
and evaluate one fresh100pilot only after actual cost/integrity gates pass.
Frozen update ceiling.71769696s, quality floors95.1720176% R1/77.6237120% MAP
and positive paired product95% lower bounds versus densePE remain unchanged.
A survivor advances to updated serving/fresh official/transfer qualification.
Full SOP+In-Shop production joint goal remains active and unmet.

[Receipt](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-fit-v1/receipt.json),
[CPU audit](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-fit-v1/cpu-audit.json),
[execution replay](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-fit-v1/execution-replay.json),
[GPU cost](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-fit-v1/l14-fit-gpu-v1-time.txt),
[frozen complete inventory](evidence/compact_metric/sop-siglip2-substrate-v1/pe-l14-fit-v1/preflight.json),
[frozen procedure](inshop_pe_l14_fit_gate_2026-09-28.md).
