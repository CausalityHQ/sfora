# Connected RS@k mechanism preflight — SKIP_NATIVE (2026-10-14)

Source: devbox/connected-rsk-source-20261014 @14243875cf7ac568b5b14c09ba40d3abddc3b679 (5 new files on 59d37521; reviews 43be9982a6204751 fixes applied; 35 stdlib tests OK; opt-in real FP32 torch test run on DGX CPU OK).
Run: DGX CPU only, systemd unit sfora-connected-rsk-preflight-v1 (invocation 855e13a08ac84776af88ce2391ad6cf9), 41.7 s, swap 0, CUDA hidden. Inputs by exact SHA: FIT canonical.npy 55d37d06, CPU-v5 CONTROL initializers 8b89897f (061) / 1747e22e (069). TRAIN FIT rows only; selection panel and official never read.

Pre-registered rule: MECHANISM_PRESENT iff >=20 boundary anchors and equal-anchor boundary-pair fraction ratio RS@k/SmoothAP >= 2 on both seeds; SKIP_NATIVE iff a qualified seed has ratio < 2.
Result: 56 boundary anchors (first-positive rank histogram 1:6270, 2:43, 3:13, >3:17; 12 no-positive), ratio 1.888 -> **SKIP_NATIVE**.
Both seeds give identical numbers because the two initializers differ only in identity.seed / initial_cpu_rng_sha256 / cpu_rng (140,622 other entries incl. A/C identical, verified); a preflight at the initializer is seed-independent, so this is one measurement.

| diagnostic | SmoothAP | RS@k | KILLed hinge (calibration) |
|---|---|---|---|
| boundary-pair fraction, equal-anchor (mass-weighted) | 0.160 (0.173) | 0.302 (0.306) | 1.0 (1.0) |
| parameter cosine(-grad L, grad margin), A/C | 0.440 | 0.608 | 1.0 |
| first-order margin change, lr 1e-4 (indicative) | 0.0183 | 0.0244 | 0.0139 |

Reading: RS@k is ~1.9x more boundary-focused than SmoothAP but far from the hinge; the pre-registered 2x mechanism bar is not met, so the ~2.5-3 h native TRAIN128 is not requested. RS@k on the connected lane is closed at this budget without GPU spend (decision on mechanism, not quality).
