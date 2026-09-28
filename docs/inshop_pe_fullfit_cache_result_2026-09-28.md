# Paired full-fit source cache: PASS

Frozen exporter `6623cc43` completed one serial DGX acquisition of both original
sources on **official In-Shop TRAIN-fit only**: 13,283 images/2,004 products,
including 12 singleton products. The class-disjoint 12,599 held images/1,993
products were excluded from decoding. No optimizer or retrieval score ran.
This supplies complete initialization data for the next quality comparison;
**it does not establish learned quality or image-to-top-k performance**.

[Gate](inshop_pe_fullfit_cache_gate_2026-09-28.md). Sole DGX user service
`sfora-pe-fullfit-cache-v3`, invocation `d8207111dfb34a73bf2403f120d2af02`,
exit 0 authenticated by the original timed-process record and completion log.
480 s/8 GiB host cap and shared GPU lock enforced; no competing GPU job.
Original FP32 weights, native FP16 inference, TF32 disabled, batch 32,
unaugmented RGB; Large/256 then PE-Core-B16/224, one tower resident at a time.

| Verified acquisition measure | SigLIP2 Large/256 control | PE-Core-B16/224 |
|---|---:|---:|
| Complete source matrix |13,283 × 1,024 FP32|13,283 × 1,024 FP32|
| Export time |118.5661 s|53.3109 s|
| Export throughput |112.0303 images/s|249.1608 images/s|
| Arm time including initialization/audits |129.0234 s|54.5514 s|
| Peak allocated CUDA |1,592,031,232 B|582,803,968 B|
| CUDA allocated after cleanup |0 B|0 B|
| First-four FP16 versus FP32 cosine minimum |0.9999576807|0.9999859333|
| Prototype-anchor cosine minimum, CPU float64 replay |0.9999999999999998|0.9999999991846333|
| Maximum unit-norm error, CPU float64 replay |1.3497055e-7|1.0756757e-7|

Export timing includes file rehash, decode, native preprocessing, encoder,
normalization, CPU transfer and atomic matrix writing. It excludes initial
model loading and subsequent anchor audit. These are single serial source-cache
costs, without a confidence interval, and have no search/packing/QPS/p99 claim.
PE/Large export-time ratio is 0.4496304977; this descriptive ratio is not a matched
production serving result. Whole timed process **187.55 s**, cumulative peak
RSS **2,799,928 KiB**; in-script whole time **184.2853 s**.

Independent saved-matrix CPU audit passed in **2.22 s**, **991,812 KiB** peak
RSS, exit 0, invocation `9c6611fedd51406cb74e6059e6124894`. It rehashed both
actual complete matrix files; checked shape/dtype/finiteness/unit norms across
all rows; reconstructed fit/held membership from the official partition;
checked path/product order and authenticated prototype anchor mapping; replayed
all 1,024 anchor cosines in float64; verified receipt precision/resource gates,
external manifest SHA and unchanged source-code files. This does not replay
FP32 image inference; those captured cosine values come from the GPU run.

CPU preflight v3 and the runnable tampered-manifest startup rejection also passed.
Focused Opus review accepted the design; the external manifest-integrity finding
was fixed before freeze. No source, precision, floor or budget rescue occurred.
All GPU/CPU/review jobs are terminal and the DGX is idle.

Matrices remain on DGX at `/home/riomus/runs/sfora-pe-fullfit-cache-v3/`,
about 52 MiB per arm. Matrix SHA-256:

- Large: `8f0b5c2db1da5637ed57eca868d7c82d365574a7e83b47a6187e0e448e17c6e8`
- PE: `0173d468cc27fd4b4f563b67165fc1d5d7ed99cf12409a73e2738d2d0eb37abe`

Receipt SHA `f71852c782bc700f5a269eba120e291e1d4d1a5c2e20b5410effa4284d121add`.
Raw manifest, review, launcher, per-arm receipts/cleanup, original process costs,
terminal journal/idle check and CPU audit are archived in
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/pe-fullfit-cache-v1/`.
Use `scripts/audit_inshop_pe_fit_features.py` with the recorded root/cache and
model/dataset paths, CUDA hidden, to replay this result.

Next gate: freeze one paired **actual augmented image-training** feasibility
comparison with each source's complete fit-only PCA/proxy/bank initialization,
matched image schedule and RGB augmentations, the qualified native FP16/scaler
route, and predeclared early stop rules. Use the existing asymmetric TRAIN-held
protocol: **6,354 queries / 6,245 gallery**, packed R@1/mAP@R and paired
product/query uncertainty. The older 97.6235% held R@1 is a stale diagnostic,
not this candidate's result or an official benchmark. Obtain the consequential
Opus/Astra critique and CPU mechanics/startup checks before that GPU run.
No automatic official evaluation or release. The production joint goal remains
active and unmet; no operator decision or access grant is needed for this route.
