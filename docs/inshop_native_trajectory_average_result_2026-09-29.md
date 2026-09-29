# Fixed native trajectory average: TRAIN GO

The prospectively fixed averaging procedure is **GO** on the previously observed
DeepFashion In-Shop TRAIN-held split. Integrity, actual updated CPU/public
serving, independent CPU quality replay and resource gates pass. The next step
is the **same** ten-snapshot rule on the preserved corrected full-TRAIN trajectory,
then updated official serving qualification and fresh confirmation. The full
production SOP/InShop quality-and-speed goal remains unmet.

| Dataset / split | Matched endpoint R1 / mAP@R | Fixed average R1 / mAP@R | Training / averaging cost | Public latency | Remaining gap / next decisive test |
| --- | --- | --- | --- | --- | --- |
| In-Shop TRAIN-held, previously observed6,354 query /6,245 gallery;1,993 held identities, disjoint13,283-image /2,004-identity fit pool; nativeFP16/B32/packed128 |97.576330% /84.973026% |97.875354% /86.021606% |Same archived2,000-update trajectory:2,317.711798s for128,000 images,55.226884 images/s; zero new training updates; CPU export/average55.128321s, capped unit57.961s |Unmeasured; qualification wall times are not latency/QPS |Full corrected official endpoint remains96.307498%,0.392502pp below necessary dated96.7 screen. No official projection from this TRAIN gain. Apply exact rule to full corrected trajectory and qualify it; strongest current comparable external, SOP, transfer and paired speed gates remain open. |

All quality numbers in the table are verified by independent CPU replay of
complete saved packed wires. Candidate-minus-endpoint product-cluster95%
intervals are R1 **+0.299024pp [+0.032203,+0.565335]** and mAP@R
**+1.048580pp [+0.774560,+1.317966]**. Query95% intervals are respectively
[+0.062952,+0.550834] and[+0.819841,+1.286889]pp. The unchanged5,000-draw
bootstrap seed179019 conditions on one archived trajectory; it does not
estimate training-seed uncertainty. The predeclared gate requires positive
product R1 lower bound and nonnegative mAP point effect; both pass.

The rule uniformly averages trainable vision and head tensors in FP64 at
steps1100,1200,...,2000, casts back to original dtype, and copies frozen
parameters and runtime buffers exactly. All ten physical checkpoint hashes,
identity/roles/source/schedule/initializers and finite serving states were
authenticated. No snapshot/window/alpha chooser, new training, bank/classifier/
optimizer/scaler/RNG averaging, or resume-from-average claim exists. The
existing `average_model_states` helper was reused unchanged.

Post-result disclosure: this TRAIN gate used the original `large-optimization-half`
trajectory, whose objective omitted rank loss on630/2,000 whole batches. The full
application used the corrected `full-valid-anchor` objective. The averaging
arithmetic was identical; the training objectives and exposure regimes were not.
The within-trajectory TRAIN GO remains verified, but it did not qualify a matched
transfer of the averaging effect to the corrected full recipe. Future corrected
training comparisons require a control with that same objective. The original
frozen specification and receipts remain unchanged.

Actual CPU endpoint/average qualification passed in26.091/23.697s. Public
B32 endpoint/average qualification passed in131.710/129.364s, including all
12,599 images, all6,354 native top10 ordinals and float32 score bits, and
independently reloaded original-processor sentinels for each role. CPU audits
passed in12.724/12.676s; paired decision in12.600s. Peak allocated CUDA was
1,528,136,704B for both public jobs. All successful units respected the fixed
120s CPU/300s GPU execution caps,8GiB cgroup guard and zero swap. CPU export
used the full8GiB allowance with mmap/file-cache reclaim; it completed within
the cap. Archived training cost is not a new paired speed measurement.

The first endpoint public attempt failed after image export, before any
quality receipt, because systemd's PATH omitted the already authenticated
`tileiras` compiler. A4.201s B32×6,245 packed native canary with the corrected
PATH passed exact ordinal and score-bit parity. The terminal failed unit,
logs and timing are preserved. The successful endpoint attempt is **unit-v2**;
the frozen decision's phase-log path explicitly symlinks its genuine v2 log
after SHA-equal preservation of the failure as `endpoint-public-failed-v1.log`.
No claim that v1 passed is made. Weights, source closure, scientific rule,
thresholds and caps were unchanged. See the explicit
[launcher and attempt mapping](evidence/compact_metric/sop-siglip2-substrate-v1/native-trajectory-average-v1/launcher-and-attempts.json).

Evidence:

- [Frozen prospective procedure](evidence/compact_metric/sop-siglip2-substrate-v1/native-trajectory-average-v1/fixed-trajectory-average-plan.md),97-entry executionSHA`765a5a01fd2163ec2a22230620c937646c7a0cdd3a1f38dedaa16971fcd64027`.
- [Export receipt](evidence/compact_metric/sop-siglip2-substrate-v1/native-trajectory-average-v1/export-receipt.json); serving weights remain `/home/riomus/runs/sfora-native-trajectory-average-weights-v1/native.pt`, SHA`e80926aecf7b6c5b9b2c2cc249d5db4dab6584e05ea71e87a76518b65eb5319b`.
- [Endpoint audit](evidence/compact_metric/sop-siglip2-substrate-v1/native-trajectory-average-v1/endpoint/cpu-audit.json), [average audit](evidence/compact_metric/sop-siglip2-substrate-v1/native-trajectory-average-v1/average/cpu-audit.json).
- [Frozen paired decision](evidence/compact_metric/sop-siglip2-substrate-v1/native-trajectory-average-v1/decision.json), SHA`51487110ff2c6f225e54e3ed1e41a2240d32afd2197b5a4b39143cb1e972c509`.

Existing separately labelled Opus5.5/Astra review evidence and negative
procedures remain preserved. This measured TRAIN improvement supports the
selected generalization intervention; it does not prove that overfitting was
the sole quality cause. No duplicate paid critique or old training was launched.
