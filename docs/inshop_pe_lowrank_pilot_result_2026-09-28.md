# Fixed rank32 PE pilot: terminal quality KILL

The reviewed procedure completed native CPU qualification, its sole17-update
mechanics attempt, and one fresh100-update TRAIN pilot. The pilot passed
execution, cost and updated-checkpoint parity but failed every quality floor.
Close this exact procedure without rank/LR/prefix/precision/budget/epoch rescue.
The complete product goal remains active.

| Dataset / split | Matched controls R@1 / mAP@R (%) | Candidate R@1 / mAP@R (%) | Training cost | Public latency | Remaining gap / next decisive test |
|---|---|---|---|---|---|
| DeepFashion In-Shop TRAIN-fit13283images/2004products; TRAIN-held6354query/6245gallery/1993products;100updates B64 same RGB exposure and packed reference scorer | Historical Large **95.6720176 /78.6237120**; historical dense PE **95.0739692 /76.3915922** | Strict-loaded rank32 PE **93.9880390 /73.3082988**, independently CPU-replayed | median3–100 **0.541347755s** <=0.71769696s;117.2032 guarded images/s;171.92s whole process;3,522,020KiB RSS;peakCUDA4,910,489,088B | Unmeasured for updated checkpoint | vs Large −1.683979pp R1/−5.315413pp MAP; close rank32 procedure. Next: one dense native encoder-optimizer intervention, then matched TRAIN quality gate; no serving promotion |

Candidate minus historical dense PE, paired5000-resample95% intervals:

| Metric | Mean delta (percentage points) | Product95% | Query95% |
|---|---:|---|---|
| R@1 | −1.085930 | [−1.427247,−0.771321] | [−1.400692,−0.786906] |
| mAP@R | −3.083293 | [−3.406621,−2.763109] | [−3.357448,−2.822796] |

The frozen floors were R@1>=95.1720176%,mAP@R>=77.6237120%, and
strictly positive product lower bounds for both metrics versus dense PE.
All failed. Repeated TRAIN-held use makes this exploratory; intervals do not
cover method selection or seed variation. Archived controls are independently
verified historical measurements, not concurrent reruns. Exposure and
unchanged source/environment/initializers/scorer/roles were authenticated;
compute is measured, not assumed matched. No official/transfer read occurred.
Packed quality here uses the inherited int8/f16 reference scorer; it is not a
new public native-search or full-pipeline latency qualification.

Sole invocationf41aa1dae1784eb4816e61462f7ee6f5 exited0 (scientifically valid
negative),171.92s whole process within300s,8GiB host/no swap/<10GB allocated
CUDA. All100 RGB hashes matched archived dense PE, scales stayed128/no skipped
updates, all native/factor/state checks passed, frozen original/prefix/foreign
rotary stayed exact. The merged updated native model reproduced GPU raw
outputs after actual strict save/reload; all12599 in-process versus loaded
compact vectors had minimum cosine0.9999999999999998, and packed per-query
hits/AP matched within1e-6. The candidate scored was the strict-loaded model.

An independent CPU saved-checkpoint, bank, packed-score, product/query interval
and decision replay passed in8.00s,1,806,524KiB RSS, original exit0. DGX was idle
after both jobs. Checkpoint retained for evidence only at
`/home/riomus/runs/sfora-pe-lowrank-pilot-100-v1/pe.pt`, SHA
`5adca041bdd6c272dbc203b4acecb51f41957efa8cadf871ad3a0e481754ac7d`.
No serving or promotion is authorized by the negative result.

[Raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/lowrank-learning-research-v1/pilot-receipt.json),
[training records](evidence/compact_metric/sop-siglip2-substrate-v1/lowrank-learning-research-v1/pilot-training.json),
[cost](evidence/compact_metric/sop-siglip2-substrate-v1/lowrank-learning-research-v1/pilot-100-v1-time.txt),
[CPU replay](evidence/compact_metric/sop-siglip2-substrate-v1/lowrank-learning-research-v1/pilot-cpu-audit.json)
and [execution log](evidence/compact_metric/sop-siglip2-substrate-v1/lowrank-learning-research-v1/pilot-100-v1.log)
are retained. No additional research/critique or alternative method ran during the
100-update pilot and its narrow result replay. The completed evidence is a quality rejection, not a model gain.

## One next intervention selected

Change the responsible encoder-adaptation algorithm: native PyTorch Muon on
only the24 upper-six-block hidden matrices, with dense native weights and
unchanged AdamW on remaining native/head/proxy parameters. Source, prefix,
initializers, data, objectives, exposure and packing remain fixed. This keeps
all native matrix coordinates available while changing matrix-gradient
preconditioning. The hypothesis is that the adaptation rule, rather than
initial-source weakness or head compression, contributes to the measured
learning gap; optimization versus capacity is still unresolved. No predicted
quality or training speed is claimed.

PyTorch2.12.1+cu130 on DGX exposes `torch.optim.Muon`. Its
[version-matched primary documentation](https://docs.pytorch.org/docs/2.12/generated/torch.optim.Muon.html)
restricts Muon to hidden2D matrices and offers `match_rms_adamw` to reuse
AdamW learning-rate/decay values; this is documentation about the algorithm,
not PE/In-Shop evidence. Use native1e-5/decay.05, canonical momentum.95,
Nesterov,5Newton–Schulz steps and documented RMS adjustment, without a LR
search or calibration probe. Other groups retain their existing optimizer.
This is the one selected next hypothesis; a complete frozen execution gate
and narrow multi-optimizer/scaler integrity check must precede training.

The [closed cached-feature ProxyMuon result](unicom_proxy_muon_f0_result_2026-08-25.md)
is negative evidence: its proxy optimizer did not meet the optimization-speed
target. It remains closed, with no proxy/head Muon, LR retuning or precision
rescue. This new candidate acts on native image-encoder hidden matrices; that
specific path has no existing measured result. If a scope audit finds it was
already tested and killed, do not run it again. The next test must train a
useful checkpoint under bounded mechanics/100-update quality gates, not
another theorem-like probe. A survivor goes directly to updated-checkpoint
serving qualification and fresh confirmation. No next training job is launched.
