# Source MAIN100: frozen exploratory packed-quality gate

Qualification552859f6fad15f093ec1d540391b594fd9ff7ea0458fe942c5523d3ebc3a4c9b
licenses ONE fresh serial native/source100x64 pair,seed179024. Same source MAIN
smoke method: Large1024/native trainable128 head/bank/fitmean/learnable1024 proxies,
freeze12+emb/BF16/AdamW1e-5 vision,1e-4 head/classifier,decay.05/nativeST ArcFace
.3/64/rank8/globalclip1. No head classification in source arm, no teacher,
auxiliary, compactor, calibration, retuning or checkpoint selection.

Use existing `run_inshop_wide_head_100.py --source-main`; old closed256 method
retains its separate qualification. First17 actual pixel hashes/losses must
reproduce qualified17 both arms, all100 paired actual inputs and source hashes.
Official In-Shop TRAIN fit13283/2004, observed held12599/1993, fixed6354 query/
6245 gallery identities and existing hashes/scorer. Exploratory, not untouched
or SOTA. Export served128 via in-process FP16 batch32 only; mandatory exact
public32 codes/norms and saved-weight/terminal-model equality before held export.

GO requires source-native mAP@R point>=.01 AND paired product-bootstrap95%
lower>0; source-native packed R@1 point>=0, report product95% interval. Native
R@1>=.922 or stop before source. Every finite/schedule/source/checkpoint/parity
guard applies. Preserve MAIN initial/rank norm>=.1, terminal MAIN>=.25 initial,
same fit-pixel eval compact variance/rank>=.9 initial; source head genuinely
inactive on singleton steps with exact AdamW preservation. Log branch norms,
clipping,saturation,centering distance and raw100 training evidence before export.

Diagnostic-inclusive train wall/median update<=1.10 native; peak<=native+1GiB;
whole source arm<=1.15 native (no calibration pass). Child cap280s each,
parent600s/external610s. No extension/rerun; completed-arm evidence retained and
failed-arm quality=null. Any negative closes this fixed configuration. PASS
permits separately frozen paired-seed TRAIN generalization only. No official,
full training,10k serving gate or default-model change yet. Completed Opus/Astra
review explicitly allowed100 after positive17; no duplicate review. Report
actual quality/uncertainty/training/export costs; serving latency/QPS absent,
never estimated from training/export. Existing package authority unchanged.

## Terminal KILL — configuration closed

Unit `sfora-inshop-source-main-100-v1`, invocation
`f3bd91edf38246fd8016eb9a31d98230`, execution exit0, method gate **KILL**.
Whole parent447.574912160s; no active Sfora GPU unit/process afterward.
ReceiptSHA `a1ee787993d3cc6c2c674361410d1a0ee62728dd47600d453592bc3d7e745b6a`.
Artifacts retained in the matching remote run directory and local
`evidence/compact_metric/sop-siglip2-substrate-v1/inshop-source-main-100-v1/`.

Both100 stable updates; paired all100 actual pixels/sources and qualified17
prefix verified, checkpoint/public32 parity exact. TRAIN holdout6354q/6245g,
seed179024, observed evaluation, all numbers below measured and verified:

| Arm | Packed R@1 | mAP@R | Training s | Images/s | Peak CUDA bytes | Held export s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native128 MAIN | 95.247088% | 77.985504% | 89.011585 | 71.900753 | 20,227,696,640 | 107.632992 |
| Source1024 MAIN, rank-only128 | 95.325779% | 78.167812% | 89.398191 | 71.589815 | 20,264,168,960 | 106.712671 |

Diagnostic-inclusive train wall; 6400 images per arm. Whole arms217.492236/
217.260466s, median update.8462432/.8506472s. Full image-to-top-k p50/p95/p99
and QPS **not measured** here; neither export nor training is serving latency.

R@1 delta+0.078691 percentage points, product-bootstrap95%
[-0.143771,+0.312022]pp,30 rescues/25 regressions,6052/6057 hits.
mAP@R delta+0.182308pp,95%[-0.015494,+0.382526]pp. Both intervals include0;
mAP point also misses frozen+1pp floor. Resource/geometry/pressure/exactness
guards all pass. Local independent metadata role-hash validation,6354 point
record means/hits and all5000 product-bootstrap draws reproduce exactly;
`terminal_verification.json` records that check.

Actual source MAIN encoder norm93.40685->38.20440, outside-span total norm
fraction.306576->.191465; serving compact variance.851326->.909563 and rank
24.23032->23.58845. There is functioning extra encoder supervision without
collapse; it does not establish a useful deployed-quality improvement at the
frozen budget. These diagnostics cannot identify a unique generalization cause.

Product decision: retain native128 MAIN training/defaults; do not promote
source MAIN, extend training, add seeds/official reads, retune proxies/loss or
benchmark its serving. Its optional guarded code remains reproducible evidence,
not a release recommendation. The reusable gradient/inactive-optimizer and
terminal checkpoint checks are verified. Joint quality+speed goal is unmet;
next work must address representation/generalization from error evidence,
not reopen this fixed loss topology. No operator decision required.
