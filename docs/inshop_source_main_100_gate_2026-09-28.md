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
