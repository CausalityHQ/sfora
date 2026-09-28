# Direct256 convergence confirmation: KILL

The original, already-launched job completed successfully; the frozen quality
gate failed. Do not launch further seeds or native/public256 serving work for
this fixed recipe. Production remains128. The full joint quality-and-speed
goal remains active and unmet.

Source `ee38e11d`, one fresh paired seed179024,1000updates×64 per arm,
native128 followed by retained256. Same SigLIP2 Large/256 backbone, original
TRAIN fit13283images/2004products, outer held12599images/1993products,
fixed6354query/6245gallery. This previously observed TRAIN panel is exploratory;
it is neither an official In-Shop result nor independent SOTA confirmation.
Same initialization's leading128 PCA rows, training inputs, schedule, objective,
optimizer, freeze12+embeddings and loaded source files. BF16 training with
FP32 parameters; nativeFP16 vision/FP32 head, TF32off, batch32 exports.

| Verified measurement | Native128 | Direct256 | Frozen decision |
|---|---:|---:|---|
| Packed asymmetric Recall@1 (%) | 97.655020 | 97.891092 | KILL: +0.236072pp < +0.5pp; lower bound0 |
| Packed asymmetric mAP@R (0–1) | 0.84916939 | 0.85719863 | KILL: +0.00802923 < +0.01 |
| Training wall excluding bank init (s) | 848.335347 | 851.174989 | Cost guard passes |
| Bank initialization (s) | 0.876402 | 0.784261 | Recorded for both |
| Training including bank init (s) | 849.211749 | 851.959251 | Pass: ≤1.10×control |
| Training presentations/s, excluding bank init | 75.441864 | 75.190179 | Training throughput only |
| Median step (s) | 0.843895 | 0.846361 | Pass: ≤1.10×control |
| Peak training CUDA allocated (bytes) | 12,988,997,120 | 13,002,782,208 | Pass: ≤control+1GiB |
| Whole child wall (s) | 952.987416 | 955.819061 | Both within1100s |
| Packed bytes/row | 130 | 258 | Format cost, no deployed256 qualification |
| Full image-to-top-k p50/p95/p99/QPS | Unmeasured | Unmeasured | No speed claim |

Paired product-bootstrap95% intervals,5000draws seed179019:
R@1 delta **+0.236072pp [0,+0.477403]pp**;
mAP@R delta **+0.00802923 [+0.00525405,+0.01090899]**.
There are40 top1 rescues and25 regressions, net15/6354 queries.
These intervals resample1993 products with all their queries; they do not
capture training-seed variation. Both point floors were frozen before launch;
a positive mAP interval does not override its missed effect-size floor.

The100-update width advantage was+0.802644pp R@1/+0.01683096 mAP@R;
at1000updates it is smaller. This supports closing this fixed recipe under
its predeclared rule. It does not establish that every wider encoder or future
representation change must fail. No retuning, extra steps, official evaluation,
deadline extension or relaxed threshold follows this result.

## Independent verification and preserved artifacts

Original DGX user service `sfora-inshop-direct-width-quality1000-v1`, invocation
`61592785a58a4985ae87964ea9b95b31`, completed in1921.294321s within the
declared2200s total budget. Journal records both terminal arms and the original
KILL decision. The transient unit was already unloaded when recovered:
its current empty/default `systemctl show` fields cannot independently recover
launch-time resource settings. No active Sfora job/GPU process remains; the
shared GPU lock is free. No training or checkpoint forward was repeated.

Original receipt SHA256
`8a25d14a58e9914d93a34809879edca71925ec09d6d834a4150d702bff9213cc`.
CPU NumPy independently scored every6354 query in each arm,3.662562s total:
every top1 outcome matched; max AP error1.1921e-7/9.9341e-8; actual saved
embeddings repacked to exactly the saved codes and inverse norms. This is
verification time, not a serving benchmark. The independent scorer uses
integer-exact float32 dots, declared float32 norm multiplication order and
stable gallery ordinals, preserving packed-score/tie semantics.

Separate metadata replay verified the partition/query/gallery roles, every
frozen criterion, all20 source files, all1000 finite loss/positive group-gradient
histories, paired pixels/schedule/PCA rows, durable pre-export and child receipt
agreement, checkpoint/fixture hashes, both geometry guards and both CI endpoints.
Original receipts attest frozen tensors/trainable changes and exact private
reload parity. This replay checks those attestations and hashes; it is not an
additional checkpoint tensor-state audit or public256/native-kernel parity.

Raw receipts, durable guards, logs, journal and runnable independent verifiers:
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/inshop-direct-width-quality1000-v1/`.
Float/packed arrays are retained locally at
`/tmp/sfora-inshop-direct-width-quality1000-v1/{128,256}` and on DGX with
checkpoints/fixtures under
`/home/riomus/runs/sfora-inshop-direct-width-quality1000-v1/result`.
Native checkpoint SHA
`e2904fd943ebca6fa12ee503d3f494972f3bc0b6de743cc505db4a4ddee2dad1`;
candidate checkpoint SHA
`015f0c8cd51519b5105931f9839f5d0106daef35ba9bdf1673b4c747a644b04c`.

## Next gate

Close direct256 at this compute/data/recipe; retain the substantive trainer,
dimension-aware scorer and durable audit work already shipped. The earlier
Opus5.5/GPT-6 Astra review `c112576b70054d1a` established quality-first
sequencing; this terminal result applies its frozen gate without a duplicate
consultation. It does not authorize serving-first work retrospectively.

A next arm needs a causally distinct TRAIN-only mechanism and cheapest frozen
falsifier, informed by the existing closed-arm ledger. None is selected by this
result. Preserve the existing model-access prerequisite; do not reopen failed
scaling/hardness gates, extrapolate this panel to official quality, or repeat a
width/loss sweep. SOP/In-Shop independent official qualification, CUB/Cars,
paired seeds and full serving speed remain outstanding product requirements.
