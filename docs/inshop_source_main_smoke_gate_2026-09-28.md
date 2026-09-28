# Frozen source MAIN encoder smoke

Opus/Astra group0056bc341ded4f98 completed: both conditional GO for mechanics
only. Full reviews are archived in evidence/compact_metric/sop-siglip2-substrate-v1/
inshop-source-main-design-review-v1.json. Known imprinted classifier plus
decoupled projection, not a claim of algorithmic novelty or SOTA.

One serial native128/source1024 MAIN pair, seed179024,17x64 actual images from
the pinned original TRAIN-fit13283/2004 schedule. Never open held/official
images. Preserve native128 initialization/model-loading RNG/loader workers4,
Large1024 embeddings and first12 blocks frozen, BF16 native straight-through
ArcFace.3/64, rank coefficient8, native detached bank/global clipping1,
AdamW vision1e-5/head1e-4/classifier1e-4, decay.05. Source proxy replaces P128
only in classifier optimizer slot; serving head and bank remain128.

Cached gate SHA f5bec2865b8aeb86f66003ada0dc5902616d98e043a3d3b8e1716ead95c41203.
Original17 native receipt SHA
83c977aa6d57fc5c0aff10824b8ffa0ac38c5d47cfb43c4e88679848e301063b.
No post-fit calibration, teacher or new source. Mean/proxy initialization hashes
persist before training images; compare exact initial head/bank/vision between
arms. Fresh native all17 pixel hashes must match original and losses atol1e-5;
failure stops before proposal. Pair must match all17 pixel hashes.

Fixed first-batch eval-mode initial/terminal probes separately measure actual
MAIN and weighted-rank encoder gradients, cosine, saturation and QR outside-span
norm. Both arms pay for probes; initial probe charged to first update, terminal
probe to diagnostic-inclusive training wall/peak. MAIN must reach source/proxy
but not head or fixed mean. Initial MAIN/rank encoder norm >=.1; terminal MAIN
encoder norm >=.25 initial. Per-step preclip group/global norms reveal clipping;
record batch-source mean distance from fixed fit mean. Do not tune coefficients.

All17 finite successful updates,15 rank-active; source head gradients genuinely
None at steps2/7 and its weights/bias/Adam moments/counters exactly unchanged.
Source head counters15; classifier and encoder update every step. Frozen encoder
parameters exact; trainable parameters changed. Compact variance/effective rank
on identical first fit pixels and eval mode must retain >=.9 initial (choose
Opus' stricter guard rather than Astra's suggested.5, before measurement).

Save terminal native vision/head, method identity, mean/proxy hashes, training
cost, all inputs/losses, gradient histories and probes before parity. Compare
saved parameter tensors exactly to terminal in-memory tensors. Compare unchanged
public32 output to matched checkpoint-derived in-process32 output, exact packed
bytes/f16 norms; no encoder64/reference32 mismatch. Source terminal pressure or
geometry failure stops before reload and preserves checkpoint evidence.

Whole pair <=120s including setup/checkpoints/public32, external watchdog.
Diagnostic-inclusive training wall and median update excluding first <=1.10
native; peak allocation <=native+1GiB. Child timeout/failure emits durable KILL
receipt retaining any completed arm; no deadline extension/rerun. Failed gates
KILL this fixed configuration. PASS licenses separately frozen100-update packed
TRAIN gate only; no held quality, full training, official reads, latency or
release/default-model change here. No operator approval handoff.

CPU implementation verification: one new regression checks native loss/gradient
equality, no mean/head MAIN gradient and real active-to-inactive AdamW state
preservation; focused source-centroid/fold regressions6/6 pass. No package source
changes; existing clean wheel authority remains valid.
