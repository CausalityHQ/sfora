# Paired native augmented TRAIN feasibility gate

Purpose: determine whether the qualified PE source/native architecture can
retain Large control quality with lower actual image-training cost. This is
a source/system replacement contrast (pretraining/width/depth/native resolution
vary), not an architecture-only or matched-backbone SOTA claim. Keep the full
production joint objective, unchanged loss and 128-D deployment geometry.

Use original FP32 Large/256 and strict PE-Core-B16/224, qualified FP16 autocast,
GradScaler initial128, TF32 disabled; freeze native embeddings/first12 Large
blocks or native PE prefix/first6 blocks including foreign rotary frequencies.
Use full audited TRAIN-fit matrices (13,283 images/2,004 products/12 singletons)
for each own image-PCA128 head, class proxies and normalized detached bank.
No held image or held score enters initialization or model selection.

Seed179032, generate existing coverage-first balanced 1,000-update schedule,
execute only its first100 batches of64. Match ordinals and augmented RGB views
exactly across arms: existing RandomResizedCrop256 scale(.8,1) + horizontal
flip, with a per-step CPU-only RNG seed and restored RNG state, followed by
native224/256 preprocessing. Rehash actual input files before each decode;
stream batches and hash every augmented RGB batch. This prefix does not cover
all fit rows; report actual distinct images/products/presentations/inactive
singleton batches. A future full1000 run must preserve the prefix exactly.

AdamW: native vision tail lr1e-5, head/proxies lr1e-4, decay.05, clip1.
ArcFace margin.3 scale64 +8 detached-member-bank rank loss. Entire rank loss
inactive on any batch containing a fit singleton, matching inherited control;
CE and bank refresh remain active. Bank writes detached normalized pre-update
raw projected vectors AFTER successful optimizer step, last duplicate view
wins. Unscale before clip/gradient checks; any skipped scaler step fails.
Finite parameters/Adam/bank every update, complete first/last gradient inventory
and all updated groups; exactly12 mathematically null Large key biases may be
zero, PE has no exception. Frozen state including warmed PE grid unchanged.

Before training, same first4 fit rows: native FP16/FP32 and cachedFP16/freshFP16
cosine floors.999, preserve values before assertion. Verify full source receipt,
manifest, file/executable hashes, external preflight SHA and init arrays; altered
manifest/code must reject before CUDA. CPU fixtures test matched augmentation,
RNG isolation, changed-file rejection, singleton policy and startup hash failure.

One serial shared-lock DGX service, Large then PE, whole process600s, host8GiB,
allocatedCUDA Large<16GB/PE<10GB, cleanup<8MiB between arms. Caps are budgets,
not projections. No retry, cap/dtype/floor/recipe rescue. Save final actual
checkpoint plus per-step loss/scale/grad/cost/RGB hashes and held vectors for
both arms. Hash checkpoint and vectors; failure logs/completed-arm evidence
persist. Save per-arm receipt before terminal quality/cost decision.

Evaluate native unaugmented final checkpoint on all12,599 TRAIN-held images,
then existing exact packed-score geometry on frozen6,354 query/6,245 gallery
roles. Assert inherited query/gallery ordinal hashes. R@1/mAP@R, paired5,000
product bootstrap95% intervals and paired query bootstrap95% intervals; these
are exploratory TRAIN-only, not official In-Shop query/gallery. Save raw per-query
hits/AP. No data augmentation at evaluation. No official/SOP/CUB/Cars/serving test.

Gross STOP: PE R@1<Large-.005 OR mAP@R<Large-.010 OR PE/Large median guarded
training-step time (steps3-100, includes file/decode/augment/preprocess and finite
checks)>.8. Also any integrity/finite/resource
failure stops. Passing is feasibility only; no superiority inference from100
updates or automatic1000/official evaluation. Longer budget/checkpoint/seed
qualification is a separately frozen next gate, not an adaptive loss search.
Independent saved checkpoint/frozen inventory/bank/scales/schedule audit and
held-score replay are required after run before claiming PASS.

Combined critique deaf616c05364878: Opus GO, Astra STOP pending concrete
launch safeguards, then GO once fixed. Fix executing trainer/module origin and
hash checks BEFORE CUDA, explicitly authenticate both original model files
before Large, reject all partial arm artifacts and exclusively reserve attempt.
Enforce whole-process wall/host/shared-lock through frozen systemd user launcher;
check CUDA peaks after calibration, every training step and held inference batch.
Record executing code and separate CPU input versus guarded update costs. Release
optimizer and last training tensors before held inference. Rerun preflight and
startup checks after changes; one fix pass, no duplicate critique. Optional
step-0 held forward is deferred:100-updates compares replacement systems and
does not isolate training gains or establish quality superiority. Serial arm
order, parameter-dependent finite-check overhead and point-estimate stop rules
remain limitations; report uncertainty and no threshold rescue.

Final CPU v2 invocation5bcb8834ad884c7f9090f3cd5dc5b603 exit0; initializers
are byte-identical to v1, same schedule and first RGB/native pixel hashes.
Trainer SHA0963cc443ef29e7d821275f4068f760dd6ed51b7ab69aa98bf0030ccf9676ecf.
External preflight SHA41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293.
Actual valid-root CLI startup passes; copying the trainer outside that root
while keeping valid root files rejects before CUDA. Fixture also rejects partial
checkpoints and second attempts without overwriting files. All three scripts
Ruff/format/syntax pass. Frozen launch-v2.sh enforces the shared lock,600s whole
process,8GiBhost and durable logs. These close the conditional GO review items.
Independent audit_inshop_pe_pair.py is ready for actual saved checkpoint/held
score/CI replay after the GPU run; it has not audited nonexistent results.
