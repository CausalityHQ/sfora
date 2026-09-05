# Fixed SigLIP depth recovery experiment

Status: speed preflight passes all six ratios; recovery quality unmeasured.
See docs/siglip_depth_speed_result_2026-09-05.md for exact receipts and the
input-access deviation. The fixed recovery specification remains unchanged.

Implementation observation during the original21849eb5 speed run: the reused
control loader eagerly decodes the full cached Cars source before selecting128
optimization images. Only that selected prefix enters either model; no quality
evaluation/optimization occurs. The original run therefore measures matched
optimization-input speed, but does NOT demonstrate strict optimization-only
pixel-access capability. Preserve that deviation rather than claiming the
stronger input-isolation requirement passed. A separate metadata-first loader
must validate full label/ID metadata before fetching selected image records for
recovery. Require its128ID/RGB output to match the timing input digest before
reuse of those timing measurements; do not overwrite or rename the original run.

## Purpose and limits

Astra consultation70951a0c33fb4fdc recommends a fixed compression baseline after
the authenticated readout audit rejected further Qwen investment. Preserve the
SigLIP representation's quality/storage advantage while reducing encoder work.
This is not an invented loss or a novel SOTA claim. Existing layer compression
and relational distillation are prior art. The operator requested autonomous
iteration; no routine approval pause is required. Work is Sfora-only.

## Exact treatment

Authenticate seed17 epoch60 checkpoint SHA256
cb9c768fbb254bb164432ac92f756ca588cb1f33ac3eea86d4057d075ce2ef6e,
5,146,653,305 bytes, against the existing immutable control receipt. The full
teacher is unchanged. Deep-copy it, physically retaining one-indexed blocks
1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27. Preserve patch/position
embeddings, final normalization, attention pooler and1152-to512 projection.
Update the student's layer-count configuration to18; serialize only18 layers.
Teacher and student must not share tensor storage. Do not mutate the teacher.
The actual pinned Transformers5.12.1 topology is
model.tower.vision_model.encoder.layers (not a second vision_model wrapper).

Inference remains384px, eager attention, BF16 tower autocast with FP32 state
and FP32 projection/normalization. The full image patch grid is unchanged.
No mixed teacher/student gallery: independently embed the complete student
gallery if the experiment reaches evaluation.

## Fixed recovery objective

Both arms start at the identical pruned seed17 checkpoint. A uses the existing
Proxy Anchor alpha32/delta0.1 loss with49 trainable normalized proxies. B adds
relational cross-entropy: qT=softmax(offdiag(T T^T)/0.1),
qS=softmax(offdiag(Z Z^T)/0.1), Lrel=mean_i sum_j(-stopgrad(qT)*log(qS)).
Coefficient1, no temperature-squared scaling. Both directions through Z Z^T
are differentiated. Teacher is frozen/eval and sees the identical materialized
augmentation. No microbatch-local relational loss.

Use full logical-batch descriptor cotangents: collect Z without graphs, make
Z and raw proxies detached leaves, differentiate PA(+CE), replay student
microbatches with descriptor cotangents, assign proxy gradients once. Reject
active dropout/training BatchNorm, nonfinite/zero-norm inputs, uncleared grads,
and replay descriptor disagreement above2e-5. Test direct versus replay loss,
all parameter gradients and one AdamW update before any scientific training.

198 final-only updates, logical batch120=30classes*4images, unchanged training
augmentation384RRC scale0.16..1 and horizontal flip0.5. Fresh paired sampler
and crops; original optimization classes0..48 only. Base LR tower/pool1e-5,
projection1e-4, proxies1e-2. AdamW betas0.9/0.999,eps1e-8,foreach=False,
WD1e-4 with existing bias/norm/proxy exclusions; clip globalnorm10.
LR multiplier k/10 for k1..10; 0.1+0.45*(1+cos(pi*(k-10)/188)) for11..198.
All surviving student parameters train; FP32 parameters/moments, BF16 tower.

## Preflight first

Separate inference-only runner authenticates the checkpoint and ordered full
Cars training manifest, then takes the first128 optimization examples sorted
by example_id. No evaluation images/labels or quality scoring in speed preflight.
Bind original RGB pixel hashes and manifest IDs, use resident original RGB
images for preprocessing-inclusive timing, and preprocessed CUDA pixels for
encoder-only timing. Both scopes use identical batch8 positions, synchronous
boundaries,10 warmups and100 samples/window. Three paired windows alternate
full/student order within every round (reverse on odd rounds/window). Keep both
models resident to eliminate model-load effects.
Do not alter attention/backend/TF32 or precision between arms. Report all raw
nanoseconds and nearest-rank p95 (sorted index94).

Each of the six ratios (three windows times two scopes) must be at most0.75.
Fail speed before recovery training. A topology/preprocessing/numerical/authority
failure is invalid execution, not a quality rejection. The preflight stores
claim_eligible=false, no quality fields, source/checkpoint/model/data authorities,
parameter/layer counts, environment and scope names. Existing audited quality
is not a result of this runner.

After speed passes, ten disposable steps per arm establish finite movement,
replay correctness and wall/memory feasibility; reload initialization after.
Six accelerator-hour total cap includes preflight, smoke, paired recovery and
evaluation. Project remaining time from measured steps with25% headroom and
30min evaluation allowance. Keep inherited RSS110GiB/CUDA96GiB, PSI0.79immediate
or0.50three5s samples, swapgrowth256MiB and300s progress stops. No restart merely
because polling times out; preserve the original terminal. No relaxed caps.

Prospective smoke clarification following Claude0adc6cb299ce4be7, before any
recovery GPU result: add the measured smoke non-step overhead again for future
reload/setup and300seconds for the two terminal checkpoint writes. Neither is
silently charged to the30min evaluation allowance. Max step time includes crop
materialization and input hashing. Microbatch120 intentionally matches the
authenticated incumbent; descriptor replay's extra forward is a deliberate
verification cost, not claimed as a memory saving at this factorization. A
failure does not trigger an automatic microbatch ladder or tolerance relaxation.
"Fresh sampler" means reset state, original seed17/epoch0..5 batch composition;
new crops use the distinct siglip-depth-recovery-crops-v1 seed domain. All3963
optimization images are decoded for the smoke, though only sampled images enter
updates. This is within the optimization-only boundary, unlike reading other
class bands. AdamW decay is coupled to its scheduled LR, as in the incumbent.

## Advancement, not publication

Seal both terminal checkpoints before one reused49..81 evaluation, tagged
exploratory-reuse/claim_eligible=false. Seed17 teacher2596/2746 hits;
student must have at least2591 and MAP loss at most0.002 (fraction). All speed
ratios pass, CPU one-worker128query exact-search p95 ratio<=1.05, and512FP32
descriptor storage is exactly5,623,808bytes for2746images.
Prefer A if it passes; otherwise advance B only if it passes. Neither pass:
close this exact recipe, no depth/epoch/loss-weight sweeps on exposed results.
Any advance requires unchanged seed29/43 pairs before stronger exploratory
conclusions. Independent evaluation and matched external baseline evidence
remain necessary for the overarching publication/SOTA goal.
## Independent review disposition

Claude6e07775e993a4a52 completed read-only review. Accepted: full descriptor
replay (proxy gradient exactly once), sibling configuration/runner, full-state
load before physical surgery, frozen external teacher, both neighbour gradients,
encoder-only timing and within-round alternating measurements. All are explicit
above. Its extra vision_model wrapper path is not valid in installed5.12.1;
real tiny SiglipVisionModel surgery/execution/serialization tests verify the
actual path. Its non-block overhead threshold is25%, not12%, from
f+(2/3)*(1-f)<=0.75. Its six-hour feasibility estimate misread historical cost:
the authenticated seed17 receipt reports73,083seconds for one60epoch seed,
not three. No claim of comfortable feasibility is accepted before measured
disposable steps. Descriptor tolerance is fixed before results and is not
retuned after observed disagreement. This review supports an engineering test,
not a novel loss or assurance of recovery quality.
