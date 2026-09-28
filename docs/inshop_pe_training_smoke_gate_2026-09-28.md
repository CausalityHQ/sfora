# PE versus Large native image-training mechanics gate

The source prototype PASS permits this separately frozen mechanics proposal.
This is the same PE source route, not a reopened Base/teacher/loss configuration.
[Qualified source evidence](inshop_pe_core_source_result_2026-09-28.md).
No full-fit cache, held score, official evaluation or optimizer has run for PE.

Use the already hash-qualified1024 official TRAIN-fit images/512 products from
`pe-core-source-v1/cpu-preflight-v2.json`. Initialize each own PCA128 head with
bias, proxies and detached1024-row bank from its stored original-FP32/native
FP16-autocast source matrix; all1024 rows belong to fit. This smoke PCA is fit
on all1024 rows, not the gallery-only source pilot PCA. It is a mechanics fixture,
not a full-fit training initializer or quality selection. Freeze both initializer
hashes before optimizer execution. Cache profile is FP16 autocast; training is
native BF16 autocast, explicitly record and bound that difference.

Matched seed179032,16 steps,64 images/step, coverage-first product-balanced
schedule using existing helper. Two images/product may repeat to fill four slots;
preserve native duplicate-refresh semantics. Use native unaugmented RGB PIL
pixels cached once per selected row in each resolution; same paths/labels and
schedule, separate native224/256 tensors. No stochastic augmentation in this
mechanics check, so it is not the later augmented trainer or a quality recipe.
Both models load original checkpoint values directly to FP32; PE strict CPU
state authority remains required. No half-to-float roundtrip.

Large: freeze `embeddings` and `encoder.layers[:12]`. PE: freeze `conv1`,
`class_embedding`, `positional_embedding`, `ln_pre`, `rope` parameters and
`transformer.resblocks[:6]`; leave last six blocks, `ln_post`, `attn_pool` and
`proj` trainable. Freeze includes pre-normalization so the early prefix does
not drift. Record complete named trainable/frozen inventories and counts;
half-depth is the matched rule, not equal parameter count or FLOPs.
This remains a source/system comparison, not an isolated learning-method effect.

Reuse native `compact_head_features` (unit source then raw affine128), ArcFace
margin0.3/scale64 plus8× detached bank SmoothAP, positives/self ordinals,
pre-update normalized-output refresh **after** optimizer, last duplicate wins.
AdamW vision1e-5, head/classifier1e-4, weight decay0.05, global clip1; constant
LR for the16-step mechanics test, no classifier freezing or new auxiliary loss.
All smoke batches have positives; no singleton-skipping branch is activated.
Keep strict production head types unchanged.

Before updates, first four PE fit images BF16-v-FP32 feature cosine must be≥0.999
and raw affine loss/gradient values finite. Both arms must show finite nonzero
head, classifier, pooling/readout and every last-half block gradient. Frozen
parameters must have no gradients; byte hashes must remain unchanged after16
steps. Trainable vision/head/classifier hashes must change. Check at first and
last step; no gradient-threshold/schedule/precision/freeze rescue on failure.

Both arms run serially within one native shared-lock DGX GPU process,180s whole
process/8GiB host, peak allocated CUDA<10GB including initialization. Record
per-step forward/backward/update timing after synchronization, raw losses,
preclip norms, images/s, initializer/pixel/schedule/code/source hashes, raw
counts and initial/final freeze integrity. First two steps are descriptive warmup;
compare medians of steps3–16, PE/control≤0.8 required. Timeout/OOM is operational
failure, not quality evidence. Any route, finite, integrity or cost failure stops
this configuration before full-fit acquisition or training; do not enlarge caps.

PASS licenses a separately frozen **full-fit** source acquisition and actual
paired image-training feasibility proposal with product/query R@1/mAP@R and
uncertainty stop rules. No learned quality, generalization, official result or
serving win is established by this16-step check. Later training must use complete
fit-only initialization/bank and matched augmentations, not this small fixture.

## Terminal dual review: corrections supersede original details

Group`0d378a7ed2f2484c`, Opus`868e830427b6438e` and
Astra`4d3ecb3b5ab4452c`, both support the mechanics gate conditionally.
Raw separately labelled results are in `pe-training-smoke-v1/dual-result.json`.
No optimizer or quality read has occurred. Mandatory corrections accepted:

- Native `Rope2D` is a plain object: freeze its nested `rope` module and audit
  its frequency parameter/state separately. Do not register it differently or
  alter native163-key checkpoint loading. Require identical shared RoPE object
  references in every block. A faithful CPU fixture and actual native PE check
  now pass; the original fixture wrongly represented this object as a module.
- Frozen integrity covers frozen roots' `state_dict()` values and the separately
  audited rotary state after initial geometry warmup. Gradient absence must
  match the exact frozen inventory. Record each trainable parameter's gradient
  at steps1/16 and reject zero/missing/nonfinite values; record/check updates
  separately for every tail block, post-normalization, pooling/projection,
  head and classifier. All losses/gradients every step and parameters, AdamW
  state and refreshed bank after **every** update must be finite.
- First four sorted scheduled/updated fit rows **in both arms** must have fresh BF16 versus
  FP32 feature cosine≥0.999 and cached FP16 versus fresh BF16 cosine≥0.999
  before updates. Report raw projected/readout differences; no exact bank parity.
- Coverage-first epoch1/seed179032/16x64 exercises256 products/512 distinct
  images,1024 presentations. All1024 source images initialize PCA/proxies/bank.
  Check actual counts and freeze schedule/updated image inventory before launch.
- Duplicate refresh's last-occurrence rule is proven by a CPU fixture with
  distinct pre-update values; unaugmented GPU duplicates cannot discriminate
  first versus last, and will not be claimed as that proof.
- Total synchronized guarded-step timing begins before pixel transfer and
  zero-grad, includes encoder/head/objective, backward, clip, AdamW, bank refresh,
  every-step finite checks and ending CUDA synchronization. Detailed first/last
  per-parameter diagnostics/hashing are outside this interval but inside the
  whole-process deadline. Use total-step median3–16 PE/control≤0.8; components
  are descriptive only. The ratio is a smoke training-cost guard, not serving.

Memory scope is corrected **before** measurement. Opus's16,940,154,368-byte
receipt is arm`freeze` with trainable embeddings, not this `freeze_emb` graph.
Its conclusion that the exact proposed arm needs16.9GB does not follow. However,
protocol-relevant true-freeze receipts independently measured12,196,663,296
bytes at17 updates and12,938,664,960 at100/1000 updates, so the original10GB
Large cap is unsupported. New exact allocated-CUDA ceilings: Large16,000,000,000
bytes (~24% above the12.94GB reference), PE10,000,000,000 bytes. These are bounds,
not predictions. Keep180s whole pair and8GiB host cap. Run fixed order Large then
PE, **only one model/optimizer/native pixel cache resident at a time**; release
all arm tensors/optimizer, collect garbage and empty CUDA allocator before PE,
reset peaks before each arm's initialization, retain per-arm peaks. Cache only
512 scheduled images' exact native FP32 pixels; no lossy uint8/half reconstruction.
Record host RSS after model/pixel setup. No failure-based cap or precision rescue.

BF16 parity failure closes that precision configuration, not the PE family.
No inference, generalization or dataset-quality claim follows the smoke.

CPU executable preflight v2 passed before any GPU optimizer: native model/source
loading, exact native512-image pixels, cached objective source/head/classifier
finite nonzero VJPs, inventories and distinct-valued duplicate refresh. Whole
process9.13s, peak host2,440,096KiB. Raw hashes/initializers are frozen in
`pe-training-smoke-v1/preflight-v2.json`; this is no encoder-gradient or quality result.

## Concrete executable review resolved before GPU

Group `f19910b5e91e4494` completed: Opus `1edcdb1a3a8a40bb`, Astra
`7a0078445d7f496d`. Raw separately labelled results archived. Required fixes:
startup now hashes every frozen file independently of lazy imports, retains
loaded-file checks, and enforces source-manifest SHA as well as initializer and
feature SHA. A fresh-process CPU check rejects changed code/source/features/init.
The original import-order failure was reproduced with CUDA hidden. Version3
CPU metadata/startup checks passed; final version4 also freezes the cleanup fixes.

Clear pinned Torch2.12.1 cuBLAS workspaces before GC/empty_cache, record live
allocation and retain the original <8MiB inter-arm cleanup guard. Opus's possible
workspace remainder was an inferred framework issue, not a measured GPU failure.
Keep CUDA phase ceilings/whole180s/8GiB unchanged. The shared lock is held inside
the service for its full Python lifetime, not around asynchronous submission.
Both matmul and convolution TF32 are disabled. Warmed `rope.freq` is included
in frozen integrity and saved separately without changing native registration.

Checkpoints on DGX are intentional replay evidence and count toward the whole
process cap: native vision/head/classifier/bank plus foreign rotary state. Host
RSS is cumulative process maximum; the PE figure can include the Large peak.
Guarded-step cost includes tensor finite-check synchronizations (more tensors in
Large); it is not a model-only training benchmark. Large key-projection bias has
a mathematical null gradient; nonzero rounding and AdamW decay there do not
prove useful learning. No such claim will be made. Raw total loss is sufficient
for this mechanics gate; loss decomposition can accompany actual quality training.

Final v4 preflight:8.89s,2,438,676KiB peak RSS, exit0; fresh-import and altered
code/source/feature/initializer rejection PASS. v2/v4 initializer bytes identical.
Exact sole GPU launcher saved as `pe-training-smoke-v1/launch-v4.sh`; v4 output.
