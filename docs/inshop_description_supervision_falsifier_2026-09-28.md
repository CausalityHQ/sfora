# In-Shop description supervision: cheapest TRAIN falsifier

Packing and preservation of the SOP source failed their gates. Test one
distinct mechanism: identity-only training lacks explicit garment-detail
supervision; existing product descriptions could supply it through a
training-only image–text term. This is established multimodal supervision,
not a novelty or SOTA claim; it adds no serving operation if adopted.

Before an encoder run, require usable color plus first-description text
for at least **95%** of the 2,004 fit products, and at least **90% unique**
captions. Use only the pinned official-TRAIN 13,283 fit images, never held
or official outcomes. Remove identity shortcuts by rejecting item-ID text;
do not feed product IDs, care instructions or size tables to a text model.
Any failure KILLs this metadata route before training.

If metadata passes, the sole next gate is a <=2-minute frozen SigLIP2 text
alignment diagnostic, using cached pretrained image features. Choose 512
distinct fit products by fixed hash, one image per product by fixed hash,
and each image's nearest different fit product as its hard impostor.
True-description versus impostor-description cosine must win on **65%**
of queries, its product-bootstrap 95% lower bound must exceed **50%**,
and it must exceed a caption permutation within clothing category by
**10 percentage points** with a positive paired lower bound. Bind every
image/cache/tokenizer/model/caption/selection hash. A metadata or alignment
pass is only a falsifier screen; it does not prove learnability or justify
full-budget training. A passing alignment requires a fixed <=2-minute
gradient/cost smoke before any paired quality training gate.

## Ingestion correction before alignment

The first CPU command exited 1 on the reader's duplicate-record guard.
Read-only audit found all 2,004 fit products present and **16 repeated
records, all identical**, with zero conflicting captions/colors. The
[initial ingestion failure receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-description-data-kill-v1.json)
is preserved. Its preliminary KILL interpretation is superseded by this
verified reader bug: exact repetitions carry no contradictory supervision.
The reader now deduplicates identical semantic text and still rejects
conflicting text and identity tokens. The regression check failed before
the fix and passed afterward. Coverage/uniqueness and alignment thresholds
are unchanged; no image–text quality or GPU result has been read.

The corrected metadata gate passed: **2,003/2,004 products (99.9501%)**
usable, with **100% distinct** usable captions. The
[eligibility receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-description-eligibility-v2.json)
SHA-256 is `8c288436623814f79bf3aa38f05c0b1f3bfd2b1af2e32c6788b75835f0ab664a`.

The alignment implementation fixes the product hash prefix to
`inshop-description-pilot-v1\0`, chooses one query image by relative-path
SHA, and breaks nearest-impostor cosine ties by original fit row order.
The caption control randomly orders fit-product captions within each
`img/SEX/CATEGORY` group (RNG 179019), then shifts the order cyclically;
it has no fixed product assignments. Tokenizer files are pinned to the
same model revision; use max-length padding/truncation of **64 tokens**,
the native frozen text pooler, float-normalized cosine and FP16 text
weights. The command has a **120-second timeout including data checks**.
Missing selected captions, nonfinite features, wrong hashes or timeout
stop it. This supervised-metadata route uses additional supplied annotation
compared with the label-only baseline; no equal-data method advantage can
be claimed without matching that supervision in the control.

## Alignment terminal: GO to gradient/cost smoke only

The first staged command `3d29466974284a6cbdea069f32e54572` exited 1
at import before GPU computation because an exporter helper was absent.
The runner now reuses the existing archived helper directory; a CPU import
preflight passed. No selection, arithmetic or decision threshold changed.
The sole retry `a0756aed956d4344bd9b92971011ab8b` exited **0** on DGX
Spark GB10; both invocations are preserved in the
[journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-description-alignment-journal.log).

| Official In-Shop TRAIN fit products; 512 distinct query products | True caption | Category-matched shuffled caption |
| --- | ---: | ---: |
| Caption cosine beats selected hard-impostor caption | **357/512 (69.7266%)** | **259/512 (50.5859%)** |
| True win product-bootstrap 95% lower bound | **65.6250%** | not a promotion metric |
| Paired gain / lower 95% | **+19.1406 / +13.0859 points** | reference |
| Diagnostic wall / peak allocated CUDA | **6.499 s / 1,265,392,128 B** | shared execution |
| Holdout R@1 / mAP@R / serving p50,p95,p99,QPS | unmeasured | unmeasured |

All frozen alignment floors pass. The
[raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-description-alignment-v1.json)
SHA-256 is `b6f77d2eaacac7e170e2d688b03db9e1ff3455f1ebfaed5829e107856f554aab`.
Local replay verified all 512 margin signs, distinct query-product rows,
paired gain and threshold decision. It is a **fit-only semantic alignment
screen**, not retrieval R@1, held generalization or proven benefit from a
training objective. No checkpoint or production default is promoted.
Next: freeze a training-only description term and its matched gradient/cost
smoke, then run that bounded smoke before any paired held-quality gate.

## Review correction and cached gradient rejection gate

Opus 5.5 / GPT-6 Astra review `65be9817860e444a` rejected one-hot caption
CE before training. Full string uniqueness does not establish distinct
token sequences or appropriate negative prototypes. Use the smallest
correction: fixed normalized text prototypes `T`, source `z`, and soft
target `q_y = softmax(32 T_y T^T)`; auxiliary loss is
`0.1 CE(32 normalize(z) T^T, q_y)`. This is a fixed surrogate, not native
SigLIP2 sigmoid training. A category-matched caption permutation is a
mandatory sham arm; prototypes and target probabilities are detached.
Map fit class names explicitly to caption indices, with one missing sentinel.

Before encoder training, run a **<=120-second CPU** diagnostic on the
same 512 preselected, class-balanced fit queries, using the exact cached
pretrained source, fit-only PCA/head/proxy initialization and frozen text
cache. This narrows the reviewer’s all-fit suggestion for a cheap rejection
screen; a pass remains insufficient evidence of quality. Freeze:

- C1: true-caption mean rank and soft CE each beat the category sham,
  with paired product-bootstrap 95% lower bounds **above zero**.
- C2: median `norm(g_true-g_sham)/norm(g_true)` at least **0.30**.
- C3: median cosine of caption-specific gradient `(g_true-g_sham)` with
  initial ArcFace pooler gradient at least **0**, and negative cosine on
  no more than **60%** of rows. This omits bank gradients and augmentation;
  it is a cheap initial conflict screen, not a training guarantee.
- C4: caption residual participation rank `trace(C)^2/trace(C^2)` exceeds
  **twice** the number of supplied clothing-category groups.
- Cache labels, source/features/metadata/fit hashes, finite unit prototypes
  and every query identity must validate. Undefined/nonfinite statistics or
  elapsed time above 120 seconds KILL the configuration before training.

No criterion is retuned after reading it. Record prototype exact duplicates
and nearest-neighbor cosine; inspect token collision/truncation before any
training if this cached gate passes. If the gate passes, freeze a serial
17-update control/true/sham smoke with actual base+rank gradient shares,
full control replay, same-input collapse checks and <=1.10 cost gates.
The original one-hot design is closed; no additional temperature search.

## Cached gate terminal: KILL this configuration

The [cached-gradient receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-caption-cached-gradients-v1.json)
SHA-256 is `4138a432bcba853a63dbec7e5035069974b2e651f73c224dd5680d11caa809b9`.
It completed on CPU in **3.847 seconds**. The initializer PCA SHA matches
the archived seed-179024 control (`f387aae1...`); class-to-caption mapping
explicitly handles the one missing caption.

| Frozen cached criterion; official TRAIN fit only | Measured | Decision |
| --- | ---: | --- |
| C1: true-vs-sham soft CE gain / lower 95% | **1.9680 / 1.8281** loss units | pass |
| C1: true-caption ordinal rank gain / lower 95% | **430.86 / 389.96** positions | pass |
| C2: median caption-specific / true gradient norm | **1.3595** | pass |
| C3: median cosine with ArcFace / negative-row fraction | **0.01480 / 44.1406%** | pass |
| C4: residual participation rank, must exceed 2 × 22 groups | **42.8523 vs 44** | **fail** |
| Weighted true aux / ArcFace gradient norm, median | **5.3426%** | diagnostic only |

Independent sample-Gram replay reproduced the participation ratio to
1e-10, using a different matrix orientation from the implementation.
**KILL the frozen global-caption configuration before encoder training**;
no threshold relaxation, 17-update smoke, 100-update quality run, TEST read
or p99 certification. C4 is a conservative resource-screening floor; this
near miss does **not** prove collapse, absent semantic information, or
inability of other text supervision to improve retrieval.

The qualified fit-only teacher cache exported once in **6.073 seconds**,
with **1,265,392,128 B** allocated CUDA. Its SHA-256 is
`d20a12cdbe1428f52ec192d182046f5f9882a2ba2e42ff82ec3aeac89a40c975`.
The [token census](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-caption-token-census-v1.json)
found **1,005/2,003** captions exceed 64 tokens, with **zero exact token
or prototype collisions**. The maximum token ID **245,848** is valid for
the pinned text embedding table **256,000 × 1,024**; parent-config generation
warnings do not describe that table. Neither cache export nor caption
classification diagnostic is an image-to-top-k or training measurement.

The opt-in library helper has runnable checks for attainable soft targets,
detached teacher gradients, missing captions and FP32 behavior under
autocast. It is retained for reproducibility; the default trainer, encoder,
128-D packed retrieval and shipped direct processor are unchanged. No
model or method is promoted by this failed gate.
