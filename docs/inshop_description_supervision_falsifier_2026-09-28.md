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
