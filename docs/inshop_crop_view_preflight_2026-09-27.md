# In-Shop TRAIN view-coverage falsifier, 27 September 2026

The selected true-freeze seed-179026 checkpoint missed 151 of 6,354 fixed
TRAIN queries against a 6,245-image held-product gallery. Detail, full and
flat query framings account for more misses than their query share, while the
training augmentation crops only 80–100% of image area. The proposed next
intervention is a wider training crop; no serving change is proposed.

Before training, reconstruct the existing held split and fixed query/gallery
roles from the SHA-256-pinned official In-Shop TRAIN partition. Verify the
prior miss-receipt SHA, all 151 miss ordinals, the 6,354/6,245 role sizes and
their frozen digests. Among queries framed `additional`, `full` or `flat`,
compare the miss rate when the gallery has another image of that product with
the **same framing** against the rate when it does not. This is a metadata-only
diagnostic on previously observed TRAIN outcomes, not a new quality estimate.

Advance to a paired 100-update augmentation screen only if each group has at
least 100 queries, both groups have at least one miss, and the no-twin miss
rate is at least **1.5×** the with-twin miss rate. Otherwise stop this crop
lane: the proposed framing mechanism lacks its predicted association. A pass
would only warrant a frozen seed-179026 control/treatment run changing
`RandomResizedCrop` area scale from `(0.8, 1.0)` to `(0.25, 1.0)` while keeping
architecture, fit/held products, seed, 100 updates, batch schedule, loss,
checkpoint selection and exact 128-D packed scorer fixed. The first input
hashes must differ by design; training wall may rise no more than 2%, peak
CUDA no more than 0.5%, and symmetric held mAP@R may fall no more than 0.002.
Predeclare the query-framing R@1 gate separately before any GPU training.

## Metadata-only result

The frozen script ran on DGX Spark CPU only. Its [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-crop-view-twin-v1/receipt.json)
has SHA-256 `ad733aaf85c4f4d4863372a52d0d212223317e0d50ced947d90fc21fec860058`.
The existing 151 misses, all six framing counts and both role digests replayed
exactly. Among additional/full/flat queries, **19/809 = 2.35%** missed with a
same-framing positive in the gallery versus **59/1,414 = 4.17%** without one,
a **1.777×** ratio. Both groups exceed 100 queries and have misses, so the
predeclared 1.5× association gate passes. This is observational and cannot
establish that crop augmentation causes the difference.

## Frozen paired 100-update gate

Run seed 179026 control `(0.8, 1.0)` and treatment `(0.25, 1.0)` serially on
the same DGX. Both use the same `freeze_emb` architecture, preflight, fit/held
products, pretrained source/PCA, 64-image schedule, BF16 precision, ArcFace
and bank loss, optimizer, 100 updates, 128-D int8/fp16 packed scorer and
held-export code. First run 17-update smokes: require finite loss and gradient,
no skipped step, equal schedule and initialization hashes, differing input
batch hashes, treatment/control wall ≤1.10 and CUDA peak ≤1.05. Stop on failure.

For the full 100-update pair, score the same fixed 6,354-query/6,245-gallery
TRAIN roles. Require all four quality rules: additional/full/flat R@1 delta
≥**+0.45 percentage points** with product-cluster bootstrap 95% lower bound
strictly above zero; front/side/back R@1 delta ≥**−0.10 pp**; whole-role R@1
delta ≥**−0.15 pp**; and symmetric 12,599-image held mAP@R delta ≥**−0.002**.
Also require treatment/control training wall including bank init ≤**1.02**
and peak allocated CUDA ≤**1.005**. A pass authorizes separate full-budget,
independent-seed TRAIN confirmation. A failure stops the crop treatment before
official query/gallery or production promotion. These thresholds are fixed
before either arm trains; the official query/gallery split is closed to this
selection.

## Terminal paired TRAIN result

The first control smoke failed at import before any optimizer update because a
transitive repository script was absent from the isolated remote staging. Its
[journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-crop-view-twin-v1/smoke/failed-import.journal.log)
has SHA-256 `3ad740f21406068e55e4df40ce76d8d024e621b2056353290dd5266bea4ebc0e`.
After staging the file and confirming the import, both 17-update services
exited 0. Their
[control](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-crop-view-twin-v1/smoke/control.json)
and [treatment](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-crop-view-twin-v1/smoke/treatment.json)
receipt SHA-256 values are `861115637d56907383b566946c037438bf4745128018512d95240b3ad38ab10d`
and `95707f5590abf2ca6cff2ea73a3e9be633490dd4c19cb67d2ecb62fe616702c3`.
The source, schedule, split, feature cache and PCA head hashes matched;
the first ten input hashes differed as designed. Losses and gradients were
finite, training wall ratio was **1.0332**, and peak CUDA ratio **1.0000**,
passing the frozen smoke cost gates.

Both serial 100-update DGX services and the native packed score service exited
0. The [control receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-crop-view-twin-v1/full/control.json),
[treatment receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-crop-view-twin-v1/full/treatment.json)
and [paired native score](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-crop-view-twin-v1/full/paired-score.json)
have SHA-256 `8807d985bfbac8dab26a2f1a7dc482b19ad9a3d3771b0bf0bda3650da8840a7c`,
`55adc855411557a5492db151842d6bc9d21ae296d2eab3f54bb359f5e4f9c9e0`
and `cc2115323b8a3e6769994ca0cd94fd25e4dc6cb97da8cbf96a44ec8e92c42403`.
Their source-bound journals are adjacent to the receipts. The same checkpoint
architecture, data, schedule, budget and scorer were used for both arms.

| TRAIN held metric, control → crop treatment | Measured | Frozen decision |
| --- | ---: | --- |
| Additional/full/flat fixed-role packed R@1 delta, 2,223 queries | **−0.1799 pp**, product-bootstrap 95% lower **−0.7756 pp** | Fail: below +0.45 pp and lower≤0 |
| Front/side/back fixed-role packed R@1 delta | **+0.1452 pp** | Pass: above −0.10 pp |
| All 6,354 fixed-role packed R@1 delta | **+0.0315 pp** | Pass: above −0.15 pp |
| Symmetric 12,599-query packed mAP@R | **0.775415 → 0.770336** (−0.005079) | Fail: below −0.002 floor |
| Training wall including bank initialization, treatment/control | **0.9993×** | Pass: ≤1.02× |
| Peak allocated CUDA, treatment/control | **1.0000×** | Pass: ≤1.005× |

The association between framing and misses was real on these TRAIN roles, but
widening the random crop did not improve the targeted ranking after 100
updates. Stop this arm before new seeds, full-budget training or official
query/gallery evaluation. The experimental crop option is removed from the
trainer; the source at the frozen experiment commit and raw receipts remain
available for audit. This is a negative result, not a revised quality or SOTA
claim.
