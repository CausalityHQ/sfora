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
