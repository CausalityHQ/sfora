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
