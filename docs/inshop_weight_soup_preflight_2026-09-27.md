# In-Shop checkpoint-average source screen, 27 September 2026

Test one uniform, FP64-accumulated weight average of existing seed-179024,
179026 and 179027 `freeze_emb` 1,000-update checkpoints. Average the vision and
128-D head state tensors only; classifier proxies are unused at inference.
These checkpoints share the pretrained SigLIP2 Large/256 files, source feature
cache, fit/held product rows, PCA head initialization, frozen embeddings and
first twelve blocks, optimizer settings, training budget and packed scorer.
The 179024 trainer source differs from the later two because the latter added
an optional rank-loss arm; its baseline recorded zero recovered rank updates.
The probe must verify state geometry, identical frozen tensors, finite values,
and each receipt/checkpoint hash before evaluating. Any mismatch stops it.

Use only official In-Shop **TRAIN**: 13,283 fit-product images and 12,599
product-disjoint held images. Encode the held panel once at native 256 pixels,
pack to the existing 128 signed int8 coordinates plus fp16 inverse norm, and
score all 12,599 self-excluded symmetric queries with stable ordinal ties.
Also score the previously fixed 6,354-query/6,245-gallery roles. Do not fit a
transform on held rows or read official query/gallery labels. This is an
exploratory feasibility test because the TRAIN held panel has been repeatedly
inspected. The idea has no method-novelty claim.

Freeze this stop rule before export. Advance only if all hold:

1. Symmetric packed R@1 is at least **98.7269%** (best source seed 98.6269%
   plus 0.10 percentage points).
2. Product-cluster bootstrap 95% lower bound for the soup's per-query R@1
   minus the *per-query three-seed mean* is strictly positive. The source mean
   R@1 is **98.5660%**.
3. Symmetric mAP@R does not regress below the three-seed mean **0.839866**.
4. Fixed asymmetric packed R@1 does not regress below the archived seed-179026
   value **97.6235%**. This is a guard, not a claim of improvement over all
   three seeds on these roles.

A pass authorizes only an independent-seed and training-cost decision. Three
trained source models cost about three times one model's training; one-model
serving speed and 130-byte gallery storage remain hypotheses until checked.
The current official query/gallery result and its seed-specific misses cannot
select the method. Failure stops before new training, official TEST, or
production promotion.
