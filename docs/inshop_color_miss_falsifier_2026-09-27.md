# In-Shop TRAIN color-miss falsifier

Before any descriptor or scorer change, test whether a fixed, cheap color
feature distinguishes the best same-product image from the top packed impostor
on the archived seed-179026 true-freeze misses. The 6,354-query/6,245-gallery
roles are product-disjoint from the 13,283 fit images and use only official
TRAIN standard retrieval pixels. The archived packed-score receipt has exactly
151 misses and supplies their query, best positive, and best impostor ordinals.

The frozen feature is an 8-bin-per-RGB-channel histogram from the central 60%
of each image, resized to 32×32. Distance is symmetric chi-square. Verify the
partition, role, receipt, and every sampled image SHA-256 before the read.
Advance only if the positive is closer than the impostor on at least **60%**
of the 151 misses **and** the median impostor-minus-positive distance exceeds
**0.02**. Otherwise stop the color lane before any model, gallery format,
scorer, or training change. A pass permits a separately frozen matched
TRAIN-only quality and image-to-top-k cost gate; it is not a quality gain,
algorithmic novelty, or official query/gallery claim.
