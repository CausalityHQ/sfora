# In-Shop TRAIN impostor-count scaling gate

The selected official In-Shop bank checkpoint is 1.218 percentage points below
the dated UNICOM 96.7% Recall@1 reference. The current seed-179026 true-freeze
checkpoint has only been measured on product-disjoint official TRAIN held roles;
its official result is unknown. Before choosing another learning arm, measure
how its packed misses change when the number of held impostor products changes.

Use the source-bound public FP16 encoder and checkpoint from the passed v2
In-Shop loader gate. The fixed TRAIN held roles are 6,354 queries and 6,245
gallery images over 1,992 products. Keep **every same-product positive** in
the gallery for each query. For other products, draw 16 product subsets at
fractions 1/4, 1/2, and 3/4 using seed 179027, plus the full set once. The
product subsets are shared across queries and sampled without replacement.
Score the exact 128-D int8/f16 packed representation with TF32 disabled and
stable score/ordinal tie order. Require the full-set result to reproduce
6,203/6,354 public FP16 hits; otherwise the diagnostic is invalid. Record all
miss counts, image export/scoring wall, peak allocated CUDA, authority hashes,
and the fixed-role digests. This is a diagnostic on one checkpoint and one
TRAIN holdout, not an official quality read or a SOTA result.

The predeclared decision is to **stop method-search claims** if misses are
approximately proportional to admitted impostor products: the mean miss count
at each smaller fraction lies within 20% relative error of fraction times the
full-set miss count. If proportionality fails, inspect the direction and
product-level concentration before choosing one causal learning treatment.
No quality gain follows from this diagnostic alone. Do not treat an
official-scale extrapolation as a measured official result.

Frozen script: `scripts/probe_inshop_impostor_scaling.py` SHA-256
`ed043a2fe334752b8e3184c228fea7c79961ab0af4e7a92021decf03d0936893`.
