# In-Shop TRAIN impostor-count scaling gate

The selected official In-Shop bank checkpoint is 1.218 percentage points below
the dated UNICOM 96.7% Recall@1 reference. The current seed-179026 true-freeze
checkpoint has only been measured on product-disjoint official TRAIN held roles;
its official result is unknown. Before choosing another learning arm, measure
how its packed misses change when the number of held impostor products changes.

Use the source-bound public FP16 encoder and checkpoint from the passed v2
In-Shop loader gate. The fixed TRAIN held roles are 6,354 queries and 6,245
gallery images over 1,993 products (1,992 potential impostor products per
query). Keep **every same-product positive** in
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

## Terminal result and decision

The sole DGX Spark GB10 unit `sfora-inshop-impostor-scaling-v1` exited 0,
invocation `fbdc028d77a14ab294a40972eec7b708`. The [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-impostor-scaling-v1/receipt.json)
has SHA-256 `817ca40b10fa670f6572008d7d794829b0b85259875c631c534061fd096371ad`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-impostor-scaling-v1/journal.log)
has SHA-256 `cbdf02a4802a6e639b8451307fb1d855cd05daa4bd327d8868603821815a839e`.
Full-gallery parity passed at **151 misses / 6,354 queries**, or **97.6235%**
packed Recall@1. The 16-draw mean miss counts were:

| Admitted product fraction | Products per draw | Mean misses | Range | Proportional prediction | Relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1/4 | 498 | **65.3125** | 52–76 | 37.75 | **73.01%** |
| 1/2 | 996 | **103.0625** | 93–114 | 75.5 | **36.51%** |
| 3/4 | 1,495 | **130.0000** | 122–138 | 113.25 | **14.79%** |
| Full | 1,993 | **151** | 151 | 151 | 0% |

The predeclared proportionality condition **fails** at 1/4 and 1/2. Misses
rise sublinearly with more impostor products; a linear official-scale
extrapolation from this holdout is invalid. The direction is consistent with
several harmful products competing for some queries, but the aggregate curve
does not prove that mechanism. An independent archived miss receipt identifies
132 distinct top-impostor images among the 151 full-gallery misses; the top ten
account for only 24/151. This argues against one small, globally dominant
impostor group. The true-freeze checkpoint has not been read officially; the
selected older bank checkpoint's 95.4823% official result cannot be replaced
by a projection from this different model.

The public encoder export took **100.595 s**, packed score diagnostic
**0.799 s**, and peak PyTorch allocated CUDA was **1.267 GB**. These are
diagnostic costs, not measured production image-to-top-k latency. Keep the
current opt-in checkpoint loader and packed scorer; do not promote a new
In-Shop default or launch another loss arm based on a linear scaling premise.

An independent replay of already committed seed-179026 and seed-179027
true-freeze baseline receipts (SHA-256
`76f1293ac158f64f1e98a9412033ca37667a2baaadb592bf6051d92b61c9c8cc`
and `48f5de680f2b86738d7632b2068abc101c115176cb34258871d39399e4ebba7c`)
compared their **same 12,599 held-only symmetric query** hit vectors. Both
receipts have the same held-row digest. Seed 179026 misses 186, seed 179027
misses 173, with **131 shared misses**, 55 rescued only by seed 179027 and
42 rescued only by seed 179026. Thus a hypothetical per-query oracle over
the two seeds would still miss 131/12,599; it is neither a deployable
single-encoder result nor evidence of an official-scale ceiling. This
overlap, together with diffuse top impostors, does not identify a causal
training change that meets the joint quality and cost target.
