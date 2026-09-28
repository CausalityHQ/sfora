# Head-first gradient-pressure gate

One mechanism: allowing the PCA head/classifier to adapt **before** encoder
updates could reduce early encoder pressure from a poorly adapted downstream
classifier. This is a training-phase change, not an auxiliary source loss or
cached lower-layer training. It leaves the deployed architecture and scorer
unchanged and makes no novelty claim.

First run one <=120-second CPU screen on the authenticated pretrained TRAIN
cache, fit-only 13,283 rows/2,004 products, seed179024 and original schedule.
Warm only the original 128-D head/classifier for **100** cached batches using
the unchanged native ArcFace margin0.3/scale64, AdamW1e-4/decay0.05 and global
clip1. Source vectors are detached. These cached vectors have no stochastic
image augmentation, so this is a mechanism falsifier rather than a faithful
full-training comparison. Warm-up does not run the image bank objective;
the subsequent image phase would retain original ArcFace+8xSmoothAP.

Before and after warm-up, rebuild the same fit member bank in each head's
geometry. On the SAME first17 cached batches, measure the complete image-phase
objective's gradient at each raw source vector, with the original singleton
rank skips. This raw-pooler gradient is a proxy for encoder pressure, not
an encoder-gradient or clipping claim. Freeze advance rules:

1. Median paired warm/initial source-gradient norm ratio <=0.75.
2. Probe mean native ArcFace loss falls at least20%, with no nonfinite loss,
   gradient, parameter or zero/undefined source-gradient statistic.
3. Normalized compact outputs on those same rows retain >=80% of initial
   centered variance and participation effective rank.
4. Partition/cache/fit/PCA/schedule authority and 100 cached-step/all17 probe
   inventories match, total CPU wall<=120s.

Any failure kills this fixed100-step head-first configuration without step
count/rate/margin search. Passing only permits a frozen <=2min GPU gradient/
cost smoke; it does not automatically authorize100/1000 image updates. A
later matched-seed TRAIN gate must account for6400 extra cached exposures,
warm-up wall and memory, full training cost, selection uncertainty and a new
independent holdout. No official labels or scores are read by this screen.
