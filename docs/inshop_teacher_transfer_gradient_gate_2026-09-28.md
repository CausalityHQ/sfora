# Trained teacher → smaller student: prospective fit-only gradient gate

The frozen Base/PCA pilot remains KILL, unchanged. This new hypothesis adds
supervision from the already trained Large retrieval teacher to a trainable
Base encoder. Plain Base is only a matched control, never a reopened candidate.
Relational distillation is existing prior art, not an invention or SOTA claim.

Dual consequential review80659f7ef3a54ac9 (Opus c7f4c5da8d724b26,
Astra0b093e03f21e4dcd) supports a bounded smoke. Adopt Astra's smaller fixed
configuration: coefficient0.1, temperature0.20, all12 Base blocks trainable.
Teacher is the pinned current seed179026 true-freeze fit-only checkpoint
ad58838e2492cd664a447308317b11a53f362a4c5a36a99f28978f32a3b71089;
Opus's different179023 suggestion is not silently substituted. Load FP32
teacher vision/head with exact state-tensor equality before GPU autocast.

First execute only the cheaper initial-gradient stage, external120-second
aggregate limit including imports/loading, peak allocated CUDA<16GiB:

- Select128 non-singleton original TRAIN fit products by SHA256 domain
  `inshop-teacher-transfer-v1\0` + label, first two images by relative-path SHA.
  No held/official evaluation. All256 images initialize Base's own PCA128,
  class proxies and detached bank. This is a smoke approximation, not the
  complete training recipe. No full fit-cache export.
- First16 selected products/two images each provide the fixed32-image probe.
  Both native processors must produce identical FP32 pixels. Native source
  parameters FP32, BF16 autocast; exact teacher checkpoint remains unchanged.
- Existing ArcFace margin0.3/scale64 plus bank SmoothAP×8 is the main loss.
  Reuse leave-self-out relational KL on normalized compact128 outputs, no
  new objective implementation. Sham cyclically shifts complete two-image
  product groups, preserving positive-pair structure but changing off-class
  relations. This checks information beyond labels, not transfer quality.
- GO only if true relation-only gradients finite/nonzero in first/last Base
  blocks and head, teacher/classifier have no relation gradients, weighted
  encoder relation/main norm ratio in[1e-4,1], teacher KL from uniform and
  pair-preserving sham each>1e-4nats, true/sham encoder-gradient relative
  difference>1e-3, authority/resource gates pass. Any failure KILL, no tuning.

A pass authorizes the previously reviewed eight batch32 updates in each of
three serial matched arms (main-only, true teacher, pair-preserving sham),
under a separately frozen aggregate120s limit/16GiB; all finite updates,
rank/variance≥50%initial, rank≥2, distinct true/main parameter updates,
initial/terminal gradient route and treatment step overhead≤3× control.
It does not authorize full training, new held reads, p99 calls or promotion.
Teacher acquisition/original training and target generation must count in
subsequent training cost. The production API/wheel stay unchanged.
