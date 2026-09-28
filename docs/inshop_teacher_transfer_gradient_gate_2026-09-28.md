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

## Initial terminal: GO to bounded smoke

Original DGX unit `sfora-inshop-teacher-transfer-gradient-v1`, invocation
`ac083ecebeba43f6a65cbe7f9ced7fd9`, exited0. All frozen gates passed:
teacher-uniform KL1.28187585nats; pair-sham KL0.26341826nats; weighted
relation/main encoder-gradient norm ratio0.01525828; true/sham encoder
gradient relative difference1.14322730. First/last/head gradient norms
0.43745269/0.21349888/0.04587354; teacher/classifier relation gradients absent.
Main wall11.731349s, peak allocated CUDA6,110,278,144bytes, teacher probe
forward0.283643s. No optimizer updates or quality/serving reads occurred.
Receipt SHA946bee5474682282489b47702fdac152afdfdf09c4e365aa475c6367948023ab.

Next eight-update smoke uses the same authenticated256-image panel. Fixed
seed179024 PCG64 picks16 products without replacement per batch, both images
consecutive; all three arms reset identical FP32 vision/head/classifier and
optimizer states. All12 Base blocks train. Class supervision/rank bank,
learning rates1e-5/1e-4, decay0.05, clipping1 stay fixed. Diagnostics use the
fixed first32 images after every update. Baseline computes teacher KL for
logging without optimizing it; true and pair-sham add the fixed0.1KL term.

For this fixed-pixel smoke only, cache256 teacher targets once and count
their generation wall separately. Per-update overhead measures student
forward/backward/optimizer, excluding common target generation and extra
diagnostics. **This does not measure online augmentation KD training cost**;
changing pixels in a later experiment requires corresponding teacher targets.
Require all24 stable updates and finite parameters, rank/variance floors on
every update, initial/terminal encoder/head relation route, unchanged teacher
probe codes, nonzero encoder/head update difference true versus main, and
treatment update p50≤3×baseline; aggregate imports/loading/diagnostics≤120s
and CUDA peak<16GiB. No immediate quality or monotonic KL criterion is added.
