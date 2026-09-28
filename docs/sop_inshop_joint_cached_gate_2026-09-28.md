# Actual joint supervision: frozen cached TRAIN-fit gate

One mechanism: actual concurrent SOP supervision may regularize the compact
In-Shop representation, despite the failed sequential warm start and the
failed synthetic In-Shop-category proxy. Test this bounded cached version
before considering an encoder experiment. This is known joint training.

## Frozen data, comparison and stopping rules

Use the immutable audited inventory receipt SHA256
`853bdac748c891e90804a52199fd90a007e0492c7041245a650dc768fec5237b`.
Its original positive-width guard was corrected explicitly in the
[inventory decision](sop_inshop_joint_inventory_gate_2026-09-28.md), before
any quality read. Target: 6,757 In-Shop TRAIN-fit rows / 995 products.
External: 5,190 SOP TRAIN-fit rows / 995 products. Validation: 6,514 images
/ 997 products, **3,440 queries / 3,074 gallery images**, wholly inside the
original In-Shop fit split. No outer holdout or official query/gallery read.

Reuse the existing cached pooling runner, adding the actual-data loader only.
All arms start with the same In-Shop-only image-PCA 1024-to-128 head, imprinted
class proxies, native ArcFace margin0.3/scale64, coefficient8 detached bank
SmoothAP, AdamW1e-4/decay0.05/global clip1. Seed179033, **100 updates per arm**,
64 cached images per update, eight CPU threads, no encoder updates or CUDA.

- Native control: 64 In-Shop presentations each update.
- Joint arm: the first 32 presentations of the same native batch plus 32 SOP
  presentations from the fixed seed179034 external coverage schedule.
- Sham: identical joint images, ordering, count, initialization procedure,
  class counts and bank shape, with a count-preserving permutation of SOP
  labels (seed179033). At least90% of external labels must change; target
  labels remain exact. Sham proxies reflect its shuffled training labels.

Each arm gets exactly6,400 presentations. Native gets twice as much target
exposure; joint has more classifiers and bank rows. This is an equal-compute
joint system comparison, not isolated identity-count causality. Sham controls
image diversity, bank/class counts and exposure, but necessarily changes
proxy initialization and semantic positives. Report all these limitations.

Hash both feature caches and metadata/inventory before use. Reject malformed
source geometry, non-finite source/loss/gradient/bank, changed split roles,
incomplete updates, or failed sham/namespace guards immediately. A process
alarm bounds main work at120s; an external140s timeout also covers imports and
receipt writing. No partial-arm quality comparison or adaptive step reduction.
Do not launch duplicate work or a GPU encoder after a failed cheap gate.

Score only after all three100-update arms finish. Use actual 130-byte packed
128-D vectors and existing stable-ordinal CPU scorer, R@1 and mAP@R.
Product-cluster bootstrap5,000 draws/seed179019 is conditional on this one
training seed and reused internal panel.

Advance only if joint minus native R@1 is at least **+0.30 percentage points**
with positive95% lower bound, and mAP@R is nonnegative; joint minus sham R@1
must be at least **+0.20 points** with positive lower bound and mAP@R
nonnegative. Otherwise KILL this fixed cached configuration without changing
seed, coefficient, budget, source, split or thresholds. A negative frozen-head
proxy is not a universal impossibility result for encoder joint learning.

A pass licenses a separately frozen bounded encoder/data/gradient/cost smoke
with matched controls, then paired seeds and untouched confirmation if useful.
It does not license default promotion, an official read, serving certification
or SOTA. Cached training/init wall is reported per arm; image-to-top-k
p50/p95/p99/QPS, encoder images/s and VRAM are **unmeasured** here.
Cross-dataset pixel duplication has not been audited and must be addressed
before any external-data generalization claim.
