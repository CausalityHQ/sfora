# SOP SigLIP2 matched compact training screen

Frozen before the 1,000-update runs. All selection uses SOP official TRAIN
identities only. This is a causal screen, not an official TEST or SOTA
evaluation.

## Source and split

- Public google/siglip2-large-patch16-256, immutable revision
  787800c8990e6f058423089178e718139608408c; model files and the
  original 59,551-row feature export are SHA-checked at launch.
- SOP official TRAIN only: seed 179019 product-disjoint partition of 53,700
  fit images (10,186 products) and 5,851 holdout query images (1,132
  products). All arms score every holdout query against the entire 59,551
  TRAIN gallery, excluding self. No TEST image chooses a setting.
- The pinned pretrained source features initialize a common fit-only centered
  1024-to-128 PCA head and class-mean proxies. The source export receipt's
  ordered-row hash must match the authenticated UNICOM L/14 SOP TRAIN archive.
  Initial head, proxies, schedule, and the first ten augmented input batch
  hashes must match across arms.

## Matched training settings

The ArcFace control, float-rank control, and Sfora deployed-code rank
treatment start from the same model, head, proxies, and seed. Each trains the
entire vision tower plus 128D head for **1,000 updates**, microbatch 64,
16 identities × 4 images, two DataLoader workers, AdamW weight decay 0.05,
vision learning rate 1e-5, head/classifier learning rate 1e-4, gradient clip
1.0, fp32 parameters with fp16 vision autocast and GradScaler initial scale
128. Random resized
crop to 256 (scale 0.8–1.0, torchvision default ratio), random horizontal
flip, and the pinned SigLIP processor are common to all arms. Evaluation uses
the pinned deterministic processor without training augmentation.

All arms include ArcFace margin 0.3 and scale 64 on normalized 128D head
features. The float-rank arm adds SmoothAP on those float features. The
deployed-code arm adds SmoothAP after signed-int8 fake quantization and f16
inverse-norm simulation with straight-through gradients. Both rank arms use
coefficient **8.0**. This coefficient was chosen solely from a 20-step
packed-rank fit-batch gradient probe at coefficient 2.0: the rank-to-ArcFace
head-gradient norm ratio had median **0.0778** and mean **0.0761** over 20
updates, so coefficient 2.0 would weakly perturb the control. Linear scaling
predicted about 0.31 median at coefficient 8.0. A separate 20-step fit-batch
check with coefficient 8.0 measured median **0.3020** and mean **0.3307**;
its minimum was 0.000037, so some random batches remain saturated. No
holdout score chose the coefficient. The two diagnostic receipts are
[coefficient 2.0](../../evidence/compact_metric/sop-siglip2-substrate-v1/train-packed-rank-gradient-v5.json)
and [coefficient 8.0](../../evidence/compact_metric/sop-siglip2-substrate-v1/train-packed-rank-gradient-v6.json).

## Evaluation and decision

Each finished arm exports the same 59,551 TRAIN descriptors using the trained
vision tower and head, packs each into 128 signed bytes plus one f16 inverse
norm, then scores the full-gallery holdout. Report Recall@1, mAP@R,
per-query outcomes, training wall time and step distribution, full export
time, score time, peak CUDA allocation and parent host RSS. The released
CuTile native top-10 must match stable full-gallery packed-score sorting and
score arithmetic for every holdout query. A zero-update run must reproduce
the frozen pretrained packed quality within 0.2 percentage points Recall@1
and 0.003 mAP@R. Any nonfinite or skipped optimizer step aborts the arm and
counts as a failure under this recipe.

Promote the deployed-code treatment beyond the screen only if its paired
product-bootstrap 95% Recall@1 interval against the ArcFace control has a
positive lower bound, its point gain is at least 1 percentage point, its
mAP@R does not regress, and training wall time is no worse than 1.15× the
control. Compare the float-rank arm under the same rule to isolate ranking
training from deployed-code training. A one-seed result cannot establish a
learning-method claim: at least three paired independent seeds, In-Shop
replication, CUB/Cars transfer, and official protocol evaluation remain
required. If the screen fails, diagnose augmentation, loss saturation,
representation geometry, precision, or scorer before choosing one revised
experiment.

## Prespecified revision after a stability failure

The first 1,000-update ArcFace control with GradScaler initial scale 1024
failed at update 131 with a nonfinite gradient norm; it produced no quality
result. Its [terminal log](../../evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-1000-v7-failure.log)
and [v7 trainer source](../../evidence/compact_metric/sop-siglip2-substrate-v1/train-trainer-source-v7.py.txt)
are retained. A repeat with gradient instrumentation reproduced the failure
at update 131: 13 nonfinite gradients were confined to the first vision
transformer layer and patch/position embeddings, while the objective and
forward outputs were finite. Its [diagnostic log](../../evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-gradient-diagnostic-v8.log)
and [v8 source](../../evidence/compact_metric/sop-siglip2-substrate-v1/train-trainer-source-v8.py.txt)
are retained. Relative to the v8 diagnostic source, the only code change
reduced the GradScaler initial scale to 128. That replay completed 150/150
finite ArcFace updates including update 131 in the
[successful diagnostic receipt](../../evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-scale128-150-v9.json)
with [v9 source](../../evidence/compact_metric/sop-siglip2-substrate-v1/train-trainer-source-v9.py.txt).
This supports fp16 backward scaling overflow, but 150 updates do not establish
long-run stability. We therefore refreeze all three 1,000-update arms with
scale 128 and the same strict nonfinite/skip-step failure policy. The failed
1024-scale control remains a reported negative result, not a completed arm.
If either rank arm aborts, that establishes a precision failure under this
recipe, not a learning-method quality comparison. The v7 zero-update gate
predates this scale revision; the exact v10 source needs its own zero-update
gate before the matched-arm quality decision.
