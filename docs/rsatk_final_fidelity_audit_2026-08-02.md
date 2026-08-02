# Final independent RS@k source-fidelity audit

Date: 2026-08-02. Performed by Claude against the current native port while the
source-cadenced Cars196 deciding run was still incomplete. No code or GPU state
was changed by the audit.

## Verdict

**NO REMAINING MISMATCH FOUND.** The audit cloned and inspected the official
Patel--Tolias--Matas source at pinned revision
`ed052029d258555df2f94dd82d6f7df60ef7cc6f` and compared it with the current
native implementation. It did not merely re-find the four already corrected
defects (retrieved-count cap, candidate-only rank exclusion, source-exhaustive
sampling plus legacy weights, and source evaluation cadence).

## Direct numerical checks

The official `RecallatK.forward` was transcribed literally, with CUDA calls
removed, and compared with `_recall_at_k_surrogate_loss` on balanced random
batches at rank temperature 0.01:

| Batch | Loss difference | Maximum gradient difference |
| --- | ---: | ---: |
| 32 examples / 8 classes | 0 | `5.0e-08` |
| 80 examples / 20 classes | `1.8e-07` | `2.8e-08` |
| 120 examples / 30 classes | `2.4e-07` | `3.3e-08` |

Those are float32 round-off. The comparison covers the cap, candidate-only
exclusion, query-slot zeroing, `min(k, positives)` normalization, and outer
per-query loss scaling.

The official ResNet-50 wrapper was also instantiated over the same backbone and
given identical GeM/head weights. Native and source 512-D outputs agreed with
maximum absolute difference **0.0**. This confirms
GeM -> flatten -> affine LayerNorm(2048) -> Linear(2048,512) -> L2
normalization, learnable GeM `p=3`, and the source's effective default linear
initialization.

## Inspected recipe surface

- Adam sees all 24,561,217 trainable parameters at learning rate `1e-4`, weight
  decay `4e-4`, and default betas/epsilon; source and native parameter groups
  have identical values.
- BatchNorm running state is frozen while affine parameters remain trainable,
  including re-freezing after every `model.train()` call.
- `MultiStepLR([80, 140], gamma=0.3)` has identical completed-epoch semantics.
- Train transforms use torchvision-default `RandomResizedCrop(224)` scale
  0.08--1.0, horizontal flip, and ImageNet normalization; evaluation uses
  resize 256 plus center crop 224.
- Sampling is 98 classes x 4 images, 14 non-reusing batches per epoch, 2,380
  total updates over 170 epochs.
- Evaluation has exactly 35 selection opportunities at completed epochs 1, 6,
  ..., 166, 170, with no validation carve-out.

Known non-recipe differences are inert: full-batch autograd replaces the
source's detached two-pass replay; evaluation batch size differs without a
batch-dependent evaluation operation; and NumPy float64 retrieval can differ
from FAISS float32 only on exact distance ties. Seed 0 versus the paper code's
seed 1 is a different draw, not a recipe mismatch.

This audit supports interpreting the completed run as a faithful single-seed
reproduction. It does not turn a single seed into uncertainty evidence and
does not authorize a novelty claim.
