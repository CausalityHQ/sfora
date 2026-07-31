# BLENDER verification — the citation that closed our only untried direction

Checked against the arXiv full text and the authors' code, *before* acting on it.
Primary sources: [arXiv:2601.20246](https://arxiv.org/pdf/2601.20246),
[code](https://github.com/facebookresearch/meta_ai_blender).

## The paper is real, and its baselines are mostly sound

**Correction to an earlier internal claim.** BLENDER was initially characterised here as
resting on weak baselines, by analogy with IDEAL. That is **wrong**, and the record
should say so:

- It correctly quotes published ResNet-50 numbers: Proxy Anchor 69.7/87.7 and HIST
  71.4/89.6 on CUB/Cars.
- **Its own Proxy Anchor baseline is *stronger* than published — 72.7/90.5.** BLENDER-PA
  reaches 74.6/92.3, so the controlled PA gains are **+1.9 CUB / +1.8 Cars**.
- The headline **+3.7 CUB is Potential Field 73.3 → 77.0**, and published PF is 73.4, so
  that baseline is sound too.
- **Retrieval is like-for-like**: ResNet-50, one 512-d embedding per image, 256 resize +
  224 centre crop. No multi-view inference, no descriptor concatenation, no generative
  model at retrieval time. The capacity objection that sank IDEAL does **not** apply.

One genuine baseline gap: on Cars, reproduced PF is **90.2 against a published 92.7**, and
BLENDER-PF reaches only 91.9 — it never recovers the published baseline there.

## Why it nonetheless does not close the direction

- **No seed count, no error bars, no confidence intervals, no paired multi-seed test.**
  "Same seed" appears only in a CLIP comparison of generated images. In a project that has
  retracted results for exactly this, single-run evidence cannot close a direction.
- **No contamination audit.** It imports Stable Diffusion 1.5, LLaVA-Next, CLIP ViT-L/14
  and Segment Anything. It does not investigate whether CUB or Cars appear in those models'
  pretraining data, and does not isolate gains attributable to imported knowledge.
- **No cost reported at all** — 20,000 LoRA/TI steps per object category and 150 samples
  per class per attribute, with no GPU model, wall-clock, GPU-hours, energy or storage.
- Expected survival under a faithful digest-pinned paired six-seed reproduction: **+1 to
  +2 CUB points, not +3.7**. The closest controlled evidence is BLENDER-PA's +1.9.

## Verdict

> BLENDER does not close expanding intra-class supervision. It provides positive,
> single-run evidence for one expensive pretrained-generative implementation. It leaves
> open simpler, contamination-controlled, cheaper, reproducible, and **non-generative**
> forms of expanded intra-class supervision.

That is the basis on which the supervision-expansion class was reopened, and it is a
narrower basis than "the paper is weak" — the paper is reasonable; it is just not
sufficient to foreclose a direction, and its route is expensive and contaminated in a way
a training-data-derived method would not be.
