# Candidate 320: confusion-edge margin (CEM)

## Gate 1 — provenance

On corrected In-Shop seed-0 embeddings, nearest foreign image and nearest
foreign proxy classes agree for 15.69% of samples. These samples have 2.3886%
leave-one-out error versus 0.1466% when the signals disagree.

## Gate 2 — verdict

**FAILED; candidate dead.** CEM constructs a directed class graph `c -> d` only
when independent sample- and proxy-level evidence agrees that `d` is the
confusing foreign class.  However, the implemented relation does not add a new
label or positive relation.  Its softplus margin increases similarity to the own
proxy and decreases similarity to one selected foreign proxy.  The edge is
therefore a static hard-negative-class selector, and its support fraction is a
hard-class weight.

Sohn, *Improved Deep Metric Learning with Multi-class N-pair Loss Objective*
(NeurIPS 2016), already introduces hard negative **class** mining.  Suh et al.,
*Stochastic Class-Based Hard Example Mining for Deep Metric Learning* (CVPR
2019), explicitly select hard negative classes from class-to-sample distances
before an instance-level search.  Agreement between an image-level and a
proxy-level estimate is a different selection heuristic, not a new supervision
primitive.  Chen et al., *Confusion-Based Metric Learning for Regularizing
Zero-Shot Image Retrieval and Clustering* (TNNLS 2022/2024), is a name-level
near-neighbour but not the decisive collision: its energy- and
diversity-confusion terms adversarially confuse local/global feature
distributions rather than selecting a foreign class.

Primary sources:

- https://proceedings.neurips.cc/paper_files/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf
- https://openaccess.thecvf.com/content_CVPR_2019/html/Suh_Stochastic_Class-Based_Hard_Example_Mining_for_Deep_Metric_Learning_CVPR_2019_paper.html
- https://doi.org/10.1109/TNNLS.2022.3185668

## Gate 3 preregistration

This preregistration was committed before the run, but Gate 2 had not actually
been closed. One corrected official In-Shop seed. Baseline final-state R@1 is 0.91370094.
Prediction: **0.9155** final-state R@1. Falsifier: **<0.9140** or failure to
beat the paired baseline. No GPU until a unit test verifies the registered
directed relation and unchanged unrelated gradients.

## Execution post-mortem

The unit tests passed and the run was mistakenly launched before the adversarial
Gate-2 audit was complete.  It was terminated at 400 / 8,580 steps as soon as
the hard-negative-class collision was established.  The partial best R@1 was
0.7734 at epoch 3 and is not a benchmark result.  No result file was produced.

The static graph also depends on a fully trained seed-0 baseline checkpoint and
training embedding export.  Thus the proposed method was not approximately 1x
training: its deciding run inherited an additional baseline-training pass and a
fixed-teacher representation.  That cost and dependence would have required
headline disclosure even if the mechanism had survived.

Process lesson: a plausible one-sentence distinction and a unit test do not
close Gate 2.  The implemented gradient must be reduced to the supervision it
actually supplies before GPU launch.  Here that reduction immediately exposed
ordinary hard-negative-class mining.
