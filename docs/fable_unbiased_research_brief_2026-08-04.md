# Outcome-only Fable research brief

Date: 2026-08-04. This brief was fixed before the next successful Fable run.
The first invocation produced no research output because the Claude session
limit was exhausted; retry it unchanged after the service resets.

## Goal

Invent a genuinely novel similarity-learning method that outperforms existing
methods on at least one standard unseen-class image-retrieval benchmark
(CUB-200-2011, Cars196, or DeepFashion In-Shop), then provide a path to validate
it rigorously and replicate it on a second dataset.

## Problem setting

Training and test classes are disjoint. Training provides images and class
labels. Standard evaluation maps each image independently to one descriptor and
uses nearest-neighbour retrieval. Do not design from official test labels or use
transductive test-set adaptation. A changed setting is allowed only if it is
identified explicitly rather than reported as a standard result.

Keep two comparison lanes distinct. In the broadly comparable ImageNet-1K CNN
lane, the recorded external horizon is VAPNet at 0.762 CUB, 0.948 Cars196, and
0.939 In-Shop with ResNet-50/GAP 2048-D and 200 epochs, without a reported seed
count; AdvRF at 0.766 CUB and 0.949 Cars196 under a broadly comparable form; and
PFML five-run ResNet-50/512-D results of 0.734 CUB and 0.927 Cars196. In the
less constrained pretrained-backbone lane, VPTSP-G reports 0.885 CUB, 0.912
Cars196, and 0.925 In-Shop with an ImageNet-21K-pretrained ViT-B/16 and a 512-D
descriptor. Its supplement also reports 0.867 CUB, 0.974 Cars196, and 0.965
In-Shop using the vision tower of a ViT-L/14 CLIP model pretrained on LAION-2B.
That web-scale result is a third lane: it imports vastly more data and capacity,
has a materially different contamination surface, and is not a controlled
comparison to the ImageNet-1K CNN setting. The recorded maxima across all three
lanes are therefore 0.885 CUB, 0.974 Cars196, and 0.965 In-Shop, but the primary
method-development target in this repository remains a reproducible gain in the
ImageNet-1K fixed-recipe lane. Backbone, pretraining corpus, descriptor
dimension, training cost, seed count, and uncertainty must accompany every SOTA
comparison.

The local corrected In-Shop Proxy Anchor final-state references are
0.9137009425 and 0.9167956112, with raw best-over-training 0.9163032775 and
0.9189056126. Two seeds are not a variance estimate.

This is the invention pass, not the collision-check pass. Do **not** read the
repository's candidate catalogue, method-search verdict, stopping adjudication,
candidate files, or prior Fable outputs before fixing your proposal. They contain
hundreds of attempted mechanisms and would anchor the search. Work from the
problem and benchmark evidence above. You may inspect the repository only if an
exact implementation fact is indispensable; otherwise state the missing fact as
a prospective diagnostic.

Solve the research problem from first principles. Choose the scientific framing
and concrete mechanism yourself. Do not merely combine named methods. Search
primary literature and official author code before claiming novelty.
Distinguish exact prior art from adjacent work; absence from a search is not
proof of novelty. Once your strongest proposal is fixed, report its closest
prior art, but leave the repository-wide catalogue collision check to the next
independent audit.

## Required deliverable

1. Diagnose what presently limits unseen-class retrieval.
2. Give the single strongest concrete new method, including exact equations or
   computation, training/inference costs, and why it should beat the relevant
   baseline.
3. Give closest primary prior art and the precise non-cosmetic novelty boundary.
4. Name the empirical fact motivating it from the evidence above, or a cheap
   prospective diagnostic if that evidence is insufficient.
5. State a numerical prediction, falsification condition, ablations, and
   second-dataset confirmation plan.
6. Adversarially critique implementation bugs, benchmark leakage, selection
   bias, capacity/recipe confounds, and failure modes.

If no method survives, identify the missing fact and the one highest-information
experiment that could reveal it. Do not lower the standard merely to return an
idea.

The research pass is read-only: repository search and primary-source web access
only; no edits, shell, benchmark access, GPU, test labels, or subagents.
