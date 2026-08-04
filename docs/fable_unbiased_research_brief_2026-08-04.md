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

The recorded external horizon is VAPNet at 0.762 CUB, 0.948 Cars196, and 0.939
In-Shop with ResNet-50/GAP 2048-D and 200 epochs, without a reported seed count;
AdvRF at 0.766 CUB and 0.949 Cars196 under a broadly comparable form; and PFML
five-run ResNet-50 results of 0.734 CUB and 0.927 Cars196. The local corrected
In-Shop Proxy Anchor final-state references are 0.9137009425 and 0.9167956112,
with raw best-over-training 0.9163032775 and 0.9189056126. Two seeds are not a
variance estimate.

Before using local evidence, read `docs/search_protocol.md`,
`docs/current_evidence_reliability_audit_321_2026-08-03.md`,
`docs/search_stopping_adjudication_353_357_2026-08-04.md`,
`docs/method_search_verdict.md`, and `docs/results.md` in full. Read other files
only when needed for exact collisions. Measurements not promoted by the
reliability audit are untrusted; a bug can invalidate a negative as easily as a
positive.

Do not let the catalogue dictate the answer. Solve the research problem from
first principles and choose the scientific framing, representation,
architecture, activation, loss, training algorithm, or other mechanism. Do not
merely combine named methods. Search primary literature and official author
code before claiming novelty. Distinguish exact prior art from adjacent work;
absence from a search is not proof of novelty.

## Required deliverable

1. Diagnose what presently limits unseen-class retrieval.
2. Give the single strongest concrete new method, including exact equations or
   computation, training/inference costs, and why it should beat the relevant
   baseline.
3. Give closest primary prior art and the precise non-cosmetic novelty boundary.
4. Name the exact verified repository measurement motivating it, or a cheap
   prospective diagnostic if current evidence is insufficient.
5. State a numerical prediction, falsification condition, ablations, and
   second-dataset confirmation plan.
6. Adversarially critique implementation bugs, benchmark leakage, selection
   bias, capacity/recipe confounds, and failure modes.

If no method survives, identify the missing fact and the one highest-information
experiment that could reveal it. Do not lower the standard merely to return an
idea.

The research pass is read-only: repository search and primary-source web access
only; no edits, shell, benchmark access, GPU, test labels, or subagents.
