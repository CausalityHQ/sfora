# OBD-SD primary-source audit: batch diffusion as a training teacher is occupied

Date: 2026-08-04.

## Primary source

Zeng et al., *Improving Deep Metric Learning via Self-Distillation and Online
Batch Diffusion Process*, **Visual Intelligence 2, 18 (2024)**:
https://doi.org/10.1007/s44267-024-00051-0

This reviewed, open-access paper was absent from the repository's prior-art
ledger even though it is a direct neighbour of several contextual-similarity
and graph-teacher proposals.

## Exact mechanism

OBD-SD freezes the student from epoch `t-1` as the teacher for epoch `t`, builds
the teacher's within-batch affinity and normalized transition matrices, and
uses the closed-form diffusion

`A_T = (1 - omega) (I - omega S_T)^(-1) D_T`

as a soft similarity target for the current student's pairwise distances. The
paper explicitly distinguishes this training-time use from test-time diffusion
reranking. It evaluates CUB-200, Cars196, and SOP, reports averages over
multiple seeds, and tests the module with several DML losses.

## Consequence

Training an ordinary deployed descriptor to reproduce a graph-diffused or
random-walk-refined similarity matrix is occupied. Changing the fixed point,
normalization, neighbourhood truncation, or teacher update cadence changes the
contextual estimator or stabilization recipe, not the supervision relation.
The source directly strengthens the Gate-2 deaths of candidate 28
(contextual-similarity distillation), candidate 146 (training-only rich
similarity distilled into a global descriptor), and equilibrium graph-teacher
variants considered in candidate 363.

This is a prior-art correction, not a new candidate or benchmark ceiling. No
implementation or GPU run follows.

