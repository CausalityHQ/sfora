# SOP proxy-clock diagnostic 253

Date: 2026-08-03. Written before the corrected SOP checkpoint or final training
embeddings existed. This is a CPU diagnostic, not a method candidate; Gate 2 has
already ruled out radial gauge control as novel.

## Measurement question

Proxy Anchor normalizes each proxy in the forward pass. Raw proxy norm is
therefore absent from cosine retrieval but scales the tangent Jacobian by
`1 / ||p_c||`. If final norms systematically track class exposure, the optimizer
has created unintended class-specific angular clocks. That would be an important
recipe measurement even though weight normalization and spherical/Riemannian
optimization already occupy the intervention.

Use only the final-state corrected official SOP checkpoint and its independently
exported final training embeddings. For every train product report image count,
raw proxy norm, inverse norm, proxy-to-class-centroid cosine, and mean
sample-to-centroid cosine. Report Pearson and Spearman correlations of class
count with norm/inverse norm and of norm with the two quality statistics, plus
top-minus-bottom proxy-norm quartile differences.

## Prediction and falsification

Predict `|Spearman(class_count, proxy_norm)| >= 0.15` and an absolute
top-minus-bottom norm-quartile difference of at least **0.03** in
proxy-to-centroid cosine. The proposed active-clock interpretation is falsified
if either condition fails. A pass records optimizer behavior but does not revive
candidate 245 or authorize GPU work, because its executable correction remains
occupied prior art.
