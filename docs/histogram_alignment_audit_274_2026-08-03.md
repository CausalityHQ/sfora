# Candidate 274: label-conditional pixel-histogram alignment

**Verdict: DEAD at Gate 1. No diagnostic, implementation, or GPU.**

A constrained Claude Haiku pass proposed computing a fixed channel-wise pixel
histogram `H_i` and adding

```text
same class:      ||H_i - H_j||^2
different class: max(0, margin - ||H_i - H_k||^2)
```

This does not satisfy the search protocol. No validated repository measurement shows
that color/tonal histograms predict identity retrieval, explain a failure, or expose a
missing relation. RSPG and ARCG show dataset-dependent relational signals, not that raw
color is useful. The proposal also retains exactly the original binary class labels and
adds a fixed-feature regularizer; it does not decide which new supervision exists.

Prior art reinforces rather than decides the death. Color histograms are foundational
content-based image-retrieval descriptors, histogram matching is standard, and Gram/
distribution alignment is established in style transfer and domain adaptation. On CUB,
Cars, and In-Shop, such a term is especially vulnerable to background, photography, and
paint/color shortcuts rather than fine-grained identity.

Candidate 274 stops at provenance. No free diagnostic is warranted merely to manufacture
the measurement that the idea lacks.
