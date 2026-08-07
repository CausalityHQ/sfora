# Pass 100 — KAN/spline embedding head (killed before GPU)

## Candidate

Replace the small Proxy Anchor embedding MLP with a Kolmogorov–Arnold
Network (learned univariate spline functions on edges), motivated as a
potentially more expressive head for fine-grained similarity.

## Gate 1: provenance

The repository has no measurement showing that the current embedding head is
the bottleneck, nor any measurement linking spline functions to the observed
transfer deficit or persistent error overlap. This is therefore not a
repo-motivated intervention. Gate 1 fails.

## Gate 2: prior art

The base KAN proposal is already an architecture replacement for MLPs
(Liu et al., 2024, https://arxiv.org/abs/2404.19756). KAN variants are already
used for representation learning and retrieval (Yu et al., 2024, KAE,
https://arxiv.org/abs/2501.00420), and KAN-based retrieval/fusion papers are
appearing in the current literature. A KAN head would therefore be a generic
architecture swap, not a defensible unexplored DML mechanism.

## Decision

**DEAD at Gates 1–2. No implementation or GPU run.** The DGX was idle, but
starting this run would spend compute on an unmotivated generic universal
approximator. The next candidate must be tied to a measured failure mode and
survive an explicit retrieval prior-art search.
