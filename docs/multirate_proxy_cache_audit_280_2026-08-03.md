# Candidate 280: multirate proxy/cache optimization

**Verdict: DEAD at Gate 2; no implementation or GPU.**

## Gate 0--1 provenance

The repository operation audit found that Proxy Anchor's sample--proxy matrix is
only about 1.043 billion multiply-accumulates (2.086 GFLOP) over the audited SOP
recipe.  BN-Inception forward/backward work dominates training.  A credible speed
method must therefore remove backbone evaluations or backward passes rather than
factorize the proxy matrix.

The cross-field proposal came from reversible reference-system propagator algorithms
in molecular dynamics: update slow expensive state less often than fast cheap state.
Applied here, an embedding cache would be refreshed by occasional image-network
forward/backward steps, with several cheap proxy-only updates between refreshes.

## Gate 2 adversarial prior-art and mechanism audit

The mechanism does not survive.

1. Wang et al., *Cross-Batch Memory for Embedding Learning* (CVPR 2020), explicitly
   train metric objectives against a memory of stale embeddings.  Thus cached,
   temporally decoupled embedding interactions are established DML machinery:
   <https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html>.
2. Gürbüz, Can, and Alatan, *Deep Metric Learning With Chance Constraints* (WACV
   2024; earlier ASAP DML), repeatedly solve proxy-based subproblems and alternate or
   reinitialize proxies.  Thus alternating proxy/model timescales are also established:
   <https://openaccess.thecvf.com/content/WACV2024/html/Gurbuz_Deep_Metric_Learning_With_Chance_Constraints_WACV_2024_paper.html>.
3. Combining a stale embedding bank with extra proxy-only block-coordinate steps does
   not create a new supervision source.  During the skipped expensive steps it cannot
   improve the image network at all; it only optimizes proxies against increasingly
   stale coordinates.  The operation-count measurement proves those added updates are
   cheap, but supplies no evidence that fewer backbone gradients reach the same quality
   in less wall time.

The precise candidate therefore reduces to established memory-bank DML plus
established alternating proxy optimization.  It fails novelty before a preregistration,
implementation, or GPU screen.

