# MET-small OPQ64 local ADC-gap codebook pilot preregistration

## Question

Powered MET-small establishes a 0.007776 mMP@5 gap between the exact float
ceiling and deterministic 64-byte OPQ64x8. This one-seed, claim-ineligible
pilot asks whether better shared codewords can recover a material part of that
fine-ordering gap while leaving the OPQ rotation, every 64-byte assignment,
and exact squared-L2 ADC runtime unchanged.

## Frozen construction

- authenticate feature SHA-256
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- recreate the deterministic 3,050-query / 35,257-gallery protocol;
- fit Faiss 1.12.0 `OPQ64_768,PQ64x8` with one OMP thread;
- freeze its rotation, 64 assignments per row, and packed codes;
- choose 5,120 gallery anchors by SHA-256 order of their stable path under the
  literal salt `local-gap-20260920`; use the first 4,096 for optimization and
  the next 1,024 only for monitoring;
- for each anchor, freeze 96 candidates: exact-float top 32, incumbent-ADC top
  32, and 32 deterministic uniform gallery ordinals; exclude the anchor itself;
- optimize only codewords for 1,000 Adam steps, learning rate 0.01, batch 64,
  from the incumbent initialization;
- minimize query-centered smooth-L1 error between exact source distance gaps
  and exact ADC gaps, normalized by the training panel's median absolute
  centered source gap, plus `0.01 * mean(U^2)`;
- parameterize movement in units of baseline per-coordinate reconstruction
  RMSE and project every codeword to a five-percent block-RMS trust region;
- use the final step with no checkpoint selection;
- compare incumbent, exact conditional-mean recentering under the same frozen
  assignments, and local-gap codewords through the same exact lookup ADC;
- report powered primary and 129-query shifted descriptive metrics,
  reconstruction MSE, train/monitor gap traces, hashes, and runtime.

No query label enters fitting. The official MET test remains sealed. This is a
mechanism pilot, not independent confirmation, because prior powered-query
results informed the choice to run it.

## Frozen decision

Expand to three codec seeds and a fresh domain only if all hold:

1. final training gap loss is at most 90% of its initial value;
2. final monitor gap loss is at most 95% of its initial value;
3. powered mMP@5 improves by at least 0.0025 over incumbent OPQ64x8;
4. powered Recall@1 changes by at least -0.002;
5. powered mMP@5 exceeds conditional-mean recentering by at least 0.001.

Otherwise close frozen-assignment codebook learning at 64 bytes and move the
intervention upstream to rotation/assignment or to a different codec family.
No custom CUDA, cuTile, or CUDA-Oxide kernel is authorized by this pilot.
