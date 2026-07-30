# Candidate 1: dual-timescale EMA

Status: iterative search protocol in progress.

## Gate 1 — provenance: PASS

The idea comes directly from the CUB EMA factorial, not an analogy.

Against the paired Proxy Anchor seed-0 baseline:

| EMA role | momentum 0.999 | momentum 0.99 |
| --- | ---: | ---: |
| distillation target; evaluate student | +0.91 pt | +0.30 pt |
| no distillation; evaluate EMA weights | +0.07 pt | +0.46 pt |

The roles prefer opposite timescales. At momentum 0.999 the average retains
5.3% of its initialization after CUB's 2,940 steps; this contaminates an
evaluated model whose embedding head began random. At momentum 0.99 the
initial contribution is effectively zero, which helps evaluation, but the
distillation target loses 0.61 pt on the hypothesis-generating seed.

A single EMA must compromise. `pa_dual_ema` maintains:

- a slow momentum-0.999 teacher used only for relational distillation; and
- a separate fast momentum-0.99 average used as the evaluated retrieval model.

This changes the temporal supervision available during training while retaining
the independently measured evaluation average. It does not rescue either of the
two refuted stories about headroom or embedding capacity.

The first attempted CUB run was stopped at epoch 5, produced no result artifact,
and is not evidence.
