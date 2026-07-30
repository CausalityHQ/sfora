# Iterative search protocol

Run this in a loop until a novel method that outperforms is found and benchmarked, or
until the evidence supports arguing that none exists. `pa_dual_ema` is candidate 1.

## Why a protocol rather than more ideas

Fifteen candidates failed and the failures were not random. Every one either

- **(a)** added regularisation to a base that was already fitting well, or
- **(b)** swapped the similarity function while leaving the supervision signal intact.

Three more died on **prior art** only *after* effort was spent. The loop below is ordered
to kill candidates cheaply, and in the order that kills most of them first.

## The gates — stop at the first one a candidate fails

**1. Provenance of the idea.** It must be motivated by a measurement *in this repo*, not
by an armchair analogy. `pa_dual_ema` is the model: it came from the factorial showing
the distillation-target role wants 0.999 while the evaluated-average role wants 0.99, so
a single EMA must lose one. Write the motivating number down.

**2. Prior art, before any GPU.** Search properly. H3 was EMAN (Cai, CVPR 2021).
Gauge-fixing was Hoffer (ICLR 2018) and Pernici (TNNLS 2022). Snapshot-Procrustes was
Huang (ICLR 2017) + Izmailov (UAI 2018). Multi-center was SoftTriple (ICCV 2019) + SwAV.
**All four were found after the work.** If it exists, record the citation in
`docs/method_search_verdict.md` and move to the next candidate. Cheap to check,
expensive to skip.

**3. Pre-register.** A number you expect, and the condition that falsifies it, committed
*before* the deciding run. Four have been registered here and all four failed — that is
the process working.

**4. Screen on In-Shop, not CUB.** In-Shop σ = 0.12 pt, so **one seed per arm is
decisive**; CUB σ = 0.57 and needs 5–17. One In-Shop run at 2.2 h beats six CUB runs for
any effect under 2 pt. Use `pa_ema_avg_bnfix`-style buffer handling wherever BatchNorm is
trainable, or the average is only half an average. CUB screening at n=1 produced a false
positive here once already (+0.52, retracted).

**5. Confirm out-of-sample.** Never quote the seeds that generated the hypothesis.
In-sample gave +0.890; out-of-sample +0.427. An sd from 2–3 runs is worthless: n=3 gave
0.153, n=6 gave 0.367.

**6. Report raw *and* selection-corrected.** Run `scripts/measure_selection_bias.py`.
Best-over-training inflates every arm by 0.35–0.84 pt and differs **between** arms by up
to 0.42 — enough to reverse a ranking. A stabler method is penalised by the protocol.

**7. Replicate on a second dataset.** Cars196 or CUB. One dataset is an observation; two
is a result.

## When a candidate dies

Write it into `docs/method_search_verdict.md` with its **mechanism**, not just its
number. That catalogue is worth more than any single arm.

Then generate the next candidate, preferring the one untried class: **change what
supervision exists**, rather than how it is scored.

## Standing fact: the ceiling is open

IDEAL's 72.3 rests on a 69.7 HIST baseline — below HIST's published 71.4 and below our
70.82 six-seed reproduction — and it uses four-view inference. So the headroom above HIST
is not credibly occupied.

Our reproductions: **HIST 0.7082, Proxy Anchor 0.6919** on CUB; **PA 0.9035, HIST 0.9038**
on In-Shop. A real single-model arm above ~0.715 on CUB, or above 0.9038 on In-Shop, is a
genuine result.

Commit and push after each gate. Report honestly whichever way each candidate goes.
