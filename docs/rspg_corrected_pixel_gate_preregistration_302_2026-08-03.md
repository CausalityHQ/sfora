# RSPG corrected-pixel operating-point gate

Date: 2026-08-03. Registered before the corrected epoch-10 pack exists.

RSPG's prior In-Shop retrieval failure and operating-point density were measured
on `img_highres` and are withdrawn as benchmark evidence. Its primary-source
novelty audit remains narrow but live: the claimed operator is the combination of
a target-excluded class-level rival signature and a positive-to-unknown gate.

The CPU gate cannot be evaluated on the corrected reference's final embedding.
RSPG constructs its graph at epoch 10, so representation stage is part of the
operator. After the corrected full reference passes its `[0.907, 0.929]` raw-best
fidelity interval, train an otherwise source-faithful Proxy Anchor seed 0 for
exactly 10 epochs / 1,440 steps on the validated 256-pixel corpus, with periodic
test evaluation disabled. Export only final epoch-10 training embeddings.

Apply the unchanged RSPG thresholds: top-8 rival overlap at least 4, JS divergence
at most 0.25 over the nearest 32 rival classes. PASS requires retained same-class
edge density in **[0.05, 0.60]** and at least **0.25** of eligible classes split
into multiple connected components. Failure kills RSPG without threshold tuning
or a candidate GPU run. Passing authorizes only a new numerical preregistration
against the corrected baseline; it does not authorize quoting or reusing the old
0.9085 threshold.

The previous favourable high-resolution epoch-10 density was seen before this
repair, so any corrected result remains adjacent to that contamination caveat.
Changing the pixel corpus is necessary to restore benchmark validity, not a claim
of a fully blind rediscovery.
