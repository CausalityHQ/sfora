# Sfora autoresearch program

The pattern is adapted from karpathy/autoresearch:
- a fixed, read-only evaluation harness
- one mutable surface per experiment
- a fixed budget per experiment
- a tab-separated results log
- keep a change only if the metric improves
- simpler is better
- keep looping without asking

This is a workflow, not a framework. Install nothing, add no paid services, and never relax a gate after seeing a result.

## Roles per iteration
1. **Root** (persistent `sfora/main`): picks the next hypothesis. It asks which perspectives matter (retrieval researcher, systems engineer, statistician, skeptical production engineer) and what each would argue, predict and object to. Simulated views are not evidence.
2. **Contextless specialist:** one fresh helper per source packet. It gets only the task, code pointers, facts, constraints and acceptance checks. It never gets the transcript or a preferred answer. It runs CPU tests on the DGX, makes one commit, and is then closed.
3. **Contextless critique** (only when the production API or wire format changes): task, diff, evidence and checks only.
4. **Root:** commits the pre-registered decision rule before any number exists, runs the experiment on the DGX under a detached `setsid` driver with flock, logs a row in `results.tsv`, commits evidence, and picks the next hypothesis.

All compute runs on the DGX (spark-2751). The devbox host only edits text and runs git.

## Loop Q: quality (research)
- **Harness (read-only):**
  - Trainer: `scripts/train_sop_siglip2_substrate_pair.py` (78546d9c bytes) with the staged src from 29a45d38.
  - TRAIN-held scorer: `evaluate_packed` / `score_packed_full_gallery`.
  - Split: SOP TRAIN, product-disjoint, seed 179019, fit 0.9 (5,851 held queries vs the full 59,551 gallery, packed int8).
- **Mutable surface:** one new wrapper `scripts/train_sop_*_pilot.py` per experiment. It monkeypatches the trainer and never edits pinned bytes.
- **Budget:** 1,000 updates at 280 px (~27 min train + ~6 min eval on GB10). At 384 px only once 280 px has passed.
- **Keep rule:** held R@1 paired product-bootstrap lower95 > 0 against the current incumbent at the same resolution and seed. Also report mAP@R; a significantly negative mAP@R is a KILL.
- **Promotion:** a kept change gets exactly ONE official SOP TEST read under a pre-registered frontier rule. Official TEST is never used for selection.
- **Crash rule:** fix trivial bugs and rerun, archiving the failed attempt under `runs/failed-attempts`. A broken idea is logged as a crash.

## Loop P: production (ship the frontier through the library)
- **Harness (read-only):**
  - `scripts/bench_sop_so400m_encode_variants_v1.py` (cd73781f) for encode speed and drift.
  - A frozen end-to-end serving check: encode → pack → search. Per-query top-1 must equal the incumbent's on the 5,851 TRAIN-held queries, or the drift budget is stated in advance.
- **Mutable surface:** the serving module for the So400m frontier (a new `src/sfora/` file; existing pinned files are never edited).
- **Metric:** end-to-end images/s at batch 64, plus p50/p99 latency at batch 1, at peak memory, under the quality gate.
- **Keep rule:** throughput or latency improves beyond run-to-run noise (3 repeats) AND the quality gate passes.
- **Milestones, in order:**
  - **P1:** So400m encoder + 512-D head + packed index API, matching the training export exactly.
  - **P2:** native int8 search for 512-D, or a measured decision that torch search is fast enough.
  - **P3:** serving bundle export/load with hashes.
  - **P4:** package, tests and CI green.
  - **P5:** documented API and a frontier table.

## results.tsv
- Location: `docs/autoresearch/results.tsv`. It is committed so it carries provenance.
- Columns: `commit loop metric value lower95 cost status description`. Tab-separated.
- `status` is one of `keep`, `discard`, `crash`, `official`.

## Stop conditions
Stop only for:
- actual completion of a milestone set by the operator
- a concrete external blocker (report it, then continue other authorized work)
- a hard resource or spend ceiling

A negative result is never a stopping point.
