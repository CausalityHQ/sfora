# SOP public query stage profile, 27 September 2026

Profile the selected seed-179024 true-freeze production `Siglip2CompactIndex`
against its 59,551-row official **TRAIN** gallery on DGX Spark GB10. Use the
same first pinned SOP TRAIN image as the completed batch-1 p99 gate. Keep the
authenticated checkpoint, FP16 mode, processor, 130-byte wire, native exact
scorer and runtime. Warm five calls, then time 100 complete calls with separate
CUDA synchronizations after transfer, vision, head/packing and native search.
Require the staged output to match the public API's top-10 ordinals and scores
on every call. Record per-stage p50/p95, hardware, source/image/checkpoint
hashes and peak allocated CUDA memory.

This is a stage attribution diagnostic. Synchronizations change its latency,
so its sum and 100-call tail are **not** a substitute for the existing 10,000
call public p99 result. If one non-vision stage consumes at least 25% of the
staged whole-call median, inspect one exact-output production optimization of
that stage. Otherwise target vision execution. Any candidate must pass exact
packed top-10 parity and a separate paired public p50/p95/p99 latency gate
before production promotion. No retrieval quality or SOTA conclusion follows
from this profile.
