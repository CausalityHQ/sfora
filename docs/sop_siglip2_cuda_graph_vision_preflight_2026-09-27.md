# SOP native vision CUDA graph feasibility, 27 September 2026

Test PyTorch's native CUDA graph replay around the existing seed-179024
true-freeze SigLIP2 Large/256 FP16 vision forward. The model, image processor,
input pixels, weights, head and native packed scorer remain unchanged. This is
an isolated vision-stage screen, not an end-to-end speed or quality claim.

Use the authenticated first 32 SOP official **TRAIN** images from the pinned
archive, with the same processor and checkpoint as the public stage profile.
Capture one static graph for batch 1 and one for batch 32. For each input,
copy the already-processed GPU pixels into the static graph input, replay,
and require the pooled FP16 tensor to be bitwise equal to an eager forward.
Measure 20 alternating ABBA/BAAB blocks of 10 eager and 10 graph calls per
position after warmup, including the static-input copy in graph timings.
Synchronize CUDA after each call. Both arms see the same preprocessed pixels;
the processor and JPEG decoder are outside this feasibility timing.

Advance only if every output is bitwise equal, batch-1 graph p50 is at most
**90%** of eager p50, batch-32 graph p95 is at most **105%** of eager p95, and
graph peak allocated CUDA memory is below **3 GB**. A pass permits a separate
exact packed top-10 and varied-image public p50/p95/p99 screen with a
single-threaded serving wrapper and explicit capture lifecycle. Failure stops
before production code. Neither case changes training throughput or retrieval
quality.
