# Matched synthetic native comparison passed

The frozen gate passes on DGX Spark. Eight fresh A–B–B–A processes performed
456 searches over synthetic59551-row/128-dimension galleries, top10, batches1/32.
Every ordinal and float32 score bit matched the independent scalar oracle,
including distinct queries, cross-group/final-block winners and all-tie galleries.
All448 main outputs are retained; the eight all-tie outputs were checked at
execution. There were no JIT misses during warm or timed calls. These are
synthetic packed fixtures with SOP TRAIN-sized geometry, not SOP image quality.

| Batch | Original hot p50/p95 ms | Candidate hot p50/p95 ms | Candidate/original | Gate |
|---|---|---|---|---|
|1|0.168968 /0.173277|0.170087 /0.172377|1.006623 /0.994805|PASS|
|32|0.235920 /0.248020|0.235119 /0.241301|0.996609 /0.972909|PASS|

Each arm/batch has100 samples pooled from two processes. Frozen tolerances
were1.05x p50 and1.10x p95. These descriptive quantiles support the native
regression guard; no product p99, uncertainty-qualified speed or QPS claim.

| Batch | Original first synchronous search ms, both processes | Candidate first search ms, both processes |
|---|---|---|
|1|876.899 /897.443|656.710 /677.993|
|32|1022.427 /1012.095|810.916 /816.485|

Each original process logged five native JIT misses: two fill, one score and
two merge kernels. Each candidate logged four: the two merge levels shared
one compiled key. The removed merge specialization explains the observed
startup reduction at this geometry. Both arms still have one score key here.
These are fresh-process timings using existing toolchains and driver state;
hardware-cold startup and cross-thread cache reuse remain unmeasured.
Constructor times ranged262.581–303.511ms for original and263.077–267.642ms
for candidate. Constructor, first search and hot search are separate intervals.

Original GPU invocation2926379a245d46598aae73b32c1975f1 exited0 normally in
18.17s GNU wall,477732KiB maximum RSS. Its journal reports38.822s CPU and
384.5M cgroup memory peak; CUDA peak memory was not measured. The600s/8GiB
controller and75s child limits came from original launch arguments, with a
shared GPU lock. All jobs are terminal; no training or encoder ran.

The first shared Cargo build falsely reused the original output for the
candidate. CPU hash verification caught this before GPU execution. Targeted
crate invalidation produced an actual candidate compile and distinct library
hashes; CPU build wall totals1.82+1.77=3.59s, within the240s ceiling.
The failed attempt is preserved, not treated as qualified candidate evidence.

The [archive receipt](evidence/compact_metric/sop-siglip2-substrate-v1/cutile-native-compare-v1/receipt.json)
includes source/library hashes, original invocations, raw timing/output reports,
native JIT logs, build controllers and resource reports. Its stdlib `verify.py`
independently replayed hashes,456 checks, process order, scalar output equality,
native miss counts, pooled quantiles and frozen gates successfully.

The preserved Rust candidate is untouched, unstaged and unpromoted. Next release
gates are shared FFI size validation before raw slices, sequential two-thread
same-handle coverage, full assurance and explicit review reconciliation. The
official SOP/In-Shop quality numbers are unchanged, and the joint quality and
full image-to-top-k speed goal remains unmet.
