# Runtime-bound compilation reuse gate

The existing candidate passed its exactness gate. This gate asks whether
real compilation follows the proposed five scalar divisibility classes,
despite the other fields in cuTile's key. It makes no full-call speed claim.

Pinned installed cuTile0.1.1 `tile_kernel.rs` SHA
`e66d90f687df12e660159e14d58700ec7e2e3c18d651fa2dd55cbb14ac841b78`
logs `CUTILE_JIT_TIMING` only on cache misses after compilation/loading.
The key includes generics, tensor strides, specialization/scalar hints,
optional constant grid and compile options. Cache is thread-local/per-device;
the assembler entrypoint writes fresh temporary bytecode/cubin files.

Run ONE fresh process using the unchanged verified binary
`5539fee25b6602dd0402f82b656be57ca85c666eeb0f635b2ed1428263e9d3df`,
with `CUTILE_JIT_TIMING=1`, filtering exactly the existing device-boundary
test. It makes16 searches: batches1/32, each with gallery sizes
10/127/128/129/132/136/257/4097. No new math, fixtures, API, thread settings,
or kernel parameters. Preserve numerical score/ordinal/tie equality.
DGX only, shared GPU lock,60s total external deadline and8GiB host cap.
Run the smallest stdlib timing-log parser check before GPU; source/binary/
assembler hashes must match the prior verification manifest.

Advance this reuse claim only if the test passes, every timing entry is
well formed and finite/nonnegative with unique module/function/key, and
`score_block_topk` compiles at most10 times: five row divisibility classes
per batch. More entries reject the proposed complete-key reuse bound; do not
retune or relax it. Report all other kernel counts and summed compilation
stages. This does not prove the cause of any extra misses or close every
future cache design. No historical compile timing is a matched control.

A pass permits separate matched original/candidate search measurements
and compiler-key inspection; it is not a measured improvement over baseline,
public promotion, official quality or full image-to-top-k latency. A failure
keeps the candidate unpromoted despite its exactness pass.
