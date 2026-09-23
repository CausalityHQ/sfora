# RC5 packed-search pointer-dispatch design

## Evidence and purpose

The retained public Python scorer validates packed arrays, then uses four
NumPy `ndpointer` argument conversions for each native search call. A frozen
one-million-row, batch-32 probe with GC disabled after warmup left 200 tracked
`ctypes.c_void_p` objects and 200 dictionaries after 50 calls. The minimal
GC trace saw a collection at timed sample 12, and a matched Nsight run
captured a 15.673 ms generation-2 GC inside an RC3 sample-12 call while CUDA
time stayed normal. The rejected RC4 merge candidate and the RC5 output-fill
candidate also had untraced sample-12 tails. Their precise causes remain
unproved, but retained ctypes objects are a concrete Python-boundary cause
worth a bounded intervention.

## Intervention and safety

Change only the search-call argument bridge in `src/sfora/cutile_int8.py`.
Retain `_packed_inputs` validation, `ctypes.CDLL`, the C ABI, lifecycle lock,
native shape batching, the gallery creation path, score arithmetic, and
return types. Bind the four search array arguments as `ctypes.c_void_p` and
pass integer addresses from `ndarray.__array_interface__["data"][0]`. Do
not use `ndarray.ctypes`, which is the suspected source of retained wrapper
objects. Input arrays remain live in local variables until the synchronous
native call returns. Output arrays are internally allocated with known dtype,
shape, and contiguity; invalid caller arrays are rejected before any pointer
is taken. No user-supplied address crosses the boundary.

## Causal and release gates

First run the same 50-search, GC-disabled object probe on the unchanged
RC3 native library with the old and new Python modules. Require the new
module to reduce the combined new tracked `ctypes.c_void_p` and dictionary
objects by at least 80% (baseline 400) without changing the public result.
If it does not, revert this intervention before further benchmarking.
Then run focused Python API and native exactness tests, including invalid
arrays, arbitrary batching, close/search concurrency, and exact f32 bits and
ordered top-10 at 1,000,000 and 1,000,003 rows for batches 1 and 32.

If the causal gate passes, compare the revised API with the original API on
the pinned RC3 library to isolate dispatch effects. Only then combine it
with the already-built 512-width native candidate and run two fresh paired
50-call public replays in opposite order against original RC3. Preserve every
sample. Promote only if each pair achieves at least 20% batch-32 p99 gain,
at most 5% batch-1 p99 regression, exact score bits and ordinals, RSS below
2 GiB, and tracked CUDA pool peak below 200 MB. Report p50/p95/p99 and
throughput for both batches. A Python process can still collect garbage due
to unrelated work; this gate demonstrates bounded behavior on the frozen
one-million-row fixture, not a universal p99 guarantee.

If the gate fails, revert the production API edit, preserve the negative
raw record, and move to a distinct measured architecture rather than tuning
the gate. Frozen Pet/In-Shop quality receipts are not retuned. A passing
terminal candidate receives clean-install/package gates and a read-only
Claude Opus 5.5 plus GPT-6 Astra review before any release decision.
