# Finite uint8 normalization CPU gate

Question: can exact table lookup remove enough CPU normalization overhead to
help public image-to-top-k serving? This is distinct from the closed spawned
worker: no IPC or thread-count change. Keep resizing, grouping, conversion to
RGB and tensor shape/strides unchanged; replace only uint8-to-float32
normalization with a 256-entry table derived from the pinned original transform.
Use existing NumPy, torchvision and torch; no product API change yet.

The current pinned Transformers backend fuses rescaling into normalization
and keeps resized pixels uint8. Require exhaustive equality of the table to
both direct and processor normalization for all256 values, plus B1/B32
contiguous/channels-last and RGB/L/P/RGBA fixtures. Reject other dtypes.
Exact public strides and pixels are mandatory. An initial fixture setup call
passed lists to a cached method requiring tuples; corrected before timing.
The fixture now passes; this is not a performance result.

Freeze one CPU screen before outcomes: same32 evenly spaced official SOP
TRAIN archive rows as the previous worker screen, source/config/archive/image
hash authority, parent20 threads, no CUDA. Decode before timing. Include RGB
conversion, unchanged processor work, output allocation and table lookup;
exclude decoder, encoder, packing and search. Warm3 calls per arm/size;5
ABBA/BAAB blocks with2 calls per position gives20 calls per arm at B1 and B32.
Check every result outside timing and keep raw nanoseconds. Restore the
previous thread count on exit. Cap120s/8GiB with process-group termination.

Advance only if pixels/strides remain exact, candidate p95<=0.8×baseline at
BOTH sizes, and median savings>=0.81ms for B1 and>=14.21ms for B32. The
savings floors equal5% of the earlier full-call medians16.203/284.217ms;
they are a plausibility screen, not an additive latency guarantee. A negative
screen closes this configuration before GPU/API work, without table-layout,
thread-count, lookup-library or threshold tuning. A pass permits Opus/Astra
review and one separately frozen exact packed/native-output and paired
public full-pipeline pilot. Production p99 still needs10,000 varied paired
calls and uncertainty. No quality or official split scoring occurs here.

Keep the complete production quality-and-speed goal active; preserve the
protected Rust work. No worker service, model recipe or training is changed.
