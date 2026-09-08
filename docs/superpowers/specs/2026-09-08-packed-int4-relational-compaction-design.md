# Packed int4 relational compaction design

## Purpose

Extend Sfora's generic relational-linear compressor with a cosine-preserving
signed-int4 storage representation. A 128-dimensional int4 vector plus one
little-endian float16 inverse code norm occupies exactly 66 bytes, matching the
released 64-dimensional int8 representation while retaining twice as many
learned directions.

The codec is generic library behavior. It does not contain dataset identities,
quality thresholds, or evaluator logic. The motivating SOP result is
exploratory; publication evidence for this new representation must come from a
fresh, non-retail domain with a frozen protocol.

## Quantization contract

Input rows are finite, nonzero, unit-normalized float32 vectors with an even
dimension of at least two. Each row is divided by its maximum absolute
coordinate, multiplied by seven, rounded half-to-even, and clamped to signed
codes `[-7, 7]`. The row scale is not persisted because a
positive scalar cancels in cosine similarity. A float16 inverse Euclidean norm
of the signed code row is stored to normalize dot products.

Two signed codes occupy one byte. The first coordinate is the low nibble and the
second coordinate is the high nibble. Nibbles use four-bit two's-complement;
the `-8` code is noncanonical and rejected. The row wire layout is packed code
bytes followed by the little-endian float16 inverse norm. No struct size or host
endianness is wire authority.

## API and validation

`PackedInt4Embeddings` owns a contiguous CPU uint8 packed-code tensor, a
contiguous CPU float16 inverse-norm tensor, and the even logical dimension.
`pack_int4_unit_embeddings` constructs it from unit rows. `restore`,
`cosine_similarity`, `to_bytes`, and `from_bytes` mirror the int8 API.

Construction and deserialization reject incorrect concrete types, devices,
shapes, odd dimensions, noncanonical `-8` nibbles, zero rows, nonfinite or
nonpositive norms, inverse norms outside a one-float16-ULP interoperability
window, and any wire length mismatch. Similarity rejects dimension mismatch and
invalid devices. Decoding uses explicit nibble operations and sign extension.

## Performance contract

Persistent bytes per vector are exactly `dimensions / 2 + 2`. The reference
scorer may unpack codes to float32 for matrix multiplication, but it must not
retain an expanded persistent representation. A future optimized nibble kernel
may replace the reference scorer only with exact differential tests and measured
latency evidence.

## Evidence boundary

Exploratory SOP evidence showed relational 128D per-vector int4 at
`MAP@R=0.407431`, `R@1=0.687597`, versus relational 64D int8 at
`0.372987/0.654177` and PCA128 int4 at `0.379712/0.651532`, all at 66 bytes.
These opened-domain numbers motivate implementation; they do not qualify the
new format. Promotion requires a preregistered non-retail evaluation, an
equal-byte PCA128-int4 control, a simple teacher-regression control, and a
projection-inclusive latency bound.
