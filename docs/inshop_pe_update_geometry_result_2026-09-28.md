# Saved update geometry result

The actual saved100-update pair does not support a gross claim that PE's native
tail parameters barely moved relative to Large. This descriptive CPU readout
does not identify functional update size or prove an optimizer/capacity cause.
It does not authorize a learning-rate sweep or reopen the stopped recipe.

| Saved parameter group | Large update/original norm (%) | PE update/original norm (%) |
|---|---:|---:|
| Median trainable native tail block | 0.145991 | 0.212091 |
| Minimum–maximum native tail block | 0.087513–0.234528 | 0.107446–0.256155 |
| Native final norm | 0.009902 | 0.005913 |
| Native pooling | 0.281871 | 0.162432 |
| PE text-space projection | — | 0.290259 |
| Compact 128-D head | 3.177582 | 3.110950 |
| Product classifier | 1.140208 | 1.142552 |

These are float64 Frobenius ratios from authenticated original/final groups,
not optimizer-step lengths or matched functional effects. PE has six trainable
blocks versus Large's twelve; width, depth, pooling and pretrained geometry
also differ. All group fingerprints match the already qualified initial/final
training receipt. Each group changed and remained finite. The executed script
and receipt digests are preserved in [raw evidence](evidence/compact_metric/sop-siglip2-substrate-v1/pe-update-geometry-v1/).

Earlier recorded post-clip vision/head/classifier gradient norms were
Large step1 0.92914/0.36595/0.05277 and step100 0.87391/0.47697/0.09370;
PE step1 0.98541/0.16511/0.04125 and step100 0.97338/0.21285/0.08500.
These do not show a head-dominated clipping budget starving PE's encoder.
They are recorded endpoint gradients, not newly recomputed gradients or a
full optimization trajectory. Classifier-gradient magnitude and supervised
loss values do not by themselves measure generalization.

The sole CPU service `sfora-pe-update-geometry-v1`, invocation
`8ecd3972c085428fba8f1603c16805da`, exited0 in10.58 seconds with4,562,444 KiB
peak host RSS (internal7.84166s). CUDA was hidden. No images, held features,
quality scoring, training or optimizer were run. The fixture failed on the
absent implementation, then passed exact zero/scaled update arithmetic and
nonfinite rejection; Ruff/syntax and local receipt ratio replay passed.

The single new Fable consultation `575a2d205fe6486d` remains responsible for
proposing one genuinely distinct, bounded encoder-adaptation mechanism from the
new held attribution evidence. No intervention is selected from this readout.
Full production quality and matched image-to-top-k speed remain unmet.
