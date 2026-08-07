# Pass 76 — NONE; CIPR dead at Gate 2

The independent review generated Complete-Invariant Positive Readout (CIPR):
replace the BN-Inception 7×7×1024 average-plus-max reduction with a spatial
G-bispectrum invariant, motivated by the corrected In-Shop positive-transfer
gap. The review returned **NONE**: no GPU run was authorized.

## Gate 1 evidence

The matched k=3 positive diagnostic reported seen/unseen values 0.788458/0.787670
(untrained, gap −0.000788), 0.866262/0.822863 (trained seed 0, −0.043399), and
0.867459/0.822963 (trained seed 1, −0.044496). A 512-D deployed artifact gave
0.826533/0.771413 (−0.055120; older, non-digest-matched artifact). Training
therefore produces the gap; the review estimates only 44.7% of within-class
contraction transfers to unseen images.

## Gate 2 failure

G-bispectral pooling is already a published deep-network mechanism (Sanborn et
al., *Bispectral Neural Networks*, ICLR 2023; Sanborn & Miolane, *Selective
G-Bispectrum and its Inversion*, NeurIPS 2024; Mathe et al., *bispectrum*, 2026).
Using it as a pooling replacement is an application, not a novel supervision
mechanism. A distillation variant collapses into relational knowledge
distillation. The compact-group premise is also a poor match to In-Shop's
pose/deformation variation. CIPR is recorded as prior art and receives no
implementation or GPU test.

The review's useful surviving measurement is that the untrained split-composition
effect is negligible; the positive-side transfer deficit is training-induced.
The next pass must still be independently generated and prior-art checked before
any GPU work.
