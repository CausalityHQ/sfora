 # Pass 81 SQLS Gate-1 coordinate-quorum measurement
 
 This is a training-only, corrected-corpus diagnostic for the frozen
 Subquorum Label Secrecy proposal. It used the seed-2 final In-Shop training
 embedding pack, excluded singleton identities, and split each remaining
 identity 80/20 by a fixed RNG seed (8,107). A nearest-class-centroid probe was
 fit on the 80% images and evaluated on the held-out 20%; no official test
 images were loaded.
 
 The baseline already leaks almost all training identity through small random
 coordinate subsets:
 
 | coordinates | random-subset accuracy | SD over 20 masks | top between/within |
 |---:|---:|---:|---:|
 | 32 | 0.93981 | 0.00304 | 0.95359 |
 | 64 | 0.98680 | 0.00132 | 0.98715 |
 | 128 | 0.99472 | 0.00048 | 0.99518 |
 | 256 | 0.99644 | 0.00037 | 0.99647 |
 | 512 | 0.99727 | 0 | 0.99727 |
 
 This is Gate-1 provenance for SQLS's premise: a small coordinate subset is
 already highly label-predictive on seen identities. It is not evidence that
 SQLS improves unseen retrieval. The reproducible diagnostic is
 `scripts/measure_coordinate_quorum.py`; the source pack and exact command are
 recorded in the run log/commit.
