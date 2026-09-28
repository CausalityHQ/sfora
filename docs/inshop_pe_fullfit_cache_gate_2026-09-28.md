# Paired full-fit source acquisition

Native FP16 training mechanics PASS `084261b2` permits this separate acquisition.
Keep the full production goal; this step supplies complete fit-only initialization
for the next actual-image quality comparison and makes no retrieval claim.

Use the fixed class-disjoint official In-Shop TRAIN split from
`preflight_inshop_siglip2_unseen_gallery.split`:13,283 fit images/2,004 products,
12 singleton products included;12,599 held images/1,993 products excluded.
Require partition SHA and exact fit-row digest `f23783...89be`. Hash every fit
image and preserve official TRAIN ordinal/path/label order. No held image decoding,
PCA, classifier fitting, optimizer, augmentation or quality score.

Original FP32 Large/256 and strict native PE-Core-B16/224 weights, same as the
qualified source/mechanics route. FP16 autocast, TF32 disabled for matmul/conv,
unaugmented native RGB inputs, batch32. Both produce finite nonzero unit FP32
1024-D descriptors. Use existing atomic `export_features`; one tower resident
at a time, Large then PE, clear cuBLAS/GC/cache and require cleanup<8MiB.
Pin mechanics receipt SHA `44fd422a70078767a1cadafa36bb8911deb68ef3759d34b091856d67483ae969`
and all recorded code/source authority before acquisition. Pin the complete
preflight manifest externally through the launcher `--preflight-sha256`; this
avoids a circular hash because the manifest itself records exporter code.
Recheck the fit ordinal digest at runtime, package versions against the original
source receipt, and the known lazy import inventory (including einops Torch helpers).

CPU preflight checks full fit membership, held disjointness, every file hash,
count/singleton invariants and the existing1024-row prototype's exact path/label
mapping into fit. Freeze complete manifest and source code before GPU. Runtime
rehashes actual decoded files against manifest. Fresh first-four full-fit FP16
versus FP32 cosine≥0.999 in each arm; save values before assertion. After each
export, require all13,283 descriptors finite/unit, and all1024 original prototype
anchor cosines≥0.999 against their authenticated stored matrices. No threshold
relaxation, dtype/cap rescue or source/checkpoint replacement on failure.

One shared-lock DGX service,480s whole process/8GiB host/allocatedCUDA<10GB,
including initialization. These are budgets, not measured predictions. Store
each arm's file SHA/count/rows/normalization/anchor and precision audit, timing,
images/s and peak CUDA immediately; retain completed arm files/receipts and failure logs without a
retry. The existing atomic writer removes its unfinished `.partial` file on
an ordinary exception; it does not promise a usable partial matrix. Full-fit matrices remain on DGX; archive compact manifest/receipts in Git.

PASS permits the separately frozen augmented matched image-training feasibility
gate with TRAIN-held R@1/mAP@R and product/query uncertainty. It does not license
automatic official evaluation or prove learned quality/generalization/serving.

Focused Opus review `4d989f8782164e88` accepted the fit-only design; its external
manifest-integrity finding is fixed above. Final CPU preflight v3 passed in
4.31 s with 969,612 KiB peak RSS, native invocation
`fea0e8b9c7d442f196782e6ac1b73e39`. Frozen manifest SHA:
`4e6c886e87ec8abca45455c5790e35e252cf60694f98d043a1c55d5d21aea3ff`.
No GPU acquisition or quality measurement is implied by this CPU result.

Runnable CPU check `audit_inshop_pe_fit_features.py --startup-only` passed:
modified product metadata is rejected by the external SHA before CUDA; the
original manifest passes metadata/code checks to the deliberately hidden-CUDA
guard. Native invocation `f861e89190a641be912bf356974a13b3`, terminal exit 0.
