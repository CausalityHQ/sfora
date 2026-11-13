# SOP pretrained backbone census (proposals v3 #3, stage 1) — in progress
scripts/screen_sop_backbone_census_v1.py (pre-registered; ADVANCE iff full-gallery packed PCA-128 R@1 >= So400m@384 + 0.5 pp).
Attempt 1 (2026-11-01): So400m@384 80.52 / holdout 91.52; AIMv2-H/14@448 62.47 / 76.94 (−18.05 pp) → CLOSE.
The job died while loading MetaCLIP-2 giant (DGX rebooted afterwards; no error captured). Attempt 2 (2026-11-13) reruns
the baseline + MetaCLIP-2 giant + DFN5B H/14@378 only (staged copy with AIMv2 removed from TOWERS; rule unchanged).
