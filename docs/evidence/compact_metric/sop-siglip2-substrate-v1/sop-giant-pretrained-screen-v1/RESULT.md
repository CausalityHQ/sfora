# SOP pretrained SigLIP2 giant-opt/16@256 vs So400m/14@384 (zero training) — CLOSE

Pre-registered: scripts/screen_sop_giant_pretrained_v1.py (a9d0ef31). Same export code/src as the So400m screen
(c7602797). giant revision 46d7129e (model shards 0ca3ce58… / dbbb979e…), features ace05f89…, width 1536.

| arm | packed PCA128 R@1 full-TRAIN gallery | held-only |
|---|---|---|
| So400m/14@384 | 80.516 | 91.523 |
| giant-opt/16@256 | 80.140 | 91.591 |
Delta: −0.376 pp full gallery (rule needs ≥ +1.0), +0.068 held-only → CLOSE. Giant encode 1516 s for 59,551 (39 img/s incl. decode).
Substrate scaling saturates beyond So400m at these resolutions; giant@384 not funded (256 tokens vs 729 confounds, but
giant@256 already costs ~ So400m@384 encode with no gain).
