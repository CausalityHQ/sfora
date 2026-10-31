# SOP add-a-member ensemble screen (TRAIN-held) — KILL
Pre-registered e658a480 (`scripts/screen_sop_ensemble_member_v2.py`). Incumbent 2-seed (179024+179025) held R@1 95.266 / mAP 84.475 (reproduces v1).
| arm | R@1 | mAP@R | dR@1 [95%] | dmAP [95%] |
|---|---|---|---|---|
| primary: + detail-crop (923c69eb) | 95.163 | 84.649 | −0.10 [−0.29, +0.08] | +0.17 [+0.02, +0.33] |
| descriptive: + 179024@2000 | 95.197 | 84.658 | −0.07 [−0.28, +0.14] | +0.18 [+0.01, +0.36] |
A third member saturates R@1 (mAP-only gain at +50 % encode). No official read. Note: driver stdout went to the
journal (systemd-run without --pipe); JSON recovered from `journalctl --user -u run-u12.service`.
