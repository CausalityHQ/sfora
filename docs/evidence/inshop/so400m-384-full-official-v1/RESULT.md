# In-Shop official: So400m@384, 512-D head, full TRAIN, one read

The model meets the In-Shop target (pre-registration 72e6fd19; audit 4482f8469eb44fe2 GO).

| Model | Official R@1 | mAP@R | Gallery bytes/row |
|---|---|---|---|
| **So400m@384 512-D (this run)** | **96.807** | **83.78** | 514 |
| Previous incumbent: Large/256 128-D, Full2000 | 96.307 | 81.88 | 130 |
| So400m@256 128-D, native256 full | 96.244 | 82.60 | 130 |
| Dated UNICOM reference | 96.7 | — | — |

- **Data:** 14,218 queries vs 12,612 gallery, packed int8, exact torch scores. Trained on 25,870 TRAIN images from 3,985 products. Query/gallery was read once, after training.
- **Cost:** training took 2,968 s (1,000 updates) with a 37.5 GB CUDA peak. Serving runs at ~42 img/s end to end at 384 (SOP P1 parity), or faster with fp16-native.
- **Not tested:** no seed replicate, and no paired interval against the incumbent. The pre-registered rule was a threshold read.
- **Artifact hashes:** checkpoint and official codes are in `artifacts.sha256`; the files stay on the DGX.
