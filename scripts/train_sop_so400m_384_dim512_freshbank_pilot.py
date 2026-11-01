#!/usr/bin/env python3
"""TRAIN-only pilot (proposals v3 #2): SOP So400m@384/512 recipe with a FRESH member bank.

Identical to train_sop_so400m_384_dim512_pilot.py except that, after the optimizer steps 150, 350, 600 and 850, every
fit row of the detached SmoothAP member bank is re-encoded with the current model (the trainer's own export_all:
eval mode, no augmentation, compact head features, L2-normalised as at bank init) instead of only being refreshed from
its last augmented batch view (~1.2 refreshes per row per run, i.e. rows are ~500 updates stale on average).
Logged per refresh: median / 10th-percentile cosine between the stale and fresh row (bank drift).
Pre-registered: held R@1 (5,851 vs 59,551, packed int8) paired product-bootstrap lower95 > 0 vs receipt 1aaf30ed
(same seed 179024) => second seed + 2-seed ensemble screen vs the incumbent ensemble; else KILL. Mechanism check: if
median drift < 0.01 at every refresh, report "mechanism absent" regardless of the headline.
"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_so400m_384_dim512_pilot as base  # noqa: E402

pair = base.pair
REFRESH_AT = (5,) if os.environ.get("FRESHBANK_SMOKE") else (150, 350, 600, 850)  # smoke exercises the path
_orig_rows = pair.member_bank_refresh_rows
_state = {"step": 0}


def _fresh_rows(batch_ordinals):
    _state["step"] += 1
    if _state["step"] in REFRESH_AT:
        f = sys._getframe(1).f_locals  # pinned main(): called right after the optimizer step
        vision, head, bank = f["vision"], f["head"], f["bank"]
        fit_paths = tuple(f["paths"][r] for r in f["fit_rows"])
        fit_labels = tuple(f["class_index"][label] for label in f["fit_labels"])
        if bank.shape[0] != len(fit_paths):
            raise ValueError("fresh bank: bank rows differ from fit rows")
        with torch.no_grad():
            fresh = F.normalize(pair.export_all(vision, head, fit_paths, fit_labels, f["processor"],
                                                workers=8, batch_size=64).float().cuda(), dim=1)
            cos = (F.normalize(bank.float(), dim=1) * fresh).sum(1)
            bank.copy_(fresh.to(bank.dtype))
        vision.train()
        head.train()
        q = torch.quantile(cos.float().cpu(), torch.tensor([0.1, 0.5]))
        print("FRESHBANK " + json.dumps({"step": _state["step"], "cos_p10": float(q[0]), "cos_median": float(q[1])}),
              flush=True)
    return _orig_rows(batch_ordinals)


pair.member_bank_refresh_rows = _fresh_rows

if __name__ == "__main__":
    assert sys.argv[sys.argv.index("--substrate") + 1] == "so400"
    pair.main()
