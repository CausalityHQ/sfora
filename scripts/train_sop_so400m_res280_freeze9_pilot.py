#!/usr/bin/env python3
"""TRAIN-only pilot: SOP So400m@280 recipe with 9 (not 13) frozen lower blocks of 27.

Identical to train_sop_so400m_res280_pilot.py (d0568eee) except ARMS["so400_280"]["frozen_blocks"] = 9.
Why: the 13/27 boundary copied the Large 12/24 ratio untuned; So400m's extra capacity may adapt better.
Pre-registered decision: held R@1 (5,851 vs full 59,551 TRAIN gallery, packed int8) paired product-bootstrap
lower95 vs the 13-frozen res280 run (receipt 9e4a05db, same seed 179024) > 0 => ONE official read at 280.
Else KILL (mAP@R and train cost reported).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_so400m_res280_pilot as base  # noqa: E402

base.pair.ARMS["so400_280"]["frozen_blocks"] = 9

if __name__ == "__main__":
    assert sys.argv[sys.argv.index("--substrate") + 1] == "so400_280"
    base.pair.main()
