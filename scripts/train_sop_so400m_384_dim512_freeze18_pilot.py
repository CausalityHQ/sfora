#!/usr/bin/env python3
"""TRAIN-only efficiency pilot: SOP So400m@384/512 recipe with 18 (not 13) frozen lower blocks of 27, seed 179024.

Why: the served 2-seed ensemble shares the frozen trunk bitwise (shared-trunk FP8 serving, official 94.042 @ 63.7 img/s,
GPU-bound). Encode cost per ensemble = (F + 2*(27-F))/27 towers: F=13 -> 1.52, F=18 -> 1.33 (~14 % faster). The
freeze-boundary curve was flat (9 vs 13: -0.22 ns; In-Shop flat), so quality may hold.
Pre-registered NON-INFERIORITY vs receipt 1aaf30ed (13 frozen, same seed): held R@1 paired delta point >= -0.10 and
lower95 >= -0.30, and dmAP@R point >= -0.30 => train seed 179025 at 18 frozen and screen the 18-shared ensemble.
Else KILL.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_so400m_384_dim512_pilot as base  # noqa: E402

base.pair.ARMS["so400"]["frozen_blocks"] = 18

if __name__ == "__main__":
    assert sys.argv[sys.argv.index("--substrate") + 1] == "so400"
    base.pair.main()
