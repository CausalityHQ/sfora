#!/usr/bin/env python3
"""TRAIN-only pilot: SOP So400m pair recipe trained AND evaluated at 280 px (400 tokens vs 729 at 384).

Runs train_sop_siglip2_substrate_pair.py (78546d9c bytes, unmodified) with one added arm `so400_280`:
the so400 arm with crop 280 and position embeddings bilinearly interpolated in every vision forward.
Deviation (documented, not hidden): head PCA init and member-bank seeds come from the existing 384 px
pretrained So400m features (same as the 384 arm); only the training/eval pixels are 280.

Pre-registered decision (committed before the run):
  candidate iff held R@1 (5,851 held vs full 59,551 TRAIN gallery, packed int8) paired product-bootstrap
  lower95 vs the Large pair control (receipt d7f7383f) > 0. Candidate => ONE official read at 280.
  Also reported, not gating: vs So400m@384 tested at 280 (92.360) and at 336 (93.796), GPU img/s 87.9 @280.
"""
import sys
from functools import partialmethod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_siglip2_substrate_pair as pair  # noqa: E402
from transformers import AutoImageProcessor  # noqa: E402
from transformers.models.siglip import modeling_siglip  # noqa: E402

CROP = 280
pair.ARMS["so400_280"] = {**pair.ARMS["so400"], "crop": CROP}
modeling_siglip.SiglipVisionEmbeddings.forward = partialmethod(
    modeling_siglip.SiglipVisionEmbeddings.forward, interpolate_pos_encoding=True)
_from_pretrained = AutoImageProcessor.from_pretrained.__func__


def _sized(cls, *args, **kwargs):
    processor = _from_pretrained(cls, *args, **kwargs)
    processor.size = {"height": CROP, "width": CROP}
    return processor


AutoImageProcessor.from_pretrained = classmethod(_sized)

if __name__ == "__main__":
    assert "--substrate" in sys.argv and sys.argv[sys.argv.index("--substrate") + 1] == "so400_280"
    pair.main()
