#!/usr/bin/env python3
"""TRAIN-only pilot: SOP So400m@384 pair recipe (so400 arm, unpatched resolution) with a 512-D head instead of 128.

Pre-registered decision (384 variant): held R@1 paired product-bootstrap lower95 vs the 280/512-D run (receipt
8bc7399e, seed 179024) > 0 => ONE official read at 384 (quality point, 514 B/row, ~47 img/s). Else KILL; the
280/512-D point (official 93.480) stays the SOP leader. Also reported vs 384/128-D (05e87eaf).

Derived verbatim from train_sop_so400m_res280_dim512_pilot.py (a9b07637) except the import of the 280 px wrapper;
its original docstring follows.


Pre-registered decision: held R@1 (5,851 held vs full 59,551 TRAIN gallery, packed int8 512-D) paired product-bootstrap lower95 vs the 128-D res280 run (receipt 9e4a05db, seed 179024) > 0 => Pareto candidate (4x gallery bytes) => ONE official read at 280. Else KILL. mAP@R reported.

Mechanics (trainer 78546d9c and src/sfora/sop_compact_training.py stay byte-unmodified):
- pair.OUTPUT_WIDTH = 512: PCA head init, head, class proxies, export check and the ArcFace mask
  (`torch.arange(OUTPUT_WIDTH)[None]` = one shard holding every head coordinate, i.e. unmasked
  ArcFace over the full embedding) all read the module global at call time.
- compact_head_features / compact_training_terms: local copies whose bodies differ from the pinned
  ones only in the accepted output_dim set (adds 512) and the output_dim default (512), because the
  trainer calls both without output_dim.
- evaluate_packed: same packed int8 wire (pack_int8_unit_embeddings, D + 2 bytes/row) and the same
  exact torch full-gallery scoring math, minus the ==128 width check; the native top-10 cross-check
  is skipped because CutilePackedInt8Gallery only supports 128-D.
- Only the ArcFace arm or --member-bank works at 512: smooth_ap_float_loss / smooth_ap_packed_loss
  (deployed_code_rank.py) are pinned 128-only. The member-bank timing receipt was measured at 128.
"""
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_siglip2_substrate_pair as pair_module  # noqa: E402
from sfora.deployed_code_rank import smooth_ap_float_loss, smooth_ap_packed_loss  # noqa: E402
from sfora.joint_relational_compaction import pack_int8_unit_embeddings  # noqa: E402
from sfora.sop_compact_training import CompactTrainingArm  # noqa: E402
from sfora.unicom_training import sharded_mask_arcface_loss  # noqa: E402

pair = pair_module
OUTPUT_WIDTH = 512
pair.OUTPUT_WIDTH = OUTPUT_WIDTH


def compact_head_features(
    source: torch.Tensor, head: nn.Linear, *, output_dim: int = 512
) -> torch.Tensor:
    """Pinned sfora.sop_compact_training.compact_head_features (29a45d38) with the So400m width
    1152 added to the accepted source widths; body otherwise byte-for-byte identical."""

    if (
        type(source) is not torch.Tensor
        or source.ndim != 2
        or source.shape[1] not in (768, 1024, 1152)
        or not source.is_floating_point()
        or not bool(torch.isfinite(source).all())
        or bool((torch.linalg.vector_norm(source.float(), dim=1) == 0).any())
        or type(head) is not nn.Linear
        or head.in_features != source.shape[1]
        or output_dim not in (128, 512, 768)
        or (output_dim == 768 and source.shape[1] != 768)
        or head.out_features != output_dim
        or source.device != head.weight.device
    ):
        raise ValueError("SOP compact source geometry differs")
    with torch.autocast(device_type=source.device.type, enabled=False):
        return head(F.normalize(source.float(), dim=1))


def compact_training_terms(
    features: torch.Tensor,
    classifier: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    arm: CompactTrainingArm,
    arcface_margin: float = 0.3,
    arcface_scale: float = 64.0,
    output_dim: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ArcFace and rank terms while preserving the same classifier.

    The rank arms add a fixed-weight SmoothAP term to the control. Every
    arm receives the same 128-dimensional head output, labels, and masks.
    Float32 objective arithmetic is explicit even under backbone AMP.
    """

    if type(arm) is not CompactTrainingArm:
        raise ValueError("SOP compact training arm differs")
    if output_dim == 768 and arm is not CompactTrainingArm.ARCFACE:
        raise ValueError("full-width control only supports ArcFace")
    if not math.isfinite(arcface_margin) or not 0 <= arcface_margin < math.pi:
        raise ValueError("SOP compact ArcFace margin differs")
    if not math.isfinite(arcface_scale) or arcface_scale <= 0:
        raise ValueError("SOP compact ArcFace scale differs")
    if (
        type(features) is not torch.Tensor
        or features.ndim != 2
        or output_dim not in (128, 512, 768)
        or features.shape[1] != output_dim
        or not features.is_floating_point()
        or not isinstance(classifier, torch.Tensor)
        or classifier.dtype != torch.float32
        or classifier.ndim != 2
        or classifier.shape[1] != output_dim
        or type(labels) is not torch.Tensor
        or labels.dtype != torch.int64
        or labels.shape != (features.shape[0],)
        or type(masks) is not torch.Tensor
        or masks.dtype != torch.int64
        or masks.shape[1] != output_dim
        or features.device != classifier.device
        or features.device != labels.device
        or features.device != masks.device
    ):
        raise ValueError("SOP compact training tensors differ")
    if min(Counter(labels.tolist()).values(), default=0) < 2:
        raise ValueError("SOP compact positive inventory differs")
    with torch.autocast(device_type=features.device.type, enabled=False):
        float_features = features.float()
        control = sharded_mask_arcface_loss(
            float_features,
            classifier,
            labels,
            masks,
            margin=arcface_margin,
            scale=arcface_scale,
        )
        if arm is CompactTrainingArm.ARCFACE:
            return control, control.new_zeros(())
        label_list = labels.tolist()
        rank = (
            smooth_ap_float_loss(float_features, label_list)
            if arm is CompactTrainingArm.FLOAT_RANK
            else smooth_ap_packed_loss(float_features, label_list)
        )
        return control, rank


@torch.inference_mode()  # type: ignore[untyped-decorator]
def score_packed_full_gallery(
    codes: torch.Tensor,
    inverse_norms: torch.Tensor,
    labels: torch.Tensor,
    held_rows: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """pair.score_packed_full_gallery with `codes.shape[1] != 128` -> `!= OUTPUT_WIDTH`; body otherwise identical."""
    if (
        codes.ndim != 2
        or codes.shape[1] != OUTPUT_WIDTH
        or inverse_norms.shape != (len(codes),)
        or labels.shape != (len(codes),)
        or held_rows.ndim != 1
        or len(held_rows) < 1
    ):
        raise ValueError("SOP SigLIP2 packed scoring geometry differs")
    code = codes.float().to(device)
    inverse = inverse_norms.float().to(device)
    gallery_labels = labels.to(device)
    query_rows = held_rows.to(device)
    relevant = torch.bincount(gallery_labels)[gallery_labels[query_rows]] - 1
    if int(relevant.min()) < 1:
        raise ValueError("SOP SigLIP2 packed scoring positive inventory differs")
    width = int(relevant.max())
    ranks = torch.arange(1, width + 1, device=device)
    per_r1: list[float] = []
    per_ap: list[float] = []
    for block in query_rows.split(64):
        scores = (code[block] @ code.T) * inverse[block, None] * inverse[None, :]
        scores[torch.arange(len(block), device=device), block] = -torch.inf
        query_labels = gallery_labels[block]
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :width]
        matches = gallery_labels[ranked] == query_labels[:, None]
        block_relevant = relevant[len(per_r1) : len(per_r1) + len(block)]
        precision = matches.cumsum(dim=1) / ranks[None, :]
        ap = (precision * matches * (ranks[None, :] <= block_relevant[:, None])).sum(
            dim=1
        ) / block_relevant
        per_r1.extend(float(value) for value in matches[:, 0].cpu().tolist())
        per_ap.extend(float(value) for value in ap.cpu().tolist())
    return {
        "recall_at_1": float(np.mean(per_r1)),
        "map_at_r": float(np.mean(per_ap)),
        "per_query_r1": per_r1,
        "per_query_ap": per_ap,
    }


def evaluate_packed(
    values: torch.Tensor,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    held_rows: np.ndarray,
    native_library: Path,
    *,
    device: torch.device = torch.device("cuda"),
) -> dict[str, Any]:
    packed = pack_int8_unit_embeddings(values)
    if len(fit_rows) != 53_700 or len(held_rows) != 5_851:
        raise ValueError("SOP SigLIP2 packed evaluation partition differs")
    scored = score_packed_full_gallery(
        packed.codes.float(),
        packed.inverse_norms,
        torch.from_numpy(labels.copy()),
        torch.from_numpy(held_rows.copy()),
        device=device,
    )
    wire_bytes = len(packed.to_bytes())
    if wire_bytes % len(packed.codes) or wire_bytes // len(packed.codes) != packed.bytes_per_vector:
        raise ValueError("SOP SigLIP2 packed wire width differs")
    return {
        "recall_at_1": scored["recall_at_1"],
        "map_at_r": scored["map_at_r"],
        "per_query_r1": scored["per_query_r1"],
        "per_query_ap": scored["per_query_ap"],
        "native_per_query_r1_equal": None,
        "native_top10_exact": None,
        "native_top10_max_score_abs_delta": None,
        "native_skipped_reason": "CutilePackedInt8Gallery supports only 128-D; 512-D scored exactly in torch",
        "gallery_wire_bytes_per_row": wire_bytes // len(packed.codes),
        # The pinned receipt records neither the wrapper nor the width; carry both in `quality`.
        "output_width": OUTPUT_WIDTH,
        "wrapper_sha256": pair.sha256(Path(__file__)),
    }


pair.compact_head_features = compact_head_features
pair.compact_training_terms = compact_training_terms
pair.score_packed_full_gallery = score_packed_full_gallery
pair.evaluate_packed = evaluate_packed

if __name__ == "__main__":
    assert "--substrate" in sys.argv and sys.argv[sys.argv.index("--substrate") + 1] == "so400"
    # smooth_ap_float_loss / smooth_ap_packed_loss are pinned 128-only; fail before loading the model.
    assert "--member-bank" in sys.argv or sys.argv[sys.argv.index("--arm") + 1] == "arcface"
    pair.main()
