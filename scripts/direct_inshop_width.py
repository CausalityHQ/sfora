"""Strict authority and private native-FP16 reload checks for direct width smoke."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from score_inshop_crop_view_pair import sha256
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_compact_training import compact_head_features


def validate_direct_width(args):
    qualification = getattr(args, "direct_width_qualification", None)
    authority = qualification or args.direct_width_receipt
    expected_sha = "5334209bf78c70b08e0dd20bd55572e57130f2b31ef5570ebdc2ce5c535b2333" if qualification else "1ac62b0edda6dd93bc808a8d2bb81fef7c89cc1c4eaef816d276c4fecb2a555f"
    blocked = ("vision_init_checkpoint", "vision_init_sha256", "wide_head_smoke_receipt",
               "wide_head_qualification", "source_centroid_smoke", "source_centroid_receipt",
               "teacher_transfer", "teacher_transfer_smoke_receipt", "teacher_model_snapshot",
               "teacher_checkpoint", "source_main_smoke", "source_main_receipt",
               "source_main_qualification", "centroid_pca_smoke", "centroid_pca_receipt",
               "centroid_pca_qualification", "centroid_pca_confirmation")
    if (args.arm != "freeze_emb" or args.seed != 179024 or args.updates != (100 if qualification else 17)
        or args.training_width not in (128, 256) or args.freeze_first_blocks != 12
        or args.vision_lr != 1e-5 or args.half_fit_products or args.tail_blocks_to_drop
        or any(getattr(args, name) is not None for name in blocked)
        or (qualification is not None and args.direct_width_receipt is not None)
        or authority is None or sha256(authority) != expected_sha):
        raise ValueError("direct-width mechanics authority differs")
    receipt = json.loads(authority.read_text())
    expected_decision = "GO_FREEZE_REAL_100_GATE" if qualification else "GO_REVIEW_ONLY"
    if receipt["decision"] != expected_decision or not all(receipt["criteria"].values()):
        raise ValueError("direct-width requires the frozen positive cached gate")


@torch.inference_mode()
def direct_checkpoint_checks(vision, head, processor, checkpoint, image_paths, destination):
    """Private profile only: FP16 vision parameters, FP32 head, exact batch32."""
    from transformers import SiglipVisionModel

    if len(image_paths) != 64 or head.out_features not in (128, 256):
        raise ValueError("direct-width reload fixture differs")
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    for name, module in (("vision", vision), ("head", head)):
        if set(module.state_dict()) != set(saved[name]) or any(
            not torch.equal(value.detach().cpu(), saved[name][key])
            for key, value in module.state_dict().items()
        ):
            raise ValueError("direct-width saved FP32 tensors differ")
    pixels = []
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    for start in (0, 32):
        pixels.append(processor(images=images[start:start+32], return_tensors="pt")["pixel_values"].half())
    torch.backends.cuda.matmul.allow_tf32 = False
    vision.half().eval()
    head.eval()

    def packed(model, projection):
        parts = []
        for batch in pixels:
            with torch.autocast("cuda", enabled=False):
                pooled = model(pixel_values=batch.cuda()).pooler_output
            parts.append(F.normalize(compact_head_features(pooled, projection, output_dim=head.out_features), dim=1).cpu())
        return pack_int8_unit_embeddings(torch.cat(parts))

    expected = packed(vision, head)
    restored = SiglipVisionModel(vision.config).cuda()
    restored.load_state_dict(saved["vision"], strict=True)
    projection = nn.Linear(1024, head.out_features).cuda()
    projection.load_state_dict(saved["head"], strict=True)
    actual = packed(restored.half().eval(), projection.eval())
    checks = {"exact_codes":torch.equal(expected.codes, actual.codes),
              "exact_inverse_norms":torch.equal(expected.inverse_norms, actual.inverse_norms)}
    if not all(checks.values()):
        raise ValueError("direct-width private native-FP16 reload differs")
    fixture = destination / "native_fp16_fit_fixture.pt"
    torch.save({"pixels":torch.cat(pixels), "codes":expected.codes, "inverse_norms":expected.inverse_norms}, fixture)
    return {"checks":checks, "profile":"private native-FP16 vision / FP32 head", "batch_size":32,
            "images":64, "wire_bytes_per_row":head.out_features+2, "fixture_sha256":sha256(fixture),
            "checkpoint_sha256":sha256(checkpoint), "image_sha256":[sha256(p) for p in image_paths]}
