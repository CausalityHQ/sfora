#!/usr/bin/env python3
"""TRAIN-held screen: offline gallery flip-averaging for the So400m@280 SOP checkpoint (query cost unchanged).

Pre-registered (committed before any flipped number exists):
  model : res280 checkpoint (receipt 9e4a05db...), encode exactly as its trainer (fp16 autocast, 280 px, interpolated
          position embeddings). flip = horizontal flip of the PIL image before the processor.
  split : SOP TRAIN, product-disjoint seed 179019 fit 0.9; 5,851 held queries vs full 59,551 gallery, self excluded.
  arms  : plain  = query plain, gallery plain (must reproduce receipt per-query R@1 exactly, else INVALID)
          gflip  = query plain, gallery = l2norm(plain + flip)   [online serving cost unchanged]
          sflip  = both sides flip-averaged                       [2x query encode; reported only]
  scoring: 128-D signed-int8 packing of each side (pack_int8_unit_embeddings), cosine via codes x inverse norms.
  decision: gflip ADVANCE iff held R@1 (gflip - plain) paired product-bootstrap lower95 > 0 -> ONE official read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import ImageOps
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_so400m_res280_pilot  # noqa: E402,F401  (registers so400_280 arm, 280 processor, pos interpolation)
import train_sop_siglip2_substrate_pair as pair  # noqa: E402
from sfora.joint_relational_compaction import pack_int8_unit_embeddings  # noqa: E402

RECEIPT = "9e4a05db43879acffce11dcee99fc768bde08106f36d23311cf0c117c99b7be9"


class Flipped(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, label = self.base[i]
        return ImageOps.mirror(image), label


@torch.inference_mode()
def score(query: torch.Tensor, gallery: torch.Tensor, labels: np.ndarray, held: np.ndarray) -> np.ndarray:
    q, g = pack_int8_unit_embeddings(query), pack_int8_unit_embeddings(gallery)
    gc, gi = g.codes.float().cuda(), g.inverse_norms.float().cuda()
    qc, qi = q.codes.float().cuda(), q.inverse_norms.float().cuda()
    lab = torch.from_numpy(labels).cuda()
    out = []
    for block in torch.from_numpy(held).cuda().split(256):
        s = (qc[block] @ gc.T) * qi[block, None] * gi[None, :]
        s[torch.arange(len(block), device="cuda"), block] = -torch.inf
        out.append((lab[s.argmax(1)] == lab[block]).cpu())
    return torch.cat(out).numpy().astype(float)


def product_interval(d, products):
    _, inv = np.unique(products, return_inverse=True)
    s, n, k, rng = np.bincount(inv, d), np.bincount(inv), inv.max() + 1, np.random.default_rng(179019)
    draws = [(c * s).sum() / (c * n).sum() for c in (np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(5000))]
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("run-dir", "model-snapshot", "archive", "dataset-root", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists() and pair.sha256(a.run_dir / "receipt.json") == RECEIPT
    receipt = json.loads((a.run_dir / "receipt.json").read_text())
    assert pair.sha256(a.run_dir / "checkpoint.pt") == receipt["checkpoint_sha256"]
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(a.model_snapshot, local_files_only=True, backend="torchvision")
    assert processor.size == {"height": 280, "width": 280}
    full = AutoModel.from_pretrained(a.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16)
    vision = full.vision_model.float().cuda().eval()
    del full
    head = nn.Linear(1152, 128).cuda().eval()
    ckpt = torch.load(a.run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    vision.load_state_dict(ckpt["vision"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    with np.load(a.archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"])
    paths = pair.paths_from_archive(a.dataset_root, relatives)
    part = pair.deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=pair.FIT_SEED)
    held = np.asarray(part.validation_row_indexes, dtype=np.int64)
    assert len(held) == 5_851
    plain = pair.export_all(vision, head, paths, tuple(map(int, labels)), processor, workers=4, batch_size=64)
    original = pair.ImageRows
    pair.ImageRows = lambda *args, **kw: Flipped(original(*args, **kw))
    flip = pair.export_all(vision, head, paths, tuple(map(int, labels)), processor, workers=4, batch_size=64)
    pair.ImageRows = original
    avg = torch.nn.functional.normalize(plain + flip, dim=1)
    r_plain = score(plain, plain, labels, held)
    result = {"schema": "sop-so400m-gallery-flip-screen-v1", "official_read": False}
    if not np.array_equal(r_plain, np.asarray(receipt["quality"]["per_query_r1"], float)):
        result["decision"] = "INVALID: plain does not reproduce receipt per-query R@1"
    else:
        arms = {"plain": r_plain, "gflip": score(plain, avg, labels, held), "sflip": score(avg, avg, labels, held)}
        for name, r in arms.items():
            result[name] = {"recall_at_1": float(r.mean()),
                            "d_vs_plain": float((r - r_plain).mean()),
                            "d_vs_plain_product_95": product_interval(r - r_plain, labels[held])}
        result["decision"] = "ADVANCE gflip" if result["gflip"]["d_vs_plain_product_95"][0] > 0 else "KILL gflip"
    a.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
