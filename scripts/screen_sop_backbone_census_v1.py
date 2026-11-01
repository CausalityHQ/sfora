#!/usr/bin/env python3
"""Quality screen (proposals v3 #3, stage 1): pretrained non-SigLIP towers vs SigLIP2 So400m@384 on SOP, zero training.

Each tower at its native processor settings; image embedding = get_image_features if available, else pooler_output,
else mean of last_hidden_state. Scored with score_corpus packed_pca128 (full_train_gallery / holdout_only R@1) on the
SOP held split (seed 179019, fit 0.9). The So400m@384 baseline is re-measured in this run.
Pre-registered: a tower ADVANCES to one fine-tuned pilot iff full R@1 >= baseline + 0.5 pp; towers that fail to
download/load are reported as SKIPPED (not a negative result).
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_siglip2_substrate_pair as pair
from score_sop_pretrained_substrate import score_corpus
from sfora.representation_ceiling import deterministic_class_partition

TOWERS = ("google/siglip2-so400m-patch14-384", "apple/aimv2-huge-patch14-448", "facebook/metaclip-2-worldwide-giant",
          "apple/DFN5B-CLIP-ViT-H-14-378")
arch, root, outdir = map(Path, sys.argv[1:4]); outdir.mkdir(parents=True, exist_ok=True)
with np.load(arch, allow_pickle=False) as z: rel = np.asarray(z["train_relative_paths"]).astype(str); lab = np.asarray(z["train_labels"], dtype=np.int64)
paths = pair.paths_from_archive(root, rel)
held = np.asarray(deterministic_class_partition(tuple(map(int, lab)), fit_fraction=0.9, seed=179019).validation_row_indexes, dtype=np.int64)
fit = np.setdiff1d(np.arange(len(lab)), held)


class Rows(torch.utils.data.Dataset):
    def __init__(self, proc): self.proc = proc
    def __len__(self): return len(paths)
    def __getitem__(self, i):
        with Image.open(paths[i]) as im: return self.proc(images=[im.convert("RGB")], return_tensors="pt")["pixel_values"][0]


res = {}
for name in TOWERS:
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModel
        snap = snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
        proc = AutoImageProcessor.from_pretrained(snap)
        model = AutoModel.from_pretrained(snap, use_safetensors=True, dtype=torch.float16).cuda().eval()
        dl = torch.utils.data.DataLoader(Rows(proc), batch_size=32, num_workers=8, pin_memory=True)
        feats, t = [], time.perf_counter()
        with torch.inference_mode():
            for px in dl:
                px = px.cuda(non_blocking=True).half()
                if hasattr(model, "get_image_features"):
                    o = model.get_image_features(pixel_values=px)
                    o = o if isinstance(o, torch.Tensor) else o.pooler_output
                else:
                    out = model(pixel_values=px)
                    o = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state.mean(1)
                feats.append(o.float().cpu())
        f = torch.cat(feats).numpy(); secs = time.perf_counter() - t
        sc = score_corpus(name, f, lab, fit, held)["score"]["packed_pca128"]
        res[name] = {"full": sc["full_train_gallery"]["recall_at_1"], "holdout": sc["holdout_only"]["recall_at_1"],
                     "width": int(f.shape[1]), "img_s": len(lab) / secs, "input": list(px.shape[-2:])}
        del model; torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        res[name] = {"skipped": repr(e)[:300]}
    print("ARM", name, json.dumps(res[name]), flush=True)
base = res[TOWERS[0]]["full"]
for n in TOWERS[1:]:
    if "full" in res[n]: res[n]["delta_pp"] = (res[n]["full"] - base) * 100; res[n]["decision"] = "ADVANCE" if res[n]["delta_pp"] >= 0.5 else "CLOSE"
(outdir / "result.json").write_text(json.dumps(res, indent=1)); print("RESULT", json.dumps(res), flush=True)
