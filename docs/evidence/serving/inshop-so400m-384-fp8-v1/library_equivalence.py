import sys, torch
from pathlib import Path
sys.path.insert(0, "scripts")
from bench_so400m_fp8_linear_v1 import swap
from sfora.so400m_compact_serving import So400mCompactEncoder
from sfora.unicom_inshop import parse_inshop_partition
from PIL import Image
R = Path("/home/riomus/runs"); snap = Path("/home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f")
ck = R / "sfora-inshop-so400m-full-c384-179024-1000-v1/checkpoint.pt"
imgs = [Image.open(r.image_path).convert("RGB") for r in parse_inshop_partition(Path("/home/riomus/datasets/inshop_official_standard")) if r.split == "query"][:128]
lib = So400mCompactEncoder.from_checkpoint(snap, ck, resolution=384, output_dim=512, device=torch.device("cuda"), fp8=True).encode_images(imgs)
ref = So400mCompactEncoder.from_checkpoint(snap, ck, resolution=384, output_dim=512, device=torch.device("cuda"), fp16_native=True)
swap(ref.vision)
out = ref.encode_images(imgs)
print("bitwise_equal", torch.equal(lib, out), "max_abs", float((lib - out).abs().max()))
