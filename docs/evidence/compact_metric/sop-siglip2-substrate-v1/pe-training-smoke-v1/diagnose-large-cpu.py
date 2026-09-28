"""CPU-only source/pixel alignment falsifier after the closed GPU parity failure."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import probe_inshop_pe_training_smoke as smoke

torch.set_num_threads(8)
args = SimpleNamespace(
    root=Path('/home/riomus/runs/sfora-pe-core-source-v1'),
    output=Path('/home/riomus/runs/sfora-pe-training-smoke-v4'),
    dataset_root=Path('/home/riomus/datasets/inshop_official_standard'),
    large_snapshot=Path('/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c'),
)
assert not torch.cuda.is_available()
frozen = json.loads((args.output / 'preflight.json').read_text())
smoke.check_startup(args, frozen)
vision, processor = smoke.load_arm(args, 'large')
vision.eval()
rows = frozen['updated_rows'][:4]
manifest = json.loads((args.root / 'cpu-preflight-v2.json').read_text())['image_manifest']
images = []
for row in rows:
    with Image.open(args.dataset_root / manifest[row]['relative_path']) as image:
        images.append(image.convert('RGB'))
single = torch.stack([processor(images=[image], return_tensors='pt')['pixel_values'][0] for image in images])
grouped = processor(images=images, return_tensors='pt')['pixel_values']
assert torch.equal(single, grouped)
with torch.no_grad():
    output = smoke.encode(vision, single, 'large').float()
cached = torch.from_numpy(np.load(args.root / 'pilot.features.npz', allow_pickle=False)['large'][rows])
result = {
    'rows': rows,
    'single_vs_grouped_pixels_exact': True,
    'cached_gpu_fp16_vs_cpu_fp32_cosine': torch.nn.functional.cosine_similarity(cached, output).tolist(),
    'cuda_used': False,
    'optimizer_updates': 0,
    'quality_read': False,
    'diagnostic_only': True,
}
smoke.save(args.output / 'large-cpu-alignment.json', result)
print(json.dumps(result))
