import hashlib
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

archive = Path('/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz')
root = Path('/home/riomus/datasets/Stanford_Online_Products')
snapshot = Path('/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c')
output = Path('/home/riomus/runs/sfora-sop-rgb-copy-screen-179024-v1/receipt.json')
assert not output.exists()
torch.set_num_threads(20)
with np.load(archive, allow_pickle=False) as source:
    relatives = np.asarray(source['train_relative_paths']).astype(str)
rows = np.linspace(0, len(relatives) - 1, 32, dtype=np.int64)
paths = [root / relatives[row] for row in rows]
images = []
for path in paths:
    with Image.open(BytesIO(path.read_bytes())) as opened:
        images.append(opened.convert('RGB'))
processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, backend='torchvision')
assert type(processor).__name__ == 'SiglipImageProcessor'
samples = {str(batch): {'copy': [], 'reuse': []} for batch in (1, 32)}
for batch in (1, 32):
    selected = images[:batch]
    copied = processor(images=[image.convert('RGB') for image in selected], return_tensors='pt')['pixel_values']
    reused = processor(images=selected, return_tensors='pt')['pixel_values']
    assert torch.equal(copied, reused)
    for block in range(20):
        order = ('copy', 'reuse', 'reuse', 'copy') if block % 2 == 0 else ('reuse', 'copy', 'copy', 'reuse')
        for arm in order:
            for _ in range(3):
                processor(images=[image.convert('RGB') for image in selected] if arm == 'copy' else selected, return_tensors='pt')
            for _ in range(10):
                start = time.perf_counter_ns()
                value = processor(images=[image.convert('RGB') for image in selected] if arm == 'copy' else selected, return_tensors='pt')
                samples[str(batch)][arm].append(time.perf_counter_ns() - start)
                assert value['pixel_values'].shape[0] == batch

def stats(values):
    ms = np.asarray(values, dtype=np.float64) / 1e6
    return {'p50_ms': float(np.median(ms)), 'p95_ms': float(np.quantile(ms, .95)), 'mean_ms': float(ms.mean())}

report = {
    'schema': 'sfora-sop-rgb-copy-processor-screen-v1',
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'image_sha256': [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
    'processor_exact': True,
    'batch': {key: {arm: stats(values) for arm, values in arms.items()} for key, arms in samples.items()},
    'raw_ns': samples,
}
output.write_text(json.dumps(report, sort_keys=True) + '\n')
print(json.dumps(report['batch'], sort_keys=True))
