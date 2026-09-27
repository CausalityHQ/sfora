import json
import resource
from contextlib import ExitStack

import numpy as np
import torch
from PIL import Image

import verify_sop_siglip2_gallery_scale as gate
from train_sop_siglip2_compact import paths_from_archive
from sfora.siglip2_compact_serving import Siglip2CompactEncoder

def rss():
    with open('/proc/self/status') as stream:
        for line in stream:
            if line.startswith('VmRSS:'):
                current = int(line.split()[1]) * 1024
                break
    return {'current': current, 'peak': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}

torch.set_num_threads(20)
torch.backends.cuda.matmul.allow_tf32 = False
training = json.loads((gate.TRAINING / 'receipt.json').read_text())
encoder = Siglip2CompactEncoder.from_checkpoint(
    model_snapshot=gate.MODEL, checkpoint=gate.TRAINING/'checkpoint.pt',
    expected_checkpoint_sha256=gate.CHECKPOINT_SHA,
    model_file_sha256=training['model_file_sha256'],
    precision='fp16_native', device=torch.device('cuda:0'),
)
with np.load(gate.ARCHIVE, allow_pickle=False) as archive:
    paths = paths_from_archive(gate.DATA, np.asarray(archive['train_relative_paths'])[:10000])
print(json.dumps({'images':0,**rss()}),flush=True)
for start in range(0,len(paths),32):
    with ExitStack() as stack:
        images=[stack.enter_context(Image.open(path)) for path in paths[start:start+32]]
        encoder.encode_images(images)
    if (start+32)%2000<32 or start+32>=len(paths):
        print(json.dumps({'images':min(start+32,len(paths)),**rss()}),flush=True)
