import json
import resource
from contextlib import ExitStack

import numpy as np
import torch
from PIL import Image

import verify_sop_siglip2_gallery_scale as gate
from train_sop_siglip2_compact import paths_from_archive
from sfora.siglip2_compact_serving import Siglip2CompactEncoder

torch.set_num_threads(20)
torch.backends.cuda.matmul.allow_tf32 = False
training=json.loads((gate.TRAINING/'receipt.json').read_text())
encoder=Siglip2CompactEncoder.from_checkpoint(
    model_snapshot=gate.MODEL,checkpoint=gate.TRAINING/'checkpoint.pt',
    expected_checkpoint_sha256=gate.CHECKPOINT_SHA,
    model_file_sha256=training['model_file_sha256'],
    precision='fp16_native',device=torch.device('cuda:0'))
with np.load(gate.ARCHIVE,allow_pickle=False) as archive:
    paths=paths_from_archive(gate.DATA,np.asarray(archive['train_relative_paths'])[43936:43968])
with ExitStack() as stack, torch.inference_mode():
    images=[stack.enter_context(Image.open(path)) for path in paths]
    bounded=encoder.encode_images(images)
    torch.cuda.synchronize()
    bounded_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    batch=encoder.processor(images=[image.convert('RGB') for image in images],return_tensors='pt')
    pixels=batch['pixel_values'].to(device='cuda',dtype=torch.float16)
    pooled=encoder.vision(pixel_values=pixels).pooler_output
    original=encoder._pack_pooled(pooled,32)
    torch.cuda.synchronize()
    original_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
print(json.dumps({'pixel_sum':sum(image.width*image.height for image in images),'packed_codes_equal':bool(torch.equal(bounded.codes,original.codes)),'inverse_norms_equal':bool(torch.equal(bounded.inverse_norms,original.inverse_norms)),'bounded_peak_rss_bytes':bounded_peak,'after_original_peak_rss_bytes':original_peak}))
