import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
import open_clip
from preflight_inshop_siglip2_unseen_gallery import split
from sfora.unicom_inshop import parse_inshop_partition

torch.set_num_threads(16)
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained=None)
weights = load_file('/home/riomus/runs/moda-fashion-vision-fp16-9f3358c2/vision_encoder.safetensors')
model.visual.load_state_dict({k.removeprefix('visual.'): v for k, v in weights.items()}, strict=True)
rows = tuple(row for row in parse_inshop_partition(Path('/home/riomus/datasets/inshop_official_standard')) if row.split == 'train')
_, held = split(tuple(row.label for row in rows))
images = []
for i in held[:32]:
    with Image.open(rows[i].image_path) as image:
        images.append(preprocess(image.convert('RGB')))
pixels = torch.stack(images).cuda()
with torch.inference_mode():
    model.visual.float().cuda().eval()
    full = torch.nn.functional.normalize(model.visual(pixels).float(), dim=1)
    model.visual.half()
    half = torch.nn.functional.normalize(model.visual(pixels.half()).float(), dim=1)
cosine = (full * half).sum(dim=1)
print(json.dumps({'rows': 32, 'minimum_cosine': float(cosine.min()), 'mean_cosine': float(cosine.mean()), 'maximum_l2': float(torch.linalg.vector_norm(full-half,dim=1).max())}))
