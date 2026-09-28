"""CPU replay of prototype packed hits, paired bootstrap and all frozen gates."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import fit_centered_pca

root = Path(sys.argv[1])
r = json.loads((root / 'pilot.json').read_text())
startup = json.loads((root / 'pilot-code-authority.json').read_text())
assert all(r['code_authority'].get(name) == expected for name, expected in startup.items())
late = {name: value for name, value in r['code_authority'].items() if name not in startup}
assert set(late) == {'isolated-deps/einops/_torch_specific.py'}
wheel = json.loads((root / 'einops-wheel.json').read_text())
with (root / wheel['filename']).open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == wheel['sha256']
with zipfile.ZipFile(root / wheel['filename']) as archive:
    assert hashlib.sha256(archive.read('einops/_torch_specific.py')).hexdigest() == next(iter(late.values()))
with (root / 'pilot.features.npz').open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == r['features_sha256']
torch.set_num_threads(8)
features = np.load(root / 'pilot.features.npz', allow_pickle=False)
for arm, receipt in r['arms'].items():
    data = features[arm]
    assert data.shape == (1024, 1024) and data.dtype == np.float32
    assert hashlib.sha256(data.tobytes()).hexdigest() == receipt['source_sha256']
    source = torch.from_numpy(data)
    query, gallery = source[:512], source[512:]
    pca = fit_centered_pca(gallery, dimensions=128)
    q, g = [pack_int8_unit_embeddings(torch.nn.functional.normalize(pca.apply(x), dim=1))
            for x in (query, gallery)]
    scores = (q.codes.float() @ g.codes.float().T) * q.inverse_norms.float()[:, None] * g.inverse_norms.float()[None, :]
    ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, 0]
    hits = (ranked == torch.arange(512)).int().tolist()
    assert hits == receipt['packed_hits'] and sum(hits)/512 == receipt['packed_r1']
    raw = ((query @ gallery.T).argmax(1) == torch.arange(512)).float().mean()
    assert float(raw) == receipt['raw_float_r1']
    assert len(receipt['encoder_batch32_ms']) == 10
    assert float(np.median(receipt['encoder_batch32_ms'])) == receipt['encoder_batch32_p50_ms']
delta = np.array(r['arms']['pe']['packed_hits']) - np.array(r['arms']['large']['packed_hits'])
rng = np.random.Generator(np.random.PCG64(179031))
# Scalar product draws independently replay the vectorized pilot's stream.
draws = [delta[rng.integers(512, size=512)].mean() for _ in range(5000)]
lower = float(np.quantile(draws, .025))
ratio = r['arms']['pe']['encoder_batch32_p50_ms'] / r['arms']['large']['encoder_batch32_p50_ms']
assert 100*lower == r['gain_lower95_pp'] and 100*delta.mean() == r['gain_pp']
assert ratio == r['encoder_batch32_p50_ratio']
criteria = {'quality_point': delta.mean() >= -.03, 'quality_lower': lower >= -.05,
            'encoder_time': ratio <= .8, 'wall': r['main_wall_seconds'] <= 120,
            'cuda': r['peak_cuda_allocated_bytes'] < 10_000_000_000}
assert criteria == r['criteria'] and all(criteria.values()) == r['advance']
assert min(r['fp16_fp32_cosine_first4']) >= .999 and r['claim_eligible'] is False
for name, key in [('cpu-preflight-v2.json','preflight_sha256'),('acquisition.json','acquisition_sha256')]:
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == r[key]
print('PASS saved-source packed/raw hits, PCG64 bootstrap, timing medians and all frozen criteria')
