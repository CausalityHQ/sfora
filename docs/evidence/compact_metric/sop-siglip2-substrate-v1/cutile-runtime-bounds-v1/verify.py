"""Check the archived execution evidence; does not rerun GPU tests."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
d = json.loads((root / 'receipt.json').read_text())
for name, expected in d['files_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
manifest = json.loads((root / 'source-manifest.json').read_text())
assert manifest['files']['rust/sfora-cutile-int8-score/src/topk.rs'] == d['topk_sha256']
assert manifest['commit'] == d['source_commit']
assert len(manifest['files']) == 11
for name, code in [('cpu-time.txt', 101), ('cpu-v2-time.txt', 0), ('gpu-time.txt', 0)]:
    assert f'Exit status: {code}' in (root / name).read_text()
assert 'CUDA_TOOLKIT_PATH is required but not set' in (root / 'cpu.log').read_text()
assert 'Finished `release` profile' in (root / 'cpu-v2.log').read_text()
log = (root / 'gpu.log').read_text()
for name in ['device_top_ten_matches_scalar_across_boundaries_batches_and_ties',
             'diagnostic_split_preserves_exact_scores_and_stable_ordinals',
             'gallery_row_limit_preserves_signed_ordinals_and_padded_shape',
             'output_ordinal_rejects_placeholders_and_out_of_gallery_values']:
    assert f'test topk::tests::{name} ... ok' in log
assert '4 passed; 0 failed; 0 ignored' in log
assert d['observed']['failed_cpu_wall_seconds'] + d['observed']['passed_cpu_wall_seconds'] < 180
assert d['observed']['gpu_wall_seconds'] < 300
assert d['claim_eligible'] is False
print('PASS raw hashes, source authority, CPU failure/correction and GPU test evidence')
