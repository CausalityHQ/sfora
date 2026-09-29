#!/usr/bin/env python3
"""Independent original-processor sentinels and common official-order authority."""
if not __debug__:
    raise SystemExit('Processor audit requires Python assertions; optimized mode is forbidden')

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image
import qualify_pe_full_valid_anchor_checkpoint as authority
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert torch.cuda.is_available() and not args.output.exists()
    torch.set_num_threads(8)
    pair = authority.pair
    assert pair.sha(Path(__file__)) == args.source_sha256
    root = Path('/home/riomus/runs/sfora-full-valid-anchor-checkpoint-v1')
    execution = '4dcc6227059da973ff4b9c400264b8a7a4e6e5af03b7f40fa5df157d4c4d6375'
    training = 'd718e4dee3971b13398a43ec56d02057491059d7e62f99362ef384d71b3a036e'
    control, source, _, proof, _, _, protocol, _, code, _ = authority.authority(root, execution, training)
    official = Path('/home/riomus/runs/sfora-full-valid-anchor-official-b32-v1')
    measured_sha = 'fde8508c057057bec0edc515e7c25c28b53686b34ec31c157dad227cdebbed84'
    assert pair.sha(official / 'receipt.json') == measured_sha
    measured = json.loads((official / 'receipt.json').read_text())
    old = json.loads((authority.cpu.previous.FULL_OFFICIAL / 'receipt.json').read_text())
    assert all(code[n] == h for n, h in old['code'].items())
    # The identical frozen evaluator consumes the identical hash-bound protocol
    # rows in order for both arms; do not infer alignment from score correlation.
    evaluator = 'qualify_pe_large_coverage_checkpoint.py'
    assert code[evaluator] == old['code'][evaluator] == pair.sha(root / evaluator)
    assert pair.sha(authority.qualified.PROTOCOL) == authority.qualified.PROTOCOL_SHA
    weights = Path('/home/riomus/runs/sfora-full-valid-anchor-native2000-v1')
    exported = json.loads((weights / 'receipt.json').read_text())
    assert pair.sha(weights / 'native.pt') == exported['checkpoint_sha256'] == measured['checkpoint_sha256']
    model, head, original_processor = authority.qualified.updated(control, source, proof, weights, exported)
    model.half().cuda(); head.cuda()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    flags = authority.qualified.teacher.qualified.numerical_flags()
    torch.cuda.reset_peak_memory_stats()
    rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    facts = {}
    for role in ('query', 'gallery'):
        rows = protocol['protocol'][role]
        images = []
        for row in rows[:32]:
            path = control.dataset_root / row['relative_path']
            assert pair.sha(path) == row['image_sha256']
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        pixels = pair.pixels(original_processor, images, 'large')
        with torch.inference_mode():
            pooled = model(pixel_values=pixels.cuda().half()).pooler_output
            actual = pack_int8_unit_embeddings(F.normalize(pair.smoke.compact_head_features(pooled, head), dim=1).cpu())
        saved = authority.qualified.selected.fp16.load_packed(official, measured, role)
        expected = PackedInt8Embeddings(saved.codes[:32].contiguous(), saved.inverse_norms[:32].contiguous())
        authority.qualified.selected.fp16.same(actual, expected)
        encoded = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
        facts[role] = {'images_checked': 32, 'pixels_sha256': pair.smoke.digest({'pixels': pixels}), 'ordered_protocol_sha256': hashlib.sha256(encoded).hexdigest()}
    assert pair.smoke.digest(authority.qualified.trained.base.whole_state(model)) == json.loads((root / 'updated-cpu-proof.json').read_text())['updated_f16_whole_sha256']
    assert pair.smoke.digest(head.state_dict()) == exported['updated_head_sha256']
    assert torch.equal(rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert flags == authority.qualified.teacher.qualified.numerical_flags() and torch.cuda.max_memory_allocated() < 10_000_000_000
    assert all(pair.sha(root / n) == h for n, h in code.items()) and pair.sha(Path(__file__)) == args.source_sha256
    pair.smoke.save(args.output, {'pass': True, 'source_sha256': args.source_sha256, 'candidate_receipt_sha256': measured_sha, 'protocol_sha256': authority.qualified.PROTOCOL_SHA, 'roles': facts, 'original_processor_from_separate_native_reload': True, 'all64_native_original_processor_packed_exact_saved_public': True, 'common_order_bound_by_identical_evaluator_and_protocol': True, 'source_whole_head_rng_preserved': True, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'quality_read': False, 'optimizer_updates': 0})
    print('PASS separate original processor64 packed sentinels and common protocol order', flush=True)


if __name__ == '__main__':
    main()
