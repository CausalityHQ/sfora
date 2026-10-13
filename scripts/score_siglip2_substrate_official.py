#!/usr/bin/env python3
"""One official final-step read per arm; run Large first, then So400 with its receipt.

--checkpoint is full-stage step2000 resume.pt. --receipt-sha256 authenticates
its adjacent receipt.json and the entire twenty-chunk chain. For So400, supply
--control (Large score receipt.json) and --control-sha256. Both sources are their
own seed-179032 pilot endpoints. No checkpoint selection; no production claim.
Parent owns one-read authorization and successful original terminal admission.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import train_siglip2_substrate_full as full

SCHEMA = 'siglip2-substrate-official-v1'
REFERENCES = {
    'evaluate_inshop_siglip2_official.py': ('6fee591bffdb2af8d77355bd4f0c282ac52979fa2125ff46bfb31df0ea985411', 'score_asymmetric'),
    'qualify_pe_large_inshop_official.py': ('d6d52da270f9c983ede3ccb23dc4374964a6e520e040758d73193c33ca5fa5a5', 'metrics'),
    'score_inshop_crop_view_pair.py': ('16e27ccaa7325b9ef7efdb3512cb95791a682afdd5ae59bd1f0847d63b87f5ed', 'bootstrap_lower'),
}
METRICS = ('per_query_r1', 'per_query_ap')


def scoring_math(context):
    """Execute only hash-bound incumbent metric definitions, preserving decorators."""
    import numpy as np
    import torch
    namespace = {'np': np, 'torch': torch, 'Counter': Counter}
    root = Path(__file__).resolve().parent
    for name, (digest, function) in REFERENCES.items():
        path = root / name
        context['pilot'].bound_file(context['guards'], path, digest)
        raw = path.read_bytes()
        full.require(hashlib.sha256(raw).hexdigest() == digest, 'metric source changed')
        nodes = [n for n in ast.parse(raw, filename=str(path)).body if isinstance(n, ast.FunctionDef) and n.name == function]
        full.require(len(nodes) == 1, 'metric definition differs')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    namespace['pair'] = SimpleNamespace(bootstrap_lower=namespace['bootstrap_lower'])
    return SimpleNamespace(**namespace)


def validate_control(control, protocol, code):
    full.require(control['schema'] == SCHEMA and control['pass'] is True and control['arm'] == 'large' and
                 control['completed_step'] == full.TOTAL_UPDATES and control['seed'] == full.SEED and
                 control['protocol'] == protocol and control['code'] == code and
                 control['selection'] is False and control['quality_read'] is True and control['exit_rehash_pass'] is True,
                 'matched final Large control required')
    for metric in METRICS:
        values = control['quality'][metric]
        full.require(len(values) == 14218 and all(math.isfinite(v) and 0 <= v <= 1 for v in values), 'control metric values differ')


def quality_gate(intervals):
    r1 = intervals['per_query_r1']
    full.require(all(math.isfinite(value) for row in intervals.values() for value in row.values()), 'nonfinite paired interval')
    return r1['mean_delta'] > 0 and r1['product_lower95'] > 0


def paired(quality, control, labels, math_source):
    import numpy as np
    intervals = {}
    for metric in METRICS:
        candidate, baseline = np.asarray(quality[metric]), np.asarray(control['quality'][metric])
        full.require(candidate.shape == baseline.shape == (14218,) and np.isfinite(candidate).all() and
                     ((candidate >= 0) & (candidate <= 1)).all(), 'paired query metrics differ')
        delta = candidate - baseline
        intervals[metric] = {'mean_delta': float(delta.mean()),
                             'product_lower95': math_source.bootstrap_lower(delta, np.asarray(labels)),
                             'product_upper95': -math_source.bootstrap_lower(-delta, np.asarray(labels))}
    return {'candidate_minus_control': intervals, 'bootstrap_draws': 5000, 'bootstrap_seed': 179019,
            'decision': 'GO' if quality_gate(intervals) else 'KILL',
            'decision_scope': 'positive paired product R@1 gain; mAP@R reported; fixed training seed'}


def export_role(context, state, rows, packing, output, role):
    import torch
    from torch.nn import functional as F
    codes, inverses = [], []
    manifest = []
    for row in rows:
        ref = full.descriptor(row.image_path.resolve())
        full.require(context['guards'].setdefault(ref['path'], ref['sha256']) == ref['sha256'], 'official image authority conflict')
        manifest.append({'path': ref['path'], 'product': row.label, 'image_sha256': ref['sha256']})
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            batch = tuple(range(start, min(start + 32, len(rows))))
            pixels, _ = full.pixels(context, state['processor'], manifest, batch)
            pooled = state['model'](pixel_values=pixels.cuda().half()).pooler_output.float()
            unit = F.normalize(state['head'](F.normalize(pooled, dim=1)), dim=1)
            full.require(torch.isfinite(unit).all().item(), 'nonfinite official embeddings')
            packed = packing.pack_int8_unit_embeddings(unit.cpu())
            codes.append(packed.codes); inverses.append(packed.inverse_norms)
            print(json.dumps({'phase': 'official_export', 'role': role, 'rows': start + len(batch)}), flush=True)
    result = packing.PackedInt8Embeddings(torch.cat(codes), torch.cat(inverses))
    path = output / (role + '.pt')
    with path.open('xb') as stream:
        torch.save({'codes': result.codes, 'inverse_norms': result.inverse_norms}, stream)
    return result, full.descriptor(path), full.json_sha(manifest)


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--arm', choices=sorted(full.WIDTHS), required=True)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--checkpoint-sha256', required=True)
    result.add_argument('--receipt-sha256', required=True)
    result.add_argument('--control', type=Path)
    result.add_argument('--control-sha256')
    result.add_argument('--partition-parser', type=Path, default=Path(__file__).resolve().parents[1] / 'src/sfora/unicom_inshop.py')
    result.add_argument('--output', type=Path, required=True)
    return result


def run(args):
    started = time.monotonic()
    full.install_backstop()
    full.require(args.output.is_absolute() and args.output.parent.resolve() == args.output.parent and
                 not args.output.exists() and not args.output.is_symlink(), 'exclusive canonical output required')
    full.require(bool(args.control) == bool(args.control_sha256) == (args.arm == 'so400'), 'So400 requires matched Large score')
    ref = full.descriptor(args.checkpoint.parent / 'receipt.json', args.receipt_sha256)
    record = full.read_json(ref)
    full.require(record['completed_step'] == full.TOTAL_UPDATES and record['checkpoint'] ==
                 full.descriptor(args.checkpoint, args.checkpoint_sha256), 'final step2000 only; checkpoint differs')
    train_code = {'trainer': full.sha(Path(full.__file__)), 'partition_parser': full.PARSER_SHA}
    full.check_chain(ref, full.TOTAL_UPDATES, args.arm, record['source'], train_code)
    code = {**train_code, 'scorer': full.sha(Path(__file__)), 'metrics': REFERENCES}
    code = full.canonical(code)
    control_ref = full.descriptor(args.control, args.control_sha256) if args.control else None
    control = full.read_json(control_ref) if control_ref else None
    context = full.admit_source(record['source'], args.arm, args.output)
    for path, digest in ((Path(__file__).resolve(), code['scorer']), (Path(full.__file__).resolve(), train_code['trainer'])):
        context['pilot'].bound_file(context['guards'], path, digest)
    before, flags = full.native_start(context)
    import torch
    all_records = full.records(context, args.partition_parser)
    query = tuple(r for r in all_records if r.split == 'query')
    gallery = tuple(r for r in all_records if r.split == 'gallery')
    qlabels, glabels = tuple(r.label for r in query), tuple(r.label for r in gallery)
    protocol = {'partition_sha256': full.PARTITION_SHA, 'queries': len(query), 'gallery': len(gallery),
                'query_order_sha256': full.json_sha([(str(r.image_path), r.label) for r in query]),
                'gallery_order_sha256': full.json_sha([(str(r.image_path), r.label) for r in gallery]),
                'width': 128, 'wire': 'int8_codes_fp16_inverse', 'precision': 'native_fp16_vision_fp32_head', 'batch': 32}
    full.require((len(query), len(gallery), len(set(qlabels))) == (14218, 12612, 3985), 'official geometry differs')
    if control:
        validate_control(control, protocol, code)
        full.require(control['training']['source'] != record['source'] and
                     control['training']['identity']['full_rows_sha256'] == record['identity']['full_rows_sha256'] and
                     control['training']['identity']['schedule_sha256'] == record['identity']['schedule_sha256'],
                     'matched training data/schedule/source lineage differs')
    math_source = scoring_math(context)
    q = context['initialized']
    packing = full.load_module('_siglip2_full_packing', q['root'] / 'joint_relational_compaction.py',
                               q['code']['joint_relational_compaction.py'])
    state, _ = full.restore_full(context, record, flags, inference=True)
    state['model'].requires_grad_(False).half().cuda().eval()
    state['head'].requires_grad_(False).cuda().eval()
    args.output.mkdir()
    qwire, qref, qhash = export_role(context, state, query, packing, args.output, 'query')
    gwire, gref, ghash = export_role(context, state, gallery, packing, args.output, 'gallery')
    if control:
        full.require(control['image_manifest_sha256'] == {'query': qhash, 'gallery': ghash}, 'paired official image bytes differ')
    del state
    gc.collect(); torch.cuda.empty_cache()
    quality, intervals, _ = math_source.metrics(math_source, qwire, gwire, qlabels, glabels, 'cuda')
    comparison = paired(quality, control, qlabels, math_source) if control else None
    full.check_chain(ref, full.TOTAL_UPDATES, args.arm, record['source'], train_code)
    if control_ref:
        full.descriptor(control_ref['path'], control_ref['sha256'])
    context['pilot'].exit_rehash(context)
    result = {'schema': SCHEMA, 'pass': True, 'arm': args.arm, 'seed': full.SEED, 'completed_step': full.TOTAL_UPDATES,
              'code': code, 'training_receipt': ref, 'training': record, 'protocol': protocol,
              'image_manifest_sha256': {'query': qhash, 'gallery': ghash}, 'wires': {'query': qref, 'gallery': gref},
              'quality': quality, 'intervals': intervals, 'control': control_ref, 'paired': comparison,
              'gap_to_dated_96_7_percentage_points': (.967 - quality['recall_at_1']) * 100,
              'quality_read': True, 'previously_observed_official_split': True, 'selection': False,
              'production_qualified': False, 'exit_rehash_pass': True, 'resource_policy': full.POLICY,
              'terminal_exit_and_both_locks_require_parent_receipt': True,
              **full.resources(context, before, started)}
    full.publish(args.output / 'receipt.json', result)
    return result


def main():
    args = parser().parse_args()
    result = run(args)
    print(json.dumps({'pass': True, 'arm': args.arm, 'quality': {k: result['quality'][k] for k in ('recall_at_1', 'map_at_r')},
                      'paired': result['paired']}), flush=True)


if __name__ == '__main__':
    main()
