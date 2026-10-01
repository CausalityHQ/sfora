#!/usr/bin/env python3
"""Score four authenticated native256 held wires after CPU/export qualification.

CLI: --execution-sha256 SHA --authority PATH --authority-sha256 SHA
--wires-authority PATH --wires-authority-sha256 SHA --output NEW_JSON.
API: validate_wire(receipt, endpoint, context, cpu) -> None; averaged_deltas
and quality_gate are unchanged late-dense definitions (Large=control,
So400=candidate). Packed scoring and bootstrap execute only the pinned named
definitions from the two full-file references, with the scorer decorator intact.
Every original CPU/export terminal, endpoint and wire closes before scoring.
No official quality, public serving or product completion is qualified here.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import time
UNIT_STARTED = time.perf_counter()

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

METRICS = ("per_query_r1", "per_query_ap")


def load_exporter(args):
    """Execute authenticated source bytes rather than an import-path/pyc helper."""
    root = Path(__file__).absolute().parent
    path = root / 'execution.json'
    with path.open('rb') as stream:
        raw = stream.read(64 * 1024**2 + 1)
    if len(raw) > 64 * 1024**2 or hashlib.sha256(raw).hexdigest() != args.execution_sha256:
        raise ValueError('held execution SHA differs before exporter load')
    code = json.loads(raw)
    path = root / 'export_siglip2_substrate_adaptation.py'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != code[path.name] or '_native256_held_exporter' in sys.modules:
        raise ValueError('held exporter source/preloaded origin differs')
    spec = importlib.util.spec_from_file_location('_native256_held_exporter', path)
    if spec is None or spec.loader is None or Path(spec.origin) != path:
        raise ValueError('held exporter import origin differs')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), 'exec'), vars(module))
    return module


def averaged_deltas(quality):
    assert set(quality) == set(export.SEEDS)
    deltas = {}
    for seed in export.SEEDS:
        assert set(quality[seed]) == {"control", "candidate"}
        deltas[seed] = {}
        for metric in METRICS:
            a, b = (quality[seed][arm][metric] for arm in ("control", "candidate"))
            assert len(a) == len(b) == 6354
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in (*a, *b))
            deltas[seed][metric] = [y - x for x, y in zip(a, b, strict=True)]
    average = {m: [(a + b) / 2 for a, b in zip(deltas[179032][m], deltas[179041][m], strict=True)] for m in METRICS}
    return deltas, average


def quality_gate(deltas, intervals):
    assert set(deltas) == set(export.SEEDS) and set(intervals) == set(METRICS)
    assert all(math.isclose(intervals[m]["mean_delta"], statistics.mean(statistics.mean(deltas[s][m]) for s in export.SEEDS),
                            rel_tol=0, abs_tol=1e-12) for m in METRICS)
    each_seed = all(statistics.mean(deltas[s]["per_query_r1"]) > 0
                    and statistics.mean(deltas[s]["per_query_ap"]) >= 0 for s in export.SEEDS)
    bounds = all(all(math.isfinite(v[k]) for k in ("mean_delta", "product_lower95", "product_upper95", "query_lower95", "query_upper95"))
                 and v["mean_delta"] >= .002 and v["product_lower95"] > 0 for v in intervals.values())
    return each_seed, bool(each_seed and bounds)


def scoring_math(context):
    """Execute only fixed ASTs, including @torch.inference_mode(), after admission."""
    import __future__
    import hashlib
    import numpy as np
    import torch
    namespace = {'__name__': '_native256_held_scoring_math', 'np': np, 'torch': torch,
                 'pack_int8_unit_embeddings': context['packing'].pack_int8_unit_embeddings}
    for name, pin in export.REFERENCES.items():
        path = export.descriptor({'path': str(context['root'] / name), 'sha256': pin['source']}, context['guards'])
        tree = ast.parse(path.read_bytes(), filename=str(path))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == pin['name']]
        selected = ast.Module(body=nodes, type_ignores=[])
        export.require(len(nodes) == 1 and hashlib.sha256(ast.dump(selected, include_attributes=False).encode()).hexdigest() == pin['ast'],
                       'fixed scoring reference AST differs')
        exec(compile(selected, str(path), 'exec', flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)
    return SimpleNamespace(**namespace)


def validate_wire(receipt, endpoint, context, cpu):
    export.require(all(receipt[k] == v for k, v in export.bind_endpoint(context, endpoint).items()) and
            receipt['schema'] == export.SCHEMA and receipt['phase'] == 'export' and receipt['source_code'] == context['code'] and
            receipt['resource_policy'] == export.policy('export') and receipt['precision'] == export.PRECISION and
            receipt['batch'] == 32 and receipt['optimizer_updates'] == 0 and receipt['frozen_split'] == context['spec']['frozen_split'] and
            (receipt['held_images'], receipt['query_images'], receipt['gallery_images'], receipt['held_products']) == (12599, 6354, 6245, 1993) and
            receipt['batch_sizes'] == {name: [32] * (len(context['frozen'][name]) // 32) + [len(context['frozen'][name]) % 32]
                                      for name in ('query', 'gallery')} and
            receipt['files'].keys() == {'held.npy', 'held.codes.npy', 'held.inverse.npy'}, 'export wire binding/split/arithmetic differs')
    export.require(all(receipt[k] is True for k in ('pass', 'strict_complete_inference_reload_exact', 'fit_raw_packed_reload_exact',
            'first_references_released_before_reload', 'full_held_independent_raw_unit_packed_exact',
            'source_head_rng_flags_preserved', 'cpu_rng_preserved', 'exit_rehash_pass')) and
            all(receipt[k] is False for k in ('quality_read', 'official_read', 'claim_eligible', 'public_serving_qualified', 'public_latency_measured')),
            'export integrity/claim scope differs')
    export.require(all(receipt[k] == cpu[k] for k in ('authority_sha256', 'execution_sha256', 'seed', 'arm', 'width', 'output_dim',
            'training_receipt', 'checkpoint', 'terminal_state_sha256', 'fp32_facts', 'native_fp16_facts', 'fit_pixels', 'fit_witness')),
            'wire differs from original FIT-only CPU proof')


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--authority', type=Path, required=True)
    p.add_argument('--authority-sha256', required=True)
    p.add_argument('--wires-authority', type=Path, required=True)
    p.add_argument('--wires-authority-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    return p


def run(args):
    global export
    export = load_exporter(args)
    # The selected original bootstrap is loaded once, before any native import.
    args.arm, args.seed = 'large', export.SEEDS[0]
    context = export.authority(args)
    context['unit_started'] = UNIT_STARTED
    collection = export.read_json({'path': str(args.wires_authority), 'sha256': args.wires_authority_sha256}, context['guards'])
    export.require(collection.keys() == {'schema', 'authority_sha256', 'execution_sha256', 'wires', 'both_locks_held'} and
            collection['schema'] == 'native256-held-wires-v1' and collection['authority_sha256'] == args.authority_sha256 and
            collection['execution_sha256'] == args.execution_sha256 and collection['both_locks_held'] is True,
            'separate frozen wire authority differs')
    export.validate_order(collection['wires'])
    terminals, wire_records = [], []
    # Complete byte/terminal admission precedes NumPy/Torch wire loading/scoring.
    admission = context['training'].FlatAdmission()
    admission.init = context['selected_context']['initialized']['init']
    for wire, endpoint in zip(collection['wires'], context['spec']['endpoints'], strict=True):
        export.require(wire.keys() == {'seed', 'arm', 'terminal', 'cpu_terminal'}, 'exact wire collection entry required')
        cpu_terminal = admission.descriptor_json(wire['cpu_terminal'], context['guards'])
        cpu = admission.descriptor_json(cpu_terminal['receipt'], context['guards'])
        export.validate_cpu(cpu, context, endpoint)
        export.zero_events(admission.admit_terminal(cpu, cpu_terminal, 120, context['guards']))
        terminal = wire['terminal']
        receipt = admission.descriptor_json(terminal['receipt'], context['guards'])
        export.require(receipt['cpu_terminal'] == wire['cpu_terminal'], 'original CPU terminal/wire link differs')
        validate_wire(receipt, endpoint, context, cpu)
        export.zero_events(admission.admit_terminal(receipt, terminal, 300, context['guards']))
        export.require(0 < receipt['peak_cuda_allocated_bytes'] < 10_000_000_000, 'complete export CUDA peak differs')
        for value in (cpu, receipt):
            for path, digest in value['input_guards'].items():
                admission.bound_file(context['guards'], path, digest)
        run = Path(terminal['receipt']['path']).parent
        export.require(Path(terminal['receipt']['path']).name == 'receipt.json' and Path(cpu_terminal['receipt']['path']).name == 'proof.json',
                       'CPU/export original receipt roles differ')
        for name, digest in receipt['files'].items():
            admission.bound_file(context['guards'], run / name, digest)
        terminals.extend((terminal, cpu_terminal))
        wire_records.append((wire, receipt, run))
    all_terminals = terminals + [e['terminal'] for e in context['spec']['endpoints']] + [v[k] for v in context['spec']['prerequisites'].values() for k in ('cpu', 'mechanics')]
    export.require(len({t['invocation_id'] for t in all_terminals}) == len(all_terminals) and
                   len({t['unit'] for t in all_terminals}) == len(all_terminals), 'distinct original whole units required')
    before = export.native_start(context, 'score')
    import numpy as np
    import torch
    for endpoint in context['spec']['endpoints']:
        export.admit_checkpoint(context, endpoint)
    arrays, inputs = {}, []
    for wire, receipt, run in wire_records:
        values = np.load(run / 'held.npy', allow_pickle=False)
        codes = np.load(run / 'held.codes.npy', allow_pickle=False)
        inverse = np.load(run / 'held.inverse.npy', allow_pickle=False)
        export.require(values.dtype == np.float32 and values.shape == (12599, 128) and np.isfinite(values).all() and
                np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0) and
                codes.dtype == np.int8 and codes.shape == values.shape and
                inverse.dtype == np.float16 and inverse.shape == (12599,) and np.isfinite(inverse).all(), 'FP32 unit/packed wire inventory differs')
        packed = context['packing'].pack_int8_unit_embeddings(torch.from_numpy(values))
        export.require(np.array_equal(codes, packed.codes.numpy()) and
                np.array_equal(inverse.view(np.uint16), packed.inverse_norms.numpy().view(np.uint16)), 'int8 codes/exact FP16 inverse bits differ')
        arrays[wire['seed'], wire['arm']] = values
        inputs.append({'seed': wire['seed'], 'arm': wire['arm'], 'terminal': wire['terminal'],
                       'cpu_terminal': wire['cpu_terminal'], 'checkpoint': receipt['checkpoint'], 'files': receipt['files']})
    fixed = scoring_math(context)
    frozen = context['frozen']
    labels = tuple(r['product'] for r in frozen['held_manifest'])
    quality = {s: {label: fixed.packed_quality(arrays[s, arm], labels, frozen['query'], frozen['gallery'], device=torch.device('cpu'))
                   for label, arm in (('control', 'large'), ('candidate', 'so400'))} for s in export.SEEDS}
    deltas, average = averaged_deltas(quality)
    intervals = {}
    for metric in METRICS:
        delta = np.asarray(average[metric])
        intervals[metric] = {'mean_delta': float(delta.mean())}
        for kind, groups in (('product', np.asarray(labels)[frozen['query']]), ('query', np.arange(6354))):
            # Original helper resets the same RNG179019/5000 draws in every call.
            intervals[metric][kind + '_lower95'] = fixed.bootstrap_lower(delta, groups)
            intervals[metric][kind + '_upper95'] = -fixed.bootstrap_lower(-delta, groups)
    each_seed, quality_go = quality_gate(deltas, intervals)
    export.rehash(context)
    result = {'schema': 'siglip2-substrate-held-score-v1', 'pass': True, 'decision': 'GO' if quality_go else 'KILL',
        'authority_sha256': args.authority_sha256, 'wires_authority_sha256': args.wires_authority_sha256,
        'execution_sha256': args.execution_sha256, 'inputs': inputs, 'quality': quality,
        'arm_mapping': {'control': 'large', 'candidate': 'so400'},
        'paired_seed_average_intervals': intervals, 'each_seed_quality_pass': each_seed, 'quality_pass': quality_go,
        'cost_pass': True, 'cost': context['costs'], 'cost_policy': export.COST_POLICY,
        'preparation': context['preparation'],
        'bootstrap_draws': 5000, 'bootstrap_seed': 179019, 'query_images': 6354, 'gallery_images': 6245,
        'fit_images': 13283, 'fit_products': 2004, 'held_products': 1993, 'updates_per_endpoint': 100,
        'schedule_seeds': list(export.SEEDS), 'augmentation': export.AUGMENTATION,
        'interval_scope': 'equal-seed per-query deltas; paired product/query resampling conditional on the two pretrained lineages',
        'independent_pretraining_seeds': False, 'intermediate_checkpoint_selection': False,
        'cost_denominator': 'fresh contemporaneous Large TRAIN100 control for each seed; whole original service and median update',
        'metric_units': 'fractions; multiply deltas by100 for percentage points',
        'quality_read': 'previously observed In-Shop TRAIN-held only', 'official_read': False, 'claim_eligible': False,
        'public_serving_qualified': False, 'public_latency_measured': False, 'global_production_goal_met': False,
        'exit_rehash_pass': True, **export.resources(context, 'score', before)}
    export.publish(args.output, result)
    return result


def main():
    args = parser().parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError, RuntimeError, SyntaxError) as error:
        raise SystemExit('Substrate held scoring rejected: ' + str(error)) from error
    print(result['decision'])


if __name__ == '__main__':
    main()
