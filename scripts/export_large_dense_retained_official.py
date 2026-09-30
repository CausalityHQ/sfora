#!/usr/bin/env python3
"""Four fixed whole-role public B32 exports; no official quality calls.

The DGX parent holds both admission locks and enforces 8GiB/no-swap,
CPU startup120/GPU role300 seconds. This process checks its own usage too.
Freeze unchanged103 + this driver as104; never modify historical manifests.
"""
import argparse
import hashlib
import inspect
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path
from unittest.mock import patch

if not __debug__:
    raise SystemExit("Qualification requires assertions")

SOURCE = Path('/home/riomus/runs/sfora-dense-retained-official-export-source-v2')
SCORE_SOURCE = Path('/home/riomus/runs/sfora-dense-retained-official-score-source-v2')
RETAINED_ROOT = Path('/home/riomus/runs/sfora-dense-retained-serving-source-v3')
RETAINED_SHA = '1b6b97a1298ffc287adc3ac0a3d57f23a5323185f00dd9010fa677a8e7434102'
SERVING = Path('/home/riomus/runs/sfora-dense-retained-serving-v3/receipt.json')
SERVING_SHA = '8476e99a5ff162be23741c050d95abea866f9c4adac363b524253cf2a2f5f238'
PROTOCOL = Path('/home/riomus/runs/sfora-large-inshop-official-qual-v2/official-cpu-proof.json')
PROTOCOL_SHA = 'be7b27c68b5e2895063d4f24b2b9d4e396a77ed79b2426596893c293df2c20f2'
PARTITION_SHA = 'cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c'
MANIFEST = 'dense-retained-official-export-execution.json'
SCORE_MANIFEST = 'dense-retained-official-score-execution.json'
DRIVER = 'export_large_dense_retained_official.py'
SCORE_DRIVER = 'score_large_dense_retained_official.py'
CONTROL_CODE_SHA = 'd2550a6e7cd59916e8d6e9e9a5b9f70dbe60efbbf37aaab3a076b529ca7badb5'
ARMS = {
    'control': (Path('/home/riomus/runs/sfora-lower-pilot-source-179032-control-v1/native.pt'),
        '947ad6c51de596be664883c44c3801a7c94c5f91492ba26ea69afa81ba33d08d',
        Path('/home/riomus/runs/sfora-lower-pilot-source-179032-control-v1/proof.json'),
        '32b163909dd00157b775b5caa5ac02770f5451fb155102a331210fd6ac7c9801'),
    'candidate': (Path('/home/riomus/runs/sfora-dense-pilot-179032-v1/native.pt'),
        '163b02268c44062dbdde2a1b07696c4d0365214ffbabfac76e575281957d362f',
        Path('/home/riomus/runs/sfora-dense-pilot-source-179032-candidate-v2/proof.json'),
        '70cee30cc424c2ec5ccce457eb5cc42550023142de1138ccf0c80b0dec6c825c'),
}
ORDER = (('control', 'gallery'), ('control', 'query'), ('candidate', 'gallery'), ('candidate', 'query'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path, expected):
    assert sha(path) == expected, 'artifact authority differs: ' + str(path)
    return json.loads(Path(path).read_text())


def extension(previous, current, name):
    assert set(current) == set(previous) | {name} and name not in previous
    assert all(current[n] == h for n, h in previous.items()), 'historical source changed'


def source_authority(root, execution, score=False):
    old = read(RETAINED_ROOT / 'dense-retained-serving-execution.json', RETAINED_SHA)
    assert len(old) == 103
    assert all(sha(RETAINED_ROOT / n) == h for n, h in old.items())
    assert read(root / 'dense-retained-serving-execution.json', RETAINED_SHA) == old
    exported = read(root / MANIFEST, sha(SOURCE / MANIFEST) if score else execution)
    extension(old, exported, DRIVER)
    assert len(exported) == 104
    if score:
        assert read(SOURCE / MANIFEST, sha(SOURCE / MANIFEST)) == exported
        assert all(sha(SOURCE / n) == h for n, h in exported.items())
        code = read(root / SCORE_MANIFEST, execution)
        extension(exported, code, SCORE_DRIVER)
        assert len(code) == 105
    else:
        code = exported
    assert all(sha(root / n) == h for n, h in code.items()), 'official source differs'
    return code, exported


def loaded_authority(root, code):
    # Covers newly imported modules too, including the __main__ driver.
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.is_relative_to(root):
                assert code.get(str(path.relative_to(root))) == sha(path), 'late import unqualified: ' + str(path)


def startup(root, execution, score=False):
    code, exported = source_authority(root, execution, score)
    sys.path.insert(0, str(root / 'src'))
    import qualify_large_dense_retained_serving as retained
    assert Path(inspect.getfile(retained)).resolve() == root / 'qualify_large_dense_retained_serving.py'
    original = retained.source_authority
    def current(r, historical):
        assert r == root and historical == RETAINED_SHA
        old, previous, old_export = original(r, historical)
        assert all(code[n] == h for n, h in old.items())
        source_authority(root, execution, score)
        return code, previous, old_export
    with patch.object(retained, 'source_authority', current):
        export, control, frozen, prior, candidate_cpu, _, active = retained.startup(root, RETAINED_SHA)
    assert active == code
    serving = read(SERVING, SERVING_SHA)
    old = read(RETAINED_ROOT / 'dense-retained-serving-execution.json', RETAINED_SHA)
    assert serving['pass'] and serving['source_code'] == old and serving['execution_sha256'] == RETAINED_SHA
    assert serving['checkpoint_sha256'] == ARMS['candidate'][1]
    assert serving['raw_two_seed_decision_sha256'] == retained.DECISION_SHA and serving['two_seed_decision'] == 'KILL'
    assert serving['native_library_sha256'] == retained.LIBRARY_SHA and serving['tileiras_sha256'] == retained.TILEIRAS_SHA
    assert serving['accepted_cpu_proof_sha256'] == ARMS['candidate'][3]
    assert serving['complete_public_B32_held_packed_exact'] and serving['source_head_all_named_buffers_rng_flags_preserved']
    assert serving['native_top10_ordinal_score_bits_exact_queries'] == 6354
    assert serving['optimizer_updates'] == 0 and not serving['quality_read'] and not serving['official_read']
    resource_receipt(serving, 300)
    decision = read(retained.SCORE_ROOT / 'decision-v2.json', retained.DECISION_SHA)
    cpus = {}
    for arm, (checkpoint, digest, proof_path, proof_sha) in ARMS.items():
        assert sha(checkpoint) == digest
        cpu = read(proof_path, proof_sha)
        assert cpu['seed'] == 179032 and cpu['arm'] == arm and cpu['completed_updates'] == 100
        assert cpu['held_images'] == 0 and cpu['numerical_flags'] is None
        assert cpu['pass'] and cpu['read_only'] and cpu['optimizer_updates'] == 0 and not cpu['quality_read']
        assert cpu['changed_driver_rejected'] and cpu['strict400_native_head_reload_and_direct_whole_calibration_exact']
        assert cpu['prefix_data_mutation_rejected_at_exit'] and cpu['cpu_cuda_rng_unchanged']
        assert Path(cpu['native_path']) == checkpoint and cpu['teacher_checkpoint_sha256'] == digest
        assert cpu['fit_manifest'] == frozen['fit_manifest']
        item = next(x for x in decision['inputs'] if x['seed'] == 179032 and x['arm'] == arm)
        run = train_wires(arm)
        receipt = read(run / 'receipt.json', item['receipt_sha256'])
        assert receipt['held_images'] == 12599
        assert receipt['cpu_authority_sha256'] == proof_sha
        assert receipt['training_receipt_sha256'] == cpu['training_receipt_sha256']
        assert receipt['training_checkpoint_sha256'] == cpu['training_checkpoint_sha256']
        assert receipt['checkpoint_sha256'] == digest
        assert all(receipt[k] == frozen[k] for k in ('held_manifest', 'query', 'gallery'))
        assert cpu['execution_sha256'] == (retained.EXPORT_SHA if arm == 'candidate' else CONTROL_CODE_SHA)
        expected_code = read((retained.EXPORT_ROOT if arm == 'candidate' else Path('/home/riomus/runs/sfora-large-lower-pilot-export-source-v1')) /
            ('dense-pilot-export-execution.json' if arm == 'candidate' else 'lower-pilot-export-execution.json'), cpu['execution_sha256'])
        assert cpu['code'] == receipt['source_code'] == expected_code
        if arm == 'control':
            name, training_sha, filename, _ = export.CONTROLS[179032]
            training_run = Path('/home/riomus/runs') / name
            training = read(training_run / 'receipt.json', training_sha)
            assert training_sha == cpu['training_receipt_sha256']
            assert training['pass'] and training['seed'] == 179032 and not training['quality_read']
            assert training.get('updates', training.get('completed_step')) == 100
            assert sha(training_run / filename) == training['checkpoint_sha256'] == cpu['training_checkpoint_sha256']
            assert cpu['teacher_whole_sha256'] == 'a116427c2c876d499f4ee3c52d824f499d3c1a69853d0a34f523e71b03a7a287'
            assert cpu['teacher_head_sha256'] == '4ff7a15a706140c6783955dfb91c73aa81bf4f095561c30a1622c1443882eeb1'
        else:
            assert cpu == candidate_cpu and cpu['training_checkpoint_sha256'] == digest
        cpus[arm] = cpu
    assert cpus['control']['environment'] == cpus['candidate']['environment']
    assert all(sha(Path(n)) == h for n, h in cpus['candidate']['environment']['native_files'].items())
    proof = read(PROTOCOL, PROTOCOL_SHA)
    assert proof['pass'] and proof['partition_sha256'] == PARTITION_SHA
    assert proof['train_query_gallery_ids_disjoint'] and proof['actual_wire_scorer_all_TRAIN_per_query_exact']
    assert proof['official_images_decoded'] == 0 and not proof['official_quality_read'] and proof['optimizer_updates'] == 0
    import evaluate_inshop_siglip2_official as official
    import torch
    torch.set_num_threads(8)
    loaded_authority(root, code)
    export.old.pair.executing_authority(root, code)
    assert sha(control.dataset_root / 'Eval/list_eval_partition.txt') == PARTITION_SHA
    records = official.parse_inshop_partition(control.dataset_root)
    groups = {role: tuple(r for r in records if r.split == role) for role in ('train', 'query', 'gallery')}
    assert tuple(len(groups[r]) for r in groups) == (25882, 14218, 12612)
    train_products = {r.label for r in groups['train']}
    assert train_products == {r['product'] for r in frozen['fit_manifest'] + frozen['held_manifest']}
    assert train_products.isdisjoint(r.label for r in groups['query'] + groups['gallery'])
    assert {r.label for r in groups['query']} == {r.label for r in groups['gallery']}
    for role in ('query', 'gallery'):
        rows = proof['protocol'][role]
        assert [(r['relative_path'], r['product']) for r in rows] == [(str(r.image_path.relative_to(control.dataset_root)), r.label) for r in groups[role]]
        assert all(sha(control.dataset_root / r['relative_path']) == r['image_sha256'] for r in rows)
    return retained, export, control, frozen, prior, cpus, proof, decision, code, exported, official


def train_wires(arm):
    return Path(f"/home/riomus/runs/sfora-{'lower' if arm == 'control' else 'dense'}-pilot-wires-179032-{arm}-{'v1' if arm == 'control' else 'v2'}")


def packed_score(official, query, gallery, qlabels, glabels):
    return official.score_asymmetric(query.codes.float(), gallery.codes.float(), tuple(qlabels), tuple(glabels),
        query_inverse=query.inverse_norms.float(), gallery_inverse=gallery.inverse_norms.float())


def replay_train(export, frozen, decision, official):
    import numpy as np
    import torch
    from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
    labels = [r['product'] for r in frozen['held_manifest']]
    for arm in ARMS:
        run = train_wires(arm)
        item = next(x for x in decision['inputs'] if x['seed'] == 179032 and x['arm'] == arm)
        receipt = read(run / 'receipt.json', item['receipt_sha256'])
        assert sha(run / 'held.npy') == receipt['held_sha256'] == item['held_sha256']
        assert sha(run / 'reference-held.npy') == receipt['reference_held_sha256']
        values = np.load(run / 'held.npy', allow_pickle=False)
        assert values.dtype == np.float32 and values.shape == (12599, 128) and np.isfinite(values).all()
        assert np.array_equal(values, np.load(run / 'reference-held.npy', allow_pickle=False))
        assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
        packed = pack_int8_unit_embeddings(torch.from_numpy(values))
        q, g = (PackedInt8Embeddings(packed.codes[frozen[r]].contiguous(), packed.inverse_norms[frozen[r]].contiguous()) for r in ('query', 'gallery'))
        actual = packed_score(official, q, g, [labels[i] for i in frozen['query']], [labels[i] for i in frozen['gallery']])
        assert actual == decision['quality']['179032'][arm], 'actual TRAIN scorer replay differs'


def resource_receipt(receipt, cap):
    assert 0 <= receipt['wall_seconds'] < cap and 0 < receipt['max_rss_bytes'] < 8 * 1024**3
    assert receipt['swaps'] == receipt['swap_bytes'] == 0
    if 'peak_cuda_allocated_bytes' in receipt:
        assert 0 <= receipt['peak_cuda_allocated_bytes'] < 10_000_000_000


def usage(started, cap):
    stats = resource.getrusage(resource.RUSAGE_SELF)
    swap = int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmSwap:'))) * 1024
    value = dict(wall_seconds=time.perf_counter() - started, max_rss_bytes=stats.ru_maxrss * 1024, swaps=stats.ru_nswap, swap_bytes=swap)
    resource_receipt(value, cap)
    return value


def alarm(cap):
    def timeout(signum, frame):
        raise TimeoutError('official retained wall cap exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(cap)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def output(arm, role):
    return Path(f'/home/riomus/runs/sfora-dense-retained-official-{arm}-{role}-v1')


def normal_exit(arm, role):
    log = SOURCE / f'sfora-dense-retained-official-{arm}-{role}-v1.log'
    assert all(s in log.read_text() for s in ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))


def binding(retained, cpus, code, execution):
    return {'execution_sha256': execution, 'source_code': code, 'retained_execution_sha256': RETAINED_SHA,
        'serving_receipt_sha256': SERVING_SHA, 'protocol_sha256': PROTOCOL_SHA, 'partition_sha256': PARTITION_SHA,
        'raw_two_seed_decision_sha256': retained.DECISION_SHA, 'two_seed_decision': 'KILL',
        'arms': {arm: {'native_path': str(v[0]), 'checkpoint_sha256': v[1], 'accepted_cpu_proof_sha256': v[3],
            'training_receipt_sha256': cpus[arm]['training_receipt_sha256'],
            'training_checkpoint_sha256': cpus[arm]['training_checkpoint_sha256'],
            'whole_sha256': cpus[arm]['teacher_whole_sha256'], 'head_sha256': cpus[arm]['teacher_head_sha256']} for arm, v in ARMS.items()},
        'native_library_sha256': retained.LIBRARY_SHA, 'tileiras_sha256': retained.TILEIRAS_SHA,
        'optimizer_updates': 0, 'claim_eligible': False, 'public_latency_measured': False,
        'global_production_goal_met': False, 'prior_official_benchmark_exposure': True}


def admission(root, expected, bound):
    proof = read(SOURCE / 'startup.json', expected)
    assert all(proof[k] == v for k, v in bound.items())
    assert proof['pass'] and proof['read_only'] and proof['changed_driver_rejected'] and proof['late_import_checked']
    assert proof['actual_TRAIN_candidate_control_per_query_exact']
    assert proof['model_loaded'] is False and proof['official_images_decoded'] == 0 and not proof['official_quality_read']
    resource_receipt(proof, 120)
    return proof


def load_role(arm, role, expected, bound, startup_sha, protocol):
    import numpy as np
    import torch
    from sfora.joint_relational_compaction import PackedInt8Embeddings
    directory = output(arm, role)
    receipt = read(directory / 'receipt.json', expected)
    assert all(receipt[k] == v for k, v in bound.items())
    assert receipt['pass'] and receipt['arm'] == arm and receipt['role'] == role
    assert receipt['startup_receipt_sha256'] == startup_sha
    assert receipt['read_only'] and receipt['optimizer_updates'] == 0 and not receipt['quality_read']
    assert receipt['official_read'] and receipt['complete_whole_role'] and receipt['public_first_batch_direct_whole_packed_exact']
    assert receipt['source_state_rng_environment_code_library_preserved']
    assert receipt['batch'] == 32 and receipt['precision'] == 'fp32_autocast'
    assert receipt['protocol_rows'] == protocol['protocol'][role]
    assert receipt['checkpoint_sha256'] == ARMS[arm][1] and receipt['accepted_cpu_proof_sha256'] == ARMS[arm][3]
    assert receipt['images'] == (14218 if role == 'query' else 12612)
    assert receipt['native_top10_ordinal_score_bits_exact_queries'] == (14218 if role == 'query' else 0)
    resource_receipt(receipt, 300)
    normal_exit(arm, role)
    arrays = []
    for field, dtype, shape in (('codes', np.int8, (receipt['images'], 128)), ('inverse', np.float16, (receipt['images'],))):
        path = directory / (field + '.npy')
        assert sha(path) == receipt[field + '_sha256']
        array = np.load(path, allow_pickle=False)
        assert array.dtype == dtype and array.shape == shape and np.isfinite(array).all()
        arrays.append(torch.from_numpy(array))
    packed = PackedInt8Embeddings(*arrays)
    norms = torch.linalg.vector_norm(packed.codes.float(), dim=1)
    assert packed.codes.min() >= -127 and (norms > 0).all()
    assert torch.equal(norms.reciprocal().half(), packed.inverse_norms)
    return packed, receipt


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-startup-only', action='store_true')
    p.add_argument('--startup-sha256')
    p.add_argument('--arm', choices=tuple(ARMS))
    p.add_argument('--role', choices=('gallery', 'query'))
    p.add_argument('--gallery-sha256')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    assert root == SOURCE and args.protocol_sha256 == PROTOCOL_SHA
    assert not args.output.exists() and not args.output.is_symlink() and args.output.parent.resolve() == args.output.parent
    started = time.perf_counter()
    cap = 120 if args.check_startup_only else 300
    alarm(cap)
    if args.check_startup_only:
        assert args.output == SOURCE / 'startup.json' and not any((args.arm, args.role, args.startup_sha256, args.gallery_sha256))
        assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1')
    else:
        assert args.arm and args.role and args.startup_sha256
        assert args.output == output(args.arm, args.role)
        assert bool(args.gallery_sha256) == (args.role == 'query')
    retained, export, control, frozen, prior, cpus, protocol, decision, code, _, official = startup(root, args.execution_sha256)
    import torch
    import numpy as np
    torch.set_num_threads(8)
    bound = binding(retained, cpus, code, args.execution_sha256)
    if args.check_startup_only:
        assert not torch.cuda.is_available()
        replay_train(export, frozen, decision, official)
        original_sha = sha
        with patch.dict(globals(), sha=lambda f: 'changed' if Path(f).resolve() == Path(__file__).resolve() else original_sha(f)):
            try:
                source_authority(root, args.execution_sha256)
            except AssertionError:
                pass
            else:
                raise AssertionError('changed official driver accepted')
        from sfora.siglip2_compact_serving import Siglip2CompactEncoder
        assert Path(inspect.getfile(Siglip2CompactEncoder)).resolve() == root / 'src/sfora/siglip2_compact_serving.py'
        startup(root, args.execution_sha256)
        loaded_authority(root, code)
        save(args.output, {**bound, 'pass': True, 'read_only': True, 'changed_driver_rejected': True,
            'late_import_checked': True, 'actual_TRAIN_candidate_control_per_query_exact': True,
            'model_loaded': False, 'official_images_decoded': 0, 'official_quality_read': False, **usage(started, cap)})
        print('PASS CPU startup actual TRAIN replay/protocol/current closure; no official quality', flush=True)
        return
    admission(root, args.startup_sha256, bound)
    # Declared whole-role order; never accept a partial predecessor.
    for arm, role in ORDER[:ORDER.index((args.arm, args.role))]:
        receipt_sha = sha(output(arm, role) / 'receipt.json')
        _, receipt = load_role(arm, role, receipt_sha, bound, args.startup_sha256, protocol)
        if role == 'query':
            assert receipt['gallery_receipt_sha256'] == sha(output(arm, 'gallery') / 'receipt.json')
    gallery = None
    if args.role == 'query':
        gallery, _ = load_role(args.arm, 'gallery', args.gallery_sha256, bound, args.startup_sha256, protocol)
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    from sfora.siglip2_compact_serving import Siglip2CompactEncoder
    from sfora.cutile_int8 import CutilePackedInt8Gallery
    from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
    loaded_authority(root, code)
    export.old.pair.executing_authority(root, code)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    pair, teacher = export.old.pair, export.teacher
    flags = teacher.qualified.numerical_flags()
    assert flags == prior['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    checkpoint, checkpoint_sha, _, cpu_sha = ARMS[args.arm]
    cpu = cpus[args.arm]
    with torch.random.fork_rng():
        encoder = Siglip2CompactEncoder.from_checkpoint(model_snapshot=control.large_snapshot,
            checkpoint=checkpoint, expected_checkpoint_sha256=checkpoint_sha, model_file_sha256=pair.smoke.MODEL_HASHES,
            precision='fp32_autocast', device=torch.device('cuda'))
    encoder.vision.requires_grad_(False)
    encoder.head.requires_grad_(False)
    models = (encoder.vision, encoder.head)
    configuration = json.dumps(encoder.vision.config.to_dict(), sort_keys=True)
    runtime = teacher.base.runtime_identity(encoder.vision)
    environment = json.loads(json.dumps(teacher.native.environment(encoder.vision, encoder.processor)))
    assert environment == cpu['environment']
    def all_buffers():
        return {f'{i}.{n}': b for i, m in enumerate(models) for n, b in m.named_buffers()}
    buffers = pair.smoke.digest(all_buffers())
    rng = export.old.fingerprint({'cpu': torch.random.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()})
    def unchanged():
        assert pair.smoke.digest(teacher.base.whole_state(encoder.vision)) == cpu['teacher_whole_sha256']
        assert pair.smoke.digest(encoder.head.state_dict()) == cpu['teacher_head_sha256']
        assert json.dumps(encoder.vision.config.to_dict(), sort_keys=True) == configuration
        assert teacher.base.runtime_identity(encoder.vision) == runtime
        assert json.loads(json.dumps(teacher.native.environment(encoder.vision, encoder.processor))) == environment
        assert pair.smoke.digest(all_buffers()) == buffers
        assert all(b.device == next(m.parameters()).device for m in models for b in m.buffers())
        assert all(p.is_cuda and p.dtype == torch.float32 and not p.requires_grad and p.grad is None for m in models for p in m.parameters())
        assert all(not m.training and not m._forward_hooks and not m._forward_pre_hooks for model in models for m in model.modules())
        assert teacher.qualified.numerical_flags() == flags
        assert rng == export.old.fingerprint({'cpu': torch.random.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()})
        assert torch.cuda.max_memory_allocated() < 10_000_000_000
        usage(started, cap)
    unchanged()
    rows = protocol['protocol'][args.role]
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            block = rows[start:start + 32]
            images, _ = pair.augmented_images(control.dataset_root, block, tuple(range(len(block))), None)
            actual = encoder.encode_images(images)
            assert actual.codes.device.type == actual.inverse_norms.device.type == 'cpu'
            if start == 0:
                pixels = pair.pixels(encoder.processor, images, 'large').cuda()
                pooled = teacher.qualified.fp16(encoder.vision, pixels)
                raw = pair.smoke.compact_head_features(pooled, encoder.head).float()
                direct = pack_int8_unit_embeddings(torch.nn.functional.normalize(raw, dim=1).cpu())
                assert torch.equal(actual.codes, direct.codes) and torch.equal(actual.inverse_norms, direct.inverse_norms)
            chunks.append(actual)
            assert torch.cuda.max_memory_allocated() < 10_000_000_000
            usage(started, cap)
            if start % 2048 == 0:
                print(json.dumps({'arm': args.arm, 'role': args.role, 'images': start + len(block)}), flush=True)
    packed = PackedInt8Embeddings(torch.cat([v.codes for v in chunks]), torch.cat([v.inverse_norms for v in chunks]))
    assert packed.codes.shape == (len(rows), 128) and packed.inverse_norms.shape == (len(rows),)
    if gallery is not None:
        gc, gi = gallery.codes.float(), gallery.inverse_norms.float()
        with CutilePackedInt8Gallery.open_packed(retained.LIBRARY, gallery) as native:
            for start in range(0, len(rows), 32):
                q = PackedInt8Embeddings(packed.codes[start:start + 32].contiguous(), packed.inverse_norms[start:start + 32].contiguous())
                scores = (q.codes.float() @ gc.T) * q.inverse_norms.float()[:, None] * gi[None, :]
                order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
                expected = (order.numpy(), scores.gather(1, order).numpy())
                assert all(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                    for a, b in zip(native.search_packed(q), expected, strict=True)), 'native CPU top10 bits differ'
                usage(started, cap)
    unchanged()
    startup(root, args.execution_sha256)
    unchanged()
    admission(root, args.startup_sha256, bound)
    if args.role == 'query':
        load_role(args.arm, 'gallery', args.gallery_sha256, bound, args.startup_sha256, protocol)
    args.output.mkdir()
    for field, values in (('codes', packed.codes), ('inverse', packed.inverse_norms)):
        np.save(args.output / (field + '.npy'), values.numpy(), allow_pickle=False)
    facts = {**bound, 'pass': True, 'arm': args.arm, 'role': args.role, 'read_only': True,
        'checkpoint_sha256': checkpoint_sha, 'accepted_cpu_proof_sha256': cpu_sha,
        'startup_receipt_sha256': args.startup_sha256, 'gallery_receipt_sha256': args.gallery_sha256,
        'batch': 32, 'precision': 'fp32_autocast', 'images': len(rows), 'protocol_rows': rows,
        'complete_whole_role': True, 'public_first_batch_direct_whole_packed_exact': True,
        'native_top10_ordinal_score_bits_exact_queries': len(rows) if gallery is not None else 0,
        'source_state_rng_environment_code_library_preserved': True, 'numerical_flags': flags,
        'environment': environment, 'quality_read': False, 'official_read': True,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'codes_sha256': sha(args.output / 'codes.npy'), 'inverse_sha256': sha(args.output / 'inverse.npy'), **usage(started, cap)}
    save(args.output / 'receipt.json', facts)
    print('PASS complete official whole-role export; no quality', flush=True)


if __name__ == '__main__':
    main()
