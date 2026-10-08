#!/usr/bin/env python3
"""Separate discarded engineering control serving qualifier; native launch UNRUN.

CLI: python -B qualify_connected_control_serving.py --authority FILE
--authority-sha256 SHA --output NEWDIR. FILE={path,sha256}, canonical regular
current bytes. Evaluator CODE={root,execution_sha256,code}, original exact two;
control_export is the original complete control061 UNIT unchanged. The original
full export launch/fullCPU authority is admitted with the original evaluator.

Authority has exactly KEYS, schema connected-control-serving-requests-authority-v1.
sources has exactly SOURCES; the root supplies actual hashes, including all three
new files. control={arm:control,seed:179061}. observation is a FILE with schema
connected-control-serving-attribution-authority-v1 and original observer fields,
replacing qualified_terminal with control_export_receipt. observer.prepare is
unchanged. native_runtime is the separate complete H-union-S authority FILE.
No selection GO, survivor, export, score or new training is performed.

Launch: both original inherited lifetime locks; body<=120s/whole<=1500s/positive
exit reserve/8GiB/zero swap/CUDA<10GB. Admission, every fresh hash, both sequential
owners, cleanup, complete nested exit and publication count against the cap.
Result schema connected-control-serving-diagnostic-v1, DISCARDED_DIAGNOSTIC,
engineering-only; quality/qualification/state reuse/optimization false. The
parent calls accept_unit(context,UNIT,authority_FILE) to authenticate the exact
new CLI, receipt, normal exit, log, invocation, cgroups, resources and both locks.
No candidate KILL or historical blocker is changed by this diagnostic.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
from types import SimpleNamespace

import qualify_connected_serving_requests as requests

SCHEMA = 'connected-control-serving-requests-authority-v1'
KEYS = {'schema','sources','evaluator','evaluation_authority','control_export',
        'control','observation','native_runtime','locks'}
SOURCES = {'control_driver','control_native','control_test','request_driver','request_test',
           'observer','observer_test','bridge','native_wrapper','packing'}
CONTROL = {'arm':'control','seed':179061}
CONTROL_RECEIPT_SHA = 'db63db8270f2bf9a75448863e7ff701b7894b0c6aaf45c748730b6ea52393407'
CONTROL_INVOCATION = '2ffefca891564bc2a74267d3511c0401'
CPU_RECEIPT_SHA = '1a732ebece8d775bffd142ff8abf32cfbc0da5ffb44c787cd0af22d8b3f7c189'
STARTED = time.perf_counter()


def require(value, message):
    if not value:
        raise ValueError(message)


def prepare_observation(fact, observer):
    """Separate control role; preserve the observer's FILE/source/gallery limits."""
    value = requests.strict_json(observer.file_bytes(fact,keep=True))
    keys = {'schema','sources','bundle','gallery','native','train_images','control_export_receipt',
        'cache_conditions','resource_policy','both_locks_held','qualification_eligible','state_reuse_eligible'}
    require(type(value) is dict and value.keys() == keys and
        value['schema'] == 'connected-control-serving-attribution-authority-v1' and
        value['qualification_eligible'] is False and value['state_reuse_eligible'] is False,
        'exact engineering control observation required')
    sources = value['sources']
    require(type(sources) is dict and sources.keys() == {'observer','test','bridge','trainer','serializer'},
        'complete observation source pins required')
    for file in sources.values(): observer.file_bytes(file)
    require(observer.canonical(sources['observer']['path']) == Path(observer.__file__).absolute(), 'current observer FILE differs')
    bundle,gallery = value['bundle'],value['gallery']
    require(type(bundle) is dict and bundle.keys() == {'directory','manifest'} and
        observer.canonical(bundle['directory']).is_dir() and
        observer.file_fact(bundle['manifest'])['path'] == str(Path(bundle['directory'])/'bundle.json'),
        'exact bundle binding required')
    require(sources['trainer']['path'] == str(Path(bundle['directory'])/'train_siglip2_connected_mlp.py') and
        sources['serializer']['path'] == str(Path(bundle['directory'])/'train_siglip2_substrate_adaptation.py'),
        'bundle source FILE paths required')
    require(type(gallery) is dict and gallery.keys() == {'file','count'} and
        type(gallery['count']) is int and gallery['count'] >= 10, 'exact gallery binding required')
    images = value['train_images']
    require(type(images) is list and len(images) == 32 and
        len({observer.file_fact(f)['path'] for f in images}) == 32, 'fixed32 distinct canonical TRAIN FILEs required')
    require(type(value['cache_conditions']) is str and value['cache_conditions'].strip() and
        value['both_locks_held'] is True, 'parent cache/lock declaration required')
    policy = value['resource_policy']
    require(type(policy) is dict and policy.keys() == observer.LIMITS.keys() | {'whole_process_seconds','exit_reserve_seconds'} and
        all(type(v) is int and v >= 0 for v in policy.values()) and
        all(policy[k] == v for k,v in observer.LIMITS.items()) and policy['exit_reserve_seconds'] > 0 and
        120+policy['exit_reserve_seconds'] < policy['whole_process_seconds'] <= 1500,
        'unchanged whole-process cap/exit reserve required')
    for file in [bundle['manifest'],gallery['file'],value['native'],value['control_export_receipt'],*images]:
        observer.file_bytes(file)
    return value


def admit_control(evaluator, context, authority, observation):
    require(authority['control'] == CONTROL and type(authority['control']['seed']) is int and
        context['launch']['stage'] == 'full' and context['launch']['panel'] == 'selection' and
        context['launch']['phase'] == 'export' and context['launch']['arm'] == 'control' and
        context['launch']['seed'] == 179061 and
        context['launch']['selected_cpu']['receipt']['sha256'] == CPU_RECEIPT_SHA,
        'original control061 full export/fullCPU authority required')
    unit = authority['control_export']
    require(unit['receipt']['sha256'] == CONTROL_RECEIPT_SHA and unit['invocation_id'] == CONTROL_INVOCATION and
        observation['control_export_receipt'] == unit['receipt'], 'original control061 UNIT/observation role required')
    exported = evaluator.accept_unit(context,unit,'export','control',179061,stage='full',panel='selection')
    endpoint = next(e for e in context['launch']['endpoints'] if (e['arm'],e['seed']) == ('control',179061))
    require(observation['bundle']['manifest'] == endpoint['bundle'], 'same original control061 bundle required')
    return endpoint,exported


def cli(authority, output, driver):
    return [driver,'--authority',authority['path'],'--authority-sha256',authority['sha256'],'--output',output]


def validate_receipt(record, authority, authority_fact):
    def exact(value, keys):
        require(type(value) is dict and value.keys() == set(keys.split()), 'exact measurement evidence required')
    def seconds(value):
        require(type(value) in (int,float) and math.isfinite(value) and value >= 0, 'finite nonnegative measurement required')
        return value
    def digest(value):
        require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value), 'measurement SHA256 required')
    def native(value, count):
        require(type(value) is list and len(value) == 2, 'complete native ID/score witness required')
        for row,fmt,width in zip(value,('q','f'),(8,4),strict=True):
            exact(row,'format shape hex')
            formats = ('q','l') if fmt == 'q' and struct.calcsize('l') == 8 else (fmt,)
            require(row['format'] in formats and row['shape'] == [count,10] and
                all(type(v) is int for v in row['shape']) and type(row['hex']) is str and
                re.fullmatch('[0-9a-f]+',row['hex']) and len(row['hex']) == count*10*width*2,
                'typed native ID/score bytes differ')
        require(all(v[0] >= 0 for v in struct.iter_unpack('<q',bytes.fromhex(value[0]['hex']))) and
            all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',bytes.fromhex(value[1]['hex']))),
            'finite native score/nonnegative ID required')
    require(record['schema'] == 'connected-control-serving-diagnostic-v1' and
        record['status'] == 'DISCARDED_DIAGNOSTIC' and record['engineering_only'] is True and
        record['authority'] == authority_fact and record['sources'] == authority['sources'] and
        record['control'] == CONTROL and record['control_export'] == authority['control_export'] and
        all(record[k] is False for k in ('quality_read','quality_eligible','qualification_eligible',
            'state_reuse_eligible','optimization_eligible','product_go')) and
        record['full_uncached_exit_pass'] is True and record['same_group_original_byte_native_parity'] is True,
        'complete discarded engineering receipt required')
    require(record['normal_terminal_required'] is True and record['owned_cleanup_requires_terminal'] is True,
        'publication is provisional until normal terminal after owned cleanup')
    require(record['invocation']['argv'] == cli(authority_fact,record['output'],authority['sources']['control_driver']['path']) and
        record['invocation']['optimize'] == 0 and record['invocation']['cuda_visible_devices'] == '0' and
        record['invocation']['cublas_workspace_config'] == ':4096:8', 'exact new control diagnostic CLI/environment required')
    expected = [(b,k) for b in (1,32) for k in ['warm_oracle','warm']+['timed']*8] + [(1,'observation'),(32,'observation')]
    require([(r['batch'],r['kind']) for r in record['calls']] == expected and
        0 < seconds(record['body_seconds']) <= 120 and record['ties']['ascending_ordinal_score_bits_exact'] is True,
        'unchanged22 public calls/oracle/tie body required')
    require(type(record['calls']) is list and
        record['oracle_semantics'] == 'instrumented first original warmup; second warmup and all8timed are unprofiled' and
        record['timing_semantics'] == 'read/decode through synchronized native top10; capture/cleanup separate; observer overhead retained' and
        record['product_p99'] == 'UNQUALIFIED; requires10000interleaved paired calls and confidence interval',
        'unchanged measurement semantics required')
    timing_keys = 'seconds read_decode_seconds public_call_seconds completion_sync_seconds native_capture_seconds image_cleanup_seconds'
    for row in record['calls']:
        exact(row,timing_keys+' native instrumented qualification_eligible batch kind')
        require(type(row['batch']) is int and row['instrumented'] is (row['kind'] in ('warm_oracle','observation')) and
            row['qualification_eligible'] is False, 'exact four instrumented calls required')
        for key in timing_keys.split(): seconds(row[key])
        require(row['seconds'] > 0 and abs(row['seconds']-sum(row[k] for k in
            ('read_decode_seconds','public_call_seconds','completion_sync_seconds'))) < 1e-6,
            'complete public request timing differs')
        native(row['native'],row['batch'])
    require(type(record['owners']) is list and len(record['owners']) == 2, 'two sequential owner charges required')
    for owner in record['owners']:
        exact(owner,'admission_seconds release_seconds')
        for value in owner.values(): seconds(value)
    charged = sum(sum(owner.values()) for owner in record['owners']) + sum(
        row['seconds']+row['native_capture_seconds']+row['image_cleanup_seconds'] for row in record['calls'])
    require(charged <= record['body_seconds']+1e-6, 'owner/request charges exceed body')
    for key in ('timed','original_warmup_oracles','observations'): exact(record[key],'1 32')
    for count in (1,32):
        rows = [r for r in record['calls'] if r['batch'] == count]
        oracle = record['original_warmup_oracles'][str(count)]
        exact(oracle,'output native')
        native(oracle['native'],count)
        report = record['observations'][str(count)]
        exact(report,'complete failures target_error tensor_occurrences fingerprints output callback_seconds counter_inspection_seconds '
            'output_capture_seconds overhead_semantics host_events exclusive_host_phase_seconds phase_semantics cuda_seconds '
            'opaque_native_subdivisions baseline_raw_unit_packed_wire_parity optimization_eligible qualification_eligible state_reuse_eligible')
        require(report['failures'] == [] and report['target_error'] is None and
            all(report[k] is False for k in ('optimization_eligible','qualification_eligible','state_reuse_eligible')) and
            report['cuda_seconds'] is None and report['opaque_native_subdivisions'] == 'UNMEASURED' and
            report['baseline_raw_unit_packed_wire_parity'] == 'UNMEASURED; parent authenticated same-group oracle required' and
            report['overhead_semantics'] == 'inspection/capture are overlapping subsets of callback time; dispatch unmeasured' and
            report['phase_semantics'] == 'event-name attribution; unresolved/opaque subdivisions stay UNMEASURED; inclusive rows overlap',
            'complete discarded fingerprint observation required')
        require(requests.witness(report,count) == oracle['output'] and
            all(row['native'] == oracle['native'] for row in rows), 'same-group original/native oracle differs')
        for key in ('callback_seconds','counter_inspection_seconds','output_capture_seconds'): seconds(report[key])
        require(report['counter_inspection_seconds']+report['output_capture_seconds'] <= report['callback_seconds']+1e-6,
            'observation callback subsets differ')
        fingerprints,leaves = report['fingerprints'],report['tensor_occurrences']
        require(type(fingerprints) is list and fingerprints and type(leaves) is list and 0 < len(leaves) <= 4096,
            'complete fingerprint/tensor records required')
        for leaf in leaves:
            exact(leaf,'fingerprint dtype shape bytes sha256')
            digest(leaf['sha256'])
            require(type(leaf['fingerprint']) is int and 0 <= leaf['fingerprint'] < len(fingerprints) and
                type(leaf['dtype']) is str and leaf['dtype'] and type(leaf['shape']) is list and
                all(type(v) is int and v >= 0 for v in leaf['shape']) and type(leaf['bytes']) is int and leaf['bytes'] >= 0,
                'typed fingerprint occurrence differs')
        for i,fp in enumerate(fingerprints):
            exact(fp,'sha256 occurrences bytes caller_filename caller_function caller_line host_seconds')
            digest(fp['sha256']); seconds(fp['host_seconds'])
            members = [leaf for leaf in leaves if leaf['fingerprint'] == i]
            require(type(fp['occurrences']) is int and fp['occurrences'] == len(members) and
                type(fp['bytes']) is int and fp['bytes'] == sum(leaf['bytes'] for leaf in members) and
                all(type(fp[k]) is str and fp[k] for k in ('caller_filename','caller_function')) and
                type(fp['caller_line']) is int and fp['caller_line'] > 0, 'fingerprint accounting differs')
        require(type(report['host_events']) is list and 0 < len(report['host_events']) <= 4096,
            'complete host timing events required')
        phases = {}
        for event in report['host_events']:
            exact(event,'kind source function line phase calls inclusive_seconds exclusive_seconds c_exceptions end_semantics')
            require(event['kind'] in ('python','c') and event['end_semantics'] == 'return_or_unwind' and
                all(type(event[k]) is str and event[k] for k in ('source','function','phase')) and
                type(event['line']) is int and event['line'] >= 0 and type(event['calls']) is int and event['calls'] > 0 and
                type(event['c_exceptions']) is int and 0 <= event['c_exceptions'] <= event['calls'] and
                seconds(event['exclusive_seconds']) <= seconds(event['inclusive_seconds']), 'host timing event differs')
            phases[event['phase']] = phases.get(event['phase'],0)+event['exclusive_seconds']
        require(type(report['exclusive_host_phase_seconds']) is dict and report['exclusive_host_phase_seconds'].keys() == phases.keys(),
            'complete host phase summary required')
        for phase,value in phases.items():
            require(math.isclose(seconds(report['exclusive_host_phase_seconds'][phase]),value,rel_tol=1e-9,abs_tol=1e-9),
                'host phase sum differs')
        summary = record['timed'][str(count)]
        exact(summary,'seconds measured_images_per_second')
        times = [row['seconds'] for row in rows if row['kind'] == 'timed']
        for value in summary['seconds']: seconds(value)
        require(type(summary['seconds']) is list and summary['seconds'] == times and
            seconds(summary['measured_images_per_second']) == count*8/sum(times), 'recomputed timed summary differs')
    exact(record['ties'],'ascending_ordinal_score_bits_exact native gallery')
    require(type(record['ties']['native']) is list and len(record['ties']['native']) == 2 and
        record['ties']['gallery'] == 'separate discarded duplicate e1; resident public gallery unchanged', 'complete native ties required')
    for count,value in zip((1,32),record['ties']['native'],strict=True):
        native(value,count)
        require(value[0]['hex'] == (struct.pack('<10q',*range(10))*count).hex() and
            value[1]['hex'] == (struct.pack('<10f',*([1.]*10))*count).hex(), 'native tied ordinal/score bits differ')


def accept_unit(context, unit, authority_fact):
    """Separate terminal receipt API; reuse the authenticated complete UNIT reader."""
    authority = requests.strict_json(requests.read_file(authority_fact))
    require(type(authority) is dict and authority.keys() == KEYS and authority['schema'] == SCHEMA,
        'exact control request authority required')
    record = requests.strict_json(requests.read_file(unit['receipt']))
    validate_receipt(record,authority,authority_fact)
    require(unit['receipt']['path'] == str(Path(record['output'])/'receipt.json') and
        record['observation'] == authority['observation'] and record['native_runtime'] == authority['native_runtime'],
        'diagnostic receipt FILE/output roles differ')
    sources = authority['sources']
    require(type(sources) is dict and sources.keys() == SOURCES, 'complete actual control source pins required')
    owned,failures = [],[]
    try:
        observer_source = requests.Source.load(sources['observer']); owned.append(observer_source)
        observer = observer_source.module
        observation = prepare_observation(authority['observation'],observer)
        require(observation['control_export_receipt'] == authority['control_export']['receipt'] and
            observation['sources']['observer'] == sources['observer'] and observation['sources']['test'] == sources['observer_test'] and
            observation['sources']['bridge'] == sources['bridge'], 'terminal control observation/source binding differs')
        native_source = requests.Source.load(sources['control_native']); owned.append(native_source)
        native = native_source.module.CombinedAuthority(context['training_context'],authority['native_runtime'],observer,requests)
        require(native.record['library'] == observation['native'], 'terminal native FILE differs')
        combined = record['combined_native']; inventory = combined['inventory']
        require(combined.keys() == {'authority','inventory','supplemental_files','historical_projection','mapped_identities'} and
            combined['authority'] == authority['native_runtime'] and combined['supplemental_files'] == native.files and
            combined['mapped_identities'] == {p:list(v) for p,v in native.identities.items()} and
            inventory.keys() == {'files','modules','native_files','packages'} and
            inventory['packages'] == context['training_context']['legacy']['selected']['packages'] and
            type(inventory['native_files']) is list and len(inventory['native_files']) == len(set(inventory['native_files'])) and
            set(inventory['native_files']) <= inventory['files'].keys() and
            native.files.keys() <= set(inventory['native_files']) and
            all({**native.historical['files'],**native.files}.get(p) == h for p,h in inventory['files'].items()) and
            all(native.historical['modules'].get(n) == p for n,p in inventory['modules'].items()),
            'complete combined terminal native evidence differs')
        projected = {**inventory,'files':{p:h for p,h in inventory['files'].items() if p not in native.files},
            'native_files':[p for p in inventory['native_files'] if p not in native.files]}
        legacy = context['training_context']['legacy']
        historical_files = legacy['selected']['source_cpu']['origins']['files'].keys() | legacy['warm_record']['origins']['files'].keys()
        require(combined['historical_projection'] == projected and
            projected['files'].keys()-historical_files == native.supplement['files'].keys() and
            native.supplement['files'].keys() <= set(projected['native_files']), 'original exact-four terminal projection differs')
        required = [authority_fact,authority['observation'],authority['native_runtime'],authority['evaluation_authority'],
            *sources.values(),*native.provenance_facts(),observation['bundle']['manifest'],observation['gallery']['file'],
            observation['control_export_receipt'],*observation['train_images']]
        for file in required:
            observer.file_bytes(file)
            require(record['input_guards'].get(file['path']) == file['sha256'], 'terminal omits frozen control FILE guards')
        require(record['resource_policy'] == observation['resource_policy'], 'frozen observation resource policy differs')
        fact = authority['evaluator']
        evaluator_source = native_source.module.load_evaluator_source(
            {'path':str(Path(fact['root'])/'evaluate_siglip2_connected_mlp.py'),
             'sha256':fact['code']['evaluate_siglip2_connected_mlp.py']},requests); owned.append(evaluator_source)
        evaluator = evaluator_source.module
        evaluator.check_code(fact,evaluator.FILES)
        require(evaluator.closure(fact['root'],fact['execution_sha256'],evaluator.FILES,{}) == fact['code'] and
            str(context['root']) == fact['root'] and context['args'].execution_sha256 == fact['execution_sha256'] and
            context['code'] == fact['code'] and str(context['args'].authority) == authority['evaluation_authority']['path'] and
            context['args'].authority_sha256 == authority['evaluation_authority']['sha256'], 'terminal original evaluator authority differs')
        admit_control(evaluator,context,authority,observation)
    except BaseException as error:
        failures.append(error)
    finally:
        for source in reversed(owned):
            if sys.modules.get(source.module.__name__) is source.module:
                del sys.modules[source.module.__name__]
            else: failures.append(ValueError('owned terminal source registry changed'))
        if failures: requests.raise_failures(failures)
    resources = record['resources']; policy = record['resource_policy']
    require(policy['whole_process_seconds'] <= 1500 and policy['exit_reserve_seconds'] > 0 and
        policy['body_seconds'] == 120 and policy['host_bytes'] == 8*1024**3 and policy['swap_bytes'] == 0 and
        policy['cuda_allocated_bytes_exclusive'] == 10_000_000_000, 'frozen diagnostic resource policy differs')
    requests.check_resources({**resources,'wall_seconds':record['whole_process_seconds']},policy,reserve=False)
    legacy = context['training_context']['legacy']
    prior = legacy['selected']['source_cpu']['invocation']
    require(all(record['invocation'][k] == prior[k] for k in ('python','python_sha256','python_version')) and
        unit['invocation_id'] not in legacy['invocations'], 'original interpreter/fresh diagnostic invocation required')
    terminal_record = {**record,'wall_seconds':record['whole_process_seconds'],**{k:resources[k] for k in
        ('process_peak_rss_kib','cgroup_before','cgroup_after')}}
    final = context['terminal_reader'](legacy['admission'],terminal_record,unit,policy['whole_process_seconds'],context['guards'])
    for value in (resources['cgroup_before'],resources['cgroup_after'],final): context['helper'].zero_events(value)
    for p,h in record['input_guards'].items():
        require(legacy['extract'].sha(Path(p)) == h, 'current diagnostic input SHA256 differs')
    require(all(record['input_guards'].get(p) == h for p,h in context['common_guards'].items()), 'diagnostic omits original guards')
    legacy['invocations'].add(unit['invocation_id'])
    return record


def run(args):
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and sys.getprofile() is None,
        'unoptimized -B unprofiled startup required')
    require(not any(n.split('.')[0] in {'torch','numpy','PIL','sfora','transformers','torchvision','safetensors'} for n in sys.modules),
        'native import preceded control admission')
    authority_fact = {'path':str(args.authority),'sha256':args.authority_sha256}
    authority = requests.strict_json(requests.read_file(authority_fact))
    require(type(authority) is dict and authority.keys() == KEYS and authority['schema'] == SCHEMA and
        authority['control'] == CONTROL and type(authority['control']['seed']) is int, 'exact separate control authority required')
    sources = authority['sources']
    require(type(sources) is dict and sources.keys() == SOURCES, 'complete actual control source pins required')
    for role,name in (('control_driver','qualify_connected_control_serving.py'),
        ('control_native','connected_control_native_authority.py'),('control_test','test_connected_control_serving.py'),
        ('request_driver','qualify_connected_serving_requests.py'),('request_test','test_connected_serving_requests.py'),
        ('observer','observe_connected_serving.py'),('observer_test','test_observe_connected_serving.py')):
        require(sources[role]['path'] == str(Path(__file__).with_name(name).absolute()), 'current source role differs: '+role)
    require(sys.argv == cli(authority_fact,str(args.output),sources['control_driver']['path']), 'fixed canonical control CLI order required')
    request_source = requests.Source(requests,sources['request_driver'])
    self_source = requests.Source(sys.modules[__name__],sources['control_driver'])
    output = args.output
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists() and not output.is_symlink(),
        'exclusive canonical new output required')
    owned,failures,context,exit_guard,api,before,guard = [],[],None,None,None,None,None
    try:
        locks = requests.Locks(authority['locks'])
        observer_source = requests.Source.load(sources['observer']); owned.append(observer_source)
        observer = observer_source.module
        observation = prepare_observation(authority['observation'],observer)
        require(observation['sources']['observer'] == sources['observer'] and
            observation['sources']['test'] == sources['observer_test'] and observation['sources']['bridge'] == sources['bridge'],
            'control observation source binding differs')
        for file in sources.values(): observer.file_bytes(file)
        native_source = requests.Source.load(sources['control_native']); owned.append(native_source)
        runtime = requests.strict_json(observer.file_bytes(authority['native_runtime'],keep=True))
        require(runtime['library'] == observation['native'], 'same frozen control native FILE required')
        native_source.module.validate_runtime_compiler(runtime,observer)
        fact = authority['evaluator']
        evaluator_source = native_source.module.load_evaluator_source(
            {'path':str(Path(fact['root'])/'evaluate_siglip2_connected_mlp.py'),
             'sha256':fact['code']['evaluate_siglip2_connected_mlp.py']},requests); owned.append(evaluator_source)
        evaluator = evaluator_source.module
        evaluator.check_code(fact,evaluator.FILES)
        require(evaluator.closure(fact['root'],fact['execution_sha256'],evaluator.FILES,{}) == fact['code'],
            'complete original evaluator CODE differs')
        eargs = SimpleNamespace(execution_sha256=fact['execution_sha256'],
            authority=Path(authority['evaluation_authority']['path']),authority_sha256=authority['evaluation_authority']['sha256'],
            phase='export',arm='control',seed=179061,output=output)
        context,exit_guard = evaluator.authority(eargs)
        context['training_context']['fit_context']['unit_started'] = STARTED
        endpoint,exported = admit_control(evaluator,context,authority,observation)
        frozen = [authority_fact,authority['observation'],authority['native_runtime'],authority['evaluation_authority'],
            *sources.values(),observation['bundle']['manifest'],observation['control_export_receipt'],
            observation['gallery']['file'],observation['native'],*observation['train_images']]
        evaluator.merge_guards(context['guards'],{f['path']:f['sha256'] for f in frozen})
        runtime_authority = native_source.module.CombinedAuthority(context['training_context'],authority['native_runtime'],observer,requests)
        evaluator.merge_guards(context['guards'],{f['path']:f['sha256'] for f in runtime_authority.provenance_facts()})
        api = runtime_authority.install(evaluator_source,context)
        policy = observation['resource_policy']
        require(policy['whole_process_seconds'] <= evaluator.policy('export')['seconds'] and
            time.perf_counter()-STARTED+120+policy['exit_reserve_seconds'] < policy['whole_process_seconds'],
            'insufficient frozen admission/body/exit headroom')
        before = evaluator.native_start(context)
        import torch
        from PIL import Image
        from sfora import cutile_int8,joint_relational_compaction
        wrapper_source = requests.Source(cutile_int8,sources['native_wrapper'])
        packing_source = requests.Source(joint_relational_compaction,sources['packing'])
        bridge_source = requests.Source.load(sources['bridge']); owned.append(bridge_source)
        torch.random.default_generator.manual_seed(179061); torch.cuda.manual_seed_all(179061)
        rng,cuda_rng = torch.random.get_rng_state().clone(),torch.cuda.get_rng_state_all()
        def guard(*,reserve=True):
            locks.check()
            for source in (self_source,request_source,observer_source,native_source,evaluator_source,
                bridge_source,wrapper_source,packing_source): source.check()
            for file in frozen: observer.file_bytes(file)
            api.authenticate()
            evaluator.guard_helpers(context)
            resources = evaluator.resources(context,before)
            resources['wall_seconds'] = time.perf_counter()-STARTED
            requests.check_resources(resources,policy,reserve=reserve)
            require(torch.equal(rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in
                zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True)), 'whole-unit RNG changed')
            return resources
        guard()
        ties = requests.native_ties(joint_relational_compaction.PackedInt8Embeddings,
            cutile_int8.CutilePackedInt8Gallery,observation['native'],observer)
        api.audit_origins(context['training_context']['legacy'])
        guard()
        def read_images(paths): return requests.decode_images(observer,Image,observation['train_images'],paths)
        with evaluator.endpoint_scope(context,endpoint):
            payloads = evaluator.authenticate_payloads(context,endpoint)
            require(payloads == context['cpu']['payload_facts'][evaluator.label(endpoint)], 'CPU-qualified control payload differs')
            rows,mapping = context['nearest_evaluator'].image_rows(context,'selection')
            selected = {r['path']:r for r in rows}
            require(all(f['path'] in selected and selected[f['path']]['image_sha256'] == f['sha256']
                for f in observation['train_images']), 'ordered TRAIN image membership/bytes differ')
            name = evaluator.label(endpoint)+'.packed.bin'
            wire = requests.read_file({'path':str(Path(exported['output'])/name),'sha256':exported['files'][name]})
            gallery_wire = b''.join(wire[i*130:(i+1)*130] for i in mapping['gallery'])
            require(len(wire) == len(rows)*130 and observation['gallery']['count'] == len(mapping['gallery']) and
                hashlib.sha256(gallery_wire).hexdigest() == observation['gallery']['file']['sha256'],
                'original control TRAIN gallery row/wire binding differs')
            wire = gallery_wire = None
            def factory():
                return bridge_source.module.ConnectedCompactIndex.from_bundle(
                    bundle_dir=Path(observation['bundle']['directory']),expected_bundle_sha256=endpoint['bundle']['sha256'],
                    gallery_path=Path(observation['gallery']['file']['path']),expected_gallery_sha256=observation['gallery']['file']['sha256'],
                    gallery_count=observation['gallery']['count'],native_library_path=Path(observation['native']['path']),
                    expected_native_library_sha256=observation['native']['sha256'])
            require(time.perf_counter()-STARTED+120+policy['exit_reserve_seconds'] < policy['whole_process_seconds'],
                'insufficient whole-process body/exit headroom')
            diagnostic = requests.request_body(factory,observer,read_images,torch.cuda.synchronize,
                [Path(f['path']) for f in observation['train_images']],observation['sources'],guard)
        guard()
        prior = context['training_context']['legacy']['selected']['source_cpu']['invocation']
        record = {'schema':'connected-control-serving-diagnostic-v1','status':'DISCARDED_DIAGNOSTIC','engineering_only':True,
            'authority':authority_fact,'sources':sources,'control':CONTROL,'control_export':authority['control_export'],
            'observation':authority['observation'],'native_runtime':authority['native_runtime'],'output':str(output),
            'ties':ties,**diagnostic,'quality_read':False,'quality_eligible':False,'qualification_eligible':False,
            'state_reuse_eligible':False,'optimization_eligible':False,'product_go':False,'resource_policy':policy,
            'normal_terminal_required':True,'owned_cleanup_requires_terminal':True,
            'invocation':{'argv':sys.argv,'python':str(Path(sys.executable).resolve()),'python_sha256':prior['python_sha256'],
                'python_version':sys.version,'pid':os.getpid(),'invocation_id':os.environ['INVOCATION_ID'],
                'optimize':sys.flags.optimize,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
    except BaseException as error:
        failures.append(error)
    finally:
        if context is not None:
            try:
                if api is None: evaluator.exit_rehash(context,exit_guard)
                else: api.evaluator_exit(context,exit_guard)
                if guard is not None: final_resources = guard(reserve=False)
            except BaseException as error: failures.append(error)
        if not failures:
            try:
                record['combined_native'] = api.evidence()
                final_resources = guard(reserve=False)
                record['full_uncached_exit_pass'] = True
                record['resources'] = final_resources
                record['input_guards'] = dict(context['guards'])
                record['whole_process_seconds'] = time.perf_counter()-STARTED
                validate_receipt(record,authority,authority_fact)
                requests.check_resources({**final_resources,'wall_seconds':record['whole_process_seconds']},policy,reserve=False)
                locks.check()
                output.mkdir()
                context['helper'].publish(output/'receipt.json',record)
            except BaseException as error: failures.append(error)
        for source in reversed(owned):
            if sys.modules.get(source.module.__name__) is source.module: del sys.modules[source.module.__name__]
            else: failures.append(ValueError('owned source registry changed at exit'))
        if not failures:
            try:
                locks.check()
                require(time.perf_counter()-STARTED < policy['whole_process_seconds'], 'publication/cleanup included whole-process cap exceeded')
            except BaseException as error: failures.append(error)
        if failures: requests.raise_failures(failures)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--authority',type=Path,required=True)
    parser.add_argument('--authority-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
