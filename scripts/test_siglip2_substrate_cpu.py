#!/usr/bin/env python3
"""Runnable stdlib-only authority negatives; NEVER execute native qualification."""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import ast
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import qualify_siglip2_substrate_cpu as driver


def rejects(call, message):
    try:
        call()
    except (ValueError, FileExistsError, KeyError) as error:
        assert message in str(error), str(error)
        return
    raise AssertionError('invalid input accepted: ' + message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    return digest(path)


def fixture(base, extract):
    """Real stdlib files and complete inventories; no tensor payload allocation."""
    config = {'model_type': 'siglip', 'vision_config': {
        'hidden_size': 1024, 'num_hidden_layers': 24, 'intermediate_size': 4096, 'image_size': 256}}
    processor = {'image_processor_type': 'SiglipImageProcessor', 'do_resize': True,
                 'size': {'height': 256, 'width': 256}, 'resample': 2, 'do_rescale': True,
                 'rescale_factor': 1 / 255, 'do_normalize': True, 'image_mean': [.5]*3, 'image_std': [.5]*3}
    config_path, processor_path = base / 'config.json', base / 'preprocessor.json'
    config_sha, processor_sha = write(config_path, config), write(processor_path, processor)
    source = base / 'vision.safetensors'
    source.write_bytes(b'fixture metadata, never loaded')
    shapes, resolved = extract.expected_vision(config, driver.ARMS['large'])
    tensors, inventory, cursor = [], {}, 0
    for original, shape in sorted(shapes.items()):
        length = 4
        for dimension in shape:
            length *= dimension
        key = original[len(extract.PREFIX):]
        offsets = [cursor, cursor + length]
        inventory[key] = {'shape': shape, 'dtype': 'F32', 'data_offsets': offsets}
        tensors.append({'original_key': original, 'derived_key': key, 'shape': shape, 'dtype': 'F32',
                        'sha256': '1'*64, 'derived_data_offsets': offsets, 'data_offsets': offsets})
        cursor += length
    source_input = {'arm': 'large', 'source_model': driver.ARMS['large'], 'revision': 'a'*40,
                    'source': {'path': str(base / 'UNREAD-full-text.safetensors'), 'bytes': 999, 'sha256': '2'*64},
                    'config': {'path': str(config_path), 'sha256': config_sha},
                    'preprocessor': {'path': str(processor_path), 'sha256': processor_sha}}
    vision = {'path': str(source), 'sha256': digest(source), 'size_bytes': source.stat().st_size,
              'tensor_count': 400, 'payload_bytes': cursor,
              'key_mapping': 'strip exactly vision_model. for installed direct SiglipVisionModel'}
    provenance_path = base / 'provenance.json'
    provenance = {'schema': extract.SCHEMA, 'pass': True, 'model_qualified': False,
                  'source_model': driver.ARMS['large'], 'revision': 'a'*40, 'output': vision,
                  'source': {'path': source_input['source']['path'], 'sha256': '2'*64, 'size_bytes': 999},
                  'config': {'path': str(config_path), 'sha256': config_sha, 'value': config},
                  'preprocessor': {'path': str(processor_path), 'sha256': processor_sha, 'value': processor},
                  'resolved_vision_inventory_config': resolved, 'tensors': tensors}
    provenance_sha = write(provenance_path, provenance)
    old = base / 'original'
    old.mkdir()
    old_code = {}
    for name in ('extract_siglip2_vision_source.py', 'test_extract_siglip2_vision_source.py'):
        (old / name).write_text('# authenticated original code\n')
        old_code[name] = digest(old / name)
    old_execution_sha = write(old / 'execution.json', old_code)
    old_inputs_sha = write(old / 'inputs.json', {'schema': 'native256-substrate-extraction-inputs-v2',
        'resource_limits': driver.POLICY, 'model_qualified': False, 'quality_read': False, 'sources': [source_input]})
    log_path = old / 'original.log'
    log_path.write_text('Running as unit: original-large.service; invocation ID: ' + 'b'*32 + '\n'
        '\tExit status: 0\nFinished with result: success\n'
        'Main processes terminated with: code=exited/status=0\n\tSwaps: 0\nMemory swap peak: 0B\n'
        'Service runtime: 14.971s\n\tMaximum resident set size (kbytes): 38840\n')
    entry = {'arm': 'large', 'source_model': driver.ARMS['large'], 'revision': 'a'*40, 'input': source_input,
             'vision': vision, 'integrity_pass': True, 'model_qualified': False,
             'resource_policy': driver.POLICY, 'host_swap_bytes': 0, 'service_seconds': 14.971,
             'native_peak_rss_kib': 38840, 'unit': 'original-large', 'invocation_id': 'b'*32,
             'original_execution_manifest': str(old / 'execution.json'), 'original_execution_sha256': old_execution_sha,
             'original_inputs': str(old / 'inputs.json'), 'original_inputs_sha256': old_inputs_sha,
             'original_log_path': str(log_path), 'original_log_sha256': digest(log_path),
             'provenance': {'path': str(provenance_path), 'sha256': provenance_sha}, 'whole_cgroup_peak_verified': False}
    sources = {'schema': 'paired-native256-vision-sources-v1', 'model_qualified': False, 'quality_read': False,
               'resource_note': 'original peak not verified', 'extraction_authority': {
                   'execution_sha256': old_execution_sha, 'inputs_sha256': old_inputs_sha},
               'sources': [entry, {**copy.deepcopy(entry), 'arm': 'so400'}]}
    sources_path = base / 'sources.json'
    sources_sha = write(sources_path, sources)
    dataset = base / 'dataset'
    (dataset / 'Img/img').mkdir(parents=True)
    for index in range(2):
        (dataset / f'Img/img/{index}.jpg').write_bytes(b'FIT witness' + bytes([index]))
    image_hash = lambda index: digest(dataset / f'Img/img/{index}.jpg') if index < 2 else '3'*64
    classes = [f'id_{index:08d}' for index in range(2004)]
    targets = [index % 2004 for index in range(13283)]
    fit = {'schema': 'native256-frozen-fit-manifest-v1', 'fit_images': 13283, 'fit_identities': 2004,
           'held_images_read': 0, 'quality_read': False, 'source_features_reused': False, 'teacher_state_reused': False,
           'dataset_root': str(dataset), 'class_names': classes, 'targets': targets,
           'rows': [{'train_row': index, 'relative_path': f'Img/img/{index}.jpg',
                     'product': classes[targets[index]], 'image_sha256': image_hash(index)} for index in range(13283)]}
    fit_path = base / 'fit.json'
    fit_sha = write(fit_path, fit)
    return SimpleNamespace(execution_sha256=digest(base / 'execution.json'), sources=sources_path,
        sources_sha256=sources_sha, fit_manifest=fit_path, fit_manifest_sha256=fit_sha,
        arm='large', output=base / 'NEW'), sources, fit, inventory, provenance


def main():
    path = Path(driver.__file__).resolve()
    # Evaluate the ACTUAL model_facts return expression with pure witnesses:
    # no Torch import, and integer id2label keys must survive JSON persistence.
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'model_facts')
    expression = compile(ast.Expression(function.body[-1].value), str(path), 'eval')
    config_value = {'id2label': {0: 'LABEL_0', 1: 'LABEL_1'}, 'hidden_size': 1024}
    model = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: config_value, _attn_implementation='sdpa'),
                            state_dict=lambda: {})
    scope = {'json': json, 'model': model, 'modules': [{'normalized_shape': (1024,)}], 'roles': [],
             'buffers': {}, 'processor': SimpleNamespace(backend='torchvision'), 'processor_config': {},
             'module_origin': lambda *_: {'class': 'stock.Processor'}, 'packages': {}}
    facts = eval(expression, scope)
    assert facts['config']['id2label'] == {'0': 'LABEL_0', '1': 'LABEL_1'}
    assert facts == json.loads(json.dumps(facts))
    config_value['hidden_size'] = 1152
    assert eval(expression, scope) != facts, 'config change erased by canonicalization'
    for argv, code, message in ((['--help'], 0, '--execution-sha256'), ([], 2, 'required')):
        result = subprocess.run([sys.executable, '-B', '-S', str(path), *argv], capture_output=True, text=True)
        assert result.returncode == code and message in result.stdout + result.stderr
    result = subprocess.run([sys.executable, '-B', '-S', '-O', str(path), '--help'], capture_output=True, text=True)
    assert result.returncode != 0 and 'optimized' in result.stderr
    with TemporaryDirectory() as directory:
        base = Path(directory).resolve()
        for name in driver.FILES:
            shutil.copyfile(path.parent / name, base / name)
        code = {name: digest(base / name) for name in driver.FILES}
        execution = write(base / 'execution.json', code)
        # Authenticate a copied closure and origin rather than changing repo files.
        with patch.object(driver, '__file__', str(base / path.name)), patch.object(sys, 'path', [str(base), *sys.path]), \
                patch.dict(sys.modules):
            sys.modules.pop('extract_siglip2_vision_source', None)
            extract, actual = driver.bootstrap(base, execution)
            assert actual == code and 'torch' not in sys.modules
            rejects(lambda: driver.bootstrap(base, '0'*64), 'execution SHA256')
            for changed in ({**code, 'extra.py': '0'*64}, {k: v for k, v in code.items() if k != path.name}):
                changed_sha = write(base / 'execution.json', changed)
                rejects(lambda: driver.bootstrap(base, changed_sha), 'exactly three')
            write(base / 'execution.json', code)
            original = (base / path.name).read_bytes()
            (base / path.name).write_bytes(original + b'\n# tamper\n')
            rejects(lambda: driver.bootstrap(base, execution), 'execution file SHA256')
            (base / path.name).write_bytes(original)
            with patch.object(driver.importlib.util, 'find_spec', return_value=SimpleNamespace(origin=str(base / 'shadow.py'))):
                rejects(lambda: driver.bootstrap(base, execution), 'import origin')
            args, sources, fit, inventory, provenance = fixture(base, extract)
            # This fixture replaces only safetensors HEADER reading, not file
            # hashes, schema checks, binding, logs, profiles, closure or origins.
            with patch.object(extract, 'read_header', return_value=(inventory, {}, b'')):
                context = driver.authority(args)
                assert len(context['expected']) == 400 and len(context['images']) == 2
                assert not Path(context['entry']['input']['source']['path']).exists()
                rejects(lambda: driver.fresh_source(context), 'origin preflight')
                for field, value in (('sources_sha256', '0'*64), ('fit_manifest_sha256', '0'*64)):
                    altered = copy.copy(args)
                    setattr(altered, field, value)
                    rejects(lambda: driver.authority(altered), 'SHA256')
                for field in ('resource_policy', 'original_log_sha256', 'source_model'):
                    altered = copy.deepcopy(sources)
                    altered['sources'][0][field] = {} if field == 'resource_policy' else '0'*64
                    args.sources_sha256 = write(args.sources, altered)
                    rejects(lambda: driver.authority(args), {'resource_policy': 'caps', 'original_log_sha256': 'SHA256',
                                                            'source_model': 'model/revision'}[field])
                args.sources_sha256 = write(args.sources, sources)
                log_path = Path(sources['sources'][0]['original_log_path'])
                log = log_path.read_text()
                log_path.write_text(log.replace('status=0', 'status=1'))
                altered = copy.deepcopy(sources)
                altered['sources'][0]['original_log_sha256'] = digest(log_path)
                args.sources_sha256 = write(args.sources, altered)
                rejects(lambda: driver.authority(args), 'normal-exit')
                log_path.write_text(log)
                args.sources_sha256 = write(args.sources, sources)
                args.output.mkdir()
                (args.output / 'preserved').write_text('keep')
                rejects(lambda: driver.authority(args), 'output already exists')
                assert (args.output / 'preserved').read_text() == 'keep'
                (args.output / 'preserved').unlink()
                args.output.rmdir()
                args.output.symlink_to(base / 'missing')
                rejects(lambda: driver.authority(args), 'output already exists')
                args.output.unlink()
                context = driver.authority(args)
                changed = Path(context['entry']['input']['config']['path'])
                contents = changed.read_bytes()
                altered_config = json.loads(contents)
                altered_config['vision_config']['hidden_size'] = 1152
                write(changed, altered_config)
                rejects(lambda: driver.authority(args), 'SHA256')
                rejects(lambda: driver.rehash(context), 'exit authority')
                changed.write_bytes(contents)
                driver.rehash(context)
            expected = context['expected']
            driver.validate_derived(extract, inventory, expected, provenance)
            for key in ('text_model.foo', 'vision_model.head.probe', 'head.extra', 'embeddings.position_ids'):
                bad = {**inventory, key: {'shape': [1], 'dtype': 'F32', 'data_offsets': [0, 4]}}
                rejects(lambda: driver.validate_derived(extract, bad, expected, provenance), 'bare-key inventory')
            bad = copy.deepcopy(provenance)
            bad['tensors'][0]['dtype'] = 'F16'
            rejects(lambda: driver.validate_derived(extract, inventory, expected, bad), 'tensor/provenance')
            config = context['config']
            rejects(lambda: extract.expected_vision(config, driver.ARMS['so400']), 'profile')
            so400 = copy.deepcopy(config)
            so400['vision_config'].update(hidden_size=1152, num_hidden_layers=27, intermediate_size=4304)
            assert len(extract.expected_vision(so400, driver.ARMS['so400'])[0]) == 448
            for field, value in (('train_row', 1), ('relative_path', '../escape'), ('product', 'wrong')):
                bad = copy.deepcopy(fit)
                bad['rows'][0][field] = value
                rejects(lambda: driver.fit_rows(extract, bad), 'row/target')
            rejects(lambda: extract.strict_json('{"x":1,"x":2}'), 'duplicate')
            rejects(lambda: extract.strict_json('{"x":NaN}'), 'JSON')
            # Installed origin checks are stdlib mocks; no package is imported.
            site = base / 'site-packages'
            native_files = {}
            versions = {dist.lower(): 'test-version' for dist in driver.PACKAGES.values()}
            for package in driver.PACKAGES:
                (site / package).mkdir(parents=True)
                init = site / package / '__init__.py'
                init.write_text('# no native imports\n')
            constructor = site / 'transformers/modeling.py'
            constructor.write_text('# observed constructor\n')
            native_files[str(constructor)] = {'sha256': digest(constructor), 'bytes': constructor.stat().st_size}
            context['sources']['native_environment'] = {'schema': 'native256-installed-source-observation-v1',
                'native_imported': False, 'model_executed': False, 'quality_read': False,
                'site_packages': str(site), 'versions': versions, 'files': native_files,
                'vision_constructor': {'path': str(constructor), 'direct_bare_state_keys_source_observed': True,
                    'assigned_self_attributes': ['config', 'embeddings', 'encoder', 'head', 'post_layernorm', 'use_head']}}
            distribution = SimpleNamespace(version='test-version', locate_file=lambda relative: site / relative)
            specs = lambda package: SimpleNamespace(origin=str(site / package / '__init__.py'))
            with patch.object(driver.importlib.metadata, 'distribution', return_value=distribution), \
                    patch.object(driver.importlib.util, 'find_spec', side_effect=specs):
                assert len(driver.package_origins(context)) == 6
                with patch.object(driver.importlib.util, 'find_spec', return_value=SimpleNamespace(origin=str(base / 'shadow.py'))):
                    rejects(lambda: driver.package_origins(context), 'import origin')
                distribution.version = 'wrong-version'
                rejects(lambda: driver.package_origins(context), 'import origin')
                distribution.version = 'test-version'
                with patch.dict(sys.modules, {'torch.fake': SimpleNamespace(__file__=str(base / 'shadow.py'))}):
                    rejects(lambda: driver.loaded_origins(driver.package_origins(context)), 'loaded native module origin')
                constructor.write_text('# tamper\n')
                rejects(lambda: driver.package_origins(context), 'size differs')
    assert not any(package in sys.modules for package in driver.PACKAGES)
    print('PASS: stdlib closure, authority/tamper, logs, profiles, extra keys, origins, exclusive output and -O; no native execution')


if __name__ == '__main__':
    main()
