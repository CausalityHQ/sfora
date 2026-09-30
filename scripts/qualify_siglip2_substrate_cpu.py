#!/usr/bin/env python3
"""Qualify ONE fresh native256 vision SOURCE on CPU, never an initializer.

The parent freezes execution.json beside this script: exactly the extractor,
this driver and its stdlib test, mapped to lowercase SHA256. --sources uses the
paired-native256-vision-sources-v1 receipt WITH authenticated original paths.
No original full/text archive, teacher, feature cache, PCA, head or optimizer
is loaded. Only two frozen FIT images are read. Output is an exclusive NEWDIR
containing fresh_vision.pt and proof.json; failed directories are not reusable.
The proof is conditional on the parent's enclosing unit normal exit/120s,
8GiB/noSwap, CUDA hidden and both locks. Actual cgroup memory is read INSIDE
that unit; old extraction summary peaks remain explicitly unverified.
Native imports are lazy. Help and negative checks need only the stdlib.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

SCHEMA = 'siglip2-substrate-source-cpu-v1'
FILES = {'extract_siglip2_vision_source.py', 'qualify_siglip2_substrate_cpu.py',
         'test_siglip2_substrate_cpu.py'}
ARMS = {'large': 'google/siglip2-large-patch16-256',
        'so400': 'google/siglip2-so400m-patch16-256'}
POLICY = {'cuda_visible_devices': '', 'host_bytes': 8 * 1024**3,
          'seconds': 120, 'swap_bytes': 0}
PACKAGES = {'torch': 'torch', 'transformers': 'transformers',
            'torchvision': 'torchvision', 'safetensors': 'safetensors',
            'numpy': 'numpy', 'PIL': 'Pillow'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(),
            'canonical regular file required: ' + str(path))
    return path


def bootstrap(root, expected):
    """Authenticate the entire local closure BEFORE importing its helper."""
    # Bootstrap cannot use the extractor until its bytes/origin are verified.
    def digest(path):
        with canonical(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    require(re.fullmatch('[0-9a-f]{64}', expected or '') is not None, 'execution SHA256 required')
    manifest = root / 'execution.json'
    require(digest(manifest) == expected, 'execution SHA256 differs')
    raw = manifest.read_bytes()
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate execution key')
            value[key] = item
        return value
    code = json.loads(raw, object_pairs_hook=pairs)
    require(isinstance(code, dict) and code.keys() == FILES, 'execution requires exactly three files')
    for name, value in code.items():
        require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
                'execution file SHA256 required')
        require(digest(root / name) == value, 'execution file SHA256 differs: ' + name)
    name = 'extract_siglip2_vision_source'
    path = root / (name + '.py')
    spec = importlib.util.find_spec(name)
    require(spec is not None and Path(spec.origin).absolute() == path,
            'extractor import origin differs')
    loaded = sys.modules.get(name)
    if loaded is None:
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    require(Path(loaded.__file__).absolute() == path, 'extractor loaded origin differs')
    return loaded, code


def bound_file(extract, guards, path, expected, size=None):
    path = canonical(path)
    extract.digest_string(expected)
    require(size is None or path.stat().st_size == size, 'bound file size differs: ' + str(path))
    require(extract.sha(path) == expected, 'bound file SHA256 differs: ' + str(path))
    prior = guards.setdefault(str(path), expected)
    require(prior == expected, 'conflicting file authority')
    return path


def read_json(extract, guards, path, expected):
    path = bound_file(extract, guards, path, expected)
    with path.open('rb') as stream:
        raw = stream.read(extract.HEADER_CAP + 1)
    require(len(raw) <= extract.HEADER_CAP, 'authority JSON too large')
    require(hashlib.sha256(raw).hexdigest() == expected, 'authority JSON changed during read')
    return extract.strict_json(raw)


def original_extraction(extract, guards, entry, authority):
    require(entry['integrity_pass'] is True and entry['model_qualified'] is False,
            'original extraction integrity required')
    require(entry['resource_policy'] == POLICY and entry['host_swap_bytes'] == 0 and
            0 < entry['service_seconds'] <= POLICY['seconds'], 'original extraction caps differ')
    require(entry['original_execution_sha256'] == authority['execution_sha256'] and
            entry['original_inputs_sha256'] == authority['inputs_sha256'], 'original authority differs')
    execution_path = canonical(entry['original_execution_manifest'])
    code = read_json(extract, guards, execution_path, entry['original_execution_sha256'])
    require(code.keys() == {'extract_siglip2_vision_source.py', 'test_extract_siglip2_vision_source.py'},
            'original execution closure differs')
    for name, digest in code.items():
        bound_file(extract, guards, execution_path.parent / name, digest)
    inputs = read_json(extract, guards, entry['original_inputs'], entry['original_inputs_sha256'])
    require(inputs['schema'] == 'native256-substrate-extraction-inputs-v2' and
            inputs['resource_limits'] == POLICY and inputs['model_qualified'] is False and
            inputs['quality_read'] is False, 'original input authority differs')
    require([item for item in inputs['sources'] if item['arm'] == entry['arm']] == [entry['input']],
            'paired source/input binding differs')
    log = bound_file(extract, guards, entry['original_log_path'], entry['original_log_sha256']).read_text()
    require(f"Running as unit: {entry['unit']}.service; invocation ID: {entry['invocation_id']}\n" in log and
            '\tExit status: 0\n' in log and 'Finished with result: success\n' in log and
            'Main processes terminated with: code=exited/status=0\n' in log and
            '\tSwaps: 0\n' in log and 'Memory swap peak: 0B\n' in log and
            f"Service runtime: {entry['service_seconds']}s\n" in log and
            f"\tMaximum resident set size (kbytes): {entry['native_peak_rss_kib']}\n" in log,
            'original normal-exit log differs')


def fit_rows(extract, fit):
    require(fit['schema'] == 'native256-frozen-fit-manifest-v1' and fit['fit_images'] == 13283 and
            fit['fit_identities'] == 2004 and fit['held_images_read'] == 0 and
            fit['quality_read'] is False and fit['source_features_reused'] is False and
            fit['teacher_state_reused'] is False, 'FIT manifest profile differs')
    rows, targets, classes = fit['rows'], fit['targets'], fit['class_names']
    require(len(rows) == len(targets) == 13283 and len(classes) == len(set(classes)) == 2004,
            'FIT inventory differs')
    seen, train_rows = set(), set()
    for row, target in zip(rows, targets):
        relative = Path(row['relative_path'])
        require(type(row['train_row']) is int and row['train_row'] >= 0 and row['train_row'] not in train_rows and
                type(target) is int and 0 <= target < 2004 and
                row['product'] == classes[target] and not relative.is_absolute() and
                '..' not in relative.parts and str(relative).startswith('Img/img/') and
                str(relative) not in seen, 'FIT row/target binding differs')
        extract.digest_string(row['image_sha256'])
        seen.add(str(relative))
        train_rows.add(row['train_row'])
    require(set(targets) == set(range(2004)), 'FIT class coverage differs')
    root = Path(fit['dataset_root'])
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'FIT root must be canonical')
    images = [(root / row['relative_path']).resolve() for row in rows[:2]]
    require(all(path.is_relative_to(root) for path in images), 'FIT image escaped dataset root')
    return images


def validate_derived(extract, inventory, expected, provenance):
    require(inventory.keys() == expected.keys(), 'derived bare-key inventory differs')
    rows = [row for row in provenance['tensors'] if row['derived_key'] is not None]
    mapping = {row['derived_key']: row for row in rows}
    require(len(rows) == len(mapping) == len(expected), 'derived provenance inventory differs')
    for name, shape in expected.items():
        row, item = mapping[name], inventory[name]
        require(row['original_key'] == extract.PREFIX + name and row['shape'] == shape and
                row['dtype'] == 'F32' and item == {'dtype': 'F32', 'shape': shape,
                                                'data_offsets': row['derived_data_offsets']},
                'derived tensor/provenance differs: ' + name)
        extract.digest_string(row['sha256'])
    return mapping


def authority(args):
    root = Path(__file__).absolute().parent
    require(root.resolve() == root, 'driver root must be canonical')
    extract, code = bootstrap(root, args.execution_sha256)
    extract.new_output(args.output)
    guards = {str(root / 'execution.json'): args.execution_sha256,
              **{str(root / name): digest for name, digest in code.items()}}
    sources = read_json(extract, guards, args.sources, args.sources_sha256)
    require(sources['schema'] == 'paired-native256-vision-sources-v1' and
            sources['model_qualified'] is False and sources['quality_read'] is False,
            'paired sources profile differs')
    require(len(sources['sources']) == 2 and {s['arm'] for s in sources['sources']} == ARMS.keys(),
            'paired source arms differ')
    entry = next(s for s in sources['sources'] if s['arm'] == args.arm)
    require(entry['source_model'] == ARMS[args.arm] and entry['input']['source_model'] == entry['source_model'] and
            entry['revision'] == entry['input']['revision'] and
            re.fullmatch('[0-9a-f]{40}', entry['revision']) is not None, 'source model/revision differs')
    original_extraction(extract, guards, entry, sources['extraction_authority'])
    provenance = read_json(extract, guards, entry['provenance']['path'], entry['provenance']['sha256'])
    require(provenance['schema'] == extract.SCHEMA and provenance['pass'] is True and
            provenance['model_qualified'] is False and provenance['source_model'] == entry['source_model'] and
            provenance['revision'] == entry['revision'] and provenance['output'] == entry['vision'],
            'selected provenance binding differs')
    original = entry['input']['source']
    require(all(provenance['source'][key] == original[other] for key, other in
                (('path', 'path'), ('sha256', 'sha256'), ('size_bytes', 'bytes'))), 'original archive binding differs')
    config = read_json(extract, guards, entry['input']['config']['path'], entry['input']['config']['sha256'])
    processor = read_json(extract, guards, entry['input']['preprocessor']['path'], entry['input']['preprocessor']['sha256'])
    for field, value in (('config', config), ('preprocessor', processor)):
        require(provenance[field] == {'path': entry['input'][field]['path'],
                                     'sha256': entry['input'][field]['sha256'], 'value': value},
                field + ' provenance differs')
    prefixed, resolved = extract.expected_vision(config, entry['source_model'])
    extract.validate_preprocessor(processor)
    require(provenance['resolved_vision_inventory_config'] == resolved, 'resolved source config differs')
    expected = {name[len(extract.PREFIX):]: shape for name, shape in prefixed.items()}
    require(len(expected) == entry['vision']['tensor_count'] == (400 if args.arm == 'large' else 448),
            'source tensor count differs')
    source = bound_file(extract, guards, entry['vision']['path'], entry['vision']['sha256'], entry['vision']['size_bytes'])
    with source.open('rb') as stream:
        inventory, metadata, _ = extract.read_header(stream)
    mapping = validate_derived(extract, inventory, expected, provenance)
    fit = read_json(extract, guards, args.fit_manifest, args.fit_manifest_sha256)
    images = fit_rows(extract, fit)
    receipt = fit['original_receipt']
    original = read_json(extract, guards, receipt['path'], receipt['sha256'])
    require(fit['rows'] == original['fit_manifest'] and fit['targets'] == original['target_products'] and
            fit['class_names'] == sorted({row['product'] for row in original['fit_manifest']}),
            'original FIT binding differs')
    for path, row in zip(images, fit['rows'][:2]):
        bound_file(extract, guards, path, row['image_sha256'])
    require(str(args.output) not in guards, 'output conflicts with authority')
    return {'args': args, 'root': root, 'extract': extract, 'code': code, 'guards': guards,
            'sources': sources, 'entry': entry, 'provenance': provenance,
            'config': config, 'processor': processor, 'expected': expected,
            'mapping': mapping, 'source': source, 'fit': fit, 'images': images}


def package_origins(context):
    """Resolve installed distribution origins with stdlib BEFORE native imports."""
    observed = context['sources']['native_environment']
    require(observed['schema'] == 'native256-installed-source-observation-v1' and
            observed['native_imported'] is False and observed['model_executed'] is False and
            observed['quality_read'] is False, 'native environment observation differs')
    site = Path(observed['site_packages'])
    require(site.is_absolute() and site.resolve() == site and site.is_dir(), 'installed site root differs')
    require(observed['versions'].keys() == {name.lower() for name in PACKAGES.values()},
            'native version inventory differs')
    for path, value in observed['files'].items():
        require(Path(path).is_relative_to(site), 'native observed file outside site root')
        bound_file(context['extract'], context['guards'], path, value['sha256'], value['bytes'])
    result = {}
    for package, distribution in PACKAGES.items():
        require(package not in sys.modules, 'native package already imported: ' + package)
        spec = importlib.util.find_spec(package)
        dist = importlib.metadata.distribution(distribution)
        expected = Path(dist.locate_file(package + '/__init__.py')).resolve()
        require(spec is not None and spec.origin is not None and Path(spec.origin).resolve() == expected and
                expected.is_file() and expected.parent.parent == site and
                dist.version == observed['versions'][distribution.lower()],
                'installed package import origin differs: ' + package)
        result[package] = {'root': str(expected.parent), 'origin': str(expected), 'version': dist.version}
        context['guards'][str(expected)] = context['extract'].sha(expected)
    constructor = observed['vision_constructor']
    require(constructor['direct_bare_state_keys_source_observed'] is True and
            constructor['path'] in observed['files'] and
            set(constructor['assigned_self_attributes']) ==
            {'config', 'embeddings', 'encoder', 'head', 'post_layernorm', 'use_head'},
            'observed direct constructor differs')
    return result


def tensor_fact(value):
    import torch
    require(value.device.type == 'cpu' and torch.isfinite(value).all().item(), 'CPU finite tensor required')
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    return {'dtype': str(value.dtype), 'shape': list(value.shape),
            'sha256': hashlib.sha256(memoryview(raw)).hexdigest()}


def numerical_flags():
    import torch
    return {'default_dtype': str(torch.get_default_dtype()), 'default_device': str(torch.get_default_device()),
            'grad_enabled': torch.is_grad_enabled(), 'inference_mode': torch.is_inference_mode_enabled(),
            'autocast_cpu': torch.is_autocast_enabled('cpu'), 'autocast_cuda': torch.is_autocast_enabled('cuda'),
            'autocast_cpu_dtype': str(torch.get_autocast_dtype('cpu')),
            'autocast_cuda_dtype': str(torch.get_autocast_dtype('cuda')),
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'cudnn_enabled': torch.backends.cudnn.enabled, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'sdpa_flash': torch.backends.cuda.flash_sdp_enabled(),
            'sdpa_math': torch.backends.cuda.math_sdp_enabled(),
            'sdpa_mem_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'sdpa_cudnn': torch.backends.cuda.cudnn_sdp_enabled()}


def loaded_origins(packages):
    for name, module in tuple(sys.modules.items()):
        package = name.split('.')[0]
        if package in packages and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(Path(packages[package]['root'])) and path.is_file(),
                    'loaded native module origin differs: ' + name)


def construct(config, context):
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel
    loaded_origins(context['packages'])
    require(module_origin(SiglipVisionModel, context['packages'])['file'] ==
            context['sources']['native_environment']['vision_constructor']['path'], 'loaded constructor origin differs')
    for cls in (SiglipVisionModel, SiglipVisionConfig):
        path = module_origin(cls, context['packages'])['file']
        require(path in context['guards'] and context['extract'].sha(path) == context['guards'][path],
                'loaded vision/config source hash differs')
    require(torch.get_default_dtype() == torch.float32 and str(torch.get_default_device()) == 'cpu',
            'FP32 CPU constructor defaults required')
    rng, flags = torch.random.get_rng_state().clone(), numerical_flags()
    resolved = SiglipVisionConfig.from_dict(config)
    resolved._attn_implementation = 'sdpa'
    with torch.random.fork_rng(devices=[]):
        model = SiglipVisionModel(resolved).float()
    require(torch.equal(rng, torch.random.get_rng_state()) and numerical_flags() == flags,
            'constructor changed CPU RNG/numerical flags')
    require(hasattr(model, 'embeddings') and hasattr(model, 'encoder') and hasattr(model, 'head') and
            not hasattr(model, 'vision_model'), 'installed direct vision layout required')
    return model


def configure_roles(model, expected, layers):
    parameters = list(model.named_parameters())
    require({name for name, _ in parameters} == expected.keys() and
            model.state_dict().keys() == expected.keys(), 'strict vision parameter/state inventory differs')
    boundary = layers - 12
    frozen = ('embeddings.',) + tuple(f'encoder.layers.{i}.' for i in range(boundary))
    rows = []
    for name, value in parameters:
        value.requires_grad_(not name.startswith(frozen))
        require(value.grad is None and list(value.shape) == expected[name] and str(value.dtype) == 'torch.float32' and
                value.device.type == 'cpu', 'vision parameter shape/dtype/device/grad differs: ' + name)
        rows.append({'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
                     'role': 'trainable' if value.requires_grad else 'frozen'})
    require(sum(row['role'] == 'trainable' for row in rows) == 205, '205 vision trainable members required')
    model.eval()
    return rows


def fresh_source(context):
    """Reusable pretrained source factory; caller supplies authenticated context.

    Native packages must have passed package_origins before the first import.
    No from_pretrained model loader is used: ONLY derived bare vision keys.
    """
    require('packages' in context, 'native origin preflight required before source factory')
    import torch
    from safetensors import safe_open
    from transformers import AutoImageProcessor
    model = construct(context['config']['vision_config'], context)
    with safe_open(str(context['source']), framework='pt', device='cpu') as disk:
        require(set(disk.keys()) == context['expected'].keys(), 'native source inventory differs')
        state = {}
        for name in disk.keys():
            value = disk.get_tensor(name)
            fact = tensor_fact(value)
            row = context['mapping'][name]
            require(fact == {'dtype': 'torch.float32', 'shape': row['shape'], 'sha256': row['sha256']},
                    'loaded source raw-byte provenance differs: ' + name)
            state[name] = value
        model.load_state_dict(state, strict=True)
        del state, value
    roles = configure_roles(model, context['expected'], model.config.num_hidden_layers)
    for name, value in model.state_dict().items():
        require(tensor_fact(value)['sha256'] == context['mapping'][name]['sha256'], 'copied source differs: ' + name)
    loaded_origins(context['packages'])
    require(module_origin(AutoImageProcessor, context['packages'])['file'] in context['guards'],
            'loaded processor factory origin differs')
    processor = AutoImageProcessor.from_pretrained(context['entry']['input']['preprocessor']['path'],
                                                  local_files_only=True, backend='torchvision')
    require(getattr(processor, 'backend', None) == 'torchvision', 'stock torchvision processor required')
    return model, processor, roles


def module_origin(cls, packages):
    path = Path(inspect.getfile(cls)).resolve()
    package = cls.__module__.split('.')[0]
    require(package in packages and path.is_relative_to(Path(packages[package]['root'])) and path.is_file(),
            'class module origin differs: ' + cls.__module__)
    return {'class': cls.__module__ + '.' + cls.__qualname__, 'file': str(path)}


def model_facts(model, processor, roles, packages):
    import torch
    modules = []
    for name, module in model.named_modules():
        require(not module.training and not module._forward_hooks and not module._forward_pre_hooks and
                not module._backward_hooks, 'source module flags/hooks differ')
        row = {'name': name, **module_origin(type(module), packages), 'training': module.training}
        row['attributes'] = {key: value for key, value in vars(module).items() if not key.startswith('_') and
                             (value is None or isinstance(value, (bool, int, float, str)) or
                              isinstance(value, (tuple, list)) and
                              all(isinstance(item, (bool, int, float, str)) for item in value))}
        if hasattr(module, 'config'):
            row['attn_implementation'] = module.config._attn_implementation
        if isinstance(module, torch.nn.LayerNorm):
            require(module.eps == model.config.layer_norm_eps, 'LayerNorm epsilon differs')
        require(not getattr(module, 'gradient_checkpointing', False), 'gradient checkpointing forbidden')
        modules.append(row)
    require(model.config._attn_implementation == 'sdpa', 'SDPA implementation required')
    parameters = list(model.named_parameters())
    require([row['name'] for row in roles] == [name for name, _ in parameters] and
            all(value.grad is None and value.requires_grad == (row['role'] == 'trainable')
                for row, (_, value) in zip(roles, parameters)), 'parameter roles/grads changed')
    buffers = dict(model.named_buffers())
    require(buffers.keys() == {'embeddings.position_ids'}, 'complete position buffer inventory required')
    position = buffers['embeddings.position_ids']
    require(position.dtype == torch.int64 and position.shape == (1, 256) and
            torch.equal(position, torch.arange(256).expand(1, -1)) and
            'position_ids' in model.embeddings._non_persistent_buffers_set, 'nonpersistent position_ids differs')
    processor_config = json.loads(processor.to_json_string())
    return json.loads(json.dumps({'config': model.config.to_dict(), 'attn_implementation': model.config._attn_implementation,
            'modules': modules, 'roles': roles,
            'vision': {name: tensor_fact(value) for name, value in model.state_dict().items()},
            'buffers': {name: {**tensor_fact(value), 'persistent': name in model.state_dict()}
                        for name, value in buffers.items()},
            'processor': {'origin': module_origin(type(processor), packages), 'config': processor_config,
                          'backend': processor.backend}}, allow_nan=False))


def pixels_and_raw(context, model, processor):
    import torch
    from PIL import Image
    images, rgb = [], []
    for path, row in zip(context['images'], context['fit']['rows'][:2]):
        require(context['extract'].sha(path) == row['image_sha256'], 'FIT image changed')
        with Image.open(path) as image:
            image = image.convert('RGB')
            rgb.append({'path': str(path), 'image_sha256': row['image_sha256'],
                        'mode': image.mode, 'size': list(image.size),
                        'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
            images.append(image)
    pixels = processor(images=images, return_tensors='pt')['pixel_values']
    for image in images:
        image.close()
    require(pixels.dtype == torch.float32 and pixels.shape == (2, 3, 256, 256), 'processed native256 pixels differ')
    pixel_fact = tensor_fact(pixels)
    with torch.no_grad():
        raw = model(pixel_values=pixels).pooler_output
    require(raw.dtype == torch.float32 and raw.shape == (2, model.config.hidden_size) and
            torch.all(torch.linalg.vector_norm(raw, dim=1) > 0).item(), 'raw source output shape/nonzero differs')
    return pixels, raw, {'images': rgb, 'pixels': pixel_fact, 'raw': tensor_fact(raw)}


def imported_origins(extract, packages):
    files, modules = {}, {}
    for name, module in tuple(sys.modules.items()):
        package = name.split('.')[0]
        if package not in packages or not getattr(module, '__file__', None):
            continue
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(Path(packages[package]['root'])) and path.is_file(),
                'loaded native module origin differs: ' + name)
        modules[name] = str(path)
        if str(path) not in files:
            files[str(path)] = extract.sha(path)
    # Include actual mapped extension/system libraries, not just Python stubs.
    native = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/') and '.so' in fields[5]:
            path = canonical(Path(fields[5]).resolve())
            native.add(str(path))
    for path in native:
        if path not in files:
            files[path] = extract.sha(path)
    return {'packages': packages, 'modules': modules, 'native_files': sorted(native), 'files': files}


def cgroup_memory():
    lines = Path('/proc/self/cgroup').read_text().splitlines()
    unified = [line[3:] for line in lines if line.startswith('0::')]
    require(len(unified) == 1 and unified[0] != '/', 'enclosing cgroup v2 unit required')
    root = Path('/sys/fs/cgroup') / unified[0].lstrip('/')
    require(root.resolve() == root and root.is_dir(), 'canonical cgroup required')
    values = {name: (root / name).read_text().strip() for name in
              ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current',
               'memory.swap.peak', 'memory.swap.max', 'memory.events')}
    require(values['memory.max'].isdigit() and int(values['memory.max']) == POLICY['host_bytes'] and
            all(values[name] == '0' for name in ('memory.swap.current', 'memory.swap.peak', 'memory.swap.max')) and
            values['memory.peak'].isdigit() and 0 < int(values['memory.peak']) <= POLICY['host_bytes'],
            'actual cgroup memory/swap caps or peak differ')
    events = dict(line.split() for line in values['memory.events'].splitlines())
    require(all(int(events.get(key, '-1')) == 0 for key in ('oom', 'oom_kill', 'max')), 'cgroup memory failure event')
    return {'path': str(root), 'values': values}


def rehash(context):
    require(fit_rows(context['extract'], context['fit']) == context['images'], 'FIT image resolution changed')
    for path, expected in context['guards'].items():
        require(context['extract'].sha(canonical(path)) == expected, 'exit authority SHA256 differs: ' + path)
    require(bootstrap(context['root'], context['args'].execution_sha256)[1] == context['code'], 'exit closure differs')


def qualify(args):
    started = time.perf_counter()
    context = authority(args)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be explicitly hidden')
    before_memory = cgroup_memory()
    packages = package_origins(context)
    context['packages'] = packages
    args.output.mkdir()  # Atomic refusal if another writer won after preflight.
    import torch
    rng, flags = torch.random.get_rng_state().clone(), numerical_flags()
    model, processor, roles = fresh_source(context)
    pixels, raw, sample = pixels_and_raw(context, model, processor)
    first = model_facts(model, processor, roles, packages)
    require(torch.equal(rng, torch.random.get_rng_state()) and flags == numerical_flags(), 'source changed RNG/flags')
    checkpoint = args.output / 'fresh_vision.pt'
    with context['extract'].exclusive(checkpoint) as stream:
        torch.save({'vision': model.state_dict(), 'buffers': dict(model.named_buffers()),
                    'config': first['config'], 'runtime': first, 'cpu_rng': rng}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    checkpoint_sha = context['extract'].sha(checkpoint)
    context['guards'][str(checkpoint)] = checkpoint_sha
    # No tensor/model references survive besides the two tiny witness outputs.
    del model, processor
    gc.collect()
    disk = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    require(disk.keys() == {'vision', 'buffers', 'config', 'runtime', 'cpu_rng'} and
            disk['config'] == first['config'] and disk['runtime'] == first and torch.equal(disk['cpu_rng'], rng),
            'complete serialized fresh source differs')
    independent = construct(disk['config'], context)
    independent.load_state_dict(disk['vision'], strict=True)
    independent_buffers = dict(independent.named_buffers())
    require(independent_buffers.keys() == disk['buffers'].keys(), 'independent buffer inventory differs')
    with torch.no_grad():
        for name, value in independent_buffers.items():
            require(tensor_fact(value) == tensor_fact(disk['buffers'][name]), 'fresh independent buffer differs: ' + name)
            value.copy_(disk['buffers'][name])
    independent_roles = configure_roles(independent, context['expected'], independent.config.num_hidden_layers)
    del disk, independent_buffers, value
    gc.collect()
    from transformers import AutoImageProcessor
    independent_processor = AutoImageProcessor.from_pretrained(context['entry']['input']['preprocessor']['path'],
                                                               local_files_only=True, backend='torchvision')
    pixels2, raw2, sample2 = pixels_and_raw(context, independent, independent_processor)
    second = model_facts(independent, independent_processor, independent_roles, packages)
    require(first == second and sample == sample2 and torch.equal(pixels, pixels2) and torch.equal(raw, raw2),
            'independent exact source/runtime/pixels/raw reload differs')
    require(torch.equal(rng, torch.random.get_rng_state()) and numerical_flags() == flags, 'reload changed RNG/flags')
    del independent, independent_processor, pixels2, raw2
    gc.collect()
    origins = imported_origins(context['extract'], packages)
    for path, expected in origins['files'].items():
        require(context['guards'].setdefault(path, expected) == expected, 'loaded origin hash changed: ' + path)
    rehash(context)
    after_memory = cgroup_memory()
    require(after_memory['path'] == before_memory['path'], 'enclosing unit changed')
    wall = time.perf_counter() - started
    require(wall < POLICY['seconds'], 'source CPU qualification exceeds 120s')
    interpreter = canonical(Path(sys.executable).resolve())
    proof = {'schema': SCHEMA, 'pass': True, 'source_qualified': True, 'model_qualified': False, 'initializer_qualified': False,
             'training_qualified': False, 'quality_qualified': False, 'quality_read': False,
             'arm': args.arm, 'source_model': context['entry']['source_model'], 'revision': context['entry']['revision'],
             'execution_sha256': args.execution_sha256, 'code': context['code'],
             'sources_sha256': args.sources_sha256, 'fit_manifest_sha256': args.fit_manifest_sha256,
             'selected_source': context['entry'], 'original_extraction_resource_note': context['sources']['resource_note'],
             'input_guards': context['guards'], 'runtime': first, 'origins': origins, 'sample': sample,
             'reload_exact': True, 'exit_rehash_pass': True, 'constructor_rng_preserved': True, 'cpu_rng': tensor_fact(rng),
             'numerical_flags': flags, 'gradients_created': False, 'optimizer_created': False, 'updates': 0,
             'checkpoint': {'path': str(checkpoint), 'sha256': checkpoint_sha},
             'resource_policy': POLICY, 'cgroup_before': before_memory, 'cgroup_after': after_memory,
             'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'wall_seconds': wall,
             'invocation': {'argv': sys.argv, 'python': str(interpreter), 'python_sha256': context['extract'].sha(interpreter),
                            'python_version': sys.version, 'optimize': sys.flags.optimize, 'pid': os.getpid(),
                            'invocation_id': os.environ.get('INVOCATION_ID'), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
             'terminal_exit_and_both_locks_require_parent_receipt': True}
    require(re.fullmatch('[0-9a-f]{32}', proof['invocation']['invocation_id'] or '') is not None,
            'original systemd invocation ID required')
    with context['extract'].exclusive(args.output / 'proof.json') as stream:
        stream.write((json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    return proof


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for field in ('execution-sha256', 'sources-sha256', 'fit-manifest-sha256'):
        result.add_argument('--' + field, required=True)
    for field in ('sources', 'fit-manifest', 'output'):
        result.add_argument('--' + field, type=Path, required=True)
    result.add_argument('--arm', choices=sorted(ARMS), required=True)
    return result


def main():
    try:
        proof = qualify(parser().parse_args())
    except (OSError, ValueError, ImportError, KeyError, TypeError, RuntimeError) as error:
        raise SystemExit('source qualification rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'source_qualified': proof['source_qualified'],
                      'proof': str(Path(proof['checkpoint']['path']).parent / 'proof.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
