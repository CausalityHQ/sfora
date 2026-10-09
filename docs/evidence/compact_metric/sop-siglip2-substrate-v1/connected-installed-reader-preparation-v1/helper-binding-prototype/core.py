"""Draft native-free installed-reader preparation; not integrated or released."""
import hashlib
import json
import sys
from pathlib import Path
from types import MethodType, ModuleType


def _serving_helper_binding(binding):
    require(type(binding) is tuple and len(binding) == 3, 'exact serving helper binding required')
    modules, guards, checker = binding
    names = ('connected_gallery_provenance', 'connected_serving_artifact', 'connected_serving_admission', 'connected_artifact_identity', 'connected_installed_environment')
    require(type(modules) is tuple and len(modules) == 5 and type(guards) is tuple and len(guards) == 5, 'complete helper source closure required')
    bridge = sys.modules.get('sfora.connected_compact_serving')
    require(bridge is not None and type(checker) is MethodType, 'genuine installed bridge checker required')
    owner = checker.__self__
    require(type(owner) is bridge.ConnectedCompactIndex and checker.__func__ is bridge.ConnectedCompactIndex._check_current and checker.__func__.__globals__ is vars(bridge), 'genuine bridge owner/method required')
    require(owner._module is sys.modules[__name__], 'bridge must own this runtime')
    root = Path(__file__).parent
    require(bridge.__file__ == str(root / 'connected_compact_serving.py') and bridge.__spec__.origin == bridge.__file__, 'bridge source sibling differs')
    require(any(row[0] is bridge for row in owner._namespaces) and any(row[0] is type(owner) for row in owner._namespaces) and any(row[0] is checker.__func__ for row in owner._callables), 'bridge checker snapshot coverage missing')
    snapshots = [row for row in owner._callables if row[0] is checker.__func__]
    require(len(snapshots) == 1, 'one authenticated checker snapshot required')
    saved = snapshots[0]
    require(checker.__func__.__code__ is saved[1] and checker.__func__.__defaults__ is saved[2] is None and checker.__func__.__kwdefaults__ is saved[3] is None and checker.__func__.__closure__ is saved[4] is None, 'bridge checker executable identity changed')
    for name, module, guard in zip(names, modules, guards, strict=True):
        require(type(module) is ModuleType and type(guard) is tuple and len(guard) == 2, 'exact helper module/guard required')
        path, sha = guard
        require(type(path) is str and path == str(root / (name + '.py')) and module.__file__ == path and module.__spec__.origin == path and module.__name__.rsplit('.', 1)[-1] == name, 'helper installed origin differs')
        require(sys.modules.get(module.__name__) is module and owner._owned.get(module.__name__) is module and any(row[0] is module for row in owner._namespaces), 'helper ownership/snapshot missing')
        bound_file({}, path, sha)
    require(checker() is None, 'genuine bridge checker return differs')
    return modules, guards, checker


def _serving_json(raw):
    value = strict_json(raw)
    # Canonical encoding preserves bool/int/float distinctions and rejects overflow.
    json.dumps(value, allow_nan=False)
    return value


def _serving_prepare(directory, *, trusted_serving_sha256, trusted_fragment_sha256, installed_environment, trusted_installed_environment_sha256, serving_helpers):
    require(type(directory) is str and Path(directory).is_absolute(), 'absolute serving directory required')
    require(type(installed_environment) is bytes and len(installed_environment) <= 64 * 1024**2 and type(trusted_installed_environment_sha256) is str and hashlib.sha256(installed_environment).hexdigest() == trusted_installed_environment_sha256, 'independent installed environment bytes differ')
    _check_runtime()
    modules, helper_guards, checker = _serving_helper_binding(serving_helpers)
    expected = _serving_json(installed_environment)
    require(type(expected) is dict and expected.keys() == {'schema', 'original_bundle_sha256', 'original_ownership_audit_sha256', 'site_packages', 'distributions', 'expected_environment'}, 'exact installed environment authority required')
    admitted = modules[2].admit_serving_artifact(directory, trusted_serving_sha256=trusted_serving_sha256, trusted_fragment_sha256=trusted_fragment_sha256)
    require(expected['original_bundle_sha256'] == trusted_fragment_sha256['origin.json'] and expected['original_ownership_audit_sha256'] == trusted_fragment_sha256['origin-owners.json'], 'installed authority original pins differ')
    origin_raw = _serving_metadata(Path(directory) / 'origin.json', trusted_fragment_sha256['origin.json'])
    owners_raw = _serving_metadata(Path(directory) / 'origin-owners.json', trusted_fragment_sha256['origin-owners.json'])
    origin = _serving_json(origin_raw)
    require(type(origin) is dict and origin.keys() == {'schema', 'code', 'files', 'endpoint_state_sha256', 'environment', 'encoder_identity', 'base_vision_sha256', 'vision_sha256', 'scope'} and origin['schema'] == BUNDLE_SCHEMA and tuple(sorted(origin['code'].items())) == _binding[0], 'original inference authority differs')
    actual = modules[4].verify_installed_environment(origin_raw, owners_raw, trusted_bundle_sha256=trusted_fragment_sha256['origin.json'], trusted_ownership_audit_sha256=trusted_fragment_sha256['origin-owners.json'], site_packages=expected['site_packages'])
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(canonical(actual) == canonical(expected), 'fresh installed environment differs from independent authority')
    env = origin['environment']
    require(type(env) is dict and env.keys() == {'packages', 'files', 'native_files', 'vision_constructor'} and env['packages'].keys() == NATIVE - {'sfora'} and env['vision_constructor'] in env['files'] and set(env['native_files']) <= set(env['files']), 'original environment shape differs')
    sites = {Path(row['root']).parent for row in env['packages'].values()}
    require(len(sites) == 1, 'one original site root required')
    site = next(iter(sites))
    anchors = {path: sha for path, sha in env['files'].items() if not Path(path).is_relative_to(site)}
    require(all(env['native_files'].get(path) == sha and actual['expected_environment']['files'].get(path) == sha and actual['expected_environment']['native_files'].get(path) == sha for path, sha in anchors.items()), 'external anchor association differs')
    guards = {}
    batch_bound_files(guards, anchors.items())
    for name, fact in admitted['files'].items():
        guards[fact['path']] = fact['sha256']
    for path, sha in helper_guards:
        bound_file(guards, path, sha)
    require(checker() is None, 'bridge changed during native-free admission')
    return {'admitted': admitted, 'origin': origin, 'origin_raw': origin_raw, 'owners_raw': owners_raw, 'installed': actual, 'installed_raw': installed_environment, 'installed_sha256': trusted_installed_environment_sha256, 'helpers': modules, 'helper_guards': helper_guards, 'checker': checker, 'guards': guards}
