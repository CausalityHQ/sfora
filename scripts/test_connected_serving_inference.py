#!/usr/bin/env python3
"""Source-only integrated artifact reader checks; native parity remains unqualified."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-installed-reader-preparation-v1'
FIXTURES = EVIDENCE / 'assembled-candidate'
RUNTIME = ROOT / 'src/sfora/connected_inference.py'


def main():
    if not __debug__:
        raise SystemExit('source checks require assertions')
    spec = importlib.util.spec_from_file_location('_serving_inverse_oracle', ROOT / 'scripts/test_connected_inference_extraction.py')
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    source = RUNTIME.read_text()
    legacy = oracle.serving_reader_inverse(source)
    record = oracle.serving_reader_record()
    for name, current, before, after in (
        ('_connected_inference_authority.py', 'authority_sha256', 'runtime_sha256', 'base_runtime_sha256'),
        ('connected_compact_serving.py', 'bridge_sha256', 'authority_sha256', 'base_authority_sha256'),
    ):
        raw = (ROOT / 'src/sfora' / name).read_text()
        if name == 'connected_compact_serving.py':
            raw = oracle.artifact_factory_inverse(raw)
        assert hashlib.sha256(raw.encode()).hexdigest() == record[current]
        restored = raw.replace(record[before], record[after])
        assert hashlib.sha256(restored.encode()).hexdigest() == record['base_'+current]
    for old, new in (('return _serving_release(endpoint)', 'return None'),
                     ('CUDA', 'ALTERED'), ('def _serving_metadata(', 'def _forged_metadata(')):
        mutant = source.replace(old, new, 1)
        if mutant == source:
            continue
        try:
            restored = oracle.serving_reader_inverse(mutant)
            assert hashlib.sha256(restored.encode()).hexdigest() == record['base_runtime_sha256']
        except AssertionError:
            pass
        else:
            raise AssertionError('whole source inverse accepted mutation')
    definitions = {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    checks = ('preparation', 'lifecycle', 'failed-load', 'payload-flow', 'public-load', 'request-prelude', 'release-setup', 'constructor', 'metadata-prototype', 'payload-identity')
    with tempfile.TemporaryDirectory(prefix='integrated-serving-tests-') as tmp:
        stage = Path(tmp)
        legacy_path = stage / 'legacy-runtime.py'
        legacy_path.write_text(legacy)
        for fixture in FIXTURES.glob('*.py'):
            if fixture.name.endswith('-check.py') or fixture.name == 'sfora-serving-prepare-fixture.py':
                raw = fixture.read_text().replace("root=Path('/home/rb/worktrees/sfora-positive-causality')", 'root=Path('+repr(str(ROOT))+')')
                if fixture.name in ('sfora-serving-lifecycle-check.py', 'sfora-serving-request-prelude-check.py'):
                    raw = raw.replace("(root/'src/sfora/connected_inference.py').read_text()", 'Path('+repr(str(legacy_path))+').read_text()')
                    raw = raw.replace("Path('src/sfora/connected_inference.py').read_text()", 'Path('+repr(str(legacy_path))+').read_text()')
                if fixture.name == 'sfora-serving-preparation-check.py':
                    lines = raw.splitlines(keepends=True)
                    i = next(i for i,line in enumerate(lines) if "raw=(root/'src/sfora/connected_inference.py').read_bytes()" in line)
                    lines[i] = "        raw=(root/'src/sfora/connected_inference.py').read_bytes()\n"
                    raw = ''.join(lines)
                if fixture.name == 'sfora-serving-prepare-fixture.py':
                    marker = '    checker_function = owner._check_current.__func__\n'
                    assert raw.count(marker) == 1
                    raw = raw.replace(marker, (EVIDENCE / 'exit-graph-fixture-check.py').read_text()+marker)
                (stage / fixture.name).write_text(raw)
            elif fixture.name != 'connected_inference_candidate.py':
                names = [n.name for n in ast.parse(fixture.read_text()).body if isinstance(n, ast.FunctionDef)]
                for name in names:
                    expected = next(n for n in ast.parse(fixture.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
                    if name == '_serving_release':
                        text = ast.get_source_segment(fixture.read_text(), expected)
                        for before, after in record['release_setup_replacements']:
                            assert text.count(before) == 1
                            text = text.replace(before, after)
                        expected = ast.parse(text).body[0]
                    if name in record.get('prototype_replacements', {}):
                        text = ast.get_source_segment(fixture.read_text(), next(n for n in ast.parse(fixture.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name))
                        if name == '_serving_release':
                            for before, after in record['release_setup_replacements']:
                                text = text.replace(before, after)
                        for before, after in record['prototype_replacements'][name]:
                            assert text.count(before) == 1
                            text = text.replace(before, after)
                        expected = ast.parse(text).body[0]
                    actual = ast.parse(definitions[name]).body[0]
                    assert ast.dump(expected) == ast.dump(actual), name
                imports = [ast.get_source_segment(fixture.read_text(), n) for n in ast.parse(fixture.read_text()).body if isinstance(n, (ast.Import, ast.ImportFrom))]
                (stage / fixture.name).write_text('\n'.join(imports)+'\n\n'+'\n\n'.join(definitions[name] for name in names)+'\n')
        (stage / 'sfora-serving-release-setup-check.py').write_bytes((EVIDENCE / 'sfora-serving-release-setup-check.py').read_bytes())
        lifecycle = stage / 'sfora-serving-lifecycle-check.py'
        text = lifecycle.read_text().replace("_processor_cache=cache)", "_processor_cache=cache,_serving_exit_callables=lambda context:(artifact,environment))")
        lifecycle.write_text(text)
        failed = stage / 'sfora-serving-failed-load-check.py'
        failed.write_text(failed.read_text().replace("_serving_full_exit=lambda context:", "_serving_full_exit=lambda context, primary=None:"))
        wrapper = stage / 'sfora-serving-public-load-check.py'
        text = wrapper.read_text().replace('def failed(context,error):', 'def failed(context,error,*,exit_attempted=False):')
        text = text.replace("assert context is prepared and error is primary;events.append('failed_exit');raise error", "assert context is prepared and error is primary and exit_attempted is (mode!='payload');events.append('failed_exit');raise error")
        wrapper.write_text(text)
        for kind in ('constructor', 'metadata-prototype', 'payload-identity'):
            filename = 'sfora-serving-'+kind+'-check.py'
            text = (EVIDENCE / filename).read_text().replace("root=Path('/home/rb/worktrees/sfora-positive-causality')", 'root=Path('+repr(str(ROOT))+')')
            if kind == 'constructor':
                text = text.replace("(root/'src/sfora/connected_inference.py').read_text()", 'Path('+repr(str(legacy_path))+').read_text()')
            (stage / filename).write_text(text)
        payload = stage / 'sfora-serving-payload-flow-check.py'
        text = payload.read_text().replace("assert prepared['origin']['encoder_identity']=={'old_file':True}", "assert prepared['origin']['encoder_identity']=={'old_file':True}\n        assert endpoint['encoder_identity'] is not endpoint['manifest']['encoder_identity']\n        assert endpoint['encoder_identity']['runtime'] is not endpoint['manifest']['encoder_identity']['runtime']")
        payload.write_text(text)
        for check in checks:
            result = subprocess.run([sys.executable, '-B', str(stage / ('sfora-serving-'+check+'-check.py'))],
                cwd=ROOT, capture_output=True, text=True, timeout=15,
                preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (1073741824,1073741824)))
            assert result.returncode == 0, check+'\n'+result.stdout+result.stderr
            print(result.stdout.strip())
    namespace = {'json':json}
    exec(compile(definitions['_serving_exact_json'], '<integrated-exact-json>', 'exec'), namespace)
    exact = namespace['_serving_exact_json']
    assert exact({'a':1,'b':[2]}, {'b':[2],'a':1})
    for a,b in ((False,0),(True,1),(1,1.0),(-0.0,0.0),([1,2],[2,1])):
        assert not exact(a,b), (a,b)
    assert not any(name.split('.')[0] in {'torch','numpy','PIL','transformers'} for name in sys.modules)
    print('PASS integrated reader definitions, exact legacy inverse and typed live equality; native UNRUN')


if __name__ == '__main__':
    main()
