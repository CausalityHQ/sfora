# Executed inside the genuine binding fixture before mutations.
import json
sys.path.insert(0, str(root/'scripts'))
try:
    from test_connected_installed_environment import EnvironmentTests
    from test_connected_gallery_provenance import fixture, encoded, sha
finally:
    sys.path.pop(0)
environment = EnvironmentTests('test_valid_complete_inverse_and_no_original_or_unselected_reads')
environment.setUp()
try:
    receipt, bundle = fixture(33)
    old_code = bundle['code']
    bundle['code'] = dict(historical)
    bundle['environment'] = environment.environment
    anchor = folder/'lib-anchor.so'
    anchor.write_bytes(b'opaque synthetic external anchor')
    old_anchor = environment.anchor
    for kind in ('files', 'native_files'):
        del bundle['environment'][kind][old_anchor]
        bundle['environment'][kind][str(anchor)] = sha(anchor.read_bytes())
    original = encoded(bundle)
    bundle_path = receipt['payload_facts']['bundle']['path']
    def update_descriptors(value):
        if type(value) is dict:
            if value.get('path') == bundle_path:
                value['sha256'] = sha(original)
            for child in value.values(): update_descriptors(child)
        elif type(value) is list:
            for child in value: update_descriptors(child)
    update_descriptors(receipt)
    receipt['binding']['training']['code'] = {name:bundle['code'][name] for name in old_code}
    receipt['input_guards'][bundle_path] = sha(original)
    prefix = bundle_path.rsplit('/',1)[0]+'/'
    for name in old_code: del receipt['input_guards'][prefix+name]
    receipt['input_guards'].update({prefix+name:value for name,value in bundle['code'].items()})
    wire = b''.join(i.to_bytes(2,'little')*65 for i in range(66))
    receipt['files']['control-179061.packed.bin'] = sha(wire)
    export = encoded(receipt)
    owners_raw = encoded(environment.audit)
    converter,_,_ = load('_draft_helpers.connected_gallery_conversion','connected_gallery_conversion.py',(root/'src/sfora/connected_gallery_conversion.py').read_bytes())
    fragment = converter.extract_gallery_members(export, original, owners_raw, wire, trusted_receipt_sha256=sha(export), trusted_bundle_sha256=sha(original), trusted_ownership_audit_sha256=sha(owners_raw))
    pins = {name:sha(raw) for name,raw in fragment.items()}
    producer = helpers[0].bind_gallery_provenance(export, original, trusted_receipt_sha256=sha(export), trusted_bundle_sha256=sha(original))
    payload = {'vision.pt':b'synthetic metadata, never native proof','endpoint.pt':b'synthetic metadata, never native proof','processor.json':b'file'}
    manifest = {'schema':'siglip2-connected-mlp-serving-v2','files':{name:{'path':name,'bytes':len(raw),'sha256':sha(raw)} for name,raw in {**fragment,**payload}.items()},'origin':{'bundle_sha256':sha(original),'endpoint_state_sha256':bundle['endpoint_state_sha256'],'fixed_sha256':producer['identity']['fixed_sha256']},'gallery':{'count':33,'dimensions':128,'bytes_per_row':130,'wire_sha256':pins['gallery.bin'],'ordered_ids_sha256':pins['gallery-ids.json'],'provenance_sha256':pins['gallery-provenance.json'],'encoder_binding_sha256':sha(encoded({key:producer[key] for key in ('identity','gallery_batches','gallery_rows')}))},'request':{'device':'cuda','batch_min':1,'batch_max':32,'dimensions':128,'k':10,'ties':'ordinal-ascending'}}
    serving = encoded(manifest)
    directory = folder/'serving'; directory.mkdir()
    for name,raw in {'serving.json':serving,**fragment,**payload}.items(): (directory/name).write_bytes(raw)
    expected = helpers[4].verify_installed_environment(original,owners_raw,trusted_bundle_sha256=sha(original),trusted_ownership_audit_sha256=sha(owners_raw),site_packages=str(environment.target))
    authority = encoded(expected)
    kwargs = dict(trusted_serving_sha256=sha(serving),trusted_fragment_sha256=pins,installed_environment=authority,trusted_installed_environment_sha256=sha(authority),serving_helpers=binding)
    prepared = runtime._serving_prepare(str(directory),**kwargs)
    assert prepared['installed']==expected and prepared['guards'][str(anchor)]==sha(anchor.read_bytes())
    assert len(prepared['admitted']['files'])==10
    for path in (directory/'endpoint.pt', directory/'origin.json', environment.target/'torch/__init__.py'):
        saved = path.read_bytes()
        path.write_bytes(saved + b'changed')
        try:
            try: runtime._serving_prepare(str(directory), **kwargs)
            except ValueError: pass
            else: raise AssertionError('current-byte mutation accepted: ' + str(path))
        finally:
            path.write_bytes(saved)
    forged = json.loads(authority)
    forged['expected_environment']['packages']['torch']['version'] = 'forged'
    forged_raw = encoded(forged)
    try:
        runtime._serving_prepare(str(directory), **{**kwargs, 'installed_environment':forged_raw, 'trusted_installed_environment_sha256':sha(forged_raw)})
    except ValueError: pass
    else: raise AssertionError('independently repinned false installed authority accepted')
    prepared_again = runtime._serving_prepare(str(directory), **kwargs)
    assert prepared_again['installed'] == prepared['installed']
    checker_function = owner._check_current.__func__
    checker_code = checker_function.__code__
    changed_during_admission = []
    runtime_lines = runtime_path.read_text().splitlines()
    def trace_checker(frame, event, arg):
        if event == 'line' and frame.f_code.co_name == '_serving_prepare' and ("bridge changed during native-free admission" in runtime_lines[frame.f_lineno-1] or runtime_lines[frame.f_lineno-1].strip() == "_serving_helper_binding(serving_helpers)"):
            checker_function.__code__ = (lambda self: None).__code__
            changed_during_admission.append(True)
        return trace_checker
    sys.settrace(trace_checker)
    try:
        try: runtime._serving_prepare(str(directory), **kwargs)
        except ValueError: pass
        else: raise AssertionError('checker changed during preparation was accepted')
    finally:
        sys.settrace(None)
        checker_function.__code__ = checker_code
    assert changed_during_admission
    anchor.write_bytes(b'opaque synthetic external mutant')
    try: runtime._serving_prepare(str(directory),**kwargs)
    except ValueError: pass
    else: raise AssertionError('external anchor mutation accepted')
    print('PASS: real ten-file admission + installed RECORD verification + bounded metadata + fresh external anchor; payload/metadata/wheel/authority/anchor mutations reject; fresh restored composition passes; native UNRUN')
finally:
    environment.doCleanups()
