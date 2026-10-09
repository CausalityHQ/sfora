"""Original predicate AST plus genuine JSON correspondence; no tensor/native proof."""
import ast, copy, hashlib, importlib.util, json, sys
from pathlib import Path
root=next(parent for parent in Path(__file__).resolve().parents if (parent/'src/sfora/connected_inference.py').is_file())
sys.path.insert(0,str(root/'scripts'))
from test_connected_installed_environment import EnvironmentTests
sys.path.pop(0)
source=(root/'src/sfora/connected_inference.py').read_text()
original=ast.parse(source)
loader=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='load_inference')
expected=next(n for n in loader.body if isinstance(n,ast.Try)).body[0]
raw=Path(__file__).with_name('candidate.py').read_text()
functions=ast.parse(raw)
validator=next(n for n in functions.body if isinstance(n,ast.FunctionDef) and n.name=='_serving_validate_payload')
assert ast.dump(validator.body[0],include_attributes=False)==ast.dump(expected,include_attributes=False)
# Original runtime loads no native modules at module scope. Read its constants only.
constants={ '__file__':str(root/'src/sfora/connected_inference.py'), '__name__':'_payload_constants'}
exec(compile(source,constants['__file__'],'exec',dont_inherit=True),constants)
def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
flags={'synthetic_flags':True}
namespace={key:constants[key] for key in ('require','INFERENCE_KEYS','INFERENCE_SCHEMA','ARMS','CONTROL_SHA256')}
namespace.update(fingerprint=fingerprint,numerical_flags=lambda:copy.deepcopy(flags),_serving_helper_binding=lambda binding: None)
exec(compile(raw,'payload-candidate','exec',dont_inherit=True),namespace)
spec=importlib.util.spec_from_file_location('_identity_candidate_check',root/'src/sfora/connected_artifact_identity.py')
identity=importlib.util.module_from_spec(spec);sys.modules[spec.name]=identity;spec.loader.exec_module(identity)
environment=EnvironmentTests('test_valid_complete_inverse_and_no_original_or_unselected_reads');environment.setUp()
try:
    origin_model={'inventory':{'fake.weight':[1]},'runtime':{'modules':[{'class':'transformers.model.Vision','file':environment.old+'/transformers/model.py','attributes':{'enabled':True,'count':1,'scale':1.0}}]}}
    processor={'origin':{'class':'transformers.model.Processor','file':environment.old+'/transformers/model.py'},'configuration':{'size':256}}
    disk=dict.fromkeys(namespace['INFERENCE_KEYS'],None)
    disk.update(schema=namespace['INFERENCE_SCHEMA'],arm='control',encoder_identity=origin_model,processor=processor,vision_sha256='2'*64,base_vision={'sha256':'3'*64},scope={'arm':'control','payload':{'scope_sha256':namespace['CONTROL_SHA256'],'class_names':['synthetic']*1008}},numerical_flags=flags)
    disk['fixed_sha256']=fingerprint({k:v for k,v in disk.items() if k!='fixed_sha256'})
    manifest={**environment.bundle,'endpoint_state_sha256':fingerprint(disk),'encoder_identity':origin_model,'vision_sha256':disk['vision_sha256'],'base_vision_sha256':disk['base_vision']['sha256']}
    installed=environment.verify(bundle=manifest)
    prepared={'origin':manifest,'installed':installed,'helpers':(None,None,None,identity,None),'helper_guards':(),'checker':None}
    before=copy.deepcopy((disk,manifest,prepared['installed']))
    result=namespace['_serving_expected_identities'](prepared,disk)
    target=str(environment.target/'transformers/model.py')
    assert result['model']['runtime']['modules'][0]['file']==target
    assert result['processor']['origin']['file']==target
    assert result['model']['runtime']['modules'][0]['attributes']==origin_model['runtime']['modules'][0]['attributes']
    assert before==(disk,manifest,prepared['installed'])
    for mutation in ('extra_key','schema','arm','fixed','flags','vision','identity','base','scope_arm','scope_sha','scope_count'):
        bad=copy.deepcopy(disk)
        if mutation=='extra_key': bad['extra']=None
        elif mutation=='schema': bad['schema']='foreign'
        elif mutation=='arm': bad['arm']='foreign'
        elif mutation=='fixed': bad['fixed_sha256']='0'*64
        elif mutation=='flags': bad['numerical_flags']={}
        elif mutation=='vision': bad['vision_sha256']='0'*64
        elif mutation=='identity': bad['encoder_identity']['inventory']={}
        elif mutation=='base': bad['base_vision']['sha256']='0'*64
        elif mutation=='scope_arm': bad['scope']['arm']='candidate'
        elif mutation=='scope_sha': bad['scope']['payload']['scope_sha256']='0'*64
        else: bad['scope']['payload']['class_names'].pop()
        if mutation!='fixed': bad['fixed_sha256']=fingerprint({k:v for k,v in bad.items() if k!='fixed_sha256'})
        altered={**prepared,'origin':{**manifest,'endpoint_state_sha256':fingerprint(bad)}}
        try: namespace['_serving_expected_identities'](altered,bad)
        except ValueError as error: assert 'complete original-scope/updated inference identity differs' in str(error)
        else: raise AssertionError(mutation+' accepted')
    selected=environment.target/'transformers/model.py';saved=selected.read_bytes();selected.write_bytes(saved+b'changed')
    try:
        try: namespace['_serving_expected_identities'](prepared,disk)
        except ValueError as error: assert 'installed source hash differs' in str(error)
        else: raise AssertionError('installed identity source mutation accepted')
    finally: selected.write_bytes(saved)
    assert namespace['_serving_expected_identities'](prepared,disk)==result
    assert not any(name.split('.')[0] in {'torch','numpy','PIL','transformers'} for name in sys.modules)
    print('PASS original full-payload predicate AST; 11 independently repinned clause negatives; genuine detached expected origin projection/materialization; source mutation reject; native and typed-tensor proof UNRUN')
finally:
    environment.doCleanups()
    if sys.modules.get(spec.name) is identity: del sys.modules[spec.name]
