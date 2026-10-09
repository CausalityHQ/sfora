"""Execute candidate's real loader body with fake Tensor/factory dependencies."""
import ast, copy, gc, json, sys, tempfile, types, weakref
from pathlib import Path
root=Path('/home/rb/worktrees/sfora-positive-causality')
class Leaf:
    def __init__(self,value=None,requires_grad=False):self.value=value;self.requires_grad=requires_grad
class Model:
    def requires_grad_(self,*args):return self
    def eval(self):return self
    def train(self):return self
    def to(self,*args):return self
    def state_dict(self):return {'opaque_weight':True}
    def parameters(self):return ()
    def buffers(self):return ()
class Processor:pass
class Pages:
    def __init__(self,stream):pass
    def copy(self,value,device):return Leaf(value.value)
class Cache:
    def cache_clear(self):pass
    def cache_info(self):return types.SimpleNamespace(currsize=0)
def require(ok,message):
    if not ok:raise ValueError(message)
with tempfile.TemporaryDirectory() as tmp:
    folder=Path(tmp);(folder/'endpoint.pt').write_bytes(b'opaque-not-a-torch-payload')
    events=[];expected={'model':{'runtime':{'expected_installed':True}},'processor':{'expected_installed':True}}
    disk={'base_vision':{'sha256':'base'},'config':{},'buffers':{},'encoder':{},'vision_sha256':'vision','head':{},'A':Leaf('A'),'C':Leaf('C'),'means':{},'mu_train':Leaf('mu'),'common_statistics':{},'processor':{'old_file':True},'arm':'control','scope':{},'mu_train_provenance':{},'encoder_identity':{'old_file':True},'numerical_flags':{}}
    prepared={'directory':str(folder),'origin':{'files':{'vision.pt':'filehash'},'vision_sha256':'vision','encoder_identity':{'old_file':True}},'installed':{'expected_environment':{'packages':{},'vision_constructor':'/synthetic-constructor'}},'guards':{}}
    def construct(context,*args):
        assert context['_serving'] is prepared and context['packages']=={}
        events.append('constructor');return Model(),Processor(),Cache(),{},expected['model']['runtime']
    def fingerprint(value):return 'vision' if value=={'opaque_weight':True} else 'readout'
    def expected_identities(context,value):
        assert context is prepared and value is disk
        events.append('authenticated expected identities');return copy.deepcopy(expected)
    ns={'sys':sys,'__name__':'_payload_flow_test','Path':Path,'copy':copy,'gc':gc,'require':require,'_serving_expected_identities':expected_identities,'construct_encoder':construct,'fingerprint':fingerprint,'CheckpointPages':Pages,'head_from':lambda *args,**kwargs:Model(),'inference_readout_tree':lambda endpoint:{'readout':True},'encoder_facts':lambda *args,**kwargs:{'vision_sha256':'vision'},'_serving_exact_json':lambda a,b:json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True),'_serving_live_identity':lambda endpoint:events.append('live identity'),'mapping_absent':lambda path:events.append('mapping'),'owned_copy':lambda value,pages,device:copy.deepcopy(value)}
    module=types.ModuleType(ns['__name__']);module.__dict__.update(ns);ns=module.__dict__;sys.modules[module.__name__]=module
    exec(compile(Path(__file__).with_name('sfora-serving-load-payload.py').read_text(),'candidate-loader-body','exec',dont_inherit=True),ns)
    torch=types.ModuleType('torch');torch.load=lambda *args,**kwargs:disk;torch.nn=types.SimpleNamespace(Parameter=Leaf);assert 'torch' not in sys.modules;sys.modules['torch']=torch
    try:
        endpoint=ns['_serving_load_payload'](prepared)
        assert endpoint['_serving'] is prepared and endpoint['processor']==expected['processor'] and endpoint['encoder_identity']==expected['model']
        assert prepared['origin']['encoder_identity']=={'old_file':True}
        assert endpoint['manifest']['environment']==prepared['installed']['expected_environment']
        assert endpoint['A'].requires_grad and endpoint['C'].requires_grad
        assert events==['authenticated expected identities','constructor','mapping','live identity']
        endpoint.clear()
    finally:
        if sys.modules.get('torch') is torch:del sys.modules['torch']
        if sys.modules.get(module.__name__) is module:del sys.modules[module.__name__]
assert not any(name.split('.')[0] in {'torch','numpy','PIL','transformers'} for name in sys.modules)
print('PASS real candidate payload-loader body executes expected identities before constructor; expected environment/model/processor retained; original metadata unchanged; original A/C role construction and mapping check retained; native UNRUN')
