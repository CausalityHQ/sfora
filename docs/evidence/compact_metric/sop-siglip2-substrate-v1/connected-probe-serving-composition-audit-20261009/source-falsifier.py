# stdlib only; run from repo root: python3 -I /tmp/probe_compose_falsifier.py
import ast, hashlib, json, pathlib
S = pathlib.Path('scripts'); R = pathlib.Path('src/sfora')
fn = lambda p: {n.name: ast.dump(n) for n in ast.parse((S/p).read_text()).body if isinstance(n, ast.FunctionDef)}
m, p = fn('evaluate_siglip2_connected_mlp.py'), fn('evaluate_siglip2_connected_probe.py')
assert m.keys() == p.keys()
# every evaluator function CombinedAuthority/derived code can reach is byte-identical in AST
for n in ('native_start','accept_unit','guard_helpers','resources','check_receipt','merge_guards','closure',
          'load_authenticated','admission_exit_guard','source_live_guard','endpoint_scope','authenticate_payloads',
          'check_launch','policy','run'):
    assert m[n] == p[n], n
# the only differing functions are ten MLP->probe renames (FILES->HISTORICAL_FILES, schema/trainer names, one bootstrap call)
assert {n for n in m if m[n] != p[n]} == {'authority','exit_rehash','load_endpoint_reader','load_bootstrap_authority','load_first_selection','load_original_owner','check_endpoint_binding','endpoint_facts','native_export','main'}
# trainer authority AST pin
t = next(n for n in ast.parse((S/'train_siglip2_connected_probe.py').read_text()).body if getattr(n,'name','') == 'authority')
assert hashlib.sha256(ast.dump(t).encode()).hexdigest() == '38827285a4eed3dcd67cf5310571a208b8ef2842be9509c0e2f076fee76d898b'
# installed probe runtime imports no third-party module the accepted MLP runtime did not
def imps(f):
    s = set()
    for n in ast.walk(ast.parse((R/f).read_text())):
        if isinstance(n, ast.Import): s |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0: s |= {n.module + '::' + a.name for a in n.names}
    return {x for x in s if x.split('::')[0].split('.')[0] in {'torch','transformers','PIL','numpy','safetensors','torchvision'}}
assert imps('connected_probe_inference.py') <= imps('connected_inference.py')
print('PASS: no static composition delta beyond the ten enumerated renames')
