#!/usr/bin/env python3
"""Separate H-union-S control native authority; no historical reader is rebound.

The root freezes an actual connected-control-native-authority-v1 FILE with
exact keys schema/library/build_receipt/build_evidence/source_manifest/supplemental.
build_evidence covers every archived files_sha256 entry. supplemental is a
nonempty exact list of {file:FILE,provenance:FILE}, including library. Each
provenance JSON has schema/file/kind/origin/evidence; kind is archived-cutile-build
or root-frozen-runtime, origin explains attribution, evidence contains actual
FILEs. ELF NEEDED alone supplies no such authority. Missing facts fail closed.

CombinedAuthority.collect calls the genuine authenticated collector, validates
the full inventory and mapped identities, then projects H for the historical
predicates. install derives only private audit/quadratic/fitter/evaluator exits;
each substitution has an exact AST inverse. Original owned API remains owned.
No hash cache, library-unmapping assumption, scientific gate or native launch.
"""
import ast
import copy
import os
from pathlib import Path
import stat
import sys
from types import FunctionType, SimpleNamespace

ARCHIVE_SHA = 'c2d6ff677c5c533f576774d8268c536483d27ff21e93c3efa8cf2319cacc7740'
BINARY_SHA = '3d1ec7968713aa0f069f742b9454976c77ad77d115cf39c0844b6f14d6b6b526'


def require(value, message):
    if not value:
        raise ValueError(message)


class CombinedAuthority:
    def __init__(self, context, fact, observer, request):
        self.context, self.fact, self.observer, self.request = context, copy.deepcopy(fact), observer, request
        self.record = self.read_json(fact)
        self.files = self.validate_runtime()
        self.identities = {p:self.identity(Path(p)) for p in self.files}
        self.admitted = False
        self.inventory = None
        original = context['nearest'].native_source_api(context)
        self.original = original
        private_audit = next(c.cell_contents for c in original.audit_origins.__closure__
            if isinstance(c.cell_contents,FunctionType) and c.cell_contents.__name__ == 'audit_origins')
        self.supplement = private_audit.__globals__['_nearest_supplement']
        self.historical = self.historical_inventory()
        require(self.files.keys().isdisjoint(self.historical['files']), 'conflicting H/S native membership')
        self.frozen = (copy.deepcopy(self.record),dict(self.files),dict(self.identities),copy.deepcopy(self.historical))

    def read_json(self, fact):
        return self.request.strict_json(self.observer.file_bytes(fact,keep=True))

    def validate_runtime(self):
        record = self.read_json(self.fact)
        require(type(record) is dict and record.keys() ==
            {'schema','library','build_receipt','build_evidence','source_manifest','supplemental'} and
            record['schema'] == 'connected-control-native-authority-v1', 'exact combined native authority required')
        require(record['build_receipt']['sha256'] == ARCHIVE_SHA and
            record['library']['sha256'] == BINARY_SHA, 'archived native binary/build FILE differs')
        build = self.read_json(record['build_receipt'])
        evidence = record['build_evidence']
        require(build['schema'] == 'sfora-cutile-threads-v1' and build['claim_eligible'] is False and
            build['decision'] == 'PASS_SEQUENTIAL_THREAD_EXACTNESS_CACHE_GATE' and
            type(evidence) is dict and evidence.keys() == build['files_sha256'].keys(),
            'complete archived build evidence required')
        for name,fact in evidence.items():
            require(Path(fact['path']).name == name and fact['sha256'] == build['files_sha256'][name],
                'archived build evidence binding differs')
            self.observer.file_bytes(fact)
        require(record['source_manifest'] == evidence['source-manifest.json'] and
            self.read_json(evidence['binaries.json'])['candidate.so'] == record['library']['sha256'],
            'archived binary/source manifest binding differs')
        manifest = self.read_json(record['source_manifest'])
        require(type(manifest) is dict and manifest and
            all(type(n) is str and not Path(n).is_absolute() and '..' not in Path(n).parts and
                type(h) is str and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
                for n,h in manifest.items()), 'complete archived source manifest required')
        self.observer.file_bytes(record['library'])
        rows = record['supplemental']
        require(type(rows) is list and rows, 'explicit supplemental native FILE provenance required')
        files = {}
        for row in rows:
            require(type(row) is dict and row.keys() == {'file','provenance'}, 'exact supplemental FILE/provenance required')
            file = row['file']
            self.observer.file_bytes(file)
            require(file['path'] not in files, 'conflicting supplemental native membership')
            files[file['path']] = file['sha256']
            proof = self.read_json(row['provenance'])
            require(type(proof) is dict and proof.keys() == {'schema','file','kind','origin','evidence'} and
                proof['schema'] == 'connected-control-native-file-provenance-v1' and proof['file'] == file and
                proof['kind'] in {'archived-cutile-build','root-frozen-runtime'} and
                type(proof['origin']) is str and proof['origin'].strip() and
                type(proof['evidence']) is list and proof['evidence'], 'actual native FILE provenance required')
            require(len({v['path'] for v in proof['evidence']}) == len(proof['evidence']) and
                all(v['path'] not in {file['path'],row['provenance']['path']} for v in proof['evidence']),
                'independent native provenance evidence required')
            for value in proof['evidence']: self.observer.file_bytes(value)
            if file == record['library']:
                require(proof['kind'] == 'archived-cutile-build' and
                    all(value in proof['evidence'] for value in
                        (record['build_receipt'],evidence['binaries.json'],record['source_manifest'])),
                    'library archived build provenance required')
            else:
                require(proof['kind'] == 'root-frozen-runtime', 'root-frozen supplemental runtime provenance required')
        require(files.get(record['library']['path']) == record['library']['sha256'], 'library missing supplemental authority')
        return files

    @staticmethod
    def identity(path):
        require(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'canonical native mapping FILE required')
        value = path.stat()
        require(stat.S_ISREG(value.st_mode), 'regular native mapping FILE required')
        return value.st_dev,value.st_ino

    def historical_inventory(self):
        legacy = self.context['legacy']
        known = {'files':{},'modules':{}}
        for proof in (legacy['selected']['source_cpu']['origins'],legacy['warm_record']['origins'],self.supplement):
            for kind in known:
                for name,value in proof[kind].items():
                    require(known[kind].setdefault(name,value) == value, 'conflicting original native origin authority')
        return known

    def check(self):
        record,files,identities,historical = self.frozen
        require(self.record == record and self.files == files and self.identities == identities and
            self.historical == historical and self.read_json(self.fact) == record and
            self.validate_runtime() == files and self.historical_inventory() == historical,
            'combined native authority/ownership changed')
        for p,identity in identities.items():
            require(self.identity(Path(p)) == identity, 'supplemental native FILE inode replaced')

    def provenance_facts(self):
        record = self.record
        facts = [self.fact,record['library'],record['build_receipt'],record['source_manifest'],*record['build_evidence'].values()]
        for row in record['supplemental']:
            facts.extend((row['file'],row['provenance'],*self.read_json(row['provenance'])['evidence']))
        return facts

    def mappings(self):
        result = {}
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith('/') or '.so' not in fields[5]: continue
            raw = fields[5]
            require(not raw.endswith(' (deleted)'), 'deleted native mapping rejected')
            path = Path(raw)
            require(str(path) == raw and path.resolve() == path and not path.is_symlink(), 'noncanonical native mapping rejected')
            major,minor = (int(v,16) for v in fields[3].split(':'))
            identity = os.makedev(major,minor),int(fields[4])
            require(identity[1] > 0 and self.identity(path) == identity, 'native mapping inode/dev differs from current FILE')
            require(result.setdefault(raw,identity) == identity, 'conflicting native mappings')
        return result

    def collect(self, extract, packages):
        self.check()
        legacy = self.context['legacy']
        require(extract is legacy['extract'] and packages is legacy['selected']['packages'], 'combined collector dependency changed')
        before = self.mappings()
        origins = legacy['source_driver'].imported_origins(extract,packages)
        after = self.mappings()
        require(before == after, 'native mapping inventory changed during genuine hash collection')
        require(type(origins) is dict and origins.keys() == {'packages','files','modules','native_files'} and
            origins['packages'] == packages and type(origins['files']) is dict and
            type(origins['modules']) is dict and type(origins['native_files']) is list and
            len(origins['native_files']) == len(set(origins['native_files'])), 'complete genuine native mapping inventory required')
        union = {**self.historical['files'],**self.files}
        require(all(union.get(p) == h for p,h in origins['files'].items()), 'unknown or changed combined native origin')
        require(all(self.historical['modules'].get(n) == p for n,p in origins['modules'].items()),
            'unknown or changed combined module origin')
        require(set(origins['native_files']) == after.keys(), 'complete genuine supplemental native mapping inventory required')
        require(after.keys() <= origins['files'].keys() and
            set(origins['modules'].values()) <= origins['files'].keys(), 'genuine inventory omits mapped/module FILE hashes')
        observed = set(origins['files']) & self.files.keys()
        require(observed <= set(origins['native_files']), 'supplemental origin missing native mapping')
        require(all(after[p] == self.identities[p] for p in observed), 'supplemental mapping inode replaced')
        if self.admitted or self.record['library']['path'] in observed:
            require(observed == self.files.keys(), 'exact supplemental native inventory required')
            self.admitted = True
        self.check()
        self.inventory = copy.deepcopy(origins)
        return {**origins,'files':{p:h for p,h in origins['files'].items() if p not in self.files},
            'native_files':[p for p in origins['native_files'] if p not in self.files]}

    def install(self, evaluator_source, evaluation_context):
        """Fixed four exit adapters and two unchanged historical wrappers, owned once."""
        context,request = self.context,self.request
        require('control_native_owned' not in context, 'untrusted combined native ownership')
        original,legacy = self.original,context['legacy']
        original_owned = context['native_source_owned']
        original_authentication = original_owned['authenticate']
        original_authentication_code = original_authentication.__code__
        nearest = context['nearest']
        nearest_source = request.Source(nearest,{'path':nearest.__file__,'sha256':context['guards'][nearest.__file__]})
        own_module = sys.modules[__name__]
        own_source = request.Source(own_module,{'path':__file__,'sha256':evaluation_context['guards'][__file__]})
        request_source = request.Source(request,{'path':request.__file__,'sha256':evaluation_context['guards'][request.__file__]})
        observer_source = request.Source(self.observer,{'path':self.observer.__file__,
            'sha256':evaluation_context['guards'][self.observer.__file__]})
        source_check,source_check_code = request.Source.check,request.Source.check.__code__
        bindings = {n:v for n,v in vars(self).items() if n not in {'admitted','inventory'}}
        owned = None
        functions,namespaces,inverses = [],[],{}
        dump = lambda node:ast.dump(node,include_attributes=False)
        def derive(module,name,changes,label):
            digest = evaluation_context['guards'].get(module.__file__,context['guards'].get(module.__file__))
            require(digest is not None, 'authenticated adapter source FILE required')
            raw = request.read_file({'path':module.__file__,'sha256':digest})
            original_node = next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name == name)
            node = copy.deepcopy(original_node)
            def substitute(tree,before,after):
                source,target = ast.parse(before,mode='eval').body,ast.parse(after,mode='eval').body
                class Change(ast.NodeTransformer):
                    count = 0
                    def visit(self,value):
                        if dump(value) == dump(source):
                            self.count += 1
                            return ast.copy_location(copy.deepcopy(target),value)
                        return super().visit(value)
                change = Change(); result = change.visit(tree)
                require(change.count == 1, 'exact adapter substitution count differs: '+label)
                return result
            for before,after in changes: node = substitute(node,before,after)
            inverse = copy.deepcopy(node)
            for before,after in reversed(changes): inverse = substitute(inverse,after,before)
            require(dump(inverse) == dump(original_node), 'adapter retained predicates differ: '+label)
            inverses[label] = (dump(original_node),dump(inverse),len(changes))
            namespace = dict(vars(module))
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),namespace)
            return namespace[name],namespace,original_node
        audit,audit_ns,_ = derive(context['old'],'audit_origins',[
            ("(expected, context['warm_record']['origins'])","(expected, context['warm_record']['origins'], _nearest_supplement)"),
            ("source.imported_origins(context['extract'], context['selected']['packages'])",
             "_control_collect(context['extract'], context['selected']['packages'])")],'audit')
        audit_ns.update(_nearest_supplement=self.supplement,_control_collect=self.collect)
        quadratic,quadratic_ns,_ = derive(context['old'],'exit_rehash',[],'quadratic')
        fitter,fitter_ns,_ = derive(context['fitter'],'exit_rehash',[
            ("context['old'].exit_rehash(context['legacy'])","_nearest_quadratic_exit(context['legacy'])")],'fitter')
        evaluator,evaluator_ns,evaluator_original = derive(evaluator_source.module,'exit_rehash',[
            ("t['nearest'].native_source_api(t)","_control_native_api(t)")],'evaluator')
        # Retain the exact original exported wrapper predicates, including exact four.
        native_node = next(n for n in ast.parse(Path(nearest.__file__).read_bytes()).body
            if isinstance(n,ast.FunctionDef) and n.name == 'native_source_api')
        wrappers = {}
        wrapper_ns = {'__builtins__':__builtins__,'require':nearest.require,'legacy':legacy,
            'context':context,'private_audit':audit,'fitter_exit':fitter,'supplement':self.supplement}
        for name in ('audit_origins','exit_rehash'):
            node = next(n for n in native_node.body if isinstance(n,ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]),nearest.__file__,'exec'),wrapper_ns)
            wrappers[name] = wrapper_ns[name]
        quadratic_ns['audit_origins'] = wrappers['audit_origins']
        fitter_ns['_nearest_quadratic_exit'] = quadratic
        def dispatch(value):
            checked_authenticate()
            require(value is context, 'owned combined native context required')
            self.admitted = True  # Full evaluator exit always requires the complete frozen S.
            return api
        def evidence():
            checked_authenticate()
            require(self.admitted, 'combined native admission incomplete')
            wrappers['audit_origins'](legacy,require_exact=True)
            return {'authority':copy.deepcopy(self.fact),'inventory':copy.deepcopy(self.inventory),
                'supplemental_files':dict(self.files),'historical_projection':copy.deepcopy(legacy['origins']),
                'mapped_identities':{p:list(v) for p,v in self.identities.items()}}
        def evaluator_exit(value,guard):
            checked_authenticate()
            require(value is evaluation_context, 'owned combined evaluator context required')
            return evaluator(value,guard)
        def authenticate():
            require(evaluation_context['training_context'] is context and context['legacy'] is legacy and
                context['nearest'] is nearest and context.get('control_native_owned') is owned and
                owned.keys() == {'api','authenticate','owner'} and owned['owner'] is self and
                owned['api'] is api and owned['authenticate'] is authenticate,
                'combined native ownership changed')
            require(request.Source.check is source_check and source_check.__code__ is source_check_code,
                'combined source checker dependency changed')
            for source in (own_source,request_source,observer_source,nearest_source,evaluator_source): source.check()
            require(context['native_source_owned'] is original_owned and
                original_owned['api'] is original and original_owned['authenticate'] is original_authentication and
                original_authentication.__code__ is original_authentication_code, 'original native authentication changed')
            require(nearest.native_source_api(context) is original, 'original native ownership changed')
            require(all(vars(self).get(n) is v for n,v in bindings.items()) and
                vars(self).keys() == bindings.keys() | {'admitted','inventory'}, 'combined authority binding changed')
            self.check()
            for values,snapshot in namespaces:
                require(values.keys() == snapshot.keys() and all(values[k] is v for k,v in snapshot.items()),
                    'combined private namespace binding changed')
            for fn,code,defaults,kw,values,cells in functions:
                require(fn.__code__ is code and fn.__defaults__ == defaults and fn.__kwdefaults__ == kw and
                    fn.__globals__ is values and len(fn.__closure__ or ()) == len(cells) and
                    all(c.cell_contents is v for c,v in zip(fn.__closure__ or (),cells,strict=True)),
                    'combined private function changed')
            require(vars(api).keys() == exported.keys() and all(vars(api)[k] is v for k,v in exported.items()),
                'combined exported API changed')
        authentication_code = authenticate.__code__
        def checked_authenticate():
            require(authenticate.__code__ is authentication_code, 'combined authentication code changed')
            authenticate()
        wrapper_ns['authenticate'] = checked_authenticate
        evaluator_ns['_control_native_api'] = dispatch
        api = SimpleNamespace(**wrappers,evaluator_exit=evaluator_exit,authenticate=authenticate,evidence=evidence,
            asts={'inverses':inverses,'evaluator_original':evaluator_original})
        exported = dict(vars(api))
        namespaces = [(values,dict(values)) for values in (audit_ns,quadratic_ns,fitter_ns,evaluator_ns,wrapper_ns)]
        owned = {'api':api,'authenticate':authenticate,'owner':self}
        context['control_native_owned'] = owned
        functions.extend((fn,fn.__code__,copy.deepcopy(fn.__defaults__),copy.deepcopy(fn.__kwdefaults__),fn.__globals__,
            tuple(c.cell_contents for c in fn.__closure__ or ())) for fn in
            (audit,quadratic,fitter,evaluator,*wrappers.values(),dispatch,evidence,evaluator_exit,authenticate,checked_authenticate))
        authenticate()
        return api
