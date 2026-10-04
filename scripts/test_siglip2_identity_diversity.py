#!/usr/bin/env python3
"""Bounded stdlib falsifiers; native numerical qualification belongs to root."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

PATH = Path(__file__).with_name('train_siglip2_identity_diversity.py')
if PATH.exists():
    spec = importlib.util.spec_from_file_location('diversity_test_driver', PATH)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
else:
    driver = SimpleNamespace()
SCOPE = Path(__file__).parent.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/identity-diversity-v1/scope.json'


class DiversityAdmissions(unittest.TestCase):
    def test_new_exact2_and_method(self):
        self.assertEqual(driver.FILES, {'train_siglip2_identity_diversity.py', 'test_siglip2_identity_diversity.py'})
        self.assertEqual(driver.SCHEMA, 'siglip2-identity-diversity-v1')
        self.assertEqual(driver.parameter_roles('candidate'), driver.parameter_roles('control'))
        self.assertEqual(driver.loss_denominators(63), (128, 126))
        self.assertEqual(driver.loss_denominators(0), (128, None))
        self.assertNotIn('hinge', driver.RECIPE)

    @unittest.skipUnless(SCOPE.exists(), 'repository metadata unavailable in portable exact2 closure')
    def test_scope_namespaces_and_mutants(self):
        raw = SCOPE.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), driver.SCOPE_SHA256)
        manifest = driver.check_scope_manifest(driver.strict_json(raw))
        for arm, classes in (('control', 1008), ('candidate', 2016)):
            scope = manifest[arm]
            driver.check_scope_arm(scope, arm)
            self.assertEqual(len(scope['class_names']), classes)
            bank = driver.ranking_bank(scope['targets'], scope['original_rows'])
            driver.check_ranking_bank(bank)
            wrong = copy.deepcopy(scope)
            wrong['rows'][1]['augmentation_id'] += 1
            with self.assertRaises(ValueError): driver.check_scope_arm(wrong, arm)
            wrong = copy.deepcopy(scope)
            wrong['targets'][0] = classes
            with self.assertRaises(ValueError): driver.check_scope_arm(wrong, arm)
            with self.assertRaises(ValueError): driver.check_scope_arm(scope, 'candidate' if arm == 'control' else 'control')
        wrong = copy.deepcopy(manifest)
        wrong['candidate']['original_rows'][1] = wrong['candidate']['original_rows'][0]
        with self.assertRaises(ValueError): driver.check_scope_manifest(wrong)

    def test_visit_arithmetic_scope_counts(self):
        for classes, expected in ((1008, {'8': 880, '9': 128}), (2016, {'4': 1888, '5': 128})):
            summary = driver.class_visits(classes, list(reversed(range(classes))))
            self.assertEqual(summary['histogram'], expected)
            self.assertEqual(sum(summary['counts']), 8192)
        with self.assertRaises(ValueError): driver.class_visits(2016, list(range(1008)))

    def test_original_image_self_and_singletons(self):
        bank = driver.ranking_bank([0, 0, 1, 2], [180, 7, 40, 90])
        fact = driver.ranking_membership(bank, [1, 2, 1])
        self.assertEqual(fact['positive'], [[0], [], [0]])
        self.assertEqual(fact['valid'], 2)
        self.assertEqual(fact['eligible_counts'], [3, 3, 3])
        with self.assertRaises(ValueError): driver.ranking_membership(bank, [4])
        with self.assertRaises(ValueError): driver.ranking_membership({**bank, 'sha256': '0'*64}, [])

    def test_scope_bundle_swap(self):
        scope = {'manifest': {'path': '/tmp/scope.json', 'sha256': driver.SCOPE_SHA256},
                 'arm': 'candidate', 'payload': {'scope_sha256': driver.ARM_SHA256['candidate']}}
        self.assertEqual(driver.scope_identity(scope)['arm'], 'candidate')
        with self.assertRaises(ValueError): driver.scope_identity({**scope, 'arm': 'control'})
        with self.assertRaises(ValueError): driver.scope_identity({**scope, 'manifest': {**scope['manifest'], 'sha256':'0'*64}})

    def test_owned_closure_tamper_and_extra_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in driver.FILES: (root/name).write_text('# owned\n')
            code = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in driver.FILES}
            (root/'execution.json').write_text(json.dumps(code))
            sha = hashlib.sha256((root/'execution.json').read_bytes()).hexdigest()
            self.assertEqual(driver.closure(root, sha, driver.FILES, {}), code)
            (root/next(iter(driver.FILES))).write_text('# tampered\n')
            with self.assertRaises(ValueError): driver.closure(root, sha, driver.FILES, {})
            code['historical_trainer.py'] = '0'*64
            (root/'execution.json').write_text(json.dumps(code))
            sha = hashlib.sha256((root/'execution.json').read_bytes()).hexdigest()
            with self.assertRaises(ValueError): driver.closure(root, sha, driver.FILES, {})

    def test_json_and_resource_contract(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError): driver.strict_json(raw)
        self.assertEqual(driver.policy('cpu')['seconds'], 500)
        self.assertEqual(driver.policy('mechanics')['seconds'], 600)
        self.assertEqual(driver.policy('train')['host_bytes'], 8*1024**3)
        self.assertEqual(driver.policy('train')['swap_bytes'], 0)
        self.assertEqual(driver.policy('train')['cuda_allocated_bytes_exclusive'], 10000000000)

    def test_retained_admission_and_loader_ast(self):
        # SHA-backed correspondence catches accidentally dropping complete source,
        # origin/exit, AdamW/scaler/RNG or vision/processor/buffer predicates.
        tree = ast.parse(PATH.read_bytes())
        nodes = {n.name:n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name, expected in RETAINED_AST.items():
            node = copy.deepcopy(nodes[name])
            for item in ast.walk(node):
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    item.value = item.value.replace('train_siglip2_identity_diversity.py', 'train_siglip2_compact_ranking.py')
            self.assertEqual(hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(), expected, name)
        native_top = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) and
                      any(a.name.split('.')[0] in driver.NATIVE for a in n.names)]
        self.assertEqual(native_top, [])
        self.assertNotIn('live_top1_hinge', nodes)
        self.assertNotIn('live_top1_indices', nodes)

    def test_launch_scope_cache_and_prerequisite_mutants(self):
        H = 'a'*64
        unit = {'receipt':{'path':'/tmp/receipt.json','sha256':H}, 'log':{'path':'/tmp/log','sha256':H},
                'unit':'test-unit','invocation_id':'a'*32,'service_seconds':1.,'native_peak_rss_kib':1.,'both_locks_held':True}
        launch = {'schema':driver.AUTHORITY_SCHEMA,'execution_sha256':H,'phase':'cpu','arm':'control','seed':179061,
                  'nearest':driver.NEAREST,'fitter':driver.FITTER,'accepted':driver.ACCEPTED,'readout':driver.READOUT,
                  'recipe':driver.RECIPE,'resource_policy':driver.policy('cpu'),'both_locks_held':True,
                  'selected_cpu':None,'selected_mechanics':None,'native_authority':{'path':'/tmp/native.json','sha256':H},
                  'scope':{'path':'/tmp/scope.json','sha256':driver.SCOPE_SHA256},
                  'candidate_cache':{'exporter':{'root':'/tmp/exporter','execution_sha256':H,
                    'code':{n:H for n in driver.EXPORTER_FILES}}, 'authority':{'path':'/tmp/export.json','sha256':H},
                    'terminal':{('proof' if k=='receipt' else k):v for k,v in unit.items()}}}
        args = SimpleNamespace(execution_sha256=H,phase='cpu',arm='control',seed=179061)
        driver.check_launch(launch,args)
        for key,value in (('schema','old'), ('candidate_cache',{}), ('scope',{'path':'/tmp/scope.json','sha256':'0'*64}),
                          ('resource_policy',{**driver.policy('cpu'),'host_bytes':9*1024**3}), ('selected_cpu',unit)):
            with self.assertRaises((ValueError,KeyError)): driver.check_launch({**launch,key:value},args)
        wrong = copy.deepcopy(launch); wrong['candidate_cache']['terminal']['both_locks_held']=False
        with self.assertRaises(ValueError): driver.check_launch(wrong,args)
        self.assertNotEqual(driver.method(launch),driver.method({**launch,'scope':{**launch['scope'],'path':'/tmp/other.json'}}))

    @unittest.skipUnless(SCOPE.exists(), 'repository metadata unavailable in portable exact2 closure')
    def test_complete_schedule_and_update_mutants(self):
        manifest=driver.strict_json(SCOPE.read_bytes())
        for arm in driver.ARMS:
            scope=manifest[arm];classes=len(scope['class_names']);order=list(range(classes))
            by_class={c:[] for c in order}
            for i,c in enumerate(scope['targets']):by_class[c].append(i)
            batches=[[by_class[order[(step*64+slot)%classes]][0] for slot in range(64)] for step in range(128)]
            masks=[[bool((step+slot)%2) for slot in range(64)] for step in range(128)]
            bank=driver.ranking_bank(scope['targets'],scope['original_rows'])
            provenance={'seed':179061,'classes':classes,'scope_sha256':driver.ARM_SHA256[arm],
                'class_order':order,'visits':driver.class_visits(classes,order),
                'schedule_json_sha256':driver.json_sha256(batches),'masks_json_sha256':driver.json_sha256(masks),
                'class_slots_sha256':driver.json_sha256([[scope['targets'][i] for i in row] for row in batches]),
                'official_image_ids_sha256':driver.json_sha256([[scope['original_rows'][i] for i in row] for row in batches]),
                'global_torch_rng_unchanged':True,'mask_seed_offset':3000001,
                'anchor_rng_after1000':{'bit_generator':'PCG64','has_uint32':0},
                'mask_rng_after1000':{'bit_generator':'PCG64','has_uint32':0}}
            record={'identity':{'arm':arm,'seed':179061},'ranking_bank':bank,'scope_schedule':batches,
                    'scope_masks':masks,'schedule_provenance':provenance}
            self.assertTrue(driver.check_scope_schedule_fact(record))
            for key,value in (('scope_masks',[[0]*64]*128), ('scope_schedule',batches[1:]+batches[:1]),
                              ('identity',{'arm':'candidate' if arm=='control' else 'control','seed':179061})):
                with self.assertRaises(ValueError):driver.check_scope_schedule_fact({**record,key:value})
            wrong=copy.deepcopy(record);wrong['schedule_provenance']['class_order'][0]=1
            with self.assertRaises(ValueError):driver.check_scope_schedule_fact(wrong)
            batch=batches[0];full=driver.ranking_membership(bank,batch)
            members=[]
            for view in driver.VIEWS:
                for offset in range(0,64,16):
                    anchors=batch[offset:offset+16];fact=driver.ranking_membership(bank,anchors)
                    members.append({'view':view,'batch':anchors,'active':fact['valid'],**fact})
            row={'step':1,'batch':batch,'full_valid':full['valid'],'full_membership_sha256':driver.json_sha256(full),
                'scale':128,'arm':arm,'gradient_norm':1.,'C_gradient_norm':1.,'preclip_norm':1.,
                'A_before_sha256':'a'*64,'A_after_sha256':'b'*64,'C_before_sha256':'a'*64,'C_after_sha256':'b'*64,
                'state_sha256':'c'*64,'core_seconds':.5,'seconds':1.,'membership':members,
                'active_anchors':sum(m['active'] for m in members),'mse':.5,'rank':.25,'loss':.75,
                'ranking_gradient_norm':1.,'ranking_C_gradient_norm':1.}
            driver.check_steps([row],1,1,bank)
            for key,value in (('hinge',0.),('loss',.5),('C_after_sha256','a'*64),('full_valid',63),('scale',64)):
                with self.assertRaises(ValueError):driver.check_steps([{**row,key:value}],1,1,bank)

    def test_complete_nested_source_guard_correspondence(self):
        tree=ast.parse(PATH.read_bytes());nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        for name,expected in RETAINED_NESTED.items():
            actual=set()
            for stmt in ast.walk(nodes[name]):
                if not isinstance(stmt,ast.stmt) or isinstance(stmt,ast.FunctionDef):continue
                stmt=copy.deepcopy(stmt)
                for item in ast.walk(stmt):
                    if isinstance(item,ast.Constant) and isinstance(item.value,str):
                        item.value=item.value.replace('train_siglip2_identity_diversity.py','train_siglip2_compact_ranking.py')
                actual.add(hashlib.sha256(ast.dump(stmt,include_attributes=False).encode()).hexdigest())
            self.assertTrue(set(expected).issubset(actual),name)

    def test_retained_update_and_vision_predicate_order(self):
        tree=ast.parse(PATH.read_bytes());nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        for name,expected in RETAINED_STATEMENTS.items():
            hashes=[]
            for stmt in nodes[name].body:
                stmt=copy.deepcopy(stmt)
                for item in ast.walk(stmt):
                    if isinstance(item,ast.Constant) and isinstance(item.value,str):
                        item.value=item.value.replace('train_siglip2_identity_diversity.py','train_siglip2_compact_ranking.py')
                hashes.append(hashlib.sha256(ast.dump(stmt,include_attributes=False).encode()).hexdigest())
            actual=[h for h in hashes if h in expected]
            self.assertEqual(actual,expected,name)

    def test_no_scope_gradient_equality_or_hinge_dispatch(self):
        tree = ast.parse(PATH.read_bytes())
        functions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        loss = ast.unparse(functions['loss_terms'])
        self.assertNotIn('hinge', loss)
        self.assertIn('ranking_gallery', loss)
        self.assertIn('smooth_ap_terms', loss)
        self.assertNotIn('FunctionType', ast.unparse(functions['cpu_gradients']))
        self.assertIn('historical', ast.unparse(functions['cpu_gradients']))
        self.assertNotIn('reset_peak_memory_stats', PATH.read_text())


RETAINED_AST = {'require': '3b218d97634fd91fd6595535193a2eb525911ab34ffd45793deaba58332ca99e', 'strict_json': '75f8d5022ff11de276d183c29d91ecdba25ecc218f7b88ec932c123426467c88', 'file_fact': '64d3e7497f697206a7793c38e1750095e0f3b60036c158a36ae689fb3f5a634a', 'bound_file': '3db94d649ee69a5e3247924c59b7880d2fc4004242122e53717965b4beff7467', 'batch_bound_files': '1fef853c3b4168a306b418dec5a7d40b1d35e3f63285d20d90272d0c6fba8ac9', 'read_json': '6b985b95eaf4a2e83a13f6b8eb2080ff9b8f9231dc8ec08a7f333229c8f7dea4', 'closure': 'bcc0d3bd8b35c3f0cbb5f928def6eaf837deb8bb9dc59a19c5c1622b12e121a3', 'load_authenticated': 'da8fa1d6dcbaacb6076d2f89ccbead51739b39336645b9b50d02f1be8fa89afa', 'policy': '6410e1634b30c2e3123aa9ad3c26185a589311226ae0780ad75ac9ed3cddf426', 'check_unit': '794886df880372a198328a31c19b840cc3f86eda8d5b4c91265555fd0a157908', 'timed': '3be58e1115fe8d07cd2eca68ba26c2b9b58da48c082f6afe4e83f4206885ccc0', 'fingerprint': 'b0a6510a7b799d7e59bcff9e90aed85955ffb71ac498b45b772ca9d12648c444', 'clone': 'a199ca1ea370e82e09f8609909c242c48c64d2c8eb0e67dabdbd4bc16bf7c2d3', 'helper_guard': '0a2be6c7589a29cfe5f3af0f2c1e3d56e74b653ab9692b31e354578e9a9d5765', 'require_no_training': '25e8a7a041a2778641802a45f437a7202c203cee6e46795cea8748944f15564d', 'parameter_roles': '4ea18a5cf379fcf0fc9e686539c1fecae948647a7984fe7c239e6d0373404909', 'check_mu_train_provenance': 'a7c29161a365bd30f8c153fc5a2afe34e2452183ed3b9c4ebb76ab37820d701c', 'own_residual': '3a149e7ce44faf03f3c6e389e1000921e5d34f9b0553ad451f3b6e9a6dbce7d9', 'fullfeature_raw_features': '3706820907338be9911fd5b51d84dd8bd3de246be547533f3d203c61ff2e4af6', 'residual_facts': '7e567c86831e6892bf9d868c0febd083b60f82a896530d3cfb3a930059973477', 'own_A': '7daa0ed42a9eb402a4a2a1c8978dd300fa3451c283f5eeea5d937a345cdf2992', 'static_tree': '744541c495643dc2182624e4621bfccb03dd7f65108e8da0babe0727fd51cbf4', 'payload': '67a9119a750659545d4630b5cbd80f1a897bab800551f2f499592dd9f31def85', 'check_optimizer': '8525a2ba87d989912eab7f9731724033cfcd747d360a2257f3e6daa46cf12eb5', 'integrity': 'c7b564cf9f8352105d115e9a0fe1925fd162ba4ac483892306464ae7060c2fcb', 'release': '73b095a67686e93c4b9717805a59c88a9bafe291b832f7825a88e8bdbf8368bc', 'save': '78a1e1b6ca09f134ae5d8703a25715f8cee7c52e798c2227a05ebc3f60adb591', 'restore': '38274c5421b261fd7db23796bd553f7e901034d93fe053088ecc5c63a4703f1b', 'loss_denominators': 'ce86de123238eee428db8c62ff8ed4c09c2cde5ad6fbe52a85139623802a78e5', 'raw_features': '0b877d75f371d1897721897b3f212d8fe68aef2613bd1e6edbee9e53703ab9f0', 'json_sha256': 'e6afa190880b91cfdb1e70946e6618bc5792b89e4b5e2bb8d72e8e549b865d98', 'ranking_bank': 'bc2b9bb03c0e4437e68e2b3e67b34dd0a8bbef6b7c2fbeee7ee613197e14905c', 'ranking_membership': 'c56dd67441cc81a47e8fcfb71032e4c751b7d5b70f2b6df19ca6236b9301813f', 'smooth_ap_terms': 'a90d81810d4295691ff1829215a07b9ab1793cd040ad92530d6bffaefa1eb698', 'ranking_gallery': '80a6ab2cf778da8b8f6c78ccb183cd877085c4b62d1ac83655aa6f86ba69e3bd', 'cached_witness': 'e2aff52bac2a3940972cf6f3fd20e2a35d7368fef3333473123d51e387abb2bf', 'write_json': 'de7f335d66c6e9d0fba69513d6ff489f974048310910e0349b0e63044df8775c', 'clone_inference_provenance': '7f0f106547748083ee34228b7929f295aca9c20370ba1739e322fdbe19680462', 'inference_outputs': 'e02047621095dd855ef0e54bb78d84b346e0770bae1dae881561550467b6ca26', 'release_inference': '5ddd8431f61917a317625523a14032375317a929ed541a5b4a25b1dd9ff6fbf4', 'deny_training_dependencies': '763cbe31be6cd080e8d6268d9d81e5fa762af28b9280f614fabc302aac9a6537', 'authenticate_bundle_environment': '2a17ba90a73c264a45e89b0f99070e1c8eeb150a4877a42e8360da6536f66a69', 'diagnostic': '50765e6ec73ef644531c6f2fc04f4fc282c886ffcf38de9e401d70fcdc9349f5', 'admit_terminal': 'c51b07ae4e4b327a509ea11cc93225bb8e8af66d166a4f780b983898a9b22499', 'exit_admission_adapter': '61e460d249e11df9b7f475342ac9c96aa824c45d4861d91fba0e0094ed71a257', 'exit_rehash': 'd10e411cedd92f43f237a0271b8d0c193362e791ebd147c9910c2dc36c2206e8', 'parser': 'f0f62dcb216f2f0db7b3712b3f8cd316cb7c134c4cbf831319f55aaf2b9b0ff6'}
RETAINED_STATEMENTS = {'update': ['4d5e05f5856c1a5a101208c444bffaac476161c006d71966f56919de16dab93d', 'd9cfa6fe7d76cfdde017fca0e2e00cd43cb88ae3e0e00e241781eecc921b2f88', '20927af6beb23021579101325cd75a1fc87c3140da2a5fc23e442c55ff137695', 'ee03ae96a9e439afdcb84faa536d920ade9e5e92800f53d098b15c386ce1374a', '58f1cf47aa114554fafed5fe0bf86b3848e309e743990543188ba764e60dc73c', '424ca1d4579dab6038f6ddd707bfdff7168134e06582f811e0a957beb72e6b3b', 'b80d61ebbf8a66bcb3def1ba7403d73d48d7a15be0b36f55ceb6c4e1c24455d5', '7deaef48e153d39aa345a2d070ae58874b7ac9f5c83ece264541f063b2e83c46', '73ed15a46f18ffc6009e5c73704ad0381d6cb0ccb6f377445de1356f6d001361', 'ca10f20f80349b90077bce130956687f86bbd201e7657b095f4c010c1e47e248', 'bfc2e2c431fe705d927c539865c8f5ae0bf876e630958fd3c3755e643e5cd2a9', '083748fc61763407503ea98b1c02802d3bcc30c0a9f4540fbffed317c7166c4a', 'dd911beea80467dfc8ce7ac61c3a8d141ec5609a585e194f02bd80aea461127a', '534c6f4c1b238c791c011796463686602631017e991a2a18aee0a04ca918f770', '0dd57e0cf72cd8c41dc9c2fdf5e3a6e7b15d4d6b45ce3d4ef78d02a353c3fddc', '6f1d52fe3ed0efeaa266358c952766ff744fd9a0040f2cdd9caf2ba09ee8780e', 'bf193c322291f40bdb8c2cea2bc71f333f618bc5b907f0c34cf39a068d798f52', '479509cd399ddf8b006c42eead9a61e50e28be0881ffbbf811cafdfc091f0953', 'a524638bbb938056e7c16726ddc40796ed444d630e9a0288a45d4c0940356804', '87bca565dfd34bc5bfd7d9738f45e8f0e7666987538fc855fcbe60323e5380d5', '04433a7bccec471d9dd8e3c03c37ec7773274a7cc3e6c1a5d91f2ca3190c5f45', 'ad1a190799df48db2ac3b585ee622880dc292227e3f7ca602a18cc957418964e', 'a1c5f489dfc781cc7b41304cf79826436f3f5744222e1a06c31ab1f2b579f79d', 'cbf7c085ed2bb78c86564d214c1dc49cd931650722ee777c68db1fd242b04240', 'e09f7604af2bffecc35a0a5e113909950b6cfa1d122fa8738b99c933ef1d0b7b', '669a0052dce5acc44bad06a9c4b634f9e42cdd6519f7c248ae6d22db4304c5af', '461385d4a2741cdb626050345848552d0f21c9f70e184f2a7507bdf59cff017e', 'b2035e7f0413e4abb8b54ffc2021a12639320e0adac84142845e44ea91fbd3bb', 'e7e0cb32ef94449b8181063af7b7ace686fd40fb2c8b8bcd5a422975d40bd556', 'ca10f20f80349b90077bce130956687f86bbd201e7657b095f4c010c1e47e248', 'ee03ae96a9e439afdcb84faa536d920ade9e5e92800f53d098b15c386ce1374a', '52d71335c8e8ae5fa4f9dce7ae9e5d3838dd504c64fc2b57507209e1e430aab4', '405a9c31479b7d88548be0991f210d5ebb5ed2129113898e40e60fb71e43e76e', '5869ad204934101fd5712ee3a3f8dc40c9f42bec11d50f6053d6d3d5cbdadca6', '95f8dab19088df3922536a353780e8d27d7aa638cd426c433c851d61a8dbec76'], 'load_inference': ['b357698030e54ba32b67e81d5e5ccd51659370344ada21259c81ba8e4d32c6a9', '378eb8e73c3cb18ef84cd07873aabf6aa2e8f6103c71f1457374519b7a467e60', '9c7f9ae5434e9a7e231f62180e624d0c689109089fc3b0a3831f176efae4a3f3', '0d6dacb480cfc0007f5b1c8a52d6c6c13dce485fa8d5590298125519d8da7a7c', 'd67739cbecd97b03eb2bc7a76baf13e9d4f632ce6845b4efb31b0623a2d30114', '4e0b5e77cfe070e2a8bd61a152aa714b69f8bd035cdaf4cfd1f618c91aa8b16d', '9aa2691b326413a61c55270e9e97d5d2f307cb739162f6704ad4d369ebe9b089', 'c84c332e75a46a48e831815c0fad1200bee5b4c4c67f734163846b18478bf9a3', 'e5bf40d3aceecd2f6fb6abe30b3a6deb367653d8aa20d0e3e2ee34f76e234b86', 'b2baa6969e14313dd2acd12c037ec5b72c5a4fe7168d3cbdcbb7412be18dd7d4', '4d5e05f5856c1a5a101208c444bffaac476161c006d71966f56919de16dab93d', '0f4ad26150b408574332b03c6f7e60aa19a45234261634547029980861c7e382', '6b262267ee94de3c40fb8efc05cb165f6b48b6c93da6eece22c55ccc5c6928ce', 'b9b94ec726006dfe2f3f341d43a6ec0380cbb706233344c097ee99fe5bbcab4d', '1699dbd84f8b9e5d78875e372075a05fb9a5c5c92533b5e00a21232f4e8b6fa9', '6b2fc000d642a1e9f58de3e804ef104ed0480847637a3824802cc5929603e423', '7eda7747dc65beadadea3da984529abea052f19685c807bd6dba82cfb4472fe2', 'ff660840da1e414a3f839c002c0cfb85ce29a9a19bb8f9ac5f48435400665d7d', 'edb42a9e235442d4827c1157dae2026dcf5be225e97d2061a4ff43db026eda3e', '6fff40904c64db1edf228e0718b341dcc670a820118a025bd8e249dc8498d98b', '991bafc091321a59bd85fbd212ca6107cda8116dbb57f986d4d88b017994155d', '060c434334ee67c564fa18dcd1084eea87b0b2be52653f51e9917bcd97a52da8', 'aa843ad08497ce2e91ad68ed4b4a4e3f1b896efa9f20980670137479d13cd272', 'faba36c3e3e6b31bd6d93e70cc28d0bebef46192ee91df109bf198d5a719bbd9', 'a8ad540c6fb1fb0850242e91f27fdeac221721b729641c6e8f7d64a96f1160ac', 'f758480b6ef22982ae6c12a7118025d205e860f83b2cf0c41fbf8db70bd0b554', '77c7d734559db134ec9be6a64132a972aaf1ec37a82565f7785fcccd45ba59d4', 'b6ad6254466a2fb49f5c96417f63cf2f1eded517ed69247214db7fb2f68dedb3', 'f777ba4617f736778fe3e07f648a1a87fd5c46179a703eb36a84ecad4daf5922', '865986e8391252ad0ac7a1725cc290ed15fa553d1755dd0bb1c85bc0ee5067fa', 'a86bb01eef5cafbdd54e90848330f75b245ee5d39cbd52146b818ce965c0ddad', '36d4bdb2e96b0651ecbb721d3b4c9d68c95d8635ef42a4063df2abef09698377', 'bd33c2fbf3d16feacd6fcc5158b673cfee83a1cc6f934930b1f1ad21ce4ba717', '95fde7dcf9c32c9cd0186c919248c219d30881ce9e39eee577a90fa947da0556', 'bc2ad0b7c949042d1cbaef772862bdeb0b7b69bf9a105053c13af6555e4b6623', 'd39b079d4f979a7b0523e34a55c4091d02839e052d68f96f7ee7b222067f523d', '7d2cf57042a27ec31c4f72f20223eec6810d83af9120f79cd16c10aa501c4f01', 'e408efa1010b371d84ea341f6a0f6ee0833025986dff610f8409467cccf21c42', 'bdb2a01e54b27c70f29609c42e250a6d03b784be1f9e936ceb0fba5a27034312', '2c9c62b9bf54399f7f21a97c774b31d3229be14a58b5f63914cd06fc7d0b70d1', 'd7aa4c43205bcb6ea71da4e104e4f092ee89b02888103ef148e6f252ffdcb3f4', '50b888fb5e67dc47e804429136e2c44ffcea7727aa963770c874537a63ffd2a6', '2830233c05c70858cf2fd4c1683585d949226f5fd6d367ef68a764f2b97b1baa'], 'export_bundle': ['f86017adc0ace5187a56ec3c072ef529c9c4ef404139446cb5653f7ac0c612c3', '4d5e05f5856c1a5a101208c444bffaac476161c006d71966f56919de16dab93d', 'f171c6b6b060bcad614549cd8cef7242bca4d91df53a225d6fa1595bfc75eab3', '2649ab7dbfa6eae369c6c7edbee296f9bbb91e697d2856821817666abbb14a70', 'acf2244acd30c355fcd478c404a68f4c462846771798422313a3ebd894b8fb56'], 'qualify_bundle': ['73d0bf35a8ff28d1acd4cd6cd453dcc0c5ad4ce115c9abf4249d9e4f4788c0ac', '4d5e05f5856c1a5a101208c444bffaac476161c006d71966f56919de16dab93d', 'eb355e0b2477a9d207a50accb1e41c8347717cb90a6348625683dd4c9e8e9888', 'a45d2f0b39caece0fd2032b5976d531bd3437a6dac6593eeff0e8ad15fa7348c', '387bde589a7eeb175fc889a95825e08765cae7861705a66a3ffec3c25d532474', '4676e109f3dbf822c0ef8957e966987b47d3677d0cef6d0cd18633f5fb32a7d7'], 'prepare_native': ['a5bf6bd721f4d9ed252eeb2227def08aaea4be7b98736102ee00f2a0a8953a2b', '4d5e05f5856c1a5a101208c444bffaac476161c006d71966f56919de16dab93d', '34ed6fe8f75af70f79763a7f0afcb2e2357f06ef1b655c091aca31d51c7e4d74', 'ca765426c9ffd917e7665a34f4b461557aa8008a2708b5096ea74930a2cd14ca', '32dbec26f3362261788d9aae94986e5a195ba519d765b5f1d49f894e81fdbeb1', '09ca8706a09b9ca00acc760e10bf0b639805722a218f873d3789cc137768a58a']}
RETAINED_NESTED = {'update': ['04433a7bccec471d9dd8e3c03c37ec7773274a7cc3e6c1a5d91f2ca3190c5f45', '083748fc61763407503ea98b1c02802d3bcc30c0a9f4540fbffed317c7166c4a', '0967d4c764710b90ef3dd92c55b2b4d68b19e9afcc98aecf6665ce03e7130684', '0ac6ce3500b6c1110f6b8e96f7780229c62d530f7596d1bc619994782d2c819f', '0dd57e0cf72cd8c41dc9c2fdf5e3a6e7b15d4d6b45ce3d4ef78d02a353c3fddc', '20927af6beb23021579101325cd75a1fc87c3140da2a5fc23e442c55ff137695', '2e99ca4363fa5aba872a08150eaf1cce84da618a10d1cfe7981be6588d26ab8c', '405a9c31479b7d88548be0991f210d5ebb5ed2129113898e40e60fb71e43e76e', '424ca1d4579dab6038f6ddd707bfdff7168134e06582f811e0a957beb72e6b3b', '461385d4a2741cdb626050345848552d0f21c9f70e184f2a7507bdf59cff017e', '479509cd399ddf8b006c42eead9a61e50e28be0881ffbbf811cafdfc091f0953', '4d2b4475b1767c5155f59c384d6adb19d0ac7023e408a47b7e98d854b74f4485', '4d5e05f5856c1a5a101208c444bffaac476161c006d71966f56919de16dab93d', '52d71335c8e8ae5fa4f9dce7ae9e5d3838dd504c64fc2b57507209e1e430aab4', '534c6f4c1b238c791c011796463686602631017e991a2a18aee0a04ca918f770', '537809e8bbce578240c6701654e2856dd8b67121483525ed995a260197c18aa8', '5869ad204934101fd5712ee3a3f8dc40c9f42bec11d50f6053d6d3d5cbdadca6', '58f1cf47aa114554fafed5fe0bf86b3848e309e743990543188ba764e60dc73c', '669a0052dce5acc44bad06a9c4b634f9e42cdd6519f7c248ae6d22db4304c5af', '69896a67869802002fcaedbd8aff37412095e3295fa211d1e639b7d8c0ece39c', '6dea4ebe2bffab269d94398249443bddd934f17adbc84ac082dc87a50b3b40f0', '6f1d52fe3ed0efeaa266358c952766ff744fd9a0040f2cdd9caf2ba09ee8780e', '73ed15a46f18ffc6009e5c73704ad0381d6cb0ccb6f377445de1356f6d001361', '79e30419a3be8307eea6470fb145212d91f07e91d91522877dac26e039f312cf', '7deaef48e153d39aa345a2d070ae58874b7ac9f5c83ece264541f063b2e83c46', '8081527f0a560f092c2dfeb9e14d0953dc9ba288594defcfbbbfe149bfad28a0', '87bca565dfd34bc5bfd7d9738f45e8f0e7666987538fc855fcbe60323e5380d5', '95f8dab19088df3922536a353780e8d27d7aa638cd426c433c851d61a8dbec76', 'a1bbf9e0db2da9337bf2ca24bc34e321c73cfc5f830749444b73ed8dd9a0e7e3', 'a1c5f489dfc781cc7b41304cf79826436f3f5744222e1a06c31ab1f2b579f79d', 'a3e31ca7ae0f9b61e1c5bc3ba648b58ff814d7eae906f6cc02494fa84f5a05fc', 'a524638bbb938056e7c16726ddc40796ed444d630e9a0288a45d4c0940356804', 'ac07d889bbfe8f581beea3eebb04cd4956702a8bc0d54c4ab71247f5c3db5208', 'ad1a190799df48db2ac3b585ee622880dc292227e3f7ca602a18cc957418964e', 'b2035e7f0413e4abb8b54ffc2021a12639320e0adac84142845e44ea91fbd3bb', 'b80d61ebbf8a66bcb3def1ba7403d73d48d7a15be0b36f55ceb6c4e1c24455d5', 'bf193c322291f40bdb8c2cea2bc71f333f618bc5b907f0c34cf39a068d798f52', 'bfc2e2c431fe705d927c539865c8f5ae0bf876e630958fd3c3755e643e5cd2a9', 'c6b533f53dde446dce6429411a444d3418555d3851dface8117544bc20169525', 'ca10f20f80349b90077bce130956687f86bbd201e7657b095f4c010c1e47e248', 'cbf7c085ed2bb78c86564d214c1dc49cd931650722ee777c68db1fd242b04240', 'd9cfa6fe7d76cfdde017fca0e2e00cd43cb88ae3e0e00e241781eecc921b2f88', 'dd911beea80467dfc8ce7ac61c3a8d141ec5609a585e194f02bd80aea461127a', 'e09f7604af2bffecc35a0a5e113909950b6cfa1d122fa8738b99c933ef1d0b7b', 'e7e0cb32ef94449b8181063af7b7ace686fd40fb2c8b8bcd5a422975d40bd556', 'ee03ae96a9e439afdcb84faa536d920ade9e5e92800f53d098b15c386ce1374a', 'f242b6c701fed789799a13a381f2f64b4d218351f7b52ee128e4dd7fa9a886b6'], 'load_inference': ['060c434334ee67c564fa18dcd1084eea87b0b2be52653f51e9917bcd97a52da8', '0d6dacb480cfc0007f5b1c8a52d6c6c13dce485fa8d5590298125519d8da7a7c', '0f4ad26150b408574332b03c6f7e60aa19a45234261634547029980861c7e382', '1699dbd84f8b9e5d78875e372075a05fb9a5c5c92533b5e00a21232f4e8b6fa9', '2830233c05c70858cf2fd4c1683585d949226f5fd6d367ef68a764f2b97b1baa', '2c16383f9b57ef57b5a1dfdf69b076429b55a7f0843b70e9e1f652e3e286e737', '2c9c62b9bf54399f7f21a97c774b31d3229be14a58b5f63914cd06fc7d0b70d1', '2ed16e32b741de087a5996461c630841d05be3b157268f6b5c2eb3469d1046cf', '36d4bdb2e96b0651ecbb721d3b4c9d68c95d8635ef42a4063df2abef09698377', '378eb8e73c3cb18ef84cd07873aabf6aa2e8f6103c71f1457374519b7a467e60', '3cc340604a83f9152b0926282460b59b9896f8fd359a9803359e99e116cb28b3', '4351459ac5d5a2d1d924c18859eef19a42f2042cf0f32150129f400fa6483de8', '4d5e05f5856c1a5a101208c444bffaac476161c006d71966f56919de16dab93d', '4e0b5e77cfe070e2a8bd61a152aa714b69f8bd035cdaf4cfd1f618c91aa8b16d', '50b888fb5e67dc47e804429136e2c44ffcea7727aa963770c874537a63ffd2a6', '5ce424038e38ffad48629d780959613daf1335d24618c42eeecdc03cbe8bee00', '6b262267ee94de3c40fb8efc05cb165f6b48b6c93da6eece22c55ccc5c6928ce', '6b2fc000d642a1e9f58de3e804ef104ed0480847637a3824802cc5929603e423', '6fff40904c64db1edf228e0718b341dcc670a820118a025bd8e249dc8498d98b', '77c7d734559db134ec9be6a64132a972aaf1ec37a82565f7785fcccd45ba59d4', '7d2cf57042a27ec31c4f72f20223eec6810d83af9120f79cd16c10aa501c4f01', '7eda7747dc65beadadea3da984529abea052f19685c807bd6dba82cfb4472fe2', '865986e8391252ad0ac7a1725cc290ed15fa553d1755dd0bb1c85bc0ee5067fa', '95fde7dcf9c32c9cd0186c919248c219d30881ce9e39eee577a90fa947da0556', '991bafc091321a59bd85fbd212ca6107cda8116dbb57f986d4d88b017994155d', '9aa2691b326413a61c55270e9e97d5d2f307cb739162f6704ad4d369ebe9b089', '9c7f9ae5434e9a7e231f62180e624d0c689109089fc3b0a3831f176efae4a3f3', 'a86bb01eef5cafbdd54e90848330f75b245ee5d39cbd52146b818ce965c0ddad', 'a8ad540c6fb1fb0850242e91f27fdeac221721b729641c6e8f7d64a96f1160ac', 'aa843ad08497ce2e91ad68ed4b4a4e3f1b896efa9f20980670137479d13cd272', 'ad1c462c1bd0b30ef99f8bba1c258e4bd46434e5e4a90586180c3a73f2e2e1b6', 'b2baa6969e14313dd2acd12c037ec5b72c5a4fe7168d3cbdcbb7412be18dd7d4', 'b357698030e54ba32b67e81d5e5ccd51659370344ada21259c81ba8e4d32c6a9', 'b6ad6254466a2fb49f5c96417f63cf2f1eded517ed69247214db7fb2f68dedb3', 'b9b94ec726006dfe2f3f341d43a6ec0380cbb706233344c097ee99fe5bbcab4d', 'bc2ad0b7c949042d1cbaef772862bdeb0b7b69bf9a105053c13af6555e4b6623', 'bd33c2fbf3d16feacd6fcc5158b673cfee83a1cc6f934930b1f1ad21ce4ba717', 'bdb2a01e54b27c70f29609c42e250a6d03b784be1f9e936ceb0fba5a27034312', 'c84c332e75a46a48e831815c0fad1200bee5b4c4c67f734163846b18478bf9a3', 'd39b079d4f979a7b0523e34a55c4091d02839e052d68f96f7ee7b222067f523d', 'd67739cbecd97b03eb2bc7a76baf13e9d4f632ce6845b4efb31b0623a2d30114', 'd7aa4c43205bcb6ea71da4e104e4f092ee89b02888103ef148e6f252ffdcb3f4', 'e408efa1010b371d84ea341f6a0f6ee0833025986dff610f8409467cccf21c42', 'e4f9b64c599a08d1dde47689de77c58f1137a698e010327673a8719e35a9be8d', 'e5bf40d3aceecd2f6fb6abe30b3a6deb367653d8aa20d0e3e2ee34f76e234b86', 'ed87a0ae05a2f975a175f2e71927e24d28dc4900dbcc4efef5768c1309ba288d', 'edb42a9e235442d4827c1157dae2026dcf5be225e97d2061a4ff43db026eda3e', 'f758480b6ef22982ae6c12a7118025d205e860f83b2cf0c41fbf8db70bd0b554', 'f777ba4617f736778fe3e07f648a1a87fd5c46179a703eb36a84ecad4daf5922', 'faba36c3e3e6b31bd6d93e70cc28d0bebef46192ee91df109bf198d5a719bbd9', 'ff660840da1e414a3f839c002c0cfb85ce29a9a19bb8f9ac5f48435400665d7d'], 'export_bundle': ['03db7bc5886c528365db67167192c82c1290fad88a365b0717a3c5234279a3c0', '0b7f866f41917266679bc2c920341e4a155ac90c3cf003e55faa8151326627a7', '0ff55e2dbf927702f81707d90ff0add08d83d8f888fabb6bc803330958cf60cd', '125e50e3fe07c0b333b44097010246dc0fa86192362cdfb9bdc41bf56388bebc', '12988ca70c49c195ecc62de1532fec088ef06663a1cf6e80658ded51f386c07f', '1549b1ac8844963dd0c95623457f1b84009ce9932ccd77c97b22da973746d67e', '1d8d49abed79043b9bf2d15e2a4fce948e5ef85d7ede9e521a511a70464ee556', '200eb4ecc938f0cba1184102260f08a30b15f0698e3b4dc2e70e33c0f510585b', '2530bdd84ff9111693f56b55701863ec441598703c86e3f29ac8073d18d80e6f', '2649ab7dbfa6eae369c6c7edbee296f9bbb91e697d2856821817666abbb14a70', '341531f3c9daf362bc0bc007fad9549f5daa06cd0f52e695d20af0fad79ade5f', '3417e1025774ca7bfa81bea5da5c5640a70bac5b6728bac32351d3bd70ecc8ba', '36d8a8257d8c55c121d9baf4df38cd5e230a56be5b1c0605a56433629db86e7c', '3ca62e67fb8091a358cd6c4d97845b0fb3d21ff2b581290272587de4ae3a1667', '42b68d60b932da591a827d52746e597e742de3dc5ee7cbe5ab2f37474dbf0ead', '4bf6d5d02384b7f8d9a5e6a1167dbde0824a9981e06de847fb76d6d90684be0c', '4d5e05f5856c1a5a101208c444bffaac476161c006d71966f56919de16dab93d', '5c5ce7dcdc09a6d736362f3af7a69fe65a424133f4184b24c54d2818e7c68d47', '5f003ea4759f08e209bf56a44de894b7c16b66f9bf17b091f72d7810ed35ea13', '5f16c2c11c31cd4dd05bb25b951484d9f71404a716f18675f8c57f997eb447c5', '6159840daafc91ce07d60531e71df5b73ff21c600e07f3e2bd5a72de25431a4d', '7ad158632d5c60c82d5acacd1f2c2e0094f2d106febb1d64bbba5e88bd688348', '838ec503901fa1a3459f856a283bba26105799fbdc3823656e242eb9395d7a91', '868b3cf4dc7dc7a633f3553367e471ae0fa90c4bf99eb697ad061be933c1c9fb', '8d45a273cfe51a34cfa784d8f4a956deb8349e6e97ce11321da8b89c7c24156b', '90a733afbf88a0f0ff148bb98679c5fc307388e80c3eba3229f4836774c699f8', '91c053482327cc255904e401ddba403aaea9e348dedf42e5ba82d8b3d188f0f8', '9ed426f37f40ee6bcfa8c1de51845d4a579457eb27aab51d8cec71f1355dd5c6', 'ac0c4709099bb985048f28b860cbfff3a81256d26c92c4bdfea3a193203b5562', 'acdc7eb72a25101f7f5186eea08de56b40db982574e3c56d61795c6925a8ec81', 'acf2244acd30c355fcd478c404a68f4c462846771798422313a3ebd894b8fb56', 'b19442cd5bbff927b67c6f3d439695deb8eae283ec7a809c352facaea5f08440', 'b25e82d90ad8f2b71fb8da6d419b25bdf2cefb89e84ed8c5f1f68d72a1059c23', 'ba27c7abadaf0787e01a58ddae38dfd9b7c8528606e25f968341ec78ae452bbb', 'c199a3378b9fe84646e1d801c4d906545aaf8464d9f3061809170a1aa71b5aa6', 'c317264cc164be978f63b4a82874bb5a5de007964a61f3024289605ebdf3d4bd', 'c6c6728ce13135246c195521341877095940aef643dc62fddef5961bc616cdd7', 'c97f7c4b4f77f87482a4361ebd83d9cec68bae1a1636bef289a0126b57652473', 'd375139cb17b394bb6204ee4b564d875862ed93edc42b79be8006d00511d892d', 'dc147913c75be5e575dbc560f74697fd9d0a87c1aa7c6edce131792fbb8090b0', 'e0d63c49d18b562ab3fbff2b988167ce2f775b932eb96bc9acae6626fe134c35', 'e1093362d229b6057dca5e1539d7a83c9a053951cab5dfde90b60b69b1e5c6ed', 'e3651c239428eac3c5a211e0813f2890d1a9be94a2f9822a2a262c2f1b0dc3de', 'eaf888e23cc61c107ec0d54a10f219382583f810784d2b5380deae2c363a1206', 'ec4e5e601858577e1f59f15045fa1da9b7cda94caa0a99cff8107db8fa724880', 'eec01fd84113ae3331c95140128dac02f1c3454df7d1dbc5c7a7246278278b41', 'eedcbe43b38b93d540b012dfe32683f858605c442e1a0c6e2ebf2ab33d969045', 'ef4bed1ea2865906f6cdd676c01a237f6752b208ea5d186f4c3fdf11db4cd6fd', 'f171c6b6b060bcad614549cd8cef7242bca4d91df53a225d6fa1595bfc75eab3', 'f1c1dfb413a975d351ba03b3318bc811dfc95fa0ee017a574c2d31899c3ffcfb', 'f2089d5c9e6a5eac41223f1893df02c15e6cad0e84f59caa57b95d239ae9c1ba', 'f34bd323df1bf4a214ecdeede2ba6883fa71218f3ecfbab7d149141839e1f39a', 'f86017adc0ace5187a56ec3c072ef529c9c4ef404139446cb5653f7ac0c612c3', 'faba36c3e3e6b31bd6d93e70cc28d0bebef46192ee91df109bf198d5a719bbd9', 'fc21608086fd3d2bf258dfcc5be75316bd1cf80ebe901b417f8bbcf91f026e07'], 'qualify_bundle': ['023afa5aba6d8259dbe83c495b050ff25a12a9df5765c7f5071ac2b916c1b66f', '06473882d1dffae5561626c7cecb49f7b918d979e348b64ad5112b027d58f1f2', '06f4d27b0729f96abf202a7f7023d9c8293855f400266c09a1565b95227797e1', '08544b83f20826a5ee2d1a686e0bcf3b4901ab0eae7a40189b37fb0cd34a5427', '0b3faa8419e6e621aab1f50c5545a0db79e6677d36c2d02524b51af793740c81', '1105f74946268aa50ff1a03962705b6b3b01e303fe432c31ee5dd39a47505de3', '1322c2172782ed7eec58da24131a270d8bf9fd8593f19894bacc0db70f44a5bd', '1514470657d497e3be49f6dc97863e9643c877af6918f6d6b11ba315cb663e35', '178e1ecbff52818cd6df488b4cc61f071f7cef97ec00480ee231b20222989989', '19f46193be6b032050546d58b0c837cf4347035cc7ac9a39d4f1c2aa555b91cd', '1d2b580da8631e860eceaaef9b15644e8f5c983c5a96d828b26dde7aaa5486c2', '22955cbf88b079686583907e613018baee2e6114853eb6a1bec06e354324e934', '26a075bbefed76f92411e982a765f78582d44af2c910f2795152a81eec39430b', '2f4f6dbadc88e1c46cf790e3044544857196309a9cf5a26676e44e31eb4840a3', '34ed6fe8f75af70f79763a7f0afcb2e2357f06ef1b655c091aca31d51c7e4d74', '3696439c55a6627dc62042926dbda1fac3d792858f8370d5eed00dd6ef9f02eb', '387bde589a7eeb175fc889a95825e08765cae7861705a66a3ffec3c25d532474', '3a7a363af16b60cfb6d3d767205456ab52d9a4ded45d59e1da4f88ba572b919d', '3b203cbfadbdd87371a7a225502308301d72f2914013a108f4d76493e990aeff', '3fee72abd632fb19f3026d54c85d1fb3be42d36977f8a25f529441fc5ef1791b', '41845e0dbe5135cf51510c81018fcc3f78e7771145415dd535d0c7e7297b1ee6', '437625465124542c21771d72a6b43d1574b44a951a7ff278a6b6e562c7368ecb', '4430a73da4a70fcafee70031bb1620b61ffc721edc264ed5564a5b5e941c9228', '4676e109f3dbf822c0ef8957e966987b47d3677d0cef6d0cd18633f5fb32a7d7', '46aa832e9e408a478dc6dcc4d5deaa1422726ad8ff9fb89048a2c171ae835852', '4d5e05f5856c1a5a101208c444bffaac476161c006d71966f56919de16dab93d', '4e2053f9a21b91577ea1fc22e84a7ed4f67340a5a2a6a75eb7918a48ef805526', '4e8981c88463f641e5a858c0cbf83d2446a9ec5a62a1a06087533e9998c71f97', '529b51d17abcc9b23266a259fdab7477212a653b850579c4ac092b8aac3b410e', '582a1f2b4607db7169c154ca273bc3dfca590b7ccc0614bebab91bcf9966759a', '58e88034c0fa3129c6f91da10b80e92c56edcc5ca14876502f54177ba3e661e8', '5e3c6b1cc72959d3c5e69f177a8689fe96943ff913a90941a847718f555be6a7', '622d643761fde126a3552422bbe7bd48cf8b649d51a3241d18fb4a02d86ef6c9', '680f6f11ececd3248365338be3db671078915e26b92bdaf313c6cdc18bd7f31e', '6a7e49eca3ed7908496e28134439f5652d5a05e3343a98d2041a3da3793da417', '702fde86feb3bd7367301739a9041addbf81e19eb6264c034072976f3ddce1c0', '73d0bf35a8ff28d1acd4cd6cd453dcc0c5ad4ce115c9abf4249d9e4f4788c0ac', '749f529ba3efc9034992a542ba24458dd4645159fec7a10e2dd0e9347685e0f1', '76316c1f0eda8b2cf241b0e4eaf5db4df02f73592533b82382d529d23fa91ba2', '7833e26b7055dbfd53a65840a753c0dec4050e31f0567d98db1808610d264899', '7eba070a93ad107f5761cf9eb71039602172ab4a199d8565999e30da3f130f60', '863bf274d5a34c4d0813febda81c98b57389ee09feffb04fb3bee12b58cc0356', '940ea06edcadd50c67280bfbaeb9445725f175a977154adc11965078226e4620', '96a6bf1ccca6572e898299011e9d70705230e00e1651356ba8ef890a61578427', '9d50b00388ad776f3168f5b4a4fc50a767a2b96b18028c76202022b4672c7d03', 'a210fdea02458b8cbb07dba2a5dfc1191ef91aa32bc3f5dcaa1c375e0842c836', 'a45d2f0b39caece0fd2032b5976d531bd3437a6dac6593eeff0e8ad15fa7348c', 'a82577e29be7f0dc62e8eb3de4921a8083f8f4743b12ae183134cae1c1592564', 'ae773b437e16e8b29a2dd6132cacbf20cea1af7a626cb8ad3867b84a3a337a6c', 'aeec618e45cd17744c0a874b1700e6378376b5812cbdea6b23e87a38119d3c0a', 'b40b0f780522b774247c55505efdcb142787d1e459d04e43a8686b8a283f540e', 'b70943fc761720fc5d3372a6fd71a1f53046ea504172c93c33d9d6005e62a5ab', 'bdfa9ba6efe26c47743cf20f152235fc9cdb8d9feb58b813f43e74be5010edda', 'd021ab49e85c8112215d302eb78bf631df3adcd20066a5a5cd907cb47eb70b38', 'd1531182cddedcff48483cacfd96daf77d383f101086fa290f34ede821128771', 'd972c210e8f346c41c1fbdd87e97e6e345905a62331d9174c5a273175e30f34b', 'db6b0372a9f5a8776344f7940d0905d640aa08fc1a605d37a104d98cb5d4728d', 'e0a0c0fdfbad845ab1e2f4d7991a696eaada3924375150ff7c5e9acb10d4ad81', 'e4d7c18a9136c9b083579b9924244e153716e0230f5ea1a40f0d9fbf15f149ee', 'e8d0c0ca4ca3dcedcaf837933ac8ec09ae25b10f6aaa5464dc93e4e16ca47b5f', 'eb355e0b2477a9d207a50accb1e41c8347717cb90a6348625683dd4c9e8e9888', 'f00ef20e15e3ea1fff5ad742c5aed85d027f8d1363adfa1b461cdf534efe1cde', 'f25dee98d7da186fad03308e01b2c46a8218880bc7f579af98471cc52190e0e7', 'fd7934d769e545c7625c76ba50d523323e9f81cbb115f8045eed187dfaf17dd5'], 'prepare_native': ['052c5b5936e0712991ed9afc4b3ed86e94cc3cb484bf97e76a393c4141de8bce', '09ca8706a09b9ca00acc760e10bf0b639805722a218f873d3789cc137768a58a', '0ca7c53399c8d54e19d87c92cabe23861b844be687d602ca57b7392dff66d814', '0f3046ba62281985273fb964d002885c52e5d9bcdae7d1f3b4d821d2c26a6eab', '1003ca56c70c495acb160bbbb0a4dbcb939b1fa2c3e84272f35168c4eb5d9cf3', '147ce359c9a4bfa1e20a03a33c3cf53eb2c5b43500943450ee3d107feaae9c5b', '186f33ae918f1105406d3472629e6d21b9db3ab502bca15835c80838e5907e6c', '199b5332ca0d3b82ae32019ccc25defa669caf5c67a642ffbe653cc82f782c6c', '1b13dafd26e57c7c1f77cd51ecd902ed18b9ae4d9f96503f172370073b34d45f', '1b24d9e5ac5495dcc31fa1c55aedcbb4dae294546dcd16b1934aa6d66676cee0', '27d19cbf736a1ca8c8e750157b48fe57b7ea20ebf71353b21bc6f88bd306e865', '2fa23ec6a0aae8013fdd77d4597a53829ccb0e3043c46937ea9fa72a9b3897e6', '32dbec26f3362261788d9aae94986e5a195ba519d765b5f1d49f894e81fdbeb1', '33eb738f6aca3655cf02a7879fadc897173b414517304857724bf3c47d2631c4', '34ed6fe8f75af70f79763a7f0afcb2e2357f06ef1b655c091aca31d51c7e4d74', '34f7d4f57fb1c98329d56c31f01c3bcedd61c9514553bfdecc03ab728863749f', '3647f2fb2b7e389eda92568f35ec747282288dd1ff1295977f92008f5db6fcf2', '3c03637d62a755c5e26f76cc2aee44842d8a84f00402b0fadcd06547cff20fec', '41d9b4948e507d1a1e23cd83422913a54e599a1812d3c7d50b10b923120b9b3d', '43bd5210c8d29bf62c92bbbcf4dfa79dff9d749bfb47b5b7e2cdeacd1dae95fd', '4bf6d5d02384b7f8d9a5e6a1167dbde0824a9981e06de847fb76d6d90684be0c', '4d5e05f5856c1a5a101208c444bffaac476161c006d71966f56919de16dab93d', '4fb1411d66deecba1775663921e9f6e1802ad7f82c64bf261e35f629603d0b94', '4fe369d60da914451416ed5295d72d34a94233ced6a720393f458ffcf112235e', '50b888fb5e67dc47e804429136e2c44ffcea7727aa963770c874537a63ffd2a6', '53c2e4f451372b4d2cb1120fc22808a2908602a56308568c1572d8addce4777b', '57b1d8b4dd1f60bcc6aecab63cb5ef670c6f3e0535f958ce8f7802928f614aca', '5de9eafceefa9a350461fbddc0e9ae5af38d725f55ef0d973682e5af76b5f027', '5e5860188f68a81bcbb6b2f7d353b494dec7a0d183a05b658148908674b5079a', '74388a8a3280f985c6bf10ab66cd5e015b56a3c4f5089190f839771cf1740688', '79fa764e255e14f791aaeeecfaefa540ad9435f0ad6c45062044f5f6d2ce9c4d', '817413a152534759364e4f0b9f22770139b1c090e8fb472edc95d9df1bbdf436', '8d45a273cfe51a34cfa784d8f4a956deb8349e6e97ce11321da8b89c7c24156b', '907c4ef0fc4d0be40deda6e46f349634f6e662321655e1085b383b2f9331a2a6', '9191e930bd663ce20e76e4c0b381cb807ee99fe0eeca49f6e0499122bca21fe2', '97662325cb293f9a21634df85cf49f65fa3a95feeb556de94e7cf37bea0eacbc', 'a5bf6bd721f4d9ed252eeb2227def08aaea4be7b98736102ee00f2a0a8953a2b', 'aa5e7a31bfbca7d927f3fa03d1c2352bdd8dea809528f8387f588b5ec08ed5ce', 'ac13485be4619905abf65f2c67d55880c55d52c314f0ca9656a96fd90aa607fc', 'b1d2633988ea7f05aa80e97239130bd2218491decfc38965bf14ce5547620776', 'b3ff60964605da68fdb1447ae8e19f5015f65a59e5be8ec08c4eae4f5e476391', 'b41f5a659717995dc9f7c25063bf99e8d0490c5aaf64d3bdce70fe41edc547cb', 'b4b62f1e4ae0aa682a14b2fc070ef00deed505773c2226d7f533dfb9e72e3b34', 'b94f28aae7ac61a4317b22e4bd9f244634f194f9e3da8392889052969f304f2b', 'bc0773f48dcf98afcc0b442126454627e0e79a3b30a5efb609d3825c15948a5e', 'c22c25320d6f831769ac5238016fabf9acadb66736cafbae097e0a43a3dd1f54', 'ca765426c9ffd917e7665a34f4b461557aa8008a2708b5096ea74930a2cd14ca', 'd03c96f38e660163a002c51e7f9a9f9e8d59c376bdc934f94934e9b2799f37ab', 'd663dd3bb05eb6a9683637f78bf1b023c6fcb3c348c7db1f038acea7e5449f1d', 'd9292c1d6cbf4bb0a6d8e73a1834b8637dc64a16f1b463be0f169a3af0a140bc', 'dc072c2502a481445b14c2cf0670199ceb735111dd5badd0a9b5c6b0b8f15448', 'e319c5a388fe7bbe85164a57889103e5f9baa0f4e255c1d1d7db5a0a132502f5', 'e6808495c776df8028468c82bc7b69c0c5e35fbb7a4bf2238c310242face37ee', 'efb94ad9bfa15d3bebfb10a2204be391de97a353a0337a4878c523ea636d922b', 'f315d1c3eb64f2961c4dc6e12f1e6c38c34bae6419cafbf45942016b4cbb6389', 'f606795d2447f4f4295dce3cca17e220c54b5924bc202a3b53ea408becb4a072', 'f8637c9dcb99861e4a9ceda4e8923c1f98bc9e755b5a9234537362ca8e8c8522', 'fa783f644c63114c026395ca2ccb9a7400345543da09529fa2e203790a187f86', 'fa9f7d1f8af0d6efad724c26b64f3b721b5740a63ca6af95c7cb9c2414944e32']}
if __name__ == '__main__': unittest.main()
