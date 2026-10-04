#!/usr/bin/env python3
"""Bounded stdlib/source falsifiers; no Torch, images, corpus or native work."""
import ast
import base64
import copy
import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import py_compile
import sys
import tarfile
import tempfile
import threading
import time
from types import FunctionType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().with_name('evaluate_siglip2_compact_ranking.py')
TRAINER_ROOT=Path(os.environ.get('SFORA_EVALUATOR_TRAINER_ROOT',str(PATH.parent)))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value

e=module('_compact_evaluation_tests',PATH)
math_helper=module('_compact_math_tests',PATH.with_name('evaluate_siglip2_genuine_views.py'))


# Exact live-top1 statements; the historical fullfeature bytes remain authoritative.
LIVE_TOP1_EDITS = ((None, "SCHEMA = 'siglip2-compact-live-top1-evaluation-v1'", "SCHEMA = 'siglip2-compact-fullfeature-residual-evaluation-v1'"), (None, "AUTHORITY_SCHEMA = 'siglip2-compact-live-top1-evaluation-launch-v1'", "AUTHORITY_SCHEMA = 'siglip2-compact-fullfeature-residual-evaluation-launch-v1'"), (None, "def check_paired_initialization(c,a):\n    for record,arm,names,shapes in ((c,'control',['A','C'],[[128,160],[128,1152]]),\n            (a,'candidate',['A','C'],[[128,160],[128,1152]])):\n        require(record['arm'] == record['identity']['arm'] == arm and\n            record['identity']['parameter_names'] == names and record['identity']['parameter_shapes'] == shapes and\n            type(record['identity']['parameter_shapes']) is list and\n            all(type(shape) is list and all(type(d) is int for d in shape)\n                for shape in record['identity']['parameter_shapes']),\n            'exact both-arm A,C optimizer roles differ')\n    require(all(c[k] == a[k] for k in ('source','initial_A_sha256','initial_C_sha256','mu_train_sha256',\n            'mu_train_provenance_sha256','initial_raw_unit_packed_sha256','numerical_flags')) and\n        all(c['identity'][k] == a['identity'][k] for k in ('static_sha256','initial_A_sha256','initial_C_sha256',\n            'mu_train_sha256','mu_train_provenance_sha256','source','numerical_flags',\n            'initial_cpu_rng_sha256','initial_cuda_rng_sha256')) and\n        [s['batch'] for s in c['steps']] == [s['batch'] for s in a['steps']],\n        'fresh same-seed complete initialization/RNG/schedule differs')", "def check_paired_initialization(c,a):\n    for record,arm,names,shapes in ((c,'control',['A'],[[128,160]]),\n            (a,'candidate',['A','C'],[[128,160],[128,1152]])):\n        require(record['arm'] == record['identity']['arm'] == arm and\n            record['identity']['parameter_names'] == names and record['identity']['parameter_shapes'] == shapes and\n            type(record['identity']['parameter_shapes']) is list and\n            all(type(shape) is list and all(type(d) is int for d in shape)\n                for shape in record['identity']['parameter_shapes']),\n            'exact control A / candidate A,C optimizer roles differ')\n    require(all(c[k] == a[k] for k in ('source','initial_A_sha256','initial_C_sha256','mu_train_sha256',\n            'mu_train_provenance_sha256','initial_raw_unit_packed_sha256','numerical_flags')) and\n        all(c['identity'][k] == a['identity'][k] for k in ('static_sha256','initial_A_sha256','initial_C_sha256',\n            'mu_train_sha256','mu_train_provenance_sha256','source','numerical_flags',\n            'initial_cpu_rng_sha256','initial_cuda_rng_sha256')) and\n        [s['batch'] for s in c['steps']] == [s['batch'] for s in a['steps']],\n        'fresh same-seed complete initialization/RNG/schedule differs')"), (None, "def check_residual_oracle(proof,arm):\n    keys={'C_exact_zero','residual_nonzero_witness','omitted_C_mutant_rejected','wrong_mu_mutant_rejected'}\n    require(arm in ARMS and isinstance(proof,dict) and proof.keys() == keys and\n        proof['C_exact_zero'] is False and\n        all(proof[k] is True for k in keys-{'C_exact_zero'}),\n        'complete both-arm nonzero C / omitted C / wrong mu oracle required')", "def check_residual_oracle(proof,arm):\n    keys={'C_exact_zero','residual_nonzero_witness','omitted_C_mutant_rejected','wrong_mu_mutant_rejected'}\n    require(arm in ARMS and isinstance(proof,dict) and proof.keys() == keys and\n        proof['C_exact_zero'] is (arm == 'control') and\n        all(proof[k] is (arm == 'candidate') for k in keys-{'C_exact_zero'}),\n        'complete role-specific nonzero C / omitted C / wrong mu oracle required')"), ('authority', "require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and\n        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-live-top1-v1' and\n        trainer.AUTHORITY_SCHEMA == 'siglip2-compact-live-top1-launch-v1' and\n        trainer.INFERENCE_SCHEMA == 'siglip2-compact-live-top1-inference-v1' and\n        trainer.BUNDLE_SCHEMA == 'siglip2-compact-live-top1-bundle-v1' and math_helper.ORDER == ORDER and\n        math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math contract differs')", "require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and\n        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-fullfeature-residual-v1' and\n        trainer.AUTHORITY_SCHEMA == 'siglip2-compact-fullfeature-residual-launch-v1' and\n        trainer.INFERENCE_SCHEMA == 'siglip2-compact-fullfeature-residual-inference-v1' and\n        trainer.BUNDLE_SCHEMA == 'siglip2-compact-fullfeature-residual-bundle-v1' and math_helper.ORDER == ORDER and\n        math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math contract differs')"), ('authenticate_payloads', "require(torch.count_nonzero(disk['C']).item() > 0,\n        'updated both-arm C must be nonzero')", "require((torch.count_nonzero(disk['C']).item() == 0) is (disk['arm'] == 'control'),\n        'updated candidate C nonzero / frozen exactzero control differs')"), (None, 'def fullfeature_oracle(context,state,features):\n    """Independent accepted concat plus explicit centered residual; never the owned wrapper."""\n    import torch\n    from torch.nn import functional as F\n    trainer,t=context[\'trainer\'],context[\'training_context\'];arm=state[\'arm\']\n    require(arm in ARMS, \'fullfeature oracle arm differs\')\n    base=trainer.helper_guard(t).raw_features(features,state[\'head_object\'],\n        torch.nn.Parameter(state[\'A\'].detach().clone()),state[\'means\'],\'concat\',t[\'legacy\'][\'quadratic\'])\n    C=state[\'C\'].detach().clone();mu=state[\'mu_train\'].detach().clone()\n    zero=torch.count_nonzero(C).item() == 0\n    require(not zero, \'updated both-arm C must be nonzero\')\n    raw=base+F.linear(features-mu,C)\n    output=packed_outputs(context,raw)\n    proof={\'C_exact_zero\':zero,\'residual_nonzero_witness\':False,\n        \'omitted_C_mutant_rejected\':False,\'wrong_mu_mutant_rejected\':False}\n    omitted=packed_outputs(context,base)\n    column=int(C.abs().sum(dim=0).argmax().item())\n    wrong_mu=mu.clone();wrong_mu[column]+=1./float(C[:,column].abs().max().item())\n    wrong=packed_outputs(context,base+F.linear(features-wrong_mu,C))\n    digest=trainer.fingerprint(t,output)\n    proof[\'residual_nonzero_witness\']=proof[\'omitted_C_mutant_rejected\']=(\n        digest != trainer.fingerprint(t,omitted))\n    proof[\'wrong_mu_mutant_rejected\']=digest != trainer.fingerprint(t,wrong)\n    check_residual_oracle(proof,arm)\n    return output,proof', 'def fullfeature_oracle(context,state,features):\n    """Independent accepted concat plus explicit centered residual; never the owned wrapper."""\n    import torch\n    from torch.nn import functional as F\n    trainer,t=context[\'trainer\'],context[\'training_context\'];arm=state[\'arm\']\n    require(arm in ARMS, \'fullfeature oracle arm differs\')\n    base=trainer.helper_guard(t).raw_features(features,state[\'head_object\'],\n        torch.nn.Parameter(state[\'A\'].detach().clone()),state[\'means\'],\'concat\',t[\'legacy\'][\'quadratic\'])\n    C=state[\'C\'].detach().clone();mu=state[\'mu_train\'].detach().clone()\n    zero=torch.count_nonzero(C).item() == 0\n    require(zero is (arm == \'control\'), \'updated candidate must have nonzero C; control must have exactzero C\')\n    raw=base if arm == \'control\' else base+F.linear(features-mu,C)\n    output=packed_outputs(context,raw)\n    proof={\'C_exact_zero\':zero,\'residual_nonzero_witness\':False,\n        \'omitted_C_mutant_rejected\':False,\'wrong_mu_mutant_rejected\':False}\n    if arm == \'candidate\':\n        omitted=packed_outputs(context,base)\n        column=int(C.abs().sum(dim=0).argmax().item())\n        wrong_mu=mu.clone();wrong_mu[column]+=1./float(C[:,column].abs().max().item())\n        wrong=packed_outputs(context,base+F.linear(features-wrong_mu,C))\n        digest=trainer.fingerprint(t,output)\n        proof[\'residual_nonzero_witness\']=proof[\'omitted_C_mutant_rejected\']=(\n            digest != trainer.fingerprint(t,omitted))\n        proof[\'wrong_mu_mutant_rejected\']=digest != trainer.fingerprint(t,wrong)\n    check_residual_oracle(proof,arm)\n    return output,proof'), (None, "TRAINING = {'root': '/home/riomus/runs/sfora-so400-live-top1-train-source-v1', 'execution_sha256': '0992550a70f38a2efe461abf6d0672fc8608ee976fca155bb16e66305da7cc81', 'code': {'train_siglip2_compact_ranking.py': 'bcabe2691a2a398b8d8a18a58d91fb8a83388b3004c6f5047ee712bcffb8e453', 'test_siglip2_compact_ranking.py': '3a511a3d256bd80e3210b06de2896c0fb93f8cd34a5a539be6f18cb00dbd823c'}}", "TRAINING = {'root': '/home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1', 'execution_sha256': '996ae38783d44c0cb01ef3bb8d5ba1817545d5296a0612da07b6c852b69bbf15', 'code': {'train_siglip2_compact_ranking.py': 'ddbbf0bc02eb62c3bb87fc29768ecbd5ec5e9d885fb53215244029413df00bad', 'test_siglip2_compact_ranking.py': '6b2e727d4aacc78e50c616024b37c34584d9b3193b8b8333ca148da57adf2a9b'}}"))


def inverse_live_top1_source(source):
    for _,new,old in LIVE_TOP1_EDITS:
        assert source.count(new)==1, 'live-top1 source statement differs'
        source=source.replace(new,old,1)
    return source


def inverse_live_top1_authority(tree):
    dump=lambda node:ast.dump(node,include_attributes=False)
    for scope,new,old in LIVE_TOP1_EDITS:
        expected=ast.parse(new).body[0];replacement=ast.parse(old).body[0]
        root=tree if scope is None else next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==scope)
        # Every declared edit is a direct statement in the named scope. Filter by
        # its identifier before the exact comparison; never traverse unrelated inventories.
        if isinstance(expected,ast.FunctionDef):
            candidates=[(i,n) for i,n in enumerate(root.body)
                if isinstance(n,ast.FunctionDef) and n.name==expected.name]
        elif isinstance(expected,ast.Assign):
            candidates=[(i,n) for i,n in enumerate(root.body)
                if isinstance(n,ast.Assign) and dump(n.targets[0])==dump(expected.targets[0])]
        else:
            candidates=[(i,n) for i,n in enumerate(root.body) if isinstance(n,ast.Expr) and
                isinstance(n.value,ast.Call) and n.value.args and
                dump(n.value.args[-1])==dump(expected.value.args[-1])]
        matches=[i for i,n in candidates if dump(n)==dump(expected)]
        assert len(matches)==1, 'live-top1 AST statement differs: '+str(scope)
        root.body[matches[0]]=copy.deepcopy(replacement)
    return tree


def inverse_live_top1_recipe(recipe):
    restored=copy.deepcopy(recipe)
    for key,(expected,old) in {'core': ('cache/target preparation + both-arm all-positive scoring + both-view forward/backward + every both-arm gallery forward/backward + both-arm fullfeature centering/transfers + candidate hinge selection/transfers/backward + optimizer', 'cache/target preparation + both-arm all-positive scoring + both-view forward/backward + every both-arm gallery forward/backward + candidate fullfeature centering/transfers + optimizer'), 'frozen': ('complete encoder448/config/buffers/processor/head/classifier/means/muTRAIN', 'complete encoder448/config/buffers/processor/head/classifier/means/muTRAIN; control C exactzero'), 'hinge': ('candidate coefficient1 relu(.05+s_nearest_wrong-s_nearest_positive) sum / (.05*2*K); control connected zero; K counts full scheduled B64 valid occurrences bothviews/micro16', None), 'hinge_coefficient': (1.0, None), 'hinge_margin': (0.05, None), 'residual': ('both arms original concat helper once + linear(actual normalized x - canonical TRAIN6355 mu,C); fresh accepted A0/zeroC; mean FP32 canonical actual CPU domain only', 'original concat helper once + candidate linear(actual normalized x - canonical TRAIN6355 mu,C); C128x1152 zero; control serialized frozen C bypass; mean FP32 canonical actual CPU domain only'), 'selection': ('detached scores for indices only; filter invalid anchors before indexing; exclude original-row self; highest score then ascending original row; gather same connected scorematrix; singleton gallery negatives connected', None), 'trainable_names': ({'candidate': ['A', 'C'], 'control': ['A', 'C']}, {'control': ['A'], 'candidate': ['A', 'C']}), 'trainable_scalars': ({'candidate': 167936, 'control': 167936}, {'control': 20480, 'candidate': 167936}), 'trainable_shapes': ({'candidate': [[128, 160], [128, 1152]], 'control': [[128, 160], [128, 1152]]}, {'control': [[128, 160]], 'candidate': [[128, 160], [128, 1152]]})}.items():
        assert key in restored and restored[key]==expected, 'live-top1 recipe statement differs: '+key
        if old is None:del restored[key]
        else:restored[key]=copy.deepcopy(old)
    return restored


def historical_fullfeature_evaluator():
    source=inverse_live_top1_source(PATH.read_text())
    assert hashlib.sha256(source.encode()).hexdigest()=='3684268825e8a03592ad355a0752b582245f1d05fd0bcb3ed084a01ef68140b0'
    restored=ModuleType('_historical_fullfeature_evaluator');restored.__file__=str(PATH)
    exec(compile(source,str(PATH),'exec'),vars(restored))
    return restored


# Exact fullfeature delta only; historical source and AST hashes below remain unchanged.
FULLFEATURE_SOURCE_EDITS = (("\nUNIT_STARTED = time.perf_counter()\nSCHEMA = 'siglip2-compact-fullfeature-residual-evaluation-v1'\nAUTHORITY_SCHEMA = 'siglip2-compact-fullfeature-residual-evaluation-launch-v1'\nFILES = {'evaluate_siglip2_compact_ranking.py', 'test_compact_ranking_evaluation.py'}\nTRAIN_FILES = {'train_siglip2_compact_ranking.py', 'test_siglip2_compact_ranking.py'}\nTRAINING = {'root': '/home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1', 'execution_sha256': '996ae38783d44c0cb01ef3bb8d5ba1817545d5296a0612da07b6c852b69bbf15', 'code': {'train_siglip2_compact_ranking.py': 'ddbbf0bc02eb62c3bb87fc29768ecbd5ec5e9d885fb53215244029413df00bad', 'test_siglip2_compact_ranking.py': '6b2e727d4aacc78e50c616024b37c34584d9b3193b8b8333ca148da57adf2a9b'}}\nARMS = ('control', 'candidate')\nSEEDS = (179061, 179069)\n", "\nUNIT_STARTED = time.perf_counter()\nSCHEMA = 'siglip2-compact-current-gallery-smooth-ap-evaluation-v1'\nAUTHORITY_SCHEMA = 'siglip2-compact-current-gallery-smooth-ap-evaluation-launch-v1'\nFILES = {'evaluate_siglip2_compact_ranking.py', 'test_compact_ranking_evaluation.py'}\nTRAIN_FILES = {'train_siglip2_compact_ranking.py', 'test_siglip2_compact_ranking.py'}\nTRAINING = {'code': {'test_siglip2_compact_ranking.py': '32f9ade5717d98e9aa5f6644ad7ac0dc5992bb21e0861730e154d0d78c8a7744', 'train_siglip2_compact_ranking.py': 'affb911bd76a760b480e616d13dc78710cc56c74ce92ddd27e1cc8aaa3483480'}, 'execution_sha256': '73cbc39e857ad99e2962a6252c8c158b232636889b5023c28a97b1be7edda68e', 'root': '/home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1'}\nARMS = ('control', 'candidate')\nSEEDS = (179061, 179069)\n"), ("            'authenticated helper live source/global binding changed')\n        bound_file({},path,context['guards'][str(path)])\ndef check_paired_initialization(c,a):\n    for record,arm,names,shapes in ((c,'control',['A'],[[128,160]]),\n            (a,'candidate',['A','C'],[[128,160],[128,1152]])):\n        require(record['arm'] == record['identity']['arm'] == arm and\n            record['identity']['parameter_names'] == names and record['identity']['parameter_shapes'] == shapes and\n            type(record['identity']['parameter_shapes']) is list and\n            all(type(shape) is list and all(type(d) is int for d in shape)\n                for shape in record['identity']['parameter_shapes']),\n            'exact control A / candidate A,C optimizer roles differ')\n    require(all(c[k] == a[k] for k in ('source','initial_A_sha256','initial_C_sha256','mu_train_sha256',\n            'mu_train_provenance_sha256','initial_raw_unit_packed_sha256','numerical_flags')) and\n        all(c['identity'][k] == a['identity'][k] for k in ('static_sha256','initial_A_sha256','initial_C_sha256',\n            'mu_train_sha256','mu_train_provenance_sha256','source','numerical_flags',\n            'initial_cpu_rng_sha256','initial_cuda_rng_sha256')) and\n        [s['batch'] for s in c['steps']] == [s['batch'] for s in a['steps']],\n        'fresh same-seed complete initialization/RNG/schedule differs')\n\n\ndef check_residual_oracle(proof,arm):\n    keys={'C_exact_zero','residual_nonzero_witness','omitted_C_mutant_rejected','wrong_mu_mutant_rejected'}\n    require(arm in ARMS and isinstance(proof,dict) and proof.keys() == keys and\n        proof['C_exact_zero'] is (arm == 'control') and\n        all(proof[k] is (arm == 'candidate') for k in keys-{'C_exact_zero'}),\n        'complete role-specific nonzero C / omitted C / wrong mu oracle required')\n\n\ndef authority(args):\n    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import preceded source admission')\n", "            'authenticated helper live source/global binding changed')\n        bound_file({},path,context['guards'][str(path)])\ndef authority(args):\n    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import preceded source admission')\n"), ("    trainer,native,math_helper,reference = (loaded[k] for k in ('training','nearest_evaluator','genuine_evaluator','reference'))\n    require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and\n        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-fullfeature-residual-v1' and\n        trainer.AUTHORITY_SCHEMA == 'siglip2-compact-fullfeature-residual-launch-v1' and\n        trainer.INFERENCE_SCHEMA == 'siglip2-compact-fullfeature-residual-inference-v1' and\n        trainer.BUNDLE_SCHEMA == 'siglip2-compact-fullfeature-residual-bundle-v1' and math_helper.ORDER == ORDER and\n        math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math contract differs')\n    first = launch['endpoints'][0]; training = launch['training']\n", "    trainer,native,math_helper,reference = (loaded[k] for k in ('training','nearest_evaluator','genuine_evaluator','reference'))\n    require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and\n        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-current-gallery-smooth-ap-v1' and\n        trainer.AUTHORITY_SCHEMA == 'siglip2-compact-current-gallery-smooth-ap-launch-v1' and\n        trainer.INFERENCE_SCHEMA == 'siglip2-compact-current-gallery-smooth-ap-inference-v1' and\n        trainer.BUNDLE_SCHEMA == 'siglip2-compact-current-gallery-smooth-ap-bundle-v1' and math_helper.ORDER == ORDER and\n        math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math contract differs')\n    first = launch['endpoints'][0]; training = launch['training']\n"), ("    for seed in seeds(launch['stage']):\n        c,a = (records[seed,arm] for arm in ARMS)\n        check_paired_initialization(c,a)\n    require(len({e['checkpoint']['sha256'] for e in launch['endpoints']}) == len(launch['endpoints']),\n        'trained state reused between endpoints')\n", "    for seed in seeds(launch['stage']):\n        c,a = (records[seed,arm] for arm in ARMS)\n        require(all(c[k] == a[k] for k in ('source','initial_A_sha256','initial_raw_unit_packed_sha256','numerical_flags')) and\n            all(c['identity'][k] == a['identity'][k] for k in ('static_sha256','initial_A_sha256','parameter_names',\n                'source','numerical_flags','initial_cpu_rng_sha256','initial_cuda_rng_sha256')) and\n            [s['batch'] for s in c['steps']] == [s['batch'] for s in a['steps']],\n            'fresh same-seed complete initialization/RNG/schedule differs')\n    require(len({e['checkpoint']['sha256'] for e in launch['endpoints']}) == len(launch['endpoints']),\n        'trained state reused between endpoints')\n"), ("            record['calibration']['same_role_forward_exact'] is True and record['calibration']['raw_unit_packed_exact'] is True,\n            'CPU full-payload/bundle/synthetic metadata qualification differs')\n        require(record['calibration']['residual_oracles'].keys() == {label(e) for e in endpoints},\n            'complete synthetic endpoint residual proofs required')\n        for endpoint in endpoints:\n            check_residual_oracle(record['calibration']['residual_oracles'][label(endpoint)],endpoint['arm'])\n    elif phase == 'export':\n        endpoint = next(e for e in endpoints if (e['seed'],e['arm']) == (seed,arm)); key=label(endpoint)\n", "            record['calibration']['same_role_forward_exact'] is True and record['calibration']['raw_unit_packed_exact'] is True,\n            'CPU full-payload/bundle/synthetic metadata qualification differs')\n    elif phase == 'export':\n        endpoint = next(e for e in endpoints if (e['seed'],e['arm']) == (seed,arm)); key=label(endpoint)\n"), ("        context['reference'].check_value_facts(record['panel_facts'],PANELS[panel][0])\n        require(len(record['train_witness']['batch']) == 16, 'actual TRAIN micro16 witness required')\n        check_residual_oracle(record['train_witness']['residual_oracle'],arm)\n    else:\n        decision = decide(context['math'],record['quality'],record['source_quality'],record['concat_quality'],stage,panel,\n", "        context['reference'].check_value_facts(record['panel_facts'],PANELS[panel][0])\n        require(len(record['train_witness']['batch']) == 16, 'actual TRAIN micro16 witness required')\n    else:\n        decision = decide(context['math'],record['quality'],record['source_quality'],record['concat_quality'],stage,panel,\n"), ("            trainer.fingerprint(t,disk,consumed=pages.consume) == endpoint['terminal_state_sha256'],\n            'complete TRAIN128 current bytes/actual identity differ')\n        members={k:trainer.fingerprint(t,disk[k]) for k in ('config','buffers','processor','head','A','means',\n            'C','mu_train','mu_train_provenance')}\n        members['arm']=trainer.fingerprint(t,ident['arm'])\n        ident=trainer.clone(t,ident)\n    del disk,pages; gc.collect()\n", "            trainer.fingerprint(t,disk,consumed=pages.consume) == endpoint['terminal_state_sha256'],\n            'complete TRAIN128 current bytes/actual identity differ')\n        members={k:trainer.fingerprint(t,disk[k]) for k in ('config','buffers','processor','head','A','means')}\n        ident=trainer.clone(t,ident)\n    del disk,pages; gc.collect()\n"), ("        trainer.fingerprint(t,{k:v for k,v in disk.items() if k != 'fixed_sha256'}) == disk['fixed_sha256'] and\n        all(trainer.fingerprint(t,disk[k]) == h for k,h in members.items()) and\n        disk['numerical_flags'] == t['flags'] and disk['arm'] == endpoint['arm'] == ident['arm'],\n        'bundle substitutes trained A/C/mu/arm or complete fixed endpoint members')\n    for name,shape in (('C',(128,1152)),('mu_train',(1152,))):\n        t['legacy']['quadratic']._check_tensor(disk[name],shape,'cpu',frozen=True)\n    require((torch.count_nonzero(disk['C']).item() == 0) is (disk['arm'] == 'control'),\n        'updated candidate C nonzero / frozen exactzero control differs')\n    require(disk['source']['initial_C_sha256'] == ident['initial_C_sha256'] and\n        disk['source']['mu_train_sha256'] == ident['mu_train_sha256'] == trainer.fingerprint(t,disk['mu_train']) and\n        disk['source']['mu_train_provenance_sha256'] == ident['mu_train_provenance_sha256'] ==\n            trainer.fingerprint(t,disk['mu_train_provenance']), 'bundle initial C / immutable TRAIN mean provenance differs')\n    require(disk['source']['accepted_A_sha256'] == t['initial_A_sha256'] and\n        disk['source']['encoder_checkpoint_sha256'] == t['initial']['provenance']['encoder']['checkpoint']['sha256'] and\n", "        trainer.fingerprint(t,{k:v for k,v in disk.items() if k != 'fixed_sha256'}) == disk['fixed_sha256'] and\n        all(trainer.fingerprint(t,disk[k]) == h for k,h in members.items()) and\n        disk['numerical_flags'] == t['flags'], 'bundle substitutes trained A or complete fixed endpoint members')\n    require(disk['source']['accepted_A_sha256'] == t['initial_A_sha256'] and\n        disk['source']['encoder_checkpoint_sha256'] == t['initial']['provenance']['encoder']['checkpoint']['sha256'] and\n"), ('\n\ndef fullfeature_oracle(context,state,features):\n    """Independent accepted concat plus explicit centered residual; never the owned wrapper."""\n    import torch\n    from torch.nn import functional as F\n    trainer,t=context[\'trainer\'],context[\'training_context\'];arm=state[\'arm\']\n    require(arm in ARMS, \'fullfeature oracle arm differs\')\n    base=trainer.helper_guard(t).raw_features(features,state[\'head_object\'],\n        torch.nn.Parameter(state[\'A\'].detach().clone()),state[\'means\'],\'concat\',t[\'legacy\'][\'quadratic\'])\n    C=state[\'C\'].detach().clone();mu=state[\'mu_train\'].detach().clone()\n    zero=torch.count_nonzero(C).item() == 0\n    require(zero is (arm == \'control\'), \'updated candidate must have nonzero C; control must have exactzero C\')\n    raw=base if arm == \'control\' else base+F.linear(features-mu,C)\n    output=packed_outputs(context,raw)\n    proof={\'C_exact_zero\':zero,\'residual_nonzero_witness\':False,\n        \'omitted_C_mutant_rejected\':False,\'wrong_mu_mutant_rejected\':False}\n    if arm == \'candidate\':\n        omitted=packed_outputs(context,base)\n        column=int(C.abs().sum(dim=0).argmax().item())\n        wrong_mu=mu.clone();wrong_mu[column]+=1./float(C[:,column].abs().max().item())\n        wrong=packed_outputs(context,base+F.linear(features-wrong_mu,C))\n        digest=trainer.fingerprint(t,output)\n        proof[\'residual_nonzero_witness\']=proof[\'omitted_C_mutant_rejected\']=(\n            digest != trainer.fingerprint(t,omitted))\n        proof[\'wrong_mu_mutant_rejected\']=digest != trainer.fingerprint(t,wrong)\n    check_residual_oracle(proof,arm)\n    return output,proof\n\n\ndef cpu_calibration(context):\n    """Updated bundle readouts versus the accepted helper, synthetic features only."""\n', '\n\ndef cpu_calibration(context):\n    """Updated bundle readouts versus the accepted helper, synthetic features only."""\n'), ("    from torch.nn import functional as F\n    t=context['training_context'];trainer=context['trainer'];legacy=t['legacy'];facts={}\n    oracles={}\n    features=F.normalize(torch.linspace(-1,1,32*1152,dtype=torch.float32).reshape(32,1152),dim=1)\n    for endpoint in context['launch']['endpoints']:\n", "    from torch.nn import functional as F\n    t=context['training_context'];trainer=context['trainer'];legacy=t['legacy'];facts={}\n    features=F.normalize(torch.linspace(-1,1,32*1152,dtype=torch.float32).reshape(32,1152),dim=1)\n    for endpoint in context['launch']['endpoints']:\n"), ("        A=torch.nn.Parameter(disk['A'].clone())\n        with torch.no_grad(),torch.autocast('cpu',enabled=False):\n            raw=trainer.fullfeature_raw_features(features,head,A,disk['means'],disk['C'],disk['mu_train'],\n                disk['arm'],legacy['quadratic'],serving)\n            first=packed_outputs(context,raw)\n            second,proof=fullfeature_oracle(context,{'head_object':head,'A':A,**disk},features.clone())\n            context['helper'].exact(tuple_outputs(first),tuple_outputs(second))\n            require(trainer.fingerprint(t,first) == trainer.fingerprint(t,second), 'synthetic updated raw/unit/pack/wire parity differs')\n        facts[label(endpoint)]=trainer.fingerprint(t,first)\n        oracles[label(endpoint)]=proof\n        require(sys.modules.pop(name,None) is serving, 'synthetic helper registry changed')\n        del head,A,raw,first,second,disk,serving;gc.collect()\n    del features\n    return {'role':'synthetic CPU FP32 updated readout','rows':32,'same_role_forward_exact':True,\n        'raw_unit_packed_exact':True,'outputs_sha256':facts,'residual_oracles':oracles}\n\n\n", "        A=torch.nn.Parameter(disk['A'].clone())\n        with torch.no_grad(),torch.autocast('cpu',enabled=False):\n            raw=serving.raw_features(features,head,A,disk['means'],'concat',legacy['quadratic'])\n            oracle=trainer.helper_guard(t).raw_features(features.clone(),head,torch.nn.Parameter(A.detach().clone()),\n                disk['means'],'concat',legacy['quadratic'])\n            first,second=packed_outputs(context,raw),packed_outputs(context,oracle)\n            context['helper'].exact(tuple_outputs(first),tuple_outputs(second))\n            require(trainer.fingerprint(t,first) == trainer.fingerprint(t,second), 'synthetic updated raw/unit/pack/wire parity differs')\n        facts[label(endpoint)]=trainer.fingerprint(t,first)\n        require(sys.modules.pop(name,None) is serving, 'synthetic helper registry changed')\n        del head,A,raw,oracle,first,second,disk,serving;gc.collect()\n    del features\n    return {'role':'synthetic CPU FP32 updated readout','rows':32,'same_role_forward_exact':True,\n        'raw_unit_packed_exact':True,'outputs_sha256':facts}\n\n\n"), ("            json.loads(state['processor_object'].to_json_string()) if k == 'processor_config' else\n            dict(state['head_object'].state_dict()) if k == 'head' else state[k])\n            for k in ('config','buffers','processor_config','head','A','means','C','mu_train','mu_train_provenance','arm')}}\n\n\n", "            json.loads(state['processor_object'].to_json_string()) if k == 'processor_config' else\n            dict(state['head_object'].state_dict()) if k == 'head' else state[k])\n            for k in ('config','buffers','processor_config','head','A','means')}}\n\n\n"), ("                with torch.autocast('cuda',enabled=False):\n                    features=F.normalize(pooled.float(),dim=1)\n                    expected,proof=fullfeature_oracle(context,state,features)\n            context['helper'].exact(tuple_outputs(values),tuple_outputs(expected))\n            require(trainer.fingerprint(t,values) == trainer.fingerprint(t,expected), 'same-role original readout/pack/wire differs')\n            del pooled,features,expected\n        fact={'rows':rows,'rgb_sha256':rgb.hexdigest(),'pixels_sha256':trainer.fingerprint(t,pixels),\n            'outputs_sha256':trainer.fingerprint(t,values),'residual_oracle':proof if oracle else None}\n        del pixels\n        return values,fact\n", "                with torch.autocast('cuda',enabled=False):\n                    features=F.normalize(pooled.float(),dim=1)\n                    A=torch.nn.Parameter(state['A'].detach().clone())\n                    raw=trainer.helper_guard(t).raw_features(features,state['head_object'],A,state['means'],\n                        'concat',t['legacy']['quadratic'])\n                    expected=packed_outputs(context,raw)\n            context['helper'].exact(tuple_outputs(values),tuple_outputs(expected))\n            require(trainer.fingerprint(t,values) == trainer.fingerprint(t,expected), 'same-role original readout/pack/wire differs')\n            del pooled,features,A,raw,expected\n        fact={'rows':rows,'rgb_sha256':rgb.hexdigest(),'pixels_sha256':trainer.fingerprint(t,pixels),\n            'outputs_sha256':trainer.fingerprint(t,values)}\n        del pixels\n        return values,fact\n"), ("        cache=t['initial']['views']['canonical'][ids]\n        head=t['legacy']['selected']['cached'].head_from('control',tensors=t['initial']['head']).requires_grad_(False).train()\n        expected,_=fullfeature_oracle(context,{'head_object':head,'A':state['A'].detach().cpu(),\n            'means':{k:v.cpu() for k,v in state['means'].items()},'C':state['C'].detach().cpu(),\n            'mu_train':state['mu_train'].detach().cpu(),'arm':state['arm']},cache)\n        raw=expected['raw']\n    difference=values['raw']-raw\n    result={'batch':ids,**fact,'cache_native_drift_max_abs':float(difference.abs().max()),\n        'cache_native_drift_l2':float(difference.double().norm()),'arithmetic_role':'CUDA FP16 vision / FP32 readout micro16',\n        'role_drift_is_diagnostic':True}\n    del values,raw,expected,cache,head,difference; gc.collect()\n    return result\n\n", "        cache=t['initial']['views']['canonical'][ids]\n        head=t['legacy']['selected']['cached'].head_from('control',tensors=t['initial']['head']).requires_grad_(False).train()\n        raw=trainer.helper_guard(t).raw_features(cache,head,torch.nn.Parameter(state['A'].detach().cpu().clone()),\n            {k:v.cpu() for k,v in state['means'].items()},'concat',t['legacy']['quadratic'])\n    difference=values['raw']-raw\n    result={'batch':ids,**fact,'cache_native_drift_max_abs':float(difference.abs().max()),\n        'cache_native_drift_l2':float(difference.double().norm()),'arithmetic_role':'CUDA FP16 vision / FP32 readout micro16',\n        'role_drift_is_diagnostic':True}\n    del values,raw,cache,head,difference; gc.collect()\n    return result\n\n"), ("            print(json.dumps({'event':'COMPACT_TIMING','stage':'endpoint_facts','boundary':'end','pass_index':pass_index,'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)\n            require(model_facts['vision_sha256'] == facts['vision_sha256'] and\n                all(model_facts['members'][k] == facts['members'][k] for k in\n                    ('config','buffers','head','A','means','C','mu_train','mu_train_provenance','arm')),\n                'independent complete frozen vision/updated A/C/mu/arm differs')\n            require(model_facts['members']['processor_config'] == facts['processor_config_sha256'],\n                'bundle-owned processor config differs')\n", "            print(json.dumps({'event':'COMPACT_TIMING','stage':'endpoint_facts','boundary':'end','pass_index':pass_index,'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)\n            require(model_facts['vision_sha256'] == facts['vision_sha256'] and\n                all(model_facts['members'][k] == facts['members'][k] for k in ('config','buffers','head','A','means')),\n                'independent complete frozen vision/updated A differs')\n            require(model_facts['members']['processor_config'] == facts['processor_config_sha256'],\n                'bundle-owned processor config differs')\n"))
FULLFEATURE_AST_EDITS = (('authority', "require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and\n    tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-fullfeature-residual-v1' and\n    trainer.AUTHORITY_SCHEMA == 'siglip2-compact-fullfeature-residual-launch-v1' and\n    trainer.INFERENCE_SCHEMA == 'siglip2-compact-fullfeature-residual-inference-v1' and\n    trainer.BUNDLE_SCHEMA == 'siglip2-compact-fullfeature-residual-bundle-v1' and math_helper.ORDER == ORDER and\n    math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math contract differs')\n", "require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and\n    tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-current-gallery-smooth-ap-v1' and\n    trainer.AUTHORITY_SCHEMA == 'siglip2-compact-current-gallery-smooth-ap-launch-v1' and\n    trainer.INFERENCE_SCHEMA == 'siglip2-compact-current-gallery-smooth-ap-inference-v1' and\n    trainer.BUNDLE_SCHEMA == 'siglip2-compact-current-gallery-smooth-ap-bundle-v1' and math_helper.ORDER == ORDER and\n    math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math contract differs')\n"), ('authority', 'check_paired_initialization(c,a)\n', "require(all(c[k] == a[k] for k in ('source','initial_A_sha256','initial_raw_unit_packed_sha256','numerical_flags')) and\n    all(c['identity'][k] == a['identity'][k] for k in ('static_sha256','initial_A_sha256','parameter_names',\n        'source','numerical_flags','initial_cpu_rng_sha256','initial_cuda_rng_sha256')) and\n    [s['batch'] for s in c['steps']] == [s['batch'] for s in a['steps']],\n    'fresh same-seed complete initialization/RNG/schedule differs')\n"), ('check_receipt', "require(record['calibration']['residual_oracles'].keys() == {label(e) for e in endpoints},\n    'complete synthetic endpoint residual proofs required')\nfor endpoint in endpoints:\n    check_residual_oracle(record['calibration']['residual_oracles'][label(endpoint)],endpoint['arm'])\n", ''), ('check_receipt', "check_residual_oracle(record['train_witness']['residual_oracle'],arm)\n", ''), ('authenticate_payloads', "members={k:trainer.fingerprint(t,disk[k]) for k in ('config','buffers','processor','head','A','means',\n    'C','mu_train','mu_train_provenance')}\nmembers['arm']=trainer.fingerprint(t,ident['arm'])\n", "members={k:trainer.fingerprint(t,disk[k]) for k in ('config','buffers','processor','head','A','means')}\n"), ('authenticate_payloads', "require(disk.keys() == trainer.INFERENCE_KEYS and disk['schema'] == trainer.INFERENCE_SCHEMA and\n    trainer.fingerprint(t,disk) == manifest['endpoint_state_sha256'] == endpoint['inference_state_sha256'] and\n    trainer.fingerprint(t,{k:v for k,v in disk.items() if k != 'fixed_sha256'}) == disk['fixed_sha256'] and\n    all(trainer.fingerprint(t,disk[k]) == h for k,h in members.items()) and\n    disk['numerical_flags'] == t['flags'] and disk['arm'] == endpoint['arm'] == ident['arm'],\n    'bundle substitutes trained A/C/mu/arm or complete fixed endpoint members')\nfor name,shape in (('C',(128,1152)),('mu_train',(1152,))):\n    t['legacy']['quadratic']._check_tensor(disk[name],shape,'cpu',frozen=True)\nrequire((torch.count_nonzero(disk['C']).item() == 0) is (disk['arm'] == 'control'),\n    'updated candidate C nonzero / frozen exactzero control differs')\nrequire(disk['source']['initial_C_sha256'] == ident['initial_C_sha256'] and\n    disk['source']['mu_train_sha256'] == ident['mu_train_sha256'] == trainer.fingerprint(t,disk['mu_train']) and\n    disk['source']['mu_train_provenance_sha256'] == ident['mu_train_provenance_sha256'] ==\n        trainer.fingerprint(t,disk['mu_train_provenance']), 'bundle initial C / immutable TRAIN mean provenance differs')\n", "require(disk.keys() == trainer.INFERENCE_KEYS and disk['schema'] == trainer.INFERENCE_SCHEMA and\n    trainer.fingerprint(t,disk) == manifest['endpoint_state_sha256'] == endpoint['inference_state_sha256'] and\n    trainer.fingerprint(t,{k:v for k,v in disk.items() if k != 'fixed_sha256'}) == disk['fixed_sha256'] and\n    all(trainer.fingerprint(t,disk[k]) == h for k,h in members.items()) and\n    disk['numerical_flags'] == t['flags'], 'bundle substitutes trained A or complete fixed endpoint members')\n"), ('cpu_calibration', 'oracles={}\n', ''), ('cpu_calibration', "raw=trainer.fullfeature_raw_features(features,head,A,disk['means'],disk['C'],disk['mu_train'],\n    disk['arm'],legacy['quadratic'],serving)\nfirst=packed_outputs(context,raw)\nsecond,proof=fullfeature_oracle(context,{'head_object':head,'A':A,**disk},features.clone())\n", "raw=serving.raw_features(features,head,A,disk['means'],'concat',legacy['quadratic'])\noracle=trainer.helper_guard(t).raw_features(features.clone(),head,torch.nn.Parameter(A.detach().clone()),\n    disk['means'],'concat',legacy['quadratic'])\nfirst,second=packed_outputs(context,raw),packed_outputs(context,oracle)\n"), ('cpu_calibration', 'oracles[label(endpoint)]=proof\n', ''), ('cpu_calibration', 'del head,A,raw,first,second,disk,serving;gc.collect()\n', 'del head,A,raw,oracle,first,second,disk,serving;gc.collect()\n'), ('cpu_calibration', "return {'role':'synthetic CPU FP32 updated readout','rows':32,'same_role_forward_exact':True,\n    'raw_unit_packed_exact':True,'outputs_sha256':facts,'residual_oracles':oracles}\n", "return {'role':'synthetic CPU FP32 updated readout','rows':32,'same_role_forward_exact':True,\n    'raw_unit_packed_exact':True,'outputs_sha256':facts}\n"), ('endpoint_facts', "return {'vision_sha256':trainer.fingerprint(t,state['model'].state_dict()),\n    'members':{k:trainer.fingerprint(t,config if k == 'config' else\n        dict(state['model'].named_buffers()) if k == 'buffers' else\n        json.loads(state['processor_object'].to_json_string()) if k == 'processor_config' else\n        dict(state['head_object'].state_dict()) if k == 'head' else state[k])\n        for k in ('config','buffers','processor_config','head','A','means','C','mu_train','mu_train_provenance','arm')}}\n", "return {'vision_sha256':trainer.fingerprint(t,state['model'].state_dict()),\n    'members':{k:trainer.fingerprint(t,config if k == 'config' else\n        dict(state['model'].named_buffers()) if k == 'buffers' else\n        json.loads(state['processor_object'].to_json_string()) if k == 'processor_config' else\n        dict(state['head_object'].state_dict()) if k == 'head' else state[k])\n        for k in ('config','buffers','processor_config','head','A','means')}}\n"), ('images_outputs', "if oracle:\n    with torch.no_grad():\n        with torch.autocast('cuda',dtype=torch.float16):\n            pooled=state['model'](pixel_values=pixels.to('cuda')).pooler_output\n        with torch.autocast('cuda',enabled=False):\n            features=F.normalize(pooled.float(),dim=1)\n            expected,proof=fullfeature_oracle(context,state,features)\n    context['helper'].exact(tuple_outputs(values),tuple_outputs(expected))\n    require(trainer.fingerprint(t,values) == trainer.fingerprint(t,expected), 'same-role original readout/pack/wire differs')\n    del pooled,features,expected\nfact={'rows':rows,'rgb_sha256':rgb.hexdigest(),'pixels_sha256':trainer.fingerprint(t,pixels),\n    'outputs_sha256':trainer.fingerprint(t,values),'residual_oracle':proof if oracle else None}\n", "if oracle:\n    with torch.no_grad():\n        with torch.autocast('cuda',dtype=torch.float16):\n            pooled=state['model'](pixel_values=pixels.to('cuda')).pooler_output\n        with torch.autocast('cuda',enabled=False):\n            features=F.normalize(pooled.float(),dim=1)\n            A=torch.nn.Parameter(state['A'].detach().clone())\n            raw=trainer.helper_guard(t).raw_features(features,state['head_object'],A,state['means'],\n                'concat',t['legacy']['quadratic'])\n            expected=packed_outputs(context,raw)\n    context['helper'].exact(tuple_outputs(values),tuple_outputs(expected))\n    require(trainer.fingerprint(t,values) == trainer.fingerprint(t,expected), 'same-role original readout/pack/wire differs')\n    del pooled,features,A,raw,expected\nfact={'rows':rows,'rgb_sha256':rgb.hexdigest(),'pixels_sha256':trainer.fingerprint(t,pixels),\n    'outputs_sha256':trainer.fingerprint(t,values)}\n"), ('native_train_witness', "expected,_=fullfeature_oracle(context,{'head_object':head,'A':state['A'].detach().cpu(),\n    'means':{k:v.cpu() for k,v in state['means'].items()},'C':state['C'].detach().cpu(),\n    'mu_train':state['mu_train'].detach().cpu(),'arm':state['arm']},cache)\nraw=expected['raw']\n", "raw=trainer.helper_guard(t).raw_features(cache,head,torch.nn.Parameter(state['A'].detach().cpu().clone()),\n    {k:v.cpu() for k,v in state['means'].items()},'concat',t['legacy']['quadratic'])\n"), ('native_train_witness', 'del values,raw,expected,cache,head,difference; gc.collect()\n', 'del values,raw,cache,head,difference; gc.collect()\n'), ('native_export', "require(model_facts['vision_sha256'] == facts['vision_sha256'] and\n    all(model_facts['members'][k] == facts['members'][k] for k in\n        ('config','buffers','head','A','means','C','mu_train','mu_train_provenance','arm')),\n    'independent complete frozen vision/updated A/C/mu/arm differs')\n", "require(model_facts['vision_sha256'] == facts['vision_sha256'] and\n    all(model_facts['members'][k] == facts['members'][k] for k in ('config','buffers','head','A','means')),\n    'independent complete frozen vision/updated A differs')\n"), (None, "SCHEMA = 'siglip2-compact-fullfeature-residual-evaluation-v1'", "SCHEMA = 'siglip2-compact-current-gallery-smooth-ap-evaluation-v1'"), (None, "AUTHORITY_SCHEMA = 'siglip2-compact-fullfeature-residual-evaluation-launch-v1'", "AUTHORITY_SCHEMA = 'siglip2-compact-current-gallery-smooth-ap-evaluation-launch-v1'"), (None, "TRAINING = {'root': '/home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1', 'execution_sha256': '996ae38783d44c0cb01ef3bb8d5ba1817545d5296a0612da07b6c852b69bbf15', 'code': {'train_siglip2_compact_ranking.py': 'ddbbf0bc02eb62c3bb87fc29768ecbd5ec5e9d885fb53215244029413df00bad', 'test_siglip2_compact_ranking.py': '6b2e727d4aacc78e50c616024b37c34584d9b3193b8b8333ca148da57adf2a9b'}}", "TRAINING = {'code': {'test_siglip2_compact_ranking.py': '32f9ade5717d98e9aa5f6644ad7ac0dc5992bb21e0861730e154d0d78c8a7744', 'train_siglip2_compact_ranking.py': 'affb911bd76a760b480e616d13dc78710cc56c74ce92ddd27e1cc8aaa3483480'}, 'execution_sha256': '73cbc39e857ad99e2962a6252c8c158b232636889b5023c28a97b1be7edda68e', 'root': '/home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1'}"))
FULLFEATURE_HELPERS = ("def check_paired_initialization(c,a):\n    for record,arm,names,shapes in ((c,'control',['A'],[[128,160]]),\n            (a,'candidate',['A','C'],[[128,160],[128,1152]])):\n        require(record['arm'] == record['identity']['arm'] == arm and\n            record['identity']['parameter_names'] == names and record['identity']['parameter_shapes'] == shapes and\n            type(record['identity']['parameter_shapes']) is list and\n            all(type(shape) is list and all(type(d) is int for d in shape)\n                for shape in record['identity']['parameter_shapes']),\n            'exact control A / candidate A,C optimizer roles differ')\n    require(all(c[k] == a[k] for k in ('source','initial_A_sha256','initial_C_sha256','mu_train_sha256',\n            'mu_train_provenance_sha256','initial_raw_unit_packed_sha256','numerical_flags')) and\n        all(c['identity'][k] == a['identity'][k] for k in ('static_sha256','initial_A_sha256','initial_C_sha256',\n            'mu_train_sha256','mu_train_provenance_sha256','source','numerical_flags',\n            'initial_cpu_rng_sha256','initial_cuda_rng_sha256')) and\n        [s['batch'] for s in c['steps']] == [s['batch'] for s in a['steps']],\n        'fresh same-seed complete initialization/RNG/schedule differs')", "def check_residual_oracle(proof,arm):\n    keys={'C_exact_zero','residual_nonzero_witness','omitted_C_mutant_rejected','wrong_mu_mutant_rejected'}\n    require(arm in ARMS and isinstance(proof,dict) and proof.keys() == keys and\n        proof['C_exact_zero'] is (arm == 'control') and\n        all(proof[k] is (arm == 'candidate') for k in keys-{'C_exact_zero'}),\n        'complete role-specific nonzero C / omitted C / wrong mu oracle required')", 'def fullfeature_oracle(context,state,features):\n    """Independent accepted concat plus explicit centered residual; never the owned wrapper."""\n    import torch\n    from torch.nn import functional as F\n    trainer,t=context[\'trainer\'],context[\'training_context\'];arm=state[\'arm\']\n    require(arm in ARMS, \'fullfeature oracle arm differs\')\n    base=trainer.helper_guard(t).raw_features(features,state[\'head_object\'],\n        torch.nn.Parameter(state[\'A\'].detach().clone()),state[\'means\'],\'concat\',t[\'legacy\'][\'quadratic\'])\n    C=state[\'C\'].detach().clone();mu=state[\'mu_train\'].detach().clone()\n    zero=torch.count_nonzero(C).item() == 0\n    require(zero is (arm == \'control\'), \'updated candidate must have nonzero C; control must have exactzero C\')\n    raw=base if arm == \'control\' else base+F.linear(features-mu,C)\n    output=packed_outputs(context,raw)\n    proof={\'C_exact_zero\':zero,\'residual_nonzero_witness\':False,\n        \'omitted_C_mutant_rejected\':False,\'wrong_mu_mutant_rejected\':False}\n    if arm == \'candidate\':\n        omitted=packed_outputs(context,base)\n        column=int(C.abs().sum(dim=0).argmax().item())\n        wrong_mu=mu.clone();wrong_mu[column]+=1./float(C[:,column].abs().max().item())\n        wrong=packed_outputs(context,base+F.linear(features-wrong_mu,C))\n        digest=trainer.fingerprint(t,output)\n        proof[\'residual_nonzero_witness\']=proof[\'omitted_C_mutant_rejected\']=(\n            digest != trainer.fingerprint(t,omitted))\n        proof[\'wrong_mu_mutant_rejected\']=digest != trainer.fingerprint(t,wrong)\n    check_residual_oracle(proof,arm)\n    return output,proof')


def inverse_fullfeature_source(source):
    if "SCHEMA = 'siglip2-compact-live-top1-evaluation-v1'" in source:
        source=inverse_live_top1_source(source)
    for new,old in FULLFEATURE_SOURCE_EDITS:
        assert source.count(new)==1, 'fullfeature source delta differs'
        source=source.replace(new,old,1)
    return source


def inverse_fullfeature_authority(tree):
    if any(isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SCHEMA' and
            n.value.value=='siglip2-compact-live-top1-evaluation-v1' for n in tree.body):
        tree=inverse_live_top1_authority(tree)
    dump=lambda n:ast.dump(n,include_attributes=False)
    for source in FULLFEATURE_HELPERS:
        expected=ast.parse(source).body[0]
        found=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==expected.name]
        assert len(found)==1 and dump(found[0])==dump(expected), 'fullfeature oracle/role helper differs'
        tree.body.remove(found[0])
    for scope,new,old in FULLFEATURE_AST_EDITS:
        expected=ast.parse(new).body;replacement=ast.parse(old).body
        root=tree if scope is None else next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==scope)
        matches=[]
        for node in ast.walk(root):
            for _,values in ast.iter_fields(node):
                if isinstance(values,list):
                    for i in range(len(values)-len(expected)+1):
                        candidate=values[i:i+len(expected)]
                        if all(isinstance(n,ast.AST) for n in candidate) and [dump(n) for n in candidate]==[dump(n) for n in expected]:
                            matches.append((values,i))
        assert len(matches)==1, 'fullfeature statement delta differs: '+str(scope)
        values,index=matches[0];values[index:index+len(expected)]=copy.deepcopy(replacement)
    return tree


# Restore current-gallery authority before every unchanged historical inverse.
CURRENT_GALLERY_AST_EDITS = (("SCHEMA = 'siglip2-compact-current-gallery-smooth-ap-evaluation-v1'",
  "SCHEMA = 'siglip2-compact-image-anchor-smooth-ap-evaluation-v1'"),
 ("AUTHORITY_SCHEMA = 'siglip2-compact-current-gallery-smooth-ap-evaluation-launch-v1'",
  "AUTHORITY_SCHEMA = 'siglip2-compact-image-anchor-smooth-ap-evaluation-launch-v1'"),
 ("TRAINING = {'code': {'test_siglip2_compact_ranking.py': "
  "'32f9ade5717d98e9aa5f6644ad7ac0dc5992bb21e0861730e154d0d78c8a7744', 'train_siglip2_compact_ranking.py': "
  "'affb911bd76a760b480e616d13dc78710cc56c74ce92ddd27e1cc8aaa3483480'}, 'execution_sha256': "
  "'73cbc39e857ad99e2962a6252c8c158b232636889b5023c28a97b1be7edda68e', 'root': "
  "'/home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1'}",
  "TRAINING = {'root': '/home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2', 'code': "
  "{'test_siglip2_compact_ranking.py': 'b56265f631a589cc40aa5e0b7cc779bb54d84451eabd87628cc850b704eeba9d', "
  "'train_siglip2_compact_ranking.py': '40c271217c74b89d3b7c20d0dfe05571746303ed6f75f7dcdcffbef3f77a2004'}, "
  "'execution_sha256': '1ed4790f6716ded3972a0e6fd86cffdd45edb7d2060cf949d07da2d4a1adf141'}"),
 ("require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS "
  'and\n'
  '        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == '
  "'siglip2-compact-current-gallery-smooth-ap-v1' and\n"
  "        trainer.AUTHORITY_SCHEMA == 'siglip2-compact-current-gallery-smooth-ap-launch-v1' and\n"
  "        trainer.INFERENCE_SCHEMA == 'siglip2-compact-current-gallery-smooth-ap-inference-v1' and\n"
  "        trainer.BUNDLE_SCHEMA == 'siglip2-compact-current-gallery-smooth-ap-bundle-v1' and "
  'math_helper.ORDER == ORDER and\n'
  "        math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math "
  "contract differs')",
  "require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS "
  'and\n'
  "        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-image-anchor-smooth-ap-v1' "
  'and\n'
  "        trainer.AUTHORITY_SCHEMA == 'siglip2-compact-image-anchor-smooth-ap-launch-v1' and\n"
  "        trainer.INFERENCE_SCHEMA == 'siglip2-compact-image-anchor-smooth-ap-inference-v1' and\n"
  "        trainer.BUNDLE_SCHEMA == 'siglip2-compact-image-anchor-smooth-ap-bundle-v1' and math_helper.ORDER "
  '== ORDER and\n'
  "        math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math "
  "contract differs')"))


def inverse_current_gallery_source(source):
    if "SCHEMA = 'siglip2-compact-live-top1-evaluation-v1'" in source:
        source=inverse_live_top1_source(source)
    if "SCHEMA = 'siglip2-compact-fullfeature-residual-evaluation-v1'" in source:
        source=inverse_fullfeature_source(source)
    for new,old in CURRENT_GALLERY_AST_EDITS:
        assert source.count(new)==1, 'prospective current-gallery authority statement differs: '+new
        source=source.replace(new,old,1)
    return source


def inverse_current_gallery_authority(tree):
    if any(isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SCHEMA' and
            n.value.value=='siglip2-compact-live-top1-evaluation-v1' for n in tree.body):
        tree=inverse_live_top1_authority(tree)
    if any(isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SCHEMA' and
            n.value.value=='siglip2-compact-fullfeature-residual-evaluation-v1' for n in tree.body):
        tree=inverse_fullfeature_authority(tree)
    with patch.dict(inverse_smooth_ap_authority.__globals__,SMOOTH_AP_AST_EDITS=CURRENT_GALLERY_AST_EDITS):
        return inverse_smooth_ap_authority(tree)


# Exact complete statements: only prospective schemas and trainer authority change.
SMOOTH_AP_AST_EDITS = (("SCHEMA = 'siglip2-compact-image-anchor-smooth-ap-evaluation-v1'", "SCHEMA = 'siglip2-compact-ranking-evaluation-v1'"), ("AUTHORITY_SCHEMA = 'siglip2-compact-image-anchor-smooth-ap-evaluation-launch-v1'", "AUTHORITY_SCHEMA = 'siglip2-compact-ranking-evaluation-launch-v1'"), ("TRAINING = {'root': '/home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2', 'code': {'test_siglip2_compact_ranking.py': 'b56265f631a589cc40aa5e0b7cc779bb54d84451eabd87628cc850b704eeba9d', 'train_siglip2_compact_ranking.py': '40c271217c74b89d3b7c20d0dfe05571746303ed6f75f7dcdcffbef3f77a2004'}, 'execution_sha256': '1ed4790f6716ded3972a0e6fd86cffdd45edb7d2060cf949d07da2d4a1adf141'}", "TRAINING = {'root': '/home/riomus/runs/sfora-so400-compact-ranking-train-source-v7', 'execution_sha256': 'ee9f77da0bac2a90b09f87fdcd7e7935de9c8dd38b663d84ca8c57271f43ce26', 'code': {'train_siglip2_compact_ranking.py': '880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c', 'test_siglip2_compact_ranking.py': '544dc5fc63adb53f312e9d0b7eae2419958adef71cc8318092bd733941654274'}}"), ("require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and\n        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-image-anchor-smooth-ap-v1' and\n        trainer.AUTHORITY_SCHEMA == 'siglip2-compact-image-anchor-smooth-ap-launch-v1' and\n        trainer.INFERENCE_SCHEMA == 'siglip2-compact-image-anchor-smooth-ap-inference-v1' and\n        trainer.BUNDLE_SCHEMA == 'siglip2-compact-image-anchor-smooth-ap-bundle-v1' and math_helper.ORDER == ORDER and\n        math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math contract differs')", "require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and\n        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-ranking-v1' and\n        trainer.BUNDLE_SCHEMA == 'siglip2-compact-ranking-bundle-v1' and math_helper.ORDER == ORDER and\n        math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math contract differs')"))

# Undo only the four prospective statements to the exact pre-image-anchor source.
IMAGE_ANCHOR_AST_EDITS = (("SCHEMA = 'siglip2-compact-image-anchor-smooth-ap-evaluation-v1'", "SCHEMA = 'siglip2-compact-smooth-ap-evaluation-v1'"), ("AUTHORITY_SCHEMA = 'siglip2-compact-image-anchor-smooth-ap-evaluation-launch-v1'", "AUTHORITY_SCHEMA = 'siglip2-compact-smooth-ap-evaluation-launch-v1'"), ("TRAINING = {'root': '/home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2', 'code': {'test_siglip2_compact_ranking.py': 'b56265f631a589cc40aa5e0b7cc779bb54d84451eabd87628cc850b704eeba9d', 'train_siglip2_compact_ranking.py': '40c271217c74b89d3b7c20d0dfe05571746303ed6f75f7dcdcffbef3f77a2004'}, 'execution_sha256': '1ed4790f6716ded3972a0e6fd86cffdd45edb7d2060cf949d07da2d4a1adf141'}", "TRAINING = {'root': '/home/riomus/runs/sfora-so400-smooth-ap-train-source-v1', 'code': {'test_siglip2_compact_ranking.py': '5834df438ac32dd98244d7b24bb19e94e173169b19ea6770ba7db2eff0c0f72b', 'train_siglip2_compact_ranking.py': '74a3cbcc3c1a82f21a36793723d57901783ef1f126617ffdea75bc6e402ba679'}, 'execution_sha256': 'bf3efd0040a6cccefde9414839a8cf4943e6a2a52c49f701b865a771def7e272'}"), ("require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and\n        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-image-anchor-smooth-ap-v1' and\n        trainer.AUTHORITY_SCHEMA == 'siglip2-compact-image-anchor-smooth-ap-launch-v1' and\n        trainer.INFERENCE_SCHEMA == 'siglip2-compact-image-anchor-smooth-ap-inference-v1' and\n        trainer.BUNDLE_SCHEMA == 'siglip2-compact-image-anchor-smooth-ap-bundle-v1' and math_helper.ORDER == ORDER and\n        math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math contract differs')", "require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and\n        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-smooth-ap-v1' and\n        trainer.AUTHORITY_SCHEMA == 'siglip2-compact-smooth-ap-launch-v1' and\n        trainer.INFERENCE_SCHEMA == 'siglip2-compact-smooth-ap-inference-v1' and\n        trainer.BUNDLE_SCHEMA == 'siglip2-compact-smooth-ap-bundle-v1' and math_helper.ORDER == ORDER and\n        math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math contract differs')"))

def inverse_smooth_ap_authority(tree):
    dump=lambda n:ast.dump(n,include_attributes=False)
    for new,old in SMOOTH_AP_AST_EDITS:
        expected=ast.parse(new).body[0]; replacement=ast.parse(old).body[0]
        matches=[]
        class Undo(ast.NodeTransformer):
            def visit(self,node):
                candidate=(isinstance(node,ast.Assign) and isinstance(expected,ast.Assign) and
                    isinstance(node.targets[0],ast.Name) and node.targets[0].id==expected.targets[0].id) or (
                    isinstance(node,ast.Expr) and isinstance(expected,ast.Expr) and
                    isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and
                    node.value.func.id=='require' and node.value.args and
                    isinstance(node.value.args[-1],ast.Constant) and
                    node.value.args[-1].value==expected.value.args[-1].value)
                if candidate and dump(node)==dump(expected):
                    matches.append(node)
                    return copy.deepcopy(replacement)
                return super().visit(node)
        tree=Undo().visit(tree)
        assert len(matches)==1, 'prospective authority statement differs: '+new
    return tree


def quality(r1,ap):
    return {'recall_at_1':sum(r1)/len(r1),'map_at_r':sum(ap)/len(ap),'per_query_r1':r1,'per_query_ap':ap}


def descriptor(path,digest='a'*64):
    return {'path':str(path),'sha256':digest}


def unit(number):
    return {'receipt':descriptor('/tmp/u'+str(number)+'/receipt.json'),'log':descriptor('/tmp/u'+str(number)+'.log'),
        'unit':'u'+str(number),'invocation_id':format(number,'032x'),'service_seconds':10,
        'native_peak_rss_kib':100,'both_locks_held':True}


def launch(stage='first',phase='cpu',panel='selection'):
    endpoints=[]
    for i,(seed,arm) in enumerate(e.endpoint_order(stage),1):
        endpoints.append({'seed':seed,'arm':arm,'launch':descriptor('/tmp/l'+str(i)),'terminal':unit(i),
            'checkpoint':descriptor('/tmp/u'+str(i)+'/resume.pt',format(i,'064x')),
            'terminal_state_sha256':'b'*64,'bundle':descriptor('/tmp/u'+str(i)+'/bundle/bundle.json'),
            'inference_state_sha256':'c'*64})
    args=SimpleNamespace(execution_sha256='d'*64,phase=phase,arm=None,seed=None)
    if phase == 'export':args.seed,args.arm=e.endpoint_order(stage)[-1]
    result={'schema':e.AUTHORITY_SCHEMA,'execution_sha256':args.execution_sha256,
        'training':copy.deepcopy(e.TRAINING),
        'nearest_evaluator':e.NEAREST_EVALUATOR,'genuine_evaluator':{'root':'/home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4',
            'execution_sha256':e.GENUINE_EXECUTION_SHA,'code':e.GENUINE_PINS},'reference':e.REFERENCE,
        'phase':phase,'arm':args.arm,'seed':args.seed,'stage':stage,'panel':panel,'endpoints':endpoints,
        'selected_cpu':None if phase=='cpu' else unit(10),'exports':{},
        'first_selection':None if stage=='first' else unit(11),
        'selection_go':None if panel=='selection' else unit(12),'both_locks_held':True,
        'selection_previously_exposed':True,'cost_policy':e.COST_POLICY,
        'resource_policies':{p:e.policy(p) for p in ('cpu','export','score')}}
    if phase=='score':result['exports']={e.label(v):unit(i+20) for i,v in enumerate(endpoints)}
    return result,args


def controls(stage):
    return {(s,a):{'service_seconds':100,'total_training_core_seconds':60} for s,a in e.endpoint_order(stage)}


def panels(stage):
    n=e.PANELS['selection'][1]
    left=[0 if i<12 else 1 for i in range(n)]; right=left.copy();right[:6]=[1]*6
    source=quality(left,[.8]*n);concat=quality(left,[.817]*n)
    q={str(s):{'control':quality(left,[.818]*n),'candidate':quality(right,[.822]*n)} for s in e.seeds(stage)}
    return q,source,concat


def intervals(q):
    _,avg=math_helper.averaged_deltas(q,'full','selection')
    return {m:{'mean_delta':sum(avg[m])/len(avg[m]),'product_lower95':.001,'product_upper95':.1,
               'query_lower95':-.001,'query_upper95':.1} for m in e.METRICS}


# Independent definition-time graphs from the supplied installed sources.
FILELOCK_IMPORTS = {
    '__init__':'from ._api import *\nfrom ._error import *\nfrom ._async_read_write import *\nfrom ._read_write import *\nfrom ._soft import *\nfrom ._soft_rw import *\nfrom ._unix import *\nfrom ._windows import *\nfrom .asyncio import *\nfrom .version import *\n',
    '_api':'from ._error import *\nfrom ._util import *\n',
    '_async_read_write':'from ._read_write import *\n',
    '_error':'', '_read_write':'from ._api import *\nfrom ._error import *\n',
    '_soft':'from ._api import *\nfrom ._util import *\n',
    '_soft_rw/__init__':'from ._async import *\nfrom ._sync import *\n',
    '_soft_rw/_async':'from ._sync import *\n',
    '_soft_rw/_sync':'from .._api import *\nfrom .._error import *\nfrom .._soft import *\nfrom .._util import *\n',
    '_unix':'from ._api import *\nfrom ._util import *\n', '_util':'',
    '_windows':'from ._api import *\nfrom ._util import *\n',
    'asyncio':'from ._api import *\nfrom ._error import *\nfrom ._soft import *\nfrom ._unix import *\nfrom ._windows import *\n',
    'version':''}
HUB_IMPORTS = {
    '_buckets':'from .utils import *\n',
    '_commit_api':'from .file_download import *\nfrom .lfs import *\nfrom tqdm.contrib.concurrent import *\n',
    '_dataset_viewer':'from .utils import *\n', '_eval_results':'',
    '_inference_endpoints':'from .errors import *\nfrom .utils import *\n',
    '_jobs_api':'from ._space_api import *\nfrom .utils._datetime import *\n',
    '_local_folder':'from .utils._fixes import *\n',
    '_snapshot_download':'from .file_download import *\nfrom .hf_api import *\nfrom tqdm.contrib.concurrent import *\n',
    '_space_api':'from .utils import *\n',
    '_upload_large_folder':'from ._commit_api import *\nfrom ._local_folder import *\nfrom .utils.sha import *\n',
    'community':'from .utils import *\n',
    'file_download':'from ._local_folder import *\nfrom .utils.sha import *\n',
    'hf_api':'from ._buckets import *\nfrom ._commit_api import *\nfrom ._dataset_viewer import *\nfrom ._eval_results import *\nfrom ._inference_endpoints import *\nfrom ._jobs_api import *\nfrom ._space_api import *\nfrom ._upload_large_folder import *\nfrom .community import *\nfrom .file_download import *\nfrom .repocard_data import *\nfrom .utils._deprecation import *\nfrom .utils._verification import *\nfrom .utils.endpoint_helpers import *\n',
    'lfs':'from .utils.sha import *\n',
    'repocard':'from .file_download import *\nfrom .hf_api import *\nfrom .repocard_data import *\n',
    'repocard_data':'from .utils import *\n', 'utils/_deprecation':'',
    'utils/_verification':'from ..file_download import *\nfrom .sha import *\n',
    'utils/endpoint_helpers':'from ..repocard_data import *\n',
    'utils/insecure_hashlib':'', 'utils/sha':'from .insecure_hashlib import *\n'}


CHARSET_IMPORTS = {
    '__init__':'from .api import value\nfrom .legacy import value\nfrom .models import value\nfrom .utils import value\nfrom .version import value\n',
    'api':'from .cd import value\nfrom .constant import value\nfrom .md import value\nfrom .models import value\nfrom .utils import value\n',
    'cd':'from .constant import value\nfrom .md import value\nfrom .models import value\nfrom .utils import value\n',
    'constant':'', 'legacy':'from .api import value\n',
    'md':'from .constant import value\nfrom .utils import value\n',
    'models':'from .constant import value\nfrom .utils import value\n',
    'utils':'from .constant import value\n', 'version':'',
}

SCIENTIFIC_DISTRIBUTIONS = {
    'scipy':('scipy','1.18.0'), 'scikit_learn':('sklearn','1.9.0'),
    'joblib':('joblib','1.5.3'), 'threadpoolctl':('threadpoolctl','3.6.0'),
    'pandas':('pandas','3.0.3'), 'python_dateutil':('dateutil','2.9.0.post0'),
    'six':('six','1.17.0'), 'narwhals':('narwhals','2.22.1'),
    'psutil':('psutil','7.2.2'), 'pyarrow':('pyarrow','24.0.0'), 'rich':('rich','15.0.0'),
}
# Representative eager edges from the installed ASTs; bodies are synthetic.
SCIENTIFIC_IMPORTS = {
    'sklearn/__init__.py':'from . import __check_build\nfrom . import base\n',
    'sklearn/__check_build/__init__.py':'from ._check_build import value\n',
    'sklearn/base.py':'from .utils import fixes, validation\nfrom .utils._repr_html import estimator\n',
    'sklearn/utils/__init__.py':'',
    'sklearn/utils/fixes.py':'import pandas\nimport scipy.sparse.linalg\nimport scipy.sparse.csgraph\n',
    'sklearn/utils/validation.py':'import joblib\nimport narwhals.stable.v2\nimport scipy.sparse\n',
    'sklearn/utils/_repr_html/__init__.py':'',
    'sklearn/utils/_repr_html/estimator.py':
        'from pathlib import Path\ncss = "".join((Path(__file__).parent / name).read_text() for name in ("estimator.css", "params.css", "features.css"))\n',
    'sklearn/metrics/__init__.py':'from . import _ranking\n',
    'sklearn/metrics/_ranking.py':'import scipy.integrate\nimport scipy.stats\nimport sklearn.preprocessing\n',
    'sklearn/preprocessing/__init__.py':'import sklearn.callback\n',
    'sklearn/callback/__init__.py':'from . import _progressbar\n',
    'sklearn/callback/_progressbar.py':'import rich.progress\n',
    'scipy/__init__.py':'from ._lib import _ccallback\n',
    'scipy/_lib/__init__.py':'',
    'scipy/_lib/_ccallback.py':'from ._ccallback_c import value\n',
    'scipy/sparse/__init__.py':'from . import _base\n',
    'scipy/sparse/_base.py':'import scipy.sparse.linalg\n',
    'scipy/sparse/_sputils.py':'',
    # The observed native initializer edge is represented by a synthetic import.
    'scipy/sparse/csgraph/__init__.py':'from . import _laplacian, _validation\n',
    'scipy/sparse/csgraph/_laplacian.py':'import scipy.sparse.linalg\nimport scipy.sparse._sputils\n',
    'scipy/sparse/csgraph/_validation.py':
        'from scipy.sparse._sputils import value\nfrom ._tools import value\n',
    'scipy/sparse/linalg/__init__.py':'from . import _interface\n',
    'scipy/sparse/linalg/_interface.py':'import scipy.linalg\n',
    'scipy/linalg/__init__.py':'from . import _misc\n',
    'scipy/linalg/_misc.py':'from . import blas\n',
    'scipy/linalg/blas.py':'from ._fblas import value\n',
    'scipy/integrate/__init__.py':'from . import _quadrature\n',
    'scipy/integrate/_quadrature.py':'import scipy.stats\n',
    'scipy/stats/__init__.py':'from . import _stats_py\n',
    'scipy/stats/_stats_py.py':'import scipy.optimize\nimport scipy.special\n',
    'scipy/optimize/__init__.py':'from . import _optimize\n',
    'scipy/optimize/_optimize.py':'',
    'scipy/special/__init__.py':'from . import _basic\n',
    'scipy/special/_basic.py':'',
    'joblib/__init__.py':'from . import _parallel_backends, parallel\n',
    'joblib/_parallel_backends.py':'from .externals import loky\n',
    'joblib/parallel.py':'from .externals.loky import process_executor\nimport threadpoolctl\n',
    'joblib/externals/__init__.py':'',
    'joblib/externals/loky/__init__.py':'from . import process_executor\n',
    'joblib/externals/loky/process_executor.py':'from .backend import utils\nimport psutil\n',
    'joblib/externals/loky/backend/__init__.py':'',
    'joblib/externals/loky/backend/utils.py':'import psutil\n',
    'threadpoolctl.py':'',
    'pandas/__init__.py':'from . import compat\nfrom ._libs import tslibs\nimport dateutil\n',
    'pandas/compat/__init__.py':'from . import pyarrow\n',
    'pandas/compat/pyarrow.py':'import pyarrow\nimport pyarrow.compute\n',
    'pandas/_libs/__init__.py':'from . import tslibs\n',
    # Simulate the three original native initializer dependencies without loading a native.
    'pandas/_libs/tslibs/__init__.py':'from .parsing import value\nimport dateutil.parser\nimport dateutil.tz\nimport dateutil.relativedelta\nimport dateutil.easter\nzone = dateutil.tz.tz.gettz("Etc/UTC")\n',
    'dateutil/__init__.py':'from . import _version\n',
    'dateutil/_version.py':'',
    'dateutil/easter.py':'import datetime\n',
    'dateutil/parser/__init__.py':'from . import _parser, isoparser\n',
    'dateutil/parser/_parser.py':'import six\nfrom .. import relativedelta, tz\n',
    'dateutil/parser/isoparser.py':'import six\nfrom .. import tz\n',
    'dateutil/relativedelta.py':'import six\nfrom . import _common\n',
    'dateutil/_common.py':'',
    'dateutil/tz/__init__.py':'from . import tz\n',
    'dateutil/tz/tz.py':
        'import six\nfrom . import _common, _factories\n'
        'def gettz(name):\n    from dateutil.zoneinfo import get_zonefile_instance\n    return get_zonefile_instance()\n',
    'dateutil/tz/_common.py':'import six\n',
    'dateutil/tz/_factories.py':'',
    'dateutil/zoneinfo/__init__.py':
        'from io import BytesIO\nfrom tarfile import TarFile\nfrom pkgutil import get_data\nfrom dateutil.tz import value\n'
        'def get_zonefile_instance():\n'
        '    with TarFile.open(fileobj=BytesIO(get_data(__name__, "dateutil-zoneinfo.tar.gz"))) as archive:\n'
        '        return archive.extractfile("Etc/UTC").read()\n',
    'six.py':'',
    'narwhals/__init__.py':'from . import dataframe, series\n',
    'narwhals/dataframe.py':'from . import dtypes\n',
    'narwhals/series.py':'from . import dtypes\n',
    'narwhals/dtypes.py':'',
    'narwhals/stable/__init__.py':'',
    'narwhals/stable/v2/__init__.py':'import narwhals\nfrom . import dependencies\n',
    'narwhals/stable/v2/dependencies.py':'',
    'psutil/__init__.py':'from . import _pslinux, _common, _ntuples\n',
    'psutil/_pslinux.py':'from ._psutil_linux import value\nfrom . import _psposix\n',
    'psutil/_common.py':'',
    'psutil/_ntuples.py':'',
    'psutil/_psposix.py':'',
    'pyarrow/__init__.py':'from .lib import value\nfrom . import ipc, types\n',
    'pyarrow/ipc.py':'from ._ipc import value\nfrom . import util\n',
    'pyarrow/types.py':'from .lib import value\n',
    'pyarrow/util.py':'',
    'pyarrow/compute.py':'from ._compute import value\n',
    'rich/__init__.py':'',
    'rich/progress.py':'from . import console, live, table\n',
    'rich/console.py':'from . import style, text\n',
    'rich/live.py':'from . import console\n',
    'rich/table.py':'from . import text\n',
    'rich/style.py':'',
    'rich/text.py':'',
}
# Definition-time local edges from the installed sources; bodies are synthetic.
ACCELERATE_IMPORTS = {
    'accelerate/__init__.py':'import accelerate.accelerator\nimport accelerate.big_modeling\nimport accelerate.data_loader\nimport accelerate.inference\nimport accelerate.launchers\nimport accelerate.parallelism_config\nimport accelerate.state\nimport accelerate.utils\n',
    'accelerate/accelerator.py':'import accelerate.big_modeling\nimport accelerate.checkpointing\nimport accelerate.data_loader\nimport accelerate.logging\nimport accelerate.optimizer\nimport accelerate.parallelism_config\nimport accelerate.scheduler\nimport accelerate.state\nimport accelerate.tracking\nimport accelerate.utils\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.modeling\nimport accelerate.utils.other\n',
    'accelerate/big_modeling.py':'import accelerate.hooks\nimport accelerate.utils\nimport accelerate.utils.constants\nimport accelerate.utils.other\n',
    'accelerate/checkpointing.py':'import accelerate.logging\nimport accelerate.state\nimport accelerate.utils\n',
    'accelerate/commands/__init__.py':'',
    'accelerate/commands/config/__init__.py':'import accelerate.commands.config.config\nimport accelerate.commands.config.config_args\nimport accelerate.commands.config.default\nimport accelerate.commands.config.update\n',
    'accelerate/commands/config/cluster.py':'import accelerate.commands.config.config_args\nimport accelerate.commands.config.config_utils\nimport accelerate.utils\nimport accelerate.utils.constants\n',
    'accelerate/commands/config/config.py':'import accelerate.commands.config.cluster\nimport accelerate.commands.config.config_args\nimport accelerate.commands.config.config_utils\nimport accelerate.commands.config.sagemaker\nimport accelerate.utils\n',
    'accelerate/commands/config/config_args.py':'import accelerate.utils\nimport accelerate.utils.constants\nimport yaml\n',
    'accelerate/commands/config/config_utils.py':'import accelerate.commands.menu\nimport accelerate.utils.dataclasses\n',
    'accelerate/commands/config/default.py':'import accelerate.commands.config.config_args\nimport accelerate.commands.config.config_utils\nimport accelerate.utils\n',
    'accelerate/commands/config/sagemaker.py':'import accelerate.commands.config.config_args\nimport accelerate.commands.config.config_utils\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.imports\n',
    'accelerate/commands/config/update.py':'import accelerate.commands.config.config_args\nimport accelerate.commands.config.config_utils\n',
    'accelerate/commands/menu/__init__.py':'import accelerate.commands.menu.selection_menu\n',
    'accelerate/commands/menu/cursor.py':'',
    'accelerate/commands/menu/helpers.py':'',
    'accelerate/commands/menu/input.py':'import accelerate.commands.menu.keymap\n',
    'accelerate/commands/menu/keymap.py':'',
    'accelerate/commands/menu/selection_menu.py':'import accelerate.commands.menu\nimport accelerate.commands.menu.cursor\nimport accelerate.commands.menu.helpers\nimport accelerate.commands.menu.input\nimport accelerate.commands.menu.keymap\nimport accelerate.utils.imports\n',
    'accelerate/data_loader.py':'import accelerate.logging\nimport accelerate.state\nimport accelerate.utils\nimport packaging.version\n',
    'accelerate/hooks.py':'import accelerate.state\nimport accelerate.utils\nimport accelerate.utils.imports\nimport accelerate.utils.memory\nimport accelerate.utils.modeling\nimport accelerate.utils.other\n',
    'accelerate/inference.py':'import accelerate.state\nimport accelerate.utils\n',
    'accelerate/launchers.py':'import accelerate.state\nimport accelerate.utils\nimport accelerate.utils.constants\n',
    'accelerate/logging.py':'import accelerate.state\n',
    'accelerate/optimizer.py':'import accelerate.state\nimport accelerate.utils\n',
    'accelerate/parallelism_config.py':'import accelerate.utils.dataclasses\nimport accelerate.utils.versions\n',
    'accelerate/scheduler.py':'import accelerate.state\n',
    'accelerate/state.py':'import accelerate.utils\nimport accelerate.utils.dataclasses\n',
    'accelerate/tracking.py':'import accelerate.logging\nimport accelerate.state\nimport accelerate.utils\nimport packaging.version\nimport yaml\n',
    'accelerate/utils/__init__.py':'import accelerate.parallelism_config\nimport accelerate.utils.ao\nimport accelerate.utils.bnb\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.environment\nimport accelerate.utils.fsdp_utils\nimport accelerate.utils.imports\nimport accelerate.utils.launch\nimport accelerate.utils.megatron_lm\nimport accelerate.utils.memory\nimport accelerate.utils.modeling\nimport accelerate.utils.offload\nimport accelerate.utils.operations\nimport accelerate.utils.other\nimport accelerate.utils.random\nimport accelerate.utils.torch_xla\nimport accelerate.utils.tqdm\nimport accelerate.utils.transformer_engine\nimport accelerate.utils.versions\n',
    'accelerate/utils/ao.py':'import accelerate.utils.imports\n',
    'accelerate/utils/bnb.py':'import accelerate.big_modeling\nimport accelerate.utils.dataclasses\nimport accelerate.utils.imports\nimport accelerate.utils.modeling\n',
    'accelerate/utils/constants.py':'',
    'accelerate/utils/dataclasses.py':'import accelerate.utils.constants\nimport accelerate.utils.environment\nimport accelerate.utils.imports\nimport accelerate.utils.versions\n',
    'accelerate/utils/environment.py':'import packaging.version\n',
    'accelerate/utils/fsdp_utils.py':'import accelerate.logging\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.modeling\nimport accelerate.utils.other\nimport accelerate.utils.versions\n',
    'accelerate/utils/imports.py':'import accelerate.utils.environment\nimport accelerate.utils.versions\nimport packaging.version\n',
    'accelerate/utils/launch.py':'import accelerate.commands.config.config_args\nimport accelerate.utils\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.other\nimport accelerate.utils.versions\n',
    'accelerate/utils/megatron_lm.py':'import accelerate.optimizer\nimport accelerate.scheduler\nimport accelerate.utils.imports\nimport accelerate.utils.operations\n',
    'accelerate/utils/memory.py':'import accelerate.utils.imports\n',
    'accelerate/utils/modeling.py':'import accelerate.state\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.imports\nimport accelerate.utils.memory\nimport accelerate.utils.offload\nimport accelerate.utils.tqdm\nimport accelerate.utils.versions\n',
    'accelerate/utils/offload.py':'',
    'accelerate/utils/operations.py':'import accelerate.state\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.imports\nimport accelerate.utils.versions\n',
    'accelerate/utils/other.py':'import accelerate.commands.config.default\nimport accelerate.logging\nimport accelerate.state\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.imports\nimport accelerate.utils.modeling\nimport accelerate.utils.transformer_engine\nimport accelerate.utils.versions\nimport packaging.version\n',
    'accelerate/utils/random.py':'import accelerate.state\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.imports\n',
    'accelerate/utils/torch_xla.py':'',
    'accelerate/utils/tqdm.py':'import accelerate.state\nimport accelerate.utils.imports\n',
    'accelerate/utils/transformer_engine.py':'import accelerate.utils.imports\nimport accelerate.utils.operations\n',
    'accelerate/utils/versions.py':'import accelerate.utils.constants\nimport packaging.version\n',
}

SCIENTIFIC_NATIVE_IMPORTS = {
    'sklearn.__check_build._check_build':'sklearn/__check_build/_check_build.cpython-313-aarch64-linux-gnu.so',
    'scipy._lib._ccallback_c':'scipy/_lib/_ccallback_c.cpython-313-aarch64-linux-gnu.so',
    'scipy.linalg._fblas':'scipy/linalg/_fblas.cpython-313-aarch64-linux-gnu.so',
    'scipy.sparse.csgraph._tools':'scipy/sparse/csgraph/_tools.cpython-313-aarch64-linux-gnu.so',
    'pandas._libs.tslibs.parsing':'pandas/_libs/tslibs/parsing.cpython-313-aarch64-linux-gnu.so',
    'psutil._psutil_linux':'psutil/_psutil_linux.abi3.so',
    'pyarrow.lib':'pyarrow/lib.cpython-313-aarch64-linux-gnu.so',
    'pyarrow._ipc':'pyarrow/_ipc.cpython-313-aarch64-linux-gnu.so',
    'pyarrow._compute':'pyarrow/_compute.cpython-313-aarch64-linux-gnu.so',
}


ID2LABEL_CONSTRUCTOR_SOURCE='self.id2label = {int(key): value for key, value in self.id2label.items()}'


class EndpointFactsFixture:
    """Exact finite constructor statement and original typed serializer; stdlib only."""
    def __init__(self):
        # Installed configuration_utils.py:270, original CPU guard be45f9ec...
        class Config:
            def __init__(self,value):
                self.values=copy.deepcopy(value);self.id2label=self.values['id2label']
                exec(compile(ID2LABEL_CONSTRUCTOR_SOURCE,'<original id2label conversion>','exec'),{}, {'self':self})
                self.values['id2label']=self.id2label
            def to_dict(self):return copy.deepcopy(self.values)
        self.frozen={'id2label':{'0':'LABEL_0','1':'LABEL_1'},'label2id':{'LABEL_0':0,'LABEL_1':1},
            'hidden_size':1152,'torch_dtype':'float32','do_sample':False,'architectures':['SiglipVisionModel'],
            'nested':{'tuple':(1,2),'list':[3,4]}}
        config=Config(self.frozen)
        self.vision={'vision':b'frozen vision'};self.buffers={'position_ids':(0,1)}
        self.head={'weight':b'frozen head'};self.processor={'backend':'torchvision','size':[256,256]}
        self.state={'model':SimpleNamespace(config=config,state_dict=lambda:self.vision,
            named_buffers=lambda:self.buffers.items()),'head_object':SimpleNamespace(state_dict=lambda:self.head),
            'processor_object':SimpleNamespace(to_json_string=lambda:json.dumps(self.processor)),
            'A':b'updated A','means':{'source':1.25},'C':b'updated C',
            'mu_train':(.25,.5),'mu_train_provenance':{'rows':6355},'arm':'candidate'}
        path=PATH.with_name('train_siglip2_substrate_adaptation.py');raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()=='a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543'
        fn=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='fingerprint')
        def synthetic_import(name,*args):
            assert name=='torch',name
            return SimpleNamespace(Tensor=()) # No real tensors or installed third-party code.
        namespace={'hashlib':hashlib,'__builtins__':dict(vars(sys.modules['builtins']),__import__=synthetic_import)}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original typed serializer only>','exec'),namespace)
        self.fingerprint=namespace['fingerprint']
        self.context={'trainer':SimpleNamespace(fingerprint=lambda _,value:self.fingerprint(value)),
            'training_context':{'initial':{'config':self.frozen}}}
    def facts(self):return e.endpoint_facts(self.context,self.state)


class PortableRuntimeFixture:
    """Installed sources and a previously admitted RECORD; no native imports."""
    def __init__(self,root,*,yaml_native=False,scientific=False,accelerate=False):
        # Legacy falsifiers keep their established installed distributions.
        # The scientific regression below uses the entire production inventory.
        self.runtime_sources={d:paths for d,paths in e.RUNTIME_SOURCES.items()
            if scientific or d not in SCIENTIFIC_DISTRIBUTIONS}
        if not accelerate:self.runtime_sources['accelerate']={'accelerate-1.14.0.dist-info/METADATA'}
        self.site=root/'site-packages';self.site.mkdir()
        packages={}
        for name in ('torch','numpy','PIL','transformers','safetensors','torchvision'):
            path=self.site/name;path.mkdir();packages[name]={'root':str(path)}
        self.package=self.site/'packaging';self.package.mkdir()
        self.sources={}
        names=('__init__.py','_elffile.py','_manylinux.py','_musllinux.py','_parser.py','_structures.py',
            '_tokenizer.py','dependency_groups.py','direct_url.py','errors.py','licenses/__init__.py',
            'licenses/_spdx.py','markers.py','metadata.py','pylock.py','requirements.py','specifiers.py',
            'tags.py','utils.py','version.py')
        for name in names:
            path=self.package/name;path.parent.mkdir(exist_ok=True)
            raw={'__init__.py':b"marker = 'source'\n",'version.py':b'from ._structures import value\n',
                 '_structures.py':b'value = 42\n'}.get(name,b'')
            path.write_bytes(raw);self.sources[path]=raw
        self.rows=[['packaging/'+str(path.relative_to(self.package)),
            'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='),str(len(raw))]
            for path,raw in self.sources.items()]
        self.record=self.site/'packaging-26.2.dist-info'/'RECORD';self.record.parent.mkdir()
        self.record.write_text(''.join(','.join(row)+'\n' for row in self.rows))
        self.bundle=root/'bundle';self.bundle.mkdir()
        self.manifest={'environment':{'packages':packages,'files':{}}}
        digest=hashlib.sha256(self.record.read_bytes()).hexdigest()
        self.context={'trainer':SimpleNamespace(admit_bundle=lambda *_:(self.manifest,{})),
            'guards':{str(self.record):digest},'required_guards':{str(self.record):digest}}
        self.endpoint={'bundle':descriptor(self.bundle/'bundle.json')}
        self.regex=self.site/'regex';self.regex.mkdir()
        self.native=self.regex/'_regex.cpython-313-aarch64-linux-gnu.so'
        self.native.write_bytes(b'original exact native file; never executed')
        self.regex_sources={self.regex/'__init__.py':b"from ._main import value\nmarker = 'source'\n",
            self.regex/'_main.py':b'from ._regex_core import value\n',self.regex/'_regex_core.py':b'value = 73\n'}
        for path,raw in self.regex_sources.items():path.write_bytes(raw)
        self.regex_rows=[['regex/'+path.name,
            'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='),str(len(raw))]
            for path,raw in {**self.regex_sources,self.native:self.native.read_bytes()}.items()]
        self.regex_record=self.site/'regex-2026.6.28.dist-info'/'RECORD';self.regex_record.parent.mkdir()
        self.regex_record.write_text(''.join(','.join(row)+'\n' for row in self.regex_rows))
        for path in (self.regex_record,self.native):
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            self.context['guards'][str(path)]=self.context['required_guards'][str(path)]=digest
        self.context['training_context']={'legacy':{'selected':{'source_cpu':{'origins':{
            'files':{str(self.native):self.context['required_guards'][str(self.native)]},
            'native_files':[str(self.native)]}}}}}
        self.metadata_versions=dict(v.split('-',1) for v in (
            'accelerate-1.14.0 aiohttp-3.14.1 filelock-3.29.4 hf_xet-1.5.1 httpx-0.28.1 '
            'huggingface_hub-1.16.1 jinja2-3.1.6 numpy-2.5.0 packaging-26.2 pillow-12.2.0 '
            'pydantic-2.13.4 pyyaml-6.0.3 regex-2026.6.28 safetensors-0.8.0 tokenizers-0.22.2 '
            'torch-2.12.1 tqdm-4.68.3').split())
        examples={'typing_extensions':{'typing_extensions.py':b'value = 21\n'},
            'tqdm':{'tqdm/__init__.py':b'from . import _monitor, _tqdm_pandas, cli, gui\nfrom .std import value\nfrom .version import __version__\n',
                'tqdm/_monitor.py':b'value = 1\n','tqdm/_tqdm_pandas.py':b'value = 2\n',
                'tqdm/cli.py':b'from .std import value\nfrom .version import __version__\n',
                'tqdm/gui.py':b'from .std import value\n',
                'tqdm/std.py':b'from ._monitor import value as monitor\nfrom .utils import value as utility\nvalue = monitor + utility\n',
                'tqdm/utils.py':b'value = 3\n',
                'tqdm/version.py':b"from importlib.metadata import version\n__version__ = version('tqdm')\nmarker = 'source'\n",
                'tqdm/auto.py':b"from .autonotebook import value as notebook\nfrom .asyncio import value as asynchronous\nvalue = notebook + asynchronous\nmarker = 'source'\n",
                'tqdm/autonotebook.py':b'from .std import value\n',
                'tqdm/asyncio.py':b'from .std import value\n'},
            'tokenizers':{'tokenizers/__init__.py':b'from .implementations import value\n',
                'tokenizers/implementations/__init__.py':b'value = 84\n'},
            'httpx':{'httpx/__init__.py':b'from ._api import value\n','httpx/_api.py':b'value = 101\n'},
            'httpcore':{'httpcore/__init__.py':b'value = 102\n'},'anyio':{'anyio/__init__.py':b'value = 103\n'},
            'h11':{'h11/__init__.py':b'value = 104\n'},'certifi':{'certifi/__init__.py':b'value = 105\n'},
            'idna':{'idna/__init__.py':b'value = 106\n'},'jinja2':{'jinja2/__init__.py':b'value = 107\n'},
            'markupsafe':{'markupsafe/__init__.py':b'value = 108\n'},
            'huggingface_hub':{'huggingface_hub/__init__.py':b'from .dataclasses import value\n',
                'huggingface_hub/dataclasses.py':b'from .errors import value\n','huggingface_hub/errors.py':b'value = 109\n',
                'huggingface_hub/utils/__init__.py':b"from ._http import value\nmarker = 'source'\n",
                'huggingface_hub/utils/_http.py':b"from ..errors import value\nmarker = 'source'\n",
                'huggingface_hub/utils/_runtime.py':b"import importlib.metadata\n_package_versions = {}\nfor name in ('aiohttp', 'hf_xet', 'Jinja2', 'httpx', 'numpy', 'Pillow', 'pydantic', 'safetensors', 'torch', 'fastai'):\n    try:\n        _package_versions[name] = importlib.metadata.version(name)\n    except importlib.metadata.PackageNotFoundError:\n        _package_versions[name] = 'N/A'\n"}}
        # Synthetic definition-time PyYAML graph, independent of the runtime list.
        yaml_imports={'__init__':'from .error import *\nfrom .tokens import *\nfrom .events import *\nfrom .nodes import *\nfrom .loader import *\nfrom .dumper import *\ntry:\n    from .cyaml import *\n    __with_libyaml__ = True\nexcept ImportError:\n    __with_libyaml__ = False\n',
            'loader':'from .reader import *\nfrom .scanner import *\nfrom .parser import *\nfrom .composer import *\nfrom .constructor import *\nfrom .resolver import *\n',
            'dumper':'from .emitter import *\nfrom .serializer import *\nfrom .representer import *\nfrom .resolver import *\n',
            'cyaml':'from yaml._yaml import CParser, CEmitter\nfrom .constructor import *\nfrom .serializer import *\nfrom .representer import *\nfrom .resolver import *\n',
            'composer':'from .error import *\nfrom .events import *\nfrom .nodes import *\n',
            'constructor':'from .error import *\nfrom .nodes import *\n',
            'emitter':'from .error import *\nfrom .events import *\n',
            'parser':'from .error import *\nfrom .tokens import *\nfrom .events import *\nfrom .scanner import *\n',
            'reader':'from .error import *\n','representer':'from .error import *\nfrom .nodes import *\n',
            'resolver':'from .error import *\nfrom .nodes import *\n',
            'scanner':'from .error import *\nfrom .tokens import *\n',
            'serializer':'from .error import *\nfrom .events import *\nfrom .nodes import *\n',
            'error':'','events':'','nodes':'value = 113\n','tokens':''}
        examples['pyyaml']={'yaml/'+n+'.py':(imports+"marker = 'source'\n").encode()
            for n,imports in yaml_imports.items()}
        examples['filelock']={'filelock/'+n+'.py':(imports+"value = 127\nmarker = 'source'\n").encode()
            for n,imports in FILELOCK_IMPORTS.items()}
        examples['huggingface_hub'].update({'huggingface_hub/'+n+'.py':
            (imports+"value = 131\nmarker = 'source'\n").encode() for n,imports in HUB_IMPORTS.items()})
        examples['huggingface_hub']['huggingface_hub/utils/_fixes.py']=b'from filelock import value\n'
        examples['tqdm'].update({'tqdm/contrib/__init__.py':b"from ..auto import value\nmarker = 'source'\n",
            'tqdm/contrib/concurrent.py':b"from ..auto import value\nmarker = 'source'\n",
            'tqdm/contrib/logging.py':b"import logging\nfrom ..std import value\nmarker = 'source'\n"})
        self.metadata_versions['charset_normalizer']='3.4.7'
        examples['charset_normalizer']={'charset_normalizer/'+n+'.py':
            (imports+"value = 151\nmarker = 'source'\n").encode() for n,imports in CHARSET_IMPORTS.items()}
        if scientific:
            for distribution,(package,version) in SCIENTIFIC_DISTRIBUTIONS.items():
                self.metadata_versions[distribution]=version
                examples[distribution]={n:(imports+"value = 137\nmarker = 'source'\n").encode()
                    for n,imports in SCIENTIFIC_IMPORTS.items() if n.split('/')[0].removesuffix('.py')==package}
            examples['scikit_learn'].update({'sklearn/utils/_repr_html/'+n+'.css':b'/* pinned definition-time style */'
                for n in ('estimator','params','features')})
            stream=io.BytesIO();raw=b'pinned UTC archive member'
            with tarfile.open(fileobj=stream,mode='w:gz') as archive:
                entry=tarfile.TarInfo('Etc/UTC');entry.size=len(raw)
                archive.addfile(entry,io.BytesIO(raw))
            examples['python_dateutil']['dateutil/zoneinfo/dateutil-zoneinfo.tar.gz']=stream.getvalue()
        if accelerate:
            examples['accelerate']={name:(body+"value = 157\nmarker = 'source'\n").encode()
                for name,body in ACCELERATE_IMPORTS.items()}
        for distribution in examples:self.metadata_versions.setdefault(distribution,'1.0')
        self.extra_sources={};self.extra_records={};self.natives=[self.native]
        for distribution in examples:
            sources={self.site/n:b'' for n in self.runtime_sources.get(distribution,())}
            sources.update({self.site/n:raw for n,raw in examples[distribution].items()})
            for path,raw in sources.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            self.extra_sources.update(sources)
            if distribution in ('tokenizers','markupsafe','charset_normalizer') or (distribution=='pyyaml' and yaml_native):
                name={'tokenizers':'tokenizers/tokenizers.abi3.so',
                    'markupsafe':'markupsafe/_speedups.cpython-313-aarch64-linux-gnu.so',
                    'charset_normalizer':'charset_normalizer/cd.cpython-313-aarch64-linux-gnu.so',
                    'pyyaml':'yaml/_yaml.cpython-313-aarch64-linux-gnu.so'}[distribution]
                path=self.site/name;path.write_bytes(b'original exact native file; never executed')
                sources[path]=path.read_bytes();self.natives.append(path)
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                self.context['guards'][str(path)]=self.context['required_guards'][str(path)]=digest
                original=self.context['training_context']['legacy']['selected']['source_cpu']['origins']
                original['files'][str(path)]=digest;original['native_files'].append(str(path))
            if scientific:
                package=SCIENTIFIC_DISTRIBUTIONS.get(distribution,('',None))[0]
                for name in SCIENTIFIC_NATIVE_IMPORTS.values():
                    if name.split('/')[0]!=package:continue
                    path=self.site/name;path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(b'original scientific native bytes; never executed')
                    sources[path]=path.read_bytes();self.natives.append(path)
                    digest=hashlib.sha256(sources[path]).hexdigest()
                    self.context['guards'][str(path)]=self.context['required_guards'][str(path)]=digest
                    original=self.context['training_context']['legacy']['selected']['source_cpu']['origins']
                    original['files'][str(path)]=digest;original['native_files'].append(str(path))
            record=self.site/(distribution+'-'+self.metadata_versions.get(distribution,'1.0')+'.dist-info')/'RECORD'
            record.parent.mkdir(exist_ok=True)
            rows=[[str(path.relative_to(self.site)),
                'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='),str(len(raw))]
                for path,raw in sources.items()]
            record.write_text(''.join(','.join(row)+'\n' for row in rows));self.extra_records[distribution]=record
            digest=hashlib.sha256(record.read_bytes()).hexdigest()
            self.context['guards'][str(record)]=self.context['required_guards'][str(record)]=digest
        self.metadata={}
        for distribution,version in self.metadata_versions.items():
            path=self.site/(distribution+'-'+version+'.dist-info')/'METADATA'
            path.parent.mkdir(exist_ok=True)
            raw=('Metadata-Version: 2.1\nName: '+distribution+'\nVersion: '+version+'\n').encode()
            path.write_bytes(raw);self.metadata[distribution]=path
            record={'packaging':self.record,'regex':self.regex_record}.get(distribution,
                self.extra_records.get(distribution,path.with_name('RECORD')))
            name=str(path.relative_to(self.site))
            rows=[r for r in csv.reader(record.read_text().splitlines()) if r[0]!=name] if record.exists() else []
            rows.append([name,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='),str(len(raw))])
            top=path.with_name('top_level.txt')
            top_raw=({'pillow':'PIL'}.get(distribution,distribution)+'\n').encode();top.write_bytes(top_raw)
            rows.append([str(top.relative_to(self.site)),
                'sha256='+base64.urlsafe_b64encode(hashlib.sha256(top_raw).digest()).decode().rstrip('='),str(len(top_raw))])
            record.write_text(''.join(','.join(row)+'\n' for row in rows))
            if distribution=='packaging':self.rows=rows
            elif distribution=='regex':self.regex_rows=rows
            if distribution not in ('packaging','regex'):self.extra_records[distribution]=record
            digest=hashlib.sha256(record.read_bytes()).hexdigest()
            self.context['guards'][str(record)]=self.context['required_guards'][str(record)]=digest

    @contextmanager
    def boundary(self):
        with patch.dict(e.RUNTIME_SOURCES,self.runtime_sources,clear=True),e.bundle_reads_only(self.context,self.endpoint):
            yield

    def add_identity_distribution(self,name,version,declared):
        package=self.site/name.replace('-','_');package.mkdir()
        source=package/'__init__.py';source.write_bytes(b"raise AssertionError('optional module executed')\n")
        metadata=self.site/(name.replace('-','_')+'-'+version+'.dist-info')/'METADATA'
        metadata.parent.mkdir();metadata.write_text('Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\n')
        paths=[source,metadata];top=metadata.with_name('top_level.txt')
        if declared is not None:top.write_text(declared);paths.append(top)
        record=metadata.with_name('RECORD')
        rows=[[str(p.relative_to(self.site)),
            'sha256='+base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('='),
            str(p.stat().st_size)] for p in paths]
        record.write_text(''.join(','.join(row)+'\n' for row in rows))
        digest=hashlib.sha256(record.read_bytes()).hexdigest()
        self.context['guards'][str(record)]=self.context['required_guards'][str(record)]=digest
        return metadata,top,record,source


class SourceAdmissionFixture:
    """Run the real context-composition block and pinned adapter, without native work.

    The old authority's actual dictionary expression supplies the pre-native
    context. Only external receipts/logs/wires are disconnected stand-ins.
    """
    def __init__(self, root):
        def pinned(name, filename, digest):
            if name not in sys.modules:
                e.load_authenticated(name, PATH.with_name(filename), digest, {})
            value=sys.modules[name]
            e.bound_file({},value.__file__,digest)
            return value
        reference=pinned('_compact_context_reference_tests','evaluate_siglip2_prototype_residual.py',
            e.REFERENCE['code']['evaluate_siglip2_prototype_residual.py'])
        self.baseline=pinned('_compact_context_baseline_tests','evaluate_siglip2_quadratic_readout.py',
            reference.EVALUATOR_PINS['evaluate_siglip2_quadratic_readout.py'])
        b=self.baseline
        self.partition={'original_cache':descriptor(root/'cache.npy',reference.FIT_SHA),
            'panels':{'selection':{'original_rows':[8,3,5]},'validation':{'original_rows':[9,4]}}}
        self.path=root/'partition.json';self.path.write_text(json.dumps(self.partition))
        part=descriptor(self.path,hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.spec={'partition':part,'source_selection':{'inventory':descriptor(root/'inventory.json',b.SOURCE_INVENTORY_SHA)}}
        fit={'class_names':['one','two'],'targets':[0,1]}
        selected={'partition':copy.deepcopy(self.partition),'source':{'standin':'authenticated source'},
            'genuine':{'prior':{'fit':fit}},'source_driver':None,'original':None,'extract':None,
            'source_cpu':{'invocation':{'invocation_id':'0'*32}}}
        self.calls=[]
        def wire_file(guards,path,digest):
            fact=next(v for v in b.SOURCE_INVENTORY['files'].values() if v['path']==str(path))
            e.require(digest==fact['sha256'],'wire descriptor differs')
            guards[str(path)]=digest;self.calls.append(('wire',str(path)))
            return SimpleNamespace(stat=lambda:SimpleNamespace(st_size=fact['bytes']))
        admission=SimpleNamespace(bound_file=wire_file)
        # Evaluate the authenticated authority's REAL context expression: no
        # synthetic top-level partition, initial state or ownership checks.
        original=PATH.with_name('train_siglip2_quadratic_readout.py')
        e.bound_file({},original,reference.ORIGINAL_PINS[original.name])
        node=next(n for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='authority')
        expression=next(n.value for n in node.body if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=='context' for t in n.targets))
        self.legacy=eval(compile(ast.Expression(expression),str(original),'eval'),
            {'args':None,'root':root,'code':{},'launch':{'partition':part},'guards':{},
             'genuine':None,'selected':selected,'admission':admission,'record':{}})
        self.training={'legacy':self.legacy,'fit_context':{'launch':{'partition':part},'guards':{}},'guards':{}}
        self.archived={'cgroup_before':{},'cgroup_after':{},'input_guards':{},'files':{},'output':str(root)}
        inventory=b.SOURCE_INVENTORY
        self.record={'schema':'siglip2-genuine-view-evaluation-v1','phase':'score','pass':True,
            'engineering_admission_pass':True,'source':selected['source'],
            'spec':{'panel':'selection','stage':'first','endpoints':[inventory['baseline_endpoint']],
                'partition':inventory['partition'],'evaluation_reference':{'root':b.REFERENCE_ROOT,
                    'execution_sha256':b.REFERENCE_EXECUTION_SHA}},
            'resource_policy':b.policy('score'),'query_images':1734,'gallery_images':1715,'panel_products':498,
            'peak_cuda_allocated_bytes':0,'files':{n:v['sha256'] for n,v in inventory['files'].items()},
            'output':str(Path(next(iter(inventory['files'].values()))['path']).parent),
            'quality':{'179061':{'control':{'recall_at_1':inventory['control_quality']['recall_at_1'],
                'map_at_r':inventory['control_quality']['map_at_r'],'per_query_r1':[1]*1670+[0]*64,
                'per_query_ap':[inventory['control_quality']['map_at_r']]*1734}}},
            'input_guards':{},'cgroup_before':{},'cgroup_after':{}}
        for k in ('strict_independent_head_reload_exact','train_raw_unit_cpu_packed_exact','rng_flags_preserved',
            'exit_rehash_pass','full_panel_raw_unit_packed_replay_exact','per_query_replay_exact'):self.record[k]=True
        for k in ('official_read','public_latency_measured','global_production_goal_met','cuda_initialized'):self.record[k]=False
        def reader(admitted,record,terminal,seconds,guards):
            e.require(admitted is admission,'original admission object differs')
            self.calls.append(('terminal',seconds));return {}
        # Copy the adapter's namespace; ORIGINAL module globals stay untouched.
        adapter_factory=FunctionType(reference.source_selection_adapter.__code__,
            {**vars(reference),'original_terminal_reader':lambda _:reader})
        def source_adapter(baseline,context):
            adapted=adapter_factory(baseline,context)  # Full-byte/live-code/AST authentication stays real.
            def receipt(value,guards):
                self.calls.append(('source_json',value['path']))
                if value==self.spec['source_selection']['inventory']:return inventory
                e.require(value==inventory['receipt'],'source receipt descriptor differs')
                return self.record
            adapted.__globals__['read_json']=receipt
            return adapted
        self.reference=SimpleNamespace(FIT_SHA=reference.FIT_SHA,original_terminal_reader=lambda _:reader,
            source_selection_adapter=source_adapter)
        self.originals=[(m,dict(vars(m))) for m in (reference,b)]

    def compose(self):
        node=next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='authority')
        def assigns(n,name):
            return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
        start=next(i for i,n in enumerate(node.body) if assigns(n,'legacy'))
        stop=next(i for i,n in enumerate(node.body) if assigns(n,'source_record'))+1
        namespace={**vars(e),'t':self.training,'reference':self.reference,'spec':self.spec,'args':None,
            'guards':{},'archived':self.archived,'baseline':self.baseline,
            'helper':SimpleNamespace(zero_events=lambda _:None),'native':SimpleNamespace(CONCAT_TERMINAL=unit(99))}
        exec(compile(ast.Module(body=node.body[start:stop],type_ignores=[]),str(PATH),'exec'),namespace)
        return namespace['s']


class GroupedMdFixture(PortableRuntimeFixture):
    """Recorded metadata, synthetic native bytes/maps; no extension execution."""
    def __init__(self,root):
        super().__init__(root)
        self.md=self.site/'charset_normalizer/md.cpython-313-aarch64-linux-gnu.so'
        self.md.write_bytes(b'UNAUTHORIZED physical wrapper; never read')
        self.cd=self.site/'charset_normalizer/cd.cpython-313-aarch64-linux-gnu.so'
        self.shared=self.site/'81d243bd2c585b0f4821__mypyc.cpython-313-aarch64-linux-gnu.so'
        self.shared.write_bytes(b'original grouped initializer stand-in; never executed')
        original=self.context['training_context']['legacy']['selected']['source_cpu']['origins']
        original['packages']=self.manifest['environment']['packages']
        self.context['training_context']['legacy']['warm_record']={'origins':{'native_files':[]}}
        self.context['training_context']['nearest']=SimpleNamespace(NATIVE_MEMBERS=set())
        self.group_digests={str(p.relative_to(self.site)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (self.cd,self.shared)}
        digest=self.group_digests[self.shared.name]
        original['native_files'].append(str(self.shared));original['files'][str(self.shared)]=digest
        self.context['guards'][str(self.shared)]=self.context['required_guards'][str(self.shared)]=digest
        self.record=self.extra_records['charset_normalizer']
        self.rows=list(csv.reader(self.record.read_text().splitlines()))
        self.rows.extend([[self.shared.name,'sha256='+base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('='),
            str(self.shared.stat().st_size)],['charset_normalizer/'+self.md.name,
            'sha256=EYzysjLHjG1gl9fj8mFGIRrs0HPeSgwZw5AWcSxih7A','201304']])
        self.write_record()
        source=PATH.with_name('qualify_siglip2_substrate_cpu.py')
        self.context['training_context']['legacy']['source_driver']=SimpleNamespace(
            __file__=str(source),__spec__=SimpleNamespace(origin=str(source)))
        self.context['guards'][str(source)]=self.context['required_guards'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        self.maps=[self.cd,self.shared];self.map_reads=0;self.md_reads=0
    def write_record(self):
        self.record.write_text(''.join(','.join(row)+'\n' for row in self.rows))
        digest=hashlib.sha256(self.record.read_bytes()).hexdigest()
        self.context['guards'][str(self.record)]=self.context['required_guards'][str(self.record)]=digest
    def module(self,path):return SimpleNamespace(__file__=str(path),__spec__=SimpleNamespace(origin=str(path)))
    @contextmanager
    def boundary(self,*,compiled=True,modules=None):
        recorded={'charset_normalizer.md':self.module(self.md if compiled else self.md.with_name('md.py'))}
        if modules is not None:recorded=modules
        original_text,original_bytes=Path.read_text,Path.read_bytes
        def text(path,*args,**kwargs):
            if str(path)=='/proc/self/maps':
                self.map_reads+=1;sys.audit('open',str(path),'r',os.O_RDONLY)
                return ''.join('1000-2000 r-xp 00000000 00:00 0 '+str(p)+'\n' for p in self.maps)
            return original_text(path,*args,**kwargs)
        def raw(path,*args,**kwargs):
            if path==self.md:
                self.md_reads+=1;raise AssertionError('unauthorized md wrapper was hashed/read')
            return original_bytes(path,*args,**kwargs)
        with patch.object(e,'GROUPED_MD_NATIVE_SHA256',self.group_digests,create=True),\
                patch.object(Path,'read_text',text),patch.object(Path,'read_bytes',raw),\
                patch.dict(sys.modules,recorded),super().boundary():
            yield


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        if self._testMethodName=='test_prospective_frozen_training_and_retired_authority':
            restored=patch.dict(globals(),e=historical_fullfeature_evaluator())
            restored.start();self.addCleanup(restored.stop)
    def test_exact_current_gallery_inverse_preserves_entire_source(self):
        source=inverse_fullfeature_source(PATH.read_text())
        restored=inverse_current_gallery_source(source)
        self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(),
            '8604043becd1d430bf90d0b5f99e02bf1c35fbd278fb755036130e811b4ddede')
        tree=inverse_current_gallery_authority(ast.parse(source))
        self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(ast.parse(restored),include_attributes=False))
        self.assertEqual(hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
            '97e8e03dbd710e065999bead5ed1f59315443a9800c8aee14e731bb3dad80c99')
        for new,_ in CURRENT_GALLERY_AST_EDITS:
            changed=source.replace(new,new.replace('siglip2','foreign',1),1)
            with self.subTest(statement=new),self.assertRaises(AssertionError):
                inverse_current_gallery_source(changed)
            with self.subTest(statement_ast=new),self.assertRaises(AssertionError):
                inverse_current_gallery_authority(ast.parse(changed))
        for kind in ('function','docstring'):
            changed=ast.parse(source)
            if kind=='function':
                next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name=='seeds').body.append(ast.Pass())
            else:changed.body[0].value.value+=' drift'
            with self.subTest(unrelated=kind):
                self.assertNotEqual(ast.dump(inverse_current_gallery_authority(changed),include_attributes=False),
                    ast.dump(tree,include_attributes=False))

    def test_exact_image_anchor_authority_inverse_preserves_entire_source(self):
        source=inverse_export_envelope_source(PATH.read_text())
        with patch.dict(inverse_smooth_ap_authority.__globals__,SMOOTH_AP_AST_EDITS=IMAGE_ANCHOR_AST_EDITS):
            restored=inverse_smooth_ap_authority(ast.parse(source))
            self.assertEqual(hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest(),
                '394c36d476ae5df03ed0ec05a8fd6ad0d9b5fad06f75692a211db258a34cdd94')
            for kind in ('function','docstring'):
                changed=ast.parse(source)
                if kind=='function':
                    next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name=='seeds').body.append(ast.Pass())
                else:changed.body[0].value.value+=' drift'
                with self.subTest(unrelated=kind):
                    self.assertNotEqual(ast.dump(inverse_smooth_ap_authority(changed),include_attributes=False),
                        ast.dump(restored,include_attributes=False))
            restored_source=source
            for new,old in IMAGE_ANCHOR_AST_EDITS:
                changed=source.replace(new,new.replace('siglip2','foreign',1),1)
                with self.subTest(statement=new),self.assertRaises(AssertionError):
                    inverse_smooth_ap_authority(ast.parse(changed))
                self.assertEqual(source.count(new),1)
                restored_source=restored_source.replace(new,old,1)
            self.assertEqual(hashlib.sha256(restored_source.encode()).hexdigest(),
                '2d9198924fe7004730f25ee8929900e0360d73a32442ad7444426cb0c447d317')

    def test_current_trainer_full_payload_and_inference_schema_denials(self):
        path=(TRAINER_ROOT/'train_siglip2_compact_ranking.py')
        trainer=e.load_authenticated('_current_gallery_payload_trainer',path,
            e.TRAINING['code'][path.name],{})
        launch={name:getattr(trainer,name.upper()) for name in ('nearest','fitter','accepted','readout','recipe')}
        launch.update(execution_sha256=e.TRAINING['execution_sha256'],native_authority=descriptor('/tmp/native'))
        ident={'source':{'owned':'source'},'method':trainer.method(launch),'parameter_names':['A','C'],
            'parameter_shapes':[[128,160],[128,1152]],'arm':'control','seed':179061,'device':'cpu','numerical_flags':{}}
        context={'launch':launch,'source':ident['source'],'flags':ident['numerical_flags']}
        payload=dict.fromkeys(trainer.PAYLOAD_KEYS)
        payload.update(schema=trainer.SCHEMA,identity=ident,source=ident['source'],counter=128,numerical_flags={})
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='check_payload')
        guard=next(n for n in node.body if isinstance(n,ast.Expr) and 'complete payload identity differs' in ast.unparse(n))
        predicate=compile(ast.Module(body=[guard],type_ignores=[]),str(path),'exec')
        def admit(saved):
            exec(predicate,{**vars(trainer),'context':context,'saved':saved,'ident':ident,'step':128})
        admit(payload)
        for schema in ('siglip2-compact-fullfeature-residual-v1','siglip2-compact-current-gallery-smooth-ap-v1',
                'siglip2-compact-image-anchor-smooth-ap-v1','siglip2-compact-smooth-ap-v1',
                'siglip2-compact-ranking-v1','foreign'):
            with self.subTest(payload_schema=schema),self.assertRaises(ValueError):
                admit({**payload,'schema':schema})
        for key in trainer.PAYLOAD_KEYS:
            changed=payload.copy();changed.pop(key)
            with self.subTest(missing_payload=key),self.assertRaises(ValueError):admit(changed)
        with self.assertRaises(ValueError):admit({**payload,'foreign':None})
        for key,bad in (('source',{}),('identity',{}),('counter',True),('numerical_flags',{'changed':True})):
            with self.subTest(payload_field=key),self.assertRaises(ValueError):admit({**payload,key:bad})
        inference=dict.fromkeys(trainer.INFERENCE_KEYS);inference['schema']=trainer.INFERENCE_SCHEMA
        e.check_inference_members(trainer,inference)
        for schema in ('siglip2-compact-fullfeature-residual-inference-v1','siglip2-compact-current-gallery-smooth-ap-inference-v1',
                'siglip2-compact-image-anchor-smooth-ap-inference-v1',
                'siglip2-compact-smooth-ap-inference-v1','siglip2-compact-ranking-inference-v1',trainer.SCHEMA,'foreign'):
            with self.subTest(inference_schema=schema),self.assertRaises(ValueError):
                e.check_inference_members(trainer,{**inference,'schema':schema})
        for key in trainer.INFERENCE_KEYS:
            changed=inference.copy();changed.pop(key)
            with self.subTest(missing_inference=key),self.assertRaises(ValueError):
                e.check_inference_members(trainer,changed)
        with self.assertRaises(ValueError):e.check_inference_members(trainer,{**inference,'optimizer':None})
        self.assertFalse(any(n.split('.')[0] in e.NATIVE for n in sys.modules))

    def test_prospective_frozen_training_and_retired_authority(self):
        self.assertEqual(e.SCHEMA,'siglip2-compact-fullfeature-residual-evaluation-v1')
        self.assertEqual(e.AUTHORITY_SCHEMA,'siglip2-compact-fullfeature-residual-evaluation-launch-v1')
        self.assertEqual(e.TRAINING,{'root': '/home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1', 'execution_sha256': '996ae38783d44c0cb01ef3bb8d5ba1817545d5296a0612da07b6c852b69bbf15', 'code': {'train_siglip2_compact_ranking.py': 'ddbbf0bc02eb62c3bb87fc29768ecbd5ec5e9d885fb53215244029413df00bad', 'test_siglip2_compact_ranking.py': '6b2e727d4aacc78e50c616024b37c34584d9b3193b8b8333ca148da57adf2a9b'}})
        value,args=launch()
        e.check_launch(value,args)
        changed=copy.deepcopy(value)
        changed['training']={'root':'/tmp/another-actual-trainer','execution_sha256':'1'*64,
            'code':dict.fromkeys(e.TRAIN_FILES,'2'*64)}
        with self.assertRaisesRegex(ValueError,'parent-frozen'):e.check_launch(changed,args)
        for bad in ('siglip2-compact-current-gallery-smooth-ap-evaluation-launch-v1',
                'siglip2-compact-image-anchor-smooth-ap-evaluation-launch-v1',
                'siglip2-compact-image-anchor-smooth-ap-launch-v1',
                'siglip2-compact-ranking-evaluation-launch-v1', 'siglip2-compact-ranking-launch-v1',
                'siglip2-compact-smooth-ap-evaluation-launch-v1','siglip2-compact-smooth-ap-launch-v1'):
            with self.subTest(schema=bad),self.assertRaises(ValueError):
                e.check_launch({**value,'schema':bad},args)
        with self.assertRaisesRegex(ValueError,'parent-frozen'):
            e.check_launch({**value,'training':{'root': '/home/riomus/runs/sfora-so400-compact-ranking-train-source-v7', 'execution_sha256': 'ee9f77da0bac2a90b09f87fdcd7e7935de9c8dd38b663d84ca8c57271f43ce26', 'code': {'train_siglip2_compact_ranking.py': '880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c', 'test_siglip2_compact_ranking.py': '544dc5fc63adb53f312e9d0b7eae2419958adef71cc8318092bd733941654274'}}},args)
        with self.assertRaisesRegex(ValueError,'parent-frozen'):
            e.check_launch({**value,'training':{'root': '/home/riomus/runs/sfora-so400-smooth-ap-train-source-v1',
                'code': {'test_siglip2_compact_ranking.py': '5834df438ac32dd98244d7b24bb19e94e173169b19ea6770ba7db2eff0c0f72b',
                    'train_siglip2_compact_ranking.py': '74a3cbcc3c1a82f21a36793723d57901783ef1f126617ffdea75bc6e402ba679'},
                'execution_sha256': 'bf3efd0040a6cccefde9414839a8cf4943e6a2a52c49f701b865a771def7e272'}},args)
        with self.assertRaisesRegex(ValueError,'parent-frozen'):
            e.check_launch({**value,'training':{'root': '/home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2', 'code': {'test_siglip2_compact_ranking.py': 'b56265f631a589cc40aa5e0b7cc779bb54d84451eabd87628cc850b704eeba9d', 'train_siglip2_compact_ranking.py': '40c271217c74b89d3b7c20d0dfe05571746303ed6f75f7dcdcffbef3f77a2004'}, 'execution_sha256': '1ed4790f6716ded3972a0e6fd86cffdd45edb7d2060cf949d07da2d4a1adf141'}},args)
        for field,bad in (('root','/tmp/another-actual-trainer'),('execution_sha256','1'*64),
                ('code',{**value['training']['code'],'train_siglip2_compact_ranking.py':'2'*64})):
            changed=copy.deepcopy(value);changed['training'][field]=bad
            with self.subTest(frozen_field=field),self.assertRaisesRegex(ValueError,'parent-frozen'):
                e.check_launch(changed,args)
        for field,bad in (('root','relative'),('execution_sha256','not-a-hash'),('code',{})):
            changed=copy.deepcopy(value);changed['training'][field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):e.check_launch(changed,args)

    def test_exact_smooth_ap_authority_inverse_preserves_entire_source(self):
        source=inverse_export_envelope_source(PATH.read_text())
        restored=inverse_smooth_ap_authority(ast.parse(source))
        self.assertEqual(hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest(),
            '1250b96b6ff6e3e615620c28323942285f6485bebbde8f3a5d3a0198d8f08ddd')
        changed=ast.parse(source)
        next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name=='seeds').body.append(ast.Pass())
        self.assertNotEqual(ast.dump(inverse_smooth_ap_authority(changed),include_attributes=False),
            ast.dump(restored,include_attributes=False))
        changed=ast.parse(source);changed.body[0].value.value+=' drift'
        self.assertNotEqual(ast.dump(inverse_smooth_ap_authority(changed),include_attributes=False),
            ast.dump(restored,include_attributes=False))
        for new,_ in SMOOTH_AP_AST_EDITS:
            changed=source.replace(new,new.replace('siglip2','foreign',1) if 'siglip2' in new else new.replace('differs','changed',1),1)
            with self.assertRaises(AssertionError):inverse_smooth_ap_authority(ast.parse(changed))

    def test_actual_prospective_trainer_source_admission_and_schema_denials(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'trainer';source.mkdir()
            for name in e.TRAIN_FILES:
                (source/name).write_bytes((TRAINER_ROOT/name).read_bytes())
            code={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in e.TRAIN_FILES}
            self.assertEqual(code,e.TRAINING['code'])
            manifest=source/'execution.json';manifest.write_text(json.dumps(code,sort_keys=True))
            digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
            value,args=launch();value['training']={'root':str(source),'execution_sha256':digest,'code':code}
            with self.assertRaisesRegex(ValueError,'parent-frozen'):e.check_launch(value,args)
            with patch.object(e,'TRAINING',copy.deepcopy(value['training'])):e.check_launch(value,args)
            guards={}
            self.assertEqual(e.closure(source,digest,e.TRAIN_FILES,guards),code)
            trainer=e.load_authenticated('_prospective_actual_trainer',source/'train_siglip2_compact_ranking.py',
                code['train_siglip2_compact_ranking.py'],guards)
            node=next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='authority')
            guard=next(n for n in node.body if isinstance(n,ast.Expr) and
                'owned trainer/paired-seed math contract differs' in ast.unparse(n))
            namespace={**vars(e),'trainer':trainer,'launch':value,
                'native':SimpleNamespace(REFERENCE=e.REFERENCE),'math_helper':math_helper}
            def admit():exec(compile(ast.Module(body=[guard],type_ignores=[]),str(PATH),'exec'),namespace)
            admit()
            self.assertEqual(trainer.PAYLOAD_KEYS,set(('A', 'C', 'mu_train', 'mu_train_provenance', 'buffers', 'classifier', 'config', 'counter', 'cpu_rng', 'cuda_rng', 'head', 'identity', 'means', 'numerical_flags', 'optimizer', 'original_rows', 'partition', 'processor', 'provenance', 'scaler', 'schedules', 'schema', 'source', 'target', 'teachers', 'views')))
            self.assertEqual(trainer.INFERENCE_KEYS,set(('A', 'C', 'mu_train', 'mu_train_provenance', 'arm', 'buffers', 'config', 'fixed_sha256', 'head', 'means', 'numerical_flags', 'processor', 'schema', 'source', 'vision_sha256')))
            targs=SimpleNamespace(execution_sha256=digest,phase='cpu',arm='control',seed=179061)
            tlaunch={'schema':trainer.AUTHORITY_SCHEMA,'execution_sha256':digest,'phase':'cpu',
                'arm':'control','seed':179061,'nearest':trainer.NEAREST,'fitter':trainer.FITTER,
                'accepted':trainer.ACCEPTED,'readout':trainer.READOUT,'recipe':copy.deepcopy(trainer.RECIPE),
                'resource_policy':trainer.policy('cpu'),'both_locks_held':True,
                'native_authority':descriptor('/tmp/original-native-authority'),
                'selected_cpu':None,'selected_mechanics':None}
            trainer.check_launch(tlaunch,targs)
            self.assertEqual(inverse_live_top1_recipe(trainer.RECIPE),{'seeds': [179061, 179069], 'rows': 6355, 'classes': 1008, 'singletons': 12, 'updates': 128, 'batch': 64, 'microbatch': 16, 'views': ['canonical', 'augmented'], 'trainable_names': {'control': ['A'], 'candidate': ['A', 'C']}, 'trainable_shapes': {'control': [[128, 160]], 'candidate': [[128, 160], [128, 1152]]}, 'trainable_scalars': {'control': 20480, 'candidate': 167936}, 'adamw': {'lr': 0.0001, 'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.05, 'amsgrad': False, 'maximize': False, 'foreach': False, 'capturable': False, 'differentiable': False, 'fused': False}, 'clip': 1.0, 'initial_scaler': 128.0, 'regression': 'both arms P[label]; both original views coordinate sum / (128*e0)', 'ranking': 'both arms backward coefficient1; all positives; SmoothAP sum / (2*K)', 'temperature': 0.01, 'teacher': 'accepted canonical T; member-inclusive P; normalize(T); both-view e0', 'mining': 'all6355 canonical rows; both arms current connected canonical readout at same pre-update A/C; exclude same original image; all same-identity positives; canonical ordinal traversal', 'schedule': 'original first128 B64 per seed; warm-authenticated; masks unused', 'readout': 'original CPU-renormalized genuine features; FP32 all; autocast disabled', 'frozen': 'complete encoder448/config/buffers/processor/head/classifier/means/muTRAIN; control C exactzero', 'residual': 'original concat helper once + candidate linear(actual normalized x - canonical TRAIN6355 mu,C); C128x1152 zero; control serialized frozen C bypass; mean FP32 canonical actual CPU domain only', 'gallery': 'both arms reconstruct connected gallery every micro16 at same pre-update A/C; both-role backward then release each graph; one optimizer step after eight micros; no serialized gallery', 'core': 'cache/target preparation + both-arm all-positive scoring + both-view forward/backward + every both-arm gallery forward/backward + candidate fullfeature centering/transfers + optimizer'})
            with self.assertRaises(ValueError):
                trainer.check_launch({**tlaunch,'recipe':{'adamw': {'amsgrad': False, 'betas': [0.9, 0.999], 'capturable': False, 'differentiable': False, 'eps': 1e-08, 'foreach': False, 'fused': False, 'lr': 0.0001, 'maximize': False, 'weight_decay': 0.05}, 'batch': 64, 'classes': 1008, 'clip': 1.0, 'core': 'cache/target preparation + both-arm all-positive scoring + both-view forward/backward + optimizer', 'frozen': 'complete encoder448/config/buffers/processor/head/classifier/means', 'initial_scaler': 128.0, 'microbatch': 16, 'mining': 'all6355 canonical frozen bank; exclude same original image; all same-identity positives; canonical ordinal traversal', 'ranking': 'both arms backward coefficient1; all positives; SmoothAP sum / (2*K)', 'readout': 'original CPU-renormalized genuine features; FP32 all; autocast disabled', 'regression': 'control P[label]; candidate canonical T[image]; both original views coordinate sum / (128*e0)', 'rows': 6355, 'schedule': 'original first128 B64 per seed; warm-authenticated; masks unused', 'seeds': [179061, 179069], 'singletons': 12, 'teacher': 'accepted canonical T; member-inclusive P; normalize(T); both-view e0', 'temperature': 0.01, 'trainable_names': ['A'], 'trainable_scalars': 20480, 'trainable_shapes': [[128, 160]], 'updates': 128, 'views': ['canonical', 'augmented']}},targs)
            for field,legacy in (('schema','siglip2-compact-ranking-launch-v1'),('recipe',{'seeds': [179061, 179069], 'rows': 6355, 'classes': 1008, 'singletons': 12, 'updates': 128, 'batch': 64, 'microbatch': 16, 'views': ['canonical', 'augmented'], 'trainable_names': ['A'], 'trainable_shapes': [[128, 160]], 'trainable_scalars': 20480, 'adamw': {'lr': 0.0001, 'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.05, 'amsgrad': False, 'maximize': False, 'foreach': False, 'capturable': False, 'differentiable': False, 'fused': False}, 'clip': 1.0, 'initial_scaler': 128.0, 'regression': 'both same-row views coordinate sum / (128*e0)', 'ranking': 'candidate coefficient1; both mine; hinge sum / (2*K*.05)', 'margin': 0.05, 'teacher': 'accepted canonical T; member-inclusive P; normalize(T); both-view e0', 'mining': 'all6355; other original image positive; wrong identity negative; ascending original-row ties', 'schedule': 'original first128 B64 per seed; warm-authenticated; masks unused', 'readout': 'original CPU-renormalized genuine features; FP32 all; autocast disabled', 'frozen': 'complete encoder448/config/buffers/processor/head/classifier/means', 'core': 'cache/target preparation + both-arm mining + both-view forward/backward + optimizer'})):
                with self.subTest(field=field),self.assertRaises(ValueError):
                    trainer.check_launch({**tlaunch,field:legacy},targs)
            retired_recipe=copy.deepcopy(trainer.RECIPE)
            retired_recipe.update(regression='both same-row views coordinate sum / (128*e0)',
                ranking='candidate coefficient1; both score all positives; SmoothAP sum / (2*K)')
            for recipe in (retired_recipe,
                    {**trainer.RECIPE,'regression':retired_recipe['regression']},
                    {**trainer.RECIPE,'ranking':retired_recipe['ranking']}):
                with self.subTest(retired_recipe=recipe),self.assertRaises(ValueError):
                    trainer.check_launch({**tlaunch,'recipe':recipe},targs)
            for name,legacy in (('SCHEMA','siglip2-compact-image-anchor-smooth-ap-v1'),
                    ('AUTHORITY_SCHEMA','siglip2-compact-image-anchor-smooth-ap-launch-v1'),
                    ('INFERENCE_SCHEMA','siglip2-compact-image-anchor-smooth-ap-inference-v1'),
                    ('BUNDLE_SCHEMA','siglip2-compact-image-anchor-smooth-ap-bundle-v1')):
                with self.subTest(closed_schema=name),patch.object(trainer,name,legacy),self.assertRaises(ValueError):admit()
            for name,legacy in (('SCHEMA','siglip2-compact-smooth-ap-v1'),
                    ('AUTHORITY_SCHEMA','siglip2-compact-smooth-ap-launch-v1'),
                    ('INFERENCE_SCHEMA','siglip2-compact-smooth-ap-inference-v1'),
                    ('BUNDLE_SCHEMA','siglip2-compact-smooth-ap-bundle-v1')):
                with self.subTest(retired_schema=name),patch.object(trainer,name,legacy),self.assertRaises(ValueError):admit()
            for name,legacy in (('SCHEMA','siglip2-compact-ranking-v1'),
                    ('AUTHORITY_SCHEMA','siglip2-compact-ranking-launch-v1'),
                    ('INFERENCE_SCHEMA','siglip2-compact-ranking-inference-v1'),
                    ('BUNDLE_SCHEMA','siglip2-compact-ranking-bundle-v1')):
                with self.subTest(schema=name),patch.object(trainer,name,legacy),self.assertRaises(ValueError):admit()
            original=source/'test_siglip2_compact_ranking.py';raw=original.read_bytes();prior=original.stat()
            original.write_bytes(raw+b' ');os.utime(original,ns=(prior.st_atime_ns,prior.st_mtime_ns))
            with self.assertRaises(ValueError):e.closure(source,digest,e.TRAIN_FILES,{})

    def test_exact_source_stage_seed_and_roles(self):
        for stage in ('first','full'):
            for phase in ('cpu','export','score'):
                value,args=launch(stage,phase);e.check_launch(value,args)
                for k,v in (('selection_previously_exposed',False),('both_locks_held',False),
                    ('stage','other'),('reference',{}),('endpoints',value['endpoints'][::-1])):
                    with self.subTest(stage=stage,phase=phase,k=k),self.assertRaises((ValueError,KeyError)):
                        e.check_launch({**value,k:v},args)
        value,args=launch()
        for k,v in (('first_selection',unit(11)),('selected_cpu',unit(10)),('panel','validation'),('seed',179069),
                    ('arm','control'),('exports',{'control-179061':unit(20)})):
            with self.subTest(k=k),self.assertRaises(ValueError):e.check_launch({**value,k:v},args)
        value,args=launch('full','score','validation');e.check_launch(value,args)
        for k in ('selection_go','first_selection','selected_cpu'):
            with self.assertRaises(ValueError):e.check_launch({**value,k:None},args)
        value,args=launch('full');value['training']['code']['foreign.py']='0'*64
        with self.assertRaises(ValueError):e.check_launch(value,args)
        value,args=launch();value['endpoints'][0]['seed']=True
        with self.assertRaises(ValueError):e.check_launch(value,args)
        value,args=launch('full');value['first_selection']=value['endpoints'][0]['terminal']
        with self.assertRaises(ValueError):e.check_launch(value,args)

    def test_pre_native_partition_and_pinned_source_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            f=SourceAdmissionFixture(Path(directory))
            self.assertTrue({'partition','partition_check','initial'}.isdisjoint(f.legacy))
            s=f.compose()
            self.assertEqual(s['partition']['panels']['selection']['original_rows'],[8,3,5])
            self.assertIs(s['fit'],f.legacy['prior']['fit']);self.assertIs(s['selected'],f.legacy)
            self.assertIs(s['source_record'],f.record);self.assertIs(s['origin_records'][-1],f.record)
            self.assertEqual(s['terminals'][-1],f.baseline.SOURCE_SCORE_TERMINAL)
            self.assertEqual(s['source_guards'][str(f.spec['source_selection']['inventory']['path'])],f.baseline.SOURCE_INVENTORY_SHA)
            self.assertEqual(s['guards'][str(f.path)],f.spec['partition']['sha256'])
            self.assertEqual([n for kind,n in f.calls if kind=='terminal'],[500,300])
            self.assertEqual(sum(kind=='wire' for kind,_ in f.calls),3)
            for module,values in f.originals:
                self.assertEqual(vars(module).keys(),values.keys())
                self.assertTrue(all(vars(module)[k] is v for k,v in values.items()))
            self.assertFalse(any(n.split('.')[0] in e.NATIVE for n in sys.modules))

    def test_partition_order_descriptor_cache_and_hash_rejected_before_source(self):
        for mutation in ('order','descriptor','cache','hash'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                f=SourceAdmissionFixture(Path(directory))
                if mutation=='order':f.legacy['selected']['partition']['panels']['selection']['original_rows'].reverse()
                elif mutation=='descriptor':f.training['fit_context']['launch']['partition']=descriptor(f.path,'0'*64)
                elif mutation=='cache':
                    f.partition['original_cache']['sha256']='0'*64
                    f.path.write_text(json.dumps(f.partition))
                    f.spec['partition']['sha256']=hashlib.sha256(f.path.read_bytes()).hexdigest()
                    f.legacy['selected']['partition']=copy.deepcopy(f.partition)
                else:f.path.write_text(json.dumps(f.partition)+' ')
                with self.assertRaisesRegex(ValueError,'ordered panel partition|SHA256'):f.compose()
                self.assertFalse(any(kind=='source_json' for kind,_ in f.calls))

    def test_first_screen_never_bootstraps(self):
        q,source,concat=panels('first');costs=e.paired_cost(controls('first'),'first')
        result=e.decide(math_helper,q,source,concat,'first','selection',{},costs)
        self.assertEqual(result['decision'],'CONTINUE');self.assertFalse(result['selection_go_admits_validation_only'])
        for replacement in (q['179061']['control'],quality(q['179061']['candidate']['per_query_r1'],[.7]*1734)):
            bad=copy.deepcopy(q);bad['179061']['candidate']=replacement
            self.assertEqual(e.decide(math_helper,bad,source,concat,'first','selection',{},costs)['decision'],'KILL')
            with self.assertRaises(ValueError):e.decide(math_helper,bad,source,concat,'first','selection',{'ci':{}},costs)
        with self.assertRaises(ValueError):e.decide(math_helper,q,source,concat,'first','selection',{'ci':{}},costs)
        records=controls('first');records[179061,'candidate']['service_seconds']=151
        self.assertEqual(e.decide(math_helper,q,source,concat,'first','selection',{},e.paired_cost(records,'first'))['decision'],'KILL')

    def test_full_equal_seed_floor_sign_and_intervals(self):
        q,source,concat=panels('full');costs=e.paired_cost(controls('full'),'full');bounds=intervals(q)
        result=e.decide(math_helper,q,source,concat,'full','selection',bounds,costs)
        self.assertEqual(result['decision'],'GO');self.assertTrue(result['selection_go_admits_validation_only'])
        self.assertFalse(result['global_production_goal_met']);self.assertFalse(result['product_go'])
        bad=copy.deepcopy(bounds);bad[e.METRICS[0]]['product_lower95']=0
        self.assertEqual(e.decide(math_helper,q,source,concat,'full','selection',bad,costs)['decision'],'KILL')
        bad=copy.deepcopy(bounds);bad[e.METRICS[0]]['mean_delta']=.7
        with self.assertRaises(ValueError):e.decide(math_helper,q,source,concat,'full','selection',bad,costs)
        bad=copy.deepcopy(q);bad['179069']['candidate']=copy.deepcopy(bad['179069']['control'])
        self.assertEqual(e.decide(math_helper,bad,source,concat,'full','selection',{},costs)['decision'],'KILL')
        with self.assertRaises(ValueError):e.decide(math_helper,bad,source,concat,'full','selection',bounds,costs)
        high=quality([1]*1734,[.9]*1734)
        self.assertEqual(e.decide(math_helper,q,high,concat,'full','selection',{},costs)['decision'],'KILL')
        self.assertEqual(e.decide(math_helper,q,source,high,'full','selection',{},costs)['decision'],'KILL')
        bad=copy.deepcopy(q);bad['179069']['candidate']['per_query_ap'][0]=float('nan')
        with self.assertRaises(ValueError):e.decide(math_helper,bad,source,concat,'full','selection',bounds,costs)

    def test_each_seed_core_and_whole_cost_boundary(self):
        records=controls('full');records[179069,'candidate'].update(service_seconds=150,total_training_core_seconds=90)
        self.assertTrue(all(v['pass'] for v in e.paired_cost(records,'full').values()))
        for key,value in (('total_training_core_seconds',90.01),('service_seconds',150.01)):
            bad=copy.deepcopy(records);bad[179069,'candidate'][key]=value
            self.assertFalse(e.paired_cost(bad,'full')['179069']['pass'])
        for bad_value in (0,True,float('nan'),float('inf'),101):
            bad=copy.deepcopy(records);bad[179061,'control']['total_training_core_seconds']=bad_value
            with self.assertRaises(ValueError):e.paired_cost(bad,'full')
        with self.assertRaises(ValueError):e.paired_cost(controls('first'),'full')

    def test_no_quality_before_readiness(self):
        calls=[]
        for key in e.READINESS:
            ready=dict.fromkeys(e.READINESS,True);ready[key]=False
            with self.assertRaises(ValueError):e.quality_after_readiness(ready,lambda:calls.append('quality'))
        self.assertEqual(calls,[])
        e.quality_after_readiness(dict.fromkeys(e.READINESS,True),lambda:calls.append('quality'))
        self.assertEqual(calls,['quality'])

    def test_diagnostic_and_native_roles(self):
        value=e.diagnostic_result([.1,-.1],[.09,-.2])
        self.assertFalse(value['mechanism_demonstrated']);self.assertFalse(value['utility_veto']);self.assertTrue(value['diagnostic_only'])
        self.assertEqual(e.batch_sizes(1734),[32]*54+[6]);self.assertEqual(e.batch_sizes(1715),[32]*53+[19])
        self.assertEqual(e.batch_sizes(1749)[-1],21);self.assertEqual(e.batch_sizes(1730)[-1],2)
        for invalid in (0,True,-1):
            with self.assertRaises(ValueError):e.batch_sizes(invalid)
        with self.assertRaises(ValueError):e.diagnostic_result([float('nan')],[0])

    def test_strict_files_closure_and_restored_mtime(self):
        for raw in ('{"a":1,"a":2}','{"a":NaN}'):
            with self.assertRaises(ValueError):e.strict_json(raw)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'worker.py';path.write_bytes(b'first')
            digest=hashlib.sha256(path.read_bytes()).hexdigest();code={'worker.py':digest}
            manifest=root/'execution.json';manifest.write_text(json.dumps(code));manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(e.closure(root,manifest_sha,code,{}),code)
            prior=path.stat();path.write_bytes(b'other');os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
            with self.assertRaises(ValueError):e.bound_file({},path,digest)
            path.write_bytes(b'first');link=root/'link';link.symlink_to(path)
            with self.assertRaises(ValueError):e.bound_file({},link,digest)
            fifo=root/'fifo';os.mkfifo(fifo)
            with self.assertRaises(ValueError):e.bound_file({},fifo,digest)
            with self.assertRaises(ValueError):e.closure(root,manifest_sha,{'worker.py','extra'}, {})

    def test_endpoint_facts_original_constructor_config_conversion_and_fresh_parity(self):
        f=EndpointFactsFixture();live=f.state['model'].config.values
        self.assertEqual(live['id2label'],{0:'LABEL_0',1:'LABEL_1'})
        self.assertTrue(all(type(k)is int for k in live['id2label']))
        original_live=copy.deepcopy(live);original_frozen=copy.deepcopy(f.frozen)
        expected={'vision_sha256':f.fingerprint(f.vision),'members':{
            'config':f.fingerprint(f.frozen),'buffers':f.fingerprint(f.buffers),
            'processor_config':f.fingerprint(f.processor),'head':f.fingerprint(f.head),
            'A':f.fingerprint(f.state['A']),'means':f.fingerprint(f.state['means']),
            **{k:f.fingerprint(f.state[k]) for k in ('C','mu_train','mu_train_provenance','arm')}}}
        for _ in range(3):self.assertEqual(f.facts(),expected)
        self.assertEqual(EndpointFactsFixture().facts(),expected)
        self.assertEqual(live,original_live);self.assertEqual(f.frozen,original_frozen)
        # Even an alias-returning stand-in must remain unchanged by comparison.
        f.state['model'].config.to_dict=lambda:live
        self.assertEqual(f.facts(),expected);self.assertEqual(f.facts(),expected)
        self.assertEqual(live,original_live);self.assertEqual(f.frozen,original_frozen)
        f.state['A']=b'different A'
        changed=f.facts();self.assertNotEqual(changed['members']['A'],expected['members']['A'])
        self.assertEqual(changed['members']['config'],expected['members']['config'])

    def test_endpoint_facts_rejects_noncanonical_frozen_and_live_label_maps(self):
        class ForeignInt(int):pass
        class ForeignString(str):pass
        class ForeignDict(dict):pass
        f=EndpointFactsFixture();live=f.state['model'].config.values
        original_live=copy.deepcopy(live);original_frozen=copy.deepcopy(f.frozen)
        cases=(
            ('live_bool',{False:'LABEL_0',1:'LABEL_1'},False),
            ('live_float',{0.0:'LABEL_0',1:'LABEL_1'},False),
            ('live_foreign_int',{ForeignInt(0):'LABEL_0',1:'LABEL_1'},False),
            ('live_strings',{'0':'LABEL_0','1':'LABEL_1'},False),
            ('live_noncanonical',{'00':'LABEL_0',1:'LABEL_1'},False),
            ('live_collision',{0:'LABEL_0','0':'LABEL_0',1:'LABEL_1'},False),
            ('live_missing',{0:'LABEL_0'},False),('live_empty',{},False),
            ('live_changed_key',{0:'LABEL_0',2:'LABEL_1'},False),
            ('live_foreign_key',{object():'LABEL_0',1:'LABEL_1'},False),
            ('live_changed_label',{0:'CHANGED',1:'LABEL_1'},False),
            ('live_non_string_label',{0:None,1:'LABEL_1'},False),
            ('live_foreign_label',{0:ForeignString('LABEL_0'),1:'LABEL_1'},False),
            ('live_map_type',[(0,'LABEL_0'),(1,'LABEL_1')],False),
            ('live_foreign_map',ForeignDict({0:'LABEL_0',1:'LABEL_1'}),False),
            ('frozen_bool',{False:'LABEL_0','1':'LABEL_1'},True),
            ('frozen_int',{0:'LABEL_0','1':'LABEL_1'},True),
            ('frozen_foreign_string',{ForeignString('0'):'LABEL_0','1':'LABEL_1'},True),
            ('frozen_noncanonical',{'00':'LABEL_0','1':'LABEL_1'},True),
            ('frozen_negative',{'-0':'LABEL_0','1':'LABEL_1'},True),
            ('frozen_signed',{'+0':'LABEL_0','1':'LABEL_1'},True),
            ('frozen_collision',{'0':'LABEL_0','00':'LABEL_0','1':'LABEL_1'},True),
            ('frozen_missing',{'0':'LABEL_0'},True),('frozen_empty',{},True),
            ('frozen_changed_key',{'0':'LABEL_0','2':'LABEL_1'},True),
            ('frozen_changed_label',{'0':'CHANGED','1':'LABEL_1'},True),
            ('frozen_foreign_label',{'0':ForeignString('LABEL_0'),'1':'LABEL_1'},True),
            ('frozen_map_type',[('0','LABEL_0'),('1','LABEL_1')],True),
            ('frozen_foreign_map',ForeignDict({'0':'LABEL_0','1':'LABEL_1'}),True))
        for case,labels,frozen in cases:
            with self.subTest(case=case):
                live.clear();live.update(copy.deepcopy(original_live))
                f.frozen.clear();f.frozen.update(copy.deepcopy(original_frozen))
                (f.frozen if frozen else live)['id2label']=labels
                with self.assertRaisesRegex(ValueError,'id2label'):f.facts()
        for frozen in (False,True):
            live.clear();live.update(copy.deepcopy(original_live));f.frozen.clear();f.frozen.update(copy.deepcopy(original_frozen))
            del (f.frozen if frozen else live)['id2label']
            with self.subTest(missing_field_frozen=frozen),self.assertRaisesRegex(ValueError,'id2label'):f.facts()
        live.clear();live.update(copy.deepcopy(original_live));f.frozen.clear();f.frozen.update(copy.deepcopy(original_frozen))
        f.state['model'].config.to_dict=lambda:ForeignDict(live)
        with self.assertRaisesRegex(ValueError,'id2label'):f.facts()

    def test_endpoint_facts_retains_all_other_complete_typed_config_changes(self):
        f=EndpointFactsFixture();live=f.state['model'].config.values
        saved=copy.deepcopy(live);expected=f.fingerprint(f.frozen)
        changes=(('hidden_size',1153),('hidden_size',1152.0),('torch_dtype',None),('do_sample',0),
            ('architectures',('SiglipVisionModel',)),('nested',{'tuple':[1,2],'list':[3,4]}),
            ('label2id',{'LABEL_0':False,'LABEL_1':1}),('foreign_field','extra'),('nested',None))
        for key,value in changes:
            with self.subTest(key=key,value=value):
                live.clear();live.update(copy.deepcopy(saved));live[key]=value
                self.assertNotEqual(f.facts()['members']['config'],expected)
        live.clear();live.update(copy.deepcopy(saved));del live['nested']
        self.assertNotEqual(f.facts()['members']['config'],expected)
        live.clear();live.update(copy.deepcopy(saved));self.assertEqual(f.facts()['members']['config'],expected)

    def test_bundle_boundary_grouped_md_metadata_fresh_cached_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            f=GroupedMdFixture(Path(directory))
            for _ in range(2):
                with f.boundary():
                    for path in (f.md,f.site/'charset_normalizer/foreign.so',f.site/'charset_normalizer/resume.pt'):
                        with self.assertRaisesRegex(ValueError,'external dependency'):os.open(path,os.O_RDONLY)
                    with self.assertRaisesRegex(ValueError,'attempted write'):os.open(f.md,os.O_WRONLY)
            self.assertEqual(f.map_reads,2);self.assertEqual(f.md_reads,0)
            self.assertNotIn(str(f.md),f.context['guards']);self.assertNotIn(str(f.md),f.context['required_guards'])
            runtime=next(iter(f.context['portable_audits'].values()))[1]
            self.assertNotIn(f.md,runtime)
            with f.boundary(compiled=False):self.assertTrue(f.md.with_name('md.py').read_bytes())
            self.assertEqual(f.map_reads,2)

    def test_bundle_boundary_grouped_md_map_denials_before_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            f=GroupedMdFixture(Path(directory));foreign=f.site/'foreign.so';foreign.write_bytes(b'foreign')
            for cached in (False,True):
                if cached:
                    with f.boundary():pass
                for maps in ([],[f.cd],[f.shared],[f.cd,f.shared,f.md],[f.cd,f.shared,foreign],
                        [f.cd,f.shared,f.site/'missing.so']):
                    with self.subTest(cached=cached,maps=maps):
                        f.maps=maps
                        with patch.object(e,'bound_file',wraps=e.bound_file) as hashes:
                            with self.assertRaises(ValueError):
                                with f.boundary():pass
                            self.assertEqual(hashes.call_count,0,'map denial must precede file hashing')
                f.maps=[f.cd,f.shared]
            self.assertEqual(f.md_reads,0)

    def test_bundle_boundary_grouped_md_identity_record_and_origin_denials(self):
        with tempfile.TemporaryDirectory() as directory:
            f=GroupedMdFixture(Path(directory));original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
            guards=copy.deepcopy(f.context['guards']);required=copy.deepcopy(f.context['required_guards'])
            proof=copy.deepcopy(original);rows=copy.deepcopy(f.rows);digests=dict(f.group_digests)
            for cached in (False,True):
                if cached:
                    with f.boundary():pass
                for case in ('group_sha','group_guard','group_native','group_file','record_guard','md_row_hash',
                        'md_row_size','md_row_missing','md_row_duplicate','md_row_path','md_byte_guard','source_guard',
                        'mixed_file','mixed_spec','foreign_file','foreign_site','relative_origin','missing_spec',
                        'foreign_module','source_spec_identity','group_mutation','source_mutation'):
                    f.context['guards']=copy.deepcopy(guards);f.context['required_guards']=copy.deepcopy(required)
                    original.clear();original.update(copy.deepcopy(proof));f.rows=copy.deepcopy(rows)
                    f.group_digests.clear();f.group_digests.update(digests);f.write_record()
                    modules={'charset_normalizer.md':f.module(f.md)};restore=None
                    if case=='group_sha':f.group_digests[f.shared.name]='a'*64
                    elif case=='group_guard':f.context['required_guards'][str(f.shared)]='a'*64
                    elif case=='group_native':original['native_files'].remove(str(f.shared))
                    elif case=='group_file':original['files'][str(f.cd)]='a'*64
                    elif case=='record_guard':f.context['required_guards'][str(f.record)]='a'*64
                    elif case.startswith('md_row_'):
                        index=next(i for i,r in enumerate(f.rows) if r[0]=='charset_normalizer/'+f.md.name)
                        if case=='md_row_hash':f.rows[index][1]='sha256='+'A'*43
                        elif case=='md_row_size':f.rows[index][2]='1'
                        elif case=='md_row_missing':f.rows.pop(index)
                        elif case=='md_row_duplicate':f.rows.append(f.rows[index])
                        else:f.rows[index][0]='charset_normalizer/foreign.so'
                        f.write_record()
                    elif case=='md_byte_guard':f.context['required_guards'][str(f.md)]='a'*64
                    elif case=='source_guard':
                        source=f.context['training_context']['legacy']['source_driver'].__file__
                        f.context['required_guards'][source]='a'*64
                    elif case=='mixed_file':modules['charset_normalizer.md'].__file__=str(f.md.with_name('md.py'))
                    elif case=='mixed_spec':modules['charset_normalizer.md'].__spec__.origin=str(f.md.with_name('md.py'))
                    elif case=='foreign_file':modules['charset_normalizer.md']=f.module(f.md.with_name('other.so'))
                    elif case=='foreign_site':modules['charset_normalizer.md']=f.module(Path(directory)/f.md.name)
                    elif case=='relative_origin':modules['charset_normalizer.md']=f.module(Path('charset_normalizer')/f.md.name)
                    elif case=='missing_spec':modules['charset_normalizer.md'].__spec__=None
                    elif case=='foreign_module':modules={'charset_normalizer.api':f.module(f.md)}
                    elif case=='source_spec_identity':
                        source=f.context['training_context']['legacy']['source_driver']
                        restore=(source.__spec__,'origin',source.__spec__.origin);source.__spec__.origin='foreign.py'
                    elif case in ('group_mutation','source_mutation'):
                        path=f.shared if case=='group_mutation' else f.md.with_name('md.py')
                        restore=(path,path.read_bytes());path.write_bytes(b'changed')
                    with self.subTest(cached=cached,case=case),self.assertRaises((ValueError,OSError)):
                        with f.boundary(modules=modules):pass
                    if restore:
                        if len(restore)==3:setattr(restore[0],restore[1],restore[2])
                        else:restore[0].write_bytes(restore[1])
                f.context['guards']=copy.deepcopy(guards);f.context['required_guards']=copy.deepcopy(required)
                original.clear();original.update(copy.deepcopy(proof));f.rows=copy.deepcopy(rows);f.write_record()
                f.group_digests.clear();f.group_digests.update(digests)
            self.assertEqual(f.md_reads,0)

    def test_bundle_boundary_denies_train_warm_optimizer_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture=PortableRuntimeFixture(root)
            owned=fixture.bundle/'vision.pt';owned.write_bytes(b'owned')
            outside=root/'resume.pt';outside.write_bytes(b'forbidden')
            with fixture.boundary():
                self.assertEqual(owned.read_bytes(),b'owned')
                for path in (outside,root/'warm.pt',root/'optimizer.pt',root/'teachers.npy'):
                    with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                with self.assertRaisesRegex(ValueError,'attempted write'):owned.write_bytes(b'changed')
            self.assertEqual(outside.read_bytes(),b'forbidden');self.assertEqual(owned.read_bytes(),b'owned')

    def test_bundle_boundary_self_maps_reads_and_proc_denials_fresh_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);f=PortableRuntimeFixture(root)
            alias=Path('/proc/self/maps');current=Path('/proc')/str(os.getpid())/'maps'
            self.assertEqual(alias.resolve(),current)
            other=Path('/proc')/str(1 if os.getpid()!=1 else 2)/'maps'
            forbidden=[other,Path('/proc/maps'),*(Path('/proc/self')/name for name in
                ('status','smaps','smaps_rollup','mem','fd','fd/999999'))]
            foreign=f.regex/'foreign.so';foreign.write_bytes(b'foreign native; never executed')
            history=[root/name for name in ('resume.pt','warm.pt','optimizer.pt','teachers.npy','checkpoint.pt')]
            for path in history:path.write_bytes(b'historical dependency')
            self.assertNotIn('portable_audits',f.context)
            for cached in (False,True):
                with self.subTest(cached=cached),f.boundary():
                    for path in (alias,current):
                        self.assertIn(b'\n',path.read_bytes())
                        fd=os.open(path,os.O_RDONLY)
                        try:self.assertTrue(os.read(fd,64))
                        finally:os.close(fd)
                        with self.assertRaisesRegex(ValueError,'attempted write'):path.write_bytes(b'forbidden')
                        for mode in ('w','a','x','r+'):
                            with self.assertRaisesRegex(ValueError,'attempted write'):
                                sys.audit('open',str(path),mode,os.O_RDONLY)
                        for flags in (os.O_WRONLY,os.O_RDWR,os.O_CREAT,os.O_TRUNC,os.O_APPEND):
                            with self.assertRaisesRegex(ValueError,'attempted write'):
                                sys.audit('open',str(path),None,flags)
                    for path in (*forbidden,foreign,*history):
                        with self.subTest(path=str(path)),self.assertRaisesRegex(ValueError,'external dependency'):
                            path.read_bytes()
                    for path in (Path('/proc'),alias.parent,current.parent,alias,current,alias.parent/'fd'):
                        with self.assertRaisesRegex(ValueError,'external dependency'):os.listdir(path)
                        with self.assertRaisesRegex(ValueError,'external dependency'):os.scandir(path)
                if not cached:audits=f.context['portable_audits'].copy()
                else:self.assertEqual(f.context['portable_audits'],audits)
            self.assertEqual(foreign.read_bytes(),b'foreign native; never executed')
            for path in history:self.assertEqual(path.read_bytes(),b'historical dependency')

    def test_bundle_boundary_imports_record_pinned_source_without_cached_code(self):
        saved={n:m for n,m in sys.modules.items() if n=='packaging' or n.startswith('packaging.')}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory));source=f.package/'__init__.py';prior=source.stat()
                source.write_bytes(b"marker = 'cached'\n");py_compile.compile(str(source),doraise=True)
                source.write_bytes(f.sources[source]);os.utime(source,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                extra=f.package/'unqualified.py';extra.write_bytes(b'value = 99\n')
                history=[f.package/name for name in ('resume.pt','warm.pt','optimizer.pt','teachers.npy','checkpoint.pt')]
                for path in history:path.write_bytes(b'forbidden')
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name=='packaging' or name.startswith('packaging.'):sys.modules.pop(name)
                        with f.boundary():
                            if direct_loader:packaging=module('packaging',source)
                            else:packaging=importlib.import_module('packaging')
                            from packaging import version
                            self.assertEqual(packaging.marker,'source','unpinned cached code executed')
                            self.assertEqual(version.value,42)
                            for path in (extra,f.record,f.site/'other.dist-info'/'METADATA',*history):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                            with self.assertRaisesRegex(ValueError,'attempted write'):source.write_bytes(b'changed')
                            with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(source))).read_bytes()
                # A stdlib directory containing site-packages grants no extra files.
                del f.context['portable_audits']
                with patch.object(e.sysconfig,'get_path',return_value=str(Path(directory))),f.boundary():
                    with self.assertRaisesRegex(ValueError,'external dependency'):extra.read_bytes()
                self.assertEqual({p:f.context['guards'][str(p)] for p in f.sources},
                    {p:hashlib.sha256(raw).hexdigest() for p,raw in f.sources.items()})
                prior=source.stat();source.write_bytes(b"marker = 'mutate'\n")
                os.utime(source,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                with self.assertRaisesRegex(ValueError,'SHA256'):
                    with f.boundary():pass
        finally:
            for name in tuple(sys.modules):
                if name=='packaging' or name.startswith('packaging.'):sys.modules.pop(name)
            sys.modules.update(saved)

    def test_bundle_boundary_rejects_missing_mutated_and_foreign_runtime(self):
        for case in ('missing_pin','mutated_record','missing_row','duplicate_row','missing_source','mutated_source',
                     'wrong_size','wrong_hash','foreign_record','source_symlink','foreign_module','wrong_module_role'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root);source=f.package/'version.py'
                if case=='missing_pin':f.context['required_guards'].clear()
                elif case=='mutated_record':f.record.write_bytes(f.record.read_bytes()+b'\n')
                elif case in ('missing_row','duplicate_row','wrong_size','wrong_hash'):
                    rows=copy.deepcopy(f.rows)
                    if case=='missing_row':rows.pop()
                    elif case=='duplicate_row':rows.append(rows[-1])
                    elif case=='wrong_hash':rows[-1][1]='sha256='+'A'*43
                    else:rows[-1][-1]='999'
                    with f.record.open('w',newline='') as stream:csv.writer(stream).writerows(rows)
                    digest=hashlib.sha256(f.record.read_bytes()).hexdigest()
                    f.context['guards'][str(f.record)]=f.context['required_guards'][str(f.record)]=digest
                elif case=='missing_source':source.unlink()
                elif case=='mutated_source':source.write_bytes(b'value = 99\n')
                elif case=='foreign_record':
                    foreign=root/'foreign'/'packaging-26.2.dist-info'/'RECORD';foreign.parent.mkdir(parents=True)
                    foreign.write_bytes(f.record.read_bytes());digest=hashlib.sha256(foreign.read_bytes()).hexdigest()
                    f.context['required_guards']={str(foreign):digest};f.context['guards']={str(foreign):digest}
                elif case=='source_symlink':
                    foreign=root/'foreign.py';foreign.write_bytes(f.sources[source]);source.unlink();source.symlink_to(foreign)
                fake=SimpleNamespace(__file__=str(root/'foreign.py'),__spec__=SimpleNamespace(origin=str(root/'foreign.py')))
                if case=='wrong_module_role':
                    fake=SimpleNamespace(__file__=str(f.package/'__init__.py'),
                        __spec__=SimpleNamespace(origin=str(f.package/'__init__.py')))
                with patch.dict(sys.modules,{'packaging.version':fake} if case in ('foreign_module','wrong_module_role') else {}):
                    with self.assertRaises((ValueError,OSError)):
                        with f.boundary():pass

    def test_bundle_boundary_sequential_lazy_sources_and_exact_original_native(self):
        prefixes=('packaging','regex')
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0] in prefixes}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory))
                sources={f.package/'__init__.py':f.sources[f.package/'__init__.py'],**f.regex_sources}
                for path,raw in sources.items():
                    prior=path.stat();path.write_bytes(raw.replace(b'source',b'cached').replace(b'73',b'99'))
                    py_compile.compile(str(path),doraise=True)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                foreign=f.regex/'foreign.so';foreign.write_bytes(b'foreign')
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name.split('.')[0] in prefixes:sys.modules.pop(name)
                        with f.boundary():
                            packaging=module('packaging',f.package/'__init__.py') if direct_loader else importlib.import_module('packaging')
                            from packaging import version
                            regex=module('regex',f.regex/'__init__.py') if direct_loader else importlib.import_module('regex')
                            self.assertEqual((packaging.marker,version.value,regex.marker,regex.value),('source',42,'source',73))
                            self.assertEqual(f.native.read_bytes(),b'original exact native file; never executed')
                            for path in (foreign,f.regex_record,f.regex/'unqualified.py',f.regex/'resume.pt',f.site/'foreign.py'):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                            for path in sources:
                                with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                            with self.assertRaisesRegex(ValueError,'attempted write'):f.native.write_bytes(b'changed')
                native=SimpleNamespace(__file__=str(f.native),__spec__=SimpleNamespace(origin=str(f.native)))
                with patch.dict(sys.modules,{'regex._regex':native}),f.boundary():pass
                for path in (*f.regex_sources,f.native):
                    raw=path.read_bytes();prior=path.stat();path.write_bytes(b'x'*len(raw))
                    os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    with self.assertRaisesRegex(ValueError,'SHA256'):
                        with f.boundary():pass
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0] in prefixes:sys.modules.pop(name)
            sys.modules.update(saved)

    def test_bundle_boundary_processor_dependency_sources_have_exact_origins(self):
        expected={'typing_extensions':21,'tokenizers':84,'httpx':101,'httpcore':102,'anyio':103,
            'h11':104,'certifi':105,'idna':106,'jinja2':107,'markupsafe':108,'huggingface_hub':109}
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0] in expected}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory))
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]),f.boundary():
                    actual={name:importlib.import_module(name).value for name in expected}
                    self.assertEqual(actual,expected)
                    for name in expected:
                        path=f.site/(name+'.py') if name=='typing_extensions' else f.site/name/'__init__.py'
                        self.assertEqual(sys.modules[name].__file__,str(path))
                        with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                    for name in ('tokenizers','httpx','huggingface_hub'):
                        with self.assertRaisesRegex(ValueError,'external dependency'):(f.site/name/'resume.pt').read_bytes()
                foreign=SimpleNamespace(__file__=str(f.site/'foreign.py'),__spec__=SimpleNamespace(origin=str(f.site/'foreign.py')))
                for name in expected:
                    with self.subTest(name=name),patch.dict(sys.modules,{name:foreign}),self.assertRaisesRegex(ValueError,'origin differs'):
                        with f.boundary():pass
                for name in expected:
                    path=f.site/(name+'.py') if name=='typing_extensions' else f.site/name/'__init__.py'
                    raw=path.read_bytes();prior=path.stat();path.write_bytes(b'x'*len(raw))
                    os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    with self.subTest(name=name),self.assertRaisesRegex(ValueError,'SHA256'):
                        with f.boundary():pass
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0] in expected:sys.modules.pop(name)
            sys.modules.update(saved)

    def test_hub_runtime_source_contract_is_finite(self):
        names=('__init__.py constants.py dataclasses.py errors.py serialization/__init__.py serialization/_base.py '
            'serialization/_torch.py utils/__init__.py utils/_auth.py utils/_cache_assets.py utils/_cache_manager.py '
            'utils/_chunk_utils.py utils/_datetime.py utils/_detect_agent.py utils/_experimental.py utils/_fixes.py '
            'utils/_git_credential.py utils/_headers.py utils/_hf_uris.py utils/_http.py utils/_lfs.py '
            'utils/_pagination.py utils/_parsing.py utils/_paths.py utils/_runtime.py utils/_safetensors.py '
            'utils/_subprocess.py utils/_telemetry.py utils/_terminal.py utils/_typing.py utils/_validators.py '
            'utils/_xet.py utils/logging.py utils/tqdm.py _buckets.py _commit_api.py _dataset_viewer.py '
            '_eval_results.py _inference_endpoints.py _jobs_api.py _local_folder.py _snapshot_download.py '
            '_space_api.py _upload_large_folder.py community.py file_download.py hf_api.py lfs.py '
            'repocard.py repocard_data.py utils/_deprecation.py utils/_verification.py '
            'utils/endpoint_helpers.py utils/insecure_hashlib.py utils/sha.py').split()
        self.assertEqual(e.RUNTIME_SOURCES['huggingface_hub'],{'huggingface_hub/'+n for n in names}|
            {'huggingface_hub-1.16.1.dist-info/METADATA'})

    def test_hub_runtime_sources_require_original_record_hash_and_size(self):
        for case in ('missing_guard','foreign_guard','mutated_record','missing_row','wrong_hash','wrong_size'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory));record=f.extra_records['huggingface_hub']
                if case=='missing_guard':f.context['required_guards'].pop(str(record))
                elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                elif case=='mutated_record':record.write_bytes(record.read_bytes()+b'\n')
                else:
                    rows=list(csv.reader(record.read_text().splitlines()))
                    row=next(r for r in rows if r[0]=='huggingface_hub/utils/__init__.py')
                    if case=='missing_row':rows.remove(row)
                    elif case=='wrong_hash':row[1]='sha256='+'A'*43
                    else:row[2]='999'
                    record.write_text(''.join(','.join(r)+'\n' for r in rows))
                    h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                with self.assertRaises(ValueError):
                    with f.boundary():pass

    def test_bundle_boundary_hub_utils_cached_source_fallback_and_denials(self):
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0]=='huggingface_hub'}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root)
                sources={p:raw for p,raw in f.extra_sources.items() if p.relative_to(f.site).as_posix() in
                    ('huggingface_hub/utils/__init__.py','huggingface_hub/utils/_http.py')}
                for path,raw in sources.items():
                    prior=path.stat();path.write_bytes(raw.replace(b'source',b'cached'))
                    py_compile.compile(str(path),doraise=True)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                history=[f.site/'huggingface_hub'/name for name in
                    ('utils/resume.pt','utils/optimizer.pt','utils/teachers.npy','utils/unqualified.py','_login.py')]
                foreign=root/'foreign.py';foreign.write_bytes(b'forbidden')
                for path in history:path.write_bytes(b'forbidden')
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name.split('.')[0]=='huggingface_hub':sys.modules.pop(name)
                        with f.boundary():
                            path=f.site/'huggingface_hub/utils/__init__.py'
                            utils=module('huggingface_hub.utils',path) if direct_loader else importlib.import_module('huggingface_hub.utils')
                            sibling=sys.modules['huggingface_hub.utils._http']
                            self.assertEqual((utils.value,utils.marker,sibling.marker),(109,'source','source'))
                            for name,value in (('huggingface_hub.utils',utils),('huggingface_hub.utils._http',sibling)):
                                origin=f.site/(name.replace('.','/')+('/__init__.py' if value is utils else '.py'))
                                self.assertIsInstance(value.__loader__,importlib.machinery.SourceFileLoader)
                                self.assertEqual(value.__file__,str(origin));self.assertEqual(value.__spec__.origin,str(origin))
                                with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(origin))).read_bytes()
                                with self.assertRaisesRegex(ValueError,'attempted write'):origin.write_bytes(b'changed')
                            for path in (*history,foreign,f.extra_records['huggingface_hub']):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                fake=SimpleNamespace(__file__=str(foreign),__spec__=SimpleNamespace(origin=str(foreign)))
                for name in ('huggingface_hub.utils','huggingface_hub.utils._http'):
                    with patch.dict(sys.modules,{name:fake}),self.assertRaisesRegex(ValueError,'origin differs'):
                        with f.boundary():pass
                for path in (f.site/'huggingface_hub/constants.py',*sources,f.site/'huggingface_hub/serialization/_torch.py'):
                    raw=path.read_bytes();prior=path.stat();path.write_bytes(raw+b'# mutation\n')
                    os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    with self.assertRaisesRegex(ValueError,'SHA256'):
                        with f.boundary():pass
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0]=='huggingface_hub':sys.modules.pop(name)
            sys.modules.update(saved)

    def test_bundle_boundary_filelock_hub_transitive_source_fallback_and_denials(self):
        prefixes=('filelock','huggingface_hub','tqdm')
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0] in prefixes}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root)
                names=({'filelock/'+n+'.py' for n in FILELOCK_IMPORTS}|
                    {'huggingface_hub/'+n+'.py' for n in HUB_IMPORTS}|
                    {'tqdm/contrib/__init__.py','tqdm/contrib/concurrent.py','tqdm/contrib/logging.py'})
                sources={f.site/n:f.extra_sources[f.site/n] for n in names}
                for path,raw in sources.items():
                    prior=path.stat();path.write_bytes(raw.replace(b'source',b'cached'))
                    py_compile.compile(str(path),doraise=True)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                forbidden=[f.site/n for n in ('filelock/resume.pt','filelock/optimizer.pt','filelock/teachers.npy',
                    'filelock/unqualified.py','filelock/foreign.so','huggingface_hub/_login.py',
                    'huggingface_hub/inference/__init__.py','huggingface_hub/utils/_xet_progress_reporting.py',
                    'tqdm/contrib/slack.py','tqdm/notebook.py')]+[root/'foreign.py']
                for path in forbidden:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'forbidden')
                original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
                original_before=copy.deepcopy(original);required_before=dict(f.context['required_guards'])
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name.split('.')[0] in prefixes:sys.modules.pop(name)
                        with f.boundary():
                            lock=module('filelock',f.site/'filelock/__init__.py') if direct_loader else importlib.import_module('filelock')
                            self.assertEqual((lock.value,lock.marker),(127,'source'))
                            for name in ('huggingface_hub._snapshot_download','huggingface_hub.repocard'):
                                self.assertEqual(importlib.import_module(name).value,131)
                            self.assertEqual(sys.modules['tqdm.contrib.concurrent'].value,8)
                            self.assertEqual(importlib.import_module('tqdm.contrib.logging').value,4)
                            for path,raw in sources.items():
                                name=str(path.relative_to(f.site)).removesuffix('.py').replace('/','.').removesuffix('.__init__')
                                value=sys.modules[name]
                                self.assertIsInstance(value.__loader__,importlib.machinery.SourceFileLoader)
                                self.assertEqual((value.__file__,value.__spec__.origin,value.marker),(str(path),str(path),'source'))
                                self.assertEqual(path.read_bytes(),raw)
                                with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                                with self.assertRaisesRegex(ValueError,'attempted write'):path.write_bytes(b'changed')
                            for path in (*forbidden,f.extra_records['filelock'],f.extra_records['huggingface_hub'],f.extra_records['tqdm']):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                            with self.assertRaisesRegex(ValueError,'external dependency'):
                                importlib.import_module('filelock.unqualified')
                self.assertEqual(original,original_before);self.assertEqual(f.context['required_guards'],required_before)
                foreign=SimpleNamespace(__file__=str(forbidden[-1]),__spec__=SimpleNamespace(origin=str(forbidden[-1])))
                for name in ('filelock','filelock._api','filelock._soft_rw._sync','huggingface_hub.hf_api',
                        'tqdm.contrib.concurrent','tqdm.contrib.logging'):
                    with patch.dict(sys.modules,{name:foreign}),self.assertRaisesRegex(ValueError,'origin differs'):
                        with f.boundary():pass
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0] in prefixes:sys.modules.pop(name)
            sys.modules.update(saved)

    def test_filelock_hub_closure_requires_original_record_hash_and_size(self):
        for distribution,name in (('filelock','filelock/_soft_rw/_sync.py'),
                ('huggingface_hub','huggingface_hub/hf_api.py'),('tqdm','tqdm/contrib/concurrent.py'),
                ('tqdm','tqdm/contrib/logging.py')):
            for case in ('missing_guard','foreign_guard','mutated_record','foreign_record','missing_row','wrong_hash','wrong_size'):
                with self.subTest(distribution=distribution,case=case),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);f=PortableRuntimeFixture(root);record=f.extra_records[distribution]
                    if case=='missing_guard':f.context['required_guards'].pop(str(record))
                    elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                    elif case=='mutated_record':record.write_bytes(record.read_bytes()+b'\n')
                    elif case=='foreign_record':
                        foreign=root/'foreign'/record.parent.name/'RECORD';foreign.parent.mkdir(parents=True)
                        foreign.write_bytes(record.read_bytes());h=hashlib.sha256(foreign.read_bytes()).hexdigest()
                        f.context['required_guards'].pop(str(record))
                        f.context['guards'][str(foreign)]=f.context['required_guards'][str(foreign)]=h
                    else:
                        rows=list(csv.reader(record.read_text().splitlines()));row=next(r for r in rows if r[0]==name)
                        if case=='missing_row':rows.remove(row)
                        elif case=='wrong_hash':row[1]='sha256='+'A'*43
                        else:row[2]='999'
                        record.write_text(''.join(','.join(r)+'\n' for r in rows))
                        h=hashlib.sha256(record.read_bytes()).hexdigest()
                        f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                    with self.assertRaises(ValueError):
                        with f.boundary():pass

    def test_filelock_hub_source_mutations_rejected_before_and_after_cached_boundary(self):
        names=({'filelock/'+n+'.py' for n in FILELOCK_IMPORTS}|
            {'huggingface_hub/'+n+'.py' for n in HUB_IMPORTS}|
            {'tqdm/contrib/__init__.py','tqdm/contrib/concurrent.py','tqdm/contrib/logging.py',
                'filelock-3.29.4.dist-info/METADATA'})
        for cached in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory))
                if cached:
                    with f.boundary():pass
                for name in sorted(names):
                    path=f.site/name;raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    try:
                        with self.subTest(name=name,cached=cached),self.assertRaisesRegex(ValueError,'SHA256'):
                            with f.boundary():pass
                    finally:
                        path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))

    def test_accelerate_runtime_inventory_is_finite(self):
        self.assertEqual(len(ACCELERATE_IMPORTS),49)
        self.assertEqual(e.RUNTIME_SOURCES['accelerate'],set(ACCELERATE_IMPORTS)|
            {'accelerate-1.14.0.dist-info/METADATA'})
        self.assertFalse(set(ACCELERATE_IMPORTS)&{'accelerate/utils/rich.py',
            'accelerate/utils/deepspeed.py','accelerate/commands/launch.py','accelerate/local_sgd.py',
            'accelerate/memory_utils.py','accelerate/test_utils/__init__.py'})

    def test_accelerate_sources_and_original_records_reject_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            f=PortableRuntimeFixture(Path(directory),accelerate=True)
            for cached in (False,True):
                f.context.pop('portable_audits',None)
                if cached:
                    with f.boundary():pass
                for name in sorted(ACCELERATE_IMPORTS):
                    path=f.site/name;raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    try:
                        with self.subTest(source=name,cached=cached),self.assertRaisesRegex(ValueError,'SHA256'):
                            with f.boundary():pass
                    finally:path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
            f.context.pop('portable_audits',None)
            record=f.extra_records['accelerate'];record_raw=record.read_bytes()
            source='accelerate/commands/config/config_args.py'
            for case in ('missing_guard','foreign_guard','mutated_record','foreign_record','missing_row','wrong_hash','wrong_size'):
                guards=dict(f.context['guards']);required=dict(f.context['required_guards'])
                if case=='missing_guard':f.context['required_guards'].pop(str(record))
                elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                elif case=='mutated_record':record.write_bytes(record_raw+b'\n')
                elif case=='foreign_record':
                    foreign=f.site.parent/'foreign'/record.parent.name/'RECORD';foreign.parent.mkdir(parents=True)
                    foreign.write_bytes(record_raw);f.context['required_guards'].pop(str(record))
                    f.context['guards'][str(foreign)]=required[str(record)]
                else:
                    rows=list(csv.reader(record_raw.decode().splitlines()));row=next(r for r in rows if r[0]==source)
                    if case=='missing_row':rows.remove(row)
                    elif case=='wrong_hash':row[1]='sha256='+'A'*43
                    else:row[2]='999'
                    record.write_text(''.join(','.join(r)+'\n' for r in rows))
                    h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                try:
                    with self.subTest(record=case),self.assertRaises(ValueError):
                        with f.boundary():pass
                finally:
                    record.write_bytes(record_raw);f.context['guards']=guards;f.context['required_guards']=required

    def test_scientific_runtime_inventory_is_finite(self):
        counts={'scipy':409,'scikit_learn':109,'joblib':38,'threadpoolctl':1,'pandas':250,
            'python_dateutil':14,'six':1,'narwhals':56,'psutil':5,'pyarrow':9,'rich':54}
        assets={'sklearn/utils/_repr_html/'+n+'.css' for n in ('estimator','params','features')}
        assets.add('dateutil/zoneinfo/dateutil-zoneinfo.tar.gz')
        for distribution,count in counts.items():
            paths=e.RUNTIME_SOURCES[distribution]
            self.assertEqual(sum(n.endswith('.py') for n in paths),count)
            self.assertTrue(all('..' not in Path(n).parts and not Path(n).is_absolute() for n in paths))
            self.assertTrue(all(n.endswith('.py') or n in assets for n in paths))
        selected=set().union(*(e.RUNTIME_SOURCES[d] for d in counts))
        self.assertEqual({n for n in selected if not n.endswith('.py')},assets)
        self.assertTrue(set(SCIENTIFIC_IMPORTS)<=selected)
        self.assertTrue({'scipy/_external/array_api_compat/numpy/fft.py',
            'sklearn/externals/array_api_compat/numpy/fft.py'}<=selected)
        forbidden={'sklearn/cluster/__init__.py','sklearn/ensemble/__init__.py',
            'sklearn/datasets/__init__.py','scipy/datasets/__init__.py','scipy/io/__init__.py',
            'pandas/plotting/_matplotlib/__init__.py','pandas/_version.py','dateutil/rrule.py',
            'dateutil/zoneinfo/rebuild.py','dateutil/zoneinfo/foreign.tar.gz',
            'narwhals/_arrow/dataframe.py','narwhals/_pandas_like/dataframe.py',
            'rich/markdown.py','rich/syntax.py','pyarrow/parquet/__init__.py'}
        self.assertFalse(selected & forbidden)
        for distribution in ('markdown_it_py','mdurl','pygments','tzdata'):
            self.assertNotIn(distribution,e.RUNTIME_SOURCES)

    def test_bundle_boundary_scientific_transitive_source_fallback_and_denials(self):
        prefixes=tuple(package for package,_ in SCIENTIFIC_DISTRIBUTIONS.values())+('accelerate','yaml','packaging','tqdm')
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0] in prefixes}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root,scientific=True,accelerate=True)
                sources={f.site/n:f.extra_sources[f.site/n] for n in set(SCIENTIFIC_IMPORTS)|set(ACCELERATE_IMPORTS)}
                for path,raw in sources.items():
                    prior=path.stat();path.write_bytes(raw.replace(b'source',b'cached'))
                    py_compile.compile(str(path),doraise=True)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                forbidden=[f.site/n for n in ('sklearn/resume.pt','scipy/optimizer.pt','pandas/teachers.npy',
                    'sklearn/cluster/__init__.py','scipy/io/__init__.py','scipy/sparse/csgraph/_optional.py',
                    'pandas/plotting/_matplotlib/__init__.py',
                    'dateutil/rrule.py','dateutil/zoneinfo/rebuild.py','dateutil/zoneinfo/foreign.tar.gz',
                    'accelerate/resume.pt','accelerate/optimizer.pt','accelerate/teachers.npy',
                    'accelerate/utils/rich.py','accelerate/utils/deepspeed.py','accelerate/commands/launch.py',
                    'accelerate/test_utils/__init__.py','accelerate/foreign.so',
                    'narwhals/_arrow/dataframe.py','rich/markdown.py','pyarrow/parquet/__init__.py',
                    'sklearn/utils/_repr_html/estimator.js','scipy/foreign.so')]+[root/'foreign.py',root/'proc/stat']
                for path in forbidden:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'forbidden')
                # Extra rows in an original RECORD still confer no optional code/native access.
                for distribution,names in (('scipy',('scipy/foreign.so',)),
                        ('python_dateutil',('dateutil/rrule.py','dateutil/zoneinfo/rebuild.py','dateutil/zoneinfo/foreign.tar.gz')),
                        ('accelerate',('accelerate/utils/rich.py','accelerate/foreign.so'))):
                    record=f.extra_records[distribution]
                    for name in names:
                        raw=(f.site/name).read_bytes()
                        record.write_text(record.read_text()+name+',sha256='+
                            base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')+','+str(len(raw))+'\n')
                    h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                originals=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
                original_before=copy.deepcopy(originals);required_before=dict(f.context['required_guards'])
                natives={name:SimpleNamespace(value=137,__file__=str(f.site/path),
                    __spec__=SimpleNamespace(origin=str(f.site/path))) for name,path in SCIENTIFIC_NATIVE_IMPORTS.items()}
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name.split('.')[0] in prefixes:sys.modules.pop(name)
                        with patch.dict(sys.modules,natives),f.boundary():
                            loaded=module('sklearn',f.site/'sklearn/__init__.py') if direct_loader else importlib.import_module('sklearn')
                            self.assertEqual((loaded.value,loaded.marker),(137,'source'))
                            self.assertEqual(sys.modules['pandas._libs.tslibs'].zone,b'pinned UTC archive member')
                            archive=f.site/'dateutil/zoneinfo/dateutil-zoneinfo.tar.gz'
                            self.assertEqual(archive.read_bytes(),f.extra_sources[archive])
                            with self.assertRaisesRegex(ValueError,'attempted write'):archive.write_bytes(b'changed')
                            importlib.import_module('sklearn.metrics')
                            self.assertEqual(importlib.import_module('accelerate').value,157)
                            for path,raw in sources.items():
                                name=str(path.relative_to(f.site)).removesuffix('.py').replace('/','.').removesuffix('.__init__')
                                loaded=sys.modules[name]
                                self.assertIsInstance(loaded.__loader__,importlib.machinery.SourceFileLoader)
                                self.assertEqual((loaded.__file__,loaded.__spec__.origin,loaded.marker),(str(path),str(path),'source'))
                                self.assertEqual(path.read_bytes(),raw)
                                with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                                with self.assertRaisesRegex(ValueError,'attempted write'):path.write_bytes(b'changed')
                            for path in (*forbidden,*(f.extra_records[d] for d in (*SCIENTIFIC_DISTRIBUTIONS,'accelerate'))):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                            for name in ('sklearn.cluster','scipy.io','scipy.sparse.csgraph._optional',
                                    'dateutil.rrule','dateutil.zoneinfo.rebuild','accelerate.utils.rich',
                                    'accelerate.commands.launch','rich.markdown'):
                                with self.assertRaisesRegex(ValueError,'external dependency'):importlib.import_module(name)
                # The complete inventory, including modules absent from the small graph,
                # has been authenticated to the fixture's original RECORD bytes.
                for distribution in (*SCIENTIFIC_DISTRIBUTIONS,'accelerate'):
                    for name in e.RUNTIME_SOURCES[distribution]:
                        path=f.site/name
                        self.assertEqual(f.context['guards'][str(path)],hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(originals,original_before);self.assertEqual(f.context['required_guards'],required_before)
                foreign=SimpleNamespace(__file__=str(forbidden[-2]),__spec__=SimpleNamespace(origin=str(forbidden[-2])))
                for name in ('sklearn.utils.validation','scipy.stats._stats_py','scipy.sparse.csgraph._validation','pandas.compat.pyarrow',
                        'dateutil.parser._parser','dateutil.easter','dateutil.zoneinfo','narwhals.stable.v2','psutil._pslinux',
                        'rich.console','accelerate.state','accelerate.utils.launch','accelerate.commands.config.config_args'):
                    with patch.dict(sys.modules,{name:foreign}),self.assertRaisesRegex(ValueError,'origin differs'):
                        with f.boundary():pass
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0] in prefixes:sys.modules.pop(name)
            sys.modules.update(saved)

    def test_scientific_runtime_mutations_and_original_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            f=PortableRuntimeFixture(Path(directory),scientific=True)
            paths=[f.site/n for n in ('sklearn/utils/validation.py','scipy/stats/_stats_py.py','scipy/sparse/csgraph/_validation.py',
                'pandas/compat/pyarrow.py','dateutil/parser/_parser.py','narwhals/stable/v2/__init__.py',
                'dateutil/easter.py','dateutil/zoneinfo/__init__.py','dateutil/zoneinfo/dateutil-zoneinfo.tar.gz',
                'joblib/externals/loky/process_executor.py','psutil/_pslinux.py','pyarrow/compute.py',
                'rich/console.py','threadpoolctl.py','six.py','sklearn/utils/_repr_html/estimator.css')]
            for cached in (False,True):
                f.context.pop('portable_audits',None)
                if cached:
                    with f.boundary():pass
                for path in paths:
                    raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    try:
                        with self.subTest(path=path.name,cached=cached),self.assertRaisesRegex(ValueError,'SHA256'):
                            with f.boundary():pass
                    finally:path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
            f.context.pop('portable_audits',None)
            for distribution,source in (('scipy','scipy/sparse/csgraph/_validation.py'),
                    ('python_dateutil','dateutil/easter.py'),
                    ('python_dateutil','dateutil/zoneinfo/__init__.py'),
                    ('python_dateutil','dateutil/zoneinfo/dateutil-zoneinfo.tar.gz')):
                record=f.extra_records[distribution];record_raw=record.read_bytes();record_sha=f.context['required_guards'][str(record)]
                for case in ('missing_guard','foreign_guard','mutated_record','foreign_record','missing_row','wrong_hash','wrong_size'):
                    guards=dict(f.context['guards']);required=dict(f.context['required_guards'])
                    if case=='missing_guard':f.context['required_guards'].pop(str(record))
                    elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                    elif case=='mutated_record':record.write_bytes(record_raw+b'\n')
                    elif case=='foreign_record':
                        foreign=f.site.parent/'foreign'/record.parent.name/'RECORD';foreign.parent.mkdir(parents=True,exist_ok=True)
                        foreign.write_bytes(record_raw);f.context['required_guards'].pop(str(record))
                        f.context['guards'][str(foreign)]=f.context['required_guards'][str(foreign)]=record_sha
                    else:
                        rows=list(csv.reader(record_raw.decode().splitlines()));row=next(r for r in rows if r[0]==source)
                        if case=='missing_row':rows.remove(row)
                        elif case=='wrong_hash':row[1]='sha256='+'A'*43
                        else:row[2]='999'
                        record.write_text(''.join(','.join(r)+'\n' for r in rows))
                        h=hashlib.sha256(record.read_bytes()).hexdigest()
                        f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                    try:
                        with self.subTest(source=source,record=case),self.assertRaises(ValueError):
                            with f.boundary():pass
                    finally:
                        record.write_bytes(record_raw);f.context['guards']=guards;f.context['required_guards']=required
            record=f.extra_records['scipy'];record_raw=record.read_bytes()
            native=f.site/SCIENTIFIC_NATIVE_IMPORTS['scipy.sparse.csgraph._tools']
            original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
            for case in ('unobserved','original_hash','required_guard','record_hash','record_size'):
                saved=copy.deepcopy(original);guards=dict(f.context['guards']);required=dict(f.context['required_guards'])
                if case=='unobserved':original['native_files'].remove(str(native))
                elif case=='original_hash':original['files'][str(native)]='a'*64
                elif case=='required_guard':f.context['required_guards'][str(native)]='a'*64
                else:
                    rows=list(csv.reader(record_raw.decode().splitlines()));row=next(r for r in rows if r[0]==str(native.relative_to(f.site)))
                    if case=='record_hash':row[1]='sha256='+'A'*43
                    else:row[2]='999'
                    record.write_text(''.join(','.join(r)+'\n' for r in rows))
                    h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                try:
                    if case=='unobserved':
                        with f.boundary(),self.assertRaisesRegex(ValueError,'external dependency'):native.read_bytes()
                    else:
                        with self.subTest(native=case),self.assertRaises(ValueError):
                            with f.boundary():pass
                finally:
                    original.clear();original.update(saved);record.write_bytes(record_raw)
                    f.context['guards']=guards;f.context['required_guards']=required;f.context.pop('portable_audits',None)

    def test_charset_source_loader_fallback_preserves_native_wrapper_denial(self):
        selected={'charset_normalizer/'+n+'.py' for n in CHARSET_IMPORTS}
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0]=='charset_normalizer'}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root)
                native=f.site/'charset_normalizer/cd.cpython-313-aarch64-linux-gnu.so'
                mapped=SimpleNamespace(value=151,__file__=str(native),__spec__=SimpleNamespace(origin=str(native)))
                for name in selected:
                    path=f.site/name;raw=f.extra_sources[path];prior=path.stat()
                    path.write_bytes(raw.replace(b'source',b'cached'));py_compile.compile(str(path),doraise=True)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                forbidden=[f.site/n for n in ('charset_normalizer/md.cpython-313-aarch64-linux-gnu.so',
                    'charset_normalizer/__main__.py','charset_normalizer/cli/__init__.py',
                    'charset_normalizer/cli/__main__.py','charset_normalizer/unqualified.py',
                    'charset_normalizer/resume.pt','charset_normalizer/optimizer.pt','charset_normalizer/teachers.npy',
                    'chardet/__init__.py')]+[root/'foreign.py']
                for path in forbidden:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'forbidden')
                record=f.extra_records['charset_normalizer'];raw=forbidden[0].read_bytes()
                record.write_text(record.read_text()+str(forbidden[0].relative_to(f.site))+',sha256='+
                    base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')+','+str(len(raw))+'\n')
                h=hashlib.sha256(record.read_bytes()).hexdigest()
                f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
                before=copy.deepcopy(original);required=dict(f.context['required_guards'])
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name.split('.')[0]=='charset_normalizer':sys.modules.pop(name)
                        with patch.dict(sys.modules,{'charset_normalizer.cd':mapped}),f.boundary():
                            loaded=module('charset_normalizer',f.site/'charset_normalizer/__init__.py') if direct_loader else importlib.import_module('charset_normalizer')
                            self.assertEqual((loaded.value,loaded.marker),(151,'source'))
                            for path in (f.site/n for n in selected if n!='charset_normalizer/cd.py'):
                                name=str(path.relative_to(f.site)).removesuffix('.py').replace('/','.').removesuffix('.__init__')
                                loaded=sys.modules[name]
                                self.assertIsInstance(loaded.__loader__,importlib.machinery.SourceFileLoader)
                                self.assertEqual((loaded.__file__,loaded.__spec__.origin,loaded.marker),(str(path),str(path),'source'))
                                with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                                with self.assertRaisesRegex(ValueError,'attempted write'):path.write_bytes(b'changed')
                            self.assertIs(sys.modules['charset_normalizer.cd'],mapped)
                            for path in (*forbidden,record):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                            for name in ('charset_normalizer.cli','charset_normalizer.unqualified','chardet'):
                                with self.assertRaisesRegex(ValueError,'external dependency'):importlib.import_module(name)
                self.assertEqual(e.RUNTIME_SOURCES['charset_normalizer'],selected)
                self.assertEqual(original,before);self.assertEqual(f.context['required_guards'],required)
                self.assertNotIn(str(forbidden[0]),f.context['guards'])
                foreign=SimpleNamespace(__file__=str(forbidden[-1]),__spec__=SimpleNamespace(origin=str(forbidden[-1])))
                for name in ('charset_normalizer','charset_normalizer.api','charset_normalizer.md','charset_normalizer.cd'):
                    with patch.dict(sys.modules,{name:foreign}),self.assertRaisesRegex(ValueError,'origin differs'):
                        with f.boundary():pass
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0]=='charset_normalizer':sys.modules.pop(name)
            sys.modules.update(saved)

    def test_charset_original_records_and_current_source_bytes_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            f=PortableRuntimeFixture(Path(directory))
            for cached in (False,True):
                f.context.pop('portable_audits',None)
                if cached:
                    with f.boundary():pass
                for name in CHARSET_IMPORTS:
                    path=f.site/('charset_normalizer/'+name+'.py');raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    try:
                        with self.subTest(source=name,cached=cached),self.assertRaisesRegex(ValueError,'SHA256'):
                            with f.boundary():pass
                    finally:path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
            f.context.pop('portable_audits',None)
            record=f.extra_records['charset_normalizer'];raw=record.read_bytes()
            for case in ('missing_guard','mutated_record','missing_row','wrong_hash','wrong_size'):
                guards=dict(f.context['guards']);required=dict(f.context['required_guards'])
                if case=='missing_guard':f.context['required_guards'].pop(str(record))
                elif case=='mutated_record':record.write_bytes(raw+b'\n')
                else:
                    rows=list(csv.reader(raw.decode().splitlines()));row=next(r for r in rows if r[0]=='charset_normalizer/api.py')
                    if case=='missing_row':rows.remove(row)
                    elif case=='wrong_hash':row[1]='sha256='+'A'*43
                    else:row[2]='999'
                    record.write_text(''.join(','.join(r)+'\n' for r in rows));h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                try:
                    with self.subTest(record=case),self.assertRaises(ValueError):
                        with f.boundary():pass
                finally:record.write_bytes(raw);f.context['guards']=guards;f.context['required_guards']=required

    def test_numpy_distribution_origin_preserves_missing_metadata(self):
        import importlib.metadata
        with tempfile.TemporaryDirectory() as directory:
            f=PortableRuntimeFixture(Path(directory));probe=f.metadata['numpy'].with_name('direct_url.json')
            required_before=dict(f.context['required_guards'])
            other=f.metadata['packaging'].with_name('direct_url.json');other.write_text('{}')
            unknown=probe.with_name('foreign.json');unknown.write_text('{}')
            for cached in (False,True):
                with self.subTest(cached=cached),patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]),f.boundary():
                    distribution=importlib.metadata.distribution('numpy')
                    self.assertIsNone(distribution.read_text('direct_url.json'))
                    self.assertIsNone(distribution.origin)
                    # The exact installed NumPy >=3.13 expression handles None
                    # by its normal AttributeError path, without importing NumPy.
                    try:editable=distribution.origin.dir_info.editable
                    except AttributeError:editable=False
                    self.assertFalse(editable)
                    with self.assertRaises(FileNotFoundError):probe.read_bytes()
                    with self.assertRaisesRegex(ValueError,'attempted write'):probe.write_text('{}')
                    for path in (other,unknown):
                        with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                    with self.assertRaisesRegex(ValueError,'external dependency'):
                        importlib.metadata.distribution('packaging').origin
            self.assertFalse(probe.exists());self.assertNotIn(str(probe),f.context['guards'])
            self.assertEqual(f.context['required_guards'],required_before)

    def test_numpy_missing_origin_rejects_new_files_symlinks_and_record_rows(self):
        import importlib.metadata
        for cached in (False,True):
            for case in ('regular','empty','symlink','dangling'):
                with self.subTest(cached=cached,case=case),tempfile.TemporaryDirectory() as directory:
                    f=PortableRuntimeFixture(Path(directory));probe=f.metadata['numpy'].with_name('direct_url.json')
                    if cached:
                        with f.boundary():pass
                    target=f.bundle/'original.json';target.write_text('{}')
                    if case=='regular':probe.write_text('{"dir_info":{"editable":true}}')
                    elif case=='empty':probe.write_bytes(b'')
                    else:probe.symlink_to(target if case=='symlink' else target.with_name('missing.json'))
                    with self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                        with f.boundary():pass
        for case in ('regular','symlink'):
            with self.subTest(live=case),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory));probe=f.metadata['numpy'].with_name('direct_url.json')
                replacement=f.bundle/'replacement.json';target=f.bundle/'original.json';target.write_text('{}')
                if case=='regular':replacement.write_text('{"dir_info":{"editable":true}}')
                else:replacement.symlink_to(target)
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]),self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                    with f.boundary():
                        # A rename simulates an external concurrent replacement;
                        # the existing write-open denial remains active.
                        replacement.replace(probe)
                        with self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                            importlib.metadata.distribution('numpy').origin
        for present in (False,True):
            with self.subTest(declared=present),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory));probe=f.metadata['numpy'].with_name('direct_url.json')
                raw=b'{"dir_info":{"editable":true}}'
                if present:probe.write_bytes(raw)
                record=f.extra_records['numpy'];record.write_text(record.read_text()+str(probe.relative_to(f.site))+
                    ',sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')+','+str(len(raw))+'\n')
                h=hashlib.sha256(record.read_bytes()).hexdigest()
                f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                with self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                    with f.boundary():pass

    def test_bundle_boundary_distribution_identity_scan_and_versions(self):
        import importlib.metadata
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);f=PortableRuntimeFixture(root)
            declared=f.add_identity_distribution('optional-declared','1.2.3','optional_declared\n')
            inferred=f.add_identity_distribution('optional-inferred','2.3.4',None)
            empty=f.add_identity_distribution('optional-empty','3.4.5','')
            original=copy.deepcopy(f.context['training_context']['legacy']['selected']['source_cpu']['origins'])
            forbidden=[root/'foreign.py',declared[3].with_name('resume.pt'),declared[3].with_name('foreign.so'),
                declared[0].with_name('entry_points.txt'),declared[0].with_name('PKG-INFO'),
                declared[3].parent/'_vendor/nested-1.0.dist-info/METADATA']
            for path in forbidden:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'forbidden')
            # Even a trusted wheel's vendored identity row grants no nested metadata root.
            vendor=forbidden[-1]
            with declared[2].open('a') as stream:
                csv.writer(stream).writerow([str(vendor.relative_to(f.site)),
                    'sha256='+base64.urlsafe_b64encode(hashlib.sha256(vendor.read_bytes()).digest()).decode().rstrip('='),
                    str(vendor.stat().st_size)])
            h=hashlib.sha256(declared[2].read_bytes()).hexdigest()
            f.context['guards'][str(declared[2])]=f.context['required_guards'][str(declared[2])]=h
            required=dict(f.context['required_guards'])
            with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                for cached in (False,True):
                    with self.subTest(cached=cached),f.boundary():
                        mapping=importlib.metadata.packages_distributions()
                        self.assertEqual(mapping['optional_declared'],['optional-declared'])
                        self.assertEqual(mapping['optional_inferred'],['optional-inferred'])
                        self.assertEqual(mapping['optional_empty'],['optional-empty'])
                        for distribution,version in (('optional-declared','1.2.3'),('optional-inferred','2.3.4'),('optional-empty','3.4.5')):
                            self.assertEqual(importlib.metadata.version(distribution),version)
                        self.assertEqual(importlib.metadata.version('torch'),'2.12.1')
                        self.assertEqual(importlib.metadata.version('Pillow'),'12.2.0')
                        for metadata,top,record,source in (declared,inferred,empty):
                            self.assertIn(str(metadata),f.context['guards'])
                            self.assertNotIn(str(source),f.context['guards'])
                            with self.assertRaisesRegex(ValueError,'external dependency'):source.read_bytes()
                            with self.assertRaisesRegex(ValueError,'external dependency'):
                                importlib.import_module(source.parent.name)
                            with self.assertRaisesRegex(ValueError,'attempted write'):metadata.write_bytes(b'changed')
                            with self.assertRaisesRegex(ValueError,'attempted write'):top.write_bytes(b'changed')
                            with self.assertRaisesRegex(ValueError,'attempted write'):record.write_bytes(b'changed')
                        with self.assertRaises(FileNotFoundError):inferred[1].read_bytes()
                        self.assertEqual(inferred[2].read_bytes(),inferred[2].read_text().encode())
                        self.assertEqual(empty[1].read_bytes(),b'')
                        with self.assertRaisesRegex(ValueError,'external dependency'):declared[2].read_bytes()
                        for path in forbidden:
                            with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                # A known absent probe is rechecked on entry, at its read, and on exit.
                inferred[1].write_text('unqualified\n')
                with self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                    with f.boundary():pass
                inferred[1].unlink()
                with self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                    with f.boundary():inferred[1].symlink_to(inferred[3])
                inferred[1].unlink()
                with f.boundary():
                    inferred[1].symlink_to(root/'missing')
                    try:
                        with self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                            inferred[1].read_bytes()
                    finally:inferred[1].unlink()
                unknown=f.site/'foreign-1.0.dist-info';unknown.mkdir();(unknown/'METADATA').write_text('Name: foreign\nVersion: 1.0\n')
                with f.boundary(),self.assertRaisesRegex(ValueError,'external dependency'):
                    importlib.metadata.packages_distributions()
            self.assertEqual(f.context['required_guards'],required)
            self.assertEqual(f.context['training_context']['legacy']['selected']['source_cpu']['origins'],original)
            for name in ('optional_declared','optional_inferred','optional_empty'):self.assertNotIn(name,sys.modules)

    def test_distribution_identity_requires_original_record_and_exact_rows(self):
        import importlib.metadata
        for identity in ('METADATA','top_level.txt'):
            for case in ('missing_guard','foreign_guard','mutated_record','foreign_record','missing_row',
                         'wrong_hash','wrong_size','duplicate_row','foreign_row','symlink'):
                with self.subTest(identity=identity,case=case),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);f=PortableRuntimeFixture(root)
                    metadata,top,record,_=f.add_identity_distribution('optional-declared','1.2.3','optional_declared\n')
                    target=metadata if identity=='METADATA' else top
                    if case=='missing_guard':f.context['required_guards'].pop(str(record))
                    elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                    elif case=='mutated_record':record.write_bytes(record.read_bytes()+b'\n')
                    elif case=='foreign_record':
                        foreign=root/'foreign'/record.parent.name/'RECORD';foreign.parent.mkdir(parents=True)
                        foreign.write_bytes(record.read_bytes());h=hashlib.sha256(foreign.read_bytes()).hexdigest()
                        f.context['required_guards'].pop(str(record))
                        f.context['guards'][str(foreign)]=f.context['required_guards'][str(foreign)]=h
                    elif case=='symlink':
                        foreign=root/'foreign.txt';foreign.write_bytes(target.read_bytes());target.unlink();target.symlink_to(foreign)
                    else:
                        rows=list(csv.reader(record.read_text().splitlines()));row=next(r for r in rows if r[0]==str(target.relative_to(f.site)))
                        if case=='missing_row':rows.remove(row)
                        elif case=='wrong_hash':row[1]='sha256='+'A'*43
                        elif case=='wrong_size':row[2]='999'
                        elif case=='duplicate_row':rows.append(row.copy())
                        else:row[0]='../foreign/'+row[0]
                        record.write_text(''.join(','.join(r)+'\n' for r in rows))
                        h=hashlib.sha256(record.read_bytes()).hexdigest()
                        f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                    with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]),self.assertRaises(ValueError):
                        with f.boundary():importlib.metadata.packages_distributions()

    def test_distribution_identity_current_bytes_rechecked_with_restored_mtime(self):
        for cached in (False,True):
            with self.subTest(cached=cached),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory))
                metadata,top,record,_=f.add_identity_distribution('optional-declared','1.2.3','optional_declared\n')
                inferred=f.add_identity_distribution('optional-inferred','2.3.4',None)
                if cached:
                    with f.boundary():pass
                for path in (metadata,top,record,inferred[0],inferred[2]):
                    raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    try:
                        with self.subTest(path=path),self.assertRaisesRegex(ValueError,'SHA256'):
                            with f.boundary():pass
                    finally:
                        path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))

    def test_bundle_boundary_yaml_transitive_source_fallback_and_denials(self):
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0]=='yaml'}
        try:
            for name in saved:sys.modules.pop(name)
            for observed_native in (False,True):
                with self.subTest(observed_native=observed_native),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);f=PortableRuntimeFixture(root,yaml_native=observed_native)
                    sources={p:raw for p,raw in f.extra_sources.items() if p.parent.name=='yaml'}
                    for path,raw in sources.items():
                        prior=path.stat();path.write_bytes(raw.replace(b'source',b'cached').replace(b'113',b'999'))
                        py_compile.compile(str(path),doraise=True)
                        path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    forbidden=[f.site/'yaml'/n for n in ('resume.pt','optimizer.pt','teachers.npy','unqualified.py','foreign.so')]
                    forbidden.append(root/'foreign.py')
                    for path in forbidden:path.write_bytes(b'forbidden')
                    native=f.site/'yaml/_yaml.cpython-313-aarch64-linux-gnu.so'
                    original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
                    original_before=copy.deepcopy(original);required_before=dict(f.context['required_guards'])
                    # Represent an already mapped original module; never load an extension.
                    mapped=SimpleNamespace(__file__=str(native),__spec__=SimpleNamespace(origin=str(native)),
                        CParser=object(),CEmitter=object())
                    with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                        for direct_loader in (True,False):
                            for name in tuple(sys.modules):
                                if name.split('.')[0]=='yaml':sys.modules.pop(name)
                            with patch.dict(sys.modules,{'yaml._yaml':mapped} if observed_native else {}),f.boundary():
                                yaml=module('yaml',f.site/'yaml/__init__.py') if direct_loader else importlib.import_module('yaml')
                                self.assertEqual((yaml.value,yaml.marker,yaml.__with_libyaml__),(113,'source',observed_native))
                                expected={'yaml'+('' if p.stem=='__init__' else '.'+p.stem) for p in sources}
                                if not observed_native:expected.remove('yaml.cyaml')
                                self.assertEqual({n for n in sys.modules if n.split('.')[0]=='yaml' and n!='yaml._yaml'},expected)
                                for name in expected:
                                    value=sys.modules[name];path=f.site/'yaml'/('__init__.py' if name=='yaml' else name.split('.')[1]+'.py')
                                    self.assertIsInstance(value.__loader__,importlib.machinery.SourceFileLoader)
                                    self.assertEqual((value.__file__,value.__spec__.origin,value.marker),(str(path),str(path),'source'))
                                for path,raw in sources.items():
                                    self.assertEqual(path.read_bytes(),raw)
                                    with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                                    with self.assertRaisesRegex(ValueError,'attempted write'):path.write_bytes(b'changed')
                                for path in (*forbidden,f.extra_records['pyyaml']):
                                    with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                                with self.assertRaisesRegex(ValueError,'external dependency'):
                                    importlib.import_module('yaml.unqualified')
                                if observed_native:
                                    self.assertEqual(native.read_bytes(),b'original exact native file; never executed')
                                    with self.assertRaisesRegex(ValueError,'attempted write'):native.write_bytes(b'changed')
                                else:
                                    with self.assertRaisesRegex(ValueError,'external dependency'):native.read_bytes()
                    self.assertEqual(original,original_before);self.assertEqual(f.context['required_guards'],required_before)
                    foreign=SimpleNamespace(__file__=str(forbidden[-1]),__spec__=SimpleNamespace(origin=str(forbidden[-1])))
                    for name in ('yaml','yaml.loader','yaml.cyaml','yaml.nodes'):
                        with patch.dict(sys.modules,{name:foreign}),self.assertRaisesRegex(ValueError,'origin differs'):
                            with f.boundary():pass
                    for name in tuple(sys.modules):
                        if name.split('.')[0]=='yaml':sys.modules.pop(name)
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0]=='yaml':sys.modules.pop(name)
            sys.modules.update(saved)

    def test_yaml_runtime_requires_original_record_hash_and_complete_rows(self):
        for case in ('missing_guard','foreign_guard','mutated_record','foreign_record','missing_row','wrong_hash','wrong_size'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root);record=f.extra_records['pyyaml']
                if case=='missing_guard':f.context['required_guards'].pop(str(record))
                elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                elif case=='mutated_record':record.write_bytes(record.read_bytes()+b'\n')
                elif case=='foreign_record':
                    foreign=root/'foreign'/record.parent.name/'RECORD';foreign.parent.mkdir(parents=True)
                    foreign.write_bytes(record.read_bytes());h=hashlib.sha256(foreign.read_bytes()).hexdigest()
                    f.context['required_guards'].pop(str(record))
                    f.context['guards'][str(foreign)]=f.context['required_guards'][str(foreign)]=h
                else:
                    rows=list(csv.reader(record.read_text().splitlines()))
                    row=next(r for r in rows if r[0]=='yaml/loader.py')
                    if case=='missing_row':rows.remove(row)
                    elif case=='wrong_hash':row[1]='sha256='+'A'*43
                    else:row[2]='999'
                    record.write_text(''.join(','.join(r)+'\n' for r in rows))
                    h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                with self.assertRaises(ValueError):
                    with f.boundary():pass

    def test_yaml_current_source_and_record_bytes_rechecked_with_restored_mtime(self):
        for cached in (False,True):
            with self.subTest(cached=cached),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory))
                if cached:
                    with f.boundary():pass
                paths=[p for p in f.extra_sources if p.parent.name=='yaml']+[f.extra_records['pyyaml'],f.metadata['pyyaml']]
                for path in paths:
                    raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    with self.subTest(path=path.name),self.assertRaisesRegex(ValueError,'SHA256'):
                        with f.boundary():pass
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))

    def test_yaml_native_authority_stays_exact_and_optional(self):
        for case in ('unobserved','missing_guard','foreign_guard','foreign_origin','not_native_files',
                     'record_hash','record_size','record_missing','module_origin','cached_guard','cached_origin','cached_bytes'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory),yaml_native=case!='unobserved')
                native=f.site/'yaml/_yaml.cpython-313-aarch64-linux-gnu.so';record=f.extra_records['pyyaml']
                original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
                if case.startswith('cached_'):
                    with f.boundary():pass
                if case=='unobserved':
                    native.write_bytes(b'unobserved')
                    h=hashlib.sha256(native.read_bytes()).hexdigest()
                    f.context['guards'][str(native)]=f.context['required_guards'][str(native)]=h
                    with record.open('a') as stream:
                        csv.writer(stream).writerow([str(native.relative_to(f.site)),
                            'sha256='+base64.urlsafe_b64encode(bytes.fromhex(h)).decode().rstrip('='),'10'])
                elif case=='missing_guard':f.context['required_guards'].pop(str(native))
                elif case in ('foreign_guard','cached_guard'):f.context['required_guards'][str(native)]='a'*64
                elif case in ('foreign_origin','cached_origin'):original['files'][str(native)]='a'*64
                elif case=='not_native_files':original['native_files'].remove(str(native))
                elif case=='cached_bytes':
                    prior=native.stat();native.write_bytes(b'x'*prior.st_size)
                    os.utime(native,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                elif case.startswith('record_'):
                    rows=list(csv.reader(record.read_text().splitlines()))
                    row=next(r for r in rows if r[0]==str(native.relative_to(f.site)))
                    if case=='record_hash':row[1]='sha256='+'A'*43
                    elif case=='record_size':row[2]='1'
                    else:rows.remove(row)
                    record.write_text(''.join(','.join(r)+'\n' for r in rows))
                if case.startswith('record_') or case=='unobserved':
                    h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                foreign=SimpleNamespace(__file__=str(native),__spec__=SimpleNamespace(origin=str(f.site/'yaml/foreign.so')))
                with patch.dict(sys.modules,{'yaml._yaml':foreign} if case=='module_origin' else {}):
                    if case in ('unobserved','not_native_files'):
                        with f.boundary():
                            with self.assertRaisesRegex(ValueError,'external dependency'):native.read_bytes()
                    else:
                        with self.assertRaises(ValueError):
                            with f.boundary():pass

    def test_bundle_boundary_tqdm_auto_source_fallback_and_metadata_reads(self):
        import importlib.metadata
        prefixes=('tqdm','huggingface_hub')
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0] in prefixes}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root)
                sources={p:raw for p,raw in f.extra_sources.items() if p.parent.name=='tqdm'}
                for path in (f.site/'tqdm/auto.py',f.site/'tqdm/version.py'):
                    raw=sources[path];prior=path.stat();path.write_bytes(raw.replace(b'source',b'cached'))
                    py_compile.compile(str(path),doraise=True)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                forbidden=[f.site/n for n in ('tqdm/resume.pt','tqdm/teachers.npy','tqdm/notebook.py',
                    'tqdm/contrib/slack.py','accelerate/__init__.py','aiohttp/__init__.py',
                    'pydantic/__init__.py','hf_xet/unqualified.so','gradio-1.0.dist-info/METADATA',
                    'tqdm-4.68.3.dist-info/entry_points.txt')]
                foreign=root/'foreign.py';forbidden.append(foreign)
                for path in forbidden:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'forbidden')
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name.split('.')[0] in prefixes:sys.modules.pop(name)
                        with f.boundary():
                            auto=module('tqdm.auto',f.site/'tqdm/auto.py') if direct_loader else importlib.import_module('tqdm.auto')
                            self.assertEqual((auto.value,auto.marker,sys.modules['tqdm.version'].__version__,
                                sys.modules['tqdm.version'].marker),(8,'source','4.68.3','source'))
                            runtime=importlib.import_module('huggingface_hub.utils._runtime')
                            self.assertEqual(runtime._package_versions['fastai'],'N/A')
                            self.assertEqual(runtime._package_versions['Pillow'],'12.2.0')
                            for distribution,version in f.metadata_versions.items():
                                self.assertEqual(importlib.metadata.version(distribution.replace('_','-')),version)
                                self.assertIn('Version: '+version,f.metadata[distribution].read_text())
                                with self.assertRaisesRegex(ValueError,'attempted write'):
                                    f.metadata[distribution].write_bytes(b'changed')
                            for path in sources:
                                name='tqdm'+('' if path.name=='__init__.py' else '.'+path.stem)
                                value=sys.modules[name]
                                self.assertIsInstance(value.__loader__,importlib.machinery.SourceFileLoader)
                                self.assertEqual((value.__file__,value.__spec__.origin),(str(path),str(path)))
                                with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                                with self.assertRaisesRegex(ValueError,'attempted write'):path.write_bytes(b'changed')
                            for path in (*forbidden,f.extra_records['tqdm'],f.extra_records['accelerate']):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                fake=SimpleNamespace(__file__=str(foreign),__spec__=SimpleNamespace(origin=str(foreign)))
                for name in ('tqdm.auto','tqdm.version','tqdm.std'):
                    with patch.dict(sys.modules,{name:fake}),self.assertRaisesRegex(ValueError,'origin differs'):
                        with f.boundary():pass
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0] in prefixes:sys.modules.pop(name)
            sys.modules.update(saved)

    def test_tqdm_metadata_rows_require_original_record_hash_and_size(self):
        for distribution in ('tqdm','accelerate','huggingface_hub','packaging'):
            for case in ('missing_guard','foreign_guard','mutated_record','missing_row','wrong_hash','wrong_size','foreign_record'):
                with self.subTest(distribution=distribution,case=case),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);f=PortableRuntimeFixture(root)
                    record=f.metadata[distribution].with_name('RECORD')
                    if case=='missing_guard':f.context['required_guards'].pop(str(record))
                    elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                    elif case=='mutated_record':record.write_bytes(record.read_bytes()+b'\n')
                    elif case=='foreign_record':
                        foreign=root/'foreign'/record.parent.name/'RECORD';foreign.parent.mkdir(parents=True)
                        foreign.write_bytes(record.read_bytes());h=hashlib.sha256(foreign.read_bytes()).hexdigest()
                        f.context['required_guards'].pop(str(record))
                        f.context['guards'][str(foreign)]=f.context['required_guards'][str(foreign)]=h
                    else:
                        rows=list(csv.reader(record.read_text().splitlines()))
                        row=next(r for r in rows if r[0]==str(f.metadata[distribution].relative_to(f.site)))
                        if case=='missing_row':rows.remove(row)
                        elif case=='wrong_hash':row[1]='sha256='+'A'*43
                        else:row[2]='999'
                        record.write_text(''.join(','.join(r)+'\n' for r in rows))
                        h=hashlib.sha256(record.read_bytes()).hexdigest()
                        f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                    with self.assertRaises(ValueError):
                        with f.boundary():pass

    def test_tqdm_metadata_and_source_mutations_are_rejected(self):
        for name in ('tqdm/auto.py','tqdm/version.py','tqdm-4.68.3.dist-info/METADATA',
                     'accelerate-1.14.0.dist-info/METADATA','huggingface_hub-1.16.1.dist-info/METADATA'):
            for cached in (False,True):
                with self.subTest(name=name,cached=cached),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);f=PortableRuntimeFixture(root)
                    if cached:
                        with f.boundary():pass
                    path=f.site/name;raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    with self.assertRaisesRegex(ValueError,'SHA256'):
                        with f.boundary():pass

    def test_bundle_boundary_native_grants_require_record_guard_and_original_cpu_origin(self):
        for case in ('unobserved','missing_guard','foreign_guard','foreign_origin','record_hash','record_size',
                     'record_missing','module_origin','cached_guard','cached_origin'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory));original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
                if case.startswith('cached_'):
                    with f.boundary():pass
                if case=='unobserved':
                    extra=f.regex/'unobserved.so';extra.write_bytes(b'foreign')
                    h=hashlib.sha256(extra.read_bytes()).hexdigest();f.context['guards'][str(extra)]=h
                    f.context['required_guards'][str(extra)]=h
                    f.regex_rows.append(['regex/unobserved.so','sha256='+base64.urlsafe_b64encode(bytes.fromhex(h)).decode().rstrip('='),'7'])
                elif case=='missing_guard':f.context['required_guards'].pop(str(f.native))
                elif case in ('foreign_guard','cached_guard'):f.context['required_guards'][str(f.native)]='a'*64
                elif case in ('foreign_origin','cached_origin'):original['files'][str(f.native)]='a'*64
                elif case in ('record_hash','record_size','record_missing'):
                    if case=='record_hash':f.regex_rows[-1][1]='sha256='+'A'*43
                    elif case=='record_size':f.regex_rows[-1][2]='1'
                    else:f.regex_rows.pop()
                if case.startswith('record_') or case=='unobserved':
                    f.regex_record.write_text(''.join(','.join(row)+'\n' for row in f.regex_rows))
                    h=hashlib.sha256(f.regex_record.read_bytes()).hexdigest()
                    f.context['guards'][str(f.regex_record)]=f.context['required_guards'][str(f.regex_record)]=h
                foreign=SimpleNamespace(__file__=str(f.native),__spec__=SimpleNamespace(origin=str(f.regex/'foreign.so')))
                with patch.dict(sys.modules,{'regex._regex':foreign} if case=='module_origin' else {}):
                    if case=='unobserved':
                        with f.boundary():
                            for path in f.natives:self.assertEqual(path.read_bytes(),b'original exact native file; never executed')
                            with self.assertRaisesRegex(ValueError,'external dependency'):extra.read_bytes()
                    else:
                        with self.assertRaises(ValueError):
                            with f.boundary():pass

    def test_partial_metadata_receipt_and_foreign_bindings_rejected(self):
        value,args=launch();flags={'threads':1};source={'actual':'source'}
        args.authority=Path('/tmp/launch');args.authority_sha256='a'*64;args.output=Path('/tmp/output')
        context={'args':args,'launch':value,'code':dict.fromkeys(e.FILES,'b'*64),
            'training_context':{'source':source,'legacy':{'selected':{'source_cpu':{'numerical_flags':flags}}}},
            'costs':e.paired_cost(controls('first'),'first'),'reference':SimpleNamespace(check_synthetic_bootstrap=lambda _:None),
            'guards':{}}
        record={'schema':e.SCHEMA,'phase':'cpu','arm':None,'seed':None,'stage':'first','panel':'selection',
            'execution_sha256':args.execution_sha256,'source_code':context['code'],'source':source,
            'cost_policy':e.COST_POLICY,'launch':value,'authority':descriptor(args.authority),
            'authority_sha256':args.authority_sha256,'binding':e.binding(context),'numerical_flags':flags,
            'output':str(args.output),'cost':context['costs'],'resource_policy':e.policy('cpu'),
            'wall_seconds':1,'process_peak_rss_kib':100,'peak_cuda_allocated_bytes':0,'cuda_initialized':False,
            'invocation':{'argv':e.cli(args),'optimize':0,'cuda_visible_devices':'','cublas_workspace_config':':4096:8'},
            'synthetic_bootstrap':{},'files':{},'quality_read':False,'metadata_only':True,
            'updated_payloads_authenticated':True,'malformed_inference_rejected':True,
            'payload_facts':{e.label(v):{} for v in value['endpoints']},
            'calibration':{'same_role_forward_exact':True,'raw_unit_packed_exact':True,
                'residual_oracles':{e.label(v):{'C_exact_zero':False,
                    'residual_nonzero_witness':True,'omitted_C_mutant_rejected':True,
                    'wrong_mu_mutant_rejected':True} for v in value['endpoints']}}}
        for key in ('pass','engineering_admission_pass','integrity_pass','resources_pass','exit_rehash_pass',
            'sequential_model_ownership','rng_flags_preserved','both_locks_held_in_parent_authority'):record[key]=True
        for key in ('official_read','global_production_goal_met','public_latency_measured','product_go'):record[key]=False
        with patch.object(e,'read_json',return_value=value):
            e.check_receipt(context,record,'cpu')
            for endpoint in value['endpoints']:
                key=e.label(endpoint)
                proof=record['calibration']['residual_oracles'][key]
                for field in proof:
                    bad=copy.deepcopy(record);bad['calibration']['residual_oracles'][key][field]=not proof[field]
                    with self.subTest(endpoint=key,field=field),self.assertRaises(ValueError):
                        e.check_receipt(context,bad,'cpu')
            for key,bad in (('schema','siglip2-compact-fullfeature-residual-evaluation-v1'),
                ('schema','siglip2-compact-image-anchor-smooth-ap-evaluation-v1'),
                ('schema','siglip2-compact-ranking-evaluation-v1'),
                ('metadata_only',False),('updated_payloads_authenticated',False),('payload_facts',{}),
                ('quality_read',True),('source_code',{}),('stage','full'),('panel','validation'),('numerical_flags',{})):
                with self.subTest(key=key),self.assertRaises((ValueError,KeyError)):
                    e.check_receipt(context,{**record,key:bad},'cpu')

    def test_whole_unit_resource_caps(self):
        for phase in ('cpu','export','score'):
            record={'resource_policy':e.policy(phase),'wall_seconds':1,'process_peak_rss_kib':100,
                'peak_cuda_allocated_bytes':1 if phase=='export' else 0,'cuda_initialized':phase=='export'}
            e.check_resource_facts(record,phase)
            for key,value in (('wall_seconds',e.policy(phase)['seconds']),('process_peak_rss_kib',8*1024**2+1),
                ('peak_cuda_allocated_bytes',10_000_000_000),('cuda_initialized',phase!='export')):
                with self.assertRaises(ValueError):e.check_resource_facts({**record,key:value},phase)

    def test_owned_native_api_exit_and_exact_membership(self):
        from test_siglip2_nearest_ranking import NativeAdmissionFixture,driver
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);f=NativeAdmissionFixture(root)
            class ExitContext(dict):
                def __getitem__(self,key):
                    accessed.add(key);return super().__getitem__(key)
            accessed=set()
            fit=f.context['fit_context']=ExitContext(f.context['fit_context'])
            phases=fit['phase_seconds'];fit.pop('unit_started')  # Authority does not supply the run clock.
            api=f.admit()
            def code_descriptor(name,names):
                target=root/name;target.mkdir()
                for n in names:(target/n).write_bytes(PATH.with_name(n).read_bytes())
                code={n:hashlib.sha256((target/n).read_bytes()).hexdigest() for n in names}
                manifest=target/'execution.json';manifest.write_text(json.dumps(code))
                return {'root':str(target),'code':code,'execution_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
            own=code_descriptor('own',e.FILES);train=code_descriptor('train',e.TRAIN_FILES) if all(PATH.with_name(n).exists() for n in e.TRAIN_FILES) else own
            refs=[code_descriptor('ref'+str(i),pins) for i,pins in enumerate((e.NEAREST_EVALUATOR['code'],e.GENUINE_PINS,e.REFERENCE['code']))]
            # No model exists; use the REAL native API, with only disconnected trainer/source snapshots stubbed.
            trainer=SimpleNamespace(require_no_training=lambda _:None,helper_guard=lambda _:None,
                admit_bundle=lambda *_:({},{}))
            f.context['nearest']=driver
            context={'trainer':trainer,'training_context':f.context,'args':SimpleNamespace(phase='export',execution_sha256=own['execution_sha256']),
                'root':Path(own['root']),'code':own['code'],'guards':{},'launch':{'training':train,
                    'nearest_evaluator':refs[0],'genuine_evaluator':refs[1],'reference':refs[2],'endpoints':[]}}
            runtime_root=root/'runtime';runtime_root.mkdir();portable=PortableRuntimeFixture(runtime_root)
            with portable.boundary():pass
            e.merge_guards(context['guards'],portable.context['guards'])
            def action():
                phases.clear();return e.exit_rehash(context)
            with patch.object(e,'guard_helpers'),patch.object(e,'NEAREST_EVALUATOR',refs[0]), \
                patch.object(e,'GENUINE_PINS',refs[1]['code']),patch.object(e,'TRAIN_FILES',train['code']),redirect_stdout(io.StringIO()):
                # Execute the actual run prefix through native_start, stopping
                # before the Torch import. The fixture used to mask this bug
                # by supplying its own unrelated fitter start timestamp.
                run_node=next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
                stop=next(i for i,n in enumerate(run_node.body) if isinstance(n,ast.Import) and
                    any(a.name=='torch' for a in n.names))
                prefix=copy.deepcopy(run_node);prefix.body=prefix.body[:stop]
                def start(value):
                    self.assertIs(value,context)
                    self.assertIs(action(),f.legacy['origins'])
                    self.assertIs(fit['unit_started'],e.UNIT_STARTED,'inherited exit must use the original evaluator start')
                    self.assertIs(fit['phase_seconds'],phases)
                def run_prefix(node):
                    namespace={**vars(e),'cli':lambda _:sys.argv,'authority':lambda _:context,'native_start':start}
                    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(PATH),'exec'),namespace)
                    namespace['run'](context['args'])
                for phase in ('cpu','export','score'):
                    context['args'].phase=phase;fit.pop('unit_started',None)
                    before=e.time.perf_counter()-e.UNIT_STARTED
                    run_prefix(prefix)
                    after=e.time.perf_counter()-e.UNIT_STARTED
                    self.assertEqual(list(phases),['old_exit_begin','old_exit_end','own_union_begin',
                        'own_union_end','closure_begin','closure_end'])
                    self.assertTrue(all(before<=v<=after for v in phases.values()))
                    self.assertEqual(list(phases.values()),sorted(phases.values()))
                required={'legacy','old','root','args','code','guards','readout','phase_seconds','unit_started'}
                self.assertTrue(required<=accessed,'real private exit context fields were not exercised')
                # Each inherited required field is consumed by the authentic
                # private adapter; neither a gate nor a source predicate is stubbed.
                for key in sorted(required):
                    saved=fit.pop(key)
                    try:
                        phases.clear()
                        with self.subTest(missing=key),self.assertRaises(KeyError) as error:action()
                        self.assertEqual(error.exception.args,(key,))
                    finally:fit[key]=saved
                target=ast.parse("context['training_context']['fit_context']['unit_started']=None").body[0].targets[0]
                for replacement in (None,'time.perf_counter()',"context['training_context']['nearest'].UNIT_STARTED"):
                    class ChangeStart(ast.NodeTransformer):
                        count=0
                        def visit_Assign(self,node):
                            if any(ast.dump(t,include_attributes=False)==ast.dump(target,include_attributes=False) for t in node.targets):
                                self.count+=1
                                if replacement is None:return None
                                node.value=ast.parse(replacement,mode='eval').body
                            return self.generic_visit(node)
                    change=ChangeStart();mutant=change.visit(copy.deepcopy(prefix))
                    self.assertEqual(change.count,1)
                    fit.pop('unit_started',None)
                    expected=KeyError if replacement is None else AssertionError
                    with self.subTest(start=replacement),self.assertRaises(expected):run_prefix(mutant)
                    if replacement is None:
                        with self.assertRaisesRegex(KeyError,'unit_started'):action()
                context['args'].phase='export';run_prefix(prefix)
                self.assertIs(action(),f.legacy['origins'])
                f.set_origins({p:h for i,(p,h) in enumerate(f.files.items()) if i})
                with self.assertRaisesRegex(ValueError,'exact four'):action()
                f.set_origins(f.files)
                with self.assertRaisesRegex(ValueError,'owned legacy'):api.audit_origins(dict(f.legacy))
                private=next(c.cell_contents for c in api.audit_origins.__closure__ if isinstance(c.cell_contents,driver.FunctionType)
                    and c.cell_contents.__name__=='audit_origins')
                with patch.dict(private.__globals__,{'_nearest_supplement':{'files':{},'modules':{}}}),self.assertRaisesRegex(ValueError,'global binding changed'):action()
                runtime=portable.package/'version.py';raw=runtime.read_bytes();prior=runtime.stat()
                runtime.write_bytes(b'changed');os.utime(runtime,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                with self.assertRaisesRegex(ValueError,'SHA256'):action()
                runtime.write_bytes(raw);os.utime(runtime,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                manifest=Path(own['root'])/'execution.json';raw=manifest.read_bytes();manifest.write_bytes(raw+b' ')
                with self.assertRaisesRegex(ValueError,'SHA256'):action()
                manifest.write_bytes(raw);f.unchanged_originals(self)

    def test_source_order_bundle_loader_and_no_global_rebinding(self):
        source=PATH.read_text();tree=ast.parse(source);nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        text=lambda name:ast.get_source_segment(source,nodes[name])
        score=text('score_exports')
        self.assertLess(score.index('archived_replay('),score.index('quality_after_readiness('))
        self.assertLess(score.index('repeated=native.read_wires('),score.index('quality_after_readiness('))
        self.assertLess(score.index("launch['stage'] == 'full' and immediate"),score.index('paired_intervals('))
        export=text('native_export');self.assertIn('for pass_index in range(2)',export)
        self.assertIn("portable.load_inference(directory,endpoint['bundle']['sha256'],'cuda')",export)
        self.assertIn('with bundle_reads_only(context,endpoint)',export);self.assertIn('require_exact=True',export)
        self.assertNotIn('native_raw',export);self.assertNotIn('cache_rows',export)
        exit_source=text('exit_rehash');self.assertIn("api.exit_rehash(t['fit_context'])",exit_source)
        self.assertIn("require_exact=context['args'].phase == 'export'",exit_source)
        for forbidden in ('reset_peak','globals()[','if False','torch.load('):
            if forbidden != 'torch.load(':self.assertNotIn(forbidden,source)
        for node in tree.body:
            if isinstance(node,ast.Import):self.assertTrue(all(n.name.split('.')[0] not in e.NATIVE for n in node.names))
        self.assertEqual(e.TRAIN_FILES,{'train_siglip2_compact_ranking.py','test_siglip2_compact_ranking.py'})


# These stand-ins exercise extracted production functions without importing Torch.
# Reintroducing either outer no_grad must reject the saved entry flags.
INFERENCE_FLAGS = {
    'autocast_cpu':False,'autocast_cpu_dtype':'torch.bfloat16','autocast_cuda':False,
    'autocast_cuda_dtype':'torch.float16','cudnn_allow_tf32':True,'cudnn_benchmark':False,
    'cudnn_deterministic':False,'cudnn_enabled':True,'default_device':'cpu',
    'default_dtype':'torch.float32','deterministic':False,'deterministic_warn_only':False,
    'float32_matmul_precision':'highest','grad_enabled':True,'inference_mode':False,
    'interop_threads':20,'matmul_allow_tf32':False,'sdpa_cudnn':True,'sdpa_flash':True,
    'sdpa_math':True,'sdpa_mem_efficient':True,'threads':8}


def extracted_functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


class InferenceContractFixture:
    """Public guard is real; only image/Tensor/native operations are stand-ins."""
    def __init__(self):
        self.grad=True;self.flags=dict(INFERENCE_FLAGS);self.entries=[];self.operations=[]
        self.closed=[];self.fail=None;owner=self
        class Device:
            type='cuda'
            def __str__(self):return self.type
        self.device=Device()
        class Tensor:
            dtype='torch.float32';layout='torch.strided'
            def __init__(self,shape=(),value=1.,requires_grad=None):
                self.shape=shape;self.value=value;self.device=owner.device
                self.requires_grad=owner.grad if requires_grad is None else requires_grad
                self.grad_fn='stand-in autograd' if self.requires_grad else None
            def clone(self):return Tensor(self.shape,self.value,self.requires_grad)
            def detach(self):return Tensor(self.shape,self.value,False)
            def cpu(self):return self
            def to(self,*a,**k):return self
            def float(self):return self
            def norm(self,**k):return Tensor((self.shape[0],),requires_grad=self.requires_grad)
            def all(self):return self
            def item(self):return self.value
            def __gt__(self,other):return self
            def __getitem__(self,key):return Tensor(self.shape[1:],self.value,self.requires_grad)
            def __setitem__(self,key,value):
                assert not value.requires_grad and value.grad_fn is None
                self.value=value.value
            def __sub__(self,other):return Tensor(self.shape,self.value-other.value,owner.grad and (self.requires_grad or other.requires_grad))
            def __add__(self,other):
                value=other.value if isinstance(other,Tensor) else other
                return Tensor(self.shape,self.value+value,owner.grad and self.requires_grad)
            def abs(self):return Tensor(self.shape,abs(self.value),self.requires_grad)
            def argmax(self):return Tensor((),0.,False)
            def max(self):return Tensor((),abs(self.value),False)
            def __mul__(self,other):return Tensor(self.shape,self.value*other.value,owner.grad and (self.requires_grad or other.requires_grad))
            def sum(self,**kwargs):return Tensor((),self.value,self.requires_grad)
            def __float__(self):return float(self.value)
        self.Tensor=Tensor
        @contextmanager
        def no_grad():
            previous=owner.grad;owner.grad=False
            try:yield
            finally:owner.grad=previous
        @contextmanager
        def autocast(*a,**k):yield
        def operation(name,shape,requires_grad=True):
            autograd=owner.grad and requires_grad
            owner.operations.append((name,autograd))
            assert not autograd, 'native inference operation has autograd enabled'
            if owner.fail == name:raise ValueError('injected '+name)
            return Tensor(shape,requires_grad=autograd)
        def normalize(value,**k):return operation('normalize',value.shape)
        def linear(features,C):
            result=operation('residual',(features.shape[0],128));result.value=features.value*C.value
            return result
        self.functional=SimpleNamespace(normalize=normalize,linear=linear)
        self.nn=SimpleNamespace(functional=self.functional,Parameter=lambda value:value)
        self.torch=SimpleNamespace(no_grad=no_grad,autocast=autocast,nn=self.nn,
            float32='torch.float32',float16='torch.float16',
            count_nonzero=lambda value:Tensor((),int(value.value!=0),False),empty=lambda shape,**k:Tensor(shape,requires_grad=False),
            empty_like=lambda value:Tensor(value.shape,requires_grad=False),
            random=SimpleNamespace(get_rng_state=lambda:Tensor((),requires_grad=False)),
            isfinite=lambda value:Tensor((),requires_grad=False),equal=lambda a,b:a.value == b.value)
        class Image:
            size=(256,256)
            def __enter__(self):return self
            def __exit__(self,*a):self.close()
            def convert(self,mode):assert mode == 'RGB';return Image()
            def tobytes(self):return b'RGB'
            def close(self):owner.closed.append(self)
        self.pil=SimpleNamespace(Image=SimpleNamespace(open=lambda path:Image()))
        def flags():
            current={**owner.flags,'grad_enabled':owner.grad}
            owner.entries.append(current)
            return current
        def model(*,pixel_values):return SimpleNamespace(pooler_output=operation('vision',(pixel_values.shape[0],1152)))
        model.modules=lambda:[SimpleNamespace(training=False,_forward_hooks={},_forward_pre_hooks={},_backward_hooks={})]
        def raw_features(features,*a):return operation('readout',(features.shape[0],128))
        def pack(value):
            result=operation('pack',value.shape,value.requires_grad)
            return SimpleNamespace(codes=result,inverse_norms=Tensor((value.shape[0],),requires_grad=result.requires_grad),to_bytes=lambda:b'wire')
        self.state={'device':'cuda','flags':dict(INFERENCE_FLAGS),'model':model,
            'processor_object':lambda *,images,return_tensors:{'pixel_values':Tensor((len(images),3,256,256),requires_grad=False)},
            'head_object':SimpleNamespace(state_dict=lambda:{'weight':b'head'}),
            'A':Tensor((128,160),requires_grad=True),'means':{},'C':Tensor((128,1152),.5,True),
            'mu_train':Tensor((1152,),.25,False),'mu_train_provenance':{'rows':6355},'arm':'candidate',
            'modules':{'qualify_siglip2_substrate_cpu.py':SimpleNamespace(numerical_flags=flags),
                'prototype_residual_readout.py':SimpleNamespace(raw_features=raw_features),
                'quadratic_readout.py':SimpleNamespace(),
                'joint_relational_compaction.py':SimpleNamespace(pack_int8_unit_embeddings=pack)}}
        extracted_functions(PATH.with_name('quadratic_readout.py'),{'_check_tensor'},
            namespace:={'_require':e.require})
        self.state['modules']['quadratic_readout.py']._check_tensor=namespace['_check_tensor']
        trainer_path=(TRAINER_ROOT/'train_siglip2_compact_ranking.py')
        inference=next(n for n in ast.parse(trainer_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='inference_outputs')
        assert hashlib.sha256(ast.dump(inference,include_attributes=False).encode()).hexdigest() == 'e02047621095dd855ef0e54bb78d84b346e0770bae1dae881561550467b6ca26'
        public={'require':e.require,'ARMS':e.ARMS}
        extracted_functions(trainer_path,{'inference_outputs','parameter_roles','fullfeature_raw_features',
            'inference_readout_tree'},public)
        self.public=public['inference_outputs']
        @contextmanager
        def bundle_reads_only(context,endpoint):yield
        def fingerprint(context,value):
            def plain(v):
                if isinstance(v,Tensor):return {'shape':v.shape,'value':v.value,'dtype':v.dtype}
                if isinstance(v,dict):return {k:plain(child) for k,child in v.items()}
                if isinstance(v,bytes):return v.hex()
                return v
            return hashlib.sha256(json.dumps(plain(value),sort_keys=True).encode()).hexdigest()
        self.state['modules']['train_siglip2_substrate_adaptation.py']=SimpleNamespace(fingerprint=lambda v:fingerprint(None,v))
        self.state['readout_sha256']=fingerprint(None,public['inference_readout_tree'](self.state))
        training={'legacy':{'packing':SimpleNamespace(pack_int8_unit_embeddings=pack),'quadratic':object()}}
        self.context={'guards':{},'training_context':training,
            'trainer':SimpleNamespace(fingerprint=fingerprint,helper_guard=lambda _:SimpleNamespace(raw_features=raw_features)),
            'portable_entry':(SimpleNamespace(inference_outputs=self.public),object()),
            'reference':SimpleNamespace(json_digest=lambda _: 'triples'),
            'helper':SimpleNamespace(exact=self.exact)}
        namespace={'require':e.require,'check_residual_oracle':e.check_residual_oracle,'ARMS':e.ARMS,
            'hashlib':hashlib,'math':__import__('math'),
            'json':json,'time':time,'UNIT_STARTED':e.UNIT_STARTED,
            'bound_file':lambda guards,path,digest:path,'bundle_reads_only':bundle_reads_only,
            'train_rows':lambda context,ids:self.rows(ids),'batch_sizes':e.batch_sizes,
            'tuple_outputs':e.tuple_outputs,
            'packed_outputs':lambda context,raw:{'raw':raw,'unit':normalize(raw,dim=1),
                'codes':Tensor(raw.shape),'inverse_norms':Tensor((raw.shape[0],)),'wire':b'wire'}}
        extracted_functions(PATH,{'images_outputs','train_diagnostic','export_pass','fullfeature_oracle'},namespace)
        self.functions=namespace

    def exact(self,left,right):
        assert len(left) == len(right) == 4
        for a,b in zip(left,right):assert (a.shape,a.value,a.requires_grad) == (b.shape,b.value,b.requires_grad)

    def rows(self,ids):return [{'path':'/stand-in/'+str(i),'image_sha256':'a'*64} for i in ids]

    @contextmanager
    def imports(self):
        with patch.dict(sys.modules,{'torch':self.torch,'torch.nn':self.nn,
                'torch.nn.functional':self.functional,'PIL':self.pil}):yield

    def call(self,mode):
        with self.imports():
            if mode == 'TRAINmicro16':
                values,fact=self.functions['images_outputs'](self.context,self.state,self.rows(range(16)),oracle=True)
                assert len(fact['rows']) == 16
                for value in values.values():
                    if isinstance(value,self.Tensor):assert not value.requires_grad and value.grad_fn is None
                return values
            if mode == 'diagnostic':
                result=self.functions['train_diagnostic'](self.context,self.state,[[0,i,i+1] for i in range(1,17)])
                assert result['margins'] == [0.]*16 and result['diagnostic_only'] and not result['utility_veto']
                return result
            values,facts,sizes=self.functions['export_pass'](self.context,self.state,self.rows(range(67)),
                {'query':list(range(33)),'gallery':list(range(33,67))})
            assert sizes == {'query':[32,1],'gallery':[32,2]}
            assert [(f['role'],len(f['rows'])) for f in facts] == [('query',32),('query',1),('gallery',32),('gallery',2)]
            for value in values:assert not value.requires_grad and value.grad_fn is None
            return values


class InferenceBoundaryTests(unittest.TestCase):
    def test_saved_flags_at_public_entry_and_internal_no_autograd(self):
        for mode,calls in (('TRAINmicro16',1),('diagnostic',2),('export',4)):
            with self.subTest(mode=mode):
                f=InferenceContractFixture();f.call(mode)
                self.assertEqual(f.entries,[INFERENCE_FLAGS]*calls)
                self.assertTrue(f.operations)
                self.assertTrue(all(not grad for _,grad in f.operations))
                self.assertEqual(f.grad,True)
                self.assertEqual(len(f.closed),{'TRAINmicro16':32,'diagnostic':36,'export':134}[mode])

    def test_complete_flag_mutations_and_disabled_caller_are_rejected(self):
        for mode in ('TRAINmicro16','diagnostic','export'):
            for key,value in INFERENCE_FLAGS.items():
                with self.subTest(mode=mode,flag=key):
                    f=InferenceContractFixture()
                    if key == 'grad_enabled':f.grad=False
                    else:f.flags[key]=not value if type(value) is bool else value+1 if type(value) is int else value+' changed'
                    with self.assertRaisesRegex(ValueError,'inference numerical flags changed'):f.call(mode)
                    self.assertEqual(f.grad,key != 'grad_enabled')
                    self.assertEqual(f.operations,[])
                    self.assertTrue(f.closed)

    def test_native_exceptions_restore_entry_mode_and_close_images(self):
        for mode in ('TRAINmicro16','diagnostic','export'):
            for stage in ('vision','readout','pack'):
                with self.subTest(mode=mode,stage=stage):
                    f=InferenceContractFixture();f.fail=stage
                    with self.assertRaisesRegex(ValueError,'injected '+stage):f.call(mode)
                    self.assertTrue(f.grad)
                    self.assertTrue(f.closed)
                    self.assertEqual(f.entries,[INFERENCE_FLAGS])
                    self.assertTrue(all(not grad for _,grad in f.operations))

    def test_exact_production_ast_inverse_of_two_outer_contexts(self):
        tree=inverse_runtime_hash_batch(ast.parse(PATH.read_text()))
        for name in ('train_diagnostic','export_pass'):
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name)
            index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.For))
            scope=ast.parse('with torch.no_grad():\n    pass').body[0]
            scope.body=[node.body[index]];node.body[index]=scope
        digest=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
        self.assertEqual(digest,'9a102dd32044ca1d6dbf2c45f69ed578c2a22237d1477b90ee869e186a6b21ea')


RUNTIME_HASH_BATCH_SOURCE = '''\
items=list(runtime.items())
hash_started=time.perf_counter()
print(json.dumps({'event':'COMPACT_RUNTIME_HASH','boundary':'begin',
    'elapsed_seconds':0.0,'item_count':len(items)}),flush=True)
try:
    for path,digest in items:
        if path.name == 'RECORD' or path.suffix == '.so':
            require(context['required_guards'].get(str(path)) == digest, 'runtime original FILE authority changed')
        if path.suffix == '.so':
            require(str(path) in original['native_files'] and original['files'].get(str(path)) == digest,
                'runtime original native origin changed')
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(bound_file,{},path,digest) for path,digest in items]
        paths=[future.result() for future in futures]
    staged=dict(context['guards'])
    for path,(_,digest) in zip(paths,items):
        require(staged.setdefault(str(path),digest) == digest, 'conflicting FILE authority')
    context['guards'].update(staged)
finally:
    print(json.dumps({'event':'COMPACT_RUNTIME_HASH','boundary':'end',
        'elapsed_seconds':time.perf_counter()-hash_started,'item_count':len(items)}),flush=True)
'''


def inverse_runtime_hash_batch(tree):
    """Undo only the exact authorized loop replacement, including its markers."""
    tree=inverse_export_audit_markers(tree)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bundle_reads_only')
    start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and
        ast.unparse(n.targets[0])=='items')
    expected=ast.parse(RUNTIME_HASH_BATCH_SOURCE).body
    actual=node.body[start:start+len(expected)]
    assert ast.dump(ast.Module(body=actual,type_ignores=[]),include_attributes=False)==ast.dump(
        ast.Module(body=expected,type_ignores=[]),include_attributes=False), 'runtime hash replacement differs'
    original=copy.deepcopy(expected[3].body[0]);original.iter=ast.parse('runtime.items()',mode='eval').body
    original.body.append(ast.parse("bound_file(context['guards'],path,digest)").body[0])
    node.body[start:start+len(expected)]=[original]
    return tree


def extracted_runtime_hash_entry():
    """Execute the real loop and unchanged evaluator reader on stdlib files."""
    tree=ast.parse(PATH.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bundle_reads_only')
    start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and
        ast.unparse(n.targets[0])=='original')
    stop=next(i for i,n in enumerate(node.body) if isinstance(n,ast.For) and
        ast.unparse(n.iter)=='tuple(sys.modules.items())')
    entry=ast.parse('def hash_entry(context,runtime):\n    pass').body[0]
    entry.body=copy.deepcopy(node.body[start:stop])
    namespace={'Path':Path,'hashlib':hashlib,'os':os,'re':__import__('re'),'time':time,'json':json}
    extracted_functions(PATH,('require','sha','bound_file'),namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[entry],type_ignores=[])),str(PATH),'exec'),namespace)
    return namespace


class RuntimeHashBatchTests(unittest.TestCase):
    def files(self,root,names):
        items=[]
        for name in names:
            path=root/name;path.write_bytes(('pinned '+name).encode())
            items.append((path,hashlib.sha256(path.read_bytes()).hexdigest()))
        return items

    def context(self,items,guards=None):
        natives={str(p):h for p,h in items if p.suffix=='.so'}
        return {'guards':{} if guards is None else guards,
            'required_guards':{str(p):h for p,h in items if p.name=='RECORD' or p.suffix=='.so'},
            'training_context':{'legacy':{'selected':{'source_cpu':{
                'origins':{'native_files':list(natives),'files':natives}}}}}}

    def run_entry(self,namespace,context,items):
        output=io.StringIO()
        with redirect_stdout(output):namespace['hash_entry'](context,SimpleNamespace(items=lambda:list(items)))
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_four_workers_read_every_occurrence_and_publish_in_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            items=self.files(Path(directory),['file'+str(i)+'.py' for i in range(9)])
            items.insert(2,items[0]);items.append(items[3])
            owner=threading.get_ident();updates=[]
            class Guards(dict):
                def update(self,values):updates.append(threading.get_ident());super().update(values)
            guards=Guards({'/existing':'a'*64});context=self.context(items,guards);before=dict(guards)
            namespace=extracted_runtime_hash_entry();reader=namespace['bound_file']
            lock=threading.Lock();barrier=threading.Barrier(4);calls=[];active=0;maximum=0;workers=[]
            def read(local,path,digest):
                nonlocal active,maximum
                with lock:
                    calls.append((path,dict(local),local is guards,threading.get_ident(),dict(guards)))
                    number=len(calls);active+=1;maximum=max(maximum,active)
                try:
                    if number<=4:
                        try:barrier.wait(timeout=.5)
                        except threading.BrokenBarrierError:pass
                    return reader(local,path,digest)
                finally:
                    with lock:active-=1
            class Executor(ThreadPoolExecutor):
                def __init__(self,*args,**kwargs):workers.append(kwargs['max_workers']);super().__init__(*args,**kwargs)
            namespace['bound_file']=read
            with patch('concurrent.futures.ThreadPoolExecutor',Executor):self.run_entry(namespace,context,items)
            self.assertEqual(workers,[4]);self.assertEqual(maximum,4);self.assertEqual(active,0)
            self.assertCountEqual([p for p,*_ in calls],[p for p,_ in items])
            self.assertTrue(all(local=={} and not shared and thread!=owner and snapshot==before
                for _,local,shared,thread,snapshot in calls))
            self.assertEqual(updates,[owner])
            self.assertEqual(list(guards),['/existing']+list(dict.fromkeys(str(p) for p,_ in items)))

    def test_first_error_joins_all_later_reads_before_return(self):
        with tempfile.TemporaryDirectory() as directory:
            items=self.files(Path(directory),['file'+str(i)+'.py' for i in range(9)])
            items[0]=(items[0][0],'0'*64)
            context=self.context(items,{'/existing':'a'*64});before=dict(context['guards'])
            namespace=extracted_runtime_hash_entry();reader=namespace['bound_file']
            lock=threading.Lock();failed=threading.Event();finished=[];threads=[]
            def read(local,path,digest):
                with lock:threads.append(threading.current_thread())
                try:
                    if path==items[1][0]:self.assertTrue(failed.wait(timeout=1))
                    return reader(local,path,digest)
                except ValueError:
                    failed.set();raise
                finally:
                    with lock:finished.append(path)
            namespace['bound_file']=read
            with self.assertRaisesRegex(ValueError,'file SHA256 differs'):
                self.run_entry(namespace,context,items)
            self.assertCountEqual(finished,[p for p,_ in items])
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(context['guards'],before)

    def test_later_file_error_does_not_publish_successful_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            items=self.files(Path(directory),['first.py','bad.py','last.py']);items[1]=(items[1][0],'0'*64)
            context=self.context(items,{'/existing':'a'*64});before=dict(context['guards'])
            with self.assertRaisesRegex(ValueError,'file SHA256 differs'):
                self.run_entry(extracted_runtime_hash_entry(),context,items)
            self.assertEqual(context['guards'],before)

    def test_guard_conflict_is_staged_after_all_successful_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            items=self.files(Path(directory),['first.py','last.py'])
            context=self.context(items,{str(items[-1][0]):'0'*64});before=dict(context['guards'])
            namespace=extracted_runtime_hash_entry();reader=namespace['bound_file'];finished=[]
            def read(local,path,digest):
                result=reader(local,path,digest);finished.append(path);return result
            namespace['bound_file']=read
            with self.assertRaisesRegex(ValueError,'conflicting FILE authority'):
                self.run_entry(namespace,context,items)
            self.assertCountEqual(finished,[p for p,_ in items]);self.assertEqual(context['guards'],before)

    def test_original_metadata_predicates_run_on_owner_before_any_read(self):
        with tempfile.TemporaryDirectory() as directory:
            items=self.files(Path(directory),['first.py','RECORD','native.so'])
            for case in ('record','native_guard','native_origin'):
                with self.subTest(case=case):
                    context=self.context(items);calls=[];owner=threading.get_ident();checks=[]
                    namespace=extracted_runtime_hash_entry();require=namespace['require']
                    def check(condition,message):
                        checks.append(threading.get_ident());return require(condition,message)
                    namespace['require']=check
                    namespace['bound_file']=lambda *args:calls.append(args)
                    if case=='record':context['required_guards'].pop(str(items[1][0]))
                    elif case=='native_guard':context['required_guards'].pop(str(items[2][0]))
                    else:context['training_context']['legacy']['selected']['source_cpu']['origins']['native_files']=[]
                    with self.assertRaisesRegex(ValueError,'runtime original'):
                        self.run_entry(namespace,context,items)
                    self.assertEqual(calls,[]);self.assertEqual(context['guards'],{})
                    self.assertTrue(checks and all(thread==owner for thread in checks))

    def test_every_entry_reads_current_bytes_even_with_restored_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            items=self.files(Path(directory),['source.py','RECORD','native.so'])
            namespace=extracted_runtime_hash_entry();context=self.context(items)
            self.run_entry(namespace,context,items);before=dict(context['guards'])
            for path,_ in items:
                with self.subTest(path=path.name):
                    raw=path.read_bytes();prior=path.stat();path.write_bytes(b'X'+raw[1:])
                    os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    with self.assertRaisesRegex(ValueError,'file SHA256 differs'):
                        self.run_entry(namespace,context,items)
                    self.assertEqual(context['guards'],before)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    self.run_entry(namespace,context,items)

    def test_bounded_timing_markers_cover_success_empty_and_error(self):
        with tempfile.TemporaryDirectory() as directory:
            good=self.files(Path(directory),['source.py'])
            for case in ('success','empty','hash_error','conflict','metadata_error'):
                with self.subTest(case=case):
                    items=[] if case=='empty' else list(good)
                    if case=='hash_error':items[0]=(items[0][0],'0'*64)
                    if case=='metadata_error':items=self.files(Path(directory),['RECORD'])
                    context=self.context(items)
                    if case=='conflict':context['guards'][str(items[0][0])]='0'*64
                    if case=='metadata_error':context['required_guards'].clear()
                    output=io.StringIO()
                    with redirect_stdout(output):
                        if case.endswith('error') or case=='conflict':
                            with self.assertRaises(ValueError):
                                extracted_runtime_hash_entry()['hash_entry'](context,SimpleNamespace(items=lambda:items))
                        else:extracted_runtime_hash_entry()['hash_entry'](context,SimpleNamespace(items=lambda:items))
                    markers=[json.loads(line) for line in output.getvalue().splitlines()]
                    self.assertEqual(len(markers),2)
                    self.assertEqual([m['boundary'] for m in markers],['begin','end'])
                    for marker in markers:
                        self.assertEqual(set(marker),{'event','boundary','elapsed_seconds','item_count'})
                        self.assertEqual(marker['event'],'COMPACT_RUNTIME_HASH')
                        self.assertEqual(marker['item_count'],len(items));self.assertGreaterEqual(marker['elapsed_seconds'],0)
                    self.assertEqual(markers[0]['elapsed_seconds'],0.0)

    def test_uncached_and_cached_entries_keep_boundary_and_fresh_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=PortableRuntimeFixture(Path(directory));owner=threading.get_ident();reads=[]
            reader=e.bound_file
            def read(guards,path,digest):
                if threading.get_ident()!=owner:reads.append((path,dict(guards)))
                return reader(guards,path,digest)
            with patch.object(e,'bound_file',read),patch.object(fixture.context['trainer'],'admit_bundle',
                    wraps=fixture.context['trainer'].admit_bundle) as admit,redirect_stdout(io.StringIO()):
                for entry in range(2):
                    reads.clear()
                    with fixture.boundary():
                        cached=next(iter(fixture.context['portable_audits'].values()))
                        self.assertTrue(cached[0][0]);self.assertTrue(sys.dont_write_bytecode)
                        with self.assertRaisesRegex(ValueError,'nested serving'):
                            with fixture.boundary():pass
                    self.assertFalse(cached[0][0]);self.assertEqual(admit.call_count,1)
                    self.assertCountEqual([p for p,_ in reads],list(cached[1]))
                    self.assertTrue(all(local=={} for _,local in reads))
                path=fixture.package/'version.py';raw=path.read_bytes();prior=path.stat()
                path.write_bytes(b'X'+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                with self.assertRaisesRegex(ValueError,'file SHA256 differs'):
                    with fixture.boundary():self.fail('mutated cached source entered serving scope')
                self.assertFalse(cached[0][0])

    def test_exact_production_ast_inverse_of_runtime_hash_loop(self):
        tree=inverse_runtime_hash_batch(ast.parse(PATH.read_text()))
        digest=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
        self.assertEqual(digest,'a35111879bd1d2a65f72860f3fdddc9db0798185aa5d72df2c22e01ba422b948')


# Exact timing statements are the only removable nodes in this inverse.
EXPORT_TIMING_SPECS = (
    ('native_export', "portable_name='_compact_export_entry_'+str(pass_index)+'_'+key.replace('-','_')", 'after', 'loader', 'begin'),
    ('native_export', "context['portable_entry']=(portable,endpoint)", 'before', 'loader', 'end'),
    ('native_export', 'model_facts=endpoint_facts(context,state)', 'before', 'endpoint_facts', 'begin'),
    ('native_export', 'model_facts=endpoint_facts(context,state)', 'after', 'endpoint_facts', 'end'),
    ('native_export', 'witness=native_train_witness(context,state,args.seed)', 'before', 'witness', 'begin'),
    ('native_export', 'witness=native_train_witness(context,state,args.seed)', 'after', 'witness', 'end'),
    ('native_export', 'diagnostic=train_diagnostic(context,state,triples)', 'before', 'diagnostic', 'begin'),
    ('native_export', 'diagnostic=train_diagnostic(context,state,triples)', 'after', 'diagnostic', 'end'),
    ('native_export', 'current,current_images,current_sizes=export_pass(context,state,rows,mapping)', 'before', 'export_pass', 'begin'),
    ('native_export', 'current,current_images,current_sizes=export_pass(context,state,rows,mapping)', 'after', 'export_pass', 'end'),
    ('native_export', "require(endpoint_facts(context,state) == model_facts, 'native forward mutated complete endpoint')", 'before', 'postpass_endpoint_facts', 'begin'),
    ('native_export', "require(endpoint_facts(context,state) == model_facts, 'native forward mutated complete endpoint')", 'after', 'postpass_endpoint_facts', 'end'),
    ('native_export', "t['nearest'].native_source_api(t).audit_origins(t['legacy'],require_exact=True)", 'before', 'postpass_audit', 'begin'),
    ('native_export', "t['nearest'].native_source_api(t).audit_origins(t['legacy'],require_exact=True)", 'after', 'postpass_audit', 'end'),
    ('export_pass', 'values,fact=images_outputs(context,state,[rows[i] for i in batch],oracle=start == 0)', 'before', 'images_outputs', 'begin'),
    ('export_pass', 'values,fact=images_outputs(context,state,[rows[i] for i in batch],oracle=start == 0)', 'after', 'images_outputs', 'end'),
    ('run', 'origins=exit_rehash(context)', 'before', 'exit_rehash', 'begin'),
    ('run', 'origins=exit_rehash(context)', 'after', 'exit_rehash', 'end'),
)
OLD_EXPORT_AUDIT = "t['nearest'].native_source_api(t).audit_origins(t['legacy'],require_exact=True)"
NEW_EXPORT_AUDIT = OLD_EXPORT_AUDIT[:-1]+",admission=t['legacy']['original'].FlatAdmission())"


def export_timing_source(function,stage,boundary):
    fields = "'pass_index':pass_index" if function=='native_export' else (
        "'role':role,'batch_start':start,'batch_size':len(batch)" if function=='export_pass' else "'phase':args.phase")
    return "print(json.dumps({'event':'COMPACT_TIMING','stage':%r,'boundary':%r,%s,'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)" % (stage,boundary,fields)


def inverse_export_envelope_source(source):
    """Normalize current authority, then undo only the 900-second export literal."""
    source=inverse_current_gallery_source(source)
    current="'seconds': 900 if phase == 'export' else 500"
    assert source.count(current)==1, 'prospective 900-second export policy differs'
    return source.replace(current,"'seconds': 600 if phase == 'export' else 500",1)


def inverse_export_envelope(tree):
    """Normalize current authority and the policy before historical AST inverses."""
    tree=inverse_current_gallery_authority(tree)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='policy']
    expected=ast.parse("""def policy(phase):
    require(phase in ('cpu','export','score'), 'fixed evaluation phase required')
    return {'seconds': 900 if phase == 'export' else 500, 'host_bytes':8*1024**3,
            'swap_bytes':0, 'cuda_visible_devices':'0' if phase == 'export' else ''}
""").body[0]
    assert len(nodes)==1 and ast.dump(nodes[0],include_attributes=False)==ast.dump(
        expected,include_attributes=False), 'prospective 900-second export policy differs'
    nodes[0].body[1].value.values[0].body.value=600
    return tree


def inverse_export_seconds(tree):
    """Restore only the prospective export-seconds literal to its original 300."""
    tree=inverse_export_envelope(tree)
    tree=inverse_smooth_ap_authority(tree)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='policy']
    expected=ast.parse("""def policy(phase):
    require(phase in ('cpu','export','score'), 'fixed evaluation phase required')
    return {'seconds': 600 if phase == 'export' else 500, 'host_bytes':8*1024**3,
            'swap_bytes':0, 'cuda_visible_devices':'0' if phase == 'export' else ''}
""").body[0]
    assert len(nodes)==1 and ast.dump(nodes[0],include_attributes=False)==ast.dump(
        expected,include_attributes=False), 'prospective export policy differs'
    nodes[0].body[1].value.values[0].body.value=300
    return tree


def inverse_export_audit_markers(tree):
    """Undo the single exact admission keyword and 18 explicitly placed prints."""
    tree=inverse_export_seconds(tree)
    dump=lambda node:ast.dump(node,include_attributes=False)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    audit=ast.parse(NEW_EXPORT_AUDIT).body[0]
    matches=[n for n in ast.walk(functions['native_export']) if dump(n)==dump(audit)]
    assert len(matches)==1, 'fresh original export admission differs'
    matches[0].value.keywords.pop()  # Exact expression comparison proved the sole new keyword.
    for function,anchor,side,stage,boundary in EXPORT_TIMING_SPECS:
        expected=ast.parse(anchor).body[0]; locations=[]
        for parent in ast.walk(functions[function]):
            for _,values in ast.iter_fields(parent):
                if isinstance(values,list):
                    locations.extend((values,i) for i,n in enumerate(values) if isinstance(n,ast.stmt) and dump(n)==dump(expected))
        assert len(locations)==1, 'timing anchor differs: '+stage
        values,index=locations[0]; index+=1 if side=='after' else -1
        marker=ast.parse(export_timing_source(function,stage,boundary)).body[0]
        assert index>=0 and dump(values[index])==dump(marker), 'timing statement differs: '+stage+'/'+boundary
        del values[index]
    return tree


class ExportOriginReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_module=module('_export_origin_fixture_tests',(TRAINER_ROOT/'test_siglip2_compact_ranking.py'))
        cls.fixture_type=fixture_module.FreshOriginAuditTests
        # Trainer2 may be staged separately; the retained native fixture stays in this source tree.
        cls.fixture_type.fixtures=module('compact_origin_fixtures',PATH.with_name('test_siglip2_nearest_ranking.py'))

    def setUp(self):
        # The existing exit test leaves its helper registered; restore it after each fixture.
        modules=patch.dict(sys.modules);modules.start();self.addCleanup(modules.stop)
        sys.modules.pop('_prototype_signed_readout',None)

    def fixture(self):
        return self.fixture_type()

    def audit_call(self):
        fn=next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='native_export')
        calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='audit_origins']
        self.assertEqual(len(calls),1)
        return compile(ast.Expression(body=calls[0]),str(PATH),'eval')

    def invoke(self,f):
        return eval(self.audit_call(),{}, {'t':f.context})

    def test_two_new_readers_preserve_first_sweep_and_identical_guards(self):
        with self.fixture().composition() as f:
            f.api.audit_origins(f.legacy,require_exact=True)
            self.assertCountEqual(f.reads,[(kind,p) for p in f.observed for kind in ('origin_sha','duplicate_sha')])
            expected=(dict(f.legacy['guards']),dict(f.legacy['prior']['guards']),copy.deepcopy(f.legacy['origins']))
            f.admission.verified.update(f.observed)
            for pass_index in range(2):
                with self.subTest(pass_index=pass_index):
                    f.reads.clear();self.invoke(f);reader=f.readers[-1]
                    self.assertIs(type(reader),f.original.FlatAdmission)
                    self.assertIsNot(reader,f.admission)
                    self.assertTrue(all(reader is not old for old in f.readers[:-1]))
                    self.assertEqual(reader.verified,set(f.observed))
                    self.assertEqual(reader.entries,{p:(h,Path(p).stat().st_size) for p,h in f.observed.items()})
                    self.assertEqual(reader.json_bytes,{})
                    self.assertCountEqual(f.reads,[('origin_sha',p) for p in f.observed])
                    self.assertEqual((dict(f.legacy['guards']),dict(f.legacy['prior']['guards']),f.legacy['origins']),expected)
            self.assertEqual(len(f.readers),2)

    def test_unknown_map_missing_fourth_and_restored_mtime_reject(self):
        for case,error in (('unknown_map','unknown or changed'),('missing_fourth','exact four'),('restored_mtime','unknown or changed')):
            with self.subTest(case=case),self.fixture().composition() as f:
                self.invoke(f)
                if case=='unknown_map':
                    unknown=f.root/'unknown.so';unknown.write_bytes(b'unknown');f.mapped.append(str(unknown))
                elif case=='missing_fourth':f.mapped.remove(next(iter(f.files)))
                else:
                    path=Path(next(iter(f.cpu['files'])));raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    self.assertEqual(path.stat().st_size,len(raw));self.assertEqual(path.stat().st_mtime_ns,prior.st_mtime_ns)
                with self.assertRaisesRegex(ValueError,error):self.invoke(f)
                self.assertIsNot(f.readers[0],f.readers[1])

    def test_original_predicates_and_guard_correspondence(self):
        self.fixture().test_private_audit_predicates_and_guard_correspondence()

    def test_exact_whole_production_ast_inverse(self):
        tree=inverse_export_audit_markers(ast.parse(PATH.read_text()))
        self.assertEqual(hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
            'f499f425a1d9a91333046d61ac2adaa5844680799965f49b4fbe1b5468498799')

    def test_timing_prints_are_flushed_cumulative_and_identify_boundaries(self):
        source=PATH.read_text();tree=ast.parse(source);dump=lambda n:ast.dump(n,include_attributes=False)
        for function,anchor,side,stage,boundary in EXPORT_TIMING_SPECS:
            node=ast.parse(export_timing_source(function,stage,boundary)).body[0]
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
            self.assertEqual(sum(dump(n)==dump(node) for n in ast.walk(fn)),1)
            flushes=[]
            class Output(io.StringIO):
                def flush(self):flushes.append(True)
            output=Output()
            with redirect_stdout(output):
                exec(compile(ast.Module(body=[node],type_ignores=[]),str(PATH),'exec'),
                    {'json':json,'time':SimpleNamespace(perf_counter=lambda:123.5),'UNIT_STARTED':100.,
                     'pass_index':1,'role':'gallery','start':32,'batch':[1,2],'args':SimpleNamespace(phase='export')})
            marker=json.loads(output.getvalue());self.assertEqual(marker['seconds'],23.5)
            self.assertEqual((marker['stage'],marker['boundary']),(stage,boundary));self.assertEqual(flushes,[True])
            if function=='native_export':self.assertEqual(marker['pass_index'],1)
            elif function=='export_pass':self.assertEqual((marker['role'],marker['batch_start'],marker['batch_size']),('gallery',32,2))
            else:self.assertEqual(marker['phase'],'export')
        output=io.StringIO()
        with redirect_stdout(output):InferenceContractFixture().call('export')
        markers=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([(m['role'],m['batch_start'],m['batch_size'],m['boundary']) for m in markers],
            [(role,start,size,boundary) for role,start,size in
             [('query',0,32),('query',32,1),('gallery',0,32),('gallery',32,2)] for boundary in ('begin','end')])
        self.assertEqual([m['seconds'] for m in markers],sorted(m['seconds'] for m in markers))


class ProspectiveExportEnvelopeTests(unittest.TestCase):
    def test_exact_900_to_600_inverse_restores_complete_source_and_ast(self):
        source=PATH.read_text()
        restored=inverse_export_envelope_source(source)
        self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(),
            '040659957c3f79dfea9ec42c6626a9909193e7cc032386a157db1dacd5ed2273')
        tree=inverse_export_envelope(ast.parse(source))
        self.assertEqual(ast.dump(tree,include_attributes=False),
            ast.dump(ast.parse(restored),include_attributes=False))
        self.assertEqual(hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
            '2c6725aa5763b221e70d4fbd4317a470044ec0f20062e42ae1a38b0c59f412c7')

    def test_export_accepts_extended_interval_and_rejects_900_boundary(self):
        record={'resource_policy':{'seconds':900,'host_bytes':8*1024**3,
            'swap_bytes':0,'cuda_visible_devices':'0'},'wall_seconds':600.198,
            'process_peak_rss_kib':8*1024**2,'peak_cuda_allocated_bytes':9_999_999_999,
            'cuda_initialized':True}
        for wall in (600,600.198,700,899.999):
            with self.subTest(accepted_export_seconds=wall):
                e.check_resource_facts({**record,'wall_seconds':wall},'export')
        for wall in (900,900.001,1200):
            with self.subTest(rejected_export_seconds=wall),self.assertRaises(ValueError):
                e.check_resource_facts({**record,'wall_seconds':wall},'export')
        for key,value in (('process_peak_rss_kib',8*1024**2+1),('peak_cuda_allocated_bytes',10_000_000_000)):
            with self.subTest(resource=key),self.assertRaises(ValueError):
                e.check_resource_facts({**record,key:value},'export')
        for phase in ('cpu','score'):
            cpu={**record,'resource_policy':{'seconds':500,'host_bytes':8*1024**3,
                'swap_bytes':0,'cuda_visible_devices':''},'wall_seconds':499.999,
                'peak_cuda_allocated_bytes':0,'cuda_initialized':False}
            e.check_resource_facts(cpu,phase)
            for wall in (500,600.198,899.999):
                with self.subTest(phase=phase,rejected_seconds=wall),self.assertRaises(ValueError):
                    e.check_resource_facts({**cpu,'wall_seconds':wall},phase)

    def test_launch_rejects_historical_and_further_increased_export_caps(self):
        for phase in ('cpu','export','score'):
            value,args=launch(phase=phase)
            value['resource_policies']={
                'cpu':{'seconds':500,'host_bytes':8*1024**3,'swap_bytes':0,'cuda_visible_devices':''},
                'export':{'seconds':900,'host_bytes':8*1024**3,'swap_bytes':0,'cuda_visible_devices':'0'},
                'score':{'seconds':500,'host_bytes':8*1024**3,'swap_bytes':0,'cuda_visible_devices':''}}
            e.check_launch(value,args)
            for cap in (600,899,901,1200):
                changed=copy.deepcopy(value);changed['resource_policies']['export']['seconds']=cap
                with self.subTest(phase=phase,export_cap=cap),self.assertRaises(ValueError):
                    e.check_launch(changed,args)
            for capped_phase in ('cpu','score'):
                changed=copy.deepcopy(value);changed['resource_policies'][capped_phase]['seconds']=900
                with self.subTest(phase=phase,capped_phase=capped_phase),self.assertRaises(ValueError):
                    e.check_launch(changed,args)

    def test_cpu_source_admission_rejects_historical_code_and_execution(self):
        source=PATH.read_text();old_source=inverse_export_envelope_source(source)
        code={PATH.name:hashlib.sha256(source.encode()).hexdigest(),
            Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        old_code={PATH.name:hashlib.sha256(old_source.encode()).hexdigest(),
            Path(__file__).name:'6a7a2ec149196fcf4cf5db7ad304991af3286fe646eee78b1102e991abe31545'}
        digest=lambda value:hashlib.sha256((json.dumps(value,sort_keys=True)+'\n').encode()).hexdigest()
        current_execution=digest(code);old_execution=digest(old_code)
        self.assertEqual(old_execution,'0fb358fcf8083ade5b1f0a27072520601b8a90327a5b94d454c9e86bc01d13c4')
        self.assertNotEqual(code,old_code);self.assertNotEqual(current_execution,old_execution)
        context={'args':SimpleNamespace(execution_sha256=current_execution),'code':code,
            'launch':{'stage':'first','panel':'selection'},'training_context':{'source':{'actual':'source'}}}
        record={'schema':e.SCHEMA,'phase':'cpu','arm':None,'seed':None,'stage':'first','panel':'selection',
            'execution_sha256':current_execution,'source_code':code,'source':context['training_context']['source'],
            'cost_policy':e.COST_POLICY}
        for key in ('pass','engineering_admission_pass','integrity_pass','resources_pass','exit_rehash_pass',
            'sequential_model_ownership','rng_flags_preserved','both_locks_held_in_parent_authority'):record[key]=True
        for key in ('official_read','global_production_goal_met','public_latency_measured','product_go'):record[key]=False
        function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='check_receipt')
        guards=[n for n in function.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
            isinstance(n.value.func,ast.Name) and n.value.func.id=='require' and n.value.args[-1].value==
            'complete source/resource evaluator receipt required']
        self.assertEqual(len(guards),1)
        predicate=compile(ast.Module(body=guards,type_ignores=[]),str(PATH),'exec')
        def admit(value):
            exec(predicate,{**vars(e),'context':context,'record':value,'phase':'cpu','arm':None,
                'seed':None,'stage':'first','panel':'selection'})
        admit(record)
        for changes in ({'source_code':old_code},{'execution_sha256':old_execution},
                {'source_code':old_code,'execution_sha256':old_execution}):
            with self.subTest(historical_fields=tuple(changes)),self.assertRaisesRegex(ValueError,'complete source/resource'):
                admit({**record,**changes})

    def test_whole_unit_deadlines_and_unchanged_resource_predicates(self):
        for phase,seconds,cuda in (('cpu',500,''),('export',900,'0'),('score',500,'')):
            record={'resource_policy':{'seconds':seconds,'host_bytes':8*1024**3,
                'swap_bytes':0,'cuda_visible_devices':cuda},'wall_seconds':seconds-1,
                'process_peak_rss_kib':100,'peak_cuda_allocated_bytes':1 if phase=='export' else 0,
                'cuda_initialized':phase=='export'}
            with self.subTest(phase=phase):
                e.check_resource_facts(record,phase)
                for wall in (seconds,seconds+1,0,-1,float('nan'),float('inf')):
                    with self.subTest(wall=wall),self.assertRaises(ValueError):
                        e.check_resource_facts({**record,'wall_seconds':wall},phase)
                for key,value in (('host_bytes',8*1024**3+1),('swap_bytes',1),('cuda_visible_devices','foreign')):
                    with self.subTest(key=key),self.assertRaises(ValueError):
                        e.check_resource_facts({**record,'resource_policy':{**record['resource_policy'],key:value}},phase)

    def test_exact_whole_production_ast_inverse_of_export_seconds(self):
        tree=inverse_export_seconds(ast.parse(PATH.read_text()))
        self.assertEqual(hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
            '190acd12ea0d4962ed7dceac47a0e6b312a7bcb10ac14b1097669380b437406f')


class FullfeatureResidualTests(unittest.TestCase):
    def setUp(self):
        if self._testMethodName in {
                'test_exact_role_inventory_preserves_all_common_initialization_guards',
                'test_independent_oracle_executes_centered_nonzero_residual_and_rejects_mutants',
                'test_candidate_oracle_receipt_cannot_use_zero_only_or_partial_proof'}:
            restored=patch.dict(globals(),e=historical_fullfeature_evaluator())
            restored.start();self.addCleanup(restored.stop)

    def test_exact_declared_delta_restores_original_source_and_AST(self):
        source=PATH.read_text();restored=inverse_fullfeature_source(source)
        self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(),
            '0128c543768d8f9eb964231777f5b1fe30e1c39b34c41dc5065441fe2d92c294')
        tree=inverse_fullfeature_authority(ast.parse(source))
        self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(ast.parse(restored),include_attributes=False))
        self.assertEqual(hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
            'e9da7ece9374c331eae89b8d2b1c199f6e4195fc4cfd2840a84a6ee17477ce47')
        for name in ('check_paired_initialization','check_residual_oracle','fullfeature_oracle'):
            changed=ast.parse(source)
            next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name==name).body.append(ast.Pass())
            with self.subTest(helper=name),self.assertRaises(AssertionError):inverse_fullfeature_authority(changed)
        changed=ast.parse(source)
        next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name=='decide').body.append(ast.Pass())
        self.assertNotEqual(ast.dump(inverse_fullfeature_authority(changed),include_attributes=False),
            ast.dump(tree,include_attributes=False))

    def pair(self):
        common={k:'a'*64 for k in ('initial_A_sha256','initial_C_sha256','mu_train_sha256',
            'mu_train_provenance_sha256','initial_raw_unit_packed_sha256')}
        common.update(source={'original':'source'},numerical_flags={'threads':8})
        identity={k:'b'*64 for k in ('static_sha256','initial_A_sha256','initial_C_sha256',
            'mu_train_sha256','mu_train_provenance_sha256','initial_cpu_rng_sha256','initial_cuda_rng_sha256')}
        identity.update(source=common['source'],numerical_flags=common['numerical_flags'])
        result=[]
        for arm,names,shapes in [('control',['A'],[[128,160]]),
                ('candidate',['A','C'],[[128,160],[128,1152]])]:
            result.append({**copy.deepcopy(common),'arm':arm,'identity':{**copy.deepcopy(identity),
                'arm':arm,'parameter_names':names,'parameter_shapes':shapes},'steps':[{'batch':[3,7]}]})
        return result

    def test_exact_role_inventory_preserves_all_common_initialization_guards(self):
        c,a=self.pair();e.check_paired_initialization(c,a)
        for index,key,value in [(0,'parameter_names',['A','C']),(1,'parameter_names',['C','A']),
                (1,'parameter_shapes',[[128,160],[128,1151]]),
                (1,'parameter_shapes',[[128.,160],[128,1152]]),(0,'arm','candidate')]:
            changed=copy.deepcopy([c,a]);changed[index]['identity'][key]=value
            with self.subTest(role=index,key=key),self.assertRaises(ValueError):
                e.check_paired_initialization(*changed)
        for location in ('record','identity'):
            for key in (c if location=='record' else c['identity']):
                if key in ('identity','steps','arm','parameter_names','parameter_shapes'):continue
                changed=copy.deepcopy(a)
                (changed if location=='record' else changed['identity'])[key]='changed'
                with self.subTest(location=location,key=key),self.assertRaises(ValueError):
                    e.check_paired_initialization(c,changed)
        changed=copy.deepcopy(a);changed['steps'][0]['batch'].reverse()
        with self.assertRaises(ValueError):e.check_paired_initialization(c,changed)

    def test_typed_live_C_mu_provenance_and_arm_fingerprints_change(self):
        f=EndpointFactsFixture()
        f.state.update(C=b'updated C',mu_train=(.25,.5),mu_train_provenance={'rows':6355},arm='candidate')
        original=f.facts()
        for key,value in [('C',b'tampered C'),('mu_train',(.25,.75)),
                ('mu_train_provenance',{'rows':6355.0}),('arm','control')]:
            saved=f.state[key];f.state[key]=value
            with self.subTest(member=key):self.assertNotEqual(f.facts()['members'][key],original['members'][key])
            f.state[key]=saved
        del f.state['C']
        with self.assertRaises(KeyError):f.facts()




    def test_checkpoint_to_bundle_C_mu_provenance_and_arm_substitution_rejected(self,arm='candidate'):
        class Tensor:
            dtype='torch.float32';device='cpu';layout='torch.strided';requires_grad=False;grad_fn=None;_version=0
            def __init__(self,shape,byte=0):self.shape=shape;self.data=bytearray([byte])*4*__import__('math').prod(shape)
            def data_ptr(self):return id(self.data)
            def detach(self):return self
            def cpu(self):return self
            def contiguous(self):return self
            def reshape(self,*a):return self
            def view(self,*a):return self
            def numpy(self):return self.data
        class Pages:
            def __init__(self,*a):pass
            def consume(self,*a):pass
        f=EndpointFactsFixture();namespace={'hashlib':hashlib}
        fn=next(n for n in ast.parse(PATH.with_name('train_siglip2_substrate_adaptation.py').read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name=='fingerprint')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original complete serializer>','exec'),namespace)
        tensor_check={'_require':e.require}
        extracted_functions(PATH.with_name('quadratic_readout.py'),{'_check_tensor'},tensor_check)
        fp=lambda _,value,**kw:namespace['fingerprint'](value,**kw)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'resume.pt';checkpoint.write_bytes(b'authenticated storage fixture')
            ident={'arm':arm,'initial_C_sha256':'a'*64}
            saved={'config':{'id2label':{'0':'label'}},'buffers':{},'processor':{'config':{'size':256}},
                'head':{},'A':Tensor((128,160),1),'means':{},'C':Tensor((128,1152),1),
                'mu_train':Tensor((1152,),1),'mu_train_provenance':{'rows':6355},'identity':ident}
            keys={'config','buffers','processor','head','A','means','C','mu_train','mu_train_provenance'}
            disk={k:copy.deepcopy(saved[k]) for k in keys}
            disk.update(schema='prospective-inference',arm=arm,numerical_flags={'threads':8},
                vision_sha256='c'*64,source={'accepted_A_sha256':'d'*64,'encoder_checkpoint_sha256':'e'*64,
                    'readout_sha256':'f'*64,'initial_C_sha256':'a'*64})
            torch=SimpleNamespace(Tensor=Tensor,uint8='uint8',
                load=lambda path,**kw:saved if Path(path)==checkpoint else disk,
                count_nonzero=lambda value:SimpleNamespace(item=lambda:sum(x!=0 for x in value.data)))
            with patch.dict(sys.modules,{'torch':torch}):
                for name in ('mu_train','mu_train_provenance'):
                    ident[name+'_sha256']=disk['source'][name+'_sha256']=fp(None,saved[name])
                trainer=SimpleNamespace(fingerprint=fp,clone=lambda _,v:copy.deepcopy(v),
                    check_payload=lambda *a:None,INFERENCE_KEYS=set(disk)|{'fixed_sha256'},
                    INFERENCE_SCHEMA='prospective-inference',READOUT={'sha256':'f'*64})
                disk['fixed_sha256']=fp(None,disk)
                manifest={'endpoint_state_sha256':fp(None,disk),'files':{'vision.pt':'e'*64},'vision_inventory':[]}
                trainer.admit_bundle=lambda *a:(manifest,{})
                endpoint={'seed':179061,'arm':arm,'checkpoint':descriptor(checkpoint,hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
                    'bundle':descriptor(root/'bundle.json'),'terminal_state_sha256':fp(None,saved),
                    'inference_state_sha256':fp(None,disk)}
                t={'legacy':{'original':SimpleNamespace(CheckpointPages=Pages),
                        'quadratic':SimpleNamespace(_check_tensor=tensor_check['_check_tensor'])},
                    'flags':disk['numerical_flags'],'initial_A_sha256':'d'*64,
                    'initial':{'provenance':{'encoder':{'checkpoint':{'sha256':'e'*64},'inventory':[]}}},
                    'old':SimpleNamespace(finite_tree=lambda _:None)}
                context={'trainer':trainer,'training_context':t,'guards':{},'records':{(179061,arm):{'identity':e.json_form(ident)}}}
                original=e.authenticate_payloads(context,endpoint)
                self.assertEqual(original['members'].get('C'),fp(t,saved['C']))
                for key,value in [('C',Tensor((128,1152),2)),('mu_train',Tensor((1152,),2)),
                        ('mu_train_provenance',{'rows':6355.0}),('arm','control' if arm=='candidate' else 'candidate')]:
                    prior=disk[key];disk[key]=value
                    disk['fixed_sha256']=fp(t,{k:v for k,v in disk.items() if k!='fixed_sha256'})
                    manifest['endpoint_state_sha256']=endpoint['inference_state_sha256']=fp(t,disk)
                    with self.subTest(substitution=key),self.assertRaisesRegex(ValueError,'bundle substitutes'):
                        e.authenticate_payloads(context,endpoint)
                    disk[key]=prior
                disk['fixed_sha256']=fp(t,{k:v for k,v in disk.items() if k!='fixed_sha256'})
                manifest['endpoint_state_sha256']=endpoint['inference_state_sha256']=fp(t,disk)
                disk['C'].data[0]^=1
                with self.assertRaisesRegex(ValueError,'bundle substitutes'):e.authenticate_payloads(context,endpoint)
                saved['C']=Tensor((128,1152),0);disk['C']=Tensor((128,1152),0)
                disk['fixed_sha256']=fp(t,{k:v for k,v in disk.items() if k!='fixed_sha256'})
                manifest['endpoint_state_sha256']=endpoint['inference_state_sha256']=fp(t,disk)
                endpoint['terminal_state_sha256']=fp(t,saved)
                with self.assertRaisesRegex(ValueError,'nonzero'):e.authenticate_payloads(context,endpoint)

    def test_current_typed_tensor_bytes_detect_data_tamper_without_version_change(self):
        class Tensor:
            dtype='torch.float32';_version=0
            def __init__(self,shape):self.shape=shape;self.data=bytearray(4*__import__('math').prod(shape))
            def data_ptr(self):return id(self.data)
            def detach(self):return self
            def cpu(self):return self
            def contiguous(self):return self
            def reshape(self,*args):return self
            def view(self,dtype):return self
            def numpy(self):return self.data
        f=EndpointFactsFixture()
        # The original typed serializer reads fresh storage bytes on every call.
        fn=next(n for n in ast.parse(PATH.with_name('train_siglip2_substrate_adaptation.py').read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name=='fingerprint')
        namespace={'hashlib':hashlib}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original current bytes>','exec'),namespace)
        f.context['trainer'].fingerprint=lambda _,value:namespace['fingerprint'](value)
        f.state.update(C=Tensor((128,1152)),mu_train=Tensor((1152,)),
            mu_train_provenance={'rows':6355},arm='candidate')
        with patch.dict(sys.modules,{'torch':SimpleNamespace(Tensor=Tensor,uint8='uint8')}):
            original=f.facts()
            for key in ('C','mu_train'):
                tensor=f.state[key];pointer=tensor.data_ptr();version=tensor._version
                tensor.data[0]^=1
                with self.subTest(member=key):
                    self.assertEqual((tensor.data_ptr(),tensor._version),(pointer,version))
                    self.assertNotEqual(f.facts()['members'][key],original['members'][key])
                tensor.data[0]^=1
            self.assertEqual(f.facts(),original)

    def test_independent_oracle_executes_centered_nonzero_residual_and_rejects_mutants(self):
        class Tensor:
            def __init__(self,data):self.data=copy.deepcopy(data)
            def detach(self):return self
            def clone(self):return Tensor(self.data)
            def abs(self):
                return Tensor([[abs(v) for v in row] for row in self.data] if isinstance(self.data[0],list)
                    else [abs(v) for v in self.data])
            def sum(self,dim):
                assert dim==0;return Tensor([sum(row[i] for row in self.data) for i in range(len(self.data[0]))])
            def argmax(self):return SimpleNamespace(item=lambda:max(range(len(self.data)),key=self.data.__getitem__))
            def max(self):return SimpleNamespace(item=lambda:max(self.data))
            def __getitem__(self,key):
                if isinstance(key,tuple):return Tensor([row[key[1]] for row in self.data])
                return self.data[key]
            def __setitem__(self,key,value):self.data[key]=value
            def __sub__(self,other):return Tensor([[v-other.data[i] for i,v in enumerate(row)] for row in self.data])
            def __add__(self,other):return Tensor([[v+other.data[j][i] for i,v in enumerate(row)] for j,row in enumerate(self.data)])
        def linear(features,C):return Tensor([[sum(x*w for x,w in zip(row,weight)) for weight in C.data] for row in features.data])
        torch=SimpleNamespace(nn=SimpleNamespace(Parameter=lambda x:x),
            count_nonzero=lambda C:SimpleNamespace(item=lambda:sum(v!=0 for row in C.data for v in row)))
        F=SimpleNamespace(linear=linear)
        calls=[];base=Tensor([[1.,2.],[3.,4.]])
        readout=lambda *args:(calls.append(args) or base.clone())
        fingerprint=lambda _,value:json.dumps(value['raw'].data,sort_keys=True)
        context={'trainer':SimpleNamespace(helper_guard=lambda _:SimpleNamespace(raw_features=readout),fingerprint=fingerprint),
            'training_context':{'legacy':{'quadratic':object()}}}
        state={'head_object':object(),'A':Tensor([[1.]]),'means':{},'arm':'candidate',
            'C':Tensor([[.2,-.3],[.1,.4]]),'mu_train':Tensor([.5,1.])}
        features=Tensor([[2.,3.],[4.,5.]])
        imports={'torch':torch,'torch.nn':SimpleNamespace(functional=F),'torch.nn.functional':F}
        with patch.dict(sys.modules,imports),patch.object(e,'packed_outputs',lambda _,raw:{'raw':raw}):
            output,proof=e.fullfeature_oracle(context,state,features)
            for row,expected in zip(output['raw'].data,[[.7,2.95],[2.5,5.95]],strict=True):
                for value,wanted in zip(row,expected,strict=True):self.assertAlmostEqual(value,wanted)
            self.assertEqual(len(calls),1);e.check_residual_oracle(proof,'candidate')
            self.assertEqual(state['mu_train'].data,[.5,1.])
            state['C']=Tensor([[0.,0.],[0.,0.]])
            with self.assertRaisesRegex(ValueError,'nonzero'):e.fullfeature_oracle(context,state,features)
            state['arm']='control';calls.clear()
            output,proof=e.fullfeature_oracle(context,state,features)
            self.assertEqual(output['raw'].data,base.data);self.assertEqual(len(calls),1)
            e.check_residual_oracle(proof,'control')
            state['C']=Tensor([[0.,.1],[0.,0.]])
            with self.assertRaisesRegex(ValueError,'zero'):e.fullfeature_oracle(context,state,features)

    def test_candidate_oracle_receipt_cannot_use_zero_only_or_partial_proof(self):
        candidate={'C_exact_zero':False,'residual_nonzero_witness':True,
            'omitted_C_mutant_rejected':True,'wrong_mu_mutant_rejected':True}
        control={key:not value for key,value in candidate.items()}
        e.check_residual_oracle(candidate,'candidate');e.check_residual_oracle(control,'control')
        for key in candidate:
            for mutant in ({**candidate,key:not candidate[key]},
                    {k:v for k,v in candidate.items() if k!=key},{**candidate,key:int(candidate[key])}):
                with self.subTest(field=key),self.assertRaises(ValueError):e.check_residual_oracle(mutant,'candidate')
        with self.assertRaises(ValueError):e.check_residual_oracle(candidate,'control')
        with self.assertRaises(ValueError):e.check_residual_oracle(candidate,'foreign')


FINAL_TRAINER_PINS = {'root': '/home/riomus/runs/sfora-so400-live-top1-train-source-v1', 'execution_sha256': '0992550a70f38a2efe461abf6d0672fc8608ee976fca155bb16e66305da7cc81', 'code': {'train_siglip2_compact_ranking.py': 'bcabe2691a2a398b8d8a18a58d91fb8a83388b3004c6f5047ee712bcffb8e453', 'test_siglip2_compact_ranking.py': '3a511a3d256bd80e3210b06de2896c0fb93f8cd34a5a539be6f18cb00dbd823c'}}


class LiveTop1Tests(unittest.TestCase):
    def pair(self):
        pair=FullfeatureResidualTests().pair()
        pair[0]['identity'].update(parameter_names=['A','C'],parameter_shapes=[[128,160],[128,1152]])
        return pair

    def test_both_arms_have_identical_ordered_optimizer_capacity(self):
        pair=self.pair();e.check_paired_initialization(*pair)
        for index in (0,1):
            for key,value in [('parameter_names',['A']),('parameter_names',['C','A']),
                    ('parameter_shapes',[[128,160]]),('parameter_shapes',[[128,160],[128,1151]]),
                    ('parameter_shapes',[[128,160],[128.,1152]])]:
                changed=copy.deepcopy(pair);changed[index]['identity'][key]=value
                with self.subTest(arm=index,key=key,value=value),self.assertRaises(ValueError):
                    e.check_paired_initialization(*changed)
        for key in ('initial_C_sha256','mu_train_sha256','mu_train_provenance_sha256'):
            for location in ('record','identity'):
                changed=copy.deepcopy(pair)
                (changed[1] if location=='record' else changed[1]['identity'])[key]='foreign'
                with self.subTest(key=key,location=location),self.assertRaises(ValueError):
                    e.check_paired_initialization(*changed)

    def test_both_updated_arm_receipts_require_all_nonzero_residual_witnesses(self):
        proof={'C_exact_zero':False,'residual_nonzero_witness':True,
            'omitted_C_mutant_rejected':True,'wrong_mu_mutant_rejected':True}
        for arm in e.ARMS:
            e.check_residual_oracle(proof,arm)
            for key in proof:
                for bad in ({**proof,key:not proof[key]},{**proof,key:int(proof[key])},
                        {k:v for k,v in proof.items() if k!=key}):
                    with self.subTest(arm=arm,key=key),self.assertRaises(ValueError):
                        e.check_residual_oracle(bad,arm)
            with self.assertRaises(ValueError):e.check_residual_oracle({**proof,'foreign':True},arm)
        with self.assertRaises(ValueError):e.check_residual_oracle(proof,'foreign')

    def test_new_method_authority_rejects_all_historical_schemas(self):
        self.assertEqual(e.SCHEMA,'siglip2-compact-live-top1-evaluation-v1')
        self.assertEqual(e.AUTHORITY_SCHEMA,'siglip2-compact-live-top1-evaluation-launch-v1')
        value,args=launch();e.check_launch(value,args)
        self.assertEqual(e.TRAINING,FINAL_TRAINER_PINS)
        with self.assertRaisesRegex(ValueError,'parent-frozen'):
            e.check_launch({**value,'training':historical_fullfeature_evaluator().TRAINING},args)
        for method in ('fullfeature-residual','current-gallery-smooth-ap','image-anchor-smooth-ap',
                'smooth-ap','ranking'):
            with self.subTest(method=method),self.assertRaises(ValueError):
                e.check_launch({**value,'schema':'siglip2-compact-'+method+'-evaluation-launch-v1'},args)
        for key,bad in [('root','/tmp/foreign-source'),('execution_sha256','0'*64),
                ('code',dict.fromkeys(e.TRAIN_FILES,'1'*64))]:
            changed=copy.deepcopy(value);changed['training'][key]=bad
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'parent-frozen'):
                e.check_launch(changed,args)


    def test_both_arm_oracles_execute_centered_nonzero_residual_and_reject_mutants(self):
        class Tensor:
            def __init__(self,data):self.data=copy.deepcopy(data)
            def detach(self):return self
            def clone(self):return Tensor(self.data)
            def abs(self):
                return Tensor([[abs(v) for v in row] for row in self.data] if isinstance(self.data[0],list)
                    else [abs(v) for v in self.data])
            def sum(self,dim):
                assert dim==0;return Tensor([sum(row[i] for row in self.data) for i in range(len(self.data[0]))])
            def argmax(self):return SimpleNamespace(item=lambda:max(range(len(self.data)),key=self.data.__getitem__))
            def max(self):return SimpleNamespace(item=lambda:max(self.data))
            def __getitem__(self,key):
                if isinstance(key,tuple):return Tensor([row[key[1]] for row in self.data])
                return self.data[key]
            def __setitem__(self,key,value):self.data[key]=value
            def __sub__(self,other):return Tensor([[v-other.data[i] for i,v in enumerate(row)] for row in self.data])
            def __add__(self,other):return Tensor([[v+other.data[j][i] for i,v in enumerate(row)] for j,row in enumerate(self.data)])
        def linear(features,C):return Tensor([[sum(x*w for x,w in zip(row,weight)) for weight in C.data] for row in features.data])
        torch=SimpleNamespace(nn=SimpleNamespace(Parameter=lambda x:x),
            count_nonzero=lambda C:SimpleNamespace(item=lambda:sum(v!=0 for row in C.data for v in row)))
        F=SimpleNamespace(linear=linear)
        calls=[];base=Tensor([[1.,2.],[3.,4.]])
        readout=lambda *args:(calls.append(args) or base.clone())
        fingerprint=lambda _,value:json.dumps(value['raw'].data,sort_keys=True)
        context={'trainer':SimpleNamespace(helper_guard=lambda _:SimpleNamespace(raw_features=readout),fingerprint=fingerprint),
            'training_context':{'legacy':{'quadratic':object()}}}
        state={'head_object':object(),'A':Tensor([[1.]]),'means':{},'arm':'candidate',
            'C':Tensor([[.2,-.3],[.1,.4]]),'mu_train':Tensor([.5,1.])}
        features=Tensor([[2.,3.],[4.,5.]])
        imports={'torch':torch,'torch.nn':SimpleNamespace(functional=F),'torch.nn.functional':F}
        with patch.dict(sys.modules,imports),patch.object(e,'packed_outputs',lambda _,raw:{'raw':raw}):
            for arm in e.ARMS:
                state['arm']=arm;calls.clear()
                state['C']=Tensor([[.2,-.3],[.1,.4]])
                output,proof=e.fullfeature_oracle(context,state,features)
                for row,expected in zip(output['raw'].data,[[.7,2.95],[2.5,5.95]],strict=True):
                    for value,wanted in zip(row,expected,strict=True):self.assertAlmostEqual(value,wanted)
                self.assertEqual(len(calls),1);e.check_residual_oracle(proof,arm)
                self.assertEqual(state['mu_train'].data,[.5,1.])
                state['C']=Tensor([[0.,0.],[0.,0.]])
                with self.subTest(arm=arm),self.assertRaisesRegex(ValueError,'nonzero'):
                    e.fullfeature_oracle(context,state,features)
                state['C']=Tensor([[.2,-.3],[.1,.4]])
                for mutant in (lambda x,c:Tensor([[0.,0.],[0.,0.]]),lambda x,c:base.clone()):
                    with self.subTest(arm=arm,mutant=mutant),patch.object(F,'linear',mutant),self.assertRaises(ValueError):
                        e.fullfeature_oracle(context,state,features)

    def test_both_arm_payloads_reject_C_mu_and_arm_substitutions(self):
        historical=FullfeatureResidualTests()
        for arm in e.ARMS:
            with self.subTest(arm=arm):
                historical.test_checkpoint_to_bundle_C_mu_provenance_and_arm_substitution_rejected(arm)

    def test_exact_live_top1_inverse_preserves_every_historical_byte_and_gate(self):
        source=PATH.read_text();restored=inverse_live_top1_source(source)
        self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(),
            '3684268825e8a03592ad355a0752b582245f1d05fd0bcb3ed084a01ef68140b0')
        tree=inverse_live_top1_authority(ast.parse(source))
        self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(ast.parse(restored),include_attributes=False))
        for scope,new,old in LIVE_TOP1_EDITS:
            changed=source.replace(new,new.replace('siglip2','foreign',1) if 'siglip2' in new else
                new.replace('require(', 'foreign_require(',1),1)
            with self.subTest(scope=scope,statement=new[:80]),self.assertRaises(AssertionError):
                inverse_live_top1_authority(ast.parse(changed))
        for name in ('policy','decide','paired_cost','bundle_reads_only','resources','exit_rehash'):
            changed=ast.parse(source)
            next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name==name).body.append(ast.Pass())
            self.assertNotEqual(ast.dump(inverse_live_top1_authority(changed),include_attributes=False),
                ast.dump(tree,include_attributes=False))


    def test_both_export_receipts_require_nonzero_C_and_complete_updated_wires(self):
        reference=module('_live_top1_value_facts',PATH.with_name('evaluate_siglip2_prototype_residual.py'))
        proof={'C_exact_zero':False,'residual_nonzero_witness':True,
            'omitted_C_mutant_rejected':True,'wrong_mu_mutant_rejected':True}
        for arm in e.ARMS:
            value,args=launch(phase='export');args.arm=value['arm']=arm
            args.authority=Path('/tmp/launch');args.authority_sha256='a'*64;args.output=Path('/tmp/output')
            endpoint=next(v for v in value['endpoints'] if v['arm']==arm);key=e.label(endpoint)
            flags={'threads':1};source={'actual':'source'};facts={'updated_C':'b'*64}
            context={'args':args,'launch':value,'code':dict.fromkeys(e.FILES,'b'*64),'guards':{},
                'training_context':{'source':source,'legacy':{'selected':{'source_cpu':{'numerical_flags':flags}}}},
                'costs':e.paired_cost(controls('first'),'first'),'reference':reference,
                'cpu':{'payload_facts':{key:facts}}}
            record={'schema':e.SCHEMA,'phase':'export','arm':arm,'seed':179061,'stage':'first','panel':'selection',
                'execution_sha256':args.execution_sha256,'source_code':context['code'],'source':source,
                'cost_policy':e.COST_POLICY,'launch':value,'authority':descriptor(args.authority),
                'authority_sha256':args.authority_sha256,'binding':e.binding(context),'numerical_flags':flags,
                'output':str(args.output),'cost':context['costs'],'resource_policy':e.policy('export'),
                'wall_seconds':1,'process_peak_rss_kib':100,'peak_cuda_allocated_bytes':1,'cuda_initialized':True,
                'invocation':{'argv':e.cli(args),'optimize':0,'cuda_visible_devices':'0','cublas_workspace_config':':4096:8'},
                'files':{key+suffix:'b'*64 for suffix in ('.raw.npy','.unit.npy','.packed.bin')},
                'batch_sizes':{'query':[32]*54+[6],'gallery':[32]*53+[19]},'payload_facts':facts,
                'inference_state_sha256':endpoint['inference_state_sha256'],
                'train_witness':{'batch':list(range(16)),'residual_oracle':proof},
                'panel_facts':{name:{'shape':shape,'dtype':dtype,'sha256':'b'*64} for name,shape,dtype in
                    [('raw',[3449,128],'torch.float32'),('unit',[3449,128],'torch.float32'),
                     ('codes',[3449,128],'torch.int8'),('inverse_norms',[3449],'torch.float16')]}}
            for name in ('pass','engineering_admission_pass','integrity_pass','resources_pass','exit_rehash_pass',
                    'sequential_model_ownership','rng_flags_preserved','both_locks_held_in_parent_authority',
                    'strict_independent_reload_exact','full_updated_state_exact','raw_unit_packed_readback_exact',
                    'train_native_witness_exact','bundle_dependency_boundary_enforced','same_role_oracle_exact',
                    'native_exact_four_post_calibration'):record[name]=True
            for name in ('official_read','global_production_goal_met','public_latency_measured','product_go'):record[name]=False
            with patch.object(e,'read_json',return_value=value):
                e.check_receipt(context,record,'export',arm,179061)
                for name in proof:
                    bad=copy.deepcopy(record);bad['train_witness']['residual_oracle'][name]=not proof[name]
                    with self.subTest(arm=arm,oracle=name),self.assertRaises(ValueError):
                        e.check_receipt(context,bad,'export',arm,179061)
                for name in record['panel_facts']:
                    bad=copy.deepcopy(record);del bad['panel_facts'][name]
                    with self.subTest(arm=arm,wire=name),self.assertRaises(ValueError):
                        e.check_receipt(context,bad,'export',arm,179061)
                for name,bad_value in [('batch_sizes',{'query':[32]*54+[6],'gallery':[32]*53+[18]}),
                        ('payload_facts',{}),('inference_state_sha256','0'*64),
                        ('strict_independent_reload_exact',False),('full_updated_state_exact',False),
                        ('native_exact_four_post_calibration',False),('raw_unit_packed_readback_exact',False),
                        ('schema','siglip2-compact-fullfeature-residual-evaluation-v1')]:
                    with self.subTest(arm=arm,field=name),self.assertRaises(ValueError):
                        e.check_receipt(context,{**record,name:bad_value},'export',arm,179061)

    def test_both_public_inference_arms_keep_saved_flags_and_connected_residual(self):
        for arm in e.ARMS:
            for mode in ('TRAINmicro16','export'):
                f=InferenceContractFixture();f.state['arm']=arm
                # The real public identity tree also includes the complete fixed head.
                public=f.public.__globals__
                f.state['readout_sha256']=f.context['trainer'].fingerprint(None,public['inference_readout_tree'](f.state))
                with self.subTest(arm=arm,mode=mode):
                    f.call(mode)
                    self.assertTrue(any(name=='residual' for name,grad in f.operations))
                    self.assertTrue(all(not grad for name,grad in f.operations))



if __name__ == '__main__':
    unittest.main()
