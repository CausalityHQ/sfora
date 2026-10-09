#!/usr/bin/env python3
"""Stdlib source/admission falsifiers for the rank-routed connected MLP. Native UNRUN.

Run: python3 -B scripts/test_siglip2_rank_routed_mlp.py --source-only
Intended envelope: timeout 120, address space 1GiB, <=15s. No Torch, numpy, images or GPU.

The new trainer is the committed connected trainer plus an enumerated byte delta
(INVERSE_EDITS + two inserted blocks). The exact inverse below restores the committed
connected trainer bytes, so every function outside the delta (admission, lifecycle,
schedule128, AdamW/scaler/clip, caps, restore, bundle, exit) is unchanged by
construction; the committed connected seam tests are reused against the new driver.
Tensors/autograd here are stdlib stand-ins: they establish the source contract only,
never native forward/gradient parity.
The fixed phase observer and exceptional cleanup seams invert first to the exact
940d37fb committed source; the historical inverse then runs without changing pins.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import textwrap
import time
from types import FunctionType, ModuleType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'train_siglip2_rank_routed_mlp.py'
CONNECTED = HERE / 'train_siglip2_connected_mlp.py'
CONNECTED_TEST = HERE / 'test_siglip2_connected_mlp.py'
CONNECTED_SHA = '79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b'
CONNECTED_TEST_SHA = '8391b3dd38a682dd327c0d0a3f0a62a5e699f7e934a6e39959600ed7f045bc25'
ORIGINAL = (HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/'
            'export-exit-scan-ab-v1-freeze/train_siglip2_identity_diversity.py')
ORIGINAL_SHA = '840c5d8277a89ccdac02c9e231cbe6eddf386e2b23915ecd1bbec1136c51dee8'
RUNTIME_BLOCK_SHA = '4a983d2c105b672cbe29dea9e3ee0d0fbf486c4d1a197d46009b4c096797a702'
ROUTE_BLOCK_SHA = '3fab2d333f38dda0f9ce600edc0ae98edcd93c387da799521574e9b762fd4b41'

OBSERVATION_BASE_SHA = '940d37fb12e4a6cc3f53c023ee31d50f8316d801fe8a6db93af929b7028bb3f1'
OBSERVATION_BLOCK_SHA = 'bc27d7d122f3456198dfaec0312e29e3be2847095b3472402c6b85cb7cbd61f4'

# Only storage scheduling may differ from the immutable CPU-v3 trainer.
SCHEDULING_BASE_SHA = '69302dab4a808a259a381d9b649ac58bd1d7a3c5dffe2b8aa6982131dd45cd5d'
SCHEDULING_BASE_AST_SHA = '5eaf64ef2da836fe4d3770c303721cff58cb82eff4384b5d344b1bae1744925d'
SCHEDULING_INVERSE_EDITS = (
    (b"""        if candidate and step == 1 and state['device'] == 'cpu':
            full_reference = route_full_reference(torch,context,state,identity,members,batch,K,view)
        ranking = [torch.zeros_like(p) for p in members] if step == 1 else None
        route = route_accumulators(torch,members) if candidate and step == 1 else None
        if route is not None and state['device'] == 'cpu':
            route['full'] = full_reference
            del full_reference
            phase_observe(context,'view_accumulators',view=view,groups=(('ranking_total',ranking_total),('routed_total',routed_total),
                ('ranking_view',ranking),('routed',route['routed']),('original',route['original']),
                ('unrouted',route['unrouted']),('encoder_ranking',route['ranking'])))
""", b"""        ranking = [torch.zeros_like(p) for p in members] if step == 1 else None
        route = route_accumulators(torch,members) if candidate and step == 1 else None
        if route is not None and state['device'] == 'cpu':
            phase_observe(context,'view_accumulators',view=view,groups=(('ranking_total',ranking_total),('routed_total',routed_total),
                ('ranking_view',ranking),('routed',route['routed']),('original',route['original']),
                ('unrouted',route['unrouted']),('encoder_ranking',route['ranking'])))
            route['full'] = route_full_reference(torch,context,state,identity,members,batch,K,view)
"""),
    (b"""            facts = {'pixels':fingerprint(context,temporary['pixels'])}
            if pass_index:
                require(facts['pixels'] == replay['pixels'], 'full B64 replay pixels/features differ')
""", b''),
    (b"                del temporary['pixels']\n", b''),
    (b"                facts['features'] = fingerprint(context,temporary['features'])\n",
     b"                facts = {name:fingerprint(context,temporary[name]) for name in ('pixels','features')}\n"),
)

# Enumerated observer/error-cleanup seams; applied before the historical repair inverse.
OBSERVATION_INVERSE_EDITS = (
    (b"    finally:\n        primary = sys.exc_info()[1]\n        phase_observe(context,'arm_cleanup_begin')\n        if primary is not None:\n            import traceback\n            traceback.clear_frames(primary.__traceback__)\n        try:\n            if state:\n                release(context,state)\n        except BaseException as cleanup:\n            if primary is None:\n                raise\n            raise primary from cleanup\n        finally:\n            phase_observe(context,'arm_cleanup_end',close=True)\n\n",
     b'    finally:\n        if state:\n            release(context,state)\n\n'),
    (b"    args = context['connected_args']\n    if args.phase == 'cpu' and arm == 'candidate' and discarded_update:\n        context['phase_observation'] = {'records':0,'bytes':0,'failed':False,'closed':False}\n    phase_observe(context,'arm_fresh_begin')\n    state = fresh(context,arm,seed,device)\n",
     b"    args = context['connected_args']\n    state = fresh(context,arm,seed,device)\n"),
    (b"    unrouted = torch.autograd.grad(original,members,retain_graph=True,allow_unused=True)\n    phase_observe(context,'micro_regression_gradients',groups=(('routed',routed),('unrouted',unrouted),('ranking',ranking)))\n    nonzero = sum(0 if g is None else torch.count_nonzero(g).item() for g in routed[2:])\n",
     b'    unrouted = torch.autograd.grad(original,members,retain_graph=True,allow_unused=True)\n    nonzero = sum(0 if g is None else torch.count_nonzero(g).item() for g in routed[2:])\n'),
    (b"    require(all(ref() is None for ref in refs), 'query/gallery split lifetime survived release')\n    phase_observe(context,'split_released')\n    return result\n",
     b"    require(all(ref() is None for ref in refs), 'query/gallery split lifetime survived release')\n    return result\n"),
    (b"    trainer,connected = context['trainer'],context['connected']\n    phase_observe(context,'split_begin')\n    query,gallery = dict(state),dict(state)\n",
     b"    trainer,connected = context['trainer'],context['connected']\n    query,gallery = dict(state),dict(state)\n"),
    (b"                raise\n        finally:\n            phase_observe(context,'full_graph_released',view=view)\n\n",
     b'                raise\n\n'),
    (b"            temporary['total'] = torch.autograd.grad(temporary['loss'],members)\n            phase_observe(context,'full_total_gradients',view=view,groups=(('regression',temporary['regression']),\n                ('ranking',temporary['ranking']),('original_A_C',temporary['original_A_C']),('total',temporary['total'])))\n            for i,name in enumerate(('A','C')):\n",
     b"            temporary['total'] = torch.autograd.grad(temporary['loss'],members)\n            for i,name in enumerate(('A','C')):\n"),
    (b"            temporary['original_A_C'] = torch.autograd.grad(temporary['original_loss'],members[:2],retain_graph=True)\n            phase_observe(context,'full_original_A_C_gradients',view=view,groups=(('regression',temporary['regression']),\n                ('ranking',temporary['ranking']),('original_A_C',temporary['original_A_C'])))\n            temporary['total'] = torch.autograd.grad(temporary['loss'],members)\n",
     b"            temporary['original_A_C'] = torch.autograd.grad(temporary['original_loss'],members[:2],retain_graph=True)\n            temporary['total'] = torch.autograd.grad(temporary['loss'],members)\n"),
    (b"            temporary['ranking'] = torch.autograd.grad(temporary['rank'],members,retain_graph=True)\n            phase_observe(context,'full_ranking_gradients',view=view,groups=(('regression',temporary['regression']),\n                ('ranking',temporary['ranking'])))\n            temporary['original_loss'] = temporary['original']+temporary['rank']\n",
     b"            temporary['ranking'] = torch.autograd.grad(temporary['rank'],members,retain_graph=True)\n            temporary['original_loss'] = temporary['original']+temporary['rank']\n"),
    (b"            temporary['regression'] = torch.autograd.grad(temporary['mse'],members,retain_graph=True,allow_unused=True)\n            phase_observe(context,'full_regression_gradients',view=view,groups=(('regression',temporary['regression']),))\n            require(all(g is None or torch.count_nonzero(g).item() == 0 for g in temporary['regression'][2:]),\n",
     b"            temporary['regression'] = torch.autograd.grad(temporary['mse'],members,retain_graph=True,allow_unused=True)\n            require(all(g is None or torch.count_nonzero(g).item() == 0 for g in temporary['regression'][2:]),\n"),
    (b"            temporary['loss'] = temporary['mse']+temporary['rank']\n            phase_observe(context,'full_loss',view=view,groups=(('raw',(temporary['raw'],temporary['detached'])),\n                ('loss',(temporary['mse'],temporary['rank'],temporary['original'],temporary['loss']))))\n            temporary['regression'] = torch.autograd.grad(temporary['mse'],members,retain_graph=True,allow_unused=True)\n",
     b"            temporary['loss'] = temporary['mse']+temporary['rank']\n            temporary['regression'] = torch.autograd.grad(temporary['mse'],members,retain_graph=True,allow_unused=True)\n"),
    (b"            temporary['features'] = F.normalize(state['model'](pixel_values=temporary['pixels']).pooler_output.float(),dim=1)\n            phase_observe(context,'full_forward',view=view,groups=(('features',(temporary['features'],)),))\n            temporary['raw'] = connected.raw_features(temporary['features'],state['head_object'],state['A'],state['means'],\n",
     b"            temporary['features'] = F.normalize(state['model'](pixel_values=temporary['pixels']).pooler_output.float(),dim=1)\n            temporary['raw'] = connected.raw_features(temporary['features'],state['head_object'],state['A'],state['means'],\n"),
    (b"        temporary['pixels'] = torch.cat(temporary.pop('chunks'),dim=0)\n        phase_observe(context,'full_pixels',view=view,groups=(('pixels',(temporary['pixels'],)),))\n        with torch.autocast('cpu',enabled=False):\n",
     b"        temporary['pixels'] = torch.cat(temporary.pop('chunks'),dim=0)\n        with torch.autocast('cpu',enabled=False):\n"),
    (b"    temporary,refs = {},[]\n    phase_observe(context,'full_begin',view=view)\n    try:\n",
     b'    temporary,refs = {},[]\n    try:\n'),
    (b"    preclip_norm = float(norm)\n    phase_observe(context,'update_complete',close=True)\n    observation_guard(context)\n    if state['device'] == 'cuda':\n",
     b"    preclip_norm = float(norm)\n    if state['device'] == 'cuda':\n"),
    (b"                                            context['legacy']['selected']['packages'])\n    phase_observe(context,'current_encoder_end')\n    integrity(context,state,identity)\n",
     b"                                            context['legacy']['selected']['packages'])\n    integrity(context,state,identity)\n"),
    (b"    optimizer.zero_grad(set_to_none=True)\n    phase_observe(context,'optimizer_gradients_cleared')\n    state['current_encoder'] = encoder_facts(state,context['legacy']['original'],context['legacy']['source_driver'],\n",
     b"    optimizer.zero_grad(set_to_none=True)\n    state['current_encoder'] = encoder_facts(state,context['legacy']['original'],context['legacy']['source_driver'],\n"),
    (b"    scaler.update()\n    phase_observe(context,'optimizer_step_end')\n    require(scaler.get_scale() == scale == (128. if state['device'] == 'cuda' else 1.) and\n",
     b"    scaler.update()\n    require(scaler.get_scale() == scale == (128. if state['device'] == 'cuda' else 1.) and\n"),
    (b"    norm = torch.nn.utils.clip_grad_norm_(members,1.,error_if_nonfinite=True)\n    phase_observe(context,'clip_end')\n    scale = scaler.get_scale()\n",
     b'    norm = torch.nn.utils.clip_grad_norm_(members,1.,error_if_nonfinite=True)\n    scale = scaler.get_scale()\n'),
    (b"    optimizer_routing = route_optimizer(torch,context,members,names,ranking_total,routed_total) if routed_total is not None else None\n    phase_observe(context,'actual_unscaled_gradients',groups=(('ranking_total',ranking_total),('routed_total',routed_total)))\n    del routed_total\n",
     b'    optimizer_routing = route_optimizer(torch,context,members,names,ranking_total,routed_total) if routed_total is not None else None\n    del routed_total\n'),
    (b"                del route_refs\n            phase_observe(context,'view_reference_accumulators_released',view=view)\n    if released:\n",
     b'                del route_refs\n    if released:\n'),
    (b"                                   **({'routing':route_view(torch,context,route)} if candidate else {})})\n            phase_observe(context,'view_correspondence_end',view=view)\n            for total,part in zip(ranking_total,ranking,strict=True):\n",
     b"                                   **({'routing':route_view(torch,context,route)} if candidate else {})})\n            for total,part in zip(ranking_total,ranking,strict=True):\n"),
    (b"            del pixels,cpu_pixels,features,raw,detached,original,mse,rank,loss,facts\n            phase_observe(context,'micro_released',view=view,micro=offset)\n        if step == 1:\n",
     b'            del pixels,cpu_pixels,features,raw,detached,original,mse,rank,loss,facts\n        if step == 1:\n'),
    (b"                scaler.scale(loss).backward()\n                phase_observe(context,'micro_backward_end',view=view,micro=offset)\n            mse_sum += float(mse.detach())\n",
     b'                scaler.scale(loss).backward()\n            mse_sum += float(mse.detach())\n'),
    (b"                    del accumulator,gradient,gradients\n                phase_observe(context,'micro_witness_end',view=view,micro=offset)\n                loss = mse+rank\n",
     b'                    del accumulator,gradient,gradients\n                loss = mse+rank\n'),
    (b"                    parity_checks += 1\n                phase_observe(context,'micro_loss',view=view,micro=offset,groups=(('loss',(mse,rank,original)),))\n                if step == 1:\n",
     b'                    parity_checks += 1\n                if step == 1:\n'),
    (b"                require(raw.requires_grad, 'live objective graph detached')\n                phase_observe(context,'micro_forward',view=view,micro=offset,groups=(('features',(features,)),('raw',(raw,detached))))\n                mse,rank,selected = trainer.loss_terms(context,state,raw,anchors,K)\n",
     b"                require(raw.requires_grad, 'live objective graph detached')\n                mse,rank,selected = trainer.loss_terms(context,state,raw,anchors,K)\n"),
    (b"            cpu_pixels,facts = context['witness'].pixels_for(trainer,context,state,state['processor_object'],anchors,view)\n            phase_observe(context,'micro_pixels',view=view,micro=offset,groups=(('pixels',(cpu_pixels,)),))\n            with torch.autocast(state['device'],enabled=False):\n",
     b"            cpu_pixels,facts = context['witness'].pixels_for(trainer,context,state,state['processor_object'],anchors,view)\n            with torch.autocast(state['device'],enabled=False):\n"),
    (b"        if route is not None and state['device'] == 'cpu':\n            phase_observe(context,'view_accumulators',view=view,groups=(('ranking_total',ranking_total),('routed_total',routed_total),\n                ('ranking_view',ranking),('routed',route['routed']),('original',route['original']),\n                ('unrouted',route['unrouted']),('encoder_ranking',route['ranking'])))\n            route['full'] = route_full_reference(torch,context,state,identity,members,batch,K,view)\n",
     b"        if route is not None and state['device'] == 'cpu':\n            route['full'] = route_full_reference(torch,context,state,identity,members,batch,K,view)\n"),
    (b"    accumulator_refs = [weakref.ref(t) for t in (*ranking_total,*routed_total)] if routed_total is not None else []\n    phase_observe(context,'outer_accumulators',groups=(('ranking_total',ranking_total),('routed_total',routed_total)))\n    for view in VIEWS:\n",
     b'    accumulator_refs = [weakref.ref(t) for t in (*ranking_total,*routed_total)] if routed_total is not None else []\n    for view in VIEWS:\n'),
    (b"    integrity(context,state,identity)\n    phase_observe(context,'update_integrity_end')\n    candidate = state['arm'] == 'candidate'\n",
     b"    integrity(context,state,identity)\n    candidate = state['arm'] == 'candidate'\n"),
    (b"    require(type(step) is int and state['counter'] == step-1 and 1 <= step <= 128, 'fixed complete update required')\n    phase_observe(context,'update_integrity_begin')\n    integrity(context,state,identity)\n",
     b"    require(type(step) is int and state['counter'] == step-1 and 1 <= step <= 128, 'fixed complete update required')\n    integrity(context,state,identity)\n"),
    (b"    check_payload(context,payload(context,state,identity),identity,state['counter'])\n    phase_observe(context,'integrity_payload_end')\n    phase_observe(context,'resource_begin')\n    resource_check(context)\n",
     b"    check_payload(context,payload(context,state,identity),identity,state['counter'])\n    resource_check(context)\n"),
    (b"    require(legacy['source_driver'].numerical_flags() == context['flags'], 'original numerical flags changed')\n    phase_observe(context,'integrity_payload_begin')\n    check_payload(context,payload(context,state,identity),identity,state['counter'])\n",
     b"    require(legacy['source_driver'].numerical_flags() == context['flags'], 'original numerical flags changed')\n    check_payload(context,payload(context,state,identity),identity,state['counter'])\n"),
    (b"            'whole new phase resource/deadline exceeded')\n    phase_observe(context,'resource_checked')\n    observation_guard(context)\n    return cgroup\n",
     b"            'whole new phase resource/deadline exceeded')\n    return cgroup\n"),
    (b"    integrity(context,state,state['identity'])\n    phase_observe(context,'fresh_complete')\n    return state\n",
     b"    integrity(context,state,state['identity'])\n    return state\n"),
    (b"        state['config'],state['buffers'],config_path,base,source_runtime=state['provenance']['encoder']['source_proof']['runtime'])\n    phase_observe(context,'model_constructed')\n    state['base_vision'] = base\n",
     b"        state['config'],state['buffers'],config_path,base,source_runtime=state['provenance']['encoder']['source_proof']['runtime'])\n    state['base_vision'] = base\n"),
    (b"    factory = {**legacy['prior'],'guards':context['guards']}\n    phase_observe(context,'model_begin')\n    model,processor,cache,base,structure = construct_encoder(legacy['source_driver'],legacy['original'],factory,\n",
     b"    factory = {**legacy['prior'],'guards':context['guards']}\n    model,processor,cache,base,structure = construct_encoder(legacy['source_driver'],legacy['original'],factory,\n"),
    (b"    state,initial_ident = load_initializer(context,qualification)\n    phase_observe(context,'initializer_end',groups=(('A_C',(state['A'],state['C'])),))\n    context.get('A_owners',{}).pop(id(state),None)\n",
     b"    state,initial_ident = load_initializer(context,qualification)\n    context.get('A_owners',{}).pop(id(state),None)\n"),
    (b"    qualification = select_initializer(context['original_cpu_record'],seed)\n    phase_observe(context,'initializer_begin')\n    state,initial_ident = load_initializer(context,qualification)\n",
     b"    qualification = select_initializer(context['original_cpu_record'],seed)\n    state,initial_ident = load_initializer(context,qualification)\n"),
)

# Exact repair splices back to the reviewed draft, before its historical byte inverse.
REPAIR_INVERSE_EDITS = (
    (b"                check_routing(context,v.get('routing'),identity)\n        actual = row.get('optimizer_routing')\n        require((isinstance(actual,dict) and actual.keys() == {'max_abs','tolerance'} and\n                 actual['max_abs'].keys() == set(identity['parameter_names']) and\n                 all(type(v) is float and math.isfinite(v) and v >= 0 for v in actual['max_abs'].values()) and\n                 actual['tolerance'] == {'rtol':context['witness'].RTOL,'atol':context['witness'].ATOL})\n                if identity['arm'] == 'candidate' and row['step'] == 1 else 'optimizer_routing' not in row,\n                'actual optimizer rank-routing witness differs')\n        projected.append({k:row[k] for k in ('step','batch','membership','full_membership_sha256','full_valid','mse','rank',\n",
     b"                check_routing(context,v.get('routing'),identity)\n        projected.append({k:row[k] for k in ('step','batch','membership','full_membership_sha256','full_valid','mse','rank',\n"),
    (b"        **({'routed_parity_checks':parity_checks} if candidate else {}),\n        **({'optimizer_routing':optimizer_routing} if candidate and step == 1 else {}),\n        'before_sha256':before,'after_sha256':after,'vision_sha256':state['current_encoder']['vision_sha256'],\n",
     b"        **({'routed_parity_checks':parity_checks} if candidate else {}),\n        'before_sha256':before,'after_sha256':after,'vision_sha256':state['current_encoder']['vision_sha256'],\n"),
    (b"    del ranking_total\n    if accumulator_refs:\n        gc.collect()\n        require(all(ref() is None for ref in accumulator_refs), 'routing optimizer accumulator lifetime survived release')\n    del accumulator_refs\n    preclip_norm = float(norm)\n",
     b'    del ranking_total\n    preclip_norm = float(norm)\n'),
    (b'    scaler.unscale_(optimizer)\n    optimizer_routing = route_optimizer(torch,context,members,names,ranking_total,routed_total) if routed_total is not None else None\n    del routed_total\n    gradient_norms = {n:float(p.grad.double().norm()) if p.grad is not None else 0. for n,p in zip(names,members,strict=True)}\n',
     b'    scaler.unscale_(optimizer)\n    gradient_norms = {n:float(p.grad.double().norm()) if p.grad is not None else 0. for n,p in zip(names,members,strict=True)}\n'),
    (b"            del total,part\n            if route is not None:\n                route_refs = [weakref.ref(t) for key in ('routed','original','unrouted','ranking') for t in route[key]]\n                for total,part in zip(routed_total,route['routed'],strict=True):\n                    total.add_(part)\n                del total,part\n                if route['full'] is not None:\n                    refs = [weakref.ref(t) for t in (*route['full']['scalars'].values(),*route['full']['gradients'])]\n                    route['full'] = None\n                    gc.collect()\n                    require(all(ref() is None for ref in refs), 'detached full reference lifetime survived release')\n                    del refs\n            del ranking,route\n            if candidate:\n                gc.collect()\n                require(all(ref() is None for ref in route_refs), 'routing view accumulator lifetime survived release')\n                del route_refs\n    if released:\n",
     b'            del total,part\n            del ranking,route\n    if released:\n'),
    (b"                    original = mse\n                    mse = routed_regression(context,state,detached,anchors,K)\n                    require(torch.equal(mse.detach(),original.detach()), 'rank-routed regression scalar differs from pinned original')\n",
     b"                    original = mse\n                    mse = context['routing'].regression_terms(context,state,detached,anchors,K)\n                    require(torch.equal(mse.detach(),original.detach()), 'rank-routed regression scalar differs from pinned original')\n"),
    (b"        route = route_accumulators(torch,members) if candidate and step == 1 else None\n        if route is not None and state['device'] == 'cpu':\n            route['full'] = route_full_reference(torch,context,state,identity,members,batch,K,view)\n        for offset in range(0,64,16):\n",
     b'        route = route_accumulators(torch,members) if candidate and step == 1 else None\n        for offset in range(0,64,16):\n'),
    (b'    ranking_total = [torch.zeros_like(p) for p in members] if step == 1 else None\n    routed_total = [torch.zeros_like(p) for p in members[:2]] if candidate and step == 1 else None\n    accumulator_refs = [weakref.ref(t) for t in (*ranking_total,*routed_total)] if routed_total is not None else []\n    for view in VIEWS:\n',
     b'    ranking_total = [torch.zeros_like(p) for p in members] if step == 1 else None\n    for view in VIEWS:\n'),
)

# Reverse of the forward edits (new bytes -> committed connected bytes), applied last-first.
INVERSE_EDITS = (
    (b"                 all(v['ranking_gradient_norms'].keys() == set(identity['parameter_names']) and\n                     all(math.isfinite(n) and n > 0 for n in v['ranking_gradient_norms'].values()) for v in row['view_gradients']))\n                if row['step'] == 1 else row['view_gradients'] == [], 'both-view original SmoothAP connectivity differs')\n        require(row.get('routed_parity_checks') == (8 if identity['arm'] == 'candidate' else None) and\n                type(row.get('routed_parity_checks')) is (int if identity['arm'] == 'candidate' else type(None)),\n                'rank-routed per-micro scalar/forward parity differs')\n        if row['step'] == 1:\n            for v in row['view_gradients']:\n                check_routing(context,v.get('routing'),identity)\n",
     b"                 all(v['ranking_gradient_norms'].keys() == set(identity['parameter_names']) and\n                     all(math.isfinite(n) and n > 0 for n in v['ranking_gradient_norms'].values()) for v in row['view_gradients']))\n                if row['step'] == 1 else row['view_gradients'] == [], 'both-view original SmoothAP connectivity differs')\n"),
    (b"        'ranking_gradient_norm':ranking_gradient_norm,'ranking_C_gradient_norm':ranking_C_gradient_norm,\n        **({'routed_parity_checks':parity_checks} if candidate else {}),\n",
     b"        'ranking_gradient_norm':ranking_gradient_norm,'ranking_C_gradient_norm':ranking_C_gradient_norm,\n"),
    (b"            del ranking,route\n    if released:\n        gc.collect()\n        require(all(ref() is None for ref in released), 'rank-routing micro graph/tensor lifetime survived release')\n        del released\n    scaler.unscale_(optimizer)\n",
     b'            del ranking\n    scaler.unscale_(optimizer)\n'),
    (b"            view_gradients.append({'view':view,'ranking_gradient_norms':norms,\n                                   **({'routing':route_view(torch,context,route)} if candidate else {})})\n",
     b"            view_gradients.append({'view':view,'ranking_gradient_norms':norms})\n"),
    (b'            if candidate and step == 1:\n                released.extend(weakref.ref(v) for v in (features,raw,detached,original,mse,rank,loss))\n            del pixels,cpu_pixels,features,raw,detached,original,mse,rank,loss,facts\n',
     b'            del pixels,cpu_pixels,features,raw,mse,rank,loss,facts\n'),
    (b'                    if candidate:\n                        route_micro(torch,context,state,members,route,gradients,original,mse,rank,features,anchors,K,selected,offset == 0)\n                    del accumulator,gradient,gradients\n                loss = mse+rank\n',
     b'                    del accumulator,gradient,gradients\n                loss = mse+rank\n'),
    (b"                mse,rank,selected = trainer.loss_terms(context,state,raw,anchors,K)\n                if candidate:\n                    original = mse\n                    mse = context['routing'].regression_terms(context,state,detached,anchors,K)\n                    require(torch.equal(mse.detach(),original.detach()), 'rank-routed regression scalar differs from pinned original')\n                    parity_checks += 1\n                if step == 1:\n",
     b'                mse,rank,selected = trainer.loss_terms(context,state,raw,anchors,K)\n                if step == 1:\n'),
    (b"                        state['mu_train'],context['legacy']['quadratic'],trainer.helper_guard(context))\n                    detached = trainer.raw_features(context,state,features.detach())\n                    require(detached.requires_grad and torch.equal(detached.detach(),raw.detach()),\n                            'rank-routed detached raw forward differs')\n                else:\n",
     b"                        state['mu_train'],context['legacy']['quadratic'],trainer.helper_guard(context))\n                else:\n"),
    (b"                features = F.normalize(state['model'](pixel_values=pixels).pooler_output.float(),dim=1)\n                detached = original = None\n                if candidate:\n",
     b"                features = F.normalize(state['model'](pixel_values=pixels).pooler_output.float(),dim=1)\n                if state['arm'] == 'candidate':\n"),
    (b'        ranking = [torch.zeros_like(p) for p in members] if step == 1 else None\n        route = route_accumulators(torch,members) if candidate and step == 1 else None\n        for offset in range(0,64,16):\n',
     b'        ranking = [torch.zeros_like(p) for p in members] if step == 1 else None\n        for offset in range(0,64,16):\n'),
    (b'    mse_sum = rank_sum = 0.\n    parity_checks,released = 0,[]\n    optimizer.zero_grad(set_to_none=True)\n',
     b'    mse_sum = rank_sum = 0.\n    optimizer.zero_grad(set_to_none=True)\n'),
    (b"    integrity(context,state,identity)\n    candidate = state['arm'] == 'candidate'\n    batch = state['schedules'][str(state['seed'])][step-1].tolist()\n",
     b"    integrity(context,state,identity)\n    batch = state['schedules'][str(state['seed'])][step-1].tolist()\n"),
    (b"    context['connected_function'] = (context['connected'].raw_features,context['connected'].raw_features.__code__)\n    context['routing'] = _regression_runtime(context)\n",
     b"    context['connected_function'] = (context['connected'].raw_features,context['connected'].raw_features.__code__)\n"),
    (b"'encoder_shapes':MLP_SHAPES,'cost_ratio':1.50,'routing':ROUTING}",
     b"'encoder_shapes':MLP_SHAPES,'cost_ratio':1.50}"),
    (b"'objective':'unchanged original CONTROL complete-gallery SmoothAP and P[label]; candidate encoder SmoothAP-only',",
     b"'objective':'unchanged original CONTROL complete-gallery SmoothAP and P[label]',"),
    (b"ROUTING = {'encoder':'SmoothAP only; regression detached','A_C':'original regression and SmoothAP','control':'unchanged',\n           'original_trainer_sha256':'840c5d8277a89ccdac02c9e231cbe6eddf386e2b23915ecd1bbec1136c51dee8',\n           'loss_terms_ast_sha256':'d9cdceea0ac03e072e6a33f86053823a786e12a3cf180e645eebe002edcffc26',\n           'regression_ast_sha256':'761203e657d9ebffa1c92e2b8bfee6d9a9af45722035722c60f143ab6fbb0d87'}\nRECIPE = {'adamw':",
     b"RECIPE = {'adamw':"),
    (b"context['connected_code']['train_siglip2_rank_routed_mlp.py'],{})",
     b"context['connected_code']['train_siglip2_connected_mlp.py'],{})"),
    (b"portable = load_authenticated(name,directory/'train_siglip2_rank_routed_mlp.py',",
     b"portable = load_authenticated(name,directory/'train_siglip2_connected_mlp.py',"),
    (b"manifest['code']['train_siglip2_rank_routed_mlp.py'],",
     b"manifest['code']['train_siglip2_connected_mlp.py'],"),
    (b"    return [str(Path(root)/'train_siglip2_rank_routed_mlp.py'),'--execution-sha256',execution,",
     b"    return [str(Path(root)/'train_siglip2_connected_mlp.py'),'--execution-sha256',execution,"),
    (b"FILES = {'train_siglip2_rank_routed_mlp.py', 'test_siglip2_rank_routed_mlp.py'}",
     b"FILES = {'train_siglip2_connected_mlp.py', 'test_siglip2_connected_mlp.py'}"),
    (b"BUNDLE_SCHEMA = 'siglip2-rank-routed-mlp-bundle-v1'",
     b"BUNDLE_SCHEMA = 'siglip2-connected-mlp-bundle-v1'"),
    (b"INFERENCE_SCHEMA = 'siglip2-rank-routed-mlp-inference-v1'",
     b"INFERENCE_SCHEMA = 'siglip2-connected-mlp-inference-v1'"),
    (b"AUTHORITY_SCHEMA = 'siglip2-rank-routed-mlp-launch-v1'",
     b"AUTHORITY_SCHEMA = 'siglip2-connected-mlp-launch-v1'"),
    (b"SCHEMA = 'siglip2-rank-routed-mlp-v1'",
     b"SCHEMA = 'siglip2-connected-mlp-v1'"),
    (b' Only candidate adds four absolute layer26 MLP parameters. Its encoder four receive SmoothAP\n only: regression is the pinned original expression over a detached trainer.raw_features\n branch; A/C keep original regression+SmoothAP; control is unchanged. Native UNRUN.\n',
     b' Only candidate adds four absolute layer26 MLP parameters; no loss changes.\n'),
    (b'"""Frozen rank-routed connected last-MLP method. Native qualification is UNRUN.',
     b'"""Frozen connected last-MLP method. Native qualification is UNRUN.'),
)


def rejects(call, text):
    try:
        call()
    except (ValueError, TypeError, KeyError) as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError('accepted mutant: ' + text)


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def cut(raw, start, end, pin):
    """Remove exactly one pinned inserted block [start, end)."""
    assert raw.count(start) == 1 and raw.count(end) == 1, 'inserted block anchors differ'
    i, j = raw.index(start), raw.index(end)
    assert i < j and sha(raw[i:j]) == pin, 'inserted block bytes differ'
    return raw[:i] + raw[j:]


def scheduling_inverse(raw):
    """Independently invert statement lists and bytes before the historical inverses."""
    tree = ast.parse(raw)
    dump = lambda node: ast.dump(node,include_attributes=False)
    for new,old in SCHEDULING_INVERSE_EDITS:
        before = ast.parse(textwrap.dedent(new.decode())).body
        after = ast.parse(textwrap.dedent(old.decode())).body
        wanted, matches = [dump(n) for n in before], []
        for node in ast.walk(tree):
            for _,value in ast.iter_fields(node):
                if isinstance(value,list) and value and all(isinstance(n,ast.stmt) for n in value):
                    for i in range(len(value)-len(before)+1):
                        if [dump(n) for n in value[i:i+len(before)]] == wanted:
                            matches.append((value,i))
        assert len(matches) == 1, 'exact scheduling AST seam differs'
        body,index = matches[0]
        body[index:index+len(before)] = after
    assert sha(dump(tree).encode()) == SCHEDULING_BASE_AST_SHA, 'unrelated scheduling production AST edit'
    for new,old in SCHEDULING_INVERSE_EDITS:
        assert raw.count(new) == 1, 'exact scheduling byte seam differs'
        raw = raw.replace(new,old)
    assert sha(raw) == SCHEDULING_BASE_SHA, 'scheduling inverse does not reproduce 69302dab source'
    assert dump(ast.parse(raw)) == dump(tree), 'scheduling byte/AST inverses disagree'
    return raw


def sequential_inverse(raw):
    raw = scheduling_inverse(raw)
    base = (HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/'
            'rank-routed-mlp-cpu-v2-freeze/train_siglip2_rank_routed_mlp.py').read_bytes()
    assert sha(base) == '2fb4f237345f346080e499cc99d97fbc40c7087f255027ed810ef02b56537b1f'
    start,end = b'def route_full_reference(',b'def route_close('
    i,j = raw.index(start),raw.index(end)
    assert raw.count(start) == raw.count(end) == 1
    assert sha(raw[i:j]) == '6daeacf539a327826b92e70839ac686fb99ea59fd104189d66cf927723cc0cc7', 'sequential full reference bytes differ'
    a,b = base.index(start),base.index(end)
    restored = raw[:i]+base[a:b]+raw[j:]
    assert sha(restored) == sha(base), 'unrelated sequential production edit'
    return restored


def observation_inverse(raw):
    """Only fixed observation/error cleanup seams may differ from the committed source base."""
    raw = cut(raw, b'OBSERVATION_LIMITS =', b'def mapping_absent(', OBSERVATION_BLOCK_SHA)
    for new, old in OBSERVATION_INVERSE_EDITS:
        assert raw.count(new) == 1, 'exact observation/cleanup seam differs'
        raw = raw.replace(new, old)
    assert sha(raw) == OBSERVATION_BASE_SHA, 'observation inverse does not reproduce 940d37fb source'
    return raw


def rank_routing_inverse(raw):
    """Invert the whole prospective delta; the result must be the committed connected trainer."""
    raw = observation_inverse(sequential_inverse(raw))
    for new, old in REPAIR_INVERSE_EDITS:
        assert raw.count(new) == 1, 'exact repair delta differs'
        raw = raw.replace(new, old)
    raw = cut(raw, b'def _regression_runtime(', b'def strict_json(raw):', RUNTIME_BLOCK_SHA)
    raw = cut(raw, b'ROUTING_KEYS = {', b'def diagnostic(row):', ROUTE_BLOCK_SHA)
    for new, old in INVERSE_EDITS:
        assert raw.count(new) == 1, 'exact rank-routing delta differs: ' + repr(new[:60])
        raw = raw.replace(new, old)
    assert sha(raw) == CONNECTED_SHA, 'inverse does not reproduce the committed connected trainer'
    return raw


def inverse_contract():
    raw = DRIVER.read_bytes()
    restored = rank_routing_inverse(raw)
    assert restored == CONNECTED.read_bytes()
    # Unrelated or partial mutations of the prospective delta must not invert.
    for before, after in ((b"<= 1.50", b"<= 1.51"), (b"features.detach()", b"features"),
                          (b"'seconds':600 if phase", b"'seconds':601 if phase"),
                          (b"mse = routed_regression(", b"mse = foreign_regression("),
                          (b"traceback.clear_frames(primary.__traceback__)", b"pass"),
                          (b"phase_observe(context,'micro_backward_end'", b"phase_observe(context,'other_phase'"),
                          (b"'records':128", b"'records':129")):
        assert raw.count(before) >= 1, before
        try:
            rank_routing_inverse(raw.replace(before, after, 1))
        except AssertionError:
            continue
        raise AssertionError('inverse accepted unrelated mutation: ' + repr(before))
    assert sha(CONNECTED_TEST.read_bytes()) == CONNECTED_TEST_SHA, 'committed connected seam tests changed'
    print('PASS exact delta inverse: observer/cleanup -> 940d37fb -> committed connected trainer; mutants rejected')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reused_connected_seams(d):
    """Committed connected stdlib seams, run against the new driver (DRIVER rebound)."""
    old = load('_connected_seams_for_rank_routing', CONNECTED_TEST)
    old.DRIVER = DRIVER
    # source_contract pins the connected file names only; everything else must hold unchanged.
    proxy = SimpleNamespace(**vars(d))
    proxy.FILES = {'train_siglip2_connected_mlp.py', 'test_siglip2_connected_mlp.py'}
    old.source_contract(proxy)
    for name in ('initializer_selection', 'authority_falsifiers', 'lifetime_restore_math', 'overlay_optimizer_seams',
                 'owned_loader_admission', 'restore_seam', 'current_encoder_seam', 'processor_release_seam',
                 'cost_terminal_falsifiers'):
        getattr(old, name)(d)
    print('PASS reused connected seams on rank-routed driver (source/authority/lifetime/overlay/loader/restore/'
          'encoder/processor/cost)')


def identity_contract(d, c):
    assert d.SCHEMA == 'siglip2-rank-routed-mlp-v1' and d.AUTHORITY_SCHEMA == 'siglip2-rank-routed-mlp-launch-v1'
    assert d.INFERENCE_SCHEMA == 'siglip2-rank-routed-mlp-inference-v1'
    assert d.BUNDLE_SCHEMA == 'siglip2-rank-routed-mlp-bundle-v1'
    assert d.FILES == {'train_siglip2_rank_routed_mlp.py', 'test_siglip2_rank_routed_mlp.py'}
    assert len({d.SCHEMA, c.SCHEMA, d.AUTHORITY_SCHEMA, c.AUTHORITY_SCHEMA}) == 4
    # Frozen schedule/optimizer/scope constants are exactly the connected ones.
    for name in ('LAUNCH_KEYS', 'STATIC_KEYS', 'PAYLOAD_KEYS', 'IDENTITY_KEYS', 'INFERENCE_KEYS', 'MLP', 'MLP_SHAPES',
                 'ADAM', 'SEEDS', 'ARMS', 'VIEWS', 'SCOPE_SHA256', 'CONTROL_SHA256', 'BACKEND_SHA', 'WITNESS_FILES',
                 'SERVING_FILES', 'NATIVE'):
        assert getattr(d, name) == getattr(c, name), name
    assert all(d.policy(p) == c.policy(p) for p in ('cpu', 'mechanics', 'train'))
    assert {k: v for k, v in d.RECIPE.items() if k not in ('objective', 'routing')} == \
           {k: v for k, v in c.RECIPE.items() if k != 'objective'}
    assert d.RECIPE['objective'].endswith('candidate encoder SmoothAP-only') and d.RECIPE['routing'] is d.ROUTING
    assert d.RECIPE['updates'] == 128 and d.RECIPE['cost_ratio'] == 1.50 and d.RECIPE['clip'] == 1.
    assert d.ROUTING['original_trainer_sha256'] == ORIGINAL_SHA == sha(ORIGINAL.read_bytes())
    # The recipe is part of the typed method identity, so old units can never satisfy new launches.
    launch = {k: None for k in ('execution_sha256', 'original_cpu', 'actual_gradient', 'witness')}
    launch['recipe'] = d.RECIPE
    assert d.method(launch) != c.method({**launch, 'recipe': c.RECIPE}) and d.method(launch)['recipe']['routing'] == d.ROUTING
    argv = d.cli('/root', '/a.json', 'a' * 64, 'b' * 64, 'train', 'candidate', d.SEEDS[0], '/out')
    assert argv[0] == '/root/train_siglip2_rank_routed_mlp.py'
    rejects(lambda: d.check_launch({'schema': c.AUTHORITY_SCHEMA}, SimpleNamespace(execution_sha256='b' * 64, phase='cpu',
            arm='control', seed=d.SEEDS[0])), 'launch')
    tree = ast.parse(DRIVER.read_text())
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and
                n.value in ('train_siglip2_connected_mlp.py', 'test_siglip2_connected_mlp.py', 'siglip2-connected-mlp-v1')]
    print('PASS identity: new schemas/FILES/CLI/recipe+routing pins in typed method; frozen constants equal connected')


class FT:
    """Nested-list tensor stand-in with only the pinned loss_terms operations."""
    dtype = 'float32'
    device = SimpleNamespace(type='cpu')
    requires_grad = True

    def __init__(self, data):
        self.data = data

    @staticmethod
    def ew(f, a, b=None):
        if isinstance(a, list):
            return [FT.ew(f, x, b[i] if isinstance(b, list) else b) for i, x in enumerate(a)]
        return f(a) if b is None else f(a, b)

    def flat(self):
        out, stack = [], [self.data]
        while stack:
            v = stack.pop(0)
            stack[0:0] = v if isinstance(v, list) else []
            if not isinstance(v, list):
                out.append(v)
        return out

    def __sub__(self, o): return FT(FT.ew(lambda x, y: x - y, self.data, o.data))
    def square(self): return FT(FT.ew(lambda x: x * x, self.data))
    def sum(self): return FT(sum(self.flat()))
    def __truediv__(self, n): return FT(FT.ew(lambda x: x / n, self.data))
    def __mul__(self, n): return FT(FT.ew(lambda x: x * n, self.data))
    def norm(self, dim): return FT([math.sqrt(sum(x * x for x in row)) for row in self.data])
    def __gt__(self, n): return FT(FT.ew(lambda x: x > n, self.data))
    def all(self): return FT(all(self.flat()))
    def item(self): return self.data
    def __getitem__(self, index): return FT([self.data[i] for i in index.data])
    @property
    def T(self): return FT([list(c) for c in zip(*self.data)])
    def __matmul__(self, o): return FT([[sum(a * b for a, b in zip(r, c)) for c in zip(*o.data)] for r in self.data])


def fake_functional():
    torch, nn = ModuleType('torch'), ModuleType('torch.nn')
    functional = ModuleType('torch.nn.functional')
    from contextlib import contextmanager

    @contextmanager
    def autocast(*a, **kw):
        yield
    torch.autocast, torch.float32, torch.nn, nn.functional = autocast, 'float32', nn, functional
    torch.tensor = lambda values, device=None: FT(list(values))
    torch.isfinite = lambda t: FT(FT.ew(math.isfinite, t.data))
    functional.normalize = lambda t, dim: FT([[x / math.sqrt(sum(y * y for y in row)) for x in row] for row in t.data])
    return {'torch': torch, 'torch.nn': nn, 'torch.nn.functional': functional}


def regression_falsifiers(d):
    """Real pinned loss_terms vs the derived regression function, plus every guard."""
    original = d.load_authenticated('_rank_routed_original_test', ORIGINAL, ORIGINAL_SHA, {})
    try:
        path = Path(original.__file__)
        witness = SimpleNamespace(TRAIN_CODE={'train_siglip2_identity_diversity.py': ORIGINAL_SHA}, RTOL=1e-5, ATOL=1e-6)
        context = {'trainer': original, 'guards': {str(path): ORIGINAL_SHA}, 'witness': witness}
        before = dict(vars(original))
        runtime = d._regression_runtime(context)
        context['routing'] = runtime
        assert vars(original).keys() == before.keys() and all(vars(original)[k] is v for k, v in before.items()), \
            'regression runtime rebound the pinned trainer'
        derived = ast.parse(runtime.source).body[0]
        names = {n.id for n in ast.walk(derived) if isinstance(n, ast.Name)}
        assert not names & {'ranking_gallery', 'smooth_ap_terms', 'ranking_membership', 'bank', 'membership', 'scores', 'rank'}
        assert "mse = (raw - target).square().sum() / (rows * state['teachers']['e0'])" in runtime.source
        # Statement-by-statement authenticity against the pinned source.
        source = function(ast.parse(ORIGINAL.read_bytes()), 'loss_terms')
        dump = lambda n: ast.dump(n, include_attributes=False)
        assert [dump(n) for n in derived.body[:2]] == [dump(source.body[0]), dump(source.body[2])]
        assert dump(derived.body[2].items[0]) == dump(source.body[5].items[0])
        assert [dump(n) for n in derived.body[2].body[:4]] == [dump(n) for n in source.body[5].body[:4]]
        assert ast.unparse(derived.body[2].body[4]) == "require(torch.isfinite(mse).item(), 'nonfinite routed regression')"
        assert ast.unparse(derived.body[3]) == 'return mse' and len(derived.body) == 4 and len(derived.body[2].body) == 5
        # Execute pinned original (ranking stubs, no positives) and the derived function on the same stand-ins.
        P = FT([[1., 0.], [0., 2.], [3., 1.]])
        state = {'teachers': {'P': P, 'e0': 2.}, 'target': FT([0, 1, 2, 1]), 'ranking_bank': 'bank'}
        anchors, raw = [0, 2, 3], FT([[1., 2.], [3., 4.], [.5, .25]])
        stubs = {'ranking_membership': lambda bank, a: {'positive': [[] for _ in a], 'eligible_counts': []},
                 'ranking_gallery': lambda c, s: FT([[1., 0.], [0., 1.]])}
        old = original.loss_terms
        ns = {**vars(original), **stubs}
        pinned = FunctionType(old.__code__, ns, 'loss_terms', old.__defaults__)
        with patch.dict(sys.modules, fake_functional()):
            mse, rank, facts = pinned(context, state, raw, anchors, 3)
            got = runtime.regression_terms(context, {k: v for k, v in state.items() if k != 'ranking_bank'}, raw, anchors, 3)
            assert got.data == mse.data > 0 and facts['active'] == 0
            expected = sum((r - t) ** 2 for row, a in zip(raw.data, anchors) for r, t in zip(row, P.data[state['target'].data[a]]))
            assert math.isclose(got.data, expected / (128 * 2.))
            rejects(lambda: runtime.regression_terms(context, state, FT([[math.inf, 1.], [1., 1.], [1., 1.]]), anchors, 3),
                    'finite nonzero FP32 raw')
            rejects(lambda: runtime.regression_terms(context, state, raw, anchors, 65), 'full B64')
        # Pins, live function, origin, bytes and guard mutants.
        for key, text in (('loss_terms_ast_sha256', 'original AST'), ('regression_ast_sha256', 'adapted AST'),
                          ('original_trainer_sha256', 'source guard')):
            with patch.dict(d.ROUTING, {key: '0' * 64}):
                rejects(lambda: d._regression_runtime(context), text)
        rejects(lambda: d._regression_runtime({**context, 'guards': {}}), 'source guard')
        rejects(lambda: d._regression_runtime({**context, 'witness': SimpleNamespace(TRAIN_CODE={
            'train_siglip2_identity_diversity.py': '0' * 64})}), 'source guard')
        with patch.object(original, 'loss_terms', lambda *a: None):
            rejects(lambda: d._regression_runtime(context), 'live function')
        foreign = FunctionType(old.__code__, dict(vars(original)), 'loss_terms', old.__defaults__)
        with patch.object(original, 'loss_terms', foreign):
            rejects(lambda: d._regression_runtime(context), 'live function')
        saved = old.__code__
        old.__code__ = (lambda *a: None).__code__
        try:
            rejects(lambda: d._regression_runtime(context), 'live function')
        finally:
            old.__code__ = saved
        old.__defaults__ = (None,)
        try:
            rejects(lambda: d._regression_runtime(context), 'live function')
        finally:
            old.__defaults__ = None
        with patch.object(original.__spec__, 'origin', '/foreign.py'):
            rejects(lambda: d._regression_runtime(context), 'origin')
        with patch.dict(sys.modules, {original.__name__: ModuleType(original.__name__)}):
            rejects(lambda: d._regression_runtime(context), 'origin')
        with patch.dict(vars(original), regression_terms=None):
            rejects(lambda: d._regression_runtime(context), 'private namespace')
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp)/'foreign.py'
            changed.write_bytes(ORIGINAL.read_bytes() + b'\n')
            with patch.object(original, '__file__', str(changed)), patch.object(original.__spec__, 'origin', str(changed)), \
                    patch.dict(context['guards'], {str(changed): ORIGINAL_SHA}):
                rejects(lambda: d._regression_runtime(context), 'bytes')
        assert vars(original)['loss_terms'] is old and old.__code__ is saved
        live_state = {**state, 'arm': 'candidate', 'device': 'cpu', 'A': FT([1.]), 'C': FT([1.])}
        with patch.object(original, 'helper_guard', lambda c: None):
            # Re-admit this explicit source stand-in; each mutant starts after construction.
            runtime = d._regression_runtime(context)
            context['routing'] = runtime
            with patch.dict(sys.modules, fake_functional()):
                assert d.routed_regression(context, live_state, raw, anchors, 3).data == mse.data
                rejects(lambda: d.routed_regression({**context}, live_state, raw, anchors, 3), 'live callable')
                with patch.dict(context, routing=SimpleNamespace(regression_terms=runtime.regression_terms)):
                    rejects(lambda: d.routed_regression(context, live_state, raw, anchors, 3), 'live callable')
                fn = runtime.regression_terms
                with patch.object(runtime, 'regression_terms', lambda *a: None):
                    rejects(lambda: d.routed_regression(context, live_state, raw, anchors, 3), 'live callable')
                saved_code = fn.__code__
                fn.__code__ = (lambda *a: None).__code__
                try:
                    rejects(lambda: d.routed_regression(context, live_state, raw, anchors, 3), 'live callable')
                finally:
                    fn.__code__ = saved_code
                fn.__defaults__ = (None,)
                try:
                    rejects(lambda: d.routed_regression(context, live_state, raw, anchors, 3), 'live callable')
                finally:
                    fn.__defaults__ = None
                for name in ('loss_denominators', 'require'):
                    with patch.dict(fn.__globals__, {name: lambda *a: None}):
                        rejects(lambda: d.routed_regression(context, live_state, raw, anchors, 3), 'private globals')
                    helper = fn.__globals__[name]
                    helper_code = helper.__code__
                    helper.__code__ = (lambda *a: None).__code__
                    try:
                        rejects(lambda: d.routed_regression(context, live_state, raw, anchors, 3), 'live callable')
                    finally:
                        helper.__code__ = helper_code
                detached = FT(raw.data)
                detached.requires_grad = False
                rejects(lambda: d.routed_regression(context, live_state, detached, anchors, 3), 'live routing context')
    finally:
        sys.modules.pop(original.__name__, None)
    print('PASS regression derivation: pinned loss_terms == derived mse (executed), statement-by-statement, guards/mutants')


class Vec:
    """Flat FP32 stand-in; arithmetic only."""
    def __init__(self, values, dtype='float32'):
        self.v, self.dtype = [float(x) for x in values], dtype
        self.requires_grad = True
        self.grad_fn = None
        self.shape = (len(self.v),)
    def _zip(self, o, f): return Vec([f(a, b) for a, b in zip(self.v, o.v, strict=True)], self.dtype)
    def __add__(self, o): return self._zip(o, lambda a, b: a + b)
    def __sub__(self, o): return self._zip(o, lambda a, b: a - b)
    def add_(self, o): self.v = [a + b for a, b in zip(self.v, o.v, strict=True)]; return self
    def detach(self):
        result = Vec(self.v, self.dtype)
        result.requires_grad = False
        return result
    clone = detach
    def float(self): return self
    def double(self): return self
    def to(self, device): return self
    def norm(self): return math.sqrt(sum(x * x for x in self.v))
    def abs(self): return Vec([abs(x) for x in self.v], self.dtype)
    def max(self): return max(self.v)


class Out:
    """Scalar loss stand-in carrying a gradient oracle (callable inputs -> gradients)."""
    def __init__(self, value, grads=None, parts=None):
        self.value, self.grads, self.parts, self.requires_grad = value, grads, parts, True
        self.v, self.shape, self.dtype, self.grad_fn = [value], (), 'float32', None
    def detach(self):
        result = Out(self.value)
        result.requires_grad = False
        return result
    def __add__(self, o): return Out(self.value + o.value, parts=(self, o))
    def __sub__(self, o): return Out(self.value - o.value)
    def abs(self): return Out(abs(self.value))
    def max(self): return self.value
    def __float__(self): return float(self.value)


class FakeTorch:
    float32 = 'float32'

    def __init__(self):
        self.retain = []
        self.nn = SimpleNamespace(Parameter=lambda t: t)
        self.autograd = SimpleNamespace(grad=self.grad)

    @staticmethod
    def autocast(*a, **kw):
        from contextlib import nullcontext
        return nullcontext()

    def grad(self, out, inputs, retain_graph=False, allow_unused=False):
        self.retain.append(retain_graph)
        if out.grads is None:
            assert len(out.parts) == 2
            parts = [self.grad(p, inputs, retain_graph, True) for p in out.parts]
            result = tuple(None if a is b is None else b if a is None else a if b is None else a+b
                           for a,b in zip(*parts,strict=True))
        else:
            result = tuple(out.grads(tuple(inputs)))
        assert allow_unused or all(g is not None for g in result), 'unused input without allow_unused'
        return tuple(None if g is None else Vec(g.v) for g in result)

    @staticmethod
    def zeros_like(p): return Vec([0.] * len(p.v))
    @staticmethod
    def tensor(value, dtype): return Out(value)
    @staticmethod
    def cat(chunks, dim): return Vec([x for chunk in chunks for x in chunk.v])
    @staticmethod
    def isfinite(x): return SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: all(math.isfinite(a) for a in x.v)))
    @staticmethod
    def allclose(a, b, rtol, atol): return all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(a.v, b.v, strict=True))
    @staticmethod
    def count_nonzero(x): return SimpleNamespace(item=lambda: sum(1 for a in x.v if a != 0))
    @staticmethod
    def equal(a, b): return a.v == b.v if isinstance(a, Vec) else a.value == b.value


class Scenario:
    """Six members (A, C, four encoder tensors) with consistent routed/original/rank gradient tables."""
    def __init__(self):
        self.torch = FakeTorch()
        self.members = [Vec([1., 2., 3.]) for _ in range(6)]
        self.index = {id(m): k for k, m in enumerate(self.members)}
        self.rank = [Vec([.5 + i, -.25, 1.]) for i in range(6)]
        ac = [Vec([2., -1., .5]), Vec([.25, .75, -2.])]
        self.routed_table = dict(enumerate(ac))
        self.original_table = dict(enumerate([*ac, *(Vec([1. + j, .5, -.5]) for j in range(4))]))
        lookup = lambda table: (lambda ins: tuple(table().get(self.index.get(id(i))) for i in ins))
        self.routed = Out(1.25, lookup(lambda: self.routed_table))
        self.original = Out(1.25, lookup(lambda: self.original_table))
        self.rank_out = Out(.5, lambda ins: tuple(self.rank[self.index[id(i)]] for i in ins))
        self.context = {'witness': SimpleNamespace(RTOL=1e-5, ATOL=1e-6), 'legacy': {'quadratic': None},
                        'trainer': SimpleNamespace(helper_guard=lambda c: None, loss_terms=self.loss_terms),
                        'connected': SimpleNamespace(raw_features=lambda features, head, A, means, C, *a:
                                                     Out(0., parts=(features, A, C)))}
        self.state = {'device': 'cpu', 'A': self.members[0], 'C': self.members[1], 'head_object': 1, 'means': 2, 'mu_train': 3}
        self.split_value, self.split_share, self.split_enc, self.leak, self.kept = .5, .4, 0., False, []

    def loss_terms(self, context, state, raw, anchors, K):
        assert state['A'] is not self.members[0] and state['C'] is not self.members[1], 'gallery must own independent A/C'
        query = [Vec([self.split_share * x for x in r.v]) for r in self.rank[:2]]
        gallery = [Vec([(1. - self.split_share) * x for x in r.v]) for r in self.rank[:2]]
        enc = [Vec([x + self.split_enc for x in r.v]) for r in self.rank[2:]]

        def grads(inputs):
            assert len(inputs) == 8 and inputs[4:] == tuple(self.members[2:]) and len({id(i) for i in inputs[:4]}) == 4
            return (*query, *gallery, *enc)
        out = Out(self.split_value, grads)
        if self.leak:
            self.kept.append(out)
        return Out(1.25, parts=(raw, state['A'], state['C'])), out, {'active': 3}

    def view(self, d, device='cpu', micros=4, **overrides):
        route = d.route_accumulators(self.torch, self.members)
        self.state['device'] = device
        if device == 'cpu':
            route['full'] = self.full()
        with patch.dict(sys.modules, {'torch': self.torch}):
            for i in range(micros):
                args = dict(torch=self.torch, context=self.context, state=self.state, members=self.members, route=route,
                            ranking=tuple(self.rank), original=self.original, mse=self.routed, rank=self.rank_out,
                            features=Out(0.), anchors=[1], K=1, selected={'active': 3}, first=i == 0)
                args.update(overrides)
                d.route_micro(**args)
        return route

    def full(self):
        ac = [Vec([2., -1., .5]), Vec([.25, .75, -2.])]
        return {'gradients':[Vec([4. * (x+y) for x,y in zip(a.v,r.v,strict=True)]) for a,r in zip(ac,self.rank[:2],strict=True)] +
                            [Vec([4. * x for x in r.v]) for r in self.rank[2:]],
                'scalars':{'mse':Out(5.),'rank':Out(2.),'loss':Out(7.)}}


def split_lifetime_falsifier(d):
    sc = Scenario()
    sc.view(d, 'cpu', micros=1)
    print('PASS split MSE owns raw/query A/C graph and dies before the lifetime gate')


def independent_reference_falsifier(d):
    sc = Scenario()
    route = sc.view(d, 'cpu')
    route['routed'][0].v[0] += 1.
    route['original'][0].v[0] += 1.
    rejects(lambda: d.route_view(sc.torch, sc.context, route), 'independent full B64 routed A')
    for key,text in (('scalars','independent full B64 mse'),('ranking','independent full B64 encoder')):
        route = sc.view(d,'cpu')
        if key == 'scalars':
            route[key]['mse'] += .25
        else:
            route[key][0].v[0] += .25
        rejects(lambda: d.route_view(sc.torch,sc.context,route),text)
    print('PASS equal micro accumulator perturbation fails independent full reference')


def optimizer_gradient_falsifier(d):
    sc = Scenario()
    names = d.parameter_roles('candidate')[0]
    route = sc.view(d, 'cuda')
    ranking = [Vec([4. * x for x in g.v]) for g in sc.rank]
    for p,g in zip(sc.members, (*route['routed'], *ranking[2:]), strict=True):
        p.grad = g.detach()
    sc.members[2].grad.v[0] += 1.
    rejects(lambda: d.route_optimizer(sc.torch, sc.context, sc.members, names, ranking, route['routed']),
            'actual unscaled optimizer '+names[2])
    sc.members[2].grad = ranking[2].detach()
    sc.members[0].grad.v[0] += 1.
    rejects(lambda: d.route_optimizer(sc.torch, sc.context, sc.members, names, ranking, route['routed']),
            'actual unscaled optimizer '+names[0])
    sc.members[0].grad = route['routed'][0].detach()
    fact = d.route_optimizer(sc.torch, sc.context, sc.members, names, ranking, route['routed'])
    assert fact['max_abs'] == {n:0. for n in names}
    print('PASS actual optimizer encoder/A/C gradient mutants fail with independent witnesses unchanged')


def routing_witness_falsifiers(d):
    for device in ('cpu', 'cuda'):
        sc = Scenario()
        route = sc.view(d, device)
        fact = d.route_view(sc.torch, sc.context, route)
        assert fact['micro_checks'] == 4 and fact['encoder_regression_gradient_nonzero'] == 0
        assert (fact['query_gallery'] is None) == (device == 'cuda') and fact['tolerance'] == {'rtol': 1e-5, 'atol': 1e-6}
        d.check_routing(sc.context, fact, {'arm': 'candidate', 'device': device})
        assert all(sc.torch.retain) and len(sc.torch.retain) == 8 + (device == 'cpu'), \
            'every extra autograd.grad must retain the graph for the real backward'
    # Gradient-table mutants.
    def mutate(sc, name):
        if name == 'encoder_nonzero': sc.routed_table[2] = Vec([1., 0., 0.])
        if name == 'a_mismatch': sc.original_table[0] = Vec([2.001, -1., .5])
        if name == 'unused': sc.original_table = {}
        if name == 'zero': sc.routed_table[0] = Vec([0., 0., 0.])
        if name == 'nonfinite': sc.original_table[1] = Vec([math.inf, 0., 0.])
        if name == 'total_cancel':  # regression parts agree, but cancellation against the ranking part breaks the total
            sc.rank[0] = Vec([-2., -.25, 1.])
            sc.original_table[0] = Vec([2.000015, -1., .5])
    for name, text in (('encoder_nonzero', 'routed regression reached the encoder'),
                       ('a_mismatch', 'correspondence differs: regression A'), ('unused', 'absent, zero or nonfinite'),
                       ('zero', 'absent, zero or nonfinite'), ('nonfinite', 'absent, zero or nonfinite'),
                       ('total_cancel', 'correspondence differs: micro A')):
        sc = Scenario()
        mutate(sc, name)
        rejects(lambda: sc.view(d, 'cuda'), text)
    sc = Scenario()
    route = sc.view(d, 'cuda')
    route['original'][0].v[0] += 1.
    rejects(lambda: d.route_view(sc.torch, sc.context, route), 'accumulated micro A')
    route = sc.view(d, 'cuda')
    route['unrouted'][2] = Vec([0., 0., 0.])
    rejects(lambda: d.route_view(sc.torch, sc.context, route), 'all four encoder tensors')
    # CPU query/gallery split mutants.
    for field, value, text in (('split_value', .75, 'split loss/membership'), ('split_share', 0., 'both reach'),
                               ('split_enc', .5, 'query route only')):
        sc = Scenario()
        setattr(sc, field, value)
        rejects(lambda: sc.view(d, 'cpu', micros=1), text)
    sc = Scenario()
    sc.leak = True
    rejects(lambda: sc.view(d, 'cpu', micros=1), 'split lifetime survived')
    sc = Scenario()
    sc.context['trainer'].loss_terms = lambda c, s, raw, a, K: (None, Out(.5, lambda ins: ()), {'active': 4})
    rejects(lambda: sc.view(d, 'cpu', micros=1), 'split loss/membership')
    # Receipt predicate mutants.
    sc = Scenario()
    good = d.route_view(sc.torch, sc.context, sc.view(d, 'cpu'))
    cpu = {'arm': 'candidate', 'device': 'cpu'}
    d.check_routing(sc.context, good, cpu)
    d.check_routing(sc.context, None, {'arm': 'control', 'device': 'cuda'})
    rejects(lambda: d.check_routing(sc.context, good, {'arm': 'control', 'device': 'cpu'}), 'control carries no')
    rejects(lambda: d.check_routing(sc.context, None, cpu), 'step1 witness')
    rejects(lambda: d.check_routing(sc.context, {**good, 'full_micro':None}, {'arm': 'candidate', 'device': 'cuda'}), 'split witness')
    for key, value in (('micro_checks', 3), ('micro_checks', True), ('encoder_regression_gradient_nonzero', 1),
                       ('tolerance', {'rtol': 1e-4, 'atol': 1e-6}), ('extra', 1),
                       ('unrouted_regression_encoder_norms', {**good['unrouted_regression_encoder_norms'], d.MLP[0]: 0.}),
                       ('A_C_total_max_abs', {'A': float('nan'), 'C': 0.}), ('accumulated_A_C_total_max_abs', {'A': 0.}),
                       ('query_gallery', {'A': 0., 'C': 0.}), ('query_gallery', None)):
        rejects(lambda: d.check_routing(sc.context, {**good, key: value}, cpu), 'witness differs')
    for value in (None, {}, {**good['full_micro'], 'batch': True}, {**good['full_micro'], 'micro': 32},
                  {**good['full_micro'], 'encoder_ranking': {}}, {**good['full_micro'], 'scalars': {'loss':0.}},
                  {**good['full_micro'], 'routed_A_C': {'A':float('nan'),'C':0.}}):
        rejects(lambda: d.check_routing(sc.context, {**good, 'full_micro':value}, cpu), 'independent full/micro')
    print('PASS routing witness: routed/original A/C correspondence, encoder-regression absent, unrouted reaches encoder, '
          'query/gallery split, full-view sums, retain_graph, lifetime, receipt mutants')


class Micro:
    """Executes the real extracted micro-loop (and lifetime gate) of update() on stand-ins."""
    def __init__(self, d, arm='candidate', step=2, device='cpu', **flags):
        tree = ast.parse(DRIVER.read_text())
        update = function(tree, 'update')
        loop = next(n for n in ast.walk(update) if isinstance(n, ast.For) and ast.unparse(n.iter) == 'range(0, 64, 16)')
        gate = next(n for n in update.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'released')
        self.loop = compile(ast.Module(body=[loop], type_ignores=[]), '<actual update micro loop>', 'exec')
        self.gate = compile(ast.Module(body=[gate], type_ignores=[]), '<actual update lifetime gate>', 'exec')
        self.d, self.arm, self.step, self.flags = d, arm, step, flags
        self.sc = sc = Scenario()
        sc.state.update(device=device, model=lambda pixel_values: SimpleNamespace(pooler_output=Vec([1., 2., 3.])),
                        processor_object=None)
        self.calls, self.backs, self.kept, self.regressions = [], [], [], []
        main = lambda state: state['A'] is sc.members[0]

        def raw_features(context, state, features):
            if flags.get('record', True):
                self.calls.append(('detached' if features is not self.last_features else 'live', id(features)))
            result = Vec([x + flags.get('raw_offset', 0.) for x in features.v])
            result.requires_grad = flags.get('detached_grad', True)
            return result

        def loss_terms(context, state, raw, anchors, K):
            if not main(state):
                return sc.loss_terms(context, state, raw, anchors, K)
            if flags.get('record', True):
                self.calls.append(('loss', raw))
            factor = len(anchors)/16
            scaled = lambda out: lambda inputs: tuple(None if g is None else Vec([factor*x for x in g.v])
                                                     for g in out.grads(inputs))
            mse = Out(1.25*factor, scaled(sc.original), parts=(raw,))
            self.kept.append(mse) if flags.get('leak') else None
            return mse, Out(.5*factor, scaled(sc.rank_out), parts=(raw,)), {'active': 3}

        def regression(context, state, detached, anchors, K):
            if flags.get('record', True):
                self.regressions.append(detached)
            factor = len(anchors)/16
            grads = lambda inputs: tuple(None if g is None else Vec([factor*x for x in g.v])
                                         for g in sc.routed.grads(inputs))
            return Out(flags.get('mse', 1.25)*factor, grads, parts=(detached,))

        def connected_raw(features, *rest):
            assert rest[-1] == 'guard' and rest[0] is sc.state['head_object']
            result = Vec(features.v)
            result.requires_grad = True
            return result

        def pixels_for(trainer, context, state, processor, anchors, view):
            return SimpleNamespace(to=lambda device: 'pixels'), 'facts'

        def normalize(t, dim):
            t.requires_grad = flags.get('features_grad', True)
            self.last_features = t
            return t
        self.last_features = None
        trainer = SimpleNamespace(raw_features=raw_features, loss_terms=loss_terms, helper_guard=lambda c: 'guard')
        context = {**sc.context, 'trainer': trainer, 'witness': SimpleNamespace(RTOL=1e-5, ATOL=1e-6, pixels_for=pixels_for),
                   'legacy': {'quadratic': None}, 'connected': SimpleNamespace(raw_features=connected_raw)}
        if arm == 'candidate':
            namespace = dict(regression.__globals__)
            regression = FunctionType(regression.__code__, namespace, regression.__name__, regression.__defaults__, regression.__closure__)
            namespace['regression_terms'] = regression
            context['routing'] = SimpleNamespace(regression_terms=regression)
            context['routing_binding'] = (context, context['routing'], regression, namespace, dict(namespace),
                vars(trainer), dict(vars(trainer)), [(regression, regression.__code__, regression.__defaults__,
                regression.__kwdefaults__, regression.__closure__, regression.__builtins__, namespace,
                regression.__name__, regression.__qualname__, regression.__module__)])
            sc.state['arm'] = arm
        sc.context.update(context)
        scaler = SimpleNamespace(scale=lambda loss: SimpleNamespace(backward=lambda: self.backs.append(loss)))
        sc.torch.isfinite = FakeTorch.isfinite
        self.ns = {**vars(d), 'torch': sc.torch, 'F': SimpleNamespace(normalize=normalize), 'state': sc.state,
                   'context': context, 'trainer': trainer, 'connected': context['connected'], 'scaler': scaler,
                   'batch': list(range(64)), 'view': 'canonical', 'step': step, 'K': 1, 'candidate': arm == 'candidate',
                   'members': sc.members, 'ranking': [Vec([0.] * 3) for _ in sc.members], 'mse_sum': 0., 'rank_sum': 0.,
                   'membership': [], 'released': [], 'parity_checks': 0,
                   'route': d.route_accumulators(sc.torch, sc.members) if arm == 'candidate' and step == 1 else None}
        if arm == 'candidate' and step == 1 and device == 'cpu':
            self.ns['route']['full'] = sc.full()

    def run(self):
        with patch.dict(sys.modules, {'torch': self.sc.torch}):
            exec(self.loop, self.ns)
        return self.ns

    def scrub(self):
        """Drop the harness's own references so only update()'s deletions decide lifetime."""
        self.backs.clear(), self.regressions.clear(), self.calls.clear()
        self.last_features = None


def sequential_reference_falsifier(d):
    """Execute the helper; two independent graphs must each see one encoder backward."""
    import gc
    import weakref
    m = Micro(d, step=1, record=False)
    sc, context, state = m.sc, m.ns['context'], m.sc.state
    state['counter'] = 0
    rng, graphs, calls, injected, phases, ac_calls, kept, audits = [23.], [], [], {}, [], [], [], []
    sc.torch.random = SimpleNamespace(get_rng_state=lambda: Vec(rng),
        set_rng_state=lambda t: rng.__setitem__(slice(None), t.v))
    context['witness'].pixels_for = lambda trainer, context, state, processor, anchors, view: (Vec(anchors), [])
    def model(pixel_values):
        gc.collect()
        assert all(ref() is None for ref in graphs), 'previous B64 graph survived replay'
        assert len(pixel_values.v) == 64
        features = Vec(pixel_values.v)
        features.forward_id = len(graphs)
        if injected.get("replay") and len(graphs) == 1:
            features.v[0] += 1.
        graphs.append(weakref.ref(features))
        rng[0] += 7.
        return SimpleNamespace(pooler_output=features)
    state['model'] = model
    raw = context['connected'].raw_features
    def connected(features, *args):
        result = raw(features, *args)
        result.graph = features
        if injected.get('raw') and features.forward_id == 1:
            result.v[0] += 1.
        return result
    context['connected'].raw_features = connected
    genuine_loss = context['trainer'].loss_terms
    def loss_terms(*args):
        original,rank,selected = genuine_loss(*args)
        original.provenance,rank.provenance = 'original','rank'
        if args[2].graph.forward_id == 1:
            if injected.get('rank'):
                rank.value += 1.
                rank.v[0] += 1.
            if injected.get('membership'):
                selected = {**selected,'active':selected['active']+1}
        return original,rank,selected
    context['trainer'].loss_terms = loss_terms
    binding = list(context['routing_binding'])
    binding[6] = dict(vars(context['trainer']))
    context['routing_binding'] = tuple(binding)
    genuine_grad = sc.torch.autograd.grad
    def grad(out, inputs, retain_graph=False, allow_unused=False):
        def reachable(value):
            if hasattr(value, 'graph'):
                return {value.graph.forward_id}
            return set().union(*(reachable(p) for p in (getattr(value, 'parts', None) or ())))
        owners = reachable(out)
        if len(inputs) == 2:
            assert tuple(inputs) == tuple(sc.members[:2]) and retain_graph is True
            assert getattr(out.parts[0],'provenance',None) == 'original' and getattr(out.parts[1],'provenance',None) == 'rank', 'original A/C objective substituted'
            assert len(owners) == 1 and next(iter(owners)) % 2 == 0, 'A/C reference did not use first graph'
            ac_calls.append(next(iter(owners)))
        encoder = any(p is member for p in inputs for member in sc.members[2:]) and bool(owners)
        if encoder:
            assert len(owners) == 1
            owner = next(iter(owners))
            assert not any(c[0] == owner for c in calls), 'one encoder backward per B64 graph required'
            calls.append((owner, 'total' if out.grads is None else 'rank', retain_graph))
            if injected.get('total') or injected.get('total_state') or injected.get('leak_state'):
                if injected.get('total_state') or injected.get('leak_state'):
                    state['A'].v[0] += 1.
                if injected.get('leak_state'):
                    kept.append(graphs[owner]())
                raise ValueError('primary total backward fault')
            if injected.get('replay_backward') and owner == 1:
                raise ValueError('primary replay backward fault')
        result = genuine_grad(out, inputs, retain_graph=retain_graph, allow_unused=allow_unused)
        if encoder and owner == 1 and (injected.get('gradient') or injected.get('zero')):
            for value in result[2:]:
                value.v = [0. if injected.get('zero') else x*2. for x in value.v]
        if encoder and owner % 2 == 0:
            for value in result:
                value.reference_total = True
        return result
    sc.torch.autograd.grad = grad
    functional = ModuleType('torch.nn.functional')
    functional.normalize = lambda t, dim: t
    nn = ModuleType('torch.nn')
    nn.functional = functional
    def saved(context, state, identity):
        audits.append(tuple(state['A'].v))
        return {'A': state['A'].v, 'C': state['C'].v, 'counter': state['counter'], 'rng': list(rng)}
    def fact(context, value):
        if isinstance(value, (Vec, Out)):
            return repr((value.shape, value.dtype, value.v))
        return repr(value)
    with patch.dict(sys.modules, {'torch': sc.torch, 'torch.nn': nn, 'torch.nn.functional': functional}), \
            patch.object(d, 'payload', saved), patch.object(d, 'fingerprint', fact), \
            patch.object(d,'phase_observe',lambda context,name,**kw:phases.append(name)):
        for view in d.VIEWS:
            result = d.route_full_reference(sc.torch, context, state, {}, sc.members, list(range(64)), 1, view)
            assert rng == [23.]
            assert result['scalars']['loss'].value == 7.
            assert [g.v for g in result['gradients']] == [g.v for g in sc.full()['gradients']]
            gc.collect()
            assert all(ref() is None for ref in graphs)
    assert ac_calls == [0,2], 'original A/C reference provenance missing'
    assert calls == [(i, kind, False) for i, kind in enumerate(('total', 'rank', 'total', 'rank'))]
    assert phases == ['full_begin','full_pixels','full_forward','full_loss','full_original_A_C_gradients',
                      'full_total_gradients','full_pass_released','replay_pixels','replay_forward',
                      'replay_ranking_gradients','replay_pass_released','full_graph_released'] * 2
    assert 113-2*9+len(phases) == 119 <= d.OBSERVATION_LIMITS['records']
    result_refs = []
    vec_detach,out_detach = Vec.detach,Out.detach
    def detached_vec(value):
        result = vec_detach(value)
        if getattr(value,'reference_total',False):
            result_refs.append(weakref.ref(result))
        return result
    def detached_out(value):
        result = out_detach(value)
        result_refs.append(weakref.ref(result))
        return result
    for mode,message in (('replay','full B64 replay pixels/features differ'),
                         ('raw','full B64 replay raw differs'),
                         ('rank','full B64 replay ranking/membership differs'),
                         ('membership','full B64 replay ranking/membership differs'),
                         ('gradient','full encoder ranking-only'),
                         ('zero','full ranking encoder gradient absent'),
                         ('replay_backward','primary replay backward fault'),
                         ('total','primary total backward fault'),
                         ('total_state','primary total backward fault'),
                         ('leak_state','primary total backward fault')):
        graphs.clear()
        calls.clear()
        ac_calls.clear()
        audits.clear()
        result_refs.clear()
        injected.clear()
        injected[mode] = True
        original_A = list(state['A'].v)
        with patch.dict(sys.modules, {'torch':sc.torch,'torch.nn':nn,'torch.nn.functional':functional}), \
                patch.object(d,'payload',saved), patch.object(d,'fingerprint',fact), \
                patch.object(Vec,'detach',detached_vec), patch.object(Out,'detach',detached_out):
            try:
                d.route_full_reference(sc.torch,context,state,{},sc.members,list(range(64)),1,d.VIEWS[0])
            except ValueError as retained_error:
                assert message in str(retained_error), str(retained_error)
                assert rng == [23.], 'failed replay did not restore RNG'
                gc.collect()
                if mode == 'leak_state':
                    assert any(ref() is not None for ref in graphs), 'external owner unexpectedly removed'
                    assert audits[-1] == tuple(state['A'].v) != tuple(original_A), 'lifetime failure skipped state audit'
                else:
                    assert all(ref() is None for ref in graphs), 'graph survived while primary traceback retained'
                assert all(ref() is None for ref in result_refs), 'detached result survived a failed second pass'
                notes = '\n'.join(getattr(retained_error,'__notes__',()))
                if mode == 'leak_state':
                    assert 'graph lifetime survived' in notes, 'external owner lifetime rejection missing'
                if mode in ('total_state','leak_state'):
                    assert 'changed initialized state/RNG' in notes, 'state mutation audit missing'
                else:
                    assert not notes, 'cleanup falsely labelled traceback ownership as a leak: '+notes
            else:
                raise AssertionError('injected failure accepted')
            finally:
                state['A'].v = original_A
                kept.clear()
        first_fault = mode in ('total','total_state','leak_state')
        assert len(graphs) == (1 if first_fault else 2), 'failure continued into another graph'
        if mode in ('replay','raw','rank','membership'):
            assert calls == [(0,'total',False)], 'mismatch reached replay backward'
        gc.collect()
        assert all(ref() is None for ref in graphs), 'failed graph retained after outer error release'
    print('PASS genuine total/rank calls on distinct released B64 graphs; native memory fit UNRUN')


def allocation_schedule_falsifier(d):
    """Reverting the prelude must put 18 unused tensors back beside the reference."""
    import gc
    import weakref
    base = (HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/'
            'rank-routed-mlp-cpu-v3-freeze/train_siglip2_rank_routed_mlp.py').read_bytes()
    assert sha(base) == '69302dab4a808a259a381d9b649ac58bd1d7a3c5dffe2b8aa6982131dd45cd5d'

    def probe(raw, arm, device, step, delayed, fault=False):
        update = function(ast.parse(raw), 'update')
        view = next(n for n in update.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'VIEWS')
        end = next(i for i,n in enumerate(view.body) if isinstance(n, ast.For))
        code = compile(ast.Module(body=view.body[:end], type_ignores=[]), '<actual per-view prelude>', 'exec')
        sc, refs, events, result_refs, result_ids = Scenario(), [], [], [], []
        members = [Vec(range(i+1)) for i in range(6 if arm == 'candidate' else 2)]
        outer = [Vec([7.] * len(p.v)) for p in members]
        routed = [Vec([11.] * len(p.v)) for p in members[:2]] if arm == 'candidate' else None
        outer_values = [list(t.v) for t in outer + (routed or [])]

        def zero(p):
            value = FakeTorch.zeros_like(p)
            refs.append(weakref.ref(value))
            events.append('allocate')
            return value

        def reference(*args):
            events.append('reference')
            assert args == (sc.torch, ns['context'], ns['state'], ns['identity'], members, ns['batch'], 1, ns['view'])
            alive = sum(ref() is not None for ref in refs)
            assert alive == (0 if delayed else 18), 'unused per-view buffers live during reference: %d' % alive
            if fault:
                raise ValueError('reference fault')
            result = sc.full()
            result_ids.append(id(result))
            result_refs.extend(weakref.ref(t) for t in (*result['gradients'], *result['scalars'].values()))
            return result

        def observe(context, name, **kw):
            events.append(name)
            assert [key for key,_ in kw['groups']] == ['ranking_total','routed_total','ranking_view',
                                                     'routed','original','unrouted','encoder_ranking']
            assert kw['groups'][0][1] is outer and kw['groups'][1][1] is routed

        def optimizer_work(*args, **kw):
            raise AssertionError('prelude performed optimizer work')

        sc.torch.zeros_like = zero
        ns = {**vars(d), 'torch':sc.torch, 'context':{}, 'state':{'device':device,'arm':arm}, 'identity':{},
              'candidate':arm == 'candidate', 'step':step, 'members':members, 'batch':list(range(64)), 'K':1,
              'ranking_total':outer, 'routed_total':routed, 'route_full_reference':reference,
              'phase_observe':observe, 'optimizer':SimpleNamespace(step=optimizer_work,zero_grad=optimizer_work),
              'scaler':SimpleNamespace(step=optimizer_work,unscale_=optimizer_work)}
        for name in d.VIEWS:
            ns['view'] = name
            refs.clear(), events.clear(), result_refs.clear(), result_ids.clear()
            if fault:
                rejects(lambda: exec(code, ns), 'reference fault')
                assert not refs and events == ['reference'] and not {'route','ranking'} & ns.keys()
                continue
            exec(code, ns)
            assert [t.v for t in outer + (routed or [])] == outer_values
            expected = len(members) + (12 if arm == 'candidate' else 0) if step == 1 else 0
            assert len(refs) == expected and all(ref() is not None for ref in refs)
            if step == 1:
                assert [t.shape for t in ns['ranking']] == [p.shape for p in members]
                assert all(t.v == [0.] * len(p.v) for t,p in zip(ns['ranking'],members,strict=True))
            else:
                assert ns['ranking'] is None
            if arm == 'candidate' and step == 1:
                route = ns['route']
                assert route.keys() == {'micro','nonzero','routed','original','unrouted','ranking','scalars','worst','split','full'}
                for key, params in (('routed',members[:2]),('original',members[:2]),
                                    ('unrouted',members[2:]),('ranking',members[2:])):
                    assert [t.shape for t in route[key]] == [p.shape for p in params]
                    assert all(t.v == [0.] * len(p.v) for t,p in zip(route[key],params,strict=True))
                assert route['micro'] == route['nonzero'] == 0 and route['split'] is None
                assert route['scalars'] == {'mse':0.,'rank':0.,'loss':0.} and route['worst'] == {'A':0.,'C':0.}
                if device == 'cpu':
                    assert id(route['full']) == result_ids[0], 'returned full dictionary was replaced'
                    assert events == (['reference'] + ['allocate']*18 + ['view_accumulators'] if delayed else
                                      ['allocate']*18 + ['view_accumulators','reference'])
                    route['full'] = None
                    gc.collect()
                    assert all(ref() is None for ref in result_refs), 'transient full reference alias survived'
                else:
                    assert route['full'] is None and events == ['allocate']*18
                del route
            else:
                assert ns['route'] is None and events == ['allocate']*expected
            del ns['ranking'], ns['route']
            gc.collect()
            assert all(ref() is None for ref in refs)
        return events

    probe(base, 'candidate', 'cpu', 1, False)
    probe(DRIVER.read_bytes(), 'candidate', 'cpu', 1, True)
    probe(DRIVER.read_bytes(), 'candidate', 'cpu', 1, True, fault=True)
    for arm,device,step in (('candidate','cuda',1),('control','cpu',1),('control','cuda',1),
                            ('candidate','cpu',2),('candidate','cuda',2),('control','cpu',2)):
        assert probe(base,arm,device,step,False) == probe(DRIVER.read_bytes(),arm,device,step,True)
    print('PASS actual view prelude: baseline 18 live buffers -> zero during reference; same buffers/full identity; failure/control/GPU')


def pixel_schedule_falsifier(d):
    """A fake forward with no saved input exposes the helper's caller ownership."""
    import gc
    import weakref

    def probe(mode=None, reference=None, scheduled=True):
        m = Micro(d, step=1, record=False)
        sc,context,state = m.sc,m.ns['context'],m.sc.state
        state['counter'] = 0
        rng,pixels,kept,forwards,hashes,gradients = [23.],[],[],[],[],[]
        sc.torch.random = SimpleNamespace(get_rng_state=lambda: Vec(rng),
            set_rng_state=lambda t: rng.__setitem__(slice(None),t.v))
        def pixels_for(trainer,context,state,processor,anchors,view):
            value = Vec(anchors).detach()
            if mode == 'pixels' and forwards:
                value.v[0] += 1.
            if mode == 'input':
                raise ValueError('input fault')
            pixels.append(weakref.ref(value))
            return value, []
        context['witness'].pixels_for = pixels_for
        def cat(chunks,dim):
            assert all(not t.requires_grad for t in chunks)
            value = FakeTorch.cat(chunks,dim).detach()
            pixels.append(weakref.ref(value))
            return value
        sc.torch.cat = cat
        def model(pixel_values):
            assert not pixel_values.requires_grad and len(pixel_values.v) == 64
            if scheduled:
                assert hashes and hashes[-1] == ('pixels',tuple(pixel_values.v)), 'pixel fingerprint must precede forward'
            forwards.append(tuple(pixel_values.v))
            rng[0] += 7.
            if mode == 'external':
                kept.append(pixel_values)
            if mode == 'forward':
                raise ValueError('forward fault')
            features = Vec(pixel_values.v)
            if mode == 'saved_input':
                features.saved_input = pixel_values
            return SimpleNamespace(pooler_output=features)
        state['model'] = model
        raw_features = context['connected'].raw_features
        def connected(features,*args):
            value = raw_features(features,*args)
            value.graph = features
            return value
        context['connected'].raw_features = connected
        genuine_grad = sc.torch.autograd.grad
        def grad(out,inputs,**kw):
            gc.collect()
            if scheduled and mode not in ('external','saved_input'):
                assert all(ref() is None for ref in pixels), 'caller pixels survived until gradients'
            if mode == 'saved_input':
                assert pixels[-1]() is not None, 'stand-in graph did not retain its saved input'
            gradients.append(len(inputs))
            if mode == 'gradient':
                raise ValueError('gradient fault')
            return genuine_grad(out,inputs,**kw)
        sc.torch.autograd.grad = grad
        def fact(context,value):
            if isinstance(value,(Vec,Out)):
                hashes.append(('pixels' if any(ref() is value for ref in pixels) else 'other',tuple(value.v)))
                return repr((value.shape,value.dtype,value.v))
            return repr(value)
        functional = ModuleType('torch.nn.functional')
        functional.normalize = lambda t,dim:t
        nn = ModuleType('torch.nn')
        nn.functional = functional
        saved = lambda context,state,identity: {'A':state['A'].v,'C':state['C'].v,'counter':state['counter'],'rng':list(rng)}
        with patch.dict(sys.modules, {'torch':sc.torch,'torch.nn':nn,'torch.nn.functional':functional}), \
                patch.object(d,'payload',saved), patch.object(d,'fingerprint',fact):
            try:
                result = (reference or d.route_full_reference)(sc.torch,context,state,{},sc.members,list(range(64)),1,d.VIEWS[0])
            except ValueError as error:
                expected = {'pixels':'full B64 replay pixels/features differ','input':'input fault','forward':'forward fault',
                            'gradient':'gradient fault','external':'full B64 graph lifetime survived release'}
                assert mode in expected and expected[mode] in str(error), str(error)
                assert rng == [23.]
                assert len(forwards) == (0 if mode == 'input' else 1), 'input fault reached another forward'
                if mode == 'pixels':
                    assert gradients == [6,2,6], 'replay pixel mismatch reached gradients'
                if mode == 'external':
                    assert any(ref() is not None for ref in pixels), 'external owner unexpectedly removed'
                else:
                    assert all(ref() is None for ref in pixels), 'pixels survived retained error traceback'
            else:
                assert mode in (None,'saved_input') and rng == [23.]
                assert forwards == [tuple(float(i) for i in range(64))]*2 and gradients == [6,2,6,6]
                assert [g.v for g in result['gradients']] == [g.v for g in sc.full()['gradients']]
                assert {k:v.value for k,v in result['scalars'].items()} == {'mse':5.,'rank':2.,'loss':7.}
                assert [v for k,v in hashes if k == 'pixels'] == forwards
            finally:
                kept.clear()
        gc.collect()
        assert all(ref() is None for ref in pixels)
        return hashes,gradients
    base = (HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/'
            'rank-routed-mlp-cpu-v3-freeze/train_siglip2_rank_routed_mlp.py').read_bytes()
    assert sha(base) == SCHEDULING_BASE_SHA
    ns = {}
    exec(compile(ast.Module(body=[function(ast.parse(base),'route_full_reference')],type_ignores=[]),
                 '<frozen CPU-v3 full reference>', 'exec'), ns)
    original = FunctionType(ns['route_full_reference'].__code__,vars(d))
    assert probe() == probe(reference=original,scheduled=False), 'fingerprints or gradient calls changed'
    for mode in ('saved_input','pixels','input','forward','gradient','external'):
        probe(mode)
    print('PASS caller pixels released before gradients; exact hashes/results; replay-before-forward/input/fault/external-owner rejects')


def full_reference_source_falsifier(d):
    def probe(mutate=False, leak=False, fault=False):
        m = Micro(d, step=1, record=False)
        sc,context,state = m.sc,m.ns['context'],m.sc.state
        state['counter'] = 0
        rng,seen,kept = [23.],[],[]
        sc.torch.random = SimpleNamespace(get_rng_state=lambda: Vec(rng), set_rng_state=lambda t: rng.__setitem__(slice(None),t.v))
        def pixels_for(trainer,context,state,processor,anchors,view):
            seen.append((view,list(anchors)))
            return Vec(anchors), []
        context['witness'].pixels_for = pixels_for
        def model(pixel_values):
            assert len(pixel_values.v) == 64
            rng[0] += 7.
            if mutate or fault:
                state['A'].v[0] += 1.
            if fault:
                raise ValueError('primary full forward fault')
            features = Vec(pixel_values.v)
            if leak:
                kept.append(features)
            return SimpleNamespace(pooler_output=features)
        state['model'] = model
        functional = ModuleType('torch.nn.functional')
        functional.normalize = lambda t,dim:t
        nn = ModuleType('torch.nn')
        nn.functional = functional
        def saved(context,state,identity):
            return {'A':state['A'].v,'C':state['C'].v,'counter':state['counter'],'rng':list(rng)}
        with patch.dict(sys.modules, {'torch':sc.torch,'torch.nn':nn,'torch.nn.functional':functional}), \
                patch.object(d,'payload',saved), patch.object(d,'fingerprint',lambda c,v:repr((v.shape,v.dtype,v.v)) if isinstance(v,(Vec,Out)) else repr(v)):
            for view in d.VIEWS:
                try:
                    full = d.route_full_reference(sc.torch,context,state,{},sc.members,list(range(64)),1,view)
                except ValueError:
                    assert rng == [23.], 'full reference failed without restoring CPU RNG'
                    raise
                assert rng == [23.] and full['scalars']['loss'].value == 7.
                for got,wanted in zip(full['gradients'],sc.full()['gradients'],strict=True):
                    assert got.v == wanted.v
                route = sc.view(d,'cpu')
                route['full'] = full
                fact = d.route_view(sc.torch,context,route)
                assert fact['full_micro']['scalars'] == {'mse':0.,'rank':0.,'loss':0.}
        assert seen == [(v,list(range(i,i+16))) for v in d.VIEWS for replay in range(2) for i in range(0,64,16)]
    probe()
    rejects(lambda: probe(mutate=True), 'changed initialized state/RNG')
    rejects(lambda: probe(leak=True), 'full B64 graph lifetime survived')
    rejects(lambda: probe(fault=True), 'primary full forward fault')
    print('PASS executed independent B64 source reference: bothviews/full-vs-micro/state+RNG/lifetime/primary-error; native UNRUN')


def live_runtime_falsifier(d):
    m = Micro(d, step=18)
    genuine = m.ns['context']['routing'].regression_terms
    def detached(*args):
        result = genuine(*args).detach()
        result.requires_grad = False
        return result
    m.ns['context']['routing'].regression_terms = detached
    rejects(m.run, 'regression live callable differs')
    assert not m.regressions and not m.backs, 'replacement evaluated before rejection'
    print('PASS live callable replacement after step17 rejected before regression/backward')


def exceptional_arm_falsifier(d):
    import gc
    import traceback
    import weakref
    def run(leak=False):
        node = copy.deepcopy(function(ast.parse(DRIVER.read_text()), 'arm_run'))
        state = {'identity':{}, 'A':Vec([1.])}
        reference = weakref.ref(state['A'])
        kept = [state['A']] if leak else []
        context = {'connected_args':SimpleNamespace(phase='cpu',output=Path('/tmp/unused-source-seam'))}
        calls = []
        def failed_update(context,state,identity,step):
            members = [state['A']]
            optimizer = SimpleNamespace(params=members)
            raise ValueError('primary native cap event')
        def release(context,state):
            calls.append('release')
            state.clear()
            gc.collect()
            d.require(reference() is None, 'previous training A still alive')
        ns = {**vars(d), 'fresh':lambda *a:state, 'update':failed_update, 'release':release}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(DRIVER),'exec'),ns)
        try:
            ns['arm_run'](context,'candidate',d.SEEDS[0],'cpu',discarded_update=True)
        except ValueError as error:
            assert str(error) == 'primary native cap event', 'cleanup replaced the original failure: '+str(error)
            assert any(frame.name == 'failed_update' for frame in traceback.extract_tb(error.__traceback__)), 'original location lost'
            if leak:
                assert str(error.__cause__) == 'previous training A still alive', 'genuine cleanup failure ignored'
                assert reference() is kept[0]
            else:
                assert error.__cause__ is None and reference() is None
        else:
            raise AssertionError('exceptional update became successful')
        assert calls == ['release'] and state == {}, 'original cleanup skipped or repeated'
    run()
    run(leak=True)
    node = copy.deepcopy(function(ast.parse(DRIVER.read_text()),'arm_run'))
    cleanup = next(n.finalbody for n in node.body if isinstance(n,ast.Try))
    seam = ast.fix_missing_locations(ast.Module(body=[ast.Try(body=[ast.Pass()],handlers=[],orelse=[],finalbody=cleanup)],type_ignores=[]))
    rejects(lambda: exec(compile(seam,str(DRIVER),'exec'),{**vars(d),'context':{},'state':{'A':Vec([1.])},
        'release':lambda *a:d.require(False,'genuine normal lifetime failure')}),'genuine normal lifetime failure')
    print('PASS extracted exceptional arm_run: traceback owners released; primary/location retained; genuine lifetime failure chained')


def observer_falsifier(d):
    import io
    import gc
    import weakref
    assert hasattr(d,'phase_observe'), 'fixed scalar phase observer missing'
    files = {'/proc/self/cgroup':'0::/test-unit\n', '/proc/self/status':'VmRSS:\t123 kB\nVmHWM:\t456 kB\n',
        '/sys/fs/cgroup/test-unit/memory.current':'789', '/sys/fs/cgroup/test-unit/memory.peak':'999',
        '/sys/fs/cgroup/test-unit/memory.max':'8589934592',
        '/sys/fs/cgroup/test-unit/memory.events':'max 7\noom 0\noom_kill 0\n',
        '/sys/fs/cgroup/test-unit/memory.stat':'anon 500\nfile 200\nkernel 89\n'}
    context = {'connected_args':SimpleNamespace(phase='cpu'), 'started':time.perf_counter(),
               'connected_code':{'train_siglip2_rank_routed_mlp.py':'1'*64},
               'phase_observation':{'records':0,'bytes':0,'failed':False,'closed':False}}
    class Owner:
        shape,dtype,requires_grad,grad_fn = (2,3),'float32',True,None
        def numel(self): return 6
        def untyped_storage(self): return SimpleNamespace(nbytes=lambda:24)
    owner = Owner()
    reference = weakref.ref(owner)
    sink = io.StringIO()
    with patch.object(Path,'open',lambda p,*a,**kw:io.StringIO(files[str(p)])), \
            patch.dict(d.os.environ,INVOCATION_ID='a'*32), patch.object(d.sys,'stdout',sink):
        d.phase_observe(context,'full_forward',view='canonical',groups=(('query',(owner,owner)),))
        row = json.loads(sink.getvalue())
        assert row['memory']['events']['max'] == 7, 'observer reset/accepted existing max events'
        assert row['process']['rss_kib'] == 123 and row['process']['hwm_kib'] == 456
        assert row['owners']['query']['storage_bytes'] == 48, 'per-tensor capacity must disclose repeated aliases'
        assert row['owners']['query']['shapes'] == [[2,3],[2,3]]
        del owner
        gc.collect()
        assert reference() is None, 'observer retained a tensor owner'
        for _ in range(127):
            d.phase_observe(context,'micro_released')
        assert context['phase_observation']['records'] == 128
        d.phase_observe(context,'overflow')
        assert context['phase_observation']['failed'] and len(sink.getvalue().encode()) <= 256*1024
    failed = {**context,'phase_observation':{'records':0,'bytes':0,'failed':False,'closed':False}}
    with patch.object(Path,'open',side_effect=OSError('observation IO failed')):
        d.phase_observe(failed,'probe')
    assert failed['phase_observation']['failed'], 'diagnostic IO silently admitted'
    interrupted = {**context,'phase_observation':{'records':0,'bytes':0,'failed':False,'closed':False}}
    with patch.object(Path,'open',side_effect=SystemExit('observer failure')):
        d.phase_observe(interrupted,'probe')
    assert interrupted['phase_observation']['error'] == 'SystemExit'
    assert context['phase_observation']['bytes'] == len(sink.getvalue().encode())
    for changes in ({'records':128},{'bytes':256*1024-1}):
        limited = {**context,'phase_observation':{'records':0,'bytes':0,'failed':False,'closed':False,**changes}}
        output = io.StringIO()
        with patch.object(Path,'open',lambda p,*a,**kw:io.StringIO(files[str(p)])), \
                patch.dict(d.os.environ,INVOCATION_ID='a'*32), patch.object(d.sys,'stdout',output):
            d.phase_observe(limited,'limited')
        assert limited['phase_observation']['failed'] and output.getvalue() == ''
    oversized = {**context,'phase_observation':{'records':0,'bytes':0,'failed':False,'closed':False}}
    output = io.StringIO()
    with patch.object(Path,'open',lambda p,*a,**kw:io.StringIO(files[str(p)])), \
            patch.dict(d.os.environ,INVOCATION_ID='a'*32), patch.object(d.sys,'stdout',output):
        d.phase_observe(oversized,'x'*4096)
    assert oversized['phase_observation']['failed'] and output.getvalue() == '', 'oversized row emitted'
    for sink_ in (SimpleNamespace(write=lambda s:len(s)-1,flush=lambda:None),
                  SimpleNamespace(write=lambda s:(_ for _ in ()).throw(OSError('stdout failure')))):
        broken = {**context,'phase_observation':{'records':0,'bytes':0,'failed':False,'closed':False}}
        with patch.object(Path,'open',lambda p,*a,**kw:io.StringIO(files[str(p)])), \
                patch.dict(d.os.environ,INVOCATION_ID='a'*32), patch.object(d.sys,'stdout',sink_):
            d.phase_observe(broken,'broken')
        assert broken['phase_observation']['failed']
    print('PASS scalar observer: raw max unchanged, scalar owners/released weakref, IO failure recorded, record/byte bounds')


def observer_guard_falsifier(d):
    calls=[]
    source=SimpleNamespace(cgroup_memory=lambda:(calls.append('cgroup') or {'path':'/sys/fs/cgroup/unit.service'}))
    context={'connected_args':SimpleNamespace(phase='cpu'),'started':time.perf_counter(),
        'phase_observation':{'records':0,'bytes':0,'failed':True,'closed':False},
        'old':SimpleNamespace(zero_events=lambda v:calls.append('zero_events')),
        'legacy':{'source_driver':source,'selected':{'genuine':{'reference':SimpleNamespace(admit_cgroup=lambda *a:calls.append('admit'))}}}}
    torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False))
    with patch.dict(sys.modules,{'torch':torch}):
        rejects(lambda:d.resource_check(context),'bounded phase observation failed')
        assert calls == ['cgroup','zero_events','admit'], 'observer skipped original guards'
        source.cgroup_memory=lambda: d.require(False,'primary cap event')
        rejects(lambda:d.resource_check(context),'primary cap event')
        try:
            raise ValueError('pending primary')
        except ValueError:
            d.observation_guard(context)
    print('PASS observer error follows original guards; primary cap/pending exception unchanged')


def observer_source_contract(d):
    tree = ast.parse(sequential_inverse(DRIVER.read_bytes()))
    expected = {'fresh':(5,5),'resource_check':(1,1),'integrity':(3,3),'update':(18,63),
                'route_full_reference':(9,9),'route_split':(2,2),'route_micro':(1,1),'arm_run':(3,3)}
    actual = {}
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef):
            continue
        sites = []
        def walk(item,weight=1):
            if isinstance(item,ast.For):
                loop = ast.unparse(item.iter)
                if loop == 'VIEWS': weight *= len(d.VIEWS)
                elif loop == 'range(0, 64, 16)': weight *= len(range(0,64,16))
                else:
                    assert not any(isinstance(c,ast.Call) and ast.unparse(c.func) == 'phase_observe'
                                   for c in ast.walk(item)), 'unbounded observation loop'
            if isinstance(item,ast.Call) and ast.unparse(item.func) == 'phase_observe':
                assert isinstance(item.args[1],ast.Constant) and type(item.args[1].value) is str
                sites.append((item.args[1].value,weight))
            for child in ast.iter_child_nodes(item): walk(child,weight)
        walk(node)
        if sites: actual[node.name] = (len(sites),sum(w for _,w in sites))
    assert actual == expected and len(d.VIEWS) == 2
    # Fresh integrity once, update integrity twice; each still calls the original resource check.
    def calls(name,target):
        return [c for c in ast.walk(function(tree,name)) if isinstance(c,ast.Call) and ast.unparse(c.func) == target]
    assert len(calls('fresh','integrity')) == 1 and len(calls('update','integrity')) == 2
    assert len(calls('integrity','resource_check')) == 1
    assert len(calls('update','route_full_reference')) == len(calls('update','route_micro')) == 1
    assert len(calls('route_micro','route_split')) == 1
    assert "if first and state['device'] == 'cpu':" in ast.unparse(function(tree,'route_micro'))
    assert 'offset == 0' in ast.unparse(function(tree,'update'))
    # Conservative union includes both normal completion and exceptional cleanup (two extra rows).
    upper = actual['arm_run'][1] + actual['fresh'][1] + 3*(actual['integrity'][1]+actual['resource_check'][1])
    upper += actual['update'][1] + 2*actual['route_full_reference'][1] + 8*actual['route_micro'][1] + 2*actual['route_split'][1]
    assert sum(v[0] for v in actual.values()) == 42 and upper == 113 <= d.OBSERVATION_LIMITS['records']
    assert d.OBSERVATION_LIMITS == {'records':128,'record_bytes':4096,'bytes':256*1024}
    observer = function(tree,'phase_observe')
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.AsyncFunctionDef)) for n in ast.walk(observer))
    call_names = {ast.unparse(c.func) for c in ast.walk(observer) if isinstance(c,ast.Call)}
    assert not any(any(word in c for word in ('clone','detach','collect','thread','reset','unlink')) for c in call_names)
    assert not any(isinstance(c.func,ast.Attribute) and c.func.attr == 'open' and (c.args or c.keywords)
                   for c in ast.walk(observer) if isinstance(c,ast.Call)), 'diagnostic file write added'
    update_text = ast.unparse(function(tree,'update'))
    assert update_text.index("phase_observe(context, 'update_complete', close=True)") < update_text.index('seconds = time.perf_counter() - tick')
    print('PASS observer source: 42 fixed call sites; conservative window <=113/128 rows; hard per-row4096/total262144 byte bounds; overhead timed')


def update_dataflow_falsifiers(d):
    # Structure of the real update(): one pinned loss_terms call, one derived regression call, one live connected call.
    tree = ast.parse(DRIVER.read_text())
    update = function(tree, 'update')
    text = ast.unparse(update)
    calls = [ast.unparse(c.func) for c in ast.walk(update) if isinstance(c, ast.Call)]
    assert calls.count('trainer.loss_terms') == 1 and calls.count('connected.raw_features') == 1
    assert calls.count('trainer.raw_features') == 2 and calls.count('routed_regression') == 1
    assert calls.count('route_micro') == 1 and calls.count('route_view') == 1 and not [c for c in calls if 'ranking_gallery' in c]
    order = ['raw = connected.raw_features(', 'detached = trainer.raw_features(context, state, features.detach())',
             'torch.equal(detached.detach(), raw.detach())', 'mse, rank, selected = trainer.loss_terms(context, state, raw, anchors, K)',
             'mse = routed_regression(context, state, detached, anchors, K)',
             'torch.equal(mse.detach(), original.detach())', 'route_micro(', 'loss = mse + rank', 'scaler.scale(loss).backward()']
    positions = [text.index(s) for s in order]
    assert positions == sorted(positions) and 'raw = trainer.raw_features(context, state, features)' in text
    assert 'regression_terms(context, state, raw' not in text and text.index('if released:') < text.index('scaler.unscale_')
    for name in ('route_micro', 'route_split'):
        grads = [c for c in ast.walk(function(tree, name)) if isinstance(c, ast.Call) and
                 ast.unparse(c.func) == 'torch.autograd.grad']
        assert grads and all(any(k.arg == 'retain_graph' and ast.unparse(k.value) == 'True' for k in c.keywords) for c in grads), name
        assert not [c for c in ast.walk(function(tree, name)) if isinstance(c, ast.Attribute) and c.attr == 'backward']
    # Executed micro loop, candidate (no autograd witness): routed mse is what reaches backward.
    m = Micro(d)
    ns = m.run()
    assert ns['parity_checks'] == 4 and len(m.backs) == 4 and len(m.regressions) == 4 and ns['mse_sum'] == 5. and ns['rank_sum'] == 2.
    assert all(loss.parts[0].value == 1.25 and loss.parts[1].value == .5 for loss in m.backs)
    assert all(kind == 'detached' for kind, _ in [c for c in m.calls if c[0] in ('live', 'detached')]), 'regression branch must see detached features'
    assert not {'pixels', 'cpu_pixels', 'features', 'raw', 'detached', 'original', 'mse', 'rank', 'loss', 'facts'} & ns.keys()
    # Control arm: unchanged single live raw_features call, original mse reaches backward, no routing state touched.
    c = Micro(d, arm='control')
    ns = c.run()
    assert ns['parity_checks'] == 0 and not c.regressions and 'routing' not in c.ns['context']
    assert [k for k, _ in c.calls if k != 'loss'] == ['live'] * 4 and all(l.parts[0].value == 1.25 for l in c.backs)
    # Mutants of the real loop.
    for flags, text in (({'mse': 1.26}, 'regression scalar differs'), ({'raw_offset': 1e-3}, 'detached raw forward differs'),
                        ({'detached_grad': False}, 'detached raw forward differs'),
                        ({'features_grad': False}, 'live encoder features disconnected')):
        rejects(lambda flags=flags: Micro(d, **flags).run(), text)
    # Step-1 CPU: real witness inside the real loop; every temporary graph object dies at the real gate.
    m = Micro(d, step=1, device='cpu')
    ns = m.run()
    assert ns['route']['micro'] == 4 and len(ns['released']) == 28 and all(m.sc.torch.retain)
    d.check_routing(m.sc.context, d.route_view(m.sc.torch, m.sc.context, ns['route']), {'arm': 'candidate', 'device': 'cpu'})
    m.scrub()
    gc_ns = {**vars(d), 'released': ns['released']}
    exec(m.gate, gc_ns)
    assert 'released' not in gc_ns
    leaked = Micro(d, step=1, device='cuda', leak=True)
    leaked_ns = leaked.run()
    leaked.scrub()
    rejects(lambda: exec(leaked.gate, {**vars(d), 'released': leaked_ns['released']}), 'lifetime survived release')
    print('PASS update dataflow: real micro loop on stand-ins (routed mse -> backward, detached branch, control unchanged, '
          'step1 witness, lifetime gate) + one loss_terms/one regression/retain_graph structure')


def check_steps_falsifiers(d):
    names = d.parameter_roles('candidate')[0]
    batch = list(range(64))

    def context(sc):
        return {'trainer': SimpleNamespace(check_steps=lambda *a: None), 'original_cpu_record': {}, 'witness': sc.context['witness']}

    def row(arm, sc, device='cuda'):
        use = d.parameter_roles(arm)[0]
        view = lambda v: {'view': v, 'ranking_gradient_norms': {n: 1. for n in use}}
        views = [view(v) for v in d.VIEWS]
        if arm == 'candidate':
            fact = d.route_view(sc.torch, sc.context, sc.view(d, device))
            for entry in views:
                entry['routing'] = copy.deepcopy(fact)
        return {'step': 1, 'arm': arm, 'batch': batch, 'gradient_norms': {n: 1. for n in use},
                'before_sha256': {n: 'a' for n in use}, 'after_sha256': {n: 'b' for n in use}, 'view_gradients': views,
                'membership': [{'active': 1}], 'full_membership_sha256': 'x', 'full_valid': 63, 'mse': 1., 'rank': 1.,
                'loss': 2., 'preclip_norm': 1., 'state_sha256': 'x', 'core_seconds': 1., 'seconds': 1.,
                'ranking_gradient_norm': 1., 'ranking_C_gradient_norm': 1., 'scale': 128.,
                **({'routed_parity_checks': 8, 'optimizer_routing': {'max_abs':{n:0. for n in use},
                    'tolerance':{'rtol':1e-5,'atol':1e-6}}} if arm == 'candidate' else {})}

    for arm in d.ARMS:
        sc = Scenario()
        identity = {'arm': arm, 'seed': d.SEEDS[0], 'device': 'cuda', 'parameter_names': d.parameter_roles(arm)[0]}
        with patch.object(d, 'select_initializer', lambda *a: {'ranking_bank': None, 'scope_schedule': [batch]}):
            d.check_steps(context(sc), [row(arm, sc)], identity)
            mutants = [('routed_parity_checks', 7), ('routed_parity_checks', True), ('routed_parity_checks', None)] \
                if arm == 'candidate' else [('routed_parity_checks', 0), ('routed_parity_checks', 8)]
            for key, value in mutants:
                wrong = row(arm, sc)
                wrong[key] = value
                rejects(lambda: d.check_steps(context(sc), [wrong], identity), 'rank-routed per-micro')
            if arm == 'candidate':
                for value in (None, {}, {'max_abs':{},'tolerance':{'rtol':1e-5,'atol':1e-6}},
                              {'max_abs':{n:float('nan') for n in names},'tolerance':{'rtol':1e-5,'atol':1e-6}},
                              {'max_abs':{n:0. for n in names},'tolerance':{'rtol':1e-4,'atol':1e-6}}):
                    wrong = row(arm,sc)
                    wrong['optimizer_routing'] = value
                    rejects(lambda: d.check_steps(context(sc),[wrong],identity),'actual optimizer rank-routing')
                wrong = row(arm, sc)
                del wrong['routed_parity_checks']
                rejects(lambda: d.check_steps(context(sc), [wrong], identity), 'rank-routed per-micro')
                wrong = row(arm, sc)
                del wrong['view_gradients'][1]['routing']
                rejects(lambda: d.check_steps(context(sc), [wrong], identity), 'step1 witness')
                wrong = row(arm, sc)
                wrong['view_gradients'][0]['routing']['micro_checks'] = 3
                rejects(lambda: d.check_steps(context(sc), [wrong], identity), 'step1 witness')
            else:
                wrong = row(arm, sc)
                wrong['view_gradients'][0]['routing'] = {}
                rejects(lambda: d.check_steps(context(sc), [wrong], identity), 'control carries no')
    print('PASS check_steps: per-micro parity count and step1 routing witness required for candidate, forbidden for control')


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--source-only', action='store_true', required=True)
    p.add_argument('--layer', choices=('allocation', 'pixels', 'sequential', 'cleanup', 'observer', 'observer-guard', 'observer-source', 'split', 'live', 'reference', 'full-source', 'optimizer', 'regression', 'routing', 'dataflow', 'inverse', 'steps', 'identity', 'seams'))
    args = p.parse_args()
    assert DRIVER.exists(), 'rank-routed trainer missing'
    started = time.perf_counter()
    before = set(sys.modules)
    d = load('_rank_routed_source_test', DRIVER)
    c = load('_connected_source_for_rank_routing', CONNECTED)
    assert not {n.split('.')[0] for n in set(sys.modules) - before} & d.NATIVE
    if args.layer:
        layers = {'allocation': allocation_schedule_falsifier, 'pixels': pixel_schedule_falsifier,
                  'sequential': sequential_reference_falsifier, 'split': split_lifetime_falsifier, 'live': live_runtime_falsifier, 'reference': independent_reference_falsifier,
                  'cleanup': exceptional_arm_falsifier,
                  'observer': observer_falsifier,
                  'observer-guard': observer_guard_falsifier,
                  'observer-source': observer_source_contract,
                  'optimizer': optimizer_gradient_falsifier,
                  'full-source': full_reference_source_falsifier,
                  'regression': regression_falsifiers,
                  'routing': routing_witness_falsifiers, 'dataflow': update_dataflow_falsifiers,
                  'inverse': lambda d: inverse_contract(), 'steps': check_steps_falsifiers,
                  'identity': lambda d: identity_contract(d, c), 'seams': reused_connected_seams}
        layers[args.layer](d)
        assert not {n.split('.')[0] for n in set(sys.modules)} & d.NATIVE
        return
    inverse_contract()
    allocation_schedule_falsifier(d)
    pixel_schedule_falsifier(d)
    exceptional_arm_falsifier(d)
    observer_falsifier(d)
    observer_guard_falsifier(d)
    observer_source_contract(d)
    identity_contract(d, c)
    regression_falsifiers(d)
    split_lifetime_falsifier(d)
    live_runtime_falsifier(d)
    independent_reference_falsifier(d)
    sequential_reference_falsifier(d)
    full_reference_source_falsifier(d)
    optimizer_gradient_falsifier(d)
    routing_witness_falsifiers(d)
    update_dataflow_falsifiers(d)
    check_steps_falsifiers(d)
    reused_connected_seams(d)
    assert not {n.split('.')[0] for n in set(sys.modules)} & d.NATIVE
    print('PASS source-only rank-routed connected MLP contracts/falsifiers; native UNRUN; %.1fs' % (time.perf_counter() - started))


if __name__ == '__main__':
    main()
