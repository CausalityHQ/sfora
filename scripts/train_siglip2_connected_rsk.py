#!/usr/bin/env python3
"""Source-only paired TRAIN arm API; native qualification and launch are UNRUN.

Call update with a freshly admitted connected CONTROL state in BOTH arms.
The private loss binding changes only the ranking term. Existing modules,
state schemas, encoder roles and optimizer remain unchanged. Returned records
have a separate experimental schema and must never be admitted as old CONTROL
receipts. The parent owns a new native authority, checkpoint provenance and
CPU/mechanics gates; this module does not launch or authorize any job.
"""
import hashlib
from pathlib import Path
import time
from types import CodeType, FunctionType, SimpleNamespace

from connected_rsk_loss import rsk_terms

SCHEMA = 'siglip2-connected-rsk-update-v1'
ARMS = ('control', 'rsk')
PINS = {
    'train_siglip2_identity_diversity.py': '840c5d8277a89ccdac02c9e231cbe6eddf386e2b23915ecd1bbec1136c51dee8',
    'train_siglip2_connected_mlp.py': '79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b',
}


def authenticate(module, filename, names):
    """Bind consumed live functions to the unchanged source, before adaptation."""
    path = Path(module.__file__)
    raw = path.read_bytes()
    if path.name != filename or hashlib.sha256(raw).hexdigest() != PINS[filename]:
        raise ValueError('frozen source bytes differ: ' + filename)
    compiled = compile(raw, str(path), 'exec', dont_inherit=True)
    codes = {c.co_name: c for c in compiled.co_consts if type(c) is CodeType}
    for name in names:
        function = getattr(module, name)
        if (type(function) is not FunctionType or function.__code__ != codes[name] or
                function.__globals__ is not vars(module) or function.__defaults__ is not None or
                function.__kwdefaults__ is not None or function.__closure__ is not None):
            raise ValueError('frozen live function differs: ' + name)


def make_loss_terms(trainer, arm):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError('fixed control or rsk experiment arm required')
    authenticate(trainer, 'train_siglip2_identity_diversity.py',
                 ('loss_denominators', 'ranking_membership', 'ranking_gallery', 'smooth_ap_terms', 'loss_terms'))
    original = trainer.loss_terms
    if arm == 'control':
        return original
    namespace = dict(original.__globals__)
    namespace['smooth_ap_terms'] = rsk_terms
    return FunctionType(original.__code__, namespace, original.__name__)


def arm_context(context, arm):
    """Project only the trainer object; retain shared guards, counters and owners."""
    trainer = context['trainer']
    loss = make_loss_terms(trainer, arm)
    private = SimpleNamespace(**{**vars(trainer), 'loss_terms': loss})
    return {**context, 'trainer': private}


def update(connected, context, state, identity, step, *, arm):
    """One original B64/two-view/micro16 update, labeled by experimental arm."""
    started = time.perf_counter()
    authenticate(connected, 'train_siglip2_connected_mlp.py', ('update', 'parameter_roles'))
    if (state['arm'] != 'control' or identity['arm'] != 'control' or
            identity['parameter_names'] != ['A', 'C'] or
            identity['parameter_shapes'] != [[128, 160], [128, 1152]]):
        raise ValueError('both experimental arms require frozen-encoder CONTROL state')
    private = arm_context(context, arm)
    try:
        row = connected.update(private, state, identity, step)
    except ValueError as error:
        if arm == 'rsk' and 'SmoothAP' in str(error):
            raise ValueError(str(error).replace('SmoothAP', 'RS@k')) from error
        raise
    return {'schema': SCHEMA, 'arm': arm, 'ranking': 'SmoothAP' if arm == 'control' else 'RS@k',
            'core_seconds': time.perf_counter() - started, 'control_update': row}
