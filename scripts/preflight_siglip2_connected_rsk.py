#!/usr/bin/env python3
"""Prospective CPU-only FIT-cache mechanism probe; never run by source tests.

No encoder, images, selection panel, optimizer step, or input-path overrides.
Inputs are pinned by the accepted identity-diversity CPU-v5 receipt d2239da8.
The A/C diagnostic is a first-order isolated ranking-gradient step at lr1e-4,
not an AdamW/B64/two-view/regression training update or a quality prediction.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = {'path': '/home/riomus/runs/sfora-so400-genuine-view-export-v2/canonical.npy',
         'sha256': '55d37d063779e95d936d2e8392b6af44478356ce1ccd29d96cc5361d7966bdb8'}
INITIALIZERS = {
    179061: {'path': '/home/riomus/runs/sfora-so400-identity-diversity-cpu-v5/initializer-control-179061.pt',
             'sha256': '8b89897fdaaad94770e40bd2e69422734709e6d2d73710aa496610bdb2882454'},
    179069: {'path': '/home/riomus/runs/sfora-so400-identity-diversity-cpu-v5/initializer-control-179069.pt',
             'sha256': '1747e22ecbf0e27b7058fa1ec40e83c411423b66a51175a9ca1abe660b95f2f7'},
}
SMOOTH_SOURCE = {'path': str(HERE / 'train_siglip2_identity_diversity.py'),
                 'sha256': '094d97259b5f417b334186305351f02d6b76983cf88dbf4021a22d0f70a235e9'}
SHAPE, LR = (6355, 1152), 1e-4
OBJECTIVES = ('smooth_ap', 'rsk', 'hinge_calibration')


def source_facts():
    return [{'path': str(HERE / name), 'sha256': hashlib.sha256((HERE / name).read_bytes()).hexdigest()}
            for name in ('connected_rsk_loss.py', 'preflight_siglip2_connected_rsk.py')]


def check_file(fact):
    path = Path(fact['path'])
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError('required pinned regular input absent: ' + str(path))
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != fact['sha256']:
        raise ValueError('input SHA256 mismatch: ' + str(path))


def npy_payload(raw):
    """Read only a C-contiguous little-endian float32 NPY; no NumPy/pickle."""
    if raw[:6] != b'\x93NUMPY' or raw[6:8] not in (b'\x01\x00', b'\x02\x00', b'\x03\x00'):
        raise ValueError('fixed numeric NPY format required')
    width = 2 if raw[6] == 1 else 4
    size = int.from_bytes(raw[8:8 + width], 'little')
    start = 8 + width + size
    header = ast.literal_eval(raw[8 + width:start].decode('utf8' if raw[6] == 3 else 'latin1').strip())
    shape = header.get('shape')
    if (header.keys() != {'descr', 'fortran_order', 'shape'} or header['descr'] != '<f4' or
            header['fortran_order'] is not False or type(shape) is not tuple or len(shape) != 2 or
            any(type(n) is not int or n <= 0 for n in shape) or len(raw) - start != math.prod(shape) * 4):
        raise ValueError('finite-layout C float32 cache required')
    return shape, memoryview(raw)[start:]


def first_positive_pair(scores, positive, eligible):
    if not positive:
        return None
    ordered = sorted(eligible, key=lambda j: (-scores[j], j))
    positives = set(positive)
    rank, best = next((rank, j) for rank, j in enumerate(ordered, 1) if j in positives)
    return best, ordered[0], rank


def gradient_fraction(gradient, positive, impostor, eligible):
    total = math.fsum(abs(gradient[j]) for j in eligible)
    return (abs(gradient[positive]) + abs(gradient[impostor])) / total if total else 0.


def top_pair_hinge(scores, best, impostor):
    return (.05 - (scores[best] - scores[impostor])).clamp(min=0.)


def summarize(rows, ranks):
    means = {name: math.fsum(r[name]['fraction'] for r in rows) / len(rows) if rows else 0.
             for name in OBJECTIVES}
    ratio = means['rsk'] / means['smooth_ap'] if means['smooth_ap'] > 0 else None
    decision = ('INCONCLUSIVE_HOLD' if len(rows) < 20 or ratio is None else
                'MECHANISM_PRESENT' if ratio >= 2. else 'SKIP_NATIVE')
    weighted, cosines = {}, {}
    for name in OBJECTIVES:
        total = math.fsum(r[name]['score_gradient_l1'] for r in rows)
        weighted[name] = math.fsum(r[name]['score_gradient_pair_l1'] for r in rows) / total if total else None
        cosines[name] = [r[name]['parameter_cosine_AC'] for r in rows if r[name]['parameter_cosine_AC'] is not None]
    return {'anchors': len(rows), 'mean_fraction': means, 'fraction_ratio': ratio,
            'first_positive_rank_histogram': {**{str(n): ranks.count(n) for n in (1, 2, 3)},
                                              '>3': sum(r is not None and r > 3 for r in ranks)},
            'no_positive_anchors': ranks.count(None),
            'mean_margin_delta_first_order': {
                name: {part: math.fsum(r[name][f'margin_delta_{part}_first_order'] for r in rows) / len(rows)
                       if rows else None for part in ('A', 'C', 'AC')} for name in OBJECTIVES},
            'diagnostics': {
                'mass_weighted_fraction': weighted,
                'mass_weighted_fraction_ratio': weighted['rsk'] / weighted['smooth_ap']
                    if weighted['smooth_ap'] and weighted['rsk'] is not None else None,
                'mean_parameter_cosine_AC': {name: math.fsum(v) / len(v) if v else None
                                             for name, v in cosines.items()},
                'parameter_cosine_count': {name: len(v) for name, v in cosines.items()}},
            'decision': decision}


def preflight_decision(results):
    if any(r['decision'] == 'SKIP_NATIVE' for r in results):
        return 'SKIP_NATIVE'
    if len(results) == 2 and all(r['decision'] == 'MECHANISM_PRESENT' for r in results):
        return 'MECHANISM_PRESENT'
    return 'INCONCLUSIVE_HOLD'


def parameter_diagnostics(loss_raw, margin_raw, bases, indices):
    """Euclidean A/C gradient geometry; no AdamW normalization or norm clipping."""
    dots, loss_norm2, margin_norm2 = [], 0., 0.
    for basis in bases:
        loss_gradient = loss_raw.double().T @ basis.double()
        margin_gradient = margin_raw.double().T @ basis[indices].double()
        dots.append(float((loss_gradient * margin_gradient).sum()))
        loss_norm2 += float(loss_gradient.square().sum())
        margin_norm2 += float(margin_gradient.square().sum())
    denominator = math.sqrt(loss_norm2 * margin_norm2)
    return {'margin_delta_A_first_order': -LR * dots[0], 'margin_delta_C_first_order': -LR * dots[1],
            'margin_delta_AC_first_order': -LR * math.fsum(dots),
            'parameter_cosine_AC': max(-1., min(1., -math.fsum(dots) / denominator)) if denominator else None}


def score_gradient_to_raw(unit, norms, anchor, gradient):
    derivative = gradient[:, None] * unit[anchor][None, :]
    derivative[anchor] += (gradient[None, :] @ unit)[0]
    return (derivative - unit * (unit * derivative).sum(1)[:, None]) / norms[:, None]


def probe_seed(torch, canonical, seed, path):
    from torch.nn import functional as F
    from connected_rsk_loss import rsk_terms
    from train_siglip2_identity_diversity import smooth_ap_terms

    disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (disk['counter'] != 0 or disk['identity']['seed'] != seed or disk['identity']['arm'] != 'control' or
            disk['scope']['arm'] != 'control' or
            disk['scope']['payload']['scope_sha256'] != '1f3ad34bbd20a3b375ccb4908f9a3da05b63b514395cb553e1c81925789b2280' or
            not torch.equal(disk['views']['canonical'], canonical) or
            tuple(disk['A'].shape) != (128, 160) or tuple(disk['C'].shape) != (128, 1152) or
            torch.count_nonzero(disk['C']).item() != 0):
        raise ValueError('exact CONTROL FIT initializer/cache correspondence required')
    labels, original_rows = disk['target'].tolist(), disk['original_rows'].tolist()
    if len(labels) != SHAPE[0] or len(original_rows) != SHAPE[0]:
        raise ValueError('full FIT row membership required')
    with torch.no_grad():
        # Exact frozen control head and signed concat readout arithmetic; no encoder.
        head = disk['head']
        features = F.normalize(canonical, dim=1)
        z = F.linear(features - head['center'], head['down.weight'])
        h0 = F.linear(features, head['primary.weight'], head['primary.bias']) + F.linear(.5 * z, head['up.weight'])
        phi = torch.cat((z, h0), dim=1) - disk['means']['concat']
        residual = canonical - disk['mu_train']
        raw = h0 + F.linear(phi, disk['A']) + F.linear(residual, disk['C'])
        norms = raw.norm(dim=1)
        if not torch.isfinite(raw).all().item() or not (norms > 1e-12).all().item():
            raise ValueError('finite initializer descriptors above normalization epsilon required')
        unit = F.normalize(raw, dim=1)
        phi, residual = phi.double(), residual.double()
    del disk, head, features, z, h0, raw
    rows, ranks = [], []
    for anchor in range(SHAPE[0]):
        eligible = [j for j in range(SHAPE[0]) if original_rows[j] != original_rows[anchor]]
        positive = [j for j in eligible if labels[j] == labels[anchor]]
        scores = (unit[anchor] @ unit.T).detach().requires_grad_(True)
        pair = first_positive_pair(scores.detach().tolist(), positive, eligible)
        ranks.append(pair[2] if pair is not None else None)
        if pair is None or pair[2] not in (2, 3):
            continue
        best, impostor, rank = pair
        indices = [anchor, best, impostor]
        margin_gradient = torch.zeros_like(scores)
        margin_gradient[best], margin_gradient[impostor] = 1., -1.
        margin_raw = score_gradient_to_raw(unit, norms, anchor, margin_gradient)[indices]
        row = {'anchor': anchor, 'original_row': original_rows[anchor], 'best_positive': best,
               'top_impostor': impostor, 'best_positive_rank': rank,
               'positive_minus_impostor': float(scores[best].detach() - scores[impostor].detach())}
        for name, term in (('smooth_ap', smooth_ap_terms), ('rsk', rsk_terms), ('hinge_calibration', None)):
            loss = top_pair_hinge(scores, best, impostor) if term is None else term(scores, positive, eligible).mean()
            gradient, = torch.autograd.grad(loss, scores)
            if not torch.isfinite(gradient).all().item():
                raise ValueError('nonfinite score gradient')
            values = gradient.tolist()
            loss_raw = score_gradient_to_raw(unit, norms, anchor, gradient)
            row[name] = {'loss': float(loss.detach()),
                'fraction': gradient_fraction(values, best, impostor, eligible),
                'score_gradient_positive': values[best], 'score_gradient_impostor': values[impostor],
                'score_gradient_pair_l1': abs(values[best]) + abs(values[impostor]),
                'score_gradient_l1': math.fsum(abs(values[j]) for j in eligible),
                **parameter_diagnostics(loss_raw, margin_raw, (phi, residual), indices)}
        rows.append(row)
    return {'seed': seed, **summarize(rows, ranks), 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise ValueError('new output file in existing directory required')
    facts = [CACHE, *INITIALIZERS.values(), SMOOTH_SOURCE, *source_facts()]
    for fact in facts:
        check_file(fact)
    raw = Path(CACHE['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CACHE['sha256']:
        raise ValueError('cache changed during read')
    shape, values = npy_payload(raw)
    if shape != SHAPE:
        raise ValueError('only the 6355-row CONTROL FIT cache is allowed')
    import torch
    torch.set_num_threads(1)
    cache = torch.frombuffer(bytearray(values), dtype=torch.float32).reshape(SHAPE)
    del raw, values
    if not torch.isfinite(cache).all().item() or not (cache.norm(dim=1) > 0).all().item():
        raise ValueError('finite nonzero canonical cache required')
    canonical = torch.nn.functional.normalize(cache, dim=1)
    del cache
    results = [probe_seed(torch, canonical, seed, fact['path']) for seed, fact in INITIALIZERS.items()]
    for fact in facts:
        check_file(fact)
    report = {'schema': 'connected-rsk-fit-mechanism-v1', 'facts': facts, 'device': 'cpu',
              'selection_read': False, 'optimizer_step': False, 'diagnostic_lr': LR,
              'diagnostic_only': 'Euclidean first-order lr1e-4; AdamW per-parameter normalization not modeled; '
                                 'cosine, mass-weighted fractions and KILLed-hinge calibration do not change the gate',
              'decision': preflight_decision(results),
              'seeds': results}
    with args.output.open('x') as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
