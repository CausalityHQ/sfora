#!/usr/bin/env python3
"""Hermetic check of stage_native256_v9_held.py on a fake completed v9 tree (stdlib only, no DGX)."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).absolute().parent
TOOL = HERE / 'stage_native256_v9_held.py'
ORDER = ((179032, 'large'), (179032, 'so400'), (179041, 'so400'), (179041, 'large'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(path, unit, runtime='3min 1.5s'):
    path.write_text(f'Running as unit: {unit}.service; invocation ID: {hashlib.md5(unit.encode()).hexdigest()}\n'
                    f'\tMaximum resident set size (kbytes): 1234\nService runtime: {runtime}\n')


def main():
    with tempfile.TemporaryDirectory() as tmp:
        runs, src = Path(tmp), Path(tmp) / 'src'
        env = {**os.environ, 'SFORA_NATIVE256_RUNS': str(runs), 'SFORA_NATIVE256_TEST_BYTES': '1'}
        run = lambda *a: subprocess.run([sys.executable, '-B', str(TOOL), *a], env=env, check=True,
                                        capture_output=True, text=True).stdout
        old, new = runs / 'sfora-native256-adaptation-source-v6', runs / 'sfora-native256-adaptation-source-v9'
        old.mkdir(), new.mkdir(), src.mkdir()
        (old / 'large-mechanics-command.sh').write_text(f"#!/bin/bash\nset -euo pipefail\ncd {old}\ntrap 'f' EXIT\nsha256sum -c x\n")
        for f in ('sfora-native256-initialized-cpu-final-footer-v1.py', 'sfora-native256-stop-cgroup-footer-v1.py'):
            (runs / f).write_text('footer')
        files = ('deployed_code_rank.py', 'reference_train_sop_siglip2_compact.py', 'reference_unicom_training.py',
                 'test_siglip2_substrate_adaptation.py', 'train_siglip2_substrate_adaptation.py')
        for f in files:
            (new / f).write_text('v9 ' + f)
        (new / 'execution.json').write_text(json.dumps({f: sha(new / f) for f in files}, indent=2, sort_keys=True) + '\n')
        (src / 'export_siglip2_substrate_adaptation.py').write_text(
            "AUTHORITY_SCHEMA = 'native256-held-authority-v1'\nCOST_POLICY = {'enforcement': 'report_only'}\n"
            "def policy(phase):\n    return {'seconds': 3600, 'phase': phase}\n"
            "from decimal import Decimal\ndef service_runtime(text):\n"
            "    return {'3min 1.5s': Decimal('181.5'), '3min 81ms': Decimal('180.081')}[text]\n")
        for f in ('score_siglip2_substrate_adaptation.py', 'test_siglip2_substrate_held.py',
                  'reference_compare_inshop_sop_warmstart_100.py', 'reference_score_inshop_crop_view_pair.py'):
            (src / f).write_text(f)
        fit = runs / 'fit.json'
        fit.write_text('{}')
        (runs / 'sfora-pe-augmented-100-v2').mkdir()
        split = runs / 'sfora-pe-augmented-100-v2' / 'preflight.json'
        split.write_text('{"held_manifest": []}')
        for arm in ('large', 'so400'):
            (new / f'authority-{arm}-mechanics-v9.json').write_text(json.dumps(
                {'qualifier_root': f'/q/{arm}', 'qualifier_execution_sha256': arm, 'qualifier_authority': {'path': arm},
                 'selected_cpu': {'unit': f'cpu-{arm}'}}))
            unit = f'sfora-native256-mechanics-{arm}-v9'
            (runs / unit).mkdir()
            (runs / unit / 'receipt.json').write_text('{}')
            log(new / f'{unit}.log', unit)
        for seed, arm in ORDER:
            unit = f'sfora-native256-train-{arm}-{seed}-v9'
            (runs / unit).mkdir()
            (runs / unit / 'resume.pt').write_bytes(b'ckpt')
            (runs / unit / 'receipt.json').write_text(json.dumps({
                'terminal_state_sha256': 'e' * 64, 'steps': [{'batch': [seed, i]} for i in range(100)],
                'identity': {'schedule_facts': {str(seed): {'shape': [100, 64]}}, 'schedule_sha256': str(seed)},
                'input_guards': {str(fit): sha(fit), '/other': 'x'}}))
            (new / f'authority-{arm}-train-{seed}-v9.json').write_text('{}')
            log(new / f'{unit}.log', unit)

        assert 'held_execution_sha256' in run('stage', str(src))
        held = runs / 'sfora-native256-held-v9'
        assert len(json.loads((held / 'execution.json').read_text())) == 10
        run('authority')
        auth = json.loads((held / 'native256-held-authority.json').read_text())
        assert [(e['seed'], e['arm']) for e in auth['endpoints']] == list(ORDER)
        assert auth['frozen_split'] == {'path': str(split), 'sha256': sha(split)}
        assert auth['schedules']['179041']['batches'][3] == [179041, 3] and auth['schedules']['179032']['sha256'] == '179032'
        assert auth['prerequisites']['so400']['cpu'] == {'unit': 'cpu-so400'}
        assert auth['prerequisites']['large']['mechanics']['service_seconds'] == 181.5
        assert auth['endpoints'][0]['checkpoint']['path'].endswith('sfora-native256-train-large-179032-v9/resume.pt')
        assert set(auth['resource_policies']) == {'cpu', 'export', 'score'} and auth['training_root'] == str(runs / 'sfora-native256-adaptation-source-v9')

        cpu_line = run('launch', 'cpu', '179032', 'large')
        argv = shlex.split(cpu_line.split(' -- ', 1)[1])
        assert '--setenv=CUDA_VISIBLE_DEVICES=' in argv and not any('CUBLAS' in a for a in argv)
        command = (held / 'sfora-native256-held-cpu-large-179032-v9-command.sh').read_text()
        assert f'cd {held}' in command and '--phase cpu --seed 179032 --arm large' in command and '--cpu-terminal' not in command
        block = command.split("<<'HASHES'\n")[1].split('HASHES\n')[0]
        subprocess.run(['sha256sum', '-c', '--quiet'], input=block, text=True, cwd=held, check=True)

        for seed, arm in ORDER:
            for phase, name in (('cpu', 'proof.json'), ('export', 'receipt.json')):
                unit = f'sfora-native256-held-{phase}-{arm}-{seed}-v9'
                (runs / unit).mkdir()
                (runs / unit / name).write_text('{}')
                log(held / f'{unit}.log', unit, '3min 81ms' if (phase, seed, arm) == ('cpu', 179032, 'so400') else '3min 1.5s')
            run('cpu-terminal', str(seed), arm)
        # h1 HOLD shape: systemd omits a zero seconds term; held v2 parses it via the staged exporter.
        assert json.loads((held / 'cpu-terminal-so400-179032.json').read_text())['service_seconds'] == 180.081
        assert json.loads((held / 'cpu-terminal-large-179032.json').read_text())['service_seconds'] == 181.5
        export_line = run('launch', 'export', '179041', 'so400')
        assert 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in export_line and '--cpu-terminal' in \
            (held / 'sfora-native256-held-export-so400-179041-v9-command.sh').read_text()
        run('wires')
        wires = json.loads((held / 'native256-held-wires.json').read_text())
        assert [(w['seed'], w['arm']) for w in wires['wires']] == list(ORDER) and wires['schema'] == 'native256-held-wires-v1'
        assert wires['wires'][0]['cpu_terminal']['path'].endswith('cpu-terminal-large-179032.json')
        score = run('launch', 'score')
        assert '--setenv=CUDA_VISIBLE_DEVICES=' in shlex.split(score.split(' -- ', 1)[1])
        assert 'native256-held-wires.json' in (held / 'sfora-native256-held-score-v9-command.sh').read_text()
    print('PASS: held stage/authority/cpu-terminal/wires/launch derived from v9 receipts; CUDA hidden for cpu/score')


if __name__ == '__main__':
    main()
