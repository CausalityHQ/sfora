#!/usr/bin/env python3
"""Hermetic check of stage_native256_v9.py against a fake v6 tree (stdlib only, no DGX)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(__file__).with_name('stage_native256_v9.py')


def run(runs, *args):
    env = {**os.environ, 'SFORA_NATIVE256_RUNS': str(runs), 'SFORA_NATIVE256_TEST_BYTES': '1'}
    return subprocess.run([sys.executable, '-B', str(TOOL), *args], env=env, check=True, capture_output=True, text=True).stdout


def main():
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp)
        old, src = runs / 'sfora-native256-adaptation-source-v6', runs / 'src'
        old.mkdir(), src.mkdir()
        files = ('deployed_code_rank.py', 'reference_train_sop_siglip2_compact.py', 'reference_unicom_training.py',
                 'test_siglip2_substrate_adaptation.py', 'train_siglip2_substrate_adaptation.py')
        for f in files:
            (old / f).write_text('v6 ' + f)
        for f in files[3:]:
            (src / f).write_text('v9 ' + f)
        for f in ('sfora-native256-initialized-cpu-final-footer-v1.py', 'sfora-native256-stop-cgroup-footer-v1.py'):
            (runs / f).write_text('footer')
        import hashlib
        code = {f: hashlib.sha256((old / f).read_bytes()).hexdigest() for f in files}
        (old / 'execution.json').write_text(json.dumps(code, indent=2, sort_keys=True) + '\n')
        header = f"#!/bin/bash\nset -euo pipefail\ncd {old}\ntrap 'footer' EXIT\n"
        for name, phase, seed in (('large-mechanics', 'mechanics', 179032), ('large-train-179041', 'train', 179041)):
            (old / f'authority-{name}-v6.json').write_text(json.dumps({
                'arm': 'large', 'phase': phase, 'seed': seed, 'execution_sha256': 'old', 'qualifier_root': '/q',
                'selected_cpu': {'unit': 'cpu'}, 'selected_mechanics': None if phase == 'mechanics' else {'old': 1},
                'resource_policy': {'seconds': 300, 'host_bytes': 8 * 1024**3}}))
            (old / f'{name}-command.sh').write_text(header + "sha256sum -c <<'HASHES'\nx\nHASHES\nrun\n")

        assert 'execution_sha256' in run(runs, 'stage', str(src))
        new = runs / 'sfora-native256-adaptation-source-v9'
        assert (new / 'deployed_code_rank.py').read_text() == 'v6 deployed_code_rank.py'
        assert (new / 'train_siglip2_substrate_adaptation.py').read_text().startswith('v9')
        run(runs, 'authority', 'mechanics', 'large')
        mech = json.loads((new / 'authority-large-mechanics-v9.json').read_text())
        assert mech['resource_policy'] == {'seconds': 3600, 'host_bytes': 8 * 1024**3} and mech['selected_mechanics'] is None
        assert mech['qualifier_root'] == '/q' and mech['selected_cpu'] == {'unit': 'cpu'}
        command = (new / 'large-mechanics-command.sh').read_text()
        assert command.startswith(header.replace(str(old), str(new))) and command.count("sha256sum -c <<'HASHES'") == 2
        assert '--phase mechanics --arm large --seed 179032' in command
        block = command.split("<<'HASHES'\n")[1].split('HASHES\n')[0]
        subprocess.run(['sha256sum', '-c', '--quiet'], input=block, text=True, cwd=new, check=True)

        # Simulated completed mechanics unit -> TRAIN authority binds it.
        out = runs / 'sfora-native256-mechanics-large-v9'
        out.mkdir()
        (out / 'receipt.json').write_text('{}')
        (new / 'sfora-native256-mechanics-large-v9.log').write_text(
            'Running as unit: sfora-native256-mechanics-large-v9.service; invocation ID: ' + 'a' * 32 + '\n'
            '\tMaximum resident set size (kbytes): 4448844\nService runtime: 2min 50.814s\n')
        run(runs, 'authority', 'train', 'large', '179041')
        train = json.loads((new / 'authority-large-train-179041-v9.json').read_text())
        selected = train['selected_mechanics']
        assert selected['service_seconds'] == 170.814 and selected['invocation_id'] == 'a' * 32
        assert selected['native_peak_rss_kib'] == 4448844 and selected['unit'] == 'sfora-native256-mechanics-large-v9'
        log = new / 'sfora-native256-mechanics-large-v9.log'
        good = log.read_text()
        for bad in (good + 'watchdog: stall (no progress 600s)\n', good.replace('2min 50.814s', '5min 716ms')):
            log.write_text(bad)
            try:
                run(runs, 'authority', 'train', 'large', '179041')
                raise AssertionError('invalid mechanics terminal admitted')
            except subprocess.CalledProcessError:
                pass
        log.write_text(good)
        launch = run(runs, 'launch', 'train', 'large', '179041')
        import shlex
        argv = shlex.split(launch.split(' -- ', 1)[1])
        assert argv[argv.index('-p', argv.index('KillMode=control-group')) + 1].startswith('ExecStopPost=/usr/bin/python3 ')
        assert argv[-2:] == ['/bin/bash', str(new / 'large-train-179041-command.sh')] and 'BACKSTOP_SECONDS=3780' in launch
        assert 'RuntimeMaxSec=3600' in launch and 'stop sfora-native256-train-large-179041-v9.service' in launch
        assert 'sfora-native256-v9-tools/run_with_progress_watchdog.sh' in launch and 'large-train-179041-command.sh' in launch
        assert ("LIVENESS_CMD='cg=$(systemctl --user show -P ControlGroup sfora-native256-train-large-179041-v9.service); "
                '[ -n "$cg" ] && cat "/sys/fs/cgroup$cg/cpu.stat"\'' in launch)

        try:  # refuses to restage over an existing root
            run(runs, 'stage', str(src))
            raise AssertionError('restage admitted')
        except subprocess.CalledProcessError:
            pass
    print('PASS: stage/authority/command/launch derived from v6; hash blocks verify; TRAIN binds v9 mechanics terminal')


if __name__ == '__main__':
    main()
