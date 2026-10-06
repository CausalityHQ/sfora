"""Restore only the byte-qualified runtime packages; retain reversible artifacts."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path('/home/riomus/runs/sfora-runtime-recovery-artifacts-v1')
AUTH_SHA = 'cd548de19967f80f34b69ed0d921b7a26ef417c985dd7ecc59a2a416eb897905'

def checked(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    checked(os.geteuid() == 0, 'root required')
    authority_path = ROOT / 'restore-authority.json'
    checked(digest(authority_path) == AUTH_SHA, 'restore authority changed')
    a = json.loads(authority_path.read_text())
    checked(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(),
            'GPU compute owner present')
    for r in a['packages'] + a['reverse_packages']:
        p = Path(r['path'])
        checked(p.is_file() and p.resolve() == p and digest(p) == r['sha256'], 'package artifact changed')
    for p, expected in a['expected_system'].items():
        extracted = ROOT / 'extracted' / p.lstrip('/')
        source = extracted if extracted.exists() else Path(p)
        checked(digest(source) == expected, 'original runtime bytes not recovered')
    boot = Path('/etc/default/grub.d/90-sfora-qualified-runtime.cfg')
    checked(not boot.exists() and not boot.is_symlink(), 'exclusive boot override required')
    checked(Path('/boot/vmlinuz-' + a['kernel']).is_file(), 'retained kernel missing')
    argv = ['apt-get', 'install', '--allow-downgrades', '--no-install-recommends',
            *[r['path'] for r in a['packages']]]
    simulation = subprocess.run([*argv[:1], '-s', *argv[1:]], check=True, text=True, stdout=subprocess.PIPE)
    (ROOT / 'final-simulation.log').write_text(simulation.stdout)
    removals = re.findall(r'^Remv (\S+)', simulation.stdout, re.M)
    checked(removals == a['permitted_remove'], 'unexpected package removal')
    changed = re.findall(r'^Inst (\S+)', simulation.stdout, re.M)
    checked(set(changed) == {r['package'] for r in a['packages']}, 'unexpected package transaction')
    subprocess.run([*argv[:2], '-y', *argv[2:]], check=True,
                   env={**os.environ, 'DEBIAN_FRONTEND': 'noninteractive', 'NEEDRESTART_MODE': 'l'})
    for r in a['packages']:
        version = subprocess.check_output(['dpkg-query', '-W', '-f=${Version}', r['package']], text=True)
        checked(version == r['old'], 'installed version differs')
    subprocess.run(['apt-mark', 'hold', *[r['package'] for r in a['packages']]], check=True)
    with boot.open('x') as stream:
        stream.write("# Sfora bounded frozen-protocol runtime; remove after procedure closeout.\n")
        stream.write('GRUB_DEFAULT=' + repr(a['boot_default']) + '\n')
    subprocess.run(['update-grub'], check=True)
    checked(a['boot_default'].split('>')[1] in Path('/boot/grub/grub.cfg').read_text(), 'boot entry missing')
    for p, expected in a['expected_system'].items():
        checked(digest(p) == expected, 'restored system bytes differ')
    checked(digest(authority_path) == AUTH_SHA, 'authority changed at exit')
    record = {'schema': 'sfora-qualified-runtime-restored-userspace-v1', 'pass': True,
              'reboot_required': True, 'authority_sha256': AUTH_SHA, 'system_hashes': a['expected_system'],
              'original_train_failure_remains_failure': True, 'quality_read': False,
              'invocation_id': os.environ['INVOCATION_ID'], 'boot_default': a['boot_default']}
    with (ROOT / 'restore-receipt.json').open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('QUALIFIED_USERSPACE_RESTORED_REBOOT_REQUIRED', flush=True)

if __name__ == '__main__':
    main()
