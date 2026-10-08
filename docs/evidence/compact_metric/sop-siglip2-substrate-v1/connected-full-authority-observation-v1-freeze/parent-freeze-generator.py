import hashlib
import json
from pathlib import Path
import runpy
import shlex

repo = Path('/home/rb/worktrees/sfora-positive-causality')
prep = Path('/home/rb/agents/handoffs/sfora-connected-full-authority-observation-v1-prep')
remote = '/home/riomus/runs/sfora-connected-full-authority-observation-source-v1'
unit = 'sfora-connected-full-authority-observation-v1'
driver = runpy.run_path(str(repo/'scripts/observe_connected_full_authority.py'))
names = ('observe_connected_full_authority.py','test_connected_full_authority_observation.py')
def sha(raw): return hashlib.sha256(raw).hexdigest()
def write(name, raw):
    with (prep/name).open('xb') as f: f.write(raw)
for name in names: write(name,(repo/'scripts'/name).read_bytes())
output = '/home/riomus/runs/sfora-connected-full-authority-observation-unused-v1'
diagnostic = '/home/riomus/runs/sfora-connected-full-authority-observation-v1.jsonl'
manifest = dict(schema=driver['SCHEMA'],observer={'path':remote+'/'+names[0],'sha256':sha((prep/names[0]).read_bytes())},
    test={'path':remote+'/'+names[1],'sha256':sha((prep/names[1]).read_bytes())},
    python=driver['PYTHON'],evaluator=driver['EVALUATOR'],execution=driver['EXECUTION'],
    authority=driver['AUTHORITY'],command=driver['COMMAND'],original_argv=driver['original_argv'](),
    argv=driver['original_argv']()[:-1]+[output],output=output,diagnostic=diagnostic,unit=unit,
    resource_policy=driver['POLICY'],both_locks_held=True,qualification_eligible=False,state_reuse_eligible=False)
raw = (json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
write('manifest.json',raw)
facts = [manifest[k] for k in ('observer','test','python','evaluator','execution','authority','command')]
facts += [driver['EVALUATOR_TEST'],{'path':remote+'/manifest.json','sha256':sha(raw)},
    {'path':'/home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py','sha256':'bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995'},
    {'path':'/home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py','sha256':'e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45'}]
table = ''.join(f"{v['sha256']}  {v['path']}\n" for v in facts)
command = '''#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
'''+"sha256sum -c <<'HASHES'\n"+table+"HASHES\n"+shlex.join(['/home/riomus/group-learning/.venv/bin/python','-B',remote+'/'+names[0],'--manifest',remote+'/manifest.json','--manifest-sha256',sha(raw)])+'\n'+"sha256sum -c <<'HASHES'\n"+table+"HASHES\n"
write('command.sh',command.encode())
launch = f'''#!/bin/bash
set -euo pipefail
test ! -e {shlex.quote(output)}
test ! -e {shlex.quote(diagnostic)}
if systemctl --user list-units --state=running --no-legend | grep -q sfora; then exit 90; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock flock -n /home/riomus/.sfora-siglip2-gpu.lock /bin/true
bash -n {remote}/command.sh
printf '%s  %s\\n' {sha(command.encode())} {remote}/command.sh | sha256sum -c
systemd-run --user --unit={unit} --wait --pipe --collect --property=RuntimeMaxSec=900 --property=MemoryMax=8589934592 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop --property='ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES= --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TOKENIZERS_PARALLELISM=false /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock /usr/bin/time -v /bin/bash -c 'set -e; sudo -n /bin/sync; sudo -n /bin/sh -c "echo 3 > /proc/sys/vm/drop_caches"; exec /bin/bash {remote}/command.sh'
'''
write('launch.sh',launch.encode())
controller = f'''#!/bin/bash
set -u
set -o noclobber
/bin/bash {remote}/launch.sh > {remote}/original.log 2>&1
sfora_launch_status=$?
printf '%s\\n' "$sfora_launch_status" > {remote}/terminal-status.txt
exit "$sfora_launch_status"
'''
write('controller.sh',controller.encode())
freeze = dict(schema='connected-authority-observation-root-freeze-v1',remote_root=remote,unit=unit,
    diagnostic=diagnostic,unused_output=output,qualification_eligible=False,state_reuse_eligible=False,
    resource_policy=driver['POLICY'],files={p.name:sha(p.read_bytes()) for p in sorted(prep.iterdir()) if p.is_file()},
    source_commit='9954f3d7',original_failure_preserved=True,stop_before_native_start=True,
    root_checks='worker5 stdlib tests PASS; root2 targeted control/pin tests PASS; full source read; no native execution')
write('freeze.json',(json.dumps(freeze,sort_keys=True,indent=2)+'\n').encode())
print(json.dumps(freeze,sort_keys=True))
