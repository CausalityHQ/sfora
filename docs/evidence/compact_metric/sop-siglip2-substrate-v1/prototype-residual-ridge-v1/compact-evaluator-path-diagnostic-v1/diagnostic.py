#!/usr/bin/env python3
"""Root-owned source-only diagnostic of the frozen evaluator; never qualifies."""
import json
from pathlib import Path
import runpy
import sys
import traceback

script = str(Path(sys.argv[1]).resolve())
sys.argv = sys.argv[1:]
reported = False

def trace(frame, event, arg):
    global reported
    if event == 'call':
        frame.f_trace_lines = False
    if event == 'call' and frame.f_code.co_name == 'native_start' and frame.f_code.co_filename == script:
        sys.settrace(None)
        raise SystemExit('DIAGNOSTIC_STOP: source authority passed; native_start not executed')
    if event == 'exception' and not reported:
        exception_type, error, _ = arg
        if isinstance(error, ValueError) and str(error) == 'canonical file required':
            reported = True
            owner = frame
            while owner is not None and owner.f_code.co_name not in ('bound_file', 'canonical'):
                owner = owner.f_back
            facts = {'diagnostic_only': True, 'message': str(error)}
            if owner is not None:
                value = owner.f_locals.get('path')
                facts.update(function=owner.f_code.co_name, source=owner.f_code.co_filename,
                             line=owner.f_lineno, path=str(value),
                             expected=str(owner.f_locals.get('expected')))
                if isinstance(value, (str, Path)):
                    path = Path(value)
                    facts.update(absolute=path.is_absolute(), resolved=str(path.resolve()),
                                 exists=path.exists(), regular_file=path.is_file(), symlink=path.is_symlink())
            print('CANONICAL_PATH_DIAGNOSTIC '+json.dumps(facts,sort_keys=True),file=sys.stderr,flush=True)
            traceback.print_stack(frame,file=sys.stderr)
            sys.settrace(None)
    return trace

sys.settrace(trace)
runpy.run_path(script,run_name='__main__')
