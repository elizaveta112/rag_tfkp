#!/usr/bin/env python3
"""Run a project command, e.g. python run.py prepare_all."""
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parent
for directory in (ROOT, ROOT/'src', ROOT/'scripts'):
    sys.path.insert(0,str(directory))
if __name__ == '__main__':
    commands={p.stem:p for directory in ('scripts','tools') for p in (ROOT/directory).glob('*.py') if not p.name.startswith('_') and p.stem!='layout_provenance'}
    commands['serve_local']=ROOT/'src/serve_local.py'
    if len(sys.argv)<2 or sys.argv[1] in ('-h','--help'):
        print('Usage: python run.py <command> [arguments]\nCommands: '+', '.join(sorted(commands)))
        raise SystemExit(0)
    name=sys.argv.pop(1).removesuffix('.py')
    if name not in commands:
        raise SystemExit('Unknown command: '+name)
    sys.argv[0]=str(commands[name])
    runpy.run_path(str(commands[name]),run_name='__main__')
