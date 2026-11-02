"""Verify historical measurements after a documented path-only migration."""
import sys
from pathlib import Path
for directory in (Path(__file__).resolve().parents[1]/'src',Path(__file__).resolve().parent):
    sys.path.insert(0,str(directory))
import json
from common import sha256,project_path


def verify_recorded_run(root, mode, state, current):
    if state == current:
        return 'current_layout'
    path=Path(root)/'artifacts/layout_migration.json'
    if not path.exists():
        raise ValueError('Run identity changed without a layout migration: '+mode)
    migration=json.loads(path.read_text())
    if state != migration['original_run_identities'].get(mode):
        raise ValueError('Unknown historical run identity: '+mode)
    if current['source_sha256'] != migration['current_source_sha256']:
        raise ValueError('Source changed after the recorded layout migration')
    if current['config_sha256'] != migration['relocated_measured_config_sha256']:
        raise ValueError('Measured config changed after layout migration')
    for key in ('version','mode','questions_sha256','index_sha256','binding_sha256'):
        if current[key] != state[key]:
            raise ValueError('Migration changed measured input: '+key)
    for name,expected in migration['preserved_files_sha256'].items():
        file=project_path(root,name)
        if file.exists() and sha256(file)!=expected:
            raise ValueError('Historical measured file changed: '+name)
    return 'historical_layout_migration'
