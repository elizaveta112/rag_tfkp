import sys as _sys
from pathlib import Path as _Path
for _directory in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1]/'src', _Path(__file__).resolve().parents[1]/'scripts'):
    _sys.path.insert(0, str(_directory))
"""Optional private run backup; outputs are filesystem copies, never model inputs."""
import os
import zipfile
from pathlib import Path
from common import ROOT

SKIP = {'assets', 'index', 'source', '.git', '__pycache__', 'checkpoint', 'archive', '.env', 'credentials.json'}


def backup_if_enabled():
    setting = os.environ.get('RAG_CHECKPOINT_DIR')
    if not setting:
        return
    directory = Path(setting).expanduser().resolve()
    if directory == ROOT or directory.is_relative_to(ROOT):
        raise ValueError('Checkpoint directory must be outside the project')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/'RAG_TFKP_checkpoint.zip'
    partial = target.with_suffix('.tmp')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            rel = path.relative_to(ROOT)
            if not path.is_file() or any(p in SKIP or p.startswith('.venv') for p in rel.parts):
                continue
            if path.suffix in {'.zip', '.pdf', '.gguf', '.safetensors', '.pyc'} or path.name == 'corpus.jsonl':
                continue
            archive.write(path, Path('rag_solution')/rel)
    partial.replace(target)


if __name__ == '__main__':
    backup_if_enabled()
    print('Checkpoint refreshed.' if os.environ.get('RAG_CHECKPOINT_DIR') else 'No checkpoint directory configured.')
