import sys as _sys
from pathlib import Path as _Path
for _directory in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1]/'src', _Path(__file__).resolve().parents[1]/'scripts'):
    _sys.path.insert(0, str(_directory))
#!/usr/bin/env python3
"""Install this source kit over an idle project, preserving heavy assets and old runs."""
import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

RELEASE = '2026-10-29-v4'


def install(archive_path, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or sum(e.file_size for e in archive.infolist()) > 100*1024**2:
            raise ValueError('Unexpected duplicate entries or oversized source archive')
        for entry in archive.infolist():
            path = Path(entry.filename)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != 'rag_solution':
                raise ValueError('Unsafe archive path')
            if ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Symlinks are not accepted in source archives')
        manifest = json.loads(archive.read('rag_solution/run_kit_manifest.json'))
        if manifest.get('release') != RELEASE or manifest.get('results_included') is not False:
            raise ValueError('Use the v4 source kit, not a result or old checkpoint archive')
        payload = {}
        for name, expected in manifest['files_sha256'].items():
            data = archive.read('rag_solution/'+name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('Source checksum mismatch: '+name)
            payload[name] = data
        payload['run_kit_manifest.json'] = archive.read('rag_solution/run_kit_manifest.json')
    previous = None
    if (root/'predict.py').exists():
        tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        previous = root.parent/'rag_previous_runs'/tag
        previous.mkdir(parents=True)
        # Back up the exact old source before replacing it. No weights/venv/PDF.
        with zipfile.ZipFile(previous/'previous_source.zip', 'w', zipfile.ZIP_DEFLATED) as backup:
            for name in payload:
                old = root/name
                if old.is_file():
                    backup.write(old, name)
        measured = ['experiments', 'index', 'corpus.jsonl', 'corpus_audit.json',
                    'retrieval_audit.json', 'dev_answers.jsonl', 'dev_trace.jsonl',
                    'experiments.csv', 'endpoint_usage.jsonl', 'request_attempts.jsonl',
                    'usage_reconciliation.json', 'completion_manifest.json',
                    'final_metrics.json', 'semantic_reviews.json', 'error_reviews.json',
                    'submission.zip', 'results_for_review.zip', 'server_resources.json',
                    'server.log', 'report.md', 'STATUS.md', 'private_review', 'smoke']
        for name in measured:
            old = root/name
            if old.exists():
                shutil.move(str(old), str(previous/name))
    root.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print('Installed', RELEASE, 'in', root)
    print('Previous results:', previous or 'no previous project')
    print('Existing .venv, model/tokenizer assets and source PDF are preserved.')
    print('Rebuild with prepare_all.py and start a fresh run; RESUME=False.')
    return previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    install(args.archive, args.root)


if __name__ == '__main__':
    main()
