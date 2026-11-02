"""Durable attempt markers prevent hiding calls after a stale checkpoint restore."""
import json
import os
import shutil
from pathlib import Path


def record_attempt(row):
    setting = os.environ.get('RAG_ATTEMPT_LOG')
    if not setting:
        return
    path = Path(setting)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    backup = os.environ.get('RAG_CHECKPOINT_DIR')
    if backup:
        live = Path(backup)/'request_attempts_live.jsonl'
        live.parent.mkdir(parents=True, exist_ok=True)
        # Persist before sending HTTP. A restore older than this file is unsafe.
        shutil.copyfile(path, live)


def validate_live_copy(path):
    backup = os.environ.get('RAG_CHECKPOINT_DIR')
    if not backup:
        return
    live = Path(backup)/'request_attempts_live.jsonl'
    saved = Path(path)
    if not live.exists() or not saved.exists() or live.read_bytes() != saved.read_bytes():
        raise ValueError('Checkpoint and live request-attempt journal differ. A later call may have occurred; do not hide its cost. Preserve the files and start a separate fresh run.')
