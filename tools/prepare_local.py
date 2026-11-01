#!/usr/bin/env python3
"""Network setup only; inference never downloads assets."""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, load_config, config_path, dump, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT/'config.json'))
    p.add_argument('--existing-model', help='Reuse verified download without a second copy')
    args = p.parse_args()
    config = load_config(args.config)
    runtime_path = config_path(config, 'runtime_json')
    runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
    binding = {'model_repo': runtime['model']['repo'], 'model_revision': runtime['model']['revision'],
        'official_runtime_path': 'runtime.json', 'official_runtime_sha256': sha256(runtime_path),
        'request_body': runtime['generation'], 'timeout_seconds': runtime['timeout_seconds'],
        'completion_budget_includes_reasoning': True, 'served_model': runtime['model']['repo'],
        'origin': 'self-defined runtime, as clarified by instructor; no provided runtime exists'}
    dump(config_path(config, 'runtime_binding'), binding)
    dest = config_path(config, 'tokenizer_dir')
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for filename in ('tokenizer.json', 'chat_template.jinja'):
        path = dest/filename
        if not path.exists():
            url = f"https://huggingface.co/{runtime['model']['repo']}/resolve/{runtime['model']['revision']}/{filename}"
            with urllib.request.urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        hashes[filename] = sha256(path)
    dump(dest/'manifest.json', {'repo': runtime['model']['repo'],
         'revision': runtime['model']['revision'], 'sha256': hashes})
    target = config_path(config, 'model_path')
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.existing_model:
        source = Path(args.existing_model).resolve()
        if target.resolve() != source:
            # A symlink avoids copying multi-GB weights in scratch environments.
            target.unlink(missing_ok=True)
            target.symlink_to(source)
    if not target.exists():
        info = runtime['weights']
        url = f"https://huggingface.co/{info['repo']}/resolve/{info['revision']}/{info['filename']}"
        partial = target.with_suffix('.partial')
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as f:
            shutil.copyfileobj(response, f, length=8*1024*1024)
        partial.replace(target)
    digest = hashlib.sha256()
    with target.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(chunk)
    if target.stat().st_size != runtime['weights']['size_bytes'] or digest.hexdigest() != runtime['weights']['sha256']:
        raise ValueError('Weights size/hash mismatch')
    print('Prepared pinned Qwen3.5-4B tokenizer and Q4_K_M weights; no paid API is needed.')


if __name__ == '__main__':
    main()
