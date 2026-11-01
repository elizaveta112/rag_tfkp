#!/usr/bin/env python3
import argparse
import gzip
import json
import platform
import resource
import sys
import time
from pathlib import Path
from common import read_jsonl, load_config, config_path, dump, sha256
from formulas import chunk_text
from retrieval import build_views


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    start = time.perf_counter()
    config = load_config(args.config)
    blocks = read_jsonl(args.corpus)
    chunks = []
    for order, b in enumerate(blocks):
        if (type(b.get('page')) is not int or b['page'] < 1 or
                not isinstance(b.get('text'), str) or not b['text'].strip() or
                not isinstance(b.get('section'), str)):
            raise ValueError(f"Invalid corpus block {b['block_id']}")
        for part, text in enumerate(chunk_text(b['text'], config['retrieval']['chunk_chars'])):
            chunks.append({'chunk_id': f'{order}:{part}', 'block_id': b['block_id'],
                           'page': b['page'], 'section': b['section'], 'text': text})
    data = {'version': 1, 'blocks': blocks, 'chunks': chunks, 'views': build_views(chunks)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_ready = False
    if config_path(config, 'runtime_binding').exists():
        from llm import load_binding, prepare_tokenizer_assets
        prepare_tokenizer_assets(config, load_binding(config), output/'tokenizer')
        tokenizer_ready = True
    target = output/'index.json.gz'
    # mtime=0 makes the compressed artifact reproducible.
    with target.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as gz:
            gz.write(json.dumps(data, ensure_ascii=False, sort_keys=True).encode())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest = {'version': 1, 'corpus_sha256': sha256(args.corpus),
                'index_sha256': sha256(target), 'blocks': len(blocks), 'chunks': len(chunks),
                'chunk_chars': config['retrieval']['chunk_chars'],
                'build_seconds': time.perf_counter()-start,
                'peak_rss_bytes': peak if sys.platform == 'darwin' else peak*1024,
                'index_bytes': sum(f.stat().st_size for f in output.rglob('*') if f.is_file()
                                   and f.name != 'manifest.json'),
                'tokenizer_ready': tokenizer_ready, 'python': platform.python_version(),
                'platform': platform.platform(), 'processor': platform.processor(),
                'source_sha256': {f: sha256(Path(__file__).parent/f)
                                  for f in ('build_index.py', 'formulas.py', 'retrieval.py')}}
    dump(output/'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), file=sys.stderr)


if __name__ == '__main__':
    main()
