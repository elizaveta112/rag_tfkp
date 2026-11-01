#!/usr/bin/env python3
"""Local Qwen3.5-4B GGUF, CPU or CUDA, exposed as Chat Completions on loopback."""
import argparse
import json
import os
import platform
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from common import ROOT, load_config, config_path, dump
from llm import load_binding, TokenCounter
from decisions import MODEL_RESPONSE_SCHEMA


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT/'config.json'))
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--gpu-layers', type=int, default=0, help='-1: offload all layers on CUDA build')
    p.add_argument('--threads', type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument('--usage-log', default=str(ROOT/'endpoint_usage.jsonl'))
    args = p.parse_args()
    config = load_config(args.config)
    binding = load_binding(config)
    runtime = json.loads(config_path(config, 'runtime_json').read_text(encoding='utf-8'))
    weights = config_path(config, 'model_path')
    if not weights.exists() or weights.stat().st_size != runtime['weights']['size_bytes']:
        raise ValueError('Run python tools/prepare_local.py first; model download is not part of prediction')
    from llama_cpp import Llama, LlamaGrammar, __version__, llama_supports_gpu_offload
    if __version__ != runtime['inference']['llama_cpp_python_version']:
        raise ValueError('Installed llama-cpp-python differs from runtime pin')
    if args.gpu_layers != 0 and not llama_supports_gpu_offload():
        raise ValueError('CUDA offload unavailable: rebuild llama-cpp-python with GGML_CUDA=ON, or choose --gpu-layers 0')
    start = time.perf_counter()
    model = Llama(model_path=str(weights), n_ctx=8192, n_batch=256,
                  n_threads=args.threads, n_threads_batch=args.threads,
                  n_gpu_layers=args.gpu_layers, verbose=False,
                  flash_attn=False, logits_all=False, seed=runtime['generation']['seed'])
    model.set_cache(None)
    counter = TokenCounter(config, binding)
    grammar = LlamaGrammar.from_json_schema(json.dumps(MODEL_RESPONSE_SCHEMA), verbose=False)
    lock = threading.Lock()
    log = Path(args.usage_log)
    log.parent.mkdir(parents=True, exist_ok=True)
    from hardware import hardware_info
    dump(ROOT/'server_resources.json', {'load_seconds': time.perf_counter()-start,
         'llama_cpp_python': __version__, 'platform': platform.platform(),
         'processor': platform.processor(), 'threads': args.threads,
         'gpu_layers_requested': args.gpu_layers, 'weights_bytes': weights.stat().st_size,
         'quantization': 'Q4_K_M', 'context': 8192, 'hardware': hardware_info(),
         'gpu_offload_supported': bool(llama_supports_gpu_offload())})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, status, obj):
            raw = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == '/health':
                self.send_json(200, {'status': 'ready', 'model': binding['model_repo']})
            else:
                self.send_json(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_json(404, {'error': 'not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length < 2_000_000:
                    raise ValueError('request size')
                body = json.loads(self.rfile.read(length))
                qid, submission = self.headers.get('X-Question-ID'), self.headers.get('X-Submission-ID')
                if not qid or not submission or body.get('model') != binding['model_repo']:
                    raise ValueError('required headers/model')
                expected = binding['request_body']
                for key, value in expected.items():
                    if body.get(key) != value:
                        raise ValueError('generation settings differ from runtime: '+key)
                prompt = counter.render(body['messages'])
                tokens = model.tokenize(prompt.encode('utf-8'), add_bos=False, special=True)
                if len(tokens)+expected['max_tokens'] > 8192:
                    raise ValueError('context budget')
                sampling = {k: expected[k] for k in ('temperature', 'top_p', 'top_k', 'min_p', 'seed')}
                with lock:
                    t = time.perf_counter()
                    # No KV cache, answer cache or RNG history is shared between questions.
                    model.reset()
                    result = model.create_completion(tokens, max_tokens=expected['max_tokens'],
                        stop=['<|im_end|>', '<|endoftext|>'], echo=False, grammar=grammar, **sampling)
                    model.reset()
                    usage = result['usage']
                    elapsed = round((time.perf_counter()-t)*1000, 3)
                    with log.open('a', encoding='utf-8') as f:
                        f.write(json.dumps({'id': qid, 'submission_id': submission,
                            'prompt_tokens': usage['prompt_tokens'], 'completion_tokens': usage['completion_tokens'],
                            'elapsed_ms': elapsed, 'finish_reason': result['choices'][0]['finish_reason']})+'\n')
                self.send_json(200, {'id': 'chatcmpl-'+uuid.uuid4().hex, 'object': 'chat.completion',
                    'model': binding['model_repo'], 'choices': [{'index': 0,
                    'message': {'role': 'assistant', 'content': result['choices'][0]['text']},
                    'finish_reason': result['choices'][0]['finish_reason']}], 'usage': usage})
            except Exception as exc:
                self.send_json(400, {'error': type(exc).__name__+': '+str(exc)[:180]})

    print(f'READY http://127.0.0.1:{args.port}/v1 Qwen3.5-4B Q4_K_M', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
