import sys as _sys
from pathlib import Path as _Path
for _directory in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1]/'src', _Path(__file__).resolve().parents[1]/'scripts'):
    _sys.path.insert(0, str(_directory))
from common import project_path
#!/usr/bin/env python3
"""Start local Qwen, run five experiments, then shut down the model process."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from common import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu-layers', type=int, default=-1)
    p.add_argument('--threads', type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument('--config', default=str(project_path(ROOT, 'config.json')))
    p.add_argument('--index', default=str(ROOT/'index'))
    p.add_argument('--corpus', default=str(project_path(ROOT, 'corpus.jsonl')))
    p.add_argument('--questions', default=str(project_path(ROOT, 'dev_questions.jsonl')))
    p.add_argument('--gold', default=str(project_path(ROOT, 'dev_gold.jsonl')))
    p.add_argument('--output-dir', default=str(ROOT/'experiments'))
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--resume', action='store_true', help='Continue exactly the same development run and retain its endpoint journal')
    p.add_argument('--smoke-test', action='store_true', help='Run one independent final-mode question before the full study')
    args = p.parse_args()
    if args.resume and args.smoke_test:
        p.error('--resume and --smoke-test cannot be combined')
    env = dict(os.environ, LLM_BASE_URL=f'http://127.0.0.1:{args.port}/v1',
        LLM_MODEL='Qwen/Qwen3.5-4B', LLM_API_KEY='LOCAL',
        SUBMISSION_ID=os.environ.get('SUBMISSION_ID', 'smagina-tfkp-local-v4'))
    usage_log = ROOT/'smoke/endpoint_usage.jsonl' if args.smoke_test else project_path(ROOT, 'endpoint_usage.jsonl')
    attempts = project_path(ROOT, 'request_attempts.jsonl')
    if args.smoke_test:
        env.pop('RAG_CHECKPOINT_DIR', None)
        env.pop('RAG_ATTEMPT_LOG', None)
    else:
        env['RAG_ATTEMPT_LOG'] = str(attempts)
    if args.resume:
        from request_journal import validate_live_copy
        validate_live_copy(attempts)
        from run_state import verify_saved_prefix
        from run_experiments import MODES
        verify_saved_prefix(args.output_dir, usage_log, MODES, attempts)
    elif not args.smoke_test:
        if attempts.exists() and attempts.stat().st_size:
            archived = ROOT/'experiments/archive'/f'request_attempts_{time.time_ns()}.jsonl'
            archived.parent.mkdir(parents=True, exist_ok=True)
            attempts.replace(archived)
        attempts.write_text('', encoding='utf-8')
        if env.get('RAG_CHECKPOINT_DIR'):
            live = Path(env['RAG_CHECKPOINT_DIR'])/'request_attempts_live.jsonl'
            live.parent.mkdir(parents=True, exist_ok=True)
            live.write_text('', encoding='utf-8')
    if not args.resume and usage_log.exists() and usage_log.stat().st_size:
        archive = ROOT/'experiments/archive'
        archive.mkdir(parents=True, exist_ok=True)
        usage_log.replace(archive/f'endpoint_usage_{time.time_ns()}.jsonl')
    with (project_path(ROOT, 'server.log')).open('w') as log:
        server = subprocess.Popen([sys.executable, str(project_path(ROOT, 'serve_local.py')),
            '--config', str(Path(args.config).resolve()), '--gpu-layers', str(args.gpu_layers),
            '--threads', str(args.threads), '--port', str(args.port), '--usage-log', str(usage_log)], stdout=log, stderr=log)
        try:
            ready = False
            for _ in range(180):
                if server.poll() is not None:
                    raise RuntimeError('Local model startup failed. Inspect server.log.')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=1) as r:
                        ready = json.load(r).get('status') == 'ready'
                    if ready:
                        break
                except OSError:
                    time.sleep(1)
            if not ready:
                raise RuntimeError('Model startup timeout; inspect server.log')
            if args.smoke_test:
                from common import read_jsonl, write_jsonl
                folder = ROOT/'smoke'
                folder.mkdir(exist_ok=True)
                write_jsonl(folder/'question.jsonl', read_jsonl(args.questions)[:1])
                subprocess.run([sys.executable, str(project_path(ROOT, 'predict.py')), '--index', str(Path(args.index).resolve()),
                    '--questions', str(folder/'question.jsonl'), '--config', str(Path(args.config).resolve()),
                    '--output', str(folder/'answer.jsonl'), '--trace', str(folder/'trace.jsonl'), '--mode', 'final'],
                    env=env, check=True)
                trace = read_jsonl(folder/'trace.jsonl')[0]
                print((folder/'answer.jsonl').read_text(), flush=True)
                if trace['status'] != 'ok' or not trace['usage_verifiable']:
                    raise RuntimeError('Smoke test failed; inspect smoke/trace.jsonl and server.log')
                print('Smoke test passed. This separate question is not part of the five measured runs.', flush=True)
                return
            print('Local Qwen ready; starting five real experiments (75 question/configuration pairs).', flush=True)
            cmd = [sys.executable, str(project_path(ROOT, 'run_experiments.py'))]
            for key in ('index', 'corpus', 'questions', 'gold', 'config', 'output_dir'):
                cmd += ['--'+key.replace('_', '-'), str(Path(getattr(args, key)).resolve())]
            if args.resume:
                cmd.append('--resume')
            subprocess.run(cmd, env=env, check=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == '__main__':
    main()
