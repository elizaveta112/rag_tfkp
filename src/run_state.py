from common import project_path
"""Explicit continuation of one interrupted development run, never default caching."""
import json
from pathlib import Path
from common import ROOT, read_jsonl, sha256, dump, load_config, config_path


def fingerprint(index, questions, config, mode):
    sources = ('predict.py', 'retrieval.py', 'formulas.py', 'llm.py', 'common.py',
               'json_math.py', 'run_state.py', 'serve_local.py', 'hardware.py', 'request_journal.py',
               'decisions.py')
    return {'version': 1, 'mode': mode, 'questions_sha256': sha256(questions),
        'config_sha256': sha256(config),
        'index_sha256': sha256(Path(index)/'index.json.gz'),
        'binding_sha256': sha256(config_path(load_config(config), 'runtime_binding')),
        'source_sha256': {name: sha256(project_path(ROOT, name)) for name in sources}}


def start_or_resume(output, trace, questions, identity, resume=False):
    """Only a complete, matching prefix may be continued; ambiguous writes abort."""
    output, trace = Path(output), Path(trace)
    manifest = Path(str(output)+'.run.json')
    if not resume:
        dump(manifest, identity)
        return 0
    present = [p.exists() for p in (output, trace, manifest)]
    if not any(present):
        dump(manifest, identity)
        return 0
    if not all(present):
        raise ValueError('Incomplete run files: preserve the endpoint journal and inspect accounting before continuation')
    if json.loads(manifest.read_text(encoding='utf-8')) != identity:
        raise ValueError('Run inputs, code, index or runtime changed; continuation is forbidden. Start a fresh experiment directory.')
    answers = read_jsonl(output) if output.stat().st_size else []
    traces = read_jsonl(trace) if trace.stat().st_size else []
    ids = [q['id'] for q in questions]
    if ([r['id'] for r in answers] != ids[:len(answers)] or
        [r['id'] for r in traces] != [r['id'] for r in answers]):
        raise ValueError('Answers and trace are not the same committed question prefix; do not silently rerun or drop charged calls')
    return len(answers)


def verify_saved_prefix(directory, journal, modes, attempts_path=None):
    """Abort on a charged call without a committed output/trace, never discard it."""
    expected = []
    expected_attempts = []
    for mode in modes:
        path = Path(directory)/f'{mode}.trace.jsonl'
        if path.exists() and path.stat().st_size:
            for row in read_jsonl(path):
                for event in row['events']:
                    expected.append((row['id'], event.get('input_tokens'), event.get('output_tokens')))
                    expected_attempts.append({'id': row['id'], 'stage': event.get('stage'),
                        'local_prompt_tokens': event.get('local_prompt_tokens'),
                        'requested_output_cap': event.get('requested_output_cap')})
    journal = Path(journal)
    rows = [json.loads(s) for s in journal.read_text().splitlines() if s.strip()] if journal.exists() else []
    actual = [(r['id'], r['prompt_tokens'], r['completion_tokens']) for r in rows]
    if expected != actual:
        raise ValueError('Saved traces and endpoint journal differ. An interrupted call may have been charged; continuation is unsafe. Preserve the files and start a separate fresh run.')
    if attempts_path is not None:
        path = Path(attempts_path)
        attempts = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
        if attempts != expected_attempts:
            raise ValueError('Request-attempt journal contains calls without a committed trace. Do not silently repeat them.')
