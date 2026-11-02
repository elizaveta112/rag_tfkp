import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
for directory in (ROOT, ROOT/'src', ROOT/'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def read_jsonl(path):
    rows, ids = [], set()
    with open(path, encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError(f'{path}:{line_no}: invalid JSON') from e
            key = row.get('id', row.get('block_id'))
            if not isinstance(key, str) or not key or key in ids:
                raise ValueError(f'{path}:{line_no}: missing/duplicate ID: {key!r}')
            ids.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: empty JSONL')
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    result = json.loads(path.read_text(encoding='utf-8'))
    result['_base'] = str(path.parent)
    caps = {'max_calls': 4, 'total_input': 16384, 'total_output': 4096, 'request_total': 8192}
    if any(type(result['limits'].get(k)) is not int or not 1 <= result['limits'][k] <= cap
           for k, cap in caps.items()):
        raise ValueError('Config cannot increase task limits')
    if result['limits'].get('prompt_safety_tokens', 0) < 0:
        raise ValueError('Negative safety margin')
    return result


def config_path(config, key):
    path = Path(config[key])
    return path if path.is_absolute() else Path(config['_base'])/path


def validate_question(q):
    if not isinstance(q.get('question'), str) or not q['question'].strip():
        raise ValueError('question must be nonempty')
    options = q.get('options')
    if not isinstance(options, list) or not 4 <= len(options) <= 6:
        raise ValueError('4–6 options required')
    ids = [o.get('id') for o in options]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('option IDs must be unique nonempty strings')
    if any(not isinstance(o.get('text'), str) or not o['text'].strip() for o in options):
        raise ValueError('option text must be nonempty')


def validate_answer(a, q, allowed, allow_no_context=False):
    required = {'id', 'answer_ids', 'abstain', 'rationale', 'citations'}
    if set(a) != required or a['id'] != q['id']:
        raise ValueError('wrong fields or question ID')
    if type(a['abstain']) is not bool:
        raise ValueError('abstain must be boolean')
    for key in ('answer_ids', 'citations'):
        if not isinstance(a[key], list) or any(not isinstance(x, str) for x in a[key]):
            raise ValueError(f'{key} must be a string list')
        if len(set(a[key])) != len(a[key]):
            raise ValueError(f'{key}: duplicates')
    if not set(a['answer_ids']) <= {o['id'] for o in q['options']}:
        raise ValueError('unknown option ID')
    if len(a['citations']) > 4 or not set(a['citations']) <= set(allowed):
        raise ValueError('unknown/not shown/excess citations')
    if not isinstance(a['rationale'], str) or not 1 <= len(a['rationale'].strip()) <= 1200:
        raise ValueError('rationale must have 1–1200 characters')
    if a['abstain']:
        if a['answer_ids'] or a['citations']:
            raise ValueError('abstention must have empty lists')
    elif not a['answer_ids'] or (not a['citations'] and not allow_no_context):
        raise ValueError('non-abstention requires answers and citations')
    return a


# Repository paths are centralized so CLI entry points work from any directory.
CONFIG_FILES = ('config.json', 'measured_config.json', 'runtime.json', 'runtime_binding.json')
DATA_FILES = ('corpus.jsonl', 'dev_gold.jsonl', 'dev_questions.jsonl', 'dev_references.jsonl', 'own_gold.jsonl', 'own_questions.jsonl')
ARTIFACT_FILES = ('completion_manifest.json', 'corpus_audit.json', 'endpoint_usage.jsonl', 'final_metrics.json', 'request_attempts.jsonl', 'selection.json', 'semantic_reviews.json', 'server.log', 'server_resources.json', 'study_provenance.json', 'usage_reconciliation.json')
CORE_MODULES = ('build_index', 'common', 'decisions', 'evaluate', 'formulas', 'hardware', 'json_math', 'llm', 'pdf_corpus', 'predict', 'request_journal', 'retrieval', 'run_state', 'serve_local')
SCRIPT_MODULES = ('checkpoint_run', 'create_run_kit', 'finalize_results', 'package_review', 'package_submission', 'prepare_all', 'prepare_data', 'prepare_gold', 'prepare_questions', 'review_results', 'run_experiments', 'run_local_experiments', 'upgrade_project', 'validate_submission', 'verify_usage')


def project_path(root, name):
    root, name = Path(root), str(name)
    if name in CONFIG_FILES:
        return root/'configs'/name
    if name in DATA_FILES:
        return root/'data'/name
    if name in ARTIFACT_FILES:
        return root/'artifacts'/name
    if name == 'requirements-core.txt':
        return root/'requirements/core.txt'
    if name.endswith('.py') and name[:-3] in CORE_MODULES:
        return root/'src'/name
    if name.endswith('.py') and name[:-3] in SCRIPT_MODULES:
        return root/'scripts'/name
    return root/name
