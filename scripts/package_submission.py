import sys as _sys
from pathlib import Path as _Path
for _directory in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1]/'src', _Path(__file__).resolve().parents[1]/'scripts'):
    _sys.path.insert(0, str(_directory))
from common import project_path
#!/usr/bin/env python3
"""Package source and required results, excluding corpus, PDF, models and caches."""
import zipfile
from pathlib import Path
from common import ROOT

EXCLUDE = {'assets', 'index', 'source', '.venv', '.git', '__pycache__', 'experiments',
           'checkpoint', 'smoke', 'measurements', 'private_review'}
DATA = {'dev_questions.jsonl', 'dev_gold.jsonl', 'dev_references.jsonl',
        'qa15_questions.jsonl', 'qa15_references.jsonl', 'own_questions.jsonl', 'own_gold.jsonl',
        'dev_answers.jsonl', 'dev_trace.jsonl', 'corpus_audit.json'}


def files():
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if not p.is_file() or any(part in EXCLUDE or part.startswith('.venv') for part in rel.parts):
            continue
        if p.suffix == '.md' and rel.as_posix() not in {'README.md', 'report.md', 'data/tfkp_15_qa.md'}:
            continue
        if p.suffix in {'.py', '.md', '.ipynb', '.txt', '.csv'} or p.name in DATA or p.name in {
                'config.json', 'runtime.json', 'runtime_binding.json', 'server_resources.json',
                'error_reviews.json', 'completion_manifest.json', 'usage_reconciliation.json',
                'final_metrics.json', 'semantic_reviews.json', 'measured_config.json',
                'selection.json', 'study_provenance.json', 'layout_migration.json', '.gitignore'}:
            yield p


def main():
    required = ['build_index.py', 'predict.py', 'requirements.txt', 'config.json', 'runtime.json',
                'README.md', 'report.md', 'dev_answers.jsonl', 'dev_trace.jsonl', 'experiments.csv',
                'own_questions.jsonl', 'own_gold.jsonl', 'prepare_data.py', 'pdf_corpus.py',
                'prepare_questions.py', 'prepare_gold.py', 'prepare_all.py', 'data/tfkp_15_qa.md',
                'requirements-core.txt', 'runtime_binding.json']
    missing = [name for name in required if not (project_path(ROOT, name)).exists()]
    if missing:
        raise ValueError('Missing required outputs: '+', '.join(missing))
    from validate_submission import validate
    validate(ROOT)
    with zipfile.ZipFile(ROOT/'submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files():
            z.write(p, p.relative_to(ROOT))
    print('Created submission.zip; review report and public GitHub visibility before submitting.')


if __name__ == '__main__':
    main()
