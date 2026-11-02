import sys as _sys
from pathlib import Path as _Path
for _directory in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1]/'src', _Path(__file__).resolve().parents[1]/'scripts'):
    _sys.path.insert(0, str(_directory))
from common import project_path
#!/usr/bin/env python3
"""Dependency-free preflight for generated deliverables; never calls a model."""
import csv
import json
from pathlib import Path
from common import ROOT, read_jsonl, sha256, validate_question, validate_answer
from evaluate import check_trace

MODES = {'no_context', 'single_rag', 'final', 'no_formula', 'no_neighbors'}


def validate(root=ROOT):
    root = Path(root)
    manifest = json.loads((project_path(root, 'completion_manifest.json')).read_text(encoding='utf-8'))
    if manifest.get('status') != 'computed':
        raise ValueError('Run all five experiments, verify_usage.py and finalize_results.py first')
    for name, expected in manifest['files_sha256'].items():
        if sha256(project_path(root, name)) != expected:
            raise ValueError('Generated files changed since finalization: '+name)
    for name, expected in manifest.get('source_sha256', {}).items():
        if sha256(project_path(root, name)) != expected:
            raise ValueError('Prediction code changed since the measured run: '+name)
    questions = read_jsonl(project_path(root, 'dev_questions.jsonl'))
    gold = read_jsonl(project_path(root, 'dev_gold.jsonl'))
    answers = read_jsonl(root/'dev_answers.jsonl')
    traces = read_jsonl(root/'dev_trace.jsonl')
    ids = [q['id'] for q in questions]
    if [a['id'] for a in answers] != ids or [t['id'] for t in traces] != ids:
        raise ValueError('Final answers and trace must contain exactly one row per question, in input order')
    if {g['id'] for g in gold} != set(ids):
        raise ValueError('Dev gold/question IDs differ')
    for q, a, t in zip(questions, answers, traces):
        validate_question(q)
        validate_answer(a, q, set(t.get('context_block_ids', [])))
        if not check_trace(t):
            raise ValueError('Unverifiable usage or exceeded limits: '+q['id'])
    with (root/'experiments.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 5 or {r['name'] for r in rows} != MODES:
        raise ValueError('experiments.csv must contain the three systems and two ablations')
    if next(r for r in rows if r['name'] == 'no_context')['Q']:
        raise ValueError('no_context Q must be empty')
    report = (root/'report.md').read_text(encoding='utf-8')
    if not report.strip() or len(report.split()) > 1500:
        raise ValueError('Report must be nonempty and at most 1500 words')
    own_q, own_g = read_jsonl(project_path(root, 'own_questions.jsonl')), read_jsonl(project_path(root, 'own_gold.jsonl'))
    if len(own_q) != 8 or {q['id'] for q in own_q} != {g['id'] for g in own_g}:
        raise ValueError('Exactly eight author questions with matching gold are required')
    from collections import Counter
    if Counter(g['category'] for g in own_g) != {c: 2 for c in
            ('conditions', 'comparison', 'application', 'insufficiency')}:
        raise ValueError('Author questions must contain two per category')
    print('PASS: final format, question IDs, citations, limits, five CSV rows, eight author questions, report length.')
    validate_selection(root, answers, traces, rows)
    print('Selection, default config, answer hashes, trace mode and reviewed metrics agree.')


def validate_selection(root, answers, traces, rows):
    selection = json.loads((project_path(root, 'selection.json')).read_text())
    selected = selection['selected_mode']
    config = json.loads((project_path(root, 'config.json')).read_text())
    measured = json.loads((project_path(root, 'measured_config.json')).read_text())
    if config['mode'] != selected or any(t.get('mode') != selected for t in traces):
        raise ValueError('Default mode, published trace and selection differ')
    if {k:v for k,v in config.items() if k!='mode'} != {k:v for k,v in measured.items() if k!='mode'}:
        raise ValueError('Inference settings changed after measurements')
    provenance = json.loads((project_path(root, 'study_provenance.json')).read_text())
    if sha256(root/'dev_answers.jsonl') != provenance['experiments'][selected]['answers_sha256']:
        raise ValueError('Published answers differ from measured selected answers')
    if sha256(root/'dev_trace.jsonl') != selection['dev_trace_sha256']:
        raise ValueError('Published trace differs from selection')
    current = next(r for r in rows if r['name']==selected)
    for key in ('Q', 'Q_automatic', 'answer_accuracy'):
        if abs(float(current[key])-selection[key]) > 1e-12:
            raise ValueError('CSV/selection mismatch: '+key)
    from review_results import matching_reviews
    reviews = matching_reviews(project_path(root, 'semantic_reviews.json'), selected, answers)
    if selection['review_complete'] and len(reviews) != len(answers):
        raise ValueError('Complete review claimed for unmatched selected answers')
    expected_review_count=sum(e['metrics']['semantic_reviewed'] for m,e in provenance['experiments'].items() if m!='no_context')
    if selection['reviewed_count'] != expected_review_count or (selection['review_complete'] and expected_review_count != 60):
        raise ValueError('Uniform review count mismatch')


if __name__ == '__main__':
    validate()
